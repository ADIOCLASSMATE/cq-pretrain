"""Document-aware packing, independent DiT budgets and consumed-only receipts."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import time

import numpy as np
import torch

from data.batch import Batch
from data.batch_layout import build_plan
from data.loss_permissions import CONTRACT as LOSS_PERMISSION_CONTRACT, generation_target


PACK_LENGTH_BUCKETS = (2048, 4096, 8192)


def padded_pack_length(occupied, capacity, padding_multiple=None):
    """Choose the physical width after packing has consumed documents in stream order."""
    if padding_multiple is None:
        for width in PACK_LENGTH_BUCKETS:
            if occupied <= width:
                return min(width, capacity)
        return capacity
    if padding_multiple == 0:
        return capacity
    multiples = (occupied + padding_multiple - 1) // padding_multiple
    return min(capacity, multiples * padding_multiple)


def sample_objective(sample_id, probability, seed, epoch, salt):
    """Independent Bernoulli draws keyed by occurrence, safe across TP/restart.

    Separate salts keep objectives independent. A new epoch redraws them;
    prefetch depth, pack boundaries and worker completion order do not.
    """
    if probability == 1:return True
    if probability == 0:return False
    digest=hashlib.sha256(f'{salt}:{seed}:{epoch}:{sample_id}'.encode()).digest()
    return int.from_bytes(digest[:8],'big') < int(probability*(1 << 64))


def token_contract(root, cfg):
    from data.tokenizer import tokenizer_root
    path=tokenizer_root(root,cfg)/'tokenizer_config.json'
    config = json.loads(path.read_text())
    added = {value['content']: int(key) for key, value in config['added_tokens_decoder'].items()}
    required = {'bos': '<|im_start|>', 'eos': '<|endoftext|>', 'boi': '<|vision_start|>',
                'eoi': '<|vision_end|>', 'image': '<|image_pad|>', 'text_mask': '<|fim_pad|>'}
    if '<null_condition>' in added or cfg.null_condition == 'token':
        required['null_condition'] = '<null_condition>'
    tokens = {name: added[text] for name, text in required.items()}
    cfg.null_token_id = tokens.get('null_condition')
    if max(tokens.values()) >= cfg.vocab_size:
        raise ValueError('Model vocabulary cannot represent the selected checkpoint tokenizer IDs')
    cfg.text_mask_token_id, cfg.image_mask_token_id = tokens['text_mask'], tokens['image']
    return tokens


class Packer:
    def __init__(self, documents, cfg, length, tokens, state=None, pin_device=None,
                 max_images=0, padding_multiple=None):
        self.documents, self.cfg, self.length, self.tokens = documents, cfg, length, tokens
        self.pin_device=pin_device
        if state and padding_multiple is None:
            # Old cursors record an integer (including zero for fixed width).
            # Resume their original layout when no new alignment was requested.
            padding_multiple = (state.get('packing') or {}).get('padding_multiple')
        if max_images < 0 or (padding_multiple is not None and padding_multiple < 0):
            raise ValueError('Packing limits must be nonnegative')
        self.max_images,self.padding_multiple=max_images,padding_multiple
        if cfg.preserve_documents and max_images:
            raise ValueError('Whole documents cannot use max_images_per_pack; use the DiT microbatch image limit')
        self.element, self.offset = 0, 0
        self.serial = 0
        self.pending_image = None
        self.prepared_images={}
        self.document_size=None
        from data.cache import caches
        self.feature_cache,self.latent_cache=caches(cfg)
        self.packing_contract=dict(preserve_documents=cfg.preserve_documents,
            long_document_policy=cfg.long_document_policy,length=length,max_images=max_images,
            padding_multiple=padding_multiple,image_loss_permissions=LOSS_PERMISSION_CONTRACT,
            visual_query_mode=cfg.visual_query_mode,visual_queries=cfg.image_condition_queries,
            visibility='time_block_bidirectional_causal_v1')
        if padding_multiple is None:
            self.packing_contract['padding_buckets'] = [n for n in PACK_LENGTH_BUCKETS if n < length] + [length]
        self.packing_contract.update(cfg.wan_objective_policy())
        self.stats=dict(batches=0,pack_seconds=0.,plan_seconds=0.,pin_seconds=0.,
                        atomic_documents=0,long_document_windows=0)
        from data.vision import NativeImageProcessor
        self.processor = NativeImageProcessor(cfg,length)
        if state:
            restored=state.get('packing')
            if restored!=self.packing_contract:
                raise ValueError('Document packing policy or limits differ on resume')
            self.element, self.offset, self.serial = state['element'], state['offset'], state['serial']

    def prepare_image(self, doc, element=None):
        # An image that does not fit is reused in the next fragment. Resume
        # re-reads the same immutable image instead of serializing large pixels.
        element=self.element if element is None else element
        key=(doc['id'],element)
        if key in self.prepared_images:return self.prepared_images[key]
        if self.pending_image is None or self.pending_image[0]!=key:
            image = self.documents.image(doc,element)
            processed = image if getattr(self.documents,'images_preprocessed',False) else self.processor(image)
            self.pending_image=(key,processed)
            grid=self.pending_image[1][2]
            if math.prod(grid)//4+4>self.length:
                raise ValueError('Processed native image exceeds the physical pack budget; increase sequence length or lower vision_min_pixels')
        item=doc['elements'][element]
        if self.feature_cache is not None and 'cache_keys' not in item:
            source=item.get('source_sha256') or item.get('locator',{}).get('sha256')
            if source is None:raise ValueError('Cached visuals require a source content hash')
            clip=dict(item.get('clip',{'kind':'image'}), qwen_grid=list(self.pending_image[1][2]),
                      target_shape=list(self.pending_image[1][0].shape))
            identities=tuple(cache.identity(source,clip) for cache in (self.feature_cache,self.latent_cache))
            cached=[cache.load(identity) for cache,identity in zip((self.feature_cache,self.latent_cache),identities)]
            item['cache_keys']=identities
            if cached[0] is not None:item['cached_features']=cached[0]['features']
            if cached[1] is not None:item['cached_latent']=cached[1]['latent']
        if self.cfg.preserve_documents:
            self.prepared_images[key]=self.pending_image[1]
        return self.pending_image[1]

    def document_extent(self,doc):
        """Exact length for every document that fits; bounded lookahead otherwise.

        Decoded images are reused, and the scan stops as soon as the document
        is known to exceed one pack. No unbounded whole-document image cache.
        """
        if self.document_size is not None and self.document_size[0]==doc['id']:
            return self.document_size[1]
        size=(2 if doc.get('templated',True) else 1)+sum(
            len(e['input_ids']) for e in doc['elements'] if e['type']=='text')
        for index,e in enumerate(doc['elements']):
            if size>self.length:break
            if e['type']=='image':
                _,_,grid=self.prepare_image(doc,index)
                item=doc['elements'][index]
                size+=len(item['video_sequence']['ids']) if 'video_sequence' in item else math.prod(grid)//4+2
        self.document_size=(doc['id'],size)
        return size

    def state_dict(self):
        return dict(documents=self.documents.state_dict(), element=self.element,
                    offset=self.offset, serial=self.serial,packing=self.packing_contract.copy())

    def next(self):
        """Fill one pack with whole documents (or one window of a long one), in stream order."""
        began = time.perf_counter()
        pack = _Pack(self.length, self.tokens)
        while pack.position < pack.capacity:
            doc = self.documents.peek()
            if doc is None or not self._admit(doc, pack):
                break
            start, templated = self._open_document(doc, pack)
            if not self._write_document(doc, pack, start, templated):
                break  # Keep image atomic / continue a long document next pack.
        if not pack.records:
            return None
        return self._finish(pack, began)

    def _admit(self, doc, pack):
        """Whether `doc` (or its continuation) starts in this pack."""
        s, cfg = pack.capacity, self.cfg
        # Decided before empty text elements are skipped below.
        continued = pack.continued = bool(self.element or self.offset)
        if not self.prepared_images:
            self.documents.prefetch_images(doc, self.element)
        if cfg.preserve_documents and not continued:
            needed = self.document_extent(doc)
            if needed > s:
                if cfg.long_document_policy == 'error':
                    raise ValueError(
                        f'Document {doc["id"]} exceeds {s} tokens; '
                        'preserving its full context needs a longer sequence')
                # Explicit long-document windows start at an empty pack.
                if pack.position:
                    return False
            elif pack.position and s - pack.position < needed:
                return False
            else:
                self.stats['atomic_documents'] += 1
        # Empty text elements are legal in real interleaved documents. Look
        # through them before reserving a BOS: the next actual element may be
        # an image that cannot fit in the remaining pack space.
        elements = doc['elements']
        while self.element < len(elements):
            first = elements[self.element]
            if first['type'] != 'text' or self.offset < len(first['input_ids']):
                break
            self.element += 1
            self.offset = 0
        first = elements[self.element] if self.element < len(elements) else None
        if first and first['type'] == 'image':
            if self.max_images and len(pack.pixels) >= self.max_images:
                return False
            _, _, grid = self.prepare_image(doc)
            if 'video_sequence' in first:
                needed = len(first['video_sequence']['ids']) + 1
            else:
                needed = math.prod(grid) // 4 + 3
        else:
            needed = 2
        return not (pack.position and s - pack.position < needed)

    def _open_document(self, doc, pack):
        """Write the opener and the receipt record. Returns (start, templated).

        A continuation receives a fresh unlabelled opener; its body tokens are
        consumed exactly once and no preceding document is visible. Natural
        interleaved documents are plain text, not conversations, so they open
        directly at their first element with no <|im_start|>.
        """
        continued = pack.continued
        start = pack.position
        templated = doc.get('templated', True)
        pack.rope_cursor = 0
        if templated:
            pack.ids[pack.position] = self.tokens['bos']
            pack.segment[pack.position] = pack.seg
            pack.position += 1
            pack.rope_cursor = 1
        record = dict(id=doc['id'], element=self.element, offset=self.offset,
                      body_tokens=0, body_spans=[], images=0, completed=False,
                      format=doc.get('format'), direction=doc.get('direction'))
        record.update({k: doc[k] for k in ('training_view', 'typed_conditions') if k in doc})
        if self.cfg.preserve_documents:
            record['window_continuation'] = continued
            long_document = self.document_size is not None and self.document_size[1] > pack.capacity
            record['window_reason'] = 'backbone_length' if continued or long_document else None
            if record['window_reason']:
                self.stats['long_document_windows'] += 1
        pack.records.append(record)
        pack.record = record
        # Boundaries and meaningful history are local to this document/window.
        # BOS, role headers and closings alone do not make a new CFG category.
        pack.instruction_start = -1
        pack.has_history = False
        # A window can start inside a text block. Its next span alone does not
        # establish a new instruction boundary.
        pack.in_text_block = continued
        return start, templated

    def _write_document(self, doc, pack, start, templated):
        """Write elements until the document ends or the pack is full.

        Returns True when the document completed and the next one may follow.
        """
        signature = getattr(self.documents, 'signature', {})
        draws = (signature.get('seed', 0), signature.get('epoch', 0))
        permissions = {index: generation_target(doc, element)
                       for index, element in enumerate(doc['elements']) if element['type'] == 'image'}
        while self.element < len(doc['elements']) and pack.position < pack.capacity:
            e = doc['elements'][self.element]
            if e['type'] == 'text':
                self._write_text(e, pack, start)
            elif e['type'] == 'image':
                if not self._write_image(doc, e, pack, start, templated, permissions, draws):
                    break
            else:
                raise ValueError(f'Unsupported corpus element: {e["type"]}')
        if self.element == len(doc['elements']) and pack.position < pack.capacity:
            self._close_document(pack)
            return True
        if pack.position == start + (1 if templated else 0):
            raise RuntimeError('Packer made no progress')
        return False

    def _write_text(self, e, pack, start):
        take = min(len(e['input_ids']) - self.offset, pack.capacity - pack.position)
        if take:
            position = pack.position
            if e.get('context_text', True):
                if self.offset == 0 and (not pack.in_text_block or e.get('instruction_start')):
                    pack.instruction_start = position if pack.has_history else start
                pack.in_text_block = True
                pack.has_history = True
            pack.record['body_spans'].append((position, take))
            pack.ids[position:position + take] = e['input_ids'][self.offset:self.offset + take]
            pack.segment[position:position + take] = pack.seg
            pack.rope[:, position:position + take, 0] = np.arange(pack.rope_cursor, pack.rope_cursor + take)
            # Fabricated chat scaffolding is fed as input but never supervised:
            # the model must not learn to emit our wrappers.
            if e.get('supervised', True):
                pack.text_indices.extend(range(position, position + take))
            pack.position += take
            pack.rope_cursor += take
            self.offset += take
            pack.body_count += take
            pack.record['body_tokens'] += take
        if self.offset == len(e['input_ids']):
            self.element += 1
            self.offset = 0

    def _write_image(self, doc, e, pack, start, templated, permissions, draws):
        """Write one atomic visual. Returns False when it does not fit this pack."""
        from data.sequence import image_sequence

        target, processed, grid = self.prepare_image(doc)
        sequence = e.get('video_sequence') or image_sequence(grid, self.tokens)
        size = len(sequence['ids'])
        if pack.position + size > pack.capacity or (self.max_images and len(pack.pixels) >= self.max_images):
            return False
        self._record_conditioning(doc, e, pack, start, templated, permissions[self.element], draws)
        position, stop = pack.position, pack.position + size
        indices = position + sequence['visual']
        pack.ids[position:stop] = sequence['ids']
        pack.segment[position:stop] = pack.seg
        pack.sigma[position:stop] = sequence['sigma'] + position
        pack.rope[:, position:stop] = sequence['positions'] + pack.rope_cursor
        pack.image_queries[indices] = True
        pack.image_indices.append(indices)
        pack.text_indices.extend((position + sequence['text_indices']).tolist())
        if permissions[self.element]:
            pack.generation_image_indices.append(len(pack.pixels))
            shape = target.shape if target.ndim == 4 else (1, *target.shape)
            pack.generation_tokens += ((shape[0] - 1) // 4 + 1) * (shape[1] // 16) * (shape[2] // 16)
        pack.pixels.append(target)
        pack.vision_pixels.append(processed)
        pack.vision_grids.append(grid)
        pack.visual_features.append(e.get('cached_features'))
        pack.visual_latents.append(e.get('cached_latent'))
        pack.visual_cache_keys.append(e.get('cache_keys', (None, None)))
        pack.rope_cursor += int(sequence['positions'].max()) + 1
        pack.position = stop
        self.element += 1
        pack.record['images'] += 1
        self.pending_image = None
        self.prepared_images.pop((doc['id'], self.element - 1), None)
        return True

    def _record_conditioning(self, doc, e, pack, start, templated, generate_image, draws):
        """Null-condition and instruction-dropout choices for one visual occurrence."""
        cfg = self.cfg
        seed, epoch = draws
        occurrence = f'{doc["id"]}:{self.element}'
        unconditional = e.get('non_cfg')
        if unconditional is None:
            unconditional = doc.get('non_cfg')
        explicit_condition = unconditional is not None
        if unconditional is None:
            unconditional = not pack.has_history or sample_objective(
                occurrence, cfg.visual_uncond_probability, seed, epoch, 'visual-uncond-v1')
        elif type(unconditional) is not bool:
            raise ValueError('non_cfg must be a boolean when specified')
        pack.generation_unconditional.append(generate_image and unconditional)
        pack.instruction_starts.append(pack.instruction_start if generate_image else -1)
        # The opener is the same token, position and state in every templated
        # document, so reading it alone carries no content.
        pack.null_positions.append(start + 1 if templated and generate_image else -1)
        eligible = (generate_image and pack.instruction_start > start
                    and not (explicit_condition or unconditional))
        pack.instruction_dropout_eligible_images += int(eligible)
        pack.instruction_dropped.append(eligible and sample_objective(
            occurrence, cfg.visual_instruction_dropout_probability, seed, epoch, 'visual-instruction-drop-v1'))
        # A later target cannot drop across this image, even when both images
        # and multiple text spans belong to one turn.
        pack.has_history, pack.instruction_start = True, -1
        pack.in_text_block = False

    def _close_document(self, pack):
        position = pack.position
        pack.ids[position] = self.tokens['eos']
        pack.segment[position] = pack.seg
        pack.rope[:, position, 0] = pack.rope_cursor
        pack.text_indices.append(position)
        pack.position += 1
        pack.record['completed'] = True
        self.documents.advance()
        self.element = self.offset = 0
        self.document_size = None
        self.prepared_images.clear()
        pack.seg += 1

    def _finish(self, pack, began):
        """Trim to the physical width, build the batch and plan, and write the receipt."""
        cfg = self.cfg
        # Do not create the obsolete dense [2S,S] mask. The exact compact plan
        # owns masks for only the branches actually evaluated by this method.
        used = padded_pack_length(pack.position, pack.capacity, self.padding_multiple)
        ids, segment, sigma = pack.ids[:used], pack.segment[:used], pack.sigma[:used]
        rope, image_queries = pack.rope[:, :used], pack.image_queries[:used]
        indices = np.full((len(pack.pixels), max(map(len, pack.image_indices), default=0)), -1, np.int64)
        for i, span in enumerate(pack.image_indices):
            indices[i, :len(span)] = span
        batch = Batch(
            ids=torch.from_numpy(ids[None]),
            generation_image_indices=torch.tensor(pack.generation_image_indices, dtype=torch.long),
            latents=torch.empty(0, cfg.latent_channels * 4),
            image_indices=torch.from_numpy(indices),
            text_indices=torch.tensor(pack.text_indices, dtype=torch.long),
            image_queries=torch.from_numpy(image_queries[None]),
            sigma=torch.from_numpy(sigma[None]),
            segments=torch.from_numpy(segment[None]),
            row_col=torch.from_numpy(rope),
            noise=torch.empty(0),
            times=torch.empty(0))
        batch.generation_unconditional = torch.tensor(pack.generation_unconditional, dtype=torch.bool)
        batch.generation_instruction_starts = torch.tensor(pack.instruction_starts, dtype=torch.long)
        batch.generation_instruction_dropped = torch.tensor(pack.instruction_dropped, dtype=torch.bool)
        batch.generation_null_positions = torch.tensor(pack.null_positions, dtype=torch.long)
        if pack.generation_image_indices:
            self._attach_generation(batch, pack)
        if pack.vision_pixels:
            self._attach_vision(batch, pack)
        plan_begin = time.perf_counter()
        batch.mainline = build_plan(batch, cfg)
        self.stats['plan_seconds'] += time.perf_counter() - plan_begin
        pin_begin = time.perf_counter()
        if self.pin_device is not None:
            batch = batch.pin_memory(self.pin_device)
        self.stats['pin_seconds'] += time.perf_counter() - pin_begin
        self.stats['batches'] += 1
        self.stats['pack_seconds'] += time.perf_counter() - began
        self.serial += 1
        receipt = dict(
            serial=self.serial,
            next_state=self.state_dict(),
            documents=pack.records,
            body_tokens=pack.body_count,
            occupied=pack.position,
            physical_length=used,
            images=len(pack.pixels),
            generation_images=batch.generation_image_count,
            condition_query_tokens=int(batch.mainline.query_image.sum()),
            unconditional_images=sum(pack.generation_unconditional),
            instruction_dropped_images=sum(pack.instruction_dropped),
            instruction_dropout_eligible_images=pack.instruction_dropout_eligible_images,
            generation_tokens=pack.generation_tokens,
            dit_microbatches=(batch.generation.dit_microbatch_count if batch.generation else 0),
            ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        return batch, receipt

    def _attach_generation(self, batch, pack):
        from data.generation import WanGenerationInputs

        cfg = self.cfg
        selected = pack.generation_image_indices
        if cfg.visual_query_mode == 'native':
            queries = [math.prod(pack.vision_grids[i]) // 4 for i in selected]
        else:
            queries = [cfg.image_condition_queries for _ in selected]
        batch.generation = WanGenerationInputs.prepare(
            [pack.pixels[i] for i in selected], queries, cfg.dit_width // cfg.dit_heads,
            cfg.generation_token_budget, cfg.generation_microbatch_images,
            cached_latents=[pack.visual_latents[i] for i in selected], rf=cfg.rf)
        batch.generation.cache_keys = tuple(pack.visual_cache_keys[i][1] for i in selected)

    def _attach_vision(self, batch, pack):
        from data.vision import VisionInputs

        cfg, config = self.cfg, self.processor.config
        batch.vision = VisionInputs.prepare(
            torch.cat(pack.vision_pixels), pack.vision_grids,
            torch.from_numpy(np.concatenate(pack.image_indices)), config)
        batch.vision.cache_keys = tuple(x[0] for x in pack.visual_cache_keys)
        if not any(x is not None for x in pack.visual_features):
            return
        cached, online_pixels, online_grids, online_indices = [], [], [], []
        offset = 0
        for features, processed, grid in zip(pack.visual_features, pack.vision_pixels, pack.vision_grids,
                                             strict=True):
            count = math.prod(grid) // 4
            if features is None:
                features = torch.zeros(count, cfg.width)
                online_pixels.append(processed)
                online_grids.append(grid)
                online_indices.append(torch.arange(offset, offset + count))
            if features.shape != (count, cfg.width):
                raise ValueError('Cached visual feature shape differs')
            cached.append(features)
            offset += count
        batch.vision.features = torch.cat(cached)
        if online_pixels:
            batch.vision.online = VisionInputs.prepare(
                torch.cat(online_pixels), online_grids, torch.cat(online_indices), config)
        # Cached inputs still carry all grids/indices for CQ planning, but raw
        # patches are not transferred for them.
        batch.vision.pixel_values = torch.empty(0)


class _Pack:
    """Arrays and per-occurrence lists of one pack while it is being filled."""

    def __init__(self, capacity, tokens):
        self.capacity = capacity
        self.ids = np.full(capacity, tokens['eos'], np.int64)
        self.segment = np.full(capacity, -1, np.int64)
        self.sigma = np.arange(capacity, dtype=np.int64)
        self.rope = np.zeros((3, capacity, 1), np.float32)
        self.image_queries = np.zeros(capacity, bool)
        self.position = self.seg = self.body_count = self.generation_tokens = 0
        self.rope_cursor = 0
        self.records, self.record = [], None
        self.text_indices, self.image_indices, self.pixels = [], [], []
        self.vision_pixels, self.vision_grids = [], []
        self.visual_features, self.visual_latents, self.visual_cache_keys = [], [], []
        self.generation_image_indices, self.generation_unconditional = [], []
        self.instruction_starts, self.instruction_dropped, self.null_positions = [], [], []
        self.instruction_dropout_eligible_images = 0
        self.instruction_start, self.has_history, self.in_text_block = -1, False, False
        self.continued = False


class ConsumedLoader:
    """One bounded packing thread overlaps workers with model execution.

    state_dict returns *acknowledged* receipts, never the producer's state.
    Checkpointing a prefetched loader therefore cannot skip unseen examples.
    Acknowledge only after an optimizer update successfully commits.
    """
    def __init__(self, packer, prefetch=2, state=None):
        if prefetch < 1: raise ValueError('Prefetch must be positive')
        self.packer, self.limit = packer, prefetch
        self.pool = ThreadPoolExecutor(max_workers=1,
            initializer=(lambda:torch.cuda.set_device(self.packer.pin_device)) if self.packer.pin_device else None)
        self.pending, self.delivered = deque(), deque()
        self.consumed = copy.deepcopy(state or packer.state_dict())
        self.closed = False
        self._fill()

    def _fill(self):
        while not self.closed and len(self.pending) < self.limit:
            self.pending.append(self.pool.submit(self.packer.next))

    def next(self):
        if self.closed: return None
        result = self.pending.popleft().result()
        if result is None:
            self.closed = True
            return None
        self.delivered.append(result[1])
        self._fill()
        return result

    def acknowledge(self, receipt):
        expected = self.delivered[0]
        if receipt['serial'] != expected['serial']:
            raise ValueError('Consumption acknowledgement must be in order')
        self.delivered.popleft()
        self.consumed = copy.deepcopy(expected['next_state'])

    def state_dict(self): return copy.deepcopy(self.consumed)

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.packer.documents.close()
