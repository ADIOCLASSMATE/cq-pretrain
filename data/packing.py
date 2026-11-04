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
from models.cq import build_plan
from data.loss_permissions import CONTRACT as LOSS_PERMISSION_CONTRACT, image_loss_allowed


def select_i2t_image(sample_id, probability, seed=0, epoch=0):
    """Independent Bernoulli draws keyed by occurrence, safe across TP/restart.

    No batch quota or inverse-probability weighting: the skipped images are
    intentionally absent from the image objective. A new epoch redraws them.
    """
    if probability == 1:return True
    if probability == 0:return False
    digest=hashlib.sha256(f'i2t-null-v1:{seed}:{epoch}:{sample_id}'.encode()).digest()
    return int.from_bytes(digest[:8],'big') < int(probability*(1 << 64))


def token_contract(root, cfg):
    from data.tokenizer import tokenizer_root
    path=tokenizer_root(root,cfg)/'tokenizer_config.json'
    config = json.loads(path.read_text())
    added = {value['content']: int(key) for key, value in config['added_tokens_decoder'].items()}
    required = {'bos': '<|im_start|>', 'eos': '<|endoftext|>', 'boi': '<|vision_start|>',
                'eoi': '<|vision_end|>', 'image': '<|image_pad|>', 'text_mask': '<|fim_pad|>'}
    tokens = {name: added[text] for name, text in required.items()}
    if max(tokens.values()) >= cfg.vocab_size:
        raise ValueError('Model vocabulary cannot represent the selected checkpoint tokenizer IDs')
    cfg.text_mask_token_id, cfg.image_mask_token_id = tokens['text_mask'], tokens['image']
    return tokens


class Packer:
    def __init__(self, documents, cfg, length, tokens, state=None, pin_device=None,
                 max_images=0, padding_multiple=0):
        if cfg.vision_encoder == 'rgb_patch' and length < cfg.image_tokens + 4:
            raise ValueError('Sequence must fit one whole image with document/image boundaries')
        self.documents, self.cfg, self.length, self.tokens = documents, cfg, length, tokens
        self.pin_device=pin_device
        if max_images < 0 or padding_multiple < 0:raise ValueError('Packing limits must be nonnegative')
        self.max_images,self.padding_multiple=max_images,padding_multiple
        if cfg.preserve_documents and max_images:
            raise ValueError('Whole documents cannot use max_images_per_pack; use the DiT microbatch image limit')
        self.element, self.offset = 0, 0
        self.serial = 0
        self.processor = None
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
        if cfg.i2t_image_loss_probability != 1:
            self.packing_contract['i2t_image_loss_probability']=cfg.i2t_image_loss_probability
            self.packing_contract['i2t_sampling']='sample_seed_epoch_sha256_v1'
        self.stats=dict(batches=0,pack_seconds=0.,plan_seconds=0.,pin_seconds=0.,atomic_captions=0,
                        atomic_documents=0,long_document_windows=0)
        if cfg.vision_encoder == 'qwen35':
            from models.vision import NativeImageProcessor
            self.processor = NativeImageProcessor(cfg,length)
        if state:
            restored=state.get('packing') if cfg.preserve_documents else state.get('packing',self.packing_contract)
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
        if self.cfg.preserve_documents:self.prepared_images[key]=self.pending_image[1]
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
        began=time.perf_counter()
        s, cfg, t = self.length, self.cfg, self.tokens
        ids = np.full(s, t['eos'], np.int64)
        segment = np.full(s, -1, np.int64)
        sigma = np.arange(s, dtype=np.int64)
        rope = np.zeros((3 if cfg.renderer=='wan' else 2, s, 1), np.float32)
        image_queries = np.zeros(s, bool)
        text_indices, image_indices, pixels, records = [], [], [], []
        generation_image_indices=[]
        i2t_null_candidates=i2t_null_images=0
        vision_pixels,vision_grids=[],[]
        visual_features,visual_latents,visual_cache_keys=[],[],[]
        position, seg, body_count = 0, 0, 0
        generation_tokens=0
        current_record = None
        while position < s:
            doc = self.documents.peek()
            if doc is None:
                break
            # Only image-first caption pairs have the fixed trained null
            # prefix. Conversation/interleaved images may have real context.
            is_null=doc.get('format')=='caption' and doc.get('direction')=='I2T'
            signature=getattr(self.documents,'signature',{})
            sampled_image=not is_null or select_i2t_image(doc['id'],cfg.i2t_image_loss_probability,
                signature.get('seed',0),signature.get('epoch',0))
            if cfg.renderer=='wan' and doc.get('direction') in ('I2T','V2T'):sampled_image=False
            # This is an objective permission, independent of source format or
            # random null-target sampling. Validate all flags before packing.
            image_loss_allowed(doc)
            image_permissions={index:image_loss_allowed(doc,element)
                for index,element in enumerate(doc['elements']) if element['type']=='image'}
            continued=bool(self.element or self.offset)
            if not self.prepared_images:self.documents.prefetch_images(doc, self.element)
            if cfg.preserve_documents and self.element==0 and self.offset==0:
                needed=self.document_extent(doc)
                if needed>s:
                    if cfg.long_document_policy=='error' or (doc.get('format')=='caption' and doc.get('direction')=='T2I'):
                        raise ValueError(f'Document {doc["id"]} exceeds {s} tokens; preserving its full context needs a longer sequence')
                    # Explicit long-document windows start at an empty pack.
                    if position:break
                elif position and s-position<needed:
                    break
                else:self.stats['atomic_documents']+=1
            # Empty text elements are legal in real interleaved documents.
            # Look through them before reserving a BOS: the next actual element
            # may be an image that cannot fit in the remaining pack space.
            while self.element<len(doc['elements']):
                first=doc['elements'][self.element]
                if first['type']!='text' or self.offset<len(first['input_ids']):break
                self.element+=1;self.offset=0
            first = doc['elements'][self.element] if self.element < len(doc['elements']) else None
            # A caption pair must keep its conditioning context. Reserving only
            # the first text token can otherwise strand its image in the next
            # pack with a new BOS. Preprocessed image lookahead is cached and
            # reused when this whole document moves to the next pack.
            if not cfg.preserve_documents and doc.get('format')=='caption' and self.element==0 and self.offset==0:
                image_elements=[i for i,e in enumerate(doc['elements']) if e['type']=='image']
                if len(image_elements)==1:
                    generate_image=sampled_image and image_permissions[image_elements[0]]
                    if self.processor is not None:
                        target,_,grid=self.prepare_image(doc,image_elements[0])
                        image_tokens=math.prod(grid)//4
                        nt=(target.shape[0]*target.shape[1]//256+cfg.image_condition_queries) if generate_image else 0
                    else:
                        image_tokens=cfg.image_tokens;nt=0
                    needed=sum(len(e['input_ids']) for e in doc['elements'] if e['type']=='text')+image_tokens+4
                    if needed>s and doc.get('direction')=='T2I':
                        raise ValueError(f'Caption pair {doc["id"]} needs {needed} tokens; cannot preserve its context in {s}')
                    if needed<=s and position and (s-position<needed or
                            (self.max_images and len(pixels)>=self.max_images) or
                            (cfg.renderer=='moss' and cfg.generation_token_budget and generation_tokens+nt>cfg.generation_token_budget)):
                        break
                    if needed<=s:self.stats['atomic_captions']+=1
            if first and first['type']=='image' and self.max_images and len(pixels)>=self.max_images:break
            if first and first['type']=='image' and self.processor is not None:
                generate_image=sampled_image and image_permissions[self.element]
                first_target,_,grid=self.prepare_image(doc)
                if generate_image and not cfg.preserve_documents and cfg.renderer=='moss' and generation_image_indices and cfg.generation_token_budget:
                    first_tokens=first_target.shape[0]*first_target.shape[1]//256+cfg.image_condition_queries
                    if generation_tokens+first_tokens>cfg.generation_token_budget:break
                needed=(len(first['video_sequence']['ids'])+1 if 'video_sequence' in first else math.prod(grid)//4+3)
            else:
                needed = cfg.image_tokens + 3 if first and first['type'] == 'image' else 2
            if position and s - position < needed:
                break
            # A continuation receives a fresh unlabelled opener; its body tokens
            # are consumed exactly once and no preceding document is visible.
            # Natural interleaved documents are plain text, not conversations,
            # so they open directly at their first element with no <|im_start|>.
            start = position
            templated = doc.get('templated', True)
            rope_cursor = 0
            if templated:
                ids[position], segment[position] = t['bos'], seg
                position += 1; rope_cursor = 1
            current_record = dict(id=doc['id'], element=self.element, offset=self.offset,
                                  body_tokens=0, body_spans=[], images=0, completed=False,
                                  format=doc.get('format'),direction=doc.get('direction'))
            if cfg.preserve_documents:
                current_record['window_continuation']=continued
                current_record['window_reason']='backbone_length' if (
                    continued or (self.document_size is not None and self.document_size[1]>s)) else None
                if current_record['window_reason']:self.stats['long_document_windows']+=1
            records.append(current_record)
            while self.element < len(doc['elements']) and position < s:
                e = doc['elements'][self.element]
                if e['type'] == 'text':
                    available = len(e['input_ids']) - self.offset
                    take = min(available, s - position)
                    if take:
                        current_record['body_spans'].append((position,take))
                        ids[position:position+take] = e['input_ids'][self.offset:self.offset+take]
                        segment[position:position+take] = seg
                        rope[:, position:position+take, 0] = np.arange(rope_cursor, rope_cursor+take)
                        # Fabricated chat scaffolding is fed as input but never
                        # supervised: the model must not learn to emit our wrappers.
                        if e.get('supervised', True):
                            text_indices.extend(range(position, position+take))
                        position += take; rope_cursor += take; self.offset += take
                        body_count += take; current_record['body_tokens'] += take
                    if self.offset == len(e['input_ids']):
                        self.element += 1; self.offset = 0
                elif e['type'] == 'image':
                    if cfg.renderer=='wan':
                        from data.sequence import image_sequence
                        target,processed,grid=self.prepare_image(doc)
                        sequence=e.get('video_sequence') or image_sequence(grid,t)
                        size=len(sequence['ids'])
                        if position+size>s or (self.max_images and len(pixels)>=self.max_images):break
                        generate_image=sampled_image and image_permissions[self.element] and e.get('generation_target',True)
                        stop=position+size;indices=position+sequence['visual']
                        ids[position:stop]=sequence['ids'];segment[position:stop]=seg
                        sigma[position:stop]=sequence['sigma']+position
                        rope[:,position:stop]=sequence['positions']+rope_cursor
                        image_queries[indices]=True;image_indices.append(indices)
                        text_indices.extend((position+sequence['text_indices']).tolist())
                        if generate_image:
                            generation_image_indices.append(len(pixels))
                            shape=target.shape if target.ndim==4 else (1,*target.shape)
                            generation_tokens+=((shape[0]-1)//4+1)*(shape[1]//16)*(shape[2]//16)
                        pixels.append(target);vision_pixels.append(processed);vision_grids.append(grid)
                        visual_features.append(e.get('cached_features'));visual_latents.append(e.get('cached_latent'))
                        visual_cache_keys.append(e.get('cache_keys',(None,None)))
                        rope_cursor+=int(sequence['positions'].max())+1;position=stop
                        self.element+=1;current_record['images']+=1
                        self.pending_image=None;self.prepared_images.pop((doc['id'],self.element-1),None)
                        continue
                    generate_image=sampled_image and image_permissions[self.element]
                    if self.max_images and len(pixels)>=self.max_images:break
                    if self.processor is not None:
                        target,processed,grid=self.prepare_image(doc)
                        height,width=grid[1]//2,grid[2]//2
                        if cfg.renderer=='moss':
                            nt=(target.shape[0]*target.shape[1]//256+cfg.image_condition_queries) if generate_image else 0
                            if not cfg.preserve_documents and generation_image_indices and cfg.generation_token_budget and generation_tokens+nt>cfg.generation_token_budget:break

                    else:
                        height=width=math.isqrt(cfg.image_tokens)
                    n = height*width
                    if s - position < n + 2:
                        break
                    ids[position], segment[position] = t['boi'], seg
                    rope[:, position, 0] = rope_cursor
                    # Structural BOI/EOI are valid A targets, never corpus tokens.
                    text_indices.append(position)
                    position += 1; rope_cursor += 1
                    end = position+n
                    ids[position:end], segment[position:end] = t['image'], seg
                    image_queries[position:end] = True
                    image_indices.append(np.arange(position, end, dtype=np.int64))
                    if generate_image:generation_image_indices.append(len(image_indices)-1)
                    if is_null:
                        i2t_null_candidates+=1
                        i2t_null_images+=int(generate_image)
                    sigma[position:end], sigma[end] = position + 1, position
                    local = np.arange(n)
                    rope[0, position:end, 0] = rope_cursor + local // width
                    rope[1, position:end, 0] = rope_cursor + local % width
                    rope_cursor += max(height,width)
                    ids[end], segment[end] = t['eoi'], seg
                    rope[:, end, 0] = rope_cursor
                    # EOI reads a strictly smaller sigma than the image body, so
                    # it is predicted without ever seeing the picture: its loss is
                    # structurally near zero and carries no signal.
                    if self.processor is not None:
                        pixels.append(target);vision_pixels.append(processed);vision_grids.append(grid)
                        if cfg.renderer=='moss':generation_tokens+=nt
                        self.pending_image=None
                        self.prepared_images.pop((doc['id'],self.element),None)
                    else:pixels.append(self.documents.image(doc, self.element))
                    position = end + 1; rope_cursor += 1
                    self.element += 1; current_record['images'] += 1
                else:
                    raise ValueError(f'Unsupported corpus element: {e["type"]}')
            if self.element == len(doc['elements']) and position < s:
                ids[position], segment[position] = t['eos'], seg
                rope[:, position, 0] = rope_cursor
                text_indices.append(position); position += 1
                current_record['completed'] = True
                self.documents.advance(); self.element = self.offset = 0
                self.document_size=None
                self.prepared_images.clear()
                seg += 1
            else:
                if position == start + (1 if templated else 0):
                    raise RuntimeError('Packer made no progress')
                break  # Keep image atomic / continue a long document next pack.
        if not records:
            return None
        # Do not create the obsolete dense [2S,S] mask. The exact compact plan
        # owns masks for only the branches actually evaluated by this method.
        n_images = len(pixels)
        if self.padding_multiple:
            used=min(s,math.ceil(position/self.padding_multiple)*self.padding_multiple)
            ids,segment,sigma=ids[:used],segment[:used],sigma[:used]
            rope,image_queries=rope[:,:used],image_queries[:used]
        indices=np.full((n_images,max(map(len,image_indices),default=cfg.image_tokens)),-1,np.int64)
        for i,span in enumerate(image_indices):indices[i,:len(span)]=span
        batch = Batch(ids=torch.from_numpy(ids[None]),
            pixels=torch.from_numpy(np.stack(pixels).transpose(0,3,1,2).copy()) if pixels and cfg.renderer not in ('moss','wan') else torch.empty(0,3,cfg.image_size,cfg.image_size,dtype=torch.uint8),
            latents=torch.empty(0, cfg.image_tokens, cfg.latent_channels),
            image_indices=torch.from_numpy(indices),
            text_indices=torch.tensor(text_indices, dtype=torch.long), image_queries=torch.from_numpy(image_queries[None]),
            sigma=torch.from_numpy(sigma[None]), segments=torch.from_numpy(segment[None]), row_col=torch.from_numpy(rope),
            noise=torch.empty(0), times=torch.empty(0), mask=torch.empty(0,dtype=torch.bool), valid_rows=torch.empty(0,dtype=torch.bool))
        if cfg.renderer in ('moss','wan') or len(generation_image_indices)!=n_images:
            batch.generation_image_indices=torch.tensor(generation_image_indices,dtype=torch.long)
        if generation_image_indices and cfg.renderer=='moss':
            from data.generation import GenerationInputs
            batch.generation=GenerationInputs.prepare([pixels[i] for i in generation_image_indices],cfg.image_condition_queries,cfg.dit_width//cfg.dit_heads,
                token_budget=cfg.generation_token_budget if cfg.preserve_documents else 0,
                max_images=cfg.generation_microbatch_images if cfg.preserve_documents else 0)
        if generation_image_indices and cfg.renderer=='wan':
            from data.generation import WanGenerationInputs
            queries=[math.prod(vision_grids[i])//4 if cfg.visual_query_mode=='native' else cfg.image_condition_queries for i in generation_image_indices]
            batch.generation=WanGenerationInputs.prepare([pixels[i] for i in generation_image_indices],queries,cfg.dit_width//cfg.dit_heads,
                cfg.generation_token_budget,cfg.generation_microbatch_images,
                cached_latents=[visual_latents[i] for i in generation_image_indices])
            batch.generation.cache_keys=tuple(visual_cache_keys[i][1] for i in generation_image_indices)
        if vision_pixels:
            from models.vision import VisionInputs
            batch.vision=VisionInputs.prepare(torch.cat(vision_pixels),vision_grids,
                torch.from_numpy(np.concatenate(image_indices)),self.processor.config)
        if batch.vision is not None and cfg.renderer=='wan':
            batch.vision.cache_keys=tuple(x[0] for x in visual_cache_keys)
            if any(x is not None for x in visual_features):
                cached=[];online_pixels=[];online_grids=[];online_indices=[];offset=0
                for features,processed,grid in zip(visual_features,vision_pixels,vision_grids):
                    count=math.prod(grid)//4
                    if features is None:
                        features=torch.zeros(count,cfg.width)
                        online_pixels.append(processed);online_grids.append(grid);online_indices.append(torch.arange(offset,offset+count))
                    if features.shape!=(count,cfg.width):raise ValueError('Cached visual feature shape differs')
                    cached.append(features);offset+=count
                batch.vision.features=torch.cat(cached)
                if online_pixels:batch.vision.online=VisionInputs.prepare(torch.cat(online_pixels),online_grids,torch.cat(online_indices),self.processor.config)
                # Cached inputs still carry all grids/indices for CQ planning,
                # but raw patches are not transferred for them.
                batch.vision.pixel_values=torch.empty(0)
        plan_begin=time.perf_counter()
        batch.mainline = build_plan(batch, cfg)
        self.stats['plan_seconds']+=time.perf_counter()-plan_begin
        pin_begin=time.perf_counter()
        if self.pin_device is not None:batch=batch.pin_memory(self.pin_device)
        self.stats['pin_seconds']+=time.perf_counter()-pin_begin
        self.stats['batches']+=1
        self.stats['pack_seconds']+=time.perf_counter()-began
        self.serial += 1
        receipt = dict(serial=self.serial, next_state=self.state_dict(), documents=records,
                       body_tokens=body_count, occupied=position, images=n_images,
                       generation_images=batch.generation_image_count,
                       i2t_null_candidates=i2t_null_candidates,i2t_null_images=i2t_null_images,
                       generation_tokens=generation_tokens,
                       dit_microbatches=(max(1,len(batch.generation.microbatches)) if batch.generation else 0),
                       ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        return batch, receipt


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
            initializer=(lambda:torch.npu.set_device(self.packer.pin_device)) if self.packer.pin_device else None)
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
