"""Batched online tokenization and local image lookahead with exact cursors."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import os
import threading
import time

import numpy as np

from data.catalog import permute
from data.sources import open_catalog
from data.images import ImageReader
from data.tokenizer import tokenizer_root, tokenizer_metadata
from data.loss_permissions import CONTRACT as LOSS_PERMISSION_CONTRACT, image_loss_allowed
import data.chat as chat


class DocumentStream:
    """DP owns shuffled contiguous blocks; CPU workers only execute tasks.

    Arrow, Rust tokenizers and Pillow release the GIL. Threads avoid spawning
    an extra torch runtime per rank/worker and copying decoded arrays through
    IPC. The worker count and all queue depths may change on exact restart.
    """
    def __init__(self, root, view, rank, world, seed, epoch=0, workers=2, prefetch=4,
                 state=None, allow_development=False, native_images=False,
                 image_size=512, image_threads=2, image_prefetch=16,
                 block_rows=2048, tokenize_rows=32, source_root=None,
                 cache_bytes=128 << 20, cfg=None, sequence_length=8192):
        if (view not in ('A', 'B') or not 0 <= rank < world or
                min(workers, prefetch, image_threads, image_prefetch, tokenize_rows) < 1):
            raise ValueError('Invalid view, topology, workers or queue bounds')
        # Limit Rust's shared Rayon pool; concurrency is owned by this pipeline.
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('RAYON_NUM_THREADS', '1')
        from tokenizers import Tokenizer
        self.catalog = open_catalog(root, source_root, block_rows, cache_bytes)
        membership = self.catalog.check(allow_development)
        self.tokenizer_info=tokenizer_metadata(cfg,self.catalog.meta['tokenizer'])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_root(root,cfg) / 'tokenizer.json'))
        self.special_ids = set(self.tokenizer_info['counting']['exclude_special_token_ids'])
        self.counting_matches_corpus=self.tokenizer_info==self.catalog.meta['tokenizer']
        self.signature = dict(snapshot=self.catalog.meta['identity'], membership=membership['identity'],
                              view=view, rank=rank, world=world, seed=seed, epoch=epoch,
                              block_rows=block_rows, native_images=native_images,
                              image_size=image_size, tokenization='qwen35_chat_template_v1',
                              tokenizer=self.tokenizer_info,
                              image_loss_permissions=LOSS_PERMISSION_CONTRACT)
        self.position = dict(block=0, row=0)
        if cfg is not None:
            self.signature['generation']={'renderer':cfg.renderer,'max_pixels':cfg.generation_max_pixels,'flow_draws':cfg.flow_draws,'channels':cfg.latent_channels,'token_budget':cfg.generation_token_budget}
            if cfg.renderer=='wan':
                self.signature['generation'].update(visual_uncond_probability=cfg.visual_uncond_probability,
                    objective_policy='wan_non_cfg_flag_v2')
            if cfg.i2t_image_loss_probability != 1:
                self.signature['generation']['i2t_image_loss_probability']=cfg.i2t_image_loss_probability
                self.signature['generation']['i2t_sampling']='sample_seed_epoch_sha256_v1'
            if cfg.preserve_documents:
                self.signature['packing']={'preserve_documents':True,'long_document_policy':cfg.long_document_policy,
                    'generation_microbatch_images':cfg.generation_microbatch_images}
            self.signature['vision']={'encoder':cfg.vision_encoder,'min_pixels':cfg.vision_min_pixels,
                                      'max_pixels':cfg.vision_max_pixels,'sequence_length':sequence_length}
        if state:
            if state['signature'] != self.signature:
                raise ValueError('Data identity/view/topology/seed/epoch differs on resume')
            self.position = copy.deepcopy(state['position'])
        self.total = len(range(rank, self.catalog.meta['units'], world))
        if not 0 <= self.position['block'] <= self.total or self.position['row'] < 0:
            raise ValueError('Invalid document cursor')
        self.submitted = self.position.copy()
        self.pending, self.rows, self.row_index = deque(), None, 0
        self.prefetch, self.tokenize_rows = prefetch, tokenize_rows
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='online-tokenize')
        self.image_pool = (ThreadPoolExecutor(max_workers=image_threads, thread_name_prefix='local-image')
                           if view == 'A' else None)
        self.image_reader = (ImageReader(self.catalog.source_root, self.catalog.source_sizes,
                                         image_size, native_images) if view == 'A' else None)
        self.image_pending, self.image_limit = {}, image_prefetch
        self.cfg, self.sequence_length = cfg, sequence_length
        self.images_preprocessed = native_images and cfg is not None
        self.image_local = threading.local()
        self.stats = dict(documents=0, text_spans=0, tokenizer_tokens=0, corpus_text_tokens=0,
                          tokenize_seconds=0., parquet_seconds=0., image_wait_seconds=0.,aspect_padded_images=0)
        self.closed = False
        self._fill()

    def _block(self, local):
        index = self.signature['rank'] + local * self.signature['world']
        physical = permute(index, self.catalog.meta['units'], self.signature['seed'], self.signature['epoch'])
        return self.catalog.block(physical)

    def _load(self, shard, start, count):
        begin = time.perf_counter()
        rows = self.catalog.read_rows(shard, start, count)
        read_seconds = time.perf_counter() - begin
        built = [chat.build(row) for row in rows]
        spans = [span['text'] for layout, _, _ in built for span in layout if span['kind'] == 'text']
        begin = time.perf_counter()
        encodings = self.tokenizer.encode_batch(spans, add_special_tokens=False) if spans else []
        tokenize_seconds = time.perf_counter() - begin
        tokens = iter(encodings)
        documents, token_count, corpus_count = [], 0, 0
        for row, (layout, direction, templated) in zip(rows, built):
            elements, counted = [], 0
            allowed = image_loss_allowed(row)
            image_permissions = [image_loss_allowed(row, info) for info in row['images']]
            for span in layout:
                if span['kind'] == 'text':
                    ids = next(tokens).ids
                    token_count += len(ids)
                    # Only real corpus text enters the published token count;
                    # fabricated scaffolding must not perturb the audit.
                    if not span['synthetic']:
                        counted += sum(token not in self.special_ids for token in ids)
                    if not len(ids):
                        continue
                    elements.append(dict(type='text', input_ids=np.asarray(ids, dtype=np.int32),
                                         supervised=span['supervised']))
                elif span['kind'] == 'image':
                    if self.signature['view'] == 'A':
                        info = row['images'][span['image_index']]
                        locator = (dict(info) if info.get('path') or info.get('frames_path') else
                                   dict(shard=row['source_shard'], offset=info['offset'],
                                        length=info['length'], sha256=info['sha256']))
                        reference = repr(sorted(locator.items()))
                        elements.append(dict(type='image', image_ref=reference, locator=locator,
                            image_loss_allowed=image_permissions[span['image_index']],
                            non_cfg=info.get('non_cfg'),
                            generation_target=(row.get('direction',direction) not in ('I2T','V2T') and
                                               span.get('role','assistant')=='assistant')))
                else:
                    raise ValueError(f'Unknown span kind: {span["kind"]}')
            if self.counting_matches_corpus and counted != row['text_tokens']:
                raise RuntimeError(f'Online tokenizer differs from corpus accounting: {row["sample_id"]}')
            corpus_count += row['text_tokens']
            documents.append(dict(id=row['sample_id'], elements=elements, text_tokens=counted,
                                  format=row['format'],direction=row.get('direction',direction),templated=templated,
                                  non_cfg=row.get('non_cfg'),
                                  corpus_text_tokens=row['text_tokens'],image_loss_allowed=allowed))
        stats = dict(documents=len(documents), text_spans=len(spans), tokenizer_tokens=token_count,
                     corpus_text_tokens=corpus_count, tokenize_seconds=tokenize_seconds,
                     parquet_seconds=read_seconds)
        return dict(rows=documents, stats=stats)

    def _fill(self):
        while not self.closed and len(self.pending) < self.prefetch and self.submitted['block'] < self.total:
            shard, start, count = self._block(self.submitted['block'])
            row = self.submitted['row']
            if not 0 <= row < count:
                raise ValueError('Document cursor extends beyond its shuffled block')
            take = min(self.tokenize_rows, count - row)
            self.pending.append(self.executor.submit(self._load, shard, start + row, take))
            self.submitted['row'] += take
            if self.submitted['row'] == count:
                self.submitted['block'] += 1
                self.submitted['row'] = 0

    def peek(self):
        if self.position['block'] >= self.total:
            return None
        if self.rows is None or self.row_index == len(self.rows):
            result = self.pending.popleft().result()
            self.rows, self.row_index = result['rows'], 0
            for key, value in result['stats'].items():
                self.stats[key] += value
            self._fill()
        return self.rows[self.row_index]

    def advance(self):
        self.row_index += 1
        self.position['row'] += 1
        _, _, count = self._block(self.position['block'])
        if self.position['row'] == count:
            self.position['block'] += 1
            self.position['row'] = 0

    def _read_image(self, locator):
        if locator.get('video'):
            from data.video import prepare_video, prepare_qwen_inputs
            from data.sequence import video_sequence
            if not hasattr(self.image_local,'video_processor'):
                from data.vision import native_processor
                self.image_local.video_processor=native_processor(self.cfg.vision_assets)
            processor=self.image_local.video_processor
            rgb,clip=prepare_video(locator,locator['settings'])
            patches,grid,times=prepare_qwen_inputs(rgb,clip,processor,locator['settings'])
            grid=tuple(grid.tolist())
            return (rgb.permute(0,2,3,1).numpy(),patches,grid), dict(
                clip=dict(clip,qwen_timestamps=times),source_sha256=clip['source_sha256'],
                video_sequence=video_sequence(processor,grid,times))
        if locator.get('path'):
            import hashlib
            import io
            from pathlib import Path
            from PIL import Image, ImageOps
            body=Path(locator['path']).read_bytes(); sha=hashlib.sha256(body).hexdigest()
            if locator.get('sha256') and sha!=locator['sha256']:raise ValueError('Image content hash changed')
            locator['sha256']=sha
            with Image.open(io.BytesIO(body)) as image:
                pixels=np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
        else:
            pixels = self.image_reader.read(locator)
        if self.images_preprocessed:
            if max(pixels.shape[:2])>200*min(pixels.shape[:2]):
                with self.image_reader.lock:self.stats['aspect_padded_images']+=1
            if not hasattr(self.image_local, 'processor'):
                from data.vision import NativeImageProcessor
                self.image_local.processor = NativeImageProcessor(self.cfg, self.sequence_length)
            return self.image_local.processor(pixels)
        return pixels

    def prefetch_images(self, doc, element):
        if self.image_pool is None:
            return
        def candidates():
            yield from doc['elements'][element:]
            for row in (self.rows or [])[self.row_index + 1:]:
                yield from row['elements']
            for future in self.pending:
                if not future.done():
                    break
                for row in future.result()['rows']:
                    yield from row['elements']
        for item in candidates():
            if len(self.image_pending) >= self.image_limit:
                break
            if item['type'] == 'image' and item['image_ref'] not in self.image_pending:
                self.image_pending[item['image_ref']] = self.image_pool.submit(self._read_image, item['locator'])

    def image(self, doc, element):
        self.prefetch_images(doc, element)
        item = doc['elements'][element]
        begin = time.perf_counter()
        future = self.image_pending.pop(item['image_ref'], None)
        if future is None:
            # Repeated references can be consumed while the lookahead is full.
            # Give the demanded image one bounded extra slot instead of dropping it.
            future = self.image_pool.submit(self._read_image, item['locator'])
        pixels = future.result()
        if item['locator'].get('video'):
            pixels,metadata=pixels
            item.update(metadata)
        self.stats['image_wait_seconds'] += time.perf_counter() - begin
        return pixels

    def state_dict(self):
        return dict(signature=self.signature.copy(), position=self.position.copy())

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.image_pool is not None:
            self.image_pool.shutdown(wait=True, cancel_futures=True)
            self.image_reader.close()
        self.catalog.close()
