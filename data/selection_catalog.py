"""Locality-preserving reads of an audited combined-corpus selection."""
from collections import OrderedDict
import json
from pathlib import Path
import threading

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.catalog import Catalog
from data.captions import CONTRACT, apply_alternatives
from data import upstream
from data.loss_permissions import image_loss_allowed
from corpus_pipeline.postprocess.reconcile import file_sha, validate_artifacts
from corpus_pipeline.production.portable_refs import logical_ref

SCHEMA = 'corpus.training_selection.v1'
ROUTES = {2: 'I2T', 4: 'T2I', 5: 'T2I', 6: 'MIXED', 8: 'TEXT'}


class SelectionCatalog(Catalog):
    COLUMNS = ['sample_id', 'source_shard', 'segments', 'images', 'format']

    def __init__(self, root, source_root=None, block_rows=2048, cache_bytes=128 << 20):
        self.path = Path(root).resolve()
        completion = json.loads((self.path / 'COMPLETE.json').read_text())
        if not completion.get('complete') or file_sha(self.path / 'manifest.json') != completion['manifest_sha256']:
            raise ValueError('Training selection manifest/completion mismatch')
        manifest = json.loads((self.path / 'manifest.json').read_text())
        if manifest.get('schema_version') != SCHEMA or min(block_rows, cache_bytes) < 1:
            raise ValueError('Invalid training selection or reader limits')
        validate_artifacts(self.path, {'artifacts': [manifest['mask']] + manifest['tokenizer_files']})
        self.mask = np.memmap(self.path / manifest['mask']['path'], dtype='uint8', mode='r')
        self.shards = manifest['files']
        self.upstream_sampling = manifest.get('upstream_sampling')
        if self.upstream_sampling not in (None, upstream.CONTRACT):
            raise ValueError('Unknown upstream sampling contract')
        self.block_rows, self.cache_bytes = block_rows, cache_bytes
        self.source_root, self.source_sizes = self.path, {}
        self.dataset_roots = str(source_root or self.path / 'source_roots.json')
        self.ends, total = [], 0
        for item in self.shards:
            path = Path(item['path'])
            if path.is_absolute() or '..' in path.parts or '://' in item['path']:
                raise ValueError('Native metadata must be inside the training release')
            total += (item['rows'] + block_rows - 1) // block_rows
            self.ends.append(total)
        if sum(x['rows'] for x in self.shards) != manifest['records']:
            raise ValueError('Training selection count mismatch')
        self.meta = dict(identity=completion['manifest_sha256'], manifest=manifest, units=total,
                         block_rows=block_rows, documents=manifest['records'],
                         tokenizer=manifest['tokenizer'], development=False)
        if any('text_alternatives' in item for item in self.shards):
            if manifest.get('caption_sampling') != CONTRACT:
                raise ValueError('Selection text alternatives require the caption sampling contract')
            self.meta['caption_sampling'] = CONTRACT
        self.local, self.caches, self.lock = threading.local(), [], threading.Lock()
        self.checked, self.policies, self.approved = set(), {}, {}
        for source in manifest['decode_policies']:
            validate_artifacts(self.path, {'artifacts': [source['policies']]})
            key = source['source_key']
            self.policies[key] = {r['asset_id']: r for r in pq.read_table(self.path / source['policies']['path']).to_pylist()}
            if 'adult_review' in source:
                validate_artifacts(self.path, {'artifacts': [source['adult_review']]})
                review = json.loads((self.path / source['adult_review']['path']).read_text())
                if review.get('allow_understanding') is not True or review.get('allow_image_loss') is not False:
                    raise ValueError('Bound DCVLM understanding permission changed')
                self.approved[key] = set(review['asset_ids'])

    def read_rows(self, shard, start, count):
        item = self.shards[shard]
        if not 0 <= start < start + count <= item['rows']:
            raise IndexError('Selected native row range is invalid')
        with self.lock:
            if shard not in self.checked:
                if file_sha(self.path / item['path']) != item['sha256']:
                    raise ValueError('Native metadata checksum changed')
                self.checked.add(shard)
        cache = self._cache()
        selected = cache.setdefault('selection', OrderedDict())
        if shard not in selected:
            mask = self.mask[item['mask_offset']:item['mask_offset'] + item['native_rows']]
            if not np.isin(mask, [0, *ROUTES]).all():
                raise ValueError('Unknown task/loss selection code')
            selected[shard] = np.flatnonzero(mask)
            if len(selected[shard]) != item['rows']:
                raise ValueError('Selected/native row count mismatch')
        selected.move_to_end(shard)
        while len(selected) > 4:
            selected.popitem(last=False)
        indices = selected[shard][start:start + count]
        begin, end = int(indices[0]), int(indices[-1]) + 1
        rows = self.read_table(shard, begin, end - begin).take(pa.array(indices - begin)).to_pylist()
        policies = self.policies.get(item['source_key'], {})
        from corpus_pipeline.dedup.dcvlm_loss_policy_reader import SAFETY_DENY_SHA256
        for index, row in zip(indices, rows):
            code = int(self.mask[item['mask_offset'] + index])
            native_permission = image_loss_allowed(row)
            row.update(sample_id=item['source_key'] + ':' + row['sample_id'],
                       text_tokens=None, direction=ROUTES[code], image_loss_allowed=bool(code & 1) and native_permission)
            if not row['images'] and row['format'] == 'caption':
                raise ValueError('Unbound image caption in training selection')
            for slot, image in enumerate(row['images']):
                image = logical_ref(image)
                if image['sha256'] in SAFETY_DENY_SHA256 and image['sha256'] not in self.approved.get(item['source_key'], set()):
                    raise ValueError('Image requires bound adult review')
                allowed = image_loss_allowed(row, image)
                image.update(_dataset_roots=self.dataset_roots, image_loss_allowed=allowed)
                if image['sha256'] in policies:
                    image['_decode_policy'] = policies[image['sha256']]
                row['images'][slot] = image
        apply_alternatives(self, shard, rows)
        upstream.attach(self, shard, rows)
        if ref := item.get('length_captions'):
            gates = cache.setdefault('length_captions', OrderedDict())
            if shard not in gates:
                validate_artifacts(self.path, {'artifacts':[ref]})
                gates[shard] = json.loads((self.path/ref['path']).read_bytes())
            gates.move_to_end(shard)
            if len(gates) > 4:
                gates.popitem(last=False)
            for row in rows:
                if row['sample_id'] in gates[shard]:
                    row['stage1_captions'] = gates[shard][row['sample_id']]
        return rows

    def close(self):
        super().close()
        self.mask._mmap.close()
