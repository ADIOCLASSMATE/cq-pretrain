"""Read caption groups without changing the immutable caption alternatives."""
import json
from pathlib import Path
import threading
from collections import OrderedDict

import numpy as np
import pyarrow as pa

from corpus_pipeline.production.common import file_sha
from data.catalog import Catalog
from data.captions import CONTRACT, apply_alternatives
from data import upstream


class GroupedCaptionCatalog(Catalog):
    COLUMNS = ['sample_id', 'record_json']
    SCHEMA = 'corpus.caption_groups.v1'
    SAMPLING_FIELD = 'caption_sampling'
    SAMPLING_CONTRACT = CONTRACT

    def __init__(self, root, source_root=None, block_rows=2048, cache_bytes=128 << 20):
        self.path = Path(root).resolve()
        manifest = json.loads((self.path / 'manifest.json').read_text())
        done = json.loads((self.path / 'COMPLETE.json').read_text())
        schema = manifest.get('schema_version')
        trajectories = schema == 'corpus.task_groups.v3'
        gated = schema == self.SCHEMA.removesuffix('v1')+'v2' or trajectories
        if (schema != self.SCHEMA and not gated or not done.get('complete') or
                done['manifest_sha256'] != file_sha(self.path / 'manifest.json') or
                manifest[self.SAMPLING_FIELD] != self.SAMPLING_CONTRACT):
            raise ValueError('Invalid caption-group publication')
        if gated and manifest.get('length_selection_contract') != 'full_document_A_qwen_8192_v1':
            raise ValueError('Unknown training length selection contract')
        from data.task_groups import STAGE1_TRAJECTORIES
        self.trajectory_sampling = manifest.get('trajectory_sampling')
        if trajectories and self.trajectory_sampling != STAGE1_TRAJECTORIES:
            raise ValueError('Unknown trajectory sampler')
        self.shards = manifest['files']
        self.upstream_sampling = manifest.get('upstream_sampling')
        if self.upstream_sampling not in (None, upstream.CONTRACT):
            raise ValueError('Unknown upstream sampling contract')
        self.block_rows, self.cache_bytes = block_rows, cache_bytes
        self.source_root, self.source_sizes = self.path, {}
        self.dataset_roots = source_root or (str(self.path / 'source_roots.json')
                                            if (self.path / 'source_roots.json').exists() else None)
        self.ends, total = [], 0
        for item in self.shards:
            path = Path(item['path'])
            if (path.is_absolute() or '..' in path.parts or min(block_rows, cache_bytes) < 1
                    or item['rows'] < int('length_selection' not in item)):
                raise ValueError('Invalid caption shard or reader limits')
            total += (item['rows'] + block_rows - 1) // block_rows
            self.ends.append(total)
        if sum(item['rows'] for item in self.shards) != manifest['records']:
            raise ValueError('Caption group count mismatch')
        self.meta = dict(identity=done['manifest_sha256'], units=total, documents=manifest['records'],
                         tokenizer={}, development=False, **{self.SAMPLING_FIELD:self.SAMPLING_CONTRACT})
        self.local, self.lock = threading.local(), threading.Lock()
        self.caches, self.checked = [], set()

    def verified_table(self, shard, start, count):
        item = self.shards[shard]
        if not 0 <= start < start + count <= item['rows']:
            raise IndexError('Selected group range is invalid')
        with self.lock:
            if shard not in self.checked:
                if file_sha(self.path / item['path']) != item['sha256']:
                    raise ValueError('Caption shard changed')
                self.checked.add(shard)
        if item.get('length_selection'):
            selection = self.length_selection(shard)
            dropped = selection['compressed_drop']
            if len(dropped):
                positions = np.arange(start, start + count)
                physical = positions + np.searchsorted(dropped, positions, side='right')
                begin = int(physical[0])
                return self.read_table(shard, begin, int(physical[-1])-begin+1).take(pa.array(physical-begin))
        return self.read_table(shard, start, count)

    def length_selection(self, shard):
        cache = self._cache().setdefault('length_selection', OrderedDict())
        if shard not in cache:
            from corpus_pipeline.postprocess.reconcile import validate_artifacts
            item = self.shards[shard]
            ref = item['length_selection']
            validate_artifacts(self.path, {'artifacts':[ref]})
            selection = json.loads((self.path/ref['path']).read_bytes())
            drop = np.asarray(selection['drop'], dtype=np.int64)
            if (len(drop) != item['native_rows'] - item['rows'] or
                    len(drop) and (drop[0] < 0 or drop[-1] >= item['native_rows'] or np.any(np.diff(drop) <= 0))):
                raise ValueError('Invalid training length selection')
            selection['compressed_drop'] = drop - np.arange(len(drop))
            cache[shard] = selection
        cache.move_to_end(shard)
        if len(cache) > 4:
            cache.popitem(last=False)
        return cache[shard]

    def read_rows(self, shard, start, count):
        records = [json.loads(row['record_json']) for row in self.verified_table(shard, start, count).to_pylist()]
        apply_alternatives(self, shard, records)
        upstream.attach(self, shard, records)
        if self.trajectory_sampling:
            for row in records:
                row['_trajectory_sampling'] = self.trajectory_sampling
        if self.shards[shard].get('length_selection'):
            gates = self.length_selection(shard)['mixed']
            for row in records:
                if row['sample_id'] in gates:
                    row.update(gates[row['sample_id']])
        if self.dataset_roots is not None:
            for row in records:
                for ref in row['images']:
                    ref['_dataset_roots'] = self.dataset_roots
        return records


class TaskGroupCatalog(GroupedCaptionCatalog):
    from data.task_groups import CONTRACT as SAMPLING_CONTRACT
    SCHEMA = 'corpus.task_groups.v1'
    SAMPLING_FIELD = 'task_sampling'


class ImageCaptionCatalog(GroupedCaptionCatalog):
    SCHEMA = 'corpus.openuni_image_captions.v1'
    COLUMNS = ['identity', 'members']

    def read_rows(self, shard, start, count):
        table = self.verified_table(shard, start, count)
        gates = self.length_selection(shard)['mixed'] if self.shards[shard].get('length_selection') else {}
        rows = []
        for identity, cell in zip(table['identity'].to_pylist(), table['members']):
            members = cell.values
            if identity in gates:
                eligible = members.field('eligible').to_numpy(zero_copy_only=False).copy()
                eligible[gates[identity]] = False
                members = pa.StructArray.from_arrays(
                    [pa.array(eligible) if name == 'eligible' else members.field(name) for name in members.type.names],
                    fields=list(members.type))
            rows.append(dict(sample_id='openuni-image:'+identity, format='image_caption_group', source_shard='',
                             caption_members=members, _dataset_roots=self.dataset_roots))
        return rows
