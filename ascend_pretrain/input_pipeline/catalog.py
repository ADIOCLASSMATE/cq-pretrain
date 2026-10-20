"""Immutable DCVLM catalog, locality-preserving shuffle and Arrow row caching."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

import pyarrow as pa
import pyarrow.parquet as pq


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def permute(index, size, seed, epoch=0):
    """Constant-memory bijection; worker completion order cannot change samples."""
    if size < 1 or not 0 <= index < size:
        raise ValueError('Invalid permutation domain/index')
    bits = max(1, (size.bit_length() + 1) // 2)
    mask = (1 << bits) - 1
    key = (int(seed) + 0x9e3779b9 * (epoch + 1)) & 0xffffffff
    while True:
        left, right = index >> bits, index & mask
        for round_id in range(6):
            z = (right + key + round_id * 0x85ebca6b) & 0xffffffff
            z = ((z ^ (z >> 16)) * 0x7feb352d) & 0xffffffff
            z = ((z ^ (z >> 15)) * 0x846ca68b) & 0xffffffff
            left, right = right, (left ^ z ^ (z >> 16)) & mask
        index = (left << bits) | right
        if index < size:
            return index


class Catalog:
    COLUMNS = ['sample_id', 'source_shard', 'segments', 'images', 'text_tokens', 'format']

    def __init__(self, root, source_root=None, block_rows=2048, cache_bytes=128 << 20):
        self.path = Path(root).resolve()
        manifest = json.loads((self.path / 'manifest.json').read_text())
        completion = json.loads((self.path / 'COMPLETE.json').read_text())
        validation = json.loads((self.path / 'reports/validation.json').read_text())
        if (manifest.get('schema_version') != 'dcvlm.pretrain.v1' or
                manifest.get('phase') != 'complete' or not completion.get('validation_passed') or
                not validation.get('passed')):
            raise RuntimeError('DCVLM dataset must be completely published and validated')
        catalog_bytes = (self.path / 'catalog.json').read_bytes()
        catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
        if catalog_sha != completion['catalog_sha256']:
            raise RuntimeError('DCVLM catalog hash differs from COMPLETE.json')
        self.shards = json.loads(catalog_bytes)
        self.source_root = Path(source_root or manifest['source_pool']).resolve()
        sources_bytes = (self.path / 'sources.json').read_bytes()
        self.source_sizes = {x['path']: int(x['bytes']) for x in json.loads(sources_bytes)['raw_shards']}
        self.tokenizer = self.path / 'tokenizer'
        lock = manifest['tokenizer_lock']
        for item in lock['files']:
            name = item.get('file', item.get('name'))
            if name not in ('tokenizer.json', 'tokenizer_config.json'):
                continue
            if hashlib.sha256((self.tokenizer / name).read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError(f'Pinned tokenizer changed: {name}')
        if block_rows < 1 or cache_bytes < 1:
            raise ValueError('Block rows and Arrow cache bytes must be positive')
        self.block_rows, self.cache_bytes = block_rows, cache_bytes
        self.ends = []
        total = 0
        for item in self.shards:
            path = Path(item['path'])
            if path.is_absolute() or '..' in path.parts or '://' in item['path']:
                raise ValueError('Expected local relative Parquet paths')
            total += (int(item['rows']) + block_rows - 1) // block_rows
            self.ends.append(total)
        identity = digest(dict(manifest=manifest, catalog_sha256=catalog_sha,
                               sources_sha256=hashlib.sha256(sources_bytes).hexdigest()))
        self.meta = dict(identity=identity, manifest=manifest, units=total, block_rows=block_rows,
                         documents=sum(int(x['rows']) for x in self.shards),
                         tokenizer=lock, development=bool(manifest.get('synthetic', False)))
        self.local = threading.local()
        self.caches, self.lock = [], threading.Lock()

    def check(self, allow_development=False):
        if self.meta['development'] and not allow_development:
            raise ValueError('Synthetic corpus requires --allow-development-data')
        return dict(identity=self.meta['identity'], complete=True,
                    development=self.meta['development'])

    def block(self, index):
        if not 0 <= index < self.meta['units']:
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        previous = self.ends[shard - 1] if shard else 0
        start = (index - previous) * self.block_rows
        return shard, start, min(self.block_rows, int(self.shards[shard]['rows']) - start)

    def _cache(self):
        if not hasattr(self.local, 'cache'):
            self.local.cache = dict(files=OrderedDict(), groups=OrderedDict(), bytes=0)
            with self.lock:
                self.caches.append(self.local.cache)
        return self.local.cache

    def read_rows(self, shard, start, count):
        """Reuse projected row groups across adjacent small tokenizer batches."""
        cache = self._cache()
        if shard not in cache['files']:
            item = self.shards[shard]
            path = self.path / item['path']
            if path.stat().st_size != item['bytes']:
                raise RuntimeError(f'Parquet size changed: {path}')
            parquet = pq.ParquetFile(path)
            ends, total = [], 0
            for i in range(parquet.num_row_groups):
                total += parquet.metadata.row_group(i).num_rows
                ends.append(total)
            if total != item['rows']:
                raise RuntimeError('Catalog/Parquet row count mismatch')
            cache['files'][shard] = (parquet, ends)
        cache['files'].move_to_end(shard)
        while len(cache['files']) > 4:
            cache['files'].popitem(last=False)[1][0].close()
        parquet, ends = cache['files'][shard]
        pieces = []
        stop = start + count
        while start < stop:
            group = bisect_right(ends, start)
            begin = ends[group - 1] if group else 0
            key = (shard, group)
            if key not in cache['groups']:
                table = parquet.read_row_group(group, columns=self.COLUMNS, use_threads=False)
                if table.nbytes <= self.cache_bytes:
                    while cache['groups'] and cache['bytes'] + table.nbytes > self.cache_bytes:
                        cache['bytes'] -= cache['groups'].popitem(last=False)[1].nbytes
                    cache['groups'][key] = table
                    cache['bytes'] += table.nbytes
            else:
                table = cache['groups'][key]
                cache['groups'].move_to_end(key)
            take = min(stop, ends[group]) - start
            pieces.append(table.slice(start - begin, take))
            start += take
        return pa.concat_tables(pieces).to_pylist()

    def close(self):
        for cache in self.caches:
            for parquet, _ in cache['files'].values():
                parquet.close()
            cache['files'].clear()
            cache['groups'].clear()
            cache['bytes'] = 0
