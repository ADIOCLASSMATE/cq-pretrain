"""Resumable SSCD candidate retrieval; never merge assets or delete task records.

Stages: catalog -> train -> build -> merge -> calibrate -> search.  One shared
IVF/PQ codebook spans every SHA bucket.  Only the small coarse quantizer may use
HNSW; the corpus does not.  The packed catalog preserves original FP32 values;
FAISS sees normalized working copies and candidates are reranked by exact cosine.

Example (parameters are starting points, NOT a recall-qualified production preset):
  python -m corpus_pipeline.postprocess.sscd_index estimate --count 90000000
  python -m corpus_pipeline.postprocess.sscd_index catalog --input RECONCILED --output INDEX
  python -m corpus_pipeline.postprocess.sscd_index train --output INDEX --train-count 1000000
  python -m corpus_pipeline.postprocess.sscd_index build --output INDEX
  python -m corpus_pipeline.postprocess.sscd_index merge --output INDEX
  python -m corpus_pipeline.postprocess.sscd_index calibrate --output INDEX --threshold YOUR_REVIEW_THRESHOLD
  python -m corpus_pipeline.postprocess.sscd_index search --output INDEX --calibration INDEX/calibration/HASH/COMPLETE.json

Calibration compares held-out real queries against streaming exact search over
the ENTIRE bound catalog. Small validation scopes cannot qualify a full catalog.
Its thresholds select review candidates, not a MOSS threshold or deletion rule.
All stages lock one output directory and commit immutable, hashed checkpoints.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np

DIMENSION = 512
VECTOR_BYTES = DIMENSION * 4
HEX64 = re.compile(r'^[0-9a-f]{64}$')
VERSION = 'corpus.sscd_index.v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def stamp(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino]


def artifact(path, root):
    path = Path(path)
    return {'path': str(path.relative_to(root)), 'sha256': file_sha(path), 'size_bytes': path.stat().st_size}


def bound_path(root, item, verify=True):
    path = (root / item['path']).resolve()
    require(path.is_relative_to(root), 'Artifact escaped root')
    require(path.is_file() and path.stat().st_size == item['size_bytes'], 'Artifact missing or size changed')
    if verify:
        require(file_sha(path) == item['sha256'], 'Artifact SHA mismatch')
    return path


def verify_receipt(root, receipt):
    for item in receipt['artifacts']:
        bound_path(root, item)


@contextmanager
def locked(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.LOCK').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def bind(directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / 'IDENTITY.json'
    if p.exists():
        require(read_json(p) == identity, 'Stage input/config/code changed; use a new output directory')
    else:
        require(not any(directory.iterdir()), 'Nonempty stage without identity')
        atomic_json(p, identity)


def stage_identity(stage, **kwargs):
    return {'schema': VERSION, 'stage': stage, 'code_sha256': file_sha(__file__), **kwargs}


def normalized(vectors):
    result = np.array(vectors, dtype=np.float32, order='C', copy=True)
    norms = np.linalg.norm(result, axis=1)
    require(np.isfinite(result).all() and np.isfinite(norms).all() and (norms > 0).all(), 'Invalid vector')
    result /= norms[:, None]
    return result


@dataclass(frozen=True)
class IndexConfig:
    nlist: int = 16384
    pq_m: int = 64
    pq_bits: int = 8
    coarse_hnsw: int = 32
    niter: int = 20
    seed: int = 20261023
    threads: int = 16

    def validate(self):
        require(type(self.nlist) is int and self.nlist >= 1, 'nlist must be positive')
        require(self.pq_m > 0 and DIMENSION % self.pq_m == 0, 'pq_m must divide 512')
        require(4 <= self.pq_bits <= 8, 'PQ bits must be between 4 and 8')
        require(self.coarse_hnsw == 0 or 4 <= self.coarse_hnsw <= 64, 'Invalid coarse HNSW')
        require(1 <= self.threads <= 64 and 1 <= self.niter <= 100, 'Invalid CPU/training bounds')

    @property
    def factory(self):
        suffix = f'_HNSW{self.coarse_hnsw}' if self.coarse_hnsw else ''
        return f'IVF{self.nlist}{suffix},PQ{self.pq_m}x{self.pq_bits}'


def estimate(count=90_000_000, config=IndexConfig(), nprobe=32):
    config.validate(); require(count > 0 and 1 <= nprobe <= config.nlist, 'Invalid estimate inputs')
    code_bytes = math.ceil(config.pq_m * config.pq_bits / 8)
    scans = count * min(1., nprobe / config.nlist)
    return {'count': count, 'dimension': DIMENSION, 'config': asdict(config),
            'packed_fp32_bytes': count * VECTOR_BYTES, 'asset_id_binary_bytes': count * 32,
            'ivfpq_codes_and_ids_bytes': count * (code_bytes + 8),
            'coarse_centroid_bytes': config.nlist * VECTOR_BYTES,
            'optional_precomputed_table_bytes': config.nlist * config.pq_m * (2 ** config.pq_bits) * 4,
            'minimum_recommended_train_count': 39 * max(config.nlist, 2 ** config.pq_bits),
            'mean_codes_scanned_per_query': scans,
            'all_query_code_bytes_lower_bound': scans * count * code_bytes,
            'notes': ['Estimates omit imbalance, metadata, temporary checkpoints and filesystem overhead.',
                      'Search estimate is code traffic, not measured wall time; CPU training/search need a pilot.',
                      'Catalog duplicates raw vectors on disk to avoid random opens of original task files.',
                      'Only candidate review edges are emitted. No merge, record deletion, or MOSS threshold.']}


def reconciliation_inputs(input_root, allow_bounded=False):
    root = Path(input_root).resolve(); complete_path = root / 'COMPLETE.json'
    receipt = read_json(complete_path)
    require(receipt.get('complete') is True and receipt.get('schema') == 'corpus.asset_reconciliation.v1',
            'Reconciliation is not complete')
    require(receipt.get('scope') == 'full_run' or (allow_bounded and receipt.get('scope') == 'bounded_validation'),
            'Bounded input requires explicit --allow-bounded')
    entries = sorted(receipt['buckets'], key=lambda x: x['bucket'])
    require([x['bucket'] for x in entries] == [f'{i:02x}' for i in range(256)], 'Expected all 256 asset buckets')
    files = []
    for entry in entries:
        path = (root / entry['receipt_path']).resolve()
        require(path == root / 'buckets' / entry['bucket'] / 'COMPLETE.json', 'Unexpected bucket receipt path')
        require(file_sha(path) == entry['receipt_sha256'], 'Bucket receipt changed')
        bucket = read_json(path)
        require(bucket.get('schema') == 'corpus.asset_reconciliation.bucket.v1'
                and bucket.get('bucket') == entry['bucket'], 'Invalid bucket receipt')
        matches = [x for x in bucket['artifacts'] if Path(x['path']).name == 'assets.parquet']
        require(len(matches) == 1, 'Missing or ambiguous assets parquet')
        assets = bound_path(root, matches[0])
        require(assets == root / 'buckets' / entry['bucket'] / 'assets.parquet', 'Unexpected asset path')
        files.append((entry['bucket'], assets, matches[0]))
    return receipt, file_sha(complete_path), files


class OriginalVectors:
    """Verify each original immutable feature file once; bounded mmap descriptors."""
    def __init__(self, verification_dir, max_open=32768, verify_workers=8):
        import resource
        require(type(max_open) is int and max_open >= 1, 'Invalid original mmap cache bound')
        require(type(verify_workers) is int and 1 <= verify_workers <= 32, 'Invalid verification worker count')
        self.directory = verification_dir; self.directory.mkdir(parents=True, exist_ok=True)
        limits = [max_open]
        fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if fd_limit != resource.RLIM_INFINITY:
            # NumPy may transiently need another descriptor while opening a map.
            fd_used = len(list(Path('/proc/self/fd').iterdir())) if Path('/proc/self/fd').exists() else 64
            limits.append((fd_limit - fd_used - 256) // 2)
        vm_limit = Path('/proc/sys/vm/max_map_count')
        process_maps = Path('/proc/self/maps')
        if vm_limit.exists() and process_maps.exists():
            with process_maps.open() as stream:
                vm_used = sum(1 for _ in stream)
            # Leave room for libraries, output mappings and transient allocation.
            limits.append((int(vm_limit.read_text()) - vm_used - 4096) // 2)
        self.max_open = min(limits)
        require(self.max_open >= 1, 'Insufficient descriptor or virtual-memory-map budget')
        self.verify_workers = verify_workers
        self.verified = {}; self.maps = OrderedDict()

    def _verify_file(self, value):
        """Only file-local evidence is written here; shared caches stay single-writer."""
        path, expected, current_stamp = value
        key = digest({'path': path, 'sha256': expected[0], 'size': expected[1]})
        evidence = self.directory / (key + '.json')
        receipt = {'path': path, 'expected': list(expected), 'stamp': current_stamp}
        valid_cache = evidence.exists() and read_json(evidence) == receipt
        if not valid_cache:
            require(file_sha(path) == expected[0] and stamp(path) == current_stamp, 'Original SSCD SHA mismatch')
            atomic_json(evidence, receipt)
        return path, expected, current_stamp

    def read(self, rows):
        vectors = np.zeros((len(rows), DIMENSION), np.float32)
        groups = {}
        for i, row in enumerate(rows):
            require(row['sscd_dimension'] == DIMENSION and row['sscd_dtype'] == 'float32'
                    and row['sscd_normalized'] is True, 'Unsupported SSCD vector schema')
            path = str(Path(row['sscd_file']).resolve())
            expected = (row['sscd_file_sha256'], row['sscd_file_size_bytes'])
            require(HEX64.fullmatch(expected[0]) and type(expected[1]) is int and expected[1] % VECTOR_BYTES == 0,
                    'Invalid original feature identity')
            groups.setdefault((path, expected), []).append((i, row['sscd_byte_offset']))
        checked, unseen = {}, []
        for path, expected in groups:
            current_stamp = stamp(path)
            require(current_stamp[0] == expected[1], 'Original feature size mismatch')
            require(path not in checked or checked[path] == (expected, current_stamp),
                    'Conflicting original feature identity within batch')
            checked[path] = (expected, current_stamp)
            cached = self.verified.get(path)
            require(cached is None or cached[:2] == (expected, current_stamp), 'Feature identity changed within catalog')
            if cached is None:
                unseen.append((path, expected, current_stamp))
        if unseen:
            with ThreadPoolExecutor(max_workers=min(self.verify_workers, len(unseen))) as executor:
                for path, expected, current_stamp in executor.map(self._verify_file, unseen):
                    self.verified[path] = (expected, current_stamp)
        for (path, expected), members in groups.items():
            # Parallel verification can take time; an early-verified file must
            # still be unchanged when the main thread actually reads its map.
            require(stamp(path) == checked[path][1], 'Original feature changed after verification')
            if path not in self.maps:
                self.maps[path] = np.memmap(path, dtype='<f4', mode='r', shape=(expected[1] // VECTOR_BYTES, DIMENSION))
            self.maps.move_to_end(path)
            while len(self.maps) > self.max_open:
                self.maps.popitem(last=False)
            offsets = [offset for _, offset in members]
            require(all(type(o) is int and o >= 0 and o % VECTOR_BYTES == 0 and o + VECTOR_BYTES <= expected[1]
                        for o in offsets), 'Invalid vector offset')
            vectors[[i for i, _ in members]] = self.maps[path][np.asarray(offsets, np.int64) // VECTOR_BYTES]
        return vectors


def catalog(input_root, output, *, block_rows=250_000, allow_bounded=False, max_new_blocks=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    require(1 <= block_rows <= 1_000_000, 'Invalid block_rows')
    output = Path(output).resolve()
    require(output != Path(input_root).resolve() and not output.is_relative_to(Path(input_root).resolve() / 'buckets'),
            'Index output overlaps reconciliation artifacts')
    with locked(output):
        upstream, upstream_hash, files = reconciliation_inputs(input_root, allow_bounded)
        directory = output / 'catalog'
        identity = stage_identity('catalog', input=str(Path(input_root).resolve()), input_sha256=upstream_hash,
                                  scope=upstream['scope'], block_rows=block_rows)
        bind(directory, identity)
        reader = OriginalVectors(directory / 'verified_originals')
        blocks, next_id, total_input, skipped, invalid, new_count = [], 0, 0, 0, 0, 0
        for bucket, path, source_artifact in files:
            previous = None
            for number, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=block_rows)):
                name = f'{bucket}-{number:06d}'; destination = directory / 'blocks' / name
                metadata_path = destination / 'COMPLETE.json'
                rows = batch.to_pylist()
                for row in rows:
                    asset_id = row['asset_id']
                    require(isinstance(asset_id, str) and HEX64.fullmatch(asset_id) and asset_id[:2] == bucket,
                            'Invalid SHA bucket asset identity')
                    require(previous is None or asset_id > previous, 'Assets must be unique and strictly sorted per bucket')
                    previous = asset_id
                binding = {'input_asset_sha256': source_artifact['sha256'], 'batch': number,
                           'input_rows': len(rows), 'start_id': next_id, 'identity_sha256': digest(identity)}
                if metadata_path.exists():
                    receipt = read_json(metadata_path)
                    require(receipt['binding'] == binding and receipt.get('complete') is True, 'Catalog checkpoint mismatch')
                    verify_receipt(output, receipt)
                else:
                    if max_new_blocks is not None and new_count >= max_new_blocks:
                        return {'complete': False, 'blocks': len(blocks), 'vectors': next_id}
                    selected = [r for r in rows if r.get('sscd_usable') is True and
                                (r.get('global_status') not in ('rejected_source', 'pending_integrity') or
                                 (r.get('has_i2t_usage') is True and r.get('i2t_integrity_pending') is False))
                                and not r.get('pixel_sha256_conflict', False)]
                    vectors = reader.read(selected) if selected else np.empty((0, DIMENSION), np.float32)
                    norms = np.linalg.norm(vectors, axis=1)
                    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (np.abs(norms - 1.) <= .01)
                    exceptions = [{'asset_id': row['asset_id'], 'reason': 'invalid_or_nonunit_sscd_vector'}
                                  for row, ok in zip(selected, valid) if not ok]
                    selected = [row for row, ok in zip(selected, valid) if ok]
                    vectors = vectors[valid]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = Path(tempfile.mkdtemp(prefix='.' + name, dir=destination.parent))
                    try:
                        vectors.astype('<f4', copy=False).tofile(temporary / 'vectors.f32')
                        with (temporary / 'asset_ids.bin').open('wb') as stream:
                            for row in selected:
                                stream.write(bytes.fromhex(row['asset_id']))
                        mapping = [{'vector_id': next_id + i, 'asset_id': r['asset_id'], 'pixel_sha256': r.get('pixel_sha256'),
                                    'source': r.get('source'), 'global_status': r.get('global_status'),
                                    'sscd_file': r['sscd_file'], 'sscd_byte_offset': r['sscd_byte_offset'],
                                    'occurrence_count': r.get('occurrence_count'), 'source_count': r.get('source_count')}
                                   for i, r in enumerate(selected)]
                        mapping_schema = pa.schema([('vector_id', pa.int64()), ('asset_id', pa.string()),
                            ('pixel_sha256', pa.string()), ('source', pa.string()), ('global_status', pa.string()),
                            ('sscd_file', pa.string()), ('sscd_byte_offset', pa.int64()),
                            ('occurrence_count', pa.int64()), ('source_count', pa.int64())])
                        pq.write_table(pa.Table.from_pylist(mapping, schema=mapping_schema), temporary / 'mapping.parquet')
                        pq.write_table(pa.Table.from_pylist(exceptions, schema=pa.schema([('asset_id', pa.string()), ('reason', pa.string())])),
                                       temporary / 'exceptions.parquet')
                        arts = []
                        for file in sorted(temporary.iterdir()):
                            item = artifact(file, temporary); item['path'] = str((destination / file.name).relative_to(output)); arts.append(item)
                        receipt = {'complete': True, 'binding': binding, 'start_id': next_id, 'count': len(selected),
                                   'skipped': len(rows) - len(selected) - len(exceptions), 'invalid': len(exceptions), 'artifacts': arts}
                        atomic_json(temporary / 'COMPLETE.json', receipt)
                        if destination.exists():
                            require(not any(destination.iterdir()), 'Uncommitted block directory exists')
                            destination.rmdir()
                        os.replace(temporary, destination)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                    new_count += 1
                blocks.append({'receipt_path': str(metadata_path.relative_to(output)), 'receipt_sha256': file_sha(metadata_path),
                               'start_id': next_id, 'count': receipt['count']})
                next_id += receipt['count']; total_input += len(rows); skipped += receipt['skipped']; invalid += receipt['invalid']
        require(total_input == upstream['counts']['assets'], 'Asset count differs from reconciliation')
        complete = {'schema': VERSION, 'stage': 'catalog', 'complete': True, 'scope': upstream['scope'],
                    'identity_sha256': digest(identity), 'input_sha256': upstream_hash, 'vectors': next_id,
                    'input_assets': total_input, 'skipped_assets': skipped, 'invalid_vectors': invalid, 'blocks': blocks}
        atomic_json(directory / 'COMPLETE.json', complete)
        return complete


class Catalog:
    def __init__(self, output, *, allow_bounded=False, max_open=1024):
        import resource
        self.root = Path(output).resolve(); self.path = self.root / 'catalog' / 'COMPLETE.json'
        self.receipt = read_json(self.path); self.sha256 = file_sha(self.path)
        require(self.receipt.get('complete') is True and self.receipt.get('stage') == 'catalog', 'Catalog incomplete')
        require(self.receipt['scope'] == 'full_run' or allow_bounded, 'Bounded catalog needs --allow-bounded')
        fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        # Each block uses two read-only mmaps; pages remain reclaimable by the OS.
        self.blocks = []; self.maps = OrderedDict(); self.verified = set()
        self.max_open = max(1, min(max_open, (fd_limit - 64) // 2 if fd_limit > 0 else max_open))
        end = 0
        for entry in self.receipt['blocks']:
            p = (self.root / entry['receipt_path']).resolve()
            require(p.is_relative_to(self.root) and file_sha(p) == entry['receipt_sha256'], 'Catalog block receipt changed')
            value = read_json(p)
            require(value['start_id'] == entry['start_id'] == end and value['count'] == entry['count'], 'Noncontiguous catalog')
            end += value['count']
            if value['count']:
                self.blocks.append(value)
        require(end == self.receipt['vectors'] and end > 0, 'Empty or invalid catalog')
        self.count = end
        self.starts = np.asarray([b['start_id'] for b in self.blocks], np.int64)

    def block(self, number):
        if number not in self.maps:
            item = self.blocks[number]
            arts = {Path(a['path']).name: a for a in item['artifacts']}
            verify = number not in self.verified
            v = bound_path(self.root, arts['vectors.f32'], verify)
            ids = bound_path(self.root, arts['asset_ids.bin'], verify)
            require(v.stat().st_size == item['count'] * VECTOR_BYTES and ids.stat().st_size == item['count'] * 32,
                    'Packed catalog layout mismatch')
            self.maps[number] = (np.memmap(v, dtype='<f4', mode='r', shape=(item['count'], DIMENSION)),
                                 np.memmap(ids, dtype='u1', mode='r', shape=(item['count'], 32)))
            self.verified.add(number)
        self.maps.move_to_end(number)
        while len(self.maps) > self.max_open:
            self.maps.popitem(last=False)
        return self.maps[number]

    def fetch(self, ids, *, asset_ids=False):
        ids = np.asarray(ids, dtype=np.int64)
        require(ids.ndim == 1 and ((ids >= 0) & (ids < self.count)).all(), 'Vector IDs outside catalog')
        result = np.empty((len(ids), DIMENSION), np.float32)
        sha = [None] * len(ids) if asset_ids else None
        block_numbers = np.searchsorted(self.starts, ids, side='right') - 1
        for number in np.unique(block_numbers):
            positions = np.flatnonzero(block_numbers == number); offsets = ids[positions] - self.starts[number]
            vectors, names = self.block(int(number)); result[positions] = vectors[offsets]
            if asset_ids:
                for position, offset in zip(positions, offsets):
                    sha[position] = names[offset].tobytes().hex()
        return (result, sha) if asset_ids else result

    def iter_vectors(self, batch_size=65536):
        for number, block in enumerate(self.blocks):
            vectors, _ = self.block(number)
            for start in range(0, block['count'], batch_size):
                yield block['start_id'] + start, vectors[start:start + batch_size]


def write_index(path, index):
    import faiss
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    faiss.write_index(index, str(temporary))
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train(output, *, config=IndexConfig(), train_count=1_000_000, allow_bounded=False):
    import faiss
    from threadpoolctl import threadpool_limits
    config.validate(); output = Path(output).resolve()
    with locked(output), threadpool_limits(limits=config.threads):
        data = Catalog(output, allow_bounded=allow_bounded)
        count = min(train_count, data.count)
        require(count >= 39 * max(config.nlist, 2 ** config.pq_bits), 'Too few training vectors for requested IVF/PQ config')
        require(count < data.count, 'Reserve held-out vectors for recall calibration')
        identity = stage_identity('train', catalog_sha256=data.sha256, config=asdict(config), train_count=count)
        directory = output / 'train'; bind(directory, identity)
        path = directory / 'COMPLETE.json'
        if path.exists():
            receipt = read_json(path); verify_receipt(output, receipt); return receipt
        ids = np.sort(np.random.default_rng(config.seed).choice(data.count, count, replace=False)).astype(np.int64)
        vectors = normalized(data.fetch(ids))
        faiss.omp_set_num_threads(config.threads)
        index = faiss.index_factory(DIMENSION, config.factory, faiss.METRIC_L2)
        index.cp.seed = config.seed; index.cp.niter = config.niter
        index.pq.cp.seed = config.seed; index.pq.cp.niter = config.niter
        index.use_precomputed_table = -1
        started = time.monotonic(); index.train(vectors)
        write_index(directory / 'trained.faiss', index); ids.tofile(directory / 'sample_ids.i64')
        receipt = {'complete': True, 'identity': identity, 'seconds': time.monotonic() - started,
                   'artifacts': [artifact(directory / name, output) for name in ('trained.faiss', 'sample_ids.i64')]}
        atomic_json(path, receipt); return receipt


def training(output, data):
    path = output / 'train' / 'COMPLETE.json'; receipt = read_json(path)
    require(receipt.get('complete') is True and receipt['identity']['catalog_sha256'] == data.sha256, 'Training/catalog mismatch')
    verify_receipt(output, receipt)
    return receipt, file_sha(path)


def build(output, *, allow_bounded=False, max_new_shards=None):
    import faiss
    output = Path(output).resolve()
    with locked(output):
        data = Catalog(output, allow_bounded=allow_bounded); trained, train_hash = training(output, data)
        cfg = trained['identity']['config']; faiss.omp_set_num_threads(cfg['threads'])
        directory = output / 'shards'
        identity = stage_identity('build', catalog_sha256=data.sha256, train_sha256=train_hash)
        bind(directory, identity)
        trained_path = output / 'train' / 'trained.faiss'; new_count = 0; shards = []
        for number, block in enumerate(data.blocks):
            receipt_path = directory / f'{number:06d}.json'
            binding = {'identity_sha256': digest(identity), 'block': number, 'start_id': block['start_id'], 'count': block['count']}
            if receipt_path.exists():
                receipt = read_json(receipt_path); require(receipt['binding'] == binding, 'Shard binding mismatch')
                verify_receipt(output, receipt)
            else:
                if max_new_shards is not None and new_count >= max_new_shards:
                    return {'complete': False, 'shards': len(shards)}
                index = faiss.read_index(str(trained_path)); index.use_precomputed_table = -1
                vectors = normalized(data.block(number)[0])
                ids = np.arange(block['start_id'], block['start_id'] + block['count'], dtype=np.int64)
                started = time.monotonic(); index.add_with_ids(vectors, ids)
                index_path = directory / f'{number:06d}.faiss'; write_index(index_path, index)
                receipt = {'complete': True, 'binding': binding, 'seconds': time.monotonic() - started,
                           'artifacts': [artifact(index_path, output)]}
                atomic_json(receipt_path, receipt); new_count += 1
            shards.append({'receipt_path': str(receipt_path.relative_to(output)), 'receipt_sha256': file_sha(receipt_path)})
        receipt = {'complete': True, 'identity': identity, 'count': data.count, 'shards': shards}
        atomic_json(directory / 'COMPLETE.json', receipt); return receipt


def merge(output, *, allow_bounded=False, checkpoint_shards=16, max_new_shards=None):
    import faiss
    require(checkpoint_shards >= 1, 'Invalid merge checkpoint interval')
    output = Path(output).resolve()
    with locked(output):
        data = Catalog(output, allow_bounded=allow_bounded); trained, _ = training(output, data)
        faiss.omp_set_num_threads(trained['identity']['config']['threads'])
        source = output / 'shards' / 'COMPLETE.json'; shards = read_json(source)
        require(shards.get('complete') is True and shards['identity']['catalog_sha256'] == data.sha256, 'Shards incomplete')
        directory = output / 'merged'; identity = stage_identity('merge', catalog_sha256=data.sha256, shards_sha256=file_sha(source))
        bind(directory, identity)
        if (directory / 'COMPLETE.json').exists():
            result = read_json(directory / 'COMPLETE.json'); verify_receipt(output, result); return result
        latest = directory / 'LATEST.json'; position = 0
        if latest.exists():
            saved = read_json(latest); require(saved['identity_sha256'] == digest(identity), 'Merge checkpoint identity mismatch')
            verify_receipt(output, saved); position = saved['shards_merged']
            index = faiss.read_index(str(output / saved['artifacts'][0]['path']))
        else:
            index = faiss.read_index(str(output / 'train' / 'trained.faiss'))
        new_count = 0
        for number in range(position, len(shards['shards'])):
            if max_new_shards is not None and new_count >= max_new_shards:
                break
            entry = shards['shards'][number]; path = output / entry['receipt_path']
            require(file_sha(path) == entry['receipt_sha256'], 'Shard receipt changed')
            receipt = read_json(path); verify_receipt(output, receipt)
            other = faiss.read_index(str(output / receipt['artifacts'][0]['path']))
            require(other.ntotal == data.blocks[number]['count'], 'Shard vector count mismatch')
            index.merge_from(other, 0); position = number + 1; new_count += 1
            if position % checkpoint_shards == 0 or position == len(shards['shards']) or new_count == max_new_shards:
                checkpoint = directory / f'merge-{position:06d}.faiss'; write_index(checkpoint, index)
                saved = {'complete': position == len(shards['shards']), 'identity_sha256': digest(identity),
                         'shards_merged': position, 'count': int(index.ntotal), 'artifacts': [artifact(checkpoint, output)]}
                atomic_json(latest, saved)
                # The latest pointer commits first; only obsolete local snapshots are removed.
                for old in directory.glob('merge-*.faiss'):
                    if old != checkpoint:
                        old.unlink()
        if position == len(shards['shards']):
            require(index.ntotal == data.count, 'Merged index coverage mismatch')
            result = {**read_json(latest), 'identity': identity, 'scope': data.receipt['scope']}
            atomic_json(directory / 'COMPLETE.json', result); return result
        return {'complete': False, 'shards_merged': position, 'count': int(index.ntotal)}


def global_index(output, data, nprobe, ef_search):
    import faiss
    path = output / 'merged' / 'COMPLETE.json'; receipt = read_json(path)
    require(receipt.get('complete') is True and receipt['identity']['catalog_sha256'] == data.sha256,
            'Global index not complete for this catalog')
    verify_receipt(output, receipt)
    index = faiss.read_index(str(output / receipt['artifacts'][0]['path']))
    require(index.ntotal == data.count and 1 <= nprobe <= index.nlist, 'Invalid global index/nprobe')
    index.nprobe = nprobe; index.use_precomputed_table = -1
    quantizer = faiss.downcast_index(index.quantizer)
    if hasattr(quantizer, 'hnsw'):
        require(ef_search >= nprobe, 'Coarse efSearch must cover nprobe')
        quantizer.hnsw.efSearch = ef_search
    return index, file_sha(path)


def candidates(data, index, query_ids, candidate_k):
    """Return both directional hits; never drop reverse-only ANN neighbors."""
    queries = normalized(data.fetch(query_ids))
    result = []
    # Fetch candidate vectors once per bounded group, grouped by packed block.
    # This avoids reopening original files or fetching repeated neighbors for
    # every query.  At most max(32768, candidate_k) vectors enter one rerank.
    group_size = max(1, 32768 // candidate_k)
    for start in range(0, len(query_ids), group_size):
        stop = min(len(query_ids), start + group_size)
        _, ids = index.search(queries[start:stop], min(candidate_k + 1, data.count))
        found_rows = []
        for row, query_id in enumerate(query_ids[start:stop]):
            found = ids[row]; found_rows.append(found[(found >= 0) & (found != query_id)][:candidate_k])
        unique = np.unique(np.concatenate(found_rows))
        raw = normalized(data.fetch(unique))
        for row, found in enumerate(found_rows):
            scores = raw[np.searchsorted(unique, found)] @ queries[start + row]
            order = np.lexsort((found, -scores))
            result.append((found[order], scores[order]))
    return result


def exact_neighbors(data, query_ids, *, top_k, threshold, ann_ids, batch_size=32768):
    queries = normalized(data.fetch(query_ids))
    best_ids = np.full((len(query_ids), top_k), -1, np.int64)
    best_scores = np.full((len(query_ids), top_k), -np.inf, np.float32)
    positive_count = np.zeros(len(query_ids), np.int64); positive_hits = np.zeros(len(query_ids), np.int64)
    ann_sets = [set(map(int, row)) for row in ann_ids]
    for offset, raw in data.iter_vectors(batch_size):
        values = queries @ normalized(raw).T
        for row, query_id in enumerate(query_ids):
            if offset <= query_id < offset + len(raw):
                values[row, query_id - offset] = -np.inf
            positive = np.flatnonzero(values[row] >= threshold) + offset
            positive_count[row] += len(positive)
            positive_hits[row] += sum(int(i) in ann_sets[row] for i in positive)
            take = min(top_k, len(raw))
            local = np.argpartition(values[row], -take)[-take:]
            merged_ids = np.concatenate((best_ids[row], local + offset))
            merged_scores = np.concatenate((best_scores[row], values[row, local]))
            order = np.lexsort((merged_ids, -merged_scores))[:top_k]
            best_ids[row] = merged_ids[order]; best_scores[row] = merged_scores[order]
    return best_ids, best_scores, positive_count, positive_hits


def calibrate(output, *, threshold, query_count=128, top_k=10, candidate_k=256, nprobe=32,
              ef_search=128, min_recall=.95, min_positive_recall=.99, seed=8273, threads=16, allow_bounded=False):
    import faiss
    from threadpoolctl import threadpool_limits
    require(-1 <= threshold <= 1 and 0 < min_recall <= 1 and 0 < min_positive_recall <= 1, 'Invalid calibration thresholds')
    require(1 <= query_count <= 4096 and 1 <= top_k <= candidate_k <= 65536 and 1 <= threads <= 64, 'Invalid calibration bounds')
    output = Path(output).resolve()
    with locked(output), threadpool_limits(limits=threads):
        data = Catalog(output, allow_bounded=allow_bounded); trained, _ = training(output, data)
        faiss.omp_set_num_threads(threads); index, index_hash = global_index(output, data, nprobe, ef_search)
        train_ids = set(np.fromfile(output / 'train' / 'sample_ids.i64', dtype=np.int64).tolist())
        require(top_k < data.count and query_count <= data.count - len(train_ids), 'Insufficient held-out queries')
        rng = np.random.default_rng(seed); selected = set(); ordered = []
        while len(ordered) < query_count:
            value = int(rng.integers(0, data.count))
            if value not in selected and value not in train_ids:
                selected.add(value); ordered.append(value)
        query_ids = np.asarray(ordered, np.int64)
        parameters = dict(threshold=threshold, query_count=query_count, top_k=top_k, candidate_k=candidate_k,
                          nprobe=nprobe, ef_search=ef_search, min_recall=min_recall,
                          min_positive_recall=min_positive_recall, seed=seed, threads=threads)
        identity = stage_identity('calibrate', catalog_sha256=data.sha256, index_sha256=index_hash, parameters=parameters)
        directory = output / 'calibration' / digest(identity)[:16]; bind(directory, identity)
        if (directory / 'COMPLETE.json').exists():
            result = read_json(directory / 'COMPLETE.json'); verify_receipt(output, result); return result
        started = time.monotonic(); approx = candidates(data, index, query_ids, candidate_k)
        ann_seconds = time.monotonic() - started; started = time.monotonic()
        exact_ids, exact_scores, positives, hits = exact_neighbors(data, query_ids, top_k=top_k, threshold=threshold,
                                                                ann_ids=[row[0] for row in approx])
        exact_seconds = time.monotonic() - started
        recall = np.asarray([len(set(map(int, truth)) & set(map(int, found[0]))) / top_k
                             for truth, found in zip(exact_ids, approx)])
        positive_recall = float(hits.sum() / positives.sum()) if positives.sum() else None
        passed = bool(recall.mean() >= min_recall and positive_recall is not None and positive_recall >= min_positive_recall)
        np.savez(directory / 'evidence.npz', query_ids=query_ids, exact_ids=exact_ids, exact_scores=exact_scores,
                 recall_at_k=recall, threshold_positive_count=positives, threshold_positive_hits=hits,
                 ann_ids=np.asarray([np.pad(row[0], (0, candidate_k - len(row[0])), constant_values=-1) for row in approx]))
        result = {'complete': True, 'passed': passed, 'identity': identity, 'scope': data.receipt['scope'],
                  'qualified_for_full_run': passed and data.receipt['scope'] == 'full_run',
                  'recall_at_k': float(recall.mean()), 'minimum_query_recall': float(recall.min()),
                  'threshold_positive_recall': positive_recall, 'threshold_positive_count': int(positives.sum()),
                  'ann_seconds': ann_seconds, 'exact_seconds': exact_seconds, 'exact_reference_vectors': data.count,
                  'interpretation': 'Held-out queries against the entire catalog; sampled evidence, not a guarantee of zero missed pairs.',
                  'artifacts': [artifact(directory / 'evidence.npz', output)]}
        atomic_json(directory / 'COMPLETE.json', result)
        atomic_json(output / 'calibration' / 'LATEST.json', {'receipt_path': str((directory / 'COMPLETE.json').relative_to(output)),
                                                           'receipt_sha256': file_sha(directory / 'COMPLETE.json')})
        return result


def search(output, calibration_path, *, query_start=0, query_end=None, batch_size=64, allow_bounded=False,
           max_new_batches=None):
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_limits
    output = Path(output).resolve()
    require(1 <= batch_size <= 1024, 'Invalid search batch size')
    with locked(output):
        data = Catalog(output, allow_bounded=allow_bounded)
        qualification = read_json(calibration_path)
        require(qualification.get('complete') is True and qualification.get('passed') is True, 'Recall calibration did not pass')
        require(qualification['identity']['catalog_sha256'] == data.sha256 and qualification['scope'] == data.receipt['scope'],
                'Calibration catalog/scope mismatch')
        require(qualification['identity']['code_sha256'] == file_sha(__file__), 'Calibration code changed')
        require(data.receipt['scope'] != 'full_run' or qualification.get('qualified_for_full_run') is True, 'Full-run calibration required')
        verify_receipt(output, qualification)
        parameters = qualification['identity']['parameters']; faiss.omp_set_num_threads(parameters['threads'])
        index, index_hash = global_index(output, data, parameters['nprobe'], parameters['ef_search'])
        require(qualification['identity']['index_sha256'] == index_hash, 'Calibration index mismatch')
        end = data.count if query_end is None else query_end
        require(0 <= query_start < end <= data.count, 'Invalid query range')
        identity = stage_identity('search', catalog_sha256=data.sha256, index_sha256=index_hash,
                                  calibration_sha256=file_sha(calibration_path), query_start=query_start,
                                  query_end=end, batch_size=batch_size)
        directory = output / 'search' / digest(identity)[:16]; bind(directory, identity)
        schema = pa.schema([('edge_id', pa.string()), ('vector_id_a', pa.int64()), ('vector_id_b', pa.int64()),
                            ('asset_id_a', pa.string()), ('asset_id_b', pa.string()), ('exact_cosine', pa.float32()),
                            ('decision', pa.string()), ('automatic_merge', pa.bool_())])
        batches, edge_count, new_count = [], 0, 0
        with threadpool_limits(limits=parameters['threads']):
            for start in range(query_start, end, batch_size):
                stop = min(end, start + batch_size); path = directory / f'{start:012d}.json'
                binding = {'identity_sha256': digest(identity), 'start': start, 'stop': stop}
                if path.exists():
                    receipt = read_json(path); require(receipt['binding'] == binding, 'Search checkpoint mismatch')
                    verify_receipt(output, receipt)
                else:
                    if max_new_batches is not None and new_count >= max_new_batches:
                        return {'complete': False, 'batches': len(batches), 'edge_rows': edge_count}
                    query_ids = np.arange(start, stop, dtype=np.int64)
                    found = candidates(data, index, query_ids, parameters['candidate_k']); edges = {}
                    for qid, (ids, scores) in zip(query_ids, found):
                        for other, score in zip(ids, scores):
                            if score >= parameters['threshold']:
                                a, b = sorted((int(qid), int(other))); edges[(a, b)] = max(float(score), edges.get((a, b), -math.inf))
                    unique = sorted({i for pair in edges for i in pair})
                    _, asset_ids = data.fetch(unique, asset_ids=True); names = dict(zip(unique, asset_ids))
                    rows = [{'edge_id': hashlib.sha256(bytes.fromhex(names[a]) + bytes.fromhex(names[b])).hexdigest(),
                             'vector_id_a': a, 'vector_id_b': b, 'asset_id_a': names[a], 'asset_id_b': names[b],
                             'exact_cosine': score, 'decision': 'requires_pair_review', 'automatic_merge': False}
                            for (a, b), score in sorted(edges.items())]
                    parquet = directory / f'{start:012d}.parquet'; temporary = parquet.with_suffix('.tmp')
                    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary); os.replace(temporary, parquet)
                    receipt = {'complete': True, 'binding': binding, 'edges': len(rows), 'artifacts': [artifact(parquet, output)]}
                    atomic_json(path, receipt); new_count += 1
                batches.append({'path': str(path.relative_to(output)), 'sha256': file_sha(path)})
                edge_count += receipt['edges']
        result = {'complete': True, 'identity': identity, 'scope': data.receipt['scope'], 'query_count': end - query_start,
                  'edge_rows': edge_count, 'batches': batches, 'automatic_merge': False, 'training_records_deleted': 0,
                  'edge_semantics': 'Canonical pair IDs; duplicate edge_id across query batches is permitted. Union by edge_id before review.',
                  'threshold_semantics': 'Recall-qualified candidate review threshold, not an automatic deletion threshold.'}
        atomic_json(directory / 'COMPLETE.json', result); return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('estimate', 'catalog', 'train', 'build', 'merge', 'calibrate', 'search'))
    parser.add_argument('--input'); parser.add_argument('--output'); parser.add_argument('--allow-bounded', action='store_true')
    parser.add_argument('--count', type=int, default=90_000_000); parser.add_argument('--block-rows', type=int, default=250_000)
    parser.add_argument('--nlist', type=int, default=16384); parser.add_argument('--pq-m', type=int, default=64)
    parser.add_argument('--pq-bits', type=int, default=8); parser.add_argument('--coarse-hnsw', type=int, default=32)
    parser.add_argument('--niter', type=int, default=20); parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--train-count', type=int, default=1_000_000); parser.add_argument('--checkpoint-shards', type=int, default=16)
    parser.add_argument('--nprobe', type=int, default=32); parser.add_argument('--ef-search', type=int, default=128)
    parser.add_argument('--threshold', type=float); parser.add_argument('--query-count', type=int, default=128)
    parser.add_argument('--top-k', type=int, default=10); parser.add_argument('--candidate-k', type=int, default=256)
    parser.add_argument('--min-recall', type=float, default=.95); parser.add_argument('--min-positive-recall', type=float, default=.99)
    parser.add_argument('--calibration'); parser.add_argument('--query-start', type=int, default=0)
    parser.add_argument('--query-end', type=int); parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-new', type=int, help='Bound new catalog blocks/build shards/merge shards/search batches for a resumable pilot')
    args = parser.parse_args(argv)
    config = IndexConfig(nlist=args.nlist, pq_m=args.pq_m, pq_bits=args.pq_bits,
                         coarse_hnsw=args.coarse_hnsw, niter=args.niter, threads=args.threads)
    if args.stage == 'estimate':
        result = estimate(args.count, config, args.nprobe)
    else:
        require(args.output is not None, '--output required')
        common = {'allow_bounded': args.allow_bounded}
        if args.stage == 'catalog':
            require(args.input is not None, '--input required')
            result = catalog(args.input, args.output, block_rows=args.block_rows, max_new_blocks=args.max_new, **common)
        elif args.stage == 'train':
            result = train(args.output, config=config, train_count=args.train_count, **common)
        elif args.stage == 'build':
            result = build(args.output, max_new_shards=args.max_new, **common)
        elif args.stage == 'merge':
            result = merge(args.output, checkpoint_shards=args.checkpoint_shards, max_new_shards=args.max_new, **common)
        elif args.stage == 'calibrate':
            require(args.threshold is not None, '--threshold explicitly required; no universal SSCD threshold')
            result = calibrate(args.output, threshold=args.threshold, query_count=args.query_count, top_k=args.top_k,
                               candidate_k=args.candidate_k, nprobe=args.nprobe, ef_search=args.ef_search,
                               min_recall=args.min_recall, min_positive_recall=args.min_positive_recall, threads=args.threads, **common)
        else:
            require(args.calibration is not None, '--calibration required')
            result = search(args.output, args.calibration, query_start=args.query_start, query_end=args.query_end,
                            batch_size=args.batch_size, max_new_batches=args.max_new, **common)
    print(canonical(result))


if __name__ == '__main__':
    main()
