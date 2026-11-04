"""Parallel, resumable SHA-bucket packing with immutable legacy-block adoption.

prepare freezes a task plan; independent worker processes steal per-block locks.
finalize publishes the existing sscd_index.Catalog contract after full coverage.
Neither original features nor previously committed catalog blocks are modified.
"""
from __future__ import annotations

import argparse
import binascii
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import tempfile
import threading
import time
import uuid
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import sscd_index as s

VERSION = 'corpus.sscd_parallel_catalog.v1'
COLUMNS = ['asset_id', 'pixel_sha256', 'source', 'global_status', 'sscd_usable',
           'pixel_sha256_conflict', 'sscd_file', 'sscd_file_sha256', 'sscd_file_size_bytes',
           'sscd_byte_offset', 'sscd_dimension', 'sscd_dtype', 'sscd_normalized',
           'occurrence_count', 'source_count']
MAPPING_SCHEMA = pa.schema([('vector_id', pa.int64()), ('asset_id', pa.string()),
    ('pixel_sha256', pa.string()), ('source', pa.string()), ('global_status', pa.string()),
    ('sscd_file', pa.string()), ('sscd_byte_offset', pa.int64()),
    ('occurrence_count', pa.int64()), ('source_count', pa.int64())])
EXCEPTION_SCHEMA = pa.schema([('asset_id', pa.string()), ('reason', pa.string())])
require = s.require
_LOCAL_LOCKS = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def runtime():
    return {'schema': VERSION, 'code_sha256': s.file_sha(__file__),
            'helper_code_sha256': s.file_sha(s.__file__), 'numpy': np.__version__, 'pyarrow': pa.__version__}


def bound(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': s.file_sha(path), 'size_bytes': path.stat().st_size}


def check_bound(value):
    path = Path(value['path'])
    require(path.is_file() and path.stat().st_size == value['size_bytes']
            and s.file_sha(path) == value['sha256'], 'Bound artifact changed: ' + str(path))
    return path


@contextmanager
def file_lock(path, *, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # This shared filesystem exposes cross-host POSIX record locks, but flock
    # was experimentally local-only. lockf is process-scoped, so a local mutex
    # also protects separate threads and prevents another fd close releasing it.
    key = os.path.abspath(path)
    with _LOCAL_LOCKS_GUARD:
        local = _LOCAL_LOCKS.setdefault(key, threading.Lock())
    if not local.acquire(blocking=blocking):
        yield False
        return
    try:
        with path.open('a+') as stream:
            acquired = False
            try:
                try:
                    fcntl.lockf(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                    acquired = True
                except OSError as exc:
                    if exc.errno not in {errno.EAGAIN, errno.EACCES}:
                        raise
                yield acquired
            finally:
                if acquired:
                    fcntl.lockf(stream, fcntl.LOCK_UN)
    finally:
        local.release()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(temporary, destination, receipt):
    for path in temporary.iterdir():
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    s.atomic_json(temporary / 'COMPLETE.json', receipt)
    _fsync_directory(temporary)
    require(not destination.exists(), 'Immutable destination already exists')
    os.rename(temporary, destination)
    _fsync_directory(destination.parent)


def _link_or_copy(source, destination, expected):
    before = s.stamp(source)
    try:
        os.link(source, destination)
        require(os.path.samestat(os.stat(source), os.stat(destination)), 'Hardlink inode differs')
        method = 'hardlink'
    except OSError:
        shutil.copyfile(source, destination)
        require(s.file_sha(destination) == expected, 'Copied artifact SHA differs')
        method = 'copy'
    require(s.stamp(source) == before, 'Source changed during adoption')
    return method


def prepare(input_root, output, *, adopt=None, block_rows=250_000, allow_bounded=False):
    input_root = Path(input_root).resolve(strict=True); output = Path(output).resolve()
    require(1 <= block_rows <= 1_000_000, 'Invalid block rows')
    require(output != input_root and not output.is_relative_to(input_root / 'buckets'), 'Output overlaps inputs')
    legacy = Path(adopt).resolve(strict=True) if adopt else None
    require(legacy is None or output != legacy, 'New catalog cannot modify legacy output')
    area = output / 'catalog_parallel'; area.mkdir(parents=True, exist_ok=True)
    with file_lock(area / 'prepare.lock'):
        completion = bound(input_root / 'COMPLETE.json'); upstream = s.read_json(completion['path'])
        require(upstream.get('complete') is True and upstream.get('schema') == 'corpus.asset_reconciliation.v1',
                'Reconciliation incomplete')
        require(upstream.get('scope') == 'full_run' or (allow_bounded and upstream.get('scope') == 'bounded_validation'),
                'Bounded reconciliation requires allow_bounded')
        identity = {**runtime(), 'input_completion': completion, 'input': str(input_root),
                    'scope': upstream['scope'], 'block_rows': block_rows,
                    'adopt': str(legacy) if legacy else None}
        plan_path = area / 'PLAN.json'
        if plan_path.exists():
            plan = s.read_json(plan_path)
            require(plan['identity'] == identity, 'Plan input/config/runtime changed; use a new output')
            return plan
        legacy_identity = None
        if legacy:
            legacy_identity = s.read_json(legacy / 'catalog' / 'IDENTITY.json')
            require(legacy_identity['input_sha256'] == completion['sha256']
                    and legacy_identity['input'] == str(input_root)
                    and legacy_identity['block_rows'] == block_rows
                    and legacy_identity['scope'] == upstream['scope'], 'Legacy catalog input differs')
        entries = sorted(upstream['buckets'], key=lambda e: e['bucket'])
        require([e['bucket'] for e in entries] == [f'{i:02x}' for i in range(256)], 'Expected all 256 buckets')
        tasks, buckets, total = [], [], 0
        for entry in entries:
            bucket = entry['bucket']; receipt_path = input_root / entry['receipt_path']
            require(receipt_path.resolve() == input_root / 'buckets' / bucket / 'COMPLETE.json'
                    and s.file_sha(receipt_path) == entry['receipt_sha256'], 'Bucket receipt binding differs')
            receipt = s.read_json(receipt_path)
            matches = [a for a in receipt['artifacts'] if Path(a['path']).name == 'assets.parquet']
            require(len(matches) == 1, 'Ambiguous source parquet')
            source = (input_root / matches[0]['path']).resolve(strict=True)
            require(source == input_root / 'buckets' / bucket / 'assets.parquet', 'Source parquet path differs')
            source_binding = {'path': str(source), 'sha256': matches[0]['sha256'],
                              'size_bytes': matches[0]['size_bytes'], 'stamp': s.stamp(source)}
            require(source_binding['stamp'][0] == source_binding['size_bytes'], 'Source parquet size differs')
            count = pq.ParquetFile(source).metadata.num_rows
            buckets.append({'bucket': bucket, 'rows': count, 'source': source_binding})
            for number, start in enumerate(range(0, count, block_rows)):
                task_id = f'{bucket}-{number:06d}'
                task = {'task_id': task_id, 'bucket': bucket, 'batch': number, 'row_start': start,
                        'input_rows': min(block_rows, count - start), 'source': source_binding, 'adoption': None}
                if legacy:
                    old_path = legacy / 'catalog' / 'blocks' / task_id / 'COMPLETE.json'
                    if old_path.exists():
                        old = s.read_json(old_path); b = old['binding']
                        require(old.get('complete') is True and b['identity_sha256'] == s.digest(legacy_identity)
                                and b['input_asset_sha256'] == source_binding['sha256']
                                and b['batch'] == number and b['input_rows'] == task['input_rows']
                                and b['start_id'] == old['start_id']
                                and old['count'] + old['skipped'] + old['invalid'] == task['input_rows'],
                                'Legacy block binding differs')
                        task['adoption'] = {'root': str(legacy), 'receipt': bound(old_path),
                                            'catalog_identity': bound(legacy / 'catalog' / 'IDENTITY.json')}
                tasks.append(task)
            total += count
        require(total == upstream['counts']['assets'], 'Parquet row coverage differs from reconciliation')
        plan = {'schema': VERSION, 'complete': True, 'identity': identity, 'tasks': tasks, 'buckets': buckets,
                'input_assets': total, 'task_count': len(tasks),
                'expected_vectors': upstream['counts'].get('sscd_usable_assets', total),
                'adopted_task_count': sum(t['adoption'] is not None for t in tasks),
                'legacy_verified_dirs': [str(legacy / 'catalog' / 'verified_originals')] if legacy else []}
        s.atomic_json(plan_path, plan)
        return plan


def load_plan(output):
    output = Path(output).resolve(strict=True)
    path = output / 'catalog_parallel' / 'PLAN.json'; plan = s.read_json(path)
    require(plan.get('complete') is True and plan.get('schema') == VERSION, 'Missing complete plan')
    require(all(plan['identity'].get(k) == v for k, v in runtime().items()), 'Parallel runtime changed')
    check_bound(plan['identity']['input_completion'])
    return output, plan, s.file_sha(path)


def _verify_immutable(path, expected_sha, expected_size, area, legacy_dirs=()):
    """Shared exact-stamp SHA receipts, with file locks to avoid repeated hashing."""
    path = str(path); current = s.stamp(path)
    require(current[0] == expected_size, 'Immutable source size changed')
    expected = [expected_sha, expected_size]
    key = s.digest({'path': path, 'sha256': expected_sha, 'size': expected_size})
    receipt = {'path': path, 'expected': expected, 'stamp': current}
    destination = area / 'verified_originals' / (key + '.json')
    with file_lock(area / 'verification_locks' / (key + '.lock')):
        if destination.exists() and s.read_json(destination) == receipt:
            require(s.stamp(path) == current, 'Source changed while checking cached verification')
            return current
        evidence = None
        for directory in legacy_dirs:
            old = Path(directory) / (key + '.json')
            if old.exists() and s.read_json(old) == receipt:
                evidence = {'kind': 'reused_exact_legacy_verification', 'receipt': bound(old)}
                break
        if evidence is None:
            require(s.file_sha(path) == expected_sha, 'Original source SHA mismatch')
            evidence = {'kind': 'full_file_sha256_verified'}
        require(s.stamp(path) == current, 'Immutable source changed during verification')
        s.atomic_json(area / 'verification_evidence' / (key + '.json'), {**receipt, **evidence})
        s.atomic_json(destination, receipt)
        return current


def read_slice(task):
    """Read projected overlapping row groups, never all metadata into Python rows."""
    parquet = pq.ParquetFile(task['source']['path'])
    start, end = task['row_start'], task['row_start'] + task['input_rows']
    selected, row_start, cursor = [], None, 0
    for index in range(parquet.metadata.num_row_groups):
        stop = cursor + parquet.metadata.row_group(index).num_rows
        if cursor < end and stop > start:
            selected.append(index)
            if row_start is None:
                row_start = cursor
        cursor = stop
    require(selected and end <= cursor, 'Parquet slice outside source')
    columns = [name for name in [*COLUMNS, 'has_i2t_usage', 'i2t_integrity_pending']
               if name in parquet.schema_arrow.names]
    table = parquet.read_row_groups(selected, columns=columns, use_threads=True).slice(start - row_start, end - start)
    require(len(table) == task['input_rows'], 'Incomplete source slice')
    if 'pixel_sha256_conflict' not in table.column_names:
        table = table.append_column('pixel_sha256_conflict', pa.array(np.zeros(len(table), dtype=bool)))
    require(set(COLUMNS) <= set(table.column_names), 'Required source columns missing')
    ids = table['asset_id'].combine_chunks()
    require(ids.null_count == 0 and pc.all(pc.match_substring_regex(ids, '^[0-9a-f]{64}$')).as_py()
            and pc.all(pc.starts_with(ids, task['bucket'])).as_py(), 'Invalid asset SHA bucket')
    if len(ids) > 1:
        require(pc.all(pc.less(ids.slice(0, len(ids) - 1), ids.slice(1))).as_py(), 'Source assets not strictly sorted')
    bounds = {'first_asset_id': ids[0].as_py(), 'last_asset_id': ids[-1].as_py()}
    status_allowed = pc.invert(pc.is_in(table['global_status'], value_set=pa.array(['rejected_source', 'pending_integrity'])))
    if {'has_i2t_usage', 'i2t_integrity_pending'} <= set(table.column_names):
        i2t_allowed = pc.fill_null(pc.and_kleene(pc.equal(table['has_i2t_usage'], True),
                                               pc.equal(table['i2t_integrity_pending'], False)), False)
        status_allowed = pc.or_kleene(status_allowed, i2t_allowed)
    eligible = pc.fill_null(pc.and_kleene(pc.equal(table['sscd_usable'], True),
        pc.and_kleene(status_allowed, pc.invert(pc.fill_null(table['pixel_sha256_conflict'], False)))), False)
    return table.filter(eligible), bounds


class ParallelVectors:
    """File-group parallel reads with lane-owned bounded descriptor/mmap caches."""
    def __init__(self, area, *, workers=8, max_open=32768, legacy_dirs=(), method='pread'):
        require(1 <= workers <= 32 and method in {'pread', 'mmap'}, 'Invalid reader options')
        self.area = Path(area); self.workers = workers; self.legacy_dirs = legacy_dirs; self.method = method
        budget = s.OriginalVectors(self.area / 'verified_originals', max_open=max_open, verify_workers=workers).max_open
        self.per_lane = max(1, budget // workers)
        require(budget >= workers, 'Descriptor budget too small for reader workers')
        self.maps = [OrderedDict() for _ in range(workers)]
        self.verified = [{} for _ in range(workers)]; self.resolved = {}
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.stats = {'files_hashed_or_shared_checked': 0, 'pread_calls': 0, 'pread_bytes': 0}
        self.stats_lock = threading.Lock()

    def close(self):
        self.executor.shutdown(wait=True)
        for cache in self.maps:
            for value in cache.values():
                if self.method == 'pread':
                    os.close(value)
            cache.clear()

    def _read_lane(self, lane, groups, output):
        cache, verified = self.maps[lane], self.verified[lane]
        calls, read_bytes, checked = 0, 0, 0
        for path, expected, positions, offsets in groups:
            before = s.stamp(path)
            require(before[0] == expected[1], 'Original feature size differs')
            if path not in verified:
                state = _verify_immutable(path, expected[0], expected[1], self.area, self.legacy_dirs)
                verified[path] = (expected, state); checked += 1
            require(verified[path] == (expected, before), 'Original feature changed within worker')
            require(np.all(offsets >= 0) and np.all(offsets % s.VECTOR_BYTES == 0)
                    and np.all(offsets <= expected[1] - s.VECTOR_BYTES), 'Invalid original vector offsets')
            if path not in cache:
                if self.method == 'pread':
                    cache[path] = os.open(path, os.O_RDONLY)
                    if hasattr(os, 'posix_fadvise'):
                        try:
                            os.posix_fadvise(cache[path], 0, 0, os.POSIX_FADV_RANDOM)
                        except OSError:
                            pass  # An optional kernel hint cannot invalidate correct reads.
                else:
                    cache[path] = np.memmap(path, dtype='<f4', mode='r', shape=(expected[1] // s.VECTOR_BYTES, s.DIMENSION))
                while len(cache) > self.per_lane:
                    _, stale = cache.popitem(last=False)
                    if self.method == 'pread':
                        os.close(stale)
            cache.move_to_end(path)
            if self.method == 'mmap':
                output[positions] = cache[path][offsets // s.VECTOR_BYTES]
            else:
                order = np.argsort(offsets, kind='stable'); ordered = offsets[order]
                begin = 0
                while begin < len(order):
                    stop = begin + 1; low = int(ordered[begin]); high = low + s.VECTOR_BYTES
                    while stop < len(order) and int(ordered[stop]) <= high + 16384 and int(ordered[stop]) + s.VECTOR_BYTES - low <= 1 << 20:
                        high = max(high, int(ordered[stop]) + s.VECTOR_BYTES); stop += 1
                    payload = os.pread(cache[path], high - low, low)
                    require(len(payload) == high - low, 'Short original feature read')
                    view = np.frombuffer(payload, dtype='<f4').reshape(-1, s.DIMENSION)
                    local = ((ordered[begin:stop] - low) // s.VECTOR_BYTES).astype(np.int64)
                    output[positions[order[begin:stop]]] = view[local]
                    calls += 1; read_bytes += len(payload); begin = stop
            require(s.stamp(path) == before, 'Original feature changed during vector read')
        with self.stats_lock:
            self.stats['files_hashed_or_shared_checked'] += checked
            self.stats['pread_calls'] += calls; self.stats['pread_bytes'] += read_bytes

    def read(self, table):
        output = np.empty((len(table), s.DIMENSION), dtype=np.float32)
        if not len(table):
            return output
        for name, expected in [('sscd_dimension', 512), ('sscd_dtype', 'float32'), ('sscd_normalized', True)]:
            require(table[name].null_count == 0 and pc.all(pc.equal(table[name], expected)).as_py(), 'Unsupported vector schema')
        paths = pc.dictionary_encode(table['sscd_file'].combine_chunks())
        hashes = pc.dictionary_encode(table['sscd_file_sha256'].combine_chunks())
        require(pa.types.is_integer(table['sscd_file_size_bytes'].type)
                and pa.types.is_integer(table['sscd_byte_offset'].type), 'Vector file size/offset must be integer')
        require(paths.null_count == hashes.null_count == table['sscd_file_size_bytes'].null_count
                == table['sscd_byte_offset'].null_count == 0, 'Missing vector identity')
        names = paths.dictionary.to_pylist(); hash_names = hashes.dictionary.to_pylist()
        codes = paths.indices.to_numpy(); hash_codes = hashes.indices.to_numpy()
        sizes = table['sscd_file_size_bytes'].combine_chunks().to_numpy(); offsets = table['sscd_byte_offset'].combine_chunks().to_numpy()
        order = np.argsort(codes, kind='stable'); boundaries = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(order)]
        lanes = [[] for _ in range(self.workers)]
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            positions = order[low:high]; first = positions[0]; raw_path = names[int(codes[first])]
            if raw_path not in self.resolved:
                self.resolved[raw_path] = str(Path(raw_path).resolve(strict=True))
            path = self.resolved[raw_path]; sha = hash_names[int(hash_codes[first])]; size = int(sizes[first])
            require(s.HEX64.fullmatch(sha) and size % s.VECTOR_BYTES == 0
                    and np.all(hash_codes[positions] == hash_codes[first]) and np.all(sizes[positions] == size),
                    'Conflicting feature-file identity')
            lanes[zlib.crc32(path.encode()) % self.workers].append((path, (sha, size), positions, offsets[positions]))
        futures = [self.executor.submit(self._read_lane, lane, groups, output) for lane, groups in enumerate(lanes)]
        for future in futures:
            future.result()
        return output


def _task_binding(plan_sha, task):
    return {'plan_sha256': plan_sha, 'task_sha256': s.digest(task), 'task_id': task['task_id']}


def _task_path(output, task):
    return output / 'catalog_parallel' / 'tasks' / task['task_id']


def _task_receipt(output, plan_sha, task, *, verify=True):
    path = _task_path(output, task) / 'COMPLETE.json'
    value = s.read_json(path)
    require(value.get('complete') is True and value['binding'] == _task_binding(plan_sha, task), 'Task receipt identity differs')
    require(value['count'] + value['skipped'] + value['invalid'] == task['input_rows'], 'Task accounting mismatch')
    if verify:
        s.verify_receipt(output, value)
    return value


def pack_task(output, plan, plan_sha, task, reader):
    destination = _task_path(output, task)
    if (destination / 'COMPLETE.json').exists():
        return _task_receipt(output, plan_sha, task)
    started = time.monotonic(); phases = {}; stats_before = dict(reader.stats)
    area = output / 'catalog_parallel'; source = task['source']
    require(s.stamp(source['path']) == source['stamp'], 'Frozen source parquet changed')
    _verify_immutable(source['path'], source['sha256'], source['size_bytes'], area)
    phases['source_verification_seconds'] = time.monotonic() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + task['task_id'] + '-', dir=destination.parent))
    adoption = None
    try:
        if task['adoption']:
            adopted = task['adoption']; old_path = check_bound(adopted['receipt']); check_bound(adopted['catalog_identity'])
            old = s.read_json(old_path); legacy = Path(adopted['root'])
            s.verify_receipt(legacy, old)
            arts = {Path(a['path']).name: a for a in old['artifacts']}
            require(set(arts) == {'vectors.f32', 'asset_ids.bin', 'mapping.parquet', 'exceptions.parquet'}, 'Legacy artifact set differs')
            mapping = pq.read_table(legacy / arts['mapping.parquet']['path'])
            count = old['count']; require(len(mapping) == count, 'Legacy mapping count differs')
            require(np.array_equal(mapping['vector_id'].to_numpy(), np.arange(old['start_id'], old['start_id'] + count)),
                    'Legacy vector IDs are not contiguous')
            mapping = mapping.set_column(mapping.schema.get_field_index('vector_id'), 'vector_id', pa.array(np.arange(count, dtype=np.int64)))
            pq.write_table(mapping.cast(MAPPING_SCHEMA), temporary / 'mapping.parquet')
            methods = {}
            for name in ('vectors.f32', 'asset_ids.bin', 'exceptions.parquet'):
                methods[name] = _link_or_copy(legacy / arts[name]['path'], temporary / name, arts[name]['sha256'])
            bounds_table, bounds = read_slice(task)
            del bounds_table
            skipped, invalid = old['skipped'], old['invalid']
            adoption = {**adopted, 'methods': methods, 'original_start_id': old['start_id']}
        else:
            phase = time.monotonic()
            selected, bounds = read_slice(task)
            phases['projected_parquet_read_seconds'] = time.monotonic() - phase
            phase = time.monotonic()
            vectors = reader.read(selected)
            phases['vector_fetch_seconds'] = time.monotonic() - phase
            norms = np.linalg.norm(vectors, axis=1)
            valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (np.abs(norms - 1.) <= .01)
            invalid = int((~valid).sum()); skipped = task['input_rows'] - len(selected)
            exceptions = selected.filter(pa.array(~valid))['asset_id']
            pq.write_table(pa.Table.from_arrays([exceptions, pa.array(['invalid_or_nonunit_sscd_vector'] * invalid)], schema=EXCEPTION_SCHEMA),
                           temporary / 'exceptions.parquet')
            selected = selected.filter(pa.array(valid)); vectors = vectors[valid]; count = len(selected)
            vectors.astype('<f4', copy=False).tofile(temporary / 'vectors.f32')
            (temporary / 'asset_ids.bin').write_bytes(binascii.unhexlify(''.join(selected['asset_id'].to_pylist())))
            columns = [pa.array(np.arange(count, dtype=np.int64))] + [selected[name].cast(MAPPING_SCHEMA.field(name).type)
                       for name in MAPPING_SCHEMA.names[1:]]
            pq.write_table(pa.Table.from_arrays(columns, schema=MAPPING_SCHEMA), temporary / 'mapping.parquet')
        require((temporary / 'vectors.f32').stat().st_size == count * s.VECTOR_BYTES
                and (temporary / 'asset_ids.bin').stat().st_size == count * 32, 'Packed layout differs')
        artifacts = []
        for path in sorted(temporary.iterdir()):
            item = s.artifact(path, temporary); item['path'] = str((destination / path.name).relative_to(output)); artifacts.append(item)
        receipt = {'schema': VERSION, 'complete': True, 'binding': _task_binding(plan_sha, task),
                   'count': count, 'skipped': skipped, 'invalid': invalid, 'input_rows': task['input_rows'],
                   'bounds': bounds, 'adoption': adoption, 'seconds': time.monotonic() - started,
                   'reader_method': reader.method, 'reader_workers': reader.workers, 'phase_seconds': phases,
                   'reader_stats_delta': {k: reader.stats[k] - stats_before[k] for k in reader.stats},
                   'artifacts': artifacts}
        require(s.stamp(source['path']) == source['stamp'], 'Source parquet changed during task')
        _publish(temporary, destination, receipt)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def status(output):
    output, plan, plan_sha = load_plan(output)
    done = []
    for task in plan['tasks']:
        if (_task_path(output, task) / 'COMPLETE.json').exists():
            done.append(_task_receipt(output, plan_sha, task, verify=False))
    published = (output / 'catalog' / 'COMPLETE.json').exists()
    return {'schema': VERSION, 'complete': len(done) == len(plan['tasks']),
            'complete_units': len(done), 'total_units': len(plan['tasks']),
            'completed_vectors': sum(r['count'] for r in done), 'total_vectors': plan['expected_vectors'],
            'ready_to_finalize': len(done) == len(plan['tasks']) and not published,
            'tasks_total': len(plan['tasks']), 'tasks_complete': len(done),
            'vectors': sum(r['count'] for r in done), 'input_assets_complete': sum(r['input_rows'] for r in done),
            'input_assets': plan['input_assets'], 'adopted_complete': sum(r['adoption'] is not None for r in done),
            'catalog_published': published}


def worker(output, *, workers=8, max_open=32768, method='pread', max_tasks=None, task_id=None,
           install_signals=True, stop_file=None):
    output, plan, plan_sha = load_plan(output); area = output / 'catalog_parallel'
    pa.set_cpu_count(workers)
    pa.set_io_thread_count(workers)
    worker_id = f'{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    stopped = threading.Event(); previous = {}
    if install_signals and threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda *_: stopped.set())
    stop_file = Path(stop_file) if stop_file else area / 'STOP'
    reader = ParallelVectors(area, workers=workers, max_open=max_open, method=method, legacy_dirs=plan['legacy_verified_dirs'])
    todo = [task for task in plan['tasks'] if task_id is None or task['task_id'] == task_id]
    require(todo or not plan['tasks'], 'Unknown task ID')
    random.Random(worker_id).shuffle(todo)
    completed = 0; started = time.monotonic()
    try:
        while not stopped.is_set() and not stop_file.exists() and (max_tasks is None or completed < max_tasks):
            available, busy = False, False
            for task in todo:
                if stopped.is_set() or stop_file.exists():
                    break
                if (_task_path(output, task) / 'COMPLETE.json').exists():
                    continue
                with file_lock(area / 'task_locks' / (task['task_id'] + '.lock'), blocking=False) as acquired:
                    if not acquired:
                        busy = True; continue
                    if (_task_path(output, task) / 'COMPLETE.json').exists():
                        continue
                    available = True
                    s.atomic_json(area / 'workers' / (worker_id + '.json'), {'state': 'running', 'task_id': task['task_id'],
                        'started_at_epoch': time.time(), 'completed_tasks': completed, 'plan_sha256': plan_sha})
                    pack_task(output, plan, plan_sha, task, reader)
                    completed += 1
                    if max_tasks is not None and completed >= max_tasks:
                        break
            if not available:
                if not busy:
                    break
                stopped.wait(1)
        result = {'worker_id': worker_id, 'completed_tasks': completed, 'seconds': time.monotonic() - started,
                  'stopped': stopped.is_set() or stop_file.exists(), 'reader_stats': reader.stats, 'state': 'finished'}
        s.atomic_json(area / 'workers' / (worker_id + '.json'), result)
        return result
    finally:
        reader.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def finalize(output):
    output, plan, plan_sha = load_plan(output)
    with file_lock(output / '.LOCK'):
        directory = output / 'catalog'
        identity = {**runtime(), 'stage': 'catalog', 'input': plan['identity']['input'],
                    'input_sha256': plan['identity']['input_completion']['sha256'],
                    'scope': plan['identity']['scope'], 'block_rows': plan['identity']['block_rows'],
                    'parallel_plan_sha256': plan_sha}
        s.bind(directory, identity)
        if (directory / 'COMPLETE.json').exists():
            result = s.read_json(directory / 'COMPLETE.json')
            require(result['identity_sha256'] == s.digest(identity), 'Final catalog identity differs')
            return result
        require(all((_task_path(output, t) / 'COMPLETE.json').exists() for t in plan['tasks']), 'Parallel tasks incomplete')
        blocks, next_id, skipped, invalid, previous = [], 0, 0, 0, {}
        for task in plan['tasks']:
            receipt = _task_receipt(output, plan_sha, task)
            require(s.stamp(task['source']['path']) == task['source']['stamp'], 'Source parquet changed before finalization')
            bounds = receipt['bounds']; bucket = task['bucket']
            require(bucket not in previous or previous[bucket] < bounds['first_asset_id'], 'Assets unsorted across task boundaries')
            previous[bucket] = bounds['last_asset_id']
            destination = directory / 'blocks' / task['task_id']; destination.parent.mkdir(parents=True, exist_ok=True)
            task_receipt = bound(_task_path(output, task) / 'COMPLETE.json')
            binding = {'input_asset_sha256': task['source']['sha256'], 'batch': task['batch'],
                       'input_rows': task['input_rows'], 'start_id': next_id, 'identity_sha256': s.digest(identity)}
            if (destination / 'COMPLETE.json').exists():
                final = s.read_json(destination / 'COMPLETE.json')
                require(final['binding'] == binding and final['parallel_task_receipt'] == task_receipt, 'Final block checkpoint differs')
                s.verify_receipt(output, final)
            else:
                temporary = Path(tempfile.mkdtemp(prefix='.' + task['task_id'] + '-', dir=destination.parent))
                try:
                    arts = {Path(a['path']).name: a for a in receipt['artifacts']}
                    mapping = pq.read_table(output / arts['mapping.parquet']['path'])
                    require(len(mapping) == receipt['count'] and np.array_equal(mapping['vector_id'].to_numpy(),
                            np.arange(receipt['count'])), 'Local vector IDs invalid')
                    mapping = mapping.set_column(0, 'vector_id', pa.array(np.arange(next_id, next_id + receipt['count'], dtype=np.int64)))
                    pq.write_table(mapping.cast(MAPPING_SCHEMA), temporary / 'mapping.parquet')
                    artifacts = []
                    for name in ('vectors.f32', 'asset_ids.bin', 'exceptions.parquet'):
                        _link_or_copy(output / arts[name]['path'], temporary / name, arts[name]['sha256'])
                        artifacts.append({**arts[name], 'path': str((destination / name).relative_to(output))})
                    item = s.artifact(temporary / 'mapping.parquet', temporary)
                    item['path'] = str((destination / 'mapping.parquet').relative_to(output)); artifacts.append(item)
                    final = {'complete': True, 'binding': binding, 'start_id': next_id,
                             'count': receipt['count'], 'skipped': receipt['skipped'], 'invalid': receipt['invalid'],
                             'parallel_task_receipt': task_receipt, 'adoption': receipt['adoption'], 'artifacts': artifacts}
                    _publish(temporary, destination, final)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            blocks.append({'receipt_path': str((destination / 'COMPLETE.json').relative_to(output)),
                           'receipt_sha256': s.file_sha(destination / 'COMPLETE.json'), 'start_id': next_id, 'count': receipt['count']})
            next_id += receipt['count']; skipped += receipt['skipped']; invalid += receipt['invalid']
        require(next_id + skipped + invalid == plan['input_assets'], 'Final coverage accounting differs')
        result = {'schema': s.VERSION, 'stage': 'catalog', 'complete': True, 'scope': plan['identity']['scope'],
                  'identity_sha256': s.digest(identity), 'input_sha256': plan['identity']['input_completion']['sha256'],
                  'vectors': next_id, 'input_assets': plan['input_assets'], 'skipped_assets': skipped,
                  'invalid_vectors': invalid, 'blocks': blocks, 'parallel_plan_sha256': plan_sha,
                  'adopted_blocks': plan['adopted_task_count']}
        s.atomic_json(directory / 'COMPLETE.json', result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'finalize', 'status'])
    parser.add_argument('--output', required=True); parser.add_argument('--input'); parser.add_argument('--adopt')
    parser.add_argument('--block-rows', type=int, default=250_000); parser.add_argument('--allow-bounded', action='store_true')
    parser.add_argument('--workers', type=int, default=8); parser.add_argument('--max-open', type=int, default=32768)
    parser.add_argument('--method', choices=['pread', 'mmap'], default='pread'); parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--task-id'); parser.add_argument('--stop-file')
    args = parser.parse_args(argv)
    if args.stage == 'prepare':
        require(args.input is not None, '--input is required')
        result = prepare(args.input, args.output, adopt=args.adopt, block_rows=args.block_rows, allow_bounded=args.allow_bounded)
        result = {k: v for k, v in result.items() if k not in {'tasks', 'buckets'}}
    elif args.stage == 'worker':
        result = worker(args.output, workers=args.workers, max_open=args.max_open, method=args.method,
                        max_tasks=args.max_tasks, task_id=args.task_id, stop_file=args.stop_file)
    elif args.stage == 'finalize':
        result = finalize(args.output); result = {k: v for k, v in result.items() if k != 'blocks'}
    else:
        result = status(args.output)
    print(s.canonical(result))


if __name__ == '__main__':
    main()
