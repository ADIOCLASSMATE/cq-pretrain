"""Parallel global SSCD retrieval with immutable per-unit receipts.

Workers never hold the output-wide lock. Every query searches the same merged
index, followed by original-FP32 cosine reranking. Completion means candidate
retrieval completed, not that duplicates were adjudicated or records deleted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

import numpy as np
from . import sscd_index as s

VERSION = 'corpus.sscd_parallel.v1'
_THREAD_LOCKS = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextmanager
def claim(path, *, blocking=False):
    """POSIX record lock verified across cluster hosts; also serialize threads.

This shared filesystem's flock is host-local. lockf is process-owned, so a
per-path thread lock must be acquired BEFORE opening the file: even closing an
unlocked descriptor could release another thread's process-owned record lock.
"""
    path = Path(path).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(str(path), threading.Lock())
    local = thread_lock.acquire(blocking=blocking)
    if not local:
        yield False
        return
    try:
        with path.open('a+') as stream:
            acquired = False
            try:
                try:
                    fcntl.lockf(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                    acquired = True
                except BlockingIOError:
                    pass
                yield acquired
            finally:
                if acquired:
                    fcntl.lockf(stream, fcntl.LOCK_UN)
    finally:
        thread_lock.release()


def identity(stage, **kwargs):
    return {'schema': VERSION, 'stage': stage, 'code_sha256': s.file_sha(__file__),
            'core_code_sha256': s.file_sha(s.__file__), **kwargs}


def check_code(value):
    s.require(value['code_sha256'] == s.file_sha(__file__) and
              value['core_code_sha256'] == s.file_sha(s.__file__), 'Parallel runtime code changed')


def shared_verified(root, item):
    """Share full-SHA verification across workers; invalidate on any stat drift.

This is an optimization for immutable run inputs, not a permission/security
boundary. The expected SHA and size remain bound in the upstream receipt.
"""
    root = Path(root).resolve(); path = s.bound_path(root, item, verify=False)
    key = s.digest(item); directory = root / 'parallel_verified'; cache = directory / (key + '.json')
    with claim(directory / (key + '.lock'), blocking=True):
        before = s.stamp(path)
        if cache.exists():
            saved = s.read_json(cache)
            if saved.get('artifact') == item and saved.get('stamp') == before:
                return path
        s.require(s.file_sha(path) == item['sha256'], 'Artifact SHA mismatch')
        s.require(before == s.stamp(path), 'Artifact changed during verification')
        s.atomic_json(cache, {'artifact': item, 'stamp': before})
    return path


class SharedCatalog(s.Catalog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.shared_stamps = {}; self.batch_checked = None

    def begin_batch(self):
        self.batch_checked = set()

    def validate_open_inputs(self):
        for entries in self.shared_stamps.values():
            for path, expected in entries:
                s.require(s.stamp(path) == expected, 'Packed catalog changed after verification')

    def block(self, number):
        if number not in self.shared_stamps:
            arts = {Path(a['path']).name: a for a in self.blocks[number]['artifacts']}
            paths = [shared_verified(self.root, arts[name]) for name in ('vectors.f32', 'asset_ids.bin')]
            self.shared_stamps[number] = [(p, s.stamp(p)) for p in paths]
            self.verified.add(number)
        if self.batch_checked is None or number not in self.batch_checked:
            for path, expected in self.shared_stamps[number]:
                s.require(s.stamp(path) == expected, 'Packed catalog changed after verification')
            if self.batch_checked is not None:
                self.batch_checked.add(number)
        return super().block(number)

    def fetch(self, ids, *, asset_ids=False):
        """Group once; avoid a full ID-array scan separately for each block."""
        ids = np.asarray(ids, dtype=np.int64)
        s.require(ids.ndim == 1 and ((ids >= 0) & (ids < self.count)).all(), 'Vector IDs outside catalog')
        result = np.empty((len(ids), s.DIMENSION), np.float32); names = [None] * len(ids) if asset_ids else None
        numbers = np.searchsorted(self.starts, ids, side='right') - 1
        order = np.argsort(numbers, kind='stable'); sorted_numbers = numbers[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_numbers)) + 1, len(ids)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if start == stop:
                continue
            positions = order[start:stop]; number = int(sorted_numbers[start]); offsets = ids[positions] - self.starts[number]
            vectors, sha = self.block(number); result[positions] = vectors[offsets]
            if asset_ids:
                for position, offset in zip(positions, offsets):
                    names[position] = sha[offset].tobytes().hex()
        return (result, names) if asset_ids else result


def candidates(data, index, query_ids, candidate_k, gather_budget=131072):
    """Full-global ANN and exact FP32 rerank, with larger bounded gather groups."""
    data.begin_batch(); queries = s.normalized(data.fetch(query_ids)); result = []
    group_size = max(1, gather_budget // candidate_k)
    for start in range(0, len(query_ids), group_size):
        stop = min(len(query_ids), start + group_size)
        _, ids = index.search(queries[start:stop], min(candidate_k + 1, data.count))
        found_rows = [row[(row >= 0) & (row != qid)][:candidate_k]
                      for row, qid in zip(ids, query_ids[start:stop])]
        unique = np.unique(np.concatenate(found_rows)); raw = s.normalized(data.fetch(unique))
        for row, found in enumerate(found_rows):
            scores = raw[np.searchsorted(unique, found)] @ queries[start + row]
            order = np.lexsort((found, -scores)); result.append((found[order], scores[order]))
    return result


def shared_training(root, data):
    path = root / 'train' / 'COMPLETE.json'; receipt = s.read_json(path)
    s.require(receipt.get('complete') is True and receipt['identity']['catalog_sha256'] == data.sha256,
              'Training/catalog mismatch')
    for item in receipt['artifacts']:
        shared_verified(root, item)
    return receipt, s.file_sha(path)


def load_global(root, data, nprobe, ef_search):
    import faiss
    path = root / 'merged' / 'COMPLETE.json'; receipt = s.read_json(path)
    s.require(receipt.get('complete') is True and receipt['identity']['catalog_sha256'] == data.sha256,
              'Global index not complete for this catalog')
    paths = [shared_verified(root, item) for item in receipt['artifacts']]
    # A full read is deliberately used: ~6.5 GB per worker for 90M PQ64 vectors.
    index = faiss.read_index(str(paths[0]))
    s.require(index.ntotal == data.count and 1 <= nprobe <= index.nlist, 'Invalid global index/nprobe')
    index.nprobe = nprobe; index.use_precomputed_table = -1
    quantizer = faiss.downcast_index(index.quantizer)
    if hasattr(quantizer, 'hnsw'):
        s.require(ef_search >= nprobe, 'Coarse efSearch must cover nprobe')
        quantizer.hnsw.efSearch = ef_search
    return index, s.file_sha(path)


def directory(root, phase):
    s.require(phase in ('build', 'search'), 'Invalid parallel phase')
    return root / ('parallel_' + phase)


def plan_path(root, phase):
    return directory(root, phase) / 'PLAN.json'


def prepare(root, phase, value):
    path = plan_path(root, phase)
    with claim(root / ('.parallel_' + phase + '.prepare.lock')) as acquired:
        s.require(acquired, 'Parallel prepare/finalize already active')
        if path.exists():
            s.require(s.read_json(path) == value, 'Parallel plan input/config/code changed; use new output')
        else:
            s.require(not (directory(root, phase) / 'COMPLETE.json').exists(), 'Complete phase without plan')
            s.atomic_json(path, value)
    return value


def prepare_build(output, *, threads=14, allow_bounded=False):
    root = Path(output).resolve(); s.require(1 <= threads <= 64, 'Invalid threads')
    data = SharedCatalog(root, allow_bounded=allow_bounded)
    _, trained_sha = shared_training(root, data)
    units = [{'number': number, 'start': block['start_id'], 'stop': block['start_id'] + block['count']}
             for number, block in enumerate(data.blocks)]
    ident = identity('build', catalog_sha256=data.sha256, train_sha256=trained_sha,
                     threads=threads, allow_bounded=allow_bounded)
    return prepare(root, 'build', {'identity': ident, 'count': data.count, 'units': units})


def load_plan(root, phase):
    value = s.read_json(plan_path(root, phase)); check_code(value['identity'])
    return value


def load_data(root, plan):
    data = SharedCatalog(root, allow_bounded=plan['identity']['allow_bounded'])
    s.require(data.sha256 == plan['identity']['catalog_sha256'] and data.count == plan['count'],
              'Parallel catalog changed')
    return data


def binding(plan, unit):
    return {'plan_sha256': s.digest(plan), **unit}


def receipt_path(root, phase, unit):
    if phase == 'build':
        return root / 'shards' / f"{unit['number']:06d}.json"
    return directory(root, phase) / 'units' / f"{unit['number']:06d}.json"


def verified_unit(root, phase, plan, unit):
    path = receipt_path(root, phase, unit); value = s.read_json(path)
    s.require(value.get('complete') is True and value.get('binding') == binding(plan, unit),
              'Parallel unit binding mismatch')
    s.verify_receipt(root, value)
    if phase == 'search':
        import pyarrow.parquet as pq
        s.require(value.get('query_count') == unit['stop'] - unit['start'], 'Search unit coverage mismatch')
        actual = sum(pq.ParquetFile(root / item['path']).metadata.num_rows for item in value['artifacts'])
        s.require(actual == value['edges'], 'Search edge row count mismatch')
    return value


def worker_progress(root, phase, worker_id, **fields):
    name = s.digest(worker_id)[:16]
    s.atomic_json(directory(root, phase) / 'workers' / (name + '.json'),
                  {'worker_id': worker_id, 'pid': os.getpid(), 'hostname': socket.gethostname(),
                   'updated_at_unix': time.time(), **fields})


def work_units(root, phase, plan, worker_id):
    units = plan['units']; offset = int(s.digest(worker_id)[:8], 16) % max(1, len(units))
    for unit in units[offset:] + units[:offset]:
        if receipt_path(root, phase, unit).exists():
            continue  # Deep verification belongs to finalize, not each worker's scan.
        lock = directory(root, phase) / 'locks' / f"{unit['number']:06d}.lock"
        with claim(lock) as acquired:
            if acquired and not receipt_path(root, phase, unit).exists():
                yield unit  # Lock remains held throughout processing and commit.


def build_worker(output, *, worker_id, max_units=None):
    import faiss
    from threadpoolctl import threadpool_limits
    root = Path(output).resolve(); plan = load_plan(root, 'build'); data = load_data(root, plan)
    trained, train_hash = shared_training(root, data)
    s.require(train_hash == plan['identity']['train_sha256'], 'Parallel training changed')
    trained_path = next(shared_verified(root, a) for a in trained['artifacts'] if Path(a['path']).name == 'trained.faiss')
    template = faiss.read_index(str(trained_path)); threads = plan['identity']['threads']
    faiss.omp_set_num_threads(threads); done = 0
    s.require(max_units is None or max_units >= 1, 'Invalid max units')
    with threadpool_limits(limits=threads):
        for unit in work_units(root, 'build', plan, worker_id):
            worker_progress(root, 'build', worker_id, state='running', unit=unit, completed_units=done)
            started = time.monotonic(); index = faiss.clone_index(template); index.use_precomputed_table = -1
            vectors = s.normalized(data.block(unit['number'])[0])
            index.add_with_ids(vectors, np.arange(unit['start'], unit['stop'], dtype=np.int64))
            s.require(index.ntotal == unit['stop'] - unit['start'], 'Built shard coverage mismatch')
            path = root / 'shards' / f"{unit['number']:06d}.faiss"; s.write_index(path, index)
            result = {'complete': True, 'binding': binding(plan, unit), 'seconds': time.monotonic() - started,
                      'artifacts': [s.artifact(path, root)]}
            s.atomic_json(receipt_path(root, 'build', unit), result); done += 1
            if max_units is not None and done >= max_units:
                break
    worker_progress(root, 'build', worker_id, state='idle', completed_units=done)
    return {'new_units': done, **status(root, 'build')}


def check_ranges(plan):
    end = 0
    for number, unit in enumerate(plan['units']):
        s.require(unit['number'] == number and unit['start'] == end and unit['stop'] > end,
                  'Parallel range coverage gap/overlap')
        end = unit['stop']
    s.require(end == plan['count'], 'Parallel range coverage incomplete')


def finalize_build(output):
    import faiss
    root = Path(output).resolve(); plan = load_plan(root, 'build'); data = load_data(root, plan)
    _, train_hash = shared_training(root, data)
    s.require(train_hash == plan['identity']['train_sha256'], 'Parallel training changed')
    with claim(root / '.parallel_build.prepare.lock') as acquired:
        s.require(acquired, 'Parallel prepare/finalize already active'); check_ranges(plan)
        s.require(len(plan['units']) == len(data.blocks), 'Shard coverage mismatch')
        refs = []
        for unit in plan['units']:
            value = verified_unit(root, 'build', plan, unit)
            index = faiss.read_index(str(root / value['artifacts'][0]['path']))
            s.require(index.ntotal == unit['stop'] - unit['start'], 'Shard vector count mismatch')
            path = receipt_path(root, 'build', unit)
            refs.append({'receipt_path': str(path.relative_to(root)), 'receipt_sha256': s.file_sha(path)})
        result = {'complete': True, 'identity': plan['identity'], 'count': data.count, 'shards': refs}
        target = root / 'shards' / 'COMPLETE.json'
        if target.exists():
            s.require(s.read_json(target) == result, 'Completed build changed')
        else:
            s.atomic_json(target, result)
        s.atomic_json(directory(root, 'build') / 'COMPLETE.json', result)
        return result


def calibrate(output, *, threshold, query_count=128, top_k=10, candidate_k=256, nprobe=32,
              ef_search=128, min_recall=.95, min_positive_recall=.99, seed=8273,
              threads=14, exact_batch_size=131072, gather_budget=131072, allow_bounded=False):
    import faiss
    from threadpoolctl import threadpool_limits
    s.require(-1 <= threshold <= 1 and .95 <= min_recall <= 1 and .99 <= min_positive_recall <= 1,
              'Recall gates may not be lowered below .95/.99')
    s.require(1 <= query_count <= 4096 and 1 <= top_k <= candidate_k <= 65536 and 1 <= threads <= 64,
              'Invalid calibration bounds')
    s.require(1 <= exact_batch_size <= 1048576, 'Invalid exact batch size')
    s.require(32768 <= gather_budget <= 1048576, 'Invalid gather budget')
    root = Path(output).resolve()
    # Only calibration orchestration is exclusive; build/search workers use unit locks.
    with claim(root / '.parallel_calibrate.lock') as acquired, threadpool_limits(limits=threads):
        s.require(acquired, 'Calibration already running')
        data = SharedCatalog(root, allow_bounded=allow_bounded); shared_training(root, data)
        faiss.omp_set_num_threads(threads); index, index_sha = load_global(root, data, nprobe, ef_search)
        training_ids = set(np.fromfile(root / 'train' / 'sample_ids.i64', dtype=np.int64).tolist())
        s.require(top_k < data.count and query_count <= data.count - len(training_ids), 'Insufficient held-out queries')
        rng = np.random.default_rng(seed); selected = set(); ordered = []
        while len(ordered) < query_count:
            value = int(rng.integers(0, data.count))
            if value not in selected and value not in training_ids:
                selected.add(value); ordered.append(value)
        query_ids = np.asarray(ordered, np.int64)
        parameters = dict(threshold=threshold, query_count=query_count, top_k=top_k, candidate_k=candidate_k,
                          nprobe=nprobe, ef_search=ef_search, min_recall=min_recall,
                          min_positive_recall=min_positive_recall, seed=seed, threads=threads,
                          exact_batch_size=exact_batch_size, gather_budget=gather_budget)
        ident = s.stage_identity('calibrate', catalog_sha256=data.sha256, index_sha256=index_sha,
                                 parameters=parameters, parallel_module_sha256=s.file_sha(__file__))
        target = root / 'calibration' / s.digest(ident)[:16]; s.bind(target, ident)
        path = target / 'COMPLETE.json'
        if path.exists():
            result = s.read_json(path); s.verify_receipt(root, result); return result
        started = time.monotonic(); approx = candidates(data, index, query_ids, candidate_k, gather_budget)
        ann_seconds = time.monotonic() - started; started = time.monotonic()
        exact_ids, exact_scores, positives, hits = s.exact_neighbors(
            data, query_ids, top_k=top_k, threshold=threshold, ann_ids=[row[0] for row in approx], batch_size=exact_batch_size)
        exact_seconds = time.monotonic() - started
        data.validate_open_inputs()
        recall = np.asarray([len(set(map(int, truth)) & set(map(int, found[0]))) / top_k
                             for truth, found in zip(exact_ids, approx)])
        positive_recall = float(hits.sum() / positives.sum()) if positives.sum() else None
        passed = bool(recall.mean() >= min_recall and positive_recall is not None and positive_recall >= min_positive_recall)
        np.savez(target / 'evidence.npz', query_ids=query_ids, exact_ids=exact_ids, exact_scores=exact_scores,
                 recall_at_k=recall, threshold_positive_count=positives, threshold_positive_hits=hits,
                 ann_ids=np.asarray([np.pad(row[0], (0, candidate_k - len(row[0])), constant_values=-1) for row in approx]))
        result = {'complete': True, 'passed': passed, 'identity': ident, 'scope': data.receipt['scope'],
                  'qualified_for_full_run': passed and data.receipt['scope'] == 'full_run',
                  'recall_at_k': float(recall.mean()), 'minimum_query_recall': float(recall.min()),
                  'threshold_positive_recall': positive_recall, 'threshold_positive_count': int(positives.sum()),
                  'ann_seconds': ann_seconds, 'exact_seconds': exact_seconds, 'exact_reference_vectors': data.count,
                  'interpretation': 'Held-out queries against every catalog vector; sampled recall evidence, not zero-miss certification.',
                  'artifacts': [s.artifact(target / 'evidence.npz', root)]}
        s.atomic_json(path, result)
        s.atomic_json(root / 'calibration' / 'LATEST.json', {'receipt_path': str(path.relative_to(root)), 'receipt_sha256': s.file_sha(path)})
        return result


def qualification(root, data, path):
    result = s.read_json(path); ident = result['identity']; params = ident['parameters']
    s.require(result.get('complete') is True and result.get('passed') is True, 'Recall calibration did not pass')
    s.require(ident['catalog_sha256'] == data.sha256 and result['scope'] == data.receipt['scope'], 'Calibration catalog/scope mismatch')
    s.require(ident['code_sha256'] == s.file_sha(s.__file__), 'Calibration core code changed')
    if 'parallel_module_sha256' in ident:
        s.require(ident['parallel_module_sha256'] == s.file_sha(__file__), 'Calibration parallel code changed')
    s.require(params['min_recall'] >= .95 and params['min_positive_recall'] >= .99,
              'Recall gates may not be lowered below .95/.99')
    s.require(result['recall_at_k'] >= params['min_recall'] and result['threshold_positive_count'] > 0 and
              result['threshold_positive_recall'] is not None and result['threshold_positive_recall'] >= params['min_positive_recall'],
              'Calibration metrics do not pass required gates')
    s.require(result['exact_reference_vectors'] == data.count, 'Calibration must cover entire reference catalog')
    s.require(data.receipt['scope'] != 'full_run' or result.get('qualified_for_full_run') is True, 'Full-run calibration required')
    s.require(ident['index_sha256'] == s.file_sha(root / 'merged' / 'COMPLETE.json'), 'Calibration index mismatch')
    s.verify_receipt(root, result)
    return result


def prepare_search(output, calibration_path, *, queries_per_unit=32768, batch_size=1024, threads=14, allow_bounded=False):
    root = Path(output).resolve()
    s.require(1 <= batch_size <= 8192 and batch_size <= queries_per_unit <= 1048576 and 1 <= threads <= 64,
              'Invalid search unit/batch/threads')
    data = SharedCatalog(root, allow_bounded=allow_bounded)
    path = Path(calibration_path).resolve(); s.require(path.is_relative_to(root), 'Calibration must be inside output')
    qualified = qualification(root, data, path)
    ident = identity('search', catalog_sha256=data.sha256, index_sha256=qualified['identity']['index_sha256'],
                     calibration_sha256=s.file_sha(path), calibration_path=str(path.relative_to(root)),
                     queries_per_unit=queries_per_unit, batch_size=batch_size, threads=threads, allow_bounded=allow_bounded)
    units = [{'number': number, 'start': start, 'stop': min(start + queries_per_unit, data.count)}
             for number, start in enumerate(range(0, data.count, queries_per_unit))]
    return prepare(root, 'search', {'identity': ident, 'count': data.count, 'units': units})


def edge_schema():
    import pyarrow as pa
    return pa.schema([('edge_id', pa.string()), ('vector_id_a', pa.int64()), ('vector_id_b', pa.int64()),
                      ('asset_id_a', pa.string()), ('asset_id_b', pa.string()), ('exact_cosine', pa.float32()),
                      ('decision', pa.string()), ('automatic_merge', pa.bool_())])


def edge_rows(data, query_ids, found, threshold):
    edges = {}
    for qid, (ids, scores) in zip(query_ids, found):
        for other, score in zip(ids, scores):
            if score >= threshold:
                a, b = sorted((int(qid), int(other))); edges[(a, b)] = max(float(score), edges.get((a, b), -float('inf')))
    unique = sorted({value for pair in edges for value in pair})
    _, names = data.fetch(unique, asset_ids=True); names = dict(zip(unique, names))
    return [{'edge_id': hashlib.sha256(bytes.fromhex(names[a]) + bytes.fromhex(names[b])).hexdigest(),
             'vector_id_a': a, 'vector_id_b': b, 'asset_id_a': names[a], 'asset_id_b': names[b],
             'exact_cosine': score, 'decision': 'requires_pair_review', 'automatic_merge': False}
            for (a, b), score in sorted(edges.items())]


def search_worker(output, *, worker_id, max_units=None):
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_limits
    root = Path(output).resolve(); plan = load_plan(root, 'search'); data = load_data(root, plan); ident = plan['identity']
    path = root / ident['calibration_path']
    s.require(s.file_sha(path) == ident['calibration_sha256'], 'Bound calibration changed')
    qualified = qualification(root, data, path); params = qualified['identity']['parameters']; threads = ident['threads']
    faiss.omp_set_num_threads(threads); index, index_sha = load_global(root, data, params['nprobe'], params['ef_search'])
    s.require(index_sha == ident['index_sha256'], 'Bound global index changed')
    schema = edge_schema(); done = 0; s.require(max_units is None or max_units >= 1, 'Invalid max units')
    with threadpool_limits(limits=threads):
        for unit in work_units(root, 'search', plan, worker_id):
            worker_progress(root, 'search', worker_id, state='running', unit=unit, completed_units=done)
            path = receipt_path(root, 'search', unit); path.parent.mkdir(parents=True, exist_ok=True)
            parquet = path.with_suffix('.parquet'); temporary = parquet.with_suffix('.tmp'); count = 0; started = time.monotonic()
            with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
                for start in range(unit['start'], unit['stop'], ident['batch_size']):
                    stop = min(start + ident['batch_size'], unit['stop']); ids = np.arange(start, stop, dtype=np.int64)
                    found = candidates(data, index, ids, params['candidate_k'], params.get('gather_budget', 131072))
                    rows = edge_rows(data, ids, found, params['threshold'])
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema)); count += len(rows)
                    worker_progress(root, 'search', worker_id, state='running', unit=unit, query_position=stop,
                                    completed_units=done, uncommitted_edge_rows=count)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            data.validate_open_inputs()
            os.replace(temporary, parquet)
            result = {'complete': True, 'binding': binding(plan, unit), 'query_count': unit['stop'] - unit['start'],
                      'edges': count, 'seconds': time.monotonic() - started, 'artifacts': [s.artifact(parquet, root)]}
            s.atomic_json(path, result); done += 1
            if max_units is not None and done >= max_units:
                break
    worker_progress(root, 'search', worker_id, state='idle', completed_units=done)
    return {'new_units': done, **status(root, 'search')}


def finalize_search(output):
    root = Path(output).resolve(); plan = load_plan(root, 'search'); data = load_data(root, plan); ident = plan['identity']
    path = root / ident['calibration_path']; s.require(s.file_sha(path) == ident['calibration_sha256'], 'Bound calibration changed')
    qualification(root, data, path)
    # Recheck the global index artifact graph, not only its COMPLETE receipt.
    s.verify_receipt(root, s.read_json(root / 'merged' / 'COMPLETE.json'))
    with claim(root / '.parallel_search.prepare.lock') as acquired:
        s.require(acquired, 'Parallel prepare/finalize already active'); check_ranges(plan)
        refs = []; count = 0
        for unit in plan['units']:
            value = verified_unit(root, 'search', plan, unit); count += value['edges']; path = receipt_path(root, 'search', unit)
            refs.append({'path': str(path.relative_to(root)), 'sha256': s.file_sha(path)})
        result = {'complete': True, 'identity': ident, 'scope': data.receipt['scope'], 'query_count': data.count,
                  'full_query_completed': True, 'candidate_edges_only': True, 'near_duplicates_resolved': False,
                  'edge_rows': count, 'units': refs, 'automatic_merge': False, 'training_records_deleted': 0,
                  'edge_semantics': 'Canonical pair IDs; union repeated edge_id across query units/batches before pair review.',
                  'threshold_semantics': 'Recall-qualified candidate threshold, not automatic deletion.'}
        path = directory(root, 'search') / 'COMPLETE.json'
        if path.exists():
            s.require(s.read_json(path) == result, 'Completed search changed')
        else:
            s.atomic_json(path, result)
        return result


def status(output, phase):
    root = Path(output).resolve(); path = plan_path(root, phase)
    if not path.exists():
        return {'phase': phase, 'prepared': False, 'complete_units': 0, 'total_units': 0,
                'ready_to_finalize': False, 'complete': False}
    plan = s.read_json(path); completed = [u for u in plan['units'] if receipt_path(root, phase, u).exists()]
    kind = 'vectors' if phase == 'build' else 'queries'
    return {'phase': phase, 'prepared': True, 'complete_units': len(completed), 'total_units': len(plan['units']),
            'completed_' + kind: sum(u['stop'] - u['start'] for u in completed), 'total_' + kind: plan['count'],
            'ready_to_finalize': len(completed) == len(plan['units']),
            'complete': (directory(root, phase) / 'COMPLETE.json').exists(),
            'counts_are_unverified_observation': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare-build', 'build-worker', 'finalize-build', 'calibrate',
                                        'prepare-search', 'search-worker', 'finalize-search', 'status'])
    parser.add_argument('--output', required=True); parser.add_argument('--allow-bounded', action='store_true')
    parser.add_argument('--worker-id', default=socket.gethostname() + '-' + str(os.getpid()))
    parser.add_argument('--threads', type=int, default=14); parser.add_argument('--max-units', type=int)
    parser.add_argument('--calibration'); parser.add_argument('--queries-per-unit', type=int, default=32768)
    parser.add_argument('--batch-size', type=int, default=1024); parser.add_argument('--phase', choices=['build', 'search'])
    parser.add_argument('--threshold', type=float, default=.75); parser.add_argument('--query-count', type=int, default=128)
    parser.add_argument('--top-k', type=int, default=10); parser.add_argument('--candidate-k', type=int, default=256)
    parser.add_argument('--nprobe', type=int, default=32); parser.add_argument('--ef-search', type=int, default=128)
    parser.add_argument('--min-recall', type=float, default=.95); parser.add_argument('--min-positive-recall', type=float, default=.99)
    parser.add_argument('--exact-batch-size', type=int, default=131072)
    parser.add_argument('--gather-budget', type=int, default=131072)
    args = parser.parse_args(argv)
    if args.stage == 'prepare-build':
        result = prepare_build(args.output, threads=args.threads, allow_bounded=args.allow_bounded)
    elif args.stage == 'build-worker':
        result = build_worker(args.output, worker_id=args.worker_id, max_units=args.max_units)
    elif args.stage == 'finalize-build': result = finalize_build(args.output)
    elif args.stage == 'calibrate':
        result = calibrate(args.output, threshold=args.threshold, query_count=args.query_count, top_k=args.top_k,
                           candidate_k=args.candidate_k, nprobe=args.nprobe, ef_search=args.ef_search,
                           min_recall=args.min_recall, min_positive_recall=args.min_positive_recall,
                           threads=args.threads, exact_batch_size=args.exact_batch_size,
                           gather_budget=args.gather_budget, allow_bounded=args.allow_bounded)
    elif args.stage == 'prepare-search':
        parser.error('--calibration required') if not args.calibration else None
        result = prepare_search(args.output, args.calibration, queries_per_unit=args.queries_per_unit,
                                batch_size=args.batch_size, threads=args.threads, allow_bounded=args.allow_bounded)
    elif args.stage == 'search-worker': result = search_worker(args.output, worker_id=args.worker_id, max_units=args.max_units)
    elif args.stage == 'finalize-search': result = finalize_search(args.output)
    else:
        parser.error('--phase required') if not args.phase else None
        result = status(args.output, args.phase)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
