"""Resumable global upper-triangle SSCD threshold search, with FP32 reranking.

The default accelerator path reads and normalizes original FP32 vectors. Half
precision is optional and requires a separately proven accumulation contract.
No source records are modified; outputs are duplicate-pair review candidates.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import math
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time

import numpy as np

from . import sscd_index as s
from . import sscd_parallel_search as p

VERSION = 'corpus.sscd_exact_npu.v1'
NORMALIZATION = 'sscd_index.normalized_float32'
PRECISION_OPTIONS = {'ACL_OP_SELECT_IMPL_MODE': 'high_precision',
                     'ACL_OPTYPELIST_FOR_IMPLMODE': 'MatMul'}
_PLAN_HASHES = {}


def runtime():
    return {'code_sha256': s.file_sha(__file__), 'core_sha256': s.file_sha(s.__file__),
            'parallel_helper_sha256': s.file_sha(p.__file__), 'numpy': np.__version__}


def upward(value):
    return float(np.nextafter(np.float64(value), np.inf))


def conservative_margin(cache, compute_dtype):
    """IEEE FP32 multiply/add bound; explicit hardware contract is also gated.

    gamma_(2d) allows separately rounded multiplication and addition. A second
    such allowance covers the authoritative CPU FP32 dot and threshold edge.
    The optional FP16 path adds full-population quantization and output rounding.
    This is not a claim that finite empirical tests prove a device arithmetic
    contract: the backend binding and full-reference check are separate gates.
    """
    s.require(compute_dtype in ('float32', 'float16'), 'Unsupported compute dtype')
    norm = float(cache['max_original_norm']); half_norm = float(cache['max_half_norm'])
    s.require(math.isfinite(norm) and 0 < norm <= 1.01 and math.isfinite(half_norm), 'Invalid normalized norm bound')
    gamma = upward((2 * s.DIMENSION * 2.0**-24) / (1 - 2 * s.DIMENSION * 2.0**-24))
    host = upward(gamma * upward(norm * norm))
    if compute_dtype == 'float32':
        total = upward(2 * host)
        return {'quantization': 0., 'device_accumulation': host, 'output_rounding': 0.,
                'cpu_comparison': host, 'total': total, 'assumption': 'FP32 multiply and accumulation; no HF32'}
    quantization = float(cache['quantization_dot_error_bound'])
    accum = upward(gamma * upward(half_norm * half_norm))
    rounding = upward(2.0**-11 * upward(half_norm * half_norm + accum) + 2.0**-25)
    total = upward(upward(upward(quantization + accum) + rounding) + host)
    return {'quantization': quantization, 'device_accumulation': accum, 'output_rounding': rounding,
            'cpu_comparison': host, 'total': total, 'assumption': 'FP16 inputs/output, FP32 accumulation'}


def pair_tasks(blocks):
    number = 0
    for left, a in enumerate(blocks):
        for right in range(left, len(blocks)):
            b = blocks[right]
            pairs = a['count'] * b['count'] if left != right else a['count'] * (a['count'] - 1) // 2
            yield {'number': number, 'left': left, 'right': right, 'pairs': pairs}
            number += 1


def task_directory(root, task):
    return root / 'tasks' / f"{task['left']:06d}" / f"{task['right']:06d}"


def prepare(index, cache, output, *, threshold=.75, compute_dtype='float32', backend_qualification=None,
            query_tile=8192, database_tile=65536, candidate_batch=32768, min_free_gib=50, allow_bounded=False):
    index, cache, root = map(lambda x: Path(x).resolve(), (index, cache, output))
    s.require(root not in (index, cache), 'Search output must be separate from immutable inputs')
    s.require(-1 <= threshold <= 1 and 1 <= query_tile <= 65536 and 1 <= database_tile <= 262144,
              'Invalid threshold/tile bounds')
    s.require(query_tile * database_tile < 2**31 and 1 <= candidate_batch <= 262144,
              'Tile exceeds bounded NPU indexing/memory contract')
    s.require(math.isfinite(min_free_gib) and min_free_gib >= 0, 'Invalid disk reserve')
    data = p.SharedCatalog(index, allow_bounded=allow_bounded)
    value = s.read_json(cache / 'COMPLETE.json')
    s.require(value.get('complete') is True and value.get('bounds_cover_all_vectors') is True,
              'Cache bounds do not cover the complete catalog')
    s.require(value['source_catalog_sha256'] == data.sha256 and value['vectors'] == data.count,
              'Cache/catalog mismatch')
    s.require(value['normalization'] == NORMALIZATION and value['dimension'] == s.DIMENSION and value['dtype'] == '<f2',
              'Cache numerical semantics changed')
    s.require(value['identity']['runtime']['helper_code_sha256'] == s.file_sha(s.__file__),
              'Cache normalization helper changed')
    blocks = value['blocks']; end = 0
    s.require(len(blocks) == len(data.blocks), 'Cache block count mismatch')
    for a, b in zip(blocks, data.blocks):
        s.require(a['start_id'] == b['start_id'] == end and a['count'] == b['count'] > 0,
                  'Cache coverage gap/overlap')
        s.require(s.file_sha(cache / a['receipt_path']) == a['receipt_sha256'], 'Cache receipt changed')
        end += a['count']
    s.require(end == data.count, 'Cache vector count mismatch')
    backend = None
    if backend_qualification:
        source = Path(backend_qualification).resolve(); qualification = s.read_json(source)
        s.require(qualification.get('qualified') is True and qualification.get('allow_hf32') is False,
                  'Backend numerical probe not qualified')
        s.require(qualification.get('requested_options') == PRECISION_OPTIONS, 'Backend precision options differ')
        if compute_dtype == 'float16':
            s.require(qualification.get('fp16_accumulation_proven') is True, 'FP16 accumulation not independently proven')
        backend = {'path': str(source), 'sha256': s.file_sha(source), 'value': qualification}
    s.require(backend is not None or (allow_bounded and data.receipt['scope'] != 'full_run'),
              'Full run requires a bound accelerator numerical qualification')
    margin = conservative_margin(value, compute_dtype)
    identity = {'schema': VERSION, 'runtime': runtime(), 'index_root': str(index), 'cache_root': str(cache),
                'catalog_sha256': data.sha256, 'cache_sha256': s.file_sha(cache / 'COMPLETE.json'),
                'scope': data.receipt['scope'], 'threshold': threshold, 'compute_dtype': compute_dtype,
                'score_semantics': NORMALIZATION, 'dot_accumulation': 'float64_then_float32', 'margin': margin,
                'candidate_threshold': float(threshold - margin['total']),
                'query_tile': query_tile, 'database_tile': database_tile, 'candidate_batch': candidate_batch,
                'min_free_bytes': int(min_free_gib * 2**30),
                'backend_qualification': backend, 'allow_bounded': allow_bounded}
    plan = {'identity': identity, 'blocks': blocks, 'vectors': data.count,
            'total_tasks': len(blocks) * (len(blocks) + 1) // 2,
            'total_pairs': data.count * (data.count - 1) // 2}
    s.require(sum(t['pairs'] for t in pair_tasks(blocks)) == plan['total_pairs'], 'Upper-triangle coverage mismatch')
    root.mkdir(parents=True, exist_ok=True)
    with p.claim(root / '.prepare.lock', blocking=True):
        path = root / 'PLAN.json'
        if path.exists(): s.require(s.read_json(path) == plan, 'Search plan changed; use new output')
        else:
            s.require(not (root / 'COMPLETE.json').exists(), 'Completed output without plan')
            s.atomic_json(path, plan)
    return plan


def load_plan(output):
    root = Path(output).resolve(); plan = s.read_json(root / 'PLAN.json'); ident = plan['identity']
    s.require(ident['runtime'] == runtime(), 'Exact search runtime changed')
    s.require(s.file_sha(Path(ident['index_root']) / 'catalog/COMPLETE.json') == ident['catalog_sha256'],
              'Source catalog changed')
    s.require(s.file_sha(Path(ident['cache_root']) / 'COMPLETE.json') == ident['cache_sha256'], 'Cache manifest changed')
    cache = s.read_json(Path(ident['cache_root']) / 'COMPLETE.json')
    s.require(plan['blocks'] == cache['blocks'] and plan['vectors'] == cache['vectors'], 'Search/cache coverage changed')
    s.require(ident['margin'] == conservative_margin(cache, ident['compute_dtype']) and
              ident['candidate_threshold'] == ident['threshold'] - ident['margin']['total'], 'Candidate error bound changed')
    qualification = ident['backend_qualification']
    if qualification:
        s.require(s.file_sha(qualification['path']) == qualification['sha256'], 'Backend qualification changed')
    return root, plan


def binding(root, task):
    path = root / 'PLAN.json'; stamp = s.stamp(path)
    cached = _PLAN_HASHES.get(str(path))
    if cached is None or cached[0] != stamp:
        cached = (stamp, s.file_sha(path)); _PLAN_HASHES[str(path)] = cached
    return {'plan_sha256': cached[1], 'task': task}


def disk_guard(root, plan):
    free = shutil.disk_usage(root).free; minimum = plan['identity']['min_free_bytes']
    if free < minimum:
        s.atomic_json(root / 'STOP', {'reason': 'disk_reserve', 'free_bytes': free,
                                     'minimum_bytes': minimum, 'updated_at': time.time()})
        raise RuntimeError('Insufficient free disk; shared STOP written before further output')


def check_task(root, task, *, full_hash=True):
    directory = task_directory(root, task); receipt = s.read_json(directory / 'COMPLETE.json')
    s.require(receipt.get('complete') is True and receipt['binding'] == binding(root, task), 'Task binding changed')
    s.require(receipt['pairs_covered'] == task['pairs'], 'Task pair coverage mismatch')
    for item in receipt['artifacts']:
        s.bound_path(root, item, verify=full_hash)
    return receipt


class VectorStore:
    def __init__(self, root, plan):
        self.root = root; self.plan = plan; ident = plan['identity']
        self.data = p.SharedCatalog(ident['index_root'], allow_bounded=ident['allow_bounded'], max_open=8)
        self.cache = Path(ident['cache_root']); self.half_stamps = {}; self.normalized = OrderedDict()

    def originals(self, number):
        if number not in self.normalized:
            self.normalized[number] = s.normalized(self.data.block(number)[0])
        self.normalized.move_to_end(number)
        while len(self.normalized) > 2: self.normalized.popitem(last=False)
        return self.normalized[number]

    def input(self, number):
        if self.plan['identity']['compute_dtype'] == 'float32': return self.originals(number)
        block = self.plan['blocks'][number]
        s.require(s.file_sha(self.cache / block['receipt_path']) == block['receipt_sha256'], 'Half block receipt changed')
        art = {key: block[key] for key in ('path', 'sha256', 'size_bytes')}
        path = p.shared_verified(self.cache, art); self.half_stamps[path] = s.stamp(path)
        return np.memmap(path, mode='r', dtype='<f2', shape=(block['count'], s.DIMENSION))

    def validate(self):
        self.data.validate_open_inputs()
        for path, stamp in self.half_stamps.items(): s.require(s.stamp(path) == stamp, 'Half cache changed during task')


class NumpyBackend:
    """Bounded test backend only; cannot qualify an accelerator/full-run search."""
    def __init__(self, plan): self.dtype = np.float32 if plan['identity']['compute_dtype'] == 'float32' else np.float16
    def upload(self, values): return np.asarray(values, dtype=self.dtype)
    def score(self, left, right): return (left.astype(np.float32) @ right.astype(np.float32).T).astype(self.dtype)
    def positions(self, scores, thresholds, max_count):
        thresholds = np.asarray(thresholds, np.float64)
        mask = scores.astype(np.float64) >= thresholds
        # Slicing the mask before nonzero bounds index allocation even if dense.
        width = mask.shape[1]; rows_per = max(1, max_count // max(1, width))
        for start in range(0, len(mask), rows_per):
            for column in range(0, width, max_count):
                rows, cols = np.nonzero(mask[start:start+rows_per, column:column+max_count])
                yield rows + start, cols + column
    def sync(self): pass
    def metadata(self): return {'backend': 'numpy_bounded_test', 'numpy': np.__version__}


class NpuBackend:
    def __init__(self, plan, device):
        import torch
        import torch_npu
        self.torch = torch; self.device = device
        torch.set_num_threads(2); torch.npu.set_device(device); torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.matmul.allow_hf32 = False; torch.npu.set_option(PRECISION_OPTIONS)
        self.dtype = torch.float32 if plan['identity']['compute_dtype'] == 'float32' else torch.float16
        expected = plan['identity']['backend_qualification']['value']
        s.require(torch.__version__ == expected['torch'] and torch_npu.__version__ == expected['torch_npu'],
                  'NPU PyTorch runtime differs from qualification')
        s.require(torch.npu.get_device_name(device) == expected['device'], 'NPU model differs from qualification')
        for key, value in expected.get('options_readback', {}).items():
            s.require(str(torch_npu._C._npu_getOption(key)) == value, 'NPU precision option readback changed')
        for path, value in expected.get('cann_files', {}).items():
            s.require(Path(path).read_text() == value, 'CANN runtime differs from qualification')
        self.details = {'backend': 'torch_npu', 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                        'device_model': torch.npu.get_device_name(device), 'precision_options': PRECISION_OPTIONS,
                        'allow_hf32': False, 'device': device}

    def upload(self, values):
        return self.torch.from_numpy(np.array(values, copy=True, order='C')).to(device=self.device, dtype=self.dtype)

    def score(self, left, right): return left @ right.T

    def positions(self, scores, thresholds, max_count):
        torch = self.torch
        # Round downward in the actual output dtype, never upward at the boundary.
        dtype = np.float32 if self.dtype == torch.float32 else np.float16
        cutoff = np.asarray(thresholds, dtype=dtype)
        cutoff = np.nextafter(cutoff, np.asarray(-np.inf, dtype=dtype))
        cutoff_tensor = torch.as_tensor(cutoff, dtype=self.dtype, device=self.device)
        mask = scores >= cutoff_tensor  # device tensor is essential for throughput
        width = mask.shape[1]
        # Tiles are constrained below INT32_MAX; the explicit dtype avoids an
        # expensive NPU int64 reduction (measured ~90 ms versus ~4 ms).
        count = int(mask.sum(dtype=torch.int32).item())
        if not count: return
        if count <= max_count:
            positions = mask.flatten().nonzero().flatten().cpu().numpy()
            yield positions // width, positions % width
            return
        # All-identical or very dense tiles must not allocate O(tile_pairs) IDs.
        rows_per = max(1, max_count // max(1, width))
        for start in range(0, len(mask), rows_per):
            for column in range(0, width, max_count):
                piece = mask[start:start+rows_per, column:column+max_count]
                positions = piece.flatten().nonzero().flatten().cpu().numpy()
                if len(positions): yield positions // piece.shape[1] + start, positions % piece.shape[1] + column

    def sync(self): self.torch.npu.synchronize()
    def metadata(self): return self.details


def backend_for(plan, device):
    if device == 'cpu':
        s.require(plan['identity']['allow_bounded'] and plan['identity']['scope'] != 'full_run',
                  'CPU mock is forbidden for full production scope')
        return NumpyBackend(plan)
    s.require(device.startswith('npu:'), 'Expected npu:N device')
    return NpuBackend(plan, device)


def pair_scores(left, right):
    """Stable FP32 dot output over the original normalized FP32 coordinates.

    Binary64 multiplication/sum avoids dependence on accelerator reduction order;
    final scores are FP32, with no half-precision evidence accepted directly.
    """
    s.require(left.shape == right.shape and left.ndim == 2, 'Invalid paired-vector shape')
    return np.einsum('ij,ij->i', left, right, dtype=np.float64).astype(np.float32)


def iter_pairs(store, backend, task, left_device=None):
    plan = store.plan; ident = plan['identity']; a, b = plan['blocks'][task['left']], plan['blocks'][task['right']]
    left_values = store.input(task['left']); right_values = store.input(task['right'])
    left_device = backend.upload(left_values) if left_device is None else left_device
    right_device = left_device if task['left'] == task['right'] else backend.upload(right_values)
    for start in range(0, a['count'], ident['query_tile']):
        stop = min(start + ident['query_tile'], a['count'])
        for column in range(0, b['count'], ident['database_tile']):
            end = min(column + ident['database_tile'], b['count'])
            if task['left'] == task['right'] and end <= start: continue
            scores = backend.score(left_device[start:stop], right_device[column:end])
            for rows, cols in backend.positions(scores, ident['candidate_threshold'], ident['candidate_batch']):
                rows = np.asarray(rows, np.int64) + start; cols = np.asarray(cols, np.int64) + column
                if task['left'] == task['right']:
                    valid = rows < cols; rows, cols = rows[valid], cols[valid]
                if not len(rows): continue
                exact = pair_scores(store.originals(task['left'])[rows], store.originals(task['right'])[cols])
                keep = exact >= ident['threshold']
                if keep.any(): yield rows[keep] + a['start_id'], cols[keep] + b['start_id'], exact[keep]
            del scores


def edge_table(store, left, right, scores):
    import pyarrow as pa
    unique = np.unique(np.r_[left, right]); _, names = store.data.fetch(unique, asset_ids=True)
    names = dict(zip(map(int, unique), names))
    rows = []
    for a, b, score in zip(left, right, scores):
        a, b = int(a), int(b); sha_a, sha_b = names[a], names[b]
        rows.append({'edge_id': hashlib.sha256(bytes.fromhex(sha_a) + bytes.fromhex(sha_b)).hexdigest(),
                     'vector_id_a': a, 'vector_id_b': b, 'asset_id_a': sha_a, 'asset_id_b': sha_b,
                     'exact_cosine': float(score), 'decision': 'requires_pair_review', 'automatic_merge': False})
    return pa.Table.from_pylist(rows, schema=p.edge_schema())


def progress(root, worker_id, **fields):
    name = hashlib.sha256(worker_id.encode()).hexdigest()[:20]
    s.atomic_json(root / 'workers' / f'{name}.json', {'worker_id': worker_id, 'host': socket.gethostname(),
                  'pid': os.getpid(), 'updated_at': time.time(), **fields})


def qualification(root, plan):
    path = root / 'calibration/COMPLETE.json'; value = s.read_json(path)
    s.require(value.get('complete') is True and value.get('passed') is True and
              value['plan_sha256'] == s.file_sha(root / 'PLAN.json'), 'Exact full-reference calibration not qualified')
    s.require(value['min_recall'] >= .95 and value['min_positive_recall'] >= .99 and
              value['recall_at_k'] >= value['min_recall'] and value['positive_counts_match'] is True and
              value['threshold_positive_recall'] is not None and
              value['threshold_positive_recall'] >= value['min_positive_recall'], 'Calibration recall requirements not met')
    s.require(s.file_sha(value['reference_path']) == value['reference_sha256'], 'Exact reference evidence changed')
    s.require(value['catalog_sha256'] == plan['identity']['catalog_sha256'] and
              value['exact_reference_vectors'] == plan['vectors'], 'Calibration does not cover the bound catalog')
    s.require(value.get('qualified_for_full_run') is True or
              (plan['identity']['allow_bounded'] and plan['identity']['scope'] != 'full_run'), 'Bounded calibration is not production qualification')
    for item in value['artifacts']: s.bound_path(root, item)
    return value


def worker(output, *, worker_id, device, max_tasks=None):
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_limits
    root, plan = load_plan(output); qualification(root, plan)
    s.require(max_tasks is None or max_tasks > 0, 'Invalid task limit')
    disk_guard(root, plan)
    backend = backend_for(plan, device); store = VectorStore(root, plan); completed = 0
    # Different workers start with different query blocks, then retain a query
    # block on-device while processing its database blocks.
    offset = int(hashlib.sha256(worker_id.encode()).hexdigest()[:12], 16) % len(plan['blocks'])
    tasks = list(pair_tasks(plan['blocks'])); tasks.sort(key=lambda t: ((t['left'] - offset) % len(plan['blocks']), t['right']))
    left_number, left_device = None, None
    with threadpool_limits(limits=2):
        for task in tasks:
            if (root / 'STOP').exists(): break
            directory = task_directory(root, task)
            if (directory / 'COMPLETE.json').exists():
                check_task(root, task, full_hash=False); continue
            with p.claim(directory.parent / (directory.name + '.lock')) as acquired:
                if not acquired: continue
                if (directory / 'COMPLETE.json').exists():
                    check_task(root, task, full_hash=False); continue
                disk_guard(root, plan)
                s.require(not directory.exists(), 'Uncommitted task destination exists')
                for stale in directory.parent.glob('.' + directory.name + '.*.tmp'):
                    if stale.is_dir(): shutil.rmtree(stale)
                temporary = Path(tempfile.mkdtemp(prefix='.' + directory.name + '.', suffix='.tmp', dir=directory.parent))
                started = time.monotonic(); count = 0
                progress(root, worker_id, state='running', task=task, completed_tasks_this_process=completed)
                try:
                    if left_number != task['left']:
                        left_device = None
                        left_device = backend.upload(store.input(task['left'])); left_number = task['left']
                    file = temporary / 'edges.parquet'
                    with pq.ParquetWriter(file, p.edge_schema(), compression='zstd') as writer:
                        for left, right, scores in iter_pairs(store, backend, task, left_device):
                            disk_guard(root, plan)
                            writer.write_table(edge_table(store, left, right, scores)); count += len(left)
                    backend.sync(); store.validate()
                    with file.open('rb') as stream: os.fsync(stream.fileno())
                    art = s.artifact(file, temporary); art['path'] = str((directory / file.name).relative_to(root))
                    receipt = {'complete': True, 'binding': binding(root, task), 'pairs_covered': task['pairs'],
                               'edges': count, 'seconds': time.monotonic() - started, 'backend': backend.metadata(),
                               'artifacts': [art], 'candidate_edges_only': True}
                    s.atomic_json(temporary / 'COMPLETE.json', receipt); os.replace(temporary, directory)
                    completed += 1
                    progress(root, worker_id, state='running', last_committed_task=task,
                             completed_tasks_this_process=completed, last_task_seconds=receipt['seconds'], last_task_edges=count)
                finally:
                    if temporary.exists(): shutil.rmtree(temporary)
            if max_tasks is not None and completed >= max_tasks: break
    progress(root, worker_id, state='idle', completed_tasks_this_process=completed)
    return {'new_tasks': completed, **status(root)}


def calibrate(output, reference, *, device, min_recall=.95, min_positive_recall=.99):
    from threadpoolctl import threadpool_limits
    root, plan = load_plan(output); ident = plan['identity']; reference = Path(reference).resolve()
    s.require(.95 <= min_recall <= 1 and .99 <= min_positive_recall <= 1, 'Recall gates may not be weakened')
    directory = root / 'calibration'; directory.mkdir(parents=True, exist_ok=True)
    with p.claim(root / '.calibration.lock', blocking=True):
        if (directory / 'COMPLETE.json').exists(): return qualification(root, plan)
        reference_receipt = s.read_json(reference.parent / 'COMPLETE.json')
        s.require(reference_receipt['identity']['catalog_sha256'] == ident['catalog_sha256'] and
                  reference_receipt['exact_reference_vectors'] == plan['vectors'], 'Reference is not full-catalog exact evidence')
        s.require(reference_receipt['identity']['parameters']['threshold'] == ident['threshold'], 'Reference threshold differs')
        matching = [a for a in reference_receipt['artifacts']
                    if (Path(ident['index_root']) / a['path']).resolve() == reference]
        s.require(len(matching) == 1 and s.file_sha(reference) == matching[0]['sha256'], 'Reference evidence hash mismatch')
        with np.load(reference) as evidence:
            qids = evidence['query_ids'].copy(); expected_ids = evidence['exact_ids'].copy()
            expected_scores = evidence['exact_scores'].copy(); expected_counts = evidence['threshold_positive_count'].copy()
        s.require(len(qids) > 0 and len(set(map(int, qids))) == len(qids), 'Invalid reference query IDs')
        backend = backend_for(plan, device); store = VectorStore(root, plan)
        q32 = s.normalized(store.data.fetch(qids)); qdevice = backend.upload(q32)
        # Include every exact-reference top-k candidate, even for queries whose
        # tenth score is below the production duplicate threshold.
        cutoffs = np.minimum(ident['threshold'], expected_scores[:, -1].astype(np.float64)) - ident['margin']['total']
        truth_sets = [set(map(int, row)) for row in expected_ids]
        hit_sets = [set() for _ in qids]; counts = np.zeros(len(qids), np.int64); started = time.monotonic()
        with threadpool_limits(limits=2):
            for number, block in enumerate(plan['blocks']):
                db = backend.upload(store.input(number))
                for offset in range(0, block['count'], ident['database_tile']):
                    end = min(offset + ident['database_tile'], block['count'])
                    values = backend.score(qdevice, db[offset:end])
                    for rows, columns in backend.positions(values, cutoffs[:, None], ident['candidate_batch']):
                        rows, columns = np.asarray(rows, np.int64), np.asarray(columns, np.int64) + offset
                        ids = columns + block['start_id']; valid = ids != qids[rows]
                        rows, columns, ids = rows[valid], columns[valid], ids[valid]
                        if not len(rows): continue
                        exact = pair_scores(q32[rows], store.originals(number)[columns])
                        # Every database vector is visited once; do not retain
                        # an unbounded dictionary for dense duplicate clusters.
                        for row, other, score in zip(rows, ids, exact):
                            row, other = int(row), int(other)
                            if other in truth_sets[row]: hit_sets[row].add(other)
                            if score >= ident['threshold']: counts[row] += 1
                    del values
                del db
                progress(root, 'calibration', state='running', completed_blocks=number+1, total_blocks=len(plan['blocks']))
        store.validate(); backend.sync()
        recalls = np.asarray([len(found)/len(truth) for truth, found in zip(expected_ids, hit_sets)])
        # Counts are exact reference totals, not only the old ANN hit counts.
        positive_recall = float(np.minimum(counts, expected_counts).sum()/expected_counts.sum()) if expected_counts.sum() else None
        counts_match = bool(np.array_equal(counts, expected_counts))
        passed = bool(recalls.mean() >= min_recall and positive_recall is not None and
                      positive_recall >= min_positive_recall and counts_match)
        evidence_out = directory / 'evidence.npz'; temporary = directory / '.evidence.tmp.npz'
        np.savez(temporary, query_ids=qids, reference_ids=expected_ids, reference_scores=expected_scores,
                 reference_positive_counts=expected_counts, retrieved_positive_counts=counts, recall_at_k=recalls)
        os.replace(temporary, evidence_out)
        result = {'schema': VERSION, 'complete': True, 'passed': passed,
                  'qualified_for_full_run': passed and ident['scope'] == 'full_run' and device != 'cpu',
                  'scope': ident['scope'], 'plan_sha256': s.file_sha(root / 'PLAN.json'),
                  'catalog_sha256': ident['catalog_sha256'], 'reference_path': str(reference),
                  'reference_sha256': s.file_sha(reference), 'exact_reference_vectors': plan['vectors'],
                  'query_count': len(qids), 'recall_at_k': float(recalls.mean()),
                  'minimum_query_recall': float(recalls.min()), 'threshold_positive_recall': positive_recall,
                  'threshold_positive_count': int(expected_counts.sum()), 'retrieved_positive_count': int(counts.sum()),
                  'positive_counts_match': counts_match, 'min_recall': min_recall,
                  'min_positive_recall': min_positive_recall, 'backend': backend.metadata(),
                  'seconds': time.monotonic()-started, 'margin': ident['margin'],
                  'interpretation': 'All bound database vectors scanned for reference queries; observed qualification, not proof of hardware arithmetic.',
                  'artifacts': [s.artifact(evidence_out, root)]}
        s.atomic_json(directory / 'COMPLETE.json', result)
        s.require(passed, 'Exact NPU reference qualification failed; workers remain gated')
        return result


def status(output):
    root, plan = load_plan(output); done = pairs = edges = 0
    for task in pair_tasks(plan['blocks']):
        path = task_directory(root, task) / 'COMPLETE.json'
        if path.exists():
            value = s.read_json(path); done += 1; pairs += task['pairs']; edges += value.get('edges', 0)
    return {'total_tasks': plan['total_tasks'], 'completed_tasks': done, 'total_pairs': plan['total_pairs'],
            'completed_pairs': pairs, 'candidate_edges': edges, 'updated_at': time.time(),
            'complete': (root / 'COMPLETE.json').exists(), 'counts_are_unverified_observation': True}


def finalize(output):
    root, plan = load_plan(output)
    with p.claim(root / '.finalize.lock', blocking=True):
        qualification(root, plan); pairs = edges = 0; manifest = hashlib.sha256()
        for task in pair_tasks(plan['blocks']):
            value = check_task(root, task); pairs += value['pairs_covered']; edges += value['edges']
            manifest.update(s.file_sha(task_directory(root, task) / 'COMPLETE.json').encode())
        s.require(pairs == plan['total_pairs'], 'Global upper-triangle coverage incomplete')
        # Validate the input artifact graph too; completed task receipts do not
        # authorize an absent or changed original vector/asset-ID file.
        data = s.Catalog(plan['identity']['index_root'], allow_bounded=plan['identity']['allow_bounded'])
        for number in range(len(data.blocks)): data.block(number)
        result = {'schema': VERSION, 'complete': True, 'all_pairs_covered': True,
                  'plan_sha256': s.file_sha(root / 'PLAN.json'),
                  'calibration_sha256': s.file_sha(root / 'calibration/COMPLETE.json'),
                  'task_receipts_sha256': manifest.hexdigest(), 'vectors': plan['vectors'],
                  'total_tasks': plan['total_tasks'], 'completed_tasks': plan['total_tasks'],
                  'total_pairs': pairs, 'completed_pairs': pairs, 'candidate_edges': edges,
                  'candidate_edges_only': True, 'near_duplicates_resolved': False,
                  'automatic_merge': False, 'source_records_deleted': False}
        path = root / 'COMPLETE.json'
        if path.exists(): s.require(s.read_json(path) == result, 'Final completion changed')
        else: s.atomic_json(path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'calibrate', 'worker', 'status', 'finalize'))
    parser.add_argument('--output', required=True); parser.add_argument('--index'); parser.add_argument('--cache')
    parser.add_argument('--threshold', type=float, default=.75); parser.add_argument('--compute-dtype', choices=('float32','float16'), default='float32')
    parser.add_argument('--backend-qualification'); parser.add_argument('--reference'); parser.add_argument('--device', default='npu:0')
    parser.add_argument('--worker-id', default=socket.gethostname()); parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--query-tile', type=int, default=8192); parser.add_argument('--database-tile', type=int, default=65536)
    parser.add_argument('--candidate-batch', type=int, default=32768); parser.add_argument('--allow-bounded', action='store_true')
    parser.add_argument('--min-free-gib', type=float, default=50)
    args = parser.parse_args()
    if args.stage == 'prepare':
        s.require(args.index and args.cache, 'prepare needs --index and --cache')
        result = prepare(args.index, args.cache, args.output, threshold=args.threshold, compute_dtype=args.compute_dtype,
                         backend_qualification=args.backend_qualification, query_tile=args.query_tile,
                         database_tile=args.database_tile, candidate_batch=args.candidate_batch,
                         min_free_gib=args.min_free_gib, allow_bounded=args.allow_bounded)
        result = {k:v for k,v in result.items() if k != 'blocks'}
    elif args.stage == 'calibrate':
        s.require(args.reference, 'calibrate needs --reference'); result = calibrate(args.output, args.reference, device=args.device)
    elif args.stage == 'worker': result = worker(args.output, worker_id=args.worker_id, device=args.device, max_tasks=args.max_tasks)
    elif args.stage == 'status': result = status(args.output)
    else: result = finalize(args.output)
    print(s.canonical(result), flush=True)


if __name__ == '__main__': main()
