"""Immutable, resumable FP16 cache of the authoritative normalized FP32 vectors.

Bounds cover every vector, not a sample. This cache is only a candidate-search
optimization: accepted similarities must be reranked from original FP32 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time

import numpy as np

from . import sscd_index as s
from .sscd_parallel_catalog import file_lock

VERSION = 'corpus.sscd_exact_cache.v1'
NORMALIZATION = 'sscd_index.normalized_float32'


def upward(value):
    return float(np.nextafter(np.float64(value), np.float64(np.inf)))


def norm_bounds(rows):
    """Conservative binary64 L2 upper bounds for stored binary32/16 numbers.

Products of two binary32 numbers are exact in binary64. Inflate the sum by
gamma_dimension (valid even for sequential summation), then round sqrt upward.
"""
    values = np.asarray(rows, dtype=np.float64)
    s.require(np.isfinite(values).all(), 'Nonfinite values in norm bound')
    gamma = values.shape[1] * 2.0**-53 / (1 - values.shape[1] * 2.0**-53)
    squares = np.sum(values * values, axis=1, dtype=np.float64)
    inflated = np.nextafter(squares / (1 - gamma), np.inf)
    return np.nextafter(np.sqrt(inflated), np.inf)


def dot_error_bound(max_norm, max_error):
    # All inputs are upper bounds. Round each nonnegative arithmetic operation up.
    first = upward(upward(2 * max_norm) * max_error)
    return upward(first + upward(max_error * max_error))


def runtime():
    from . import sscd_parallel_catalog as c
    return {'code_sha256': s.file_sha(__file__), 'helper_code_sha256': s.file_sha(s.__file__),
            'lock_helper_code_sha256': s.file_sha(c.__file__), 'numpy': np.__version__}


def prepare(source, output, *, chunk_rows=8192, allow_bounded=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    s.require(source != output and 1 <= chunk_rows <= 131072, 'Invalid cache output/chunk size')
    data = s.Catalog(source, allow_bounded=allow_bounded)
    blocks = []
    for number, block in enumerate(data.blocks):
        item = next(a for a in block['artifacts'] if a['path'].endswith('/vectors.f32'))
        path = s.bound_path(source, item, verify=False)
        s.require(item['size_bytes'] == block['count'] * s.VECTOR_BYTES, 'Source shape mismatch')
        blocks.append({'number': number, 'start_id': block['start_id'], 'count': block['count'],
                       'source_artifact': item, 'source_stamp': s.stamp(path)})
    identity = {'schema': VERSION, 'runtime': runtime(), 'source_root': str(source),
                'source_catalog_sha256': data.sha256, 'normalization': NORMALIZATION,
                'dimension': s.DIMENSION, 'dtype': '<f2', 'chunk_rows': chunk_rows,
                'scope': data.receipt['scope']}
    plan = {'identity': identity, 'vectors': data.count, 'blocks': blocks}
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / '.prepare.lock'):
        path = output / 'PLAN.json'
        if path.exists():
            s.require(s.read_json(path) == plan, 'Cache source/config/runtime changed; use new output')
        else:
            s.require(not (output / 'COMPLETE.json').exists(), 'Complete cache without plan')
            s.atomic_json(path, plan)
    return plan


def load_plan(output):
    root = Path(output).resolve(); plan = s.read_json(root / 'PLAN.json'); ident = plan['identity']
    s.require(ident['runtime'] == runtime(), 'Cache runtime changed')
    source = Path(ident['source_root'])
    s.require(s.file_sha(source / 'catalog/COMPLETE.json') == ident['source_catalog_sha256'],
              'Source catalog changed')
    end = 0
    for number, block in enumerate(plan['blocks']):
        s.require(block['number'] == number and block['start_id'] == end and block['count'] > 0,
                  'Cache block coverage gap/overlap')
        end += block['count']
    s.require(end == plan['vectors'] and end > 0, 'Cache coverage incomplete')
    return root, plan


def block_dir(root, number):
    return root / 'blocks' / f'{number:06d}'


def verify_block(root, plan, block, *, full_hash=True):
    path = block_dir(root, block['number']) / 'COMPLETE.json'; receipt = s.read_json(path)
    s.require(receipt['identity'] == plan['identity'] and receipt['source_block'] == block,
              'Cache block identity changed')
    s.require(receipt['start_id'] == block['start_id'] and receipt['count'] == block['count'],
              'Cache block range changed')
    source = Path(plan['identity']['source_root']) / block['source_artifact']['path']
    s.require(s.stamp(source) == block['source_stamp'], 'Original vector stamp changed')
    item = receipt['artifacts'][0]; target = s.bound_path(root, item, verify=full_hash)
    s.require(item['size_bytes'] == block['count'] * s.DIMENSION * 2 and
              s.stamp(target) == receipt['output_stamp'], 'Cached vector stamp/shape changed')
    for key in ('max_original_norm', 'max_half_norm', 'max_quantization_l2_error'):
        s.require(np.isfinite(receipt[key]) and receipt[key] >= 0, 'Invalid cache error bound')
    return receipt


def write_block(root, plan, block):
    target = block_dir(root, block['number'])
    if (target / 'COMPLETE.json').exists():
        verify_block(root, plan, block, full_hash=False); return False
    s.require(not target.exists(), 'Uncommitted destination directory exists')
    temporary = target.with_name('.' + target.name + '.partial')
    if temporary.exists():
        shutil.rmtree(temporary)  # only this locked task's uncommitted output
    temporary.mkdir(parents=True)
    source = s.bound_path(Path(plan['identity']['source_root']), block['source_artifact'], verify=False)
    s.require(s.stamp(source) == block['source_stamp'], 'Original vector stamp changed')
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    maxima = dict(max_original_norm=0., max_half_norm=0., max_quantization_l2_error=0.)
    started = time.monotonic(); remaining = block['count']
    vector_path = temporary / 'vectors.f16'
    with source.open('rb') as incoming, vector_path.open('wb') as outgoing:
        while remaining:
            rows = min(remaining, plan['identity']['chunk_rows'])
            payload = incoming.read(rows * s.VECTOR_BYTES)
            s.require(len(payload) == rows * s.VECTOR_BYTES, 'Truncated original vectors')
            source_hash.update(payload)
            original = s.normalized(np.frombuffer(payload, dtype='<f4').reshape(rows, s.DIMENSION))
            half = original.astype('<f2')
            s.require(np.isfinite(half).all(), 'FP16 overflow/nonfinite cache')
            difference = original.astype(np.float64) - half.astype(np.float64)
            for key, values in (('max_original_norm', original), ('max_half_norm', half),
                                ('max_quantization_l2_error', difference)):
                maxima[key] = max(maxima[key], float(norm_bounds(values).max()))
            encoded = half.tobytes(order='C'); outgoing.write(encoded); output_hash.update(encoded)
            remaining -= rows
        s.require(not incoming.read(1), 'Original vectors have trailing bytes')
        outgoing.flush(); os.fsync(outgoing.fileno())
    s.require(source_hash.hexdigest() == block['source_artifact']['sha256'], 'Original vector SHA mismatch')
    s.require(s.stamp(source) == block['source_stamp'], 'Original changed during conversion')
    artifact = {'path': str((target / 'vectors.f16').relative_to(root)),
                'sha256': output_hash.hexdigest(), 'size_bytes': block['count'] * s.DIMENSION * 2}
    receipt = {'schema': VERSION, 'complete': True, 'identity': plan['identity'], 'source_block': block,
               'start_id': block['start_id'], 'count': block['count'], 'dimension': s.DIMENSION,
               'dtype': '<f2', 'normalization': NORMALIZATION, 'original_sha256_verified': source_hash.hexdigest(),
               **maxima, 'quantization_dot_error_bound': dot_error_bound(maxima['max_original_norm'],
                                                                       maxima['max_quantization_l2_error']),
               'artifacts': [artifact], 'output_stamp': s.stamp(vector_path),
               'seconds': time.monotonic() - started}
    s.atomic_json(temporary / 'COMPLETE.json', receipt)
    os.replace(temporary, target)
    fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def worker(output, *, max_blocks=None, threads=2):
    from threadpoolctl import threadpool_limits
    s.require(1 <= threads <= 16 and (max_blocks is None or max_blocks > 0), 'Invalid worker bounds')
    root, plan = load_plan(output); completed = 0
    with threadpool_limits(limits=threads):
        for block in plan['blocks']:
            if max_blocks is not None and completed >= max_blocks:
                break
            with file_lock(root / 'locks' / f"{block['number']:06d}.lock", blocking=False) as acquired:
                if acquired and write_block(root, plan, block):
                    completed += 1
                    print(s.canonical({'event': 'cache_block_complete', 'number': block['number'],
                                       'vectors': block['count']}), flush=True)
    return {'new_blocks': completed, **status(root)}


def status(output):
    root, plan = load_plan(output)
    done = [b for b in plan['blocks'] if (block_dir(root, b['number']) / 'COMPLETE.json').exists()]
    return {'blocks_complete': len(done), 'blocks_total': len(plan['blocks']),
            'vectors_complete': sum(b['count'] for b in done), 'vectors_total': plan['vectors'],
            'complete': (root / 'COMPLETE.json').exists(), 'counts_are_unverified_observation': True}


def finalize(output):
    root, plan = load_plan(output)
    with file_lock(root / '.finalize.lock'):
        rows, maxima = [], dict(max_original_norm=0., max_half_norm=0., max_quantization_l2_error=0.)
        for block in plan['blocks']:
            value = verify_block(root, plan, block, full_hash=True)
            path = block_dir(root, block['number']) / 'COMPLETE.json'
            rows.append({'start_id': block['start_id'], 'count': block['count'],
                         **value['artifacts'][0], 'receipt_path': str(path.relative_to(root)),
                         'receipt_sha256': s.file_sha(path)})
            for key in maxima:
                maxima[key] = max(maxima[key], value[key])
        result = {'schema': VERSION, 'complete': True, 'identity': plan['identity'],
                  'source_catalog_sha256': plan['identity']['source_catalog_sha256'],
                  'vectors': plan['vectors'], 'dimension': s.DIMENSION, 'dtype': '<f2',
                  'normalization': NORMALIZATION, 'blocks': rows, **maxima,
                  'quantization_dot_error_bound': dot_error_bound(maxima['max_original_norm'],
                                                                  maxima['max_quantization_l2_error']),
                  'bounds_cover_all_vectors': True, 'fp32_rerank_required': True,
                  'note': 'Add hardware-appropriate FP32 accumulation and output-rounding bounds to quantization_dot_error_bound before candidate screening.'}
        path = root / 'COMPLETE.json'
        if path.exists():
            s.require(s.read_json(path) == result, 'Completed cache changed')
        else:
            s.atomic_json(path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'status', 'finalize'])
    parser.add_argument('--input'); parser.add_argument('--output', required=True)
    parser.add_argument('--chunk-rows', type=int, default=8192)
    parser.add_argument('--threads', type=int, default=2); parser.add_argument('--max-blocks', type=int)
    parser.add_argument('--allow-bounded', action='store_true'); args = parser.parse_args()
    if args.stage == 'prepare':
        if not args.input:
            parser.error('--input required')
        result = prepare(args.input, args.output, chunk_rows=args.chunk_rows, allow_bounded=args.allow_bounded)
        result = {'prepared': True, 'vectors': result['vectors'], 'blocks': len(result['blocks'])}
    elif args.stage == 'worker':
        result = worker(args.output, max_blocks=args.max_blocks, threads=args.threads)
    elif args.stage == 'status':
        result = status(args.output)
    else:
        result = finalize(args.output)
        result = {key: value for key, value in result.items() if key != 'blocks'}
    print(s.canonical(result), flush=True)


if __name__ == '__main__':
    main()
