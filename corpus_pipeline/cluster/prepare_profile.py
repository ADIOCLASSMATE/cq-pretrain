"""Read-only CPU profile of the frozen production image preparation.

This diagnostic never loads models, claims tasks, publishes screening results,
or changes official preprocessing. It compares instrumented tensors bit for bit
with ResidentInferenceWorker._prepare and benchmarks the unmodified method.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import struct
import time

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')

import numpy as np
from PIL import Image, ImageOps

from corpus_pipeline.production.common import atomic_json, file_sha


def sample_assets(root, count):
    import pyarrow.parquet as pq
    plan = json.loads((root / 'plan.json').read_text())
    dataset = Path(plan['snapshot_root'])
    manifest = json.loads((dataset / 'manifest.json').read_text())
    pool = Path(manifest['source_pool'])
    assets = []
    for number in (30, 31):
        path = dataset / 'data' / f'{number:06d}.parquet'
        table = pq.read_table(path, columns=['source', 'source_shard', 'images'])
        indices = np.linspace(0, min(len(table), 8192) - 1, count // 2, dtype=int)
        for row_number in indices:
            row = table.slice(int(row_number), 1).to_pylist()[0]
            if row['source'] != 'as100m' or len(row['images']) != 1:
                raise ValueError('Expected a single-image AS100M native row')
            image = row['images'][0]
            ref = {**image, 'kind': 'tar', 'path': str(pool / row['source_shard'])}
            assets.append({'sha256': ref['sha256'], 'original_ref': ref,
                           'diagnostic_parquet': str(path), 'diagnostic_row': int(row_number)})
    return assets


def prepare_detail(asset, worker):
    timings = {}
    def measured(name, function):
        start = time.perf_counter()
        value = function()
        timings[name] = time.perf_counter() - start
        return value

    ref = asset['original_ref']
    def read():
        with Path(ref['path']).open('rb') as stream:
            stream.seek(ref['offset'])
            return stream.read(ref['length'])
    blob = measured('tar_open_seek_read', read)
    if len(blob) != ref['length']:
        raise ValueError('short image read')
    # The reference performs both hashes: once in read_asset_bytes, then _read_blob.
    for name in ('sha256_read_asset_bytes', 'sha256_read_blob'):
        if measured(name, lambda: hashlib.sha256(blob).hexdigest()) != ref['sha256']:
            raise ValueError('native image SHA256 mismatch')
    with measured('pil_open_header', lambda: Image.open(io.BytesIO(blob))) as opened:
        metadata = {'width': opened.width, 'height': opened.height, 'mode': opened.mode,
                    'format': opened.format, 'compressed_bytes': len(blob)}
        if min(opened.size) < 128:
            return {'status': 'too_small', 'metadata': metadata, 'seconds': timings}, None
        if opened.width * opened.height > 100_000_000 or getattr(opened, 'n_frames', 1) != 1:
            raise ValueError('native image exceeds production limits')
        measured('pil_decode_load', opened.load)
        rgb = measured('detector_rgb_convert', lambda: opened.convert('RGB'))
        tensors = {}
        for name in ('cf384', 'unifd'):
            value = rgb
            for index, transform in enumerate(worker.bundles[name]['transform'].transforms):
                value = measured(f'{name}_{index}_{type(transform).__name__}',
                                 lambda transform=transform, value=value: transform(value))
            tensors[name] = value
        rgba = measured('sscd_exif_transpose_rgba', lambda: ImageOps.exif_transpose(opened).convert('RGBA'))
        measured('sscd_rgba_load', rgba.load)
        pixel_bytes = measured('sscd_rgba_tobytes', rgba.tobytes)
        pixel_sha = measured('sscd_pixel_sha256', lambda: hashlib.sha256(
            b'RGBA' + struct.pack('>II', *rgba.size) + pixel_bytes).hexdigest())
        image = measured('sscd_alpha_composite_rgb', lambda: Image.alpha_composite(
            Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB'))
        resized = measured('sscd_pil_resize_320', lambda: image.resize((320, 320), Image.Resampling.BILINEAR))
        array = measured('sscd_numpy_float_normalize', lambda: ((
            np.asarray(resized, dtype=np.float32) / 255.0 - np.array([.485, .456, .406], np.float32)) /
            np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1).copy())
        tensors['sscd'] = measured('sscd_torch_from_numpy', lambda: worker.torch.from_numpy(array))
    return {'status': 'prepared', 'metadata': metadata, 'pixel_sha256': pixel_sha,
            'seconds': timings}, tensors


def cpu_stat():
    path = Path('/sys/fs/cgroup/cpu.stat')
    return dict((name, int(value)) for name, value in
                (line.split() for line in path.read_text().splitlines())) if path.is_file() else {}


def run(root, output, count, repeats, decode_threads, intraop_threads):
    import torch
    from corpus_pipeline.detector_calibration.infer import preprocessing
    from .inference import ResidentInferenceWorker, MODEL_NAMES
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # _prepare only accesses these fields; model allocation/inference is excluded.
    worker = ResidentInferenceWorker.__new__(ResidentInferenceWorker)
    worker.torch, worker.identity_sha256 = torch, 'cpu-preparation-diagnostic-only'
    worker.bundles = {name: {'transform': preprocessing(name)[0]} for name in ('cf384', 'unifd')}
    assets = sample_assets(root, count)
    atomic_json(output / 'SAMPLES.json', assets)
    detailed, totals = [], {}
    for asset in assets:
        result, tensors = prepare_detail(asset, worker)
        expected, reference = worker._prepare(asset, MODEL_NAMES)
        if result['status'] != expected['status']:
            raise AssertionError('Instrumented preparation changed outcome')
        if tensors is not None:
            if result['pixel_sha256'] != expected['pixel_sha256'] or any(
                    not torch.equal(tensors[name], reference[name]) for name in MODEL_NAMES):
                raise AssertionError('Instrumented preparation changed official pixels')
        detailed.append({'sha256': asset['sha256'], **result})
        for name, seconds in result['seconds'].items():
            totals[name] = totals.get(name, 0.) + seconds
    atomic_json(output / 'PHASES.json', {'images': detailed, 'summed_phase_seconds': totals,
        'official_tensor_bitwise_equal': True, 'threads': 1,
        'scope': 'sequential_mixed_initial_and_warm_filesystem_cache_cpu_only'})
    print(json.dumps({'stage': 'phases', 'images': len(assets), 'phase_seconds': totals}), flush=True)
    grid = []
    for repetition in range(repeats):
        configurations = [(t, d) for t in intraop_threads for d in decode_threads]
        if repetition % 2:
            configurations.reverse()
        for intra, decode in configurations:
            torch.set_num_threads(intra)
            before = cpu_stat()
            process_start, started = time.process_time(), time.perf_counter()
            with ThreadPoolExecutor(max_workers=decode) as pool:
                values = list(pool.map(lambda asset: worker._prepare(asset, MODEL_NAMES), assets))
            elapsed = time.perf_counter() - started
            cpu_elapsed = time.process_time() - process_start
            after = cpu_stat()
            statuses = {status: sum(item[0]['status'] == status for item in values)
                        for status in set(item[0]['status'] for item in values)}
            if any(status not in ('prepared', 'too_small') for status in statuses):
                raise AssertionError('Preparation grid produced an error')
            row = {'intraop_threads': intra, 'decode_threads': decode, 'repetition': repetition,
                   'elapsed_seconds': elapsed, 'images_per_second': len(assets) / elapsed,
                   'process_cpu_seconds': cpu_elapsed, 'mean_busy_cpu_cores': cpu_elapsed / elapsed,
                   'cgroup_cpu_stat_delta': {key: after[key] - before[key] for key in before},
                   'statuses': statuses}
            del values
            grid.append(row)
            atomic_json(output / 'GRID.json', grid)
            print(json.dumps(row), flush=True)
    result = {'success': True, 'hostname': socket.gethostname(), 'images': len(assets),
        'sample_strategy': '32 uniformly spaced native rows each from s00030/s00031, maximum row8191',
        'scope': 'bounded_cpu_preparation_profile_not_npu_or_end_to_end_throughput',
        'cold_cache_claimed': False, 'official_tensor_bitwise_equal': True,
        'production_queue_mutations': 0, 'model_inference_calls': 0,
        'inference_sha256': file_sha(Path(__file__).with_name('inference.py')),
        'profile_script_sha256': file_sha(__file__), 'grid': grid,
        'cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        'summed_phase_seconds': totals, 'completed_at': time.time()}
    atomic_json(output / 'COMPLETE.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--decode-threads', type=int, nargs='+', default=[1, 4, 8, 16])
    parser.add_argument('--intraop-threads', type=int, nargs='+', default=[1, 4])
    args = parser.parse_args()
    if args.count < 2 or args.count % 2 or args.count > 256 or args.repeats < 1:
        parser.error('count must be an even number in [2,256], repeats positive')
    if any(n < 1 or n > 32 for n in args.decode_threads + args.intraop_threads):
        parser.error('thread counts must be in [1,32]')
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.root.resolve(), args.output.resolve(), args.count, args.repeats,
        args.decode_threads, args.intraop_threads)


if __name__ == '__main__':
    main()
