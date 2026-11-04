"""Bounded real-image bitwise preparation check and CPU throughput comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import time

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')

from corpus_pipeline.production.common import atomic_json, file_sha


def run(root, output, count):
    import torch
    from corpus_pipeline.detector_calibration.infer import preprocessing
    from .inference import ResidentInferenceWorker, MODEL_NAMES
    from .fast_prepare import prepare_asset
    from .prepare_profile import sample_assets
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    worker = ResidentInferenceWorker.__new__(ResidentInferenceWorker)
    worker.torch, worker.identity_sha256 = torch, 'cpu-preparation-diagnostic-only'
    worker.bundles = {name: {'transform': preprocessing(name)[0]} for name in ('cf384', 'unifd')}
    assets = sample_assets(root, count)
    atomic_json(output / 'SAMPLES.json', assets)
    for asset in assets:
        expected, reference = worker._prepare(asset, MODEL_NAMES)
        actual, tensors = prepare_asset(worker, asset, MODEL_NAMES)
        expected.pop('decode_seconds')
        actual.pop('decode_seconds')
        if expected != actual or expected['status'] != 'prepared' or any(
                not torch.equal(reference[name], tensors[name]) for name in MODEL_NAMES):
            raise AssertionError('Native image official tensor/result mismatch')
    atomic_json(output / 'BITWISE.json', {'images': len(assets), 'all_result_metadata_equal': True,
        'all_three_model_inputs_bitwise_equal': True, 'reference_inference_sha256':
        file_sha(Path(__file__).with_name('inference.py')),
        'fast_prepare_sha256': file_sha(Path(__file__).with_name('fast_prepare.py'))})
    rows = []
    for repetition in range(2):
        cases = [('reference', lambda a: worker._prepare(a, MODEL_NAMES)),
                 ('fast', lambda a: prepare_asset(worker, a, MODEL_NAMES))]
        if repetition:
            cases.reverse()
        for threads in (4, 8, 16):
            for name, prepare in cases:
                start, cpu_start = time.perf_counter(), time.process_time()
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    result = list(pool.map(prepare, assets))
                seconds = time.perf_counter() - start
                if any(item[0]['status'] != 'prepared' for item in result):
                    raise AssertionError('Preparation benchmark error')
                row = {'name': name, 'repetition': repetition, 'decode_threads': threads,
                       'intraop_threads': 4, 'images': len(assets), 'elapsed_seconds': seconds,
                       'images_per_second': len(assets) / seconds,
                       'process_cpu_seconds': time.process_time() - cpu_start}
                del result
                rows.append(row)
                atomic_json(output / 'GRID.json', rows)
                print(json.dumps(row), flush=True)
    result = {'success': True, 'hostname': socket.gethostname(), 'images': len(assets), 'grid': rows,
        'all_three_model_inputs_bitwise_equal': True, 'all_result_metadata_equal': True,
        'model_inference_calls': 0, 'production_queue_mutations': 0,
        'scope': 'cpu_preparation_only_bounded_as100m_not_npu_or_end_to_end_throughput',
        'cold_cache_claimed': False, 'completed_at': time.time(),
        'reference_inference_sha256': file_sha(Path(__file__).with_name('inference.py')),
        'fast_prepare_sha256': file_sha(Path(__file__).with_name('fast_prepare.py'))}
    atomic_json(output / 'COMPLETE.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    args = parser.parse_args()
    if args.count < 2 or args.count % 2 or args.count > 256:
        parser.error('count must be an even number in [2,256]')
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.root.resolve(), args.output.resolve(), args.count)


if __name__ == '__main__':
    main()
