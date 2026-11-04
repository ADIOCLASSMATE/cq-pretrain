"""Isolated CPU experiment: incremental RGBA hashing without changing production."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import types

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import numpy as np
from PIL import Image
from . import fast_prepare
from .prepare_profile import sample_assets
from corpus_pipeline.production.common import atomic_json, file_sha


def stream_sscd(opened):
    if not fast_prepare._opaque_rgb(opened) or opened.getexif().get(274) not in (None, 1):
        return fast_prepare._sscd_input(opened)
    rgba = opened.convert('RGBA')
    rgba.load()
    digest = hashlib.sha256()
    digest.update(b'RGBA' + struct.pack('>II', *rgba.size))
    digest.update(rgba.tobytes())
    tensor = np.asarray(opened.resize((320, 320), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tensor = ((tensor - np.array([.485, .456, .406], np.float32)) /
              np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1).copy()
    return tensor, {'pixel_sha256': digest.hexdigest(), 'oriented_width': opened.width,
                    'oriented_height': opened.height}


def run(root, output):
    import torch
    from corpus_pipeline.detector_calibration.infer import preprocessing
    from .inference import ResidentInferenceWorker, MODEL_NAMES
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    worker = ResidentInferenceWorker.__new__(ResidentInferenceWorker)
    worker.torch, worker.identity_sha256 = torch, 'cpu-hash-experiment-only'
    worker.bundles = {name: {'transform': preprocessing(name)[0]} for name in ('cf384', 'unifd')}
    assets = sample_assets(root, 64)
    # Copy only this function's globals; the imported production module and its
    # function objects are never monkeypatched, even inside this isolated process.
    candidate = types.FunctionType(fast_prepare.prepare_asset.__code__,
        {**fast_prepare.prepare_asset.__globals__, 'sscd_input': stream_sscd}, 'experimental_prepare')
    for asset in assets:
        a, at = fast_prepare.prepare_asset(worker, asset, MODEL_NAMES)
        b, bt = candidate(worker, asset, MODEL_NAMES)
        a.pop('decode_seconds'); b.pop('decode_seconds')
        if a != b or a['status'] != 'prepared' or any(not torch.equal(at[name], bt[name]) for name in MODEL_NAMES):
            raise AssertionError('incremental hashing changed input tensors or metadata')
    rows = []
    for repetition in range(2):
        cases = [('current', fast_prepare.prepare_asset), ('stream_hash', candidate)]
        if repetition:
            cases.reverse()
        for name, prepare in cases:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=16) as pool:
                values = list(pool.map(lambda asset: prepare(worker, asset, MODEL_NAMES), assets))
            elapsed = time.perf_counter() - started
            if any(row[0]['status'] != 'prepared' for row in values):
                raise AssertionError('preparation grid failed')
            del values
            rows.append({'name': name, 'repetition': repetition, 'images': len(assets),
                         'seconds': elapsed, 'images_per_second': len(assets) / elapsed})
    elapsed = {name: sum(row['seconds'] for row in rows if row['name'] == name) for name in ('current', 'stream_hash')}
    result = {'success': True, 'images': len(assets), 'decode_workers': 16, 'cpu_threads': 1,
              'three_model_tensors_and_metadata_bitwise_equal': True, 'grid': rows,
              'pooled_speedup': elapsed['current'] / elapsed['stream_hash'],
              'scope': 'bounded_cpu_only_rgba_hash_experiment_not_npu_or_production',
              'frozen_fast_prepare_sha256': file_sha(fast_prepare.__file__), 'production_mutations': 0}
    atomic_json(output / 'COMPLETE.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=False)
    run(args.root, args.output)
