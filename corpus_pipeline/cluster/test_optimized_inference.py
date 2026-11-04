"""CPU scheduling/semantic tests; numerical NPU qualification is a separate gate."""
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import unittest

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import numpy as np
import torch

from .inference import ResidentInferenceWorker
from .optimized_inference import OptimizedResidentInferenceWorker
from .frontpass import checked_inference


def fixture_worker(cls):
    """Use real CPU tensor arithmetic with tiny deterministic stand-in models."""
    worker = cls.__new__(cls)
    worker.torch, worker.device, worker.dtype = torch, torch.device('cpu'), torch.float32
    worker.batch_size, worker.names, worker.closed = 2, ('cf384', 'unifd', 'sscd'), False
    worker.bundles = {
        'cf384': {'model': lambda x: x.sum(dim=1, keepdim=True)},
        'unifd': {'model': lambda x: x.mean(dim=1, keepdim=True)},
        'sscd': {'model': lambda x: x.sum(dim=1, keepdim=True).repeat(1, 512)},
    }
    worker.pool = ThreadPoolExecutor(max_workers=2)
    worker.stats = {'input_images': 0, 'complete_images': 0, 'batches': 0, 'decode_seconds': 0.,
                    'model_seconds': {name: 0. for name in worker.names},
                    'model_batches': {name: 0 for name in worker.names}}
    def prepare(asset, names):
        result = {'image_sha256': asset['sha256'], 'status': 'prepared', 'scores': {}}
        if asset.get('bad'):
            result.update(status='error', error_stage='decode', error_type='ValueError')
            return result, None
        return result, {name: torch.tensor([float(asset['sha256']) + 1., 2.]) for name in names}
    worker._prepare = prepare
    if isinstance(worker, OptimizedResidentInferenceWorker):
        worker.coordinator = ThreadPoolExecutor(max_workers=1)
        worker._active_stream = False
        worker.stats.update(preparation_wait_seconds=0., inference_seconds=0., stream_seconds=0.,
                            input_batch_histogram={}, valid_batch_histogram={})
    return worker


class OptimizedInferenceTests(unittest.TestCase):
    def test_exact_reference_arithmetic_order_decode_failures_and_tail(self):
        assets = [{'sha256': str(i), 'bad': i == 2} for i in range(7)]
        for names in (('cf384', 'unifd', 'sscd'), ('sscd',), ('cf384', 'unifd')):
            reference = fixture_worker(ResidentInferenceWorker)
            optimized = fixture_worker(OptimizedResidentInferenceWorker)
            try:
                a = list(reference.iter_results(assets, models=names))
                b = list(optimized.iter_results(assets, models=names))
                for left, right in zip(a, b):
                    if left.get('sscd', {}).get('embedding') is not None:
                        np.testing.assert_array_equal(left['sscd'].pop('embedding'), right['sscd'].pop('embedding'))
                    self.assertEqual(left, right)
                self.assertEqual([r['image_sha256'] for r in b], [str(i) for i in range(7)])
                self.assertEqual(optimized.stats['input_batch_histogram'], {'2': 3, '1': 1})
                self.assertEqual(optimized.stats['valid_batch_histogram'], {'2': 2, '1': 2})
            finally:
                reference.close(); optimized.close()

    def test_next_cpu_prepare_overlaps_main_thread_forward_and_is_bounded(self):
        worker = fixture_worker(OptimizedResidentInferenceWorker)
        main = threading.get_ident()
        next_started = threading.Event()
        consumed = []
        original_prepare = worker._prepare_batch
        original_infer = worker._infer_prepared
        def prepare(assets, names):
            self.assertNotEqual(threading.get_ident(), main)
            if assets[0]['sha256'] == '2':
                next_started.set()
            return original_prepare(assets, names)
        def infer(prepared, names):
            self.assertEqual(threading.get_ident(), main)
            self.assertTrue(next_started.wait(3), 'next CPU batch did not begin while main forwards')
            return original_infer(prepared, names)
        def source():
            for i in range(100):
                consumed.append(i)
                yield {'sha256': str(i)}
        worker._prepare_batch, worker._infer_prepared = prepare, infer
        stream = worker.iter_batches(source())
        try:
            batch, outputs = next(stream)
            self.assertEqual([r['sha256'] for r in batch], ['0', '1'])
            self.assertEqual(consumed, [0, 1, 2, 3])
            stream.close()
            self.assertFalse(worker._active_stream)
            self.assertEqual(consumed, [0, 1, 2, 3])
            # Cancellation frees the stream; a fresh caller can reuse the worker.
            self.assertEqual(len(worker.infer_batch([{'sha256': '7'}])), 1)
        finally:
            stream.close(); worker.close()

    def test_preparation_exception_drains_stream_and_can_retry(self):
        worker = fixture_worker(OptimizedResidentInferenceWorker)
        original = worker._prepare_batch
        worker._prepare_batch = lambda *args: (_ for _ in ()).throw(ValueError('prepare failed'))
        try:
            with self.assertRaisesRegex(ValueError, 'prepare failed'):
                list(worker.iter_results([{'sha256': '0'}]))
            self.assertFalse(worker._active_stream)
            worker._prepare_batch = original
            self.assertEqual(len(worker.infer_batch([{'sha256': '1'}])), 1)
        finally:
            worker.close()

    def test_model_error_validator_closes_prefetch_and_keeps_failure_visible(self):
        worker = fixture_worker(OptimizedResidentInferenceWorker)
        worker.bundles['cf384']['model'] = lambda _: (_ for _ in ()).throw(RuntimeError('model failed'))
        try:
            with self.assertRaisesRegex(RuntimeError, 'model forward failed: cf384'):
                list(checked_inference(worker, [{'sha256': str(i)} for i in range(8)], models=worker.names))
            self.assertFalse(worker._active_stream)
        finally:
            worker.close()

    def test_oversize_batch_and_reentrant_stream_rejected(self):
        worker = fixture_worker(OptimizedResidentInferenceWorker)
        stream = worker.iter_batches([{'sha256': str(i)} for i in range(4)])
        try:
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                worker.infer_batch([{'sha256': str(i)} for i in range(3)])
            next(stream)
            with self.assertRaisesRegex(RuntimeError, 'single-consumer'):
                worker.infer_batch([{'sha256': '5'}])
        finally:
            stream.close(); worker.close()


if __name__ == '__main__':
    unittest.main()
