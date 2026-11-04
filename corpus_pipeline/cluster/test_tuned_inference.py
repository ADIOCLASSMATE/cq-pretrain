"""Bounded scheduling and execution-context tests; no production model evidence."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
import threading
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .inference import ResidentInferenceWorker
from .optimized_inference import OptimizedResidentInferenceWorker
from .tuned_inference import TunedResidentInferenceWorker
from .test_optimized_inference import fixture_worker


def tuned_fixture(prefetch_batches=2):
    worker = fixture_worker(TunedResidentInferenceWorker)
    worker.prefetch_batches = prefetch_batches
    worker.stats['max_outstanding_preparation_batches'] = 0
    return worker


class TunedInferenceTests(unittest.TestCase):
    def test_config_validation_happens_before_loading_any_model(self):
        with patch.object(OptimizedResidentInferenceWorker, '__init__') as base:
            for key, values in [('decode_workers', [0, 33, True, 1.5]),
                                ('cpu_threads', [0, 17, False, '4']),
                                ('prefetch_batches', [0, 5, True, 1.5])]:
                for value in values:
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        TunedResidentInferenceWorker(device='cpu', **{key: value})
            with self.assertRaisesRegex(ValueError, 'separate'):
                TunedResidentInferenceWorker(device='cpu', threads=4)
            with self.assertRaisesRegex(ValueError, 'boolean'):
                TunedResidentInferenceWorker(device='cpu', fast_prepare=1)
            base.assert_not_called()

    def test_constructor_separates_cpu_threads_from_decode_pool_without_model_changes(self):
        identity = {'test_model': 'unchanged'}
        old_pool = ThreadPoolExecutor(max_workers=1)
        seen = {}
        def base(instance, *args, **kwargs):
            seen.update(kwargs)
            instance.pool = old_pool
            instance.torch = torch
            instance.closed = instance._active_stream = False
            instance.identity = identity
            instance.identity_sha256 = 'same-model-identity'
            instance.execution_identity = {'base_inference_code_sha256': 'bound', 'model_identity_unchanged': True}
            instance.stats = {}
        with patch.object(OptimizedResidentInferenceWorker, '__init__', base):
            worker = TunedResidentInferenceWorker(device='cpu', decode_workers=7, cpu_threads=2, prefetch_batches=3)
        try:
            self.assertEqual(seen['threads'], 2)
            self.assertEqual(worker.pool._max_workers, 7)
            self.assertIs(worker.identity, identity)
            self.assertEqual(worker.identity_sha256, 'same-model-identity')
            self.assertEqual(worker.execution_identity['maximum_prepared_or_preparing_batches'], 4)
            self.assertFalse(worker.execution_identity['fast_prepare'])
            self.assertIsNone(worker.execution_identity['fast_prepare_code_sha256'])
            self.assertEqual(len(worker.execution_identity['tuned_inference_code_sha256']), 64)
            with self.assertRaises(RuntimeError):
                old_pool.submit(lambda: None)
        finally:
            worker.pool.shutdown()

    def test_idle_reconfiguration_reuses_models_and_rejects_active_or_closed_worker(self):
        worker = tuned_fixture(1)
        worker.execution_identity = {'model_identity_unchanged': True}
        original_bundles = worker.bundles
        original_models = {name: bundle['model'] for name, bundle in worker.bundles.items()}
        stream = worker.iter_batches([{'sha256': str(i)} for i in range(4)])
        settings = {'batch_size': 4, 'decode_workers': 3, 'cpu_threads': 1, 'prefetch_batches': 2}
        try:
            next(stream)
            with self.assertRaisesRegex(RuntimeError, 'idle'):
                worker.configure(**settings)
            stream.close()
            identity = worker.configure(**settings)
            self.assertIs(worker.bundles, original_bundles)
            self.assertTrue(all(worker.bundles[name]['model'] is model for name, model in original_models.items()))
            self.assertEqual(identity['batch_size'], 4)
            self.assertEqual(worker.pool._max_workers, 3)
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertEqual(len(worker.infer_batch([{'sha256': str(i)} for i in range(4)])), 4)
        finally:
            stream.close(); worker.close()
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            worker.configure(**settings)

    def test_optional_fast_prepare_is_explicit_bound_and_reversible(self):
        worker = tuned_fixture(1)
        worker.execution_identity = {'model_identity_unchanged': True}
        settings = {'batch_size': 2, 'decode_workers': 2, 'cpu_threads': 1, 'prefetch_batches': 1}
        calls = []
        def fast(instance, asset, names):
            calls.append((instance, asset, names))
            return 'fast-result'
        fake_module = SimpleNamespace(prepare_asset=fast)
        try:
            with patch.dict(sys.modules, {'corpus_pipeline.cluster.fast_prepare': fake_module}), \
                 patch('corpus_pipeline.cluster.tuned_inference.file_sha', return_value='h' * 64):
                identity = worker.configure(**settings, fast_prepare=True)
            self.assertTrue(identity['fast_prepare'])
            self.assertEqual(identity['fast_prepare_code_sha256'], 'h' * 64)
            self.assertEqual(TunedResidentInferenceWorker._prepare(worker, {'sha256': 'asset'}, ('sscd',)), 'fast-result')
            self.assertIs(calls[0][0], worker)
            identity = worker.configure(**settings)
            self.assertFalse(identity['fast_prepare'])
            self.assertIsNone(identity['fast_prepare_code_sha256'])
            with patch.object(OptimizedResidentInferenceWorker, '_prepare', return_value='reference-result') as reference:
                self.assertEqual(TunedResidentInferenceWorker._prepare(worker, {}, ('sscd',)), 'reference-result')
                reference.assert_called_once()
        finally:
            worker.close()

    def test_all_prefetch_depths_preserve_exact_reference_arithmetic(self):
        assets = [{'sha256': str(i), 'bad': i == 3} for i in range(9)]
        reference = fixture_worker(ResidentInferenceWorker)
        try:
            expected = list(reference.iter_results(assets))
        finally:
            reference.close()
        for depth in range(1, 5):
            worker = tuned_fixture(depth)
            try:
                actual = list(worker.iter_results(assets))
                self.assertEqual(len(expected), len(actual))
                for a, b in zip(expected, actual):
                    if a.get('sscd', {}).get('embedding') is not None:
                        np.testing.assert_array_equal(a['sscd']['embedding'], b['sscd']['embedding'])
                    self.assertEqual({k: v for k, v in a.items() if k != 'sscd'},
                                     {k: v for k, v in b.items() if k != 'sscd'})
                self.assertLessEqual(worker.stats['max_outstanding_preparation_batches'], depth + 1)
            finally:
                worker.close()

    def test_current_plus_future_bound_and_cpu_only_prep_and_main_forward(self):
        for depth in (1, 2, 4):
            worker = tuned_fixture(depth)
            consumed, preparing_threads, forward_threads = [], [], []
            next_started = threading.Event()
            original_prepare, original_forward = worker._prepare_batch, worker._infer_prepared
            main_thread = threading.get_ident()
            def assets():
                for i in range(100):
                    consumed.append(i)
                    yield {'sha256': str(i)}
            def prepare(batch, names):
                preparing_threads.append(threading.get_ident())
                if batch[0]['sha256'] == '2':
                    next_started.set()
                return original_prepare(batch, names)
            def forward(prepared, names):
                forward_threads.append(threading.get_ident())
                self.assertTrue(next_started.wait(3))
                return original_forward(prepared, names)
            worker._prepare_batch, worker._infer_prepared = prepare, forward
            stream = worker.iter_batches(assets())
            try:
                next(stream)
                self.assertEqual(len(consumed), 2 * (depth + 1))
                self.assertEqual(forward_threads, [main_thread])
                self.assertTrue(all(t != main_thread for t in preparing_threads))
                stream.close()
                self.assertEqual(len(consumed), 2 * (depth + 1))
                self.assertFalse(worker._active_stream)
            finally:
                stream.close(); worker.close()

    def test_forward_explicit_device_context_restored_even_on_failure(self):
        worker = TunedResidentInferenceWorker.__new__(TunedResidentInferenceWorker)
        state = {'device': 'npu:7'}
        calls = []
        @contextmanager
        def device_context(device):
            previous = state['device']; state['device'] = device
            calls.append(('enter', threading.get_ident()))
            try:
                yield
            finally:
                state['device'] = previous
                calls.append(('exit', threading.get_ident()))
        worker.device = 'npu:2'
        worker.torch = SimpleNamespace(npu=SimpleNamespace(device=device_context))
        def forward(*args):
            self.assertEqual(state['device'], 'npu:2')
            raise RuntimeError('fixture device failure')
        with patch.object(OptimizedResidentInferenceWorker, '_infer_prepared', forward):
            with self.assertRaisesRegex(RuntimeError, 'fixture device failure'):
                worker._infer_prepared([], ('sscd',))
        self.assertEqual(state['device'], 'npu:7')
        self.assertEqual(calls, [('enter', threading.get_ident()), ('exit', threading.get_ident())])

    def test_source_exception_cancels_prefetch_and_worker_remains_reusable(self):
        worker = tuned_fixture(2)
        def broken():
            yield {'sha256': '0'}
            yield {'sha256': '1'}
            raise ValueError('source failed')
        try:
            with self.assertRaisesRegex(ValueError, 'source failed'):
                list(worker.iter_results(broken()))
            self.assertFalse(worker._active_stream)
            self.assertEqual(len(worker.infer_batch([{'sha256': '5'}])), 1)
        finally:
            worker.close()


if __name__ == '__main__':
    unittest.main()
