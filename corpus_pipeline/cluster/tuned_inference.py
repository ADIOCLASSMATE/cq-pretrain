"""Tunable, bounded CPU feeding around the qualified inference implementation.

This module changes scheduling only. Reference preprocessing, model arithmetic,
precision, and model identity are inherited unchanged. Production use requires a
separate execution qualification; no production launcher imports this module.
PyTorch CPU thread count is process-global: independent simultaneous configurations
must use distinct processes rather than constructing workers in separate threads.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
import time

from .optimized_inference import OptimizedResidentInferenceWorker
from corpus_pipeline.production.common import file_sha


class TunedResidentInferenceWorker(OptimizedResidentInferenceWorker):
    """One device, one caller, separately sized CPU pool and bounded prefetch.

    decode_workers: 1..32 concurrent original image preparations.
    cpu_threads: 1..16 PyTorch intra-op CPU threads per process.
    prefetch_batches: 1..4 future batches, in addition to the current batch.
    """

    def __init__(self, *args, decode_workers=8, cpu_threads=2, prefetch_batches=2, fast_prepare=False, **kwargs):
        if 'threads' in kwargs:
            raise ValueError('use separate decode_workers and cpu_threads, not threads')
        batch_size = kwargs.get('batch_size', 32)
        self._validate_configuration(batch_size, decode_workers, cpu_threads, prefetch_batches)
        if type(fast_prepare) is not bool:
            raise ValueError('fast_prepare must be a boolean')
        super().__init__(*args, threads=cpu_threads, **kwargs)
        self.configure(batch_size=batch_size, decode_workers=decode_workers,
                       cpu_threads=cpu_threads, prefetch_batches=prefetch_batches, fast_prepare=fast_prepare)
        self.stats.update(max_outstanding_preparation_batches=0)

    @staticmethod
    def _validate_configuration(batch_size, decode_workers, cpu_threads, prefetch_batches):
        for name, value, maximum in (('batch_size', batch_size, 1024),
                                      ('decode_workers', decode_workers, 32),
                                      ('cpu_threads', cpu_threads, 16),
                                      ('prefetch_batches', prefetch_batches, 4)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in 1..{maximum}')

    def configure(self, *, batch_size, decode_workers, cpu_threads, prefetch_batches, fast_prepare=False):
        """Reuse resident models between serial, separately measured configurations.

        Reject an active/closed stream. CPU thread count is process-global; this
        method is intentionally not a cross-worker or multi-thread controller.
        Counters remain cumulative; max_outstanding_preparation_batches is a
        lifetime high-water mark, so it must not be treated as an additive timer.
        """
        self._validate_configuration(batch_size, decode_workers, cpu_threads, prefetch_batches)
        if type(fast_prepare) is not bool:
            raise ValueError('fast_prepare must be a boolean')
        if self.closed or self._active_stream:
            raise RuntimeError('configure requires an open idle inference worker')
        prepare_impl, prepare_sha = None, None
        if fast_prepare:
            from .fast_prepare import prepare_asset
            prepare_impl = prepare_asset
            prepare_sha = file_sha(Path(__file__).with_name('fast_prepare.py'))
        new_pool = ThreadPoolExecutor(max_workers=decode_workers, thread_name_prefix='image-decode')
        try:
            self.torch.set_num_threads(cpu_threads)
        except BaseException:
            new_pool.shutdown(wait=True, cancel_futures=True)
            raise
        old_pool = self.pool
        self.pool = new_pool
        old_pool.shutdown(wait=True, cancel_futures=True)
        self.batch_size = batch_size
        self.decode_workers, self.cpu_threads = decode_workers, cpu_threads
        self.prefetch_batches = prefetch_batches
        self.fast_prepare_enabled, self._fast_prepare_impl = fast_prepare, prepare_impl
        self.execution_identity = {
            **self.execution_identity,
            'version': 'tuned_image_inference.v1',
            'optimized_inference_code_sha256': file_sha(Path(__file__).with_name('optimized_inference.py')),
            'tuned_inference_code_sha256': file_sha(__file__),
            'batch_size': batch_size,
            'decode_workers': decode_workers, 'cpu_threads': cpu_threads,
            'prefetch_batches': prefetch_batches,
            'fast_prepare': fast_prepare, 'fast_prepare_code_sha256': prepare_sha,
            'maximum_prepared_or_preparing_batches': prefetch_batches + 1,
            'explicit_caller_device_context': True,
        }
        return dict(self.execution_identity)

    def _prepare(self, asset, names):
        if self.fast_prepare_enabled:
            return self._fast_prepare_impl(self, asset, names)
        return super()._prepare(asset, names)

    def _infer_prepared(self, prepared, names):
        # CPU preparation threads never enter a device context. Explicitly bind
        # the caller while forwarding, then restore any previous caller device.
        context = (self.torch.npu.device(self.device)
                   if str(self.device).startswith('npu') else nullcontext())
        with context:
            return super()._infer_prepared(prepared, names)

    def iter_batches(self, assets, *, models=None):
        names = self._names(models)
        if self._active_stream:
            raise RuntimeError('inference worker is single-consumer')
        self._active_stream = True
        started = time.perf_counter()
        iterator, pending, exhausted = iter(assets), deque(), False
        capacity = self.prefetch_batches + 1

        def fill():
            nonlocal exhausted
            while not exhausted and len(pending) < capacity:
                batch = list(islice(iterator, self.batch_size))
                if not batch:
                    exhausted = True
                    break
                pending.append((batch, self.coordinator.submit(self._prepare_batch, batch, names)))
                self.stats['max_outstanding_preparation_batches'] = max(
                    self.stats.get('max_outstanding_preparation_batches', 0), len(pending))

        try:
            fill()
            while pending:
                batch, future = pending.popleft()
                wait_started = time.perf_counter()
                prepared, preparation_seconds = future.result()
                self.stats['preparation_wait_seconds'] += time.perf_counter() - wait_started
                self.stats['decode_seconds'] += preparation_seconds
                # A finished Future owns its result; release it before yielding
                # so only current tensors and bounded future batches can live.
                del future
                predictions = self._infer_prepared(prepared, names)
                del prepared
                yield batch, predictions
                # Current tensors have been released. Refill before selecting
                # the next current batch, keeping capacity=N+1 at all times.
                fill()
        finally:
            for _, future in pending:
                future.cancel()
            for _, future in pending:
                if not future.cancelled():
                    try:
                        future.result()
                    except Exception:
                        pass  # Keep original inference/preparation/consumer error.
            pending.clear()
            self.stats['stream_seconds'] += time.perf_counter() - started
            self._active_stream = False
