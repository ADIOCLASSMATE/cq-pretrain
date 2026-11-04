"""Bounded CPU preparation overlap around the frozen FP32 inference reference.

Only the caller thread touches NPU tensors. The coordinator prepares at most one
future batch with the reference CPU preprocessing pool. The forward below copies
the qualified reference arithmetic deliberately: changing the reference file
requires a new numerical qualification, as does changing this execution wrapper.
The model identity stays unchanged; execution identity is recorded separately.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
import time

import numpy as np

from .inference import ResidentInferenceWorker
from corpus_pipeline.production.common import file_sha


class OptimizedResidentInferenceWorker(ResidentInferenceWorker):
    """Single consumer, current batch plus one prepared future batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-prepare")
        self._active_stream = False
        self.execution_identity = {
            "version": "overlapped_image_inference.v1", "prefetch_batches": 1,
            "base_inference_code_sha256": file_sha(Path(__file__).with_name("inference.py")),
            "optimized_inference_code_sha256": file_sha(__file__),
            "model_identity_unchanged": True,
        }
        self.stats.update(preparation_wait_seconds=0.0, inference_seconds=0.0,
                          stream_seconds=0.0, input_batch_histogram={}, valid_batch_histogram={})

    def _names(self, models):
        if self.closed:
            raise RuntimeError("worker is closed")
        names = self.names if models is None else tuple(models)
        if not names or len(set(names)) != len(names) or set(names) - set(self.names):
            raise ValueError("requested models must be a unique subset of resident models")
        return names

    def _prepare_batch(self, assets, names):
        started = time.perf_counter()
        prepared = list(self.pool.map(lambda asset: self._prepare(asset, names), assets))
        # This includes I/O, decode, transform, and contention with NPU host work.
        return prepared, time.perf_counter() - started

    def _infer_prepared(self, prepared, names):
        """Exactly the frozen reference forward and validation, on caller thread."""
        started = time.perf_counter()
        valid = [(result, tensors) for result, tensors in prepared if tensors is not None]
        for name in names:
            if not valid:
                break
            start = time.perf_counter()
            batch, values = None, None
            try:
                batch = self.torch.stack([tensors[name] for _, tensors in valid]).to(device=self.device, dtype=self.dtype)
                self.synchronize()
                with self.torch.inference_mode():
                    values = self.bundles[name]["model"](batch).float()
                self.synchronize()
                if name == "sscd":
                    vectors = values.cpu().numpy()
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    if (vectors.shape != (len(valid), 512) or not np.isfinite(vectors).all()
                            or (norms < 1e-8).any()):
                        raise ValueError("invalid SSCD descriptor")
                    vectors = (vectors / norms).astype(np.float32)
                    for (result, _), vector in zip(valid, vectors):
                        result["sscd"] = {"status": "ok", "embedding": vector, "dimensions": 512}
                else:
                    logits = values.flatten()
                    if logits.shape != (len(valid),) or not self.torch.isfinite(logits).all().item():
                        raise ValueError("invalid detector logits")
                    probabilities = logits.sigmoid().cpu().tolist()
                    for (result, _), logit, prob in zip(valid, logits.cpu().tolist(), probabilities):
                        result["scores"][name] = {"status": "ok", "logit": logit, "prob": prob}
            except Exception as exc:
                failure = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:400]}
                for result, _ in valid:
                    if name == "sscd":
                        result["sscd"] = dict(failure)
                    else:
                        result["scores"][name] = dict(failure)
            finally:
                del batch, values
                self.stats["model_seconds"][name] += time.perf_counter() - start
                self.stats["model_batches"][name] += 1
        for result, _ in valid:
            result["status"] = "ok" if all((result.get("sscd", {}) if name == "sscd" else
                result["scores"].get(name, {})).get("status") == "ok" for name in names) else "incomplete"
        self.stats["batches"] += 1
        self.stats["input_images"] += len(prepared)
        self.stats["complete_images"] += sum(result["status"] == "ok" for result, _ in prepared)
        for key, count in (("input_batch_histogram", len(prepared)), ("valid_batch_histogram", len(valid))):
            histogram = self.stats[key]
            histogram[str(count)] = histogram.get(str(count), 0) + 1
        self.stats["inference_seconds"] += time.perf_counter() - started
        return [result for result, _ in prepared]

    def iter_batches(self, assets, *, models=None):
        names = self._names(models)
        if self._active_stream:
            raise RuntimeError("inference worker is single-consumer")
        self._active_stream = True
        started = time.perf_counter()
        pending = None
        iterator = iter(assets)
        try:
            batch = list(islice(iterator, self.batch_size))
            if not batch:
                return
            pending = self.coordinator.submit(self._prepare_batch, batch, names)
            while batch:
                wait_started = time.perf_counter()
                prepared, prepare_seconds = pending.result()
                self.stats["preparation_wait_seconds"] += time.perf_counter() - wait_started
                self.stats["decode_seconds"] += prepare_seconds
                pending = None
                next_batch = list(islice(iterator, self.batch_size))
                if next_batch:
                    pending = self.coordinator.submit(self._prepare_batch, next_batch, names)
                predictions = self._infer_prepared(prepared, names)
                # Do not retain current CPU tensors while consumer persists output.
                del prepared
                yield batch, predictions
                batch = next_batch
        finally:
            # Cancellation/validation errors cannot leave preparation leaking into
            # another stream or hold image buffers beyond this bounded lifetime.
            if pending is not None:
                if not pending.cancel():
                    try:
                        pending.result()
                    except Exception:
                        pass  # Preserve the original consumer/preparation error.
            self.stats["stream_seconds"] += time.perf_counter() - started
            self._active_stream = False

    def iter_results(self, assets, *, models=None):
        batches = self.iter_batches(assets, models=models)
        try:
            for _, predictions in batches:
                yield from predictions
        finally:
            batches.close()

    def infer_batch(self, assets, *, models=None):
        assets = list(islice(iter(assets), self.batch_size + 1))
        if len(assets) > self.batch_size:
            raise ValueError("infer_batch exceeds configured batch_size; use iter_results")
        return list(self.iter_results(assets, models=models))

    def close(self):
        if not self.closed:
            self.coordinator.shutdown(wait=True, cancel_futures=True)
            super().close()
