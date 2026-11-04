"""Local CPU admission and measurements; no model policy or platform operations."""
from __future__ import annotations

import asyncio
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import math
import os
from pathlib import Path
import threading
import time

ADMISSION_SCHEMA = "corpus.screening_admission.v1"


def cpu_limits(root=Path("/sys/fs/cgroup")):
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    result = {"affinity_cpus": affinity, "cpu_quota": None, "memory_limit_bytes": None}
    try:
        quota, period = (root / "cpu.max").read_text().split()
        result["cpu_max"] = quota + " " + period
        if quota != "max":
            result["cpu_quota"] = int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        value = (root / "memory.max").read_text().strip()
        result["memory_limit_bytes"] = int(value) if value != "max" else None
    except (OSError, ValueError):
        pass
    result["suggested_image_threads"] = max(1, min(16, affinity, math.floor(result["cpu_quota"] or affinity)))
    return result


def cgroup_sample(root=Path("/sys/fs/cgroup")):
    result = {"monotonic": time.monotonic(), "process_cpu_seconds": time.process_time()}
    try:
        result["cpu_stat"] = {key: int(value) for key, value in
                              (line.split() for line in (root / "cpu.stat").read_text().splitlines())}
        result["memory_current_bytes"] = int((root / "memory.current").read_text())
    except (OSError, ValueError):
        pass
    return result


def timing_summary(values):
    values = sorted(values)
    if not values:
        return {"samples": 0}
    return {"samples": len(values), "p50_ms": values[int((len(values) - 1) * .5)],
            "p95_ms": values[int((len(values) - 1) * .95)], "max_ms": values[-1]}


class MeasuredExecutor(ThreadPoolExecutor):
    """Bound asyncio.to_thread work and measure image preparation without changing it.

    The output image bytes still come from production.client.image_payload.
    A separate one-thread executor handles registry I/O, so that readiness
    refreshes never sit behind thousands of image-preparation tasks.
    """
    def __init__(self, workers):
        super().__init__(max_workers=workers, thread_name_prefix="screen-image")
        self.workers = workers
        self.metrics_lock = threading.Lock()
        self.counts = Counter()
        self.elapsed = deque(maxlen=4096)
        self.waits = deque(maxlen=4096)

    def submit(self, fn, /, *args, **kwargs):
        target = fn.args[0] if isinstance(fn, partial) and fn.args and getattr(fn.func, "__name__", "") == "run" else fn
        is_image = getattr(target, "__name__", "") == "image_payload"
        if not is_image:
            return super().submit(fn, *args, **kwargs)
        queued = time.monotonic()
        with self.metrics_lock:
            self.counts["submitted"] += 1
        def observed():
            started = time.monotonic()
            with self.metrics_lock:
                self.counts["running"] += 1
                self.counts["peak_running"] = max(self.counts["peak_running"], self.counts["running"])
                self.waits.append((started - queued) * 1000)
            try:
                value = fn(*args, **kwargs)
                with self.metrics_lock:
                    self.counts["successful"] += 1
                return value
            except BaseException:
                with self.metrics_lock:
                    self.counts["failed"] += 1
                raise
            finally:
                elapsed = (time.monotonic() - started) * 1000
                with self.metrics_lock:
                    self.counts["running"] -= 1
                    self.counts["completed"] += 1
                    self.counts["preparation_ms_total"] += elapsed
                    self.elapsed.append(elapsed)
        return super().submit(observed)

    def stats(self):
        with self.metrics_lock:
            value = dict(self.counts)
            value.update(threads=self.workers, queued=self.counts["submitted"] - self.counts["completed"] - self.counts["running"],
                         preparation=timing_summary(self.elapsed), queue_wait=timing_summary(self.waits))
            return value


class AdmissionControl:
    """Mutable *scheduling* limit, never part of the image/model identity.

    A missing/malformed configured file pauses new admissions instead of falling
    back to maximum concurrency. A reduction drains existing requests. No expiry
    releases occupied capacity and no change modifies image decisions.
    """
    def __init__(self, ceiling, path=None):
        self.ceiling = ceiling
        self.path = Path(path) if path else None
        self.limit = ceiling if path is None else 0
        self.status = "fixed" if path is None else "awaiting_control"
        self.sha256 = None

    def refresh(self):
        if self.path is None:
            return
        try:
            if self.path.stat().st_size > 4096:
                raise ValueError("oversized")
            raw = self.path.read_bytes()
            value = json.loads(raw)
            limit = value.get("max_inflight")
            if (value.get("schema") != ADMISSION_SCHEMA or type(limit) is not int
                    or not 0 <= limit <= self.ceiling):
                raise ValueError("invalid")
            import hashlib
            self.limit, self.status, self.sha256 = limit, "paused" if limit == 0 else "active", hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, TypeError, AttributeError):
            self.limit, self.status, self.sha256 = 0, "invalid_control_paused", None

    def stats(self):
        return {"max_inflight": self.limit, "ceiling": self.ceiling,
                "status": self.status, "control_sha256": self.sha256}


class WorkerTelemetry:
    def __init__(self, executor):
        self.executor = executor
        self.limits = cpu_limits()
        self.previous = cgroup_sample()
        self.lag = deque(maxlen=4096)

    async def watch_loop(self):
        while True:
            expected = time.monotonic() + .25
            await asyncio.sleep(.25)
            self.lag.append(max(0, time.monotonic() - expected) * 1000)

    def stats(self):
        current = cgroup_sample()
        seconds = max(.001, current["monotonic"] - self.previous["monotonic"])
        cpu_delta = {name: value - self.previous.get("cpu_stat", {}).get(name, value)
                     for name, value in current.get("cpu_stat", {}).items()}
        result = {"limits": self.limits, "window_seconds": seconds,
                  "cgroup_cpu_stat_delta": cpu_delta,
                  "cgroup_cpu_cores_used": cpu_delta.get("usage_usec", 0) / seconds / 1e6,
                  "process_cpu_cores_used": (current["process_cpu_seconds"] - self.previous["process_cpu_seconds"]) / seconds,
                  "memory_current_bytes": current.get("memory_current_bytes"),
                  "event_loop_lag": timing_summary(self.lag), "image_preparation": self.executor.stats()}
        self.previous = current
        return result
