"""Bounded H2D/latent prefetch on a separate device stream, with safe lifetimes."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
import time

import torch


def tensors(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name), seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item, seen)


@dataclass
class DeviceItem:
    batch: object
    receipt: dict
    ready: object
    host_keepalive: tuple
    prepare_seconds: float
    future: object = None

    def __getitem__(self, index):
        return (self.batch, self.receipt)[index]


class DeviceLoader:
    """Consumption is acknowledged only after a successful optimizer update.

    next() retrieves an enqueued item without blocking on the accelerator.
    acquire() inserts its event dependency immediately before its forward,
    allowing GA microbatches to overlap with preparation of later items.
    """
    def __init__(self, host_loader, preparer, prefetch=2):
        if prefetch < 0:
            raise ValueError('Device prefetch must be nonnegative; zero is the synchronous control')
        self.host, self.preparer = host_loader, preparer
        self.limit, self.pending = prefetch, deque()
        self.is_npu = preparer.device.startswith('npu')
        self.stream = torch.npu.Stream(device=preparer.device) if self.is_npu and prefetch else None
        if self.stream is not None:
            self.stream.wait_stream(torch.npu.current_stream(preparer.device))
        self.closed, self.stopped = False, False
        self.wait_events = []
        self.stats = dict(host_wait_seconds=0., acquired=0, not_ready_on_acquire=0,
                          max_queued=0, prepare_enqueue_seconds=0.,acquire_wait_seconds=0.,
                          future_submit_seconds=0.)
        def initialize():
            if self.is_npu:
                torch.npu.set_device(preparer.device)
        self.pool = ThreadPoolExecutor(max_workers=1, initializer=initialize,
                                       thread_name_prefix='latent-prefetch') if prefetch else None
        self.waiting = deque()
        self.inflight = deque()

    def _prepare(self, item):
        host_batch, receipt = item.batch, item.receipt
        begin = time.perf_counter()
        if self.stream is not None:
            with torch.npu.stream(self.stream):
                batch, randoms = self.preparer.prepare(host_batch, receipt)
                ready = torch.npu.Event()
                ready.record(self.stream)
        else:
            batch, randoms = self.preparer.prepare(host_batch, receipt)
            ready = None
        return DeviceItem(batch, receipt, ready, (host_batch, randoms), time.perf_counter() - begin)

    def _fill(self):
        begin=time.perf_counter()
        while self.pool is not None and self.waiting and len(self.inflight) < self.limit:
            item=self.waiting.popleft()
            item.future=self.pool.submit(self._prepare,item)
            self.inflight.append(item)
        self.stats['max_queued'] = max(self.stats['max_queued'], len(self.inflight))
        self.stats['future_submit_seconds']+=time.perf_counter()-begin

    def next_group(self, count):
        """Collect CPU metadata for exact loss normalization, bound device lookahead.

        Every update consumes exactly `count` packs, so the effective batch is
        identical across updates and DP replicas. Gradient accumulation must
        not expand the NPU queue to the whole update: only `limit` upcoming
        microbatches have device preparations in flight.
        """
        if self.waiting or self.inflight:
            raise RuntimeError('Acquire every previous group item before requesting another')
        if count < 1:
            raise ValueError('Require a positive microbatch count')
        begin = time.perf_counter()
        group = []
        for _ in range(count):
            pair = None if self.closed else self.host.next()
            if pair is None:
                self.closed = True
                break
            item = DeviceItem(pair[0], pair[1], None, (), 0.)
            group.append(item)
            self.waiting.append(item)
        group.extend([None]*(count-len(group)))
        self.stats['host_wait_seconds'] += time.perf_counter() - begin
        self._fill()
        return group

    def acquire(self, item):
        begin=time.perf_counter()
        if self.pool is not None:
            if self.inflight[0] is not item:raise ValueError('Acquire device batches in order')
            prepared=item.future.result()
            item.future=None
            self.inflight.popleft()
        else:
            if self.waiting.popleft() is not item:raise ValueError('Acquire device batches in order')
            prepared=self._prepare(item)
        self.stats['acquire_wait_seconds']+=time.perf_counter()-begin
        # Retain pinned transfer sources until the consumer releases this item.
        item.host_keepalive=prepared.host_keepalive
        self.stats['prepare_enqueue_seconds']+=prepared.prepare_seconds
        if prepared.ready is not None:
            stream = torch.npu.current_stream(self.preparer.device)
            self.stats['not_ready_on_acquire'] += int(not prepared.ready.query())
            start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            start.record(stream)
            stream.wait_event(prepared.ready)
            end.record(stream)
            self.wait_events.append((start, end))
            # Objects can leave Python scope before forward/backward finishes.
            # Record every plan/vision/batch tensor on the consuming stream.
            for tensor in tensors(prepared.batch):
                if tensor.device.type == 'npu' and tensor.numel():
                    tensor.record_stream(stream)
        self.stats['acquired'] += 1
        self._fill()
        return prepared.batch

    def device_wait_seconds(self):
        """Call only after the caller's normal end-of-step synchronization."""
        seconds = sum(start.elapsed_time(end) for start, end in self.wait_events) / 1000
        self.wait_events.clear()
        return seconds

    def acknowledge(self, receipt):
        self.host.acknowledge(receipt)

    def state_dict(self):
        return self.host.state_dict()

    def close(self):
        if self.stopped:
            return
        self.closed = self.stopped = True
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.stream is not None:
            self.stream.synchronize()
        self.host.close()
        self.pending.clear()
        self.inflight.clear()
        self.waiting.clear()
