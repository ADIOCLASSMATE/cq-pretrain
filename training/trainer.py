"""One optimizer update: microbatch forward/backward, gradient sync, optimizer and EMA."""

from contextlib import nullcontext
import math
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

import distributed.accel as accel
import distributed.megatron as P
import utils.checkpoint as checkpoint_io
from training.loss import compose_losses
from training.options import data_loss_scales, learning_rate
from training.renderer_schedule import backward_deferred_renderer

ACCEL = accel.accel_type()
DEV = accel.device_module()


class Trainer:
    def __init__(self, args, cfg, run, model, wrapped, optimizer, ema, pipeline=None, synthetic_batch=None):
        self.args, self.cfg, self.run = args, cfg, run
        self.model, self.wrapped, self.optimizer, self.ema = model, wrapped, optimizer, ema
        self.pipeline, self.synthetic_batch = pipeline, synthetic_batch
        self.ordered_buckets = None
        if pipeline is not None:
            from distributed.communication import OrderedGradientBuckets, OrderedParameterGathers

            self.ordered_buckets = OrderedGradientBuckets(wrapped)
            # Registers ordered parameter-gather hooks on construction.
            self.ordered_gathers = OrderedParameterGathers(wrapped)

    def _set_learning_rate(self, step):
        lr = learning_rate(step, self.args)
        for opt in checkpoint_io._optimizers(self.optimizer):
            for group in opt.optimizer.param_groups:
                group["lr"] = lr * group.get("lr_mult", 1.0)
        return lr

    def step(self, step, completed_step, on_epoch_end):
        """Run update number `step` (zero based). Returns None when the data is exhausted."""
        args = self.args
        lr = self._set_learning_rate(step)
        self.optimizer.zero_grad()
        self.wrapped.zero_grad_buffer()
        if self.ordered_buckets:
            self.ordered_buckets.begin()
        DEV.reset_peak_memory_stats()
        begin = self.last_begin = time.perf_counter()
        result = SimpleNamespace(
            lr=lr,
            begin=begin,
            counts=None,
            micro_batches=None,
            receipts=[],
            data_wait_seconds=0.0,
            data_prepare_seconds=0.0,
            phase_events=[],
            renderer_forward_events=[],
            loss_totals=[torch.zeros((), device=ACCEL, dtype=torch.float32) for _ in range(3)],
        )
        if self.pipeline:
            micro_batches, counts, wait = self.pipeline.next_group(
                self.run.rank, completed_step, self.run.output, on_epoch_end
            )
            result.counts, result.data_wait_seconds = counts, wait
            if micro_batches is None:
                return None
            result.micro_batches = micro_batches
            accumulation_steps = len(micro_batches)
        else:
            accumulation_steps = args.grad_accum
        for micro_step in range(accumulation_steps):
            self._microbatch(result, micro_step, accumulation_steps)
        if self.ordered_buckets:
            self.ordered_buckets.finish()
        P.finish_grad_sync(self.wrapped)
        self._optimizer_step(result)
        DEV.synchronize()
        result.data_device_wait_seconds = self.pipeline.device_wait_seconds() if self.pipeline else 0.0
        result.local_seconds = time.perf_counter() - begin
        if args.phase_timing:
            self._phase_times(result)
        self._reduce(result)
        for receipt in result.receipts:
            self.pipeline.acknowledge(receipt)
        return result

    def _microbatch(self, result, micro_step, accumulation_steps):
        args, cfg = self.args, self.cfg
        scales = None
        if self.pipeline:
            item = result.micro_batches[micro_step]
            begin = time.perf_counter()
            if item is None:
                batch, scales = self.pipeline.padding_batch, (0.0, 0.0)
            else:
                cpu_batch, receipt = item
                counts = result.counts
                scales = data_loss_scales(
                    cpu_batch, counts["supervised_text_tokens"], counts["generation_images"], self.run.dp
                )
                batch = self.pipeline.acquire(item)
                result.receipts.append(receipt)
            result.data_prepare_seconds += time.perf_counter() - begin
        else:
            batch = self.synthetic_batch
        # Real data scales each microbatch by its global share; synthetic data averages over GA.
        divisor = 1 if self.pipeline else args.grad_accum
        last = micro_step + 1 == accumulation_steps
        context = nullcontext() if last else self.wrapped.no_sync()
        with context:
            if args.phase_timing:
                events = [DEV.Event(enable_timing=True) for _ in range(3)]
                result.renderer_forward_events.append([])
                events[0].record()
            deferred = bool(batch.generation is not None and batch.generation.microbatches)
            outputs = self.wrapped(batch, reference=args.split_reference, compute_image_loss=not deferred)
            losses = compose_losses(outputs, cfg, scales)
            if self.ordered_buckets:
                self._mark_absent_buckets(batch)
            if args.phase_timing:
                events[1].record()
            if deferred:
                backward_deferred_renderer(
                    self.model,
                    batch,
                    losses,
                    1.0 if self.pipeline else 1 / args.grad_accum,
                    self.wrapped,
                    result.renderer_forward_events[-1] if args.phase_timing else None,
                    image_factor=cfg.image_weight * (scales[1] if scales is not None else 1.0),
                )
            else:
                (losses["loss"] / divisor).backward()
            if args.phase_timing:
                events[2].record()
                result.phase_events.append(events)
        # The objective accumulates every microbatch. Logging only the last
        # scaled loss would shrink it by GA and misweight layouts.
        text_scale, image_scale = scales if scales is not None else (1 / args.grad_accum, 1 / args.grad_accum)
        totals = result.loss_totals
        totals[0].add_(losses["loss"].detach().float() / divisor)
        totals[1].add_(losses["text_loss"].detach().float() * text_scale)
        totals[2].add_(losses["image_loss"].detach().float() * image_scale)

    def _mark_absent_buckets(self, batch):
        from distributed.communication import finish_parameter_gathers

        buckets = self.ordered_buckets
        finish_parameter_gathers(self.wrapped)
        if not len(batch.image_indices):
            buckets.absent_images()
        elif not batch.generation_image_count:
            buckets.absent_renderer()
        if not batch.text_indices.numel():
            buckets.absent_text()

    def _optimizer_step(self, result):
        if self.args.phase_timing:
            result.optimizer_events = [DEV.Event(enable_timing=True) for _ in range(2)]
            result.optimizer_events[0].record()
        updated, grad_norm, _ = self.optimizer.step()
        if self.ema is not None and updated and math.isfinite(float(grad_norm)):
            self.ema.update()
        if self.args.phase_timing:
            result.optimizer_events[1].record()
        result.updated, result.grad_norm = updated, grad_norm

    @staticmethod
    def _phase_times(result):
        # DiT forward/backward interleave in the staged schedule. Add its
        # measured forward time to forward_ms, not backward_ms.
        renderer = [sum(a.elapsed_time(b) for a, b in pairs) for pairs in result.renderer_forward_events]
        result.renderer_forward_ms = renderer
        result.pack_forward_ms = [
            e[0].elapsed_time(e[1]) + r for e, r in zip(result.phase_events, renderer, strict=True)
        ]
        result.pack_backward_ms = [
            e[1].elapsed_time(e[2]) - r for e, r in zip(result.phase_events, renderer, strict=True)
        ]

    def _reduce(self, result):
        """Step time and peak memory are maxima over ranks; losses are DP means."""
        elapsed = torch.tensor(result.local_seconds, device=ACCEL, dtype=torch.float32)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        result.seconds = elapsed.item()
        result.local_memory = DEV.max_memory_allocated()
        memory = torch.tensor(result.local_memory, device=ACCEL, dtype=torch.int64)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        result.memory = memory.item()
        values = torch.stack(result.loss_totals)
        dist.all_reduce(values, group=P.state().get_data_parallel_group())
        values /= self.run.dp
        finite = torch.isfinite(values).all().item() and math.isfinite(float(result.grad_norm))
        if not finite or not result.updated:
            raise FloatingPointError("Nonfinite loss or skipped optimizer update")
        result.values = values.tolist()
