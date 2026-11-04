"""Separate entry: bounded 64-rank benchmark followed by 3-step profiling.

The unprofiled benchmark runs first. Only rank 0 records; all 64 ranks execute
the same workload/collectives. Profiling overhead is never used as throughput.
"""
from contextlib import nullcontext
import gc
import os
from pathlib import Path
import traceback

import torch
import torch.distributed as dist

import scripts.validation.delta64_probe as probe
import training.mcore as P


original_benchmark = probe.benchmark


def profile_training(output, layers, length, ga):
    import torch_npu
    from torch_npu import profiler as prof
    rank = dist.get_rank()
    for hybrid in (False, True):
        kind = 'hybrid_3_1' if hybrid else 'gated_full_attention'
        torch.manual_seed(710)
        model = probe.DecoderProbe(layers, hybrid=hybrid).to(device='npu', dtype=torch.bfloat16)
        wrapped, optimizer = P.make_optimizer(model, lr=1e-4)
        x = probe.data(720 + rank, length)

        def step():
            with torch.autograd.profiler.record_function('phase_zero_grad'):
                optimizer.zero_grad(); wrapped.zero_grad_buffer()
            for micro in range(ga):
                with wrapped.no_sync() if micro + 1 < ga else nullcontext():
                    with torch.autograd.profiler.record_function('phase_forward'):
                        y = wrapped(x).float()
                        loss = (y.square().mean() + .1 * y.mean()) / ga
                    with torch.autograd.profiler.record_function('phase_backward_including_recompute'):
                        loss.backward()
            with torch.autograd.profiler.record_function('phase_finish_grad_sync'):
                P.finish_grad_sync(wrapped)
            with torch.autograd.profiler.record_function('phase_optimizer'):
                updated, norm, _ = optimizer.step()
            with torch.autograd.profiler.record_function('phase_param_sync'):
                wrapped.start_param_sync(force_sync=True)
            assert updated

        for _ in range(2):
            step()
        torch.npu.synchronize(); dist.barrier()
        if rank == 0:
            probe.log('profiling_start', kind=kind, layers=layers, length=length, steps=3)
            config = prof._ExperimentalConfig(
                profiler_level=prof.ProfilerLevel.Level1,
                aic_metrics=prof.AiCMetrics.PipeUtilization,
                data_simplification=False)
            recorder = prof.profile(
                activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                schedule=prof.schedule(wait=0, warmup=0, active=3, repeat=1),
                on_trace_ready=prof.tensorboard_trace_handler(str(Path(output) / 'profiling' / kind)),
                record_shapes=True, profile_memory=True, with_stack=False,
                experimental_config=config)
        else:
            recorder = nullcontext()
        with recorder as trace:
            for index in range(3):
                step()
                torch.npu.synchronize()
                if rank == 0:
                    trace.step()
                    probe.log('profiling_step', kind=kind, step=index)
        dist.barrier()
        if rank == 0:
            probe.log('profiling_done', kind=kind)
        wrapped.disable_forward_pre_hook()
        del wrapped, optimizer, model, x, step, recorder, trace
        gc.collect(); torch.npu.empty_cache()


def benchmark_and_profile(output, lengths, layers, warmup, steps, ga):
    result = original_benchmark(output, lengths, layers, warmup, steps, ga)
    profile_training(output, layers, max(lengths), ga)
    return result


if __name__ == '__main__':
    probe.benchmark = benchmark_and_profile
    try:
        probe.main()
    except Exception:
        probe.log('failed', traceback=traceback.format_exc())
        raise
