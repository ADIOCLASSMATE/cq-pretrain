"""CUDA step profiling: upstream torch.profiler with the CUDA activity."""

import torch


def profiler(output, warmup):
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=warmup, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def check_output_path(path):
    """Nothing to reject here; the Ascend parser is the stricter one."""


def finalize(output):
    """Traces are written by the trace handler as the profiler exits."""
