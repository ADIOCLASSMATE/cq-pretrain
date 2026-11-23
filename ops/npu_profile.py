"""Ascend step profiling: torch_npu.profiler, parsed after the collectives end.

Level1 records the operator breakdown this project reads (op_statistic.csv):
which kernel owns how much device time per step. The trace is written during
training with ``analyse_flag=False`` and parsed in ``finalize`` once every rank
has left its collectives -- synchronous mid-step parsing stalls the profiled
rank long enough for the others to hit the HCCL execution timeout.
"""

import os

import torch_npu

# Python stacks are what turn an operator breakdown into a list of lines to fix,
# but they are recorded per dispatch and this model issues about two million of
# them per update: one active step is already a multi-gigabyte trace, so the
# window narrows when they are requested.
_WITH_STACK = os.environ.get("PRETRAIN_PROFILE_STACK", "0") == "1"


def profiler(output, warmup):
    return torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=warmup, warmup=1, active=1 if _WITH_STACK else 2, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output), analyse_flag=False),
        record_shapes=True,
        profile_memory=True,
        with_stack=_WITH_STACK,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
    )


def check_output_path(path):
    """The profiler swallows callback-construction errors.

    Reject an output path its parser cannot accept before the training model is
    built, rather than discovering it after a full run.
    """
    from torch_npu.utils._path_manager import PathManager

    PathManager._input_path_common_check(str(path))


def finalize(output):
    from pathlib import Path

    from torch_npu.profiler.profiler import analyse

    analyse(str(output), max_process_number=4)
    if not list(Path(output).glob("*/ASCEND_PROFILER_OUTPUT/op_statistic.csv")):
        raise RuntimeError(f"Profiling did not export {output}; inspect the offline parser errors")
