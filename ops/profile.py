"""Step profiling, dispatched to the selected accelerator's profiler.

Both backends expose the same three calls, so the training loop never names a
vendor profiler: ``check_output_path`` (before the model is built),
``profiler`` (the context manager the loop steps) and ``finalize`` (parsing
after the collectives have ended).
"""

import distributed.accel as accel

if accel.BACKEND == "npu":
    from ops import npu_profile as _backend
else:
    from ops import cuda_profile as _backend


def check_output_path(path):
    return _backend.check_output_path(path)


def profiler(output, warmup):
    return _backend.profiler(output, warmup)


def finalize(output):
    return _backend.finalize(output)
