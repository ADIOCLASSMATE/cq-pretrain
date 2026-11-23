"""The single place this project names its accelerator.

Two backends are supported and they are chosen once, at import:
``cuda`` (NVIDIA, the L40S production stack) and ``npu`` (Ascend 910B with
CANN/HCCL). ``PRETRAIN_BACKEND`` forces the choice; otherwise an importable
and available ``torch_npu`` selects the NPU.

Everything else in the tree asks this module rather than naming a vendor:
``device_module()`` for streams/events/RNG, ``dist_backend()`` for the process
group, and ``accel_type()`` for the device string tensors are moved to. The
operator dispatch in ``ops.fused``/``ops.gated_delta``/``ops.attention`` keys
off ``BACKEND`` as well, so a kernel is never chosen by guessing from a tensor.
"""

import os

import torch

_VALID = ("cuda", "npu")


def _detect():
    requested = os.environ.get("PRETRAIN_BACKEND", "").lower()
    if requested in _VALID:
        return requested
    if requested:
        raise ValueError(f"PRETRAIN_BACKEND must be one of {_VALID}; got {requested!r}")
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    return "cuda"


BACKEND = _detect()
IS_NPU = BACKEND == "npu"


def device_module():
    return torch.npu if IS_NPU else torch.cuda


def dist_backend():
    return "hccl" if IS_NPU else "nccl"


def accel_type():
    return BACKEND


def on_accelerator(x):
    """True when this tensor lives on the accelerator this process trains with."""
    return x.device.type == BACKEND


def collective_version():
    """Provenance of the collective library this backend rendezvouses over."""
    if IS_NPU:
        import torch_npu

        cann = None
        try:
            from fla_npu._guard import detect_cann_version

            cann = detect_cann_version()
        except Exception:
            pass
        return dict(library="hccl", torch_npu=torch_npu.__version__, cann=cann)
    return dict(library="nccl", version=list(torch.cuda.nccl.version()))
