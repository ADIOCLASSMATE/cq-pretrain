"""Accelerator backend selection: CUDA (default) or Ascend NPU.

Chosen once at import. PRETRAIN_BACKEND forces it; otherwise torch_npu presence
decides. All model code keeps its existing ``device.type`` checks; this only
drives distributed init, device-module and the GDN kernel dispatch.
"""
import os
import torch


def _detect():
    b = os.environ.get("PRETRAIN_BACKEND", "").lower()
    if b in ("cuda", "npu"):
        return b
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
    return "npu" if IS_NPU else "cuda"
