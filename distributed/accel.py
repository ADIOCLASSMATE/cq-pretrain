"""The single place this branch names its accelerator.

Only CUDA is supported here; the Ascend NPU backend and its HCCL/MindSpeed
stack were removed. Call sites go through ``device_module()`` rather than
hard-coding ``torch.cuda`` so the indirection survives if that ever changes.
"""

import torch

BACKEND = "cuda"


def device_module():
    return torch.cuda


def dist_backend():
    return "nccl"


def accel_type():
    return "cuda"
