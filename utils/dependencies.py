"""Explicit versioned runtime selection; never silently fall back to an old stack."""

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


# One pinned stack per accelerator. Nothing falls back: a wrong stack is a
# different set of kernels, so it must be named explicitly and recorded.
STACKS = dict(cuda="cuda", npu="core17")


def name():
    import distributed.accel as accel

    default = STACKS[accel.BACKEND]
    value = os.environ.get("PRETRAIN_STACK", default)
    if value != default:
        raise ValueError(f"Backend {accel.BACKEND} requires the pinned {default!r} stack; got {value!r}")
    return value


def source_directories():
    """Vendor checkouts the Ascend stack puts on sys.path (CANN 9.0 / MCore 0.17)."""
    deps = PACKAGE / ".deps" / name()
    return [deps / component for component in ("Megatron-LM", "MegatronAdaptor", "TransformerEngineNPU", "MindSpeed")]


def lock_path():
    return PACKAGE / "configs" / f"deps.{name()}.lock.json"


def installed_versions():
    """Record the actual environment alongside the declared dependency lock."""
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for package in (
        "torch",
        "triton",
        "pytorch-triton",
        "flash-linear-attention",
        "fla-core",
        "megatron-core",
        "apex",
        "transformer-engine",
        "transformers",
        "diffusers",
        "pyarrow",
        "av",
        "torch_npu",
        "mindspeed",
        "megatron_adaptor",
        "flash-linear-attention-npu-a2",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
