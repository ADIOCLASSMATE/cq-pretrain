"""Published CUDA norm kernels, wired to this repo's existing dispatch points.

No kernel is written here. ``rms_norm`` uses flash-attn's Triton RMSNorm and
``gated_rms_norm`` uses the same upstream ``fla.modules.fused_norm_gate``
entry point the Ascend path already calls, so both sides of the port invoke the
same published implementation. Normalization is over the last dimension only;
every caller keeps its own weight convention (the backbone passes ``1+weight``).
"""

import functools
import os

_ENABLED = os.environ.get("PRETRAIN_CUDA_FUSED_NORM", "1") == "1"


def enabled(x, fused=True):
    import torch

    if not (
        _ENABLED
        and fused
        and x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not torch.are_deterministic_algorithms_enabled()
    ):
        return False
    return _kernels_present()


@functools.lru_cache(maxsize=1)
def _kernels_present():
    """Both kernels ship with the pinned image; never fail a step over a import."""
    try:
        from flash_attn.ops.triton.layer_norm import rms_norm_fn  # noqa: F401
        from fla.modules.fused_norm_gate import rms_norm_gated  # noqa: F401
    except Exception:
        return False
    return True

def rms_norm(x, weight, eps):
    """x * rsqrt(mean(x^2)+eps) * weight, fused; weight may be FP32 with BF16 x."""
    from flash_attn.ops.triton.layer_norm import rms_norm_fn

    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    out = rms_norm_fn(flat, weight, None, eps=eps)
    return out.reshape(shape)


def gated_rms_norm(x, gate, weight, eps):
    """rmsnorm(x) * weight * silu(gate), fused by upstream FLA."""
    from fla.modules.fused_norm_gate import rms_norm_gated

    shape = x.shape
    out = rms_norm_gated(
        x.reshape(-1, shape[-1]),
        gate.reshape(-1, shape[-1]),
        weight,
        None,
        activation="swish",
        eps=eps,
    )
    return out.reshape(shape)


_OPS_ENABLED = os.environ.get("PRETRAIN_CUDA_FUSED_OPS", "0") == "1"
# The SwiGLU fusion carries its own switch: sharing one with the causal
# convolution would make any A/B measure both and leave run.json unable to
# attribute the result to either.
_SWIGLU_ENABLED = os.environ.get("PRETRAIN_CUDA_FUSED_SWIGLU", "0") == "1"


def ops_enabled(x, fused=True):
    """Gate for the fused SwiGLU / causal convolution below."""
    import torch

    return (
        _OPS_ENABLED
        and fused
        and x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16)
        and not torch.are_deterministic_algorithms_enabled()
    )


# Upstream also ships a fused SwiGLU. It is deliberately not wired here: it
# reads its two inputs as contiguous, while this MLP's gate/up are strided
# views of one packed gate_up projection, so passing them directly returns
# silently wrong values (relative error 1.414, no exception raised). Copying
# them first is correct but leaves only 1.15x over the reference pair, which
# is not worth a kernel that fails quietly on this model's real layout.
# Measured in evaluations/benchmarks/cuda_l40s_tuning_20261117/fused_ops.json.


def causal_conv1d(x, weight, cu_seqlens):
    """Depthwise causal convolution + silu over packed documents.

    x is [1, T, D]; weight is [D, W]; cu_seqlens holds the same document
    boundaries the reference loop iterates over, so documents stay independent
    and no token reads across a document edge.
    """
    from fla.modules.convolution import causal_conv1d as _conv

    return _conv(x, weight=weight, activation="silu", cu_seqlens=cu_seqlens)[0]


@functools.lru_cache(maxsize=1)
def _compiled_swiglu():
    import torch
    from torch.nn import functional as F

    def swiglu(projected):
        gate, up = projected.chunk(2, -1)
        return F.silu(gate) * up

    # Packed token counts vary per batch; static shapes would recompile for
    # every new length. Inductor reads both strided halves in one pass.
    return torch.compile(swiglu, dynamic=True, fullgraph=True)


def swiglu_enabled(x):
    """True when the Inductor SwiGLU fusion should replace the reference pair."""
    import torch

    return (
        _SWIGLU_ENABLED
        and x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not torch.are_deterministic_algorithms_enabled()
    )


def swiglu(projected):
    """silu(gate) * up over a packed [.., 2*ffn] projection, fused by Inductor.

    Same expression as the reference below it in models.backbone_qwen35.MLP;
    this only changes how many passes over memory it takes.
    """
    return _compiled_swiglu()(projected)
