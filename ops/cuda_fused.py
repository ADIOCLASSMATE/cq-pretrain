"""Published CUDA norm kernels, wired to this repo's existing dispatch points.

No kernel is written here. ``rms_norm`` uses flash-attn's Triton RMSNorm and
``gated_rms_norm`` uses the same upstream ``fla.modules.fused_norm_gate``
entry point the Ascend path already calls, so both sides of the port invoke the
same published implementation. Normalization is over the last dimension only;
every caller keeps its own weight convention (the backbone passes ``1+weight``).
"""

import os

_ENABLED = os.environ.get("PRETRAIN_CUDA_FUSED_NORM", "1") == "1"


def enabled(x, fused=True):
    import torch

    return (
        _ENABLED
        and fused
        and x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not torch.are_deterministic_algorithms_enabled()
    )


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
