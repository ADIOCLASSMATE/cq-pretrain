"""Ascend 910B norm/activation/convolution kernels behind this repo's dispatch.

No kernel is written here. Normalization and SwiGLU call TorchNPU's published
operators; the gated norm and the packed causal convolution call the pinned
AscendC GDN backend (``ops.npu_gated_delta``), which is the same backend the
recurrence itself uses. The function names, arguments and gates mirror
``ops.cuda_fused`` so ``ops.fused`` can select a backend without either side
knowing about the other.
"""

import functools
import os

import torch

import ops.audit as audit

_ENABLED = os.environ.get("PRETRAIN_NPU_FUSED_NORM", "1") == "1"
_OPS_ENABLED = os.environ.get("PRETRAIN_NPU_FUSED_OPS", "1") == "1"
_SWIGLU_ENABLED = os.environ.get("PRETRAIN_NPU_FUSED_SWIGLU", "1") == "1"
# The fused gated norm wins on full 8K packs and loses on short ones; measured
# on A2 at 131072 rows (8K x 16 heads). Below the crossover the kernel-level
# RMSNorm still runs, with the gate applied eagerly.
_GATED_ROW_FLOOR = int(os.environ.get("PRETRAIN_NPU_GATED_NORM_ROW_FLOOR", "131072"))


def _usable(x, fused=True):
    return (
        fused
        and x.device.type == "npu"
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and _kernels_present()
    )


@functools.lru_cache(maxsize=1)
def _kernels_present():
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return True


def enabled(x, fused=True):
    return _ENABLED and _usable(x, fused)


def ops_enabled(x, fused=True):
    return (
        _OPS_ENABLED
        and fused
        and x.device.type == "npu"
        and x.dtype in (torch.float16, torch.bfloat16)
        and not torch.are_deterministic_algorithms_enabled()
    )


def swiglu_enabled(x):
    return _SWIGLU_ENABLED and _usable(x)


def wgrad(total_input, grad_output, main_grad):
    """BF16 GEMM accumulated directly into FP32 through TorchNPU's ATB binding."""
    import torch_npu

    audit.record("wgrad", "npu_matmul_add_fp32")
    torch_npu._npu_matmul_add_fp32(grad_output.contiguous(), total_input.contiguous(), main_grad)


def rotary_enabled(x):
    return _usable(x)


def rotary(x, cos, sin):
    import torch_npu

    audit.record("rotary", "npu_rotary_mul")
    return torch_npu.npu_rotary_mul(x, cos, sin)


def rms_norm(x, weight, eps):
    """x * rsqrt(mean(x^2)+eps) * weight, fused by TorchNPU."""
    import torch_npu

    if torch.are_deterministic_algorithms_enabled():
        audit.record("rms_norm", "npu_rms_norm_deterministic")
        return _DeterministicRMSNorm.apply(x, weight, eps)
    audit.record("rms_norm", "npu_rms_norm")
    return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]


class _DeterministicRMSNorm(torch.autograd.Function):
    """Avoid CANN 9.0's deterministic BF16 small-head dgamma error.

    Training uses native BF16 autograd. Deterministic validation keeps the
    fused dx and reduces dgamma explicitly in FP32. Switching deterministic
    mode after a kernel is warm is not a valid probe: this runtime caches its
    initial kernel selection.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        import torch_npu

        y, rstd = torch_npu.npu_rms_norm(x, weight, epsilon=eps)
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, grad):
        import torch_npu

        x, weight, rstd = ctx.saved_tensors
        dx, _ = torch_npu.npu_rms_norm_backward(grad.contiguous(), x, weight, rstd)
        dw = (grad.float() * x.float() * rstd).reshape(-1, x.shape[-1]).sum(0)
        return dx.to(x.dtype), dw.to(weight.dtype), None


def gated_rms_norm(x, gate, weight, eps):
    """rmsnorm(x) * weight * silu(gate)."""
    from torch.nn import functional as F

    import ops.npu_gated_delta as delta

    rows = x.numel() // x.shape[-1]
    if rows >= _GATED_ROW_FLOOR:
        audit.record("gated_rms_norm", "fla_npu_rms_norm_gated")
        shape = x.shape
        out = delta.call(
            "rms_norm_gated", x.reshape(rows, shape[-1]), gate.reshape(rows, shape[-1]), weight, None, eps=eps
        )
        return out.reshape(shape)
    # Still a kernel for the normalization; only the gate multiply is eager.
    audit.record("gated_rms_norm", "npu_rms_norm+silu")
    return (rms_norm(x, weight, eps).float() * F.silu(gate.float())).to(x.dtype)


def swiglu(projected):
    """silu(gate) * up over a packed [.., 2*ffn] projection."""
    import torch_npu

    audit.record("swiglu", "npu_swiglu")
    return torch_npu.npu_swiglu(projected, dim=-1)


def causal_conv1d(x, weight, plan):
    """Depthwise causal convolution + silu over packed documents.

    x is [1, T, D]; weight is [D, W]. The AscendC kernel takes the same packed
    document boundaries the reference loop iterates over, plus their CPU copy:
    training must not read packing metadata back from the device each layer.
    """
    import ops.npu_gated_delta as delta

    audit.record("causal_conv1d", "ascendc_causal_conv1d")
    return delta.conv(x, weight, activation="silu", cu_seqlens=plan.content_cu, cu_seqlens_cpu=plan.content_bounds)[0]
