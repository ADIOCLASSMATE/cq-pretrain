"""One dispatch surface for the fused norm/activation/convolution kernels.

The model asks for an operation; this module decides which backend provides it
(``ops.cuda_fused`` or ``ops.npu_fused``) from ``distributed.accel.BACKEND``.
Call sites never name a vendor, and the two backends never import each other.

Every gate here returns False for a CPU tensor, which is what routes the
correctness tests and the CQ oracle to the reference expressions in
``models.backbone_qwen35``. On an accelerator, a reference path taken anyway is
recorded by ``ops.audit`` and, under ``PRETRAIN_REQUIRE_FUSED=1``, raises.
"""

import distributed.accel as accel

if accel.BACKEND == "npu":
    from ops import npu_fused as _backend
else:
    from ops import cuda_fused as _backend


def wgrad(total_input, grad_output, main_grad):
    """Vendor GEMM that accumulates a BF16 product into an FP32 buffer."""
    return _backend.wgrad(total_input, grad_output, main_grad)


def rotary_enabled(x):
    """True when a vendor rotary kernel replaces the rotate-half expression."""
    return _backend.rotary_enabled(x)


def rotary(x, cos, sin):
    return _backend.rotary(x, cos, sin)


def enabled(x, fused=True):
    """True when the fused RMSNorm / gated RMSNorm should run for this tensor."""
    return _backend.enabled(x, fused)


def rms_norm(x, weight, eps):
    return _backend.rms_norm(x, weight, eps)


def gated_rms_norm(x, gate, weight, eps):
    return _backend.gated_rms_norm(x, gate, weight, eps)


def ops_enabled(x, fused=True):
    """True when the fused packed causal convolution should run."""
    return _backend.ops_enabled(x, fused)


def causal_conv1d(x, weight, plan):
    return _backend.causal_conv1d(x, weight, plan)


def swiglu_enabled(x):
    """True when a fused SwiGLU should replace chunk + silu + multiply."""
    return _backend.swiglu_enabled(x)


def swiglu(projected):
    return _backend.swiglu(projected)
