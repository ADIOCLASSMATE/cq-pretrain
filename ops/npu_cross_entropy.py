"""Chunked linear cross entropy backed by TorchNPU's fused CE operator.

No kernel is written here. The chunking, recompute and MCore gradient protocol
are exactly those of ``ops.cross_entropy._LinearCE``; the only change is that
each chunk's log-softmax and its gradient are evaluated by
``npu_cross_entropy_loss`` instead of several PyTorch passes over a
materialized FP32 ``[chunk, vocab]`` tensor.

Vocabulary tensor parallelism stays on the original path: this operator
reduces within one process, while the original all-reduces the maxima, the
selected logit and the partition function across the TP group.
"""

import os

import torch
from torch.nn import functional as F

import ops.audit as audit
from ops.functional import fused_wgrad

_ENABLED = os.environ.get("PRETRAIN_NPU_FUSED_CE", "1") == "1"


def available(hidden, tp_size):
    """True only where this path is both applicable and equivalent."""
    if not (_ENABLED and hidden.device.type == "npu" and tp_size == 1):
        return False
    if torch.are_deterministic_algorithms_enabled():
        return False
    try:
        import torch_npu
    except Exception:
        return False
    return callable(getattr(torch_npu, "npu_cross_entropy_loss", None))


class _FusedLinearCE(torch.autograd.Function):
    """Mean loss over rows; no [rows, vocab] tensor is retained in the graph."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        import torch_npu

        audit.record("cross_entropy", "npu_cross_entropy_loss")
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk):
            logits = F.linear(hidden[start : start + chunk], weight).float()
            # The operator returns its summed loss as a [1] tensor; `total` is
            # 0-dim, and aclnnInplaceAdd requires the destination shape to match.
            total += torch_npu.npu_cross_entropy_loss(logits, targets[start : start + chunk], reduction="sum")[
                0
            ].sum()
            del logits
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk = chunk
        ctx.main_grad = (
            getattr(weight, "main_grad", None)
            if getattr(weight, "_ce_direct_accumulation", False) and hasattr(weight, "grad_added_to_main_grad")
            else None
        )
        if ctx.main_grad is not None and ctx.main_grad.dtype != torch.float32:
            raise ValueError("Direct CE accumulation requires MCore FP32 gradients")
        return total / max(1, hidden.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        import torch_npu

        hidden, weight, targets = ctx.saved_tensors
        dh = torch.empty_like(hidden)
        # FP32 chunk accumulation avoids rounding a BF16 gradient once per chunk.
        dw = ctx.main_grad if ctx.main_grad is not None else torch.zeros_like(weight, dtype=torch.float32)
        factor = grad_output.float() / max(1, hidden.shape[0])
        for start in range(0, hidden.shape[0], ctx.chunk):
            h = hidden[start : start + ctx.chunk]
            chunk_targets = targets[start : start + ctx.chunk]
            logits = F.linear(h, weight).float()
            _, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(logits, chunk_targets, reduction="sum")
            gradient = torch_npu.npu_cross_entropy_loss_backward(
                factor.reshape(1), log_prob, chunk_targets, reduction="sum"
            ).to(hidden.dtype)
            dh[start : start + ctx.chunk] = gradient @ weight
            fused_wgrad(h, gradient, dw)
            del logits, log_prob, gradient
        if ctx.main_grad is not None:
            # Match ColumnParallelLinear's hook protocol; main_grad already holds
            # the result, so MCore must not add this dummy gradient again.
            weight.grad_added_to_main_grad = True
            factory = torch.zeros_like if getattr(weight, "zero_out_wgrad", False) else torch.empty_like
            gradient = factory(weight)
        else:
            gradient = dw.to(weight.dtype)
        return dh, gradient, None, None


def linear_cross_entropy(hidden, weight, targets, chunk):
    return _FusedLinearCE.apply(hidden, weight, targets, chunk)
