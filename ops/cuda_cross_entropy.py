"""Chunked linear cross entropy backed by a published fused CE kernel.

No kernel is written here. The chunking, recompute and MCore gradient protocol
are exactly those of ``ops.cross_entropy._LinearCE``; the only change is that
each chunk's softmax/log-sum-exp and its gradient are evaluated by flash-attn's
Triton cross-entropy instead of eight PyTorch passes over a materialized FP32
``[chunk, vocab]`` tensor.

Numerics are unchanged by construction: ``F.linear`` on BF16 inputs already
rounds the logits to BF16 before the old path's ``.float()`` upcast, so feeding
the same BF16 logits to a kernel that reduces in FP32 sees identical values.

Vocabulary tensor parallelism stays on the original path: this kernel reduces
within one process, while the old path all-reduces maxima, the selected logit
and the partition function across the TP group.
"""

import os

import torch
from torch.nn import functional as F

from ops.functional import fused_wgrad

_ENABLED = os.environ.get("PRETRAIN_CUDA_FUSED_CE", "0") == "1"


def available(hidden, tp_size):
    """True only where this path is both applicable and equivalent."""
    if not (_ENABLED and hidden.device.type == "cuda" and tp_size == 1):
        return False
    if torch.are_deterministic_algorithms_enabled():
        return False
    try:
        from flash_attn.ops.triton.cross_entropy import cross_entropy_loss  # noqa: F401
    except Exception:
        return False
    return True


def _losses(logits, targets, inplace):
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    return cross_entropy_loss(logits, targets, inplace_backward=inplace)[0]


class _FusedLinearCE(torch.autograd.Function):
    """Mean loss over rows; no [rows, vocab] tensor is retained in the graph."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk):
            logits = F.linear(hidden[start : start + chunk], weight)
            total += _losses(logits, targets[start : start + chunk], False).sum()
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
        hidden, weight, targets = ctx.saved_tensors
        dh = torch.empty_like(hidden)
        # FP32 chunk accumulation avoids rounding a BF16 gradient once per chunk.
        dw = ctx.main_grad if ctx.main_grad is not None else torch.zeros_like(weight, dtype=torch.float32)
        factor = grad_output.float() / max(1, hidden.shape[0])
        for start in range(0, hidden.shape[0], ctx.chunk):
            h = hidden[start : start + ctx.chunk]
            logits = F.linear(h, weight)
            # The kernel's own backward writes dlogits into the logits buffer,
            # so the chunk costs one [chunk, vocab] BF16 tensor and no more.
            with torch.enable_grad():
                leaf = logits.detach().requires_grad_(True)
                scaled = (_losses(leaf, targets[start : start + ctx.chunk], True) * factor).sum()
            (gradient,) = torch.autograd.grad(scaled, leaf)
            dh[start : start + ctx.chunk] = gradient @ weight
            fused_wgrad(h, gradient, dw)
            del logits, leaf, scaled, gradient
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
