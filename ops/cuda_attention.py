"""Packed varlen attention on CUDA, through flash-attn's own varlen entry point.

No kernel is written here. The CPU planners already describe each pack as a
list of independent segments, which is exactly the layout
``flash_attn_varlen_func`` takes, so one call replaces the one-call-per-segment
loop in ``ops.attention.varlen_reference`` -- the same change the Ascend side
makes through ``npu_fusion_attention``'s TND layout.

flash-attn lives in the training image (``/usr/local/lib/python3.12/
dist-packages``) rather than in ``.venv-cuda``, so its absence from the venv
listing says nothing; ``available`` decides by importing it.

The kernel takes cumulative boundaries that begin at zero, while this project's
planners emit ends only. Converting them allocates two small int32 tensors per
call, so the conversion is cached per (boundaries, device): a pack reuses one
layout across all 30 DiT layers and both recomputes.
"""

import functools
import os

import torch

import ops.audit as audit

_ENABLED = os.environ.get("PRETRAIN_CUDA_VARLEN", "1") == "1"


@functools.lru_cache(maxsize=1)
def _kernel():
    """The varlen entry point, or None when this image does not ship it."""
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception:
        return None
    return flash_attn_varlen_func


def available(q, fused=True):
    """True only where the kernel both exists and accepts these tensors."""
    if not (_ENABLED and fused and q.device.type == "cuda"):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        # The reference stays the implementation for FP32/FP64 oracles.
        return False
    if q.shape[-1] % 8 or q.shape[-1] > 256:
        return False
    if torch.are_deterministic_algorithms_enabled():
        return False
    return _kernel() is not None


@functools.lru_cache(maxsize=32)
def _offsets(ends, device):
    """Cumulative boundaries starting at zero, as the kernel wants them."""
    return torch.tensor((0, *ends), dtype=torch.int32, device=device)


def varlen(q, k, v, cu_q, cu_k, scale=None):
    """q is [Tq,H,D]; k and v are [Tk,H,D]. Segment i reads segment i only."""
    audit.record("varlen_attention", "flash_attn_varlen")
    device = q.device
    offsets_q, offsets_k = _offsets(tuple(cu_q), device), _offsets(tuple(cu_k), device)
    lengths_q = [b - a for a, b in zip((0, *cu_q), cu_q)]
    lengths_k = [b - a for a, b in zip((0, *cu_k), cu_k)]
    query, key, value = (t if t.is_contiguous() else t.contiguous() for t in (q, k, v))
    return _kernel()(
        query,
        key,
        value,
        offsets_q,
        offsets_k,
        max(lengths_q),
        max(lengths_k),
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
    )
