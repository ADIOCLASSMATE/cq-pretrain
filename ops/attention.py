"""Exact tiled attention with one output buffer and FP32 K/V accumulation."""

import torch
from torch.nn import functional as F

import distributed.accel as accel
import ops.audit as audit
from ops import functional as K
from ops import flex_cq


def segment_lengths(ends):
    """Per-segment lengths from cumulative ends (the form the CPU planners emit)."""
    return tuple(b - a for a, b in zip((0, *ends), ends))


def varlen(q, k, v, cu_q, cu_k, fused=True, scale=None):
    """Attention over packed segments. q is [Tq,H,D]; k and v are [Tk,H,D].

    Segment i of the query reads segment i of the key/value and nothing else.
    ``cu_q`` and ``cu_k`` are the cumulative segment ends prepared on CPU, the
    same boundaries ``varlen_reference`` iterates over; the last entry is the
    total length.

    One kernel call replaces one call per segment. Native-resolution packing
    gives every visual its own length, so the per-segment loop ran on every
    real pack, and on a host-bound accelerator its launch count rather than
    its arithmetic set the step time.
    """
    if fused and accel.IS_NPU and q.device.type == "npu":
        from ops import npu_attention

        return npu_attention.varlen(q, k, v, cu_q, cu_k, scale)
    if q.device.type == "cuda":
        from ops import cuda_attention

        if cuda_attention.available(q, fused):
            return cuda_attention.varlen(q, k, v, cu_q, cu_k, scale)
        # SDPA is a fused kernel here; only the per-segment launches remain.
        audit.record("varlen_attention", "sdpa_per_segment")
    else:
        audit.reference("varlen_attention", q, "no packed varlen kernel on this backend")
    return varlen_reference(q, k, v, cu_q, cu_k, scale)


def varlen_reference(q, k, v, cu_q, cu_k, scale=None):
    """One SDPA call per segment: the oracle a fused varlen form is gated against.

    Equal-length segments take a single batched call, which is what the
    evaluation suite's same-resolution sampling relies on; training packs are
    native resolution and take the loop.
    """
    heads, dim = q.shape[1], q.shape[2]
    lengths_q, lengths_k = segment_lengths(cu_q), segment_lengths(cu_k)
    if len(lengths_q) > 1 and len(set(lengths_q)) == 1 and len(set(lengths_k)) == 1:
        count, lq, lk = len(lengths_q), lengths_q[0], lengths_k[0]
        query = q.reshape(count, lq, heads, dim).transpose(1, 2)
        key, value = (t.reshape(count, lk, heads, dim).transpose(1, 2) for t in (k, v))
        result = F.scaled_dot_product_attention(query, key, value, scale=scale)
        return result.transpose(1, 2).reshape(-1, heads, dim)
    outputs = []
    start_q = start_k = 0
    for end_q, end_k in zip(cu_q, cu_k):
        query = q[start_q:end_q].transpose(0, 1)[None]
        key, value = (t[start_k:end_k].transpose(0, 1)[None] for t in (k, v))
        outputs.append(F.scaled_dot_product_attention(query, key, value, scale=scale)[0].transpose(0, 1))
        start_q, start_k = end_q, end_k
    return torch.cat(outputs)


def attend(q, k, v, mask, valid, plan, fused, predicate=None):
    # The predicate is the same visibility rule as `mask`, so this only changes
    # which kernel evaluates it; `fused=False` keeps the numerical reference.
    if fused and accel.IS_NPU and q.device.type == "npu":
        from ops import npu_attention

        return npu_attention.attend(q, k, v, mask, valid, plan)
    if fused and flex_cq.usable(q, predicate, mask):
        return flex_cq.attention(q, k, v, predicate, valid)
    if plan is None:
        return K.attention(q, k, v, mask, valid, fused)
    return tiled_attention(q, k, v, mask, valid, plan, fused)


def tiled_attention(q, k, v, mask, valid_rows, plan, fused=True, scale=None):
    if not plan.matches(mask):
        # In-place edits/replaced masks invalidate data-side metadata.
        return K.attention(q, k, v, mask, valid_rows, fused, scale)
    pieces = []
    for index, (start, end, key_end) in enumerate(plan.tiles):
        key_start = plan.key_starts[index] if plan.key_starts else 0
        tile_mask = plan.tile_masks[index] if plan.tile_masks else mask[:, :, start:end, key_start:key_end].contiguous()
        valid = valid_rows[:, start:end] if valid_rows is not None else None
        pieces.append(
            K.attention(q[start:end], k[key_start:key_end], v[key_start:key_end], tile_mask, valid, fused, scale)
        )
    return torch.cat(pieces, dim=0)
