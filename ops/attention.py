"""Exact tiled attention with one output buffer and FP32 K/V accumulation."""

import torch
from ops import functional as K
from ops import flex_cq


def attend(q, k, v, mask, valid, plan, fused, predicate=None):
    # The predicate is the same visibility rule as `mask`, so this only changes
    # which kernel evaluates it; `fused=False` keeps the numerical reference.
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
