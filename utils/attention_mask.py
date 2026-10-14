"""CQ_IT visibility: content <= sigma, query < sigma, tied image ranks.

Boolean masks returned here use True for *allowed*, matching PyTorch SDPA.
The source implementation returns the inverse on Ascend; that conversion belongs
at the attention backend boundary, not in data or model visibility rules.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def joint_image_sigma(token_types, sigma=None):
    """Preserve source text/boundary order while tying each complete image."""
    image = token_types.eq(1)
    positions = torch.arange(token_types.shape[1], device=token_types.device)[None].expand_as(token_types)
    previous_image = F.pad(image[:, :-1], (1, 0), value=False)
    starts = torch.where(image & ~previous_image, positions, 0).cummax(-1).values
    if sigma is None:
        values = positions.float()
        values = torch.where(~image & previous_image, starts.float(), values)
        image_values = starts.float() + 1
    else:
        if sigma.shape != token_types.shape:
            raise ValueError("sigma must align with token_types")
        values = sigma.to(device=token_types.device, dtype=torch.float32)
        image_values = values.gather(1, starts)
    return torch.where(image, image_values, values)


def build_cq_masks(batch, boi_token_id=None):
    """Return independent query/content masks, including packing and CFG."""
    token_types = batch["token_types"]
    sigma = joint_image_sigma(token_types, batch.get("sigma"))
    valid = token_types.ne(3)
    segments = batch.get("segment_ids")
    if segments is None:
        segments = torch.where(valid, 0, -1)
    elif segments.shape != token_types.shape:
        raise ValueError("segment_ids must align with token_types")
    segments = torch.where(valid, segments.to(token_types.device), -1)
    segment_mask = (segments[:, :, None] == segments[:, None, :])
    segment_mask &= (segments[:, :, None] >= 0) & (segments[:, None, :] >= 0)
    query = (sigma[:, None, :] < sigma[:, :, None]) & segment_mask
    content = (sigma[:, None, :] <= sigma[:, :, None]) & segment_mask
    uncond = batch.get("image_uncond_mask")
    uncond_rows = batch.get("image_uncond_rows")
    if uncond is not None or uncond_rows is not None:
        if uncond is not None:
            if uncond.shape != token_types.shape:
                raise ValueError("image_uncond_mask must align with token_types")
            uncond = uncond.to(device=token_types.device, dtype=torch.bool)
            image_span_ids = segments
        else:
            if boi_token_id is None:
                raise ValueError("row-level CFG dropout requires boi_token_id")
            uncond = uncond_rows.to(token_types.device).bool()[:, None] & token_types.eq(1)
            image_span_ids = batch["input_ids"].eq(boi_token_id).long().cumsum(1)
        same_image = token_types[:, None, :].eq(1) & (
            image_span_ids[:, None, :] == image_span_ids[:, :, None]
        )
        cfg_mask = ~uncond[:, :, None] | same_image
        query &= cfg_mask
        content &= cfg_mask
    return query, content
