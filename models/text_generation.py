"""Same-position CQ text decoding with an append-only content KV cache."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .backbone import ContentCache
from .position import build_row_col_position_ids
from utils.attention_mask import joint_image_sigma


def _sample(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k:
        threshold = logits.topk(min(top_k, logits.shape[-1]), -1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1:
        sorted_logits, order = logits.sort(descending=True)
        remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.masked_fill(torch.zeros_like(remove).scatter(1, order, remove), -torch.inf)
    return torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)


@torch.no_grad()
def generate_text(model, batch, *, max_new_tokens=128, temperature=0.0,
                  top_k=0, top_p=1.0, eos_token_id=None, use_cache=True):
    """Generate text after a supplied multimodal prefix; return new token IDs.

    First predict a mask query without committing it. Then commit the sampled
    token as content. This preserves the training model's same-position target.
    ``use_cache=False`` is the complete-sequence reference path.
    """
    if max_new_tokens < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid text sampling settings")
    state = dict(batch)
    # A prepared training batch may carry augmentation and CFG-dropout state.
    # Decode from the observed clean prefix, then grow only inference fields.
    for key in ("content_image_latents", "image_uncond_rows", "image_uncond_mask"):
        state.pop(key, None)
    ids = state["input_ids"]
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("Text generation needs a nonempty [B,L] prefix")
    state["token_types"] = state.get("token_types", torch.zeros_like(ids))
    state["sigma"] = joint_image_sigma(state["token_types"], state.get("sigma"))
    state["segment_ids"] = state.get("segment_ids", torch.where(state["token_types"].ne(3), 0, -1))
    valid = state["token_types"].ne(3) & state["segment_ids"].ge(0)
    if bool((~valid.any(-1)).any()):
        raise ValueError("Each prompt must contain valid context")
    # A decoding request is one independent prefix per row, never packed tasks.
    for row in range(ids.shape[0]):
        if state["segment_ids"][row][valid[row]].unique().numel() != 1:
            raise ValueError("Unpack independent segments before text generation")
    eos = model.config.eos_token_id if eos_token_id is None else eos_token_id
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else (eos if eos is not None else 0)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        cache = ContentCache() if use_cache else None
        if use_cache:
            model.backbone(state, cache=cache, cache_mode="append", query_stream=False)
        generated = []
        finished = torch.zeros(ids.shape[0], device=ids.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            next_rank = state["sigma"].masked_fill(~valid, -torch.inf).amax(-1, keepdim=True) + 1
            next_segment = state["segment_ids"].masked_fill(~valid, -1).amax(-1, keepdim=True)
            query_batch = {
                "input_ids": torch.full((ids.shape[0], 1), model.config.mask_token_id, device=ids.device, dtype=ids.dtype),
                "token_types": finished[:, None].to(ids.dtype) * 3,
                "sigma": next_rank,
                "segment_ids": torch.where(finished[:, None], -1, next_segment),
            }
            full = _append(state, query_batch, model.config.image_latent_dim)
            positions = build_row_col_position_ids(full["token_types"], model.config.image_tokens_per_img)[:, :, -1:]
            if use_cache:
                allowed = (state["sigma"][:, None] < next_rank[..., None]) & valid[:, None] & ~finished[:, None, None]
                hidden = model.backbone(query_batch, cache=cache, cache_mode="read", positions=positions,
                                        masks=(allowed, allowed))["query_hidden"]
            else:
                full.pop("position_ids", None)
                hidden = model.backbone(full)["query_hidden"][:, -1:]
            next_token = _sample(model.lm_head(hidden[:, 0]), temperature, top_k, top_p)
            next_token = torch.where(finished, pad, next_token)
            generated.append(next_token)
            query_batch["input_ids"] = next_token[:, None]
            if use_cache:
                key_sigma = torch.cat((state["sigma"], next_rank), -1)
                key_valid = torch.cat((valid, ~finished[:, None]), -1)
                allowed = (key_sigma[:, None] <= next_rank[..., None]) & key_valid[:, None] & ~finished[:, None, None]
                model.backbone(query_batch, cache=cache, cache_mode="append", positions=positions,
                               masks=(allowed, allowed), query_stream=False)
            state = _append(state, query_batch, model.config.image_latent_dim)
            state.pop("position_ids", None)
            valid = torch.cat((valid, ~finished[:, None]), -1)
            if eos is not None:
                finished |= next_token.eq(eos)
            if bool(finished.all()):
                break
        return torch.stack(generated, -1) if generated else ids.new_empty((ids.shape[0], 0))
    finally:
        for module, mode in modes:
            module.training = mode


def _append(prefix, token, latent_dim):
    result = dict(prefix)
    for key in ("input_ids", "token_types", "sigma", "segment_ids"):
        result[key] = torch.cat((prefix[key], token[key]), -1)
    if prefix.get("image_latents") is not None:
        latent = prefix["image_latents"]
        result["image_latents"] = torch.cat((latent, latent.new_zeros((latent.shape[0], 1, latent_dim))), 1)
    if prefix.get("image_latent_mask") is not None:
        mask = prefix["image_latent_mask"]
        result["image_latent_mask"] = torch.cat((mask, torch.zeros_like(mask[:, :1])), 1)
    result.pop("labels", None)
    return result
