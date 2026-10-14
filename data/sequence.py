"""Shared sequence-order utilities for ImageNet flow batches."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def scalar_int(item: Mapping[str, Any], key: str) -> int:
    """Read scalar sample metadata without coupling callers to its container."""

    value = item[key]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"sample field {key!r} must be scalar, got shape={tuple(value.shape)}"
            )
        value = value.item()
    return int(value)


def build_selfless_sigma(
    item: Mapping[str, Any],
    *,
    image_tokens: int,
) -> torch.Tensor:
    """Build one sample's strict generation order.

    The physical sequence is ``prefix, BOI, image, EOI, suffix, EOS`` while
    sigma deliberately orders EOI before every image token. Consequently EOI
    cannot see image content. Image queries can see prefix/BOI/EOI but no
    target-image content. All image tokens have the same sigma for joint
    whole-image denoising; suffix text can read the complete image content.
    """

    length = int(item["input_ids"].shape[0])
    prompt_len = scalar_int(item, "prompt_len")
    suffix_len = scalar_int(item, "suffix_len")
    image_start = scalar_int(item, "image_start")
    eoi_pos = image_start + int(image_tokens)
    suffix_start = eoi_pos + 1
    eos_pos = length - 1
    if image_start != prompt_len + 1:
        raise ValueError(
            "image_start must immediately follow prefix and BOI: "
            f"image_start={image_start}, prompt_len={prompt_len}"
        )
    if eoi_pos >= length or eos_pos < eoi_pos:
        raise ValueError(
            f"invalid image span for segment length={length}: "
            f"start={image_start}, image_tokens={image_tokens}"
        )
    expected_length = prompt_len + int(image_tokens) + suffix_len + 3
    if length != expected_length:
        raise ValueError(
            "serialized sample metadata does not match sequence length: "
            f"length={length}, expected={expected_length}"
        )

    sigma = torch.empty(length, dtype=torch.long)
    if prompt_len:
        sigma[:prompt_len] = torch.arange(prompt_len, dtype=torch.long)
    sigma[prompt_len] = prompt_len
    sigma[eoi_pos] = prompt_len + 1

    image_sigma_order = str(item.get("image_sigma_order", "joint")).strip().lower()
    if image_sigma_order != "joint":
        raise ValueError("UMM v3 uses joint image sigma; autoregressive image orders are not supported")
    reveal_order = torch.zeros(int(image_tokens), dtype=torch.long)
    sigma[image_start:eoi_pos] = prompt_len + 2 + reveal_order

    suffix_sigma_start = prompt_len + int(image_tokens) + 2
    if suffix_len:
        sigma[suffix_start:eos_pos] = suffix_sigma_start + torch.arange(
            suffix_len, dtype=torch.long
        )
    sigma[eos_pos] = suffix_sigma_start + suffix_len
    return sigma


def make_image_sequence(prefix_ids, suffix_ids, *, image_tokens, boi_id, eoi_id, mask_id, eos_id, max_seq_length=None, label_suffix=False):
    """Serialize prefix, BOI, image slots, EOI, suffix, EOS without shifting labels."""
    prefix_len, suffix_len = int(prefix_ids.numel()), int(suffix_ids.numel())
    length = prefix_len + image_tokens + suffix_len + 3
    if max_seq_length is not None and length > max_seq_length:
        raise ValueError("Serialized text exceeds max_seq_length; caption truncation is not supported")
    image_block = torch.full((image_tokens + 2,), mask_id, dtype=torch.long)
    image_block[0], image_block[-1] = boi_id, eoi_id
    image_types = torch.ones(image_tokens + 2, dtype=torch.uint8)
    image_types[[0, -1]] = 2
    input_ids = torch.cat([prefix_ids, image_block, suffix_ids, torch.tensor([eos_id])])
    token_types = torch.cat([torch.zeros(prefix_len, dtype=torch.uint8), image_types, torch.zeros(suffix_len, dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)])
    labels = torch.full_like(input_ids, -100)
    if label_suffix:
        if suffix_len == 0:
            raise ValueError("I2T caption supervision requires a non-empty suffix")
        suffix_start = prefix_len + image_tokens + 2
        labels[suffix_start:suffix_start + suffix_len] = suffix_ids
        labels[-1] = eos_id
    return dict(input_ids=input_ids, token_types=token_types, labels=labels,
                prompt_len=torch.tensor(prefix_len), suffix_len=torch.tensor(suffix_len),
                image_start=torch.tensor(prefix_len + 1))
