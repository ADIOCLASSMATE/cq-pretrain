"""Padding-only collation: one logical video sample per row, no document packing."""

import torch
from torch.nn.utils.rnn import pad_sequence


def validate_representation_batch(cached, online, *, branch):
    """Each branch uses one representation per batch; absent inference targets are allowed."""
    if len(cached) != len(online):
        raise ValueError(f"{branch} inputs must have matching counts")
    if any(x is not None and y is not None for x, y in zip(cached, online)):
        raise ValueError(f"Each {branch} video must supply cached or online input, not both")
    if any(x is not None for x in cached) and any(x is not None for x in online):
        raise ValueError(
            f"{branch} batch cannot mix cached and online videos; use one mode "
            "for the entire branch batch (complete caches or select online)"
        )


def collate_video_samples(samples, *, pad_token_id=0, pad_to_multiple_of=8):
    if not samples:
        raise ValueError("Empty batch")
    features = [x.get("video_features") for x in samples]
    pixels = [x.get("pixel_values_videos") for x in samples]
    validate_representation_batch(features, pixels, branch="Qwen")
    targets = [i for i, x in enumerate(samples) if x["task"] == "t2v"]
    latents = [samples[i].get("wan_latent") for i in targets]
    rgb = [samples[i].get("wan_rgb") for i in targets]
    validate_representation_batch(latents, rgb, branch="Wan")
    ids = pad_sequence(
        [x["input_ids"] for x in samples], batch_first=True, padding_value=pad_token_id
    )
    length = (
        (ids.shape[1] + pad_to_multiple_of - 1)
        // pad_to_multiple_of
        * pad_to_multiple_of
    )
    ids = torch.nn.functional.pad(ids, (0, length - ids.shape[1]), value=pad_token_id)
    valid = (
        torch.arange(length)[None]
        < torch.tensor([len(x["input_ids"]) for x in samples])[:, None]
    )
    labels = torch.full_like(ids, -100)
    mm_types = torch.zeros_like(ids)
    for i, item in enumerate(samples):
        labels[i, : len(item["labels"])] = item["labels"]
        mm_types[i, : len(item["input_ids"])] = item["mm_token_type_ids"]
    tasks = [x["task"] for x in samples]
    result = dict(
        input_ids=ids,
        attention_mask=valid,
        labels=labels,
        mm_token_type_ids=mm_types,
        video_grid_thw=torch.stack([x["video_grid_thw"] for x in samples]),
        video_observed_mask=torch.ones(len(samples), dtype=torch.bool),
        target_video_ids=torch.tensor(targets, dtype=torch.long),
        wan_latents=latents,
        wan_rgb=rgb,
        tasks=tasks,
        task=tasks[0] if len(set(tasks)) == 1 else "mixed",
        sample_ids=[x["sample_id"] for x in samples],
        clips=[x["clip"] for x in samples],
    )
    if any(x is not None for x in pixels):
        result["pixel_values_videos"] = torch.cat([x for x in pixels if x is not None])
    else:
        result["video_features"] = features
    return result
