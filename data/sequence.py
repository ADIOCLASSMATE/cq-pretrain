"""Native Qwen chat/video serialization for same-position CQ supervision."""

from __future__ import annotations
from copy import deepcopy
import torch
from transformers.video_utils import VideoMetadata


def native_video_metadata(timestamps):
    # The processor pairs frames then averages their timestamps. Express exact
    # PTS in seconds so nonuniform sampling does not become a fictitious CFR.
    return VideoMetadata(
        total_num_frames=len(timestamps), fps=1.0, frames_indices=list(timestamps)
    )


def make_video_sequence(
    messages, processor, grid_thw, timestamps, *, max_seq_length=None, prompt_only=False
):
    messages = deepcopy(messages)
    if prompt_only:
        # Accept both supervised manifests and unlabeled inference requests.
        # Never put a reference assistant answer into the generated prefix.
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("V2T generation requires a final user instruction")
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=prompt_only,
        enable_thinking=False,
    )
    if text.count(processor.video_token) != 1:
        raise ValueError(
            "Stage one requires exactly one video placeholder in each sample"
        )
    replacement = processor.replace_video_token(
        {
            "video_grid_thw": grid_thw[None],
            "video_metadata": [native_video_metadata(timestamps)],
        },
        0,
    )
    text, _ = processor.get_text_with_replacements(
        [text], videos_replacements=[replacement]
    )
    encoded = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"][0]
    expected = int(grid_thw.prod()) // processor.video_processor.merge_size**2
    if int(ids.eq(processor.video_token_id).sum()) != expected:
        raise ValueError("Serialized video slots do not match native ViT features")
    if max_seq_length is not None and ids.numel() > max_seq_length:
        raise ValueError(
            f"Sequence has {ids.numel()} tokens, exceeding max_seq_length={max_seq_length}; no partial-video truncation"
        )
    labels = ids.clone()
    labels[ids.eq(processor.video_token_id)] = -100
    if prompt_only:
        labels.fill_(-100)
    return dict(
        input_ids=ids,
        labels=labels,
        mm_token_type_ids=ids.eq(processor.video_token_id).long() * 2,
    )
