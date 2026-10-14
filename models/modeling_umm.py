"""Composition of a CQ_IT multimodal backbone and a full-image Z2 DiT.

The model returns predictions. Objectives, optimizers and ODE integration live
outside the model so training and inference use the same generator function.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

import torch
from torch import nn

from .backbone import CQITBackbone
from .initialization import initialize_normal
from .generator import Z2Generator


@dataclass
class UMMConfig:
    vocab_size: int = 151940
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    pad_token_id: int | None = None
    mask_token_id: int = 151936
    image_mask_token_id: int = 151937
    boi_token_id: int = 151938
    eoi_token_id: int = 151939
    eos_token_id: int | None = 151645
    image_latent_dim: int = 16
    image_tokens_per_img: int = 256
    image_input_noise_strength: float = 0.01
    image_flow_width: int = 864
    image_flow_depth: int = 12
    joint_dit_head_dim: int = 72
    joint_dit_intermediate: int = 3456
    image_flow_time_scale: float = 1000.0
    image_flow_grad_checkpointing: bool = True
    gradient_checkpointing: bool = True
    backbone_type: str = "cq_it"
    generator_type: str = "z2"

    def __post_init__(self):
        if self.backbone_type != "cq_it" or self.generator_type != "z2":
            raise ValueError("This release supports the selected CQ_IT + Z2 baseline")
        if min(self.hidden_size, self.num_hidden_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim) <= 0:
            raise ValueError("Backbone dimensions must be positive")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("GQA requires divisible head counts and an even head_dim")
        if self.hidden_act != "silu":
            raise ValueError("The selected Qwen checkpoint uses SiLU")
        if math.isqrt(self.image_tokens_per_img) ** 2 != self.image_tokens_per_img:
            raise ValueError("Image tokens must form a complete square grid")
        for name in ("mask_token_id", "image_mask_token_id", "boi_token_id", "eoi_token_id", "pad_token_id", "eos_token_id"):
            token = getattr(self, name)
            if token is not None and not 0 <= token < self.vocab_size:
                raise ValueError(f"{name}={token} is outside vocab_size={self.vocab_size}; add special tokens before model construction")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values, *, strict=True):
        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if strict and unknown:
            raise ValueError(f"Unknown UMM model fields: {sorted(unknown)}")
        return cls(**{key: value for key, value in values.items() if key in known})


def build_backbone(config):
    return CQITBackbone(config)


def build_generator(config):
    return Z2Generator(config)


def visual_indices(batch, image_tokens_per_img):
    """Map target image spans to their batch rows and physical token indices."""
    spans = batch.get("image_span_table")
    if spans is None or spans.ndim != 2 or spans.shape[1] < 4 or spans.shape[0] == 0:
        raise ValueError("Visual prediction needs nonempty image_span_table [M,>=4]")
    spans = spans.to(device=batch["input_ids"].device, dtype=torch.long)
    rows, starts, ends = spans[:, 0], spans[:, 2], spans[:, 3]
    if not bool((ends - starts).eq(image_tokens_per_img).all()):
        raise ValueError("Each visual span must contain one complete image grid")
    if bool(((rows < 0) | (rows >= batch["input_ids"].shape[0]) | (starts < 0) | (ends > batch["input_ids"].shape[1])).any()):
        raise ValueError("Image span lies outside the input sequence")
    indices = starts[:, None] + torch.arange(image_tokens_per_img, device=rows.device)
    if not bool(batch["token_types"][rows[:, None], indices].eq(1).all()):
        raise ValueError("Visual spans must refer only to image token slots")
    return rows, indices


class UMM(nn.Module):
    def __init__(self, config: UMMConfig, *, backbone=None, generator=None):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config) if backbone is None else backbone
        self.condition_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.generator = build_generator(config) if generator is None else generator
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        initialize_normal(self.condition_projector.weight, config.initializer_range)
        nn.init.zeros_(self.condition_projector.bias)
        initialize_normal(self.lm_head.weight, config.initializer_range)
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embed_tokens.weight

    def resize_token_embeddings(self, vocab_size):
        """Resize once after tokenizer special tokens are added; preserve tying."""
        old = self.backbone.embed_tokens
        replacement = nn.Embedding(vocab_size, self.config.hidden_size, self.config.pad_token_id,
                                   device=old.weight.device, dtype=old.weight.dtype)
        initialize_normal(replacement.weight, self.config.initializer_range)
        count = min(vocab_size, old.num_embeddings)
        with torch.no_grad():
            replacement.weight[:count].copy_(old.weight[:count])
        self.backbone.embed_tokens = replacement
        if not self.config.tie_word_embeddings:
            output = nn.Linear(self.config.hidden_size, vocab_size, bias=False,
                               device=old.weight.device, dtype=old.weight.dtype)
            initialize_normal(output.weight, self.config.initializer_range)
            with torch.no_grad():
                output.weight[:count].copy_(self.lm_head.weight[:count])
            self.lm_head = output
        else:
            self.lm_head.out_features = vocab_size
        self.config.vocab_size = vocab_size
        self.tie_weights()
        return replacement

    def encode_condition(self, batch):
        hidden = self.backbone(batch)["query_hidden"]
        return self._condition_from_hidden(hidden, batch)

    def _condition_from_hidden(self, hidden, batch):
        rows, indices = visual_indices(batch, self.config.image_tokens_per_img)
        selected = hidden[rows[:, None], indices]
        return self.condition_projector(selected.to(self.condition_projector.weight.dtype))

    def forward(self, batch, noisy_latents=None, timesteps=None, *, compute_text=None):
        """Share one CQ pass between text predictions and optional flow velocity.

        With noisy latents, labels enable text output by default. Training can
        explicitly disable it for a flow-only microbatch.
        """
        if (noisy_latents is None) != (timesteps is None):
            raise ValueError("noisy_latents and timesteps must be supplied together")
        if compute_text is None:
            compute_text = noisy_latents is None or batch.get("labels") is not None
        hidden = self.backbone(batch)["query_hidden"]
        output = {}
        if noisy_latents is not None:
            condition = self._condition_from_hidden(hidden, batch)
            if noisy_latents.shape[0] % condition.shape[0]:
                raise ValueError("RF samples must repeat complete condition batches")
            repeats = noisy_latents.shape[0] // condition.shape[0]
            condition = condition.repeat(repeats, 1, 1)
            output.update(velocity=self.generator(noisy_latents, timesteps, condition), condition=condition)
        if compute_text:
            labels = batch.get("labels")
            if labels is None:
                output.update(text_logits=self.lm_head(hidden), text_labels=None)
            else:
                valid = labels.ne(-100)
                if "token_types" in batch:
                    valid &= batch["token_types"].eq(0) | batch["token_types"].eq(2)
                output.update(text_logits=self.lm_head(hidden[valid]), text_labels=labels[valid])
        return output
