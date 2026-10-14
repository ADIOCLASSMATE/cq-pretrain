# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0; see licenses/Apache-2.0.txt.
# Modified for UMM v3 from unified-mm commit 193ec90300af5fee4bc4b7e8a426492a908a0447.
"""CQ_IT: shared Qwen weights, independent content/query states.

Only content produces keys and values. The content mask is sigma <=; the
query mask is sigma <. Both masks include segment/padding/CFG restrictions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .position import build_row_col_position_ids, rotate_half
from .initialization import initialize_normal
from utils.attention_mask import build_cq_masks


def attend(query, key, value, allowed, scale):
    """SDPA/NPU dispatch, including well-defined zero output for empty rows."""
    if key.shape[-2] == 0:
        return torch.zeros_like(query)
    valid = allowed.any(-1, keepdim=True)
    safe = allowed.clone()
    safe[..., 0] |= ~valid.squeeze(-1)
    if query.device.type == "npu":
        import torch_npu
        result = torch_npu.npu_fusion_attention(
            query, key, value, head_num=query.shape[1], input_layout="BNSD",
            atten_mask=~safe[:, None], sparse_mode=0, scale=scale,
        )[0]
    else:
        result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=safe[:, None], dropout_p=0.0,
            is_causal=False, scale=scale, enable_gqa=query.shape[1] != key.shape[1],
        )
    return result * valid[:, None].to(result.dtype)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, x):
        if x.device.type == "npu":
            import torch_npu
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Alternating row/column frequency pairs; (p,p) is ordinary text RoPE."""
    def __init__(self, config):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        ), persistent=False)

    @torch.no_grad()
    def forward(self, x, positions):
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            if positions.ndim == 2:
                frequencies = positions[:, :, None].float() * self.inv_freq.float()
            elif positions.ndim == 3 and positions.shape[0] == 2:
                axes = torch.arange(self.inv_freq.numel(), device=positions.device) % 2
                selected = positions.index_select(0, axes).permute(1, 2, 0)
                frequencies = selected.float() * self.inv_freq.float()
            else:
                raise ValueError("positions must have shape [B,L] or [2,B,L]")
            phase = torch.cat((frequencies, frequencies), -1)
            return phase.cos().to(x.dtype), phase.sin().to(x.dtype)


class ImageTokenEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_proj = nn.Linear(config.image_latent_dim, config.hidden_size)
        self.reset_parameters(config)

    def reset_parameters(self, config):
        initialize_normal(self.z_proj.weight, config.initializer_range / math.sqrt(2 * config.image_latent_dim))
        nn.init.zeros_(self.z_proj.bias)

    def forward(self, latents):
        return self.z_proj(latents.to(self.z_proj.weight.dtype))


@dataclass
class ContentCache:
    """Inference-only per-layer content KV. Query states are never committed."""
    layers: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def append(self, layer, key, value):
        if layer == len(self.layers):
            self.layers.append((key, value))
        else:
            old_key, old_value = self.layers[layer]
            self.layers[layer] = (torch.cat((old_key, key), -2), torch.cat((old_value, value), -2))
        return self.layers[layer]

    @property
    def length(self):
        return self.layers[0][0].shape[-2] if self.layers else 0


class CQAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, content, query, rope, query_allowed, content_allowed, cache=None, cache_mode=None):
        shape = (*content.shape[:-1], -1, self.head_dim)
        cos, sin = (part[:, None] for part in rope)
        def rotate(x):
            return x * cos + rotate_half(x) * sin
        content_q = rotate(self.q_norm(self.q_proj(content).view(shape)).transpose(1, 2))
        query_q = None if query is None else rotate(self.q_norm(self.q_proj(query).view(shape)).transpose(1, 2))
        if cache_mode == "read":
            key, value = cache.layers[self.layer_idx]
        else:
            key = rotate(self.k_norm(self.k_proj(content).view(shape)).transpose(1, 2))
            value = self.v_proj(content).view(shape).transpose(1, 2)
            if cache_mode == "append":
                key, value = cache.append(self.layer_idx, key, value)
        def apply(q, allowed):
            result = attend(q, key, value, allowed, self.scaling)
            return self.o_proj(result.transpose(1, 2).reshape(*content.shape[:-1], -1).contiguous())
        content_out = apply(content_q, content_allowed)
        query_out = None if query_q is None else apply(query_q, query_allowed)
        return content_out, query_out


class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CQDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = CQAttention(config, layer_idx)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, content, query, rope, query_allowed, content_allowed, cache=None, cache_mode=None):
        content_attn, query_attn = self.self_attn(
            self.input_layernorm(content), None if query is None else self.input_layernorm(query),
            rope, query_allowed, content_allowed, cache, cache_mode,
        )
        content = content + content_attn
        content = content + self.mlp(self.post_attention_layernorm(content))
        if query is not None:
            query = query + query_attn
            query = query + self.mlp(self.post_attention_layernorm(query))
        return content, query


class CQITBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.image_token_embedder = ImageTokenEmbedder(config)
        self.layers = nn.ModuleList([CQDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.apply(self._init_weights)
        self.image_token_embedder.reset_parameters(config)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            initialize_normal(module.weight, self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def embed_content(self, batch):
        ids = batch["input_ids"]
        types = batch.get("token_types", torch.zeros_like(ids))
        image = types.eq(1)
        content = self.embed_tokens(ids.masked_fill(image, self.config.image_mask_token_id))
        latents = batch.get("content_image_latents", batch.get("image_latents"))
        if latents is not None:
            if latents.shape != (*ids.shape, self.config.image_latent_dim):
                raise ValueError("image_latents must align with [B,L,latent_dim]")
            latents = latents.to(self.image_token_embedder.z_proj.weight.dtype)
            if self.training and self.config.image_input_noise_strength > 0 and "content_image_latents" not in batch:
                latents = latents + torch.randn_like(latents) * self.config.image_input_noise_strength
            observed = image & batch.get("image_latent_mask", image).bool()
            content = torch.where(observed[..., None], self.image_token_embedder(latents), content)
        elif bool(image.any()):
            raise ValueError("image_latents are required when the batch contains image slots")
        return content

    def forward(self, batch, *, return_content=False, cache=None, cache_mode=None,
                positions=None, masks=None, query_stream=True):
        if cache_mode is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Content cache is available only in eval/no_grad inference")
            if cache is None or cache_mode not in {"read", "append"}:
                raise ValueError("cache_mode requires a ContentCache and read/append mode")
        ids = batch["input_ids"]
        types = batch.get("token_types", torch.zeros_like(ids))
        content = self.embed_content(batch)
        if query_stream:
            mask_ids = torch.full_like(ids, self.config.mask_token_id)
            mask_ids = mask_ids.masked_fill(types.eq(1), self.config.image_mask_token_id)
            query = self.embed_tokens(mask_ids)
        else:
            query = None
        query_allowed, content_allowed = masks if masks is not None else build_cq_masks(batch, self.config.boi_token_id)
        if positions is None:
            positions = batch.get("position_ids")
        if positions is None:
            positions = build_row_col_position_ids(types, self.config.image_tokens_per_img)
        rope = self.rotary_emb(content, positions)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                content, query = checkpoint(layer, content, query, rope, query_allowed, content_allowed, use_reentrant=False)
            else:
                content, query = layer(content, query, rope, query_allowed, content_allowed, cache, cache_mode)
        output = {"query_hidden": self.norm(query) if query is not None else None}
        if return_content or not query_stream:
            output["content_hidden"] = self.norm(content)
        return output
