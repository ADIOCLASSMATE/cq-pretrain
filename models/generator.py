"""Z2: Z+B rectified-flow conditions with DiT blocks and a four-times-width FFN.

The backbone still produces one fixed condition per image position. Both Q/K/V
and the FFN read their branch's common modulated, non-affine LayerNorm output.
Local row/column RoPE is retained from B; no target content enters this head.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .position import build_local_row_col_rope, rotate_half
from .initialization import initialize_normal as _normal_init_fp32_, initialize_xavier as _xavier_uniform_init_fp32_


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self._frequency_cache = {}

    def _apply(self, fn, recurse=True):
        self._frequency_cache.clear()
        return super()._apply(fn, recurse=recurse)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, freqs=None):
        half = dim // 2
        if freqs is None:
            freqs = torch.exp(
                -math.log(max_period)
                * torch.arange(
                    start=0,
                    end=half,
                    dtype=torch.float32,
                    device=t.device,
                )
                / half
            )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        cache_key = (t.device.type, t.device.index)
        freqs = self._frequency_cache.get(cache_key)
        if freqs is None:
            half = self.frequency_embedding_size // 2
            freqs = torch.exp(
                -math.log(10000)
                * torch.arange(
                    start=0,
                    end=half,
                    dtype=torch.float32,
                    device=t.device,
                )
                / half
            )
            self._frequency_cache[cache_key] = freqs
        t_freq = self.timestep_embedding(
            t,
            self.frequency_embedding_size,
            freqs=freqs,
        )
        t_freq = t_freq.to(device=self.mlp[0].weight.device, dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)




class DiTSelfAttention(nn.Module):
    """Full image MHA; a single conditioned input supplies all three projections."""

    def __init__(self, width, head_dim):
        super().__init__()
        self.num_heads = width // head_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, hidden, rope):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).reshape(
            batch, length, 3, self.num_heads, self.head_dim,
        ).unbind(2)
        cos, sin = (part.unsqueeze(2) for part in rope)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if hidden.device.type == "npu":
            import torch_npu
            output = torch_npu.npu_fusion_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                head_num=self.num_heads, input_layout="BSND", scale=self.scale,
                keep_prob=1.0, sparse_mode=0,
            )[0]
        else:
            output = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False, scale=self.scale,
            ).transpose(1, 2)
        return self.proj(output.reshape(batch, length, width))


class AdaLNZeroDiTBlock(nn.Module):
    def __init__(self, width, head_dim, intermediate):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = DiTSelfAttention(width, head_dim)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, intermediate), nn.GELU(approximate="tanh"),
            nn.Linear(intermediate, width),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, hidden, condition, rope):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        hidden = hidden + gate_attn * self.attn(
            modulate(self.norm1(hidden), shift_attn, scale_attn), rope,
        )
        return hidden + gate_mlp * self.mlp(
            modulate(self.norm2(hidden), shift_mlp, scale_mlp),
        )


class Z2Generator(nn.Module):
    """Parameter-matched DiT head retaining Z+B's per-position h+t conditions."""

    def __init__(self, config):
        super().__init__()
        self.in_channels = self.out_channels = int(config.image_latent_dim)
        self.model_channels = width = int(config.image_flow_width)
        self.num_res_blocks = int(config.image_flow_depth)
        self.head_dim = int(getattr(config, "joint_dit_head_dim", 72))
        self.intermediate = int(getattr(config, "joint_dit_intermediate", 4 * width))
        self.image_tokens_per_img = int(config.image_tokens_per_img)
        if width <= 0 or self.head_dim <= 0 or width % self.head_dim or self.head_dim % 4:
            raise ValueError("DiT width must divide into positive RoPE heads with head_dim divisible by 4")
        if self.num_res_blocks < 1 or self.intermediate < 1:
            raise ValueError("DiT depth and FFN intermediate must be positive")
        if self.image_tokens_per_img < 1 or math.isqrt(self.image_tokens_per_img) ** 2 != self.image_tokens_per_img:
            raise ValueError("DiT requires a complete square image-token grid")
        self.num_heads = width // self.head_dim
        self.mlp_ratio = self.intermediate / width
        self.grad_checkpointing = bool(getattr(config, "image_flow_grad_checkpointing", True))
        self.endpoint_time = float(getattr(config, "image_flow_time_scale", 1000.0))
        self.flow_head_attention_contract = "joint_bidirectional"
        self.conditioning_mode = "adaln"
        self.input_proj = nn.Linear(self.in_channels, width)
        self.cond_embed = nn.Linear(config.hidden_size, width)
        self.time_embed = TimestepEmbedder(width)
        self.blocks = nn.ModuleList([
            AdaLNZeroDiTBlock(width, self.head_dim, self.intermediate)
            for _ in range(self.num_res_blocks)
        ])
        self.final_layer = FinalLayer(width, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _xavier_uniform_init_fp32_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        _normal_init_fp32_(self.time_embed.mlp[0].weight, std=0.02)
        _normal_init_fp32_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def position_contract(self):
        return dict(
            schema="z2_position_v1", architecture="single_stream_z2",
            additive_image_position=False, rope_mode="row_col_2d",
            rope_axis_dims=[self.head_dim // 2, self.head_dim // 2], rotate_value=False,
            cross_token_attention=True, shared_time_per_image=True, uses_clean_target_content=False,
        )

    @staticmethod
    def cache_contract():
        return dict(
            schema="z2_fixed_condition_v1", content_cache=False,
            backbone_condition_cached=True, refresh_backbone_each_velocity=False,
            conditioning="backbone_plus_time_adaln", head_kv_source="modulated_current_noisy_stream",
        )

    def forward(self, x_t, times, condition):
        if (x_t.ndim != 3 or condition.ndim != 3 or x_t.shape[:2] != condition.shape[:2]
                or x_t.shape[1:] != (self.image_tokens_per_img, self.in_channels)
                or condition.shape[-1] != self.cond_embed.in_features):
            raise ValueError("Z2 requires aligned complete noisy-image and condition grids")
        if times.shape != (x_t.shape[0],):
            raise ValueError("Z2 requires one shared time per image")
        dtype = self.input_proj.weight.dtype
        hidden = self.input_proj(x_t.to(dtype))
        modulation = self.cond_embed(condition.to(dtype)) + self.time_embed(
            times * self.endpoint_time,
        )[:, None, :]
        positions = torch.arange(self.image_tokens_per_img, device=hidden.device)[None]
        rope = build_local_row_col_rope(
            positions, image_tokens_per_img=self.image_tokens_per_img, head_dim=self.head_dim,
            axis_dims=(self.head_dim // 2, self.head_dim // 2), dtype=hidden.dtype,
        )
        for block in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, modulation, rope, use_reentrant=False)
            else:
                hidden = block(hidden, modulation, rope)
        return self.final_layer(hidden, modulation)
