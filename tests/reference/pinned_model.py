# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0; see licenses/Apache-2.0.txt.
# Modified for UMM v3 from unified-mm commit 193ec90300af5fee4bc4b7e8a426492a908a0447.
"""Frozen arithmetic reference extracted from unified-mm 193ec90300af5fee4bc4b7e8a426492a908a0447.

Model class bodies below are verbatim source. Only unavailable framework
annotations/decorators and the attention kernel dispatch are supplied locally.
The independent attention kernel computes softmax(QK^T)V directly in FP32.
No v3 implementation is imported here.
"""
from __future__ import annotations
from typing import Optional, Unpack, TypedDict
from transformers.cache_utils import Cache, DynamicCache
try:
    import torch_npu
except ImportError:
    torch_npu = None  # Reference tests execute only the original CPU arithmetic.
class FlashAttentionKwargs(TypedDict):
    pass
class TransformersKwargs(TypedDict):
    pass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
ACT2FN = {"silu": F.silu}
Qwen3Config = object
GradientCheckpointingLayer = nn.Module
CANONICAL_IMAGE_LATENT_DIM = 16
CANONICAL_IMAGE_TOKENS = 256
class Qwen3PreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def post_init(self):
        pass  # Numerical tests load identical state_dicts before forward.
class BaseModelOutputWithPast(dict):
    def __getattr__(self, name):
        return self[name]
def validate_model_image_layout(config):
    pass  # Tests use a complete 2x2 grid to keep arithmetic verification small.
def auto_docstring(obj=None, **kwargs):
    return obj if obj is not None else (lambda fn: fn)
ROPE_INIT_FUNCTIONS = {}
def use_kernel_forward_from_hub(*args, **kwargs):
    return lambda cls: cls
def dynamic_rope_update(fn):
    return fn
def deprecate_kwarg(*args, **kwargs):
    return lambda fn: fn

def compiled_flex_attention(query, key, value, attention_mask, scaling, enable_gqa):
    if key.shape[1] != query.shape[1]:
        key = key.repeat_interleave(query.shape[1] // key.shape[1], 1)
        value = value.repeat_interleave(query.shape[1] // value.shape[1], 1)
    if isinstance(attention_mask, tuple):
        safe, valid_rows = attention_mask
        allowed = (~safe) & valid_rows
    else:
        allowed = attention_mask[:, None]
    valid = allowed.any(-1, keepdim=True)
    scores = (query.float() @ key.float().transpose(-1,-2)) * scaling
    scores = scores.masked_fill(~allowed, -torch.inf)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    return ((scores.softmax(-1) @ value.float()) * valid).to(query.dtype)



def build_row_col_position_ids(
    token_types: torch.Tensor,
    image_tokens_per_img: int,
) -> torch.Tensor:
    """Build two-axis multimodal coordinates with exact pure-text 1D compatibility.

    Text, special, and padding tokens receive ``(p, p)``. Each contiguous image
    span is anchored at the current running cursor and laid out on a square grid.
    After an image span, the cursor advances by the canonical spatial extent
    instead of the flattened image-token count.
    """

    if token_types.ndim != 2:
        raise ValueError(f"token_types must be [B,L], got {tuple(token_types.shape)}")
    image_tokens_per_img = int(image_tokens_per_img)
    side = int(math.isqrt(image_tokens_per_img))
    if side * side != image_tokens_per_img:
        raise ValueError(
            f"row/column image positions require a square grid, got {image_tokens_per_img} tokens"
        )
    canonical_spatial_extent = side

    batch_size, seq_len = token_types.shape
    image_mask = token_types == 1
    previous_is_image = torch.cat(
        [
            torch.zeros(batch_size, 1, device=token_types.device, dtype=torch.bool),
            image_mask[:, :-1],
        ],
        dim=1,
    )
    span_starts = image_mask & ~previous_is_image
    span_count = span_starts.long().cumsum(dim=1)

    non_image = (~image_mask).long()
    non_image_before = non_image.cumsum(dim=1) - non_image
    sequence_positions = torch.arange(
        seq_len,
        device=token_types.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)
    start_locations = torch.where(
        span_starts,
        sequence_positions,
        torch.zeros_like(sequence_positions),
    )
    latest_start = torch.cummax(start_locations, dim=1).values
    local = sequence_positions - latest_start

    image_anchor = non_image_before + (
        (span_count - 1).clamp_min(0) * canonical_spatial_extent
    )
    text_position = non_image_before + span_count * canonical_spatial_extent
    rows = image_anchor + local.div(side, rounding_mode="floor")
    cols = image_anchor + (local % side)
    row_ids = torch.where(image_mask, rows, text_position)
    col_ids = torch.where(image_mask, cols, text_position)
    return torch.stack([row_ids, col_ids], dim=0)

def _interleaved_coordinate_axes(axis_dims: tuple[int, int], device) -> torch.Tensor:
    remaining = [int(axis_dims[0]) // 2, int(axis_dims[1]) // 2]
    axes = []
    while remaining[0] or remaining[1]:
        for axis in (0, 1):
            if remaining[axis]:
                axes.append(axis)
                remaining[axis] -= 1
    return torch.tensor(axes, device=device, dtype=torch.long)

def build_local_row_col_rope(
    positions: torch.Tensor,
    *,
    image_tokens_per_img: int,
    head_dim: int,
    axis_dims: tuple[int, int],
    rope_theta: float = 10000.0,
    dtype: torch.dtype | None = None,
    validate_positions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build interleaved local-grid row/column RoPE cos/sin in FP32."""

    if positions.ndim != 2:
        raise ValueError(f"positions must be [B,L], got {tuple(positions.shape)}")
    image_tokens_per_img = int(image_tokens_per_img)
    side = int(math.isqrt(image_tokens_per_img))
    if side * side != image_tokens_per_img:
        raise ValueError(
            "Flow-head row/column RoPE requires a square image-token grid, "
            f"got {image_tokens_per_img}."
        )
    head_dim = int(head_dim)
    axis_dims = tuple(int(item) for item in axis_dims)
    if len(axis_dims) != 2 or sum(axis_dims) != head_dim:
        raise ValueError(
            f"axis_dims={axis_dims} must contain two entries summing to {head_dim}."
        )
    if any(dim <= 0 or dim % 2 for dim in axis_dims):
        raise ValueError(f"axis_dims must be positive and even, got {axis_dims}.")
    positions = positions.to(dtype=torch.long)
    if validate_positions and positions.numel():
        min_position = int(positions.min().item())
        max_position = int(positions.max().item())
        if min_position < 0 or max_position >= image_tokens_per_img:
            raise ValueError(
                f"positions must be in [0,{image_tokens_per_img}), "
                f"got min={min_position}, max={max_position}."
            )

    rows = positions.div(side, rounding_mode="floor")
    cols = positions % side
    coordinates = torch.stack([rows, cols], dim=-1).float()
    coordinate_axes = _interleaved_coordinate_axes(axis_dims, positions.device)
    selected_coordinates = coordinates.index_select(-1, coordinate_axes)
    inv_freq = 1.0 / (
        float(rope_theta)
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=positions.device,
                dtype=torch.float32,
            )
            / float(head_dim)
        )
    )
    frequencies = selected_coordinates.float() * inv_freq
    phases = torch.cat([frequencies, frequencies], dim=-1)
    cos = phases.cos()
    sin = phases.sin()
    if dtype is not None:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)

def _xavier_uniform_init_fp32_(tensor: torch.Tensor):
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(value)
        tensor.copy_(value.to(dtype=tensor.dtype))

def _normal_init_fp32_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0):
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        torch.nn.init.normal_(value, mean=mean, std=std)
        tensor.copy_(value.to(dtype=tensor.dtype))

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

class Z2FlowHead(nn.Module):
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

class ImageTokenEmbedder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_size,
        image_tokens_per_img=256,
        initializer_range=0.02,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.image_tokens_per_img = int(image_tokens_per_img)
        self.initializer_range = float(initializer_range)
        self.z_proj = nn.Linear(self.latent_dim, self.hidden_size, bias=True)
        self.last_init_stats = None
        self._reset_parameters()

    def _reset_parameters(self):
        # Finalized pure-2D projection initialization.
        weight_std = self.initializer_range / math.sqrt(
            2.0 * float(self.latent_dim)
        )
        init_stats = {
            "init_mode": "pure_2d",
            "z_proj_weight_std": float(weight_std),
            "additive_image_position": False,
        }

        _normal_init_fp32_(self.z_proj.weight, mean=0.0, std=float(weight_std))
        nn.init.zeros_(self.z_proj.bias)
        self.last_init_stats = init_stats

    def _reset_image_slots(self):
        self._reset_parameters()

    @property
    def weight_dtype(self):
        return self.z_proj.weight.dtype

    def embed_latents(self, latents):
        x = latents.to(dtype=self.z_proj.weight.dtype)
        return self.z_proj(x)

    def embed_mask(
        self,
        shape,
        mask_embedding,
        *,
        debug_finite: bool = False,
        debug_label: str = "",
    ):
        x = mask_embedding.to(dtype=self.z_proj.weight.dtype)
        return x.expand(tuple(shape) + (x.shape[-1],))

    def forward(self, latents):
        return self.embed_latents(latents)

def _compute_default_rope_parameters(config: Qwen3Config, device=None):
    base = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    return inv_freq, 1.0

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if getattr(hidden_states, "is_npu", False):
            return torch_npu.npu_rms_norm(
                hidden_states, self.weight, epsilon=self.variance_epsilon
            )[0]
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def apply_rotary_pos_emb(X0_q, XT_q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    X0_q_embed = (X0_q * cos) + (rotate_half(X0_q) * sin)
    XT_q_embed = (XT_q * cos) + (rotate_half(XT_q) * sin) if XT_q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return X0_q_embed, XT_q_embed, k_embed

def _debug_require_finite_tensor(
    enabled: bool,
    label: str,
    component: str,
    tensor: torch.Tensor | None,
) -> None:
    if not enabled or tensor is None:
        return
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    finite_values = tensor[finite].float()
    nonfinite_rows = None
    nonfinite_axis1 = None
    nonfinite_last_dim = None
    if tensor.ndim >= 1:
        row_finite = finite.reshape(tensor.shape[0], -1).all(dim=1)
        row_indices = (~row_finite).nonzero(as_tuple=True)[0].detach().cpu().tolist()
        nonfinite_rows = {
            "count": len(row_indices),
            "first": row_indices[:64],
        }
    if tensor.ndim >= 2:
        axis1_nonfinite = (
            (~finite).movedim(1, 0).reshape(tensor.shape[1], -1).any(dim=1)
        )
        axis1_indices = (
            axis1_nonfinite.nonzero(as_tuple=True)[0].detach().cpu().tolist()
        )
        nonfinite_axis1 = {
            "count": len(axis1_indices),
            "first": axis1_indices[:64],
        }
        last_nonfinite = (~finite).reshape(-1, tensor.shape[-1]).any(dim=0)
        last_indices = last_nonfinite.nonzero(as_tuple=True)[0].detach().cpu().tolist()
        nonfinite_last_dim = {
            "count": len(last_indices),
            "first": last_indices[:64],
        }
    finite_range = (
        [float(finite_values.min().item()), float(finite_values.max().item())]
        if finite_values.numel()
        else None
    )
    raise FloatingPointError(
        "non-finite tensor inside selfless-flow backbone: "
        f"label={label!r}, component={component!r}, shape={tuple(tensor.shape)}, "
        f"nonfinite_rows={nonfinite_rows}, "
        f"nonfinite_axis1={nonfinite_axis1}, nonfinite_last_dim={nonfinite_last_dim}, "
        f"nonfinite={int((~finite).sum().item())}/{tensor.numel()}, "
        f"finite_range={finite_range}, "
        f"finite_max_abs={float(finite_values.abs().max().item()) if finite_values.numel() else None}"
    )

def compute_image_local_positions(
    token_types: torch.Tensor,
    image_tokens_per_img: int,
) -> torch.Tensor:
    is_image = token_types == 1
    local_positions = torch.full(
        token_types.shape,
        -1,
        device=token_types.device,
        dtype=torch.long,
    )
    if not is_image.any():
        return local_positions

    for batch_idx in range(is_image.shape[0]):
        positions = is_image[batch_idx].nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        gaps = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=True)[0] + 1
        boundaries = torch.cat(
            [
                torch.zeros(1, device=positions.device, dtype=torch.long),
                gaps,
                torch.full(
                    (1,), positions.numel(), device=positions.device, dtype=torch.long
                ),
            ]
        )
        for start_idx, end_idx in zip(
            boundaries[:-1].tolist(), boundaries[1:].tolist()
        ):
            span = positions[start_idx:end_idx]
            span_len = span.numel()
            if span_len > image_tokens_per_img:
                raise ValueError(
                    f"image span length {span_len} exceeds image_tokens_per_img={image_tokens_per_img}"
                )
            local_positions[batch_idx, span] = torch.arange(
                span_len,
                device=token_types.device,
                dtype=torch.long,
            )
    return local_positions

def _to_bool_atten_mask(attention_mask) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a stored dense additive/bool mask into npu_fusion_attention bool mask (True = disallow).

    npu_fusion_attention 在全 mask 行（所有 kv 均 disallow）上会产生 NaN/垃圾值，
    而 FlexAttention 对这些行返回严格的 0。这里先放行 kv[0] 避免 NaN，再由
    调用方使用 valid_rows 将输出恢复为严格的 0。

    无 sync 版本：始终在 GPU 上执行 clamp，避免 bool(row_all.any()) 的
    device→host sync（每层 2 次 × 28 层 = 56 次 sync，~200ms/step）。
    """
    if attention_mask.dtype == torch.bool:
        mask = attention_mask
    else:
        mask = attention_mask < 0
    row_all = mask.all(dim=-1, keepdim=True)
    # 无条件执行：对非全 mask 行是 no-op（mask[..., 0] & True = mask[..., 0]）
    mask = mask.clone()
    mask[..., 0] = mask[..., 0] & ~row_all.squeeze(-1)
    return mask, ~row_all

class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        gate_mode = str(
            getattr(config, "backbone_attention_output_gate", "none")
        ).strip().lower()
        if gate_mode not in {"none", "per_head_identity_sigmoid"}:
            raise ValueError(
                "backbone_attention_output_gate must be one of "
                "{'none', 'per_head_identity_sigmoid'}, "
                f"got {gate_mode!r}."
            )
        self.backbone_attention_output_gate = gate_mode
        self.backbone_attention_gate_scale = 2.0
        self.last_gate_stats: dict[str, torch.Tensor] = {}

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        if self.backbone_attention_output_gate != "none":
            # nn.Linear initializes its weight randomly. Restore the CPU RNG after
            # construction so enabling the identity gate cannot perturb the
            # initialization of any pre-existing downstream module.
            with torch.random.fork_rng(devices=[]):
                self.attn_output_gate_proj = nn.Linear(
                    config.hidden_size,
                    config.num_attention_heads,
                    bias=False,
                )
            self.reset_attention_output_gate()
        else:
            self.attn_output_gate_proj = None

    def reset_attention_output_gate(self) -> None:
        if self.attn_output_gate_proj is None:
            return
        with torch.no_grad():
            nn.init.zeros_(self.attn_output_gate_proj.weight)

    @staticmethod
    def _gate_summary(
        gate: torch.Tensor,
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        values = gate.detach().float()
        return {
            f"{prefix}/mean": values.mean(),
            f"{prefix}/std": values.std(unbiased=False),
            f"{prefix}/mean_abs_delta": (values - 1.0).abs().mean(),
            f"{prefix}/saturation_low": (values < 0.1).float().mean(),
            f"{prefix}/saturation_high": (values > 1.9).float().mean(),
        }

    @staticmethod
    def _masked_gate_mean(
        gate: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.to(device=gate.device, dtype=torch.bool)
        values = gate.detach().float()
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(values)
        weights = mask.to(dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def _record_gate_stats(
        self,
        gate: torch.Tensor,
        *,
        stream: str,
        token_types: torch.Tensor | None,
        flow_sigma: torch.Tensor | None,
        detailed: bool,
    ) -> None:
        self.last_gate_stats.update(self._gate_summary(gate, prefix=stream))
        if not detailed or token_types is None:
            return

        token_types = token_types.to(device=gate.device)
        for token_type, label in (
            (0, "text"),
            (1, "image"),
            (2, "special"),
            (3, "padding"),
        ):
            value = self._masked_gate_mean(gate, token_types.eq(token_type))
            if value is not None:
                self.last_gate_stats[f"{stream}/{label}/mean"] = value

        if flow_sigma is None:
            return
        flow_sigma = flow_sigma.to(device=gate.device, dtype=torch.float32)
        if tuple(flow_sigma.shape) != tuple(token_types.shape):
            raise ValueError(
                "flow_sigma must align with token_types when detailed backbone "
                f"gate statistics are requested: {tuple(flow_sigma.shape)} != "
                f"{tuple(token_types.shape)}."
            )
        image_mask = token_types.eq(1)
        positive_inf = torch.full_like(flow_sigma, torch.inf)
        negative_inf = torch.full_like(flow_sigma, -torch.inf)
        row_min = torch.where(image_mask, flow_sigma, positive_inf).amin(
            dim=1, keepdim=True
        )
        row_max = torch.where(image_mask, flow_sigma, negative_inf).amax(
            dim=1, keepdim=True
        )
        relative_order = (flow_sigma - row_min) / (row_max - row_min).clamp_min(
            1.0e-12
        )
        for quartile in range(4):
            lower = quartile / 4.0
            upper = (quartile + 1) / 4.0
            if quartile == 3:
                quartile_mask = image_mask & relative_order.ge(lower)
            else:
                quartile_mask = (
                    image_mask
                    & relative_order.ge(lower)
                    & relative_order.lt(upper)
                )
            value = self._masked_gate_mean(gate, quartile_mask)
            if value is not None:
                self.last_gate_stats[
                    f"{stream}/image_order_q{quartile + 1}/mean"
                ] = value

    def _apply_attention_output_gate(
        self,
        attention_output: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        *,
        stream: str,
        token_types: torch.Tensor | None,
        flow_sigma: torch.Tensor | None,
        record_stats: bool,
        detailed_stats: bool,
    ) -> torch.Tensor | None:
        if attention_output is None or hidden_states is None:
            return attention_output
        if self.attn_output_gate_proj is None:
            return attention_output
        gate = (
            torch.sigmoid(self.attn_output_gate_proj(hidden_states))
            * self.backbone_attention_gate_scale
        )
        if record_stats:
            self._record_gate_stats(
                gate,
                stream=stream,
                token_types=token_types,
                flow_sigma=flow_sigma,
                detailed=detailed_stats,
            )
        return attention_output * gate.transpose(1, 2).unsqueeze(-1)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        X0_hidden_states: torch.Tensor,
        XT_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        content_attention_mask: torch.Tensor | None = None,
        past_key_values: Optional[Cache] = None,
        cache_read_only: bool = False,
        cache_write_mask: Optional[torch.BoolTensor] = None,
        cache_write_prefix: Optional[int] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        debug_finite = bool(kwargs.pop("debug_finite_backbone", False))
        debug_label = str(kwargs.pop("debug_backbone_label", ""))
        record_gate_stats = bool(
            kwargs.pop("record_backbone_gate_stats", False)
        )
        detailed_gate_stats = (
            str(kwargs.pop("backbone_gate_stats_level", "summary")).strip().lower()
            == "detailed"
        )
        self.last_gate_stats.clear()
        prefix = f"layers.{self.layer_idx}.self_attn"
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            f"{prefix}.input_x0",
            X0_hidden_states,
        )
        input_shape = X0_hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        XT_query_states = (
            self.q_proj(XT_hidden_states).view(hidden_shape)
            if XT_hidden_states is not None
            else None
        )
        X0_query_states = self.q_proj(X0_hidden_states).view(hidden_shape)
        X0_key_states = self.k_proj(X0_hidden_states).view(hidden_shape)
        X0_value_states = self.v_proj(X0_hidden_states).view(hidden_shape)
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.q_proj", X0_query_states
        )
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.k_proj", X0_key_states
        )
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.v_proj", X0_value_states
        )
        XT_query_states = (
            self.q_norm(XT_query_states).transpose(1, 2)
            if XT_query_states is not None
            else None
        )
        X0_query_states = self.q_norm(X0_query_states).transpose(1, 2)
        X0_key_states = self.k_norm(X0_key_states).transpose(1, 2)
        X0_value_states = X0_value_states.transpose(1, 2)
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.q_norm", X0_query_states
        )
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.k_norm", X0_key_states
        )

        cos, sin = position_embeddings
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.rope_cos", cos
        )
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.rope_sin", sin
        )
        (
            X0_query_states,
            XT_query_states,
            X0_key_states,
        ) = apply_rotary_pos_emb(
            X0_query_states, XT_query_states, X0_key_states, cos, sin
        )
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.rotary_q", X0_query_states
        )
        _debug_require_finite_tensor(
            debug_finite, debug_label, f"{prefix}.rotary_k", X0_key_states
        )

        if past_key_values is not None and cache_read_only:
            cache_layer = past_key_values.layers[self.layer_idx]
            if not cache_layer.is_initialized:
                raise RuntimeError(
                    "Cannot read an uninitialized backbone KV cache at "
                    f"layer {self.layer_idx}."
                )
            X0_key_states, X0_value_states = past_key_values.read(self.layer_idx)
        elif past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "cache_write_mask": cache_write_mask,
                "cache_write_prefix": cache_write_prefix,
            }
            X0_key_states, X0_value_states = past_key_values.update(
                X0_key_states, X0_value_states, self.layer_idx, cache_kwargs
            )

        # 检查是否需要 GQA (Grouped Query Attention)
        enable_gqa = self.config.num_attention_heads != self.config.num_key_value_heads
        x0_attention_mask = (
            attention_mask
            if content_attention_mask is None
            else content_attention_mask
        )

        X0_attn_output = compiled_flex_attention(
            X0_query_states,
            X0_key_states,
            X0_value_states,
            x0_attention_mask,
            self.scaling,
            enable_gqa,
        )
        XT_attn_output = (
            compiled_flex_attention(
                XT_query_states,
                X0_key_states,
                X0_value_states,
                attention_mask,
                self.scaling,
                enable_gqa,
            )
            if XT_query_states is not None
            else None
        )
        attn_weights = None

        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            f"{prefix}.flex_attention_output",
            X0_attn_output,
        )
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            f"{prefix}.flex_attention_output_xt",
            XT_attn_output,
        )

        X0_attn_output = self._apply_attention_output_gate(
            X0_attn_output,
            X0_hidden_states,
            stream="x0",
            token_types=kwargs.get("token_types", None),
            flow_sigma=kwargs.get("flow_sigma", None),
            record_stats=record_gate_stats,
            detailed_stats=detailed_gate_stats,
        )
        XT_attn_output = self._apply_attention_output_gate(
            XT_attn_output,
            XT_hidden_states,
            stream="xt",
            token_types=kwargs.get("token_types", None),
            flow_sigma=kwargs.get("flow_sigma", None),
            record_stats=record_gate_stats,
            detailed_stats=detailed_gate_stats,
        )

        # attn_output 形状: (batch, heads, seq_len, head_dim)
        # 需要转换回 (batch, seq_len, hidden_size)
        X0_attn_output = (
            X0_attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        )
        XT_attn_output = (
            XT_attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            if XT_attn_output is not None
            else None
        )
        X0_attn_output = self.o_proj(X0_attn_output)
        XT_attn_output = (
            self.o_proj(XT_attn_output) if XT_attn_output is not None else None
        )
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            f"{prefix}.o_proj",
            X0_attn_output,
        )

        return X0_attn_output, XT_attn_output, attn_weights

class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        X0_hidden_states: torch.Tensor,
        XT_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        content_attention_mask: torch.Tensor | None = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        X0_residual = X0_hidden_states
        XT_residual = XT_hidden_states if XT_hidden_states is not None else None
        X0_hidden_states = self.input_layernorm(X0_hidden_states)
        XT_hidden_states = (
            self.input_layernorm(XT_hidden_states)
            if XT_hidden_states is not None
            else None
        )
        # Self Attention
        X0_hidden_states, XT_hidden_states, _ = self.self_attn(
            X0_hidden_states=X0_hidden_states,
            XT_hidden_states=XT_hidden_states,
            attention_mask=attention_mask,
            content_attention_mask=content_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        X0_hidden_states = X0_residual + X0_hidden_states
        XT_hidden_states = (
            XT_residual + XT_hidden_states if XT_hidden_states is not None else None
        )

        # Fully Connected
        X0_residual = X0_hidden_states
        XT_residual = XT_hidden_states if XT_hidden_states is not None else None
        X0_hidden_states = self.post_attention_layernorm(X0_hidden_states)
        XT_hidden_states = (
            self.post_attention_layernorm(XT_hidden_states)
            if XT_hidden_states is not None
            else None
        )
        X0_hidden_states = self.mlp(X0_hidden_states)
        XT_hidden_states = (
            self.mlp(XT_hidden_states) if XT_hidden_states is not None else None
        )
        X0_hidden_states = X0_residual + X0_hidden_states
        XT_hidden_states = (
            XT_residual + XT_hidden_states if XT_hidden_states is not None else None
        )
        return X0_hidden_states, XT_hidden_states

class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    @staticmethod
    def compute_default_rope_parameters(config: Qwen3Config, device=None):
        return _compute_default_rope_parameters(config, device)

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type in (None, "default"):
            self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(
                "default", _compute_default_rope_parameters
            )
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
                .to(x.device)
            )
            position_ids_expanded = position_ids[:, None, :].float()
        elif position_ids.ndim == 3 and position_ids.shape[0] == 2:
            num_frequency_pairs = int(self.inv_freq.numel())
            coordinate_axes = (
                torch.arange(
                    num_frequency_pairs,
                    device=position_ids.device,
                    dtype=torch.long,
                )
                % 2
            )
            selected_positions = position_ids.index_select(0, coordinate_axes).permute(
                1, 0, 2
            )
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[1], -1, 1)
                .to(x.device)
            )
            position_ids_expanded = selected_positions.float()
        else:
            raise ValueError(
                f"position_ids must have shape [B,L] or [2,B,L], got {tuple(position_ids.shape)}"
            )

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            if position_ids.ndim == 2:
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
            else:
                freqs = (
                    inv_freq_expanded.float() * position_ids_expanded.float()
                ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen3Model(Qwen3PreTrainedModel):
    supported_training_objectives = frozenset({"selfless_dual_stream"})

    def __init__(self, config: Qwen3Config):
        validate_model_image_layout(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.training_objective = str(
            getattr(config, "training_objective", "selfless_dual_stream")
        ).strip().lower()
        if self.training_objective not in self.supported_training_objectives:
            raise ValueError(
                "training_objective must be selfless_dual_stream, "
                f"got {self.training_objective!r}"
            )

        self.image_latent_dim = int(
            getattr(config, "image_latent_dim", CANONICAL_IMAGE_LATENT_DIM)
        )
        image_tokens_per_img = int(
            getattr(config, "image_tokens_per_img", CANONICAL_IMAGE_TOKENS)
        )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.image_token_embedder = ImageTokenEmbedder(
            latent_dim=self.image_latent_dim,
            hidden_size=config.hidden_size,
            image_tokens_per_img=image_tokens_per_img,
            initializer_range=getattr(config, "initializer_range", 0.02),
        )
        self.image_input_noise_strength = float(
            getattr(config, "image_input_noise_strength", 1.0e-2)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        self.reset_backbone_attention_output_gates()
        self.reset_image_token_embedder()

    def reset_backbone_attention_output_gates(self) -> None:
        for layer in self.layers:
            layer.self_attn.reset_attention_output_gate()

    def backbone_attention_gate_stats(self) -> dict[str, torch.Tensor]:
        stats: dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            for name, value in layer.self_attn.last_gate_stats.items():
                stats[f"backbone_gate/layer_{layer_idx:02d}/{name}"] = value
        return stats

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def reset_image_token_embedder(self):
        self.image_token_embedder._reset_parameters()
        return self.image_token_embedder.last_init_stats

    def _image_mask_token_id(self) -> int:
        return int(
            getattr(self.config, "image_mask_token_id", self.config.mask_token_id)
        )

    def _image_mask_embedding(self, device, dtype):
        token_id = self._image_mask_token_id()
        if token_id < 0 or token_id >= self.embed_tokens.num_embeddings:
            raise ValueError(
                f"image_mask_token_id={token_id} is outside embedding vocab size "
                f"{self.embed_tokens.num_embeddings}."
            )
        token = torch.tensor([token_id], device=device, dtype=torch.long)
        return self.embed_tokens(token)[0].to(dtype=dtype)

    def image_local_positions(self, token_types: torch.Tensor) -> torch.Tensor:
        return compute_image_local_positions(
            token_types,
            self.image_token_embedder.image_tokens_per_img,
        )

    def _validate_image_latents(self, image_latents: torch.Tensor):
        if image_latents.shape[-1] != self.image_latent_dim:
            raise ValueError(
                f"image_latents last dimension ({image_latents.shape[-1]}) must match "
                f"config.image_latent_dim ({self.image_latent_dim})."
            )

    def _maybe_add_image_input_noise(self, image_latents: torch.Tensor) -> torch.Tensor:
        strength = self.image_input_noise_strength
        if not self.training or strength <= 0.0:
            return image_latents
        return image_latents + torch.randn_like(image_latents) * float(strength)

    def _build_x0_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        token_types: torch.Tensor | None,
        image_latents: torch.Tensor | None,
        image_latent_mask: torch.Tensor | None,
        image_spans_present: bool | None = None,
        image_latents_are_noisy: bool = False,
        debug_finite: bool = False,
        debug_label: str = "",
        image_context: dict | None = None,
    ) -> torch.Tensor:
        if token_types is None:
            return self.embed_tokens(input_ids)

        token_types = token_types.to(input_ids.device)
        is_image = token_types == 1
        safe_input_ids = input_ids.masked_fill(is_image, self._image_mask_token_id())
        inputs_embeds = self.embed_tokens(safe_input_ids)
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            "input_embeddings.safe_token_lookup",
            inputs_embeds,
        )

        has_image = image_spans_present
        if has_image is None:
            has_image = bool(is_image.any())
        if has_image:
            if image_latents is None:
                raise ValueError(
                    "image_latents must be provided when token_types contains image tokens."
                )
            self._validate_image_latents(image_latents)
            image_latents = image_latents.to(
                device=input_ids.device,
                dtype=self.image_token_embedder.weight_dtype,
            )
            projected_latents = self.image_token_embedder(
                self._maybe_add_image_input_noise(image_latents)
                if not image_latents_are_noisy
                else image_latents
            ).to(dtype=inputs_embeds.dtype)
            if image_latent_mask is None:
                inputs_embeds = torch.where(
                    is_image.unsqueeze(-1),
                    projected_latents,
                    inputs_embeds,
                )
            else:
                image_latent_mask = image_latent_mask.to(
                    device=input_ids.device, dtype=torch.bool
                )
                use_image_latent = is_image & image_latent_mask
                use_image_mask = is_image & ~use_image_latent
                image_mask_embedding = self._image_mask_embedding(
                    device=input_ids.device,
                    dtype=self.image_token_embedder.weight_dtype,
                )
                _debug_require_finite_tensor(
                    debug_finite,
                    debug_label,
                    "input_embeddings.image_mask_token",
                    image_mask_embedding,
                )
                image_mask_values = image_mask_embedding.to(
                    dtype=inputs_embeds.dtype
                ).view(1, 1, -1).expand_as(inputs_embeds)
                inputs_embeds = torch.where(
                    use_image_mask.unsqueeze(-1),
                    image_mask_values,
                    inputs_embeds,
                )
                inputs_embeds = torch.where(
                    use_image_latent.unsqueeze(-1),
                    projected_latents,
                    inputs_embeds,
                ).to(dtype=inputs_embeds.dtype)
            _debug_require_finite_tensor(
                debug_finite,
                debug_label,
                "input_embeddings.observed_image_latents",
                projected_latents,
            )
        elif self.training:
            dummy_latent = torch.zeros(
                1,
                self.image_latent_dim,
                device=input_ids.device,
                dtype=self.image_token_embedder.weight_dtype,
            )
            inputs_embeds = (
                inputs_embeds + self.image_token_embedder(dummy_latent).sum() * 0.0
            )

        return inputs_embeds

    def _build_xt_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        token_types: torch.Tensor | None,
        image_spans_present: bool | None = None,
    ) -> torch.Tensor:
        xt_input_ids = torch.full_like(input_ids, self.config.mask_token_id)
        inputs_embeds = self.embed_tokens(xt_input_ids)
        if token_types is None:
            return inputs_embeds

        token_types = token_types.to(input_ids.device)
        is_image = token_types == 1
        has_image = image_spans_present
        if has_image is None:
            has_image = bool(is_image.any())
        if has_image:
            image_mask_embedding = self._image_mask_embedding(
                device=input_ids.device,
                dtype=self.image_token_embedder.weight_dtype,
            )
            image_mask_values = image_mask_embedding.to(
                dtype=inputs_embeds.dtype
            ).view(1, 1, -1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(
                is_image.unsqueeze(-1),
                image_mask_values,
                inputs_embeds,
            )
        return inputs_embeds

    def _resolve_text_ar_mode(
        self,
        *,
        requested: bool | None,
        calculate_likelihood: bool | None,
        require_image_query_stream: bool | None,
        token_types: torch.Tensor | None,
        image_latent_mask: torch.Tensor | None,
    ) -> bool:
        """Return whether this call uses an ablation-specific text AR stream.

        The production Selfless model never enables this mode.  A dedicated
        ablation model can override the hook without adding a behavior switch
        to the baseline implementation.
        """

        del (
            requested,
            calculate_likelihood,
            require_image_query_stream,
            token_types,
            image_latent_mask,
        )
        return False

    def _needs_query_stream(
        self,
        *,
        calculate_likelihood: bool | None,
        require_image_query_stream: bool | None,
        text_ar_mode: bool,
        image_spans_present: bool | None,
    ) -> bool:
        """Select the XT stream while preserving the baseline default."""

        del require_image_query_stream, text_ar_mode, image_spans_present
        return bool(self.training or calculate_likelihood)

    def _prepare_stream_attention_masks(
        self,
        attention_mask,
        content_attention_mask,
        *,
        text_ar_mode: bool,
        input_ids: torch.LongTensor | None,
        token_types: torch.Tensor | None,
        text_segment_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor | None,
    ):
        """Allow an isolated model class to supply a text-only content mask."""

        del (
            text_ar_mode,
            input_ids,
            token_types,
            text_segment_ids,
            past_key_values,
            cache_position,
        )
        return attention_mask, content_attention_mask

    def _finalize_stream_hidden(
        self,
        X0_hidden_states: torch.Tensor,
        XT_hidden_states: torch.Tensor | None,
        *,
        use_query_stream: bool,
        text_ar_mode: bool,
        token_types: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor | None,
    ) -> torch.Tensor:
        """Choose and normalize the stream returned by the baseline model."""

        del text_ar_mode, token_types, past_key_values, cache_position
        hidden_states = XT_hidden_states if use_query_stream else X0_hidden_states
        if hidden_states is None:
            raise RuntimeError("The selected Selfless stream was not constructed.")
        return self.norm(hidden_states)

    # @check_model_inputs
    @auto_docstring
    def forward(
        self,
        X0_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        content_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        X0_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        calculate_likelihood: Optional[bool] = None,
        return_x0_hidden_state: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        Args:
            X0_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for Flex Attention. Should be a block mask of shape `(batch_size, sequence_length)`.
            calculate_likelihood (`bool`, *optional*):
                Whether to calculate the likelihood of the input sequence. If `True`, the model will compute the likelihood using the XT stream, which is necessary for training. If `False`, the model will only compute the hidden states for the X0 stream, which can be used for efficient decoding.
        """
        if (X0_input_ids is None) ^ (X0_inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if attention_mask is None:
            raise ValueError(
                "attention_mask must be provided for selfless sigma-causal attention."
            )

        require_image_query_stream = kwargs.pop(
            "_require_image_query_stream", None
        )
        requested_text_ar_mode = kwargs.pop("_text_ar_mode", None)
        text_segment_ids = kwargs.pop("_text_segment_ids", None)
        text_ar_mode = self._resolve_text_ar_mode(
            requested=requested_text_ar_mode,
            calculate_likelihood=calculate_likelihood,
            require_image_query_stream=require_image_query_stream,
            token_types=kwargs.get("token_types", None),
            image_latent_mask=kwargs.get("image_latent_mask", None),
        )
        attention_mask, content_attention_mask = (
            self._prepare_stream_attention_masks(
                attention_mask,
                content_attention_mask,
                text_ar_mode=text_ar_mode,
                input_ids=X0_input_ids,
                token_types=kwargs.get("token_types", None),
                text_segment_ids=text_segment_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
        )

        # Every decoder layer consumes the same dense attention mask, and the
        # X0/XT streams share it as well.  Preparing it inside each attention
        # call repeated all/clone/index operations 56 times per microbatch on
        # the 28-layer training path.  Keep the safe mask and fully-masked-row
        # indicator device-resident and reuse them across all layers/streams.
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = _to_bool_atten_mask(attention_mask)
        if isinstance(content_attention_mask, torch.Tensor):
            content_attention_mask = _to_bool_atten_mask(
                content_attention_mask
            )

        debug_finite_backbone = bool(kwargs.get("debug_finite_backbone", False))
        debug_backbone_label = str(kwargs.get("debug_backbone_label", ""))
        token_types = kwargs.get("token_types", None)
        image_span_table = kwargs.pop("image_span_table", None)
        kwargs.pop("image_local_positions", None)
        image_spans_present = (
            image_span_table.shape[0] > 0
            if image_span_table is not None
            else None
        )

        if X0_inputs_embeds is None:
            X0_inputs_embeds = self._build_x0_inputs_embeds(
                input_ids=X0_input_ids,
                token_types=token_types,
                image_latents=kwargs.get("image_latents", None),
                image_latent_mask=kwargs.get("image_latent_mask", None),
                image_spans_present=image_spans_present,
                image_latents_are_noisy=bool(
                    kwargs.get("image_latents_are_noisy", False)
                ),
                debug_finite=debug_finite_backbone,
                debug_label=debug_backbone_label,
                image_context={
                    "attention_mask": attention_mask,
                    "content_attention_mask": content_attention_mask,
                    "past_key_values": past_key_values,
                    "cache_position": cache_position,
                    "cache_read_only": kwargs.get("cache_read_only", False),
                    "cache_write_prefix": kwargs.get("cache_write_prefix"),
                    "cache_write_mask": kwargs.get("cache_write_mask"),
                    "use_cache": use_cache,
                },
            )
        _debug_require_finite_tensor(
            debug_finite_backbone,
            debug_backbone_label,
            "input_embeddings.x0",
            X0_inputs_embeds,
        )

        use_query_stream = self._needs_query_stream(
            calculate_likelihood=calculate_likelihood,
            require_image_query_stream=require_image_query_stream,
            text_ar_mode=text_ar_mode,
            image_spans_present=image_spans_present,
        )
        if use_query_stream:
            XT_inputs_embeds = self._build_xt_inputs_embeds(
                input_ids=X0_input_ids,
                token_types=token_types,
                image_spans_present=image_spans_present,
            )
        else:
            XT_inputs_embeds = None

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + X0_inputs_embeds.shape[1],
                device=X0_inputs_embeds.device,
            )

        if position_ids is None:
            if token_types is not None:
                position_ids = build_row_col_position_ids(
                    token_types.to(device=X0_inputs_embeds.device),
                    self.image_token_embedder.image_tokens_per_img,
                )
            else:
                position_ids = cache_position.unsqueeze(0)

        X0_hidden_states = X0_inputs_embeds
        XT_hidden_states = XT_inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(X0_hidden_states, position_ids)
        for component_idx, component in enumerate(position_embeddings):
            _debug_require_finite_tensor(
                debug_finite_backbone,
                debug_backbone_label,
                f"rotary_embedding.{component_idx}",
                component,
            )
        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            X0_hidden_states, XT_hidden_states = decoder_layer(
                X0_hidden_states,
                XT_hidden_states,
                attention_mask,
                content_attention_mask=content_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            _debug_require_finite_tensor(
                debug_finite_backbone,
                debug_backbone_label,
                f"layers.{layer_idx}.output_x0",
                X0_hidden_states,
            )
            _debug_require_finite_tensor(
                debug_finite_backbone,
                debug_backbone_label,
                f"layers.{layer_idx}.output_xt",
                XT_hidden_states,
            )

        hidden_states = self._finalize_stream_hidden(
            X0_hidden_states,
            XT_hidden_states,
            use_query_stream=use_query_stream,
            text_ar_mode=text_ar_mode,
            token_types=token_types,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        if return_x0_hidden_state:
            output["x0_last_hidden_state"] = self.norm(X0_hidden_states)
        return output

class OriginalSampler:
    def sample(self, z, temperature=1., cfg=1., cfg_schedule="constant", solver=None,
                   num_steps=None, initial_noise=None, return_trace=False, debug_finite=False, **unused):
            steps = self.num_sampling_steps if num_steps is None else int(num_steps)
            solver = str(solver or self.solver).lower()
            if solver not in {"heun", "euler"} or steps < 1:
                raise ValueError("Joint DiT requires Heun/Euler and positive steps")
            paired = float(cfg) != 1.
            if paired and z.shape[0] % 2:
                raise ValueError("CFG conditions must contain paired conditional/unconditional rows")
            batch = z.shape[0] // (2 if paired else 1)
            shape = (batch, self.image_tokens_per_img, self.in_channels)
            if initial_noise is not None and tuple(initial_noise.shape) != shape:
                raise ValueError(f"initial_noise must have shape {shape}")
            x = (torch.randn(shape, device=z.device, dtype=torch.float32) if initial_noise is None
                 else initial_noise.to(device=z.device, dtype=torch.float32).clone()) * float(temperature)
            times = torch.linspace(0., 1., steps + 1, device=z.device, dtype=torch.float32)
            for step in range(steps):
                t = times[step].expand(batch)
                t_next = times[step + 1].expand(batch)
                dt = times[step + 1] - times[step]
                cfg_now = self._scheduled_cfg(float(cfg), cfg_schedule, step / steps)
                velocity = self._guided_velocity(x, t, z, cfg_now).float()
                predictor = x + dt * velocity
                if solver == "heun":
                    cfg_next = self._scheduled_cfg(float(cfg), cfg_schedule, (step + 1) / steps)
                    next_velocity = self._guided_velocity(predictor, t_next, z, cfg_next).float()
                    x = x + .5 * dt * (velocity + next_velocity)
                else:
                    x = predictor
                if debug_finite and not torch.isfinite(x).all():
                    raise FloatingPointError(f"Nonfinite joint DiT state at step {step}")
            output = x.to(self.net.input_proj.weight.dtype)
            calls = steps * (2 if solver == "heun" else 1)
            return (output, {"solver": solver, "steps": steps, "flow_head_calls": calls}) if return_trace else output

class OriginalGuidance:


    @staticmethod

    def _scheduled_cfg(cfg: float, schedule: str | None, progress: float) -> float:
            schedule = str(schedule or "constant").lower()
            if cfg == 1.0 or schedule in {"constant", "none", "off", ""}:
                return float(cfg)
            if schedule == "linear":
                progress = min(1.0, max(0.0, float(progress)))
                return 1.0 + (float(cfg) - 1.0) * progress
            raise ValueError(
                f"Unknown image flow cfg_schedule={schedule!r}; expected constant or linear."
            )

    def _guided_velocity(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            z: torch.Tensor,
            cfg: float,
        ) -> torch.Tensor:
            if cfg == 1.0 and z.shape[0] == x.shape[0]:
                return self.velocity(x, t, z)
            if z.shape[0] != x.shape[0] * 2:
                raise ValueError(
                    "CFG requires conditional/unconditional conditions concatenated "
                    f"on the first axis, got x={tuple(x.shape)}, z={tuple(z.shape)}"
                )
            paired_x = torch.cat([x, x], dim=0)
            paired_t = torch.cat([t, t], dim=0)
            conditional, unconditional = self.velocity(paired_x, paired_t, z).chunk(
                2,
                dim=0,
            )
            return unconditional + float(cfg) * (conditional - unconditional)
