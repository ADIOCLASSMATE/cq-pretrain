# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""Original Wan2.1 diffusion backbone and attention helpers in one module.

Upstream: https://github.com/Wan-Video/Wan2.1/tree/9737cba9c1c3c4d04b33fcad41c111989865d315
Sources:
    wan/modules/model.py
    SHA256: c1572ade3bf7345bd4c00d4a19535788fab62f9c08b5aeb15f6d01f4435bb47a
    wan/modules/attention.py
    SHA256: 23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc

The upstream implementations are unchanged; imports and exports are consolidated
so the model does not depend on a separate Wan checkout. Diffusers still provides
configuration and pretrained loading. The original CUDA/FlashAttention execution
requirements are preserved; NPU adaptation is a separate step.

Load the T2V-1.3B checkpoint with WanModel.from_pretrained(local_model_path,
local_files_only=True); its config selects the 1.3B dimensions instead of the
upstream constructor defaults. Inputs are latent videos, timesteps and encoded
text context; the VAE, text encoder and sampling pipeline are separate components.
"""

import math
import warnings

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

__all__ = ['WanModel', 'flash_attention', 'attention']


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False



def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
# CQ BEGIN: UMM conditioning, loading and accelerator-compatible Wan operations
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import ModelMixin

def initialize_normal(tensor, std):
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        nn.init.normal_(value, std=std)
        tensor.copy_(value.to(tensor.dtype))



__all__ = ["Wan21Generator"]


class _WanCheckpoint(WanModel):
    # Upstream ignores these fields on BOTH serialization and deserialization.
    # Honor them when supplied by a checkpoint or explicit loading overrides,
    # without modifying the original class or its parameter implementation.
    ignore_for_config = []


def _linear(layer, x):
    """Also support explicitly converted BF16 weights outside autocast."""
    return layer(x.to(layer.weight.dtype))


def _attention(q, k, v, key_lengths, backend):
    # Keep padded queries. The upstream flash helper reshapes its result to
    # [batch, padded_query_length, ...], so compact query lengths cannot be used.
    q, k = q.to(v.dtype), k.to(v.dtype)
    flash_available = FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
    can_flash = (q.device.type == "cuda" and flash_available and q.shape[-1] <= 256
                 and v.dtype in (torch.float16, torch.bfloat16))
    if backend == "flash" and not can_flash:
        raise ValueError("flash attention requires CUDA, FlashAttention, FP16/BF16 and head_dim <= 256")
    if backend == "flash" or (backend == "auto" and can_flash):
        return flash_attention(q, k, v, k_lens=key_lengths)

    allowed = torch.arange(k.shape[1], device=q.device)[None] < key_lengths[:, None]
    if backend == "npu" or (backend == "auto" and q.device.type == "npu"):
        if q.device.type != "npu":
            raise ValueError("npu attention requires NPU tensors")
        import torch_npu
        # NPU uses True for blocked, unlike SDPA. Materialize the query axis
        # because fusion attention does not implement every SDPA broadcast case.
        blocked = (~allowed[:, None, None]).expand(-1, 1, q.shape[1], -1).contiguous()
        return torch_npu.npu_fusion_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            head_num=q.shape[2], input_layout="BSND", atten_mask=blocked,
            scale=q.shape[-1] ** -0.5, keep_prob=1.0, sparse_mode=0,
        )[0]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=allowed[:, None, None], dropout_p=0.0, is_causal=False,
    ).transpose(1, 2)


def _apply_rope(x, rope):
    """Real arithmetic equivalent of Wan's adjacent complex-pair rotation."""
    cos, sin = rope
    real, imag = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((real * cos - imag * sin, imag * cos + real * sin), -1).flatten(-2)


class _LayerNorm(WanLayerNorm):
    def forward(self, x):
        return F.layer_norm(
            x.float(), self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(), self.eps,
        ).to(x.dtype)


class _SelfAttention(WanSelfAttention):
    def forward(self, x, seq_lens, rope):
        shape = (*x.shape[:2], self.num_heads, self.head_dim)
        q = self.norm_q(_linear(self.q, x)).view(shape)
        k = self.norm_k(_linear(self.k, x)).view(shape)
        v = _linear(self.v, x).view(shape)
        output = _attention(_apply_rope(q, rope), _apply_rope(k, rope), v,
                            seq_lens, self.attention_backend)
        return _linear(self.o, output.flatten(2))


class _CrossAttention(WanT2VCrossAttention):
    def forward(self, x, context, context_lens):
        shape = (x.shape[0], -1, self.num_heads, self.head_dim)
        q = self.norm_q(_linear(self.q, x)).view(shape)
        k = self.norm_k(_linear(self.k, context)).view(shape)
        v = _linear(self.v, context).view(shape)
        return _linear(self.o, _attention(q, k, v, context_lens,
                                         self.attention_backend).flatten(2))


class _AttentionBlock(WanAttentionBlock):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, backend):
        # Construct the native parameter layout once, using local execution
        # adapters. No global patches and no second allocation of Wan weights.
        nn.Module.__init__(self)
        self.norm1 = _LayerNorm(dim, eps)
        self.self_attn = _SelfAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.self_attn.attention_backend = backend
        self.norm3 = _LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = _CrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.cross_attn.attention_backend = backend
        self.norm2 = _LayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, rope, context, context_lens):
        shift, scale, gate, ffn_shift, ffn_scale, ffn_gate = (
            self.modulation.float() + e.float()
        ).chunk(6, dim=1)
        y = self.self_attn(self.norm1(x).float() * (1 + scale) + shift, seq_lens, rope)
        x = x.float() + y.float() * gate
        x = x + self.cross_attn(self.norm3(x), context, context_lens).float()
        y = _linear(self.ffn[0], self.norm2(x).float() * (1 + ffn_scale) + ffn_shift)
        y = _linear(self.ffn[2], self.ffn[1](y))
        return x + y.float() * ffn_gate


class _Head(Head):
    def __init__(self, dim, out_dim, patch_size, eps):
        super().__init__(dim, out_dim, patch_size, eps)
        self.norm = _LayerNorm(dim, eps)

    def forward(self, x, e):
        shift, scale = (self.modulation.float() + e.float().unsqueeze(1)).chunk(2, dim=1)
        return _linear(self.head, self.norm(x).float() * (1 + scale) + shift)


class _FP32Linear(nn.Linear):
    """Keep Wan time arithmetic in FP32 while exposing normal module hooks.

    Parameter storage and checkpoint keys are unchanged. Calling the module
    lets a distributed optimizer gather its weights before they are read.
    """

    def forward(self, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs.float(), self.weight.float(),
                            None if self.bias is None else self.bias.float())


class Wan21Generator(WanModel):
    """A T2V-layout Wan network with a trainable backbone-condition adapter.

    ``condition_dim`` is the backbone hidden width. ``text_dim`` remains the
    original Wan condition width so all pretrained DiT tensors can be retained.
    ``text_len`` is accepted for source-config compatibility, but never limits or
    pads UMM conditions. Only global attention and the T2V parameter layout are
    supported; native I2V's additional latent/CLIP inputs require another API.
    """

    ignore_for_config = []  # Unlike upstream, persist every architecture field.
    _no_split_modules = ["_AttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self, condition_dim: int, model_type="t2v", patch_size=(1, 2, 2),
        text_len=512, in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256,
        text_dim=4096, out_dim=16, num_heads=16, num_layers=32,
        window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6,
        attention_backend="auto", gradient_checkpointing=False,
    ):
        ModelMixin.__init__(self)
        if model_type != "t2v":
            raise ValueError("Wan21Generator supports only the Wan T2V parameter layout")
        if tuple(window_size) != (-1, -1):
            raise ValueError("Wan21Generator currently supports global attention only")
        if attention_backend not in {"auto", "sdpa", "flash", "npu"}:
            raise ValueError("attention_backend must be auto, sdpa, flash or npu")
        sizes = (condition_dim, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim, num_heads, num_layers)
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in sizes):
            raise ValueError("All Wan dimensions and layer counts must be positive integers")
        if dim % num_heads or (dim // num_heads) % 2 or freq_dim % 2:
            raise ValueError("Wan requires divisible heads, even head_dim and even freq_dim")
        if len(patch_size) != 3 or any(not isinstance(p, int) or isinstance(p, bool) or p <= 0 for p in patch_size):
            raise ValueError("patch_size must contain three positive integers")
        if in_dim != out_dim:
            raise ValueError("Flow prediction requires matching input and output latent channels")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.condition_dim = condition_dim
        self.model_type, self.patch_size = model_type, tuple(patch_size)
        self.text_len, self.in_dim, self.dim = text_len, in_dim, dim
        self.ffn_dim, self.freq_dim, self.text_dim = ffn_dim, freq_dim, text_dim
        self.out_dim, self.num_heads, self.num_layers = out_dim, num_heads, num_layers
        self.window_size, self.qk_norm, self.cross_attn_norm, self.eps = tuple(window_size), qk_norm, cross_attn_norm, eps
        self.attention_backend = attention_backend
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        self._rope_cache = {}

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(_FP32Linear(freq_dim, dim), nn.SiLU(), _FP32Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), _FP32Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            _AttentionBlock(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, attention_backend)
            for _ in range(num_layers)
        ])
        self.head = _Head(dim, out_dim, self.patch_size, eps)
        self.init_weights()
        self.condition_proj = nn.Linear(condition_dim, text_dim)
        initialize_normal(self.condition_proj.weight, 0.02)
        nn.init.zeros_(self.condition_proj.bias)

    def _apply(self, fn, recurse=True):
        self._rope_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def _rope(self, grids, padded_length, device):
        """Wan axis frequencies, generated on CPU in FP64 then stored as FP32.

        Real tables keep NPU execution free of complex/FP64 tensors. They are not
        state-dict buffers and cannot be silently rounded by model.bfloat16().
        """
        length = max(max(grid) for grid in grids)
        key = (device.type, device.index)
        cached = self._rope_cache.get(key)
        if cached is None or cached[0] < length:
            d = self.dim // self.num_heads
            axis_dims = (d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6))
            tables = []
            for width in axis_dims:
                coordinates = torch.arange(length, device="cpu", dtype=torch.float64)
                frequencies = 1.0 / torch.pow(10000, torch.arange(0, width, 2, device="cpu", dtype=torch.float64) / width)
                angles = torch.outer(coordinates, frequencies)
                tables.append((angles.cos().to(device=device, dtype=torch.float32),
                               angles.sin().to(device=device, dtype=torch.float32)))
            cached = (length, tables)
            self._rope_cache[key] = cached
        cos_rows, sin_rows = [], []
        for t, h, w in grids:
            for part, rows in ((0, cos_rows), (1, sin_rows)):
                axes = [table[part] for table in cached[1]]
                values = torch.cat((
                    axes[0][:t, None, None].expand(t, h, w, -1),
                    axes[1][None, :h, None].expand(t, h, w, -1),
                    axes[2][None, None, :w].expand(t, h, w, -1),
                ), -1).reshape(t * h * w, -1)
                rows.append(F.pad(values, (0, 0, 0, padded_length - t * h * w), value=1.0 - part))
        return torch.stack(cos_rows)[:, :, None], torch.stack(sin_rows)[:, :, None]

    def _validate_inputs(self, latents, times, condition, mask):
        if not isinstance(latents, (list, tuple)) or not latents:
            raise ValueError("noisy_latents must be a nonempty list of [C,T,H,W] tensors")
        count, device = len(latents), self.patch_embedding.weight.device
        for latent in latents:
            if (not isinstance(latent, torch.Tensor) or latent.ndim != 4
                    or latent.shape[0] != self.in_dim or not latent.is_floating_point()):
                raise ValueError("Each noisy latent must be floating [in_dim,T,H,W]")
            if any(size <= 0 or size % patch for size, patch in zip(latent.shape[1:], self.patch_size)):
                raise ValueError("Latent T/H/W must be positive and divisible by patch_size")
            if latent.device != device:
                raise ValueError("Latents and generator must be on the same device")
        if not isinstance(times, torch.Tensor) or times.shape != (count,) or not bool(torch.isfinite(times).all()):
            raise ValueError("timesteps must be finite with one native Wan timestep per video")
        if (not isinstance(condition, torch.Tensor) or condition.ndim != 3
                or condition.shape[0] != count or condition.shape[1] < 1
                or condition.shape[2] != self.condition_dim or not condition.is_floating_point()):
            raise ValueError("condition must be floating [videos,L,condition_dim] with L > 0")
        if condition.device != device:
            raise ValueError("Condition and generator must be on the same device")
        if mask is None:
            mask = torch.ones(condition.shape[:2], device=device, dtype=torch.bool)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != condition.shape[:2]:
            raise ValueError("condition_mask must be boolean [videos,L], True for valid tokens")
        mask = mask.to(device)
        if not bool(mask.any(-1).all()):
            raise ValueError("Every video needs at least one valid condition token, including CFG branches")
        return mask

    def _prepare_condition(self, condition, mask):
        # Compact BEFORE projection: masked holes and even NaNs in unused slots
        # cannot affect the forward pass or the projection's weight gradients.
        rows = [row[valid] for row, valid in zip(condition, mask)]
        lengths = torch.tensor([len(row) for row in rows], device=condition.device, dtype=torch.long)
        hidden = _linear(self.condition_proj, torch.cat(rows))
        hidden = _linear(self.text_embedding[0], hidden)
        hidden = _linear(self.text_embedding[2], self.text_embedding[1](hidden))
        return pad_sequence(hidden.split([len(row) for row in rows]), batch_first=True), lengths

    def _embed_time(self, timesteps, device):
        # Match the native Sequential calls; _FP32Linear preserves the previous
        # arithmetic after model.bfloat16() and under an outer autocast.
        with torch.autocast(device_type=device.type, enabled=False):
            half = self.freq_dim // 2
            frequencies = torch.pow(10000.0, -torch.arange(half, device=device, dtype=torch.float32) / half)
            angles = timesteps.to(device=device, dtype=torch.float32)[:, None] * frequencies[None]
            hidden = torch.cat((angles.cos(), angles.sin()), -1)
            hidden = self.time_embedding(hidden)
            modulation = self.time_projection(hidden)
        return hidden, modulation.unflatten(1, (6, self.dim))

    def forward(self, noisy_latents, timesteps, condition, *, condition_mask=None):
        mask = self._validate_inputs(noisy_latents, timesteps, condition, condition_mask)
        embedded = [self.patch_embedding(latent[None].to(self.patch_embedding.weight.dtype))
                    for latent in noisy_latents]
        grids = [tuple(row.shape[2:]) for row in embedded]
        rows = [row.flatten(2).transpose(1, 2)[0] for row in embedded]
        hidden = pad_sequence(rows, batch_first=True)
        seq_lens = torch.tensor([len(row) for row in rows], device=hidden.device, dtype=torch.long)
        rope = self._rope(grids, hidden.shape[1], hidden.device)
        context, context_lens = self._prepare_condition(condition, mask)
        time_embedding, modulation = self._embed_time(timesteps, hidden.device)
        for block in self.blocks:
            args = (hidden, modulation, seq_lens, rope, context, context_lens)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = self._gradient_checkpointing_func(block, *args)
            else:
                hidden = block(*args)
        prediction = self.head(hidden, time_embedding)
        return [row.float() for row in self.unpatchify(prediction, torch.tensor(grids, device="cpu"))]

    @torch.no_grad()
    def sample(self, condition, *, condition_mask=None, latent_shapes=None, initial_noise=None,
               steps=20, solver="euler", cfg=1.0, unconditional=None, unconditional_mask=None,
               time_shift=1.0, timestep_scale=1000.0, rng=None):
        """Integrate dx/dsigma = noise-clean from sigma=1 to 0.

        Unlike the legacy image sampler, Wan predicts noise minus clean. The
        model receives sigma*timestep_scale, and the integration step is negative.
        time_shift explicitly controls the monotone shifted sigma schedule.
        """
        if not isinstance(steps, int) or steps < 1 or solver not in {"euler", "heun"}:
            raise ValueError("Use positive integer steps and Euler or Heun")
        if (not all(math.isfinite(float(x)) for x in (cfg, time_shift, timestep_scale))
                or time_shift <= 0 or timestep_scale <= 0):
            raise ValueError("Sampling scales must be finite; time scales must be positive")
        if initial_noise is None:
            if latent_shapes is None or len(latent_shapes) != condition.shape[0]:
                raise ValueError("Supply one Wan latent shape per target video")
            latents = [torch.randn(tuple(shape), device=condition.device, dtype=torch.float32, generator=rng)
                       for shape in latent_shapes]
        else:
            if not isinstance(initial_noise, (list, tuple)) or len(initial_noise) != condition.shape[0]:
                raise ValueError("initial_noise must contain one tensor per target video")
            if latent_shapes is not None and [tuple(x.shape) for x in initial_noise] != [tuple(x) for x in latent_shapes]:
                raise ValueError("initial_noise shapes do not match latent_shapes")
            latents = [x.to(device=condition.device, dtype=torch.float32).clone() for x in initial_noise]
        times = torch.linspace(1., 0., steps + 1, device=condition.device, dtype=torch.float32)
        times = time_shift * times / (1 + (time_shift - 1) * times)
        mask = self._validate_inputs(latents, times[0].expand(len(latents)) * timestep_scale, condition, condition_mask)
        if cfg != 1.0:
            if unconditional is None or unconditional.shape != condition.shape:
                raise ValueError("CFG needs aligned conditional and unconditional query sequences")
            unconditional_mask = self._validate_inputs(
                latents, times[0].expand(len(latents)), unconditional, unconditional_mask)

        def velocity(values, sigma):
            timestep = (sigma * timestep_scale).expand(len(values))
            if cfg == 1.0:
                return self(values, timestep, condition, condition_mask=mask)
            predictions = self(
                values + values, torch.cat((timestep, timestep)), torch.cat((condition, unconditional)),
                condition_mask=torch.cat((mask, unconditional_mask)),
            )
            return [u + cfg * (c - u) for c, u in zip(predictions[:len(values)], predictions[len(values):])]

        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            for i in range(steps):
                dt = times[i + 1] - times[i]
                first = velocity(latents, times[i])
                predictor = [x + dt * v.float() for x, v in zip(latents, first)]
                if solver == "heun":
                    second = velocity(predictor, times[i + 1])
                    latents = [x + dt * 0.5 * (v.float() + w.float())
                               for x, v, w in zip(latents, first, second)]
                else:
                    latents = predictor
            return latents
        finally:
            for module, mode in modes:
                module.training = mode

    @classmethod
    def from_wan_pretrained(cls, pretrained_model_name_or_path, *, condition_dim,
                            attention_backend="auto", gradient_checkpointing=False,
                            local_files_only=True, **kwargs):
        """Strictly import an original Wan T2V checkpoint, including shards.

        Extra kwargs go to WanModel.from_pretrained (e.g. torch_dtype, subfolder,
        or upstream architecture overrides omitted by its ignore_for_config).
        Device-map/offload loading is excluded: load first, then place/shard the
        complete generator. No second full allocation of Wan weights is made.
        """
        if kwargs.get("device_map") is not None or kwargs.get("offload_folder") is not None:
            raise ValueError("Load Wan on one device before placing/sharding the adapted generator")
        if kwargs.get("ignore_mismatched_sizes", False) or "output_loading_info" in kwargs:
            raise ValueError("from_wan_pretrained always validates all original Wan tensors")
        source, info = _WanCheckpoint.from_pretrained(
            pretrained_model_name_or_path, local_files_only=local_files_only,
            output_loading_info=True, **kwargs,
        )
        errors = {name: info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                  if info.get(name)}
        if errors:
            raise ValueError(f"Incomplete or incompatible Wan pretrained weights: {errors}")
        names = ("model_type", "patch_size", "text_len", "in_dim", "dim", "ffn_dim", "freq_dim",
                 "text_dim", "out_dim", "num_heads", "num_layers", "window_size", "qk_norm", "cross_attn_norm", "eps")
        architecture = {name: getattr(source, name) for name in names}
        with torch.device("meta"):
            result = cls(condition_dim=condition_dim, attention_backend=attention_backend,
                         gradient_checkpointing=gradient_checkpointing, **architecture)
        missing, unexpected = result.load_state_dict(source.state_dict(), strict=False, assign=True)
        if set(missing) != {"condition_proj.weight", "condition_proj.bias"} or unexpected:
            raise ValueError(f"Wan parameter layout mismatch: missing={missing}, unexpected={unexpected}")
        weight = source.patch_embedding.weight
        result.condition_proj = nn.Linear(condition_dim, result.text_dim, device=weight.device, dtype=weight.dtype)
        initialize_normal(result.condition_proj.weight, 0.02)
        nn.init.zeros_(result.condition_proj.bias)
        return result.eval()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a complete adapted checkpoint; never silently reinitialize it."""
        return_info = kwargs.pop("output_loading_info", False)
        if kwargs.get("ignore_mismatched_sizes", False):
            raise ValueError("Adapted generator checkpoints must match all parameter shapes")
        model, info = super().from_pretrained(pretrained_model_name_or_path, output_loading_info=True, **kwargs)
        errors = {name: info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                  if info.get(name)}
        if errors:
            raise ValueError(f"Incomplete adapted generator checkpoint: {errors}")
        return (model, info) if return_info else model
# CQ END
