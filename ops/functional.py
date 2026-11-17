"""CUDA training operators and their explicit PyTorch numerical references.

Each entry point has one fused path and one reference implementation. The
reference is not dead code: it is the oracle every GPU gate compares against
and the only path on CPU, so it stays a literal transcription of the math.
"""

import os
import torch
from torch.nn import functional as F

_COMPILE_MODULATION = os.environ.get("PRETRAIN_COMPILE_MODULATION", "0") == "1"


def modulated_norm_reference(hidden, scale, shift, eps=1e-6, upcast=False):
    normalized = F.layer_norm(hidden.float() if upcast else hidden, (hidden.shape[-1],), eps=eps).to(
        hidden.dtype
    )
    return normalized * (1 + scale) + shift


def gated_add_reference(hidden, branch, gate):
    return hidden + gate * branch


def modulated_norm(hidden, scale, shift, fused=True, eps=1e-6, upcast=False):
    if _COMPILE_MODULATION and fused and hidden.device.type == "cuda":
        from ops.cuda_compile import compiled

        return compiled("modulated_norm")(hidden, scale, shift, eps, upcast)
    return modulated_norm_reference(hidden, scale, shift, eps, upcast)


def gated_add(hidden, branch, gate, fused=True):
    if _COMPILE_MODULATION and fused and hidden.device.type == "cuda":
        from ops.cuda_compile import compiled

        return compiled("gated_add")(hidden, branch, gate)
    return gated_add_reference(hidden, branch, gate)


def fused_wgrad(total_input, grad_output, main_grad):
    """Accumulate the BF16 GEMM product straight into MCore's FP32 buffer."""
    import fused_weight_gradient_mlp_cuda

    if total_input.numel() and grad_output.numel():
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            total_input.contiguous(), grad_output.contiguous(), main_grad
        )


def rms_norm(x, weight, eps, fused=True):
    if not x.numel():
        return x * weight.to(x.dtype)
    from ops.cuda_fused import enabled as _cuda_fused, rms_norm as _cuda_rms_norm

    if _cuda_fused(x, fused):
        return _cuda_rms_norm(x, weight, eps)
    normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return normalized * weight


def rotary(x, cos, sin, fused=True):
    # `fused` is kept so call sites can still request reference mode explicitly;
    # this rotate-half form is both the reference and the only implementation.
    first, second = x.chunk(2, -1)
    return x * cos + torch.cat((-second, first), -1) * sin


_ALLOW_CACHE = {}


def _allow_mask(mask):
    """Cache ~mask per forbid-mask tensor; recompute if it is replaced or edited.

    Identity plus _version is the same validity test AttentionPlan.matches uses,
    so an in-place edit or a different mask object cannot return a stale inverse.
    """
    key = id(mask)
    entry = _ALLOW_CACHE.get(key)
    if entry is not None:
        owner, version, allow = entry
        if owner is mask and mask._version == version:
            return allow
    allow = ~mask
    if len(_ALLOW_CACHE) > 16:
        _ALLOW_CACHE.clear()
    _ALLOW_CACHE[key] = (mask, mask._version, allow)
    return allow


def attention(q, k, v, mask=None, valid_rows=None, fused=True, scale=None):
    """Q/K/V: [sequence,batch,heads,dim]; bool mask True means forbidden."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    query, key, value = (t.permute(1, 2, 0, 3) for t in (q, k, v))
    repeat = q.shape[2] // k.shape[2]
    if repeat != 1:
        if query.device.type == "cuda":
            # GQA: broadcast the shared KV heads as a view. repeat_interleave
            # would materialize 4x the K and V activations here (8 query heads
            # over 2 KV heads), and SDPA accepts the expanded stride.
            b, h, t, d = key.shape
            key, value = (
                x[:, :, None].expand(b, h, repeat, t, d).reshape(b, h * repeat, t, d)
                for x in (key, value)
            )
        else:
            # Explicit repeat in the CPU numerical reference only.
            key, value = key.repeat_interleave(repeat, 1), value.repeat_interleave(repeat, 1)
    # SDPA takes True = attend, the inverse of this project's forbid-mask.
    # Inverting allocates another full [1,1,2S,S] bool (128 MB at S=8192) on
    # every call, so reuse the per-batch inverse the CPU planner cached.
    allow = None if mask is None else _allow_mask(mask)
    result = F.scaled_dot_product_attention(
        query, key, value, attn_mask=allow, dropout_p=0.0, scale=scale
    )
    result = result.permute(2, 0, 1, 3).flatten(-2).contiguous()
    if valid_rows is not None:
        result = result * valid_rows.transpose(0, 1).unsqueeze(-1).to(result.dtype)
    return result


def rope_tables(row_col, head_dim, theta, dtype):
    # Same rotate-half convention as the original Qwen-derived backbone.
    frequencies = theta ** (-torch.arange(0, head_dim, 2, device=row_col.device).float() / head_dim)
    axis = torch.arange(head_dim // 2, device=row_col.device) % 2
    positions = torch.where(axis[None, None] == 0, row_col[0, :, :, None], row_col[1, :, :, None])
    angles = positions.float() * frequencies
    angles = torch.cat((angles, angles), -1).unsqueeze(2)
    return angles.cos().to(dtype), angles.sin().to(dtype)
