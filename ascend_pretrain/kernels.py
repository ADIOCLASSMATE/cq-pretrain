"""NPU training kernels. CPU implementations are explicit numerical references."""
import os
import torch
from torch.nn import functional as F

_COMPILE_MODULATION = os.environ.get('PRETRAIN_COMPILE_MODULATION', '0') == '1'
_ATTENTION_LAYOUT = os.environ.get('PRETRAIN_ATTENTION_LAYOUT', 'SBH')
if _ATTENTION_LAYOUT not in ('SBH', 'BNSD'):
    raise ValueError('PRETRAIN_ATTENTION_LAYOUT must be SBH or BNSD')


def modulated_norm(hidden, scale, shift, fused=True):
    if _COMPILE_MODULATION and fused and hidden.device.type == 'npu':
        from .compiled_kernels import compiled
        return compiled('modulated_norm')(hidden, scale, shift)
    return F.layer_norm(hidden, (hidden.shape[-1],), eps=1e-6) * (1 + scale) + shift


def fused_wgrad(total_input, grad_output, main_grad):
    """BF16 GEMM accumulated directly into FP32 through TorchNPU's ATB binding."""
    import torch_npu
    if total_input.numel() and grad_output.numel():
        torch_npu._npu_matmul_add_fp32(grad_output.contiguous(), total_input.contiguous(), main_grad)


def npu_ops(x, fused=True):
    if fused and x.device.type == 'npu':
        import torch_npu
        return torch_npu
    return None


def rms_norm(x, weight, eps, fused=True):
    ops = npu_ops(x, fused)
    if ops:
        if torch.are_deterministic_algorithms_enabled():
            return _DeterministicRMSNorm.apply(x, weight, eps)
        return ops.npu_rms_norm(x, weight, epsilon=eps)[0]
    normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return normalized * weight


class _DeterministicRMSNorm(torch.autograd.Function):
    """Avoid CANN 9.0's deterministic BF16 small-head dgamma error.

    The ordinary training path uses native BF16 autograd. Deterministic
    validation retains fused dx and explicitly reduces dgamma in FP32.
    Switching deterministic mode after warming a kernel is not a valid probe:
    this runtime caches its initial kernel selection.
    """
    @staticmethod
    def forward(ctx, x, weight, eps):
        import torch_npu
        y, rstd = torch_npu.npu_rms_norm(x, weight, epsilon=eps)
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, grad):
        import torch_npu
        x, weight, rstd = ctx.saved_tensors
        dx, _ = torch_npu.npu_rms_norm_backward(grad.contiguous(), x, weight, rstd)
        dw = (grad.float() * x.float() * rstd).reshape(-1, x.shape[-1]).sum(0)
        return dx.to(x.dtype), dw.to(weight.dtype), None


class _AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        import torch_npu
        norm, rstd, summed = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon=eps)
        ctx.save_for_backward(summed, weight, rstd)
        return norm, summed

    @staticmethod
    def backward(ctx, grad_norm, grad_sum):
        import torch_npu
        summed, weight, rstd = ctx.saved_tensors
        dx, dw = torch_npu.npu_rms_norm_backward(grad_norm.contiguous(), summed, weight, rstd)
        dx = dx + grad_sum
        return dx, dx, dw, None


def add_rms_norm(x, residual, weight, eps, fused=True):
    ops = npu_ops(x, fused)
    if ops:
        # torch_npu 2.6 exposes the forward kernel without an autograd formula.
        # Supply its exact residual chain rule using the fused RMSNorm backward.
        return _AddRMSNorm.apply(x, residual, weight, eps)
    summed = x + residual
    return rms_norm(summed, weight, eps, False), summed


def swiglu(x, fused=True):
    ops = npu_ops(x, fused)
    if ops:
        return ops.npu_swiglu(x, dim=-1)
    gate, value = x.chunk(2, -1)
    return F.silu(gate) * value


def rotary(x, cos, sin, fused=True):
    ops = npu_ops(x, fused)
    if ops:
        return ops.npu_rotary_mul(x, cos, sin)
    first, second = x.chunk(2, -1)
    return x * cos + torch.cat((-second, first), -1) * sin


def attention(q, k, v, mask=None, valid_rows=None, fused=True):
    """Q/K/V: [sequence,batch,heads,dim]; bool mask True means forbidden."""
    ops = npu_ops(q, fused)
    if ops:
        if _ATTENTION_LAYOUT == 'BNSD':
            query, key, value = (t.permute(1, 2, 0, 3).contiguous() for t in (q, k, v))
            result = ops.npu_fusion_attention(query, key, value, head_num=q.shape[2],
                input_layout='BNSD', atten_mask=mask, scale=q.shape[-1] ** -.5,
                keep_prob=1., sparse_mode=0)[0].permute(2, 0, 1, 3).flatten(-2).contiguous()
        else:
            result = ops.npu_fusion_attention(
                q.flatten(-2).contiguous(), k.flatten(-2).contiguous(), v.flatten(-2).contiguous(),
                head_num=q.shape[2], input_layout='SBH', atten_mask=mask,
                scale=q.shape[-1] ** -.5, keep_prob=1., sparse_mode=0)[0]
    else:
        query, key, value = (t.permute(1, 2, 0, 3) for t in (q, k, v))
        # Explicit repeat only in the numerical reference, never in the NPU path.
        repeat = q.shape[2] // k.shape[2]
        if repeat != 1:
            key, value = key.repeat_interleave(repeat, 1), value.repeat_interleave(repeat, 1)
        result = F.scaled_dot_product_attention(query, key, value,
            attn_mask=None if mask is None else ~mask, dropout_p=0.)
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
