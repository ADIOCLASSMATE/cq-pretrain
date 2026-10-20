"""Ascend 910B GDN training backend with explicit autograd and no eager fallback.

Backend ABI: flashserve/flash-linear-attention-npu at
f623f2b02ddd7a06b115d8061cc805275761a8b6. The A2 fused forward uses exp,
head-first QKV, and external L2 normalization. Its backward is an AscendC
operator chain; the A5-only composite backward must never be dispatched here.
The API accepts FLA's [B,T,H,D] layout and CPU packed-sequence metadata.
"""
from collections import Counter
from functools import lru_cache
import os

import torch
from torch.nn import functional as F


CALLS = Counter()
_OPS = {}
MODE = os.getenv('DELTA_ASCENDC_MODE', 'fused')
POINTWISE = os.getenv('DELTA_ASCENDC_POINTWISE', '1') == '1'


def initialize():
    if os.getenv('TRITON_ALL_BLOCKS_PARALLEL', '0') != '0':
        raise RuntimeError('TRITON_ALL_BLOCKS_PARALLEL must be 0: 3.2.1 drops logical programs on this A2 path')
    from fla_npu.ops import ascendc
    from fla.modules.backends.triton_ascend.l2norm import l2norm_fwd_npu, l2norm_bwd_npu
    from fla.ops.utils.solve_tril import solve_tril
    from fla.ops.utils.backends.triton_ascend.cumsum import chunk_local_cumsum_npu
    from fla.modules.fused_norm_gate import rms_norm_gated
    names = ('chunk_gated_delta_rule_fwd', 'recompute_w_u_fwd',
             'chunk_gated_delta_rule_fwd_h', 'chunk_bwd_dv_local',
             'chunk_gated_delta_rule_bwd_dhu', 'chunk_bwd_dqkwg',
             'prepare_wy_repr_bwd_da', 'prepare_wy_repr_bwd_full',
             'chunk_local_cumsum', 'chunk_scaled_dot_kkt', 'chunk_fwd_o',
             'causal_conv1d_fn', 'causal_conv1d_bwd')
    _OPS.update({name: getattr(ascendc, name) for name in names})
    _OPS.update(l2_fwd=l2norm_fwd_npu, l2_bwd=l2norm_bwd_npu, solve=solve_tril,
                reverse_cumsum_native=chunk_local_cumsum_npu, rms_norm_gated=rms_norm_gated)
    if MODE not in ('fused', 'cached'):
        raise ValueError(f'Unknown DELTA_ASCENDC_MODE={MODE}')


def call(name, *args, **kwargs):
    CALLS[name] += 1
    return _OPS[name](*args, **kwargs)


def gated_norm(y, z, weight):
    # A2 microbenchmarks: the fused kernel wins at 131072 rows (8K x 16),
    # but loses at 2K. Keep the faster small-shape implementation.
    if POINTWISE and y.numel() // y.shape[-1] >= 131072:
        return call('rms_norm_gated', y, z, weight, None, eps=1e-6)
    from .kernels import rms_norm
    return (rms_norm(y, weight, 1e-6).float() * F.silu(z.float())).to(y.dtype)


def reverse_cumsum(dg, bounds):
    if POINTWISE and bounds is None and dg.shape[-1] >= 2048:
        return call('reverse_cumsum_native', dg, 64, reverse=True, head_first=True)
    return call('chunk_local_cumsum', dg, 64,
                cu_seqlens=None if bounds is None else list(bounds),
                chunk_indices_out=chunk_indices(bounds, 2048), reverse=True)


@lru_cache(maxsize=128)
def chunk_indices(bounds, size):
    if bounds is None:
        return None
    return [item for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
            for j in range((b - a + size - 1) // size) for item in (i, j)]


def host_bounds(cu_seqlens, cu_seqlens_cpu):
    if cu_seqlens is None:
        return None
    if cu_seqlens_cpu is None:
        raise ValueError('Supply cu_seqlens_cpu: training must not copy metadata from NPU in each layer')
    if isinstance(cu_seqlens_cpu, torch.Tensor):
        if cu_seqlens_cpu.device.type != 'cpu':
            raise ValueError('cu_seqlens_cpu must be on CPU')
        cu_seqlens_cpu = cu_seqlens_cpu.tolist()
    bounds = tuple(int(i) for i in cu_seqlens_cpu)
    if not bounds or bounds[0] != 0 or any(b < a for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Invalid packed sequence boundaries')
    return bounds


@lru_cache(maxsize=64)
def device_metadata(bounds, device):
    if bounds is None:
        return None, None
    cu = torch.tensor(bounds, device=device, dtype=torch.long)
    indices = {str(size): torch.tensor(chunk_indices(bounds, size), device=device,
                                      dtype=torch.long).view(-1, 2)
               for size in (16, 32, 64, 128, 1216)}
    return cu, indices


class _Delta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, bounds, normalize, scale):
        ctx.input_dtypes = [x.dtype for x in (q, k, v, g, beta)]
        q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
        qr = kr = None
        if normalize:
            q, qr = call('l2_fwd', q)
            k, kr = call('l2_fwd', k)
        meta = dict(cu_seqlens=None if bounds is None else list(bounds),
                    chunk_indices=chunk_indices(bounds, 64))
        bf = beta.transpose(1, 2).contiguous().float()
        w = h = vn = None
        if MODE == 'fused':
            out, _, gs, a, *_ = call(
                'chunk_gated_delta_rule_fwd', q, k, v, g.contiguous(), beta.contiguous(),
                scale=scale, chunk_size=64, disable_recompute=True, **meta)
            gs = gs.transpose(1, 2).contiguous()
        else:
            gs = call('chunk_local_cumsum', g.transpose(1, 2).contiguous().float(), 64,
                      cu_seqlens=meta['cu_seqlens'],
                      chunk_indices_out=chunk_indices(bounds, 2048))
            a = call('chunk_scaled_dot_kkt', k, gs, bf, chunk_size=64, **meta)
            # Public A2 solve_tri has a narrower tail/packed contract than GDN.
            # Use the explicit, tested Triton triangular solve; never catch and fall back.
            cu, ci = device_metadata(bounds, str(q.device))
            a = call('solve', a.transpose(1, 2).contiguous(), cu_seqlens=cu,
                     chunk_indices=None if ci is None else ci['64'], output_dtype=k.dtype)
            a = a.transpose(1, 2).contiguous()
            w, u = call('recompute_w_u_fwd', k, v, bf, a, 64, g=gs, **meta)
            h, vn, _ = call('chunk_gated_delta_rule_fwd_h', k, w, u, g=gs,
                            chunk_size=64, **meta)
            out = call('chunk_fwd_o', q, k, vn, h, scale, g=gs,
                       chunk_size=64, output_layout='BNSD', **meta).transpose(1, 2).contiguous()
        ctx.save_for_backward(q, k, v, gs, bf, a, qr, kr, w, h, vn)
        ctx.meta, ctx.bounds, ctx.scale, ctx.normalize = meta, bounds, scale, normalize
        return out

    @staticmethod
    def backward(ctx, dy):
        q, k, v, g, beta, a, qr, kr, w, h, vn = ctx.saved_tensors
        meta, scale = ctx.meta, ctx.scale
        if w is None:
            w, u = call('recompute_w_u_fwd', k, v, beta, a, 64, g=g, **meta)
            h, vn, _ = call('chunk_gated_delta_rule_fwd_h', k, w, u, g=g,
                            chunk_size=64, **meta)
        dy = dy.transpose(1, 2).contiguous()
        du = call('chunk_bwd_dv_local', q, k, dy, g, scale, 64, A=a, **meta)
        dh, _, du = call('chunk_gated_delta_rule_bwd_dhu', q, k, w, dy, du,
                         scale, 64, g=g, use_exp2=False, **meta)
        dq, dk, dw, dg = call('chunk_bwd_dqkwg', q, k, vn, g, h, dy, dh, du,
                              64, scale=scale, use_exp2=False, **meta)
        da = call('prepare_wy_repr_bwd_da', k, v, beta, a, dw, du, g,
                  chunk_size=64, **meta)
        dk2, dv, db, dg2 = call('prepare_wy_repr_bwd_full', k, v, beta, a, da, dw, du,
                               g, 64, **meta)
        dk = dk + dk2
        dg = reverse_cumsum((dg + dg2).contiguous(), ctx.bounds)
        if ctx.normalize:
            dq = call('l2_bwd', q, qr, dq.contiguous())
            dk = call('l2_bwd', k, kr, dk.contiguous())
        grads = [x.transpose(1, 2).contiguous().to(dtype)
                 for x, dtype in zip((dq, dk, dv, dg, db), ctx.input_dtypes)]
        return *grads, None, None, None


def delta(q, k, v, g, beta, *, use_qk_l2norm_in_kernel=False,
          cu_seqlens=None, cu_seqlens_cpu=None, initial_state=None,
          output_final_state=False, scale=None):
    if initial_state is not None or output_final_state:
        raise NotImplementedError('This training adapter requires zero initial state and no state output')
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('QKV must share BF16/FP16 dtype')
    if q.shape != k.shape or q.shape != v.shape or (q.shape[-1] != 128 or q.shape[-2] not in (4,8,16,32)):
        raise ValueError('GDN requires equal expanded QKV, head dimension 128 and supported local heads')
    bounds = host_bounds(cu_seqlens, cu_seqlens_cpu)
    if bounds is not None and (q.shape[0] != 1 or bounds[-1] != q.shape[1]):
        raise ValueError('Packed inputs must have batch 1 and boundaries covering every token')
    return _Delta.apply(q, k, v, g, beta, bounds, use_qk_l2norm_in_kernel,
                        q.shape[-1] ** -.5 if scale is None else float(scale)), None


class _Conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, activation, bounds):
        x = x.contiguous()
        weight = weight.transpose(0, 1).contiguous()
        packed = bounds is not None
        op_x = x.reshape(-1, x.shape[-1]) if packed else x
        meta = None if bounds is None else torch.tensor(bounds, dtype=torch.int32, device=x.device)
        y = call('causal_conv1d_fn', op_x, weight, None, activation=None,
                 query_start_loc=meta)
        ctx.save_for_backward(op_x, weight, y)
        ctx.shape, ctx.bounds, ctx.activation = x.shape, bounds, activation
        return (F.silu(y) if activation else y).reshape_as(x)

    @staticmethod
    def backward(ctx, dy):
        x, weight, y = ctx.saved_tensors
        dx, dw, _, _ = call('causal_conv1d_bwd', x, y if ctx.activation else None,
                            weight, dy.reshape_as(x).contiguous(),
                            query_start_loc=ctx.bounds, activation=int(ctx.activation),
                            input_layout='TND' if ctx.bounds is not None else 'BSND')
        return dx.reshape(ctx.shape), dw.transpose(0, 1).contiguous(), None, None


def conv(x, weight, *, activation=None, cu_seqlens=None, cu_seqlens_cpu=None):
    if activation not in (None, 'silu', 'swish'):
        raise ValueError(f'Unsupported activation {activation}')
    return _Conv.apply(x, weight, activation is not None,
                       host_bounds(cu_seqlens, cu_seqlens_cpu)), None


def provenance():
    import hashlib
    from pathlib import Path
    import fla_npu
    from fla_npu.ops import ascendc
    names = ('libfla_npu_stable.so', 'opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so')
    package = Path(fla_npu.__file__).parent
    files = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names}
    assert all(v == 'stable' for v in ascendc.BACKENDS.values()), ascendc.BACKENDS
    assert not ascendc.FALLBACKS, ascendc.FALLBACKS
    return dict(mode=MODE, pointwise=POINTWISE, l2norm='fla_native_ub_tiled',
                triton_all_blocks_parallel=os.getenv('TRITON_ALL_BLOCKS_PARALLEL', '0'),
                package=str(package), calls=dict(CALLS),
                dispatch=ascendc.BACKENDS, fallback=ascendc.FALLBACKS, sha256=files)
