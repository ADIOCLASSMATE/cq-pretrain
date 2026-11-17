"""ABI wiring for upstream FLA 0.6; no kernels or CQ planning live here."""
from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _host_boundaries(bounds):
    # Reuse the existing CPU plan, avoiding FLA's CUDA-to-host metadata copy
    # at every layer. This tensor is metadata only and never mutated by FLA.
    return torch.tensor(bounds, dtype=torch.long, device='cpu')


def chunk_gated_delta(q, k, v, g, beta, *, cu_seqlens, bounds):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # FLA 0.6 takes [B,T,H,D] exclusively: head_first was removed. The CPU
    # plan already supplies packed varlen write/read events in precisely this
    # order. g remains log-decay FP32; beta is the precomputed sigmoid.
    return chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta.to(q.dtype),
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=_host_boundaries(tuple(bounds)),
    )
