"""ABI wiring for upstream FLA 0.4; no kernels or CQ planning live here."""


def chunk_gated_delta(q, k, v, g, beta, *, cu_seqlens):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # FLA 0.4 takes [B,T,H,D] by default. The CPU
    # plan already supplies packed varlen write/read events in precisely this
    # order. g remains log-decay FP32; beta is the precomputed sigmoid.
    return chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta.to(q.dtype),
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
    )
