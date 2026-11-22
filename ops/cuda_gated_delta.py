"""ABI wiring for upstream FLA 0.4; no kernels or CQ planning live here."""

import os

# Upstream's l2norm kernel takes its row count as a compile-time constant and
# re-autotunes per ceil(rows / 2048). Real packs change their event count on
# every pack, so normalizing inside chunk_gated_delta_rule rebuilt 50 kernel
# variants per new count: about 20 s per pack and rank, and a shared cache
# directory growing by thousands of entries per step. The same upstream l2norm
# is applied here to q and k padded to a fixed row bucket. Zero rows normalize
# to zero and are sliced away, so values and gradients are those of the
# in-kernel path while at most ceil(max_rows / bucket) variants ever exist.
_L2_ROW_BUCKET = int(os.environ.get("PRETRAIN_GDN_L2_ROW_BUCKET", "65536"))


def _unit(x):
    from fla.modules.l2norm import l2norm
    from torch.nn import functional as F

    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    rows = flat.shape[0]
    if _L2_ROW_BUCKET > 0:
        extent = -(-rows // _L2_ROW_BUCKET) * _L2_ROW_BUCKET
        if extent != rows:
            flat = F.pad(flat, (0, 0, 0, extent - rows))
    return l2norm(flat)[:rows].reshape(shape)


def chunk_gated_delta(q, k, v, g, beta, *, cu_seqlens, output_final_state=False):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # FLA 0.4 takes [B,T,H,D] by default. The CPU
    # plan already supplies packed varlen write/read events in precisely this
    # order. g remains log-decay FP32; beta is the precomputed sigmoid.
    return chunk_gated_delta_rule(
        _unit(q), _unit(k), v, g=g, beta=beta.to(q.dtype),
        use_qk_l2norm_in_kernel=False, cu_seqlens=cu_seqlens, output_final_state=output_final_state,
    )
