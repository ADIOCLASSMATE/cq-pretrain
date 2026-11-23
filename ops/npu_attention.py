"""Ascend FlashAttention (npu_fusion_attention) for the CQ mask layout.

Two entry points, both fused:

``attention`` runs one kernel over the whole packed sequence with this
project's forbid-mask. ``attend`` runs the CPU planner's exact tiling, which
keeps a dense [1,1,2S,S] mask from ever being materialized on device: each tile
carries only the mask its queries need. The tiled form writes into one output
buffer and accumulates K/V gradients in FP32 across tiles, so its values and
gradients match the single-kernel form.

Layout: SBH is the measured default on A2; BNSD is kept for comparison because
the two differ in transpose cost rather than in result.
"""

import os

import torch

import ops.audit as audit

# Lets a run fall back to the per-segment loop, which is also the oracle, so
# the two backends can be A/B-ed through the same switch.
_VARLEN = os.environ.get("PRETRAIN_NPU_VARLEN", "1") == "1"
LAYOUT = os.environ.get("PRETRAIN_ATTENTION_LAYOUT", "SBH")
if LAYOUT not in ("SBH", "BNSD"):
    raise ValueError("PRETRAIN_ATTENTION_LAYOUT must be SBH or BNSD")


def attend(q, k, v, mask, valid_rows, plan, scale=None):
    """Tile when the planner supplied per-tile masks; otherwise one kernel."""
    if plan is not None and plan.matches(mask) and plan.tile_masks:
        audit.record("attention", "npu_fusion_attention_tiled")
        result = _TiledFA.apply(q, k, v, plan, scale)
        if valid_rows is not None:
            result = result * valid_rows.transpose(0, 1).unsqueeze(-1).to(result.dtype)
        return result
    return attention(q, k, v, mask, valid_rows, scale)


def varlen_available(q):
    """True when the TND kernel should run instead of the per-segment loop."""
    return _VARLEN and q.device.type == "npu" and not torch.are_deterministic_algorithms_enabled()


def varlen(q, k, v, cu_q, cu_k, scale=None):
    """Packed segments in one TND kernel. Q is [T,N,D]; cu_* are cumulative ends.

    TND is this kernel's own packed layout: it takes the segment boundaries the
    CPU planner already produced and runs every segment in one launch, instead
    of one launch per segment. Same per-row math as the reference loop.
    """
    import torch_npu

    audit.record("attention", "npu_fusion_attention_varlen")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return torch_npu.npu_fusion_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        head_num=q.shape[1],
        input_layout="TND",
        actual_seq_qlen=tuple(int(end) for end in cu_q),
        actual_seq_kvlen=tuple(int(end) for end in cu_k),
        scale=scale,
        keep_prob=1.0,
        sparse_mode=0,
    )[0]


def attention(q, k, v, mask=None, valid_rows=None, scale=None):
    """Q/K/V: [sequence,batch,heads,dim]; bool mask True means forbidden."""
    import torch_npu

    audit.record("attention", f"npu_fusion_attention_{LAYOUT}")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if LAYOUT == "BNSD":
        query, key, value = (t.permute(1, 2, 0, 3).contiguous() for t in (q, k, v))
        result = (
            torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                head_num=q.shape[2],
                input_layout="BNSD",
                atten_mask=mask,
                scale=scale,
                keep_prob=1.0,
                sparse_mode=0,
            )[0]
            .permute(2, 0, 1, 3)
            .flatten(-2)
            .contiguous()
        )
    else:
        result = torch_npu.npu_fusion_attention(
            q.flatten(-2).contiguous(),
            k.flatten(-2).contiguous(),
            v.flatten(-2).contiguous(),
            head_num=q.shape[2],
            input_layout="SBH",
            atten_mask=mask,
            scale=scale,
            keep_prob=1.0,
            sparse_mode=0,
        )[0]
    if valid_rows is not None:
        result = result * valid_rows.transpose(0, 1).unsqueeze(-1).to(result.dtype)
    return result


class _TiledFA(torch.autograd.Function):
    """One output buffer and FP32 K/V gradient accumulation across exact tiles."""

    @staticmethod
    def forward(ctx, q, k, v, plan, scale):
        import torch_npu

        ctx.shapes = (q.shape, k.shape, v.shape)
        ctx.heads = q.shape[2]
        ctx.scale = q.shape[-1] ** -0.5 if scale is None else scale
        query, key, value = [t.flatten(-2).contiguous() for t in (q, k, v)]
        output = torch.empty_like(query)
        statistics = []
        ctx.random = []
        ctx.tiles = plan.tiles
        ctx.key_starts = plan.key_starts or (0,) * len(plan.tiles)
        for (start, end, key_end), key_start, mask in zip(plan.tiles, ctx.key_starts, plan.tile_masks):
            result = torch_npu.npu_fusion_attention(
                query[start:end],
                key[key_start:key_end],
                value[key_start:key_end],
                head_num=ctx.heads,
                input_layout="SBH",
                atten_mask=mask,
                scale=ctx.scale,
                keep_prob=1.0,
                sparse_mode=0,
            )
            output[start:end].copy_(result[0])
            statistics.extend((result[1], result[2]))
            ctx.random.append(result[4:7])
        ctx.save_for_backward(query, key, value, output, *statistics, *plan.tile_masks)
        return output

    @staticmethod
    def backward(ctx, grad):
        import torch_npu

        query, key, value, output, *saved = ctx.saved_tensors
        count = len(ctx.tiles)
        stats = saved[: 2 * count]
        masks = saved[2 * count :]
        dq = torch.empty_like(query)
        dk = torch.zeros_like(key, dtype=torch.float32)
        dv = torch.zeros_like(value, dtype=torch.float32)
        for index, (start, end, key_end) in enumerate(ctx.tiles):
            key_start = ctx.key_starts[index]
            seed, offset, numels = ctx.random[index]
            g = torch_npu.npu_fusion_attention_grad(
                query[start:end],
                key[key_start:key_end],
                value[key_start:key_end],
                grad[start:end].contiguous(),
                head_num=ctx.heads,
                input_layout="SBH",
                atten_mask=masks[index],
                softmax_max=stats[index * 2],
                softmax_sum=stats[index * 2 + 1],
                attention_in=output[start:end],
                scale_value=ctx.scale,
                keep_prob=1.0,
                sparse_mode=0,
                seed=seed,
                offset=offset,
                numels=numels,
            )
            dq[start:end].copy_(g[0])
            dk[key_start:key_end].add_(g[1].float())
            dv[key_start:key_end].add_(g[2].float())
        return (
            dq.reshape(ctx.shapes[0]),
            dk.to(key.dtype).reshape(ctx.shapes[1]),
            dv.to(value.dtype).reshape(ctx.shapes[2]),
            None,
            None,
        )
