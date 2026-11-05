"""Exact tiled attention with one output buffer and FP32 K/V accumulation."""

import torch
from ops import functional as K


def attend(q, k, v, mask, valid, plan, fused):
    if plan is None:
        return K.attention(q, k, v, mask, valid, fused)
    return tiled_attention(q, k, v, mask, valid, plan, fused)


def tiled_attention(q, k, v, mask, valid_rows, plan, fused=True, scale=None):
    if not plan.matches(mask):
        # In-place edits/replaced masks invalidate data-side metadata.
        return K.attention(q, k, v, mask, valid_rows, fused, scale)
    if fused and q.device.type == "npu" and K._ATTENTION_LAYOUT == "SBH" and plan.tile_masks:
        result = _TiledFA.apply(q, k, v, plan, scale)
        if valid_rows is not None:
            result = result * valid_rows.T.unsqueeze(-1).to(result.dtype)
        return result
    pieces = []
    for index, (start, end, key_end) in enumerate(plan.tiles):
        key_start = plan.key_starts[index] if plan.key_starts else 0
        tile_mask = (
            plan.tile_masks[index]
            if plan.tile_masks
            else mask[:, :, start:end, key_start:key_end].contiguous()
        )
        valid = valid_rows[:, start:end] if valid_rows is not None else None
        pieces.append(
            K.attention(
                q[start:end], k[key_start:key_end], v[key_start:key_end], tile_mask, valid, fused, scale
            )
        )
    return torch.cat(pieces, dim=0)


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
