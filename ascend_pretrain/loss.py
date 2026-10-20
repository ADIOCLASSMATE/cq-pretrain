"""Chunked, recomputed linear cross entropy with vocabulary tensor parallelism.

No [batch, sequence, vocabulary] tensor is retained in the training graph.
The custom backward returns ordinary parameter gradients, preserving MCore's
gradient hooks, tied embeddings, accumulation and distributed optimizer.
"""
import os
import torch
import torch.distributed as dist
from . import parallel
from .kernels import fused_wgrad


def logits_and_probabilities(hidden, weight, targets, group, rank, size):
    logits = torch.nn.functional.linear(hidden, weight).float()
    maxima = logits.max(-1).values
    if size > 1:
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
    logits.sub_(maxima[:, None])
    local_targets = targets - rank * weight.shape[0]
    owned = (local_targets >= 0) & (local_targets < weight.shape[0])
    safe_targets = local_targets.clamp(0, weight.shape[0] - 1)
    selected = logits.gather(1, safe_targets[:, None]).squeeze(1) * owned
    if size > 1:
        dist.all_reduce(selected, group=group)
    logits.exp_()
    denominator = logits.sum(-1)
    if size > 1:
        dist.all_reduce(denominator, group=group)
    probabilities = logits.div_(denominator[:, None])
    loss = denominator.log() - selected
    return loss, probabilities, safe_targets, owned


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        ps = parallel.state()
        size = parallel.tp_size()
        group = ps.get_tensor_model_parallel_group() if ps else None
        rank = ps.get_tensor_model_parallel_rank() if ps else 0
        fused_ce = (hidden.device.type == 'npu' and size == 1
                    and os.environ.get('PRETRAIN_FUSED_CE', '1') == '1')
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk):
            if fused_ce:
                import torch_npu
                logits = torch.nn.functional.linear(hidden[start:start+chunk], weight).float()
                loss = torch_npu.npu_cross_entropy_loss(
                    logits, targets[start:start+chunk], reduction='sum')[0]
            else:
                loss, _, _, _ = logits_and_probabilities(hidden[start:start+chunk], weight,
                    targets[start:start+chunk], group, rank, size)
            total += loss.sum()
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk, ctx.group, ctx.rank, ctx.size = chunk, group, rank, size
        ctx.fused_ce = fused_ce
        ctx.main_grad = (getattr(weight,'main_grad',None)
                         if getattr(weight,'_ce_direct_accumulation',False)
                         and hasattr(weight,'grad_added_to_main_grad') else None)
        if ctx.main_grad is not None and ctx.main_grad.dtype != torch.float32:
            raise ValueError('Direct CE accumulation requires MCore FP32 gradients')
        return total / max(1, hidden.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, targets = ctx.saved_tensors
        dh = torch.empty_like(hidden)
        # FP32 chunk accumulation avoids rounding a BF16 gradient once per chunk.
        dw = ctx.main_grad if ctx.main_grad is not None else torch.zeros_like(weight, dtype=torch.float32)
        factor = grad_output.float() / max(1, hidden.shape[0])
        for start in range(0, hidden.shape[0], ctx.chunk):
            h = hidden[start:start+ctx.chunk]
            if ctx.fused_ce:
                import torch_npu
                logits = torch.nn.functional.linear(h, weight).float()
                chunk_targets = targets[start:start+ctx.chunk]
                _, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(logits, chunk_targets, reduction='sum')
                gradient = torch_npu.npu_cross_entropy_loss_backward(
                    factor.reshape(1), log_prob, chunk_targets, reduction='sum').to(hidden.dtype)
                del logits, log_prob
            else:
                _, prob, index, owned = logits_and_probabilities(h, weight,
                    targets[start:start+ctx.chunk], ctx.group, ctx.rank, ctx.size)
                prob.scatter_add_(1, index[:, None], -owned[:, None].to(prob.dtype))
                gradient = (prob * factor).to(hidden.dtype)
            dh[start:start+ctx.chunk] = gradient @ weight
            # ATB accumulates BF16 products directly into FP32, with no
            # full FP32 cast or separately materialized dW for each chunk.
            if hidden.device.type == 'npu':
                fused_wgrad(h, gradient, dw)
            else:
                dw.addmm_(gradient.float().T, h.float())
        if ctx.size > 1:
            dist.all_reduce(dh, group=ctx.group)
        if ctx.main_grad is not None:
            # Match ColumnParallelLinear's hook protocol. The dummy gradient
            # runs AccumulateGrad/DP readiness on the autograd thread; MCore
            # does not add it because main_grad already contains the result.
            weight.grad_added_to_main_grad=True
            factory=torch.zeros_like if getattr(weight,'zero_out_wgrad',False) else torch.empty_like
            gradient=factory(weight)
        else:gradient=dw.to(weight.dtype)
        return dh, gradient, None, None


def linear_cross_entropy(hidden, weight, targets, chunk):
    return _LinearCE.apply(hidden, weight, targets, chunk)
