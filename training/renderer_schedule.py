"""Bound DiT activations while backpropagating the shared CQ backbone once.

The flow objective averages over visuals, independent of chunk and pixel counts.
Every chunk retains the parent layout's loss weights and pre-sampled flow data.
Only the final chunk marks renderer gradients ready for MCore's DP reduction.
"""

from contextlib import contextmanager, nullcontext
import torch
from distributed.accel import device_module


@contextmanager
def suspend_gradient_readiness(wrapped):
    """Nest safely inside an outer GA no_sync; never reset its False to True.

    MCore 0.17 no_sync unconditionally restores True, so nesting that context
    would launch premature reductions on non-final accumulation microbatches.
    The normal post hooks still accumulate every partial gradient in main_grad.
    """
    if wrapped is None:
        yield
        return
    groups = wrapped.bucket_groups + wrapped.expert_parallel_bucket_groups
    previous = [g.is_last_microbatch for g in groups]
    for group in groups:
        group.is_last_microbatch = False
    try:
        yield
    finally:
        for group, state in zip(groups, previous):
            group.is_last_microbatch = state


@contextmanager
def shared_flow_kv(renderer, conditions, layout, split_draws):
    """Project once per visual group and backpropagate all draw gradients through KV once."""
    if not split_draws:
        yield None
        return
    kv = renderer.project_condition_kv(renderer.project_conditions(conditions, layout))
    leaves = tuple(tuple(t.detach().requires_grad_(t.requires_grad) for t in pair) for pair in kv)
    yield leaves
    outputs, gradients = [], []
    for pair, detached in zip(kv, leaves, strict=True):
        for t, leaf in zip(pair, detached, strict=True):
            if t.requires_grad:
                outputs.append(t)
                gradients.append(leaf.grad)
    if outputs:
        torch.autograd.backward(outputs, gradients)


def backward_deferred_renderer(
    model, batch, losses, backward_scale=1.0, wrapped=None, forward_events=None, *, image_factor
):
    """Consume a deferred JointModel forward, update losses for ordinary logging.

    Graph storage is one backbone plus one DiT chunk. CQ gradients are bridged
    explicitly, so detaching each temporary chunk leaf does NOT freeze CQ or
    the backbone. A single multi-root backward joins CE and all CQ gradients.
    """
    conditions = losses.pop("image_conditions")
    layout = batch.generation
    if not layout.microbatches:
        raise ValueError("Missing CPU-planned DiT microbatches")
    gradient = torch.empty_like(conditions) if conditions.requires_grad else None
    image_loss = torch.zeros((), device=conditions.device, dtype=torch.float32)
    for index, part in enumerate(layout.microbatches):
        rows = slice(part.start, part.stop)
        leaf = conditions[rows].detach().requires_grad_(gradient is not None)
        group_sync = suspend_gradient_readiness(wrapped) if index+1 < len(layout.microbatches) else nullcontext()
        with group_sync, shared_flow_kv(model.renderer, leaf, part.layout, part.draws_per_chunk < layout.rf) as kv:
            for first_draw in part.draw_ranges(layout.rf):
                chunk = part.bind(batch, first_draw)
                draws = chunk.generation.rf
                # DP readiness belongs to the last contribution, including the shared KV backward.
                sync = suspend_gradient_readiness(wrapped) if first_draw+draws < layout.rf else nullcontext()
                with sync:
                    if forward_events is not None:
                        pair = [device_module().Event(enable_timing=True) for _ in range(2)]
                        pair[0].record()
                    chunk_loss = model.renderer(leaf, chunk, condition_kv=kv) * (draws / layout.rf)
                    if forward_events is not None:
                        pair[1].record()
                        forward_events.append(pair)
                    (chunk_loss * (image_factor * backward_scale)).backward()
                image_loss.add_(chunk_loss.detach().float())
                del chunk_loss, chunk
        del kv
        if gradient is not None:
            if leaf.grad is None:
                raise RuntimeError("Renderer did not propagate its CQ gradient")
            gradient[rows].copy_(leaf.grad)
    roots = [losses["loss"] * backward_scale]
    upstream = [None]
    if gradient is not None:
        roots.append(conditions)
        upstream.append(gradient)
    torch.autograd.backward(roots, upstream)
    losses["loss"] = losses["loss"].detach() + image_loss * (image_factor)
    losses["text_loss"] = losses["text_loss"].detach()
    losses["image_loss"] = image_loss
