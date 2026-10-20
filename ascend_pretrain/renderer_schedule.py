"""Bound DiT activations while backpropagating the shared CQ backbone once.

The MOSS objective is an equal average over images, not over chunks or pixels.
Every chunk retains the parent layout's loss weights and pre-sampled flow data.
Only the final chunk marks renderer gradients ready for MCore's DP reduction.
"""
from contextlib import contextmanager, nullcontext
import torch


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
    groups=wrapped.bucket_groups+wrapped.expert_parallel_bucket_groups
    previous=[g.is_last_microbatch for g in groups]
    for group in groups:group.is_last_microbatch=False
    try:yield
    finally:
        for group,state in zip(groups,previous):group.is_last_microbatch=state


def backward_deferred_renderer(model,batch,losses,backward_scale=1.,wrapped=None,forward_events=None):
    """Consume a deferred JointModel forward, update losses for ordinary logging.

    Graph storage is one backbone plus one DiT chunk. CQ gradients are bridged
    explicitly, so detaching each temporary chunk leaf does NOT freeze CQ or
    the backbone. A single multi-root backward joins CE and all CQ gradients.
    """
    conditions=losses.pop('_image_conditions')
    image_scale=losses.pop('_image_scale')
    parts=batch.generation.microbatches
    if not parts:raise ValueError('Missing CPU-planned DiT microbatches')
    gradient=torch.empty_like(conditions) if conditions.requires_grad else None
    image_loss=torch.zeros((),device=conditions.device,dtype=torch.float32)
    for index,part in enumerate(parts):
        leaf=conditions[part.start:part.stop].detach().requires_grad_(gradient is not None)
        # Other DP replicas may have different chunk counts. Register exactly
        # one final renderer contribution, independent of the number of chunks.
        context=suspend_gradient_readiness(wrapped) if index+1<len(parts) else nullcontext()
        with context:
            if forward_events is not None:
                pair=[torch.npu.Event(enable_timing=True) for _ in range(2)]
                pair[0].record()
            chunk_loss=model.renderer(leaf,part.bind(batch))
            if forward_events is not None:
                pair[1].record();forward_events.append(pair)
            (chunk_loss*(model.cfg.image_weight*image_scale*backward_scale)).backward()
        image_loss.add_(chunk_loss.detach().float())
        if gradient is not None:
            if leaf.grad is None:raise RuntimeError('Renderer did not propagate its CQ gradient')
            gradient[part.start:part.stop].copy_(leaf.grad)
        del chunk_loss,leaf
    roots=[losses['loss']*backward_scale]
    upstream=[None]
    if gradient is not None:
        roots.append(conditions);upstream.append(gradient)
    torch.autograd.backward(roots,upstream)
    losses['loss']=losses['loss'].detach()+image_loss*(model.cfg.image_weight*image_scale)
    losses['text_loss']=losses['text_loss'].detach()
    losses['image_loss']=image_loss
