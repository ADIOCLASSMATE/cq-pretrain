"""Execute the shared CQ backbone using a prepared layout."""

import torch
from torch.utils.checkpoint import checkpoint


def backbone(model, batch, reference=False, vision_features=None, cache=None):
    cfg = model.cfg
    plan = batch.mainline
    b, s = batch.ids.shape
    if plan is None or plan.method != cfg.method:
        raise ValueError("Batch must be prepared for the selected mainline method")
    content = model.embedding(batch.ids).reshape(-1, cfg.width)
    if len(batch.image_indices):
        if cfg.vision_encoder == "qwen35":
            if batch.vision is None:
                raise ValueError("Qwen vision input is missing")
            patches = model.patch_embed(batch.vision) if vision_features is None else vision_features
            visual_indices = batch.vision.content_indices
        else:
            patches = model.patch_embed(batch.pixels).flatten(2).transpose(1, 2).reshape(-1, cfg.width)
            visual_indices = batch.image_indices.flatten()
    content = content.reshape(b, s, cfg.width).transpose(0, 1)
    query_ids = torch.where(plan.query_image, cfg.image_mask_token_id, cfg.text_mask_token_id)
    query = model.embedding(query_ids).transpose(0, 1)
    generation = None
    if cfg.method == "Bi_I+CQ_T" and len(plan.generation_positions):
        gids = batch.ids[:, plan.generation_positions].masked_fill(
            plan.generation_image, cfg.image_mask_token_id
        )
        generation = model.embedding(gids).transpose(0, 1)
    parts = (content, query) if generation is None else (content, generation[plan.prefix :], query)
    hidden = torch.cat(parts).contiguous()
    if len(batch.image_indices):
        # Cat owns this buffer. TP embedding outputs can be custom-autograd
        # views, so replacing their content in-place would invalidate backward.
        hidden[:, 0].index_copy_(0, visual_indices, patches)
    rope = (plan.rope_cos.to(hidden.dtype), plan.rope_sin.to(hidden.dtype))
    for index, layer in enumerate(model.layers):
        recompute = cfg.checkpoint_backbone or index >= cfg.layers - cfg.checkpoint_backbone_layers
        if (
            (index + 1) % cfg.full_attention_interval == 0
            and index // cfg.full_attention_interval < cfg.keep_attention_activations
        ):
            recompute = False
        if cache is not None:
            hidden = layer(hidden, rope, plan, s, reference, cache[index])
        elif recompute and model.training and torch.is_grad_enabled():
            hidden = checkpoint(layer, hidden, rope, plan, s, reference, use_reentrant=True)
        else:
            hidden = layer(hidden, rope, plan, s, reference)
    keys = s + (len(plan.generation_positions) - plan.prefix if generation is not None else 0)
    query = model.final_norm(hidden[keys:], cfg.fused and not reference)
    output = query.new_zeros((plan.output_size or b * s, cfg.width)).index_copy_(
        0, plan.output_positions, query.transpose(0, 1).reshape(-1, cfg.width)
    )
    if generation is not None:
        generation = torch.cat((hidden[: plan.prefix], hidden[s:keys]))
        condition = generation.transpose(0, 1).reshape(-1, cfg.width)[plan.generation_condition]
        condition = model.final_norm(condition, cfg.fused and not reference)
        output.index_copy_(0, plan.condition_indices.flatten(), condition.reshape(-1, cfg.width))
    return output
