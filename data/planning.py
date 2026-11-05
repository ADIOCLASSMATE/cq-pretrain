"""Prepare CQ visibility, GDN events and rotary coordinates once on CPU."""

from dataclasses import replace
import math
import os
import torch
from infra.attention_plan import plan_attention
from infra.plans import MainlinePlan
from infra.layout import build_delta_plan, rope_tables


def content_view(batch, cfg):
    if cfg.content_visibility == "causal":
        ranks = torch.arange(batch.ids.shape[1]).expand_as(batch.sigma)
        return replace(batch, sigma=ranks)
    return batch


def safe_mask(allowed):
    valid = allowed.any(-1)
    mask = ~allowed
    mask[:, :, 0] &= valid
    return mask[:, None].contiguous(), valid


def document_cq_mask(batch, query_positions, query_unconditional=None):
    """Exact CQ masks; compare sigma only inside each independent document.

    The current NPU attention adapter still consumes a dense Boolean mask,
    but constructing full-pack int64 comparison grids needlessly costs CPU
    time and several extra full-size temporary masks at an 8K pack length.
    """
    s = batch.ids.shape[1]
    segments = batch.segments[0]
    sigma = batch.sigma[0]
    query_segments = segments[query_positions]
    mask = torch.ones(s + len(query_positions), s, dtype=torch.bool)
    valid = torch.zeros(s + len(query_positions), dtype=torch.bool)
    for segment in segments.unique().tolist():
        if segment < 0:
            continue
        positions = (segments == segment).nonzero().flatten()
        start, stop = int(positions[0]), int(positions[-1]) + 1
        if stop - start != len(positions):
            raise ValueError("A packed document must occupy contiguous content positions")
        sigmas = sigma[start:stop]
        mask[start:stop, start:stop] = sigmas[None, :] > sigmas[:, None]
        valid[start:stop] = True
        rows = (query_segments == segment).nonzero().flatten()
        qsigmas = sigma[query_positions[rows]]
        mask[s + rows, start:stop] = sigmas[None, :] >= qsigmas[:, None]
        valid[s + rows] = qsigmas > sigmas.min()
    if query_unconditional is not None:
        rows = s + query_unconditional.nonzero().flatten()
        mask[rows] = True
        valid[rows] = False
    # Same finite sentinel as safe_mask; invalid rows are zeroed after FA.
    mask[~valid, 0] = False
    return mask[None, None], valid[None]


def finalize_plan(batch, cfg, plan):
    tile = int(os.environ.get("PRETRAIN_ATTENTION_TILE", "0"))
    if tile < 0:
        raise ValueError("PRETRAIN_ATTENTION_TILE must be nonnegative")
    if tile:
        plan.cq_plan = plan_attention(plan.cq_mask, tile)
        if plan.gen_suffix_mask.numel():
            plan.gen_plan = plan_attention(plan.gen_suffix_mask, tile)
    plan.delta_plan = build_delta_plan(batch, plan, cfg)
    plan.rope_cos, plan.rope_sin = rope_tables(batch, plan, cfg)
    return plan


def build_plan(batch, cfg):
    """CPU-only preparation for the fixed-shape synthetic interleaved benchmark.

    The fake loader uses matching span locations on each batch row. Segment
    IDs and positions remain per row. Real variable layouts need their own
    collation/bucketing; no mask or position tensors are built in forward.
    """
    if cfg.renderer == "wan":
        return build_visual_plan(batch, cfg)
    if cfg.vision_encoder == "qwen35":
        return build_native_plan(batch, cfg)
    if batch.ids.device.type != "cpu":
        raise ValueError("Prepare mainline plans on CPU before device transfer")
    batch = content_view(batch, cfg)
    b, s = batch.ids.shape
    image = batch.image_queries
    if not torch.equal(image, image[:1].expand_as(image)):
        raise ValueError("The synthetic layout requires aligned image spans within a microbatch")
    idx = torch.arange(s)
    starts = (image[0] & ~torch.nn.functional.pad(image[0, :-1], (1, 0), value=False)).nonzero().flatten()
    n = cfg.image_tokens
    q = cfg.image_condition_queries
    side = math.isqrt(n)
    qs = math.isqrt(q)
    local = torch.arange(q)
    # Canonical cell centers, matching Extension.dream_condition at Q64/Q256.
    row = (local // qs + 0.5) * (side / qs) - 0.5
    col = (local % qs + 0.5) * (side / qs) - 0.5
    anchors = (local // qs * side // qs) * side + (local % qs * side // qs)
    selected = (starts[:, None] + anchors).flatten()
    # Only supervised positions need a readout; see build_native_plan.
    text = batch.text_indices[(batch.text_indices < s) & ~image[0][batch.text_indices.clamp(max=s - 1)]]
    if not len(text):
        text = idx[(~image[0]) & batch.segments.ge(0).any(0)]
    # A fully empty rank uses a one-token zero-loss graph to keep DDP hooks.
    if not len(text) and not len(starts):
        text = idx[:1]
    query_positions = torch.cat((text, selected)) if cfg.method == "CQ_IT" else text
    query_image = image[:, query_positions]
    positions = batch.row_col[:, query_positions].float().clone()
    # Use each image's actual RoPE origin, preserving intervening image grids.
    qrope = (
        torch.stack(
            [
                batch.row_col[:, int(start)].float()[:, None, :] + torch.stack((row, col))[:, :, None]
                for start in starts
            ],
            dim=1,
        ).reshape(2, -1, b)
        if len(starts)
        else torch.empty(2, 0, b)
    )
    if cfg.method == "CQ_IT":
        positions[:, len(text) :] = qrope
    cq_same = (batch.segments[:, :, None] == batch.segments[:, None, :]) & batch.segments[:, :, None].ge(0)
    c_allowed = (batch.sigma[:, None, :] <= batch.sigma[:, :, None]) & cq_same
    q_allowed = batch.sigma[:, None, :] < batch.sigma[:, query_positions, None]
    q_allowed &= (batch.segments[:, query_positions, None] == batch.segments[:, None, :]) & batch.segments[
        :, query_positions, None
    ].ge(0)
    cq_mask, cq_valid = safe_mask(torch.cat((c_allowed, q_allowed), 1))
    conditions = (selected[None] + torch.arange(b)[:, None] * s).reshape(b * len(starts), q)
    # A physically single masked-image sequence, compacting only target blocks.
    keep = ~image[0].clone()
    keep[selected] = True
    generation = idx[keep]
    generation = (
        generation[generation <= selected[-1]]
        if len(selected) and cfg.method == "Bi_I+CQ_T"
        else generation[:0]
    )
    g_image = image[:, generation]
    grope = batch.row_col[:, generation].float().clone()
    glocal = image[0, generation].nonzero().flatten()
    if len(generation):
        grope[:, glocal] = qrope
    prefix = int(starts[0]) if len(generation) else 0
    gsegment = batch.segments[:, generation]
    block = (g_image & ~torch.nn.functional.pad(g_image[:, :-1], (1, 0), value=False)).long().cumsum(-1)
    same_image = g_image[:, :, None] & g_image[:, None, :] & (block[:, :, None] == block[:, None, :])
    gidx = torch.arange(len(generation))
    g_allowed = (gidx[None, :] <= gidx[:, None])[None] | same_image
    g_allowed &= (gsegment[:, :, None] == gsegment[:, None, :]) & gsegment[:, :, None].ge(0)
    gen_mask, gen_valid = (
        safe_mask(g_allowed)
        if len(generation)
        else (torch.empty(b, 1, 0, 0, dtype=torch.bool), torch.empty(b, 0, dtype=torch.bool))
    )
    gen_conditions = (
        (glocal[None] + torch.arange(b)[:, None] * len(generation)).reshape(b * len(starts), q)
        if len(generation)
        else torch.empty(0, q, dtype=torch.long)
    )
    output_positions = (query_positions[None] + torch.arange(b)[:, None] * s).flatten()
    plan = MainlinePlan(
        cfg.method,
        prefix,
        query_positions,
        query_image,
        positions,
        generation,
        g_image,
        grope,
        output_positions,
        conditions,
        gen_conditions,
        cq_mask,
        cq_valid,
        gen_mask,
        gen_valid,
        gen_mask[:, :, prefix:].contiguous(),
    )
    return finalize_plan(batch, cfg, plan)


def build_native_plan(batch, cfg):
    """Native image content and fixed-Q conditions occupy separate index spaces.

    The physical pack contains only merged ViT content tokens. Image queries
    write to virtual output slots after S, so even N<Q images retain all Q
    independent conditions without repeating/overwriting physical outputs.
    """
    if batch.ids.device.type != "cpu" or batch.ids.shape[0] != 1:
        raise ValueError("Prepare native-resolution plans on CPU with micro_batch=1")
    batch = content_view(batch, cfg)
    _, s = batch.ids.shape
    q = cfg.image_condition_queries
    qs = math.isqrt(q)
    image = batch.image_queries[0]
    idx = torch.arange(s)
    # Only supervised positions need a readout. Chat scaffolding (role headers,
    # generated instructions) is fed as content but never predicted, so building
    # a query for it costs a full 32-layer forward/backward and is then dropped.
    # CQ queries never write recurrent/KV state, so pruning them is exactly
    # equivalent for every content representation -- not an approximation.
    text = batch.text_indices[(batch.text_indices < s) & ~image[batch.text_indices.clamp(max=s - 1)]]
    if not len(text):
        text = idx[(~image) & batch.segments[0].ge(0)]
    spans = [indices[indices.ge(0)] for indices in batch.image_indices]
    if spans and (batch.vision is None or len(batch.vision.grid_thw) != len(spans)):
        raise ValueError("Native image spans require matching vision geometry")
    image_readouts = torch.arange(len(spans))
    if cfg.method == "CQ_IT" and batch.generation_image_indices is not None:
        image_readouts = batch.generation_image_indices
        if (
            image_readouts.device.type != "cpu"
            or image_readouts.dtype != torch.long
            or image_readouts.ndim != 1
        ):
            raise ValueError("Generation image indices must be a CPU int64 vector")
        if image_readouts.numel() and (
            int(image_readouts.min()) < 0 or int(image_readouts.max()) >= len(spans)
        ):
            raise ValueError("Generation image index is outside the understanding image set")
        if image_readouts.unique().numel() != image_readouts.numel():
            raise ValueError("Generation image indices must not repeat")
    if not len(text) and not spans:
        text = idx[:1]
    local = torch.arange(q)
    anchors = []
    ropes = []
    for indices, (_, h, w) in zip(spans, batch.vision.grid_thw if spans else ()):
        h, w = h // 2, w // 2
        if len(indices) != h * w or not torch.equal(
            indices, torch.arange(int(indices[0]), int(indices[0]) + h * w)
        ):
            raise ValueError("An image must occupy one contiguous merged spatial grid")
        anchors.append(indices[0].repeat(q))
        offsets = torch.stack(((local // qs + 0.5) * (h / qs) - 0.5, (local % qs + 0.5) * (w / qs) - 0.5))
        ropes.append(batch.row_col[:, indices[0]].float()[:, None] + offsets[:, :, None])
    selected = torch.cat(anchors) if anchors else torch.empty(0, dtype=torch.long)
    qrope = torch.cat(ropes, dim=1) if ropes else torch.empty(2, 0, 1)
    # All images remain clean understanding content. CQ readouts never write
    # KV/GDN state, so omitted generation targets need no image queries at all.
    # Keep the historical virtual output slots below for stable image indexing.
    if cfg.method == "CQ_IT":
        selected = selected.reshape(len(spans), q)[image_readouts].flatten()
        qrope = qrope.reshape(2, len(spans), q, 1)[:, image_readouts].reshape(2, -1, 1)
    query_positions = torch.cat((text, selected)) if cfg.method == "CQ_IT" else text
    query_image = batch.image_queries[:, query_positions]
    query_rope = batch.row_col[:, query_positions].float().clone()
    if cfg.method == "CQ_IT":
        query_rope[:, len(text) :] = qrope
    if cfg.preserve_documents:
        cq_mask, cq_valid = document_cq_mask(batch, query_positions)
    else:
        same = (batch.segments[:, :, None] == batch.segments[:, None, :]) & batch.segments[:, :, None].ge(0)
        c_allowed = (batch.sigma[:, None, :] <= batch.sigma[:, :, None]) & same
        q_allowed = batch.sigma[:, None, :] < batch.sigma[:, query_positions, None]
        q_allowed &= (
            batch.segments[:, query_positions, None] == batch.segments[:, None, :]
        ) & batch.segments[:, query_positions, None].ge(0)
        cq_mask, cq_valid = safe_mask(torch.cat((c_allowed, q_allowed), 1))
    conditions = torch.arange(s, s + len(spans) * q).reshape(len(spans), q)
    output_positions = (
        torch.cat((text, conditions[image_readouts].flatten())) if cfg.method == "CQ_IT" else text
    )
    pieces = []
    cursor = 0
    if cfg.method == "Bi_I+CQ_T":
        for indices, anchor in zip(spans, anchors):
            pieces.extend((idx[cursor : int(indices[0])], anchor))
            cursor = int(indices[-1]) + 1
    generation = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)
    g_image = batch.image_queries[:, generation]
    grope = batch.row_col[:, generation].float().clone()
    glocal = g_image[0].nonzero().flatten()
    if len(generation):
        grope[:, glocal] = qrope
    prefix = int(spans[0][0]) if len(generation) else 0
    gsegment = batch.segments[:, generation]
    block = (g_image & ~torch.nn.functional.pad(g_image[:, :-1], (1, 0), value=False)).long().cumsum(-1)
    same_image = g_image[:, :, None] & g_image[:, None, :] & (block[:, :, None] == block[:, None, :])
    gidx = torch.arange(len(generation))
    allowed = (gidx[None, :] <= gidx[:, None])[None] | same_image
    allowed &= (gsegment[:, :, None] == gsegment[:, None, :]) & gsegment[:, :, None].ge(0)
    gen_mask, gen_valid = (
        safe_mask(allowed)
        if len(generation)
        else (torch.empty(1, 1, 0, 0, dtype=torch.bool), torch.empty(1, 0, dtype=torch.bool))
    )
    gen_condition = glocal.reshape(len(spans), q) if len(generation) else torch.empty(0, q, dtype=torch.long)
    plan = MainlinePlan(
        cfg.method,
        prefix,
        query_positions,
        query_image,
        query_rope,
        generation,
        g_image,
        grope,
        output_positions,
        conditions,
        gen_condition,
        cq_mask,
        cq_valid,
        gen_mask,
        gen_valid,
        gen_mask[:, :, prefix:].contiguous(),
        output_size=s + len(spans) * q,
    )
    return finalize_plan(batch, cfg, plan)


def visual_query_coordinates(grid, count, mode):
    t, h, w = grid
    h //= 2
    w //= 2
    if mode == "native":
        return (
            torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"))
            .reshape(3, -1)
            .float()
        )
    nt = min(t, count)
    nh = max(1, math.ceil(math.sqrt(count / nt * h / w)))
    nw = math.ceil(count / (nt * nh))
    axes = [(torch.arange(n) + 0.5) * (size / n) - 0.5 for n, size in zip((nt, nh, nw), (t, h, w))]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij")).reshape(3, -1)
    chosen = ((torch.arange(count) + 0.5) * coordinates.shape[1] / count).long()
    return coordinates[:, chosen]


def build_visual_plan(batch, cfg):
    """One ragged query plan for both images and videos; understanding has no visual Q."""
    if batch.ids.device.type != "cpu" or batch.ids.shape[0] != 1 or cfg.method != "CQ_IT":
        raise ValueError("Unified Wan training uses CPU-planned CQ_IT with microbatch=1")
    batch = content_view(batch, cfg)
    s = batch.ids.shape[1]
    spans = [x[x.ge(0)] for x in batch.image_indices]
    text = batch.text_indices
    selected = (
        torch.arange(len(spans)) if batch.generation_image_indices is None else batch.generation_image_indices
    )
    grids = batch.vision.grid_thw if spans else ()
    counts = [
        (math.prod(g) // 4 if cfg.visual_query_mode == "native" else cfg.image_condition_queries)
        for g in grids
    ]
    maxq = max((counts[i] for i in selected.tolist()), default=0)
    conditions = torch.full((len(spans), maxq), -1, dtype=torch.long)
    offset = s
    for i in selected.tolist():
        n = counts[i]
        conditions[i, :n] = torch.arange(offset, offset + n)
        offset += n
    positions = []
    ropes = []
    outputs = []
    unconditional = torch.zeros(len(text) + offset - s, dtype=torch.bool)
    c = batch.row_col.float()
    if c.shape[0] == 2:
        temporal = c[0].clone()
        for span in spans:
            temporal[span] = c[0, span[0]]
        c = torch.cat((temporal[None], c))
    for i in selected.tolist():
        span = spans[i]
        grid = grids[i]
        n = counts[i]
        if len(span) != math.prod(grid) // 4:
            raise ValueError("Visual spans differ from the native grid")
        coordinates = visual_query_coordinates(grid, n, cfg.visual_query_mode)
        # Native timestamps/temporal positions may be nonuniform. Interpolate
        # query time in the exact sampled native time-block coordinates.
        bases = c[:, span.reshape(grid[0], -1)[:, 0], 0]
        ti = coordinates[0].clamp(0, grid[0] - 1)
        lo = ti.long()
        hi = (lo + 1).clamp(max=grid[0] - 1)
        interpolated = bases[:, lo] + (bases[:, hi] - bases[:, lo]) * (ti - lo)
        coordinates[0] = interpolated[0]
        coordinates[1:] += interpolated[1:]
        if batch.generation_unconditional is not None and batch.generation_unconditional[i]:
            start = len(text) + int(conditions[i, 0]) - s
            unconditional[start : start + n] = True
        positions.append(span[0].repeat(n))
        ropes.append(coordinates[:, :, None])
        outputs.append(conditions[i, :n])
    qpos = torch.cat((text, *positions))
    qrope = torch.cat((c[:, text], *ropes), dim=1)
    image = torch.cat(
        (torch.zeros(len(text), dtype=torch.bool), torch.ones(len(qpos) - len(text), dtype=torch.bool))
    )[None]
    mask, valid = document_cq_mask(batch, qpos, unconditional)
    empty = torch.empty(0, dtype=torch.long)
    em = torch.empty(1, 1, 0, 0, dtype=torch.bool)
    plan = MainlinePlan(
        "CQ_IT",
        0,
        qpos,
        image,
        qrope,
        empty,
        torch.empty(1, 0, dtype=torch.bool),
        torch.empty(3, 0, 1),
        torch.cat((text, *outputs)),
        conditions,
        torch.empty(0, 0, dtype=torch.long),
        mask,
        valid,
        em,
        torch.empty(1, 0, dtype=torch.bool),
        em,
        output_size=offset,
        query_unconditional=unconditional,
    )
    return finalize_plan(batch, cfg, plan)
