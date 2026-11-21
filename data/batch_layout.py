"""Prepare CQ visibility, GDN events and rotary coordinates once on CPU."""

from dataclasses import replace
import math
import os
import torch
from ops.attention_layout import plan_attention, document_cq_mask, cq_mask_vectors
from models.cq_layout import MainlinePlan
from ops.gated_delta_layout import build_delta_plan, rope_tables


def content_view(batch, cfg):
    if cfg.content_visibility == "causal":
        ranks = torch.arange(batch.ids.shape[1]).expand_as(batch.sigma)
        return replace(batch, sigma=ranks)
    return batch


def attach_cq_predicate(batch, plan, valid, unconditional=None):
    """Same mask, expressed as vectors so a kernel need not read a dense tensor."""
    plan.cq_predicate = cq_mask_vectors(batch, plan.query_positions, valid, unconditional, plan.query_read_sigma)
    return plan


def finalize_plan(batch, cfg, plan):
    tile = int(os.environ.get("PRETRAIN_ATTENTION_TILE", "0"))
    if tile < 0:
        raise ValueError("PRETRAIN_ATTENTION_TILE must be nonnegative")
    if tile:
        plan.cq_plan = plan_attention(plan.cq_mask, tile)
    plan.delta_plan = build_delta_plan(batch, plan, cfg)
    plan.rope_cos, plan.rope_sin = rope_tables(batch, plan, cfg)
    return plan


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


def build_plan(batch, cfg):
    """One ragged query plan for both images and videos; understanding has no visual Q."""
    if batch.ids.device.type != "cpu" or batch.ids.shape[0] != 1:
        raise ValueError("Unified Wan training uses CPU-planned CQ_IT with microbatch=1")
    batch = content_view(batch, cfg)
    s = batch.ids.shape[1]
    spans = [x[x.ge(0)] for x in batch.image_indices]
    text = batch.text_indices
    selected = batch.generation_image_indices
    if selected.ndim != 1 or selected.dtype != torch.long:
        raise ValueError("Expected a vector of visual occurrence indices")
    targets = selected.tolist()
    if len(set(targets)) != len(targets) or any(i < 0 or i >= len(spans) for i in targets):
        raise ValueError("Generation occurrence indices must be distinct and in range")
    grids = batch.vision.grid_thw if spans else ()
    counts = [
        (math.prod(g) // 4 if cfg.visual_query_mode == "native" else cfg.image_condition_queries)
        for g in grids
    ]
    maxq = max((counts[i] for i in targets), default=0)
    conditions = torch.full((len(spans), maxq), -1, dtype=torch.long)
    offset = s
    for i in targets:
        n = counts[i]
        conditions[i, :n] = torch.arange(offset, offset + n)
        offset += n
    positions = []
    ropes = []
    outputs = []
    unconditional = torch.zeros(len(text) + offset - s, dtype=torch.bool)
    read_sigma = torch.empty(len(unconditional), dtype=batch.sigma.dtype)
    read_sigma[:len(text)] = batch.sigma[0, text]
    c = batch.row_col.float()
    bidirectional = getattr(cfg, "query_interaction", "none") == "bidirectional"
    null_at_opener = getattr(cfg, "null_condition", "masked") == "bos"
    blocks = []
    block_rows = torch.zeros_like(unconditional)
    for i in targets:
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
        start = len(text) + int(conditions[i, 0]) - s
        rows = slice(start, start + n)
        read_position = span[0]
        if batch.generation_instruction_dropped is not None and batch.generation_instruction_dropped[i]:
            if batch.generation_instruction_starts is None or (
                batch.generation_unconditional is not None and batch.generation_unconditional[i]
            ):
                raise ValueError("History queries need a conditional target and an instruction boundary")
            read_position = batch.generation_instruction_starts[i]
            if not 0 <= read_position < span[0] or batch.segments[0, read_position] != batch.segments[0, span[0]]:
                raise ValueError("Instruction cutoff must precede the target in the same document")
            if batch.image_queries[0, read_position:span[0]].any():
                raise ValueError("An instruction cutoff cannot discard an intervening visual")
        read_sigma[rows] = batch.sigma[0, read_position]
        if batch.generation_unconditional is not None and batch.generation_unconditional[i]:
            opener = -1
            if null_at_opener and batch.generation_null_positions is not None:
                opener = int(batch.generation_null_positions[i])
            if opener >= 0:
                # Read exactly the document opener. Its token, position and
                # state are the same in every document, so nothing leaks; the
                # Q coordinates are re-based so their offset to it is fixed too.
                if not 0 < opener <= span[0] or batch.segments[0, opener - 1] != batch.segments[0, span[0]]:
                    raise ValueError("The null read position must follow the opener of the target's document")
                read_sigma[rows] = batch.sigma[0, opener]
                coordinates = coordinates - coordinates[0, 0] + c[0, opener - 1, 0] + 1
            else:
                unconditional[rows] = True
                read_sigma[rows] = 0
        if bidirectional:
            # C visible to this target is one contiguous prefix of its document:
            # ranks never decrease along positions, and Q reads strictly below
            # its cutoff. The same range drives both attention and GDN.
            document = (batch.segments[0] == batch.segments[0, span[0]]).nonzero().flatten()
            first, last = int(document[0]), int(document[-1]) + 1
            visible = batch.sigma[0, first:last] < read_sigma[start]
            count = 0 if unconditional[start] else int(visible.sum())
            if count and not visible[:count].all():
                raise ValueError("A bidirectional query block needs a contiguous visible prefix")
            blocks.append((s + start, s + start + n, first, first + count))
            block_rows[rows] = True
        positions.append(span[0].repeat(n))
        ropes.append(coordinates[:, :, None])
        outputs.append(conditions[i, :n])
    qpos = torch.cat((text, *positions))
    qrope = torch.cat((c[:, text], *ropes), dim=1)
    image = torch.cat(
        (torch.zeros(len(text), dtype=torch.bool), torch.ones(len(qpos) - len(text), dtype=torch.bool))
    )[None]
    # Block rows are served by their own [prefix ; block] attention and GDN
    # sequence. In the shared C kernels they read nothing and cost nothing.
    hidden_from_content = unconditional | block_rows
    mask, valid = document_cq_mask(batch, qpos, hidden_from_content, read_sigma)
    plan = MainlinePlan(
        query_positions=qpos, query_image=image, query_rope=qrope,
        output_positions=torch.cat((text, *outputs)), condition_indices=conditions,
        cq_mask=mask, cq_valid=valid,
        output_size=offset,
        query_unconditional=unconditional,
        query_read_sigma=read_sigma,
        query_blocks=tuple(blocks),
    )
    attach_cq_predicate(batch, plan, valid, hidden_from_content)
    return finalize_plan(batch, cfg, plan)
