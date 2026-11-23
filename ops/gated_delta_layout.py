"""CPU plans for write-once content and read-only CQ GatedDeltaNet streams.

Within each document, content writes follow sigma order. A tied image group
is fully written before any of its content readouts. CQ queries read before
the matching sigma group and never alter either recurrent or convolution state.
"""

from bisect import bisect_left
from dataclasses import dataclass, fields
import torch


@dataclass
class DeltaPlan:
    content_order: torch.Tensor
    event_q: torch.Tensor
    event_write: torch.Tensor
    output_event: torch.Tensor
    output_hidden: torch.Tensor
    query_source: torch.Tensor
    query_history: torch.Tensor
    content_cu: torch.Tensor
    event_cu: torch.Tensor
    content_bounds: tuple
    event_bounds: tuple
    hidden_length: int
    hidden_event: torch.Tensor
    padding_hidden: torch.Tensor
    # Queries from this index of query_source on also write, inside a private
    # sequence of their own target. len(query_source) when no query writes.
    query_write_start: int = -1

    def to(self, device, non_blocking=False):
        return DeltaPlan(
            **{
                f.name: (
                    getattr(self, f.name).to(device, non_blocking=non_blocking)
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )

    def pin_memory(self, device):
        return DeltaPlan(
            **{
                f.name: (
                    getattr(self, f.name).pin_memory(device=device)
                    if isinstance(getattr(self, f.name), torch.Tensor) and getattr(self, f.name).numel()
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )


def build_delta_plan(batch, plan, cfg):
    if batch.ids.device.type != "cpu" or batch.ids.shape[0] != 1:
        raise ValueError("Prepare Qwen3.5 CQ plans on CPU, microbatch=1")
    s = batch.ids.shape[1]
    keys = s
    null_anchor = getattr(plan, "null_anchor", -1)
    nq = len(plan.query_positions)
    sigmas = batch.sigma[0].tolist()
    segments = batch.segments[0].tolist()
    # Each segment has one content stream. CQ reads never write to it.
    docs = {}
    for i, doc in enumerate(segments):
        if doc >= 0:
            docs.setdefault(doc, []).append((sigmas[i], i))
    if not docs:
        docs[0] = [(0, 0)]  # zero-loss EOF graph
    query_by_doc = {}
    unconditional = plan.query_unconditional.tolist() if plan.query_unconditional is not None else [False] * nq
    read_sigmas = plan.query_read_sigma.tolist() if plan.query_read_sigma is not None else None
    blocks = getattr(plan, "query_blocks", ())
    in_block = set()
    for row_start, row_stop, _, _ in blocks:
        in_block.update(range(row_start - keys, row_stop - keys))
    for j, p in enumerate(plan.query_positions.tolist()):
        if j in in_block:
            continue  # served by its target's private sequence below
        segment = segments[p]
        sigma = sigmas[p] if read_sigmas is None else read_sigmas[j]
        if unconditional[j] or (segment < 0 and read_sigmas is not None):
            sigma = None  # Empty prefix also for documents with a nonzero first rank.
        elif p == null_anchor and sigma == 1:
            pass  # Read after the singleton segment's only write.
        elif read_sigmas is not None and not 0 <= sigma <= sigmas[p]:
            raise ValueError("A query cutoff cannot read past its physical position")
        query_by_doc.setdefault(max(0, segment), {}).setdefault(sigma, []).append((j, keys + j))
    order = []
    event_q = []
    event_write = []
    out_events = []
    out_hidden = []
    qsource = []
    qhistory = []
    cb = [0]
    eb = [0]
    content_index = {}
    for doc, items in docs.items():
        groups = {}
        for rank, index in sorted(items):
            groups.setdefault(rank, []).append(index)
        if null_anchor >= 0 and items == [(0, null_anchor)]:
            groups[1] = []  # Terminal reads see the token and its convolution history.
        queries = query_by_doc.get(doc, {})
        # A numerical cutoff can lie between writes (including zero in a
        # later packed document). Read just before the first rank >= cutoff.
        missing = [r for r in queries if r is not None and r not in groups] if read_sigmas is not None else []
        if missing:
            ranks = list(groups)
            for cutoff in missing:
                before = bisect_left(ranks, cutoff)
                if before == len(ranks):
                    raise ValueError("Query cutoff exceeds the document's last content rank")
                queries.setdefault(ranks[before], []).extend(queries.pop(cutoff))
        if None in queries:
            # Explicit non-CFG Q reads before any C write, with empty conv
            # history and zero recurrent state, regardless of target position.
            first = next(iter(groups))
            queries[first] = queries.pop(None) + queries.get(first, [])
        history = []
        for sigma, group_items in groups.items():
            for _, hidden_idx in queries.get(sigma, []):
                qi = len(qsource)
                qsource.append(hidden_idx)
                prev = history[-(cfg.linear_conv_kernel - 1) :] if cfg.linear_conv_kernel > 1 else []
                qhistory.append([-1] * (cfg.linear_conv_kernel - 1 - len(prev)) + prev)
                event_q.append(("q", qi))
                event_write.append(-1)
                out_events.append(len(event_q) - 1)
                out_hidden.append(hidden_idx)
            members = []
            for hidden_idx in group_items:
                ci = len(order)
                order.append(hidden_idx)
                content_index[hidden_idx] = ci
                members.append((ci, hidden_idx))
                event_q.append(("c", ci))
                event_write.append(ci)
                history.append(ci)
                if len(group_items) == 1:
                    out_events.append(len(event_q) - 1)
                    out_hidden.append(hidden_idx)
            if len(group_items) > 1:
                # Bidirectional content visibility within the image group.
                for ci, hidden_idx in members:
                    event_q.append(("c", ci))
                    event_write.append(-1)
                    out_events.append(len(event_q) - 1)
                    out_hidden.append(hidden_idx)
        cb.append(len(order))
        eb.append(len(event_q))
    # Bidirectional visual Q. The content stream must not see these writes, so
    # each target gets a sequence of its own: replay the C prefix it may read
    # (the same k/v/decay as in the document, hence the same state), let every
    # Q of the target write, then read them all. This is the tied-group rule
    # content image blocks already follow, applied to a private continuation.
    write_start = len(qsource)
    for row_start, row_stop, key_start, key_stop in blocks:
        prefix = [content_index[i] for _, i in sorted((sigmas[i], i) for i in range(key_start, key_stop))]
        for ci in prefix:
            event_q.append(("c", ci))
            event_write.append(ci)
        prev = prefix[-(cfg.linear_conv_kernel - 1) :] if cfg.linear_conv_kernel > 1 else []
        tail = [-1] * (cfg.linear_conv_kernel - 1 - len(prev)) + prev
        members = []
        for hidden_idx in range(row_start, row_stop):
            qi = len(qsource)
            qsource.append(hidden_idx)
            qhistory.append(tail)
            members.append((qi, hidden_idx))
            event_q.append(("q", qi))
            event_write.append(("q", qi - write_start))
        for qi, hidden_idx in members:
            event_q.append(("q", qi))
            event_write.append(-1)
            out_events.append(len(event_q) - 1)
            out_hidden.append(hidden_idx)
        eb.append(len(event_q))
    nc = len(order)
    writers = len(qsource) - write_start
    long = lambda xs: torch.tensor(xs, dtype=torch.long)
    eq = [i if kind == "c" else nc + i for kind, i in event_q]
    # Write slots: [content ; writing queries ; one zero row meaning "no write"].
    event_write = [nc + i[1] if isinstance(i, tuple) else nc + writers if i < 0 else i for i in event_write]
    hidden_event = torch.zeros(keys + nq, dtype=torch.long)
    hidden_event[long(out_hidden)] = long(out_events)
    padding = torch.ones(keys + nq, dtype=torch.bool)
    padding[long(out_hidden)] = False
    return DeltaPlan(
        long(order),
        long(eq),
        long(event_write),
        long(out_events),
        long(out_hidden),
        long(qsource),
        long([[nc if i < 0 else i for i in row] for row in qhistory]).reshape(len(qsource), cfg.linear_conv_kernel - 1),
        torch.tensor(cb, dtype=torch.int32),
        long(eb),
        tuple(cb),
        tuple(eb),
        keys + nq,
        hidden_event,
        padding.nonzero().flatten(),
        write_start,
    )


def rope_tables(batch, plan, cfg):
    """Official interleaved temporal/height/width partial RoPE for static images."""
    return position_rope(torch.cat((batch.row_col, plan.query_rope), dim=1), cfg)


def position_rope(pos, cfg):
    pos = pos.double()
    dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    inv = cfg.rope_theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    frequencies = pos[..., None] * inv
    angles = frequencies[0].clone()
    for axis in (1, 2):
        stop = 3 * cfg.mrope_section[axis]
        angles[..., axis:stop:3] = frequencies[axis, ..., axis:stop:3]
    angles = torch.cat((angles, angles), dim=-1)[:, :, None, :]
    return angles.cos().float(), angles.sin().float()
