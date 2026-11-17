"""Drive CQ attention from the plan's predicate instead of a dense mask.

No kernel is written here. ``torch.nn.attention.flex_attention`` is PyTorch's
published block-sparse attention entry point; this module only supplies the CQ
visibility rule as its ``mask_mod`` and reuses the block mask across the layers
of one batch.

The rule is exactly the one ``document_cq_mask`` materializes --
``tests/test_cq_predicate.py`` requires the two to agree bitwise -- so the only
thing that changes is that fully-masked blocks are skipped instead of being
multiplied by a [1,1,2S,S] boolean tensor. Grouped-query heads are handled by
``enable_gqa`` rather than by expanding K/V.

Tile sizes are not guessed from a shared-memory formula: this backbone's
head_dim is 256 and Ada (sm89) offers ~99 KB per SM against Hopper's 228 KB, so
the upstream defaults do not compile here. The ladder below is tried largest
first and the first configuration that compiles on this card is cached. If none
compiles the module disables itself and callers keep the dense mask, because a
slower attention is always better than a failed training step.
"""

import os

import torch

_ENABLED = os.environ.get("PRETRAIN_FLEX_ATTENTION", "0") == "1"
_BLOCK = int(os.environ.get("PRETRAIN_FLEX_BLOCK", "128"))
_DISABLED_REASON = None
_COMPILED = None
_OPTIONS = {}

# Descending working-set size; (forward M, forward N, backward M, backward N).
_LADDER = (
    (128, 128, 64, 64),
    (128, 64, 64, 32),
    (64, 64, 32, 32),
    (64, 32, 32, 32),
    (32, 32, 32, 32),
    (32, 32, 16, 16),
    (16, 16, 16, 16),
)


def _options(fm, fn, bm, bn):
    return dict(BLOCK_M=fm, BLOCK_N=fn,
                BLOCK_M1=bn, BLOCK_N1=bm, BLOCK_M2=bm, BLOCK_N2=bn,
                num_stages=1, num_warps=4)


def compiled():
    global _COMPILED
    if _COMPILED is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED = torch.compile(flex_attention, dynamic=False)
    return _COMPILED


def usable(q, predicate, mask):
    """Only take this path when it is exactly equivalent and actually supported."""
    if not _ENABLED or _DISABLED_REASON or predicate is None or q.device.type != "cuda":
        return False
    if torch.are_deterministic_algorithms_enabled():
        return False
    # The plan's predicate describes this exact [rows, keys] pair.
    rows, keys = len(predicate.row_sigma), len(predicate.key_sigma)
    if q.shape[0] != rows or mask is None or tuple(mask.shape[-2:]) != (rows, keys):
        return False
    return predicate.key_sigma.device.type == "cuda"


def mask_mod(predicate):
    key_sigma = predicate.key_sigma
    key_segment = predicate.key_segment
    row_sigma = predicate.row_sigma
    row_segment = predicate.row_segment
    row_is_query = predicate.row_is_query
    row_unconditional = predicate.row_unconditional
    row_invalid = predicate.row_invalid

    def cq(b, h, q_idx, kv_idx):
        rseg = row_segment[q_idx]
        same = (rseg == key_segment[kv_idx]) & (rseg >= 0)
        rsig = row_sigma[q_idx]
        ksig = key_sigma[kv_idx]
        # Queries read strictly earlier ranks; content also reads its own rank,
        # which is what makes a tied image group bidirectional.
        ordered = torch.where(row_is_query[q_idx], ksig < rsig, ksig <= rsig)
        allowed = same & ordered & ~row_unconditional[q_idx]
        # A row with no readable key keeps the same finite sentinel the dense
        # builder uses; its output is zeroed afterwards through valid_rows.
        return allowed | (row_invalid[q_idx] & (kv_idx == 0))

    return cq


def block_mask(predicate, rows, keys):
    """One block mask per batch, reused by every full-attention layer."""
    cached = predicate.block_mask
    if cached is not None and tuple(cached.shape[-2:]) == (rows, keys):
        return cached
    from torch.nn.attention.flex_attention import create_block_mask

    built = create_block_mask(
        mask_mod(predicate), B=None, H=None, Q_LEN=rows, KV_LEN=keys,
        device=predicate.key_sigma.device, BLOCK_SIZE=_BLOCK, _compile=True,
    )
    predicate.block_mask = built
    return built


def _run(query, key, value, bm, scale, gqa, options, needs_backward):
    out = compiled()(query, key, value, block_mask=bm, scale=scale,
                     enable_gqa=gqa, kernel_options=options)
    if needs_backward:
        # Force the backward kernel to compile now, so a shared-memory failure
        # is discovered during the ladder search and not mid-training.
        torch.autograd.grad(out.sum(), [t for t in (query, key, value)
                                        if t.requires_grad], retain_graph=True)
    return out


def attention(q, k, v, predicate, valid_rows=None, scale=None):
    """q/k/v are [sequence, batch, heads, dim]; returns [sequence, batch, heads*dim]."""
    rows, keys = q.shape[0], k.shape[0]
    query, key, value = (t.permute(1, 2, 0, 3).contiguous() for t in (q, k, v))
    bm = block_mask(predicate, rows, keys)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    gqa = q.shape[2] != k.shape[2]
    signature = (q.shape[-1], q.element_size(), gqa, torch.is_grad_enabled())

    chosen = _OPTIONS.get(signature)
    if chosen is not None:
        out = compiled()(query, key, value, block_mask=bm, scale=scale,
                         enable_gqa=gqa, kernel_options=chosen)
    else:
        needs_backward = torch.is_grad_enabled() and any(
            t.requires_grad for t in (query, key, value))
        errors = []
        out = None
        for fm, fn, bmm, bnn in _LADDER:
            options = _options(fm, fn, bmm, bnn)
            try:
                out = _run(query, key, value, bm, scale, gqa, options, needs_backward)
            except Exception as error:  # OutOfResources and its compile wrappers
                errors.append(f"{fm}x{fn}/{bmm}x{bnn}: {type(error).__name__}")
                continue
            _OPTIONS[signature] = options
            break
        if out is None:
            raise RuntimeError(
                "No FlexAttention tile configuration compiles on this device: "
                + "; ".join(errors))
        if needs_backward:
            # The probe above built a throwaway graph; redo the forward so the
            # returned tensor owns the graph the caller will back-propagate.
            out = compiled()(query, key, value, block_mask=bm, scale=scale,
                             enable_gqa=gqa, kernel_options=_OPTIONS[signature])

    result = out.permute(2, 0, 1, 3).flatten(-2).contiguous()
    if valid_rows is not None:
        result = result * valid_rows.transpose(0, 1).unsqueeze(-1).to(result.dtype)
    return result


def disable(reason):
    """Stop offering this path; callers fall back to the dense mask."""
    global _DISABLED_REASON
    _DISABLED_REASON = reason
    return reason


def chosen_options():
    return {str(k): v for k, v in _OPTIONS.items()}


def sparsity(predicate, rows, keys):
    """Fraction of blocks the mask lets the kernel skip."""
    bm = block_mask(predicate, rows, keys)
    try:
        return round(float(bm.sparsity()) / 100.0, 4)
    except Exception:
        return None
