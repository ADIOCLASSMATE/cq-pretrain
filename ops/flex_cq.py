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
"""

import os
from functools import lru_cache

import torch

_ENABLED = os.environ.get("PRETRAIN_FLEX_ATTENTION", "0") == "1"
_BLOCK = int(os.environ.get("PRETRAIN_FLEX_BLOCK", "128"))


@lru_cache(maxsize=1)
def _compiled():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False)


def usable(q, predicate, mask):
    """Only take this path when it is exactly equivalent and actually supported."""
    if not _ENABLED or predicate is None or q.device.type != "cuda":
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
    if cached is not None and cached.shape[-2:] == (rows, keys):
        return cached
    from torch.nn.attention.flex_attention import create_block_mask

    built = create_block_mask(
        mask_mod(predicate), B=None, H=None, Q_LEN=rows, KV_LEN=keys,
        device=predicate.key_sigma.device, BLOCK_SIZE=_BLOCK, _compile=True,
    )
    predicate.block_mask = built
    return built


def attention(q, k, v, predicate, valid_rows=None, scale=None):
    """q/k/v are [sequence, batch, heads, dim]; returns [sequence, batch, heads*dim]."""
    rows, keys = q.shape[0], k.shape[0]
    query, key, value = (t.permute(1, 2, 0, 3).contiguous() for t in (q, k, v))
    out = _compiled()(
        query, key, value,
        block_mask=block_mask(predicate, rows, keys),
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        enable_gqa=q.shape[2] != k.shape[2],
    )
    result = out.permute(2, 0, 1, 3).flatten(-2).contiguous()
    if valid_rows is not None:
        result = result * valid_rows.transpose(0, 1).unsqueeze(-1).to(result.dtype)
    return result


def sparsity(predicate, rows, keys):
    """Fraction of 128x128 blocks the mask lets the kernel skip."""
    bm = block_mask(predicate, rows, keys)
    try:
        return float(bm.sparsity()) / 100.0
    except Exception:
        return None
