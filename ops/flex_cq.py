"""Run the CQ mask predicate with PyTorch's FlexAttention.

No kernel is written here. The CQ visibility rule is evaluated as a ``mask_mod``
predicate so the published block-sparse kernel can skip fully-masked blocks and
never read a dense ``[1,1,2S,S]`` mask tensor; GQA is handled by the kernel's own
``enable_gqa`` rather than by materializing repeated K/V.

Tile sizes are *not* searched at runtime. FlexAttention's defaults assume far
more shared memory than Ada (sm89) provides, and a failed Triton compile leaves
Inductor's async worker pool unusable for the rest of the process -- so an
in-process retry ladder deadlocks training. Instead ``scripts/validation/
probe_flex_tile.py`` probes candidates offline, one isolated process each, and
the winner is recorded in ``configs/flex_tiles.json``. At runtime we only look
the answer up. Anything unrecognized falls back to the dense-mask path, which
stays the numerical reference.
"""

import functools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ENABLED = os.environ.get("PRETRAIN_FLEX_ATTENTION", "0") == "1"


@functools.lru_cache(maxsize=1)
def _table():
    path = ROOT / "configs" / "flex_tiles.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


@functools.lru_cache(maxsize=8)
def tile_options(head_dim, capability):
    """Recorded tile config for this head size and GPU, or None to stay dense."""
    entry = _table().get(f"sm{capability}", {}).get(str(head_dim))
    if not entry:
        return None
    back = min(entry["block_m"], entry["block_n"])
    options = {
        "BLOCK_M": entry["block_m"], "BLOCK_N": entry["block_n"],
        "BLOCK_M1": back, "BLOCK_N1": back, "BLOCK_M2": back, "BLOCK_N2": back,
    }
    # Fewer pipeline stages is what makes the larger tile fit Ada's shared
    # memory at head_dim=256; it is part of the recorded answer, not a default.
    for key in ("num_stages", "num_warps"):
        if entry.get(key):
            options[key] = entry[key]
    return options


@functools.lru_cache(maxsize=1)
def _compiled():
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False)


def usable(q, predicate, mask):
    """True only when a tile config was recorded for this GPU and head size."""
    import torch

    if not (_ENABLED and predicate is not None and q.device.type == "cuda"):
        return False
    if torch.are_deterministic_algorithms_enabled():
        return False
    # The predicate must describe exactly the rows/keys of the dense mask it
    # replaces; otherwise fall back rather than silently attend over the wrong
    # extent.
    if mask is not None and tuple(mask.shape[-2:]) != (len(predicate.row_sigma),
                                                       len(predicate.key_sigma)):
        return False
    if q.shape[0] != len(predicate.row_sigma):
        return False
    major, minor = torch.cuda.get_device_capability(q.device)
    return tile_options(q.shape[-1], f"{major}{minor}") is not None


def block_mask(predicate, rows, keys, device):
    """Build (and cache on the predicate) the BlockMask for this batch.

    The same mask serves every full-attention layer in the step, so it is built
    once per batch rather than once per layer.
    """
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    if predicate.block_mask is not None:
        return predicate.block_mask
    key_sigma = predicate.key_sigma
    key_segment = predicate.key_segment
    row_sigma = predicate.row_sigma
    row_segment = predicate.row_segment
    row_is_query = predicate.row_is_query
    row_unconditional = predicate.row_unconditional
    row_invalid = predicate.row_invalid

    def mask_mod(b, h, q_idx, kv_idx):
        same = (row_segment[q_idx] == key_segment[kv_idx]) & (row_segment[q_idx] >= 0)
        strict = key_sigma[kv_idx] < row_sigma[q_idx]
        loose = key_sigma[kv_idx] <= row_sigma[q_idx]
        ok = torch.where(row_is_query[q_idx], strict, loose)
        value = same & ok & ~row_unconditional[q_idx]
        # Same finite sentinel as the dense builder: an all-masked row keeps
        # key 0 so the kernel has a defined softmax; it is zeroed afterwards.
        return value | (row_invalid[q_idx] & (kv_idx == 0))

    mask = create_block_mask(mask_mod, None, None, rows, keys,
                             device=str(device), BLOCK_SIZE=(128, 128), _compile=True)
    predicate.block_mask = mask
    return mask


def attention(q, k, v, predicate, valid_rows, scale=None):
    """q: [sequence, batch, heads, dim]; k/v may have fewer (GQA) heads."""
    import torch

    major, minor = torch.cuda.get_device_capability(q.device)
    options = tile_options(q.shape[-1], f"{major}{minor}")
    if options is None:
        raise ValueError("No recorded FlexAttention tile for this GPU and head size")
    rows, keys = q.shape[0], k.shape[0]
    query, key, value = (t.permute(1, 2, 0, 3) for t in (q, k, v))
    result = _compiled()(
        query.contiguous(), key.contiguous(), value.contiguous(),
        block_mask=block_mask(predicate, rows, keys, q.device),
        enable_gqa=q.shape[2] != k.shape[2],
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        kernel_options=options,
    )
    result = result.permute(2, 0, 1, 3).flatten(-2).contiguous()
    if valid_rows is not None:
        result = result * valid_rows.transpose(0, 1).unsqueeze(-1).to(result.dtype)
    return result
