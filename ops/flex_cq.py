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
# Real packs differ in length and query count on every step. The compiled
# kernel is shape-specialized, and once Dynamo's recompile limit is hit it
# silently runs FlexAttention's eager dense fallback, which materializes
# [heads, rows, keys] scores and exhausts the card. Rows and keys are therefore
# padded up to a multiple of this bucket: padded keys are invisible, padded
# rows keep only the key-0 sentinel block and are sliced off, so the
# block-sparse kernel skips nearly all of the padding and a run compiles two
# or three extents in total.
_BUCKET = int(os.environ.get("PRETRAIN_FLEX_BUCKET", "8192"))


def padded_extent(rows, keys):
    if _BUCKET <= 0:
        return rows, keys
    return -(-rows // _BUCKET) * _BUCKET, -(-keys // _BUCKET) * _BUCKET


@functools.lru_cache(maxsize=1)
def _table():
    path = ROOT / "configs" / "flex_tiles.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


@functools.lru_cache(maxsize=8)
def device_tile_options(head_dim):
    """Tile config for the GPU this process runs on, or None to stay dense."""
    import torch

    if not (_ENABLED and torch.cuda.is_available()):
        return None
    return tile_options(head_dim, "%d%d" % torch.cuda.get_device_capability())


def tile_options(head_dim, capability):
    """Recorded tile config for this head size and GPU, or None to stay dense."""
    entry = _table().get(f"sm{capability}", {}).get(str(head_dim))
    if not entry:
        return None
    back = min(entry["block_m"], entry["block_n"])
    options = {
        "BLOCK_M": entry["block_m"],
        "BLOCK_N": entry["block_n"],
        "BLOCK_M1": back,
        "BLOCK_N1": back,
        "BLOCK_M2": back,
        "BLOCK_N2": back,
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
    if mask is not None and tuple(mask.shape[-2:]) != (len(predicate.row_sigma), len(predicate.key_sigma)):
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
    from torch.nn.functional import pad

    extra_keys = keys - len(predicate.key_sigma)
    extra_rows = rows - len(predicate.row_sigma)
    if extra_keys < 0 or extra_rows < 0:
        raise ValueError("BlockMask extent is smaller than the CQ predicate")
    # Segment -2 matches no row (rows need segment >= 0); padded rows are
    # invalid, so they see exactly the key-0 sentinel like any all-masked row.
    key_sigma = pad(predicate.key_sigma, (0, extra_keys))
    key_segment = pad(predicate.key_segment, (0, extra_keys), value=-2)
    row_sigma = pad(predicate.row_sigma, (0, extra_rows))
    row_segment = pad(predicate.row_segment, (0, extra_rows), value=-1)
    row_is_query = pad(predicate.row_is_query, (0, extra_rows), value=False)
    row_unconditional = pad(predicate.row_unconditional, (0, extra_rows), value=False)
    row_invalid = pad(predicate.row_invalid, (0, extra_rows), value=True)

    def mask_mod(b, h, q_idx, kv_idx):
        same = (row_segment[q_idx] == key_segment[kv_idx]) & (row_segment[q_idx] >= 0)
        strict = key_sigma[kv_idx] < row_sigma[q_idx]
        loose = key_sigma[kv_idx] <= row_sigma[q_idx]
        ok = torch.where(row_is_query[q_idx], strict, loose)
        value = same & ok & ~row_unconditional[q_idx]
        # Same finite sentinel as the dense builder: an all-masked row keeps
        # key 0 so the kernel has a defined softmax; it is zeroed afterwards.
        return value | (row_invalid[q_idx] & (kv_idx == 0))

    mask = create_block_mask(mask_mod, None, None, rows, keys, device=str(device), BLOCK_SIZE=(128, 128), _compile=True)
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
    padded_rows, padded_keys = padded_extent(rows, keys)
    if padded_rows != rows:
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, 0, 0, padded_rows - rows))
    if padded_keys != keys:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 0, 0, padded_keys - keys))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 0, 0, padded_keys - keys))
    query, key, value = (t.permute(1, 2, 0, 3) for t in (q, k, v))
    result = _compiled()(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        block_mask=block_mask(predicate, padded_rows, padded_keys, q.device),
        enable_gqa=q.shape[2] != k.shape[2],
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        kernel_options=options,
    )
    result = result.permute(2, 0, 1, 3)[:rows].flatten(-2).contiguous()
    if valid_rows is not None:
        result = result * valid_rows.transpose(0, 1).unsqueeze(-1).to(result.dtype)
    return result
