"""Exact CPU query tiling and reusable device masks."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class AttentionPlan:
    tiles: tuple
    mask_pointer: int
    mask_version: int
    shape: tuple
    owner: object = None
    tile_masks: tuple = ()
    key_starts: tuple = ()

    def matches(self, mask):
        return self.owner is mask and mask._version == self.mask_version and tuple(mask.shape) == self.shape

    def bind(self, mask):
        if tuple(mask.shape) != self.shape or mask.dtype != torch.bool:
            raise ValueError("Bind a plan only to a same-shape copy of its original mask")
        starts = self.key_starts or (0,) * len(self.tiles)
        masks = tuple(
            mask[:, :, start:end, key_start:key_end].contiguous()
            for (start, end, key_end), key_start in zip(self.tiles, starts)
        )
        # Reading a just-enqueued NPU allocation's pointer can drain the host
        # launch queue. Python ownership plus version/shape is a stricter and
        # sufficient identity check; no device pointer is needed in forward.
        pointer = mask.data_ptr() if mask.device.type == "cpu" else 0
        return AttentionPlan(
            self.tiles,
            pointer,
            mask._version,
            tuple(mask.shape),
            owner=mask,
            tile_masks=masks,
            key_starts=self.key_starts,
        )


def plan_attention(mask, chunk=1024, trim_key_start=False):
    """Data-side CPU preparation, valid for arbitrary sigma/document patterns.

    All batches and heads share a conservative bound. An irregular mask may
    yield full key lengths; it is never approximated by a causal/band mask.
    """
    if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.ndim != 4 or chunk < 1:
        raise ValueError("Prepare the plan on a CPU boolean [B,H,Q,K] mask")
    tiles = []
    key_starts = []
    for start in range(0, mask.shape[-2], chunk):
        end = min(start + chunk, mask.shape[-2])
        visible = (~mask[:, :, start:end]).any(dim=(0, 1, 2)).nonzero().flatten()
        key_end = int(visible[-1]) + 1 if visible.numel() else 1
        key_starts.append(int(visible[0]) if trim_key_start and visible.numel() else 0)
        tiles.append((start, end, key_end))
    return AttentionPlan(
        tuple(tiles),
        mask.data_ptr(),
        mask._version,
        tuple(mask.shape),
        owner=mask,
        key_starts=tuple(key_starts) if trim_key_start else (),
    )
