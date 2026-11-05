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
