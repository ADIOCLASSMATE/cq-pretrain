"""Packed Wan latent geometry shared by image and video execution."""
from dataclasses import dataclass, fields, replace
from itertools import accumulate
import math

import torch


def wan_rope(grid, head_dim):
    coordinates = torch.stack(torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64) for n in grid), indexing="ij"
    ), -1).reshape(-1, 3)
    widths = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))
    angles = torch.cat([
        coordinates[:, i, None] * 10000 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
        for i, d in enumerate(widths)
    ], -1)
    return angles.cos().float()[:, None], angles.sin().float()[:, None]


def patch_latent(z):
    c, t, h, w = z.shape
    return z.reshape(c, t, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(-1, c * 4)


def unpatch_latent(patches, grid, channels=16):
    t, h, w = grid
    return patches.reshape(t, h, w, channels, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(channels, t, h * 2, w * 2)


@dataclass
class WanLayout:
    grids: tuple
    lengths: tuple
    query_counts: tuple
    video_ends: tuple
    text_ends: tuple
    video_sample_indices: torch.Tensor
    condition_indices: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    def to(self, device, non_blocking=False):
        return replace(self, **{
            f.name: v.to(device, non_blocking=non_blocking)
            for f in fields(self) if isinstance(v := getattr(self, f.name), torch.Tensor)
        })

    @classmethod
    def prepare(cls, grids, query_counts, head_dim, condition_width=None):
        grids, counts = tuple(grids), tuple(query_counts)
        lengths = tuple(math.prod(g) for g in grids)
        width = max(counts) if condition_width is None else condition_width
        samples = torch.repeat_interleave(torch.arange(len(grids)), torch.tensor(lengths))
        indices = torch.cat([torch.arange(q) + i * width for i, q in enumerate(counts)])
        cos, sin = zip(*(wan_rope(g, head_dim) for g in grids))
        return cls(grids, lengths, counts, tuple(accumulate(lengths)), tuple(accumulate(counts)),
                   samples, indices, torch.cat(cos), torch.cat(sin))
