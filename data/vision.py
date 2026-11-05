"""Native image preprocessing and prepared vision geometry."""

from dataclasses import dataclass, fields
import math
import torch
from utils.assets import asset_path, vision_config


def native_processor(path):
    import sys
    from contextlib import ExitStack
    from unittest.mock import patch

    with ExitStack() as imports:
        if "torch_npu" not in sys.modules:
            for name in (
                "transformers.utils.import_utils.is_torch_npu_available",
                "transformers.utils.is_torch_npu_available",
            ):
                imports.enter_context(patch(name, return_value=False))
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(str(asset_path(path)), local_files_only=True)


@dataclass
class VisionInputs:
    pixel_values: torch.Tensor
    content_indices: torch.Tensor
    pos_indices: torch.Tensor
    pos_weights: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    grid_thw: tuple
    lengths: tuple
    ends: tuple
    features: torch.Tensor | None = None
    online: object = None
    cache_keys: tuple = ()

    def to(self, device, non_blocking=False):
        return VisionInputs(
            **{
                f.name: (
                    getattr(self, f.name).to(device, non_blocking=non_blocking)
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else self.online.to(device, non_blocking=non_blocking)
                    if f.name == "online" and self.online is not None
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )

    def pin_memory(self, device):
        return VisionInputs(
            **{
                f.name: (
                    getattr(self, f.name).pin_memory(device=device)
                    if isinstance(getattr(self, f.name), torch.Tensor) and getattr(self, f.name).numel()
                    else self.online.pin_memory(device)
                    if f.name == "online" and self.online is not None
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )

    @classmethod
    def prepare(cls, pixel_values, grid_thw, content_indices, cfg):
        """Exact Qwen bilinear absolute positions + 2D rotary, in merge order."""
        if pixel_values.device.type != "cpu":
            raise ValueError("Prepare vision geometry on CPU before transferring the batch")
        grid = tuple(tuple(map(int, g)) for g in grid_thw)
        side = math.isqrt(cfg["num_position_embeddings"])
        if side * side != cfg["num_position_embeddings"]:
            raise ValueError("The pretrained absolute embedding grid must be square")
        merge = cfg["spatial_merge_size"]
        rotary_dim = cfg["hidden_size"] // cfg["num_heads"] // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        all_indices, all_weights, all_angles, lengths = [], [], [], []
        for t, h, w in grid:
            if t < 1 or h < merge or w < merge or h % merge or w % merge:
                raise ValueError("Visual grids require positive time and spatial multiples of merge_size")
            order = (
                torch.arange(h * w)
                .reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            row, col = order // w, order % w
            y, x = torch.linspace(0, side - 1, h), torch.linspace(0, side - 1, w)
            y0, x0 = y.long(), x.long()
            y1, x1 = (y0 + 1).clamp(max=side - 1), (x0 + 1).clamp(max=side - 1)
            dy, dx = y - y0, x - x0
            indices = torch.stack(
                [
                    (a[:, None] * side + b[None, :]).flatten()[order]
                    for a, b in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
                ]
            )
            weights = torch.stack(
                [
                    (a[:, None] * b[None, :]).flatten()[order]
                    for a, b in ((1 - dy, 1 - dx), (1 - dy, dx), (dy, 1 - dx), (dy, dx))
                ]
            )
            angle = torch.stack((row, col), dim=-1).float()[..., None] * inv_freq
            angle = angle.flatten(1)
            all_indices.append(indices.repeat(1, t))
            all_weights.append(weights.repeat(1, t))
            all_angles.append(torch.cat((angle, angle), dim=-1).repeat(t, 1))
            lengths.extend([h * w] * t)
        if not grid:
            raise ValueError("An empty image batch does not need a VisionInputs payload")
        if sum(lengths) != len(pixel_values) or sum(lengths) // merge**2 != content_indices.numel():
            raise ValueError("Pixel patches, merged image tokens and content indices disagree")
        angles = torch.cat(all_angles)
        ends = tuple(torch.tensor(lengths).cumsum(0).tolist())
        # CPU data workers build reproducible FP32 tables via FP64 trig. Some
        # CPU vector-math dispatches in the pinned torch build lose ~1e-4 on
        # FP32 cos; pretrained vision activations amplify that error.
        trig = angles.double()
        return cls(
            pixel_values,
            content_indices,
            torch.cat(all_indices, 1),
            torch.cat(all_weights, 1),
            trig.cos().float(),
            trig.sin().float(),
            grid,
            tuple(lengths),
            ends,
        )


class NativeImageProcessor:
    """Use the official fast processor, with an explicit packed-token budget."""

    def __init__(self, cfg, sequence_length):
        # Transformers 4.57 probes installed torch_npu while importing its CPU
        # image utilities, even on loader/test hosts without an Ascend driver.
        # Suppress only that optional probe during the CPU-only import. Do not
        # alter the installed package or a live NPU process's capability checks.
        import sys
        from contextlib import ExitStack
        from unittest.mock import patch

        with ExitStack() as imports:
            if "torch_npu" not in sys.modules:
                for name in (
                    "transformers.utils.import_utils.is_torch_npu_available",
                    "transformers.utils.is_torch_npu_available",
                ):
                    imports.enter_context(patch(name, return_value=False))
            from transformers import Qwen2VLImageProcessorFast
        self.config = vision_config(cfg.vision_assets)
        # One merged token covers 32x32 pixels. Reserve BOS/BOI/EOI/EOS.
        maximum = min(cfg.vision_max_pixels, (sequence_length - 4) * 32 * 32)
        if maximum < cfg.vision_min_pixels:
            raise ValueError("Sequence length is too short for vision_min_pixels plus boundaries")
        self.processor = Qwen2VLImageProcessorFast.from_pretrained(
            str(asset_path(cfg.vision_assets)),
            local_files_only=True,
            min_pixels=cfg.vision_min_pixels,
            max_pixels=maximum,
        )
        if (self.processor.patch_size, self.processor.temporal_patch_size, self.processor.merge_size) != (
            16,
            2,
            2,
        ):
            raise ValueError("Processor geometry differs from Qwen3.5-0.8B")
        self.target_size = cfg.image_size
        self.generation_dynamic = cfg.renderer in ("moss", "wan")
        self.generation_max_pixels = cfg.generation_max_pixels
        self.pad_color = tuple(round(x * 255) for x in self.processor.image_mean)

    def supported_aspect(self, image):
        """Retain extreme strips that the official smart_resize rejects.

        Pad only the narrow side to its documented 200:1 limit. Ordinary
        inputs are passed through unchanged; generation keeps the original.
        """
        from PIL import ImageOps

        w, h = image.size
        if max(w, h) <= 200 * min(w, h):
            return image
        width, height = max(w, math.ceil(h / 200)), max(h, math.ceil(w / 200))
        left, top = (width - w) // 2, (height - h) // 2
        return ImageOps.expand(image, (left, top, width - w - left, height - h - top), fill=self.pad_color)

    def __call__(self, native_rgb):
        import numpy as np
        from PIL import Image, ImageOps

        original = Image.fromarray(native_rgb)
        result = self.processor(images=self.supported_aspect(original), return_tensors="pt", device="cpu")
        grid = tuple(map(int, result["image_grid_thw"][0].tolist()))
        # Both paths start from the original image. MOSS retains the full
        # image with proportional resize and edge padding; only the legacy
        # square KL16 target below uses a center crop.
        if self.generation_dynamic:
            from data.generation import target_image

            target = target_image(original, self.generation_max_pixels)
        else:
            target = ImageOps.fit(
                original, (self.target_size, self.target_size), Image.Resampling.LANCZOS, centering=(0.5, 0.5)
            )
        return np.asarray(target, dtype=np.uint8).copy(), result["pixel_values"], grid
