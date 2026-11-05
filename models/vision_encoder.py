"""Native Qwen visual encoder and post-merger video features."""
from copy import deepcopy

import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5VisionModel


class _VisionWeights(Qwen3_5VisionModel):
    # Loading the vision part of a full conditional-generation checkpoint is
    # deliberate. Missing vision tensors are still checked strictly below.
    _keys_to_ignore_on_load_unexpected = [r"^(model\.)?language_model\.", r"^lm_head\.", r"^mtp\."]


class Qwen35VideoEncoder(nn.Module):
    # These native methods only need config.vision_config and each other; reuse
    # them verbatim rather than reconstructing the multimodal position rules.
    get_vision_position_ids = Qwen3_5Model.get_vision_position_ids
    get_rope_index = Qwen3_5Model.get_rope_index

    def __init__(self, config, *, visual=None):
        super().__init__()
        self.config = deepcopy(config)
        self.visual = Qwen3_5VisionModel(self.config.vision_config) if visual is None else visual
        if self.visual.config.out_hidden_size != self.config.text_config.hidden_size:
            raise ValueError("Qwen visual merger output must match the language hidden size")

    @classmethod
    def from_pretrained(cls, path, *, config=None, attn_implementation="sdpa", **kwargs):
        config = deepcopy(config) if config is not None else Qwen3_5Config.from_pretrained(path, local_files_only=True)
        kwargs.setdefault("local_files_only", True)
        visual, info = _VisionWeights.from_pretrained(
            path, config=config.vision_config, attn_implementation=attn_implementation,
            key_mapping={r"^model\.visual\.": "", r"^visual\.": ""},
            output_loading_info=True, **kwargs,
        )
        errors = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                  if info[key]}
        if errors:
            raise ValueError(f"Incompatible Qwen3.5 vision weights: {errors}")
        return cls(config, visual=visual).eval()

    def token_counts(self, grid_thw):
        if (not isinstance(grid_thw, torch.Tensor) or grid_thw.ndim != 2 or grid_thw.shape[1] != 3
                or grid_thw.dtype not in (torch.int32, torch.int64)):
            raise ValueError("video_grid_thw must be an integer [videos,3] tensor")
        merge = self.visual.spatial_merge_size
        if bool((grid_thw <= 0).any()) or bool((grid_thw[:, 1:] % merge != 0).any()):
            raise ValueError("Video grids must be positive with H/W divisible by spatial_merge_size")
        return grid_thw.prod(-1) // merge**2

    def forward(self, pixel_values_videos, video_grid_thw):
        counts = self.token_counts(video_grid_thw)
        cfg = self.visual.config
        width = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2
        shape = (int(video_grid_thw.prod(-1).sum()), width)
        if (not isinstance(pixel_values_videos, torch.Tensor) or pixel_values_videos.shape != shape
                or not pixel_values_videos.is_floating_point()):
            raise ValueError(f"pixel_values_videos must contain native flattened patches with shape {shape}")
        if pixel_values_videos.device != self.visual.device:
            raise ValueError("Video patches and the Qwen visual encoder must be on the same device")
        if not counts.numel():
            return ()
        output = self.visual(pixel_values_videos.to(self.visual.dtype), grid_thw=video_grid_thw, return_dict=True)
        return output.pooler_output.split(counts.tolist())
