"""Video input/sampling adapter for main's packed Wan execution layers."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from models.wan_layers import Wan21Generator as WanLayers
from ops.wan_layout import WanLayout, patch_latent, unpatch_latent


class Wan21Generator(WanLayers, ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _keep_in_fp32_modules = ["time_embedding", "time_projection"]

    @register_to_config
    def __init__(
        self, condition_dim: int, model_type="t2v", patch_size=(1, 2, 2),
        text_len=512, in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256,
        text_dim=4096, out_dim=16, num_heads=16, num_layers=32,
        window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6,
        attention_backend="auto", gradient_checkpointing=False,
    ):
        if model_type != "t2v":
            raise ValueError("Wan21Generator requires the T2V parameter layout")
        if tuple(window_size) != (-1, -1):
            raise ValueError("Wan uses global, bidirectional clip attention")
        if tuple(patch_size) != (1, 2, 2) or in_dim != out_dim or not (qk_norm and cross_attn_norm):
            raise ValueError("Use native Wan2.1 T2V patching and normalization")
        if attention_backend not in {"auto", "sdpa", "npu"}:
            raise ValueError("attention_backend must be auto, sdpa or npu")
        cfg = SimpleNamespace(width=condition_dim, dit_width=dim, dit_ffn=ffn_dim,
            dit_heads=num_heads, dit_layers=num_layers, latent_channels=in_dim,
            wan_freq_dim=freq_dim, norm_eps=eps, fused=True,
            checkpoint_dit=gradient_checkpointing, checkpoint_dit_layers=0)
        super().__init__(cfg)
        self.condition_dim, self.in_dim, self.dim = condition_dim, in_dim, dim
        self.freq_dim, self.num_heads = freq_dim, num_heads
        self.patch_size = (1, 2, 2)
        self.attention_backend = attention_backend
        self.gradient_checkpointing = gradient_checkpointing

    def enable_gradient_checkpointing(self, **kwargs):
        self.gradient_checkpointing = self.cfg.checkpoint_dit = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = self.cfg.checkpoint_dit = False

    def _layout(self, latents, condition, mask):
        grids = tuple((x.shape[1], x.shape[2] // 2, x.shape[3] // 2) for x in latents)
        layout = WanLayout.prepare(grids, mask.sum(-1).tolist(), self.dim // self.num_heads, condition.shape[1])
        layout.condition_indices = mask.flatten().nonzero().flatten()
        return layout.to(condition.device, non_blocking=True)

    def _predict_videos(self, latents, times, condition, layout, projected=None, cached_kv=None):
        half = self.freq_dim // 2
        frequencies = 10000.0 ** (-torch.arange(half, device=times.device, dtype=torch.float32) / half)
        angles = times.float()[:, None] * frequencies[None]
        time_features = torch.cat((angles.cos(), angles.sin()), -1)
        patches = torch.cat([patch_latent(x) for x in latents])
        prediction = self.predict(patches, condition, time_features, layout, projected, cached_kv)
        return [unpatch_latent(x, grid, self.in_dim) for x, grid in zip(prediction.split(layout.lengths), layout.grids)]

    def forward(self, noisy_latents, timesteps, condition, *, condition_mask=None):
        mask = self._validate_inputs(noisy_latents, timesteps, condition, condition_mask)
        layout = self._layout(noisy_latents, condition, mask)
        return self._predict_videos(noisy_latents, timesteps, condition, layout)

    def _validate_inputs(self, latents, times, condition, mask):
        if not isinstance(latents, (list, tuple)) or not latents:
            raise ValueError("noisy_latents must be a nonempty list of [C,T,H,W] tensors")
        count, device = len(latents), self.patch_embedding.weight.device
        for latent in latents:
            if (not isinstance(latent, torch.Tensor) or latent.ndim != 4
                    or latent.shape[0] != self.in_dim or not latent.is_floating_point()):
                raise ValueError("Each noisy latent must be floating [in_dim,T,H,W]")
            if any(size <= 0 or size % patch for size, patch in zip(latent.shape[1:], self.patch_size)):
                raise ValueError("Latent T/H/W must be positive and divisible by patch_size")
            if latent.device != device:
                raise ValueError("Latents and generator must be on the same device")
        if not isinstance(times, torch.Tensor) or times.shape != (count,):
            raise ValueError("timesteps must be finite with one native Wan timestep per video")
        if (not isinstance(condition, torch.Tensor) or condition.ndim != 3
                or condition.shape[0] != count or condition.shape[1] < 1
                or condition.shape[2] != self.condition_dim or not condition.is_floating_point()):
            raise ValueError("condition must be floating [videos,L,condition_dim] with L > 0")
        if condition.device != device:
            raise ValueError("Condition and generator must be on the same device")
        if mask is None:
            mask = torch.ones(condition.shape[:2], device=device, dtype=torch.bool)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != condition.shape[:2]:
            raise ValueError("condition_mask must be boolean [videos,L], True for valid tokens")
        mask = mask.cpu()
        if not bool(mask.any(-1).all()):
            raise ValueError("Every video needs at least one valid condition token, including CFG branches")
        return mask

    @torch.no_grad()
    def sample(self, condition, *, condition_mask=None, latent_shapes=None, initial_noise=None,
               steps=20, solver="euler", cfg=1.0, unconditional=None, unconditional_mask=None,
               time_shift=1.0, timestep_scale=1000.0, rng=None):
        """Integrate dx/dsigma = noise-clean from sigma=1 to 0.

        Unlike the legacy image sampler, Wan predicts noise minus clean. The
        model receives sigma*timestep_scale, and the integration step is negative.
        time_shift explicitly controls the monotone shifted sigma schedule.
        """
        if not isinstance(steps, int) or steps < 1 or solver not in {"euler", "heun"}:
            raise ValueError("Use positive integer steps and Euler or Heun")
        if (not all(math.isfinite(float(x)) for x in (cfg, time_shift, timestep_scale))
                or time_shift <= 0 or timestep_scale <= 0):
            raise ValueError("Sampling scales must be finite; time scales must be positive")
        if initial_noise is None:
            if latent_shapes is None or len(latent_shapes) != condition.shape[0]:
                raise ValueError("Supply one Wan latent shape per target video")
            latents = [torch.randn(tuple(shape), device=condition.device, dtype=torch.float32, generator=rng)
                       for shape in latent_shapes]
        else:
            if not isinstance(initial_noise, (list, tuple)) or len(initial_noise) != condition.shape[0]:
                raise ValueError("initial_noise must contain one tensor per target video")
            if latent_shapes is not None and [tuple(x.shape) for x in initial_noise] != [tuple(x) for x in latent_shapes]:
                raise ValueError("initial_noise shapes do not match latent_shapes")
            latents = [x.to(device=condition.device, dtype=torch.float32).clone() for x in initial_noise]
        times = torch.linspace(1., 0., steps + 1, device=condition.device, dtype=torch.float32)
        times = time_shift * times / (1 + (time_shift - 1) * times)
        mask = self._validate_inputs(latents, times[0].expand(len(latents)) * timestep_scale, condition, condition_mask)
        if cfg != 1.0:
            if unconditional is None or unconditional.shape != condition.shape:
                raise ValueError("CFG needs aligned conditional and unconditional query sequences")
            unconditional_mask = self._validate_inputs(
                latents, times[0].expand(len(latents)), unconditional, unconditional_mask)

        contexts = condition if cfg == 1.0 else torch.cat((condition, unconditional))
        masks = mask if cfg == 1.0 else torch.cat((mask, unconditional_mask))
        layout = self._layout(latents if cfg == 1.0 else latents + latents, contexts, masks)
        projected = self.project_conditions(contexts, layout)
        cached_kv = self.cache_condition_kv(projected)

        def velocity(values, sigma):
            inputs = values if cfg == 1.0 else values + values
            timestep = (sigma * timestep_scale).expand(len(inputs))
            predictions = self._predict_videos(inputs, timestep, contexts, layout, projected, cached_kv)
            if cfg == 1.0:
                return predictions
            return [u + cfg * (c - u) for c, u in zip(predictions[:len(values)], predictions[len(values):])]

        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            for i in range(steps):
                dt = times[i + 1] - times[i]
                first = velocity(latents, times[i])
                predictor = [x + dt * v.float() for x, v in zip(latents, first)]
                if solver == "heun":
                    second = velocity(predictor, times[i + 1])
                    latents = [x + dt * 0.5 * (v.float() + w.float())
                               for x, v, w in zip(latents, first, second)]
                else:
                    latents = predictor
            return latents
        finally:
            for module, mode in modes:
                module.training = mode

    @classmethod
    def from_wan_pretrained(cls, path, *, condition_dim, torch_dtype=torch.float32,
                            attention_backend="auto", gradient_checkpointing=False,
                            local_files_only=True, **architecture):
        settings = {k: v for k, v in json.loads((Path(path) / "config.json").read_text()).items()
                    if not k.startswith("_")}
        settings.update(architecture, condition_dim=condition_dim, attention_backend=attention_backend,
                        gradient_checkpointing=gradient_checkpointing)
        model = cls(**settings)
        model.load_pretrained(path)
        from models.qwen35_layers import cast_for_training
        if torch_dtype == torch.bfloat16:
            cast_for_training(model)
        else:
            model.to(dtype=torch_dtype)
        return model.eval()

    @classmethod
    def from_pretrained(cls, path, *, torch_dtype=torch.float32, local_files_only=True,
                        output_loading_info=False):
        """Strict local loading without Accelerate rewriting TP parameter metadata."""
        from safetensors import safe_open
        from models.qwen35_layers import cast_for_training
        root = Path(path)
        model = cls.from_config(json.loads((root / "config.json").read_text()))
        if torch_dtype == torch.bfloat16:
            cast_for_training(model)
        else:
            model.to(dtype=torch_dtype)
        index = root / "diffusion_pytorch_model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["diffusion_pytorch_model.safetensors"]
        targets, seen = model.state_dict(), set()
        for filename in files:
            with safe_open(str(root / filename), framework="pt", device="cpu") as source, torch.no_grad():
                for name in source.keys():
                    if name not in targets or source.get_slice(name).get_shape() != list(targets[name].shape):
                        raise ValueError(f"Incomplete adapted generator checkpoint: incompatible {name}")
                    targets[name].copy_(source.get_tensor(name))
                    seen.add(name)
        if seen != set(targets):
            raise ValueError(f"Incomplete adapted generator checkpoint: missing {sorted(set(targets) - seen)}")
        info = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
        model.eval()
        return (model, info) if output_loading_info else model
