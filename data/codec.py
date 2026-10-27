"""Frozen Wan2.1 VAE; checkpoint normalization is part of the cache contract."""

from __future__ import annotations
from pathlib import Path
import torch


class WanVideoCodec:
    def __init__(self, vae):
        self.vae = vae.eval().requires_grad_(False)
        self.mean = torch.tensor(vae.config.latents_mean).view(16, 1, 1, 1)
        self.std = torch.tensor(vae.config.latents_std).view(16, 1, 1, 1)

    @classmethod
    def from_pretrained(cls, path, *, device="cpu", dtype=torch.float32):
        from diffusers import AutoencoderKLWan
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

        path = Path(path)
        if path.is_dir() and (path / "Wan2.1_VAE.pth").exists():
            path = path / "Wan2.1_VAE.pth"
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            # Reuse Diffusers' official key conversion, with strict local loading.
            state = convert_wan_vae_to_diffusers(state)
            with torch.device("meta"):
                vae = AutoencoderKLWan()
            vae.load_state_dict(state, strict=True, assign=True)
        else:
            vae = AutoencoderKLWan.from_pretrained(
                path, local_files_only=True, torch_dtype=dtype
            )
        return cls(vae.to(device=device, dtype=dtype))

    @torch.no_grad()
    def encode(self, rgb):
        if (
            rgb.ndim != 4
            or rgb.shape[0] != 3
            or (rgb.shape[1] - 1) % 4
            or any(x % 16 for x in rgb.shape[2:])
        ):
            raise ValueError("Wan RGB must be [3,4k+1,H,W] with H/W divisible by 16")
        if not bool(torch.isfinite(rgb).all()) or bool((rgb.abs() > 1.0001).any()):
            raise ValueError("Wan RGB must be finite in [-1,1]")
        parameter = next(self.vae.parameters())
        # Native Wan uses the posterior mean, not a fresh posterior sample.
        with torch.autocast(parameter.device.type, enabled=False):
            z = self.vae.encode(rgb[None].to(parameter)).latent_dist.mode()[0].float()
        return (z - self.mean.to(z)) / self.std.to(z)

    @torch.no_grad()
    def decode(self, latents):
        parameter = next(self.vae.parameters())
        videos = []
        for z in latents:
            raw = z.float()
            raw = raw * self.std.to(raw) + self.mean.to(raw)
            with torch.autocast(parameter.device.type, enabled=False):
                videos.append(
                    self.vae.decode(raw[None].to(parameter))
                    .sample[0]
                    .float()
                    .clamp(-1, 1)
                )
        return videos
