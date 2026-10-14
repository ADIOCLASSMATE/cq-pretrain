"""KL16 image codec and the scaled posterior cache contract.

The codec module/checkpoint remain explicit external assets, as in the source
repository. Importing the model does not import a VAE implementation.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


POSTERIOR_FORMAT = "imagenet_kl16_scaled_posterior_v1"
POSTERIOR_LAYOUT = "scaled_mean_then_scaled_std"


def kl16_layout(image_size: int) -> tuple[int, int]:
    if int(image_size) < 16 or int(image_size) % 16:
        raise ValueError("KL16 image_size must be a positive multiple of 16")
    side = int(image_size) // 16
    return side, side * side


class KL16Codec:
    """Encode [B,3,H,W] pixels in [0,1]; decode [B,N,16] latents.

    A supplied ``vae`` must provide ``encode`` returning mean/std and ``decode``
    returning [-1,1] pixels. Injection supports small CPU integration tests and
    avoids a global dependency on the original machine's VAE checkout.
    """

    def __init__(self, vae, *, scaling_factor: float = 0.2325):
        if not math.isfinite(scaling_factor) or scaling_factor <= 0:
            raise ValueError("VAE scaling_factor must be finite and positive")
        self.vae = vae.eval()
        self.scaling_factor = float(scaling_factor)
        for parameter in vae.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, module_root, checkpoint, *, device="cpu", dtype=torch.float32,
                        scaling_factor=0.2325):
        module_path = Path(module_root) / "models" / "vae.py"
        checkpoint = Path(checkpoint)
        if not module_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"KL16 VAE assets missing: {module_path}, {checkpoint}")
        spec = importlib.util.spec_from_file_location("umm_kl16_vae", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load KL16 VAE: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        vae = module.AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=str(checkpoint))
        return cls(vae.to(device=device, dtype=dtype), scaling_factor=scaling_factor)

    def _device_dtype(self):
        tensor = next(iter(self.vae.parameters()), None)
        if tensor is None:
            tensor = next(iter(self.vae.buffers()), torch.empty((), dtype=torch.float32))
        return tensor.device, tensor.dtype

    @torch.no_grad()
    def encode_posterior(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[2] != images.shape[3]:
            raise ValueError("KL16 expects square images with shape [B,3,H,W]")
        side, tokens = kl16_layout(images.shape[-1])
        if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
            raise ValueError("codec pixels must be finite and in [0,1]")
        device, dtype = self._device_dtype()
        posterior = self.vae.encode(images.to(device=device, dtype=dtype).mul(2).sub(1))
        stats = torch.cat((posterior.mean, posterior.std), dim=1).mul(self.scaling_factor)
        if stats.shape != (images.shape[0], 32, side, side):
            raise ValueError(f"unexpected KL16 posterior shape: {tuple(stats.shape)}")
        stats = stats.permute(0, 2, 3, 1).reshape(images.shape[0], tokens, 32).contiguous()
        self.validate_posterior(stats)
        return stats

    @staticmethod
    def validate_posterior(stats):
        if stats.ndim < 2 or stats.shape[-1] != 32:
            raise ValueError("posterior must end in [N,32]: scaled mean then scaled std")
        if not bool(torch.isfinite(stats).all()) or bool((stats[..., 16:] < 0).any()):
            raise ValueError("posterior contains nonfinite values or negative std")

    @staticmethod
    def sample_posterior(stats, *, generator=None, noise=None):
        KL16Codec.validate_posterior(stats)
        mean, std = stats.float().chunk(2, dim=-1)
        if noise is None:
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        if noise.shape != mean.shape:
            raise ValueError("posterior noise shape mismatch")
        return mean + std * noise.to(mean)

    @torch.no_grad()
    def encode(self, images, *, generator=None):
        return self.sample_posterior(self.encode_posterior(images), generator=generator)

    @torch.no_grad()
    def decode(self, latents):
        if latents.ndim != 3 or latents.shape[-1] != 16:
            raise ValueError("KL16 latent shape must be [B,N,16]")
        side = math.isqrt(latents.shape[1])
        if side * side != latents.shape[1] or side < 1:
            raise ValueError("KL16 tokens must form a nonempty square spatial grid")
        if not bool(torch.isfinite(latents).all()):
            raise FloatingPointError("nonfinite generated latents")
        device, dtype = self._device_dtype()
        grid = latents.reshape(-1, side, side, 16).permute(0, 3, 1, 2)
        images = self.vae.decode(grid.to(device=device, dtype=dtype) / self.scaling_factor)
        if not bool(torch.isfinite(images).all()):
            raise FloatingPointError("nonfinite decoded image")
        return images.float().clamp(-1, 1).add(1).div(2)
