"""Pluggable frozen latent encoders and stateless per-consumption randomness."""
import hashlib
import os
import time
from pathlib import Path

import torch
from torch import nn

from ..vendor.mar.vae import Encoder


class FrozenKL16(nn.Module):
    def __init__(self, checkpoint, dtype=torch.float32, device='npu', scale=.2325, chunk=2):
        super().__init__()
        self.encoder = Encoder(ch_mult=(1, 1, 2, 2, 4), z_channels=16)
        self.quant_conv = nn.Conv2d(32, 32, 1)
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True, mmap=True)['model']
        state = {key: value for key, value in state.items() if key.startswith(('encoder.', 'quant_conv.'))}
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False).eval().to(device=device, dtype=dtype)
        self.scale, self.chunk = float(scale), int(chunk)
        self.dtype, self.device = dtype, device
        if str(device).startswith('npu') and dtype==torch.float32:
            torch.npu.conv.allow_hf32=False
            torch.npu.matmul.allow_hf32=False

    def norm_silu(self, x, norm):
        import torch_npu
        return torch_npu.npu_group_norm_silu(x,norm.weight,norm.bias,norm.num_groups,eps=norm.eps)[0]

    def resblock(self, x, block):
        h=block.conv1(self.norm_silu(x,block.norm1))
        h=block.conv2(self.norm_silu(h,block.norm2))
        if block.in_channels!=block.out_channels:
            x=block.conv_shortcut(x) if block.use_conv_shortcut else block.nin_shortcut(x)
        return x+h

    def fast_encoder(self, x):
        # The original frozen encoder retains an unused list of every hidden
        # state. Release each level immediately and fuse GroupNorm + SiLU.
        enc=self.encoder;h=enc.conv_in(x)
        for level in range(enc.num_resolutions):
            stage=enc.down[level]
            for i in range(enc.num_res_blocks):
                h=self.resblock(h,stage.block[i])
                if len(stage.attn):h=stage.attn[i](h)
            if level!=enc.num_resolutions-1:h=stage.downsample(h)
        h=self.resblock(h,enc.mid.block_1)
        h=enc.mid.attn_1(h)
        h=self.resblock(h,enc.mid.block_2)
        return enc.conv_out(self.norm_silu(h,enc.norm_out))

    @torch.no_grad()
    def encode(self, pixels):
        moments = []
        for start in range(0, len(pixels), self.chunk):
            x = pixels[start:start+self.chunk].to(self.dtype)
            encoded=self.fast_encoder(x) if x.device.type=='npu' else self.encoder(x)
            moments.append(self.quant_conv(encoded).float())
        result = torch.cat(moments)
        mean, logvar = result.chunk(2, 1)
        # Match the existing scaled posterior contract; do not substitute the
        # posterior mode or confuse the accounting 512 with this 32x32 grid.
        return (mean * self.scale, logvar.clamp(-30, 20).mul(.5).exp() * self.scale)


def consumption_seed(signature, serial):
    key = f"{signature['snapshot']}:{signature['seed']}:{signature['epoch']}:{signature['rank']}:{serial}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & ((1 << 63)-1)


class FakeLatentEncoder(nn.Module):
    """Deterministic test double; exercises actual pixels, kernels and sampling.

    It is deliberately not a substitute for the compute cost or targets of a
    future VAE. The renderer's configured channels and spatial stride define
    its output; the loader contains no hard-coded KL16 dimensions.
    """
    def __init__(self, cfg, device='cpu', work=0):
        super().__init__()
        self.stride, self.work = cfg.patch_size, int(work)
        weight = torch.arange(cfg.latent_channels * 3, dtype=torch.float32).reshape(-1, 3, 1, 1)
        self.register_buffer('projection', (weight.remainder(11) - 5).div(15).to(device))

    def encode(self, pixels):
        from torch.nn import functional as F
        mean = F.conv2d(F.avg_pool2d(pixels.float(), self.stride), self.projection)
        for _ in range(self.work):
            mean = F.avg_pool2d(mean, 3, stride=1, padding=1)
        return mean, torch.full_like(mean, .1)


class DevicePreparer:
    def __init__(self, cfg, vae_path=None, device='npu', vae_dtype=torch.float32, vae_chunk=2,
                 encoder=None, encoder_options=None):
        self.cfg, self.device = cfg, str(device)
        self.stats=dict(batches=0,random_cpu_seconds=0.,batch_transfer_enqueue_seconds=0.,
                        encoder_enqueue_seconds=0.,random_transfer_enqueue_seconds=0.)
        self.encoder_name = encoder or ('kl16' if vae_path else None)
        options = dict(encoder_options or {})
        if self.encoder_name == 'fake':
            self.vae = FakeLatentEncoder(cfg, device, **options)
        elif self.encoder_name == 'moss':
            from .moss_vae import FrozenMOSS
            self.vae=FrozenMOSS(cfg,device,vae_dtype,chunk=vae_chunk,**options)
        elif self.encoder_name == 'kl16':
            if not vae_path:
                raise ValueError('The kl16 encoder requires an explicit checkpoint')
            self.vae = FrozenKL16(vae_path, vae_dtype, device, chunk=vae_chunk, **options)
        elif self.encoder_name is not None:
            import importlib
            module, separator, name = self.encoder_name.partition(':')
            if not separator:
                raise ValueError('Latent encoder must be fake, kl16, or module:factory')
            self.vae = getattr(importlib.import_module(module), name)(
                cfg=cfg, device=device, dtype=vae_dtype, **options)
        else:
            self.vae = None
        if isinstance(self.vae, nn.Module):
            self.vae.requires_grad_(False).eval()

    @torch.no_grad()
    def prepare(self, host_batch, receipt):
        begin=time.perf_counter()
        # Only this private CPU generator supplies sampling/flow randomness.
        # Advancing a prefetch queue never advances the model's RNG state.
        signature = receipt['next_state']['documents']['signature']
        seed = consumption_seed(signature, receipt['serial'])
        generator = torch.Generator(device='cpu').manual_seed(seed)
        if host_batch.generation is not None:
            from .moss_vae import prepare_dynamic
            return prepare_dynamic(self,host_batch,generator)
        n, cfg = len(host_batch.image_indices), self.cfg
        side = cfg.image_size // cfg.patch_size
        randoms = [torch.randn(n, cfg.latent_channels, side, side, generator=generator),
                   torch.randn(n * cfg.flow_draws, cfg.image_tokens, cfg.latent_channels, generator=generator),
                   torch.randn(n * cfg.flow_draws, generator=generator).sigmoid().clamp(1e-5, 1-1e-5)]
        if os.environ.get('PRETRAIN_CPU_NOISE_CAST','1')=='1':
            # Mixed dtype + device Tensor.to is not an asynchronous DMA path
            # on the pinned NPU runtime. Round on CPU before pin/copy instead.
            randoms[1]=randoms[1].to(torch.bfloat16)
        if self.device.startswith('npu'):
            randoms = [x.pin_memory(device=self.device) if x.numel() else x for x in randoms]
        posterior_noise, noise, times = randoms
        self.stats['random_cpu_seconds']+=time.perf_counter()-begin
        begin=time.perf_counter()
        batch = host_batch.to(self.device, non_blocking=True)
        self.stats['batch_transfer_enqueue_seconds']+=time.perf_counter()-begin
        begin=time.perf_counter()
        if n:
            if self.vae is None:
                raise ValueError('Choose a latent encoder explicitly; use fake only for pipeline tests')
            rgb = batch.pixels.float().div_(127.5).sub_(1)
            mean, std = self.vae.encode(rgb)
            expected = (n, cfg.latent_channels, side, side)
            if tuple(mean.shape) != expected or tuple(std.shape) != expected:
                raise ValueError(f'Encoder must return scaled posterior mean/std {expected}')
            sample = mean + std * posterior_noise.to(self.device, non_blocking=True)
            batch.latents = sample.flatten(2).transpose(1, 2).contiguous().to(torch.bfloat16)
            batch.pixels = rgb.to(torch.bfloat16)
        else:
            batch.pixels = batch.pixels.to(torch.bfloat16)
        self.stats['encoder_enqueue_seconds']+=time.perf_counter()-begin
        begin=time.perf_counter()
        batch.noise = noise.to(self.device, dtype=torch.bfloat16, non_blocking=True)
        batch.times = times.to(self.device, non_blocking=True)
        self.stats['random_transfer_enqueue_seconds']+=time.perf_counter()-begin
        self.stats['batches']+=1
        return batch, tuple(randoms)

    def __call__(self, batch, receipt):
        return self.prepare(batch, receipt)[0]
