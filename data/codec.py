"""Wan latent preparation and stateless per-consumption randomness."""
import hashlib
import math
import time

import torch


def consumption_seed(signature, serial):
    key = f"{signature['snapshot']}:{signature['seed']}:{signature['epoch']}:{signature['rank']}:{serial}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & ((1 << 63)-1)


class DevicePreparer:
    def __init__(self, cfg, device='cpu', vae_dtype=torch.float32, encoder=None):
        self.cfg, self.device = cfg, str(device)
        self.stats=dict(batches=0,random_cpu_seconds=0.,batch_transfer_enqueue_seconds=0.,
                        encoder_enqueue_seconds=0.)
        self.encoder_name = encoder
        self.vae = None
        if encoder == 'wan':
            from data.wan_codec import WanVideoCodec
            self.vae = WanVideoCodec.from_pretrained(cfg.wan_vae_assets, device=device, dtype=vae_dtype)
        elif encoder not in (None, 'fake'):
            raise ValueError(f'Unknown Wan encoder: {encoder}')

    @torch.no_grad()
    def prepare(self, host_batch, receipt):
        begin=time.perf_counter()
        # Only this private CPU generator supplies sampling/flow randomness.
        # Advancing a prefetch queue never advances the model's RNG state.
        signature = receipt['next_state']['documents']['signature']
        seed = consumption_seed(signature, receipt['serial'])
        generator = torch.Generator(device='cpu').manual_seed(seed)
        if host_batch.generation is not None:
            return prepare_wan(self,host_batch,generator)
        if host_batch.generation_image_count:
            raise ValueError('Generation targets require a Wan layout')
        batch = host_batch.to(self.device, non_blocking=True)
        self.stats['batch_transfer_enqueue_seconds'] += time.perf_counter()-begin
        self.stats['batches'] += 1
        return batch, ()

    def __call__(self, batch, receipt):
        return self.prepare(batch, receipt)[0]


@torch.no_grad()
def prepare_wan(preparer,host_batch,generator):
    from data.generation import patch_latent
    began=time.perf_counter();layout=host_batch.generation
    noise=torch.randn(layout.rf*sum(layout.lengths),preparer.cfg.latent_channels*4,generator=generator)
    times=torch.randn(layout.rf*len(layout.lengths),generator=generator).sigmoid().clamp(1e-5,1-1e-5)
    frequencies=10000**(-torch.arange(preparer.cfg.wan_freq_dim//2,dtype=torch.float64)/(preparer.cfg.wan_freq_dim//2))
    phase=(times*1000).double()[:,None]*frequencies
    features=torch.cat((phase.cos(),phase.sin()),-1).float()
    randoms=[noise,times,features]
    if preparer.device.startswith(('npu', 'cuda')):randoms=[x.pin_memory(device=preparer.device) for x in randoms]
    preparer.stats['random_cpu_seconds']+=time.perf_counter()-began
    begin=time.perf_counter();batch=host_batch.to(preparer.device,non_blocking=True)
    preparer.stats['batch_transfer_enqueue_seconds']+=time.perf_counter()-begin
    clean=[];begin=time.perf_counter()
    cached=batch.generation.cached_latents or (None,)*len(layout.lengths)
    for pixels,z,grid in zip(batch.generation.pixels,cached,layout.grids):
        if z is None:
            if preparer.encoder_name=='fake':
                from torch.nn import functional as F
                pooled=F.avg_pool2d(pixels[:,::4].permute(1,0,2,3).float(),16).mean(1)
                z=pooled[:,None].expand(-1,preparer.cfg.latent_channels*4,-1,-1).permute(0,2,3,1).reshape(-1,preparer.cfg.latent_channels*4)/255
            elif preparer.vae is None:
                raise ValueError("An uncached Wan target requires its frozen VAE")
            else:z=patch_latent(preparer.vae.encode(pixels.float().div(127.5).sub(1)))
        elif z.ndim==4:z=patch_latent(z)
        if z.shape!=(math.prod(grid),preparer.cfg.latent_channels*4):raise ValueError('Cached Wan latent geometry differs')
        clean.append(z.float())
    batch.latents=torch.cat(clean)
    batch.noise,batch.times,batch.generation.time_features=[x.to(preparer.device,non_blocking=True) for x in randoms]
    batch.generation.pixels=()
    preparer.stats['encoder_enqueue_seconds']+=time.perf_counter()-begin
    preparer.stats['batches']+=1
    return batch,tuple(randoms)
