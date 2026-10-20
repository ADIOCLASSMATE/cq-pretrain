"""Frozen official FLUX.2 posterior, 2x2 packing and learned BN statistics."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import torch
from torch import nn


def flux2_class():
    # CPU preparation hosts have torch_npu installed but no device driver.
    with ExitStack() as imports:
        if 'torch_npu' not in sys.modules:
            for name in ('transformers.utils.import_utils.is_torch_npu_available',
                         'transformers.utils.is_torch_npu_available',
                         'diffusers.utils.import_utils.is_torch_npu_available',
                         'diffusers.utils.is_torch_npu_available'):
                imports.enter_context(patch(name,return_value=False))
        from diffusers import AutoencoderKLFlux2
    return AutoencoderKLFlux2


def pack(x):
    b,c,h,w=x.shape
    return x.reshape(b,c,h//2,2,w//2,2).permute(0,1,3,5,2,4).reshape(b,c*4,h//2,w//2)


class FrozenMOSS(nn.Module):
    def __init__(self,cfg,device='cpu',dtype=torch.float32,chunk=1,**options):
        super().__init__()
        if options:raise ValueError(f'Unknown MOSS VAE options: {options}')
        root=Path(cfg.moss_assets)
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['revision']!='1ceede80badddd6a7812069619edaa5633587192':raise ValueError('Unexpected MOSS VAE revision')
        self.vae=flux2_class().from_pretrained(str(root/'vae'),local_files_only=True,torch_dtype=dtype)
        self.vae.requires_grad_(False).eval().to(device=device,dtype=dtype)
        # Encoding needs no decoder parameters or device memory.
        self.vae.decoder.to('cpu');self.vae.post_quant_conv.to('cpu')
        self.dtype,self.chunk=dtype,chunk
        if str(device).startswith('npu') and dtype==torch.float32:
            torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False

    @torch.no_grad()
    def encode(self,pixels):
        means=[];stds=[]
        mean=self.vae.bn.running_mean.float().view(1,-1,1,1)
        inv_std=(self.vae.bn.running_var.float()+self.vae.config.batch_norm_eps).rsqrt().view(1,-1,1,1)
        for i in range(0,len(pixels),self.chunk):
            posterior=self.vae.encode(pixels[i:i+self.chunk].to(self.dtype)).latent_dist
            means.append((pack(posterior.mean.float())-mean)*inv_std)
            stds.append(pack(posterior.std.float())*inv_std)
        return torch.cat(means),torch.cat(stds)


@torch.no_grad()
def prepare_dynamic(preparer,host_batch,generator):
    begin=time.perf_counter();layout=host_batch.generation
    count=sum(layout.lengths);channels=preparer.cfg.latent_channels
    posterior_noise=torch.randn(count,channels,generator=generator)
    noise=torch.randn(count,channels,generator=generator)
    times=torch.randn(len(layout.lengths),generator=generator).sigmoid().clamp(1e-5,1-1e-5)
    phase=(times*1000).double()[:,None]*10000**(-torch.arange(128,dtype=torch.float64)/128)
    features=torch.cat((phase.cos(),phase.sin()),-1).float().to(torch.bfloat16)
    randoms=[posterior_noise,noise,times,features]
    if preparer.device.startswith('npu'):randoms=[x.pin_memory(device=preparer.device) for x in randoms]
    preparer.stats['random_cpu_seconds']+=time.perf_counter()-begin
    begin=time.perf_counter();batch=host_batch.to(preparer.device,non_blocking=True)
    preparer.stats['batch_transfer_enqueue_seconds']+=time.perf_counter()-begin
    begin=time.perf_counter();means=[None]*len(layout.lengths);stds=[None]*len(layout.lengths)
    for pixels,indices in zip(batch.generation.pixels,layout.groups):
        mean,std=preparer.vae.encode(pixels.float().div(127.5).sub(1))
        for local,index in enumerate(indices):
            h,w=layout.grids[index]
            if tuple(mean[local].shape)!=(channels,h,w):raise ValueError('MOSS VAE geometry mismatch')
            means[index]=mean[local].flatten(1).T;stds[index]=std[local].flatten(1).T
    posterior_noise,noise,times,features=[x.to(preparer.device,non_blocking=True) for x in randoms]
    batch.latents=torch.cat(means)+torch.cat(stds)*posterior_noise
    batch.noise,batch.times=noise,times
    batch.generation.time_features=features
    # Release transferred uint8 targets after encoder kernels are enqueued;
    # allocator stream ownership is recorded by the consumer loader.
    batch.generation.pixels=()
    preparer.stats['encoder_enqueue_seconds']+=time.perf_counter()-begin
    preparer.stats['batches']+=1
    return batch,tuple(randoms)
