"""Aspect-preserving targets and CPU-built ragged Wan generation geometry."""
from ops.wan_layout import WanLayout, wan_rope, patch_latent, unpatch_latent
from dataclasses import dataclass, fields
import math
from types import SimpleNamespace
import numpy as np
import torch


def target_image(image, max_pixels=1048576):
    """Resize proportionally only when necessary, then edge-pad by <16 pixels.

    No cropping, stretching, square buckets, or enlargement of small images.
    max_pixels=0 means native resolution (unbounded memory/compute).
    """
    from PIL import Image
    width, height = image.size
    def shape(scale):
        w, h = max(1, int(width*scale)), max(1, int(height*scale))
        return w, h, math.ceil(w/16)*16, math.ceil(h/16)*16
    scale = min(1., math.sqrt(max_pixels/(width*height))) if max_pixels else 1.
    if max_pixels and shape(scale)[2]*shape(scale)[3] > max_pixels:
        lo, hi = 0., scale
        for _ in range(40):
            mid = (lo+hi)/2
            _, _, w, h = shape(mid)
            if w*h <= max_pixels:lo = mid
            else:hi = mid
        scale = lo
    w, h, canvas_w, canvas_h = shape(scale)
    if (w,h) != image.size:image = image.resize((w,h), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.uint8)
    pad_h, pad_w = canvas_h-h, canvas_w-w
    return np.pad(rgb, ((pad_h//2,pad_h-pad_h//2),(pad_w//2,pad_w-pad_w//2),(0,0)), mode='edge').copy()


@dataclass
class WanGenerationInputs:
    pixels: tuple
    grids: tuple
    lengths: tuple
    query_counts: tuple
    video_ends: tuple
    text_ends: tuple
    video_sample_indices: torch.Tensor
    condition_indices: torch.Tensor
    loss_weights: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    time_features: torch.Tensor
    cached_latents: tuple = ()
    cache_keys: tuple = ()
    microbatches: tuple = ()

    def _map(self,fn):
        return type(self)(**{f.name:(fn(v) if isinstance(v,torch.Tensor) else
            tuple(fn(x) if isinstance(x,torch.Tensor) else x for x in v)
            if f.name in ('pixels','cached_latents') else
            tuple(x._map(fn) for x in v) if f.name=='microbatches' else v)
            for f in fields(self) for v in (getattr(self,f.name),)})

    def to(self,device,non_blocking=False):
        return self._map(lambda x:x.to(device,non_blocking=non_blocking))

    def pin_memory(self,device):
        return self._map(lambda x:x.pin_memory(device=device) if x.numel() else x)

    @classmethod
    def prepare(cls,images,queries,head_dim=128,token_budget=0,max_images=0,cached_latents=()):
        # Input is uint8 [H,W,3] or [T,H,W,3]; all output paths use [C,T,H,W].
        arrays=[x[None] if x.ndim==3 else x for x in images]
        pixels=tuple(torch.from_numpy(np.moveaxis(x,-1,0).copy()) for x in arrays)
        grids=tuple(((x.shape[0]-1)//4+1,x.shape[1]//16,x.shape[2]//16) for x in arrays)
        if any((x.shape[0]-1)%4 or x.shape[1]%16 or x.shape[2]%16 for x in arrays):
            raise ValueError('Wan targets require 4k+1 frames and spatial multiples of 16')
        counts=tuple([queries]*len(images) if isinstance(queries,int) else queries)
        geometry=WanLayout.prepare(grids,counts,head_dim)
        lengths=geometry.lengths
        weights=torch.cat([torch.full((n,),1/(len(images)*n)) for n in lengths])
        out=cls(pixels=pixels,loss_weights=weights,time_features=torch.empty(0),
                cached_latents=tuple(cached_latents),**vars(geometry))
        ranges=[];start=used=0
        for i,n in enumerate(lengths):
            if token_budget and n>token_budget:raise ValueError(f'Visual target needs {n} Wan tokens, above budget {token_budget}')
            if i>start and ((token_budget and used+n>token_budget) or (max_images and i-start>=max_images)):
                ranges.append((start,i));start=i;used=0
            used+=n
        ranges.append((start,len(images)))
        if len(ranges)>1:out.microbatches=tuple(WanGenerationSlice.prepare(out,a,b) for a,b in ranges)
        return out


@dataclass
class WanGenerationSlice:
    start: int
    stop: int
    latent_start: int
    latent_stop: int
    layout: WanGenerationInputs

    def _map(self,fn):
        return type(self)(self.start,self.stop,self.latent_start,self.latent_stop,self.layout._map(fn))

    @classmethod
    def prepare(cls,p,a,b):
        lo=sum(p.lengths[:a]);hi=sum(p.lengths[:b]);counts=p.query_counts[a:b]
        # Keep the parent padded condition width: renderer chunks slice samples,
        # and never pad/synchronize again on the device.
        condition=torch.cat([torch.arange(q)+i*max(p.query_counts) for i,q in enumerate(counts)])
        layout=WanGenerationInputs((),p.grids[a:b],p.lengths[a:b],counts,
            tuple(np.cumsum(p.lengths[a:b]).tolist()),tuple(np.cumsum(counts).tolist()),
            p.video_sample_indices[lo:hi]-a,condition,p.loss_weights[lo:hi],p.cos[lo:hi],p.sin[lo:hi],torch.empty(0))
        return cls(a,b,lo,hi,layout)

    def bind(self,batch):
        from dataclasses import replace
        layout=replace(self.layout,time_features=batch.generation.time_features[self.start:self.stop])
        return SimpleNamespace(generation=layout,latents=batch.latents[self.latent_start:self.latent_stop],
            noise=batch.noise[self.latent_start:self.latent_stop],times=batch.times[self.start:self.stop])
