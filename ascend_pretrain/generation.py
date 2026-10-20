"""Aspect-preserving targets and CPU-built ragged MOSS generation geometry."""
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


def moss_rope(height, width, text_length, head_dim=128, base=10000.):
    # Match MOSS's FP32 phase and sin/cos, before casting to activation dtype.
    bands = 1/(base**(torch.arange(head_dim//8,dtype=torch.float32)/(head_dim//8)))
    h,w = torch.meshgrid(torch.arange(height),torch.arange(width),indexing='ij')
    image = torch.stack((torch.zeros_like(h),torch.full_like(h,text_length),h,w),-1).flatten(0,1).float()
    image[:,2] -= (height-1)/2; image[:,3] -= (width-1)/2
    text = torch.zeros(text_length,4);text[:,1] = torch.arange(text_length)
    phase = (torch.cat((image,text))[:,:,None]*bands).flatten(1)
    return torch.cat((phase.cos(),phase.cos()),-1)[:,None],torch.cat((phase.sin(),phase.sin()),-1)[:,None]


@dataclass
class GenerationInputs:
    pixels: tuple
    groups: tuple
    grids: tuple
    lengths: tuple
    ends: tuple
    text_ends: tuple
    video_ends: tuple
    video_rows: torch.Tensor
    video_indices: torch.Tensor
    text_indices: torch.Tensor
    sample_indices: torch.Tensor
    video_sample_indices: torch.Tensor
    loss_weights: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    time_features: torch.Tensor
    microbatches: tuple = ()

    def _map(self, fn):
        values = {}
        for f in fields(self):
            value = getattr(self,f.name)
            values[f.name] = fn(value) if isinstance(value,torch.Tensor) else (
                tuple(fn(x) for x in value) if f.name=='pixels' else
                tuple(x._map(fn) for x in value) if f.name=='microbatches' else value)
        return type(self)(**values)

    def to(self,device,non_blocking=False):
        return self._map(lambda x:x.to(device,non_blocking=non_blocking))

    def pin_memory(self,device):
        return self._map(lambda x:x.pin_memory(device=device) if x.numel() else x)

    @classmethod
    def prepare(cls,images,queries,head_dim=128,token_budget=0,max_images=0):
        if not images or queries < 1 or token_budget < 0 or max_images < 0:
            raise ValueError('Require images, positive CQ count and nonnegative DiT budgets')
        grouped={}
        for i,x in enumerate(images):grouped.setdefault(x.shape[:2],[]).append(i)
        pixels=tuple(torch.from_numpy(np.stack([images[i] for i in group]).transpose(0,3,1,2).copy()) for group in grouped.values())
        groups=tuple(tuple(g) for g in grouped.values())
        grids=tuple((x.shape[0]//16,x.shape[1]//16) for x in images)
        lengths=tuple(h*w for h,w in grids)
        ends=[]; video=[]; text=[];samples=[];vsamples=[];weights=[];cos=[];sin=[];cursor=0
        for i,((h,w),n) in enumerate(zip(grids,lengths)):
            # Match MOSS: noisy image tokens first, then CQ conditions.
            video.extend(range(cursor,cursor+n));text.extend(range(cursor+n,cursor+n+queries))
            samples.extend([i]*(n+queries));vsamples.extend([i]*n)
            weights.extend([1/(len(images)*n)]*n)
            c,s=moss_rope(h,w,queries,head_dim)
            cos.append(c);sin.append(s)
            cursor+=n+queries;ends.append(cursor)
        # Cumulative latent-token counts: image i owns video rows
        # [video_ends[i-1], video_ends[i]) once conditions are removed.
        video_ends=[];running=0
        for n in lengths:running+=n;video_ends.append(running)
        layout=cls(pixels,groups,grids,lengths,tuple(ends),tuple((i+1)*queries for i in range(len(images))),
            tuple(video_ends),torch.tensor(video,dtype=torch.long),
            torch.tensor(video,dtype=torch.long),torch.tensor(text,dtype=torch.long),
            torch.tensor(samples,dtype=torch.long),torch.tensor(vsamples,dtype=torch.long),
            torch.tensor(weights,dtype=torch.float32),torch.cat(cos),torch.cat(sin),torch.empty(0))
        layout.plan_microbatches(token_budget,max_images)
        return layout

    def plan_microbatches(self,token_budget=0,max_images=0):
        """Build only small local indices on CPU; large rope/loss tensors stay shared.

        A single image/CQ pair is indivisible. An oversized pair is an error,
        rather than silently violating the promised activation-memory budget.
        """
        if self.cos.device.type != 'cpu':
            raise ValueError('Plan DiT microbatches in the CPU dataloader')
        if token_budget < 0 or max_images < 0:raise ValueError('Negative DiT budget')
        ranges=[];start=0;used=0
        previous=0
        for i,end in enumerate(self.ends):
            count=end-previous;previous=end
            if token_budget and count>token_budget:
                raise ValueError(f'Image/CQ pair needs {count} tokens, above DiT budget {token_budget}')
            if i>start and ((token_budget and used+count>token_budget) or
                            (max_images and i-start>=max_images)):
                ranges.append((start,i));start=i;used=0
            used+=count
        ranges.append((start,len(self.lengths)))
        # Preserve the existing single-forward fast path for small packs.
        self.microbatches=tuple(GenerationSlice.prepare(self,a,b) for a,b in ranges) if len(ranges)>1 else ()


@dataclass
class GenerationSlice:
    """CPU-planned image boundaries and local indices, with no device readback."""
    start: int
    stop: int
    latent_start: int
    latent_stop: int
    token_start: int
    token_stop: int
    ends: tuple
    text_ends: tuple
    video_ends: tuple
    video_rows: torch.Tensor
    video_indices: torch.Tensor
    text_indices: torch.Tensor
    sample_indices: torch.Tensor
    video_sample_indices: torch.Tensor

    def _map(self,fn):
        return type(self)(**{f.name:fn(getattr(self,f.name)) if isinstance(getattr(self,f.name),torch.Tensor)
                             else getattr(self,f.name) for f in fields(self)})

    @classmethod
    def prepare(cls,layout,start,stop):
        first=layout.ends[start-1] if start else 0
        last=layout.ends[stop-1]
        v0=sum(layout.lengths[:start]);v1=v0+sum(layout.lengths[start:stop])
        t0=layout.text_ends[start-1] if start else 0
        t1=layout.text_ends[stop-1]
        video_ends=[];running=0
        for n in layout.lengths[start:stop]:running+=n;video_ends.append(running)
        rows=layout.video_indices[v0:v1]-first
        return cls(start,stop,v0,v1,first,last,
            tuple(x-first for x in layout.ends[start:stop]),
            tuple(x-t0 for x in layout.text_ends[start:stop]),
            tuple(video_ends),rows,
            rows,layout.text_indices[t0:t1]-first,
            layout.sample_indices[first:last]-start,layout.video_sample_indices[v0:v1]-start)

    def bind(self,batch):
        """Views only: no new noise, normalization change, copy or accelerator sync."""
        p=batch.generation;a,b=self.start,self.stop
        lo,hi=self.latent_start,self.latent_stop
        layout=GenerationInputs((),(),p.grids[a:b],p.lengths[a:b],self.ends,self.text_ends,
            self.video_ends,self.video_rows,
            self.video_indices,self.text_indices,self.sample_indices,self.video_sample_indices,
            p.loss_weights[lo:hi],p.cos[self.token_start:self.token_stop],p.sin[self.token_start:self.token_stop],
            p.time_features[a:b])
        return SimpleNamespace(generation=layout,latents=batch.latents[lo:hi],
                               noise=batch.noise[lo:hi],times=batch.times[a:b])
