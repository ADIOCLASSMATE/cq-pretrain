"""Device-resident synthetic interleaved batches; no disk or decoder in timed steps."""
from dataclasses import dataclass, fields
import copy
import math
import os
import torch


@dataclass
class Batch:
    ids: torch.Tensor                 # [B,S], content including clean images
    pixels: torch.Tensor              # [images,3,H,W]
    latents: torch.Tensor             # [images,patches,C], generation targets only
    image_indices: torch.Tensor       # [images,max_input_tokens], -1 padding for native resolution
    text_indices: torch.Tensor        # supervised text positions in flattened [B,S]
    image_queries: torch.Tensor       # [B,S] boolean; query uses learned image mask
    sigma: torch.Tensor               # [B,S], image blocks have tied ranks
    segments: torch.Tensor            # [B,S], independent packed documents
    row_col: torch.Tensor             # [2,S,B], 2D multimodal RoPE
    noise: torch.Tensor               # [draws*images,patches,C]
    times: torch.Tensor               # [draws*images]
    mask: torch.Tensor                # [B,1,2S,S], True forbids reading
    valid_rows: torch.Tensor          # [B,2S]
    attention_plan: object = None     # Exact, optional data-side mask tiling
    mainline: object = None
    generation: object = None
    vision: object = None             # Packed Qwen pixels + CPU-prepared spatial layout

    def to(self, device, non_blocking=False):
        result=Batch(**{f.name:getattr(self,f.name).to(device,non_blocking=non_blocking) for f in fields(self)
                        if f.name not in ('attention_plan','mainline','vision','generation')})
        if self.vision is not None:result.vision=self.vision.to(device,non_blocking=non_blocking)
        if self.generation is not None:result.generation=self.generation.to(device,non_blocking=non_blocking)
        if self.mainline is not None:
            result.mainline=self.mainline.to(device,non_blocking=non_blocking)
        if self.attention_plan is not None and self.attention_plan.matches(self.mask):
            result.attention_plan=self.attention_plan.bind(result.mask)
        return result

    def pin_memory(self,device='npu'):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message="The argument 'device' of Tensor.pin_memory",category=DeprecationWarning)
            result=Batch(**{f.name:(getattr(self,f.name).pin_memory(device=device) if getattr(self,f.name).numel()
                                   else getattr(self,f.name)) for f in fields(self)
                           if f.name not in ('attention_plan','mainline','vision','generation')})
            if self.vision is not None:result.vision=self.vision.pin_memory(device)
            if self.generation is not None:result.generation=self.generation.pin_memory(device)
            if self.mainline is not None:result.mainline=self.mainline.pin_memory(device)
        return result

    def __deepcopy__(self,memo):
        result=Batch(**{f.name:copy.deepcopy(getattr(self,f.name),memo) for f in fields(self)
                        if f.name not in ('attention_plan','mainline')})
        result.mainline=copy.deepcopy(self.mainline,memo)
        memo[id(self)]=result
        if self.attention_plan is not None and self.attention_plan.matches(self.mask):
            result.attention_plan=self.attention_plan.bind(result.mask)
        return result


def make_mask(sigma, segments):
    same = (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] >= 0)
    content = (sigma[:, None, :] <= sigma[:, :, None]) & same
    query = (sigma[:, None, :] < sigma[:, :, None]) & same
    allowed = torch.cat((content, query), dim=1)
    valid = allowed.any(-1)
    # Fused attention on an empty row must have a safe key; zero it afterward.
    mask = ~allowed
    mask[:, :, 0] &= valid
    return mask[:, None].contiguous(), valid


def fake_batch(config, length=8192, batch_size=1, images_per_sequence=2,
               seed=42, device='cpu', dtype=torch.float32):
    if config.vision_encoder == 'qwen35':
        from .vision import fake_vision_batch
        return fake_vision_batch(config,length,batch_size,images_per_sequence,seed,device,dtype)
    config.validate()
    patches = config.image_tokens
    if images_per_sequence < 1 or length < images_per_sequence * (patches + 3) + 2:
        raise ValueError('Sequence must hold complete images, BOI/EOI and text between them')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.randint(5, config.vocab_size, (batch_size, length), generator=generator)
    ids[:, 0] = 1
    image_queries = torch.zeros_like(ids, dtype=torch.bool)
    sigma = torch.arange(length)[None].expand(batch_size, -1).clone()
    positions = torch.zeros(2, batch_size, length, dtype=torch.long)
    image_indices = []
    gap = (length - images_per_sequence * (patches + 2)) // (images_per_sequence + 1)
    side = math.isqrt(patches)
    for b in range(batch_size):
        cursor = 0
        last = 0
        for i in range(images_per_sequence):
            start = gap * (i + 1) + i * (patches + 2) + 1
            end = start + patches
            ids[b, start - 1], ids[b, end] = 2, 3
            ids[b, start:end] = 0
            image_queries[b, start:end] = True
            sigma[b, start:end], sigma[b, end] = start + 1, start
            image_indices.append(torch.arange(start, end) + b * length)
            positions[:, b, last:start] = torch.arange(cursor, cursor + start - last)[None]
            cursor += start - last
            local = torch.arange(patches)
            positions[0, b, start:end] = cursor + local // side
            positions[1, b, start:end] = cursor + local % side
            cursor += side
            last = end
        positions[:, b, last:] = torch.arange(cursor, cursor + length - last)[None]
    images = batch_size * images_per_sequence
    text_mask = ~image_queries
    text_mask[:, 0] = False
    segments = torch.zeros_like(ids)
    mask, valid = make_mask(sigma, segments)
    latent_shape = (images, patches, config.latent_channels)
    noise_shape = (images * config.flow_draws, patches, config.latent_channels)
    times = torch.randn(images * config.flow_draws, generator=generator).sigmoid().clamp(1e-5, 1-1e-5)
    result = Batch(ids=ids, pixels=torch.randn(images, 3, config.image_size, config.image_size,
                     generator=generator, dtype=dtype),
        latents=torch.randn(*latent_shape, generator=generator, dtype=dtype),
        image_indices=torch.stack(image_indices), text_indices=text_mask.flatten().nonzero().flatten(),
        image_queries=image_queries, sigma=sigma, segments=segments,
        row_col=positions.permute(0, 2, 1).contiguous(),
        noise=torch.randn(*noise_shape, generator=generator, dtype=dtype), times=times,
        mask=mask, valid_rows=valid)
    from .mainline import build_plan
    result.mainline=build_plan(result,config)
    return result.to(device)
