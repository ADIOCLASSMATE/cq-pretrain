"""Qwen3.5 vision weights with native-resolution inputs and 910B varlen attention.

The vision math/state-dict follows Hugging Face's Qwen3.5 vision tower (also
Qwen3-VL with no DeepStack). Patch convolution is an equivalent GEMM. Geometry
is prepared on CPU, outside forward; the NPU path has no per-image attention
loop or device-to-host length/position reads. No language-model weights enter
this module. See QWEN35_VISION.md for the retained CQ backbone contract.
"""
from dataclasses import dataclass, fields
from contextlib import nullcontext
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from . import parallel as P


ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'


def asset_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def vision_config(path):
    cfg = json.loads((asset_path(path) / 'config.json').read_text())['vision_config']
    if cfg.get('deepstack_visual_indexes'):
        raise ValueError('This adapter targets the Qwen3.5-0.8B tower without DeepStack')
    if (cfg['patch_size'], cfg['temporal_patch_size'], cfg['spatial_merge_size']) != (16, 2, 2):
        raise ValueError('Expected Qwen3.5 patch16 / temporal2 / merge2 checkpoint')
    if cfg['hidden_act'] != 'gelu_pytorch_tanh':
        raise ValueError('Unsupported vision MLP activation')
    return cfg


@dataclass
class VisionInputs:
    pixel_values: torch.Tensor
    content_indices: torch.Tensor
    pos_indices: torch.Tensor
    pos_weights: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    grid_thw: tuple
    lengths: tuple
    ends: tuple

    def to(self, device, non_blocking=False):
        return VisionInputs(**{f.name: (getattr(self,f.name).to(device,non_blocking=non_blocking)
            if isinstance(getattr(self,f.name),torch.Tensor) else getattr(self,f.name)) for f in fields(self)})

    def pin_memory(self, device):
        return VisionInputs(**{f.name: (getattr(self,f.name).pin_memory(device=device)
            if isinstance(getattr(self,f.name),torch.Tensor) and getattr(self,f.name).numel()
            else getattr(self,f.name)) for f in fields(self)})

    @classmethod
    def prepare(cls, pixel_values, grid_thw, content_indices, cfg):
        """Exact Qwen bilinear absolute positions + 2D rotary, in merge order."""
        if pixel_values.device.type != 'cpu':
            raise ValueError('Prepare vision geometry on CPU before transferring the batch')
        grid = tuple(tuple(map(int,g)) for g in grid_thw)
        side = math.isqrt(cfg['num_position_embeddings'])
        if side * side != cfg['num_position_embeddings']:
            raise ValueError('The pretrained absolute embedding grid must be square')
        merge = cfg['spatial_merge_size']
        rotary_dim = cfg['hidden_size'] // cfg['num_heads'] // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0,rotary_dim,2).float()/rotary_dim))
        all_indices, all_weights, all_angles, lengths = [], [], [], []
        for t,h,w in grid:
            if t != 1 or h < merge or w < merge or h % merge or w % merge:
                raise ValueError('Still images require t=1 and spatial grids divisible by merge_size')
            order = torch.arange(h*w).reshape(h//merge,merge,w//merge,merge).permute(0,2,1,3).flatten()
            row, col = order//w, order%w
            y,x = torch.linspace(0,side-1,h),torch.linspace(0,side-1,w)
            y0,x0 = y.long(),x.long()
            y1,x1 = (y0+1).clamp(max=side-1),(x0+1).clamp(max=side-1)
            dy,dx = y-y0,x-x0
            indices = torch.stack([(a[:,None]*side+b[None,:]).flatten()[order]
                for a,b in ((y0,x0),(y0,x1),(y1,x0),(y1,x1))])
            weights = torch.stack([(a[:,None]*b[None,:]).flatten()[order]
                for a,b in ((1-dy,1-dx),(1-dy,dx),(dy,1-dx),(dy,dx))])
            angle = torch.stack((row,col),dim=-1).float()[...,None] * inv_freq
            angle = angle.flatten(1)
            all_indices.append(indices); all_weights.append(weights)
            all_angles.append(torch.cat((angle,angle),dim=-1)); lengths.append(h*w)
        if not grid:
            raise ValueError('An empty image batch does not need a VisionInputs payload')
        if sum(lengths) != len(pixel_values) or sum(lengths)//merge**2 != content_indices.numel():
            raise ValueError('Pixel patches, merged image tokens and content indices disagree')
        angles = torch.cat(all_angles)
        ends = tuple(torch.tensor(lengths).cumsum(0).tolist())
        # CPU data workers build reproducible FP32 tables via FP64 trig. Some
        # CPU vector-math dispatches in the pinned torch build lose ~1e-4 on
        # FP32 cos; pretrained vision activations amplify that error.
        trig=angles.double()
        return cls(pixel_values,content_indices,torch.cat(all_indices,1),torch.cat(all_weights,1),
                   trig.cos().float(),trig.sin().float(),grid,tuple(lengths),ends)


class NativeImageProcessor:
    """Use the official fast processor, with an explicit packed-token budget."""
    def __init__(self, cfg, sequence_length):
        # Transformers 4.57 probes installed torch_npu while importing its CPU
        # image utilities, even on loader/test hosts without an Ascend driver.
        # Suppress only that optional probe during the CPU-only import. Do not
        # alter the installed package or a live NPU process's capability checks.
        import sys
        from contextlib import ExitStack
        from unittest.mock import patch
        with ExitStack() as imports:
            if 'torch_npu' not in sys.modules:
                for name in ('transformers.utils.import_utils.is_torch_npu_available',
                             'transformers.utils.is_torch_npu_available'):
                    imports.enter_context(patch(name,return_value=False))
            from transformers import Qwen2VLImageProcessorFast
        self.config = vision_config(cfg.vision_assets)
        # One merged token covers 32x32 pixels. Reserve BOS/BOI/EOI/EOS.
        maximum = min(cfg.vision_max_pixels, (sequence_length-4)*32*32)
        if maximum < cfg.vision_min_pixels:
            raise ValueError('Sequence length is too short for vision_min_pixels plus boundaries')
        self.processor = Qwen2VLImageProcessorFast.from_pretrained(
            str(asset_path(cfg.vision_assets)), local_files_only=True,
            min_pixels=cfg.vision_min_pixels, max_pixels=maximum)
        if (self.processor.patch_size,self.processor.temporal_patch_size,self.processor.merge_size) != (16,2,2):
            raise ValueError('Processor geometry differs from Qwen3.5-0.8B')
        self.target_size = cfg.image_size
        self.generation_dynamic = cfg.renderer=='moss'
        self.generation_max_pixels = cfg.generation_max_pixels
        self.pad_color=tuple(round(x*255) for x in self.processor.image_mean)

    def supported_aspect(self, image):
        """Retain extreme strips that the official smart_resize rejects.

        Pad only the narrow side to its documented 200:1 limit. Ordinary
        inputs are passed through unchanged; generation keeps the original.
        """
        from PIL import ImageOps
        w,h=image.size
        if max(w,h)<=200*min(w,h):return image
        width,height=max(w,math.ceil(h/200)),max(h,math.ceil(w/200))
        left,top=(width-w)//2,(height-h)//2
        return ImageOps.expand(image,(left,top,width-w-left,height-h-top),fill=self.pad_color)

    def __call__(self, native_rgb):
        import numpy as np
        from PIL import Image, ImageOps
        original = Image.fromarray(native_rgb)
        result = self.processor(images=self.supported_aspect(original),return_tensors='pt',device='cpu')
        grid = tuple(map(int,result['image_grid_thw'][0].tolist()))
        # Both paths start from the original image. MOSS retains the full
        # image with proportional resize and edge padding; only the legacy
        # square KL16 target below uses a center crop.
        if self.generation_dynamic:
            from .generation import target_image
            target=target_image(original,self.generation_max_pixels)
        else:
            target = ImageOps.fit(original,(self.target_size,self.target_size),
                                  Image.Resampling.LANCZOS,centering=(.5,.5))
        return np.asarray(target,dtype=np.uint8).copy(),result['pixel_values'],grid


class PatchProjection(nn.Module):
    """Conv3d checkpoint layout, GEMM execution and normal module gather hooks."""
    def __init__(self,cfg):
        super().__init__()
        kernel=(cfg['temporal_patch_size'],cfg['patch_size'],cfg['patch_size'])
        self.weight=nn.Parameter(torch.empty(cfg['hidden_size'],cfg['in_channels'],*kernel))
        self.bias=nn.Parameter(torch.empty(cfg['hidden_size']))
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        bound=1/math.sqrt(cfg['in_channels']*math.prod(kernel))
        nn.init.uniform_(self.bias,-bound,bound)

    def forward(self,pixels):
        return F.linear(pixels.to(self.weight.dtype),self.weight.flatten(1),self.bias)


class VisionPatchEmbed(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # Avoid torch_npu's automatic Conv3d weight-format conversion. Keep
        # the exact public weight names and invoke proj to preserve DDP hooks.
        self.proj=PatchProjection(cfg)

    def forward(self,pixels):
        return self.proj(pixels)


def rotary(x,cos,sin):
    a,b=x.float().chunk(2,dim=-1)
    return (x.float()*cos[:,None].float()+torch.cat((-b,a),dim=-1)*sin[:,None].float()).to(x.dtype)


def column(cfg,in_features,out_features,stride=1):
    if cfg.get('_tensor_parallel') and P.tp_size()>1:
        return P.ColumnLinear(in_features,out_features,bias=True,stride=stride)
    return nn.Linear(in_features,out_features)


def row(cfg,in_features,out_features):
    if cfg.get('_tensor_parallel') and P.tp_size()>1:
        return P.RowLinear(in_features,out_features,bias=True)
    return nn.Linear(in_features,out_features)


class VisionAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        width=cfg['hidden_size'];self.dim=width//cfg['num_heads']
        self.heads=cfg['num_heads']//(P.tp_size() if cfg.get('_tensor_parallel') else 1)
        self.qkv=column(cfg,width,3*width,stride=3)
        self.proj=row(cfg,width,width)

    def forward(self,x,layout,fused):
        q,k,v=self.qkv(x).reshape(len(x),3,self.heads,self.dim).unbind(1)
        q,k=rotary(q,layout.cos,layout.sin),rotary(k,layout.cos,layout.sin)
        if x.device.type=='npu' and fused:
            import torch_npu
            result=torch_npu.npu_fusion_attention(q.contiguous(),k.contiguous(),v.contiguous(),
                self.heads,input_layout='TND',scale=self.dim**-.5,keep_prob=1.,
                pre_tockens=2147483647,next_tockens=2147483647,sparse_mode=0,
                actual_seq_qlen=layout.ends,actual_seq_kvlen=layout.ends)[0]
        else:
            # Reference/CPU path. Never mix keys from different packed images.
            outputs=[]
            for qi,ki,vi in zip(q.split(layout.lengths),k.split(layout.lengths),v.split(layout.lengths)):
                outputs.append(F.scaled_dot_product_attention(qi.transpose(0,1)[None],
                    ki.transpose(0,1)[None],vi.transpose(0,1)[None],dropout_p=0.,is_causal=False)[0].transpose(0,1))
            result=torch.cat(outputs)
        return self.proj(result.reshape(len(x),-1))


class VisionMLP(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.linear_fc1=column(cfg,cfg['hidden_size'],cfg['intermediate_size'])
        self.linear_fc2=row(cfg,cfg['intermediate_size'],cfg['hidden_size'])

    def forward(self,x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x),approximate='tanh'))


class VisionBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.norm1=nn.LayerNorm(cfg['hidden_size'],eps=1e-6)
        self.norm2=nn.LayerNorm(cfg['hidden_size'],eps=1e-6)
        self.attn=VisionAttention(cfg);self.mlp=VisionMLP(cfg)

    def forward(self,x,layout,fused):
        x=x+self.attn(self.norm1(x),layout,fused)
        return x+self.mlp(self.norm2(x))


class VisionMerger(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.width=cfg['hidden_size']*cfg['spatial_merge_size']**2
        self.norm=nn.LayerNorm(cfg['hidden_size'],eps=1e-6)
        self.linear_fc1=column(cfg,self.width,self.width)
        self.linear_fc2=row(cfg,self.width,cfg['out_hidden_size'])

    def forward(self,x):
        return self.linear_fc2(F.gelu(self.linear_fc1(self.norm(x).reshape(-1,self.width))))


class QwenVisionTower(nn.Module):
    def __init__(self,cfg,fused=True,recompute=False,freeze_encoder=False,freeze_merger=False):
        super().__init__()
        self.cfg,self.fused,self.recompute=cfg,fused,recompute and not freeze_encoder
        self.freeze_encoder,self.freeze_merger=freeze_encoder,freeze_merger
        self.patch_embed=VisionPatchEmbed(cfg)
        self.pos_embed=nn.Embedding(cfg['num_position_embeddings'],cfg['hidden_size'])
        self.blocks=nn.ModuleList([VisionBlock(cfg) for _ in range(cfg['depth'])])
        self.merger=VisionMerger(cfg)
        if freeze_encoder:
            for module in (self.patch_embed,self.pos_embed,self.blocks):module.requires_grad_(False)
        if freeze_merger:self.merger.requires_grad_(False)
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_encoder:
            for module in (self.patch_embed,self.pos_embed,self.blocks):module.eval()
        if self.freeze_merger:self.merger.eval()
        return self

    def encode(self,layout):
        x=self.patch_embed(layout.pixel_values)
        values=self.pos_embed(layout.pos_indices)*layout.pos_weights.to(x.dtype)[...,None]
        # Preserve the official accumulation order, including BF16 rounding.
        x=x+(values[0]+values[1]+values[2]+values[3])
        for block in self.blocks:
            if self.recompute and self.training and torch.is_grad_enabled():
                x=checkpoint(block,x,layout,self.fused,use_reentrant=True)
            else:x=block(x,layout,self.fused)
        return x

    def forward(self,layout):
        # Frozen visual features are ordinary tensors, not inference tensors:
        # a trainable merger must be able to save them for its weight gradient.
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            x=self.encode(layout)
        return self.merger(x)


class QwenVisionInput(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        vc=vision_config(cfg.vision_assets)
        vc['_tensor_parallel']=cfg.vision_tensor_parallel
        self.visual=QwenVisionTower(vc,cfg.fused,cfg.checkpoint_vision,
                                   cfg.freeze_vision,cfg.freeze_vision_merger)
        self.bridge=(nn.Identity() if vc['out_hidden_size']==cfg.width else
                     nn.Linear(vc['out_hidden_size'],cfg.width,bias=False))

    def load_pretrained(self,path):
        from safetensors.torch import load_file
        path=asset_path(path)
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['model_id']!=MODEL_ID or manifest['revision']!=REVISION:
            raise ValueError('Vision assets do not match the pinned Qwen3.5-0.8B checkpoint')
        values=load_file(str(path/'vision.safetensors'))
        targets={name.replace('.impl.','.'):p for name,p in self.visual.named_parameters()}
        if set(targets)!=set(values):raise ValueError('Vision checkpoint keys differ')
        with torch.no_grad():
            for name,p in targets.items():p.copy_(P.checkpoint_shard(values[name],p))
        return manifest

    def forward(self,layout):
        return self.bridge(self.visual(layout))


def fake_vision_batch(cfg,length,batch_size,images,seed,device,dtype):
    """Native rectangular synthetic inputs through the same production packer."""
    import numpy as np
    from .input_pipeline.packing import Packer
    if batch_size!=1:
        raise ValueError('Native-resolution packing currently uses micro_batch=1; use gradient accumulation')
    class Documents:
        def __init__(self):
            rng=np.random.default_rng(seed);self.index=0
            self.pixels=[rng.integers(0,256,size=(cfg.image_size,cfg.image_size*(1+i%2),3),dtype=np.uint8)
                         for i in range(images)]
            elements=[]
            for i in range(images):elements.extend((dict(type='text',input_ids=[5,6,7]),dict(type='image',index=i)))
            elements.append(dict(type='text',input_ids=[8]*length))
            self.doc=dict(id='synthetic',elements=elements)
        def peek(self):return self.doc if self.index==0 else None
        def advance(self):self.index+=1
        def prefetch_images(self,*args):pass
        def image(self,doc,element):return self.pixels[doc['elements'][element]['index']]
        def state_dict(self):return dict(index=self.index)
    packed=Packer(Documents(),cfg,length,dict(bos=1,eos=2,boi=3,eoi=4,image=cfg.image_mask_token_id)).next()
    batch,_=packed
    if len(batch.image_indices)!=images:
        raise ValueError('Synthetic sequence is too short to fit the requested native images')
    generator=torch.Generator().manual_seed(seed)
    batch.pixels=batch.pixels.to(dtype).div(127.5).sub(1)
    batch.latents=torch.randn(images,cfg.image_tokens,cfg.latent_channels,generator=generator,dtype=dtype)
    batch.noise=torch.randn(images*cfg.flow_draws,cfg.image_tokens,cfg.latent_channels,generator=generator,dtype=dtype)
    batch.times=torch.randn(images*cfg.flow_draws,generator=generator).sigmoid().clamp(1e-5,1-1e-5)
    return batch.to(device)
