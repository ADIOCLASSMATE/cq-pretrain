"""Native Wan 2.1 math with packed TND attention and tensor-parallel projections.

Images are single-frame videos. Geometry, RoPE and sequence boundaries are
prepared on CPU; the device path never expands a quadratic padding mask.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import training.mcore as P
import models.kernels as K


def linear(layer, x):
    return layer(x.to(layer.weight.dtype))


class FP32Linear(nn.Linear):
    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return F.linear(x.float(), self.weight.float(), self.bias.float())


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(), self.eps).to(x.dtype)


class RMSNorm(nn.Module):
    """Wan normalizes the entire projected width, including other TP heads."""
    def __init__(self, width, eps):
        super().__init__(); self.eps = eps
        self.weight = P.mark_sharded(nn.Parameter(torch.ones(width // P.tp_size())))

    def forward(self, x):
        variance = x.float().square().mean(-1, keepdim=True)
        if P.tp_size() > 1:
            from torch.distributed.nn.functional import all_reduce
            variance = all_reduce(variance, group=P.state().get_tensor_model_parallel_group()) / P.tp_size()
        return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


def packed_attention(q, k, v, q_ends, k_ends):
    q, k = q.to(v.dtype), k.to(v.dtype)
    if q.device.type == 'npu':
        import torch_npu
        return torch_npu.npu_fusion_attention(q.contiguous(), k.contiguous(), v.contiguous(),
            q.shape[1], input_layout='TND', scale=q.shape[-1]**-.5, keep_prob=1.,
            sparse_mode=0, pre_tockens=2147483647, next_tockens=2147483647,
            actual_seq_qlen=q_ends, actual_seq_kvlen=k_ends)[0]
    result=[]; qa=ka=0
    for qb,kb in zip(q_ends,k_ends):
        a,b,c=(t.transpose(0,1)[None] for t in (q[qa:qb],k[ka:kb],v[ka:kb]))
        result.append(F.scaled_dot_product_attention(a,b,c)[0].transpose(0,1))
        qa,ka=qb,kb
    return torch.cat(result)


def rotary(x, rope):
    cos,sin=rope
    real,imag=x.float().unflatten(-1,(-1,2)).unbind(-1)
    return torch.stack((real*cos-imag*sin,imag*cos+real*sin),-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg, cross=False):
        super().__init__(); self.cross=cross
        dim=cfg.dit_width; self.heads=cfg.dit_heads//P.tp_size(); self.head_dim=dim//cfg.dit_heads
        self.q=P.ColumnLinear(dim,dim,bias=True)
        self.kv=P.ColumnLinear(dim,2*dim,bias=True,stride=2)
        self.o=P.RowLinear(dim,dim,bias=True)
        self.norm_q=RMSNorm(dim,cfg.norm_eps); self.norm_k=RMSNorm(dim,cfg.norm_eps)

    def forward(self, x, layout, context=None):
        q=self.norm_q(linear(self.q,x)).reshape(-1,self.heads,self.head_dim)
        k,v=linear(self.kv,x if context is None else context).chunk(2,-1)
        k=self.norm_k(k).reshape(-1,self.heads,self.head_dim)
        v=v.reshape(-1,self.heads,self.head_dim)
        if context is None:q,k=rotary(q,(layout.cos,layout.sin)),rotary(k,(layout.cos,layout.sin))
        y=packed_attention(q,k,v,layout.video_ends,
                           layout.video_ends if context is None else layout.text_ends)
        return self.o(y.flatten(1))


class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); dim=cfg.dit_width
        self.fused=cfg.fused
        self.norm1=LayerNorm(dim,eps=cfg.norm_eps,elementwise_affine=False)
        self.self_attn=Attention(cfg)
        self.norm3=LayerNorm(dim,eps=cfg.norm_eps)
        self.cross_attn=Attention(cfg,cross=True)
        self.norm2=LayerNorm(dim,eps=cfg.norm_eps,elementwise_affine=False)
        self.ffn=nn.Sequential(P.ColumnLinear(dim,cfg.dit_ffn,bias=True),nn.GELU(approximate='tanh'),
                              P.RowLinear(cfg.dit_ffn,dim,bias=True))
        self.modulation=nn.Parameter(torch.randn(1,6,dim)/math.sqrt(dim))

    def forward(self,x,modulation,context,layout):
        # Sample-wise modulation is gathered once before all layers/recomputes.
        shift,scale,gate,fs,fc,fg=(modulation+self.modulation.float()).unbind(1)
        y=self.self_attn(K.modulated_norm(x,scale,shift,self.fused,self.norm1.eps,upcast=True),layout)
        x=K.gated_add(x.float(),y.float(),gate,self.fused)
        x=x+self.cross_attn(self.norm3(x),layout,context).float()
        y=linear(self.ffn[0],K.modulated_norm(x,fc,fs,self.fused,self.norm2.eps,upcast=True))
        return K.gated_add(x,linear(self.ffn[2],self.ffn[1](y)).float(),fg,self.fused)


class Head(nn.Module):
    def __init__(self,cfg):
        super().__init__(); dim=cfg.dit_width
        self.fused=cfg.fused
        self.modulation=nn.Parameter(torch.randn(1,2,dim)/math.sqrt(dim))
        self.norm=LayerNorm(dim,eps=cfg.norm_eps,elementwise_affine=False)
        self.head=nn.Linear(dim,cfg.latent_channels*4)

    def forward(self,x,time):
        shift,scale=(time[:,None].float()+self.modulation.float()).unbind(1)
        value=linear(self.head,K.modulated_norm(x,scale,shift,self.fused,self.norm.eps,upcast=True))
        # Native head emits [patch_h,patch_w,C]; Conv3d input patches use [C,h,w].
        return value.reshape(-1,2,2,value.shape[-1]//4).permute(0,3,1,2).flatten(1)


class PatchEmbedding(nn.Module):
    """Preserve native Conv3d weights while executing nonoverlapping patches by GEMM."""
    def __init__(self,cfg):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(cfg.dit_width,cfg.latent_channels,1,2,2))
        self.bias=nn.Parameter(torch.zeros(cfg.dit_width))
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))

    def forward(self,patches):
        return F.linear(patches.to(self.weight.dtype),self.weight.flatten(1),self.bias)


class Wan21Generator(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        dim=cfg.dit_width
        self.patch_embedding=PatchEmbedding(cfg)
        self.condition_proj=nn.Linear(cfg.width,cfg.wan_text_dim)
        self.text_embedding=nn.Sequential(P.ColumnLinear(cfg.wan_text_dim,dim,bias=True),
            nn.GELU(approximate='tanh'),P.RowLinear(dim,dim,bias=True))
        self.time_embedding=nn.Sequential(FP32Linear(cfg.wan_freq_dim,dim),nn.SiLU(),FP32Linear(dim,dim))
        self.time_projection=nn.Sequential(nn.SiLU(),FP32Linear(dim,6*dim))
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.dit_layers)])
        self.head=Head(cfg)
        nn.init.normal_(self.condition_proj.weight,std=.02); nn.init.zeros_(self.condition_proj.bias)

    def project_conditions(self,conditions,layout):
        context=linear(self.condition_proj,conditions.reshape(-1,self.cfg.width).index_select(0,layout.condition_indices))
        context=linear(self.text_embedding[0],context)
        return linear(self.text_embedding[2],self.text_embedding[1](context))

    def predict(self,xt,conditions,time_features,layout,projected_conditions=None):
        time=self.time_embedding(time_features.float())
        e=self.time_projection(time).reshape(-1,6,self.cfg.dit_width)
        modulation=e.index_select(0,layout.video_sample_indices).contiguous()
        head_time=time.index_select(0,layout.video_sample_indices)
        context=self.project_conditions(conditions,layout) if projected_conditions is None else projected_conditions
        x=self.patch_embedding(xt)
        for i,block in enumerate(self.blocks):
            recompute=self.cfg.checkpoint_dit or i>=self.cfg.dit_layers-self.cfg.checkpoint_dit_layers
            if recompute and self.training and torch.is_grad_enabled():
                x=checkpoint(block,x,modulation,context,layout,use_reentrant=True)
            else:x=block(x,modulation,context,layout)
        return self.head(x,head_time).float()

    def forward(self,conditions,batch):
        layout=batch.generation
        sigma=batch.times.index_select(0,layout.video_sample_indices)[:,None]
        xt=(1-sigma)*batch.latents.float()+sigma*batch.noise.float()
        prediction=self.predict(xt,conditions,layout.time_features,layout)
        error=(prediction-(batch.noise.float()-batch.latents.float())).square().mean(-1)
        return (error*layout.loss_weights).sum()

    def load_pretrained(self,path):
        from safetensors import safe_open
        root=Path(path); config=json.loads((root/'config.json').read_text())
        for key,expected in dict(dim=self.cfg.dit_width,ffn_dim=self.cfg.dit_ffn,
            num_heads=self.cfg.dit_heads,num_layers=self.cfg.dit_layers,in_dim=self.cfg.latent_channels).items():
            if config[key]!=expected:raise ValueError(f'Wan checkpoint {key} differs from configuration')
        index=root/'diffusion_pytorch_model.safetensors.index.json'
        if index.exists():files=sorted(set(json.loads(index.read_text())['weight_map'].values()))
        else:files=[p.name for p in root.glob('*.safetensors')]
        if not files:raise FileNotFoundError(f'No native Wan safetensors in {root}')
        targets={}
        for name,param in self.named_parameters():
            if name.startswith('condition_proj.'):continue
            clean=name.replace('.impl.','.')
            if '.kv.' in clean:
                for label,view in zip(('k','v'),param.chunk(2,0)):
                    targets[clean.replace('.kv.',f'.{label}.')]=P.mark_sharded(view)
            else:targets[clean]=param
        seen=set()
        for filename in files:
            with safe_open(str(root/filename),framework='pt',device='cpu') as source,torch.no_grad():
                for name in source.keys():
                    if name not in targets:raise ValueError(f'Unexpected Wan weight {name}')
                    target=targets[name];value=P.checkpoint_shard(source.get_tensor(name),target)
                    if value.shape!=target.shape:raise ValueError(f'Wan shape mismatch: {name}')
                    target.copy_(value);seen.add(name)
        if seen!=set(targets):raise ValueError(f'Missing Wan weights: {sorted(set(targets)-seen)}')
        return dict(path=str(root),loaded_tensors=len(seen),config=config)
