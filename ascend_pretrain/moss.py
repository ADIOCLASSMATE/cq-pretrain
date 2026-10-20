"""MOSS-Image-Base math/weights, packed TND attention and tensor parallelism.

Adapted from OpenMOSS-Team/MOSS-Image-Base@1ceede80badddd6a7812069619edaa5633587192
pipeline_moss_image.py (Apache-2.0). Retains SITU, four-axis RoPE, QK RMSNorm,
shared time modulation and both text refinement blocks. CQ supplies conditions.
Each packed image stream is [noisy image tokens, CQ conditions], matching
the original MOSS order and RoPE coordinates.
"""
import json
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from . import parallel as P, kernels as K

_pointwise_setting=os.environ.get('PRETRAIN_MOSS_FUSED_POINTWISE','0')
FUSED_POINTWISE=_pointwise_setting!='0'
POINTWISE_KINDS=set(('situ','affine','gated_add') if _pointwise_setting=='1' else _pointwise_setting.split(','))


class RMSNorm(nn.Module):
    def __init__(self,dim):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim))

    def forward(self,x):
        return K.rms_norm(x,self.weight,1e-6)


def situ(hidden,gate):
    # SITU is part of the pretrained architecture; do not replace with SwiGLU.
    if FUSED_POINTWISE and 'situ' in POINTWISE_KINDS and hidden.device.type=='npu':
        from .moss_kernels import situ as fused
        return fused(hidden,gate)
    return (4*torch.tanh(gate/4)*torch.sigmoid(gate))*(25*torch.tanh(hidden/25))


def affine(hidden,scale,shift):
    if FUSED_POINTWISE and 'affine' in POINTWISE_KINDS and hidden.device.type=='npu':
        from .moss_kernels import affine as fused
        return fused(hidden,scale,shift)
    return hidden*(1+scale)+shift


def gated_add(hidden,branch,gate):
    if FUSED_POINTWISE and 'gated_add' in POINTWISE_KINDS and hidden.device.type=='npu':
        from .moss_kernels import gated_add as fused
        return fused(hidden,branch,gate)
    return hidden+branch*gate


def packed_attention(q,k,v,rope,heads,ends):
    dim=q.shape[-1]//heads
    q,k,v=(x.reshape(-1,heads,dim) for x in (q,k,v))
    cos,sin=(x.to(q.dtype) for x in rope)
    # rotary_mul expects [S,B,N,D]; retain the fused exact rotate-half kernel.
    q=K.rotary(q[:,None].contiguous(),cos[:,None],sin[:,None])[:,0]
    k=K.rotary(k[:,None].contiguous(),cos[:,None],sin[:,None])[:,0]
    if q.device.type=='npu':
        import torch_npu
        y=torch_npu.npu_fusion_attention(q.contiguous(),k.contiguous(),v.contiguous(),heads,
            input_layout='TND',scale=dim**-.5,keep_prob=1.,sparse_mode=0,
            pre_tockens=2147483647,next_tockens=2147483647,
            actual_seq_qlen=ends,actual_seq_kvlen=ends)[0]
    else:
        rows=[];start=0
        for end in ends:
            a,b,c=(x[start:end].transpose(0,1)[None] for x in (q,k,v))
            rows.append(F.scaled_dot_product_attention(a,b,c)[0].transpose(0,1));start=end
        y=torch.cat(rows)
    return y.flatten(1)


class TextBlock(nn.Module):
    def __init__(self,dim,ffn_dim,heads):
        super().__init__();self.heads=heads//P.tp_size()
        self.norm1,self.norm2=RMSNorm(dim),RMSNorm(dim)
        self.q_norm,self.k_norm=RMSNorm(dim//heads),RMSNorm(dim//heads)
        self.qkv=P.ColumnLinear(dim,3*dim,bias=True,stride=3)
        self.proj=P.RowLinear(dim,dim,bias=True)
        self.ffn_gate,self.ffn_up=P.ColumnLinear(dim,ffn_dim,bias=True),P.ColumnLinear(dim,ffn_dim,bias=True)
        self.ffn_down=P.RowLinear(ffn_dim,dim,bias=True)

    def forward(self,x,rope,ends):
        q,k,v=self.qkv(self.norm1(x)).chunk(3,-1)
        q=self.q_norm(q.unflatten(-1,(self.heads,-1))).flatten(1)
        k=self.k_norm(k.unflatten(-1,(self.heads,-1))).flatten(1)
        x=x+self.proj(packed_attention(q,k,v,rope,self.heads,ends))
        y=self.norm2(x)
        return x+self.ffn_down(situ(self.ffn_up(y),self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self,text_dim,dim,ffn_dim,heads):
        super().__init__();self.proj=nn.Linear(text_dim,dim)
        self.blocks=nn.ModuleList([TextBlock(dim,ffn_dim,heads) for _ in range(2)])

    def forward(self,x,rope,ends):
        x=self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:x=block(x,rope,ends)
        return x


class CrossAttention(nn.Module):
    """Latents (query) read CQ conditions (key/value); no RoPE on either side.

    Conditions are an unordered set of learned reads of the image, not a
    spatial sequence, so rotary position would impose a false geometry. The
    gate is zero-initialized: at step zero this block is exactly the identity
    and the renderer reproduces the pretrained MOSS computation, so the
    control starts from the same place as the baseline.

    The projections must be initialized here, explicitly. This block has no
    counterpart in the official checkpoint, so `official_tensors` skips it; and
    a fully pretrained run sets `perform_initialization=False`, so MCore never
    initializes it either. Left to those two paths the weights stay at whatever
    the allocator returned, which measured as exact zeros -- and a zero to_out
    makes d(loss)/d(gate) zero as well, so the gate can never open. The branch
    is then permanently dead while training looks healthy, because the renderer
    simply learns unconditional generation.
    """
    def __init__(self,dim,heads):
        super().__init__();self.heads=heads//P.tp_size();self.dim=dim//heads
        self.norm_q,self.norm_kv=RMSNorm(dim),RMSNorm(dim)
        self.q_norm,self.k_norm=RMSNorm(dim//heads),RMSNorm(dim//heads)
        self.to_q=P.ColumnLinear(dim,dim)
        self.to_kv=P.ColumnLinear(dim,2*dim,stride=2)
        self.to_out=P.RowLinear(dim,dim)
        with torch.no_grad():
            for linear in (self.to_q,self.to_kv,self.to_out):
                P.normal_init(linear.impl.weight)
        # Zero gate keeps step 0 an exact identity; nonzero projections keep the
        # gradient path to that gate alive so it can actually open.
        self.gate=nn.Parameter(torch.zeros(dim))

    def forward(self,x,memory,layout):
        q=self.q_norm(self.to_q(self.norm_q(x)).unflatten(-1,(self.heads,-1))).flatten(1)
        k,v=self.to_kv(self.norm_kv(memory)).chunk(2,-1)
        k=self.k_norm(k.unflatten(-1,(self.heads,-1))).flatten(1)
        y=cross_packed_attention(q,k,v,self.heads,layout)
        return x+self.to_out(y)*self.gate


def cross_packed_attention(q,k,v,heads,layout):
    """Per-image blocks: image i's latents see only image i's conditions."""
    dim=q.shape[-1]//heads
    q=q.reshape(-1,heads,dim);k=k.reshape(-1,heads,dim);v=v.reshape(-1,heads,dim)
    if q.device.type=='npu':
        import torch_npu
        y=torch_npu.npu_fusion_attention(q.contiguous(),k.contiguous(),v.contiguous(),heads,
            input_layout='TND',scale=dim**-.5,keep_prob=1.,sparse_mode=0,
            pre_tockens=2147483647,next_tockens=2147483647,
            actual_seq_qlen=layout.video_ends,actual_seq_kvlen=layout.text_ends)[0]
    else:
        rows=[];qs=0;ks=0
        for qe,ke in zip(layout.video_ends,layout.text_ends):
            a,b,c=(x.transpose(0,1)[None] for x in (q[qs:qe],k[ks:ke],v[ks:ke]))
            rows.append(F.scaled_dot_product_attention(a,b,c)[0].transpose(0,1));qs=qe;ks=ke
        y=torch.cat(rows)
    return y.flatten(1)


class Block(nn.Module):
    def __init__(self,dim,ffn_dim,heads,fused_qkv=False,cross=False):
        super().__init__();self.heads=heads//P.tp_size()
        self.fused_qkv=fused_qkv
        self.cross_attn=CrossAttention(dim,heads) if cross else None
        self.norm1,self.norm2=RMSNorm(dim),RMSNorm(dim)
        self.q_norm,self.k_norm=RMSNorm(dim//heads),RMSNorm(dim//heads)
        self.proj=nn.ModuleDict({'qkv':P.ColumnLinear(dim,3*dim,stride=3)} if fused_qkv else
                               {name:P.ColumnLinear(dim,dim) for name in ('q','k','v')})
        self.proj['o']=P.RowLinear(dim,dim)
        self.proj['ffn_up_gate']=P.ColumnLinear(dim,2*ffn_dim,stride=2)
        self.proj['ffn_down']=P.RowLinear(ffn_dim,dim)

    def forward(self,x,rope,modulation,ends,memory=None,layout=None):
        shift_a,scale_a,gate_a,shift_f,scale_f,gate_f=modulation
        y=affine(self.norm1(x),scale_a,shift_a)
        q,k,v=self.proj['qkv'](y).chunk(3,-1) if self.fused_qkv else tuple(self.proj[name](y) for name in ('q','k','v'))
        q=self.q_norm(q.unflatten(-1,(self.heads,-1))).flatten(1)
        k=self.k_norm(k.unflatten(-1,(self.heads,-1))).flatten(1)
        x=gated_add(x,self.proj['o'](packed_attention(q,k,v,rope,self.heads,ends)),gate_a)
        if self.cross_attn is not None:x=self.cross_attn(x,memory,layout)
        y=affine(self.norm2(x),scale_f,shift_f)
        hidden,gate=self.proj['ffn_up_gate'](y).chunk(2,-1)
        return gated_add(x,self.proj['ffn_down'](situ(hidden,gate)),gate_f)


class Rope(nn.Module):
    def __init__(self,dim):
        super().__init__();self.register_buffer('bands',1/(10000**(torch.arange(dim//8,dtype=torch.float32)/(dim//8))))


class MOSSRenderer(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        dim,ffn,heads=cfg.dit_width,cfg.dit_ffn,cfg.dit_heads
        self.video_embedder=nn.Linear(cfg.latent_channels,dim)
        self.text_embedder=TextProjection(cfg.width,dim,ffn,heads)
        self.rope=Rope(dim//heads)
        self.time_embedding=nn.Sequential(nn.Linear(256,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.shared_block_modulation=nn.Linear(dim,6*dim)
        self.cross_conditioning=getattr(cfg,'moss_cross_attention',False)
        self.layers=nn.ModuleList([Block(dim,ffn,heads,cfg.moss_fused_qkv,self.cross_conditioning)
                                   for _ in range(cfg.dit_layers)])
        self.final_norm_video=RMSNorm(dim)
        self.final_linear_video=nn.Linear(dim,cfg.latent_channels,bias=False)
        self.video_head_modulation=nn.Linear(dim,2*dim)

    def project_conditions(self,conditions,layout,rope=None):
        dtype=self.video_embedder.weight.dtype
        rope=tuple(x.to(dtype) for x in (layout.cos,layout.sin)) if rope is None else rope
        text_rope=tuple(x.index_select(0,layout.text_indices) for x in rope)
        return self.text_embedder(conditions.flatten(0,1),text_rope,layout.text_ends)

    def predict(self,xt,conditions,time_features,layout,projected_conditions=None):
        dtype=self.video_embedder.weight.dtype
        time=self.time_embedding(time_features.to(dtype)).float()
        mod=self.shared_block_modulation(time.to(dtype)).view(-1,6,self.cfg.dit_width)
        # Gather time modulation once outside 42 repeated/recomputed blocks.
        # CANN otherwise materializes each strided [S,6,D] slice in every
        # layer and its recomputation. Materialize these six views once.
        modulation=tuple(x.contiguous() for x in mod.index_select(0,layout.sample_indices).unbind(1))
        head=self.video_head_modulation(time.to(dtype)).view(-1,2,self.cfg.dit_width)
        rope=(layout.cos.to(dtype),layout.sin.to(dtype))
        text=self.project_conditions(conditions,layout,rope) if projected_conditions is None else projected_conditions
        video=self.video_embedder(xt.to(dtype))
        if self.cross_conditioning:
            # Latents self-attend only among themselves; conditions arrive
            # through cross attention. Modulation and RoPE are restricted to
            # the video rows, so the stream carries no condition positions.
            x=video
            ends=layout.video_ends
            modulation=tuple(m.index_select(0,layout.video_rows) for m in modulation)
            rope=(rope[0].index_select(0,layout.video_indices),
                  rope[1].index_select(0,layout.video_indices))
        else:
            x=video.new_zeros(layout.ends[-1],self.cfg.dit_width)
            x=x.index_copy(0,layout.video_indices,video).index_copy(0,layout.text_indices,text)
            ends=layout.ends
        for index,layer in enumerate(self.layers):
            recompute=self.cfg.checkpoint_dit or index>=self.cfg.dit_layers-self.cfg.checkpoint_dit_layers
            if self.training and torch.is_grad_enabled() and recompute:
                # Pass modulation tensors directly: reentrant checkpoint does
                # not discover grad inputs nested inside a tuple.
                def run(hidden,memory,*mods,layer=layer):
                    return layer(hidden,rope,mods,ends,memory,layout)
                x=checkpoint(run,x,text,*modulation,use_reentrant=True)
            else:x=layer(x,rope,modulation,ends,text,layout)
        x=x if self.cross_conditioning else x.index_select(0,layout.video_indices)
        x=self.final_norm_video(x)
        head=head.index_select(0,layout.video_sample_indices)
        x=affine(x,head[:,1],head[:,0])
        return self.final_linear_video(x)

    def forward(self,conditions,batch):
        if self.cfg.flow_draws!=1:raise ValueError('MOSS CQ training uses exactly one flow draw per image')
        layout=batch.generation
        sigma=batch.times.index_select(0,layout.video_sample_indices)[:,None]
        target,noise=batch.latents.float(),batch.noise.float()
        # Official MOSS uses noise=1, data=0; velocity is noise minus data.
        xt=(1-sigma)*target+sigma*noise
        prediction=self.predict(xt,conditions,layout.time_features,layout)
        mse=(prediction.float()-(noise-target)).square().mean(-1)
        return (mse*layout.loss_weights).sum()

    def official_tensors(self):
        """Writable views under the original MOSS checkpoint names."""
        target={}
        for name,tensor in self.named_parameters():
            name=name.replace('.impl.','.')
            # Cross-attention conditioning is this project's addition and has
            # no counterpart in the official checkpoint. Keep strict matching
            # for every pretrained tensor by excluding it here; its zero gate
            # makes the untrained block an identity at initialization.
            if '.cross_attn.' in name:continue
            if name.endswith('.proj.qkv.weight'):
                for label,part in zip(('q','k','v'),tensor.chunk(3,dim=0)):
                    target[name.replace('.qkv.',f'.{label}.')]=P.mark_sharded(part)
            else:target[name]=tensor
        target['rope.bands']=self.rope.bands
        return target

    def load_pretrained(self,path):
        from safetensors import safe_open
        root=Path(path);manifest=json.loads((root/'manifest.json').read_text())
        if (manifest['model_id'],manifest['revision'])!=('OpenMOSS-Team/MOSS-Image-Base','1ceede80badddd6a7812069619edaa5633587192'):
            raise ValueError('Unexpected MOSS checkpoint')
        index=json.loads((root/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
        target=self.official_tensors()
        if set(target)!=set(index):raise ValueError(f'MOSS key mismatch: missing={set(index)-set(target)}, extra={set(target)-set(index)}')
        for shard in sorted(set(index.values())):
            with safe_open(str(root/'transformer'/shard),framework='pt',device='cpu') as source,torch.no_grad():
                for name,tensor in target.items():
                    if index[name]!=shard:continue
                    value=P.checkpoint_shard(source.get_tensor(name),tensor)
                    if value.shape!=tensor.shape:raise ValueError(f'MOSS shape mismatch: {name}: {value.shape} vs {tensor.shape}')
                    tensor.copy_(value)
        return dict(model_id=manifest['model_id'],revision=manifest['revision'],files=manifest['files'],loaded_tensors=len(target))
