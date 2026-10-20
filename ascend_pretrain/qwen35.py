"""Qwen3.5 parameterization with CQ read-only streams and image blocks."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from . import kernels as K
from . import parallel as P


class RMSNorm(nn.Module):
    def __init__(self,width,eps=1e-6):
        super().__init__();self.weight=nn.Parameter(torch.zeros(width));self.eps=eps

    def forward(self,x,fused=True):
        # Qwen3.5 checkpoints store the offset from one, including Q/K norms.
        if x.dtype==torch.float64:
            return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.eps)*(1+self.weight)
        weight=1+self.weight.float()
        if x.device.type=='npu' and fused:
            return K.rms_norm(x,weight,self.eps,True)
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)*weight).to(x.dtype)


class MLP(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.fused_projections=cfg.qwen_fused_projections
        if self.fused_projections:self.gate_up_proj=P.ColumnLinear(cfg.width,2*cfg.ffn,stride=2)
        else:
            self.gate_proj=P.ColumnLinear(cfg.width,cfg.ffn)
            self.up_proj=P.ColumnLinear(cfg.width,cfg.ffn)
        self.down_proj=P.RowLinear(cfg.ffn,cfg.width)

    def forward(self,x):
        if self.fused_projections:
            projected=self.gate_up_proj(x)
            if x.device.type=='npu':
                import torch_npu
                activated=torch_npu.npu_swiglu(projected,dim=-1)
            else:
                gate,up=projected.chunk(2,-1);activated=F.silu(gate)*up
            return self.down_proj(activated)
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


def recurrent_delta(q,k,v,g,beta,bounds):
    """Differentiable CPU oracle. NPU training must use the fused backend."""
    q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)*q.shape[-1]**-.5
    k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    outputs=[]
    for begin,end in zip(bounds,bounds[1:]):
        state=q.new_zeros(q.shape[0],q.shape[2],q.shape[3],v.shape[3])
        for i in range(begin,end):
            state=state*g[:,i].exp()[...,None,None]
            residual=v[:,i]-(state*k[:,i,:,:,None]).sum(-2)
            state=state+k[:,i,:,:,None]*(residual*beta[:,i,:,None])[...,None,:]
            outputs.append((state*q[:,i,:,:,None]).sum(-2))
    return torch.stack(outputs,dim=1)


class CausalConv(nn.Module):
    def __init__(self,channels,kernel):
        super().__init__();self.weight=nn.Parameter(torch.empty(channels,1,kernel))
        nn.init.uniform_(self.weight,-.5,.5)

    def forward(self,x,plan):
        if x.device.type=='npu':
            from . import delta_ascendc
            return delta_ascendc.conv(x,self.weight[:,0],activation='silu',
                cu_seqlens=plan.content_cu,cu_seqlens_cpu=plan.content_bounds)[0]
        results=[]
        for a,b in zip(plan.content_bounds,plan.content_bounds[1:]):
            y=F.conv1d(x[:,a:b].transpose(1,2),self.weight,padding=self.weight.shape[-1]-1,
                       groups=x.shape[-1])[...,:b-a]
            results.append(F.silu(y).transpose(1,2))
        return torch.cat(results,dim=1)


class GatedNorm(nn.Module):
    def __init__(self,dim,eps):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps

    def forward(self,x,z):
        if x.device.type=='npu':
            from .delta_ascendc import gated_norm
            return gated_norm(x,z,self.weight)
        compute=x if x.dtype==torch.float64 else x.float()
        normalized=compute*torch.rsqrt(compute.square().mean(-1,keepdim=True)+self.eps)
        return (normalized.to(x.dtype)*self.weight*F.silu(z.to(compute.dtype))).to(x.dtype)


class GatedDelta(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        tp=P.tp_size()
        key=cfg.linear_heads*cfg.linear_head_dim
        value=(cfg.linear_value_heads or cfg.linear_heads)*cfg.linear_head_dim
        size=value
        h=cfg.linear_value_heads or cfg.linear_heads
        if cfg.qwen_fused_projections:
            self.in_proj_all=P.ColumnLinear(cfg.width,2*key+2*value+2*h)
            P.mark_sharded(self.in_proj_all.weight,splits=(key,key,value,value,h,h))
        else:
            self.in_proj_qkv=P.ColumnLinear(cfg.width,2*key+value)
            P.mark_sharded(self.in_proj_qkv.weight,splits=(key,key,value))
            self.in_proj_z=P.ColumnLinear(cfg.width,size)
            self.in_proj_a=P.ColumnLinear(cfg.width,h)
            self.in_proj_b=P.ColumnLinear(cfg.width,h)
        self.conv1d=CausalConv((2*key+value)//tp,cfg.linear_conv_kernel)
        P.mark_sharded(self.conv1d.weight,splits=(key,key,value))
        self.A_log=P.mark_sharded(nn.Parameter(torch.rand((cfg.linear_value_heads or cfg.linear_heads)//tp).mul(15.99).add(.01).log()))
        self.dt_bias=P.mark_sharded(nn.Parameter(torch.ones((cfg.linear_value_heads or cfg.linear_heads)//tp)))
        self.norm=GatedNorm(cfg.linear_head_dim,cfg.norm_eps)
        self.out_proj=P.RowLinear(size,cfg.width)

    def forward(self,x,plan):
        if x.shape[1]!=1:raise ValueError('CQ GDN currently requires microbatch 1')
        cfg=self.cfg;tp=P.tp_size();hk=cfg.linear_heads//tp
        h=(cfg.linear_value_heads or cfg.linear_heads)//tp;d=cfg.linear_head_dim;size=hk*d
        flat=x[:,0]
        if cfg.qwen_fused_projections:
            raw,z,a,beta=self.in_proj_all(flat).split((2*size+h*d,h*d,h,h),-1)
        else:
            raw=self.in_proj_qkv(flat);z=self.in_proj_z(flat)
            a=self.in_proj_a(flat);beta=self.in_proj_b(flat)
        updates=raw.index_select(0,plan.content_order)
        convolved=self.conv1d(updates[None],plan)[0]
        q,k,v=convolved.split((size,size,h*d),-1)
        q,k,v=q.reshape(-1,hk,d),k.reshape(-1,hk,d),v.reshape(-1,h,d)
        # Query convolution reads the previous real-content cache; neither the
        # target nor other queries can influence this cache. Do not recover it
        # by subtracting the target from a rounded convolution output.
        raw_q=raw.index_select(0,plan.query_source)[:,:size]
        weight=self.conv1d.weight[:size,0]
        query=raw_q*weight[:,-1]
        history=torch.cat((updates[:,:size],updates.new_zeros(1,size)),dim=0)
        for j in range(cfg.linear_conv_kernel-1):
            query=query+history.index_select(0,plan.query_history[:,j])*weight[:,j]
        query=F.silu(query).reshape(-1,hk,d)
        if h!=hk:
            q,k,query=(t.repeat_interleave(h//hk,dim=1) for t in (q,k,query))
        q=torch.cat((q,query),dim=0).index_select(0,plan.event_q)[None]
        zeros=k.new_zeros(1,h,d)
        k=torch.cat((k,zeros),dim=0).index_select(0,plan.event_write)[None]
        v=torch.cat((v,zeros),dim=0).index_select(0,plan.event_write)[None]
        control_dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
        a=a.to(control_dtype)
        g=-self.A_log.to(control_dtype).exp()*F.softplus(a+self.dt_bias.to(control_dtype))
        beta=beta.sigmoid().to(control_dtype)
        g=g.index_select(0,plan.content_order);beta=beta.index_select(0,plan.content_order)
        control_zero=g.new_zeros(1,h)
        g=torch.cat((g,control_zero)).index_select(0,plan.event_write)[None]
        beta=torch.cat((beta,control_zero)).index_select(0,plan.event_write)[None]
        if x.device.type=='npu':
            from . import delta_ascendc
            y=delta_ascendc.delta(q,k,v,g,beta,use_qk_l2norm_in_kernel=True,
                cu_seqlens=plan.event_cu,cu_seqlens_cpu=plan.event_bounds)[0]
        else:y=recurrent_delta(q,k,v,g,beta,plan.event_bounds)
        y=y[0].index_select(0,plan.output_event)
        selected=x.new_zeros(plan.hidden_length,h,d).index_copy(0,plan.output_hidden,y)
        z=z.reshape(-1,h,d)
        return self.out_proj(self.norm(selected,z).flatten(-2))[:,None]


class GatedAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.q_proj=P.ColumnLinear(cfg.width,2*cfg.heads*cfg.head_dim)
        if cfg.qwen_fused_projections:
            self.kv_proj=P.ColumnLinear(cfg.width,2*cfg.kv_heads*cfg.head_dim,stride=2)
        else:
            self.k_proj=P.ColumnLinear(cfg.width,cfg.kv_heads*cfg.head_dim)
            self.v_proj=P.ColumnLinear(cfg.width,cfg.kv_heads*cfg.head_dim)
        self.o_proj=P.RowLinear(cfg.heads*cfg.head_dim,cfg.width)
        self.q_norm=RMSNorm(cfg.head_dim,cfg.norm_eps);self.k_norm=RMSNorm(cfg.head_dim,cfg.norm_eps)

    def forward(self,x,rope,plan,content_length,reference=False):
        from .mainline import attend
        cfg=self.cfg;s=content_length
        g=len(plan.generation_positions)-plan.prefix if len(plan.generation_positions) else 0
        keys=s+g;n,b,_=x.shape
        q,gate=self.q_proj(x).reshape(n,b,cfg.heads//P.tp_size(),2*cfg.head_dim).chunk(2,-1)
        q=self.q_norm(q,cfg.fused and not reference)
        k,v=(self.kv_proj(x[:keys]).chunk(2,-1) if cfg.qwen_fused_projections else
             (self.k_proj(x[:keys]),self.v_proj(x[:keys])))
        k=self.k_norm(k.reshape(keys,b,cfg.kv_heads//P.tp_size(),cfg.head_dim),cfg.fused and not reference)
        v=v.reshape(keys,b,cfg.kv_heads//P.tp_size(),cfg.head_dim)
        dim=int(cfg.head_dim*cfg.partial_rotary_factor)
        def rotate(t,cos,sin):
            return torch.cat((K.rotary(t[...,:dim].contiguous(),cos,sin,cfg.fused and not reference),t[...,dim:]),-1)
        q=rotate(q,*rope);k=rotate(k,rope[0][:keys],rope[1][:keys])
        cq=q if not g else torch.cat((q[:s],q[keys:]))
        y=attend(cq,k[:s],v[:s],plan.cq_mask,plan.cq_valid,None if reference else plan.cq_plan,cfg.fused and not reference)
        if g:
            kg=torch.cat((k[:plan.prefix],k[s:keys]));vg=torch.cat((v[:plan.prefix],v[s:keys]))
            gy=attend(q[s:keys],kg,vg,plan.gen_suffix_mask,plan.gen_valid[:,plan.prefix:],
                None if reference else plan.gen_plan,cfg.fused and not reference)
            y=torch.cat((y[:s],gy,y[s:]))
        return self.o_proj(y*gate.flatten(-2).sigmoid())


class QwenBlock(nn.Module):
    def __init__(self,cfg,index):
        super().__init__();self.cfg=cfg
        self.input_layernorm=RMSNorm(cfg.width,cfg.norm_eps)
        self.post_attention_layernorm=RMSNorm(cfg.width,cfg.norm_eps)
        if (index+1)%cfg.full_attention_interval:
            self.linear_attn=GatedDelta(cfg)
        else:self.self_attn=GatedAttention(cfg)
        self.mlp=MLP(cfg)

    def forward(self,hidden,rope,plan,content_length,reference=False):
        norm=self.input_layernorm(hidden,self.cfg.fused and not reference)
        mixed=(self.linear_attn(norm,plan.delta_plan) if hasattr(self,'linear_attn') else
               self.self_attn(norm,rope,plan,content_length,reference))
        hidden=hidden+mixed
        return hidden+self.mlp(self.post_attention_layernorm(hidden,self.cfg.fused and not reference))


def asset_path(path):
    path=Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1]/path


def cast_for_training(model,device='cpu'):
    # Official Qwen checkpoint keeps these recurrence-sensitive parameters in
    # FP32. Preserve their values before casting the bulk weights to BF16.
    keep={name:p.detach().clone() for name,p in model.named_parameters()
          if name.endswith(('.linear_attn.A_log','.linear_attn.norm.weight'))}
    rope_bands={name:b.detach().clone() for name,b in model.named_buffers()
                if name.endswith('.rope.bands')}
    model.to(device=device,dtype=torch.bfloat16)
    for name,p in model.named_parameters():
        if name in keep:p.data=keep[name].to(device=device,dtype=torch.float32)
    for name,b in model.named_buffers():
        if name in rope_bands:b.data=rope_bands[name].to(device=device,dtype=torch.float32)
    return model


def asset_manifest(path,verify=False):
    from .prepare_qwen35 import sha256
    root=asset_path(path);data=json.loads((root/'manifest.json').read_text())
    allowed={('Qwen/Qwen3.5-0.8B','2fc06364715b967f1860aea9cf38778875588b17'),
             ('Qwen/Qwen3.5-9B','c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
             # Base checkpoint: identical architecture, vocab and merges; it only
             # lacks the four post-training tokens (<think>, tool responses).
             ('Qwen/Qwen3.5-9B-Base','68c46c4b3498877f3ef123c856ecfde50c39f404')}
    if (data['model_id'],data['revision']) not in allowed:raise ValueError('Unexpected Qwen3.5 checkpoint identity')
    if verify:
        for name,digest in data['files'].items():
            if sha256(root/name)!=digest:raise ValueError(f'Qwen3.5 asset checksum changed: {name}')
    return data


def unfused_parameters(name,tensor,cfg):
    """Writable checkpoint/gradient views preserving official tensor names."""
    clean=name.replace('.impl.','.')
    if clean.endswith('.mlp.gate_up_proj.weight'):
        return {clean.replace('gate_up_proj',label):P.mark_sharded(view)
                for label,view in zip(('gate_proj','up_proj'),tensor.chunk(2,0))}
    if clean.endswith('.linear_attn.in_proj_all.weight'):
        key=cfg.linear_heads*cfg.linear_head_dim;heads=cfg.linear_value_heads or cfg.linear_heads
        value=heads*cfg.linear_head_dim
        views=tensor.split(tuple(x//P.tp_size() for x in (2*key+value,value,heads,heads)),0)
        result={clean.replace('in_proj_all',label):P.mark_sharded(view)
                for label,view in zip(('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b'),views)}
        P.mark_sharded(views[0],splits=(key,key,value))
        return result
    if clean.endswith('.self_attn.kv_proj.weight'):
        return {clean.replace('kv_proj',label):P.mark_sharded(view)
                for label,view in zip(('k_proj','v_proj'),tensor.chunk(2,0))}
    return {clean:tensor}


def load_pretrained(model,path):
    """Strictly load all official language and vision tensors into TP views."""
    from safetensors import safe_open
    root=asset_path(path);manifest=asset_manifest(root)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    target={}
    parameters={key:value for name,tensor in model.named_parameters()
                for key,value in unfused_parameters(name,tensor,model.cfg).items()}
    for clean,tensor in parameters.items():
        if not isinstance(tensor,torch.Tensor):continue
        if clean.startswith('embedding.'):key='model.language_model.embed_tokens.'+clean.removeprefix('embedding.')
        elif clean=='lm_head.weight':key='lm_head.weight'
        elif clean.startswith('layers.'):key='model.language_model.'+clean
        elif clean.startswith('final_norm.'):key='model.language_model.norm.'+clean.removeprefix('final_norm.')
        elif clean.startswith('patch_embed.visual.'):key='model.visual.'+clean.removeprefix('patch_embed.visual.')
        else:continue
        target[key]=tensor
    expected={k for k in index if k.startswith(('model.language_model.','model.visual.')) or (k=='lm_head.weight' and not model.cfg.tie_word_embeddings)}
    if expected!=set(target):
        raise ValueError(f'Official checkpoint key mismatch: missing={sorted(expected-set(target))}, extra={sorted(set(target)-expected)}')
    for shard in sorted({index[k] for k in target}):
        with safe_open(str(root/shard),framework='pt',device='cpu') as source,torch.no_grad():
            for key,tensor in target.items():
                if index[key]!=shard:continue
                value=P.checkpoint_shard(source.get_tensor(key),tensor)
                if value.shape!=tensor.shape:raise ValueError(f'Checkpoint shape mismatch: {key}')
                tensor.copy_(value)
    return dict(**manifest,loaded_tensors=len(target),loaded_parameters=sum(x.numel() for x in target.values()))
