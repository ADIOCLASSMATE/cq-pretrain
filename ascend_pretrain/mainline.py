"""Static data-side layouts for exact CQ_IT / Bi_I+CQ_T joint training.

Bi follows the interleaved MaskImage reference: all supervised target images
are masked in the generation branch; the clean understanding branch is kept
separate. A causal prefix before the first image is shared algebraically.
"""
from dataclasses import dataclass, fields
import math
import os
import torch
from .attention_plan import plan_attention


def safe_mask(allowed):
    valid=allowed.any(-1)
    mask=~allowed
    mask[:,:,0] &= valid
    return mask[:,None].contiguous(),valid


def document_cq_mask(batch,query_positions):
    """Exact CQ masks; compare sigma only inside each independent document.

    The current NPU attention adapter still consumes a dense Boolean mask,
    but constructing full-pack int64 comparison grids needlessly costs CPU
    time and several extra full-size temporary masks at an 8K pack length.
    """
    s=batch.ids.shape[1];segments=batch.segments[0];sigma=batch.sigma[0]
    query_segments=segments[query_positions]
    mask=torch.ones(s+len(query_positions),s,dtype=torch.bool)
    valid=torch.zeros(s+len(query_positions),dtype=torch.bool)
    for segment in segments.unique().tolist():
        if segment<0:continue
        positions=(segments==segment).nonzero().flatten()
        start,stop=int(positions[0]),int(positions[-1])+1
        if stop-start!=len(positions):
            raise ValueError('A packed document must occupy contiguous content positions')
        sigmas=sigma[start:stop]
        mask[start:stop,start:stop]=sigmas[None,:]>sigmas[:,None]
        valid[start:stop]=True
        rows=(query_segments==segment).nonzero().flatten()
        qsigmas=sigma[query_positions[rows]]
        mask[s+rows,start:stop]=sigmas[None,:]>=qsigmas[:,None]
        valid[s+rows]=qsigmas>sigmas.min()
    # Same finite sentinel as safe_mask; invalid rows are zeroed after FA.
    mask[~valid,0]=False
    return mask[None,None],valid[None]


@dataclass
class MainlinePlan:
    method: str
    prefix: int
    query_positions: torch.Tensor
    query_image: torch.Tensor
    query_rope: torch.Tensor
    generation_positions: torch.Tensor
    generation_image: torch.Tensor
    generation_rope: torch.Tensor
    output_positions: torch.Tensor
    condition_indices: torch.Tensor
    generation_condition: torch.Tensor
    cq_mask: torch.Tensor
    cq_valid: torch.Tensor
    gen_mask: torch.Tensor
    gen_valid: torch.Tensor
    gen_suffix_mask: torch.Tensor
    cq_plan: object = None
    gen_plan: object = None
    output_size: int = 0
    delta_plan: object = None
    rope_cos: object = None
    rope_sin: object = None

    def to(self,device,non_blocking=False):
        values={f.name:(getattr(self,f.name).to(device,non_blocking=non_blocking) if isinstance(getattr(self,f.name),torch.Tensor)
                        else getattr(self,f.name)) for f in fields(self) if not f.name.endswith('_plan')}
        out=MainlinePlan(**values)
        for name in ('cq','gen'):
            plan=getattr(self,name+'_plan')
            if plan is not None:
                setattr(out,name+'_plan',plan.bind(out.gen_suffix_mask if name=='gen' else out.cq_mask))
        if self.delta_plan is not None:out.delta_plan=self.delta_plan.to(device,non_blocking=non_blocking)
        return out

    def pin_memory(self,device):
        from dataclasses import replace
        values={f.name:(getattr(self,f.name).pin_memory(device=device) if isinstance(getattr(self,f.name),torch.Tensor)
                        and getattr(self,f.name).numel() else getattr(self,f.name)) for f in fields(self)
                if not f.name.endswith('_plan')}
        out=MainlinePlan(**values)
        for name in ('cq','gen'):
            plan=getattr(self,name+'_plan')
            if plan is not None:
                mask=out.gen_suffix_mask if name=='gen' else out.cq_mask
                setattr(out,name+'_plan',replace(plan,mask_pointer=mask.data_ptr(),mask_version=mask._version,owner=mask,tile_masks=()))
        if self.delta_plan is not None:out.delta_plan=self.delta_plan.pin_memory(device)
        return out

    def __deepcopy__(self,memo):
        import copy
        values={f.name:copy.deepcopy(getattr(self,f.name),memo) for f in fields(self)
                if not f.name.endswith('_plan')}
        out=MainlinePlan(**values);memo[id(self)]=out
        for name in ('cq','gen'):
            plan=getattr(self,name+'_plan')
            if plan is not None:
                setattr(out,name+'_plan',plan.bind(out.gen_suffix_mask if name=='gen' else out.cq_mask))
        out.delta_plan=copy.deepcopy(self.delta_plan,memo)
        return out


def finalize_plan(batch,cfg,plan):
    from .qwen35_layout import build_delta_plan,rope_tables
    plan.delta_plan=build_delta_plan(batch,plan,cfg)
    plan.rope_cos,plan.rope_sin=rope_tables(batch,plan,cfg)
    return plan


def build_plan(batch,cfg):
    """CPU-only preparation for the fixed-shape synthetic interleaved benchmark.

    The fake loader uses matching span locations on each batch row. Segment
    IDs and positions remain per row. Real variable layouts need their own
    collation/bucketing; no mask or position tensors are built in forward.
    """
    if cfg.vision_encoder=='qwen35':
        return build_native_plan(batch,cfg)
    if batch.ids.device.type!='cpu':
        raise ValueError('Prepare mainline plans on CPU before device transfer')
    b,s=batch.ids.shape
    image=batch.image_queries
    if not torch.equal(image,image[:1].expand_as(image)):
        raise ValueError('The synthetic layout requires aligned image spans within a microbatch')
    idx=torch.arange(s)
    starts=(image[0] & ~torch.nn.functional.pad(image[0,:-1],(1,0),value=False)).nonzero().flatten()
    n=cfg.image_tokens;q=cfg.image_condition_queries
    side=math.isqrt(n);qs=math.isqrt(q)
    local=torch.arange(q)
    # Canonical cell centers, matching Extension.dream_condition at Q64/Q256.
    row=(local//qs+.5)*(side/qs)-.5
    col=(local%qs+.5)*(side/qs)-.5
    anchors=(local//qs*side//qs)*side+(local%qs*side//qs)
    selected=(starts[:,None]+anchors).flatten()
    # Only supervised positions need a readout; see build_native_plan.
    text=batch.text_indices[(batch.text_indices<s)&~image[0][batch.text_indices.clamp(max=s-1)]]
    if not len(text):text=idx[(~image[0]) & batch.segments.ge(0).any(0)]
    # A fully empty rank uses a one-token zero-loss graph to keep DDP hooks.
    if not len(text) and not len(starts):text=idx[:1]
    query_positions=torch.cat((text,selected)) if cfg.method=='CQ_IT' else text
    query_image=image[:,query_positions]
    positions=batch.row_col[:,query_positions].float().clone()
    # Use each image's actual RoPE origin, preserving intervening image grids.
    qrope=(torch.stack([batch.row_col[:,int(start)].float()[:,None,:]+torch.stack((row,col))[:,:,None]
                       for start in starts],dim=1).reshape(2,-1,b) if len(starts) else torch.empty(2,0,b))
    if cfg.method=='CQ_IT':positions[:,len(text):]=qrope
    cq_same=(batch.segments[:,:,None]==batch.segments[:,None,:]) & batch.segments[:,:,None].ge(0)
    c_allowed=(batch.sigma[:,None,:]<=batch.sigma[:,:,None]) & cq_same
    q_allowed=(batch.sigma[:,None,:]<batch.sigma[:,query_positions,None])
    q_allowed &= (batch.segments[:,query_positions,None]==batch.segments[:,None,:]) & batch.segments[:,query_positions,None].ge(0)
    cq_mask,cq_valid=safe_mask(torch.cat((c_allowed,q_allowed),1))
    conditions=(selected[None]+torch.arange(b)[:,None]*s).reshape(b*len(starts),q)
    # A physically single masked-image sequence, compacting only target blocks.
    keep=~image[0].clone();keep[selected]=True
    generation=idx[keep]
    generation=generation[generation<=selected[-1]] if len(selected) and cfg.method=='Bi_I+CQ_T' else generation[:0]
    g_image=image[:,generation]
    grope=batch.row_col[:,generation].float().clone()
    glocal=image[0,generation].nonzero().flatten()
    if len(generation):grope[:,glocal]=qrope
    prefix=int(starts[0]) if len(generation) else 0
    gsegment=batch.segments[:,generation]
    block=(g_image & ~torch.nn.functional.pad(g_image[:,:-1],(1,0),value=False)).long().cumsum(-1)
    same_image=g_image[:,:,None]&g_image[:,None,:]&(block[:,:,None]==block[:,None,:])
    gidx=torch.arange(len(generation))
    g_allowed=(gidx[None,:]<=gidx[:,None])[None] | same_image
    g_allowed &= (gsegment[:,:,None]==gsegment[:,None,:]) & gsegment[:,:,None].ge(0)
    gen_mask,gen_valid=(safe_mask(g_allowed) if len(generation) else
                       (torch.empty(b,1,0,0,dtype=torch.bool),torch.empty(b,0,dtype=torch.bool)))
    gen_conditions=((glocal[None]+torch.arange(b)[:,None]*len(generation)).reshape(b*len(starts),q)
                    if len(generation) else torch.empty(0,q,dtype=torch.long))
    output_positions=(query_positions[None]+torch.arange(b)[:,None]*s).flatten()
    plan=MainlinePlan(cfg.method,prefix,query_positions,query_image,positions,generation,g_image,grope,
        output_positions,conditions,gen_conditions,cq_mask,cq_valid,gen_mask,gen_valid,
        gen_mask[:,:,prefix:].contiguous())
    tile=int(os.environ.get('PRETRAIN_ATTENTION_TILE','0'))
    if tile<0:raise ValueError('PRETRAIN_ATTENTION_TILE must be nonnegative')
    if tile:
        plan.cq_plan=plan_attention(cq_mask,tile)
        # Generation prefix outputs are shared with clean content; skip them.
        if len(generation):plan.gen_plan=plan_attention(plan.gen_suffix_mask,tile)
    return finalize_plan(batch,cfg,plan)


def build_native_plan(batch,cfg):
    """Native image content and fixed-Q conditions occupy separate index spaces.

    The physical pack contains only merged ViT content tokens. Image queries
    write to virtual output slots after S, so even N<Q images retain all Q
    independent conditions without repeating/overwriting physical outputs.
    """
    if batch.ids.device.type!='cpu' or batch.ids.shape[0]!=1:
        raise ValueError('Prepare native-resolution plans on CPU with micro_batch=1')
    _,s=batch.ids.shape;q=cfg.image_condition_queries;qs=math.isqrt(q)
    image=batch.image_queries[0];idx=torch.arange(s)
    # Only supervised positions need a readout. Chat scaffolding (role headers,
    # generated instructions) is fed as content but never predicted, so building
    # a query for it costs a full 32-layer forward/backward and is then dropped.
    # CQ queries never write recurrent/KV state, so pruning them is exactly
    # equivalent for every content representation -- not an approximation.
    text=batch.text_indices[(batch.text_indices<s)&~image[batch.text_indices.clamp(max=s-1)]]
    if not len(text):text=idx[(~image)&batch.segments[0].ge(0)]
    spans=[indices[indices.ge(0)] for indices in batch.image_indices]
    if spans and (batch.vision is None or len(batch.vision.grid_thw)!=len(spans)):
        raise ValueError('Native image spans require matching vision geometry')
    if not len(text) and not spans:text=idx[:1]
    local=torch.arange(q);anchors=[];ropes=[]
    for indices,(_,h,w) in zip(spans,batch.vision.grid_thw if spans else ()):
        h,w=h//2,w//2
        if len(indices)!=h*w or not torch.equal(indices,torch.arange(int(indices[0]),int(indices[0])+h*w)):
            raise ValueError('An image must occupy one contiguous merged spatial grid')
        anchors.append(indices[0].repeat(q))
        offsets=torch.stack(((local//qs+.5)*(h/qs)-.5,(local%qs+.5)*(w/qs)-.5))
        ropes.append(batch.row_col[:,indices[0]].float()[:,None]+offsets[:,:,None])
    selected=torch.cat(anchors) if anchors else torch.empty(0,dtype=torch.long)
    qrope=torch.cat(ropes,dim=1) if ropes else torch.empty(2,0,1)
    query_positions=torch.cat((text,selected)) if cfg.method=='CQ_IT' else text
    query_image=batch.image_queries[:,query_positions]
    query_rope=batch.row_col[:,query_positions].float().clone()
    if cfg.method=='CQ_IT':query_rope[:,len(text):]=qrope
    if cfg.preserve_documents:
        cq_mask,cq_valid=document_cq_mask(batch,query_positions)
    else:
        same=(batch.segments[:,:,None]==batch.segments[:,None,:])&batch.segments[:,:,None].ge(0)
        c_allowed=(batch.sigma[:,None,:]<=batch.sigma[:,:,None])&same
        q_allowed=(batch.sigma[:,None,:]<batch.sigma[:,query_positions,None])
        q_allowed&=(batch.segments[:,query_positions,None]==batch.segments[:,None,:])&batch.segments[:,query_positions,None].ge(0)
        cq_mask,cq_valid=safe_mask(torch.cat((c_allowed,q_allowed),1))
    conditions=torch.arange(s,s+len(spans)*q).reshape(len(spans),q)
    output_positions=torch.cat((text,conditions.flatten())) if cfg.method=='CQ_IT' else text
    pieces=[];cursor=0
    if cfg.method=='Bi_I+CQ_T':
        for indices,anchor in zip(spans,anchors):
            pieces.extend((idx[cursor:int(indices[0])],anchor));cursor=int(indices[-1])+1
    generation=torch.cat(pieces) if pieces else torch.empty(0,dtype=torch.long)
    g_image=batch.image_queries[:,generation]
    grope=batch.row_col[:,generation].float().clone()
    glocal=g_image[0].nonzero().flatten()
    if len(generation):grope[:,glocal]=qrope
    prefix=int(spans[0][0]) if len(generation) else 0
    gsegment=batch.segments[:,generation]
    block=(g_image&~torch.nn.functional.pad(g_image[:,:-1],(1,0),value=False)).long().cumsum(-1)
    same_image=g_image[:,:,None]&g_image[:,None,:]&(block[:,:,None]==block[:,None,:])
    gidx=torch.arange(len(generation))
    allowed=(gidx[None,:]<=gidx[:,None])[None]|same_image
    allowed&=(gsegment[:,:,None]==gsegment[:,None,:])&gsegment[:,:,None].ge(0)
    gen_mask,gen_valid=(safe_mask(allowed) if len(generation) else
        (torch.empty(1,1,0,0,dtype=torch.bool),torch.empty(1,0,dtype=torch.bool)))
    gen_condition=glocal.reshape(len(spans),q) if len(generation) else torch.empty(0,q,dtype=torch.long)
    plan=MainlinePlan(cfg.method,prefix,query_positions,query_image,query_rope,generation,g_image,grope,
        output_positions,conditions,gen_condition,cq_mask,cq_valid,gen_mask,gen_valid,
        gen_mask[:,:,prefix:].contiguous(),output_size=s+len(spans)*q)
    tile=int(os.environ.get('PRETRAIN_ATTENTION_TILE','0'))
    if tile<0:raise ValueError('PRETRAIN_ATTENTION_TILE must be nonnegative')
    if tile:
        plan.cq_plan=plan_attention(cq_mask,tile)
        if len(generation):plan.gen_plan=plan_attention(plan.gen_suffix_mask,tile)
    return finalize_plan(batch,cfg,plan)


def attend(q,k,v,mask,valid,plan,fused):
    from . import kernels as K
    if plan is None:return K.attention(q,k,v,mask,valid,fused)
    from .attention_plan import tiled_attention
    return tiled_attention(q,k,v,mask,valid,plan,fused)


def packed_layer(layer,hidden,rope,plan,content_length,reference=False):
    return layer(hidden,rope,plan,content_length,reference)


def backbone(model,batch,reference=False,vision_features=None):
    from . import kernels as K
    from torch.utils.checkpoint import checkpoint
    cfg=model.cfg;plan=batch.mainline;b,s=batch.ids.shape
    if plan is None or plan.method!=cfg.method:
        raise ValueError('Batch must be prepared for the selected mainline method')
    content=model.embedding(batch.ids).reshape(-1,cfg.width)
    if len(batch.image_indices):
        if cfg.vision_encoder=='qwen35':
            if batch.vision is None:raise ValueError('Qwen vision input is missing')
            patches=model.patch_embed(batch.vision) if vision_features is None else vision_features
            content=content.index_copy(0,batch.vision.content_indices,patches)
        else:
            patches=model.patch_embed(batch.pixels).flatten(2).transpose(1,2).contiguous()
            content=content.index_copy(0,batch.image_indices.flatten(),patches.reshape(-1,cfg.width))
    content=content.reshape(b,s,cfg.width).transpose(0,1)
    query_ids=torch.where(plan.query_image,cfg.image_mask_token_id,cfg.text_mask_token_id)
    query=model.embedding(query_ids).transpose(0,1)
    generation=None
    if cfg.method=='Bi_I+CQ_T' and len(plan.generation_positions):
        gids=batch.ids[:,plan.generation_positions].masked_fill(plan.generation_image,cfg.image_mask_token_id)
        generation=model.embedding(gids).transpose(0,1)
    parts=(content,query) if generation is None else (content,generation[plan.prefix:],query)
    hidden=torch.cat(parts).contiguous()
    rope=(plan.rope_cos.to(hidden.dtype),plan.rope_sin.to(hidden.dtype))
    for index,layer in enumerate(model.layers):
        recompute=cfg.checkpoint_backbone or index>=cfg.layers-cfg.checkpoint_backbone_layers
        if (index+1)%cfg.full_attention_interval==0 and index//cfg.full_attention_interval<cfg.keep_attention_activations:
            recompute=False
        if recompute and model.training and torch.is_grad_enabled():
            hidden=checkpoint(packed_layer,layer,hidden,rope,plan,s,reference,use_reentrant=True)
        else:hidden=packed_layer(layer,hidden,rope,plan,s,reference)
    keys=s+(len(plan.generation_positions)-plan.prefix if generation is not None else 0)
    query=model.final_norm(hidden[keys:],cfg.fused and not reference)
    output=query.new_zeros((plan.output_size or b*s,cfg.width)).index_copy(0,plan.output_positions,query.transpose(0,1).reshape(-1,cfg.width))
    if generation is not None:
        generation=torch.cat((hidden[:plan.prefix],hidden[s:keys]))
        condition=generation.transpose(0,1).reshape(-1,cfg.width)[plan.generation_condition]
        condition=model.final_norm(condition,cfg.fused and not reference)
        output=output.index_copy(0,plan.condition_indices.flatten(),condition.reshape(-1,cfg.width))
    return output
