"""CPU plans for write-once content and read-only CQ GatedDeltaNet streams.

Within each document, content writes follow sigma order. A tied image group
is fully written before any of its content readouts. CQ queries read before
the matching sigma group and never alter either recurrent or convolution state.
"""
from dataclasses import dataclass, fields
import torch


@dataclass
class DeltaPlan:
    content_order: torch.Tensor
    event_q: torch.Tensor
    event_write: torch.Tensor
    output_event: torch.Tensor
    output_hidden: torch.Tensor
    query_source: torch.Tensor
    query_history: torch.Tensor
    content_cu: torch.Tensor
    event_cu: torch.Tensor
    content_bounds: tuple
    event_bounds: tuple
    hidden_length: int

    def to(self, device, non_blocking=False):
        return DeltaPlan(**{f.name:(getattr(self,f.name).to(device,non_blocking=non_blocking)
            if isinstance(getattr(self,f.name),torch.Tensor) else getattr(self,f.name)) for f in fields(self)})

    def pin_memory(self,device):
        return DeltaPlan(**{f.name:(getattr(self,f.name).pin_memory(device=device)
            if isinstance(getattr(self,f.name),torch.Tensor) and getattr(self,f.name).numel()
            else getattr(self,f.name)) for f in fields(self)})


def build_delta_plan(batch,plan,cfg):
    if batch.ids.device.type!='cpu' or batch.ids.shape[0]!=1:
        raise ValueError('Prepare Qwen3.5 CQ plans on CPU, microbatch=1')
    s=batch.ids.shape[1]
    generation=plan.generation_positions.tolist()
    suffix=len(generation)-plan.prefix if generation else 0
    keys=s+suffix; nq=len(plan.query_positions)
    sigmas=batch.sigma[0].tolist();segments=batch.segments[0].tolist()
    # Source indices address the shared hidden=[C,G_suffix,Q] GEMM layout.
    branches=[[(i,segments[i],sigmas[i],True) for i in range(s) if segments[i]>=0]]
    if generation:
        # Masked-generation branch has its own causal/image-block order.
        gi=plan.generation_image[0].tolist(); gsig=[];group=-1;was_image=False
        for image in gi:
            if not image or not was_image:group+=1
            gsig.append(group);was_image=image
        branches.append([(i if i<plan.prefix else s+i-plan.prefix,segments[p],gsig[i],i>=plan.prefix)
                         for i,p in enumerate(generation) if segments[p]>=0])
    if not branches[0]:branches[0]=[(0,0,0,True)]  # zero-loss EOF graph
    query_by_doc={}
    for j,p in enumerate(plan.query_positions.tolist()):
        segment=segments[p]
        if segment<0:segment=0
        query_by_doc.setdefault(segment,{}).setdefault(sigmas[p],[]).append((j,keys+j))
    order=[];event_q=[];event_write=[];out_events=[];out_hidden=[]
    qsource=[];qhistory=[];cb=[0];eb=[0]
    for branch_id,branch in enumerate(branches):
        docs={}
        for item in branch:docs.setdefault(item[1],[]).append(item)
        for doc,items in docs.items():
            groups={}
            for item in sorted(items,key=lambda x:(x[2],x[0])):groups.setdefault(item[2],[]).append(item)
            history=[]
            for sigma,group_items in groups.items():
                if branch_id==0:
                    for _,hidden_idx in query_by_doc.get(doc,{}).get(sigma,[]):
                        qi=len(qsource);qsource.append(hidden_idx)
                        prev=history[-(cfg.linear_conv_kernel-1):] if cfg.linear_conv_kernel>1 else []
                        qhistory.append([-1]*(cfg.linear_conv_kernel-1-len(prev))+prev)
                        event_q.append(('q',qi));event_write.append(-1)
                        out_events.append(len(event_q)-1);out_hidden.append(hidden_idx)
                members=[]
                for hidden_idx,_,_,emit in group_items:
                    ci=len(order);order.append(hidden_idx);members.append((ci,hidden_idx,emit))
                    event_q.append(('c',ci));event_write.append(ci);history.append(ci)
                    if len(group_items)==1 and emit:
                        out_events.append(len(event_q)-1);out_hidden.append(hidden_idx)
                if len(group_items)>1:
                    # Bidirectional content visibility within the image group.
                    for ci,hidden_idx,emit in members:
                        event_q.append(('c',ci));event_write.append(-1)
                        if emit:out_events.append(len(event_q)-1);out_hidden.append(hidden_idx)
            cb.append(len(order));eb.append(len(event_q))
    nc=len(order)
    long=lambda xs:torch.tensor(xs,dtype=torch.long)
    eq=[i if kind=='c' else nc+i for kind,i in event_q]
    return DeltaPlan(long(order),long(eq),long([nc if i<0 else i for i in event_write]),
        long(out_events),long(out_hidden),long(qsource),
        long([[nc if i<0 else i for i in row] for row in qhistory]).reshape(len(qsource),cfg.linear_conv_kernel-1),
        long(cb),long(eb),tuple(cb),tuple(eb),keys+nq)


def rope_tables(batch,plan,cfg):
    """Official interleaved temporal/height/width partial RoPE for static images."""
    c=batch.row_col.float();ct=c[0].clone()
    for indices in batch.image_indices:
        indices=indices[indices.ge(0)]
        if len(indices):ct[indices]=c[0,indices[0]]
    q=plan.query_rope.float();qt=q[0].clone()
    qt=torch.where(plan.query_image[0,:,None],ct[plan.query_positions],qt)
    positions=[torch.stack((ct,c[0],c[1]))]
    if len(plan.generation_positions):
        g=plan.generation_rope.float();gt=g[0].clone()
        for indices in plan.generation_condition:
            if len(indices):gt[indices]=g[0,indices[0]]
        positions.append(torch.stack((gt,g[0],g[1]))[:,plan.prefix:])
    positions.append(torch.stack((qt,q[0],q[1])))
    pos=torch.cat(positions,dim=1).double()
    dim=int(cfg.head_dim*cfg.partial_rotary_factor)
    inv=cfg.rope_theta**(-torch.arange(0,dim,2,dtype=torch.float64)/dim)
    frequencies=pos[...,None]*inv
    angles=frequencies[0].clone()
    for axis in (1,2):
        stop=3*cfg.mrope_section[axis]
        angles[...,axis:stop:3]=frequencies[axis,...,axis:stop:3]
    angles=torch.cat((angles,angles),dim=-1)[:,:,None,:]
    return angles.cos().float(),angles.sin().float()
