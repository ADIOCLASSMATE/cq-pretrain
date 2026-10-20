"""Exact query tiling: remove only key tails forbidden by every row in a tile."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class AttentionPlan:
    tiles: tuple
    mask_pointer: int
    mask_version: int
    shape: tuple
    owner: object = None
    tile_masks: tuple = ()

    def matches(self, mask):
        return (self.owner is mask and mask._version==self.mask_version
                and tuple(mask.shape)==self.shape)

    def bind(self, mask):
        if tuple(mask.shape)!=self.shape or mask.dtype!=torch.bool:
            raise ValueError('Bind a plan only to a same-shape copy of its original mask')
        masks=tuple(mask[:,:,start:end,:key_end].contiguous() for start,end,key_end in self.tiles)
        # Reading a just-enqueued NPU allocation's pointer can drain the host
        # launch queue. Python ownership plus version/shape is a stricter and
        # sufficient identity check; no device pointer is needed in forward.
        pointer=mask.data_ptr() if mask.device.type=='cpu' else 0
        return AttentionPlan(self.tiles,pointer,mask._version,tuple(mask.shape),owner=mask,tile_masks=masks)


def plan_attention(mask, chunk=1024):
    """Data-side CPU preparation, valid for arbitrary sigma/document patterns.

    All batches and heads share a conservative bound. An irregular mask may
    yield full key lengths; it is never approximated by a causal/band mask.
    """
    if mask.device.type!='cpu' or mask.dtype!=torch.bool or mask.ndim!=4 or chunk<1:
        raise ValueError('Prepare the plan on a CPU boolean [B,H,Q,K] mask')
    tiles=[]
    for start in range(0,mask.shape[-2],chunk):
        end=min(start+chunk,mask.shape[-2])
        visible=(~mask[:,:,start:end]).any(dim=(0,1,2)).nonzero().flatten()
        key_end=int(visible[-1])+1 if visible.numel() else 1
        tiles.append((start,end,key_end))
    return AttentionPlan(tuple(tiles),mask.data_ptr(),mask._version,tuple(mask.shape),owner=mask)


def tiled_attention(q,k,v,mask,valid_rows,plan,fused=True):
    from .kernels import attention,_ATTENTION_LAYOUT
    if not plan.matches(mask):
        # In-place edits/replaced masks invalidate data-side metadata.
        return attention(q,k,v,mask,valid_rows,fused)
    if fused and q.device.type=='npu' and _ATTENTION_LAYOUT=='SBH' and plan.tile_masks:
        result=_TiledFA.apply(q,k,v,plan)
        if valid_rows is not None:
            result=result*valid_rows.T.unsqueeze(-1).to(result.dtype)
        return result
    pieces=[]
    for index,(start,end,key_end) in enumerate(plan.tiles):
        tile_mask=plan.tile_masks[index] if plan.tile_masks else mask[:,:,start:end,:key_end].contiguous()
        valid=valid_rows[:,start:end] if valid_rows is not None else None
        pieces.append(attention(q[start:end],k[:key_end],v[:key_end],tile_mask,valid,fused))
    return torch.cat(pieces,dim=0)


class _TiledFA(torch.autograd.Function):
    """One output buffer and FP32 K/V gradient accumulation across exact tiles."""
    @staticmethod
    def forward(ctx,q,k,v,plan):
        import torch_npu
        ctx.shapes=(q.shape,k.shape,v.shape)
        ctx.heads=q.shape[2];ctx.scale=q.shape[-1]**-.5
        query,key,value=[t.flatten(-2).contiguous() for t in (q,k,v)]
        output=torch.empty_like(query)
        statistics=[];ctx.random=[];ctx.tiles=plan.tiles
        for (start,end,key_end),mask in zip(plan.tiles,plan.tile_masks):
            result=torch_npu.npu_fusion_attention(query[start:end],key[:key_end],value[:key_end],
                head_num=ctx.heads,input_layout='SBH',atten_mask=mask,scale=ctx.scale,keep_prob=1.,sparse_mode=0)
            output[start:end].copy_(result[0])
            statistics.extend((result[1],result[2]))
            ctx.random.append(result[4:7])
        ctx.save_for_backward(query,key,value,output,*statistics,*plan.tile_masks)
        return output

    @staticmethod
    def backward(ctx,grad):
        import torch_npu
        query,key,value,output,*saved=ctx.saved_tensors
        count=len(ctx.tiles);stats=saved[:2*count];masks=saved[2*count:]
        dq=torch.empty_like(query)
        dk=torch.zeros_like(key,dtype=torch.float32);dv=torch.zeros_like(value,dtype=torch.float32)
        for index,(start,end,key_end) in enumerate(ctx.tiles):
            seed,offset,numels=ctx.random[index]
            g=torch_npu.npu_fusion_attention_grad(query[start:end],key[:key_end],value[:key_end],
                grad[start:end].contiguous(),head_num=ctx.heads,input_layout='SBH',atten_mask=masks[index],
                softmax_max=stats[index*2],softmax_sum=stats[index*2+1],attention_in=output[start:end],
                scale_value=ctx.scale,keep_prob=1.,sparse_mode=0,seed=seed,offset=offset,numels=numels)
            dq[start:end].copy_(g[0]);dk[:key_end].add_(g[1].float());dv[:key_end].add_(g[2].float())
        return dq.reshape(ctx.shapes[0]),dk.to(key.dtype).reshape(ctx.shapes[1]),dv.to(value.dtype).reshape(ctx.shapes[2]),None
