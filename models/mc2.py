"""Opt-in row-parallel MC2 forward with the full training chain rule.

TorchNPU's MatmulAllReduce has no training autograd contract. This adapter
supplies dX/dW and retains MCore's FP32 accumulation and gradient-ready hooks.
Bias is added after the reduction, exactly as in RowParallelLinear.
"""
import torch
import torch.distributed as dist


def topology_supports_mc2(tp,local_rank,local_world,mapping,topology):
    """MC2 needs a complete local HCCS TP group; fragmented PHB pairs cannot launch it."""
    if tp==1 or local_world%tp:return False
    physical={}
    for line in mapping.splitlines():
        fields=line.split()
        if len(fields)>=3 and fields[0].isdigit() and fields[2].isdigit():
            physical[int(fields[2])]='NPU'+fields[0]
    lines=[line.split() for line in topology.splitlines() if line.strip()]
    if not lines:return False
    columns=[name for name in lines[0] if name.startswith('NPU')]
    rows={line[0]:dict(zip(columns,line[1:])) for line in lines[1:] if line[0] in columns}
    start=local_rank//tp*tp
    group=[physical.get(i) for i in range(start,start+tp)]
    return all(a is not None and b is not None and (a==b or rows.get(a,{}).get(b)=='HCCS')
               for a in group for b in group)


def configure_mc2(tp):
    import json
    import os
    import subprocess
    if os.environ.get('PRETRAIN_MC2_ROW','0')!='1':return
    reason='HCCS TP group'
    try:
        supported=topology_supports_mc2(tp,int(os.environ.get('LOCAL_RANK',0)),
            int(os.environ.get('LOCAL_WORLD_SIZE',1)),
            subprocess.check_output(['npu-smi','info','-m'],text=True,timeout=10),
            subprocess.check_output(['npu-smi','info','-t','topo'],text=True,timeout=10))
    except (OSError,subprocess.SubprocessError):
        supported=False
    flag=torch.tensor(int(supported),device='npu')
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    enabled=bool(flag.item());os.environ['PRETRAIN_MC2_ROW']='1' if enabled else '0'
    if not enabled:reason='TP1, non-HCCS fragments, or unavailable topology; use ordinary TP collectives'
    if dist.get_rank()==0:print(json.dumps(dict(mc2=dict(requested=True,enabled=enabled,reason=reason))),flush=True)


class _RowAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,hcom):
        ctx.save_for_backward(x,weight)
        ctx.direct=hasattr(weight,'main_grad') and hasattr(weight,'grad_added_to_main_grad')
        if x.device.type=='npu':
            import torch_npu
            y=torch_npu.npu_mm_all_reduce_base(x.reshape(-1,x.shape[-1]).contiguous(),weight.T,hcom)
            return y.reshape(*x.shape[:-1],weight.shape[0])
        if hcom is not None:raise ValueError('CPU numerical reference is local-only')
        return torch.nn.functional.linear(x,weight)

    @staticmethod
    def backward(ctx,grad):
        x,weight=ctx.saved_tensors
        g=grad.reshape(-1,grad.shape[-1]).contiguous();a=x.reshape(-1,x.shape[-1]).contiguous()
        dx=(g@weight).reshape_as(x) if ctx.needs_input_grad[0] else None
        dw=None
        if ctx.needs_input_grad[1]:
            if ctx.direct:
                if weight.main_grad.dtype!=torch.float32:raise TypeError('MC2 training requires FP32 main_grad')
                if x.device.type=='npu':
                    from models.kernels import fused_wgrad
                    fused_wgrad(a,g,weight.main_grad)
                else:weight.main_grad.addmm_(g.float().T,a.float())
                dw=torch.zeros_like(weight) if getattr(weight,'zero_out_wgrad',False) else torch.empty_like(weight)
                weight.grad_added_to_main_grad=True
            else:dw=g.T@a
        return dx,dw,None


def row_forward(module,x):
    if module.sequence_parallel or not module.input_is_parallel:
        raise ValueError('MC2 row adapter requires full-sequence, sharded-feature input')
    if not hasattr(module,'_cq_hcom'):
        import training.mcore as P
        group=P.state().get_tensor_model_parallel_group()
        module._cq_hcom=group._get_backend(torch.device('npu')).get_hccl_comm_name(dist.get_rank())
    output=_RowAllReduce.apply(x,module.weight,module._cq_hcom)
    if module.skip_bias_add:return output,module.bias
    return (output+module.bias if module.bias is not None else output),None
