"""Opt-in row-parallel MC2 forward with the full training chain rule.

TorchNPU's MatmulAllReduce has no training autograd contract. This adapter
supplies dX/dW and retains MCore's FP32 accumulation and gradient-ready hooks.
Bias is added after the reduction, exactly as in RowParallelLinear.
"""
import torch
import torch.distributed as dist


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
                    from .kernels import fused_wgrad
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
        from . import parallel as P
        group=P.state().get_tensor_model_parallel_group()
        module._cq_hcom=group._get_backend(torch.device('npu')).get_hccl_comm_name(dist.get_rank())
    output=_RowAllReduce.apply(x,module.weight,module._cq_hcom)
    if module.skip_bias_add:return output,module.bias
    return (output+module.bias if module.bias is not None else output),None
