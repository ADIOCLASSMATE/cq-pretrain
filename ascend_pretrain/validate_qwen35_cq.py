"""64-NPU CQ GDN fusion accuracy and isolation gate before real-data training."""
import argparse
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
from .config import tiny_config
from .data import fake_batch
from .qwen35 import GatedDelta,RMSNorm,recurrent_delta


def relerr(actual,expected):
    a=actual.detach().cpu().double();b=expected.detach().cpu().double()
    return float((a-b).norm()/b.norm().clamp_min(1e-8))


def error_detail(actual,expected):
    a=actual.detach().cpu().double();b=expected.detach().cpu().double()
    return dict(relative=relerr(a,b),max_absolute=float((a-b).abs().max()),
                reference_norm=float(b.norm()),reference_max=float(b.abs().max()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--key-heads',type=int,default=16);p.add_argument('--value-heads',type=int,default=16)
    a=p.parse_args()
    import torch_npu
    from . import delta_ascendc as backend
    torch.set_num_threads(1);device=f'npu:{os.environ["LOCAL_RANK"]}';torch.npu.set_device(device)
    backend.initialize();dist.init_process_group('hccl');rank=dist.get_rank();world=dist.get_world_size()
    if world!=64:raise ValueError('CQ fusion validation requires all 64 NPUs')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cases=[]
    # Check FP32 offset-gamma + BF16 activations, including dgamma.
    torch.manual_seed(1);norm=RMSNorm(256);norm.weight.data.normal_(0,.1)
    ref=copy.deepcopy(norm);norm=norm.to(device)
    x=torch.randn(31,256,dtype=torch.bfloat16,device=device,requires_grad=True)
    xr=x.detach().cpu().requires_grad_();y=norm(x);yr=ref(xr)
    u=torch.randn_like(y);(y*u).sum().backward();(yr*u.cpu()).sum().backward()
    norm_errors=dict(output=relerr(y,yr),dx=relerr(x.grad,xr.grad),dw=relerr(norm.weight.grad,ref.weight.grad))
    if max(norm_errors.values())>.025:raise AssertionError(norm_errors)
    for method in ('CQ_IT','Bi_I+CQ_T'):
        for length in (61,64,65):
            torch.manual_seed(101+rank)
            cfg=tiny_config(linear_heads=a.key_heads,linear_value_heads=a.value_heads,linear_head_dim=128,method=method)
            batch=fake_batch(cfg,length,1,2);plan=batch.mainline.delta_plan
            layer=GatedDelta(cfg);reference=copy.deepcopy(layer).double()
            # Preserve FP32 control/norm values, BF16 bulk weights as in training.
            layer=layer.to(device=device,dtype=torch.bfloat16)
            layer.A_log.data=reference.A_log.detach().float().to(device)
            layer.norm.weight.data=reference.norm.weight.detach().float().to(device)
            reference.load_state_dict({k:v.detach().cpu().double() for k,v in layer.state_dict().items()})
            x=torch.randn(plan.hidden_length,1,cfg.width,device=device,dtype=torch.bfloat16,requires_grad=True)
            xr=x.detach().cpu().double().requires_grad_();dp=plan.to(device)
            captured={};original=backend.delta
            def capture(*values,**kwargs):
                output=original(*values,**kwargs)
                for value in values:value.retain_grad()
                output[0].retain_grad();captured.update(values=values,output=output[0])
                return output
            with patch.object(backend,'delta',capture):y=layer(x,dp)
            reference_control={}
            def reference_capture(*values):
                values[3].retain_grad();reference_control['g']=values[3]
                return recurrent_delta(*values)
            with patch('ascend_pretrain.qwen35.recurrent_delta',reference_capture):yr=reference(xr,plan)
            u=torch.randn_like(y)
            (y*u).sum().backward();(yr*u.cpu().double()).sum().backward()
            errors={name:relerr(param.grad,dict(reference.named_parameters())[name].grad)
                    for name,param in layer.named_parameters()}
            errors.update(output=relerr(y,yr),dx=relerr(x.grad,xr.grad))
            details={name:error_detail(param.grad,dict(reference.named_parameters())[name].grad)
                     for name,param in layer.named_parameters()}
            # Isolate the fused recurrence from BF16 projections, convolution,
            # normalization and gate rounding in the complete layer.
            cr=[v.detach().cpu().double().requires_grad_() for v in captured['values']]
            cy=recurrent_delta(*cr,plan.event_bounds)
            cy.backward(captured['output'].grad.detach().cpu().double())
            core={name:error_detail(v.grad,r.grad) for name,v,r in zip(
                  ('q','k','v','g','beta'),captured['values'],cr)}
            # Signed token contributions can nearly cancel for A_log/dt_bias.
            # Report raw relative errors, and bound the numerical error against
            # the independent FP64 sum of absolute contributions (conditioned
            # gradient error). This also detects incorrect gradients when the
            # final reference gradient is near zero.
            with torch.no_grad():
                rg=reference_control['g'];dg=rg.grad
                ad=reference.in_proj_a(xr[:,0]).index_select(0,plan.content_order)+reference.dt_bias
                derivative=-reference.A_log.exp()*ad.sigmoid()
                derivative=torch.cat((derivative,derivative.new_zeros(1,a.value_heads))).index_select(0,plan.event_write)[None]
                scales=dict(A_log=(dg*rg).abs().sum((0,1)).norm(),
                            dt_bias=(dg*derivative).abs().sum((0,1)).norm())
                conditioned={name:float((getattr(layer,name).grad.cpu().double()-getattr(reference,name).grad).norm()/scale.clamp_min(1e-8))
                             for name,scale in scales.items()}
            # Perturb every read-only query; content and generation branch are invariant.
            keys=length+len(batch.mainline.generation_positions)-batch.mainline.prefix if len(batch.mainline.generation_positions) else length
            changed=x.detach().clone();changed[keys:]+=2
            with torch.no_grad():after=layer(changed,dp)
            torch.testing.assert_close(after[:keys],y[:keys],atol=0,rtol=0)
            passed=(max(v for k,v in errors.items() if k not in conditioned)<=.025
                    and max(c['relative'] for c in core.values())<=.035 and max(conditioned.values())<=.025)
            cases.append(dict(method=method,length=length,events=len(plan.event_q),errors=errors,
                              details=details,core=core,conditioned_control_error=conditioned,passed=passed))
    torch.npu.synchronize()
    result=dict(passed=all(c['passed'] for c in cases),rank=rank,world=world,norm_errors=norm_errors,cases=cases,backend=backend.provenance())
    (out/f'rank_{rank:05d}.json').write_text(json.dumps(result,indent=2));dist.barrier()
    if rank==0:
        reports=[json.loads((out/f'rank_{i:05d}.json').read_text()) for i in range(world)]
        (out/'summary.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in reports),world=world,cases_per_rank=len(cases),
            worst_relative_error=max(v for r in reports for c in r['cases'] for v in c['errors'].values()),
            worst_core_relative_error=max(v['relative'] for r in reports for c in r['cases'] for v in c['core'].values()),
            worst_conditioned_control_error=max(v for r in reports for c in r['cases'] for v in c['conditioned_control_error'].values()),
            norm_worst_relative_error=max(v for r in reports for v in r['norm_errors'].values())),indent=2))
    passed=torch.tensor(int(result['passed']),device=device);dist.all_reduce(passed,op=dist.ReduceOp.MIN)
    success=bool(passed.item());dist.destroy_process_group()
    if not success:raise AssertionError('CQ GDN accuracy gate failed; all 64 diagnostic reports were saved')


if __name__=='__main__':main()
