"""Actual 8K CQ attention equivalence and training time on a single 910B."""
import argparse
import json
from pathlib import Path
import time
import torch
from data.attention_plan import plan_attention, tiled_attention
from configs.config import ModelConfig
from data.batch import fake_batch
from models.kernels import attention


def cpu_checks():
    torch.manual_seed(31)
    tensors=[torch.randn(s,2,h,16,requires_grad=True) for s,h in ((30,4),(17,2),(17,2))]
    mask=torch.ones(2,1,30,17,dtype=torch.bool)
    for i in range(30):mask[:,:,i,:(i%17)+1]=False
    mask[:,:,0]=True;mask[:,:,0,-1]=False  # A deliberately noncausal visible key.
    valid=torch.ones(2,30,dtype=torch.bool)
    plan=plan_attention(mask,4).bind(mask)
    assert plan.tiles[0][2]==17
    dy=torch.randn(30,2,64)
    expected=attention(*tensors,mask,valid,False)
    actual=tiled_attention(*tensors,mask,valid,plan,False)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    a=torch.autograd.grad(actual,tensors,dy);b=torch.autograd.grad(expected,tensors,dy)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-4)
    mask[:,:,0,0]=False
    assert not plan.matches(mask)
    torch.testing.assert_close(tiled_attention(*tensors,mask,valid,plan,False),
                               attention(*tensors,mask,valid,False),atol=0,rtol=0)
    assert not plan.matches(mask.clone())
    return dict(arbitrary_future_keys=True,forward_and_backward=True,stale_mask_fallback=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    checks=cpu_checks()
    if args.cpu_only:
        args.output.write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
        print(json.dumps(dict(passed=True,checks=checks)))
        return
    import torch_npu
    torch.npu.set_device(0);torch.set_num_threads(4);torch.manual_seed(42)
    batch=fake_batch(ModelConfig(),device='cpu',dtype=torch.bfloat16)
    plans={n:plan_attention(batch.mask,n) for n in (1024,2048,4096)}
    batch=batch.to('npu');plans={n:p.bind(batch.mask) for n,p in plans.items()}
    inputs=[torch.randn(s,1,h,128,device='npu',dtype=torch.bfloat16,requires_grad=True)
            for s,h in ((16384,16),(8192,8),(8192,8))]
    dy=torch.randn(16384,1,2048,device='npu',dtype=torch.bfloat16)
    def run(plan):
        if plan is None:out=attention(*inputs,batch.mask,batch.valid_rows)
        else:out=tiled_attention(*inputs,batch.mask,batch.valid_rows,plan)
        grads=torch.autograd.grad(out,inputs,dy)
        return out,grads
    reference=run(None)
    rows=[]
    for chunk,plan in [(0,None),*plans.items()]:
        out,grads=run(plan)
        pairs=[(out,reference[0]),*zip(grads,reference[1])]
        errors=[float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)) for a,b in pairs]
        assert max(errors)<.015,(chunk,errors)
        for _ in range(3):run(plan)
        torch.npu.synchronize();start=time.perf_counter()
        for _ in range(15):run(plan)
        torch.npu.synchronize()
        rows.append(dict(chunk=chunk,milliseconds=(time.perf_counter()-start)*1000/15,
                         relative_l2=errors,tiles=plan.tiles if plan else None))
    args.output.write_text(json.dumps(dict(passed=True,checks=checks,rows=rows),indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
