"""64-NPU numerical gates and isolated timings for exact MOSS functions."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch_npu
from . import moss_kernels as fused


def reference(kind,a,b,c):
    if kind=='situ':return (4*torch.tanh(b/4)*torch.sigmoid(b))*(25*torch.tanh(a/25))
    if kind=='affine':return a*(1+b)+c
    return a+b*c


def error(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double()
    return dict(relative=float((a-b).norm()/b.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    device=f'npu:{os.environ["LOCAL_RANK"]}';torch.npu.set_device(device)
    dist.init_process_group('hccl');rank=dist.get_rank()
    if dist.get_world_size()!=64:raise ValueError('Requires 64 NPUs')
    torch.manual_seed(100+rank)
    reports=[]
    for dtype in (torch.float32,torch.bfloat16):
        for length,width,stride in ((129,128,1),(257,2048,2),(65,3072,6),(131,4096,2),(137,3072,1)):
            for kind in ('situ','affine','gated_add'):
                storage=[torch.randn(length,width*stride,dtype=dtype,device=device)*scale for scale in (30,7,3)]
                inputs=[x[:, :width].detach().requires_grad_() for x in storage]
                oracle=[x.detach().cpu().double().requires_grad_() for x in inputs]
                probe=torch.randn_like(inputs[0]);expected=reference(kind,*oracle)
                actual=(fused.situ(*inputs[:2]) if kind=='situ' else getattr(fused,kind)(*inputs))
                actual.backward(probe);expected.backward(probe.cpu().double())
                errors=[error(actual,expected)]+[error(a.grad,b.grad) for a,b in zip(inputs,oracle) if b.grad is not None]
                limit=1e-4 if dtype==torch.float32 else .006
                reports.append(dict(kind=kind,dtype=str(dtype),shape=[length,width],stride=stride,
                    errors=errors,passed=all(x['relative']<limit for x in errors)))
    timings={}
    for kind in ('situ','affine','gated_add'):
        shapes=((4096,2048,2),(4096,4096,2)) if kind=='situ' else ((8192,3072,1),)
        cases=[]
        for length,width,stride in shapes:
            inputs=[torch.randn(length,width*stride,device=device,dtype=torch.bfloat16)[:,:width].requires_grad_() for _ in range(3)]
            probe=torch.randn_like(inputs[0])
            def execute(optimized):
                for x in inputs:x.grad=None
                y=(fused.situ(*inputs[:2]) if kind=='situ' else getattr(fused,kind)(*inputs)) if optimized else reference(kind,*inputs)
                y.backward(probe)
            pair=dict(shape=[length,width],stride=stride)
            for optimized in (False,True):
                for _ in range(3):execute(optimized)
                torch.npu.synchronize()
                begin,end=[torch.npu.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(20):execute(optimized)
                end.record();torch.npu.synchronize()
                pair['fused_ms' if optimized else 'eager_ms']=begin.elapsed_time(end)/20
            pair['speedup']=pair['eager_ms']/pair['fused_ms'];cases.append(pair)
        timings[kind]=dict(eager_ms=sum(c['eager_ms'] for c in cases)/len(cases),
            fused_ms=sum(c['fused_ms'] for c in cases)/len(cases),
            speedup=min(c['speedup'] for c in cases),shapes=cases)
    passed=all(x['passed'] for x in reports)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    (root/f'rank_{rank:05d}.json').write_text(json.dumps(dict(passed=passed,cases=reports,timings=timings),indent=2))
    gate=torch.tensor(int(passed),device=device);dist.all_reduce(gate,op=dist.ReduceOp.MIN)
    dist.barrier()
    if rank==0:
        import statistics
        all_reports=[json.loads((root/f'rank_{r:05d}.json').read_text()) for r in range(64)]
        summary=dict(passed=bool(gate.item()),world=64,
            cases=sum(len(r['cases']) for r in all_reports),
            worst_relative=max(e['relative'] for r in all_reports for case in r['cases'] for e in case['errors']),
            timings={kind:{key:statistics.median(r['timings'][kind][key] for r in all_reports)
                           for key in ('eager_ms','fused_ms','speedup')} for kind in timings})
        (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    success=bool(gate.item());dist.destroy_process_group()
    if not success:raise AssertionError('MOSS kernel precision gate failed')


if __name__=='__main__':main()
