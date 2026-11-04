"""Bounded 64-NPU measurements of the actual BF16/FP32-accumulating GEMMs.

These are isolated kernel measurements, not training throughput or an official
hardware peak. Run as a separate stage, never alongside training on its devices.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import torch
import torch.distributed as dist
import torch_npu


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=f'npu:{os.environ["LOCAL_RANK"]}';torch.npu.set_device(device)
    dist.init_process_group('hccl')
    rank=dist.get_rank();world=dist.get_world_size()
    if world!=64:raise ValueError('Requires 64 NPUs')
    torch.manual_seed(913+rank)
    torch.npu.matmul.allow_hf32=False
    a=torch.randn(256,256,device=device,dtype=torch.bfloat16)
    g=torch.randn_like(a);dw=torch.zeros(256,256,device=device,dtype=torch.float32)
    torch_npu._npu_matmul_add_fp32(g,a,dw)
    expected=g.cpu().double().T@a.cpu().double()
    relative=float((dw.cpu().double()-expected).norm()/expected.norm())
    if relative>1e-4:raise AssertionError(f'FP32 accumulation mismatch: {relative}')
    del a,g,dw
    cases=[]
    for label,m,k,n in (('qwen_mlp_tp2',4096,4096,6144),
            ('moss_up_gate_tp2',8192,3072,8192),('vision_mlp_tp2',4096,1152,2152),
            ('large_dense_control',8192,8192,8192)):
        a=torch.randn(m,k,device=device,dtype=torch.bfloat16)
        w=torch.randn(n,k,device=device,dtype=torch.bfloat16)
        g=torch.randn(m,n,device=device,dtype=torch.bfloat16)
        dw=torch.zeros(n,k,device=device,dtype=torch.float32)
        operations={
            'forward':lambda:torch.nn.functional.linear(a,w),
            'input_gradient':lambda:torch.nn.functional.linear(g,w.T),
            'fp32_weight_gradient':lambda:torch_npu._npu_matmul_add_fp32(g,a,dw)}
        for kind,operation in operations.items():
            for _ in range(8):operation()
            torch.npu.synchronize();dist.barrier()
            elapsed=[]
            for _ in range(3):
                begin,end=[torch.npu.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(20):operation()
                end.record();torch.npu.synchronize()
                elapsed.append(begin.elapsed_time(end)/20)
            ms=statistics.median(elapsed)
            cases.append(dict(label=label,kind=kind,m=m,k=k,n=n,milliseconds=ms,
                              tflops=2*m*n*k/(ms/1000)/1e12))
        del a,w,g,dw,operations
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/f'rank_{rank:05d}.json').write_text(json.dumps(dict(rank=rank,
        device=torch.npu.get_device_name(),fp32_gradient_relative=relative,cases=cases),indent=2)+'\n')
    dist.barrier()
    if rank==0:
        reports=[json.loads((args.output/f'rank_{r:05d}.json').read_text()) for r in range(world)]
        summary=dict(world=world,passed=True,
            caveat='Observed isolated GEMM throughput; not end-to-end training, hardware peak, or an upper bound on future implementations.',
            fp32_gradient_worst_relative=max(r['fp32_gradient_relative'] for r in reports),cases=[])
        for i,case in enumerate(cases):
            values=[r['cases'][i]['tflops'] for r in reports]
            summary['cases'].append({k:v for k,v in case.items() if k not in ('milliseconds','tflops')}|
                dict(median_tflops=statistics.median(values),min_tflops=min(values),max_tflops=max(values),
                     summed_64_devices_pflops=sum(values)/1000))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
