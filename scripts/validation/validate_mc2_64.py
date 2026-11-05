"""64-NPU TP2 forward/gradient gates and timings for MatmulAllReduce."""
import argparse
import json
import os
from pathlib import Path
import statistics
import torch
import torch.distributed as dist
import torch_npu
import infra.parallel as P
from configs.config import RunConfig
from infra.mc2 import _RowAllReduce


def relative(a,b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    P.initialize(2,RunConfig.load(Path(__file__).resolve().parents[2]/'configs/legacy/qwen35_9b_moss.json'))
    rank=dist.get_rank();group=P.state().get_tensor_model_parallel_group()
    if dist.get_world_size()!=64:raise ValueError('Requires 64 NPUs')
    hcom=group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
    torch.manual_seed(830+rank);reports=[]
    for m,k,n in ((129,256,512),(1024,2048,4096),(4096,4096,3072),(8192,4096,3072)):
        a=torch.randn(m,k,device='npu',dtype=torch.bfloat16)
        w=torch.randn(n,k,device='npu',dtype=torch.bfloat16)
        def eager():
            y=torch.nn.functional.linear(a,w);dist.all_reduce(y,group=group);return y
        def fused():return _RowAllReduce.apply(a,w,hcom)
        expected=eager();actual=fused();error=relative(actual,expected)
        result=dict(shape=[m,k,n],output_relative=error,passed=error<.01)
        if m==129:
            a.requires_grad_();w.requires_grad_()
            w.main_grad=torch.zeros_like(w,dtype=torch.float32)
            w.grad_added_to_main_grad=False;w.zero_out_wgrad=True
            g=torch.randn(m,n,device='npu',dtype=torch.bfloat16)
            dist.broadcast(g,src=rank-rank%2,group=group)
            for _ in range(2):fused().backward(g)
            ax=a.detach().cpu().double();wx=w.detach().cpu().double();gx=g.cpu().double()
            result['input_gradient_relative']=relative(a.grad.cpu(),2*(gx@wx))
            result['fp32_weight_gradient_relative']=relative(w.main_grad.cpu(),2*(gx.T@ax))
            result['hook_ready']=bool(w.grad_added_to_main_grad)
            result['passed'] &= (result['input_gradient_relative']<.01 and
                result['fp32_weight_gradient_relative']<1e-4 and result['hook_ready'])
            a=a.detach();w=w.detach()
        for name,operation in (('eager_ms',eager),('fused_ms',fused)):
            for _ in range(5):operation()
            torch.npu.synchronize();dist.barrier()
            samples=[]
            for _ in range(3):
                begin,end=[torch.npu.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(20):operation()
                end.record();torch.npu.synchronize();samples.append(begin.elapsed_time(end)/20)
            result[name]=statistics.median(samples)
        result['speedup']=result['eager_ms']/result['fused_ms'];reports.append(result)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/f'rank_{rank:05d}.json').write_text(json.dumps(dict(cases=reports),indent=2)+'\n')
    dist.barrier()
    gate=torch.tensor(int(all(r['passed'] for r in reports)),device='npu')
    dist.all_reduce(gate,op=dist.ReduceOp.MIN)
    if rank==0:
        ranks=[json.loads((args.output/f'rank_{r:05d}.json').read_text()) for r in range(64)]
        summary=dict(passed=bool(gate.item()),world=64,tp=2,cases=[])
        for i,case in enumerate(reports):
            summary['cases'].append(dict(shape=case['shape'],
                worst_output_relative=max(r['cases'][i]['output_relative'] for r in ranks),
                **{key:statistics.median(r['cases'][i][key] for r in ranks) for key in ('eager_ms','fused_ms','speedup')}))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    success=bool(gate.item());dist.destroy_process_group()
    if not success:raise AssertionError('MC2 precision gate failed')


if __name__=='__main__':main()
