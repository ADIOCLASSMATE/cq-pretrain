"""Compare GDN providers on identical image CQ event plans, including backward."""
import argparse
import json
import statistics
import time
import sys
from pathlib import Path
import torch
import torch_npu
from configs.config import RunConfig
from data.batch import fake_batch
from models.backbone_qwen35 import GatedDelta
from ops import gated_delta_net as delta_ascendc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[2048,8192])
    p.add_argument('--iterations',type=int,default=12)
    args=p.parse_args()
    torch.npu.set_device(0); torch.manual_seed(872)
    delta_ascendc.initialize()
    from utils.dependencies import source_directories
    for path in source_directories():sys.path.insert(0,str(path))
    # Match the training stack without changing tensor-parallel math for this layer benchmark.
    previous=sys.argv; sys.argv=[previous[0],"--optimization-level","0"]
    compile_function=torch.compile
    try:
        import megatron_adaptor
        import mindspeed.megatron_adaptor
    finally:
        sys.argv=previous; torch.compile=compile_function
    cfg=RunConfig(renderer='legacy',vision_assets='assets/Qwen3.5-0.8B-vision',
        qwen_fused_projections=True,vision_max_pixels=262144)
    model=GatedDelta(cfg).to('npu',torch.bfloat16)
    model.A_log.data=model.A_log.data.float(); model.norm.weight.data=model.norm.weight.data.float()
    report={'device':torch.npu.get_device_name(0),'config':cfg.to_dict(),'results':[]}
    for length in args.lengths:
        batch=fake_batch(cfg,length=length,images_per_sequence=2)
        plan=batch.mainline.delta_plan.to('npu')
        x=torch.randn(plan.hidden_length,1,cfg.width,device='npu',dtype=torch.bfloat16,requires_grad=True)
        upstream=torch.randn_like(x)/100
        outputs={}; row={'content_tokens':length,'query_tokens':len(batch.mainline.query_positions),'events':len(plan.event_q)}
        for backend in ('ascendc','mindspeed'):
            cfg.gdn_backend=backend
            def step():
                model.zero_grad(set_to_none=True);x.grad=None
                y=model(x,plan);(y*upstream).sum().backward();return y
            try:
                for _ in range(3):step()
                torch.npu.synchronize(); torch.npu.reset_peak_memory_stats()
                durations=[]
                for _ in range(args.iterations):
                    torch.npu.synchronize();begin=time.perf_counter();y=step();torch.npu.synchronize()
                    durations.append((time.perf_counter()-begin)*1000)
                outputs[backend]={'y':y.detach().float().cpu(),'dx':x.grad.detach().float().cpu(),
                    'dw':model.in_proj_all.weight.grad.detach().float().cpu()}
                row[backend]={'median_ms':statistics.median(durations),'samples_ms':durations,
                    'peak_bytes':torch.npu.max_memory_allocated()}
            except Exception as exc:
                row[backend]={'error':str(exc)}
        if len(outputs)==2:
            row['relative_rms']={name:float((outputs['ascendc'][name]-outputs['mindspeed'][name]).square().mean().sqrt()/outputs['ascendc'][name].square().mean().sqrt().clamp_min(1e-12)) for name in ('y','dx','dw')}
        report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    if any('error' in r[b] for r in report['results'] for b in ('ascendc','mindspeed')):raise SystemExit(1)

if __name__=='__main__':main()
