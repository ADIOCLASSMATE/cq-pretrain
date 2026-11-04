"""64-NPU split-renderer gradient/communication oracle, not a speed benchmark.

Uses small synthetic inputs to isolate the execution protocol. Full-size speed
stages use a prepared dataset with native task order and pretrained weights separately.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from . import parallel as P
from .config import tiny_config
from .data import fake_batch
from .generation import GenerationInputs
from .input_pipeline.parallel import OrderedGradientBuckets, OrderedParameterGathers, finish_parameter_gathers
from .model import JointModel
from .qwen35 import cast_for_training
from .renderer_schedule import backward_deferred_renderer


def error(actual, expected):
    actual=actual.detach().cpu().float();expected=expected.detach().cpu().float()
    return dict(relative=float((actual-expected).norm()/expected.norm().clamp_min(1e-8)),
                max_abs=float((actual-expected).abs().max()),reference_norm=float(expected.norm()))


def make_batch(cfg, micro):
    n=3+(P.dp_rank()+micro)%3
    generator=torch.Generator().manual_seed(770+P.dp_rank()*2+micro)
    batch=fake_batch(cfg,65,1,n,seed=93+P.dp_rank()*2+micro,dtype=torch.bfloat16)
    images=[np.zeros((32,48 if i%2 else 32,3),dtype=np.uint8) for i in range(n)]
    layout=GenerationInputs.prepare(images,cfg.image_condition_queries,32,token_budget=18)
    batch.latents=torch.randn(sum(layout.lengths),128,generator=generator)
    batch.noise=torch.randn(batch.latents.shape,generator=generator)
    batch.times=torch.linspace(.2,.8,n)
    phase=(batch.times*1000).double()[:,None]*10000**(-torch.arange(128,dtype=torch.float64)/128)
    layout.time_features=torch.cat((phase.cos(),phase.sin()),-1).float().to(torch.bfloat16)
    batch.generation=layout
    return batch.to('npu')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tp',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cfg=tiny_config(width=128,heads=4,kv_heads=4,ffn=256,linear_heads=16,linear_value_heads=32,
        linear_head_dim=128,tie_word_embeddings=False,renderer='moss',dit_width=128,dit_heads=4,
        dit_ffn=256,dit_layers=2,latent_channels=128,flow_draws=1,checkpoint_backbone=True,
        checkpoint_dit=True,moss_fused_qkv=True,qwen_fused_projections=True)
    P.initialize(args.tp,cfg)
    if dist.get_world_size()!=64:raise ValueError('Require exactly 64 NPUs')
    torch.manual_seed(42)
    model=cast_for_training(JointModel(cfg)).to('npu').train()
    wrapped,optimizer=P.make_optimizer(model,5e-5,bucket_size=1000000)
    buckets=OrderedGradientBuckets(wrapped);gathers=OrderedParameterGathers(wrapped)
    batches=[make_batch(cfg,micro) for micro in range(2)]
    reports=[]

    def accumulate(staged):
        optimizer.zero_grad();wrapped.zero_grad_buffer();buckets.begin()
        values=[];cq_gradients=[];backbone_visits=[]
        for micro,batch in enumerate(batches):
            with wrapped.no_sync() if micro==0 else nullcontext():
                captured=[]
                def capture(module,inputs):
                    inputs[0].retain_grad();captured.append(inputs[0])
                handle=None if staged else model.renderer.register_forward_pre_hook(capture)
                losses=wrapped(batch,return_hidden=True,loss_scales=(.7,.3),defer_renderer=staged)
                if handle is not None:handle.remove()
                visits=[];losses['hidden'].register_hook(lambda grad:visits.append(True))
                if staged:
                    cq=losses['_image_conditions'];cq.retain_grad()
                else:cq=captured[0]
                finish_parameter_gathers(wrapped)
                if staged:backward_deferred_renderer(model,batch,losses,.5,wrapped)
                else:(losses['loss']*.5).backward()
                if visits!=[True]:raise AssertionError('Backbone backward did not execute exactly once')
                backbone_visits.append(len(visits))
                cq_gradients.append(cq.grad.detach().cpu())
                values.append({k:losses[k].detach().cpu() for k in ('loss','text_loss','image_loss')})
                del losses,cq,captured
        buckets.finish();P.finish_grad_sync(wrapped)
        gradients={name:p.main_grad.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
        return values,cq_gradients,gradients,backbone_visits

    for recompute in (True,False):
        cfg.checkpoint_backbone=recompute;cfg.checkpoint_dit=recompute
        expected=accumulate(False);actual=accumulate(True)
        outputs={f'{i}/{k}':error(actual[0][i][k],expected[0][i][k]) for i in range(2) for k in expected[0][i]}
        cq={str(i):error(a,b) for i,(a,b) in enumerate(zip(actual[1],expected[1]))}
        gradients={name:error(value,expected[2][name]) for name,value in actual[2].items()}
        passed=all(e['relative']<.03 or e['max_abs']<1e-4 for e in outputs.values()) and all(
            e['relative']<.05 or e['max_abs']<1e-4 for e in [*cq.values(),*gradients.values()])
        updated,norm,_=optimizer.step()
        if not updated or not math.isfinite(float(norm)):raise AssertionError('Optimizer update is not finite')
        wrapped.start_param_sync(force_sync=True)
        reports.append(dict(recompute=recompute,passed=passed,outputs=outputs,cq=cq,gradients=gradients,
                            grad_norm=float(norm),backbone_visits=actual[3]))
    root=args.output;root.mkdir(parents=True,exist_ok=True);rank=dist.get_rank()
    report=dict(rank=rank,tp=args.tp,passed=all(r['passed'] for r in reports),cases=reports,
        chunk_counts=[len(b.generation.microbatches) for b in batches])
    (root/f'rank_{rank:05d}.json').write_text(json.dumps(report,indent=2))
    dist.barrier()
    if rank==0:
        ranks=[json.loads((root/f'rank_{i:05d}.json').read_text()) for i in range(64)]
        summary=dict(passed=all(r['passed'] for r in ranks),world=64,tp=args.tp,
            scope='Small synthetic correctness oracle; real NPU kernels, TP/DP and MCore optimizer; not throughput',
            chunk_counts=sorted({c for r in ranks for c in r['chunk_counts']}),
            worst_gradient_relative=max(e['relative'] for r in ranks for c in r['cases'] for e in c['gradients'].values()),
            checks=['recompute on/off','GA2 with unequal DP chunk counts','joint loss',
                    'CQ gradients','all trainable parameter gradients','one backbone backward',
                    'ordered DP reduce-scatter/parameter all-gather','finite optimizer updates'])
        (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    gate=torch.tensor(int(report['passed']),device='npu');dist.all_reduce(gate,op=dist.ReduceOp.MIN)
    passed=bool(gate.item());dist.destroy_process_group()
    if not passed:raise AssertionError('Split-renderer correctness gate failed; per-parameter evidence saved')


if __name__=='__main__':main()
