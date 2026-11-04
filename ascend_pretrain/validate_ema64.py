"""64-NPU EMA correctness/retention oracle; not a throughput benchmark."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
from contextlib import nullcontext
import torch
from torch import nn
import torch.distributed as dist

from . import parallel as P
from . import checkpoint as checkpoint_io
from .checkpoint_catalog import latest,prune
from .compare_data_restart import equal
from .config import tiny_config
from .ema import RendererEMA


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.up=P.ColumnLinear(128,512);self.down=P.RowLinear(512,128)
        self.to(dtype=torch.bfloat16)
        # Native FP32 model parameters alias their optimizer shards.
        self.gain=nn.Parameter(torch.ones(128,dtype=torch.float32))
    def forward(self,x):
        return (self.down(torch.nn.functional.silu(self.up(x))).float()*self.gain).to(torch.bfloat16)


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Linear(128,128,bias=False).to(torch.bfloat16)
        self.renderer=Renderer()
    def forward(self,x):return self.renderer(self.backbone(x))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tp',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cfg=tiny_config(width=128,heads=4,kv_heads=4,ffn=256,linear_heads=16,linear_value_heads=32,
                    linear_head_dim=128,dit_width=128,dit_heads=4,dit_ffn=256)
    P.initialize(args.tp,cfg)
    if dist.get_world_size()!=64:raise ValueError('Require 64 NPUs')
    torch.manual_seed(42)
    model=Model().to('npu').train();wrapped,optimizer=P.make_optimizer(model,5e-5,bucket_size=4096)
    ema=RendererEMA(model,optimizer,wrapped,.9999)
    reference=[x.detach().cpu().double() for x in ema.shadows]
    rank=dist.get_rank();root=args.output;root.mkdir(parents=True,exist_ok=True)
    topology=dict(tp=args.tp,dp=P.dp_size(),pp=1,cp=1,optimizer_instances=1)
    schedule={'ema':.9999,'lr':5e-5}
    generator=torch.Generator().manual_seed(73+P.dp_rank())
    inputs=[torch.randn(17,1,128,generator=generator).to(device='npu',dtype=torch.bfloat16) for _ in range(2)]
    losses=[]

    def update():
        nonlocal reference
        optimizer.zero_grad();wrapped.zero_grad_buffer()
        for i,x in enumerate(inputs):
            with wrapped.no_sync() if i==0 else nullcontext():
                loss=wrapped(x).float().square().mean()/2
                loss.backward()
        P.finish_grad_sync(wrapped)
        updated,norm,_=optimizer.step()
        if not updated or not math.isfinite(float(norm)):raise AssertionError('Optimizer failed')
        ema.update();wrapped.start_param_sync(force_sync=True)
        reference=[(.9999*previous+.0001*current.detach().cpu().double()).float().double()
                   for previous,current in zip(reference,ema.masters)]
        for actual,expected in zip(ema.shadows,reference):
            torch.testing.assert_close(actual.detach().cpu(),expected.float(),rtol=2e-6,atol=5e-7)
        losses.append(float(loss.detach()))

    for step in range(1,5):
        update()
        checkpoint_io.save(root/'checkpoints'/f'step_{step:07d}',model,optimizer,step,cfg,topology,schedule,ema=ema)
        if rank==0:prune(root/'checkpoints',3)
        dist.barrier()
    if len(list((root/'checkpoints').glob('step_*')))!=3:raise AssertionError('Rolling retention failed')

    online_model=checkpoint_io._cpu(copy.deepcopy(model.state_dict()))
    online_optimizer=checkpoint_io._cpu([checkpoint_io._optimizer_state(o) for o in checkpoint_io._optimizers(optimizer)])
    online_ema=checkpoint_io._cpu(copy.deepcopy(ema.state_dict()))
    pieces=[dict(layout=entry,value=value.float()) for entry,value in zip(ema.layout,reference)]
    gathered=[None]*P.dp_size()
    dist.all_gather_object(gathered,pieces,group=P.state().get_data_parallel_group())
    expected={name:torch.zeros_like(p.detach().cpu(),dtype=torch.float32)
              for name,p in model.named_parameters() if name.startswith('renderer.')}
    coverage={name:torch.zeros(p.numel(),dtype=torch.int32) for name,p in expected.items()}
    for entries in gathered:
        for entry in entries:
            layout=entry['layout'];name=layout['name'];start=layout['start'];end=layout['end']
            expected[name].flatten()[start:end].copy_(entry['value'].flatten());coverage[name][start:end]+=1
    if not all(torch.all(x==1).item() for x in coverage.values()):raise AssertionError('DP shard coverage is not exactly one')
    wrapped.disable_forward_pre_hook(param_sync=False)
    try:
        with ema.average_parameters():
            for name,p in model.named_parameters():
                want=expected[name].to(p.dtype) if name.startswith('renderer.') else online_model[name]
                torch.testing.assert_close(p.detach().cpu(),want,rtol=2e-6 if p.dtype==torch.float32 else 0,atol=5e-7 if p.dtype==torch.float32 else 0)
            if not torch.isfinite(model(inputs[0])).all().item():raise AssertionError('EMA inference failed')
    finally:wrapped.enable_forward_pre_hook()
    equal(checkpoint_io._cpu(model.state_dict()),online_model)
    equal(checkpoint_io._cpu([checkpoint_io._optimizer_state(o) for o in checkpoint_io._optimizers(optimizer)]),online_optimizer)
    equal(checkpoint_io._cpu(ema.state_dict()),online_ema)

    with torch.no_grad():
        for p in model.parameters():p.zero_()
        for master in ema.masters:master.zero_()
        for shadow in ema.shadows:shadow.zero_()
    ema.num_updates=999
    checkpoint_path,_=latest(root/'checkpoints')
    step=checkpoint_io.load(checkpoint_path,model,optimizer,cfg,topology,schedule,ema=ema)
    if step!=4 or ema.num_updates!=4:raise AssertionError('EMA update counter did not resume')
    equal(checkpoint_io._cpu(model.state_dict()),online_model)
    equal(checkpoint_io._cpu([checkpoint_io._optimizer_state(o) for o in checkpoint_io._optimizers(optimizer)]),online_optimizer)
    equal(checkpoint_io._cpu(ema.state_dict()),online_ema)
    update()
    if ema.num_updates!=5:raise AssertionError('EMA did not continue after resume')
    report=dict(passed=True,rank=rank,tp=args.tp,world=64,ema=ema.metadata(),local_ema_bytes=ema.local_bytes,
        checks=['FP32 EMA vs CPU recurrence','DP shard coverage','EMA materialization','online model/master/Adam exact restoration',
                'rolling three checkpoints','schema6 exact EMA restore','training and EMA continue after resume'],losses=losses)
    (root/f'rank_{rank:05d}.json').write_text(json.dumps(report,indent=2))
    dist.barrier()
    if rank==0:
        reports=[json.loads((root/f'rank_{i:05d}.json').read_text()) for i in range(64)]
        summary=dict(passed=all(r['passed'] for r in reports),tp=args.tp,world=64,
                     scope='Small real MCore distributed optimizer; EMA correctness, not model throughput',checks=report['checks'])
        (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
