"""All-parameter TP vs dense NPU oracle; separate from real-data throughput."""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import training.mcore as P
from configs.config import tiny_config
from data.batch import fake_batch
from data.generation import GenerationInputs
from models.backbone_qwen35 import cast_for_training, unfused_parameters


def ce_check(device):
    from training.loss import linear_cross_entropy
    torch.manual_seed(77+P.dp_rank())
    full=torch.randn(4096,128,dtype=torch.bfloat16,device=device).requires_grad_()
    weight=torch.nn.Parameter(full.detach().chunk(P.tp_size())[P.state().get_tensor_model_parallel_rank()].clone())
    P.mark_sharded(weight)
    weight._ce_direct_accumulation=True;weight.grad_added_to_main_grad=False
    weight.main_grad=torch.zeros_like(weight,dtype=torch.float32)
    def hook(p):
        if not p.grad_added_to_main_grad:raise AssertionError('CE bypassed direct accumulation')
        p.grad=None
    weight.register_post_accumulate_grad_hook(hook)
    errors={}
    def compare(a,b):
        a=a.detach().cpu().float();b=b.detach().cpu().float()
        return dict(relative=float((a-b).norm()/b.norm().clamp_min(1e-8)),max_abs=float((a-b).abs().max()))
    for i,(n,scale) in enumerate(((37,.3),(65,.7))):
        x=torch.randn(n,128,dtype=torch.bfloat16,device=device).requires_grad_()
        other=x.detach().clone().requires_grad_();targets=torch.randint(4096,(n,),device=device)
        actual=linear_cross_entropy(x,weight,targets,17)*scale
        expected=torch.nn.functional.cross_entropy(torch.nn.functional.linear(other,full).float(),targets)*scale
        actual.backward();expected.backward()
        errors[f'loss{i}']=compare(actual,expected);errors[f'input{i}']=compare(x.grad,other.grad)
    errors['accumulated_weight']=compare(weight.main_grad,P.checkpoint_shard(full.grad,weight))
    return errors


def vision_check(device,freeze_encoder=False):
    from models.vision import QwenVisionTower, VisionInputs
    vc=dict(depth=2,hidden_act='gelu_pytorch_tanh',hidden_size=128,intermediate_size=256,
        in_channels=3,num_heads=4,num_position_embeddings=64,out_hidden_size=128,
        patch_size=16,spatial_merge_size=2,temporal_patch_size=2,_tensor_parallel=False)
    reference=QwenVisionTower(vc,recompute=True,freeze_encoder=freeze_encoder).to(dtype=torch.bfloat16)
    layout=VisionInputs.prepare(torch.randn(108,1536),[(1,8,6),(1,6,10)],torch.arange(27),vc).to(device)
    other=copy.deepcopy(layout)
    layout.pixel_values.requires_grad_();other.pixel_values.requires_grad_()
    actual=QwenVisionTower(dict(vc,_tensor_parallel=True),recompute=True,
                          freeze_encoder=freeze_encoder).to(dtype=torch.bfloat16)
    source=dict(reference.named_parameters())
    with torch.no_grad():
        for name,p in actual.named_parameters():p.copy_(P.checkpoint_shard(source[name.replace('.impl.','.')],p))
    reference=reference.to(device);actual=actual.to(device)
    source=dict(reference.named_parameters())
    expected=reference(other);result=actual(layout);probe=torch.randn_like(result)
    (expected*probe).sum().backward();(result*probe).sum().backward()
    def compare(a,b):
        a=a.detach().cpu().float();b=b.detach().cpu().float()
        return dict(relative=float((a-b).norm()/b.norm().clamp_min(1e-8)),max_abs=float((a-b).abs().max()))
    errors={'output':compare(result,expected)}
    if freeze_encoder:
        if layout.pixel_values.grad is not None or other.pixel_values.grad is not None:
            raise AssertionError('Frozen vision leaked an input gradient')
        if actual.recompute or actual.blocks.training:raise AssertionError('Frozen encoder retained training/recompute')
    else:errors['input_gradient']=compare(layout.pixel_values.grad,other.pixel_values.grad)
    for name,p in actual.named_parameters():
        ref=source[name.replace('.impl.','.')]
        if not p.requires_grad:
            if p.grad is not None or ref.grad is not None:raise AssertionError(f'Frozen gradient: {name}')
        else:errors[name]=compare(p.grad,P.checkpoint_shard(ref.grad,p))
    return errors


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--tp',type=int,required=True);args=p.parse_args()
    cfg=tiny_config(width=128,heads=4,kv_heads=4,ffn=256,linear_heads=16,linear_value_heads=32,linear_head_dim=128,
        tie_word_embeddings=False,renderer='moss',dit_width=128,dit_heads=4,dit_ffn=256,dit_layers=2,
        latent_channels=128,flow_draws=1,checkpoint_backbone=True,checkpoint_dit=True,moss_fused_qkv=True,qwen_fused_projections=True)
    P.initialize(args.tp,cfg,initialize_weights=False);P._config.gradient_accumulation_fusion=False
    rank=dist.get_rank();device=f'npu:{os.environ["LOCAL_RANK"]}'
    if dist.get_world_size()!=64:raise ValueError('Require 64 NPUs')
    torch.manual_seed(42+P.dp_rank())
    from models.modeling_umm import JointModel
    P._active=False
    reference_cfg=copy.deepcopy(cfg);reference_cfg.moss_fused_qkv=False;reference_cfg.qwen_fused_projections=False
    reference=cast_for_training(JointModel(reference_cfg))
    batch=fake_batch(cfg,33,1,2,dtype=torch.bfloat16)
    layout=GenerationInputs.prepare([np.zeros((32,48,3),np.uint8),np.zeros((16,32,3),np.uint8)],cfg.image_condition_queries,32)
    batch.latents=torch.randn(sum(layout.lengths),128)
    batch.noise=torch.randn_like(batch.latents)
    batch.times=torch.tensor([.25,.75])
    phase=(batch.times*1000).double()[:,None]*10000**(-torch.arange(128,dtype=torch.float64)/128)
    layout.time_features=torch.cat((phase.cos(),phase.sin()),-1).float().to(torch.bfloat16)
    batch.generation=layout
    source={n.replace('.impl.','.'):v for n,v in reference.named_parameters()}
    P._active=True
    model=cast_for_training(JointModel(cfg))
    with torch.no_grad():
        for name,param in model.named_parameters():
            for clean,view in unfused_parameters(name,param,cfg).items():
                value=(torch.cat([source[clean.replace('.qkv.',f'.{label}.')] for label in ('q','k','v')])
                       if clean.endswith('.proj.qkv.weight') else source[clean])
                view.copy_(P.checkpoint_shard(value,view))
    model=model.to(device);reference=reference.to(device);batch=batch.to(device)
    output=model(batch,return_hidden=True);output['loss'].backward()
    for name,param in model.named_parameters():
        if name.endswith(('.q_norm.weight','.k_norm.weight','.linear_attn.norm.weight')):
            dist.all_reduce(param.grad,group=P.state().get_tensor_model_parallel_group())
    P._active=False
    import models.moss as moss
    moss.FUSED_POINTWISE=False
    expected=reference(batch,return_hidden=True);expected['loss'].backward()
    P._active=True
    def relative(a,b):
        a=a.detach().cpu().float();b=b.detach().cpu().float()
        return dict(relative=float((a-b).norm()/b.norm().clamp_min(1e-8)),max_abs=float((a-b).abs().max()),ref_norm=float(b.norm()))
    errors={name:relative(output[name],expected[name]) for name in ('loss','text_loss','image_loss','hidden')}
    refs={n.replace('.impl.','.'):v for n,v in reference.named_parameters()}
    grads={}
    for name,param in model.named_parameters():
        if param.grad is None:raise AssertionError(f'Absent parameter gradient: {name}')
        views=unfused_parameters(name,param,cfg);grad_views=unfused_parameters(name,param.grad,cfg)
        for clean,view in views.items():
            other=(torch.cat([refs[clean.replace('.qkv.',f'.{label}.')].grad for label in ('q','k','v')])
                   if clean.endswith('.proj.qkv.weight') else refs[clean].grad)
            if other is None:raise AssertionError(f'Absent reference gradient: {clean}')
            grads[clean]=relative(grad_views[clean],P.checkpoint_shard(other,view))
    vision=vision_check(device)
    frozen_vision=vision_check(device,freeze_encoder=True)
    ce=ce_check(device)
    passed=max(v['relative'] for v in errors.values())<.03 and all(v['relative']<.05 or v['max_abs']<1e-4 for v in [*grads.values(),*vision.values(),*frozen_vision.values()]) and all(v['relative']<.03 or v['max_abs']<1e-4 for v in ce.values())
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    report=dict(rank=rank,tp=args.tp,passed=passed,outputs=errors,gradients=grads,vision=vision,frozen_vision=frozen_vision,ce=ce)
    (root/f'rank_{rank:05d}.json').write_text(json.dumps(report,indent=2))
    dist.barrier()
    if rank==0:
        reports=[json.loads((root/f'rank_{i:05d}.json').read_text()) for i in range(64)]
        summary=dict(passed=all(r['passed'] for r in reports),tp=args.tp,world=64,
            worst_output_relative=max(v['relative'] for r in reports for v in r['outputs'].values()),
            worst_gradient_relative=max(v['relative'] for r in reports for v in r['gradients'].values()))
        (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    gate=torch.tensor(int(passed),device=device);dist.all_reduce(gate,op=dist.ReduceOp.MIN)
    success=bool(gate.item());dist.destroy_process_group()
    if not success:raise AssertionError('TP equivalence gate failed; all per-parameter diagnostics saved')


if __name__=='__main__':main()
