"""Compare actual TP Wan outputs and every parameter gradient with the full CPU model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from configs.config import ModelConfig
from models.generator_wan21 import Wan21Generator, FP32Linear
from data.generation import WanGenerationInputs
from scripts.validation.validate_wan import rms
import training.mcore as P


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--tp',type=int,default=2)
    args=parser.parse_args();torch.manual_seed(73)
    cfg=ModelConfig(width=64,layers=4,heads=2,kv_heads=2,head_dim=32,ffn=128,
        dit_width=128,dit_heads=2,dit_layers=2,dit_ffn=256,wan_text_dim=64,wan_freq_dim=32)
    full=Wan21Generator(cfg).to(dtype=torch.bfloat16)
    for module in full.modules():
        if isinstance(module,FP32Linear):module.float()
    full.float();values=dict(full.named_parameters())
    layout=WanGenerationInputs.prepare([np.zeros((64,64,3),np.uint8),
        np.zeros((5,64,96,3),np.uint8)],[17,64],64)
    x=torch.randn(sum(layout.lengths),64,dtype=torch.bfloat16).float().requires_grad_()
    c=torch.randn(2,64,64,dtype=torch.bfloat16).float().requires_grad_();f=torch.randn(2,32)
    expected=full.predict(x,c,f,layout);upstream=torch.randn_like(expected)/100
    (expected*upstream).sum().backward()
    P.initialize(args.tp,cfg)
    # This isolates TP math. The training-loop probe separately checks fused main_grad accumulation.
    P._config.gradient_accumulation_fusion=False
    model=Wan21Generator(cfg).to(dtype=torch.bfloat16)
    for module in model.modules():
        if isinstance(module,FP32Linear):module.float()
    with torch.no_grad():
        for name,param in model.named_parameters():param.copy_(P.checkpoint_shard(values[name],param))
    model.to('npu')
    xn=x.detach().to('npu',torch.bfloat16).requires_grad_();cn=c.detach().to('npu',torch.bfloat16).requires_grad_()
    actual=model.predict(xn,cn,f.to('npu'),layout.to('npu'))
    (actual*upstream.to('npu')).sum().backward();torch.npu.synchronize()
    errors={name:rms(param.grad.cpu(),P.checkpoint_shard(values[name].grad,param)) for name,param in model.named_parameters()}
    result=dict(rank=dist.get_rank(),tp=args.tp,forward_rms=rms(actual.detach().cpu(),expected),
        input_gradient_rms=rms(xn.grad.cpu(),x.grad),condition_gradient_rms=rms(cn.grad.cpu(),c.grad),
        max_parameter_gradient_rms=max(errors.values()),worst_parameters=sorted(errors.items(),key=lambda x:-x[1])[:8])
    result['passed']=max(result[k] for k in ('forward_rms','input_gradient_rms','condition_gradient_rms','max_parameter_gradient_rms'))<.05
    rows=[None]*dist.get_world_size();dist.all_gather_object(rows,result)
    if dist.get_rank()==0:
        args.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)
    dist.destroy_process_group()
    if not all(x['passed'] for x in rows):raise SystemExit(1)


if __name__=='__main__':main()
