"""Bounded Ascend forward/backward comparison with the FP32 CPU Wan path."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch_npu
from configs.config import RunConfig
from data.generation import WanGenerationInputs
from models.generator_wan21 import Wan21Generator, FP32Linear


def rms(actual,expected):
    return float((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-8))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();torch.npu.set_device(0);torch.manual_seed(19)
    cfg=RunConfig(width=64,dit_width=128,dit_heads=2,dit_layers=2,dit_ffn=256,wan_freq_dim=32,checkpoint_dit=True)
    model=Wan21Generator(cfg).to(dtype=torch.bfloat16)
    for module in model.modules():
        if isinstance(module,FP32Linear):module.float()
    reference=copy.deepcopy(model).float()
    model.to('npu')
    layout=WanGenerationInputs.prepare([np.zeros((64,64,3),np.uint8),np.zeros((5,64,96,3),np.uint8)],[64,27],64)
    x=torch.randn(sum(layout.lengths),64,dtype=torch.bfloat16)
    condition=torch.randn(2,64,64,dtype=torch.bfloat16)
    features=torch.randn(2,32)
    xn=x.to('npu').requires_grad_();cn=condition.to('npu').requires_grad_()
    xc=x.float().requires_grad_();cc=condition.float().requires_grad_()
    y=model.predict(xn,cn,features.to('npu'),layout.to('npu'))
    expected=reference.predict(xc,cc,features,layout)
    upstream=torch.randn_like(expected).div(100)
    (y*upstream.to('npu')).sum().backward();(expected*upstream).sum().backward()
    torch.npu.synchronize()
    result=dict(device=torch.npu.get_device_name(0),forward_rms=rms(y.detach().cpu(),expected),
        input_gradient_rms=rms(xn.grad.cpu(),xc.grad),condition_gradient_rms=rms(cn.grad.cpu(),cc.grad),
        peak_bytes=torch.npu.max_memory_allocated())
    result['passed']=max(result[k] for k in ('forward_rms','input_gradient_rms','condition_gradient_rms'))<.03
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
