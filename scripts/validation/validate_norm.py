"""RMSNorm oracle; each deterministic mode must use a fresh process."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from infra.kernels import rms_norm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.npu.set_device(0)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.manual_seed(42)
    rows=[]
    for shape in ((48,1,1,16),(48,1,2,16),(48,1,4,16),(8192,1,2,128),(8192,1,8,128)):
        for magnitude in (1.,1e-5):
            x=torch.randn(shape,device='npu',dtype=torch.bfloat16,requires_grad=True)
            w=torch.ones(shape[-1],device='npu',dtype=torch.bfloat16,requires_grad=True)
            dy=torch.randn_like(x)*magnitude
            y=rms_norm(x,w,1e-6)
            dx,dw=torch.autograd.grad(y,(x,w),dy)
            xr=x.detach().cpu().float().requires_grad_()
            wr=w.detach().cpu().float().requires_grad_()
            yr=xr*torch.rsqrt(xr.square().mean(-1,keepdim=True)+1e-6)*wr
            gx,gw=torch.autograd.grad(yr,(xr,wr),dy.cpu().float())
            errors=[float((a.cpu().float()-b).norm()/b.norm().clamp_min(1e-30)) for a,b in ((dx,gx),(dw,gw))]
            assert max(errors)<.01,(shape,magnitude,errors)
            rows.append(dict(shape=shape,gradient_magnitude=magnitude,gradient_relative_l2=errors))
    args.output.write_text(json.dumps(dict(passed=True,deterministic=args.deterministic,checks=rows),indent=2)+'\n')
    print(json.dumps(dict(passed=True,output=str(args.output))))


if __name__=='__main__':
    main()
