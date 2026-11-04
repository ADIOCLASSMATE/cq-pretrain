"""Bounded, standalone numerical localization; never used for timing."""
import torch
import torch_npu
import models.delta_ascendc as b
from scripts.validation.delta64_probe import error_stats

torch.npu.set_device(0)
b.initialize()
torch.manual_seed(301)
q, k, v = [torch.randn(1,129,16,128).bfloat16().npu().transpose(1,2).contiguous() for _ in range(3)]
g = (-torch.rand(1,129,16)*.2).npu()
beta = (torch.rand(1,129,16)*.8+.1).npu()
q, _ = b.call('l2_fwd', q)
k, _ = b.call('l2_fwd', k)
o, _, gf, af, *_ = b.call('chunk_gated_delta_rule_fwd',q,k,v,g,beta,disable_recompute=True)
gs = b.call('chunk_local_cumsum',g.transpose(1,2).contiguous(),64)
aa = b.call('chunk_scaled_dot_kkt',k,gs,beta.transpose(1,2).contiguous(),chunk_size=64)
ac = b.call('solve',aa.transpose(1,2).contiguous(),output_dtype=k.dtype).transpose(1,2).contiguous()
print('gate',error_stats(gs,gf.transpose(1,2)),flush=True)
print('inverse',error_stats(ac,af),flush=True)
print('inverse_per_head',[error_stats(ac[:,h],af[:,h])['relative_l2'] for h in range(16)],flush=True)
print('inverse_diag_head0',torch.stack([ac[0,0,torch.arange(129),torch.arange(129)%64],af[0,0,torch.arange(129),torch.arange(129)%64]]).cpu(),flush=True)
refg=torch.empty_like(gs.cpu())
refa=torch.empty_like(aa.cpu()).double()
for s in range(0,129,64):
 e=min(s+64,129); n=e-s
 rg=g[:,s:e].cpu().transpose(1,2).double().cumsum(-1)
 refg[:,:,s:e]=rg
 rk=k[:,:,s:e].cpu().double()
 kk=(rk@rk.transpose(-1,-2)) * beta[:,s:e].cpu().transpose(1,2).double().unsqueeze(-1)
 kk=kk*torch.exp(rg.unsqueeze(-1)-rg.unsqueeze(-2))
 kk=kk.tril(-1)
 refa[:,:,s:e].zero_(); refa[:,:,s:e,:n]=kk
print('gate_cpu',error_stats(gs,refg),flush=True)
print('kkt_cpu',error_stats(aa,refa),flush=True)
print('KKT sample',aa[0,0,:4,:4].cpu(),refa[0,0,:4,:4],flush=True)
print('inverse sample',ac[0,0,:4,:4].cpu(),af[0,0,:4,:4].cpu(),flush=True)
for name,a,gate in [('fused',af,gf.transpose(1,2).contiguous()),('cached',ac,gs)]:
 w,u=b.call('recompute_w_u_fwd',k,v,beta.transpose(1,2).contiguous(),a,64,g=gate)
 h,vn,_=b.call('chunk_gated_delta_rule_fwd_h',k,w,u,g=gate,chunk_size=64)
 out=b.call('chunk_fwd_o',q,k,vn,h,128**-.5,g=gate,chunk_size=64)
 print(name,'out_vs_fused',error_stats(out.transpose(1,2),o),flush=True)
