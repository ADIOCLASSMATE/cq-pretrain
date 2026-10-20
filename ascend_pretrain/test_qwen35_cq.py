"""Independent recurrence, pretrained-shape and CQ visibility checks."""
import ast
import copy
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from .config import tiny_config
from .data import fake_batch
from .mainline import build_plan
from .qwen35 import GatedDelta,recurrent_delta,QwenBlock,RMSNorm,unfused_parameters


def official_recurrent():
    path=Path(__file__).resolve().parents[1]/'assets/Qwen3.5-0.8B/reference_modeling_qwen3_5.py'
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
              ('l2norm','torch_recurrent_gated_delta_rule')]
    for node in selected:node.decorator_list=[]
    namespace={'torch':torch};exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['torch_recurrent_gated_delta_rule']


def pure_text_reference(layer,x):
    """Official causal Qwen GDN, without CQ plans/events or shared adapter math."""
    h,d=layer.cfg.linear_heads,layer.cfg.linear_head_dim
    raw=layer.in_proj_qkv(x)
    conv=F.silu(F.conv1d(raw.T[None],layer.conv1d.weight,
        padding=layer.cfg.linear_conv_kernel-1,groups=raw.shape[-1])[...,:len(x)]).transpose(1,2)
    q,k,v=[t.reshape(1,len(x),h,d) for t in conv.chunk(3,-1)]
    g=-layer.A_log.float().exp()*F.softplus(layer.in_proj_a(x).float()+layer.dt_bias.float())
    beta=layer.in_proj_b(x).sigmoid()
    y,_=official_recurrent()(q,k,v,g[None],beta[None],use_qk_l2norm_in_kernel=True)
    z=layer.in_proj_z(x).reshape(len(x),h,d)
    y=y[0];y=y.float()*torch.rsqrt(y.float().square().mean(-1,keepdim=True)+layer.cfg.norm_eps)
    y=y*layer.norm.weight*F.silu(z.float())
    return layer.out_proj(y.flatten(-2))


class QwenCQContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_official_recurrence_and_gradients(self):
        torch.manual_seed(12)
        values=[torch.randn(1,65,2,8) for _ in range(3)]
        values.extend((-torch.rand(1,65,2),torch.rand(1,65,2)))
        a=[v.requires_grad_() for v in values];b=[v.detach().clone().requires_grad_() for v in a]
        y=recurrent_delta(*a,(0,65));z=official_recurrent()(*b,use_qk_l2norm_in_kernel=True)[0]
        torch.testing.assert_close(y,z,atol=2e-7,rtol=2e-5)
        upstream=torch.randn_like(y);(y*upstream).sum().backward();(z*upstream).sum().backward()
        for x,w in zip(a,b):torch.testing.assert_close(x.grad,w.grad,atol=3e-6,rtol=2e-4)

    def test_text_content_is_official_qwen(self):
        cfg=tiny_config(fused=False);batch=fake_batch(cfg,17,1,1)
        batch.image_queries.zero_();batch.image_indices=batch.image_indices[:0]
        batch.sigma=torch.arange(17)[None];batch.segments.zero_()
        batch.mainline=build_plan(batch,cfg)
        plan=batch.mainline.delta_plan;torch.manual_seed(42)
        layer=GatedDelta(cfg);ref=copy.deepcopy(layer)
        x=torch.randn(plan.hidden_length,1,cfg.width,requires_grad=True)
        r=x[:17,0].detach().clone().requires_grad_()
        actual=layer(x,plan)[:17,0];expected=pure_text_reference(ref,r)
        torch.testing.assert_close(actual,expected,atol=3e-7,rtol=2e-5)
        upstream=torch.randn_like(actual);(actual*upstream).sum().backward();(expected*upstream).sum().backward()
        torch.testing.assert_close(x.grad[:17,0],r.grad,atol=1e-6,rtol=3e-4)
        self.assertEqual(x.grad[17:].count_nonzero(),0)
        for (name,p),(_,r) in zip(layer.named_parameters(),ref.named_parameters()):
            torch.testing.assert_close(p.grad,r.grad,atol=2e-6,rtol=4e-4,msg=name)

    def test_query_and_future_do_not_write_state(self):
        cfg=tiny_config(fused=False);batch=fake_batch(cfg,24,1,1)
        plan=batch.mainline.delta_plan;layer=GatedDelta(cfg).double()
        x=torch.randn(plan.hidden_length,1,cfg.width,dtype=torch.float64)
        before=layer(x,plan)
        changed=x.clone();changed[24:]+=torch.randn_like(changed[24:])*10
        after=layer(changed,plan)
        torch.testing.assert_close(before[:24],after[:24],atol=0,rtol=0)
        image=batch.image_indices[0];target_sigma=batch.sigma[0,image[0]]
        changed=x.clone();changed[image]+=torch.randn_like(changed[image])*10
        after=layer(changed,plan)
        qs=batch.mainline.query_positions
        keep=(batch.sigma[0,qs]<=target_sigma).nonzero().flatten()+24
        torch.testing.assert_close(before[keep],after[keep],atol=0,rtol=0)
        # Each image output has access to the final group state, including its
        # last spatial token, while CQ's own image queries remain strictly prior.
        changed=x.clone();changed[image[-1]]+=3*torch.randn_like(changed[image[-1]])
        after=layer(changed,plan)
        self.assertGreater((before[image[0]]-after[image[0]]).abs().max().item(),1e-8)

    def test_layout_and_offset_norm(self):
        cfg=tiny_config();layers=[QwenBlock(cfg,i) for i in range(24)]
        self.assertEqual(sum(hasattr(x,'linear_attn') for x in layers),18)
        self.assertEqual(sum(hasattr(x,'self_attn') for x in layers),6)
        norm=RMSNorm(16);norm.weight.data.normal_()
        x=torch.randn(5,16,dtype=torch.bfloat16)
        expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*(1+norm.weight.float())).bfloat16()
        torch.testing.assert_close(norm(x),expected,atol=0,rtol=0)

    def test_fused_projection_outputs_and_every_gradient(self):
        cfg=tiny_config(fused=False,linear_value_heads=4)
        batch=fake_batch(cfg,17,1,1);plan=batch.mainline
        fused_cfg=copy.deepcopy(cfg);fused_cfg.qwen_fused_projections=True
        for index in (0,3):
            torch.manual_seed(12)
            reference=QwenBlock(cfg,index).double();actual=QwenBlock(fused_cfg,index).double()
            source={'layers.0.'+n.replace('.impl.','.'):p for n,p in reference.named_parameters()}
            with torch.no_grad():
                for name,param in actual.named_parameters():
                    for key,view in unfused_parameters('layers.0.'+name,param,fused_cfg).items():view.copy_(source[key])
            x=torch.randn(plan.delta_plan.hidden_length,1,cfg.width,dtype=torch.float64,requires_grad=True)
            other=x.detach().clone().requires_grad_();rope=(plan.rope_cos.double(),plan.rope_sin.double())
            result=actual(x,rope,plan,17);expected=reference(other,rope,plan,17)
            torch.testing.assert_close(result,expected,atol=2e-10,rtol=2e-9)
            probe=torch.randn_like(result);(result*probe).sum().backward();(expected*probe).sum().backward()
            torch.testing.assert_close(x.grad,other.grad,atol=2e-10,rtol=2e-9)
            for name,param in actual.named_parameters():
                for key,view in unfused_parameters('layers.0.'+name,param.grad,fused_cfg).items():
                    torch.testing.assert_close(view,source[key].grad,atol=2e-9,rtol=2e-8,msg=key)


if __name__=='__main__':unittest.main()
