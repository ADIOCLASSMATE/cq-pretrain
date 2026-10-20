"""Compare ragged training against the original public MOSS transformer source."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import tiny_config
from .generation import GenerationInputs
from .moss import MOSSRenderer


def official_transformer():
    root=Path(__file__).resolve().parent
    assets=Path(json.loads((root/'configs/qwen35_9b_moss.json').read_text())['moss_assets'])
    source=assets/'pipeline_moss_image.py'
    if not source.is_file():raise unittest.SkipTest('Pinned official MOSS source is not installed')
    manifest=json.loads((assets/'manifest.json').read_text())
    if manifest['revision']!='1ceede80badddd6a7812069619edaa5633587192':
        raise AssertionError('Unexpected MOSS reference revision')
    names={'_RMSNorm','_situ','_validate_skip_layers','_rotate','_attention',
           '_TextBlock','_TextProjection','_Block','_RoPE','MOSSImageTransformer'}
    nodes=[n for n in ast.parse(source.read_text()).body
           if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes}!=names:raise AssertionError('Incomplete official MOSS reference')
    # Remove only Diffusers registration/packaging. Every forward function,
    # attention operation, normalization and positional calculation stays original.
    for node in nodes:
        if node.name=='MOSSImageTransformer':
            node.bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())]
            for child in node.body:
                if isinstance(child,ast.FunctionDef):child.decorator_list=[]
    tree=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    scope={'torch':torch,'nn':nn,'F':F,'Transformer2DModelOutput':SimpleNamespace}
    exec(compile(tree,str(source),'exec'),scope)
    return scope['MOSSImageTransformer']


def gradient_views(model):
    result={}
    for name,p in model.named_parameters():
        name=name.replace('.impl.','.')
        if name.endswith('.proj.qkv.weight'):
            for label,grad in zip(('q','k','v'),p.grad.chunk(3,0)):
                result[name.replace('.qkv.',f'.{label}.')]=grad
        else:result[name]=p.grad
    return result


class MOSSReferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference_class=official_transformer()

    def test_ragged_flow_loss_and_every_gradient_against_official_source(self):
        for recompute in (False,True):
            with self.subTest(recompute=recompute):
                torch.manual_seed(45)
                cfg=tiny_config(renderer='moss',latent_channels=128,flow_draws=1,
                    dit_width=128,dit_heads=4,dit_ffn=256,dit_layers=2,
                    moss_fused_qkv=True,checkpoint_dit=recompute,fused=False)
                model=MOSSRenderer(cfg)
                reference=self.reference_class(dim=128,ffn_dim=256,num_heads=4,num_layers=2,
                    text_dim=cfg.width,freq_dim=256,latent_channels=128)
                reference.config=SimpleNamespace(latent_channels=128,freq_dim=256)
                reference.load_state_dict({n:t.detach().clone() for n,t in model.official_tensors().items()},strict=True)
                images=[np.zeros((32,48,3),np.uint8),np.zeros((16,64,3),np.uint8)]
                layout=GenerationInputs.prepare(images,cfg.image_condition_queries,32)
                target=torch.randn(sum(layout.lengths),128)
                noise=torch.randn_like(target)
                times=torch.tensor([.2,.7])
                phase=(times*1000).double()[:,None]*10000**(-torch.arange(128,dtype=torch.float64)/128)
                layout.time_features=torch.cat((phase.cos(),phase.sin()),-1).float()
                conditions=torch.randn(2,cfg.image_condition_queries,cfg.width,requires_grad=True)
                other=conditions.detach().clone().requires_grad_()
                batch=SimpleNamespace(generation=layout,latents=target,noise=noise,times=times)
                actual=model(conditions,batch)
                expected=[];start=0
                for i,((h,w),length) in enumerate(zip(layout.grids,layout.lengths)):
                    clean,eps=target[start:start+length],noise[start:start+length]
                    xt=((1-times[i])*clean+times[i]*eps).T.reshape(1,128,h,w)
                    prediction=reference(xt,times[i:i+1]*1000,other[i:i+1],return_dict=False)[0]
                    velocity=(eps-clean).T.reshape_as(prediction)
                    expected.append((prediction.float()-velocity).square().mean())
                    start+=length
                expected=torch.stack(expected).mean()
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
                actual.backward();expected.backward()
                torch.testing.assert_close(conditions.grad,other.grad,atol=2e-6,rtol=2e-4)
                refs=dict(reference.named_parameters())
                grads=gradient_views(model)
                self.assertEqual(set(grads),set(refs))
                for name,grad in grads.items():
                    self.assertIsNotNone(grad,name);self.assertIsNotNone(refs[name].grad,name)
                    torch.testing.assert_close(grad,refs[name].grad,atol=3e-6,rtol=4e-4,msg=name)


if __name__=='__main__':unittest.main()
