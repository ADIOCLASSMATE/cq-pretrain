import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from .ema import RendererEMA
from . import checkpoint as checkpoint_io


def fixture():
    model=torch.nn.Module();model.renderer=torch.nn.Module()
    model.renderer.weight=torch.nn.Parameter(torch.arange(24,dtype=torch.bfloat16).reshape(4,6)/16)
    model.renderer.gain=torch.nn.Parameter(torch.ones(4,dtype=torch.float32))
    model.backbone=torch.nn.Linear(2,2,bias=False).to(torch.bfloat16)
    weight=model.renderer.weight;gain=model.renderer.gain
    master=weight.detach().flatten()[3:17].float().clone().requires_grad_()
    alias=gain.detach()[1:3].requires_grad_()
    inner=torch.optim.AdamW([master,alias],lr=.01)
    opt=SimpleNamespace(optimizer=inner,model_float16_groups=[[weight]],
        shard_fp32_from_float16_groups=[[master]],model_fp32_groups=[[gain]],shard_fp32_groups=[[alias]],
        _get_model_param_range_map=lambda p:{'param':SimpleNamespace(start=3,end=17) if p is weight else SimpleNamespace(start=1,end=3)})
    group=SimpleNamespace(params=list(model.parameters()),start_param_sync=lambda **kwargs:None)
    wrapped=SimpleNamespace(bucket_groups=[group],expert_parallel_bucket_groups=[],use_forward_hook=True,
                            remove_forward_pre_hook_handles={},
                            start_param_sync=lambda **kwargs:None)
    return model,opt,wrapped


class EMATests(unittest.TestCase):
    def test_fp32_recurrence_partial_shards_and_exception_restore(self):
        model,opt,wrapped=fixture();ema=RendererEMA(model,opt,wrapped,.9999)
        self.assertEqual(ema.local_bytes,16*4)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        expected=[s.double().clone() for s in ema.shadows]
        with torch.no_grad():
            for i in range(4):
                for master in ema.masters:master.add_(.01*(i+1))
                expected=[(.9999*e+.0001*m.detach().double()).float().double()
                          for e,m in zip(expected,ema.masters)]
                ema.update()
        for actual,want in zip(ema.shadows,expected):
            torch.testing.assert_close(actual,want.float(),rtol=2e-6,atol=2e-7)
            self.assertEqual(actual.dtype,torch.float32)
        ema._publish()
        online_model={n:p.detach().clone() for n,p in model.named_parameters()}
        online_masters=[m.detach().clone() for m in ema.masters]
        wrapped.remove_forward_pre_hook_handles['sentinel']=object()
        with self.assertRaisesRegex(RuntimeError,'Disable DDP'):
            with ema.average_parameters():pass
        wrapped.remove_forward_pre_hook_handles.clear()
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with ema.average_parameters():
                torch.testing.assert_close(model.renderer.weight.flatten()[3:17],ema.shadows[0].to(torch.bfloat16),rtol=0,atol=0)
                torch.testing.assert_close(model.renderer.gain[1:3],ema.shadows[1],rtol=0,atol=0)
                torch.testing.assert_close(model.backbone.weight,before['backbone.weight'],rtol=0,atol=0)
                with self.assertRaisesRegex(RuntimeError,'Checkpointing'):ema.state_dict()
                raise RuntimeError('deliberate')
        for actual,expected in zip(ema.masters,online_masters):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for name,param in model.named_parameters():torch.testing.assert_close(param,online_model[name],rtol=0,atol=0)
        self.assertFalse(ema.active);self.assertEqual(ema.num_updates,4)

    def test_strict_state_and_checkpoint_resume(self):
        model,opt,wrapped=fixture();ema=RendererEMA(model,opt,wrapped,.9999)
        with torch.no_grad():
            for m in ema.masters:m.add_(.1)
        ema.update();ema._publish();state=copy.deepcopy(ema.state_dict())
        changed=copy.deepcopy(state);changed['decay']=.99
        with self.assertRaisesRegex(ValueError,'differs'):ema.load_state_dict(changed)
        changed=copy.deepcopy(state);changed['layout'][0]['start']+=1
        with self.assertRaisesRegex(ValueError,'differs'):ema.load_state_dict(changed)
        parallel=SimpleNamespace(get_tensor_model_parallel_rank=lambda:0,get_data_parallel_group=lambda:None)
        npu=SimpleNamespace(get_rng_state=torch.get_rng_state,set_rng_state=torch.set_rng_state)
        cfg=SimpleNamespace(to_dict=lambda:{'tiny_ema':True});topology={'tp':1,'dp':1}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(checkpoint_io.P,'state',return_value=parallel), \
             patch.object(checkpoint_io.P,'dp_rank',return_value=0), \
             patch.object(checkpoint_io.dist,'get_rank',return_value=0), \
             patch.object(checkpoint_io.dist,'get_world_size',return_value=1), \
             patch.object(checkpoint_io.dist,'get_global_rank',return_value=0), \
             patch.object(checkpoint_io.dist,'barrier'),patch.object(checkpoint_io.dist,'broadcast'), \
             patch.object(torch,'npu',npu,create=True):
            path=Path(temporary)/'step_0000001'
            checkpoint_io.save(path,model,opt,1,cfg,topology,{'ema':.9999},ema=ema)
            with torch.no_grad():
                for s in ema.shadows:s.zero_()
            ema.num_updates=999
            self.assertEqual(checkpoint_io.load(path,model,opt,cfg,topology,{'ema':.9999},ema=ema),1)
            self.assertEqual(ema.num_updates,1)
            for actual,expected in zip(ema.shadows,state['shadows']):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'preserve'):
                checkpoint_io.load(path,model,opt,cfg,topology,{'ema':.9999})
            manifest_path=path/'manifest.json'
            manifest=json.loads(manifest_path.read_text());manifest.pop('ema')
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'schema and renderer EMA'):
                checkpoint_io.load(path,model,opt,cfg,topology,{'ema':.9999})


if __name__=='__main__':unittest.main()
