"""CPU semantic, weight-layout, gradient, packing and native-resolution contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from .config import tiny_config
from .data import fake_batch
from .input_pipeline.packing import Packer
from .input_pipeline.runtime import empty_batch
from .model import JointModel
from .vision import (NativeImageProcessor, QwenVisionTower, VisionInputs, asset_path, vision_config)


class Documents:
    def __init__(self,rows,index=0):self.rows,self.index=rows,index
    def peek(self):return self.rows[self.index] if self.index<len(self.rows) else None
    def advance(self):self.index+=1
    def prefetch_images(self,*args):pass
    def image(self,doc,element):return doc['elements'][element]['pixels']
    def state_dict(self):return dict(index=self.index)


class NativeVisionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory()
        cls.assets=Path(cls.temp.name)
        cls.vc=dict(depth=2,hidden_act='gelu_pytorch_tanh',hidden_size=32,
            intermediate_size=64,in_channels=3,num_heads=4,num_position_embeddings=64,
            out_hidden_size=64,patch_size=16,spatial_merge_size=2,temporal_patch_size=2,
            deepstack_visual_indexes=[])
        (cls.assets/'config.json').write_text(json.dumps(dict(vision_config=cls.vc)))
        # Same public processing parameters, smaller explicit test pixel budget.
        (cls.assets/'preprocessor_config.json').write_text(json.dumps(dict(
            image_processor_type='Qwen2VLImageProcessorFast',patch_size=16,temporal_patch_size=2,
            merge_size=2,image_mean=[.5]*3,image_std=[.5]*3,
            size=dict(shortest_edge=65536,longest_edge=16777216))))
        cls.tokens=dict(bos=1,eos=2,boi=3,eoi=4,image=0)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def cfg(self,**kwargs):
        return tiny_config(vision_encoder='qwen35',vision_assets=str(self.assets),
            vision_min_pixels=1024,vision_max_pixels=16384,image_condition_queries=16,
            fused=False,**kwargs)

    def layout(self):
        processor=NativeImageProcessor(self.cfg(),128)
        rng=np.random.default_rng(71)
        inputs=[rng.integers(0,256,(h,w,3),dtype=np.uint8) for h,w in ((64,128),(96,64),(192,64))]
        processed=[processor(x) for x in inputs]
        counts=[g[1]*g[2]//4 for _,_,g in processed]
        layout=VisionInputs.prepare(torch.cat([p for _,p,_ in processed]),[g for _,_,g in processed],
                                    torch.arange(sum(counts)),self.vc)
        return processor,inputs,layout

    def test_official_preprocess_and_native_shape(self):
        processor,images,layout=self.layout()
        from transformers import Qwen2VLImageProcessorFast
        official=Qwen2VLImageProcessorFast.from_pretrained(str(self.assets),local_files_only=True,
            min_pixels=1024,max_pixels=16384)
        expected=official(images=[Image.fromarray(x) for x in images],return_tensors='pt')
        torch.testing.assert_close(layout.pixel_values,expected['pixel_values'],rtol=0,atol=0)
        # The final grid exceeds the learned 8x8 test embedding table.
        self.assertEqual(layout.grid_thw,((1,4,8),(1,6,4),(1,12,4)))
        for x in images:self.assertEqual(processor(x)[0].shape,(32,32,3))
        # The max-pixel limit shrinks the complete image; it does not center crop.
        huge=np.zeros((512,1024,3),dtype=np.uint8)
        _,_,grid=processor(huge)
        self.assertLessEqual(grid[1]*grid[2]*256,16384)
        self.assertGreater(grid[2],grid[1])

    def test_official_tower_forward_backward_and_isolation(self):
        _,_,layout=self.layout()
        with patch('transformers.utils.import_utils.is_torch_npu_available',return_value=False):
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        config=Qwen3VLVisionConfig(**self.vc);config._attn_implementation='sdpa'
        reference=Qwen3VLVisionModel(config)
        torch.manual_seed(902)
        tower=QwenVisionTower(self.vc,fused=False)
        reference.load_state_dict(tower.state_dict(),strict=True)
        layout.pixel_values.requires_grad_()
        ref_pixels=layout.pixel_values.detach().clone().requires_grad_()
        gathers=[]
        hook=tower.patch_embed.proj.register_forward_pre_hook(lambda *args:gathers.append(True))
        actual=tower(layout)
        hook.remove();self.assertEqual(gathers,[True])
        expected,_=reference(ref_pixels,torch.tensor(layout.grid_thw))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        upstream=torch.randn_like(actual)
        (actual*upstream).sum().backward();(expected*upstream).sum().backward()
        torch.testing.assert_close(layout.pixel_values.grad,ref_pixels.grad,atol=2e-6,rtol=3e-4)
        for (name,p),(rn,r) in zip(tower.named_parameters(),reference.named_parameters()):
            self.assertEqual(name,rn)
            torch.testing.assert_close(p.grad,r.grad,atol=8e-6,rtol=3e-4,msg=name)
        changed=copy.deepcopy(layout);changed.pixel_values=layout.pixel_values.detach().clone()
        changed.pixel_values[:layout.lengths[0]].add_(17)
        with torch.no_grad():result=tower(changed)
        torch.testing.assert_close(result[layout.lengths[0]//4:],actual[layout.lengths[0]//4:],atol=0,rtol=0)

    def test_extreme_strip_retains_pixels(self):
        from PIL import ImageOps
        processor=NativeImageProcessor(self.cfg(),128)
        for height,width in ((1,511),(511,1)):
            pixels=np.arange(height*width*3,dtype=np.uint8).reshape(height,width,3)
            original=Image.fromarray(pixels);padded=processor.supported_aspect(original)
            self.assertLessEqual(max(padded.size)/min(padded.size),200)
            x,y=(padded.width-width)//2,(padded.height-height)//2
            np.testing.assert_array_equal(np.asarray(padded)[y:y+height,x:x+width],pixels)
            target,processed,grid=processor(pixels)
            expected=ImageOps.fit(original,(32,32),Image.Resampling.LANCZOS,centering=(.5,.5))
            np.testing.assert_array_equal(target,np.asarray(expected))
            self.assertEqual(len(processed),grid[1]*grid[2]);self.assertTrue(torch.isfinite(processed).all())

    def test_joint_methods_strict_queries_and_gradients(self):
        for method in ('CQ_IT','Bi_I+CQ_T'):
            cfg=self.cfg(method=method)
            batch=fake_batch(cfg,64)
            self.assertGreater(cfg.image_condition_queries,batch.image_indices.shape[1])
            self.assertEqual(batch.mainline.condition_indices.shape,(2,16))
            self.assertEqual(batch.mainline.condition_indices.unique().numel(),32)
            torch.manual_seed(211);model=JointModel(cfg)
            torch.nn.init.normal_(model.renderer.output.weight,std=.02)
            reference=copy.deepcopy(model)
            a=model(batch,return_hidden=True);b=reference(batch,reference=True,return_hidden=True)
            for name in a:torch.testing.assert_close(a[name],b[name],atol=3e-6,rtol=2e-4,msg=method+'/'+name)
            a['loss'].backward();b['loss'].backward()
            for (name,p),(rn,r) in zip(model.named_parameters(),reference.named_parameters()):
                self.assertEqual(name,rn);self.assertIsNotNone(p.grad,name)
                torch.testing.assert_close(p.grad,r.grad,atol=3e-6,rtol=3e-4,msg=method+'/'+name)
            changed=copy.deepcopy(batch)
            changed.vision.pixel_values[:batch.vision.lengths[0]].add_(17)
            with torch.no_grad():result=model.backbone(changed)
            # Changing the clean target cannot change its generation condition.
            torch.testing.assert_close(result[batch.mainline.condition_indices[0]],
                a['hidden'][batch.mainline.condition_indices[0]],atol=0,rtol=0)
            after=int(batch.image_indices[0].max())+2
            self.assertGreater((result[after]-a['hidden'][after]).abs().max().item(),1e-6)
            # Query-to-query attention remains absent in CQ and bidirectional in Bi.
            from . import mainline
            original=mainline.packed_layer
            qstart=(64+len(batch.mainline.query_positions)-32 if method=='CQ_IT' else
                    64+int(batch.mainline.generation_condition[0,0])-batch.mainline.prefix)
            def perturb(layer,hidden,*rest):
                if layer is model.layers[0]:
                    hidden=hidden.clone();hidden[qstart,0]+=torch.linspace(-2,2,cfg.width)
                return original(layer,hidden,*rest)
            with patch.object(mainline,'packed_layer',perturb),torch.no_grad():perturbed=model.backbone(batch)
            delta=(perturbed[batch.mainline.condition_indices[0,-1]]-a['hidden'][batch.mainline.condition_indices[0,-1]]).abs().max().item()
            if method=='CQ_IT':self.assertEqual(delta,0)
            else:self.assertGreater(delta,1e-6)

    def test_fragmentation_resume_and_document_isolation(self):
        rows=[dict(id='a',elements=[dict(type='text',input_ids=[5]*22),
            dict(type='image',pixels=np.zeros((64,128,3),dtype=np.uint8)),dict(type='text',input_ids=[6]*5)]),
            dict(id='b',elements=[dict(type='text',input_ids=[7]*3),
            dict(type='image',pixels=np.full((64,64,3),200,dtype=np.uint8)),dict(type='text',input_ids=[8]*8)])]
        cfg=self.cfg();packer=Packer(Documents(rows),cfg,32,self.tokens)
        first=packer.next();state=packer.state_dict();expected=[]
        while (item:=packer.next()) is not None:expected.append(item)
        resumed=Packer(Documents(rows,state['documents']['index']),cfg,32,self.tokens,state)
        actual=[]
        while (item:=resumed.next()) is not None:actual.append(item)
        self.assertEqual(len(actual),len(expected))
        self.assertEqual(sum(x[1]['body_tokens'] for x in [first]+actual),38)
        self.assertEqual(sum(x[1]['images'] for x in [first]+actual),2)
        for (a,ar),(b,br) in zip(actual,expected):
            self.assertEqual(ar,br);torch.testing.assert_close(a.ids,b.ids,atol=0,rtol=0)
            if a.vision is not None:
                torch.testing.assert_close(a.vision.pixel_values,b.vision.pixel_values,atol=0,rtol=0)
        # Two separate documents in one physical pack remain isolated.
        for method in ('CQ_IT','Bi_I+CQ_T'):
            cfg=self.cfg(method=method)
            batch,_=Packer(Documents(rows),cfg,96,self.tokens).next()
            model=JointModel(cfg)
            with torch.no_grad():before=model.backbone(batch)
            changed=copy.deepcopy(batch)
            changed.ids[changed.segments==0]=9
            changed.vision.pixel_values[:batch.vision.lengths[0]].add_(13)
            with torch.no_grad():after=model.backbone(changed)
            keep=batch.segments[0].eq(1).nonzero().flatten()
            torch.testing.assert_close(after[keep],before[keep],atol=0,rtol=0)
            torch.testing.assert_close(after[batch.mainline.condition_indices[1]],before[batch.mainline.condition_indices[1]],atol=0,rtol=0)

    def test_recompute_and_empty_text_rank(self):
        cfg=self.cfg();batch=fake_batch(cfg,64)
        model=JointModel(cfg);reference=copy.deepcopy(model)
        model.patch_embed.visual.recompute=True
        model(batch)['loss'].backward();reference(batch)['loss'].backward()
        for (name,p),(rn,r) in zip(model.named_parameters(),reference.named_parameters()):
            torch.testing.assert_close(p.grad,r.grad,atol=0,rtol=0,msg=name)
        model.zero_grad(set_to_none=True)
        empty=empty_batch(cfg,self.tokens).to('cpu')
        result=model(empty);self.assertEqual(result['loss'].item(),0)
        result['loss'].backward()
        # Optional visual/renderer modules stay under patch_embed/renderer, so
        # the existing ordered DDP buckets can mark all their absent grads ready.
        from types import SimpleNamespace
        from .input_pipeline.parallel import OrderedGradientBuckets
        ready=set()
        group=SimpleNamespace(register_grad_ready=lambda p,force:ready.add(id(p)))
        wrapped=SimpleNamespace(module=model,param_to_bucket_group={p:group for p in model.parameters()},force_all_reduce=False)
        OrderedGradientBuckets.absent_images(SimpleNamespace(wrapped=wrapped))
        self.assertEqual(ready,{id(p) for p in model.parameters() if p.grad is None})

    def test_frozen_vision_preserves_outputs_and_downstream_training(self):
        for freeze_merger in (False,True):
            with self.subTest(freeze_merger=freeze_merger):
                cfg=self.cfg(freeze_vision=True,freeze_vision_merger=freeze_merger,
                             checkpoint_vision=True)
                batch=fake_batch(cfg,64)
                batch.vision.pixel_values.requires_grad_()
                torch.manual_seed(211);model=JointModel(cfg).train()
                torch.nn.init.normal_(model.renderer.output.weight,std=.02)
                reference=copy.deepcopy(model)
                # Same forward with a fully differentiable encoder is the
                # oracle for every remaining trainable parameter's gradient.
                tower=reference.patch_embed.visual
                tower.freeze_encoder=False;tower.requires_grad_(True);tower.train()
                if freeze_merger:tower.merger.requires_grad_(False)
                reference_batch=copy.deepcopy(batch)
                frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
                self.assertTrue(frozen)
                self.assertFalse(model.patch_embed.visual.blocks.training)
                self.assertFalse(model.patch_embed.visual.recompute)
                self.assertEqual(model.patch_embed.visual.merger.training,not freeze_merger)
                with patch('ascend_pretrain.vision.checkpoint',side_effect=AssertionError('Frozen encoder recomputed')):
                    actual=model(batch)
                    actual['loss'].backward()
                expected=reference(reference_batch);expected['loss'].backward()
                torch.testing.assert_close(actual['loss'],expected['loss'],atol=0,rtol=0)
                self.assertIsNone(batch.vision.pixel_values.grad)
                refs=dict(reference.named_parameters())
                for name,p in model.named_parameters():
                    if not p.requires_grad:self.assertIsNone(p.grad,name)
                    else:
                        self.assertIsNotNone(p.grad,name)
                        torch.testing.assert_close(p.grad,refs[name].grad,atol=0,rtol=0,msg=name)
                trainable=[p for p in model.parameters() if p.requires_grad]
                optimizer=torch.optim.AdamW(trainable,lr=1e-3)
                tracked={id(p) for group in optimizer.param_groups for p in group['params']}
                self.assertTrue(all(id(p) not in tracked for p in model.parameters() if not p.requires_grad))
                before=model.renderer.output.weight.detach().clone()
                optimizer.step()
                self.assertFalse(torch.equal(before,model.renderer.output.weight))
                for name,p in model.named_parameters():
                    if name in frozen:torch.testing.assert_close(p,frozen[name],atol=0,rtol=0,msg=name)
                # Pure-text ranks may skip both optional modules. Frozen
                # parameters have no DDP buckets and must never be marked ready.
                from types import SimpleNamespace
                from .input_pipeline.parallel import OrderedGradientBuckets
                ready=set();group=SimpleNamespace(register_grad_ready=lambda p,force:ready.add(id(p)))
                wrapped=SimpleNamespace(module=model,param_to_bucket_group={p:group for p in trainable},force_all_reduce=False)
                OrderedGradientBuckets.absent_images(SimpleNamespace(wrapped=wrapped))
                expected_ready={id(p) for module in (model.patch_embed,model.renderer)
                                for p in module.parameters() if p.requires_grad}
                self.assertEqual(ready,expected_ready)


if __name__=='__main__':unittest.main()
