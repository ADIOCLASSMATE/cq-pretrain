"""I2T subsampling saves generation work without dropping understanding data."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import test_qwen35_vision as fixtures
from .checkpoint import initialization_model_config
from .input_pipeline.packing import Packer,select_i2t_image
from .input_pipeline.vae import DevicePreparer
from .model import JointModel
from .renderer_schedule import backward_deferred_renderer


class Documents(fixtures.Documents):
    signature=dict(snapshot='sampling-fixture',seed=42,epoch=0,rank=0)

    def state_dict(self):
        return dict(index=self.index,signature=self.signature.copy())


def row(name,direction='I2T',height=64,width=64):
    text=lambda ids:dict(type='text',input_ids=np.asarray(ids,np.int32))
    image=dict(type='image',pixels=np.full((height,width,3),len(name)*13%256,np.uint8))
    elements=[text([11]),image,text([21,22,23])] if direction=='I2T' else [text([31,32]),image]
    return dict(id=name,format='caption',direction=direction,elements=elements)


class I2TImageSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NativeVisionContracts.setUpClass();cls.fixture=fixtures.NativeVisionContracts()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):fixtures.NativeVisionContracts.tearDownClass()

    def cfg(self,**kwargs):
        values=dict(renderer='moss',latent_channels=128,flow_draws=1,dit_width=128,dit_heads=4,
                    dit_ffn=256,dit_layers=2,preserve_documents=True,generation_token_budget=64,
                    i2t_image_loss_probability=.2,cfg_null_condition='i2t_prefix')
        values.update(kwargs)
        return self.fixture.cfg(**values)

    def packer(self,rows,cfg=None,state=None,length=256):
        docs=Documents(rows,state['documents']['index'] if state else 0)
        return Packer(docs,cfg or self.cfg(),length,self.fixture.tokens,state)

    def selected_id(self,selected):
        return next(f'sample-{i}' for i in range(100) if
                    select_i2t_image(f'sample-{i}',.2,42,0)==selected)

    def test_bernoulli_is_replayable_and_redraws_each_epoch(self):
        ids=[f'doc-{i}' for i in range(10000)]
        draws=[select_i2t_image(x,.2,42,0) for x in ids]
        self.assertTrue(1800 < sum(draws) < 2200)
        self.assertEqual(draws,[select_i2t_image(x,.2,42,0) for x in ids])
        self.assertNotEqual(draws,[select_i2t_image(x,.2,42,1) for x in ids])
        self.assertNotEqual(draws,[select_i2t_image(x,.2,43,0) for x in ids])
        self.assertFalse(any(select_i2t_image(x,0) for x in ids))
        self.assertTrue(all(select_i2t_image(x,1) for x in ids))

    def test_targets_are_compacted_but_all_understanding_images_and_text_remain(self):
        rows=[row(self.selected_id(False)),row('conditional','T2I',32,64),
              row(self.selected_id(True),height=64,width=96)]
        cfg=self.cfg();batch,receipt=self.packer(rows,cfg).next()
        full,_=self.packer(rows,self.cfg(i2t_image_loss_probability=1)).next()
        self.assertEqual(batch.generation_image_indices.tolist(),[1,2])
        self.assertEqual(batch.generation_image_count,2)
        self.assertEqual(receipt['images'],3)
        self.assertEqual((receipt['i2t_null_candidates'],receipt['i2t_null_images']),(2,1))
        self.assertEqual(receipt['generation_images'],2)
        self.assertEqual(batch.generation.grids,((2,4),(4,6)))
        for field in ('ids','text_indices','image_indices','sigma','row_col'):
            torch.testing.assert_close(getattr(batch,field),getattr(full,field),rtol=0,atol=0)
        torch.testing.assert_close(batch.vision.pixel_values,full.vision.pixel_values,rtol=0,atol=0)
        torch.testing.assert_close(batch.generation.loss_weights.sum(),torch.tensor(1.))
        self.assertEqual(sum(len(x) for x in batch.generation.pixels),2)
        preparer=DevicePreparer(cfg,device='cpu',encoder='fake')
        sizes=[]
        encode=preparer.vae.encode
        with patch.object(preparer.vae,'encode',side_effect=lambda x:(sizes.append(len(x)),encode(x))[1]):
            ready,_=preparer.prepare(batch,receipt)
        self.assertEqual(sum(sizes),2)
        self.assertEqual(ready.latents.shape,(32,128))
        model=JointModel(cfg);captured=[]
        handle=model.renderer.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        losses=model(ready,return_hidden=True);handle.remove()
        expected=losses['hidden'][ready.mainline.condition_indices[ready.generation_image_indices]]
        torch.testing.assert_close(captured[0],expected,rtol=0,atol=0)
        losses['loss'].backward()
        self.assertTrue(any(p.grad is not None for p in model.renderer.parameters()))

    def test_zero_targets_skip_vae_and_dit_but_keep_understanding_gradient(self):
        cfg=self.cfg(i2t_image_loss_probability=0)
        batch,receipt=self.packer([row('understand')],cfg).next()
        self.assertEqual(len(batch.image_indices),1)
        self.assertEqual(batch.generation_image_count,0)
        self.assertIsNone(batch.generation)
        preparer=DevicePreparer(cfg,device='cpu',encoder='fake')
        with patch.object(preparer.vae,'encode',side_effect=AssertionError('VAE should be skipped')):
            ready,randoms=preparer.prepare(batch,receipt)
        self.assertEqual(randoms,())
        model=JointModel(cfg)
        with patch.object(model.renderer,'forward',side_effect=AssertionError('DiT should be skipped')):
            losses=model(ready)
        self.assertEqual(losses['image_loss'].item(),0)
        self.assertGreater(losses['text_loss'].item(),0)
        losses['loss'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.patch_embed.parameters()))
        self.assertTrue(all(p.grad is None for p in model.renderer.parameters()))

    def test_non_caption_context_is_not_mistaken_for_null(self):
        rows=[row('conversation'),row('interleaved')]
        for entry in rows:entry['format']=entry['id'];entry['direction']=None
        batch,receipt=self.packer(rows,self.cfg(i2t_image_loss_probability=0)).next()
        self.assertEqual(batch.generation_image_count,2)
        self.assertEqual(receipt['i2t_null_candidates'],0)

    def test_restart_keeps_selection_and_rejects_changed_probability(self):
        rows=[row(f'item-{i}') for i in range(20)]
        cfg=self.cfg();packer=self.packer(rows,cfg,length=24)
        _,first=packer.next();expected,receipt=packer.next()
        resumed=self.packer(rows,cfg,first['next_state'],length=24)
        actual,replayed=resumed.next()
        self.assertEqual(receipt,replayed)
        torch.testing.assert_close(actual.generation_image_indices,expected.generation_image_indices)
        with self.assertRaisesRegex(ValueError,'packing policy'):
            self.packer(rows,self.cfg(i2t_image_loss_probability=.5),first['next_state'],length=24)

    def test_selected_targets_keep_chunked_loss_and_all_gradients(self):
        cfg=self.cfg(generation_token_budget=32)
        rows=[row(self.selected_id(False)),row('a','T2I'),row('b','T2I'),row('c','T2I')]
        batch,receipt=self.packer(rows,cfg).next()
        batch,_=DevicePreparer(cfg,device='cpu',encoder='fake').prepare(batch,receipt)
        self.assertEqual(len(batch.generation.microbatches),3)
        torch.manual_seed(29);reference=JointModel(cfg);staged=copy.deepcopy(reference)
        expected=reference(batch);expected['loss'].backward()
        actual=staged(batch,defer_renderer=True)
        backward_deferred_renderer(staged,batch,actual)
        torch.testing.assert_close(actual['loss'],expected['loss'],atol=3e-6,rtol=2e-5)
        for (name,p),(_,q) in zip(reference.named_parameters(),staged.named_parameters()):
            if p.grad is None:self.assertIsNone(q.grad,name)
            else:torch.testing.assert_close(p.grad,q.grad,atol=5e-6,rtol=5e-4,msg=name)

    def test_new_stage_can_change_sampling_but_invalid_configuration_fails(self):
        old=self.cfg(i2t_image_loss_probability=1,cfg_null_condition='empty_t2i').to_dict()
        old.pop('i2t_image_loss_probability');old.pop('cfg_null_condition')
        self.assertEqual(initialization_model_config(old),initialization_model_config(self.cfg().to_dict()))
        for p in (-.1,1.1,float('nan')):
            with self.assertRaisesRegex(ValueError,'probability'):self.cfg(i2t_image_loss_probability=p).validate()


if __name__=='__main__':unittest.main()
