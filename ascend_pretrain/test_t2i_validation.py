import copy
import unittest
import numpy as np
import torch

from .test_qwen35_vision import NativeVisionContracts
from .qualitative import make_batch,moss_sigmas
from .mainline import backbone
from .model import JointModel
from .vision import NativeImageProcessor
from .input_pipeline.packing import Packer


class ValidationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NativeVisionContracts.setUpClass()
        cls.fixture=NativeVisionContracts()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):NativeVisionContracts.tearDownClass()

    def cfg(self):
        return self.fixture.cfg(layers=4,renderer='moss',latent_channels=128,flow_draws=1,
            dit_width=128,dit_heads=4,dit_ffn=256,dit_layers=2)

    def test_pruned_image_queries_match_full_and_cannot_read_target(self):
        torch.manual_seed(27);cfg=self.cfg();tokens=self.fixture.tokens
        image=np.random.default_rng(9).integers(0,256,(64,96,3),dtype=np.uint8)
        processed=NativeImageProcessor(cfg,128)(image)
        full=make_batch(cfg,tokens,[11,12,13],processed,'T2I',prune=False)
        lean=make_batch(cfg,tokens,[11,12,13],processed,'T2I')
        model=JointModel(cfg).eval()
        with torch.no_grad():
            expected=backbone(model,full)[full.mainline.condition_indices]
            actual=backbone(model,lean,vision_features=torch.zeros(len(lean.vision.content_indices),cfg.width))[lean.mainline.condition_indices]
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        self.assertEqual(len(lean.mainline.query_positions),cfg.image_condition_queries)

    def test_pruned_text_query_matches_full_with_cached_vision(self):
        torch.manual_seed(33);cfg=self.cfg();tokens=self.fixture.tokens
        image=np.random.default_rng(3).integers(0,256,(64,64,3),dtype=np.uint8)
        processed=NativeImageProcessor(cfg,128)(image)
        full=make_batch(cfg,tokens,[19,20,21],processed,'I2T',prune=False)
        lean=make_batch(cfg,tokens,[19,20,21],processed,'I2T')
        model=JointModel(cfg).eval()
        with torch.no_grad():
            expected=backbone(model,full)[full.ids.shape[1]-1]
            features=model.patch_embed(lean.vision)
            actual=backbone(model,lean,vision_features=features)[lean.ids.shape[1]-1]
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        self.assertEqual(len(lean.mainline.query_positions),1)

    def test_caption_reservation_keeps_text_and_image_together(self):
        cfg=self.cfg();tokens=self.fixture.tokens
        processed=NativeImageProcessor(cfg,48)(np.zeros((64,64,3),np.uint8))
        class Documents:
            images_preprocessed=True
            def __init__(self):
                self.index=0
                self.docs=[dict(id='previous',format='text',elements=[dict(type='text',input_ids=np.arange(39)+10)]),
                    dict(id='caption',format='caption',direction='T2I',elements=[dict(type='text',input_ids=np.array([7,8,9])),dict(type='image')])]
            def peek(self):return self.docs[self.index] if self.index<len(self.docs) else None
            def advance(self):self.index+=1
            def prefetch_images(self,*args):pass
            def image(self,*args):return processed
            def state_dict(self):return dict(index=self.index)
        packer=Packer(Documents(),cfg,48,tokens)
        first,r1=packer.next();second,r2=packer.next()
        self.assertEqual([r['id'] for r in r1['documents']],['previous'])
        self.assertEqual(r2['documents'][0]['element'],0)
        self.assertTrue(r2['documents'][0]['completed'])
        start=int(second.image_indices[0,0])
        self.assertEqual(second.ids[0,1:4].tolist(),[7,8,9])
        self.assertTrue((second.mainline.cq_mask[0,0,48+second.mainline.query_image[0].nonzero().flatten()[0],1:4]==False).all())
        self.assertGreater(start,3)

    def test_condition_projection_cache_and_schedule(self):
        cfg=self.cfg();torch.manual_seed(7)
        model=JointModel(cfg).eval()
        from .generation import GenerationInputs
        layout=GenerationInputs.prepare([np.zeros((32,48,3),np.uint8)],cfg.image_condition_queries,32)
        condition=torch.randn(1,cfg.image_condition_queries,cfg.width)
        xt=torch.randn(sum(layout.lengths),cfg.latent_channels);time=torch.randn(1,256)
        with torch.no_grad():
            expected=model.renderer.predict(xt,condition,time,layout)
            projected=model.renderer.project_conditions(condition,layout)
            actual=model.renderer.predict(xt,None,time,layout,projected)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        grid=moss_sigmas(30)
        self.assertEqual(grid[0].item(),1.);self.assertEqual(grid[-1].item(),0.)
        self.assertTrue(torch.all(grid[:-1]>grid[1:]))

    def test_selective_attention_recompute_preserves_all_backbone_gradients(self):
        torch.manual_seed(153);cfg=self.cfg();cfg.checkpoint_backbone=True
        other=copy.deepcopy(cfg);other.keep_attention_activations=1
        baseline=JointModel(cfg).train();selected=JointModel(other).train()
        selected.load_state_dict(baseline.state_dict(),strict=True)
        processed=NativeImageProcessor(cfg,128)(np.random.default_rng(5).integers(0,256,(64,64,3),dtype=np.uint8))
        batch=make_batch(cfg,self.fixture.tokens,[13,14,15],processed,'I2T',prune=False)
        expected=backbone(baseline,batch);actual=backbone(selected,copy.deepcopy(batch))
        target=torch.randn_like(expected)
        (expected-target).square().mean().backward()
        (actual-target).square().mean().backward()
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        for (name,p),(other_name,q) in zip(baseline.named_parameters(),selected.named_parameters()):
            self.assertEqual(name,other_name)
            if p.grad is None:self.assertIsNone(q.grad,name)
            else:torch.testing.assert_close(q.grad,p.grad,atol=2e-6,rtol=2e-5,msg=name)


if __name__=='__main__':unittest.main()
