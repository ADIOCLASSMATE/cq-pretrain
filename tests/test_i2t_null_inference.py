"""CFG null CQ must match the image-first training branch on CPU."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import tests.test_qwen35_vision as fixtures
import data.chat as chat
from data.packing import Packer
from models.cq import backbone
from models.modeling_umm import JointModel
from evaluations.sampling import OneDocument, make_batch, make_null_batch
from data.vision import NativeImageProcessor


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):
        return SimpleNamespace(ids=[10+ord(c)%96 for c in text])


class I2TNullInferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NativeVisionContracts.setUpClass()
        cls.fixture=fixtures.NativeVisionContracts()
        cls.tokenizer=CharacterTokenizer()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        fixtures.NativeVisionContracts.tearDownClass()

    def cfg(self,**kwargs):
        return self.fixture.cfg(renderer='moss',latent_channels=128,flow_draws=1,
            dit_width=128,dit_heads=4,dit_ffn=256,dit_layers=2,**kwargs)

    def training_batch(self,cfg,processed,caption):
        row=dict(sample_id='null-training-check',format='caption',segments=[
            dict(kind='image',image_index=0),dict(kind='text',text=caption)])
        spans,direction,templated=chat.build(row)
        elements=[dict(type='image') if s['kind']=='image' else
            dict(type='text',input_ids=np.asarray(self.tokenizer.encode(s['text']).ids,np.int32),
                 supervised=s['supervised']) for s in spans]
        source=OneDocument(elements,processed)
        source.doc.update(format='caption',direction=direction,templated=templated)
        batch,receipt=Packer(source,cfg,256,self.fixture.tokens).next()
        self.assertTrue(receipt['documents'][0]['completed'])
        return batch

    def test_null_matches_training_i2t_and_excludes_caption_and_target_pixels(self):
        for preserve in (False,True):
            with self.subTest(preserve_documents=preserve):
                torch.manual_seed(427)
                cfg=self.cfg(preserve_documents=preserve,cfg_null_condition='i2t_prefix')
                processor=NativeImageProcessor(cfg,256)
                pixels=np.random.default_rng(7).integers(0,256,(64,96,3),dtype=np.uint8)
                actual_image=processor(pixels)
                black_image=processor(np.zeros_like(pixels))
                train=self.training_batch(cfg,actual_image,'A red bicycle beside the store.')
                other=self.training_batch(cfg,black_image,'Completely different caption.')
                null=make_null_batch(cfg,self.fixture.tokens,black_image,self.tokenizer)
                self.assertEqual(len(null.mainline.query_positions),cfg.image_condition_queries)
                self.assertTrue(null.mainline.query_image.all())
                model=JointModel(cfg).eval()
                with torch.no_grad():
                    expected=backbone(model,train)[train.mainline.condition_indices]
                    changed=backbone(model,other)[other.mainline.condition_indices]
                    empty=torch.zeros(len(null.vision.content_indices),cfg.width)
                    actual=backbone(model,null,vision_features=empty)[null.mainline.condition_indices]
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(changed,expected,atol=2e-6,rtol=2e-5)

    def test_default_null_keeps_old_empty_t2i_template(self):
        cfg=self.cfg()
        self.assertEqual(cfg.cfg_null_condition,'empty_t2i')
        processed=NativeImageProcessor(cfg,256)(np.zeros((64,96,3),np.uint8))
        expected=make_batch(cfg,self.fixture.tokens,[],processed,'T2I',tokenizer=self.tokenizer)
        actual=make_null_batch(cfg,self.fixture.tokens,processed,self.tokenizer)
        for field in ('ids','sigma','row_col','image_indices'):
            torch.testing.assert_close(getattr(actual,field),getattr(expected,field),rtol=0,atol=0)
        torch.testing.assert_close(actual.mainline.cq_mask,expected.mainline.cq_mask,rtol=0,atol=0)

    def test_i2t_null_uses_only_the_trained_user_prefix(self):
        cfg=self.cfg(cfg_null_condition='i2t_prefix')
        processed=NativeImageProcessor(cfg,256)(np.zeros((64,96,3),np.uint8))
        null=make_null_batch(cfg,self.fixture.tokens,processed,self.tokenizer)
        row=null.ids.shape[1]
        visible=null.ids[0][~null.mainline.cq_mask[0,0,row]].tolist()
        self.assertEqual(visible,[self.fixture.tokens['bos']]+
            self.tokenizer.encode(chat.ROLE_USER).ids+[self.fixture.tokens['boi'],self.fixture.tokens['eoi']])

    def test_unknown_null_mode_fails_instead_of_silently_changing_guidance(self):
        cfg=self.cfg(cfg_null_condition='unknown')
        with self.assertRaisesRegex(ValueError,'Unknown CFG null condition'):
            make_null_batch(cfg,self.fixture.tokens,None,self.tokenizer)


if __name__=='__main__':unittest.main()
