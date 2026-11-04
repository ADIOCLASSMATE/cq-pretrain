"""Image-target permissions survive loading and never remove understanding."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .config import tiny_config
from .input_pipeline.loss_permissions import image_loss_allowed
from .input_pipeline.packing import Packer
from .input_pipeline.prepare import make_fake
from .input_pipeline.stream import DocumentStream
from .input_pipeline.vae import DevicePreparer
from .model import JointModel
from .test_i2t_image_sampling import Documents, row
from . import test_qwen35_vision as fixtures


class PermissionLoadingTests(unittest.TestCase):
    def test_strict_and_never_short_circuits_invalid_fields(self):
        self.assertTrue(image_loss_allowed({}, {}))
        self.assertFalse(image_loss_allowed({'image_loss_allowed':False}, {'image_loss_allowed':True}))
        self.assertFalse(image_loss_allowed({'image_loss_allowed':True,'dcvlm_image_loss_allowed':False}))
        for invalid in (None, 0, 1, 'false', [], {}):
            for name in ('image_loss_allowed','dcvlm_image_loss_allowed'):
                with self.subTest(value=invalid,name=name),self.assertRaisesRegex(ValueError,'boolean'):
                    image_loss_allowed({'image_loss_allowed':False}, {name:invalid})

    def test_parquet_flags_reach_documents_and_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'corpus'
            make_fake(root,Path('assets/tokenizer'),rows=3,shard_rows=3)
            path=root/'data/00000.parquet'
            rows=pq.read_table(path).to_pylist()
            for index,entry in enumerate(rows):
                entry['image_loss_allowed']=index!=0
                entry['dcvlm_image_loss_allowed']=index!=1
                for info in entry['images']:info['image_loss_allowed']=index!=2
            pq.write_table(pa.Table.from_pylist(rows),path)
            catalog=json.loads((root/'catalog.json').read_text())
            catalog[0]['bytes']=path.stat().st_size
            (root/'catalog.json').write_text(json.dumps(catalog))
            done=json.loads((root/'COMPLETE.json').read_text())
            done['catalog_sha256']=hashlib.sha256((root/'catalog.json').read_bytes()).hexdigest()
            (root/'COMPLETE.json').write_text(json.dumps(done))
            for view in ('A','B'):
                stream=DocumentStream(root,view,0,1,42,workers=1,prefetch=1,
                                      block_rows=3,tokenize_rows=3,allow_development=True)
                try:
                    docs=[]
                    while (doc:=stream.peek()) is not None:
                        docs.append(doc);stream.advance()
                    self.assertEqual([x['image_loss_allowed'] for x in docs],[False,False,True])
                    self.assertEqual(len(docs),3)
                    if view=='A':
                        self.assertEqual([e['image_loss_allowed'] for d in docs
                                          for e in d['elements'] if e['type']=='image'],[False]*3)
                finally:stream.close()


class PermissionPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NativeVisionContracts.setUpClass();cls.fixture=fixtures.NativeVisionContracts()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):fixtures.NativeVisionContracts.tearDownClass()

    def cfg(self,**kwargs):
        values=dict(renderer='moss',latent_channels=128,flow_draws=1,dit_width=128,dit_heads=4,
                    dit_ffn=256,dit_layers=2,preserve_documents=True,generation_token_budget=64,
                    i2t_image_loss_probability=1,cfg_null_condition='i2t_prefix')
        values.update(kwargs)
        return self.fixture.cfg(**values)

    def packer(self,rows,cfg,state=None,length=256):
        docs=Documents(rows,state['documents']['index'] if state else 0)
        return Packer(docs,cfg,length,self.fixture.tokens,state)

    def test_mixed_image_flags_override_probability_one_in_every_format(self):
        cfg=self.cfg()
        for fmt in ('caption','conversation','interleaved'):
            with self.subTest(format=fmt):
                rows=[row('record-denied'),row('legacy-denied'),row('image-denied'),row('allowed')]
                rows[0]['image_loss_allowed']=False
                rows[0]['elements'][1]['image_loss_allowed']=True
                rows[1]['dcvlm_image_loss_allowed']=False
                rows[2]['elements'][1]['image_loss_allowed']=False
                for item in rows:item['format']=fmt
                batch,receipt=self.packer(rows,cfg).next()
                self.assertEqual(batch.generation_image_indices.tolist(),[3])
                self.assertEqual(receipt['images'],4)
                control=copy.deepcopy(rows)
                for item in control:
                    item.pop('image_loss_allowed',None);item.pop('dcvlm_image_loss_allowed',None)
                    for element in item['elements']:element.pop('image_loss_allowed',None)
                full,_=self.packer(control,cfg).next()
                for name in ('ids','text_indices','image_indices','sigma','row_col'):
                    torch.testing.assert_close(getattr(batch,name),getattr(full,name),rtol=0,atol=0)
                torch.testing.assert_close(batch.vision.pixel_values,full.vision.pixel_values,rtol=0,atol=0)

    def test_per_image_permissions_inside_one_document_and_restart(self):
        entry=row('multi');entry['format']='interleaved'
        entry['elements']=sum((row('image-'+str(i))['elements'] for i in range(5)),[])
        for index,element in enumerate(e for e in entry['elements'] if e['type']=='image'):
            element['image_loss_allowed']=index%2==1
        cfg=self.cfg();packer=self.packer([entry],cfg,length=24)
        _,first=packer.next();expected,receipt=packer.next()
        resumed=self.packer([entry],cfg,first['next_state'],length=24)
        actual,replayed=resumed.next()
        self.assertEqual(receipt,replayed)
        torch.testing.assert_close(actual.generation_image_indices,expected.generation_image_indices)
        old_state=copy.deepcopy(first['next_state'])
        old_state['packing'].pop('image_loss_permissions')
        with self.assertRaisesRegex(ValueError,'packing policy'):
            self.packer([entry],cfg,old_state,length=24)
        batch,_=self.packer([entry],cfg).next()
        self.assertEqual(batch.generation_image_indices.tolist(),[1,3])

    def test_denied_moss_image_skips_generation_and_keeps_understanding_gradient(self):
        cfg=self.cfg();entry=row('understand');entry['image_loss_allowed']=False
        batch,receipt=self.packer([entry],cfg).next()
        preparer=DevicePreparer(cfg,device='cpu',encoder='fake')
        with patch.object(preparer.vae,'encode',side_effect=AssertionError('VAE must be skipped')):
            ready,_=preparer.prepare(batch,receipt)
        model=JointModel(cfg)
        with patch.object(model.renderer,'forward',side_effect=AssertionError('DiT must be skipped')):
            losses=model(ready)
        self.assertEqual(losses['image_loss'].item(),0)
        self.assertGreater(losses['text_loss'].item(),0)
        losses['loss'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.patch_embed.parameters()))
        self.assertTrue(all(p.grad is None for p in model.renderer.parameters()))

    def test_invalid_flags_are_rejected_even_under_record_denial(self):
        cfg=self.cfg();entry=row('invalid');entry['image_loss_allowed']=False
        entry['elements'][1]['image_loss_allowed']='false'
        with self.assertRaisesRegex(ValueError,'boolean'):self.packer([entry],cfg).next()

    def test_legacy_renderer_encodes_only_allowed_targets_and_preserves_all_pixels(self):
        for vision in ('rgb_patch','qwen35'):
            for allowed in (False,True):
                with self.subTest(vision=vision,one_target=allowed):
                    cfg=(tiny_config(fused=False) if vision=='rgb_patch' else
                         self.fixture.cfg(renderer='legacy',preserve_documents=False))
                    rows=[row('denied',height=32,width=32),row('other',height=32,width=32)]
                    rows[0]['image_loss_allowed']=False;rows[1]['image_loss_allowed']=allowed
                    batch,receipt=self.packer(rows,cfg).next()
                    self.assertEqual(batch.generation_image_indices.tolist(),[1] if allowed else [])
                    raw=batch.pixels.clone()
                    preparer=DevicePreparer(cfg,device='cpu',encoder='fake');encoded=[]
                    encode=preparer.vae.encode
                    with patch.object(preparer.vae,'encode',side_effect=lambda x:(encoded.append(x.clone()),encode(x))[1]):
                        ready,_=preparer.prepare(batch,receipt)
                    self.assertEqual(sum(len(x) for x in encoded),int(allowed))
                    self.assertEqual(len(ready.latents),int(allowed))
                    expected=raw.float().div(127.5).sub(1)
                    torch.testing.assert_close(ready.pixels,expected.to(torch.bfloat16),rtol=0,atol=0)
                    if allowed:torch.testing.assert_close(encoded[0],expected[1:],rtol=0,atol=0)
                    # CPU model uses FP32 parameters; production runs autocast.
                    ready.pixels=ready.pixels.float()
                    model=JointModel(cfg)
                    if allowed:
                        losses=model(ready)
                        self.assertGreater(losses['image_loss'].item(),0)
                    else:
                        with patch.object(model.renderer,'forward',side_effect=AssertionError('DiT must be skipped')):
                            losses=model(ready)
                        self.assertEqual(losses['image_loss'].item(),0)
                    losses['loss'].backward()
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.patch_embed.parameters()))
                    if not allowed:self.assertTrue(all(p.grad is None for p in model.renderer.parameters()))


if __name__=='__main__':unittest.main()
