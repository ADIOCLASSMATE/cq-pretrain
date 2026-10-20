"""Local DCVLM coverage, lossless packing, exact restart and encoder contracts."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .config import tiny_config
from .input_pipeline.catalog import Catalog,permute
from .input_pipeline.prepare import make_fake
from .input_pipeline.stream import DocumentStream
from .input_pipeline.packing import Packer,ConsumedLoader,token_contract
from .input_pipeline.vae import DevicePreparer
from .input_pipeline.device import DeviceLoader

TOKENIZER=Path(os.environ.get('DCVLM_TOKENIZER','assets/tokenizer'))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)/'corpus'
        make_fake(cls.root,TOKENIZER,rows=33,shard_rows=11,edge_cases=True)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def stream(self,view='A',**kwargs):
        values=dict(rank=0,world=1,seed=9,workers=2,prefetch=3,image_size=32,
                    block_rows=7,tokenize_rows=3,allow_development=True)
        values.update(kwargs)
        return DocumentStream(self.root,view,**values)

    def loader(self,state=None,**kwargs):
        cfg=tiny_config(vocab_size=151936)
        tokens=token_contract(self.root,cfg)
        docs=self.stream(state=state['documents'] if state else None,**kwargs)
        return ConsumedLoader(Packer(docs,cfg,512,tokens,state,max_images=2,padding_multiple=32),3,state)

    def collect(self,stream):
        out=[]
        try:
            while (doc:=stream.peek()) is not None:
                out.append(copy.deepcopy(doc));stream.advance()
        finally:stream.close()
        return out

    def test_permutation_and_dp_exact_coverage(self):
        for size in (1,2,3,7,31,100,513):
            self.assertEqual(sorted(permute(i,size,42) for i in range(size)),list(range(size)))
        parts=[self.collect(self.stream(rank=r,world=4,workers=1+r%2)) for r in range(4)]
        ids=[d['id'] for part in parts for d in part]
        self.assertEqual(len(ids),33);self.assertEqual(len(set(ids)),33)

    def test_views_and_online_token_ids(self):
        a=self.collect(self.stream());b=self.collect(self.stream('B',workers=1,tokenize_rows=5))
        self.assertEqual([d['id'] for d in a],[d['id'] for d in b])
        for left,right in zip(a,b):
            l=[e['input_ids'].tolist() for e in left['elements'] if e['type']=='text']
            r=[e['input_ids'].tolist() for e in right['elements']]
            self.assertEqual(l,r)
        self.assertTrue(any(151644 in e['input_ids'] for d in a for e in d['elements'] if e['type']=='text'))

    def test_lossless_packing(self):
        documents=self.collect(self.stream())
        expected={d['id']:[int(t) for e in d['elements'] if e['type']=='text' for t in e['input_ids']] for d in documents}
        image_total=sum(e['type']=='image' for d in documents for e in d['elements'])
        actual={key:[] for key in expected};images=0
        loader=self.loader()
        try:
            while (pair:=loader.next()) is not None:
                batch,receipt=pair
                self.assertLessEqual(len(batch.image_indices),2)
                self.assertLessEqual(batch.ids.shape[1],512)
                self.assertEqual(batch.ids.shape[1]%32,0)
                for record in receipt['documents']:
                    for start,count in record['body_spans']:
                        actual[record['id']].extend(batch.ids[0,start:start+count].tolist())
                images+=receipt['images'];loader.acknowledge(receipt)
        finally:loader.close()
        self.assertEqual(actual,expected);self.assertEqual(images,image_total)

    def test_consumed_restart_with_different_workers(self):
        loader=self.loader()
        try:
            first=loader.next();second=loader.next()
            with self.assertRaises(ValueError):loader.acknowledge(second[1])
            loader.acknowledge(first[1]);state=loader.state_dict()
            expected=[second]+[loader.next() for _ in range(4)]
        finally:loader.close()
        other=self.loader(state,workers=1,tokenize_rows=5,prefetch=1)
        try:
            for pair in expected:
                replay=other.next()
                self.assertEqual(pair[1],replay[1]);self.assertTrue(torch.equal(pair[0].ids,replay[0].ids))
                self.assertTrue(torch.equal(pair[0].pixels,replay[0].pixels))
                other.acknowledge(replay[1])
        finally:other.close()

    def test_resume_rejects_changed_visual_resolution_or_pack_length(self):
        cfg=tiny_config(vocab_size=151936)
        stream=self.stream(cfg=cfg,sequence_length=512)
        try:
            stream.peek();stream.advance();state=stream.state_dict()
        finally:stream.close()
        changed=copy.deepcopy(cfg);changed.vision_max_pixels=262144
        with self.assertRaisesRegex(ValueError,'differs on resume'):
            self.stream(cfg=changed,sequence_length=512,state=state)
        with self.assertRaisesRegex(ValueError,'differs on resume'):
            self.stream(cfg=cfg,sequence_length=1024,state=state)

    def test_latent_prefetch_rng_and_contract(self):
        cfg=tiny_config(vocab_size=151936,latent_channels=8)
        before=torch.random.get_rng_state().clone();results=[]
        for depth in (0,2):
            host=self.loader();device=DeviceLoader(host,DevicePreparer(cfg,device='cpu',encoder='fake'),depth)
            try:
                group=device.next_group(3);result=[]
                for item in group:
                    batch=device.acquire(item)
                    result.append((batch.latents,batch.noise,batch.times))
                    device.acknowledge(item[1])
                self.assertLessEqual(device.stats['max_queued'],depth)
                results.append(result)
            finally:device.close()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        for a,b in zip(*results):
            for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))

    def test_fail_closed_local_integrity(self):
        stream=self.stream()
        try:
            while (doc:=stream.peek()) is not None:
                items=[e for e in doc['elements'] if e['type']=='image']
                if items:break
                stream.advance()
            locator=items[0]['locator']
            self.assertEqual(stream.image_reader.read(locator).shape,(32,32,3))
            with self.assertRaises(RuntimeError):stream.image_reader.read(dict(locator,sha256='0'*64))
            with self.assertRaises(ValueError):stream.image_reader.read(dict(locator,shard='https://example.invalid/a.tar'))
            with self.assertRaises(ValueError):stream.image_reader.read(dict(locator,offset=-1))
        finally:stream.close()


if __name__=='__main__':unittest.main()
