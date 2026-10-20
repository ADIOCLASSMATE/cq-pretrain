"""Semantic/gradient oracles for whole documents and independently chunked DiT."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from . import test_qwen35_vision as fixtures
from .generation import GenerationInputs
from .input_pipeline.packing import Packer,ConsumedLoader
from .model import JointModel
from .mainline import build_plan
from .renderer_schedule import backward_deferred_renderer,suspend_gradient_readiness


def text(ids):return dict(type='text',input_ids=np.asarray(ids,dtype=np.int32))


def picture(h=64,w=64,seed=0):
    return dict(type='image',pixels=np.random.default_rng(seed).integers(0,256,(h,w,3),dtype=np.uint8))


class Documents(fixtures.Documents):
    def __init__(self,rows,index=0):super().__init__(rows,index);self.image_reads=0
    def image(self,*args):self.image_reads+=1;return super().image(*args)
    def close(self):pass


class DocumentPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NativeVisionContracts.setUpClass();cls.fixture=fixtures.NativeVisionContracts()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):fixtures.NativeVisionContracts.tearDownClass()

    def cfg(self,**kwargs):
        values=dict(renderer='moss',latent_channels=128,flow_draws=1,dit_width=128,dit_heads=4,
            dit_ffn=256,dit_layers=2,preserve_documents=True,generation_token_budget=64,
            freeze_vision=True,freeze_vision_merger=False,tie_word_embeddings=False)
        values.update(kwargs)
        return self.fixture.cfg(**values)

    def packer(self,rows,cfg=None,length=128,state=None):
        docs=Documents(rows,state['documents']['index'] if state else 0)
        return Packer(docs,cfg or self.cfg(),length,self.fixture.tokens,state)

    def pair(self,cfg=None):
        rows=[dict(id='interleaved',format='interleaved',elements=[text([11,12]),picture(),
            text([13,14,15]),picture(32,64,1),text([16]),picture(64,96,2),text([17,18])])]
        batch,_=self.packer(rows,cfg).next()
        generator=torch.Generator().manual_seed(1703)
        n=sum(batch.generation.lengths)
        batch.latents=torch.randn(n,128,generator=generator)
        batch.noise=torch.randn(n,128,generator=generator)
        batch.times=torch.tensor([.2,.5,.8])
        batch.generation.time_features=torch.randn(3,256,generator=generator)
        return batch

    def test_document_moves_intact_and_decode_is_reused(self):
        rows=[dict(id='previous',format='text',elements=[text([9]*110)]),
              dict(id='whole',format='interleaved',elements=[text([11,12]),picture(),text([13]),
                  picture(32,64),text([14]),picture(64,96),text([15])])]
        packer=self.packer(rows)
        first,r1=packer.next();second,r2=packer.next()
        self.assertEqual([x['id'] for x in r1['documents']],['previous'])
        self.assertEqual([x['id'] for x in r2['documents']],['whole'])
        self.assertTrue(r2['documents'][0]['completed'])
        self.assertFalse(r2['documents'][0]['window_continuation'])
        self.assertIsNone(r2['documents'][0]['window_reason'])
        self.assertEqual(r2['images'],3)
        self.assertGreater(r2['generation_tokens'],64)
        self.assertEqual(r2['dit_microbatches'],2)
        self.assertEqual(packer.documents.image_reads,3)
        self.assertEqual([(p.start,p.stop) for p in second.generation.microbatches],[(0,2),(2,3)])
        replay=self.packer(rows,state=r1['next_state'])
        reproduced,r3=replay.next()
        self.assertEqual(r3,r2)
        torch.testing.assert_close(reproduced.ids,second.ids,rtol=0,atol=0)
        for a,b in zip(reproduced.generation.pixels,second.generation.pixels):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertIsNone(packer.next())

    def test_slicing_retains_global_weights_samples_and_zero_copy_large_views(self):
        batch=self.pair();parts=batch.generation.microbatches
        weights=[];targets=[];noise=[];times=[]
        for part in parts:
            chunk=part.bind(batch);p=chunk.generation
            weights.append(p.loss_weights);targets.append(chunk.latents);noise.append(chunk.noise);times.append(chunk.times)
            self.assertEqual(chunk.latents.untyped_storage().data_ptr(),batch.latents.untyped_storage().data_ptr())
            self.assertEqual(p.cos.untyped_storage().data_ptr(),batch.generation.cos.untyped_storage().data_ptr())
            self.assertLessEqual(p.ends[-1],64)
            for i,end in enumerate(p.ends):
                start=p.ends[i-1] if i else 0;n=p.lengths[i]
                self.assertEqual(p.video_indices[p.video_sample_indices==i].tolist(),list(range(start,start+n)))
                self.assertEqual(p.text_indices[i*16:(i+1)*16].tolist(),list(range(start+n,end)))
        for chunks,expected in ((weights,batch.generation.loss_weights),(targets,batch.latents),
                                (noise,batch.noise),(times,batch.times)):
            torch.testing.assert_close(torch.cat(chunks),expected,rtol=0,atol=0)
        torch.testing.assert_close(torch.cat(weights).sum(),torch.tensor(1.))
        moved=batch.generation.to('cpu')
        self.assertEqual(len(moved.microbatches),len(parts))

    def test_indivisible_pair_and_single_chunk_fast_path(self):
        image=np.zeros((64,64,3),dtype=np.uint8)
        with self.assertRaisesRegex(ValueError,'above DiT budget'):
            GenerationInputs.prepare([image],16,32,token_budget=31)
        layout=GenerationInputs.prepare([image],16,32,token_budget=32)
        self.assertEqual(layout.microbatches,())
        with self.assertRaisesRegex(ValueError,'max_images_per_pack'):
            Packer(Documents([]),self.cfg(),128,self.fixture.tokens,max_images=1)

    def test_document_mask_is_bitwise_equal_to_full_comparisons(self):
        for method in ('CQ_IT','Bi_I+CQ_T'):
            cfg=self.cfg(method=method)
            rows=[dict(id='a',elements=[text([]),text([11]),picture(),text([12])]),
                  dict(id='b',elements=[text([21,22]),picture(32,64),text([23])])]
            batch,receipt=self.packer(rows,cfg).next()
            self.assertFalse(receipt['documents'][0]['window_continuation'])
            full_cfg=copy.deepcopy(cfg);full_cfg.preserve_documents=False
            expected=build_plan(batch,full_cfg)
            torch.testing.assert_close(batch.mainline.cq_mask,expected.cq_mask,rtol=0,atol=0)
            torch.testing.assert_close(batch.mainline.cq_valid,expected.cq_valid,rtol=0,atol=0)

    def test_explicit_backbone_windows_keep_coverage_and_consumed_restart(self):
        rows=[dict(id='long',format='interleaved',elements=[text([11]*270),picture(),
              text([12]*40),picture(32,64),text([13]*17)])]
        with self.assertRaisesRegex(ValueError,'exceeds'):
            self.packer(rows,self.cfg(long_document_policy='error')).next()
        loader=ConsumedLoader(self.packer(rows),prefetch=3)
        try:
            first=loader.next();loader.acknowledge(first[1]);saved=loader.state_dict()
            remaining=[]
            while (pair:=loader.next()) is not None:remaining.append(pair);loader.acknowledge(pair[1])
        finally:loader.close()
        all_pairs=[first]+remaining
        actual=[]
        for batch,receipt in all_pairs:
            for record in receipt['documents']:
                for start,n in record['body_spans']:actual.extend(batch.ids[0,start:start+n].tolist())
                self.assertEqual(record['window_reason'],'backbone_length')
        self.assertEqual(actual,[11]*270+[12]*40+[13]*17)
        self.assertEqual(sum(r['images'] for _,r in all_pairs),2)
        replay=self.packer(rows,state=saved)
        for expected,receipt in remaining:
            batch,r=replay.next();self.assertEqual(r,receipt)
            torch.testing.assert_close(batch.ids,expected.ids,rtol=0,atol=0)

    def test_chunked_joint_loss_every_gradient_and_one_backbone_backward(self):
        for recompute in (False,True):
            with self.subTest(recompute=recompute):
                cfg=self.cfg(checkpoint_dit=recompute,checkpoint_backbone=recompute)
                torch.manual_seed(482)
                expected_model=JointModel(cfg).train();actual_model=copy.deepcopy(expected_model)
                batch=self.pair(cfg);reference_cq=[]
                def capture(module,args):
                    args[0].retain_grad();reference_cq.append(args[0])
                handle=expected_model.renderer.register_forward_pre_hook(capture)
                expected=expected_model(batch,loss_scales=(.7,.3));handle.remove()
                (expected['loss']*.4).backward()
                actual=actual_model(batch,loss_scales=(.7,.3),return_hidden=True,defer_renderer=True)
                cq=actual['_image_conditions'];cq.retain_grad();backward_visits=[]
                actual['hidden'].register_hook(lambda x:backward_visits.append(True))
                backward_deferred_renderer(actual_model,batch,actual,.4)
                self.assertEqual(backward_visits,[True])
                for name in ('loss','text_loss','image_loss'):
                    torch.testing.assert_close(actual[name],expected[name],atol=3e-6,rtol=2e-5,msg=name)
                torch.testing.assert_close(cq.grad,reference_cq[0].grad,atol=3e-6,rtol=3e-4)
                self.assertTrue(any('merger' in n and p.grad is not None for n,p in actual_model.named_parameters()))
                for (name,p),(other,q) in zip(expected_model.named_parameters(),actual_model.named_parameters()):
                    self.assertEqual(name,other)
                    if p.grad is None:self.assertIsNone(q.grad,name)
                    else:torch.testing.assert_close(q.grad,p.grad,atol=4e-6,rtol=4e-4,msg=name)

    def test_packed_documents_match_independent_and_cannot_influence_other(self):
        cfg=self.cfg();torch.manual_seed(3);model=JointModel(cfg).eval()
        rows=[dict(id='a',format='interleaved',elements=[text([11,12]),picture(),text([13])]),
              dict(id='b',format='interleaved',elements=[text([21,22,23]),picture(32,64,2),text([24,25])])]
        packed,receipt=self.packer(rows,cfg).next()
        with torch.no_grad():full=model.backbone(packed)
        for i,row in enumerate(rows):
            single,_=self.packer([row],cfg).next()
            with torch.no_grad():alone=model.backbone(single)
            positions=(packed.segments[0]==i).nonzero().flatten()
            q=single.mainline.query_positions[~single.mainline.query_image[0]]
            torch.testing.assert_close(full[positions[q]],alone[q],atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(full[packed.mainline.condition_indices[i]],
                alone[single.mainline.condition_indices[0]],atol=2e-6,rtol=2e-5)
        altered=copy.deepcopy(packed)
        altered.ids[0,1:3]=torch.tensor([31,32])
        altered.vision.pixel_values[:packed.vision.lengths[0]].add_(7)
        with torch.no_grad():after=model.backbone(altered)
        keep=(packed.segments[0]==1)&~packed.image_queries[0]
        torch.testing.assert_close(after[:packed.ids.shape[1]][keep],full[:packed.ids.shape[1]][keep],atol=0,rtol=0)
        torch.testing.assert_close(after[packed.mainline.condition_indices[1]],full[packed.mainline.condition_indices[1]],atol=0,rtol=0)

    def test_mcore_post_hooks_accumulate_all_chunks_but_ready_only_once(self):
        # Execute the installed MCore hook implementation, without initializing
        # its NPU runtime. This tests its real counting/FP32-main-grad contract.
        deps=Path(__file__).resolve().parents[1]/'ascend_pretrain/.deps/core17/Megatron-LM/megatron/core/distributed'
        if not deps.is_dir():self.skipTest('Pinned MCore source unavailable')
        def method(filename,name):
            tree=ast.parse((deps/filename).read_text())
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
            node.decorator_list=[]
            scope={'torch':torch,'Optional':__import__('typing').Optional,'is_graph_capturing':lambda:False}
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(deps/filename),'exec'),scope)
            return scope[name]
        register=method('param_and_grad_buffer.py','register_grad_ready')
        hook_factory=method('distributed_data_parallel.py','_make_backward_post_hook')
        cfg=self.cfg();torch.manual_seed(44)
        reference=JointModel(cfg);actual=copy.deepcopy(reference)
        params=[p for p in actual.parameters() if p.requires_grad]
        completed=[]
        group=SimpleNamespace(is_last_microbatch=True,is_first_batch=False,params=params,
            param_to_bucket={p:None for p in params},ddp_config=SimpleNamespace(overlap_grad_reduce=True),
            per_param_grad_ready_counts={},golden_per_param_grad_ready_counts={p:1 for p in params},
            start_grad_sync=lambda **kw:completed.append(True))
        group.register_grad_ready=lambda p,force:register(group,p,force)
        wrapped=SimpleNamespace(module=actual,bucket_groups=[group],expert_parallel_bucket_groups=[],
            param_to_bucket_group={p:group for p in params},force_all_reduce=False,ddp_config=group.ddp_config)
        handles=[]
        for p in params:
            p.main_grad=torch.zeros_like(p,dtype=torch.float32);p.grad_added_to_main_grad=False
            handles.append(p.register_post_accumulate_grad_hook(hook_factory(wrapped,p)))
        try:
            for micro in range(2):
                group.is_last_microbatch=bool(micro)
                batch=self.pair(cfg)
                if micro:batch.generation.plan_microbatches(40)
                (reference(batch)['loss']*.5).backward()
                losses=actual(batch,defer_renderer=True)
                backward_deferred_renderer(actual,batch,losses,.5,wrapped)
                self.assertEqual(group.is_last_microbatch,bool(micro))
                if micro==0:
                    self.assertEqual(completed,[]);self.assertEqual(group.per_param_grad_ready_counts,{})
            self.assertEqual(completed,[True])
            self.assertEqual(group.per_param_grad_ready_counts,group.golden_per_param_grad_ready_counts)
            for (name,p),(_,q) in zip(reference.named_parameters(),actual.named_parameters()):
                if p.requires_grad:torch.testing.assert_close(q.main_grad,p.grad,atol=4e-6,rtol=4e-4,msg=name)
        finally:
            for h in handles:h.remove()

    def test_readiness_restored_after_exception_with_nested_ga(self):
        groups=[SimpleNamespace(is_last_microbatch=x) for x in (False,True)]
        wrapped=SimpleNamespace(bucket_groups=groups,expert_parallel_bucket_groups=[])
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with suspend_gradient_readiness(wrapped):
                self.assertEqual([g.is_last_microbatch for g in groups],[False,False])
                raise RuntimeError('sentinel')
        self.assertEqual([g.is_last_microbatch for g in groups],[False,True])

    def test_resume_rejects_changed_document_semantics(self):
        rows=[dict(id='a',elements=[text([11,12])])]
        packer=self.packer(rows);state=packer.state_dict()
        for altered in (dict(state,packing=dict(state['packing'],long_document_policy='error')),
                        {k:v for k,v in state.items() if k!='packing'}):
            with self.assertRaisesRegex(ValueError,'differ on resume'):
                self.packer(rows,state=altered)


if __name__=='__main__':unittest.main()
