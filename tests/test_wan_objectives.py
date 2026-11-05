"""Complete native tasks, empty CQ reads and ragged generation supervision."""
from training.loss import compose_losses
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.codec import DevicePreparer
from data.packing import Packer, sample_objective
from data.stream import DocumentStream
from evaluations.sampling import make_batch, make_null_batch
from models.cq import backbone
from data.planning import build_plan
from models.modeling_umm import JointModel
from data.vision import NativeImageProcessor
from tests import test_qwen35_vision as fixtures


@pytest.fixture
def setup():
    fixtures.NativeVisionContracts.setUpClass()
    fixture=fixtures.NativeVisionContracts()
    cfg=fixture.cfg(renderer='wan',latent_channels=16,flow_draws=1,dit_width=64,
        dit_heads=2,dit_ffn=128,dit_layers=2,wan_freq_dim=32,
        visual_query_mode='native',cfg_null_condition='empty',preserve_documents=True)
    yield cfg,fixture.tokens
    fixtures.NativeVisionContracts.tearDownClass()


def text(*ids):return dict(type='text',input_ids=np.array(ids,dtype=np.int32))


def picture(h=64,w=96,**flags):
    return dict(type='image',pixels=np.zeros((h,w,3),np.uint8),**flags)


class Documents(fixtures.Documents):
    signature=dict(seed=42,epoch=3,view='A',rank=0,snapshot='wan-objectives')
    def __init__(self,rows,index=0):super().__init__(rows,index);self.read=[]
    def state_dict(self):return dict(index=self.index,signature=self.signature.copy())
    def image(self,doc,element):
        self.read.append(doc['id'])
        return super().image(doc,element)


def test_native_roles_survive_stream_and_input_images_never_become_targets():
    # Run the real online adapter with a trivial tokenizer and in-memory catalog.
    def segment(role,turn,kind):
        return dict(role=role,turn=turn,kind=kind,text='body',image_index=0)
    rows=[dict(sample_id='understand',format='conversation',segments=[segment('user',0,'image'),segment('assistant',1,'text')]),
          dict(sample_id='generate',format='conversation',segments=[segment('user',0,'image'),segment('user',0,'text'),segment('assistant',1,'image')]),
          dict(sample_id='interleaved',format='interleaved',segments=[segment(None,None,'text'),segment(None,None,'image')])]
    for row in rows:row.update(images=[dict(path='test.png')],source_shard='',text_tokens=1)
    rows[0]['non_cfg']=True;rows[1]['non_cfg']=False;rows[2]['images'][0]['non_cfg']=True
    stream=object.__new__(DocumentStream)
    stream.catalog=SimpleNamespace(read_rows=lambda *args:rows)
    stream.tokenizer=SimpleNamespace(encode_batch=lambda spans,**kw:[SimpleNamespace(ids=[11]) for _ in spans])
    stream.special_ids=set();stream.counting_matches_corpus=False;stream.signature={'view':'A'}
    documents=stream._load(0,0,3)['rows']
    assert [d['direction'] for d in documents]==['I2T','T2I',None]
    assert [[e['generation_target'] for e in d['elements'] if e['type']=='image']
            for d in documents]==[[False],[False,True],[True]]
    assert all(e['supervised'] for e in documents[-1]['elements'] if e['type']=='text')
    assert [d['non_cfg'] for d in documents]==[True,False,None]
    assert [[e['non_cfg'] for e in d['elements'] if e['type']=='image']
            for d in documents]==[[None],[None,None],[True]]


def test_native_single_and_multi_turn_dialogues_are_all_retained(setup):
    cfg,tokens=setup;cfg=replace(cfg,visual_uncond_probability=.1)
    rows=[dict(id=str(i),format='conversation',direction='T2I',elements=
        [text(11),picture()] if i%2 else [text(11),picture(),text(12),picture()]) for i in range(20)]
    rows += [dict(id='i2t',format='conversation',direction='I2T',elements=[picture(),text(13)]),
             dict(id='natural',format='interleaved',elements=[text(14),picture(),text(15)])]
    docs=Documents(rows);packer=Packer(docs,cfg,64,tokens);seen=[]
    while (item:=packer.next()) is not None:
        batch,receipt=item;seen.extend(d['id'] for d in receipt['documents'])
        assert all(d['completed'] for d in receipt['documents'])
        assert receipt['condition_query_tokens']==(sum(batch.generation.query_counts) if batch.generation else 0)
    assert seen==[str(i) for i in range(20)]+['i2t','natural']
    assert set(docs.read)==set(seen)
    selected=[i for i in range(1000) if sample_objective(f'{i}:1',.1,42,3,'visual-uncond-v1')]
    assert 80<len(selected)<120
    assert selected!=[i for i in range(1000) if sample_objective(f'{i}:1',.1,42,4,'visual-uncond-v1')]


@pytest.mark.parametrize('leading_text',[False,True])
def test_interleaved_uses_real_prefix_without_random_cfg_dropout(setup,leading_text):
    cfg,tokens=setup;cfg=replace(cfg,visual_uncond_probability=1.)
    elements=([text(21,22)] if leading_text else [])+[picture(),text(11,12),picture(),text(13)]
    rows=[dict(id='natural',format='interleaved',templated=False,elements=elements)]
    batch,receipt=Packer(Documents(rows),cfg,64,tokens).next()
    assert batch.generation_image_indices.tolist()==[0,1]
    assert batch.generation_unconditional.tolist()==[not leading_text,False]
    assert receipt['unconditional_images']==int(not leading_text)
    torch.manual_seed(23);model=JointModel(cfg).eval()
    features=torch.randn(len(batch.vision.content_indices),cfg.width)
    indices=batch.mainline.condition_indices
    with torch.no_grad():
        expected=backbone(model,batch,vision_features=features)[indices]
        changed=features.clone();changed[:batch.generation.query_counts[0]]+=3
        actual=backbone(model,batch,vision_features=changed)[indices]
        torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
        assert not torch.allclose(actual[1],expected[1])  # I2 reads the real preceding I1.
        modified=copy.deepcopy(batch);modified.ids[modified.ids==11]=27
        actual=backbone(model,modified,vision_features=features)[indices]
        torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
        assert not torch.allclose(actual[1],expected[1])  # I2 also reads the preceding text.


def test_dynamic_targets_permissions_and_text_image_losses(setup):
    cfg,tokens=setup
    rows=[dict(id='i2t',direction='I2T',elements=[picture(128,128),text(11)]),
          dict(id='natural',format='interleaved',elements=[text(12),picture(),text(13),
              picture(96,96),text(14),picture(image_loss_allowed=False),text(15)])]
    batch,receipt=Packer(Documents(rows),cfg,128,tokens).next()
    assert batch.generation_image_indices.tolist()==[1,2]
    assert batch.generation.query_counts==(6,9)
    assert batch.mainline.condition_indices.shape==(4,9)
    assert batch.mainline.condition_indices[[0,3]].eq(-1).all()
    assert receipt['condition_query_tokens']==15
    ready,_=DevicePreparer(cfg,device='cpu',encoder='fake').prepare(batch,receipt)
    model=JointModel(cfg);losses=compose_losses(model(ready), model.cfg)
    assert losses['text_loss']>0 and losses['image_loss']>0
    losses['loss'].backward()
    assert model.embedding.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.renderer.parameters())


def test_i2t_skips_visual_queries_vae_and_renderer(setup,monkeypatch):
    cfg,tokens=setup
    batch,receipt=Packer(Documents([dict(id='i2t',direction='I2T',non_cfg=True,
        elements=[picture(non_cfg=True),text(11)])]),cfg,32,tokens).next()
    assert batch.generation_image_count==0 and not batch.mainline.query_image.any()
    assert batch.generation is None
    preparer=DevicePreparer(cfg,device='cpu',encoder='fake')
    def forbidden(*a,**kw):raise AssertionError('Understanding must skip image generation')
    monkeypatch.setattr(preparer.vae,'encode',forbidden)
    ready,_=preparer.prepare(batch,receipt)
    model=JointModel(cfg);monkeypatch.setattr(model.renderer,'forward',forbidden)
    losses=compose_losses(model(ready), model.cfg);losses['loss'].backward()
    assert losses['image_loss']==0 and losses['text_loss']>0
    assert all(p.grad is None for p in model.renderer.parameters())


@pytest.mark.parametrize('fmt',['conversation','interleaved'])
@pytest.mark.parametrize('mode',['fixed','native'])
def test_explicit_non_cfg_is_independent_of_format_and_target_position(setup,fmt,mode):
    cfg,tokens=setup;cfg=replace(cfg,visual_uncond_probability=1.,visual_query_mode=mode)
    rows=[dict(id='mixed',format=fmt,direction='T2I',non_cfg=True,elements=[
        text(11),picture(non_cfg=False),text(12),picture(96,96),text(13),picture(non_cfg=False),text(14)])]
    batch,receipt=Packer(Documents(rows),cfg,64,tokens).next()
    assert batch.generation_unconditional.tolist()==[False,True,False]
    assert batch.generation_image_count==3 and receipt['unconditional_images']==1
    plan=batch.mainline;indices=[row[row.ge(0)] for row in plan.condition_indices]
    torch.manual_seed(29);model=JointModel(cfg).eval()
    features=torch.randn(len(batch.vision.content_indices),cfg.width,requires_grad=True)
    expected=backbone(model,batch,vision_features=features)
    expected[indices[1]].square().sum().backward()
    assert features.grad.count_nonzero()==0
    assert model.embedding.weight.grad[[11,12,13,14]].count_nonzero()==0
    assert model.embedding.weight.grad[cfg.image_mask_token_id].abs().sum()>0
    changed=copy.deepcopy(batch);changed.ids[changed.ids==11]=27
    with torch.no_grad():
        actual=backbone(model,changed,vision_features=features.detach()+3)
        torch.testing.assert_close(actual[indices[1]],expected[indices[1]],atol=0,rtol=0)
        assert not torch.allclose(actual[indices[0]],expected[indices[0]])
        assert not torch.allclose(actual[indices[2]],expected[indices[2]])
        # An unspecified model flag means ordinary reads, regardless of the
        # training dropout probability. Only visibility changes, never geometry.
        ordinary=copy.deepcopy(batch);ordinary.generation_unconditional=None
        ordinary.mainline=build_plan(ordinary,cfg)
        for name in ('query_positions','query_rope','condition_indices'):
            torch.testing.assert_close(getattr(ordinary.mainline,name),getattr(plan,name),atol=0,rtol=0)
        normal=backbone(model,ordinary,vision_features=features.detach())
        assert not torch.allclose(normal[indices[1]],expected[indices[1]])
        ordinary.generation_unconditional=torch.zeros(3,dtype=torch.bool)
        ordinary.mainline=build_plan(ordinary,cfg)
        torch.testing.assert_close(backbone(model,ordinary,vision_features=features.detach()),normal,atol=0,rtol=0)


def test_empty_condition_matches_inference_without_content_or_gradient_leak(setup):
    cfg,tokens=setup;cfg=replace(cfg,visual_uncond_probability=1.)
    rows=[dict(id='previous',elements=[text(21,22,23)]),
          dict(id='t2i',format='conversation',direction='T2I',elements=[text(11,12),picture(),text(13)])]
    batch,receipt=Packer(Documents(rows),cfg,64,tokens).next()
    assert receipt['unconditional_images']==1
    plan=batch.mainline;s=batch.ids.shape[1]
    assert not plan.cq_valid[0,s:][plan.query_image[0]].any()
    delta=plan.delta_plan
    image_hidden=s+plan.query_image[0].nonzero().flatten()
    null_history=delta.query_history[torch.isin(delta.query_source,image_hidden)]
    assert null_history.eq(len(delta.content_order)).all()
    torch.manual_seed(9);model=JointModel(cfg).eval()
    features=torch.randn(len(batch.vision.content_indices),cfg.width,requires_grad=True)
    output=backbone(model,batch,vision_features=features)
    condition=output[plan.condition_indices]
    condition.square().sum().backward()
    assert features.grad.count_nonzero()==0
    changed=copy.deepcopy(batch);changed.ids[changed.ids==11]=27
    with torch.no_grad():
        actual=backbone(model,changed,vision_features=features.detach()+7)
        torch.testing.assert_close(actual[plan.condition_indices],condition,atol=0,rtol=0)
        assert not torch.allclose(actual[batch.text_indices],output[batch.text_indices])
        conditional=copy.deepcopy(batch);conditional.generation_unconditional.zero_()
        conditional.mainline=build_plan(conditional,cfg)
        normal=backbone(model,conditional,vision_features=features.detach())
        torch.testing.assert_close(normal[batch.text_indices],output[batch.text_indices],atol=1e-6,rtol=1e-6)
        processed=NativeImageProcessor(cfg,128)(rows[1]['elements'][1]['pixels'])
        null=make_null_batch(cfg,tokens,processed,tokenizer=None)
        inferred=backbone(model,null,vision_features=features.detach()*0)
        torch.testing.assert_close(inferred[null.mainline.condition_indices],condition,atol=1e-6,rtol=1e-6)
        explicit=make_batch(cfg,tokens,[31,32,33],processed,'T2I',non_cfg=True)
        assert explicit.mainline.query_unconditional.all()  # Flag survives inference pruning.
        actual=backbone(model,explicit,vision_features=features.detach()+4)
        torch.testing.assert_close(actual[explicit.mainline.condition_indices],condition,atol=1e-6,rtol=1e-6)
        positive=make_batch(cfg,tokens,[11,12],processed,'T2I')
        assert not positive.generation_unconditional.any()  # No random CFG dropout in validation.


def test_unconditional_choices_resume_exactly(setup):
    cfg,tokens=setup;cfg=replace(cfg,visual_uncond_probability=.5)
    rows=[dict(id=str(i),format='conversation',direction='T2I',elements=[text(11),picture(),text(12)]) for i in range(40)]
    packer=Packer(Documents(rows),cfg,24,tokens)
    _,first=packer.next();expected,receipt=packer.next()
    state=first['next_state'];docs=Documents(rows,state['documents']['index'])
    actual,replayed=Packer(docs,cfg,24,tokens,state).next()
    assert receipt==replayed
    torch.testing.assert_close(actual.mainline.cq_mask,expected.mainline.cq_mask)
    torch.testing.assert_close(actual.generation_unconditional,expected.generation_unconditional)
    with pytest.raises(ValueError,match='packing policy'):
        Packer(docs,replace(cfg,visual_uncond_probability=0.),24,tokens,state)
    # A fresh stage may reuse fixed-Q weights without restoring its old data cursor.
    from utils.checkpoint import initialization_model_config
    previous=replace(cfg,visual_query_mode='fixed',cfg_null_condition='empty_t2i').to_dict()
    previous.pop('visual_uncond_probability')
    assert initialization_model_config(previous)==initialization_model_config(cfg.to_dict())


def test_model_accepts_config_without_data_or_loss_policy(setup):
    from dataclasses import fields
    from configs.config import ModelConfig
    cfg,tokens=setup
    batch,receipt=Packer(Documents([dict(id='target',direction='T2I',
        elements=[text(11),picture()])]),cfg,32,tokens).next()
    ready,_=DevicePreparer(cfg,device='cpu',encoder='fake').prepare(batch,receipt)
    model_cfg=ModelConfig(**{f.name:getattr(cfg,f.name) for f in fields(ModelConfig)})
    assert not hasattr(model_cfg,'text_weight')
    assert not hasattr(model_cfg,'visual_uncond_probability')
    model=JointModel(model_cfg)
    outputs=model(ready)
    assert set(outputs)=={'text_loss','image_loss'}
    assert outputs['text_loss']>0 and outputs['image_loss']>0
    losses=compose_losses(outputs,cfg,(.7,.3))
    losses['loss'].backward()
    assert model.renderer.text_embedding[0].weight.grad.abs().sum()>0
    assert model.renderer.text_embedding[2].weight.grad.abs().sum()>0
