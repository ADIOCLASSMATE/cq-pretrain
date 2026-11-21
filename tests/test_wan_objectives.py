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
from models.cq_backbone import backbone
from data.batch_layout import build_plan
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
    flags.setdefault('generation_target',True)
    return dict(type='image',pixels=np.zeros((h,w,3),np.uint8),**flags)


class Documents(fixtures.Documents):
    signature=dict(seed=42,epoch=3,view='A',rank=0,snapshot='wan-objectives')
    def __init__(self,rows,index=0):super().__init__(rows,index);self.read=[]
    def state_dict(self):return dict(index=self.index,signature=self.signature.copy())
    def image(self,doc,element):
        self.read.append(doc['id'])
        return super().image(doc,element)


def read_documents(rows, encode=lambda value: [11], grant_field='generation_target'):
    stream = object.__new__(DocumentStream)
    stream.catalog = SimpleNamespace(read_rows=lambda *args: rows)
    stream.tokenizer = SimpleNamespace(encode_batch=lambda spans, **kw: [SimpleNamespace(ids=encode(s)) for s in spans])
    stream.special_ids = set()
    stream.counting_matches_corpus = False
    stream.signature = {'view': 'A','generation_target_field':grant_field}
    return stream._load(0, 0, len(rows))['rows']


def test_native_occurrences_keep_declared_permissions_and_null_flags():
    # Run the real online adapter with a trivial tokenizer and in-memory catalog.
    def segment(role,turn,kind):
        return dict(role=role,turn=turn,kind=kind,text='body',image_index=0)
    rows=[dict(sample_id='understand',format='conversation',segments=[segment('user',0,'image'),segment('assistant',1,'text')]),
          dict(sample_id='generate',format='conversation',segments=[segment('user',0,'image'),segment('user',0,'text'),segment('assistant',1,'image')]),
          dict(sample_id='interleaved',format='interleaved',segments=[segment(None,None,'text'),segment(None,None,'image')])]
    for row in rows:row.update(images=[dict(path='test.png')],source_shard='',text_tokens=1)
    rows[0].update(direction='I2T',non_cfg=True);rows[1]['non_cfg']=False;rows[2]['images'][0]['non_cfg']=True
    rows[1]['segments'][0]['image_loss_allowed']=False
    rows[1]['segments'][-1]['generation_target']=True
    rows[2]['segments'][-1]['generation_target']=True
    documents=read_documents(rows)
    assert [d['direction'] for d in documents]==['I2T',None,None]
    assert [[e['generation_target'] for e in d['elements'] if e['type']=='image']
            for d in documents]==[[False],[False,True],[True]]
    assert all(e['supervised'] for e in documents[-1]['elements'] if e['type']=='text')
    assert [d['non_cfg'] for d in documents]==[True,False,None]
    assert [[e['non_cfg'] for e in d['elements'] if e['type']=='image']
            for d in documents]==[[True],[False,False],[True]]


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
def test_full_interleaved_targets_read_their_own_real_prefix(setup,leading_text):
    cfg,tokens=setup;cfg=replace(cfg,visual_uncond_probability=0.,visual_instruction_dropout_probability=0.)
    elements=([text(21,22)] if leading_text else [])+[picture(),text(11,12),picture(),text(13)]
    rows=[dict(id='natural',format='interleaved',templated=False,elements=elements)]
    batch,receipt=Packer(Documents(rows),cfg,64,tokens).next()
    assert batch.generation_image_indices.tolist()==[0,1]
    assert batch.generation_unconditional.tolist()==[not leading_text,False]
    assert receipt['unconditional_images']==int(not leading_text)
    assert receipt['instruction_dropout_eligible_images']==1 and receipt['instruction_dropped_images']==0
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
    rows=[dict(id='i2t',direction='I2T',elements=[picture(128,128,image_loss_allowed=False),text(11)]),
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
        elements=[picture(non_cfg=True,image_loss_allowed=False),text(11)])]),cfg,32,tokens).next()
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


@pytest.mark.parametrize('fmt', ['conversation', 'causal_sequence'])
@pytest.mark.parametrize('visibility', ['visual_block', 'causal'])
def test_history_queries_keep_prefix_and_cannot_read_instruction_or_target(setup, fmt, visibility):
    cfg, tokens = setup
    cfg = replace(cfg, content_visibility=visibility, visual_uncond_probability=0.,
                  visual_instruction_dropout_probability=1.)
    def words(role, turn, value):
        return dict(kind='text', role=role, turn=turn, text=value, supervised=True)
    def visual(role, turn, index):
        return dict(kind='image', role=role, turn=turn, image_index=index, generation_target=index==1)
    row = dict(sample_id='edit', format=fmt, direction='T2I', images=[dict(path='source'), dict(path='target')],
               source_shard='', segments=[words('user',0,'older'), words('assistant',1,'reply'),
               visual('user',2,0), words('user',2,'change'), words('user',2,'colour'), visual('assistant',3,1)])
    vocab = dict(older=[11], reply=[12], change=[21,22], colour=[23])
    doc, = read_documents([row], lambda s: vocab.get(s, [51]))
    for e in doc['elements']:
        if e['type'] == 'image':
            e['pixels'] = picture()['pixels']
    docs = [dict(id='other', elements=[text(30)]), doc]
    hist, receipt = Packer(Documents(docs), cfg, 64, tokens).next()
    full, _ = Packer(Documents(docs), replace(cfg,visual_instruction_dropout_probability=0.), 64, tokens).next()
    assert receipt['instruction_dropout_eligible_images'] == receipt['instruction_dropped_images'] == 1
    assert receipt['unconditional_images'] == 0
    assert hist.generation_image_indices.tolist() == [1]
    assert hist.generation_instruction_dropped.tolist() == [False, True]
    cut = int((hist.ids[0] == 21).nonzero().item())
    assert hist.generation_instruction_starts.tolist() == [-1, cut]
    plan = hist.mainline
    assert plan.query_read_sigma[plan.query_image[0]].eq(cut).all()
    assert torch.equal(plan.cq_predicate.dense(), plan.cq_mask)
    assert torch.equal(hist.text_indices, full.text_indices)
    for key in ('query_positions', 'query_rope', 'condition_indices', 'rope_cos', 'rope_sin'):
        assert torch.equal(getattr(plan, key), getattr(full.mainline, key))

    torch.manual_seed(312)
    model = JointModel(cfg).eval()
    features = torch.randn(len(hist.vision.content_indices), cfg.width, requires_grad=True)
    source_rows = hist.vision.content_indices < cut
    target = plan.condition_indices[1]
    output = backbone(model, hist, vision_features=features)
    output[target].square().sum().backward()
    assert features.grad[source_rows].abs().sum() > 0
    assert features.grad[~source_rows].count_nonzero() == 0
    assert model.embedding.weight.grad[[21,22,23,30]].count_nonzero() == 0
    with torch.no_grad():
        changed = replace(hist, ids=hist.ids.clone())
        changed.ids[0, cut:] = 32
        changed.ids[hist.ids == 30] = 33  # Adjacent packed document.
        noisy = features.clone()
        noisy[~source_rows] += torch.randn_like(noisy[~source_rows])
        assert torch.equal(backbone(model, changed, vision_features=noisy)[target], output[target])
        full_output = backbone(model, full, vision_features=features)
        assert not torch.allclose(backbone(model, replace(full, ids=changed.ids), vision_features=noisy)[target],
                                  full_output[target])
        assert torch.equal(full_output[hist.text_indices], output[hist.text_indices])
        older = hist.ids.clone()
        older[older == 11] = 33
        assert not torch.allclose(backbone(model, replace(hist, ids=older), vision_features=features)[target], output[target])
        noisy[source_rows] += torch.randn_like(noisy[source_rows])
        assert not torch.allclose(backbone(model, hist, vision_features=noisy)[target], output[target])


def test_instruction_dropout_eligibility_and_exact_replay(setup):
    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=.25, visual_instruction_dropout_probability=.5)
    prompt = dict(text(12), instruction_start=True, context_text=True)
    history = dict(text(11), context_text=True)
    source = picture(generation_target=False)
    rows = [dict(id=str(i), direction='T2I', elements=[history, prompt, picture()]) for i in range(60)]
    packer = Packer(Documents(rows), cfg, 24, tokens)
    baseline = Packer(Documents(rows), replace(cfg,visual_instruction_dropout_probability=0.), 24, tokens)
    selected = []
    while (item := packer.next()) is not None:
        batch, receipt = item
        old, old_receipt = baseline.next()
        assert torch.equal(batch.generation_unconditional, old.generation_unconditional)
        ids = [doc['id'] for doc in receipt['documents']]
        draws = [not null and sample_objective(f'{key}:2', .5, 42, 3, 'visual-instruction-drop-v1')
                 for key, null in zip(ids, batch.generation_unconditional.tolist())]
        selected.extend(draws)
        assert batch.generation_instruction_dropped.tolist() == draws
        assert receipt['instruction_dropout_eligible_images'] == len(ids) - receipt['unconditional_images']
        assert receipt['instruction_dropped_images'] == sum(draws)
        if receipt['serial'] == 1:
            state = receipt['next_state']
            assert old_receipt['next_state']['packing']['objective_policy'] == 'wan_occurrence_prefix_v1'
            assert 'visual_instruction_dropout_probability' not in old_receipt['next_state']['packing']
            replay = Packer(Documents(rows, state['documents']['index']), cfg, 24, tokens, state)
            with pytest.raises(ValueError, match='packing policy'):
                Packer(Documents(rows), cfg, 24, tokens, old_receipt['next_state'])
        else:
            restored, replayed = replay.next()
            assert replayed == receipt and torch.equal(restored.mainline.query_read_sigma, batch.mainline.query_read_sigma)
    assert 0 < sum(selected) < len(selected)

    cfg = replace(cfg, visual_uncond_probability=0., visual_instruction_dropout_probability=1.)
    cases = [
        [dict(text(50),context_text=False), prompt, picture()],                        # Only a wrapper before the instruction.
        [history, prompt, source, picture()],        # Image inside the proposed suffix.
        [history, prompt, picture(image_loss_allowed=False)], # Permission denied.
        [history, prompt, picture(non_cfg=False)],            # Explicit full condition.
        [history, prompt, picture(non_cfg=True)],             # Explicit null condition.
        [source, picture()],                                  # No intervening text.
    ]
    for elements in cases:
        _, receipt = Packer(Documents([dict(id='excluded', elements=elements)]), cfg, 48, tokens).next()
        assert receipt['instruction_dropped_images'] == receipt['instruction_dropout_eligible_images'] == 0


@pytest.mark.parametrize('visibility', ['visual_block', 'causal'])
def test_pruned_edit_conditions_match_training_readouts(setup, visibility):
    from evaluations.sampling import make_history_batch, prune_queries

    cfg, tokens = setup
    cfg = replace(cfg, content_visibility=visibility)
    processor = NativeImageProcessor(cfg, 128)
    source = np.random.default_rng(83).integers(0, 256, (64,96,3), dtype=np.uint8)
    tokenizer = SimpleNamespace(encode=lambda value, **kw: SimpleNamespace(ids=[51]))
    full_doc = make_batch(cfg, tokens, [21,22], processor(np.zeros_like(source)), 'T2I', prune=False,
                          tokenizer=tokenizer, context=processor(source))
    full = prune_queries(replace(full_doc), cfg, full_doc.mainline.query_image[0])
    hist = make_history_batch(full, cfg)
    torch.manual_seed(53)
    model = JointModel(cfg).eval()
    with torch.no_grad():
        for sampled in (full, hist):
            train = replace(full_doc, generation_instruction_dropped=sampled.generation_instruction_dropped)
            train.mainline = build_plan(train, cfg)
            expected = model(train, compute_image_loss=False)['image_conditions']
            # A pruned EDIT pack has exactly the target's visual Qs in output order.
            actual = backbone(model, sampled)[sampled.mainline.output_positions][None]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_last_text_group_ignores_role_and_windows_do_not_invent_history(setup):
    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=0., visual_instruction_dropout_probability=1.)
    for role, extra in [(None, []), ('user', [dict(kind='text', text='tool result', role='tool', turn=2)])]:
        segments = [dict(kind='text', text='earlier', role='assistant', turn=0),
                    dict(kind='text', text='instruction', role=role, turn=1), *extra,
                    dict(kind='image', image_index=0, role='assistant', turn=3,generation_target=True)]
        row = dict(sample_id='ambiguous', format='conversation', image_loss_allowed=True, images=[dict(path='target')], segments=segments)
        doc, = read_documents([row])
        for e in doc['elements']:
            if e['type']=='image':
                e['pixels'] = picture()['pixels']
        batch, receipt = Packer(Documents([doc]), cfg, 48, tokens).next()
        assert batch.generation_image_count == receipt['instruction_dropout_eligible_images'] == 1
    for new_group in (False,True):
        doc = dict(id='windowed', format='conversation', elements=[
            dict(text(*([11]*30)), context_text=True),
            dict(text(12), context_text=True, instruction_start=new_group), picture()])
        packer = Packer(Documents([doc]), replace(cfg,preserve_documents=False), 24, tokens)
        packer.next()
        batch, receipt = packer.next()
        assert batch.generation_image_count == 1
        assert receipt['instruction_dropout_eligible_images'] == receipt['instruction_dropped_images'] == int(new_group)


def test_malformed_history_flags_cannot_hide_a_source_image(setup):
    from evaluations.sampling import make_history_batch
    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=0., visual_instruction_dropout_probability=1.)
    doc = dict(id='edit',elements=[dict(text(11),context_text=True), picture(generation_target=False),
               dict(text(12),context_text=True,instruction_start=True),picture()])
    batch, _ = Packer(Documents([doc]), cfg, 64, tokens).next()
    early = batch.generation_instruction_starts.clone()
    early[-1] = 1
    with pytest.raises(ValueError, match='intervening visual'):
        build_plan(replace(batch,generation_instruction_starts=early), cfg)
    with pytest.raises(ValueError, match='conditional target'):
        make_history_batch(replace(batch,generation_unconditional=torch.tensor([False,True])),cfg)
    with pytest.raises(ValueError, match='generation target'):
        make_history_batch(replace(batch,generation_image_indices=torch.empty(0,dtype=torch.long)),cfg)


def test_validation_source_and_history_images_match_native_exif_orientation(tmp_path):
    import hashlib
    from PIL import Image
    from evaluations.sampling import context_image, encode_history

    image=Image.fromarray(np.random.default_rng(301).integers(0,256,(48,80,3),dtype=np.uint8))
    exif=Image.Exif()
    exif[274]=6  # display the stored image rotated 90 degrees clockwise
    path=tmp_path/'rotated.jpg'
    image.save(path,exif=exif)
    item=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    stream=object.__new__(DocumentStream)
    stream.images_preprocessed=False
    expected=stream._read_image(dict(item))
    assert expected.shape==(80,48,3)
    with Image.open(path) as stored:
        np.testing.assert_array_equal(expected,np.asarray(stored.transpose(Image.Transpose.ROTATE_270)))
    for checksum in (None,item['sha256']):
        np.testing.assert_array_equal(context_image(path,checksum,np.copy),expected)
    tokenizer=SimpleNamespace(encode=lambda *a,**kw:SimpleNamespace(ids=[11]))
    history=encode_history(dict(history=[dict(kind='image',role='user',turn=0,image_index=0)],images=[item]),
                           tokenizer,np.copy)
    np.testing.assert_array_equal(next(e['prepared'] for e in history if e['type']=='image'),expected)
    with pytest.raises(ValueError,match='changed'):
        context_image(path,'0'*64,np.copy)
    with pytest.raises(ValueError,match='changed'):
        context_image(path,'',np.copy)


@pytest.mark.parametrize('query_mode', ['native', 'fixed'])
def test_sampling_scales_share_positive_geometry_and_skip_redundant_forwards(setup, tmp_path, monkeypatch, query_mode):
    from PIL import Image
    from data.generation import patch_latent
    from evaluations import sampling

    cfg, tokens = setup
    cfg = replace(cfg, visual_query_mode=query_mode, image_condition_queries=64)
    processor = NativeImageProcessor(cfg, 128)
    image_path = tmp_path / 'input.png'
    Image.fromarray(picture()['pixels']).save(image_path)
    tokenizer = SimpleNamespace(encode=lambda value, **kw: SimpleNamespace(ids=[19,20] if value else []))
    monkeypatch.setattr(sampling, 'accel_type', lambda: 'cpu')
    monkeypatch.setattr(sampling, 'P', SimpleNamespace(state=lambda: SimpleNamespace(get_tensor_model_parallel_rank=lambda:1)))
    batches, expected_rows, projected, predicted, decoded = {}, [], [], [], []
    def read_condition(model, batch, vision_features):
        name = ('null' if batch.mainline.query_unconditional.all() else
                'hist' if batch.generation_instruction_dropped.any() else 'full')
        assert name not in batches
        batches[name] = batch, vision_features
        plan = batch.mainline
        expected_rows.append(plan.output_positions[plan.query_image[0]])
        hidden = torch.full((plan.output_size, cfg.width), {'full':7., 'hist':3., 'null':-2.}[name])
        hidden[:, 1] = torch.arange(plan.output_size)
        return hidden
    monkeypatch.setattr(sampling, 'backbone', read_condition)
    def project(raw, layout):
        # Channel 1 identifies every backbone readout. Inspect the exact rows
        # that Wan will project, including their order, for full/hist/null.
        actual_rows = raw.reshape(-1, cfg.width).index_select(0, layout.condition_indices)[:, 1]
        assert torch.equal(actual_rows, expected_rows[-1]), (actual_rows, expected_rows[-1])
        projected.append(raw)
        return raw.flatten()[0]
    def predict(xt, raw, times, layout, condition, cached_kv):
        assert cached_kv is condition and raw.flatten()[0] == condition
        predicted.append(condition.item())
        return torch.full_like(xt, condition.item())
    renderer = SimpleNamespace(project_conditions=project, cache_condition_kv=lambda c:c, predict=predict)
    model = SimpleNamespace(cfg=cfg, renderer=renderer, embedding=torch.nn.Embedding(1, cfg.width),
                            patch_embed=lambda vision:torch.ones(len(vision.content_indices), cfg.width))
    def decode(latents):
        decoded.append(patch_latent(latents[0]))
        return [torch.zeros(3,1,32,32)]
    def run(task, scales, branches, velocity, prompt='edit', **fields):
        batches.clear(); expected_rows.clear(); projected.clear(); predicted.clear()
        case = dict(task=task, id='sample', image=str(image_path), prompt=prompt, seed=39,
                    height=64, width=96, steps=2, shift=1., **scales, **fields)
        sampling.generate(model, case, tokenizer, tokens, processor, SimpleNamespace(decode=decode), tmp_path)
        assert set(batches) == set(branches)
        assert len(projected) == len(branches) and len(predicted) == case['steps'] * len(branches)
        actual = decoded[-1]
        expected = torch.randn(actual.shape, generator=torch.Generator().manual_seed(case['seed']))
        for _ in range(2):
            expected = expected - .5 * velocity
        assert torch.equal(actual, expected)
        return actual

    # The single-scale expression and both equal-scale fast paths stay exact.
    single = run('EDIT', dict(guidance=2.), ['full','null'], 16.)
    assert torch.equal(run('EDIT', dict(s_ctx=2., s_ins=2.), ['full','null'], 16.), single)
    run('EDIT', dict(s_ctx=1., s_ins=1.), ['full'], 7.)
    # The omitted scale inherits guidance: -2 + .5*(3+2) + 2*(7-3).
    run('EDIT', dict(guidance=2., s_ctx=.5), ['full','null','hist'], 8.5)
    full, features = batches['full']
    hist, hist_features = batches['hist']
    assert hist_features is features and torch.equal(full.ids, hist.ids)
    assert torch.equal(full.mainline.query_rope, hist.mainline.query_rope)
    assert torch.equal(full.mainline.query_positions, hist.mainline.query_positions)
    cut = hist.generation_instruction_starts[-1].item()
    assert hist.mainline.query_read_sigma.eq(cut).all()
    assert hist.mainline.cq_mask[0,0,hist.ids.numel():,cut:].all()
    run('T2I', dict(s_ctx=.5, s_ins=2.), ['full','null'], 16.)
    run('EDIT', dict(s_ctx=.5, s_ins=2.), ['full','null'], 2.5, prompt='')
    past = [dict(role='user',turn=0,kind='text',text='older request'),
            dict(role='assistant',turn=1,kind='image',image_index=0),
            dict(role='user',turn=2,kind='text',text='another request'),
            dict(role='assistant',turn=3,kind='image',image_index=0)]
    run('T2I', dict(s_ctx=.5,s_ins=2.), ['full','null','hist'], 8.5, history=past, images=[dict(path=str(image_path))])
    full, features = batches['full']
    hist, hist_features = batches['hist']
    assert full.generation_image_indices.tolist() == [2] and features is hist_features and features.eq(1).all()
    cut = int(hist.generation_instruction_starts[-1])
    assert (hist.image_indices[:2] < cut).all() and hist.mainline.query_read_sigma.eq(cut).all()
    run('T2I', dict(s_ctx=.5,s_ins=2.), ['full','null','hist'], 8.5, history=past[:1])
    for invalid in (float('nan'), float('inf'), 10**400, True):
        with pytest.raises(ValueError, match='finite'):
            run('T2I', dict(s_ctx=invalid), [], 0.)
    with pytest.raises(ValueError, match='finite'):
        run('T2I', dict(guidance=float('nan'),s_ctx=1.,s_ins=2.), [], 0.)
