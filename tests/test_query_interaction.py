"""Bidirectional visual queries and the opener null condition (ablation axes)."""
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from configs.config import tiny_config
from data.batch import Batch
from data.batch_layout import build_plan
from data.codec import DevicePreparer
from data.packing import Packer
from evaluations.sampling import prune_queries
from models.backbone_qwen35 import QwenBlock
from models.modeling_umm import JointModel
from training.loss import compose_losses
from tests.test_wan_objectives import Documents, picture, setup, text  # noqa: F401  (pytest fixture)


def stack(cfg, seed=3):
    torch.manual_seed(seed)
    return torch.nn.ModuleList([QwenBlock(cfg, i) for i in range(cfg.layers)]).eval()


@torch.no_grad()
def run(blocks, plan, s, hidden, rope=None):
    rope = rope or (plan.rope_cos, plan.rope_sin)
    for block in blocks:
        hidden = block(hidden, rope, plan, s)
    return hidden


def packed(cfg, tokens, rows, length=96):
    batch, receipt = Packer(Documents(rows), cfg, length, tokens).next()
    assert all(d['completed'] for d in receipt['documents'])
    return batch


def two_targets():
    return [dict(id='plain', elements=[text(21, 22)]),
            dict(id='talk', format='conversation', direction='T2I', elements=[
                text(11, 12, 13), picture(64, 96), text(14, 15), picture(32, 64), text(16)])]


def test_default_axes_change_nothing(setup):
    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=0.)
    assert (cfg.query_interaction, cfg.null_condition) == ('none', 'masked')
    assert not {'query_interaction', 'null_condition'} & set(cfg.wan_objective_policy())
    assert replace(cfg, query_interaction='bidirectional', null_condition='bos').wan_objective_policy()[
        'query_interaction'] == 'bidirectional'
    plan = packed(cfg, tokens, two_targets()).mainline
    assert plan.query_blocks == ()
    assert plan.delta_plan.query_write_start == len(plan.delta_plan.query_source)
    with pytest.raises(ValueError, match='query_interaction'):
        replace(cfg, query_interaction='causal').validate_model()
    with pytest.raises(ValueError, match='null_condition'):
        replace(cfg, null_condition='learned').validate_model()


@pytest.mark.parametrize('visibility', ['visual_block', 'causal'])
def test_block_reads_its_prefix_and_itself_and_nothing_else(setup, visibility):
    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=0., content_visibility=visibility)
    base = packed(cfg, tokens, two_targets())
    batch = packed(replace(cfg, query_interaction='bidirectional'), tokens, two_targets())
    plan, s = batch.mainline, batch.ids.shape[1]
    for name in ('query_positions', 'query_rope', 'condition_indices', 'output_positions'):
        torch.testing.assert_close(getattr(plan, name), getattr(base.mainline, name), rtol=0, atol=0)
    assert len(plan.query_blocks) == 2 and plan.query_blocks[-1][1] == s + len(plan.query_positions)
    # The second target's prefix holds the first target's clean image; its own content is excluded.
    (a0, a1, ak0, ak1), (b0, b1, bk0, bk1) = plan.query_blocks
    first_image, second_image = (x[x.ge(0)] for x in batch.image_indices)
    assert ak0 == bk0 and ak1 <= int(first_image[0]) < int(first_image[-1]) < bk1 <= int(second_image[0])
    blocks = stack(cfg)
    hidden = torch.randn(s + len(plan.query_positions), 1, cfg.width)
    expected = run(blocks, plan, s, hidden)
    plain = run(blocks, base.mainline, s, hidden)
    # Content and text queries are untouched by the visual-Q interaction.
    torch.testing.assert_close(expected[:a0], plain[:a0], rtol=0, atol=1e-6)
    assert not torch.allclose(expected[a0:], plain[a0:])
    for row0, row1, key0, key1 in plan.query_blocks:
        outside = torch.ones(s, dtype=torch.bool)
        outside[key0:key1] = False
        changed = hidden.clone()
        changed[:s][outside] += torch.randn_like(changed[:s][outside])
        torch.testing.assert_close(run(blocks, plan, s, changed)[row0:row1], expected[row0:row1], rtol=0, atol=0)
        changed = hidden.clone()
        changed[key1 - 1] += 1
        assert not torch.allclose(run(blocks, plan, s, changed)[row0:row1], expected[row0:row1])
    # One late query moves every query of its own target, including earlier
    # ones, and nothing else: not content, not text queries, not other targets.
    changed = hidden.clone()
    changed[b1 - 1] += 1
    actual = run(blocks, plan, s, changed)
    assert all(not torch.allclose(actual[row], expected[row]) for row in range(b0, b1))
    torch.testing.assert_close(actual[:b0], expected[:b0], rtol=0, atol=0)
    # Without the interaction each query is a function of content alone.
    actual = run(blocks, base.mainline, s, changed)
    torch.testing.assert_close(actual[:b1 - 1], plain[:b1 - 1], rtol=0, atol=0)


def test_block_equals_a_tied_content_block_placed_after_the_prefix():
    """With no convolution history, appended queries are exactly a content image block."""
    cfg = tiny_config(width=32, visual_query_mode='native', linear_conv_kernel=1,
                      query_interaction='bidirectional', fused=False)
    empty = torch.empty(0)

    def positions(s, image):
        rope = torch.arange(s).float()[None, :, None].repeat(3, 1, 1)
        local = torch.arange(4)
        rope[0, image, 0] = float(image[0])
        rope[1, image, 0] = (image[0] + local // 2).float()
        rope[2, image, 0] = (image[0] + local % 2).float()
        return rope

    # A: prefix 0..2, the clean target at 3..6, later text; four queries at the tail.
    s = 9
    image = torch.arange(3, 7)
    sigma = torch.arange(s)[None].clone()
    sigma[0, 3:7] = 3
    a = Batch(torch.zeros(1, s, dtype=torch.long), empty, image[None], torch.tensor([1, 8]),
              torch.zeros(1, s, dtype=torch.bool), sigma, torch.zeros(1, s, dtype=torch.long),
              positions(s, image), empty, empty, vision=SimpleNamespace(grid_thw=((1, 4, 4),)),
              generation_image_indices=torch.tensor([0]))
    plan = build_plan(a, cfg)
    (row0, row1, key0, key1), = plan.query_blocks
    assert (row0, row1, key0, key1) == (s + 2, s + 6, 0, 3)
    # B: the same prefix followed by the query vectors as a tied content block.
    t = 7
    b = Batch(torch.zeros(1, t, dtype=torch.long), empty, image[None], torch.tensor([1]),
              torch.zeros(1, t, dtype=torch.bool), sigma[:, :t].clone(), torch.zeros(1, t, dtype=torch.long),
              positions(t, image), empty, empty, vision=SimpleNamespace(grid_thw=((1, 4, 4),)),
              generation_image_indices=torch.empty(0, dtype=torch.long))
    reference = build_plan(b, replace(cfg, query_interaction='none'))
    assert reference.query_blocks == ()
    blocks = stack(cfg, seed=11)
    hidden = torch.randn(s + 6, 1, cfg.width)
    as_content = torch.cat((hidden[:3], hidden[row0:row1], hidden[s:s + 1]))
    cos, sin = reference.rope_cos.clone(), reference.rope_sin.clone()
    cos[3:7], sin[3:7] = plan.rope_cos[row0:row1], plan.rope_sin[row0:row1]
    expected = run(blocks, reference, t, as_content, (cos, sin))[3:7]
    actual = run(blocks, plan, s, hidden)[row0:row1]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('interaction,null', list(product(['none', 'bidirectional'], ['masked', 'bos'])))
def test_null_condition_is_one_function_of_the_grid(setup, interaction, null):
    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=1., query_interaction=interaction, null_condition=null)
    rows = [dict(id='short', format='conversation', direction='T2I', elements=[text(11), picture()]),
            dict(id='long', format='conversation', direction='T2I',
                 elements=[text(12, 13, 14, 15, 16, 17), picture(), text(18)])]
    batch = packed(cfg, tokens, rows)
    assert batch.generation_unconditional.tolist() == [True, True]
    plan, s = batch.mainline, batch.ids.shape[1]
    visual = plan.query_image[0].nonzero().flatten()
    n = len(visual) // 2
    first, second = s + visual[:n], s + visual[n:]
    openers = batch.generation_null_positions - 1
    assert batch.ids[0, openers].eq(tokens['bos']).all() and openers[0] != openers[1]
    if null == 'bos':
        assert not plan.query_unconditional.any()
        assert plan.query_read_sigma[visual].tolist() == sum(([int(p)] * n for p in openers + 1), [])
    else:
        assert plan.query_unconditional[visual].all()
    blocks = stack(cfg)
    hidden = torch.randn(s + len(plan.query_positions), 1, cfg.width)
    # As in the model: every visual query starts from one embedding and every
    # document opens with one token.
    hidden[s + visual] = hidden[s + visual[0]].clone()
    hidden[openers] = hidden[openers[0]].clone()
    expected = run(blocks, plan, s, hidden)
    # Two documents with different prompts and prefix lengths give one null condition.
    torch.testing.assert_close(expected[first], expected[second], rtol=1e-5, atol=1e-6)
    changed = hidden.clone()
    other = torch.ones(s, dtype=torch.bool)
    other[openers] = False
    changed[:s][other] += torch.randn_like(changed[:s][other])
    actual = run(blocks, plan, s, changed)
    torch.testing.assert_close(actual[first], expected[first], rtol=0, atol=0)
    torch.testing.assert_close(actual[second], expected[second], rtol=0, atol=0)
    if null == 'bos':
        changed = hidden.clone()
        changed[openers] += 1
        assert not torch.allclose(run(blocks, plan, s, changed)[first], expected[first])
    same_vector = torch.allclose(expected[first], expected[first][:1].expand_as(expected[first]), atol=1e-6)
    # Position enters only through rotary scores, and scores only reweight
    # values. Identical queries that read nothing, one key, or only each other
    # therefore stay one repeated vector. They are told apart only when they
    # weigh the opener against one another.
    assert same_vector == (not (interaction == 'bidirectional' and null == 'bos'))


def test_pruned_inference_plan_matches_the_training_pack(setup):
    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=0., query_interaction='bidirectional')
    batch = packed(cfg, tokens, two_targets())
    plan, s = batch.mainline, batch.ids.shape[1]
    blocks = stack(cfg)
    hidden = torch.randn(s + len(plan.query_positions), 1, cfg.width)
    keep = plan.query_image[0].nonzero().flatten()
    expected = run(blocks, plan, s, hidden)[s + keep]
    pruned = prune_queries(batch, cfg, plan.query_image[0]).mainline
    assert pruned.query_blocks[0][0] == s and pruned.query_blocks[-1][1] == s + len(keep)
    actual = run(blocks, pruned, s, torch.cat((hidden[:s], hidden[s + keep])))[s:]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match='as a whole'):
        prune_queries(packed(cfg, tokens, two_targets()), cfg, keep[1:])


@pytest.mark.parametrize('null', ['masked', 'bos'])
def test_joint_losses_train_through_interacting_queries(setup, null):
    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=.5, query_interaction='bidirectional', null_condition=null)
    rows = [dict(id=f'doc{i}', format='conversation', direction='T2I', non_cfg=bool(i % 2),
                 elements=[text(11 + i, 12), picture()]) for i in range(4)]
    batch, receipt = Packer(Documents(rows), cfg, 96, tokens).next()
    assert receipt['unconditional_images'] == 2
    ready, _ = DevicePreparer(cfg, device='cpu', encoder='fake').prepare(batch, receipt)
    torch.manual_seed(5)
    model = JointModel(cfg)
    losses = compose_losses(model(ready), cfg)
    assert torch.isfinite(losses['loss']) and losses['image_loss'] > 0 and losses['text_loss'] > 0
    losses['loss'].backward()
    assert model.embedding.weight.grad[cfg.image_mask_token_id].abs().sum() > 0
    gdn, attention = model.layers[0].linear_attn, model.layers[cfg.full_attention_interval - 1].self_attn
    assert gdn.conv1d.weight.grad.abs().sum() > 0
    projection = attention.kv_proj if cfg.qwen_fused_projections else attention.k_proj
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in projection.parameters())


@pytest.mark.parametrize('interaction', ['none', 'bidirectional'])
@pytest.mark.parametrize('full_pack', [False, True])
def test_shared_null_token_for_templated_and_natural_documents(setup, interaction, full_pack):
    from data.batch_layout import content_view
    from evaluations.sampling import make_null_batch

    cfg, tokens = setup
    cfg = replace(cfg, query_interaction=interaction, null_condition='token', null_token_id=29)
    rows = [dict(id='talk', templated=True, elements=[text(11, 12), picture(non_cfg=True)]),
            dict(id='natural', templated=False, elements=[text(16, 17, 18), picture(non_cfg=True)]),
            dict(id='conditional', templated=True, elements=[text(19, 20), picture(non_cfg=False)])]
    batch, receipt = Packer(Documents(rows), cfg, 96, tokens).next()
    if full_pack:
        batch, receipt = Packer(Documents(rows), cfg, receipt['occupied'], tokens).next()
    assert len(receipt['documents']) == 3 and all(d['completed'] for d in receipt['documents'])
    ids_before = batch.ids.clone()
    view = content_view(batch, cfg)
    plan = batch.mainline
    anchor = plan.null_anchor
    assert anchor == batch.ids.shape[1] - int(not full_pack)
    assert view.ids[0, anchor] == 29 and view.segments.eq(view.segments[0, anchor]).sum() == 1
    assert not batch.ids.eq(29).any()
    # Use the real forward/readout so gradients include the private token's embedding.
    model = JointModel(cfg).eval()
    def features(batch):
        hidden = model.backbone(batch, reference=True)
        return batch.mainline.gather_image_conditions(hidden, batch.generation_image_indices)
    first = features(batch)
    torch.testing.assert_close(first[0], first[1], atol=2e-5, rtol=2e-5)
    loss = first[:2].square().mean()
    gradient, = torch.autograd.grad(loss, model.embedding.weight)
    assert gradient[29].abs().sum() > 0
    assert gradient[[tokens['bos'], 11, 12, 16, 17, 18]].count_nonzero() == 0
    null_batch = make_null_batch(batch, cfg)
    torch.testing.assert_close(features(null_batch)[:2], first[:2], atol=2e-5, rtol=2e-5)
    with torch.no_grad():
        model.embedding.weight[29].add_(0.5)
    changed = features(batch)
    assert not torch.allclose(changed[0], first[0])
    torch.testing.assert_close(changed[2], first[2], atol=1e-6, rtol=1e-6)
    assert torch.equal(batch.ids, ids_before)
