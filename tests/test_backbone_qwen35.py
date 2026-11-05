"""Video adapter checks against native Qwen and main's shared CQ layers."""
from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from models.backbone_qwen35 import Qwen35CQBackbone, cq_plan
from tests.test_modeling_umm_video import configuration


def config(layers=4):
    return configuration(layers=layers).qwen_config.text_config


def batch():
    return dict(input_ids=torch.tensor([[3, 4, 61, 60, 60, 60, 60, 62, 7]]),
                token_types=torch.tensor([[0, 0, 2, 1, 1, 1, 1, 2, 0]]),
                position_ids=torch.arange(9)[None, None].expand(3, 1, -1))


def test_causal_content_matches_native_outputs_and_parameter_gradients():
    torch.manual_seed(51)
    cfg = config()
    cfg.content_visibility = 'causal'
    reference = Qwen3_5TextModel(deepcopy(cfg)).eval()
    model = Qwen35CQBackbone(cfg).eval()
    with torch.no_grad():
        for name, target in model.native_parameters().items():
            target.copy_(reference.state_dict()[name])
    ids = torch.tensor([[3, 4, 6, 7, 8, 9]])
    expected = reference(ids, use_cache=False).last_hidden_state
    actual = model(dict(input_ids=ids), query_stream=False)['content_hidden']
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=3e-4)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    from models.qwen35_layers import unfused_parameters
    gradients = {key: value for name, parameter in model.named_parameters() if parameter.grad is not None
                 for key, value in unfused_parameters(name, parameter.grad, model.cfg).items()}
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(gradients[name], parameter.grad, atol=3e-5, rtol=5e-4, msg=name)


@pytest.mark.parametrize('checkpointed', [False, True])
def test_readonly_visual_queries_never_leak_target_or_future_content(checkpointed):
    torch.manual_seed(52)
    model = Qwen35CQBackbone(config(8)).train()
    model.gradient_checkpointing = checkpointed
    source = torch.randn(1, 9, 32, requires_grad=True)
    inputs = dict(batch(), inputs_embeds=source)
    output = model(inputs, return_content=True)
    probe = torch.randn_like(output['query_hidden'][:, 3:7])
    (output['query_hidden'][:, 3:7] * probe).sum().backward()
    assert source.grad[:, :3].abs().sum() > 0
    assert torch.count_nonzero(source.grad[:, 3:]) == 0
    with torch.no_grad():
        changed = source.detach().clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
        other = model(dict(inputs, inputs_embeds=changed), return_content=True)
    torch.testing.assert_close(other['query_hidden'][:, 3:7], output['query_hidden'][:, 3:7], rtol=0, atol=0)


def test_non_cfg_has_no_content_gradients_through_attention_conv_or_gdn():
    torch.manual_seed(53)
    model = Qwen35CQBackbone(config(8)).train()
    inputs = batch()
    inputs['visual_uncond_mask'] = inputs['token_types'].eq(1)
    source = torch.randn(1, 9, 32, requires_grad=True)
    output = model(dict(inputs, inputs_embeds=source))['query_hidden']
    (output[:, 3:7] * torch.randn_like(output[:, 3:7])).sum().backward()
    assert torch.count_nonzero(source.grad) == 0
    assert model.embed_tokens.weight.grad[60].abs().sum() > 0


@pytest.mark.parametrize('visibility', ['visual_block', 'causal'])
def test_visual_content_visibility_and_readonly_query_history(visibility):
    torch.manual_seed(54)
    cfg = config()
    cfg.content_visibility = visibility
    model = Qwen35CQBackbone(cfg)
    inputs = batch()
    plan, _ = cq_plan(inputs, model.cfg, inputs['position_ids'])
    allowed = ~plan.cq_mask[0, 0]
    # Visual queries all read before the target group, irrespective of C policy.
    assert not allowed[9 + 3:9 + 7, 3:].any()
    assert allowed[3, 6].item() == (visibility == 'visual_block')
    assert not allowed[3:7, 7:].any()
    source = torch.randn(1, 9, 32, requires_grad=True)
    content = model(dict(inputs, inputs_embeds=source), return_content=True)['content_hidden']
    (content[:, 3] * torch.randn_like(content[:, 3])).sum().backward()
    assert (source.grad[:, 6].abs().sum() > 0).item() == (visibility == 'visual_block')
    assert torch.count_nonzero(source.grad[:, 7:]) == 0


def test_text_query_matches_prefix_inference():
    torch.manual_seed(55)
    model = Qwen35CQBackbone(config(8)).eval()
    ids = torch.tensor([[3, 4, 6, 7, 8]])
    output = model(dict(input_ids=ids))['query_hidden']
    # Later tokens cannot change a query's prefix result.
    for index in range(ids.shape[1]):
        prefix = model(dict(input_ids=ids[:, :index + 1]))['query_hidden'][:, -1]
        torch.testing.assert_close(output[:, index], prefix, rtol=1e-5, atol=2e-6)


def test_packed_documents_match_separate_forwards_and_padding_cannot_leak():
    torch.manual_seed(56)
    model = Qwen35CQBackbone(config(8)).eval()
    ids = torch.tensor([[3, 4, 6, 7, 8, 0]])
    segments = torch.tensor([[0, 0, 0, 1, 1, -1]])
    types = torch.tensor([[0, 0, 0, 0, 0, 3]])
    packed = model(dict(input_ids=ids, segment_ids=segments, token_types=types))['query_hidden']
    for start, end in [(0, 3), (3, 5)]:
        separate = model(dict(input_ids=ids[:, start:end]))['query_hidden']
        torch.testing.assert_close(packed[:, start:end], separate, atol=2e-6, rtol=1e-5)
    assert torch.count_nonzero(packed[:, 5]) == 0
