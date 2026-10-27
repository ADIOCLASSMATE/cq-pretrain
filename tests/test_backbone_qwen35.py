"""CQ query isolation and comparisons against native Qwen3.5 arithmetic."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models import backbone_qwen35 as cq
from models.references import modeling_qwen35 as native
from models.references.modeling_qwen35 import (
    Qwen3_5Attention, Qwen3_5TextConfig, Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding,
    causal_conv1d_fn, l2norm,
)
from models.backbone_qwen35 import build_cq_masks


def config(layers=8):
    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
        pad_token_id=0,
    )
    cfg.mask_token_id, cfg.image_mask_token_id, cfg.boi_token_id = 59, 60, 61
    cfg._attn_implementation = "sdpa"
    return cfg


def visual_batch():
    # Legacy sigma metadata must no longer reorder physical content.
    return {
        "input_ids": torch.tensor([[3, 4, 61, 60, 60, 60, 60, 62, 7]]),
        "token_types": torch.tensor([[0, 0, 2, 1, 1, 1, 1, 2, 0]]),
        "sigma": torch.tensor([[0, 1, 2, 4, 4, 4, 4, 3, 8]]),
        "position_ids": torch.arange(9)[None, None].expand(3, 1, -1),
    }


def layout_for(batch, cfg):
    query_allowed, content_allowed = build_cq_masks(batch, cfg.boi_token_id)
    return cq._gdn_layout(batch, query_allowed, content_allowed,
                          cfg.linear_conv_kernel_dim), query_allowed


def capture_queries(monkeypatch):
    captured = []
    original = cq._cq_interleave_readonly_queries

    def record(content_q, query_q, *args):
        captured.append((content_q, query_q))
        return original(content_q, query_q, *args)

    monkeypatch.setattr(cq, "_cq_interleave_readonly_queries", record)
    return captured


def test_query_is_independent_of_other_queries():
    torch.manual_seed(51)
    cfg = config(4)
    layer = cq.Qwen35CQBackbone(cfg).eval().layers[0].linear_attn
    batch = visual_batch()
    layout, allowed = layout_for(batch, cfg)
    context = cq._CQContext(layout, allowed)
    content = torch.randn(1, 9, cfg.hidden_size)
    query = torch.randn_like(content, requires_grad=True)
    content_out, query_out = layer(content, cq_context=context, query_hidden_states=query)
    changed = query.detach().clone()
    changed[:, 4] = torch.randn_like(changed[:, 4]) * 3
    changed_content, changed_query = layer(content, cq_context=context, query_hidden_states=changed)
    others = torch.arange(9) != 4
    torch.testing.assert_close(content_out, changed_content, rtol=0, atol=0)
    torch.testing.assert_close(query_out[:, others], changed_query[:, others], rtol=0, atol=0)
    assert not torch.allclose(query_out[:, 4], changed_query[:, 4])
    gradient, = torch.autograd.grad((query_out[:, 4] * torch.randn_like(query_out[:, 4])).sum(), query)
    assert gradient[:, 4].abs().max() > 0
    assert torch.count_nonzero(gradient[:, others]) == 0


@pytest.mark.parametrize("unconditional", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_query_q_matches_native_qkv_q_for_visible_window(monkeypatch, unconditional, dtype):
    torch.manual_seed(52)
    cfg = config(4)
    layer = cq.Qwen35CQBackbone(cfg).eval().to(dtype).layers[0].linear_attn
    batch = visual_batch()
    if unconditional:
        batch["image_uncond_rows"] = torch.tensor([True])
    layout, allowed = layout_for(batch, cfg)
    captured = capture_queries(monkeypatch)
    content = torch.randn(1, 9, cfg.hidden_size, dtype=dtype)
    query = torch.randn_like(content)
    with torch.no_grad():
        layer(content, cq_context=cq._CQContext(layout, allowed), query_hidden_states=query)
        actual = layout.unpack(captured[0][1].flatten(-2), content.shape[:2])
        expected = []
        for position in range(content.shape[1]):
            history = torch.where(allowed[0, position])[0]
            history = history[-(cfg.linear_conv_kernel_dim - 1):]
            # Independent oracle: run the complete native QKV projection and
            # convolution on the visible content window plus this query alone.
            window = torch.cat((content[:, history], query[:, position:position + 1]), dim=1)
            projected = layer.in_proj_qkv(window).transpose(1, 2)
            convolved = causal_conv1d_fn(projected, layer.conv1d.weight.squeeze(1),
                                         layer.conv1d.bias, activation=layer.activation)
            q = convolved[:, :layer.key_dim, -1].reshape(1, layer.num_k_heads, layer.head_k_dim)
            q = q.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=1)
            expected.append(q.flatten(1))
        expected = torch.stack(expected, dim=1)
    tolerance = dict(rtol=0.02, atol=3e-6) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(actual, expected, **tolerance)
    # Both streams subsequently use this same per-head normalization/scaling
    # inside the native delta-rule operator, including repeated value heads.
    shape = (1, 9, layer.num_v_heads, layer.head_k_dim)
    actual = l2norm(actual.float().view(shape)) / layer.head_k_dim ** 0.5
    expected = l2norm(expected.float().view(shape)) / layer.head_k_dim ** 0.5
    torch.testing.assert_close(actual, expected, rtol=tolerance["rtol"], atol=2e-4 if dtype == torch.bfloat16 else 2e-6)


def test_same_source_q_matches_qkv_q_outputs_and_gradients(monkeypatch):
    torch.manual_seed(53)
    cfg = config(4)
    layer = cq.Qwen35CQBackbone(cfg).eval().layers[0].linear_attn
    batch = {"input_ids": torch.arange(1, 8)[None], "token_types": torch.zeros(1, 7, dtype=torch.long)}
    layout, allowed = layout_for(batch, cfg)
    captured = capture_queries(monkeypatch)
    shared_input = torch.randn(1, 7, cfg.hidden_size, requires_grad=True)
    layer(shared_input, cq_context=cq._CQContext(layout, allowed),
          query_hidden_states=shared_input)
    content_q, query_q = captured[0]
    torch.testing.assert_close(query_q, content_q, rtol=2e-5, atol=2e-7)
    probe = torch.randn_like(content_q)
    sources = (shared_input, layer.in_proj_qkv.weight, layer.conv1d.weight)
    expected = torch.autograd.grad((content_q * probe).sum(), sources, retain_graph=True)
    actual = torch.autograd.grad((query_q * probe).sum(), sources)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_default_queries_have_no_target_leak_and_use_allowed_content(checkpointed):
    torch.manual_seed(54)
    model = cq.Qwen35CQBackbone(config()).train()
    model.gradient_checkpointing = checkpointed
    batch = {**visual_batch(), "inputs_embeds": torch.randn(1, 9, 32, requires_grad=True)}
    query = model(batch)["query_hidden"]
    changed = batch["inputs_embeds"].detach().clone()
    changed[:, 3:7] = torch.randn_like(changed[:, 3:7]) * 10
    with torch.no_grad():
        alternative = model({**batch, "inputs_embeds": changed})["query_hidden"]
        torch.testing.assert_close(query[:, 3:7], alternative[:, 3:7], rtol=0, atol=0)
        assert not torch.allclose(query[:, 8], alternative[:, 8])
        prefix = batch["inputs_embeds"].detach().clone()
        prefix[:, 0] = torch.randn_like(prefix[:, 0]) * 10
        conditioned = model({**batch, "inputs_embeds": prefix})["query_hidden"]
        assert not torch.allclose(query[:, 3:7], conditioned[:, 3:7])
        uncond = {**batch, "image_uncond_rows": torch.tensor([True])}
        first = model(uncond)["query_hidden"]
        second = model({**uncond, "inputs_embeds": prefix})["query_hidden"]
        torch.testing.assert_close(first[:, 3:7], second[:, 3:7], rtol=0, atol=0)
    (query[:, 3:7] * torch.randn_like(query[:, 3:7])).sum().backward()
    gradient = batch["inputs_embeds"].grad
    assert gradient[:, [0, 1, 2, 7]].abs().max() > 0
    assert torch.count_nonzero(gradient[:, [3, 4, 5, 6, 8]]) == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_content_still_matches_native_text_model():
    torch.manual_seed(55)
    cfg = config(4)
    native = Qwen3_5TextModel(deepcopy(cfg)).eval()
    model = cq.Qwen35CQBackbone(cfg).eval()
    model.load_state_dict(native.state_dict(), strict=True)
    ids = torch.arange(1, 8)[None]
    with torch.no_grad():
        actual = model({"input_ids": ids}, return_content=True)["content_hidden"]
        content_only = model({"input_ids": ids}, query_stream=False)["content_hidden"]
        expected = native(input_ids=ids, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(actual, content_only, rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_full_attention_matches_native_backend(backend):
    torch.manual_seed(57)
    cfg = config(4)
    cfg._attn_implementation = backend
    reference = Qwen3_5Attention(cfg, 0).eval()
    layer = cq.Qwen3_5Attention(cfg, 0).eval()
    layer.load_state_dict(reference.state_dict(), strict=True)
    batch = visual_batch()
    hidden_states = torch.randn(1, 9, cfg.hidden_size)
    rope = Qwen3_5TextRotaryEmbedding(cfg)(hidden_states, batch["position_ids"])
    query_allowed, content_allowed = build_cq_masks(batch, cfg.boi_token_id)
    mask = torch.zeros(1, 1, 9, 9).masked_fill(~content_allowed[:, None], torch.finfo(torch.float32).min)
    # Native self-attention with identical Q sources is an independent oracle
    # for each visibility policy. Empty query rows must aggregate to zero.
    query_mask = torch.zeros_like(mask).masked_fill(~query_allowed[:, None], torch.finfo(torch.float32).min)
    with torch.no_grad():
        expected, expected_weights = reference(hidden_states, rope, mask)
        expected_query, _ = reference(hidden_states, rope, query_mask)
        expected_query = expected_query * query_allowed.any(-1, keepdim=True)
        (content, query), weights = layer(hidden_states, rope, content_allowed,
                                         cq_context=cq._CQContext(None, query_allowed),
                                         query_hidden_states=hidden_states)
    torch.testing.assert_close(content, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(query, expected_query, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def test_packed_segments_and_padding_match_independent_runs():
    torch.manual_seed(56)
    model = cq.Qwen35CQBackbone(config(4)).eval()
    batch = {
        "input_ids": torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0],
                                   [12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0]]),
        "token_types": torch.tensor([[0] * 9 + [3] * 2, [0] * 3 + [3] * 8]),
        "segment_ids": torch.tensor([[0] * 5 + [1] * 4 + [-1] * 2, [0] * 3 + [-1] * 8]),
        "sigma": torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 0], [0, 1, 2] + [0] * 8]),
    }
    with torch.no_grad():
        packed = model(batch, return_content=True)
        for row, start, end in [(0, 0, 5), (0, 5, 9), (1, 0, 3)]:
            local = {key: batch[key][row:row + 1, start:end] for key in ("input_ids", "token_types", "sigma")}
            independent = model(local, return_content=True)
            for key in packed:
                torch.testing.assert_close(packed[key][row:row + 1, start:end], independent[key],
                                           rtol=1e-4, atol=2e-6)
        for values in packed.values():
            assert torch.count_nonzero(values[batch["token_types"].eq(3)]) == 0


def test_removing_marked_cq_additions_recovers_native_source():
    """Keep the user's source-comparison contract enforceable as CQ evolves."""
    reference = Path(native.__file__).read_bytes()
    original_lines = []
    inside = False
    for line in Path(cq.__file__).read_bytes().splitlines(keepends=True):
        if line.strip().startswith(b"# CQ BEGIN:"):
            assert not inside, "CQ blocks must not be nested"
            inside = True
        elif line.strip() == b"# CQ END":
            assert inside, "CQ END must match CQ BEGIN"
            inside = False
        elif not inside:
            original_lines.append(line)
    assert not inside, "Unterminated CQ block"
    assert b"".join(original_lines) == reference


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_native_path_retains_prefill_and_incremental_cache(backend):
    torch.manual_seed(58)
    cfg = config(4)
    cfg._attn_implementation = backend
    reference = Qwen3_5TextModel(deepcopy(cfg)).eval()
    model = cq.Qwen3_5TextModel(deepcopy(cfg)).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.arange(1, 9)[None]
    with torch.no_grad():
        expected = reference(input_ids=ids[:, :5], use_cache=True, output_hidden_states=True)
        actual = model(input_ids=ids[:, :5], use_cache=True, output_hidden_states=True)
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
        assert len(actual.hidden_states) == len(expected.hidden_states)
        for a, b in zip(actual.hidden_states, expected.hidden_states):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for i in range(5, 8):
            actual = model(input_ids=ids[:, i:i + 1], past_key_values=actual.past_key_values, use_cache=True)
            expected = reference(input_ids=ids[:, i:i + 1], past_key_values=expected.past_key_values, use_cache=True)
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
        full = reference(input_ids=ids, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual.last_hidden_state, full[:, -1:], rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("length", [70, 129])
def test_content_matches_native_across_chunks(backend, length):
    torch.manual_seed(59)
    cfg = config(8)
    cfg._attn_implementation = backend
    reference = Qwen3_5TextModel(deepcopy(cfg)).eval()
    model = cq.Qwen35CQBackbone(cfg).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    source = torch.randn(1, length, cfg.hidden_size)
    native_input = source.clone().requires_grad_()
    cq_input = source.clone().requires_grad_()
    ids = torch.ones(1, length, dtype=torch.long)
    expected = reference(inputs_embeds=native_input, use_cache=False).last_hidden_state
    actual = model({"input_ids": ids, "inputs_embeds": cq_input}, return_content=True)["content_hidden"]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-6)
    probe = torch.randn_like(actual)
    native_grad, = torch.autograd.grad((expected * probe).sum(), native_input)
    cq_grad, = torch.autograd.grad((actual * probe).sum(), cq_input)
    torch.testing.assert_close(cq_grad, native_grad, rtol=1e-4, atol=3e-6)


def test_full_model_query_independence_and_target_visibility_across_chunks():
    torch.manual_seed(60)
    cfg = config(8)
    model = cq.Qwen3_5TextModel(cfg).eval()
    length = 129
    types = torch.zeros(1, length, dtype=torch.long)
    types[:, 60:100] = 1
    sigma = torch.arange(length)[None].float()
    sigma[:, 60:100] = 60
    batch = {"input_ids": torch.ones_like(types), "token_types": types, "sigma": sigma}
    layout, query_allowed = layout_for(batch, cfg)
    _, content_allowed = build_cq_masks(batch, cfg.boi_token_id)
    content = torch.randn(1, length, cfg.hidden_size, requires_grad=True)
    query = torch.randn_like(content, requires_grad=True)
    context = cq._CQContext(layout, query_allowed)

    def run(content, query):
        return model(inputs_embeds=content, query_hidden_states=query, cq_context=context,
                     attention_mask={"full_attention": content_allowed, "linear_attention": None},
                     use_cache=False)

    output = run(content, query)
    changed_query = query.detach().clone()
    changed_query[:, 65] *= 5
    with torch.no_grad():
        changed = run(content, changed_query)
    others = torch.arange(length) != 65
    torch.testing.assert_close(output.last_hidden_state, changed.last_hidden_state, rtol=0, atol=0)
    torch.testing.assert_close(output.query_hidden_state[:, others], changed.query_hidden_state[:, others], rtol=0, atol=0)
    assert not torch.allclose(output.query_hidden_state[:, 65], changed.query_hidden_state[:, 65])
    qgrad, = torch.autograd.grad(
        (output.query_hidden_state[:, 65] * torch.randn_like(output.query_hidden_state[:, 65])).sum(),
        query, retain_graph=True,
    )
    assert torch.count_nonzero(qgrad[:, others]) == 0
    assert qgrad[:, 65].abs().max() > 0

    changed_content = content.detach().clone()
    changed_content[:, 60:100] *= 5
    with torch.no_grad():
        changed = run(changed_content, query)
    torch.testing.assert_close(output.query_hidden_state[:, 60:100], changed.query_hidden_state[:, 60:100],
                               rtol=0, atol=0)
    cgrad, = torch.autograd.grad(
        (output.query_hidden_state[:, 60:100] * torch.randn_like(output.query_hidden_state[:, 60:100])).sum(), content,
    )
    assert cgrad[:, :60].abs().max() > 0
    assert torch.count_nonzero(cgrad[:, 100:]) == 0
    assert torch.count_nonzero(cgrad[:, 60:100]) == 0


def test_checkpoint_preserves_both_streams_and_parameter_gradients():
    torch.manual_seed(61)
    model = cq.Qwen35CQBackbone(config()).train()
    source = torch.randn(1, 9, 32)
    probes = {name: torch.randn_like(source) for name in ("query_hidden", "content_hidden")}
    snapshots = []
    for checkpointed in (False, True):
        model.gradient_checkpointing = checkpointed
        model.zero_grad(set_to_none=True)
        inputs = source.clone().requires_grad_()
        outputs = model({**visual_batch(), "inputs_embeds": inputs}, return_content=True)
        sum((outputs[key] * probes[key]).sum() for key in probes).backward()
        snapshots.append((
            {key: value.detach().clone() for key, value in outputs.items()}, inputs.grad.clone(),
            {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None},
        ))
    plain, checkpointed = snapshots
    assert plain[2].keys() == checkpointed[2].keys()
    for key in plain[0]:
        torch.testing.assert_close(plain[0][key], checkpointed[0][key], rtol=0, atol=0)
    torch.testing.assert_close(plain[1], checkpointed[1], rtol=1e-5, atol=2e-6)
    for name in plain[2]:
        torch.testing.assert_close(plain[2][name], checkpointed[2][name], rtol=1e-5, atol=2e-6)


def test_umm_adapter_loads_native_text_weights(tmp_path):
    torch.manual_seed(62)
    reference = Qwen3_5TextModel(config(4)).eval()
    reference.save_pretrained(tmp_path)
    model, info = cq.Qwen35CQBackbone.from_pretrained(tmp_path, output_loading_info=True)
    assert not any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert model.state_dict().keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("entry", ["argument", "config", "forward"])
@pytest.mark.parametrize("mode", ["after_update", "unknown"])
def test_cq_rejects_non_before_update_modes(entry, mode):
    cfg = config(4)
    if entry == "config":
        cfg.mask_query_read = mode
    if entry == "forward":
        model = cq.Qwen35CQBackbone(cfg).eval()
        model.config.mask_query_read = mode
    with pytest.raises(ValueError, match="requires mask_query_read='before_update'"):
        if entry == "forward":
            model({"input_ids": torch.ones(1, 2, dtype=torch.long)})
        else:
            cq.Qwen35CQBackbone(cfg, **({"mask_query_read": mode} if entry == "argument" else {}))


def test_query_visibility_cannot_be_overridden_to_include_targets():
    cfg = config(4)
    model = cq.Qwen35CQBackbone(cfg).eval()
    batch = {**visual_batch(), "inputs_embeds": torch.randn(1, 9, cfg.hidden_size)}
    _, content_allowed = build_cq_masks(batch, cfg.boi_token_id)
    with pytest.raises(ValueError, match="Explicit masks must match"):
        model(batch, masks=(content_allowed, content_allowed))
    with pytest.raises(ValueError, match="exclude the entire target group"):
        cq._gdn_layout(batch, content_allowed, content_allowed, cfg.linear_conv_kernel_dim)


def test_cq_attention_recording_retains_native_weights():
    torch.manual_seed(63)
    cfg = config()
    cfg._attn_implementation = "eager"
    model = cq.Qwen3_5TextModel(cfg).eval()
    batch = visual_batch()
    layout, query_allowed = layout_for(batch, cfg)
    _, content_allowed = build_cq_masks(batch, cfg.boi_token_id)
    content = torch.randn(1, 9, cfg.hidden_size)
    query = torch.randn_like(content)
    arguments = dict(
        inputs_embeds=content, cq_context=cq._CQContext(layout, query_allowed),
        attention_mask={"full_attention": content_allowed, "linear_attention": None},
        position_ids=batch["position_ids"], use_cache=False, output_attentions=True,
        output_hidden_states=True,
    )
    with torch.no_grad():
        output = model(**arguments, query_hidden_states=query)
        content_only = model(**arguments)
        changed = model(**arguments, query_hidden_states=query * 5)
    assert len(output.attentions) == cfg.layer_types.count("full_attention")
    assert len(output.hidden_states) == cfg.num_hidden_layers + 1
    for actual, expected, altered in zip(output.attentions, content_only.attentions, changed.attentions):
        assert actual.shape == (1, cfg.num_attention_heads, 9, 9)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(actual, altered, rtol=0, atol=0)
        assert torch.count_nonzero(actual.masked_select(~content_allowed[:, None])) == 0
        torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual.sum(-1)), rtol=1e-5, atol=1e-6)
    assert output.query_hidden_state.shape == content.shape
    assert all(hidden.shape == content.shape for hidden in output.hidden_states)


def lm_config(kind):
    cfg = config(4)
    if kind == "Qwen3_5ForConditionalGeneration":
        cfg = cq.Qwen3_5Config(
            text_config=cfg.to_dict(),
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "out_hidden_size": 32, "num_position_embeddings": 16},
            image_token_id=60, video_token_id=58, vision_start_token_id=61,
        )
    cfg._attn_implementation = "sdpa"
    return cfg


@pytest.mark.parametrize("kind", ["Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"])
@pytest.mark.parametrize("query_only", [False, True])
def test_native_lm_heads_reject_cq_entry(kind, query_only, monkeypatch):
    model = getattr(cq, kind)(lm_config(kind)).train()
    arguments = ({"query_hidden_states": torch.randn(1, 3, 32)} if query_only else
                 {"cq_context": cq._CQContext(None, torch.zeros(1, 3, 3, dtype=torch.bool))})

    def unexpected_forward(*args, **kwargs):
        pytest.fail("CQ must be rejected before running the native content/loss path")

    monkeypatch.setattr(model.model, "forward", unexpected_forward)
    ids = torch.tensor([[3, 4, 5]])
    with pytest.raises(ValueError, match="Qwen35CQBackbone with UMM query heads"):
        model(input_ids=ids, labels=ids, use_cache=False, **arguments)
    model.eval()
    with pytest.raises(ValueError, match="Qwen35CQBackbone with UMM query heads"):
        model(input_ids=ids, use_cache=False, **arguments)


@pytest.mark.parametrize("kind", ["Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"])
def test_native_lm_logits_and_shifted_loss_are_unchanged(kind):
    torch.manual_seed(64)
    cfg = lm_config(kind)
    reference = getattr(native, kind)(deepcopy(cfg)).eval()
    model = getattr(cq, kind)(deepcopy(cfg)).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    with torch.no_grad():
        expected = reference(input_ids=ids, labels=ids, use_cache=False)
        actual = model(input_ids=ids, labels=ids, use_cache=False)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    shifted_loss = torch.nn.functional.cross_entropy(actual.logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
    torch.testing.assert_close(actual.loss, shifted_loss, rtol=0, atol=0)


def test_raw_text_model_rejects_orphan_or_misaligned_queries():
    cfg = config(4)
    model = cq.Qwen3_5TextModel(cfg)
    content = torch.randn(1, 3, cfg.hidden_size)
    with pytest.raises(ValueError, match="requires cq_context"):
        model(inputs_embeds=content, query_hidden_states=content, use_cache=False)
    with pytest.raises(ValueError, match="must match content shape"):
        model(inputs_embeds=content, query_hidden_states=content[:, :2], use_cache=False,
              cq_context=cq._CQContext(None, torch.zeros(1, 3, 3, dtype=torch.bool)))




@pytest.mark.parametrize("tail", [(), (5,), (2, 4)])
def test_cq_row_permutation_forward_backward_and_batch_isolation(tail):
    torch.manual_seed(107)
    batch, length = 3, 11
    # Exercise non-contiguous source and incoming gradients as used by the scan.
    source = torch.randn(length, batch, *tail, dtype=torch.float64).transpose(0, 1).requires_grad_()
    indices = torch.stack([torch.randperm(length) for _ in range(batch)])
    actual = cq._cq_gather(source, indices)
    expected = torch.stack([torch.stack([source[row, int(i)] for i in indices[row]]) for row in range(batch)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn(length, batch, *tail, dtype=source.dtype).transpose(0, 1)
    gradient, = torch.autograd.grad(actual, source, probe)
    expected_gradient = torch.empty_like(source)
    for row in range(batch):
        for output_position, input_position in enumerate(indices[row].tolist()):
            expected_gradient[row, input_position] = probe[row, output_position]
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    restored = cq._cq_gather(actual, indices.argsort(dim=1))
    torch.testing.assert_close(restored, source, rtol=0, atol=0)


def test_backend_configuration_preserves_parameters_and_audits_layers():
    model = cq.Qwen35CQBackbone(config(4))
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.configure_gdn_backend("mindspeed")
    assert model.config.to_dict()["qwen_gdn_backend"] == "mindspeed"
    assert all(layer.linear_attn.qwen_gdn_backend == "mindspeed"
               for layer in model.layers if hasattr(layer, "linear_attn"))
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
    model.configure_gdn_backend("native")
    identity = model.gdn_backend_identity()
    assert identity["backend"] == "native"
    assert identity["function"] == "torch_chunk_gated_delta_rule"
    assert identity["cq_index_implementation"] == "batched_row_index_select_v1"
    model.layers[0].linear_attn.qwen_gdn_backend = "mindspeed"
    with pytest.raises(RuntimeError, match="disagrees"):
        model.gdn_backend_identity()
    with pytest.raises(ValueError, match="qwen_gdn_backend"):
        model.configure_gdn_backend("auto")


def test_explicit_backend_dispatch_is_local_to_cq(monkeypatch):
    model = cq.Qwen35CQBackbone(config(4)).eval()
    calls = []
    native_scan = cq._cq_select_chunk_gdn("native")

    def record(*args, **kwargs):
        calls.append(args[0].shape)
        return native_scan(*args, **kwargs)

    original_symbol = cq.torch_chunk_gated_delta_rule
    monkeypatch.setattr(cq, "_cq_mindspeed_chunk_gated_delta_rule", record)
    model.configure_gdn_backend("mindspeed")
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        model({"input_ids": ids})
    assert len(calls) == 3
    assert all(shape[1] == 6 for shape in calls)
    assert cq.torch_chunk_gated_delta_rule is original_symbol
    # Even a layer configured for CQ MindSpeed retains its native non-CQ path.
    with torch.no_grad():
        model.layers[0].linear_attn(torch.randn(1, 3, model.config.hidden_size))
    assert len(calls) == 3


def test_mindspeed_backend_does_not_silently_fall_back_on_cpu():
    model = cq.Qwen35CQBackbone(config(4)).eval()
    model.configure_gdn_backend("mindspeed")
    with pytest.raises(ValueError, match="requires NPU tensors"):
        model({"input_ids": torch.tensor([[3, 4, 5]])})
