"""Nontrivial numerical comparisons with the pinned source arithmetic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.modeling_umm import UMM, UMMConfig
from models.backbone import CQDecoderLayer, RotaryEmbedding
from models.generator import Z2Generator
from models.position import build_row_col_position_ids
from models.text_generation import generate_text
from models.sampling import integrate_flow, sample_visual_latents
from utils.attention_mask import build_cq_masks

spec = importlib.util.spec_from_file_location("pinned_model", Path(__file__).parent / "reference" / "pinned_model.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def config(**updates):
    values = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                  mask_token_id=26, image_mask_token_id=27, boi_token_id=28, eoi_token_id=29,
                  eos_token_id=30, pad_token_id=31, image_latent_dim=4, image_tokens_per_img=4,
                  image_flow_width=16, image_flow_depth=2, joint_dit_head_dim=8,
                  joint_dit_intermediate=64, image_input_noise_strength=0,
                  image_flow_grad_checkpointing=False, gradient_checkpointing=False)
    values.update(updates)
    return UMMConfig(**values)


def batch():
    return dict(input_ids=torch.tensor([[2, 3, 28, 27, 27, 27, 27, 29, 4]]),
                token_types=torch.tensor([[0, 0, 2, 1, 1, 1, 1, 2, 0]]),
                image_latents=torch.randn(1, 9, 4),
                image_latent_mask=torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
                segment_ids=torch.zeros(1, 9, dtype=torch.long),
                image_span_table=torch.tensor([[0, 0, 3, 7]]), task="t2i")


def activate_head(head):
    # Zero-initialized output would make a false equivalence test; activate all
    # modulation branches and the final projection before comparing arithmetic.
    with torch.no_grad():
        for name, parameter in head.named_parameters():
            if "adaLN_modulation" in name or "final_layer.linear" in name:
                parameter.normal_(0, .08)


def test_source_qwen_layer_rope_outputs_and_gradients():
    torch.manual_seed(11)
    cfg = config()
    original_cfg = SimpleNamespace(**cfg.to_dict(), layer_types=["full_attention"] * 2,
                                   attention_dropout=0, max_position_embeddings=128)
    old = reference.Qwen3DecoderLayer(original_cfg, 0)
    new = CQDecoderLayer(cfg, 0)
    new.load_state_dict(old.state_dict(), strict=True)
    content = torch.randn(1, 9, cfg.hidden_size, requires_grad=True)
    query = torch.randn_like(content, requires_grad=True)
    position = build_row_col_position_ids(batch()["token_types"], 4)
    torch.testing.assert_close(position, reference.build_row_col_position_ids(batch()["token_types"], 4))
    rope = RotaryEmbedding(cfg)(content, position)
    original_rope = reference.Qwen3RotaryEmbedding(original_cfg)(content, position)
    for a, b in zip(rope, original_rope):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    query_mask, content_mask = build_cq_masks(batch(), cfg.boi_token_id)
    actual = new(content, query, rope, query_mask, content_mask)
    expected = old(content, query, query_mask, content_attention_mask=content_mask, position_embeddings=rope)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=3e-7, rtol=2e-6)
    actual_grads = torch.autograd.grad(sum(x.square().sum() for x in actual), (content, query, *new.parameters()))
    expected_grads = torch.autograd.grad(sum(x.square().sum() for x in expected), (content, query, *old.parameters()))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=8e-6, rtol=5e-5)


def test_source_generator_output_gradient_and_initialization():
    cfg = config()
    torch.manual_seed(12)
    expected = reference.Z2FlowHead(cfg)
    torch.manual_seed(12)
    actual = Z2Generator(cfg)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
    activate_head(expected)
    actual.load_state_dict(expected.state_dict(), strict=True)
    x = torch.randn(2, 4, 4, requires_grad=True)
    c = torch.randn(2, 4, 16, requires_grad=True)
    t = torch.tensor([.2, .8])
    prediction = actual(x, t, c)
    target = expected(x, t, c)
    assert prediction.abs().max() > .01
    torch.testing.assert_close(prediction, target, rtol=0, atol=0)
    a = torch.autograd.grad(prediction.square().sum(), (x, c, *actual.parameters()))
    b = torch.autograd.grad(target.square().sum(), (x, c, *expected.parameters()))
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("solver", ["euler", "heun"])
@pytest.mark.parametrize("cfg_scale,schedule", [(1., "constant"), (3.5, "constant"), (3.5, "linear")])
def test_complete_ode_matches_source(solver, cfg_scale, schedule):
    torch.manual_seed(4)
    head = Z2Generator(config()).eval()
    activate_head(head)
    condition = torch.randn(1 if cfg_scale == 1 else 2, 4, 16)
    noise = torch.randn(1, 4, 4)
    old = type("Sampler", (reference.OriginalSampler, reference.OriginalGuidance), {})()
    old.net, old.image_tokens_per_img, old.in_channels = head, 4, 4
    old.num_sampling_steps, old.solver = 3, solver
    old.velocity = head
    expected = old.sample(condition, cfg=cfg_scale, cfg_schedule=schedule, initial_noise=noise)
    actual = integrate_flow(head, condition, steps=3, solver=solver, cfg=cfg_scale,
                            cfg_schedule=schedule, initial_noise=noise)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_condition_has_no_target_leak_and_uses_text():
    torch.manual_seed(42)
    model = UMM(config()).eval()
    original = batch()
    condition = model.encode_condition(original)
    changed = dict(original, image_latents=torch.randn_like(original["image_latents"]) * 100)
    torch.testing.assert_close(condition, model.encode_condition(changed), rtol=0, atol=0)
    changed = dict(original, input_ids=original["input_ids"].clone())
    changed["input_ids"][0, 0] = 9
    assert not torch.allclose(condition, model.encode_condition(changed))
    assert model.lm_head.weight is model.backbone.embed_tokens.weight


def test_flow_forward_repeats_condition_once_and_backpropagates():
    model = UMM(config()).train()
    activate_head(model.generator)
    calls = []
    handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    output = model(batch(), noisy_latents=torch.randn(3, 4, 4), timesteps=torch.tensor([.2, .5, .8]))
    handle.remove()
    assert len(calls) == 1
    assert output["velocity"].shape == (3, 4, 4)
    for row in range(1, 3):
        torch.testing.assert_close(output["condition"][0], output["condition"][row], rtol=0, atol=0)
    output["velocity"].square().mean().backward()
    assert model.backbone.layers[0].self_attn.k_proj.weight.grad.abs().sum() > 0


def test_sampler_cfg_single_backbone_call_and_mode_restore():
    model = UMM(config()).train()
    activate_head(model.generator)
    calls = []
    handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    latent, trace = sample_visual_latents(model, batch(), steps=3, solver="heun", cfg=3.5,
                                          initial_noise=torch.randn(1, 4, 4), return_trace=True)
    handle.remove()
    assert latent.shape == (1, 4, 4)
    assert trace["flow_head_calls"] == 6 and len(calls) == 1
    assert model.training


def test_text_same_position_special_labels_and_cache_equivalence():
    torch.manual_seed(6)
    model = UMM(config()).eval()
    inputs = batch()
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["labels"][inputs["token_types"].eq(1)] = -100
    output = model(inputs)
    assert output["text_labels"].tolist() == [2, 3, 28, 29, 4]
    cached = generate_text(model, inputs, max_new_tokens=5, use_cache=True, eos_token_id=-1)
    complete = generate_text(model, inputs, max_new_tokens=5, use_cache=False, eos_token_id=-1)
    torch.testing.assert_close(cached, complete, rtol=0, atol=0)


def test_backbone_checkpointing_matches_eager_gradients():
    torch.manual_seed(2)
    eager = UMM(config()).train()
    checkpointed = UMM(config(gradient_checkpointing=True, image_flow_grad_checkpointing=True)).train()
    activate_head(eager.generator)
    checkpointed.load_state_dict(eager.state_dict())
    inputs = batch()
    noise = torch.randn(1, 4, 4)
    times = torch.tensor([.3])
    expected = eager(inputs, noise, times)["velocity"]
    actual = checkpointed(inputs, noise, times)["velocity"]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    expected.sum().backward()
    actual.sum().backward()
    for (name, a), (_, b) in zip(eager.named_parameters(), checkpointed.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_complete_source_backbone_condition_logits_and_gradients():
    torch.manual_seed(13)
    cfg = config()
    original_cfg = SimpleNamespace(**cfg.to_dict(), layer_types=["full_attention"] * 2,
                                   attention_dropout=0, max_position_embeddings=128)
    model = UMM(cfg).eval()
    original = reference.Qwen3Model(original_cfg).eval()
    original.load_state_dict(model.backbone.state_dict(), strict=True)
    inputs = batch()
    query_mask, content_mask = build_cq_masks(inputs, cfg.boi_token_id)
    expected = original(
        X0_input_ids=inputs["input_ids"], token_types=inputs["token_types"],
        image_latents=inputs["image_latents"], image_latent_mask=inputs["image_latent_mask"],
        attention_mask=~query_mask[:, None], content_attention_mask=~content_mask[:, None],
        image_span_table=inputs["image_span_table"], calculate_likelihood=True,
        return_x0_hidden_state=True,
    )
    actual = model.backbone(inputs, return_content=True)
    torch.testing.assert_close(actual["query_hidden"], expected.last_hidden_state, atol=6e-7, rtol=5e-6)
    torch.testing.assert_close(actual["content_hidden"], expected["x0_last_hidden_state"], atol=6e-7, rtol=5e-6)
    expected_condition = model.condition_projector(expected.last_hidden_state[:, 3:7])
    actual_condition = model.encode_condition(inputs)
    torch.testing.assert_close(actual_condition, expected_condition, atol=1e-7, rtol=5e-6)
    torch.testing.assert_close(model(inputs)["text_logits"], model.lm_head(expected.last_hidden_state), atol=1e-7, rtol=5e-6)
    original_params = list(original.parameters())
    new_params = list(model.backbone.parameters())
    first = torch.autograd.grad(expected_condition.square().sum(), original_params, allow_unused=True)
    second = torch.autograd.grad(actual_condition.square().sum(), new_params, allow_unused=True)
    for a, b in zip(first, second):
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-6, rtol=8e-5)


def test_text_cache_logits_match_full_sequence_with_padded_multimodal_prefix():
    torch.manual_seed(23)
    model = UMM(config()).eval()
    prompt = batch()
    prompt["token_types"][0, -1] = 3
    prompt["segment_ids"][0, -1] = -1
    captured = []
    handle = model.lm_head.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
    generate_text(model, prompt, max_new_tokens=5, eos_token_id=-1, use_cache=True)
    cached = captured[:]
    captured.clear()
    generate_text(model, prompt, max_new_tokens=5, eos_token_id=-1, use_cache=False)
    handle.remove()
    assert len(cached) == len(captured) == 5
    for actual, expected in zip(cached, captured):
        torch.testing.assert_close(actual, expected, atol=5e-8, rtol=2e-5)


@pytest.mark.parametrize("use_cache", [False, True])
def test_text_inference_uses_clean_prefix_after_training_preparation(use_cache):
    from training.loss import prepare_batch
    model = UMM(config(image_input_noise_strength=.1)).train()
    original = batch()
    prepared = prepare_batch(original, "cpu", {"model": {"image_uncond_prob": 1., "image_input_noise_strength": .1}})
    prepared["content_image_latents"] += 100
    prepared["image_uncond_mask"] = prepared["token_types"].eq(1)
    captured = []
    handle = model.lm_head.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
    expected = generate_text(model, original, max_new_tokens=3, eos_token_id=-1, use_cache=use_cache)
    clean_logits = captured[:]
    captured.clear()
    actual = generate_text(model, prepared, max_new_tokens=3, eos_token_id=-1, use_cache=use_cache)
    handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for clean, trained in zip(clean_logits, captured):
        torch.testing.assert_close(clean, trained, rtol=0, atol=0)
    assert "content_image_latents" in prepared and "image_uncond_mask" in prepared


@pytest.mark.parametrize("operation", ["image", "text"])
@pytest.mark.parametrize("failure", [False, True])
def test_inference_restores_each_module_mode_on_success_and_error(operation, failure):
    model = UMM(config()).train()
    model.backbone.eval()
    model.backbone.layers[0].train()
    model.generator.blocks[0].eval()
    before = [module.training for module in model.modules()]
    inputs = batch()
    if failure:
        if operation == "image":
            inputs["image_span_table"][0, 3] -= 1
        else:
            inputs["image_latents"] = inputs["image_latents"][:, :-1]
    def generate():
        if operation == "image":
            return sample_visual_latents(model, inputs, initial_noise=torch.zeros(1, 4, 4), steps=1)
        return generate_text(model, inputs, max_new_tokens=1)
    if failure:
        with pytest.raises(ValueError):
            generate()
    else:
        generate()
    assert [module.training for module in model.modules()] == before
