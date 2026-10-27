"""Opt-in CQ integration on the explicitly selected MindSpeed NPU backend.

These exercise the actual upstream kernel after MCore patch registration, not
the CPU fallback. Use tests.validation.validate_cq_mcore for the four-rank check.
"""

from copy import deepcopy
import os

import pytest
import torch

from models import backbone_qwen35 as cq
from models.modeling_umm import UMM
from models.references import modeling_qwen35 as native
from tests.test_modeling_umm_video import configuration, video_batch


@pytest.fixture
def network(monkeypatch):
    requested = os.environ.get("UMM_VIDEO_DEVICE", "")
    if not requested.startswith("npu"):
        pytest.skip("Set UMM_VIDEO_DEVICE=npu:<rank> to execute the MindSpeed kernel")
    import torch_npu  # noqa: F401

    torch.npu.set_device(torch.device(requested))
    torch.manual_seed(721)
    cfg = configuration(layers=4, freeze=False)
    # Real Qwen3.5 GDN head width; attention/ViT/Wan remain small for isolation.
    cfg.qwen_config.text_config.linear_key_head_dim = 128
    cfg.qwen_config.text_config.linear_value_head_dim = 128
    cfg.qwen_config.text_config.qwen_gdn_backend = "mindspeed"
    cfg.wan_config["attention_backend"] = "npu"
    model = UMM(cfg).to(device=requested, dtype=torch.bfloat16).eval()
    model.backbone.configure_gdn_backend("mindspeed")
    with torch.no_grad():
        model.generator.head.head.weight.normal_(std=0.05)
    calls = []
    kernel = cq._cq_mindspeed_chunk_gated_delta_rule

    def audited(*args, **kwargs):
        result = kernel(*args, **kwargs)
        calls.append(1)
        return result

    monkeypatch.setattr(cq, "_cq_mindspeed_chunk_gated_delta_rule", audited)
    with torch.autocast("npu", dtype=torch.bfloat16):
        yield model
    assert calls, "Test did not execute the MindSpeed GDN kernel"


def test_content_matches_native_under_explicit_backend(network):
    reference = native.Qwen3_5Model(deepcopy(network.config.qwen_config)).to(
        device=next(network.parameters()).device, dtype=torch.bfloat16
    ).eval()
    reference.visual.load_state_dict(network.vision_encoder.visual.state_dict())
    reference.language_model.load_state_dict({
        k: v for k, v in network.backbone.state_dict().items()
        if not k.startswith("vision_encoder.")
    })
    batch = video_batch(next(network.parameters()).device)
    with torch.no_grad():
        actual = network(batch, return_hidden_states=True)
        expected = reference(**{k: batch[k] for k in (
            "input_ids", "attention_mask", "mm_token_type_ids",
            "video_grid_thw", "pixel_values_videos",
        )}, use_cache=False)
    # The upstream kernel rounds intermediate tensors to BF16, whereas the
    # reference scan accumulates FP32. This is arithmetic parity, not bit parity.
    torch.testing.assert_close(
        actual["content_hidden"], expected.last_hidden_state, rtol=0.04, atol=0.02
    )


@pytest.mark.parametrize("checkpointed", [False, True])
def test_video_no_leak_and_lm_wan_alignment(network, checkpointed):
    device = next(network.parameters()).device
    network.train()
    network.backbone.gradient_checkpointing = checkpointed
    batch = video_batch(device, requires_grad=True)
    noisy = [torch.randn(4, 3, 4, 6, device=device)]
    timesteps = torch.tensor([700.0], device=device)
    output = network(batch, noisy, timesteps, return_hidden_states=True)
    prepared = network.prepare_inputs(batch)
    slots = prepared["video_token_indices"][0]
    assert output["query_hidden"].shape == output["content_hidden"].shape
    torch.testing.assert_close(
        output["condition"][0], output["query_hidden"][0, slots], rtol=0, atol=0
    )
    valid = batch["labels"].ne(-100)
    torch.testing.assert_close(
        output["text_logits"], network.lm_head(output["query_hidden"][valid]), rtol=0, atol=0
    )
    torch.testing.assert_close(output["text_labels"], batch["labels"][valid])
    assert output["velocity"][0].shape == noisy[0].shape
    loss = (output["velocity"][0] * torch.randn_like(output["velocity"][0])).sum()
    pixels, embeddings, bridge = torch.autograd.grad(
        loss, (batch["pixel_values_videos"], network.backbone.embed_tokens.weight,
               network.generator.condition_proj.weight), retain_graph=True,
    )
    assert torch.count_nonzero(pixels) == 0
    assert embeddings[network.config.video_mask_token_id].abs().max() > 0
    assert bridge.abs().max() > 0 and torch.isfinite(bridge).all()
    text_loss = (output["text_logits"] * torch.randn_like(output["text_logits"])).sum()
    vision, = torch.autograd.grad(text_loss, network.vision_encoder.visual.patch_embed.proj.weight)
    assert vision.abs().max() > 0 and torch.isfinite(vision).all()
    network.eval()
    changed = {**batch, "pixel_values_videos": torch.randn_like(batch["pixel_values_videos"]) * 5}
    unobserved = {k: v for k, v in batch.items() if k != "pixel_values_videos"}
    unobserved["video_observed_mask"] = torch.tensor([False], device=device)
    with torch.no_grad():
        for alternative in (changed, unobserved):
            generated = network(alternative, noisy, timesteps, return_hidden_states=True)
            torch.testing.assert_close(generated["condition"], output["condition"], rtol=0, atol=0)
            torch.testing.assert_close(generated["velocity"][0], output["velocity"][0], rtol=0, atol=0)


def test_all_post_prefix_content_is_excluded_and_prefix_receives_gradients(network):
    device = next(network.parameters()).device
    prepared = network.prepare_inputs(video_batch(device))
    slots = prepared["video_token_indices"][0]
    boundary = int(slots[0])
    source = prepared["inputs_embeds"].detach().requires_grad_()
    original = network.backbone({**prepared, "inputs_embeds": source})["query_hidden"]
    grad, = torch.autograd.grad(
        (original[:, slots] * torch.randn_like(original[:, slots])).sum(), source
    )
    assert torch.count_nonzero(grad[:, boundary:]) == 0
    assert grad[:, :boundary].abs().max() > 0
    changed = source.detach().clone()
    changed[:, boundary:] = torch.randn_like(changed[:, boundary:]) * 5
    with torch.no_grad():
        alternative = network.backbone({**prepared, "inputs_embeds": changed})["query_hidden"]
    torch.testing.assert_close(original[:, slots], alternative[:, slots], rtol=0, atol=0)


def test_query_independence_across_gdn_chunk_boundaries(network):
    device = next(network.parameters()).device
    cfg = network.backbone.config
    length = 129
    types = torch.zeros(1, length, dtype=torch.long, device=device)
    types[:, 60:100] = 1
    batch = {"input_ids": torch.ones_like(types), "token_types": types}
    query_allowed, content_allowed = cq.build_cq_masks(batch, cfg.boi_token_id)
    layout = cq._gdn_layout(batch, query_allowed, content_allowed, cfg.linear_conv_kernel_dim)
    context = cq._CQContext(layout, query_allowed)
    content = torch.randn(1, length, cfg.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    query = torch.randn_like(content, requires_grad=True)

    def run(values):
        return cq.Qwen3_5TextModel.forward(
            network.backbone, inputs_embeds=content, query_hidden_states=values, cq_context=context,
            attention_mask={"full_attention": content_allowed, "linear_attention": None}, use_cache=False,
        )

    output = run(query)
    changed = query.detach().clone()
    changed[:, 65] *= 5
    with torch.no_grad():
        other = run(changed)
    others = torch.arange(length, device=device) != 65
    torch.testing.assert_close(output.last_hidden_state, other.last_hidden_state, rtol=0, atol=0)
    torch.testing.assert_close(output.query_hidden_state[:, others], other.query_hidden_state[:, others], rtol=0, atol=0)
    assert not torch.equal(output.query_hidden_state[:, 65], other.query_hidden_state[:, 65])
    gradient, = torch.autograd.grad(
        (output.query_hidden_state[:, 65] * torch.randn_like(output.query_hidden_state[:, 65])).sum(), query
    )
    assert torch.count_nonzero(gradient[:, others]) == 0
    assert gradient[:, 65].abs().max() > 0


def test_text_generation_matches_same_position_training(network):
    device = next(network.parameters()).device
    prefix = video_batch(device)
    decoded = network.generate_text(prefix, max_new_tokens=3, eos_token_id=-1, return_logits=True)
    length = prefix["input_ids"].shape[1]
    full = torch.cat((prefix["input_ids"], decoded["sequences"]), 1)
    labels = full.clone()
    labels[:, :length] = -100
    teacher = {k: v for k, v in prefix.items() if k in ("pixel_values_videos", "video_grid_thw")}
    teacher.update(input_ids=full, labels=labels)
    network.train()
    output = network(teacher)
    torch.testing.assert_close(output["text_labels"], decoded["sequences"].flatten())
    torch.testing.assert_close(
        output["text_logits"], decoded["logits"].flatten(0, 1), rtol=0.02, atol=0.003
    )
