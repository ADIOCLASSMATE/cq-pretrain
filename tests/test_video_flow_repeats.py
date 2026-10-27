"""Repeated flow draws reuse CQ while preserving loss weighting and gradients."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from configs.config import DEFAULTS, load_config, validate_config
from data.dataset import VideoDataset
from data.packing import collate_video_samples
from training.loss import forward_loss, prepare_batch
from tests.video_fixtures import make_fixture


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(2)
    return make_fixture(tmp_path)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_flow_repeats_requires_positive_integer(value):
    config = deepcopy(DEFAULTS)
    config["flow"]["repeats"] = value
    with pytest.raises(ValueError, match="flow.repeats must be a positive integer"):
        validate_config(config)


def test_flow_repeats_defaults_and_cli_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    assert load_config(path)["flow"]["repeats"] == 1
    assert load_config(path, ["flow.repeats=4"])["flow"]["repeats"] == 4


def test_repeated_flow_matches_independent_draws_and_gradients(fixture):
    config, processor, model, _, _ = fixture
    config["flow"]["repeats"] = repeats = 3
    data = VideoDataset(config["data"]["sources"][0], processor, config)
    # Two targets separated by a V2T row; differing latent sizes test both
    # target ordering and averaging by videos instead of by latent elements.
    samples = [data[0], data[2], data[1]]
    samples[2]["wan_latent"] = samples[2]["wan_latent"].repeat(1, 1, 1, 2) + 0.5
    batch = collate_video_samples(samples)
    batch["video_uncond_mask"] = torch.tensor([False, False, True])
    batch = prepare_batch(batch, "cpu", config)
    original_flags = batch["video_uncond_mask"].clone()
    model.train()
    reference = deepcopy(model)
    captured, calls = {}, dict(backbone=0, lm_head=0, wan=0)

    def backbone_hook(module, args, output):
        calls["backbone"] += 1
        captured["prepared"], captured["streams"] = args[0], output

    def lm_hook(module, args, output):
        calls["lm_head"] += 1

    def wan_hook(module, args, output):
        calls["wan"] += 1
        captured["noisy"] = [x.detach().clone() for x in args[0]]
        captured["times"] = args[1].detach().clone()

    def output_hook(module, args, output):
        captured["output"] = output

    handles = [
        model.backbone.register_forward_hook(backbone_hook),
        model.lm_head.register_forward_hook(lm_hook),
        model.generator.register_forward_hook(wan_hook),
        model.register_forward_hook(output_hook),
    ]
    try:
        actual = forward_loss(model, batch, config, torch.Generator().manual_seed(19))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == dict(backbone=1, lm_head=1, wan=1)
    assert actual["video_target_count"] == 2
    assert actual["text_target_count"] == int(batch["labels"].ne(-100).sum())
    torch.testing.assert_close(batch["video_uncond_mask"], original_flags, rtol=0, atol=0)
    assert captured["prepared"]["input_ids"].shape == batch["input_ids"].shape
    output = captured["output"]
    assert output["target_video_ids"].tolist() == [0, 2] * repeats
    assert len(captured["noisy"]) == len(output["velocity"]) == repeats * 2
    prepared, streams = captured["prepared"], captured["streams"]
    for index, video_id in enumerate([0, 2] * repeats):
        row = prepared["video_batch_indices"][video_id]
        slots = prepared["video_token_indices"][video_id]
        torch.testing.assert_close(
            output["condition"][index, :len(slots)],
            streams["query_hidden"][row, slots],
            rtol=0,
            atol=0,
        )

    # Recover the independent Gaussian draws from the inputs given to Wan.
    # This also catches swapped video order or a mismatched timestep scale.
    sigmas = captured["times"] / config["flow"]["timestep_scale"]
    targets = [
        (noisy - batch["wan_latents"][i % 2]) / sigmas[i]
        for i, noisy in enumerate(captured["noisy"])
    ]
    for video_id in range(2):
        assert torch.unique(sigmas[video_id::2]).numel() == repeats
        assert not torch.equal(targets[video_id], targets[video_id + 2])

    # Oracle: run the same draws separately through the unexpanded model.
    # CE is included once and each video/draw contributes equal flow weight.
    video_losses = []
    for draw in range(repeats):
        start, end = draw * 2, (draw + 1) * 2
        expected = reference(
            batch,
            captured["noisy"][start:end],
            captured["times"][start:end],
            compute_text=draw == 0,
        )
        if draw == 0:
            text_loss = F.cross_entropy(expected["text_logits"].float(), expected["text_labels"])
        for i, prediction in enumerate(expected["velocity"], start):
            torch.testing.assert_close(prediction, output["velocity"][i], rtol=1e-5, atol=1e-6)
            video_losses.append(F.mse_loss(prediction.float(), targets[i]))
    video_loss = torch.stack(video_losses).mean()
    expected_loss = config["model"]["lambda_text"] * text_loss + config["model"]["lambda_video"] * video_loss
    torch.testing.assert_close(actual["text_loss"], text_loss.detach(), rtol=0, atol=0)
    torch.testing.assert_close(actual["video_loss"], video_loss.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual["loss"], expected_loss, rtol=1e-5, atol=1e-6)
    actual["loss"].backward()
    expected_loss.backward()
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-4, atol=2e-6, msg=name)


def test_v2t_is_unaffected_by_flow_repeats(fixture):
    config, processor, model, _, _ = fixture
    data = VideoDataset(config["data"]["sources"][0], processor, config)
    batch = prepare_batch(collate_video_samples([data[2], data[3]]), "cpu", config)
    rng = torch.Generator().manual_seed(23)
    state = rng.get_state()
    with torch.no_grad():
        once = forward_loss(model, batch, config, rng)
        config["flow"]["repeats"] = 4
        repeated = forward_loss(model, batch, config, rng)
    assert once["video_target_count"] == repeated["video_target_count"] == 0
    assert once["text_target_count"] == repeated["text_target_count"]
    torch.testing.assert_close(once["loss"], repeated["loss"], rtol=0, atol=0)
    torch.testing.assert_close(rng.get_state(), state, rtol=0, atol=0)


@pytest.mark.parametrize("legacy_input", ["mixed", "pixel_video_ids"])
def test_model_entry_rejects_legacy_mixed_inputs(fixture, legacy_input):
    config, processor, model, _, _ = fixture
    source = config["data"]["sources"][0]
    cached = VideoDataset(source, processor, config)[0]
    config["data"]["representations"]["qwen"] = "online"
    online = VideoDataset(source, processor, config)[0]
    batch = collate_video_samples([online])
    if legacy_input == "mixed":
        batch["video_features"] = [cached["video_features"]]
        message = "whole batch, not both"
    else:
        batch["pixel_video_ids"] = torch.tensor([0])
        message = "pixel_video_ids is unsupported"
    with pytest.raises(ValueError, match=message):
        model.prepare_inputs(batch)
    with pytest.raises(ValueError, match=message):
        model.generate_video(batch)
