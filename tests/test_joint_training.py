"""Joint supervision must share a CQ pass and preserve both source objectives."""
from copy import deepcopy

import pytest
import torch

from data.dataloader import build_dataloaders, build_validation_loader
from evaluations.validation import validate_losses
from models.modeling_umm import UMM
from training.loss import forward_loss
from test_data import Tokenizer, assert_batch_equal, config as data_config
from test_model_core import activate_head, config as tiny_config
from test_training import manual_loss, settings, tasks


def joint_batch():
    batch = tasks()[1]
    batch["task"] = "joint"
    batch["labels"][0, -2:] = batch["input_ids"][0, -2:]
    return batch


def test_same_sequence_ce_and_flow_share_one_backbone_pass_and_match_source_gradients(tmp_path):
    torch.manual_seed(7)
    cfg = settings(tmp_path)
    model = UMM(tiny_config())
    activate_head(model.generator)
    reference = deepcopy(model)
    batch = joint_batch()
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    result = forward_loss(model, batch, cfg, torch.Generator().manual_seed(17))
    hook.remove()
    assert len(calls) == 1
    expected_text = manual_loss(reference, dict(batch, task="i2t"), cfg["model"], None)
    expected_flow = manual_loss(reference, dict(batch, task="t2i"), cfg["model"], torch.Generator().manual_seed(17))
    torch.testing.assert_close(result["loss"], expected_text + expected_flow, rtol=0, atol=0)
    assert result["text_target_count"] == 2 and result["image_target_count"] == 6
    result["loss"].backward()
    (expected_text + expected_flow).backward()
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert actual.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-5, msg=name)
    assert model.generator.input_proj.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("modality", ["text", "image"])
@pytest.mark.parametrize("explicit_switch", [False, True])
def test_joint_batch_with_one_active_modality_is_finite(tmp_path, modality, explicit_switch):
    cfg = settings(tmp_path)
    model = UMM(tiny_config())
    batch = joint_batch()
    if explicit_switch:
        batch.update(compute_text_loss=modality == "text", compute_image_loss=modality == "image")
    elif modality == "text":
        batch["image_loss_mask"].zero_()
    else:
        batch["labels"].fill_(-100)
    result = forward_loss(model, batch, cfg, torch.Generator().manual_seed(4))
    inactive = "image" if modality == "text" else "text"
    assert result[f"{modality}_target_count"] > 0
    assert result[f"{inactive}_target_count"] == 0 and result[f"{inactive}_loss"] == 0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()


@pytest.mark.parametrize("packed", [False, True])
def test_joint_loader_mixed_masks_validation_and_consumed_resume(tmp_path, packed):
    cfg = data_config(tmp_path)
    cfg["data"]["schedule"] = ["joint"]
    cfg["data"]["sources"]["joint"] = dict(micro_batch_size=4)
    cfg["data"]["image"]["packing"] = dict(enabled=packed, nominal_capacity=64)
    train, validation = build_dataloaders(cfg, Tokenizer())
    assert train.image_loaders["joint"].dataset.caption_sequence_modes == ("t2i", "i2t")
    iterator = iter(train)
    first = next(iterator)
    assert first["task"] == "joint"
    assert first["image_loss_mask"].any() and first["labels"].ne(-100).any()
    train.mark_consumed(first)
    state = train.state_dict()
    expected = next(iterator)
    restored, _ = build_dataloaders(cfg, Tokenizer())
    restored.load_state_dict(state)
    assert_batch_equal(next(iter(restored)), expected)
    standalone = build_validation_loader(cfg, Tokenizer())
    for actual, expected in zip(standalone, validation):
        assert_batch_equal(actual, expected)


def test_joint_validation_uses_separate_modality_denominators(tmp_path):
    cfg = settings(tmp_path)
    model = UMM(tiny_config())
    batches = [joint_batch(), joint_batch(), joint_batch()]
    batches[1]["image_loss_mask"].zero_()
    batches[2]["labels"].fill_(-100)
    observations = []
    def evaluate(model, batch):
        result = forward_loss(model, batch, cfg)
        observations.append(result)
        return result
    metrics = validate_losses(model, batches, evaluate, schedule=["joint"])
    assert metrics["val/loss"] == pytest.approx(sum(float(x["loss"]) for x in observations) / 3)
    for modality in ("text", "image"):
        denominator = sum(x[f"{modality}_target_count"] for x in observations)
        numerator = sum(float(x[f"{modality}_loss"]) * x[f"{modality}_target_count"] for x in observations)
        assert metrics[f"val/loss_joint_{modality}"] == pytest.approx(numerator / denominator)
        assert metrics[f"val/joint_{modality}_target_tokens"] == denominator
    assert "val/ppl_joint" not in metrics and "val/joint_target_tokens" not in metrics
