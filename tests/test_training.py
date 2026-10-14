"""Real UMM gradients and updates across a complete mixed-task window."""
from copy import deepcopy
import math

import pytest
import torch
from torch.nn import functional as F

from configs.config import DEFAULTS, merge_config
from models.modeling_umm import UMM
from training.loss import forward_loss, prepare_batch, prepare_flow, sample_times
from training.optimizer import build_optimizer_and_scheduler, parameter_role, wsd_multiplier
from training.train import build_accelerator
from test_model_core import activate_head, batch as visual_batch, config as tiny_config


def settings(tmp_path, **model_updates):
    return merge_config(DEFAULTS, {
        "model": {**tiny_config().to_dict(), "image_flow_batch_mul": 3,
                  "image_uncond_prob": 0., "lambda_text": .07, "lambda_image": 1.3,
                  **model_updates},
        "training": {"output_dir": str(tmp_path), "max_train_steps": 4,
                     "gradient_accumulation_steps": 4, "mixed_precision": "no",
                     "cpu": True, "save_every": 0, "val_every": 0, "log_every": 1,
                     "max_grad_norm": 1000., "use_ema": False},
        "optimizer": {"warmup_steps": 0, "decay_steps": 0},
    })


def tasks():
    text = dict(input_ids=torch.tensor([[2, 3, 4, 30]]),
                token_types=torch.tensor([[0, 0, 0, 2]]),
                labels=torch.tensor([[-100, 3, 4, 30]]), task="text")
    image = visual_batch()
    image["image_loss_mask"] = torch.tensor([[0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=torch.bool)
    image["labels"] = torch.full_like(image["input_ids"], -100)
    i2t = deepcopy(image)
    i2t["task"] = "i2t"
    i2t["image_loss_mask"].zero_()
    i2t["labels"][0, -2:] = i2t["input_ids"][0, -2:]
    return [text, image, deepcopy(text), i2t]


def manual_source_times(count, values, generator):
    if values["image_flow_time_sampling"] == "uniform":
        time = torch.rand(count, generator=generator)
    else:
        time = torch.sigmoid(torch.randn(count, generator=generator) * values["image_flow_logit_std"] + values["image_flow_logit_mean"])
        mixture = values["image_flow_time_uniform_mix"]
        if mixture > 0:
            selected = torch.rand(count, generator=generator) < mixture
            time = torch.where(selected, torch.rand(count, generator=generator), time)
    return time.clamp(values["image_flow_time_eps"], 1 - values["image_flow_time_eps"])


def manual_loss(model, inputs, cfg, generator):
    """Independent expressions from source CE and JointDiTFlowLoss.forward."""
    if inputs["task"] != "t2i":
        hidden = model.backbone(inputs)["query_hidden"]
        selected = inputs["labels"].ne(-100) & (inputs["token_types"].eq(0) | inputs["token_types"].eq(2))
        logits = model.lm_head(hidden[selected])
        return F.cross_entropy(logits, inputs["labels"][selected], reduction="sum") / selected.sum() * cfg["lambda_text"]
    target = inputs["image_latents"][:, 3:7].float().repeat(cfg["image_flow_batch_mul"], 1, 1)
    time = manual_source_times(target.shape[0], cfg, generator)
    epsilon = torch.randn(target.shape, generator=generator)
    noisy = (1 - time[:, None, None]) * epsilon + time[:, None, None] * target
    condition = model.encode_condition(inputs).repeat(cfg["image_flow_batch_mul"], 1, 1)
    prediction = model.generator(noisy, time, condition)
    mask = inputs["image_loss_mask"][:, 3:7].repeat(cfg["image_flow_batch_mul"], 1)
    error = (prediction.float() - (target - epsilon)).square().mean(-1)
    return (error * mask).sum() / mask.sum() * cfg["lambda_image"]


@pytest.mark.parametrize("distribution,mixture", [("uniform", 0.), ("logit_normal", 0.), ("logit_normal", .35)])
def test_time_draws_and_noise_match_source_order(tmp_path, distribution, mixture):
    cfg = settings(tmp_path, image_flow_time_sampling=distribution, image_flow_time_uniform_mix=mixture)["model"]
    a, b = (torch.Generator().manual_seed(92) for _ in range(2))
    actual = sample_times(200, torch.device("cpu"), cfg, a)
    expected = manual_source_times(200, cfg, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(a.get_state(), b.get_state(), rtol=0, atol=0)
    inputs = tasks()[1]
    a.manual_seed(17)
    b.manual_seed(17)
    noisy, times, target_velocity, mask = prepare_flow(inputs, cfg, a)
    target = inputs["image_latents"][:, 3:7].float().repeat(cfg["image_flow_batch_mul"], 1, 1)
    times_expected = manual_source_times(target.shape[0], cfg, b)
    noise = torch.randn(target.shape, generator=b)
    torch.testing.assert_close(times, times_expected, rtol=0, atol=0)
    torch.testing.assert_close(noisy, (1 - times[:, None, None]) * noise + times[:, None, None] * target, rtol=0, atol=0)
    torch.testing.assert_close(target_velocity, target - noise, rtol=0, atol=0)
    assert mask.sum() == 6 and times.shape == (3,)


def test_mixed_window_loss_gradient_and_optimizer_match_manual_reference(tmp_path):
    torch.manual_seed(4)
    cfg = settings(tmp_path)
    model = UMM(tiny_config()).train()
    activate_head(model.generator)
    reference = deepcopy(model)
    opt, scheduler = build_optimizer_and_scheduler(model, cfg)
    ref_opt, ref_scheduler = build_optimizer_and_scheduler(reference, cfg)
    a, b = (torch.Generator().manual_seed(51) for _ in range(2))
    for inputs in tasks():
        actual = forward_loss(model, inputs, cfg, a)
        expected = manual_loss(reference, inputs, cfg["model"], b)
        torch.testing.assert_close(actual["loss"], expected, rtol=0, atol=0)
        expected_count = 6 if inputs["task"] == "t2i" else int(inputs["labels"].ne(-100).sum())
        assert actual["target_count"] == expected_count
        (actual["loss"] / 4).backward()
        (expected / 4).backward()
    for (name, actual), (ref_name, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    opt.step()
    ref_opt.step()
    scheduler.step()
    ref_scheduler.step()
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, reference.state_dict()[name], rtol=0, atol=0)


def test_optimizer_has_unique_tied_parameters_and_source_roles(tmp_path):
    cfg = settings(tmp_path)
    model = UMM(tiny_config())
    optimizer, _ = build_optimizer_and_scheduler(model, cfg)
    seen = []
    for group in optimizer.param_groups:
        for name, parameter in zip(group["param_names"], group["params"]):
            seen.append(id(parameter))
            role = parameter_role(name)
            assert group["lr"] == cfg["optimizer"][f"{role}_lr"]
            no_decay = role in {"projector", "embedding"} or "bias" in name or "norm.weight" in name
            expected = 0 if no_decay else cfg["optimizer"]["generator_weight_decay" if role == "generator" else "weight_decay"]
            assert group["weight_decay"] == expected
    assert len(set(seen)) == len(seen) == len(list(model.parameters()))
    assert seen.count(id(model.lm_head.weight)) == 1


@pytest.mark.parametrize("step", [0, 1, 2, 4, 7, 8, 9, 10])
def test_wsd_boundaries_match_source(step):
    warmup, decay, total, minimum = 2, 3, 10, .1
    expected = step / 2 if step < 2 else (1 if step < 7 else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 7) / 3)))
    assert wsd_multiplier(step, warmup, decay, total, minimum) == expected


def test_prepared_content_jitter_is_not_resampled_or_applied_to_targets(tmp_path):
    cfg = settings(tmp_path, image_input_noise_strength=.01)
    model = UMM(tiny_config(image_input_noise_strength=.01)).train()
    inputs = tasks()[1]
    clean = inputs["image_latents"].clone()
    prepared = prepare_batch(inputs, torch.device("cpu"), cfg, torch.Generator().manual_seed(8))
    assert "content_image_latents" in prepared
    assert not torch.equal(prepared["content_image_latents"], clean)
    torch.testing.assert_close(prepared["image_latents"], clean, rtol=0, atol=0)
    before = torch.get_rng_state().clone()
    first = model.backbone.embed_content(prepared)
    second = model.backbone.embed_content(prepared)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_accelerator_accumulation_is_one_actual_umm_update(tmp_path):
    cfg = settings(tmp_path)
    accelerator = build_accelerator(cfg)
    torch.manual_seed(71)
    model = UMM(tiny_config())
    activate_head(model.generator)
    reference = deepcopy(model)
    opt, scheduler = build_optimizer_and_scheduler(model, cfg)
    ref_opt, ref_scheduler = build_optimizer_and_scheduler(reference, cfg)
    model, opt, scheduler = accelerator.prepare(model, opt, scheduler)
    a, b = (torch.Generator().manual_seed(11) for _ in range(2))
    syncs = []
    for inputs in tasks():
        with accelerator.accumulate(model):
            result = forward_loss(model, inputs, cfg, a)
            accelerator.backward(result["loss"])
            (manual_loss(reference, inputs, cfg["model"], b) / 4).backward()
            syncs.append(accelerator.sync_gradients)
            if accelerator.sync_gradients:
                opt.step()
                scheduler.step()
                opt.zero_grad()
    assert syncs == [False, False, False, True]
    ref_opt.step()
    ref_scheduler.step()
    for name, actual in accelerator.unwrap_model(model).state_dict().items():
        torch.testing.assert_close(actual, reference.state_dict()[name], rtol=0, atol=0)
    accelerator.free_memory()


@pytest.mark.parametrize("joint", [False, True])
def test_real_trainer_mixed_loader_resume_matches_next_update(tmp_path, joint):
    from data.dataloader import build_dataloaders
    from models.modeling_umm import UMMConfig
    from training.ema import RankShardedEMA, build_sharded_ema_layout
    from training.trainer import Trainer
    from test_data import Tokenizer, config as data_config

    cfg = merge_config(settings(tmp_path / "output"), data_config(tmp_path / "dataset"))
    cfg["model"].update(vocab_size=256, image_mask_token_id=6, eos_token_id=2, pad_token_id=0)
    cfg["training"].update(max_train_steps=3, use_ema=True)
    cfg["optimizer"].update(warmup_steps=0, decay_steps=1)
    if joint:
        cfg["data"]["schedule"] = ["joint"]
        cfg["data"]["sources"]["joint"] = dict(micro_batch_size=4)
        cfg["training"]["gradient_accumulation_steps"] = 1

    def construct():
        accelerator = build_accelerator(cfg)
        model = UMM(UMMConfig.from_dict(cfg["model"], strict=False))
        activate_head(model.generator)
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
        loader, validation = build_dataloaders(cfg, Tokenizer())
        layout = build_sharded_ema_layout(model, world_size=1, chunk_numel=8192)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        loader.prepare_with_accelerator(accelerator)
        ema = RankShardedEMA(layout, rank=0, decay=.9, update_after_step=0)
        ema.bind(accelerator.unwrap_model(model))
        ema.initialize_from_model()
        trainer = Trainer(model, optimizer, scheduler, loader, accelerator, cfg,
                          validation_loader=validation, ema=ema)
        return trainer, accelerator

    torch.manual_seed(56)
    first, accelerator = construct()
    first.fit(stop_after_steps=1)
    checkpoint = first.output_dir / "checkpoint-1"
    assert checkpoint.is_dir() and first.train_loader.state_dict()["micro_step"] == (1 if joint else 4)
    first.fit(stop_after_steps=2)
    expected = {name: value.clone() for name, value in accelerator.unwrap_model(first.model).state_dict().items()}
    expected_ema = {name: value.clone() for name, value in first.ema.shards.items()}
    expected_data = first.train_loader.state_dict()
    accelerator.free_memory()
    del first, accelerator

    torch.manual_seed(9876)
    restored, accelerator = construct()
    restored.resume(checkpoint)
    assert restored.global_step == 1
    restored.fit(stop_after_steps=2)
    for name, value in accelerator.unwrap_model(restored.model).state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    for name, value in restored.ema.shards.items():
        torch.testing.assert_close(value, expected_ema[name], rtol=0, atol=0)
    assert restored.train_loader.state_dict() == expected_data
    assert restored.micro_step == 0 and restored.global_step == 2
    accelerator.free_memory()


def test_sampling_discards_training_augmentation_before_cfg(tmp_path):
    from models.sampling import sample_visual_latents
    cfg = settings(tmp_path, image_input_noise_strength=.01)
    model = UMM(tiny_config(image_input_noise_strength=.01)).train()
    activate_head(model.generator)
    original = tasks()[1]
    prepared = prepare_batch(original, torch.device("cpu"), cfg, torch.Generator().manual_seed(18))
    prepared["content_image_latents"] = prepared["content_image_latents"] + 100
    noise = torch.randn(1, 4, 4)
    expected = sample_visual_latents(model, original, initial_noise=noise, cfg=3.5, steps=2)
    actual = sample_visual_latents(model, prepared, initial_noise=noise, cfg=3.5, steps=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_text_zero_latent_placeholder_does_not_draw_image_jitter(tmp_path):
    cfg = settings(tmp_path, image_input_noise_strength=.01, image_uncond_prob=.1)
    text = tasks()[0]
    text["image_latents"] = torch.zeros(1, 4, 4)
    text["image_latent_mask"] = torch.zeros(1, 4, dtype=torch.bool)
    model = UMM(tiny_config(image_input_noise_strength=.01)).train()
    a, b = (torch.Generator().manual_seed(95) for _ in range(2))
    prepared = prepare_batch(text, torch.device("cpu"), cfg, a)
    # Source CFG sampling still draws a row coin even when has_image is false.
    torch.rand(1, generator=b)
    torch.testing.assert_close(a.get_state(), b.get_state(), rtol=0, atol=0)
    before = torch.get_rng_state().clone()
    model.backbone(prepared)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_full_image_training_random_draw_order_matches_source(tmp_path):
    cfg = settings(tmp_path, image_input_noise_strength=.01, image_uncond_prob=.25)
    inputs = tasks()[1]
    a, b = (torch.Generator().manual_seed(31) for _ in range(2))
    prepared = prepare_batch(inputs, torch.device("cpu"), cfg, a)
    noisy, times, velocity, _ = prepare_flow(prepared, cfg["model"], a)
    expected_uncond = torch.rand(1, generator=b) < .25
    expected_content = inputs["image_latents"] + torch.randn(inputs["image_latents"].shape, generator=b) * .01
    target = inputs["image_latents"][:, 3:7].repeat(3, 1, 1)
    expected_times = manual_source_times(3, cfg["model"], b)
    noise = torch.randn(target.shape, generator=b)
    torch.testing.assert_close(prepared["image_uncond_rows"], expected_uncond, rtol=0, atol=0)
    torch.testing.assert_close(prepared["content_image_latents"], expected_content, rtol=0, atol=0)
    torch.testing.assert_close(times, expected_times, rtol=0, atol=0)
    torch.testing.assert_close(noisy, (1 - times[:, None, None]) * noise + times[:, None, None] * target, rtol=0, atol=0)
    torch.testing.assert_close(velocity, target - noise, rtol=0, atol=0)
    torch.testing.assert_close(a.get_state(), b.get_state(), rtol=0, atol=0)
