"""Same-position CE and whole-image rectified-flow targets.

Task weights are applied once per microbatch. The trainer/Accelerate divides
by accumulation steps; target counts are used only for reporting raw losses.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from models.modeling_umm import visual_indices


def sample_times(count, device, model_config, generator=None):
    mode = model_config.get("image_flow_time_sampling", "logit_normal")
    def uniform():
        return torch.rand((count,), device=device, generator=generator)
    if mode == "uniform":
        times = uniform()
    elif mode == "logit_normal":
        times = torch.sigmoid(torch.randn((count,), device=device, generator=generator) *
                              float(model_config.get("image_flow_logit_std", 1.0)) +
                              float(model_config.get("image_flow_logit_mean", 0.0)))
        mixture = float(model_config.get("image_flow_time_uniform_mix", 0.1))
        if mixture:
            use_uniform = uniform() < mixture
            times = torch.where(use_uniform, uniform(), times)
    else:
        raise ValueError(f"Unknown time sampling: {mode}")
    epsilon = float(model_config.get("image_flow_time_eps", 1e-5))
    return times.clamp(epsilon, 1.0 - epsilon)


def prepare_flow(batch, model_config, generator=None, *, allow_empty=False):
    rows, indices = visual_indices(batch, int(model_config["image_tokens_per_img"]))
    target = batch["image_latents"][rows[:, None], indices].float()
    repeats = int(model_config.get("image_flow_batch_mul", 4))
    target = target.repeat(repeats, 1, 1)
    times = sample_times(target.shape[0], target.device, model_config, generator)
    noise = torch.randn(target.shape, device=target.device, dtype=torch.float32, generator=generator)
    noisy = (1 - times[:, None, None]) * noise + times[:, None, None] * target
    mask = batch.get("image_loss_mask")
    if mask is None:
        weights = torch.ones(target.shape[:2], device=target.device, dtype=torch.bool)
    else:
        weights = mask[rows[:, None], indices].repeat(repeats, 1).bool()
    if not allow_empty and not bool(weights.any()):
        raise ValueError("T2I batch has no supervised visual tokens")
    return noisy, times, target - noise, weights


def prepare_batch(batch, device, config, generator=None, *, image_input_dtype=torch.float32):
    """Prepare CFG and content jitter in the image input projector's dtype.

    Training/evaluation callers pass the prepared model's actual parameter
    dtype; it can differ from both posterior-cache dtype and autocast dtype.
    Clean posterior targets retain their original values and dtype.
    """
    result = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
              for key, value in batch.items()}
    if "image_uncond_mask" not in result and "image_uncond_rows" not in result:
        probability = float(config["model"].get("image_uncond_prob", 0.1))
        if probability:
            token_types = result["token_types"]
            result["image_uncond_rows"] = (
                torch.rand(token_types.shape[0], device=device, generator=generator) < probability
            ) & token_types.eq(1).any(1)
    # The source draws input jitter before RF time/noise. Store content
    # augmentation separately so loss targets remain the clean posterior draw.
    strength = float(config["model"].get("image_input_noise_strength", 0.01))
    latents = result.get("image_latents")
    if strength > 0 and latents is not None and "content_image_latents" not in result:
        if bool(result["token_types"].eq(1).any()):
            latents = latents.to(dtype=image_input_dtype)
            result["content_image_latents"] = latents + torch.randn(
                latents.shape, device=latents.device, dtype=latents.dtype, generator=generator,
            ) * strength
        else:
            # Text-only batches carry placeholder latents in the unified schema.
            # The source has no visual input here and draws no augmentation noise.
            result["content_image_latents"] = latents
    return result


def forward_loss(model, batch, config, generator=None):
    """Independent same-position CE and image RF supervision in one model call.

    ``joint`` keeps the source caption dataset's mixed T2I/I2T masks. Either
    modality can be empty in an individual batch. Explicit compute_*_loss
    switches also support both targets within a single serialized sample.
    """
    task = batch.get("task", "t2i")
    if task == "climbmix":
        task = "text"
    model_config = config["model"]
    if task not in {"text", "t2i", "i2t", "joint"}:
        raise ValueError(f"Unknown training task: {task}")
    compute_text = bool(batch.get("compute_text_loss", task in {"text", "i2t", "joint"}))
    compute_image = bool(batch.get("compute_image_loss", task in {"t2i", "joint"}))
    noisy = times = None
    if compute_image:
        noisy, times, target_velocity, weights = prepare_flow(
            batch, model_config, generator, allow_empty=task == "joint",
        )
    if not compute_text and not compute_image:
        raise ValueError("At least one training objective must be enabled")
    output = model(batch, noisy_latents=noisy, timesteps=times, compute_text=compute_text)
    zero = next(iter(output.values())).sum() * 0.0
    image_loss = text_loss = zero
    image_count = text_count = 0
    if compute_image:
        errors = (output["velocity"].float() - target_velocity).square().mean(-1)
        count = weights.sum()
        image_loss = (errors * weights).sum() / count.clamp_min(1)
        image_count = int(count)
    if compute_text:
        logits, labels = output["text_logits"], output["text_labels"]
        if labels is None:
            raise ValueError("Text training requires same-position labels")
        text_count = labels.numel()
        if text_count == 0 and task != "joint":
            raise ValueError(f"{task} batch has no supervised text tokens")
        text_loss = F.cross_entropy(logits.float(), labels, reduction="sum") / max(text_count, 1)
    if text_count + image_count == 0:
        raise ValueError(f"{task} batch has no supervised targets")
    loss = (text_loss * float(model_config.get("lambda_text", 0.05))
            + image_loss * float(model_config.get("lambda_image", 1.0)))
    result = dict(loss=loss, task=task, text_loss=text_loss, image_loss=image_loss,
                  text_target_count=text_count, image_target_count=image_count)
    # Joint losses have two independent denominators, never a combined token mean.
    if task != "joint":
        result.update(raw_loss=image_loss if task == "t2i" else text_loss,
                      target_count=image_count if task == "t2i" else text_count)
    return result
