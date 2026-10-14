"""Whole-image rectified-flow integration; one fixed backbone evaluation."""
from __future__ import annotations

import torch

from .modeling_umm import visual_indices


def scheduled_cfg(cfg, schedule, progress):
    schedule = str(schedule or "constant").lower()
    if cfg == 1.0 or schedule in {"constant", "none", "off", ""}:
        return float(cfg)
    if schedule == "linear":
        return 1.0 + (float(cfg) - 1.0) * min(1.0, max(0.0, float(progress)))
    raise ValueError("cfg_schedule must be constant or linear")


def guided_velocity(generator, x, t, condition, cfg):
    if condition.shape[0] == x.shape[0] and cfg == 1.0:
        return generator(x, t, condition)
    if condition.shape[0] != 2 * x.shape[0]:
        raise ValueError("CFG requires conditional then unconditional conditions")
    conditional, unconditional = generator(
        torch.cat((x, x)), torch.cat((t, t)), condition,
    ).chunk(2)
    return unconditional + float(cfg) * (conditional - unconditional)


@torch.no_grad()
def integrate_flow(generator, condition, *, steps=10, solver="heun", cfg=1.0,
                   cfg_schedule="constant", initial_noise=None, temperature=1.0,
                   debug_finite=False, return_trace=False):
    """Update the complete latent grid at every Euler/Heun evaluation."""
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("Use a positive step count and Euler or Heun")
    paired = float(cfg) != 1.0
    if paired and condition.shape[0] % 2:
        raise ValueError("CFG requires paired condition rows")
    batch = condition.shape[0] // (2 if paired else 1)
    shape = (batch, generator.image_tokens_per_img, generator.in_channels)
    if initial_noise is not None and tuple(initial_noise.shape) != shape:
        raise ValueError(f"initial_noise must have shape {shape}")
    x = (torch.randn(shape, device=condition.device, dtype=torch.float32) if initial_noise is None
         else initial_noise.to(device=condition.device, dtype=torch.float32).clone()) * float(temperature)
    times = torch.linspace(0.0, 1.0, steps + 1, device=condition.device, dtype=torch.float32)
    for step in range(steps):
        t, t_next = times[step].expand(batch), times[step + 1].expand(batch)
        dt = times[step + 1] - times[step]
        velocity = guided_velocity(generator, x, t, condition, scheduled_cfg(cfg, cfg_schedule, step / steps)).float()
        predictor = x + dt * velocity
        if solver == "heun":
            next_velocity = guided_velocity(generator, predictor, t_next, condition,
                                            scheduled_cfg(cfg, cfg_schedule, (step + 1) / steps)).float()
            x = x + 0.5 * dt * (velocity + next_velocity)
        else:
            x = predictor
        if debug_finite and not bool(torch.isfinite(x).all()):
            raise FloatingPointError(f"Nonfinite joint DiT state at step {step}")
    output = x.to(generator.input_proj.weight.dtype)
    trace = {"solver": solver, "steps": steps, "flow_head_calls": steps * (2 if solver == "heun" else 1)}
    return (output, trace) if return_trace else output


def generation_batch(batch, config, paired):
    """Mask target content and build a batched CFG copy without mutating input."""
    result = dict(batch)
    # Training augmentation is not an observed image input at generation time.
    result.pop("content_image_latents", None)
    ids = batch["input_ids"]
    rows, indices = visual_indices(batch, config.image_tokens_per_img)
    if rows.numel() != ids.shape[0] or not torch.equal(rows.sort().values, torch.arange(ids.shape[0], device=ids.device)):
        raise ValueError("Image sampling requires exactly one target image per batch row")
    target = torch.zeros_like(ids, dtype=torch.bool)
    target[rows[:, None], indices] = True
    context = batch["token_types"].eq(1) & ~target
    latents = batch.get("image_latents")
    if latents is None:
        if bool(context.any()):
            raise ValueError("Observed context images require latent values")
        latents = torch.zeros((*ids.shape, config.image_latent_dim), device=ids.device)
    observed = batch.get("image_latent_mask", batch["token_types"].eq(1))
    if bool((context & ~observed.bool()).any()):
        raise ValueError("All context image tokens must be observed")
    result["image_latents"] = torch.where(target[..., None], 0.0, latents)
    result["image_latent_mask"] = context
    result.pop("image_uncond_rows", None)
    result.pop("image_uncond_mask", None)
    if paired:
        for key in ("input_ids", "token_types", "sigma", "segment_ids", "image_latents", "image_latent_mask", "labels"):
            if key in result and result[key] is not None:
                result[key] = torch.cat((result[key], result[key]), 0)
        if result.get("position_ids") is not None:
            result["position_ids"] = torch.cat((result["position_ids"], result["position_ids"]), 1)
        second = batch["image_span_table"].clone()
        second[:, 0] += ids.shape[0]
        result["image_span_table"] = torch.cat((batch["image_span_table"], second), 0)
        result["image_uncond_mask"] = torch.cat((torch.zeros_like(target), target), 0)
    return result


@torch.no_grad()
def sample_visual_latents(model, batch, *, steps=10, solver="heun", cfg=1.0,
                          cfg_schedule="constant", initial_noise=None, temperature=1.0,
                          debug_finite=False, return_trace=False):
    """Return [target_images, image_tokens, latent_dim], with eval mode restored."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        prepared = generation_batch(batch, model.config, paired=float(cfg) != 1.0)
        condition = model.encode_condition(prepared)
        output, trace = integrate_flow(
            model.generator, condition, steps=steps, solver=solver, cfg=cfg,
            cfg_schedule=cfg_schedule, initial_noise=initial_noise, temperature=temperature,
            debug_finite=debug_finite, return_trace=True,
        )
        trace.update(backbone_calls=1, backbone_condition_cached=True, cfg_batched=float(cfg) != 1.0,
                     shared_time_per_image=True, generation_mode="joint_full_image_flow")
        return (output, trace) if return_trace else output
    finally:
        for module, mode in modes:
            module.training = mode
