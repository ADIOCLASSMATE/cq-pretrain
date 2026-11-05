"""Same-position text CE and whole-video Wan rectified flow, no token shift."""

from __future__ import annotations
import torch
from torch.nn import functional as F
from data.packing import validate_representation_batch


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [to_device(x, device) for x in value]
    if isinstance(value, tuple):
        return tuple(to_device(x, device) for x in value)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


class LazyVideoCodec:
    """VAE exists outside trainable UMM and loads only for online Wan targets."""

    def __init__(self, config, device):
        self.config, self.device, self.codec = config, device, None

    def get(self):
        if self.codec is None:
            from data.codec import WanVideoCodec

            settings = self.config["codec"]
            self.codec = WanVideoCodec.from_pretrained(
                settings.get("path") or self.config["model"]["wan_path"],
                device=self.device,
                dtype=getattr(torch, settings["dtype"]),
            )
        return self.codec

    def encode(self, rgb):
        return self.get().encode(rgb)


def sample_times(count, device, settings, generator=None):
    def uniform():
        return torch.rand(count, device=device, generator=generator)

    if settings["time_sampling"] == "uniform":
        times = uniform()
    else:
        times = (
            torch.randn(count, device=device, generator=generator)
            * float(settings["logit_std"])
            + float(settings["logit_mean"])
        ).sigmoid()
        mixture = float(settings["uniform_mix"])
        if mixture:
            times = torch.where(uniform() < mixture, uniform(), times)
    epsilon = float(settings["time_eps"])
    return times.clamp(epsilon, 1 - epsilon)


def prepare_batch(batch, device, config, generator=None, *, codec=None):
    result = to_device({k: v for k, v in batch.items() if k != "_data_state"}, device)
    from models.backbone_qwen35 import cpu_metadata
    result["_cpu_layout_inputs"] = cpu_metadata(batch)
    latents = result.get("wan_latents", [])
    pixels = result.pop("wan_rgb", [None] * len(latents))
    if len(latents) != len(result["target_video_ids"]) or len(pixels) != len(latents):
        raise ValueError("Wan target inputs must follow target_video_ids order")
    validate_representation_batch(latents, pixels, branch="Wan")
    clean = []
    for z, rgb in zip(latents, pixels):
        if z is None:
            if codec is None or rgb is None:
                raise ValueError("Online Wan target requires RGB and a frozen VAE")
            z = codec.encode(rgb)
        if z.ndim != 4 or not bool(torch.isfinite(z).all()):
            raise ValueError("Wan target must be finite [C,T,H,W]")
        clean.append(z.float())
    result["wan_latents"] = clean
    if "video_uncond_mask" not in result:
        drop = torch.zeros(
            len(result["video_grid_thw"]), device=device, dtype=torch.bool
        )
        drop[result["target_video_ids"]] = (
            torch.rand(len(clean), device=device, generator=generator)
            < config["model"]["video_uncond_prob"]
        )
        result["video_uncond_mask"] = drop
    return result


def supervision_counts(batch):
    labels = batch["labels"]
    valid = labels.ne(-100)
    # CPU collator carries native multimodal types; CQ types are built later by
    # prepare_inputs. Native video placeholders (2) must never be text targets.
    if "mm_token_type_ids" in batch:
        valid &= batch["mm_token_type_ids"].eq(0)
    elif "token_types" in batch:
        valid &= batch["token_types"].ne(1) & batch["token_types"].ne(3)
    if "attention_mask" in batch:
        valid &= batch["attention_mask"].bool()
    return int(valid.sum()), len(batch["target_video_ids"])


def global_loss_weights(batches, runtime):
    """Each text token / target video has equal weight across DP and GA.

    DDP averages gradients across DP ranks; multiply local mean losses by D*n/N
    and D*m/M. There is no additional division by the number of microbatches.
    Flow repeats average independent draws per video and do not change M.
    """
    counts = [supervision_counts(batch) for batch in batches]
    totals = torch.tensor(counts, device=runtime.device, dtype=torch.int64).sum(0)
    totals = runtime.reduce(totals, reduction="sum").cpu().tolist()
    return [tuple(runtime.num_processes * count / total if total else 0.0
                  for count, total in zip(row, totals)) for row in counts]


def forward_loss(model, batch, config, generator=None, *, loss_weights=(1.0, 1.0)):
    clean = batch.get("wan_latents", [])
    repeats = config["flow"].get("repeats", 1)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("flow.repeats must be a positive integer")
    # Draw-major order matches UMM's repeated conditions. Reuse the original
    # batch/CFG decision; each copy receives independent flow time and noise.
    draws = clean * repeats
    times = sample_times(
        len(draws), batch["input_ids"].device, config["flow"], generator
    )
    noisy, target = [], []
    for z, s in zip(draws, times):
        noise = torch.randn(
            z.shape, device=z.device, dtype=torch.float32, generator=generator
        )
        # Wan: s=0 is clean, s=1 is noise; sampling integrates from 1 to 0.
        noisy.append((1 - s) * z + s * noise)
        target.append(noise - z)
    output = model(
        batch,
        noisy_latents=noisy or None,
        timesteps=times * config["flow"]["timestep_scale"] if noisy else None,
        compute_text=True,
        compute_text_loss=True,
        flow_repeats=repeats,
    )
    if target and len(output["velocity"]) != len(target):
        raise ValueError("Wan predictions must match every repeated flow target")
    labels = output["text_labels"]
    text_loss = output["text_loss"]
    video_loss = (
        torch.stack(
            [
                F.mse_loss(pred.float(), truth)
                for pred, truth in zip(output["velocity"], target)
            ]
        ).mean()
        if target
        else text_loss.new_zeros(())
    )
    loss = (
        config["model"]["lambda_text"] * loss_weights[0] * text_loss
        + config["model"]["lambda_video"] * loss_weights[1] * video_loss
    )
    return dict(
        loss=loss,
        metric_loss=config["model"]["lambda_text"] * text_loss.detach()
                    + config["model"]["lambda_video"] * video_loss.detach(),
        text_loss=text_loss.detach(),
        video_loss=video_loss.detach(),
        text_target_count=labels.numel(),
        video_target_count=len(clean),
    )
