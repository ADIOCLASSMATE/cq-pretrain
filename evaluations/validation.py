"""Training-protocol validation without changing training RNG or module modes."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
import math
import random

import torch
import torch.distributed as dist

from utils.checkpoint import preserved_rng_state


def loss_validation_options(config):
    """Shared training/standalone validation seed and optional diagnostic cap."""
    evaluation = config.get("evaluation", {})
    return dict(seed=int(evaluation.get("seed", 424242)),
                max_batches=evaluation.get("max_batches", config.get("training", {}).get("validation_max_batches")))


def loss_batch_seed(seed, task, batch_index):
    # The first three offsets are the pinned source protocol. Joint is a
    # separate opt-in stream and must not reuse any single-task random draws.
    offsets = {"text": 0, "climbmix": 0, "t2i": 1000003, "i2t": 2000006, "joint": 3000009}
    return int(seed) + offsets[task] + int(batch_index)


@contextmanager
def evaluation_state(model, *, training=False, ema=None):
    modes = [(module, module.training) for module in model.modules()]
    try:
        with preserved_rng_state(), (ema.applied_to(model) if ema is not None else nullcontext()), torch.no_grad():
            model.train(training)
            yield
    finally:
        for module, mode in modes:
            module.training = mode


def seed_evaluation(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    for name in ("cuda", "npu"):
        device = getattr(torch, name, None)
        if device is not None and device.is_available():
            device.manual_seed_all(seed)


def loss_metric_row(output, *, device, dtype):
    """Microbatch totals with separate CE and RF denominators.

    The first four entries retain single-task reporting compatibility. Joint
    batches leave the combined token numerator/denominator empty and report
    their weighted objective as a microbatch mean instead.
    """
    get = output.get if isinstance(output, dict) else lambda key, default=None: getattr(output, key, default)
    def scalar(key, default=0):
        return torch.as_tensor(get(key, default), device=device, dtype=dtype).detach()
    loss = scalar("loss")
    raw, count = scalar("raw_loss"), scalar("target_count")
    text_count, image_count = scalar("text_target_count"), scalar("image_target_count")
    values = torch.stack((raw * count, count, loss, torch.ones_like(loss),
                          scalar("text_loss") * text_count, text_count,
                          scalar("image_loss") * image_count, image_count))
    if not bool(torch.isfinite(values).all()) or bool((values[[1, 5, 7]] < 0).any()) or count + text_count + image_count <= 0:
        raise ValueError("nonfinite/empty loss metrics")
    return values


def reduce_loss_metrics(totals, *, schedule):
    """Keep token means separate from schedule-weighted microbatch means.

    Each source supplies [raw_loss * targets, targets, weighted_loss, batches]
    and optionally [CE * text_targets, text_targets, RF * image_targets, image_targets].
    This preserves ``unified_schedule_microbatch_mean_v1`` from the source.
    """
    slots = Counter(schedule)
    if not slots or set(totals) != set(slots):
        raise ValueError("validation must cover every active schedule source exactly")
    metrics = {}
    contribution = 0.0
    for source, frequency in slots.items():
        row = torch.as_tensor(totals[source], dtype=torch.float64)
        if row.shape not in {(4,), (8,)} or not bool(torch.isfinite(row).all()) or row[3] <= 0:
            raise ValueError(f"invalid or incomplete validation totals for {source}")
        batch_mean = float(row[2] / row[3])
        if source == "joint":
            if row.shape != (8,) or row[5] + row[7] <= 0 or bool((row[[5, 7]] < 0).any()):
                raise ValueError("joint validation requires separate text/image totals")
            raw = batch_mean
            for modality, numerator, denominator in (("text", row[4], row[5]), ("image", row[6], row[7])):
                mean = float(numerator / denominator.clamp_min(1))
                metrics[f"val/loss_joint_{modality}"] = mean
                metrics[f"val/joint_{modality}_target_tokens"] = int(denominator)
                if modality == "text" and denominator > 0:
                    metrics["val/ppl_joint_text"] = math.exp(min(mean, 100))
        else:
            if row[1] <= 0:
                raise ValueError(f"invalid or incomplete validation totals for {source}")
            raw = float(row[0] / row[1])
            metrics[f"val/{source}_target_tokens"] = int(row[1])
        weighted = batch_mean * frequency / len(schedule)
        metrics[f"val/loss_{source}"] = raw
        metrics[f"val/{source}_microbatches"] = int(row[3])
        metrics[f"val/mean_microbatch_loss_{source}"] = batch_mean
        metrics[f"val/weighted_contribution_{source}"] = weighted
        if source in {"text", "climbmix", "i2t"}:
            metrics[f"val/ppl_{source}"] = math.exp(min(raw, 100))
        contribution += weighted
    metrics["val/loss"] = contribution
    return metrics


def validate_losses(model, batches, forward_loss, *, schedule, device=None,
                    max_batches=None, seed=424242, ema=None, distributed=False):
    """Evaluate a fixed global batch iterable with the training objective.

    ``forward_loss(model, batch)`` returns a weighted loss and modality totals;
    single-task raw_loss/target_count results are also supported.
    Every batch has a ``task`` string. With distributed=True each rank receives
    the same global iterable and owns each task's batch index modulo world size, so uneven
    target counts and changes in evaluation partition do not alter averaging.
    The iterable must be independent of the training data cursor.
    """
    device = torch.device(device or next(model.parameters()).device)
    active = tuple(dict.fromkeys(schedule))
    total = torch.zeros((len(active), 8), dtype=torch.float32 if device.type == "npu" else torch.float64, device=device)
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed and dist.is_initialized() else (0, 1)
    with evaluation_state(model, training=True, ema=ema):
        seed_evaluation(seed)
        iterator = iter(batches)
        source_batches = Counter()
        index = -1
        while True:
            index += 1
            if max_batches is not None and index >= max_batches:
                break
            try:
                batch = next(iterator)
            except StopIteration:
                break
            task = batch["task"]
            if task not in active:
                raise ValueError(f"validation contains unscheduled source: {task}")
            batch_index = source_batches[task]
            source_batches[task] += 1
            if batch_index % world != rank:
                continue
            # Dataset posterior draws and packed CFG use stable per-sample
            # seeds. Model-side CFG, jitter and RF draws use the source seed.
            seed_evaluation(loss_batch_seed(seed, task, batch_index))
            output = forward_loss(model, batch)
            total[active.index(task)] += loss_metric_row(output, device=device, dtype=total.dtype)
        if world > 1:
            dist.all_reduce(total)
    return reduce_loss_metrics(dict(zip(active, total.cpu())), schedule=schedule)
