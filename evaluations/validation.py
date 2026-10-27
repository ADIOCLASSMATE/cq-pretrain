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
    return dict(
        seed=int(evaluation.get("seed", 424242)),
        max_batches=evaluation.get(
            "max_batches", config.get("training", {}).get("validation_max_batches")
        ),
    )


def loss_batch_seed(seed, task, batch_index):
    offsets = {"t2v": 1000003, "v2t": 2000006, "mixed": 3000009}
    return int(seed) + offsets[task] + int(batch_index)


@contextmanager
def evaluation_state(model, *, training=False, ema=None):
    modes = [(module, module.training) for module in model.modules()]
    try:
        with (
            preserved_rng_state(),
            ema.applied_to(model) if ema is not None else nullcontext(),
            torch.no_grad(),
        ):
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
    def scalar(key):
        return torch.as_tensor(output[key], device=device, dtype=dtype).detach()

    loss, ce, rf = scalar("metric_loss" if "metric_loss" in output else "loss"), scalar("text_loss"), scalar("video_loss")
    text, videos = scalar("text_target_count"), scalar("video_target_count")
    row = torch.stack(
        (
            loss * 0,
            loss * 0,
            loss,
            torch.ones_like(loss),
            ce * text,
            text,
            rf * videos,
            videos,
        )
    )
    if not bool(torch.isfinite(row).all()) or text + videos <= 0:
        raise ValueError("Nonfinite or empty loss metrics")
    return row


def reduce_loss_metrics(totals, *, schedule):
    slots = Counter(schedule)
    if set(totals) != set(slots):
        raise ValueError("Metrics must cover every scheduled stream")
    result, combined = {}, 0.0
    for task, frequency in slots.items():
        row = torch.as_tensor(totals[task], dtype=torch.float64)
        if row.shape != (8,) or not bool(torch.isfinite(row).all()) or row[3] <= 0:
            raise ValueError(f"Incomplete metrics for {task}")
        mean = float(row[2] / row[3])
        combined += mean * frequency / len(schedule)
        result[f"val/loss_{task}"] = mean
        result[f"val/{task}_microbatches"] = int(row[3])
        for modality, numerator, count in (
            ("text", row[4], row[5]),
            ("video", row[6], row[7]),
        ):
            result[f"val/{task}_{modality}_targets"] = int(count)
            if count > 0:
                value = float(numerator / count)
                result[f"val/loss_{task}_{modality}"] = value
                if modality == "text":
                    result[f"val/ppl_{task}_text"] = math.exp(min(value, 100))
    result["val/loss"] = combined
    return result


def validate_losses(
    model,
    batches,
    forward_loss,
    *,
    schedule,
    device=None,
    max_batches=None,
    seed=424242,
    ema=None,
    distributed=False,
):
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
    total = torch.zeros(
        (len(active), 8),
        dtype=torch.float32 if device.type == "npu" else torch.float64,
        device=device,
    )
    rank, world = (
        (dist.get_rank(), dist.get_world_size())
        if distributed and dist.is_initialized()
        else (0, 1)
    )
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
            # Reproducible per-stream CFG and flow draws, independent of rank.
            seed_evaluation(loss_batch_seed(seed, task, batch_index))
            output = forward_loss(model, batch)
            total[active.index(task)] += loss_metric_row(
                output, device=device, dtype=total.dtype
            )
        if world > 1:
            dist.all_reduce(total)
    return reduce_loss_metrics(dict(zip(active, total.cpu())), schedule=schedule)
