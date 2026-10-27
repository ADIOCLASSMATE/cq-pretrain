"""Scorer interfaces; imports of UMM and accelerator libraries are demand-driven."""

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path


def score_loss(runtime, task, output_dir, *, distributed=False):
    """Use the exact training validation objective, batching and reduction.

    Batch diagnostics are observations of the same forward, not a separate loss
    implementation. They identify global per-stream batches even with DP.
    """
    import torch
    import torch.distributed as dist
    from data.dataloader import build_validation_loader
    from training.loss import LazyVideoCodec, prepare_batch, forward_loss
    from evaluations.validation import loss_validation_options, validate_losses

    settings = task.get("score") or {}
    supported = {"input", "objective", "metrics", "seed", "max_batches", "save_batches"}
    if set(settings) - supported:
        raise ValueError(f"Unsupported loss score parameters: {sorted(set(settings) - supported)}")
    if set(task.get("data") or {}) - {"sources"}:
        raise ValueError("loss data supports sources only; objective and batching inherit training")
    if settings.get("input", "dataset") != "dataset":
        raise ValueError("loss currently supports score.input=dataset, not offline statistics")
    if settings.get("objective", "training") != "training":
        raise ValueError("loss objective must be training; the supervised objective is not overridden")
    requested = settings.get("metrics", ["text_ce", "video_flow", "text_ppl"])
    if not requested or set(requested) - {"text_ce", "video_flow", "text_ppl"}:
        raise ValueError("loss metrics must be text_ce, video_flow and/or text_ppl")
    config = deepcopy(runtime.config)
    options = loss_validation_options(config)
    for key in ("seed", "max_batches"):
        if settings.get(key) is not None:
            options[key] = settings[key]
    if type(options["seed"]) is not int:
        raise ValueError("loss seed must be an integer")
    cap = options["max_batches"]
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("loss max_batches must be a positive integer or null")
    model = runtime.get_model()
    loader = build_validation_loader(config, runtime.processor, (task.get("data") or {}).get("sources"))
    codec = runtime.codec if hasattr(runtime, "codec") else LazyVideoCodec(config, runtime.device)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if distributed and dist.is_initialized() else int(os.environ.get("RANK", "0"))
    diagnostics = directory / f"batches.rank-{rank}.jsonl"
    save_batches = settings.get("save_batches", True)

    def indexed():
        counts = Counter()
        for batch in loader:
            task_name = batch["task"]
            yield dict(batch, _eval_batch_index=counts[task_name])
            counts[task_name] += 1

    def evaluate(current_model, batch):
        index = batch["_eval_batch_index"]
        source = {key: value for key, value in batch.items() if key != "_eval_batch_index"}
        prepared = prepare_batch(source, runtime.device, config, codec=codec)
        with runtime.autocast():
            output = forward_loss(current_model, prepared, config)
        if save_batches:
            row = dict(task=batch["task"], batch_index=index, sample_ids=batch.get("sample_ids", []))
            row.update({key: float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else value
                        for key, value in output.items()})
            with diagnostics.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return output

    if save_batches:
        diagnostics.write_text("")
    metrics = validate_losses(model, indexed(), evaluate, schedule=config["data"]["schedule"],
                              device=runtime.device, distributed=distributed, **options)
    # Preserve the training-compatible val/* keys and aggregate loss. Selected
    # metric families control reporting only; they never change supervision.
    result = {key: value for key, value in metrics.items()
              if ("/ppl_" not in key or "text_ppl" in requested)
              and (not ("/loss_" in key and key.endswith("_text")) or "text_ce" in requested)
              and (not ("/loss_" in key and key.endswith("_video")) or "video_flow" in requested)}
    if rank == 0:
        temporary = directory / ".metrics.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(directory / "metrics.json")
    return result


def score_vbench(runtime, task, output_dir, *, distributed=False):
    """Run the explicit external scoring adapter without loading the UMM runtime."""
    from evaluations.benchmarks.vbench.adapter import run_vbench_score
    return run_vbench_score(task, output_dir)


SCORERS = {"loss": score_loss, "vbench": score_vbench}


def get_scorer(name):
    try:
        return SCORERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown scorer {name!r}; available: {', '.join(SCORERS)}") from exc
