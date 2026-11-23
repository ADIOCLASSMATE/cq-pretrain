"""Per-rank and global step logs and the end-of-run summary. Formats are unchanged."""

import json
import math
import resource
import statistics

import torch
import torch.distributed as dist

import distributed.accel as accel
import distributed.megatron as P

ACCEL = accel.accel_type()

# Global step-row keys taken directly from the DP-summed counts.
_ROW_COUNTS = (
    "body_tokens",
    "occupied_tokens",
    "images_consumed",
    "generation_images",
    "generation_tokens",
    "documents_completed",
    "t2i_captions_completed",
    "i2t_captions_completed",
    "active_microbatches",
    "supervised_text_tokens",
    "condition_query_tokens",
    "unconditional_images",
    "instruction_dropped_images",
    "instruction_dropout_eligible_images",
    "backbone_processed_slots",
    "padding_tokens",
    "dit_microbatches",
)


def write_rank_timing(args, run, step, result):
    """timing_rank_*.jsonl: this rank's wall time, data waits and pack shapes."""
    packs = []
    for batch, receipt in (x for x in result.micro_batches if x):
        packs.append(
            dict(
                length=batch.ids.shape[1],
                text=batch.text_indices.numel(),
                queries=len(batch.mainline.query_positions),
                images=len(batch.image_indices),
                generation_images=batch.generation_image_count,
                vision_patches=sum(batch.vision.lengths) if batch.vision else 0,
                generation_tokens=receipt.get("generation_tokens", 0),
                dit_microbatches=receipt.get("dit_microbatches", 1 if batch.generation_image_count else 0),
                generation_grids=batch.generation.grids if batch.generation else None,
            )
        )
    row = dict(
        step=step,
        seconds=result.local_seconds,
        host_wait=result.data_wait_seconds,
        prepare_host=result.data_prepare_seconds,
        device_wait=result.data_device_wait_seconds,
        packs=packs,
    )
    if args.phase_timing:
        row.update(forward_ms=sum(result.pack_forward_ms), backward_ms=sum(result.pack_backward_ms))
        timings = zip(packs, result.pack_forward_ms, result.pack_backward_ms, result.renderer_forward_ms, strict=True)
        for pack, forward, backward, renderer in timings:
            pack.update(forward_ms=forward, backward_ms=backward, deferred_renderer_forward_ms=renderer)
    with (run.output / f"timing_rank_{run.rank:05d}.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def write_rank_step(args, run, host, step, result, warmup, data_epoch):
    """steps_rank_*.jsonl and consumed_rank_*.jsonl."""
    totals = result.loss_totals
    record = dict(
        step=step,
        rank=run.rank,
        tp_rank=host["tp_rank"],
        dp_rank=P.dp_rank(),
        hostname=host["hostname"],
        seconds=result.local_seconds,
        peak_memory_gib=result.local_memory / 2**30,
        loss=totals[0].item(),
        text_loss=totals[1].item(),
        image_loss=totals[2].item(),
        grad_norm=float(result.grad_norm),
        optimizer_updated=bool(result.updated),
        global_tokens_per_step=run.global_batch * args.sequence_length,
        warmup=warmup,
    )
    if data_epoch is not None:
        record["data_epoch"] = data_epoch
    with (run.output / f"steps_rank_{run.rank:05d}.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    if args.log_all_ranks:
        print(json.dumps(dict(rank_step=record)), flush=True)
    if result.counts is not None:
        with (run.output / f"consumed_rank_{run.rank:05d}.jsonl").open("a") as trace:
            trace.write(json.dumps(dict(step=step, receipts=result.receipts)) + "\n")


def global_row(args, cfg, run, step, result, warmup, ema, data_epoch):
    """The steps.jsonl row: maxima over ranks for time and memory, DP-summed data counts."""
    seconds = result.seconds
    loss, text_loss, image_loss = result.values
    row = dict(
        step=step,
        seconds=seconds,
        tokens_per_second=run.global_batch * args.sequence_length / seconds,
        peak_memory_gib=result.memory / 2**30,
        loss=loss,
        text_loss=text_loss,
        image_loss=image_loss,
        grad_norm=float(result.grad_norm),
        warmup=warmup,
        lr=result.lr,
    )
    if ema is not None:
        row.update(renderer_ema_decay=ema.decay, renderer_ema_updates=ema.num_updates)
    counts = result.counts
    if counts is not None:
        row.pop("tokens_per_second")
        row["capacity_slots_per_second"] = counts["active_microbatches"] * args.sequence_length / seconds
        resources = torch.tensor(
            [
                result.data_wait_seconds,
                result.data_prepare_seconds,
                result.data_device_wait_seconds,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            ],
            device=ACCEL,
        )
        dist.all_reduce(resources, op=dist.ReduceOp.MAX)
        waits = resources.tolist()
        row.update({key: counts[key] for key in _ROW_COUNTS})
        row.update(
            flow_samples=counts["generation_images"] * cfg.rf,
            data_wait_seconds=waits[0],
            data_prepare_seconds=waits[1],
            data_device_wait_seconds=waits[2],
            cpu_peak_rss_gib=waits[3],
            images_per_second=counts["images_consumed"] / seconds,
            generation_images_per_second=counts["generation_images"] / seconds,
            documents_per_second=counts["documents_completed"] / seconds,
            body_tokens_per_second=counts["body_tokens"] / seconds,
            occupied_tokens_per_second=counts["occupied_tokens"] / seconds,
        )
    if args.phase_timing:
        phases = torch.tensor(
            [
                sum(result.pack_forward_ms),
                sum(result.pack_backward_ms),
                result.optimizer_events[0].elapsed_time(result.optimizer_events[1]),
            ],
            device=ACCEL,
        )
        dist.all_reduce(phases, op=dist.ReduceOp.MAX)
        row.update(zip(("forward_ms", "backward_ms", "optimizer_ms"), phases.tolist(), strict=True))
    if data_epoch is not None:
        row["data_epoch"] = data_epoch
    if run.rank == 0:
        with (run.output / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return row


def write_summary(args, cfg, run, measurements, wall_seconds, completed_step, parameters, persistence, ema, real_data):
    """summary.json on rank zero."""
    if not measurements:
        if not args.until_data_end:
            raise RuntimeError(
                "Dataset ended before any measured updates; reduce warmup or enlarge the explicit development snapshot"
            )
        # A preempted final update can resume directly at EOF. It must
        # complete successfully rather than trigger an endless restart.
        record = dict(
            completed_updates=completed_step,
            status="finite_dataset_complete",
            measured_updates=0,
            resumed_from=args.resume,
        )
        (run.output / "summary.json").write_text(json.dumps(record, indent=2))
        return
    total, trainable = parameters
    tokens_per_update = run.global_batch * args.sequence_length
    mean_seconds = sum(x["seconds"] for x in measurements) / len(measurements)
    ordered = sorted(x["seconds"] for x in measurements)
    summary = dict(
        world=run.world,
        **run.topology,
        method="CQ_IT",
        image_condition_queries=cfg.image_condition_queries,
        completed_updates=completed_step,
        steps=len(measurements),
        mean_seconds=mean_seconds,
        tokens_per_second=tokens_per_update / mean_seconds,
        tokens_per_second_per_card=tokens_per_update / mean_seconds / run.world,
        peak_memory_gib=max(x["peak_memory_gib"] for x in measurements),
        global_batch=run.global_batch,
        sequence_length=args.sequence_length,
        parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
        freeze_vision=cfg.freeze_vision,
        freeze_vision_merger=cfg.freeze_vision_merger,
        vision_max_pixels=cfg.vision_max_pixels,
        generation_max_pixels=cfg.generation_max_pixels,
        wall_seconds=wall_seconds,
        wall_tokens_per_second=tokens_per_update * len(measurements) / wall_seconds,
        median_seconds=statistics.median(ordered),
        p95_seconds=ordered[math.ceil(0.95 * len(measurements)) - 1],
        bucket_size=args.bucket_size,
        overlap_param_gather=not args.no_param_overlap,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        images=args.images,
        checkpoint_backbone=cfg.checkpoint_backbone,
        checkpoint_dit=cfg.checkpoint_dit,
        checkpoint_backbone_layers=cfg.checkpoint_backbone_layers,
        checkpoint_dit_layers=cfg.checkpoint_dit_layers,
        split_reference=args.split_reference,
        profiling=args.profile,
    )
    summary["validation_probes"] = args.validation_probes
    summary["validation_seconds"] = sum(
        e["seconds"] for e in persistence.validation_events if e["step"] > persistence.start_step
    )
    summary["qualitative_validation"] = bool(args.validation_suite)
    summary["eval_snapshot_dir"] = str(persistence.snapshot_root)
    summary["eval_snapshot_steps"] = [e["step"] for e in persistence.snapshot_events if not e["skipped"]]
    summary["eval_snapshot_seconds"] = sum(e["seconds"] for e in persistence.snapshot_events)
    if ema is not None:
        summary["renderer_ema"] = dict(**ema.metadata(), local_storage_bytes=ema.local_bytes)
    if args.phase_timing:
        for key in ("forward_ms", "backward_ms", "optimizer_ms"):
            summary[key] = statistics.mean(x[key] for x in measurements)
    if real_data:
        _real_data_summary(args, run, summary, measurements, wall_seconds)
    (run.output / "summary.json").write_text(json.dumps(summary, indent=2))


def _real_data_summary(args, run, summary, measurements, wall_seconds):
    summary.pop("images")
    summary["max_images_per_pack"] = args.max_images_per_pack
    renamed = (
        ("tokens_per_second", "capacity_slots_per_second"),
        ("tokens_per_second_per_card", "capacity_slots_per_second_per_card"),
        ("wall_tokens_per_second", "wall_capacity_slots_per_second"),
    )
    for old, new in renamed:
        summary[new] = summary.pop(old)
    summary["data_mode"] = args.data_mode
    for key, source in (
        ("body_tokens_per_second", "body_tokens"),
        ("occupied_tokens_per_second", "occupied_tokens"),
        ("images_per_second", "images_consumed"),
        ("generation_images_per_second", "generation_images"),
        ("documents_per_second", "documents_completed"),
        ("generation_tokens_per_second", "generation_tokens"),
    ):
        summary[key] = sum(x[source] for x in measurements) / wall_seconds
    total_seconds = sum(x["seconds"] for x in measurements)
    total_capacity = sum(x["active_microbatches"] for x in measurements) * args.sequence_length
    summary["capacity_slots_per_second"] = total_capacity / total_seconds
    summary["capacity_slots_per_second_per_card"] = total_capacity / total_seconds / run.world
    summary["wall_capacity_slots_per_second"] = total_capacity / wall_seconds
    for key in (
        "occupied_tokens",
        "body_tokens",
        "images_consumed",
        "generation_images",
        "flow_samples",
        "supervised_text_tokens",
        "condition_query_tokens",
        "backbone_processed_slots",
        "padding_tokens",
        "active_microbatches",
        "dit_microbatches",
    ):
        values = [x[key] for x in measurements]
        summary[key + "_per_update"] = dict(mean=statistics.mean(values), min=min(values), max=max(values))
    summary["fake_latents"] = args.latent_encoder == "fake"
    for key in ("data_wait_seconds", "data_prepare_seconds", "data_device_wait_seconds", "cpu_peak_rss_gib"):
        summary[key] = statistics.mean(x[key] for x in measurements)
