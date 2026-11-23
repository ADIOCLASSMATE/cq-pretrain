"""Distributed CQ + Wan pretraining on a published release or synthetic controls.

The pieces live in their own modules:
  options      command line, learning-rate schedule, loss scales
  configure    config resolution from options, model YAML and data release
  model_setup  process group, restart selection, model and optimizer
  contract     the exact resume contract (training_state) and run.json
  pipeline     per-epoch data loaders and DP-agreed microbatch groups
  trainer      one optimizer update
  persistence  restart checkpoints, evaluation snapshots, validation
  metrics      step logs and summary.json
"""

import json
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

import distributed.accel as accel
import utils.checkpoint as checkpoint_io
from data.batch import fake_batch
import distributed.megatron as P
from training import configure, contract, metrics, model_setup
from training.options import data_loss_scales, learning_rate, parse_args  # noqa: F401  (public API)
from training.persistence import Persistence
from training.pipeline import DataPipeline
from training.trainer import Trainer

ACCEL = accel.accel_type()
DEV = accel.device_module()


def _restore(args, run, model, optimizer, cfg, state, ema, real_data):
    """Load the selected restart checkpoint. Returns (start_step, saved data cursor)."""
    resumed = {} if real_data else None
    start_step = 0
    if args.resume:
        start_step = checkpoint_io.load(
            args.resume, model, optimizer, cfg, run.topology, state, data_state_out=resumed, ema=ema
        )
    if run.rank == 0:
        restored = []
        if args.resume:
            restored = ["model", "optimizer", "FP32 masters", "CPU/NPU RNG", "consumed data cursor"]
            if ema is not None:
                restored.append("renderer FP32 EMA and update counter")
        print(json.dumps(dict(resume=dict(path=args.resume, step=start_step, restored=restored))), flush=True)
    return start_step, resumed


def _remaining_budget(args, start_step):
    """Steps left under --total-steps. True when the budget was already met."""
    if args.total_steps is None:
        return False
    if args.total_steps < 1:
        raise ValueError("total_steps must be positive")
    args.steps = max(0, args.total_steps - start_step)
    if not args.steps:
        return True
    args.warmup = min(args.warmup, args.steps - 1)
    return False


def _profile_root(run):
    return run.output / "profile" / f"rank{run.rank}"


def _profiler(args, run, complete):
    if not (args.profile and run.rank in run.profile_ranks and not complete):
        return nullcontext(), False
    import ops.profile as profiling

    return profiling.profiler(_profile_root(run), args.warmup), True


def main():
    args = parse_args()
    if args.profile:
        import ops.profile as profiling

        # A profiler swallows callback-construction errors; reject an output
        # path its parser cannot accept before any weight is built.
        profiling.check_output_path(args.output)
    cfg, data, assets = configure.resolve(args)
    run = model_setup.initialize(args, cfg)
    roots = model_setup.storage_roots(args, run.output)
    model_setup.select_restart(args, run.rank, roots[0])
    model, loaded, startup = model_setup.build_model(args, cfg, run)
    parameters = model_setup.count_parameters(model, args.tp)
    wrapped, optimizer, ema, scope = model_setup.build_optimizer(args, cfg, model)
    state = contract.training_state(args, cfg, ema, assets, data)
    start_step, resumed = _restore(args, run, model, optimizer, cfg, state, ema, data is not None)
    complete = _remaining_budget(args, start_step)

    # Different DP replicas see different data. TP ranks must see identical data.
    pipeline = synthetic = None
    if data is not None and not complete:
        pipeline = DataPipeline(args, cfg, data, resumed)
    elif not complete:
        synthetic = fake_batch(
            cfg, args.sequence_length, args.micro_batch, args.images, seed=args.seed + P.dp_rank(), device=ACCEL
        )
    record, host = contract.run_record(
        args, cfg, run, model, ema, state, assets, data, pipeline, loaded, startup, parameters, scope
    )
    trainer = Trainer(args, cfg, run, model, wrapped, optimizer, ema, pipeline, synthetic)
    persistence = Persistence(args, cfg, run, model, wrapped, optimizer, ema, state, record, loaded, roots)
    persistence.pipeline = pipeline
    persistence.start_step = start_step
    persistence.last_saved_step = start_step if args.resume else -1
    preparer = pipeline.preparer if pipeline else None
    profile, profile_this_rank = _profiler(args, run, complete)
    DEV.synchronize()

    # A restart checkpoint can commit just before its evaluation export is
    # interrupted. Revisit that boundary before consuming the next batch.
    if args.eval_snapshot_every and start_step > 0 and start_step % args.eval_snapshot_every == 0:
        persistence.save_eval_snapshot(start_step)
    if complete:
        if run.rank == 0:
            print(json.dumps(dict(status="budget_already_complete", step=start_step)), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return
    if args.validate_first:
        persistence.validate(start_step, preparer)

    completed_step = start_step
    measurements = []
    measured_begin = None
    begin = time.perf_counter()
    with profile as profiler:
        for local_step in range(args.steps):
            if local_step == args.warmup:
                DEV.synchronize()
                dist.barrier()
                measured_begin = time.perf_counter()
            result = trainer.step(
                start_step + local_step,
                completed_step,
                lambda step=completed_step: persistence.save_if_unsaved(step),
            )
            begin = trainer.last_begin
            if result is None:
                args.steps = local_step
                break
            completed_step = start_step + local_step + 1
            warmup = local_step < args.warmup
            data_epoch = pipeline.epoch if pipeline and args.data_epochs > 1 else None
            if pipeline:
                metrics.write_rank_timing(args, run, completed_step, result)
            metrics.write_rank_step(args, run, host, completed_step, result, warmup, data_epoch)
            row = metrics.global_row(args, cfg, run, completed_step, result, warmup, ema, data_epoch)
            if not warmup:
                measurements.append(row)
            if args.validation_probes:
                torch.save(
                    checkpoint_io.validation_probes(model, optimizer),
                    run.output / f"probe_rank_{run.rank:05d}_step_{completed_step:07d}.pt",
                )
            persistence.after_step(completed_step)
            if profile_this_rank:
                profiler.step()
            persistence.after_step_validation(completed_step, preparer)
        events = persistence.validation_events
        if args.validate_final and (not events or events[-1]["step"] != completed_step):
            persistence.validate(completed_step, preparer)
        if args.save_final and completed_step > persistence.last_saved_step:
            persistence.save_checkpoint(completed_step)
        # Includes gradient zeroing, metric collectives and logging/checkpoints.
        DEV.synchronize()
        wall = torch.tensor(time.perf_counter() - (measured_begin or begin), device=ACCEL)
        dist.all_reduce(wall, op=dist.ReduceOp.MAX)
        wall_seconds = wall.item()
    if run.rank == 0:
        metrics.write_summary(
            args,
            cfg,
            run,
            measurements,
            wall_seconds,
            completed_step,
            parameters,
            persistence,
            ema,
            pipeline is not None,
        )
    if pipeline:
        pipeline.close(run.output, run.rank)
    dist.barrier()
    dist.destroy_process_group()
    if profile_this_rank:
        # Parsing a trace is minutes of CPU work on one rank. Only here, with
        # every collective finished, can it not stall the others into a timeout.
        import ops.profile as profiling

        profiling.finalize(_profile_root(run))


if __name__ == "__main__":
    main()
