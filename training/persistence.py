"""Restart checkpoints, inference-only evaluation snapshots and in-training validation."""

from contextlib import nullcontext
import json
import time

import torch.distributed as dist

import distributed.accel as accel
import utils.checkpoint as checkpoint_io

DEV = accel.device_module()


class Persistence:
    def __init__(self, args, cfg, run, model, wrapped, optimizer, ema, state, record, loaded, roots):
        self.args, self.cfg, self.run = args, cfg, run
        self.model, self.wrapped, self.optimizer, self.ema = model, wrapped, optimizer, ema
        self.state, self.record, self.loaded = state, record, loaded
        self.checkpoint_root, self.snapshot_root = roots
        self.pipeline = None
        self.last_saved_step = -1
        self.snapshot_events = []
        self.validation_events = []
        self._warned_validation_assets = False
        self.validation_clock = run.output / "validation_clock.json"
        self.last_validation = time.time()
        if run.rank == 0 and self.validation_clock.exists():
            self.last_validation = json.loads(self.validation_clock.read_text())["time"]

    # Restart checkpoints ---------------------------------------------------

    def save_checkpoint(self, step):
        from utils.checkpoint_catalog import durable_json, prune

        begin = time.perf_counter()
        self.wrapped.start_param_sync(force_sync=True)
        path = self.checkpoint_root / f"step_{step:07d}"
        checkpoint_io.save(
            path,
            self.model,
            self.optimizer,
            step,
            self.cfg,
            self.run.topology,
            self.state,
            data_state=self.pipeline.state_dict() if self.pipeline else None,
            ema=self.ema,
        )
        if self.run.rank == 0:
            removed = prune(self.checkpoint_root, self.args.keep_checkpoints)
            event = dict(
                step=step,
                path=str(path),
                keep=self.args.keep_checkpoints,
                seconds=time.perf_counter() - begin,
                removed=removed,
            )
            durable_json(self.checkpoint_root / "latest.json", event)
            with (self.run.output / "checkpoints.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(dict(checkpoint_committed=event)), flush=True)
        dist.barrier()
        self.last_saved_step = step

    def save_if_unsaved(self, step):
        """Epoch boundaries and normal completion: save unless this step already is."""
        if step > self.last_saved_step and (self.args.save_every or self.args.save_final):
            self.save_checkpoint(step)

    # Evaluation snapshots --------------------------------------------------

    def save_eval_snapshot(self, step):
        from utils.eval_snapshot import save

        record, state = self.record, self.state
        provenance = dict(
            source_sha256=record["source_sha256"],
            dependencies=record["dependencies"],
            runtime={key: record[key] for key in ("torch", "stack", "cuda")},
            assets={key: state[key] for key in ("qwen35_assets", "vision_assets", "wan_assets") if key in state},
            inference_inputs={
                key: state.get("data", {}).get(key) for key in ("tokenizer", "latent_encoder", "vae_dtype")
            },
            resumed_from=self.args.resume,
            initial_weights=self.loaded,
        )
        event = save(
            self.snapshot_root / f"step_{step:07d}",
            self.model,
            self.wrapped,
            step,
            self.cfg,
            self.run.topology,
            ema=self.ema,
            provenance=provenance,
        )
        self.snapshot_events.append(event)
        if self.run.rank == 0:
            with (self.run.output / "eval_snapshots.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(dict(eval_snapshot=event)), flush=True)

    # In-training validation ------------------------------------------------

    def validation_due(self, step):
        args = self.args
        due = bool(args.validate_every and step % args.validate_every == 0)
        if args.validate_every_hours:
            # Rank zero owns the wall clock; every rank must take the same branch.
            elapsed = time.time() - self.last_validation
            flag = [self.run.rank == 0 and elapsed >= args.validate_every_hours * 3600]
            dist.broadcast_object_list(flag, src=0)
            due = due or flag[0]
        return due

    def missing_validation_assets(self):
        """Names of suite files that are absent, or None when the suite is usable.

        Every rank reads the same filesystem and so reaches the same answer,
        which is why this can gate a collective: either all of them validate or
        none do. It runs before any collective for the same reason.

        Validation is a regression canary. It used to abort the update instead,
        and the platform restarted the job and replayed from the last
        checkpoint -- losing every step since it, once per validation interval,
        silently. A canary must never be able to stop training.
        """
        from evaluations.sampling import suite_assets

        try:
            missing = [str(p) for p in suite_assets(self.args.validation_suite) if not p.is_file()]
        except (OSError, ValueError, KeyError, TypeError) as error:
            missing = [f"unreadable suite: {error}"]
        if not missing:
            return None
        if self.run.rank == 0 and not self._warned_validation_assets:
            print(
                json.dumps(dict(validation_skipped=dict(suite=str(self.args.validation_suite), missing=missing[:8]))),
                flush=True,
            )
        self._warned_validation_assets = True
        return missing

    def validate(self, step, preparer):
        if not self.args.validation_suite:
            return
        output, rank = self.run.output, self.run.rank
        report = output / "validation" / f"step_{step:07d}"
        # A preempted job restarts at a step it may already have reported.
        done = [rank == 0 and (report / "results.json").exists()]
        dist.broadcast_object_list(done, src=0)
        if done[0]:
            return
        if self.missing_validation_assets() is not None:
            return
        from evaluations.sampling import run_suite

        begin = time.perf_counter()
        self.wrapped.start_param_sync(force_sync=True)
        hooks = self.wrapped.use_forward_hook
        if hooks:
            self.wrapped.disable_forward_pre_hook(param_sync=False)
        weights = dict(backbone="online", renderer="ema" if self.ema is not None else "online")
        if self.ema is not None:
            weights["ema"] = self.ema.metadata()
        try:
            with self.ema.average_parameters() if self.ema is not None else nullcontext():
                event = run_suite(self.model, self.args.validation_suite, report, preparer, weight_sources=weights)
        finally:
            if hooks:
                self.wrapped.enable_forward_pre_hook()
        DEV.synchronize()
        dist.barrier()
        event["sampling_seconds"] = event["seconds"]
        event["seconds"] = time.perf_counter() - begin
        event["step"] = step
        self.validation_events.append(event)
        if rank == 0:
            from utils.file_sync import atomic_json

            result = json.loads((report / "results.json").read_text())
            result["end_to_end_seconds"] = event["seconds"]
            atomic_json(report / "results.json", result)
            with (output / "validation.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            self.last_validation = time.time()
            atomic_json(self.validation_clock, dict(time=self.last_validation, step=step))
            print(json.dumps(dict(validation=event)), flush=True)

    # Per-step boundary -----------------------------------------------------

    def after_step(self, step):
        """Saves once update `step` is committed. The profiler step and validation follow."""
        args = self.args
        if (args.save_every and step % args.save_every == 0) or (args.save_first and step == 1):
            self.save_checkpoint(step)
        if args.eval_snapshot_every and step % args.eval_snapshot_every == 0:
            self.save_eval_snapshot(step)

    def after_step_validation(self, step, preparer):
        if self.validation_due(step):
            self.validate(step, preparer)
