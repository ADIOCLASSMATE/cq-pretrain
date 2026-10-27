"""Training updates; numerical model and data transformations stay elsewhere."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext

import torch

from .loss import forward_loss, prepare_batch, LazyVideoCodec, global_loss_weights
from evaluations.validation import (
    loss_metric_row,
    reduce_loss_metrics,
    loss_validation_options,
)
from utils.checkpoint import save_checkpoint, load_checkpoint, export_weights
from utils.distributed import run_io_phase
from configs.config import resolve_output_paths
from utils.checkpoint_catalog import commit_and_prune
from utils.checkpoint import validate_checkpoint
from utils.run_logging import RunLogger, resolve_attempt


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        accelerator,
        config,
        *,
        validation_loader=None,
        ema=None,
        logger=None,
    ):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.accelerator, self.config, self.ema = accelerator, config, ema
        self.codec = LazyVideoCodec(config, accelerator.device)
        self.global_step = 0
        paths = resolve_output_paths(config)
        self.output_dir = paths["output_dir"]
        self.checkpoint_dir = paths["checkpoint_dir"]
        if logger is None:
            attempt = run_io_phase(
                accelerator,
                lambda: str(resolve_attempt(config)),
                description="Create log attempt",
                main_process_only=True,
            )
            logger = RunLogger(attempt, enabled=accelerator.is_main_process)
        self.logger = logger
        self.schedule = tuple(config["data"]["schedule"])
        self.accumulation = int(config["training"]["gradient_accumulation_steps"])
        self.micro_step = 0

    def resume(self, path):
        restored = load_checkpoint(
            path,
            model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            accelerator=self.accelerator,
            expected_config=self.config,
        )
        state = restored.get("task_state") or {}
        if tuple(state.get("schedule", self.schedule)) != self.schedule:
            raise ValueError("Checkpoint task schedule differs from this run")
        if int(state.get("micro_step", 0)) != 0:
            raise ValueError("Only complete optimizer windows can be resumed")
        run_io_phase(
            self.accelerator,
            lambda: self.train_loader.load_state_dict(restored["data_state"]),
            description="Restore consumed data cursor",
        )
        self.global_step = int(restored["global_step"])
        return restored

    def save(self):
        try:
            return self._save()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run_io_phase(
                self.accelerator,
                lambda: self.logger.record(
                    "checkpoints.jsonl",
                    {
                        "event": "save_failed",
                        "step": self.global_step,
                        "error": error,
                    },
                ),
                description="Record checkpoint failure",
                main_process_only=True,
            )
            raise

    def _save(self):
        if self.micro_step:
            raise RuntimeError("Checkpoint must be saved at an optimizer boundary")
        path = self.checkpoint_dir / f"checkpoint-{self.global_step}"
        started = time.monotonic()
        save_checkpoint(
            path,
            model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            data_state=self.train_loader.state_dict(),
            task_state={"schedule": list(self.schedule), "micro_step": 0},
            ema=self.ema,
            accelerator=self.accelerator,
            config=self.config,
        )

        def publish_catalog():
            removed = commit_and_prune(
                path, self.config["training"].get("keep_checkpoints", 3)
            )
            self.logger.record(
                "checkpoints.jsonl",
                {
                    "event": "saved",
                    "step": self.global_step,
                    "path": str(path),
                    "seconds": time.monotonic() - started,
                    "removed": removed,
                    "keep_checkpoints": self.config["training"].get(
                        "keep_checkpoints", 3
                    ),
                },
            )

        run_io_phase(
            self.accelerator,
            publish_catalog,
            description="Publish catalog and prune",
            main_process_only=True,
        )
        return path

    def validate(self):
        from evaluations.validation import validate_losses

        if self.validation_loader is None:
            return {}

        def validation_forward(model, batch):
            batch = prepare_batch(
                batch, self.accelerator.device, self.config, codec=self.codec
            )
            with self.accelerator.autocast():
                return forward_loss(model, batch, self.config)

        context = self.accelerator.evaluation_parameters() if getattr(self.accelerator, "is_mcore", False) else nullcontext()
        with context:
            return validate_losses(
                self.accelerator.unwrap_model(self.model), self.validation_loader,
                validation_forward, schedule=self.schedule, device=self.accelerator.device,
                **loss_validation_options(self.config), distributed=True,
            )

    def _write_log(self, payload):
        def write():
            row = self.logger.record(
                "metrics.jsonl", {"step": self.global_step, **payload}
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)

        run_io_phase(
            self.accelerator, write, description="Write metrics", main_process_only=True
        )

    def fit(self, *, stop_after_steps=None):
        settings = self.config["training"]
        limit = int(settings["max_train_steps"])
        if stop_after_steps is not None:
            limit = min(limit, int(stop_after_steps))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        iterator = iter(self.train_loader)
        active = tuple(dict.fromkeys(self.schedule))
        metric_dtype = (
            torch.float32 if self.accelerator.device.type == "npu" else torch.float64
        )
        totals = torch.zeros(
            len(active), 8, device=self.accelerator.device, dtype=metric_dtype
        )
        started = time.monotonic()
        window_start = self.global_step

        def flush_metrics():
            nonlocal started, window_start
            if window_start == self.global_step:
                return
            reduced = self.accelerator.reduce(totals, reduction="sum")
            logs = {
                "seconds": time.monotonic() - started,
                "step_start": window_start + 1,
                "step_end": self.global_step,
                "optimizer_updates": self.global_step - window_start,
            }
            metrics = reduce_loss_metrics(
                dict(zip(active, reduced.cpu())), schedule=self.schedule
            )
            logs.update(
                {
                    key.replace("val/", "train/", 1): value
                    for key, value in metrics.items()
                }
            )
            logs["lr"] = self.scheduler.get_last_lr()
            counts = reduced.sum(0)
            logs["train/objective_global"] = sum(
                float(self.config["model"][key]) * float(counts[numerator] / counts[denominator])
                if counts[denominator] > 0 else 0.0
                for key, numerator, denominator in (("lambda_text", 4, 5), ("lambda_video", 6, 7))
            )
            if getattr(self.accelerator, "is_mcore", False):
                logs["grad_norm"] = self.accelerator.last_grad_norm
                logs["absent_final_microbatch_gradients"] = self.accelerator.last_absent_gradients
            self._write_log(logs)
            totals.zero_()
            window_start = self.global_step
            started = time.monotonic()

        while self.global_step < limit:
            try:
                if self.micro_step == 0:
                    # Count supervision over the whole optimizer window on CPU.
                    # Fetching does not commit the resume cursor or build graphs.
                    window = [next(iterator) for _ in range(self.accumulation)]
                    weights = global_loss_weights(window, self.accelerator)
                source_batch = window[self.micro_step]
            except StopIteration as exc:
                raise RuntimeError(
                    "Training loader ended before the requested optimizer boundary"
                ) from exc
            task = source_batch["task"]
            expected = self.schedule[
                (self.global_step * self.accumulation + self.micro_step)
                % len(self.schedule)
            ]
            if task != expected:
                raise RuntimeError(
                    f"Task schedule mismatch: expected {expected}, received {task}"
                )
            batch = prepare_batch(
                source_batch, self.accelerator.device, self.config, codec=self.codec
            )
            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    result = forward_loss(self.model, batch, self.config,
                                          loss_weights=weights[self.micro_step])
                finite = torch.isfinite(result["loss"]).to(dtype=torch.int32)
                finite = self.accelerator.reduce(finite, reduction="sum")
                if int(finite) != self.accelerator.num_processes:
                    raise FloatingPointError(
                        "At least one rank produced a non-finite loss"
                    )
                # CPU Accelerate divides by GA; MCore uses the already
                # globally normalized loss directly. Do not divide it twice.
                backward_loss = result["loss"]
                if not getattr(self.accelerator, "is_mcore", False):
                    backward_loss = backward_loss * self.accumulation
                self.accelerator.backward(backward_loss)
                totals[active.index(task)] += loss_metric_row(
                    result, device=totals.device, dtype=totals.dtype
                )
                self.micro_step += 1
                if self.accelerator.sync_gradients:
                    if self.micro_step != self.accumulation:
                        raise RuntimeError("Unexpected gradient accumulation boundary")
                    if (
                        str(self.accelerator.distributed_type).split(".")[-1]
                        not in {"DEEPSPEED", "MEGATRON"}
                        and float(settings["max_grad_norm"]) > 0
                    ):
                        norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), float(settings["max_grad_norm"])
                        )
                        if not bool(torch.isfinite(norm)):
                            raise FloatingPointError("Non-finite gradient norm")
                    self.optimizer.step()
                    if self.accelerator.optimizer_step_was_skipped:
                        raise FloatingPointError(
                            "Optimizer skipped an update; resume from the last complete checkpoint"
                        )
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self.micro_step = 0
                    if self.ema is not None:
                        if getattr(self.accelerator, "is_mcore", False):
                            self.accelerator.synchronize_parameters()
                        self.ema.maybe_update(self.global_step)
            self.train_loader.mark_consumed(source_batch)
            if self.micro_step:
                continue
            if self.global_step % int(settings["log_every"]) == 0:
                flush_metrics()
            if (
                int(settings["val_every"])
                and self.global_step % int(settings["val_every"]) == 0
            ):
                self._write_log(self.validate())
            if (
                int(settings["save_every"])
                and self.global_step % int(settings["save_every"]) == 0
            ):
                flush_metrics()
                self.save()
        flush_metrics()
        final = self.checkpoint_dir / f"checkpoint-{self.global_step}"
        exists = run_io_phase(
            self.accelerator,
            final.exists,
            description="Locate final checkpoint",
            main_process_only=True,
        )
        if not exists:
            final = self.save()
        else:
            run_io_phase(
                self.accelerator,
                lambda: validate_checkpoint(final, expected_config=self.config),
                description="Validate final checkpoint",
                main_process_only=True,
            )

        if getattr(self.accelerator, "is_mcore", False):
            self.accelerator.synchronize_parameters()

        def export_final_weights():
            export_weights(
                self.checkpoint_dir / "exports" / "weights.pt",
                model=self.accelerator.unwrap_model(self.model),
                config=self.config,
            )
            if self.ema is not None and self.ema.started:
                export_weights(
                    self.checkpoint_dir / "exports" / "weights-ema.pt",
                    ema_checkpoint=final / "ema",
                    config=self.config,
                )

        run_io_phase(
            self.accelerator,
            export_final_weights,
            description="Export final weights",
            main_process_only=True,
        )
        # run_io_phase already synchronizes all ranks and propagates export errors.
        return {"global_step": self.global_step, "checkpoint": str(final)}
