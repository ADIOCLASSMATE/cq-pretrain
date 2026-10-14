"""Training updates; numerical model and data transformations stay elsewhere."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from .loss import forward_loss, prepare_batch
from evaluations.validation import loss_metric_row, reduce_loss_metrics, loss_validation_options
from utils.checkpoint import save_checkpoint, load_checkpoint, export_weights
from utils.distributed import run_io_phase


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, accelerator, config,
                 *, validation_loader=None, ema=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.accelerator, self.config, self.ema = accelerator, config, ema
        self.global_step = 0
        self.output_dir = Path(config["training"]["output_dir"])
        self.schedule = tuple(config["data"]["schedule"])
        self.accumulation = int(config["training"]["gradient_accumulation_steps"])
        self.micro_step = 0

    def resume(self, path):
        restored = load_checkpoint(path, model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer, scheduler=self.scheduler, ema=self.ema,
            accelerator=self.accelerator, expected_config=self.config)
        state = restored.get("task_state") or {}
        if tuple(state.get("schedule", self.schedule)) != self.schedule:
            raise ValueError("Checkpoint task schedule differs from this run")
        if int(state.get("micro_step", 0)) != 0:
            raise ValueError("Only complete optimizer windows can be resumed")
        self.train_loader.load_state_dict(restored["data_state"])
        self.global_step = int(restored["global_step"])

    def save(self):
        if self.micro_step:
            raise RuntimeError("Checkpoint must be saved at an optimizer boundary")
        path = self.output_dir / f"checkpoint-{self.global_step}"
        save_checkpoint(path, model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer, scheduler=self.scheduler, global_step=self.global_step,
            data_state=self.train_loader.state_dict(),
            task_state={"schedule": list(self.schedule), "micro_step": 0},
            ema=self.ema, accelerator=self.accelerator, config=self.config)
        return path

    def validate(self):
        from evaluations.validation import validate_losses
        if self.validation_loader is None:
            return {}
        def validation_forward(model, batch):
            batch = prepare_batch(batch, self.accelerator.device, self.config,
                image_input_dtype=model.backbone.image_token_embedder.z_proj.weight.dtype)
            with self.accelerator.autocast():
                return forward_loss(model, batch, self.config)
        return validate_losses(self.accelerator.unwrap_model(self.model), self.validation_loader,
            validation_forward, schedule=self.schedule, device=self.accelerator.device,
            **loss_validation_options(self.config), distributed=True)

    def _write_log(self, payload):
        if not self.accelerator.is_main_process:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"step": self.global_step, **payload}, ensure_ascii=False)
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def fit(self, *, stop_after_steps=None):
        settings = self.config["training"]
        limit = int(settings["max_train_steps"])
        if stop_after_steps is not None:
            limit = min(limit, int(stop_after_steps))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        iterator = iter(self.train_loader)
        active = tuple(dict.fromkeys(self.schedule))
        metric_dtype = torch.float32 if self.accelerator.device.type == "npu" else torch.float64
        totals = torch.zeros(len(active), 8, device=self.accelerator.device, dtype=metric_dtype)
        started = time.monotonic()
        while self.global_step < limit:
            try:
                source_batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("Training loader ended before the requested optimizer boundary") from exc
            task = source_batch.get("task", "t2i")
            expected = self.schedule[(self.global_step * self.accumulation + self.micro_step) % len(self.schedule)]
            if task != expected:
                raise RuntimeError(f"Task schedule mismatch: expected {expected}, received {task}")
            unwrapped = self.accelerator.unwrap_model(self.model)
            batch = prepare_batch(source_batch, self.accelerator.device, self.config,
                image_input_dtype=unwrapped.backbone.image_token_embedder.z_proj.weight.dtype)
            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    result = forward_loss(self.model, batch, self.config)
                finite = torch.isfinite(result["loss"]).to(dtype=torch.int32)
                finite = self.accelerator.reduce(finite, reduction="sum")
                if int(finite) != self.accelerator.num_processes:
                    raise FloatingPointError("At least one rank produced a non-finite loss")
                self.accelerator.backward(result["loss"])
                totals[active.index(task)] += loss_metric_row(result, device=totals.device, dtype=totals.dtype)
                self.micro_step += 1
                if self.accelerator.sync_gradients:
                    if self.micro_step != self.accumulation:
                        raise RuntimeError("Unexpected gradient accumulation boundary")
                    if str(self.accelerator.distributed_type).split(".")[-1] != "DEEPSPEED" and float(settings["max_grad_norm"]) > 0:
                        norm = self.accelerator.clip_grad_norm_(self.model.parameters(), float(settings["max_grad_norm"]))
                        if not bool(torch.isfinite(norm)):
                            raise FloatingPointError("Non-finite gradient norm")
                    self.optimizer.step()
                    if self.accelerator.optimizer_step_was_skipped:
                        raise FloatingPointError("Optimizer skipped an update; resume from the last complete checkpoint")
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self.micro_step = 0
                    if self.ema is not None:
                        self.ema.maybe_update(self.global_step)
            self.train_loader.mark_consumed(source_batch)
            if self.micro_step:
                continue
            if self.global_step % int(settings["log_every"]) == 0:
                reduced = self.accelerator.reduce(totals, reduction="sum")
                logs = {"seconds": time.monotonic() - started}
                metrics = reduce_loss_metrics(dict(zip(active, reduced.cpu())), schedule=self.schedule)
                logs.update({key.replace("val/", "train/", 1): value for key, value in metrics.items()})
                logs["lr"] = self.scheduler.get_last_lr()
                self._write_log(logs)
                totals.zero_()
                started = time.monotonic()
            if int(settings["val_every"]) and self.global_step % int(settings["val_every"]) == 0:
                self._write_log(self.validate())
            if int(settings["save_every"]) and self.global_step % int(settings["save_every"]) == 0:
                self.save()
        final = self.output_dir / f"checkpoint-{self.global_step}"
        if not final.exists():
            final = self.save()
        def export_final_weights():
            export_weights(self.output_dir / "weights.pt", model=self.accelerator.unwrap_model(self.model), config=self.config)
            if self.ema is not None and self.ema.started:
                export_weights(self.output_dir / "weights-ema.pt", ema_checkpoint=final / "ema", config=self.config)
        run_io_phase(self.accelerator, export_final_weights, description="Export final weights", main_process_only=True)
        self.accelerator.wait_for_everyone()
        return {"global_step": self.global_step, "checkpoint": str(final)}
