"""Small YAML configuration loader with explicit training invariants."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import yaml


DEFAULTS = {
    "model": {"backbone_type": "cq_it", "generator_type": "z2", "lambda_text": 0.05,
              "lambda_image": 1.0, "image_flow_batch_mul": 4,
              "image_flow_time_sampling": "logit_normal", "image_flow_logit_mean": 0.0,
              "image_flow_logit_std": 1.0, "image_flow_time_uniform_mix": 0.1,
              "image_flow_time_eps": 1e-5, "image_uncond_prob": 0.1,
              "image_mask_init": "copy_text"},
    "data": {"schedule": ["text", "t2i", "text", "i2t"]},
    "training": {"output_dir": "output/umm", "max_train_steps": 31800,
                 "gradient_accumulation_steps": 4, "mixed_precision": "bf16",
                 "seed": 42, "max_grad_norm": 1.0, "batch_size": 16,
                 "save_every": 2000, "log_every": 10, "val_every": 0,
                 "validation_max_batches": None, "use_ema": True,
                 "ema_decay": 0.9997, "ema_update_after_step": 0,
                 "ema_shard_chunk_numel": 4194304, "trainable_scope": "full",
                 "dataloader_workers": 0},
    "optimizer": {"backbone_lr": 3e-4, "generator_lr": 5e-4, "projector_lr": 5e-4,
                  "embedding_lr": 3e-4, "weight_decay": 0.01,
                  "generator_weight_decay": 0.01, "betas": [0.9, 0.95], "eps": 1e-8,
                  "warmup_steps": 199, "decay_steps": 7950, "min_lr_ratio": 0.1},
    "sampling": {"steps": 10, "solver": "heun", "cfg": 3.5, "temperature": 1.0,
                 "seed": 42},
    "evaluation": {"seed": 424242, "loss_image_per_class": 2}, "codec": {},
}


def merge_config(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, overrides=None):
    """Read one complete YAML and optional dotted key=value CLI overrides."""
    with Path(path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    config = merge_config(DEFAULTS, values)
    for override in overrides or ():
        if "=" not in override:
            raise ValueError("Overrides must use section.key=value")
        key, raw_value = override.split("=", 1)
        parts = key.split(".")
        if parts[0] not in DEFAULTS or len(parts) < 2:
            raise ValueError(f"Unknown override section: {key}")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Override traverses a scalar: {key}")
        target[parts[-1]] = yaml.safe_load(raw_value)
    validate_config(config)
    return config


def validate_config(config):
    train, model, optimizer = config["training"], config["model"], config["optimizer"]
    schedule = config["data"]["schedule"]
    if not schedule or set(schedule) - {"text", "t2i", "i2t", "joint"}:
        raise ValueError("data.schedule must contain text, t2i, i2t and/or joint")
    if model["backbone_type"] != "cq_it" or model["generator_type"] != "z2":
        raise ValueError("This release implements the selected CQ_IT + Z2 baseline")
    if model.get("image_mask_init", "copy_text") not in {"copy_text", "random"}:
        raise ValueError("image_mask_init must be copy_text or random")
    accumulation = int(train["gradient_accumulation_steps"])
    if accumulation < 1 or (len(set(schedule)) > 1 and accumulation != len(schedule)):
        raise ValueError("Mixed training needs one complete source schedule per accumulation window")
    if train["mixed_precision"] not in {"no", "bf16"}:
        raise ValueError("Use FP32 or BF16 with FP32 gradient accumulation")
    if int(train["max_train_steps"]) < 1 or int(train["log_every"]) < 1:
        raise ValueError("max_train_steps and log_every must be positive")
    if any(int(train[name]) < 0 for name in ("save_every", "val_every", "ema_update_after_step")):
        raise ValueError("Save, validation and EMA start steps must be nonnegative")
    if train["trainable_scope"] not in {"full", "generator"}:
        raise ValueError("trainable_scope must be full or generator")
    if train["trainable_scope"] == "generator" and set(schedule) != {"t2i"}:
        raise ValueError("Generator-only optimization requires a T2I-only schedule")
    if not 0 <= float(train["ema_decay"]) < 1:
        raise ValueError("ema_decay must be in [0,1)")
    for task in set(schedule):
        names = ("lambda_text", "lambda_image") if task == "joint" else (
            "lambda_image" if task == "t2i" else "lambda_text",)
        for name in names:
            if not math.isfinite(float(model[name])) or float(model[name]) <= 0:
                raise ValueError(f"Active task {task} requires positive finite {name}")
    for name in ("image_uncond_prob", "image_flow_time_uniform_mix"):
        if not 0 <= float(model[name]) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if int(model["image_flow_batch_mul"]) < 1 or not 0 <= float(model["image_flow_time_eps"]) < 0.5:
        raise ValueError("Invalid RF Monte Carlo count or time epsilon")
    if model["image_flow_time_sampling"] not in {"uniform", "logit_normal"}:
        raise ValueError("Unknown flow time sampling distribution")
    warmup, decay, steps = (int(optimizer["warmup_steps"]), int(optimizer["decay_steps"]), int(train["max_train_steps"]))
    if min(warmup, decay) < 0 or warmup + decay > steps:
        raise ValueError("WSD warmup_steps + decay_steps must fit max_train_steps")
    if not 0 <= float(optimizer["min_lr_ratio"]) <= 1:
        raise ValueError("min_lr_ratio must be in [0,1]")
    if config["sampling"]["solver"] not in {"euler", "heun"} or int(config["sampling"]["steps"]) < 1:
        raise ValueError("Sampling requires Euler/Heun and positive steps")
    evaluation = config["evaluation"]
    if int(evaluation["seed"]) < 0:
        raise ValueError("evaluation.seed must be nonnegative")
    per_class = evaluation.get("loss_image_per_class")
    if per_class is not None and (isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 1):
        raise ValueError("evaluation.loss_image_per_class must be positive or null for the full split")


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
