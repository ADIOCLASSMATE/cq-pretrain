"""Disjoint parameter roles and the source warmup/stable/cosine schedule."""

from __future__ import annotations

import math

import torch


def parameter_role(name):
    if name.startswith("generator.condition_proj."):
        return "projector"
    if name.startswith("generator."):
        return "generator"
    if name.startswith("backbone.vision_encoder."):
        return "vision"
    if name in {"backbone.embed_tokens.weight", "lm_head.weight"}:
        return "embedding"
    return "backbone"


def apply_trainable_scope(model, settings):
    switches = dict(
        backbone=settings["train_language"],
        embedding=settings["train_language"],
        generator=settings["train_wan"],
        projector=settings["train_condition_proj"],
        vision=not settings["freeze_vision"],
    )
    for name, param in model.named_parameters():
        param.requires_grad_(switches[parameter_role(name)])


def parameter_groups(model, config):
    settings = config["optimizer"]
    groups = {}
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            raise RuntimeError(f"Duplicate optimizer parameter: {name}")
        seen.add(id(param))
        role = parameter_role(name)
        no_decay = role in {"projector", "embedding"} or any(
            fragment in name
            for fragment in (
                "bias",
                "layer_norm.weight",
                "layernorm.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "norm.weight",
            )
        )
        decay = (
            0.0
            if no_decay
            else float(
                settings[
                    "generator_weight_decay" if role == "generator" else "weight_decay"
                ]
            )
        )
        lr = float(settings[f"{role}_lr"])
        key = (role, lr, decay)
        group = groups.setdefault(
            key,
            {
                "params": [],
                "param_names": [],
                "lr": lr,
                "weight_decay": decay,
                "role": role,
            },
        )
        group["params"].append(param)
        group["param_names"].append(name)
    if not groups:
        raise ValueError("No trainable parameters")
    return list(groups.values())


def build_optimizer_and_scheduler(model, config):
    settings = config["optimizer"]
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config),
        betas=tuple(settings["betas"]),
        eps=float(settings["eps"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: wsd_multiplier(
            step,
            int(settings["warmup_steps"]),
            int(settings["decay_steps"]),
            int(config["training"]["max_train_steps"]),
            float(settings["min_lr_ratio"]),
        ),
    )
    return optimizer, scheduler


def wsd_multiplier(step, warmup, decay, total, minimum):
    if step < warmup:
        return step / max(1, warmup)
    if step < total - decay:
        return 1.0
    progress = (step - (total - decay)) / max(1, decay)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))
