"""Load configuration and dependencies, then hand control to Trainer."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from configs.config import load_config, save_config
from models.modeling_umm import UMM, UMMConfig
from utils.weights import read_weights, check_source_architecture, load_into_model


def load_tokenizer(config):
    from transformers import AutoTokenizer
    values = config["model"]
    path = values.get("tokenizer_path") or values.get("model_path")
    if not path:
        raise ValueError("model.tokenizer_path or model.model_path is required for data loading")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if "<|mdm_mask|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    tokens = [token for token in ("<|boi|>", "<|eoi|>", "<|img_mask|>") if token not in tokenizer.get_vocab()]
    values["_added_image_mask_token"] = "<|img_mask|>" in tokens
    if tokens:
        tokenizer.add_tokens(tokens, special_tokens=True)
    for name, token in (("mask_token_id", "<|mdm_mask|>"), ("boi_token_id", "<|boi|>"),
                        ("eoi_token_id", "<|eoi|>"), ("image_mask_token_id", "<|img_mask|>")):
        values[name] = tokenizer.convert_tokens_to_ids(token)
    base_config = Path(path) / "config.json"
    base_vocab = json.loads(base_config.read_text()).get("vocab_size", 0) if base_config.is_file() else 0
    values["vocab_size"] = max(int(values.get("vocab_size", 0)), int(base_vocab), len(tokenizer))
    values["eos_token_id"] = tokenizer.eos_token_id
    values["pad_token_id"] = tokenizer.pad_token_id
    return tokenizer


def load_model(config):
    added_image_mask = config["model"].pop("_added_image_mask_token", False)
    values = dict(config["model"])
    mask_init = values.get("image_mask_init", "copy_text")
    if mask_init not in {"copy_text", "random"}:
        raise ValueError("image_mask_init must be copy_text or random")
    checkpoint = values.get("checkpoint")
    base_path = values.get("model_path")
    state, saved = (None, {})
    if checkpoint:
        state, saved = read_weights(checkpoint)
        check_source_architecture(saved)
    elif base_path and not values.get("from_scratch", False):
        state, saved = read_weights(base_path)
    elif base_path and (Path(base_path) / "config.json").is_file():
        saved = json.loads((Path(base_path) / "config.json").read_text())
    saved_model = saved.get("model", saved)
    check_source_architecture(saved)
    check_source_architecture(values)
    known = {field.name for field in fields(UMMConfig)}
    architecture = {key: value for key, value in saved_model.items() if key in known}
    # Qwen v5 may store its original RoPE base in rope_parameters.
    if "rope_theta" not in architecture and isinstance(saved_model.get("rope_parameters"), dict):
        if "rope_theta" in saved_model["rope_parameters"]:
            architecture["rope_theta"] = saved_model["rope_parameters"]["rope_theta"]
    if checkpoint:
        for key, value in values.items():
            if key in known and key in architecture and value != architecture[key]:
                raise ValueError(f"Checkpoint model.{key}={architecture[key]!r} conflicts with configuration {value!r}")
        architecture.update({key: value for key, value in values.items() if key in known and key not in architecture})
    else:
        architecture.update({key: value for key, value in values.items() if key in known})
    model_config = UMMConfig.from_dict(architecture)
    model = UMM(model_config)
    is_full = False
    if state is not None:
        is_full = any(key.startswith(("generator.", "image_flow_head.")) for key in state)
        if is_full:
            check_source_architecture(saved)
        load_into_model(model, state, backbone_only=not checkpoint and not is_full)
    # Only initialize newly introduced tokens when starting from a base model.
    # Full UMM exports and training resumes retain their learned embeddings.
    if added_image_mask and not checkpoint and not is_full:
        with torch.no_grad():
            embedding = model.backbone.embed_tokens.weight
            image_row = embedding[model_config.image_mask_token_id]
            if mask_init == "copy_text":
                image_row.copy_(embedding[model_config.mask_token_id])
            else:
                # Keep all subsequent training RNG draws identical between the
                # two initialization experiments, including reserved vocab rows.
                with torch.random.fork_rng(devices=[]):
                    image_row.normal_(std=model_config.initializer_range)
    config["model"]["image_mask_init"] = mask_init
    config["model"].update(model_config.to_dict())
    return model


def build_accelerator(config):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin
    train = config["training"]
    accelerator = Accelerator(
        cpu=bool(train.get("cpu", False)), mixed_precision=train["mixed_precision"],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=int(train["gradient_accumulation_steps"]), sync_with_dataloader=False,
        ),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.state.deepspeed_plugin is not None:
        settings = accelerator.state.deepspeed_plugin.deepspeed_config
        if int(settings.get("zero_optimization", {}).get("stage", 0)) != 2:
            raise ValueError("Selected sharded EMA/DeepSpeed baseline supports ZeRO-2")
        settings["gradient_accumulation_steps"] = train["gradient_accumulation_steps"]
        settings["train_micro_batch_size_per_gpu"] = train["batch_size"]
        settings["gradient_clipping"] = train["max_grad_norm"]
        settings.setdefault("data_types", {})["grad_accum_dtype"] = "fp32"
    return accelerator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without loading model/data")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    from accelerate.utils import set_seed
    from data.dataloader import build_dataloaders
    from .optimizer import apply_trainable_scope, build_optimizer_and_scheduler
    from .ema import RankShardedEMA, build_sharded_ema_layout
    from .trainer import Trainer
    accelerator = build_accelerator(config)
    set_seed(int(config["training"]["seed"]), device_specific=True)
    tokenizer = load_tokenizer(config)
    model = load_model(config)
    apply_trainable_scope(model, config["training"]["trainable_scope"])
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    train_loader, validation_loader = build_dataloaders(config, tokenizer)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    train_loader = train_loader.prepare_with_accelerator(accelerator)
    ema = None
    if config["training"]["use_ema"]:
        # ZeRO-2 keeps complete parameters, but BF16 preparation changes their
        # dtype. Record the final model layout before binding the FP32 shards.
        layout = build_sharded_ema_layout(accelerator.unwrap_model(model),
            world_size=accelerator.num_processes,
            chunk_numel=int(config["training"]["ema_shard_chunk_numel"]))
        ema = RankShardedEMA(layout, rank=accelerator.process_index,
            decay=float(config["training"]["ema_decay"]),
            update_after_step=int(config["training"]["ema_update_after_step"]))
        ema.bind(accelerator.unwrap_model(model))
        ema.initialize_from_model(global_step=0)
    output = Path(config["training"]["output_dir"])
    def save_run_metadata():
        output.mkdir(parents=True, exist_ok=True)
        save_config(config, output / "config.yaml")
        tokenizer.save_pretrained(output / "tokenizer")
    from utils.distributed import run_io_phase
    run_io_phase(accelerator, save_run_metadata, description="Save run metadata", main_process_only=True)
    accelerator.wait_for_everyone()
    trainer = Trainer(model, optimizer, scheduler, train_loader, accelerator, config,
                      validation_loader=validation_loader, ema=ema)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
