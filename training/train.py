"""Train the Qwen3.5 CQ + Wan2.1 video pipeline on CPU, CUDA or Ascend."""

from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
import sys
from pathlib import Path
import torch
from configs.config import load_config, save_config, resolve_output_paths


def _rename_legacy_mask_tokens(tokenizer):
    """Rename saved CQ tokens in memory without moving their embedding rows."""
    from tokenizers import AddedToken, Tokenizer

    vocab = tokenizer.get_vocab()
    renames = {
        old: new
        for old, new in (
            ("<|mdm_mask|>", "<|text_mask|>"),
            ("<|video_mask|>", "<|vision_mask|>"),
        )
        if old in vocab
    }
    if not renames:
        return
    if any(new in vocab for new in renames.values()):
        raise ValueError(
            "Tokenizer contains both legacy and renamed CQ mask tokens; resolve their IDs before loading"
        )

    # Replace exact token strings in the backend vocabulary/added-token records.
    # Appending aliases instead would allocate new IDs and lose checkpoint alignment.
    state = tokenizer.backend_tokenizer.to_str()
    for old, new in renames.items():
        state = state.replace(json.dumps(old), json.dumps(new))
    tokenizer._tokenizer = Tokenizer.from_str(state)

    def rename(value):
        if isinstance(value, AddedToken):
            state = value.__getstate__()
            state["content"] = renames.get(state["content"], state["content"])
            return AddedToken(**state)
        if isinstance(value, str):
            return renames.get(value, value)
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(rename(item) for item in value)
        return value

    # Keep HF metadata consistent so save_pretrained cannot restore the old names.
    tokenizer.init_kwargs = rename(tokenizer.init_kwargs)
    tokenizer._special_tokens_map = rename(tokenizer._special_tokens_map)
    tokenizer._extra_special_tokens = rename(tokenizer._extra_special_tokens)


def load_processor(config):
    from transformers import AutoProcessor

    settings = config["model"]
    # Export-local tokenizer metadata preserves any reserved mask-token IDs.
    # An explicit processor path still takes precedence for testing/migration.
    checkpoint = settings.get("checkpoint")
    adjacent = Path(checkpoint).expanduser().resolve().parent / "processor" if checkpoint else None
    path = settings.get("processor_path") or (
        str(adjacent) if adjacent is not None and adjacent.is_dir() else settings["qwen_path"]
    )
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = processor.tokenizer
    _rename_legacy_mask_tokens(tokenizer)
    # Native image/video placeholders keep their original roles and IDs.
    for key, token in (
        ("mask_token_id", "<|text_mask|>"),
        ("video_mask_token_id", "<|vision_mask|>"),
    ):
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
        settings[key] = tokenizer.convert_tokens_to_ids(token)
    return processor


def load_model(config):
    # MCore initializes transport and preserves native kernel dispatch before
    # constructing any trainable model/optimizer.
    from models.modeling_umm import UMM, UMMConfig
    settings = config["model"]
    checkpoint = settings.get("checkpoint")
    if checkpoint:
        from utils.weights import read_weights, load_into_model

        state, saved = read_weights(checkpoint)
        architecture = deepcopy(saved["model"]["architecture"])
        if (
            not saved["model"]["freeze_vision"]
            and config["data"]["representations"]["qwen"] != "online"
        ):
            raise ValueError(
                "This export trained ViT; use online Qwen inputs with its learned visual weights"
            )
        for key in ("mask_token_id", "video_mask_token_id"):
            if architecture[key] != settings[key]:
                raise ValueError(f"Export and processor disagree about {key}")
        architecture.update(
            freeze_vision=settings["freeze_vision"],
            gradient_checkpointing=settings["gradient_checkpointing"],
        )
        architecture["wan_config"].update(
            attention_backend=settings["wan_attention_backend"],
            gradient_checkpointing=settings["gradient_checkpointing"],
        )
        model = UMM(UMMConfig.from_dict(architecture))
        load_into_model(model, state)
    elif settings["from_scratch"]:
        architecture = deepcopy(settings.get("architecture", {}))
        if not architecture:
            raise ValueError(
                "from_scratch requires an explicit model.architecture (test YAML)"
            )
        architecture.update(
            mask_token_id=settings["mask_token_id"],
            video_mask_token_id=settings["video_mask_token_id"],
            freeze_vision=settings["freeze_vision"],
            gradient_checkpointing=settings["gradient_checkpointing"],
        )
        architecture["wan_config"].update(
            attention_backend=settings["wan_attention_backend"],
            gradient_checkpointing=settings["gradient_checkpointing"],
        )
        model = UMM(UMMConfig.from_dict(architecture))
    else:
        model = UMM.from_pretrained_components(
            settings["qwen_path"],
            settings["wan_path"],
            mask_token_id=settings["mask_token_id"],
            video_mask_token_id=settings["video_mask_token_id"],
            freeze_vision=settings["freeze_vision"],
            gradient_checkpointing=settings["gradient_checkpointing"],
            attn_implementation=settings["attn_implementation"],
            wan_attention_backend=settings["wan_attention_backend"],
            dtype=torch.float32,
        )
        # New mask tokens can occupy reserved existing vocab rows. Initialize
        # them explicitly once; a full export/resume always restores its rows.
        with torch.no_grad():
            embeddings = model.backbone.embed_tokens.weight
            for token_id in (
                settings["mask_token_id"],
                settings["video_mask_token_id"],
            ):
                embeddings[token_id].normal_(
                    std=model.backbone.config.initializer_range
                )
    # UMM and its backbone own separate config copies. Persist the effective
    # backend in both, so exports and all three loading paths use one choice.
    gdn_backend = settings.get("qwen_gdn_backend", "native")
    model.backbone.configure_gdn_backend(gdn_backend)
    model.config.qwen_config.text_config.qwen_gdn_backend = gdn_backend
    model.backbone.config._attn_implementation = settings["attn_implementation"]
    model.vision_encoder.visual.config._attn_implementation = settings[
        "attn_implementation"
    ]
    config["model"]["architecture"] = model.config.to_dict()
    return model


def register_device(name):
    if name == "npu":
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU requested but unavailable")
    elif name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def build_accelerator(config):
    """CPU regression runtime; production training uses MCoreRuntime."""
    train = config["training"]
    if train["device"] != "cpu":
        raise ValueError("Accelerate is only for CPU regression; use MindSpeed for NPU training")
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() == "true":
        raise ValueError("DeepSpeed training is no longer supported")
    register_device(train["device"])
    from accelerate import Accelerator
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
    )

    accelerator = Accelerator(
        cpu=train["device"] == "cpu",
        mixed_precision=train["mixed_precision"],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=int(train["gradient_accumulation_steps"]),
            sync_with_dataloader=False,
        ),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.device.type != train["device"]:
        raise RuntimeError(
            f"Requested {train['device']}, Accelerate selected {accelerator.device}"
        )
    return accelerator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume")
    recovery.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="Stop at this optimizer boundary without changing the saved training budget",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        parser.error("--stop-after-steps must be positive")
    if config["training"]["deterministic"]:
        # Same restart-validation settings as cq-pretrain. Set before any NPU
        # kernel/communicator initialization; changing warmed kernels is unsafe.
        os.environ["HCCL_DETERMINISTIC"] = "True"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        torch.use_deterministic_algorithms(True)
    paths = resolve_output_paths(config)
    output, checkpoint_root = paths["output_dir"], paths["checkpoint_dir"]
    from accelerate.utils import set_seed
    from data.dataloader import build_dataloaders
    from .optimizer import apply_trainable_scope, build_optimizer_and_scheduler
    from .ema import RankShardedEMA, build_sharded_ema_layout
    from .trainer import Trainer
    from utils.distributed import run_io_phase

    from contextlib import ExitStack
    from utils.checkpoint_catalog import (
        DirectoryLock,
        discover,
        latest,
        quarantine_invalid,
        durable_json,
    )
    from utils.checkpoint import validate_checkpoint
    from utils.resume_contract import runtime_contract, management_changes
    from utils.run_logging import RunLogger, resolve_attempt

    if config["distributed"]["backend"] == "mindspeed":
        from .mcore import MCoreRuntime
        accelerator = MCoreRuntime(config)
    else:
        accelerator = build_accelerator(config)
    attempt = run_io_phase(
        accelerator,
        lambda: str(resolve_attempt(config)),
        description="Create log attempt",
        main_process_only=True,
    )
    logger = RunLogger(attempt, enabled=accelerator.is_main_process)
    trainer = None
    with ExitStack() as locks:
        try:
            run_io_phase(
                accelerator,
                lambda: logger.start(
                    config,
                    list(argv) if argv is not None else sys.argv[1:],
                    runtime_contract(accelerator),
                ),
                description="Record launch",
                main_process_only=True,
            )

            def select_resume():
                # Lock both metadata and checkpoint roots; locks are released even on failed recovery.
                for root in sorted({output, checkpoint_root}):
                    locks.enter_context(DirectoryLock(root))
                rejected = []
                if args.resume:
                    selected = str(Path(args.resume).expanduser().absolute())
                    validate_checkpoint(selected)
                elif args.auto_resume:
                    selected, rejected = latest(checkpoint_root)
                else:
                    selected = None
                valid, invalid = discover(checkpoint_root)
                if selected is None:
                    if (
                        valid
                        or invalid
                        or (checkpoint_root / ".training_started.json").exists()
                        or (checkpoint_root / "latest.json").exists()
                    ):
                        raise FileExistsError(
                            "Experiment has training history; use --resume/--auto-resume with its checkpoint_dir, or a new exp_id"
                        )
                elif valid:
                    source = Path(selected).resolve()
                    if source != valid[0][0].resolve():
                        raise ValueError(
                            "Resuming an older/external checkpoint into existing history requires a new checkpoint_dir"
                        )
                logger.record(
                    "checkpoints.jsonl",
                    {
                        "event": "resume_selection",
                        "path": selected,
                        "automatic": args.auto_resume,
                        "rejected": rejected,
                    },
                )
                return selected

            resume_path = run_io_phase(
                accelerator,
                select_resume,
                description="Select resume checkpoint",
                main_process_only=True,
            )
            # Newly initialized condition/mask parameters must agree across ranks.
            set_seed(int(config["training"]["seed"]), device_specific=False)
            processor = load_processor(config)
            train_loader, validation_loader = build_dataloaders(config, processor)
            model = load_model(config)
            apply_trainable_scope(model, config["model"])
            if getattr(accelerator, "is_mcore", False):
                model, optimizer, scheduler = accelerator.prepare_model(model)
            else:
                optimizer, scheduler = build_optimizer_and_scheduler(model, config)
                model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
            train_loader.prepare_with_accelerator(accelerator)
            # Avoid creating an AcceleratorState on the MCore path.
            set_seed(int(config["training"]["seed"]) + accelerator.process_index,
                     device_specific=False)
            ema = None
            if config["training"]["use_ema"]:
                layout = build_sharded_ema_layout(
                    accelerator.unwrap_model(model),
                    world_size=accelerator.num_processes,
                    chunk_numel=int(config["training"]["ema_shard_chunk_numel"]),
                )
                ema = RankShardedEMA(
                    layout,
                    rank=accelerator.process_index,
                    decay=float(config["training"]["ema_decay"]),
                    update_after_step=int(config["training"]["ema_update_after_step"]),
                )
                ema.bind(accelerator.unwrap_model(model))
                ema.initialize_from_model(global_step=0)
            trainer = Trainer(
                model,
                optimizer,
                scheduler,
                train_loader,
                accelerator,
                config,
                validation_loader=validation_loader,
                ema=ema,
                logger=logger,
            )
            restored = trainer.resume(resume_path) if resume_path else None

            def metadata():
                changes = (
                    management_changes(restored["metadata"]["config"], config)
                    if restored
                    else []
                )
                # Do not alter shared metadata or rejected checkpoints until every rank restored successfully.
                if resume_path:
                    moved = quarantine_invalid(checkpoint_root)
                    if moved:
                        logger.record(
                            "checkpoints.jsonl", {"event": "quarantine", "moved": moved}
                        )
                output.mkdir(parents=True, exist_ok=True)
                save_config(config, output / "config.yaml")
                processor.save_pretrained(checkpoint_root / "exports" / "processor")
                logger.ready(
                    config,
                    resume=resume_path,
                    step=trainer.global_step,
                    changes=changes,
                    legacy_runtime=bool(
                        restored and "runtime_contract" not in restored["metadata"]
                    ),
                )
                durable_json(
                    checkpoint_root / ".training_started.json",
                    {
                        "attempt_id": Path(attempt).name,
                        "exp_id": config["training"]["exp_id"],
                        "start_step": trainer.global_step,
                        "resume": resume_path,
                    },
                )

            run_io_phase(
                accelerator,
                metadata,
                description="Save validated run metadata",
                main_process_only=True,
            )
            result = trainer.fit(stop_after_steps=args.stop_after_steps)
            run_io_phase(
                accelerator,
                lambda: logger.finish(
                    status="completed",
                    step=trainer.global_step,
                    checkpoint=result["checkpoint"],
                    exports=str(checkpoint_root / "exports"),
                ),
                description="Record completion",
                main_process_only=True,
            )
        except BaseException as exc:
            if accelerator.is_main_process:
                logger.finish(
                    status="failed",
                    step=trainer.global_step if trainer else None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
        finally:
            accelerator.end_training()


if __name__ == "__main__":
    main()
