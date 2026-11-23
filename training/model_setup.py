"""Distributed initialization, restart selection, model construction and optimizer."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

import distributed.accel as accel
import distributed.megatron as P
import utils.checkpoint as checkpoint_io
from models.backbone_qwen35 import cast_for_training, load_pretrained

ACCEL = accel.accel_type()
DEV = accel.device_module()


def initialize(args, cfg):
    """Start the process group and fix the batch geometry. Returns the topology record."""
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    complete_pretrained = bool(cfg.text_assets) and not args.random_init
    skip_initialization = bool(args.resume or args.init_checkpoint) or complete_pretrained
    P.initialize(
        args.tp, cfg, optimizer_instances=args.optimizer_instances, initialize_weights=not skip_initialization
    )
    torch.manual_seed(args.seed)
    DEV.manual_seed_all(args.seed)
    rank, world = dist.get_rank(), dist.get_world_size()
    profile_ranks = {int(r) for r in args.profile_ranks.split(",")}
    if args.profile and (not profile_ranks or min(profile_ranks) < 0 or max(profile_ranks) >= world):
        raise ValueError("Profile ranks must be a nonempty subset of the training world")
    dp = P.dp_size()
    if args.global_batch:
        if args.global_batch % (dp * args.micro_batch):
            raise ValueError("global_batch must be divisible by DP * micro_batch")
        args.grad_accum = args.global_batch // (dp * args.micro_batch)
    if args.data_prefetch == 0:
        args.data_prefetch = max(4, args.grad_accum + 4)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    return SimpleNamespace(
        rank=rank,
        world=world,
        dp=dp,
        global_batch=args.micro_batch * dp * args.grad_accum,
        output=output,
        profile_ranks=profile_ranks,
        complete_pretrained=complete_pretrained,
        topology=dict(tp=args.tp, dp=dp, pp=1, cp=1, optimizer_instances=args.optimizer_instances),
    )


def storage_roots(args, output):
    """Restart checkpoints and evaluation snapshots must never share a retention root."""
    checkpoint_root = args.checkpoint_dir or output
    snapshot_root = args.eval_snapshot_dir or (
        args.checkpoint_dir.parent / "eval_snapshots" if args.checkpoint_dir else output / "eval_snapshots"
    )
    if args.eval_snapshot_every:
        checkpoint_path, snapshot_path = checkpoint_root.resolve(), snapshot_root.resolve()
        inside_steps = snapshot_path.is_relative_to(checkpoint_path) and snapshot_path.relative_to(
            checkpoint_path
        ).parts[0].startswith("step_")
        if snapshot_path == checkpoint_path or inside_steps:
            raise ValueError(
                "Evaluation snapshots must be outside checkpoint step directories and their retention root"
            )
    return checkpoint_root, snapshot_root


def select_restart(args, rank, checkpoint_root):
    """--auto-resume: rank zero picks the newest committed checkpoint and cleans the rest."""
    if not args.auto_resume:
        return
    from utils.checkpoint_catalog import latest, prune, remove_incomplete

    selection = [None]
    if rank == 0:
        try:
            path, rejected = latest(checkpoint_root)
            remove_incomplete(checkpoint_root)
            if path:
                prune(checkpoint_root, args.keep_checkpoints)
            selection[0] = dict(path=path, skipped=rejected)
        except Exception as error:
            selection[0] = dict(error=str(error))
    dist.broadcast_object_list(selection, src=0)
    if "error" in selection[0]:
        raise RuntimeError(selection[0]["error"])
    args.resume = selection[0]["path"]
    if rank == 0:
        print(json.dumps(dict(checkpoint_selection=selection[0])), flush=True)


def build_model(args, cfg, run):
    """Construct the joint model and load initial weights. Returns (model, loaded, startup)."""
    from models.modeling_umm import JointModel

    skip_initialization = bool(args.resume or args.init_checkpoint) or run.complete_pretrained
    begin = time.perf_counter()
    if skip_initialization:
        from transformers.initialization import no_init_weights

        construction = no_init_weights()
    else:
        construction = nullcontext()
    with construction:
        model = cast_for_training(JointModel(cfg))
    startup = dict(
        construct_seconds=time.perf_counter() - begin, skipped_random_initialization=skip_initialization
    )
    loaded = None
    if not args.resume and not args.init_checkpoint and not args.random_init:
        if cfg.text_assets:
            loaded = load_pretrained(model, cfg.text_assets)
        else:
            model.patch_embed.load_pretrained(cfg.vision_assets)
        loaded = dict(backbone=loaded, renderer=model.renderer.load_pretrained(cfg.wan_assets))
    model = model.to(device=ACCEL).train()
    if args.init_checkpoint and not args.resume:
        loaded = checkpoint_io.initialize_model(args.init_checkpoint, model, cfg)
    if cfg.null_token_id is not None and not (args.resume or args.init_checkpoint):
        _initialize_null_token(model, cfg, args.seed)
        startup["initialized_null_token_id"] = cfg.null_token_id
    startup["weights_to_device_seconds"] = time.perf_counter() - begin
    if run.rank == 0:
        print(json.dumps(dict(startup=startup)), flush=True)
    if args.data_root and args.data_mode == "B":
        model.patch_embed.requires_grad_(False)
        model.renderer.requires_grad_(False)
    return model, loaded, startup


def _initialize_null_token(model, cfg, seed):
    """Initialize the extension row once, with the same private RNG in every arm."""
    first = getattr(model.embedding, "vocab_start_index", 0)
    local = cfg.null_token_id - first
    if 0 <= local < model.embedding.weight.shape[0]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = torch.randn(cfg.width, generator=generator, dtype=torch.float32) * 0.02
        with torch.no_grad():
            model.embedding.weight[local].copy_(value)


def count_parameters(model, tp):
    """Total and trainable counts; TP shards and replicated tensors are each counted once."""
    unique = [
        p
        for p in model.parameters()
        if getattr(p, "tensor_model_parallel", False) or P.state().get_tensor_model_parallel_rank() == 0
    ]
    count = torch.tensor(
        [sum(p.numel() for p in unique), sum(p.numel() for p in unique if p.requires_grad)],
        device=ACCEL,
        dtype=torch.int64,
    )
    if tp > 1:
        dist.all_reduce(count, group=P.state().get_tensor_model_parallel_group())
    return count.tolist()


def build_optimizer(args, cfg, model):
    """Distributed optimizer, optional renderer EMA and the trainable-scope record."""
    wrapped, optimizer = P.make_optimizer(
        model,
        args.lr,
        bucket_size=args.bucket_size,
        overlap_param_gather=not args.no_param_overlap,
        optimizer_offload_fraction=args.optimizer_offload_fraction,
        grad_reduce_in_fp32=not args.bf16_grad_reduce,
    )
    ema = None
    if args.renderer_ema_decay:
        if args.data_mode == "B":
            raise ValueError("EMA requires a trainable renderer")
        from training.ema import RendererEMA

        ema = RendererEMA(model, optimizer, wrapped, args.renderer_ema_decay)
    frozen_in_buckets = [
        name for name, p in model.named_parameters() if not p.requires_grad and p in wrapped.param_to_bucket_group
    ]
    if frozen_in_buckets:
        raise RuntimeError(f"Frozen parameters entered gradient buckets: {frozen_in_buckets}")
    tower = model.patch_embed.visual
    scope = dict(
        freeze_vision=cfg.freeze_vision,
        freeze_vision_merger=cfg.freeze_vision_merger,
        frozen_parameters_in_gradient_buckets=len(frozen_in_buckets),
        vision_encoder_training=tower.blocks.training,
        vision_encoder_recompute=tower.recompute,
    )
    return wrapped, optimizer, ema, scope
