"""Command-line options, learning-rate schedule and loss normalization."""

import argparse
import math
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--vision-assets", help="Local pinned Qwen3.5 vision assets; training never downloads weights")
    parser.add_argument("--text-assets", help="Local pinned Qwen3.5 language checkpoint and tokenizer")
    parser.add_argument("--wan-assets", help="Local native Wan2.1 DiT checkpoint")
    parser.add_argument("--wan-vae-assets", help="Local frozen Wan2.1 VAE checkpoint")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Explicit architecture control; do not load pretrained language/vision weights",
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use fake_batch with the unchanged production model and CPU planners",
    )
    parser.add_argument("--image-condition-queries", type=int)
    parser.add_argument("--visual-query-mode", choices=("native", "fixed"))
    parser.add_argument(
        "--query-interaction",
        choices=("none", "bidirectional"),
        help="Ablation: let the visual queries of one target read each other (model.query_interaction)",
    )
    parser.add_argument(
        "--null-condition",
        choices=("masked", "bos", "token"),
        help="Ablation: what an unconditional visual query reads (model.null_condition)",
    )

    # Topology, batch and step budget.
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--images", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--total-steps", type=int, help="Stop at this absolute update count, including resume")
    parser.add_argument(
        "--data-epochs", type=int, default=1, help="Complete this many independently shuffled finite epochs"
    )
    parser.add_argument(
        "--until-data-end",
        action="store_true",
        help="Traverse the remaining finite DCVLM epoch instead of a slot-based step budget",
    )
    parser.add_argument("--warmup", type=int, default=3)

    # Optimization.
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--renderer-ema-decay",
        type=float,
        default=0.0,
        help="FP32 distributed optimizer-shard renderer EMA; zero disables",
    )
    parser.add_argument("--min-lr", type=float, help="Cosine floor; defaults to one tenth of --lr")
    parser.add_argument("--lr-decay-style", choices=("constant", "cosine", "wsd"), default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=100000,
        help="Step at which the LR reaches min-lr (cosine and wsd)",
    )
    parser.add_argument(
        "--lr-decay-start",
        type=int,
        help="wsd only: first step of the cosine decay; omit to stay stable. "
        "Both wsd decay settings may change on resume",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="New-stage model-only initialization; fresh optimizer and data cursor",
    )

    # Validation, checkpoints and snapshots.
    parser.add_argument("--validation-suite", type=Path, help="Fixed local T2I/I2T qualitative suite")
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument(
        "--validate-every-hours",
        type=float,
        default=0.0,
        help="Also validate once this much wall time has passed since the last validation; "
        "the clock is stored beside the reports and survives restarts",
    )
    parser.add_argument("--validate-first", action="store_true")
    parser.add_argument("--validate-final", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, help="Shared checkpoint root, independent of attempt logs")
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument(
        "--eval-snapshot-every",
        type=int,
        default=1000,
        help="Save inference-only weights every N absolute updates; zero disables",
    )
    parser.add_argument(
        "--eval-snapshot-dir",
        type=Path,
        help="Retained evaluation weights; defaults beside checkpoint-dir, or under output",
    )
    parser.add_argument("--save-first", action="store_true", help="Save after the first successful update")
    parser.add_argument("--save-final", action="store_true", help="Save the last update on normal completion")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume")
    resume.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume the newest committed checkpoint in checkpoint-dir",
    )
    parser.add_argument(
        "--validation-probes",
        action="store_true",
        help="Write model/Adam/master probes after each update; exclude this run from throughput estimates",
    )

    # Execution and measurement.
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-ranks", default="0", help="Comma separated global ranks")
    parser.add_argument("--bucket-size", type=int, default=40_000_000, help="DDP bucket elements")
    parser.add_argument(
        "--bf16-grad-reduce",
        action="store_true",
        help="Reduce gradients in BF16 instead of FP32: halves DP allreduce bytes and the "
        "resident main_grad buffer. Changes reduction numerics; validate loss parity.",
    )
    parser.add_argument("--no-param-overlap", action="store_true")
    parser.add_argument("--phase-timing", action="store_true", help="Measure forward/backward/optimizer device time")
    parser.add_argument("--log-all-ranks", action="store_true", help="Print successful step receipts on every rank")
    parser.add_argument("--checkpoint-backbone-layers", type=int, default=0)
    parser.add_argument("--checkpoint-dit-layers", type=int, default=0)
    parser.add_argument("--keep-attention-activations", type=int, default=0)
    parser.add_argument(
        "--optimizer-instances",
        type=int,
        default=1,
        help="Replicated optimizer groups for hierarchical DP synchronization",
    )
    parser.add_argument(
        "--optimizer-offload-fraction",
        type=float,
        default=0.0,
        help="Fraction of Adam state/compute offloaded by upstream MCore (0 disables)",
    )
    parser.add_argument("--ce-chunk", type=int, help="Text output projection/CE chunk size")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic kernels for exact restart validation",
    )
    recompute = parser.add_mutually_exclusive_group()
    recompute.add_argument("--no-checkpoint", action="store_true")
    recompute.add_argument("--checkpoint", action="store_true", help="Recompute backbone and DiT layers")
    parser.add_argument("--split-reference", action="store_true")

    # Data.
    parser.add_argument(
        "--data-root",
        help="Published corpus, JSONL or mixed-source YAML; tokenize raw text online",
    )
    parser.add_argument("--data-mode", choices=("A", "B"), default="A")
    parser.add_argument("--allow-development-data", action="store_true")
    parser.add_argument("--data-workers", type=int, default=2)
    parser.add_argument("--data-prefetch", type=int, default=0, help="CPU packs ahead; zero selects GA+4 (minimum 4)")
    parser.add_argument("--source-root", help="Optional local TAR pool relocation")
    parser.add_argument("--tokenize-rows", type=int, default=32)
    parser.add_argument("--shuffle-block-rows", type=int, default=4)
    parser.add_argument("--image-workers", type=int, default=2)
    parser.add_argument("--image-prefetch", type=int, default=16)
    parser.add_argument(
        "--device-prefetch",
        type=int,
        default=2,
        help="Prepared NPU batches ahead; zero is synchronous control",
    )
    parser.add_argument(
        "--preserve-documents",
        action="store_true",
        help="Pack complete documents; budget DiT image/CQ pairs independently",
    )
    parser.add_argument(
        "--long-document-policy",
        choices=("window", "error"),
        help="Policy only for documents longer than the backbone context",
    )
    parser.add_argument(
        "--dit-microbatch-tokens", type=int, help="Wan latent-token budget per chunk across all RF draws"
    )
    parser.add_argument(
        "--dit-microbatch-images",
        type=int,
        help="Optional limit on image draws per DiT chunk; zero is unlimited",
    )
    parser.add_argument("--rf", type=int, help="Rectified-flow draws per generation target (data.rf, default 4)")
    parser.add_argument(
        "--timestep-shift", type=float, help="Wan shift applied to logit-normal training times (default 1)"
    )
    parser.add_argument(
        "--max-images-per-pack",
        type=int,
        default=None,
        help="Legacy fragment packing only; defaults to zero for whole documents",
    )
    parser.add_argument(
        "--pack-padding-multiple",
        type=int,
        help="Override automatic 2K/4K/8K pack widths with a multiple; zero keeps the full sequence length",
    )
    parser.add_argument("--latent-encoder", choices=("wan", "fake"), default="wan")
    parser.add_argument("--vae-dtype", choices=("float32", "bfloat16"), default="float32")
    return parser


def parse_args(argv=None):
    """Parse options; an experiment YAML and the release geometry provide defaults."""
    parser = build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config and Path(preliminary.config).suffix in (".yaml", ".yml"):
        import yaml

        document = yaml.safe_load(Path(preliminary.config).read_text())
        for key, value in document.get("environment", {}).items():
            os.environ.setdefault(key, str(value))
        defaults = document.get("training", {})
        destinations = {action.dest for action in parser._actions}
        if set(defaults) - destinations:
            parser.error(f"Unknown training settings: {set(defaults) - destinations}")
        parser.set_defaults(**defaults)
    data_root = preliminary.data_root or parser.get_default("data_root")
    if data_root:
        from data.sources import release_geometry

        geometry = release_geometry(data_root)
        parser.set_defaults(
            **{key: geometry[key] for key in ("sequence_length", "shuffle_block_rows") if key in geometry}
        )
    args = parser.parse_args(argv)
    _validate(parser, args)
    return args


def _validate(parser, args):
    if not args.output:
        parser.error("Set --output or training.output in the experiment config")
    for key in ("validation_suite", "init_checkpoint", "checkpoint_dir", "eval_snapshot_dir"):
        if getattr(args, key) is not None:
            setattr(args, key, Path(getattr(args, key)))
    if args.min_lr is None:
        args.min_lr = args.lr * 0.1
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.min_lr) or not 0 <= args.min_lr <= args.lr:
        parser.error("Require finite lr > 0 and 0 <= min-lr <= lr")
    if not math.isfinite(args.renderer_ema_decay) or not 0 <= args.renderer_ema_decay < 1:
        parser.error("Require finite 0 <= renderer-ema-decay < 1")
    if (
        args.lr_decay_style == "wsd"
        and args.lr_decay_start is not None
        and not (args.lr_warmup_steps <= args.lr_decay_start < args.lr_decay_steps)
    ):
        parser.error("wsd requires lr-warmup-steps <= lr-decay-start < lr-decay-steps")
    if args.lr_decay_start is not None and args.lr_decay_style != "wsd":
        parser.error("--lr-decay-start applies only to --lr-decay-style wsd")
    if args.eval_snapshot_every < 0:
        parser.error("Require eval-snapshot-every >= 0")
    if not 0 <= args.optimizer_offload_fraction <= 1:
        parser.error("Require optimizer-offload-fraction in [0, 1]")


def learning_rate(step, args):
    if step < args.lr_warmup_steps:
        return args.lr * (step + 1) / args.lr_warmup_steps
    if args.lr_decay_style == "constant":
        return args.lr
    start = args.lr_warmup_steps
    if args.lr_decay_style == "wsd":
        # Warmup, then a stable plateau, then cosine to min-lr. The decay can be
        # scheduled later, on resume, once the stopping point is known.
        start = getattr(args, "lr_decay_start", None)
        if start is None or step < start:
            return args.lr
    progress = min(1.0, max(0.0, (step - start) / max(1, args.lr_decay_steps - start)))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def data_loss_scales(batch, global_text_tokens, global_generation_images, dp):
    """Global target averages after DP reduction, independent of pack sizes.

    Understanding-only images remain in the backbone but must not change the
    renderer denominator. A replica with zero selected images contributes zero.
    """
    return (
        dp * batch.text_indices.numel() / max(1, global_text_tokens),
        dp * batch.generation_image_count / max(1, global_generation_images),
    )
