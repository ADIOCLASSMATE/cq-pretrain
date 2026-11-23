"""Resolve the run configuration from options, the model YAML and the data release."""

from pathlib import Path
from types import SimpleNamespace

from configs.config import RunConfig, tiny_config

ROOT = Path(__file__).resolve().parents[1]

# Options that override a config field of the same meaning when given.
_CONFIG_OVERRIDES = (
    ("vision_assets", "vision_assets"),
    ("text_assets", "text_assets"),
    ("wan_assets", "wan_assets"),
    ("wan_vae_assets", "wan_vae_assets"),
    ("image_condition_queries", "image_condition_queries"),
    ("visual_query_mode", "visual_query_mode"),
    ("query_interaction", "query_interaction"),
    ("null_condition", "null_condition"),
)


def check_budget(args):
    """Validate the step budget; --until-data-end replaces it with the finite dataset."""
    if args.save_every < 0 or args.keep_checkpoints < 1:
        raise ValueError("Require save-every >= 0 and keep-checkpoints >= 1")
    if args.data_epochs < 1 or (args.data_epochs > 1 and not args.until_data_end):
        raise ValueError("Multiple data epochs require --until-data-end and a positive epoch count")
    if args.until_data_end:
        if not args.data_root:
            raise ValueError("--until-data-end requires a finite --data-root")
        args.total_steps = None
        args.steps = (1 << 63) - 1
    budget = args.total_steps if args.total_steps is not None else args.steps
    if budget <= args.warmup or args.warmup < 0 or args.micro_batch < 1 or args.grad_accum < 1:
        raise ValueError("Require steps > warmup and positive batch/accumulation sizes")
    if args.config and args.tiny:
        raise ValueError("Select either an explicit model config or the tiny validation model")
    args.random_init = args.random_init or args.tiny


def resolve(args):
    """Return (cfg, data, assets). `data` is None for synthetic runs."""
    check_budget(args)
    cfg = (
        tiny_config(linear_heads=16, linear_head_dim=128)
        if args.tiny
        else RunConfig.load(args.config or ROOT / "configs/qwen35_2b_wan.yaml")
    )
    _apply_options(cfg, args)
    data = open_release(args, cfg) if args.data_root else None
    _apply_recompute(cfg, args)
    cfg.validate(args.tp)
    if cfg.visual_cache and args.random_init:
        raise ValueError("Pretrained visual caches cannot be used with a randomly initialized encoder")
    return cfg, data, asset_manifests(args, cfg)


def _apply_options(cfg, args):
    for option, field in _CONFIG_OVERRIDES:
        if getattr(args, option) is not None:
            setattr(cfg, field, getattr(args, option))
    if args.preserve_documents:
        cfg.preserve_documents = True
    if args.long_document_policy is not None:
        cfg.long_document_policy = args.long_document_policy
    if args.dit_microbatch_tokens is not None:
        cfg.generation_token_budget = args.dit_microbatch_tokens
    if args.dit_microbatch_images is not None:
        cfg.generation_microbatch_images = args.dit_microbatch_images
    if args.rf is not None:
        cfg.rf = args.rf
    if args.timestep_shift is not None:
        cfg.timestep_shift = args.timestep_shift
    dit_limits = args.dit_microbatch_tokens is not None or args.dit_microbatch_images is not None
    if not cfg.preserve_documents and dit_limits:
        raise ValueError("DiT microbatch flags require --preserve-documents or the whole-document config")
    if args.max_images_per_pack is None:
        args.max_images_per_pack = 0 if cfg.preserve_documents else 4
    if args.synthetic and args.data_root:
        raise ValueError("--synthetic and --data-root are mutually exclusive")
    real_data = args.data_root or args.synthetic
    if cfg.preserve_documents and (not real_data or args.max_images_per_pack):
        raise ValueError("Whole-document training needs real data and max-images-per-pack=0; bound DiT separately")


def open_release(args, cfg):
    """Open the published catalog; its geometry overrides the pixel caps."""
    from data.packing import token_contract
    from data.sources import open_catalog

    catalog = open_catalog(args.data_root, args.source_root, args.shuffle_block_rows)
    membership = catalog.check(args.allow_development_data)
    geometry = catalog.meta.get("training_geometry", {})
    for name in ("vision_max_pixels", "generation_max_pixels"):
        if name in geometry:
            setattr(cfg, name, geometry[name])
    tokens = token_contract(catalog.path, cfg)
    if args.micro_batch != 1:
        raise ValueError("Real variable-layout adapter currently requires the requested microbatch=1")
    return SimpleNamespace(catalog=catalog, membership=membership, tokens=tokens)


def _apply_recompute(cfg, args):
    if args.ce_chunk is not None:
        cfg.ce_chunk = args.ce_chunk
    cfg.keep_attention_activations = args.keep_attention_activations
    if args.no_checkpoint:
        cfg.checkpoint_backbone = cfg.checkpoint_dit = False
    if args.checkpoint:
        cfg.checkpoint_backbone = cfg.checkpoint_dit = True
    if args.checkpoint_backbone_layers or args.checkpoint_dit_layers:
        cfg.checkpoint_backbone = cfg.checkpoint_dit = False
        cfg.checkpoint_backbone_layers = args.checkpoint_backbone_layers
        cfg.checkpoint_dit_layers = args.checkpoint_dit_layers


def asset_manifests(args, cfg):
    from utils.assets import asset_manifest, vision_config

    text = asset_manifest(cfg.text_assets) if cfg.text_assets else None
    if args.micro_batch != 1:
        raise ValueError("Native-resolution packing requires micro_batch=1")
    vision_config(cfg.vision_assets)
    if args.tiny and not args.vision_assets:
        vision = dict(synthetic_tower=vision_config(cfg.vision_assets))
    else:
        vision = asset_manifest(cfg.vision_assets)
    return SimpleNamespace(text=text, vision=vision)
