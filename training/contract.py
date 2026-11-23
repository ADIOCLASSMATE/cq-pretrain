"""The resume contract and the run record.

`training_state` is stored in every checkpoint and compared exactly on resume:
anything that changes the meaning of already-taken steps belongs here.
`run_record` is the descriptive run.json; it is never compared.
"""

import hashlib
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist

import distributed.accel as accel
import distributed.megatron as P

ROOT = Path(__file__).resolve().parents[1]
DEV = accel.device_module()

# Source files whose bytes define the backbone numerics.
BACKBONE_SOURCES = (
    "models/backbone_qwen35.py",
    "ops/gated_delta_layout.py",
    "models/cq_backbone.py",
    "models/modeling_umm.py",
    "data/batch_layout.py",
    "models/cq_layout.py",
    "ops/attention.py",
    "ops/attention_layout.py",
    "ops/flex_cq.py",
    "ops/cuda_gated_delta.py",
    "ops/cuda_fused.py",
    "ops/cuda_cross_entropy.py",
    "ops/cuda_compile.py",
    "training/renderer_schedule.py",
    "training/loss.py",
    "ops/cross_entropy.py",
    "ops/functional.py",
    "distributed/megatron.py",
)


def sha256_of(*relative_paths):
    return hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in relative_paths)).hexdigest()


def training_state(args, cfg, ema, assets, data):
    state = {
        key: getattr(args, key)
        for key in (
            "lr",
            "min_lr",
            "lr_decay_style",
            "lr_warmup_steps",
            "lr_decay_steps",
            "seed",
            "sequence_length",
            "images",
            "micro_batch",
            "grad_accum",
        )
    }
    if args.lr_decay_style == "wsd":
        # The tail of a WSD schedule is decided while training; it never
        # changes steps already taken, so it is not part of the resume contract.
        state.pop("lr_decay_steps")
    state["renderer_schedule_sha256"] = sha256_of("training/renderer_schedule.py")
    state["initialization_checkpoint"] = str(args.init_checkpoint) if args.init_checkpoint else None
    state["deterministic"] = args.deterministic
    state["optimizer_offload_fraction"] = args.optimizer_offload_fraction
    state["cuda_gdn_sha256"] = sha256_of("ops/cuda_gated_delta.py")
    state["grad_reduce_in_fp32"] = not args.bf16_grad_reduce
    state["kernel_options"] = kernel_options(args, cfg)
    state["qwen35_assets"] = assets.text
    if ema is not None:
        state["renderer_ema"] = dict(
            schema=ema.schema,
            module="renderer",
            decay=ema.decay,
            dtype="float32",
            adapter_sha256=sha256_of("training/ema.py"),
        )
    from data.cache import encoder_identity

    state["wan_assets"] = dict(
        generator=encoder_identity(cfg.wan_assets), vae=encoder_identity(cfg.wan_vae_assets)
    )
    state["wan_adapter_sha256"] = sha256_of("models/generator_wan21.py")
    state["backbone_adapter_sha256"] = sha256_of(*BACKBONE_SOURCES)
    state["vision_assets"] = assets.vision
    state["vision_adapter_sha256"] = sha256_of("models/vision_encoder.py", "models/cq_backbone.py")
    if data is not None:
        state["data"] = data_contract(args, cfg, data)
    return state


def kernel_options(args, cfg):
    """A kernel switch that changes speed or numerics must be attributable from the run record alone."""
    import ops.cuda_fused as cuda_fused
    from ops import cuda_cross_entropy, flex_cq
    from ops.functional import _COMPILE_MODULATION

    cuda = accel.BACKEND == "cuda"
    return dict(
        compile_modulation=_COMPILE_MODULATION,
        attention_tile=os.environ.get("PRETRAIN_ATTENTION_TILE", "0"),
        cuda_fused_norm=cuda_fused._ENABLED,
        backend=accel.BACKEND,
        cuda_fused_swiglu=cuda_fused._SWIGLU_ENABLED if cuda else None,
        cuda_fused_ops=cuda_fused._OPS_ENABLED,
        grad_reduce_in_fp32=not args.bf16_grad_reduce,
        flex_attention=flex_cq._ENABLED,
        flex_tiles=(
            flex_cq.tile_options(cfg.head_dim, "%d%d" % torch.cuda.get_device_capability())
            if flex_cq._ENABLED and cuda
            else None
        ),
        cuda_fused_ce=cuda_cross_entropy._ENABLED,
    )


def data_contract(args, cfg, data):
    from data.tokenizer import tokenizer_metadata

    online_tokenizer = tokenizer_metadata(cfg, data.catalog.meta["tokenizer"])
    contract = dict(
        snapshot=data.catalog.meta["identity"],
        membership=data.membership["identity"],
        mode=args.data_mode,
        vae_dtype=args.vae_dtype,
        tokenizer=online_tokenizer,
        tokenizer_counting_matches_corpus=online_tokenizer == data.catalog.meta["tokenizer"],
        latent_encoder=args.latent_encoder if args.data_mode == "A" else None,
        max_images_per_pack=args.max_images_per_pack,
        pack_padding_multiple=args.pack_padding_multiple,
    )
    if args.data_epochs > 1:
        contract["epochs"] = args.data_epochs
    adapter_files = sorted((ROOT / "data").glob("*.py"))
    contract["adapter_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in adapter_files)).hexdigest()
    return contract


def run_record(args, cfg, run, model, ema, state, assets, data, pipeline, loaded, startup, parameters, scope):
    """Write run.json (rank zero) and return the record and this rank's placement."""
    total, trainable = parameters
    record = dict(
        torch=torch.__version__,
        device=DEV.get_device_name(DEV.current_device()),
        total_memory=DEV.get_device_properties(DEV.current_device()).total_memory,
        world_size=run.world,
        topology=run.topology,
        model=cfg.to_dict(),
        parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
        training_scope=scope,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        global_batch=run.global_batch,
        capacity_slots_per_update_upper_bound=run.global_batch * args.sequence_length,
        training_backend="Megatron Core / NCCL",
        attention_backend="upstream flash-linear-attention GatedDeltaNet + PyTorch SDPA",
        measured_on=accel.accel_type(),
        cuda=torch.version.cuda,
        synthetic=not bool(args.data_root),
        pretrained_weights=loaded,
        qwen35_assets=assets.text,
        startup=startup,
    )
    if ema is not None:
        record["renderer_ema"] = dict(
            **ema.metadata(), local_storage_bytes=ema.local_bytes, local_optimizer_shards=len(ema.shadows)
        )
    record["kernel_options"] = state["kernel_options"]
    if pipeline is not None:
        record["data"] = dict(
            snapshot=data.catalog.meta,
            membership=data.membership,
            online_tokenizer=state["data"]["tokenizer"],
            tokenizer_counting_matches_corpus=pipeline.docs.counting_matches_corpus,
            mode=args.data_mode,
            latent_grid="dynamic",
            actual_generation_image_tokens="dynamic",
            generation_max_pixels=cfg.generation_max_pixels,
            generation_token_budget=cfg.generation_token_budget,
            rf=cfg.rf,
            timestep_shift=cfg.timestep_shift,
            query_interaction=cfg.query_interaction,
            null_condition=cfg.null_condition,
            fake_latents=args.latent_encoder == "fake",
            local_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    from utils.experiment import source_hashes
    import utils.dependencies as stack

    record["source_sha256"] = source_hashes()
    record["stack"] = stack.name()
    record["dependencies"] = json.loads(stack.lock_path().read_text())
    record["installed_versions"] = stack.installed_versions()
    record["nccl"] = torch.cuda.nccl.version()
    host = dict(
        rank=run.rank,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        hostname=socket.gethostname(),
        tp_rank=P.state().get_tensor_model_parallel_rank(),
        dp_rank=P.dp_rank(),
    )
    hosts = [None] * run.world
    dist.all_gather_object(hosts, host)
    record["rank_placement"] = hosts
    if run.rank == 0:
        (run.output / "run.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    return record, host
