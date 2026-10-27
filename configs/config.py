"""Qwen3.5 + Wan2.1 configuration; test overrides live in configs/tests/."""

from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path
import yaml

MODEL_ROOT = "/inspire/sj-ssd3/project/high-dimensionaldata/public/models"
DEFAULTS = {
    "distributed": {
        "backend": "mindspeed",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "overlap_grad_reduce": True,
        "overlap_param_gather": True,
        "bucket_size": 40000000,
        "timeout_minutes": 60,
        "num_processes": 4,
    },
    "model": {
        "qwen_path": f"{MODEL_ROOT}/Qwen3.5-2B",
        "wan_path": f"{MODEL_ROOT}/Wan2.1-T2V-1.3B",
        "checkpoint": None,
        "from_scratch": False,
        "freeze_vision": True,
        "gradient_checkpointing": True,
        "attn_implementation": "sdpa",
        "qwen_gdn_backend": "native",
        "wan_attention_backend": "sdpa",
        "train_language": True,
        "train_wan": True,
        "train_condition_proj": True,
        "lambda_text": 0.05,
        "lambda_video": 1.0,
        "video_uncond_prob": 0.1,
    },
    "data": {
        "sources": [],
        "validation_sources": [],
        "mixing": "balanced",
        "task_weights": {"t2v": 1, "v2t": 1},
        "max_seq_length": 8192,
        "pad_to_multiple_of": 8,
        "representations": {"qwen": "prefer_cache", "wan": "prefer_cache"},
        "video": {
            "num_frames": 9,
            "fps": 4.0,
            "height": 432,
            "width": 768,
            "qwen_num_frames": None,
            "qwen_processor": {},
        },
    },
    "training": {
        "exp_id": "umm_video",
        "output_root": "./outputs",
        "output_dir": None,
        "log_dir": None,
        "checkpoint_dir": None,
        "keep_checkpoints": 3,
        "device": "npu",
        "max_train_steps": 100000,
        "gradient_accumulation_steps": 8,
        "mixed_precision": "bf16",
        "seed": 42,
        "deterministic": False,
        "max_grad_norm": 1.0,
        "batch_size": 1,
        "save_every": 2000,
        "log_every": 10,
        "val_every": 0,
        "validation_max_batches": None,
        "use_ema": True,
        "ema_decay": 0.9997,
        "ema_update_after_step": 0,
        "ema_shard_chunk_numel": 4194304,
        "dataloader_workers": 2,
        "dataloader_prefetch_factor": 2,
    },
    "optimizer": {
        "backbone_lr": 1e-5,
        "generator_lr": 1e-4,
        "projector_lr": 1e-4,
        "embedding_lr": 1e-5,
        "vision_lr": 1e-5,
        "weight_decay": 0.1,
        "generator_weight_decay": 0.0,
        "betas": [0.9, 0.95],
        "eps": 1e-6,
        "warmup_steps": 1000,
        "decay_steps": 10000,
        "min_lr_ratio": 0.1,
    },
    "flow": {
        "repeats": 1,
        "time_sampling": "logit_normal",
        "logit_mean": 0.0,
        "logit_std": 1.0,
        "uniform_mix": 0.0,
        "time_eps": 1e-5,
        "timestep_scale": 1000.0,
    },
    "sampling": {
        "steps": 50,
        "solver": "euler",
        "cfg": 5.0,
        "time_shift": 5.0,
        "seed": 42,
        "max_new_tokens": 128,
        "temperature": 0.0,
    },
    "cache_identity": {},
    "evaluation": {
        "seed": 424242,
        "tasks": [],
        "eval_id": None,
        "output_dir": None,
        "log_dir": None,
        "num_processes": None,
        "max_batches": None,
    },
    "codec": {"path": None, "dtype": "float32"},
}


def merge_config(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        result[key] = (
            merge_config(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def load_config(path, overrides=None, *, purpose="train"):
    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError("Configuration must be a mapping of supported sections")
    config = merge_config(DEFAULTS, values)
    for override in overrides or ():
        if "=" not in override:
            raise ValueError("Overrides must use section.key=value")
        key, raw = override.split("=", 1)
        parts = key.split(".")
        if len(parts) < 2 or parts[0] not in DEFAULTS:
            raise ValueError(f"Unknown override: {key}")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Override traverses a scalar: {key}")
        target[parts[-1]] = yaml.safe_load(raw)
    # Task YAML paths are relative to their containing run configuration. Other
    # existing data/model paths retain their established project-root semantics.
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    tasks = evaluation.get("tasks", [])
    if isinstance(tasks, list):
        config["evaluation"]["tasks"] = [
            str((Path(path).resolve().parent / Path(item).expanduser()).resolve())
            if isinstance(item, str) else item
            for item in tasks
        ]
    validate_config(config, purpose=purpose)
    return config


def validate_config(config, *, purpose="train"):
    if purpose not in {"train", "eval"}:
        raise ValueError("Configuration purpose must be train or eval")
    train, model, data, opt, flow = (
        config[k] for k in ("training", "model", "data", "optimizer", "flow")
    )
    if train["mixed_precision"] is False:  # YAML 1.1 parses unquoted `no` as false.
        train["mixed_precision"] = "no"
    resolve_output_paths(config)
    if type(train["deterministic"]) is not bool:
        raise ValueError("training.deterministic must be boolean")
    distributed = config["distributed"]
    if set(distributed) - set(DEFAULTS["distributed"]):
        raise ValueError("Unknown distributed setting")
    if distributed["backend"] not in {"accelerate", "mindspeed"}:
        raise ValueError("distributed.backend must be accelerate or mindspeed")
    if distributed["backend"] == "mindspeed":
        if train["device"] != "npu" or train["mixed_precision"] != "bf16":
            raise ValueError("MindSpeed requires NPU and BF16")
        if any(distributed[key] != 1 for key in (
            "tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"
        )):
            raise ValueError("This UMM adapter supports TP=PP=CP=1 with DP and a distributed optimizer")
    for key in ("overlap_grad_reduce", "overlap_param_gather"):
        if type(distributed[key]) is not bool:
            raise ValueError(f"distributed.{key} must be boolean")
    for key in ("bucket_size", "timeout_minutes", "num_processes"):
        if type(distributed[key]) is not int or distributed[key] < 1:
            raise ValueError(f"distributed.{key} must be positive")
    keep = train.get("keep_checkpoints", 3)
    if keep is not None and (type(keep) is not int or keep < 1):
        raise ValueError("training.keep_checkpoints must be a positive integer or null")
    allowed_model = set(DEFAULTS["model"]) | {
        "architecture",
        "processor_path",
        "mask_token_id",
        "video_mask_token_id",
    }
    if set(model) - allowed_model:
        raise ValueError(
            f"Unknown model settings (legacy image configs are unsupported): {sorted(set(model) - allowed_model)}"
        )
    if model["qwen_gdn_backend"] not in {"native", "mindspeed"}:
        raise ValueError("model.qwen_gdn_backend must be native or mindspeed")
    if model["qwen_gdn_backend"] == "mindspeed" and (
        train["device"] != "npu" or train["mixed_precision"] != "bf16"
    ):
        raise ValueError("MindSpeed GDN requires NPU and BF16")
    if set(data) - (set(DEFAULTS["data"]) | {"schedule"}):
        raise ValueError("Unsupported data settings; use the unified video schema")
    if data["mixing"] not in {"balanced", "preserve"}:
        raise ValueError("data.mixing must be balanced or preserve")
    weights = data["task_weights"]
    if (
        set(weights) - {"t2v", "v2t"}
        or any(type(x) is not int or x < 0 for x in weights.values())
        or not sum(weights.values())
    ):
        raise ValueError(
            "task_weights must be nonnegative integer T2V/V2T exposure counts"
        )
    data["schedule"] = (
        [t for t, n in weights.items() for _ in range(n)]
        if data["mixing"] == "balanced"
        else ["mixed"]
    )
    if purpose == "train":
        accumulation = int(train["gradient_accumulation_steps"])
        if accumulation < 1 or accumulation % len(data["schedule"]):
            raise ValueError(
                "gradient_accumulation_steps must contain whole task schedule cycles"
            )
    for key in (("batch_size", "max_train_steps", "log_every") if purpose == "train" else ("batch_size",)):
        if int(train[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if train["device"] not in {"npu", "cuda", "cpu"} or train[
        "mixed_precision"
    ] not in {"no", "bf16"}:
        raise ValueError("Use cpu/cuda/npu and mixed_precision=no/bf16")
    if any(
        int(train[k]) < 0
        for k in (
            "dataloader_workers",
            "save_every",
            "val_every",
            "ema_update_after_step",
        )
    ):
        raise ValueError("Worker and interval counts must be nonnegative")
    if not 0 <= float(train["ema_decay"]) < 1:
        raise ValueError("ema_decay must be in [0,1)")
    for kind, mode in data["representations"].items():
        if kind not in {"qwen", "wan"} or mode not in {
            "cache",
            "online",
            "prefer_cache",
        }:
            raise ValueError("Representations must use cache, online, or prefer_cache")
    if not model["freeze_vision"] and data["representations"]["qwen"] != "online":
        raise ValueError("Unfrozen ViT requires data.representations.qwen=online")
    video = data["video"]
    if int(video["num_frames"]) < 5 or (int(video["num_frames"]) - 1) % 4:
        raise ValueError("Wan video frame count must be 4k+1, k >= 1")
    if float(video["fps"]) <= 0 or any(
        int(video[k]) < 16 or int(video[k]) % 16 for k in ("height", "width")
    ):
        raise ValueError("Use positive fps and H/W divisible by 16")
    if video["qwen_num_frames"] is not None and not 2 <= int(
        video["qwen_num_frames"]
    ) <= int(video["num_frames"]):
        raise ValueError("Qwen frame count must be within [2, num_frames]")
    if int(data["max_seq_length"]) < 1 or int(data["pad_to_multiple_of"]) < 1:
        raise ValueError("Sequence and padding sizes must be positive")
    for key in ("lambda_text", "lambda_video"):
        if not math.isfinite(float(model[key])) or float(model[key]) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if type(flow.get("repeats", 1)) is not int or flow.get("repeats", 1) < 1:
        raise ValueError("flow.repeats must be a positive integer")
    for value in (model["video_uncond_prob"], flow["uniform_mix"]):
        if not 0 <= float(value) <= 1:
            raise ValueError("CFG and time mixture probabilities must lie in [0,1]")
    if (
        flow["time_sampling"] not in {"uniform", "logit_normal"}
        or not 0 <= float(flow["time_eps"]) < 0.5
    ):
        raise ValueError("Invalid flow time settings")
    if float(flow["logit_std"]) <= 0 or float(flow["timestep_scale"]) <= 0:
        raise ValueError("Flow scale/std must be positive")
    if purpose == "train" and (min(int(opt["warmup_steps"]), int(opt["decay_steps"])) < 0 or int(
        opt["warmup_steps"]
    ) + int(opt["decay_steps"]) > int(train["max_train_steps"])):
        raise ValueError("Warmup + decay must fit max_train_steps")
    if not 0 <= float(opt["min_lr_ratio"]) <= 1:
        raise ValueError("min_lr_ratio must be in [0,1]")
    evaluation = config["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) - set(DEFAULTS["evaluation"]):
        raise ValueError("Unknown evaluation setting")
    if not isinstance(evaluation["tasks"], list) or any(
        not isinstance(item, str) or not item for item in evaluation["tasks"]
    ):
        raise ValueError("evaluation.tasks must be a list of task YAML paths")
    # Preserve the legacy validation cap: dict.get(key, fallback) does not
    # inherit when the new optional key exists with a null default.
    if evaluation.get("max_batches") is None:
        evaluation["max_batches"] = train.get("validation_max_batches")
    for key in ("num_processes", "max_batches"):
        value = evaluation.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"evaluation.{key} must be positive or null")
    if type(evaluation["seed"]) is not int or evaluation["seed"] < 0:
        raise ValueError("evaluation.seed must be a nonnegative integer")
    if (
        config["sampling"]["solver"] not in {"euler", "heun"}
        or int(config["sampling"]["steps"]) < 1
    ):
        raise ValueError("Sampling requires Euler/Heun and positive steps")


def resolve_output_paths(config):
    """Resolve management paths independently of the launch working directory."""
    train = config["training"]
    project = Path(__file__).resolve().parents[1]

    def absolute(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else project / path).resolve()

    if train.get("output_dir"):
        output = absolute(train["output_dir"])
    else:
        exp_id = train.get("exp_id", "umm_video")
        if (
            not isinstance(exp_id, str)
            or not exp_id
            or exp_id in {".", ".."}
            or "/" in exp_id
            or "\\" in exp_id
        ):
            raise ValueError("training.exp_id must be a single nonempty directory name")
        output = absolute(train.get("output_root", "./outputs")) / exp_id
    logs = absolute(train["log_dir"]) if train.get("log_dir") else output / "logs"
    checkpoints = (
        absolute(train["checkpoint_dir"])
        if train.get("checkpoint_dir")
        else output / "checkpoints"
    )
    if (
        logs == checkpoints
        or logs in checkpoints.parents
        or checkpoints in logs.parents
    ):
        raise ValueError("log_dir and checkpoint_dir must be separate directory trees")
    return {"output_dir": output, "log_dir": logs, "checkpoint_dir": checkpoints}


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
