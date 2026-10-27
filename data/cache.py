"""Versioned, weights-only visual caches shared by online/offline preparation."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch

QWEN_FORMAT = "umm_qwen35_video_v1"
WAN_FORMAT = "umm_wan21_latent_v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def checkpoint_id(path):
    path = Path(path).resolve()
    files = (
        [path]
        if path.is_file()
        else sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix in {".json", ".pth", ".bin", ".safetensors"}
        )
    )
    return digest([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files])


def cache_identities(config):
    return {
        "qwen": checkpoint_id(config["model"]["qwen_path"]),
        "wan": checkpoint_id(
            config["codec"].get("path") or config["model"]["wan_path"]
        ),
    }


def load_cache(path, kind, *, checkpoint=None, preprocessing=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = QWEN_FORMAT if kind == "qwen" else WAN_FORMAT
    if not isinstance(payload, dict) or payload.get("format") != expected:
        raise ValueError(f"{path}: expected {expected}")
    meta = payload.get("metadata", {})
    for key in ("clip", "preprocessing", "checkpoint"):
        if key not in meta:
            raise ValueError(f"{path}: cache metadata lacks {key}")
    if checkpoint is not None and meta["checkpoint"] != checkpoint:
        raise ValueError(f"{path}: cache checkpoint differs")
    if preprocessing is not None and meta["preprocessing"] != preprocessing:
        raise ValueError(f"{path}: cache preprocessing differs; regenerate this cache")
    tensor = payload.get("features" if kind == "qwen" else "latent")
    if (
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_floating_point()
        or not torch.isfinite(tensor).all()
    ):
        raise ValueError(f"{path}: invalid cached tensor")
    if kind == "qwen":
        grid = payload.get("grid_thw")
        if (
            not isinstance(grid, torch.Tensor)
            or grid.shape != (3,)
            or grid.dtype != torch.long
            or bool((grid <= 0).any())
        ):
            raise ValueError(f"{path}: invalid Qwen grid")
        if tensor.ndim != 2 or tensor.shape[0] != int(grid.prod()) // 4:
            raise ValueError(f"{path}: feature count does not match native merger grid")
        if not meta.get("qwen_timestamps"):
            raise ValueError(f"{path}: missing sampled frame timestamps")
    else:
        if (
            tensor.ndim != 4
            or tensor.shape[0] != 16
            or tensor.shape[2] % 2
            or tensor.shape[3] % 2
        ):
            raise ValueError(f"{path}: expected Wan latent [16,T,H,W], even H/W")
        clip = meta["clip"]
        shape = (
            16,
            (clip["num_frames"] - 1) // 4 + 1,
            clip["height"] // 8,
            clip["width"] // 8,
        )
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{path}: latent shape does not match cached clip")
    return payload


def save_cache(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {path}")
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
