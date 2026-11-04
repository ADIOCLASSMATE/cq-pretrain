"""Versioned, weights-only visual caches shared by online/offline preparation."""

from __future__ import annotations
import hashlib
import json
from functools import lru_cache
import os
import tempfile
from .manifest import file_sha256
from pathlib import Path
import torch

QWEN_FORMAT = "umm_qwen35_video_v1"
WAN_FORMAT = "umm_wan21_latent_v1"
FORMATS_V2 = {"qwen": "umm_qwen35_video_v2", "wan": "umm_wan21_latent_v2"}


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


def load_cache(path, kind, *, checkpoint=None, preprocessing=None, contract=None, sample_id=None, info=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = QWEN_FORMAT if kind == "qwen" else WAN_FORMAT
    if not isinstance(payload, dict) or payload.get("format") not in {expected, FORMATS_V2[kind]}:
        raise ValueError(f"{path}: expected {expected}")
    meta = payload.get("metadata", {})
    for key in ("clip", "preprocessing", "checkpoint"):
        if key not in meta:
            raise ValueError(f"{path}: cache metadata lacks {key}")
    v2 = payload["format"] == FORMATS_V2[kind]
    if contract is not None and (not v2 or meta.get("encoder") != contract):
        raise ValueError(f"{path}: cache encoder contract differs")
    if sample_id is not None and meta.get("sample_id") != sample_id:
        raise ValueError(f"{path}: cache sample_id differs")
    if v2 and preprocessing is not None:
        preprocessing = branch_preprocessing(preprocessing, kind)
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
    if v2:
        actual = tensor_info(tensor)
        if meta.get("tensor") != actual:
            raise ValueError(f"{path}: stored tensor metadata differs")
        if info is not None:
            for key, value in actual.items():
                if info.get(key) != value:
                    raise ValueError(f"{path}: index tensor {key} differs")
            if info.get("format") != payload["format"]:
                raise ValueError(f"{path}: index cache format differs")
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
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic create: another worker must never replace a completed cache.
        os.link(name, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        Path(name).unlink(missing_ok=True)


def tensor_info(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def branch_preprocessing(settings, kind):
    return {key: value for key, value in settings.items()
            if kind == "qwen" or not key.startswith("qwen_")}


@lru_cache(maxsize=16)
def _content_identity(files):
    return digest([(Path(name).name, file_sha256(name)) for name, _, _ in files])


def content_identity(path, kind):
    path = Path(path).resolve()
    if kind == "wan" and (path / "Wan2.1_VAE.pth").is_file():
        path = path / "Wan2.1_VAE.pth"
    files = [path] if path.is_file() else sorted(
        p for p in path.iterdir() if p.suffix in {".json", ".pth", ".bin", ".safetensors"}
    )
    if not files:
        raise ValueError(f"No model files at {path}")
    return _content_identity(tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files))


def encoder_contracts(config):
    result = {
        "qwen": dict(weights=content_identity(config["model"]["qwen_path"], "qwen"),
                     implementation="qwen35_visual_post_merger_v1",
                     compute_dtype="bfloat16" if config["training"]["mixed_precision"] == "bf16" else "float32",
                     attention=config["model"]["attn_implementation"], storage="native_output"),
        "wan": dict(weights=content_identity(config["codec"].get("path") or config["model"]["wan_path"], "wan"),
                    implementation="wan21_posterior_mode_normalized_v1",
                    compute_dtype=config["codec"]["dtype"], storage="float32"),
    }
    processor = config["model"].get("processor_path")
    if processor and Path(processor).resolve() != Path(config["model"]["qwen_path"]).resolve():
        result["qwen"]["processor"] = content_identity(processor, "qwen")
    return result
