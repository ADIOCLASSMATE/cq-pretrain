"""Native Qwen asset paths and pinned checkpoint identity."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def vision_config(path):
    cfg = json.loads((asset_path(path) / "config.json").read_text())["vision_config"]
    if cfg.get("deepstack_visual_indexes"):
        raise ValueError("This adapter targets the Qwen3.5 tower without DeepStack")
    if (cfg["patch_size"], cfg["temporal_patch_size"], cfg["spatial_merge_size"]) != (16, 2, 2):
        raise ValueError("Expected Qwen3.5 patch16 / temporal2 / merge2 checkpoint")
    if cfg["hidden_act"] != "gelu_pytorch_tanh":
        raise ValueError("Unsupported vision MLP activation")
    return cfg


def asset_manifest(path, verify=False):
    root = asset_path(path)
    data = json.loads((root / "manifest.json").read_text())
    allowed = {
        ("Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
        ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
        ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
        # Base checkpoint: identical architecture, vocab and merges; it only
        # lacks the four post-training tokens (<think>, tool responses).
        ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"),
    }
    if (data["model_id"], data["revision"]) not in allowed:
        raise ValueError("Unexpected Qwen3.5 checkpoint identity")
    if verify:
        for name, digest in data["files"].items():
            if sha256(root / name) != digest:
                raise ValueError(f"Qwen3.5 asset checksum changed: {name}")
    return data
