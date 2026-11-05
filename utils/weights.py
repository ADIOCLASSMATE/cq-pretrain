"""Strict UMM video exports. Native components load in their owning modules."""

from pathlib import Path
import torch


def read_weights(path):
    path = Path(path)
    if path.is_dir():
        path = path / "weights.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "umm_weights_v2" or not isinstance(
        payload.get("model"), dict
    ):
        raise ValueError(
            "Expected a UMM weights.pt export; use --resume for training checkpoints"
        )
    saved = payload.get("config") or {}
    architecture = saved.get("model", {}).get("architecture")
    if (
        not architecture
        or architecture.get("backbone_type") != "qwen35"
        or architecture.get("generator_type") != "wan21"
    ):
        raise ValueError(
            "Expected Qwen3.5 + Wan2.1 export; legacy weight formats are unsupported"
        )
    return payload["model"], saved


def load_into_model(model, state):
    model.load_state_dict(state, strict=True)
