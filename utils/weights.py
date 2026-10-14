"""Explicit source-to-v3 weight conversion, independent of training state."""
from __future__ import annotations

import json
from pathlib import Path

import torch


SOURCE_COMMIT = "193ec90300af5fee4bc4b7e8a426492a908a0447"


def convert_state_dict(state):
    """Rename the selected joint Z2 parameters; never silently drop tensors."""
    converted = {}
    for key, tensor in state.items():
        if key.startswith("model."):
            target = "backbone." + key[len("model."):]
        elif key.startswith("image_flow_condition_proj."):
            target = "condition_projector." + key[len("image_flow_condition_proj."):]
        elif key.startswith("image_flow_head.net."):
            target = "generator." + key[len("image_flow_head.net."):]
        else:
            target = key
        if target in converted:
            raise ValueError(f"Weight mapping collision: {key} -> {target}")
        converted[target] = tensor
    return converted


def read_weights(path):
    """Return (state, saved config) from local HF or v3 exported weights."""
    path = Path(path)
    if path.is_file():
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path)), {}
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "model" in payload:
            return payload["model"], payload.get("config") or {}
        if "state_dict" in payload:
            return payload["state_dict"], payload.get("config") or {}
        return payload, {}
    if not path.is_dir():
        raise FileNotFoundError(f"Model weights not found: {path}")
    saved = json.loads((path / "config.json").read_text()) if (path / "config.json").is_file() else {}
    files = sorted(path.glob("*.safetensors"))
    if files:
        from safetensors.torch import load_file
        state = {}
        for file in files:
            for key, value in load_file(str(file)).items():
                if key in state:
                    raise ValueError(f"Duplicate tensor {key} in {file}")
                state[key] = value
        return state, saved
    for name in ("weights.pt", "model.pt", "pytorch_model.bin"):
        if (path / name).is_file():
            state, embedded = read_weights(path / name)
            return state, embedded or saved
    raise FileNotFoundError(f"No exported model weights in {path}; use --resume for training checkpoints")


def check_source_architecture(saved):
    values = saved.get("model", saved)
    if values.get("backbone_attention_output_gate", "none") != "none":
        raise ValueError("Attention output gate is not part of the selected baseline")
    if values.get("training_image_sigma_order", "joint") != "joint":
        raise ValueError("Selected baseline requires joint image sigma")
    for name in ("rope_scaling", "rope_parameters"):
        rope = values.get(name)
        if rope and (rope.get("rope_type", rope.get("type", "default")) != "default" or
                     set(rope) - {"rope_type", "type", "rope_theta"}):
            raise ValueError(f"Unsupported {name}: scaled RoPE requires a separate implementation")
    if values.get("joint_dit_head_type", "z2") != "z2":
        raise ValueError("Only the selected Z2 head can be converted without changing architecture")
    if any(values.get(key, False) for key in (
        "joint_dit_dynamic_xt", "joint_dit_dynamic_xt_bidirectional", "joint_dit_text_single_stream",
    )):
        raise ValueError("This checkpoint is an ablation, not the fixed-condition CQ_IT baseline")
    for key, expected in {"flow_condition_contract": "backbone_xt_fixed",
                          "flow_head_attention_contract": "joint_bidirectional",
                          "dual_stream_attention_contract": "xlnet_content_diagonal"}.items():
        if key in values and values[key] != expected:
            raise ValueError(f"Source checkpoint has incompatible {key}={values[key]}")


def load_into_model(model, state, *, backbone_only=False):
    state = convert_state_dict(state)
    target = model.state_dict()
    if model.config.tie_word_embeddings:
        embedding = state.get("backbone.embed_tokens.weight")
        head = state.get("lm_head.weight")
        if embedding is not None and head is not None and not torch.equal(embedding, head):
            raise ValueError("Checkpoint embedding and output head disagree despite tied weights")
        if embedding is not None:
            state["lm_head.weight"] = embedding
        elif head is not None:
            state["backbone.embed_tokens.weight"] = head
    if backbone_only:
        allowed_missing = {key for key in target if not key.startswith("backbone.") or
                           key.startswith("backbone.image_token_embedder.")}
        for key in ("backbone.embed_tokens.weight", "lm_head.weight"):
            if key in state and state[key].shape != target[key].shape:
                old = state[key]
                if old.ndim != 2 or old.shape[1] != target[key].shape[1] or old.shape[0] > target[key].shape[0]:
                    raise ValueError(f"Incompatible pretrained vocabulary: {key}")
                expanded = target[key].clone()
                expanded[:old.shape[0]].copy_(old)
                state[key] = expanded
        if model.config.tie_word_embeddings and "backbone.embed_tokens.weight" in state:
            state["lm_head.weight"] = state["backbone.embed_tokens.weight"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(f"Incomplete base weights: missing={set(missing)-allowed_missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(state, strict=True)
    model.tie_weights()
