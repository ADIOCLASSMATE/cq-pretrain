"""Versioned resume invariants; unclassified configuration fields stay strict."""

from copy import deepcopy
import json

SCHEMA = "umm_resume_contract_v1"
MANAGEMENT = {
    "training": {
        "exp_id",
        "output_root",
        "output_dir",
        "log_dir",
        "checkpoint_dir",
        "keep_checkpoints",
        "save_every",
        "log_every",
        "val_every",
        "validation_max_batches",
    },
    "data": {"validation_sources"},
    "evaluation": {
        "seed", "tasks", "eval_id", "output_dir", "log_dir", "num_processes", "max_batches",
    },
    "sampling": {
        "steps",
        "solver",
        "cfg",
        "time_shift",
        "seed",
        "max_new_tokens",
        "temperature",
    },
}


def normalized(config):
    result = json.loads(json.dumps(config or {}))
    if isinstance(result.get("flow"), dict):
        result["flow"].setdefault("repeats", 1)
    model = result.get("model")
    if isinstance(model, dict):
        model.setdefault("qwen_gdn_backend", "native")
        # Old exports have neither field; their original implementation is
        # native. A new explicit backend remains a strict training invariant.
        text = model.get("architecture", {}).get("qwen_config", {}).get("text_config")
        if isinstance(text, dict):
            text.setdefault("qwen_gdn_backend", "native")
    return result


def training_contract(config):
    result = normalized(config)
    for section, keys in MANAGEMENT.items():
        if isinstance(result.get(section), dict):
            for key in keys:
                result[section].pop(key, None)
            if not result[section]:
                del result[section]
    return {"schema": SCHEMA, "config": result}


def differences(saved, current, prefix=""):
    if isinstance(saved, dict) and isinstance(current, dict):
        result = []
        for key in sorted(saved.keys() | current.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in saved or key not in current:
                value = saved.get(key, current.get(key))
                if isinstance(value, dict) and value:
                    result.extend(
                        differences(saved.get(key, {}), current.get(key, {}), name)
                    )
                else:
                    result.append(
                        {
                            "field": name,
                            "saved": saved.get(key, "<missing>"),
                            "current": current.get(key, "<missing>"),
                        }
                    )
            else:
                result.extend(differences(saved[key], current[key], name))
        return result
    return (
        []
        if saved == current
        else [{"field": prefix, "saved": saved, "current": current}]
    )


def validate_contract(saved_config, current_config, saved_contract=None):
    expected = training_contract(saved_config)
    # Compare canonical defaults on both sides, including contracts written
    # before the backend field existed, without weakening unknown-field checks.
    if saved_contract is not None and (
        not isinstance(saved_contract, dict)
        or set(saved_contract) != {"schema", "config"}
        or saved_contract["schema"] != SCHEMA
        or normalized(saved_contract["config"]) != expected["config"]
    ):
        raise ValueError(
            "saved resume_contract disagrees with its config or has an unsupported schema"
        )
    changes = differences(
        expected["config"], training_contract(current_config)["config"]
    )
    if changes:
        detail = "\n".join(
            f"  {x['field']}: {x['saved']!r} -> {x['current']!r}" for x in changes
        )
        raise ValueError(
            "checkpoint training configuration contract differs:\n" + detail
        )
    return management_changes(saved_config, current_config)


def management_changes(saved, current):
    def select(config):
        return {
            section: {k: v for k, v in config.get(section, {}).items() if k in keys}
            for section, keys in MANAGEMENT.items()
        }

    return differences(select(normalized(saved)), select(normalized(current)))


def runtime_contract(accelerator):
    if accelerator is None:
        return {"backend": "torch", "world_size": 1}
    if getattr(accelerator, "is_mcore", False):
        return accelerator.runtime_contract()
    return normalized(
        {
            "backend": str(getattr(accelerator, "distributed_type", "NO")).split(".")[
                -1
            ],
            "world_size": int(accelerator.num_processes),
            "device": accelerator.device.type,
            "mixed_precision": accelerator.mixed_precision,
        }
    )
