"""Transactional training checkpoints and standalone raw/EMA weight export.

All ranks participate. Publication is a same-filesystem directory rename after
the payload inventory is checked. Resume requires the original world size and
training contract. Production uses MCore; plain Torch/Accelerator support is
retained for CPU regression. DeepSpeed/FSDP checkpoints are not supported.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

import torch

from utils.distributed import run_io_phase
from utils.checkpoint_catalog import (
    SCHEMA,
    durable_json,
    fsync_directory,
    inspect_checkpoint,
)
from utils.resume_contract import (
    training_contract,
    validate_contract,
    runtime_contract,
    differences,
)


def _runtime(accelerator=None):
    if accelerator is not None:
        kind = str(getattr(accelerator, "distributed_type", "NO")).upper()
        if "FSDP" in kind or ("MEGATRON" in kind and not getattr(accelerator, "is_mcore", False)):
            raise ValueError(
                "UMM EMA/checkpoint requires replicated parameters (MCore or CPU regression)"
            )
        plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
        if plugin is not None or "DEEPSPEED" in kind:
            raise ValueError("DeepSpeed checkpoints are no longer supported")
        return accelerator
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("distributed checkpointing requires the training Accelerator")
    return SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)


def capture_rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    for name in ("cuda", "npu"):
        device = getattr(torch, name, None)
        if device is not None and device.is_available():
            state[name] = device.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    for name in ("cuda", "npu"):
        if name in state:
            device = getattr(torch, name, None)
            if device is None or not device.is_available():
                raise RuntimeError(f"checkpoint requires unavailable {name} RNG")
            device.set_rng_state_all([value.cpu() for value in state[name]])


@contextmanager
def preserved_rng_state():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


_write_json = durable_json


def _inventory(directory):
    return {
        str(p.relative_to(directory)): p.stat().st_size
        for p in directory.rglob("*")
        if p.is_file()
        and p.name not in {"checkpoint_files.json", "checkpoint_complete.json"}
    }


def validate_checkpoint(
    path, *, world_size=None, expected_config=None, expected_runtime=None
):
    metadata = inspect_checkpoint(path, require_name=False)
    if world_size is not None and int(metadata["world_size"]) != int(world_size):
        raise ValueError("exact resume requires unchanged world size")
    if expected_config is not None:
        validate_contract(
            metadata["config"], expected_config, metadata.get("resume_contract")
        )
    if expected_runtime is not None and metadata.get("runtime_contract") is not None:
        changes = differences(metadata["runtime_contract"], expected_runtime)
        if changes:
            raise ValueError(
                "checkpoint runtime/topology contract differs: " + json.dumps(changes)
            )
    return metadata


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    global_step,
    data_state,
    task_state=None,
    ema=None,
    accelerator=None,
    config=None,
):
    """Save after a completed optimizer step, with the consumed data cursor.

    Accelerator owns wrapped optimizer/scaler/scheduler serialization. The
    native path is for a single process and explicitly stores those objects.
    Caller supplies JSON-compatible configuration describing resume invariants.
    """
    runtime = _runtime(accelerator)
    path = Path(path)
    if int(global_step) < 0:
        raise ValueError("global_step must be nonnegative")
    if ema is not None and int(ema.global_step) != int(global_step):
        raise ValueError("EMA and training global_step differ")
    if accelerator is not None:
        if not getattr(accelerator, "_models", ()) or not getattr(
            accelerator, "_optimizers", ()
        ):
            raise ValueError(
                "prepare model and optimizer with Accelerator before checkpointing"
            )
        if scheduler is not None and not getattr(accelerator, "_schedulers", ()):
            raise ValueError(
                "prepare/register the scheduler with Accelerator before checkpointing"
            )

    def prepare():
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
        staging.mkdir()
        return str(staging)

    staging = Path(
        run_io_phase(
            runtime, prepare, description="prepare checkpoint", main_process_only=True
        )
    )
    backend = "megatron" if getattr(accelerator, "is_mcore", False) else "accelerator" if accelerator is not None else "torch"
    if accelerator is not None:
        run_io_phase(
            runtime,
            lambda: accelerator.save_state(str(staging / backend)),
            description="save training backend state",
        )
    else:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
            },
            staging / "training_state.pt",
        )
    rank_payload = {
        "rng": capture_rng_state(),
        "data_state": data_state,
        "task_state": task_state,
        "global_step": int(global_step),
    }
    run_io_phase(
        runtime,
        lambda: torch.save(
            rank_payload, staging / f"runtime_rank_{runtime.process_index:05d}.pt"
        ),
        description="save rank runtime",
    )
    if ema is not None:
        ema.save_checkpoint(staging / "ema", runtime, global_step=global_step)

    def publish():
        metadata = {
            "schema": SCHEMA,
            "global_step": int(global_step),
            "world_size": int(runtime.num_processes),
            "config": config,
            "backend": backend,
            "ema": ema is not None,
            "resume_contract": training_contract(config),
            "runtime_contract": runtime_contract(accelerator),
        }
        _write_json(staging / "metadata.json", metadata)
        files = _inventory(staging)
        if metadata["backend"] == "accelerator" and not any(
            k.startswith("accelerator/") for k in files
        ):
            raise ValueError("Accelerator did not write state")
        _write_json(staging / "checkpoint_files.json", files)
        _write_json(
            staging / "checkpoint_complete.json",
            {"schema": SCHEMA, "global_step": int(global_step)},
        )
        validate_checkpoint(staging, world_size=runtime.num_processes)
        if path.exists():
            raise FileExistsError(f"checkpoint appeared during publication: {path}")
        os.replace(staging, path)
        fsync_directory(path.parent)

    run_io_phase(
        runtime, publish, description="publish checkpoint", main_process_only=True
    )
    return path


def load_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    ema=None,
    accelerator=None,
    expected_config=None,
):
    """Restore an internally generated, trusted training checkpoint."""
    runtime = _runtime(accelerator)
    path = Path(path)
    metadata = run_io_phase(
        runtime,
        lambda: validate_checkpoint(
            path,
            world_size=runtime.num_processes,
            expected_config=expected_config,
            expected_runtime=runtime_contract(accelerator),
        ),
        description="checkpoint preflight",
        main_process_only=True,
    )
    if bool(metadata["ema"]) != (ema is not None):
        raise ValueError("resume must preserve EMA enabled/disabled state")
    backend = "megatron" if getattr(accelerator, "is_mcore", False) else "accelerator" if accelerator is not None else "torch"
    if metadata["backend"] != backend:
        raise ValueError("resume requires the same training backend")
    rank_payload = run_io_phase(
        runtime,
        lambda: torch.load(
            path / f"runtime_rank_{runtime.process_index:05d}.pt",
            map_location="cpu",
            weights_only=False,
        ),
        description="read runtime state",
    )
    if int(rank_payload["global_step"]) != int(metadata["global_step"]):
        raise ValueError("rank runtime step differs from published checkpoint")
    if accelerator is not None:
        run_io_phase(
            runtime,
            lambda: accelerator.load_state(str(path / backend)),
            description="restore training backend state",
        )
    else:
        payload = torch.load(
            path / "training_state.pt", map_location="cpu", weights_only=True
        )
        state = model.state_dict()
        if set(state) != set(payload["model"]) or any(
            state[k].shape != payload["model"][k].shape for k in state
        ):
            raise ValueError("checkpoint model structure mismatch")
        if (payload["scheduler"] is None) != (scheduler is None):
            raise ValueError("scheduler presence changed on resume")
        saved_groups = payload["optimizer"]["param_groups"]
        current_groups = optimizer.state_dict()["param_groups"]
        if len(saved_groups) != len(current_groups) or any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(saved_groups, current_groups)
        ):
            raise ValueError("optimizer parameter grouping changed on resume")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
    if ema is not None:
        ema.load_checkpoint(
            path / "ema", runtime, expected_global_step=metadata["global_step"]
        )
    restore_rng_state(rank_payload["rng"])
    return {
        "global_step": int(metadata["global_step"]),
        "data_state": rank_payload["data_state"],
        "task_state": rank_payload["task_state"],
        "metadata": metadata,
    }


def export_weights(path, *, model=None, ema_checkpoint=None, config=None, dtype=None):
    """Write a standalone full state dict (including tied aliases) atomically.

    Export accepts either a replicated model or a complete EMA shard directory.
    The result contains weights/config only; it is not a resumable checkpoint.
    """
    if (model is None) == (ema_checkpoint is None):
        raise ValueError("choose exactly one of model or ema_checkpoint")
    if ema_checkpoint is not None:
        from training.ema import merge_sharded_ema_state_dict

        source = merge_sharded_ema_state_dict(ema_checkpoint)
    else:
        source = model.state_dict()
    from training.ema import _tensor_view_identity

    state, storage = {}, {}
    for key, value in source.items():
        identity = _tensor_view_identity(value)
        if identity not in storage:
            storage[identity] = (
                value.detach()
                .to(
                    device="cpu",
                    dtype=dtype
                    if dtype is not None and value.is_floating_point()
                    else value.dtype,
                )
                .clone()
            )
        state[key] = storage[identity]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    torch.save(
        {
            "schema": "umm_weights_v1",
            "model": state,
            "config": config,
            "weights": "ema" if ema_checkpoint is not None else "raw",
        },
        temporary,
    )
    os.replace(temporary, path)
    return path
