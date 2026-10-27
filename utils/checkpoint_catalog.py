"""Checkpoint discovery, commit metadata and retention (standard library only)."""

from contextlib import AbstractContextManager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import socket
import uuid

SCHEMA = "umm_training_checkpoint_v1"
NAME = re.compile(r"checkpoint-(\d+)")


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class DirectoryLock(AbstractContextManager):
    """One writer per root. Keep the inode: unlinking a lock permits split ownership."""

    def __init__(self, root):
        self.root = Path(root)
        self.handle = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        handle = (self.root / ".writer.lock").open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(f"Training writer already owns {self.root}") from exc
        self.handle = handle
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
        handle.flush()
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def inspect_checkpoint(path, *, require_name=True):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Checkpoint must be a real directory")
    marker = json.loads((path / "checkpoint_complete.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    inventory = json.loads((path / "checkpoint_files.json").read_text())
    if not all(isinstance(value, dict) for value in (marker, metadata, inventory)):
        raise ValueError(
            "checkpoint metadata, completion marker and inventory must be JSON objects"
        )
    if metadata.get("schema") != SCHEMA or marker.get("schema") != SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    step, world = metadata["global_step"], metadata["world_size"]
    if type(step) is not int or step < 0 or type(world) is not int or world < 1:
        raise ValueError("invalid checkpoint step/world size")
    if marker.get("global_step") != step:
        raise ValueError("checkpoint completion step mismatch")
    if require_name or NAME.fullmatch(path.name):
        if path.name != f"checkpoint-{step}":
            raise ValueError("checkpoint directory/step mismatch")
    required = {
        "metadata.json",
        *(f"runtime_rank_{rank:05d}.pt" for rank in range(world)),
    }
    if metadata["backend"] == "torch":
        required.add("training_state.pt")
    elif metadata["backend"] == "accelerator":
        if not any(name.startswith("accelerator/") for name in inventory):
            raise ValueError("missing Accelerator state")
    elif metadata["backend"] == "megatron":
        required.add("megatron/model.pt")
        required.update(f"megatron/optimizer_rank_{rank:05d}.pt" for rank in range(world))
    else:
        raise ValueError("unsupported checkpoint backend")
    if metadata["ema"]:
        required.add("ema/ema_manifest.json")
        required.update(
            f"ema/ema_shard_rank_{rank:05d}.safetensors" for rank in range(world)
        )
    if not required <= set(inventory):
        raise ValueError(
            f"missing checkpoint payload: {sorted(required - set(inventory))}"
        )
    for name, size in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid checkpoint inventory path")
        payload = path / relative
        if any(
            (path / Path(*relative.parts[:i])).is_symlink()
            for i in range(1, len(relative.parts) + 1)
        ):
            raise ValueError(f"symlink in checkpoint payload: {name}")
        if (
            type(size) is not int
            or size <= 0
            or not payload.is_file()
            or payload.stat().st_size != size
        ):
            raise ValueError(f"missing or truncated checkpoint file: {name}")
    return metadata


def discover(root):
    candidates = sorted(
        (p for p in Path(root).glob("checkpoint-*") if NAME.fullmatch(p.name)),
        key=lambda p: int(NAME.fullmatch(p.name)[1]),
        reverse=True,
    )
    valid, rejected = [], []
    for path in candidates:
        try:
            metadata = inspect_checkpoint(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append(
                {
                    "path": str(path),
                    "reason": str(exc),
                    "committed": (path / "checkpoint_complete.json").exists(),
                }
            )
        else:
            valid.append((path, metadata))
    return valid, rejected


def latest(root):
    valid, rejected = discover(root)
    if valid:
        return str(valid[0][0]), rejected
    root = Path(root)
    if (
        rejected
        or (root / "latest.json").exists()
        or (root / ".training_started.json").exists()
    ):
        raise RuntimeError(
            f"No intact checkpoint; refusing to restart training from initialization: {rejected}"
        )
    return None, rejected


def quarantine_invalid(root):
    """After successful recovery only; preserve rejected payloads for inspection."""
    root = Path(root)
    _, rejected = discover(root)
    paths = [Path(item["path"]) for item in rejected]
    paths.extend(
        p
        for p in root.glob(".checkpoint-*.partial-*")
        if re.fullmatch(r"\.checkpoint-\d+\.partial-[0-9a-f]+", p.name)
    )
    moved = []
    for path in paths:
        destination = root / "quarantine" / f"{path.name}-{uuid.uuid4().hex[:8]}"
        destination.parent.mkdir(exist_ok=True)
        os.replace(path, destination)
        moved.append({"from": str(path), "to": str(destination)})
    if moved:
        fsync_directory(root / "quarantine")
        fsync_directory(root)
    return moved


def commit_and_prune(path, keep):
    """Only call under the root's writer lock, after publishing a new checkpoint."""
    path = Path(path)
    metadata = inspect_checkpoint(path)
    if keep is not None and (type(keep) is not int or keep < 1):
        raise ValueError("Retain at least one checkpoint, or use null")
    valid, _ = discover(path.parent)
    if not valid or valid[0][0] != path:
        raise ValueError(
            "Refusing to publish an older checkpoint; branch into a new checkpoint_dir"
        )
    durable_json(
        path.parent / "latest.json",
        {"path": path.name, "global_step": metadata["global_step"]},
    )
    removed = []
    for candidate, _ in valid[keep:] if keep is not None else ():
        shutil.rmtree(candidate)
        removed.append(str(candidate))
    fsync_directory(path.parent)
    return removed
