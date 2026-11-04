"""Atomic manifests, durable per-part journals, and reproducible identities."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def journal_latest(path, *, repair_tail=False, per_variant=False):
    """A torn last append is recoverable; malformed committed lines are errors."""
    path = Path(path)
    latest = {}
    if not path.exists():
        return latest
    with path.open("r+b" if repair_tail else "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if repair_tail:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            row = json.loads(line)
            key = (row["sample_id"], row["variant"]) if per_variant else row["sample_id"]
            latest[key] = row
    return latest


def append_journal(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def require_readable(value):
    """Public records contain no hash/checksum fields, including nested fields."""
    if isinstance(value, dict):
        for key, item in value.items():
            if any(token in key.lower() for token in ("sha256", "hash", "fingerprint", "crc32")):
                raise ValueError(f"Integrity field leaked into readable metadata: {key}")
            require_readable(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_readable(item)


@contextmanager
def exclusive_run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def runtime_identity():
    import av
    import numpy
    import platform
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": numpy.__version__, "av": av.__version__,
            "ffmpeg_libraries": {k: list(v) for k, v in av.library_versions.items()}}


def code_identity():
    directory = Path(__file__).parent
    return {name: sha256(directory / name) for name in
            ("openvidhd.py", "archive.py", "video.py", "manifest.py")}
