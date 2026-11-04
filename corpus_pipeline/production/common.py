from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_production_registry(path):
    registry = json.loads(Path(path).read_text())
    if registry.get("qualification_only") is True or registry.get("published_training") is False:
        raise ValueError("qualification-only endpoint registry cannot run or publish production data")
    return registry


@contextmanager
def atomic_text(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path, value):
    with atomic_text(path) as stream:
        stream.write(canonical(value) + "\n")


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with atomic_text(path) as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
