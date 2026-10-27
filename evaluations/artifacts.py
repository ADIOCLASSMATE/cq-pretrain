"""Atomic sample receipts and integrity checks shared by local/DP evaluation."""

import hashlib
import json
from pathlib import Path

from utils.checkpoint_catalog import durable_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_identity(path):
    path = Path(path)
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while being read: {path}")
    return {"size": after.st_size, "sha256": sha.hexdigest()}


def artifact_path(directory, relative):
    root = Path(directory).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Artifact escapes task directory: {relative}")
    return path


def receipt_path(directory, request):
    return Path(directory) / "receipts" / f"{digest(request['id'])}.json"


def read_receipt(directory, request):
    """Only a committed receipt with all intact files counts as a completion."""
    try:
        row = json.loads(receipt_path(directory, request).read_text())
        if row["request_sha256"] != digest(request) or not row["artifacts"]:
            return None
        for key, relative in row["artifacts"].items():
            if file_identity(artifact_path(directory, relative)) != row["integrity"][key]:
                return None
        return row
    except (OSError, ValueError, KeyError, TypeError):
        return None


def commit_receipt(directory, request, result, *, rank, attempt_id):
    row = dict(result, request_sha256=digest(request), rank=rank, attempt_id=attempt_id)
    row["integrity"] = {key: file_identity(artifact_path(directory, path))
                        for key, path in row["artifacts"].items()}
    receipt = receipt_path(directory, request)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    durable_json(receipt, row)
    return row


def consolidate(directory, requests):
    rows = [row for request in requests if (row := read_receipt(directory, request)) is not None]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(directory / "manifest.jsonl")
    return rows
