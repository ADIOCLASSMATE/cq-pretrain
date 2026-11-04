"""Task-neutral index I/O and atomic, lossless latent-reference updates."""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

MANAGED_FIELDS = {"qwen_cache", "wan_cache", "latent_info", "latent_profile"}


def read_manifest(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    elif path.suffix == ".json":
        value = json.loads(path.read_text())
        rows = value if isinstance(value, list) else [value]
    else:
        raise ValueError("Dataset paths must identify a JSONL or JSON manifest")
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def original_digest(row):
    value = {k: v for k, v in row.items() if k not in MANAGED_FIELDS}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def resolve_video(row, manifest):
    if "videos" in row:
        if len(row["videos"]) != 1:
            raise ValueError("Stage one supports exactly one video per sample")
        video = deepcopy(row["videos"][0])
    else:
        video = {
            k: row[k]
            for k in (
                "frames_path",
                "fps",
                "fps_num",
                "fps_den",
                "num_frames",
                "height",
                "width",
                "qwen_cache",
                "wan_cache",
            )
            if row.get(k) is not None
        }
        video["path"] = row.get("video_path")
    video.setdefault("id", "v0")
    for key in ("path", "frames_path", "qwen_cache", "wan_cache"):
        if video.get(key):
            video[key] = str((Path(manifest).resolve().parent / video[key]).resolve())
    return video


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        Path(tmp).unlink(missing_ok=True)


def merge_latent_updates(path, updates, originals):
    """Merge worker updates under a lock; preserve order and every original field."""
    path = Path(path)
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        rows = read_manifest(path)
        ids = [str(row["sample_id"]) for row in rows]
        if len(ids) != len(set(ids)) or ids != list(originals):
            raise ValueError("Manifest sample membership changed during extraction")
        changed = False
        for row, sample_id in zip(rows, ids):
            if original_digest(row) != originals[sample_id]:
                raise ValueError(f"Original sample fields changed: {sample_id}")
            if sample_id in updates:
                update = updates[sample_id]
                before = deepcopy(row)
                if set(update) - MANAGED_FIELDS:
                    raise ValueError("Only latent fields may be updated")
                for key, value in update.items():
                    if key == "latent_info":
                        row[key] = {**row.get(key, {}), **value}
                    else:
                        row[key] = value
                changed |= row != before
        if changed:
            atomic_write(
                path,
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            )


def source_profiles(source, rows=None):
    path = Path(source["path"]).expanduser().resolve()
    rows = read_manifest(path) if rows is None else rows
    profiles = {}
    for row in rows:
        ref = row.get("latent_profile")
        if ref:
            absolute = str((path.parent / ref).resolve())
            if absolute not in profiles:
                value = json.loads(Path(absolute).read_text())
                if value.get("format") != "umm_latent_profile_v2":
                    raise ValueError(f"Invalid latent profile: {absolute}")
                profiles[absolute] = value
    return profiles
