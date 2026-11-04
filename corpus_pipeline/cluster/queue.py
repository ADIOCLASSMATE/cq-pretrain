"""Small shard-task queue on a shared filesystem, without SQLite/WAL.

The persistent ``.queue.lock`` serializes metadata transitions with flock. Each
lease also owns an atomically created claim directory; its random token fences
preempted workers. Task manifests contain parquet paths and exclusive row ranges,
not materialized corpus records. Workers write only beneath
``artifacts/<task_id>/<token>/`` and publish a checksummed completion receipt.

All mutations run on the control host: remote workers use queue_rpc. The
filesystem's flock is only assumed to coordinate processes on that same host.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid


SCHEMA = "corpus_shard_queue.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class QueueError(RuntimeError):
    pass


class LeaseLost(QueueError):
    """The caller no longer owns the current unexpired lease."""


class ArtifactError(QueueError):
    """An output is missing, changed, outside this attempt, or has a bad hash."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _clock(now):
    result = time.time() if now is None else float(now)
    if not math.isfinite(result):
        raise ValueError("time must be finite")
    return result


def _timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _clock(value)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must include a timezone")
    return parsed.timestamp()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _task_id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("task_id must be a safe filename of 1..128 characters")
    return value


def _root(path):
    return Path(path).expanduser().resolve()


@contextmanager
def _locked(root, *, create=False):
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise QueueError(f"queue root does not exist: {root}")
    # Never unlink this file: replacing its inode would split the lock domain.
    with (root / ".queue.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _config(root):
    path = root / "queue.json"
    if not path.is_file():
        raise QueueError("queue has not completed initialization")
    result = _read(path)
    if result.get("schema") != SCHEMA or result.get("initialized") is not True:
        raise QueueError("unsupported or incomplete queue metadata")
    return result


def _state_path(root, task_id):
    return root / "states" / (_task_id(task_id) + ".json")


def _manifest(root, config, ordinal):
    item = _read(root / config["catalog"] / f"{ordinal:012d}.json")
    if item.get("ordinal") != ordinal or item.get("task_sha256") != _digest(item.get("task")):
        raise QueueError("task manifest identity or checksum changed")
    _task_id(item.get("task_id"))
    return item


@lru_cache(maxsize=8)
def _catalog_index(root, catalog, manifest_sha256, task_count):
    """Read-only index, built from and checked against the frozen catalog.

    Kept in memory only; it does not change existing queue schemas or cursors.
    Cache each original task digest too, so later edits cannot substitute a new
    task with a recomputed per-file checksum.
    """
    index, hasher = {}, hashlib.sha256()
    for ordinal in range(task_count):
        item = _manifest(root, {'catalog': catalog}, ordinal)
        if item['task_id'] in index:
            raise QueueError('duplicate task ID in frozen catalog')
        index[item['task_id']] = (ordinal, item['task_sha256'])
        hasher.update((_canonical(item) + '\n').encode())
    if hasher.hexdigest() != manifest_sha256:
        raise QueueError('frozen catalog digest changed')
    return index


def catalog_index(root):
    """Return a copy of the verified ID -> (ordinal, task digest) mapping."""
    root = _root(root)
    config = _config(root)
    return dict(_catalog_index(root, config['catalog'], config['manifest_sha256'],
                               config['task_count']))


def _task(task, ordinal):
    if not isinstance(task, dict) or not task:
        raise ValueError("each task must be a nonempty small manifest object")
    # JSON round trip rejects nonserializable and NaN values without mutating input.
    task = json.loads(_canonical(task))
    if "records" in task or "segments" in task:
        raise ValueError("queue tasks must reference shards, not contain corpus rows")
    has_start, has_end = "row_start" in task, "row_end" in task
    if has_start != has_end:
        raise ValueError("row_start and row_end must be supplied together")
    if has_start:
        start = _integer(task["row_start"], "row_start")
        end = _integer(task["row_end"], "row_end", 1)
        if end <= start:
            raise ValueError("row_end must be greater than row_start (exclusive)")
        if not any(isinstance(task.get(key), str) and task[key].strip()
                   for key in ("parquet_path", "path", "shard_path")):
            raise ValueError("a row range requires a parquet_path/path/shard_path")
    task_id = _task_id(task.get("task_id", f"{ordinal:012d}-{_digest(task)[:20]}"))
    return {"task_id": task_id, "ordinal": ordinal, "task": task,
            "task_sha256": _digest(task)}


def init_queue(root, tasks, *, max_attempts=3, lease_seconds=900,
               deadline_at=None, now=None):
    """Freeze an iterable of small tasks; an identical re-init is idempotent.

    A partially written catalog is never made visible: queue.json is the final
    atomic initialization commit. Input rows are never read or spooled here.
    """
    root, now = _root(root), _clock(now)
    _integer(max_attempts, "max_attempts", 1)
    lease_seconds = float(lease_seconds)
    if not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive and finite")
    deadline_at = _timestamp(deadline_at)
    with _locked(root, create=True):
        existing = _config(root) if (root / "queue.json").exists() else None
        catalog = root / "catalogs" / uuid.uuid4().hex
        if not existing:
            catalog.mkdir(parents=True)
        seen, count, hasher = set(), 0, hashlib.sha256()
        try:
            for ordinal, value in enumerate(tasks):
                item = _task(value, ordinal)
                if item["task_id"] in seen:
                    raise ValueError(f"duplicate task_id: {item['task_id']}")
                seen.add(item["task_id"])
                hasher.update((_canonical(item) + "\n").encode())
                if not existing:
                    _atomic(catalog / f"{ordinal:012d}.json", item)
                count += 1
            if existing:
                expected = (count, hasher.hexdigest(), max_attempts, lease_seconds)
                actual = (existing["task_count"], existing["manifest_sha256"],
                          existing["max_attempts"], existing["lease_seconds"])
                if actual != expected or (deadline_at is not None and
                                          deadline_at != existing.get("deadline_at")):
                    raise QueueError("existing queue identity differs; use a new root")
                return existing
            config = {"schema": SCHEMA, "initialized": True,
                      "created_at": now, "deadline_at": deadline_at,
                      "lease_seconds": lease_seconds, "max_attempts": max_attempts,
                      "task_count": count, "manifest_sha256": hasher.hexdigest(),
                      "catalog": str(catalog.relative_to(root)),
                      "row_range_semantics": "row_end_exclusive",
                      "artifact_directory": "artifacts/<task_id>/<token>/"}
            for directory in ("states", "claims", "retry", "artifacts"):
                (root / directory).mkdir(exist_ok=True)
            _atomic(root / "cursor.json", {"next_ordinal": 0})
            _atomic(root / "queue.json", config)
            return config
        except BaseException:
            if not existing and catalog.exists() and not (root / "queue.json").exists():
                shutil.rmtree(catalog)
            raise


def _retry_marker(root, state):
    _atomic(root / "retry" / (state["task_id"] + ".json"),
            {"task_id": state["task_id"], "ordinal": state["ordinal"]})


def _remove_claim(root, task_id):
    directory = root / "claims" / task_id
    if directory.exists():
        # Only queue metadata is stored here; actual outputs live under artifacts.
        shutil.rmtree(directory)


def _recover_expired(root, config, now):
    reclaimed = []
    for directory in list((root / "claims").iterdir()):
        if not directory.is_dir():
            raise QueueError(f"unexpected entry in claim directory: {directory}")
        path = _state_path(root, directory.name)
        if not path.exists():
            # mkdir succeeded but the worker died before writing its first state.
            _remove_claim(root, directory.name)
            continue
        state = _read(path)
        if state["status"] == "running" and state["lease_expires_at"] <= now:
            failure = {"reason": "lease_expired", "at": now, "token": state["token"],
                       "attempt": state["attempt"]}
            state["failures"].append(failure)
            state.update(status="pending" if state["attempt"] < config["max_attempts"]
                         else "failed", updated_at=now, last_failure=failure)
            _atomic(path, state)
            reclaimed.append(state["task_id"])
        if state["status"] != "running":
            if state["status"] == "pending":
                # Marker before removing claim makes a crash in this transition recoverable.
                _retry_marker(root, state)
            _remove_claim(root, state["task_id"])
    return reclaimed


def _next_task(root, config):
    for path in (root / "retry").iterdir():
        item = _read(path)
        state_path = _state_path(root, item["task_id"])
        state = _read(state_path) if state_path.exists() else None
        if state and state["status"] == "pending":
            return _manifest(root, config, item["ordinal"]), state, None
        path.unlink()
    cursor = _read(root / "cursor.json")["next_ordinal"]
    while cursor < config["task_count"]:
        item = _manifest(root, config, cursor)
        state_path = _state_path(root, item["task_id"])
        state = _read(state_path) if state_path.exists() else None
        if state is None or state["status"] == "pending":
            return item, state, cursor + 1
        cursor += 1
        _atomic(root / "cursor.json", {"next_ordinal": cursor})
    return None, None, None


def claim(root, worker_id, *, now=None):
    root, now = _root(root), _clock(now)
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ValueError("worker_id must be a nonempty string")
    with _locked(root):
        config = _config(root)
        _recover_expired(root, config, now)
        item, previous, cursor = _next_task(root, config)
        if item is None:
            return None
        return _claim_item(root, config, item, previous, worker_id, now, cursor)


def _claim_item(root, config, item, previous, worker_id, now, cursor=None):
    """Caller holds the control-host queue lock."""
    task_id, token = item['task_id'], uuid.uuid4().hex
    directory = root / 'claims' / task_id
    directory.mkdir()
    state = {**item, 'status': 'running', 'worker_id': worker_id, 'token': token,
             'attempt': (previous['attempt'] if previous else 0) + 1,
             'claimed_at': now, 'heartbeat_at': now, 'updated_at': now,
             'lease_expires_at': now + config['lease_seconds'],
             'failures': previous.get('failures', []) if previous else []}
    _atomic(_state_path(root, task_id), state)
    _atomic(directory / 'lease.json', state)
    if cursor is not None:
        _atomic(root / 'cursor.json', {'next_ordinal': cursor})
    (root / 'retry' / (task_id + '.json')).unlink(missing_ok=True)
    return {**state, 'artifact_dir': str(root / 'artifacts' / task_id / token)}


def claim_task(root, worker_id, task_id, *, now=None):
    """Claim one frozen task by ID without advancing the sequential cursor.

    Running, completed and terminal failed tasks return None. Expired leases
    recover under the same bounded-retry and fencing rules as ordinary claim.
    No worker-supplied path, range or task body is accepted.
    """
    root, now = _root(root), _clock(now)
    _task_id(task_id)
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ValueError('worker_id must be a nonempty string')
    config = _config(root)
    index = _catalog_index(root, config['catalog'], config['manifest_sha256'], config['task_count'])
    if task_id not in index:
        raise ValueError('task ID is absent from frozen catalog')
    ordinal, expected_digest = index[task_id]
    with _locked(root):
        if _config(root) != config:
            raise QueueError('queue identity changed during targeted claim')
        _recover_expired(root, config, now)
        item = _manifest(root, config, ordinal)
        if item['task_id'] != task_id or item['task_sha256'] != expected_digest:
            raise QueueError('targeted task identity changed')
        path = _state_path(root, task_id)
        previous = _read(path) if path.exists() else None
        if previous and previous['status'] != 'pending':
            return None
        return _claim_item(root, config, item, previous, worker_id, now)


def _lease(root, task_id, token, now, *, allow_complete=False):
    path = _state_path(root, task_id)
    if not path.is_file():
        raise LeaseLost("task has no current lease")
    state = _read(path)
    if not isinstance(token, str) or state.get("token") != token:
        raise LeaseLost("lease token was superseded")
    if allow_complete and state["status"] == "completed":
        return state
    if state["status"] != "running" or state["lease_expires_at"] <= now:
        raise LeaseLost("lease is not running or has expired")
    return state


def heartbeat(root, task_id, token, *, now=None):
    root, now = _root(root), _clock(now)
    with _locked(root):
        config = _config(root)
        state = _lease(root, task_id, token, now)
        state.update(heartbeat_at=now, updated_at=now,
                     lease_expires_at=now + config["lease_seconds"])
        _atomic(_state_path(root, task_id), state)
        directory = root / "claims" / task_id
        directory.mkdir(exist_ok=True)
        _atomic(directory / "lease.json", state)
        return {"task_id": task_id, "token": token,
                "lease_expires_at": state["lease_expires_at"]}


def fail(root, task_id, token, error, *, retryable=True, now=None):
    root, now = _root(root), _clock(now)
    if not isinstance(error, str) or not error.strip():
        raise ValueError("error must be a nonempty string")
    with _locked(root):
        config = _config(root)
        state = _lease(root, task_id, token, now)
        failure = {"reason": error, "at": now, "token": token,
                   "attempt": state["attempt"]}
        state["failures"].append(failure)
        state.update(status="pending" if retryable and state["attempt"] <
                     config["max_attempts"] else "failed", updated_at=now,
                     last_failure=failure)
        _atomic(_state_path(root, task_id), state)
        if state["status"] == "pending":
            _retry_marker(root, state)
        _remove_claim(root, task_id)
        return state


def _file_signature(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _artifacts(root, task_id, token, artifacts):
    if not isinstance(artifacts, (list, tuple)) or not artifacts:
        raise ArtifactError("completion requires at least one output artifact")
    namespace = (root / "artifacts" / task_id / token).resolve()
    if not namespace.is_relative_to(root):
        raise ArtifactError("artifact namespace escapes queue root")
    checked, signatures, seen = [], {}, set()
    for value in artifacts:
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ArtifactError("artifact must contain path and sha256")
        sha = value.get("sha256")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ArtifactError("artifact sha256 must contain 64 lowercase hex digits")
        original = Path(value["path"])
        path = (original if original.is_absolute() else root / original).resolve()
        if not path.is_relative_to(namespace) or path == namespace:
            raise ArtifactError("output must belong to artifacts/<task_id>/<token>/")
        if path in seen or not path.is_file():
            raise ArtifactError("duplicate, missing or non-file artifact")
        seen.add(path)
        before = _file_signature(path)
        hasher = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 << 20), b""):
                    hasher.update(chunk)
            after = _file_signature(path)
        except OSError as exc:
            raise ArtifactError(f"cannot read output: {path}") from exc
        if before != after or hasher.hexdigest() != sha:
            raise ArtifactError(f"output changed or sha256 mismatch: {path}")
        if "size_bytes" in value and value["size_bytes"] != after[2]:
            raise ArtifactError(f"output byte count mismatch: {path}")
        signatures[path] = after
        checked.append({"path": str(path.relative_to(root)), "sha256": sha,
                        "size_bytes": after[2]})
    checked.sort(key=lambda value: value["path"])
    return checked, signatures


def complete(root, task_id, token, *, artifacts, processed_records,
             processed_images, metadata=None, now=None):
    """Validate actual outputs then atomically commit the current lease.

    Hashing is outside the metadata lock so large outputs do not stall heartbeats.
    Lease ownership and file identity are checked again immediately before commit.
    Workers must keep their heartbeat thread alive while completion hashes run.
    """
    root = _root(root)
    _integer(processed_records, "processed_records")
    _integer(processed_images, "processed_images")
    metadata = json.loads(_canonical(metadata or {}))
    with _locked(root):
        _config(root)
        state = _lease(root, task_id, token, _clock(now), allow_complete=True)
        task = state["task"]
        if "row_start" in task and processed_records != task["row_end"] - task["row_start"]:
            raise QueueError("processed_records must equal the entire exclusive task row range")
    checked, signatures = _artifacts(root, task_id, token, artifacts)
    intent = {"task_id": task_id, "token": token, "task_sha256": state["task_sha256"],
              "artifacts": checked, "processed_records": processed_records,
              "processed_images": processed_images, "metadata": metadata}
    intent_sha = _digest(intent)
    with _locked(root):
        finished_at = _clock(now)
        state = _lease(root, task_id, token, finished_at, allow_complete=True)
        for path, signature in signatures.items():
            if not path.is_file() or _file_signature(path) != signature:
                raise ArtifactError(f"output changed before commit: {path}")
        if state["status"] == "completed":
            if state["receipt"]["intent_sha256"] != intent_sha:
                raise QueueError("completed receipt differs; completion is immutable")
            return state["receipt"]
        receipt = {"schema": SCHEMA, "status": "completed", **intent,
                   "worker_id": state["worker_id"], "attempt": state["attempt"],
                   "completed_at": finished_at, "intent_sha256": intent_sha}
        state.update(status="completed", updated_at=finished_at, receipt=receipt)
        _atomic(_state_path(root, task_id), state)  # Sole authoritative completion commit.
        _remove_claim(root, task_id)
        return receipt


def status(root, *, now=None):
    root, now = _root(root), _clock(now)
    with _locked(root):
        config = _config(root)
        recovered = _recover_expired(root, config, now)
        counts = {"pending": config["task_count"], "running": 0, "failed": 0,
                  "completed": 0}
        processed_records = processed_images = failed_attempts = attempts = 0
        for path in (root / "states").glob("*.json"):
            state = _read(path)
            kind = state["status"]
            if kind not in counts:
                raise QueueError(f"invalid task status in {path}")
            if kind != "pending":
                counts["pending"] -= 1
                counts[kind] += 1
            failed_attempts += len(state["failures"])
            attempts += state["attempt"]
            if kind == "completed":
                processed_records += state["receipt"]["processed_records"]
                processed_images += state["receipt"]["processed_images"]
        deadline = config.get("deadline_at")
        remaining = None if deadline is None else deadline - now
        return {"schema": SCHEMA, "root": str(root), "as_of": now,
                "task_count": config["task_count"], "counts": counts,
                "processed_records": processed_records, "processed_images": processed_images,
                "attempts": attempts, "failed_attempts": failed_attempts,
                "recovered_expired_tasks": recovered,
                "created_at": config["created_at"],
                "elapsed_seconds": max(0, now - config["created_at"]),
                "deadline_at": deadline, "seconds_to_deadline": remaining,
                "deadline_exceeded": remaining is not None and remaining < 0,
                "all_tasks_completed": counts["completed"] == config["task_count"],
                "all_tasks_terminal": not counts["pending"] and not counts["running"],
                "artifact_integrity_scope": "sha256_verified_at_completion_not_rehashed_by_status"}


def _tasks_from_file(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        data = _read(path)
        tasks = data["tasks"] if isinstance(data, dict) else data
        if not isinstance(tasks, list):
            raise ValueError("manifest must be a JSON task array/object or JSONL")
        yield from tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--root", required=True)
    initialize.add_argument("--manifest", required=True)
    initialize.add_argument("--max-attempts", type=int, default=3)
    initialize.add_argument("--lease-seconds", type=float, default=900)
    deadlines = initialize.add_mutually_exclusive_group()
    deadlines.add_argument("--deadline-at", help="timezone-qualified ISO timestamp")
    deadlines.add_argument("--deadline-hours", type=float)
    acquire = sub.add_parser("claim")
    acquire.add_argument("--root", required=True)
    acquire.add_argument("--worker-id", required=True)
    for name in ("heartbeat", "complete", "fail"):
        item = sub.add_parser(name)
        item.add_argument("--root", required=True)
        item.add_argument("--task-id", required=True)
        item.add_argument("--token", required=True)
        if name == "complete":
            item.add_argument("--artifacts", required=True, help="JSON file containing path/sha256 list")
            item.add_argument("--processed-records", type=int, required=True)
            item.add_argument("--processed-images", type=int, required=True)
            item.add_argument("--metadata", help="optional JSON metadata file")
        if name == "fail":
            item.add_argument("--error", required=True)
            item.add_argument("--no-retry", action="store_true")
    inspect = sub.add_parser("status")
    inspect.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            deadline = args.deadline_at
            if args.deadline_hours is not None:
                if not math.isfinite(args.deadline_hours) or args.deadline_hours <= 0:
                    raise ValueError("deadline-hours must be positive and finite")
                existing_path = _root(args.root) / "queue.json"
                started = _read(existing_path)["created_at"] if existing_path.is_file() else time.time()
                deadline = started + args.deadline_hours * 3600
            result = init_queue(args.root, _tasks_from_file(args.manifest),
                                max_attempts=args.max_attempts, lease_seconds=args.lease_seconds,
                                deadline_at=deadline)
        elif args.command == "claim":
            result = claim(args.root, args.worker_id)
        elif args.command == "heartbeat":
            result = heartbeat(args.root, args.task_id, args.token)
        elif args.command == "complete":
            result = complete(args.root, args.task_id, args.token,
                              artifacts=_read(args.artifacts),
                              processed_records=args.processed_records,
                              processed_images=args.processed_images,
                              metadata=_read(args.metadata) if args.metadata else None)
        elif args.command == "fail":
            result = fail(args.root, args.task_id, args.token, args.error,
                          retryable=not args.no_retry)
        else:
            result = status(args.root)
        print(_canonical(result))
        return 0
    except (QueueError, ValueError, OSError) as exc:
        print(_canonical({"status": "error", "error_type": type(exc).__name__,
                          "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
