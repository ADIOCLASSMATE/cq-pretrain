"""Resumable, single-writer Codex image-review farm.

Workers only write their private batch directory. The parent owns all SQLite
transitions, adaptive admission, exports, and strict provenance auditing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import threading
import time
import uuid

MODEL = "gpt-5.6-sol"
EFFORT = "low"
TRANSPORT_ERRORS = {"timeout", "codex_exit", "transport_error", "network_error"}
SCHEMA = "corpus.codex_review_farm.v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "farm.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("farm already has a writer or finalizer") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def raw_results(raw):
    """The worker's schema is an object containing the ordered results array."""
    values = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(values, list):
        raise ValueError("raw response has no items array")
    counts = Counter(value.get("item_id") for value in values if isinstance(value, dict))
    return {value["item_id"]: value for value in values
            if isinstance(value, dict) and isinstance(value.get("item_id"), str)
            and counts[value["item_id"]] == 1}, counts


class Farm:
    """Use only inside ``exclusive(root)``; no worker may access this connection."""

    def __init__(self, root, *, worker=None, clock=time.time):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.worker = worker
        config_path = self.root / "config.json"
        self.config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if self.config.get("model", MODEL) != MODEL or self.config.get("reasoning_effort", EFFORT) != EFFORT:
            raise ValueError("model and reasoning effort are fixed by the acceptance contract")
        self.config.update(model=MODEL, reasoning_effort=EFFORT)
        self.db = sqlite3.connect(self.root / "state.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS items(
            item_id TEXT PRIMARY KEY,source_key TEXT UNIQUE NOT NULL,input_json TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,queue_order INTEGER NOT NULL,status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,retry_not_before REAL NOT NULL DEFAULT 0,
            batch_id TEXT,result_json TEXT,result_sha256 TEXT,error_type TEXT,last_error TEXT,
            created_at REAL NOT NULL,updated_at REAL NOT NULL);
          CREATE INDEX IF NOT EXISTS items_queue ON items(status,retry_not_before,queue_order);
          CREATE INDEX IF NOT EXISTS items_batch ON items(batch_id);
          CREATE TABLE IF NOT EXISTS batches(
            batch_id TEXT PRIMARY KEY,item_ids_json TEXT NOT NULL,status TEXT NOT NULL,
            directory TEXT NOT NULL,started_at REAL NOT NULL,completed_at REAL,
            model TEXT NOT NULL,reasoning_effort TEXT NOT NULL,metadata_json TEXT,
            outcome_sha256 TEXT,error_type TEXT,is_probe INTEGER NOT NULL DEFAULT 0);
        """)

    def close(self):
        self.db.close()

    def metadata(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        self.db.execute("INSERT INTO metadata VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, canonical(value)))

    def load_worker(self):
        if self.worker is None:
            self.worker = importlib.import_module("corpus_pipeline.codex_review.worker")
        return self.worker

    def initialize(self, manifest):
        manifest = Path(manifest).resolve()
        binding = {"path": str(manifest), "sha256": file_sha(manifest)}
        existing = self.metadata("source")
        if existing:
            if binding != {key: existing[key] for key in binding}:
                raise ValueError("source manifest changed; use a new farm root")
            self.check_coverage()
            return self.status()
        if self.db.execute("SELECT count(*) FROM items").fetchone()[0]:
            raise ValueError("unbound populated database")
        now = self.clock()
        total = 0
        with self.db:
            with manifest.open(encoding="utf-8") as stream:
                for total, line in enumerate(stream, 1):
                    item = json.loads(line)
                    if not isinstance(item.get("item_id"), str) or not item["item_id"]:
                        raise ValueError(f"missing stable item ID at line {total}")
                    if "source_ref" not in item:
                        raise ValueError(f"missing source_ref at line {total}")
                    queue_order = item.get("queue_order", total - 1)
                    if type(queue_order) is not int:
                        raise ValueError("queue_order must be an integer")
                    self.db.execute("""INSERT INTO items(item_id,source_key,input_json,input_sha256,
                      queue_order,status,created_at,updated_at) VALUES (?,?,?,?,?,'pending',?,?)""",
                                    (item["item_id"], canonical(item["source_ref"]), canonical(item),
                                     digest(item), queue_order, now, now))
            if not total:
                raise ValueError("empty source manifest")
            self.put("source", {**binding, "count": total})
            self.put("schema", SCHEMA)
        return self.status()

    def pin_runtime(self):
        # A deliberate code/config migration must use a separate, reviewed root.
        worker = self.load_worker()
        files = {str(Path(__file__).resolve()): file_sha(__file__)}
        for name in ("worker.py", "inputs.py"):
            path = Path(__file__).parent / name
            if path.exists():
                files[str(path.resolve())] = file_sha(path)
        binding = {"code": files, "config": self.config, "model": MODEL, "reasoning_effort": EFFORT}
        previous = self.metadata("runtime_binding")
        if previous is not None and previous != binding:
            raise ValueError("runtime code or config changed; explicit migration required")
        with self.db:
            self.put("runtime_binding", binding)
        return worker

    def status(self):
        counts = dict(self.db.execute("SELECT status,count(*) FROM items GROUP BY status"))
        return {"schema_version": SCHEMA, "time": self.clock(), "root": str(self.root),
                "counts": counts, "total": sum(counts.values()), "expected": self.metadata("source", {}).get("count"),
                "attempts": dict(self.db.execute("SELECT attempts,count(*) FROM items GROUP BY attempts")),
                "batches": dict(self.db.execute("SELECT status,count(*) FROM batches GROUP BY status")),
                "adaptive": self.metadata("adaptive", {}), "model": MODEL, "reasoning_effort": EFFORT,
                "complete": bool(self.metadata("complete", {}).get("complete")),
                "run": self.metadata("last_run", {})}

    def recover(self):
        # The process flock proves no old parent remains. Successful stored items
        # are never changed. Each worker receives a new ephemeral batch identity.
        now = self.clock()
        with self.db:
            self.db.execute("""UPDATE items SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'retry' END,
                retry_not_before=?,error_type='interrupted',last_error='previous parent stopped',updated_at=?
                WHERE status='running'""", (self.max_attempts, now, now))
            self.db.execute("UPDATE batches SET status='interrupted',completed_at=?,error_type='interrupted' WHERE status='running'", (now,))
            adaptive = self.metadata("adaptive", {})
            if adaptive.get("circuit") == "half_open":
                adaptive.update(circuit="open", retry_at=max(now, adaptive.get("retry_at", 0)), probe_batch=None)
                self.put("adaptive", adaptive)

    def claim(self, batch_size, *, probe=False):
        # A previously partial/invalid item is retried alone to prevent another
        # long list from starving the same image. Fresh batches remain bounded.
        rows = self.db.execute("""SELECT * FROM items WHERE status IN ('pending','retry')
           AND retry_not_before<=? ORDER BY attempts DESC,queue_order,item_id LIMIT ?""",
                               (self.clock(), batch_size)).fetchall()
        if not rows:
            return None
        if rows[0]["attempts"] > 0:
            rows = rows[:1]
        batch_id = uuid.uuid4().hex
        directory = self.root / "batches" / batch_id
        directory.mkdir(parents=True, exist_ok=False)
        items = [json.loads(row["input_json"]) for row in rows]
        atomic_json(directory / "input.json", items)
        now = self.clock()
        with self.db:
            self.db.execute("""INSERT INTO batches(batch_id,item_ids_json,status,directory,started_at,
               model,reasoning_effort,is_probe) VALUES (?,?,'running',?,?,?,?,?)""",
                            (batch_id, canonical([x["item_id"] for x in items]), str(directory), now, MODEL, EFFORT, int(probe)))
            self.db.executemany("""UPDATE items SET status='running',attempts=attempts+1,batch_id=?,
                updated_at=? WHERE item_id=?""", [(batch_id, now, x["item_id"]) for x in items])
        return batch_id, items, directory

    def _artifact(self, metadata, key, directory):
        record = metadata.get("artifacts", {}).get(key)
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError(f"missing artifact {key}")
        path = Path(record["path"])
        if not path.is_absolute():
            path = directory / path
        path = path.resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha(path) != record["sha256"]:
            raise ValueError(f"artifact outside batch or hash mismatch: {key}")
        return path

    def commit(self, batch_id, outcome):
        worker = self.load_worker()
        batch = self.db.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        if batch["status"] != "running":
            raise ValueError("batch already committed")
        expected = json.loads(batch["item_ids_json"])
        directory = Path(batch["directory"])
        if not isinstance(outcome, dict):
            outcome = {"results": [], "errors": {}, "transport_ok": False,
                       "error_type": "worker_exception", "metadata": {}}
        atomic_json(directory / "outcome.json", outcome)
        metadata = outcome.get("metadata") or {}
        accepted, errors = {}, dict(outcome.get("errors") or {})
        try:
            raw_path = self._artifact(metadata, "last_message", directory)
            raw, raw_counts = raw_results(json.loads(raw_path.read_text()))
        except Exception as error:
            raw, raw_counts = {}, {}
            for item_id in expected:
                errors.setdefault(item_id, f"raw_binding:{type(error).__name__}:{error}")
        values = outcome.get("results") or []
        duplicates = Counter(value.get("item_id") for value in values if isinstance(value, dict))
        for result in values:
            item_id = result.get("item_id") if isinstance(result, dict) else None
            if item_id not in expected:
                continue
            try:
                if duplicates[item_id] != 1 or raw_counts.get(item_id) != 1:
                    raise ValueError("duplicate or missing raw item")
                worker.validate_result(result, item_id)
                if digest(result) != digest(raw[item_id]):
                    raise ValueError("result differs from raw last message")
                accepted[item_id] = result
                errors.pop(item_id, None)
            except Exception as error:
                errors[item_id] = f"validation_error:{type(error).__name__}:{error}"
        now = self.clock()
        transport_ok = bool(outcome.get("transport_ok")) or bool(accepted)
        error_type = outcome.get("error_type") or ("partial_response" if len(accepted) != len(expected) else None)
        with self.db:
            for item_id in expected:
                if item_id in accepted:
                    result = accepted[item_id]
                    self.db.execute("""UPDATE items SET status='succeeded',result_json=?,result_sha256=?,
                       error_type=NULL,last_error=NULL,updated_at=? WHERE item_id=? AND batch_id=?""",
                                    (canonical(result), digest(result), now, item_id, batch_id))
                else:
                    attempts = self.db.execute("SELECT attempts FROM items WHERE item_id=?", (item_id,)).fetchone()[0]
                    delay = min(self.config.get("retry_max_seconds", 300),
                                self.config.get("retry_base_seconds", 5) * 2 ** min(attempts - 1, 10))
                    delay += random.uniform(0, max(0, delay * 0.2))
                    message = str(errors.get(item_id, "missing item result"))[:4000]
                    deterministic_decode = message.startswith("decode_error:") and not metadata.get("alternative_decoder_available", False)
                    self.db.execute("""UPDATE items SET status=?,retry_not_before=?,error_type=?,last_error=?,
                       updated_at=? WHERE item_id=? AND batch_id=?""",
                                    ("failed" if attempts >= self.max_attempts or deterministic_decode else "retry", now + delay,
                                     "decode_error" if deterministic_decode else error_type or "validation_error", message, now, item_id, batch_id))
            self.db.execute("""UPDATE batches SET status=?,completed_at=?,metadata_json=?,outcome_sha256=?,
               error_type=? WHERE batch_id=?""", ("succeeded" if len(accepted) == len(expected) else
                    "partial" if accepted else "failed", now, canonical(metadata), file_sha(directory / "outcome.json"), error_type, batch_id))
            self.adapt(transport_ok, error_type, bool(batch["is_probe"]))
        return len(accepted)

    def adapt(self, healthy, error_type, probe):
        state = self.metadata("adaptive")
        now = self.clock()
        if healthy:
            if probe:
                state.update(circuit="closed", retry_at=0, failures=0, signature=None, probe_batch=None,
                             effective_workers=1, healthy_streak=0, circuit_trips=0)
            elif state["circuit"] == "closed":
                state["healthy_streak"] += 1
                state.update(failures=0, signature=None)
                if state["healthy_streak"] >= self.config.get("increase_every", 4):
                    state["effective_workers"] = min(state["worker_ceiling"], state["effective_workers"] + 1)
                    state["healthy_streak"] = 0
        elif error_type in TRANSPORT_ERRORS:
            state["healthy_streak"] = 0
            state["failures"] = state.get("failures", 0) + 1 if state.get("signature") == error_type else 1
            state["signature"] = error_type
            threshold = self.config.get("circuit_threshold", 3)
            if probe or (state["circuit"] == "closed" and state["failures"] >= threshold):
                trips = state.get("circuit_trips", 0) + 1
                delay = min(self.config.get("circuit_max_seconds", 600),
                            self.config.get("circuit_base_seconds", 30) * 2 ** min(trips - 1, 8))
                state.update(circuit="open", retry_at=now + delay, circuit_trips=trips, probe_batch=None,
                             effective_workers=max(1, state["effective_workers"] // 2))
        elif probe:
            # A local error cannot establish healthy network transport.
            state.update(circuit="open", retry_at=now + self.config.get("circuit_base_seconds", 30), probe_batch=None)
        self.put("adaptive", state)

    def run(self, *, workers=64, initial_workers=4, batch_size=8, max_batches=None, stop_event=None):
        if not (1 <= workers <= 64 and 1 <= initial_workers <= workers and 1 <= batch_size <= 16):
            raise ValueError("require workers<=64, initial_workers<=workers, batch_size<=16")
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be positive")
        self.max_attempts = int(self.config.get("max_attempts", 5))
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        worker = self.pin_runtime()
        if not self.metadata("source"):
            raise ValueError("farm is not initialized")
        self.check_coverage()
        self.recover()
        adaptive = self.metadata("adaptive", {"effective_workers": initial_workers, "healthy_streak": 0,
            "circuit": "closed", "retry_at": 0, "failures": 0, "signature": None, "circuit_trips": 0})
        adaptive.update(worker_ceiling=workers, effective_workers=min(workers, adaptive["effective_workers"]))
        run_info = {"id": uuid.uuid4().hex, "pid": os.getpid(), "started_at": self.clock(), "workers": workers,
                    "initial_workers": initial_workers, "batch_size": batch_size, "max_batches": max_batches}
        with self.db:
            self.put("adaptive", adaptive)
            self.put("last_run", run_info)
            self.put("complete", {"complete": False, "reason": "strict audit and export required"})
        atomic_json(self.root / "runs" / f"{run_info['id']}.json", run_info)
        stopped = stop_event if stop_event is not None else threading.Event()
        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, lambda *_: stopped.set())
        futures, submitted, last_heartbeat = {}, 0, 0.0
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="codex-review") as executor:
                while True:
                    state = self.metadata("adaptive")
                    refill = not stopped.is_set() and (max_batches is None or submitted < max_batches)
                    if refill:
                        limit = state["effective_workers"] if state["circuit"] == "closed" else 0
                        probe = state["circuit"] == "open" and self.clock() >= state["retry_at"] and not futures
                        if probe:
                            limit = 1
                        while not stopped.is_set() and len(futures) < limit and (max_batches is None or submitted < max_batches):
                            claimed = self.claim(batch_size, probe=probe)
                            if not claimed:
                                break
                            batch_id, items, directory = claimed
                            if probe:
                                with self.db:
                                    state.update(circuit="half_open", probe_batch=batch_id)
                                    self.put("adaptive", state)
                            futures[executor.submit(worker.run_batch, items, directory, dict(self.config))] = batch_id
                            submitted += 1
                    if futures:
                        done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                        for future in done:
                            batch_id = futures.pop(future)
                            try:
                                outcome = future.result()
                            except Exception as error:
                                outcome = {"results": [], "errors": {}, "transport_ok": False,
                                           "error_type": "worker_exception", "metadata": {},
                                           "exception": f"{type(error).__name__}:{error}"}
                            self.commit(batch_id, outcome)
                    elif not refill:
                        break
                    else:
                        unfinished = self.db.execute("SELECT count(*) FROM items WHERE status IN ('pending','retry')").fetchone()[0]
                        if not unfinished:
                            break
                        stopped.wait(0.2)
                    if self.clock() - last_heartbeat >= 5:
                        atomic_json(self.root / "HEARTBEAT.json", {**self.status(), "active_batches": len(futures),
                                                                  "submitted_batches": submitted, "draining": stopped.is_set()})
                        last_heartbeat = self.clock()
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            with self.db:
                self.put("last_run", {**run_info, "finished_at": self.clock(), "interrupted": stopped.is_set(),
                                      "submitted_batches": submitted})
            atomic_json(self.root / "HEARTBEAT.json", {**self.status(), "active_batches": 0, "draining": False})
        return self.status()

    def check_coverage(self):
        source = self.metadata("source")
        if not source or file_sha(source["path"]) != source["sha256"]:
            raise ValueError("source manifest missing or changed")
        total = 0
        with Path(source["path"]).open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                row = self.db.execute("SELECT input_sha256,input_json FROM items WHERE item_id=?", (item["item_id"],)).fetchone()
                if row is None or row[0] != digest(item) or digest(json.loads(row[1])) != row[0]:
                    raise ValueError("source/database coverage or input hash mismatch")
                total += 1
        if total != source["count"] or total != self.db.execute("SELECT count(*) FROM items").fetchone()[0]:
            raise ValueError("source coverage count mismatch")

    def audit(self, *, require_complete=True, export_manifest=None):
        worker = self.pin_runtime()
        self.check_coverage()
        successes = self.db.execute("SELECT count(*) FROM items WHERE status='succeeded'").fetchone()[0]
        total = self.metadata("source")["count"]
        if require_complete and successes != total:
            raise ValueError(f"incomplete coverage: {successes}/{total} succeeded")
        seen_threads, audited, validators = set(), 0, {}
        for batch in self.db.execute("""SELECT * FROM batches WHERE batch_id IN
             (SELECT DISTINCT batch_id FROM items WHERE status='succeeded') ORDER BY batch_id"""):
            if batch["status"] not in {"succeeded", "partial"} or batch["model"] != MODEL or batch["reasoning_effort"] != EFFORT:
                raise ValueError("success references an invalid batch")
            directory = Path(batch["directory"])
            metadata = json.loads(batch["metadata_json"])
            if metadata.get("model") != MODEL or metadata.get("reasoning_effort") != EFFORT:
                raise ValueError("batch runtime metadata mismatch")
            if file_sha(directory / "outcome.json") != batch["outcome_sha256"]:
                raise ValueError("batch outcome changed")
            paths = {key: self._artifact(metadata, key, directory) for key in
                     ("command", "prompt", "schema", "last_message", "events", "stderr", "decode_evidence")}
            command = json.loads(paths["command"].read_text())
            if isinstance(command, dict):
                command = command.get("argv", command.get("command"))
            def argument(flag):
                return command[command.index(flag) + 1] if flag in command else None
            if (not isinstance(command, list) or "--ephemeral" not in command or "resume" in command or "fork" in command
                    or argument("--model") != MODEL or 'model_reasoning_effort="low"' not in command
                    or argument("--sandbox") != "read-only" or argument("--cd") != str(directory)
                    or Path(argument("--output-schema") or "").resolve() != paths["schema"]
                    or Path(argument("--output-last-message") or "").resolve() != paths["last_message"]):
                raise ValueError("command provenance mismatch")
            raw = json.loads(paths["last_message"].read_text())
            mapping, counts = raw_results(raw)
            # Partial responses can be valid per item despite failing an array's
            # minItems/maxItems contract. Validate each accepted result against
            # both the worker contract and the exact stored per-item schema.
            schema_hash = metadata["artifacts"]["schema"]["sha256"]
            if schema_hash not in validators:
                import jsonschema
                schema = json.loads(paths["schema"].read_text())
                item_schema = schema.get("properties", {}).get("items", {}).get("items")
                if not isinstance(item_schema, dict):
                    raise ValueError("schema has no result item definition")
                jsonschema.Draft202012Validator.check_schema(item_schema)
                validators[schema_hash] = jsonschema.Draft202012Validator(item_schema)
            validator = validators[schema_hash]
            attachments = metadata.get("attachments")
            decode = json.loads(paths["decode_evidence"].read_text())
            if isinstance(decode, dict):
                decode = decode.get("attachments", decode.get("items"))
            if not isinstance(attachments, list) or not isinstance(decode, list):
                raise ValueError("missing attachment decode evidence")
            decoded = {item["item_id"]: item for item in decode}
            attachment_map = {item["item_id"]: item for item in attachments}
            if len(decoded) != len(decode) or len(attachment_map) != len(attachments):
                raise ValueError("duplicate attachment identity")
            expected = json.loads(batch["item_ids_json"])
            if metadata.get("claimed_item_ids") != expected or metadata.get("attached_item_ids") != [item["item_id"] for item in attachments]:
                raise ValueError("claimed/attached item provenance mismatch")
            if json.loads((directory / "input.json").read_text()) != [json.loads(self.db.execute(
                    "SELECT input_json FROM items WHERE item_id=?", (item_id,)).fetchone()[0]) for item_id in expected]:
                raise ValueError("batch input differs from canonical source")
            if [item["item_id"] for item in attachments] != [item_id for item_id in expected if item_id in attachment_map]:
                raise ValueError("attachment ordering mismatch")
            returned = [item.get("item_id") if isinstance(item, dict) else None for item in raw["items"]]
            attached_ids = [item["item_id"] for item in attachments]
            if returned != [item_id for item_id in attached_ids if item_id in returned]:
                raise ValueError("raw response order, unexpected identity or duplicate mismatch")
            cmd_images = [command[index + 1] for index, value in enumerate(command[:-1]) if value in ("--image", "-i")]
            if [str(Path(image).resolve()) for image in cmd_images] != [str(Path(item["path"]).resolve()) for item in attachments]:
                raise ValueError("CLI attachments differ from decode metadata")
            threads, usages, last_agent_messages = [], [], []
            with gzip.open(paths["events"], "rt", encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get("type") == "thread.started":
                        threads.append(event.get("thread_id"))
                    if event.get("type") == "turn.completed":
                        usages.append(event.get("usage"))
                    item = event.get("item", {})
                    if item.get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"}:
                        raise ValueError("unexpected tool use in review event stream")
                    if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                        last_agent_messages.append(item.get("text"))
            if (len(threads) != 1 or not threads[0] or threads[0] != metadata.get("thread_id")
                    or threads[0] in seen_threads or not usages or usages[-1] != metadata.get("usage")):
                raise ValueError("event thread/usage mismatch or reused thread")
            if not last_agent_messages or json.loads(last_agent_messages[-1]) != raw:
                raise ValueError("event final response differs from last message")
            seen_threads.add(threads[0])
            for row in self.db.execute("SELECT * FROM items WHERE batch_id=? AND status='succeeded'", (batch["batch_id"],)):
                item_id = row["item_id"]
                result = json.loads(row["result_json"])
                worker.validate_result(result, item_id)
                validator.validate(result)
                if (item_id not in expected or counts.get(item_id) != 1 or digest(result) != row["result_sha256"]
                        or digest(mapping[item_id]) != row["result_sha256"]):
                    raise ValueError("raw/result identity or hash mismatch")
                image, evidence = attachment_map.get(item_id), decoded.get(item_id)
                if not image or not evidence or image.get("decoded") is not True or evidence.get("decoded") is not True:
                    raise ValueError("successful result lacks real decode evidence")
                image_path = Path(image["path"]).resolve()
                if (not image_path.is_relative_to(directory) or file_sha(image_path) != image.get("sha256")
                        or evidence.get("sha256") != image.get("sha256")
                        or min(image.get("width", 0), image.get("height", 0)) <= 0):
                    raise ValueError("attachment hash/dimensions mismatch")
                audited += 1
        if audited != successes:
            raise ValueError("success audit coverage mismatch")
        if export_manifest is not None:
            self.audit_export(export_manifest, successes)
        report = {"schema_version": SCHEMA, "mode": "strict", "passed": True,
                  "complete_coverage": successes == total, "expected": total, "succeeded": successes,
                  "audited_items": audited, "audited_batches": len(seen_threads), "time": self.clock(),
                  "source": self.metadata("source"), "runtime_binding_sha256": digest(self.metadata("runtime_binding")),
                  "skipped_gates": [], "export_checked": export_manifest is not None}
        atomic_json(self.root / "AUDIT.json", report)
        return report

    def audit_export(self, manifest_path, expected_successes):
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("source") != self.metadata("source")
                or manifest.get("runtime_binding_sha256") != digest(self.metadata("runtime_binding"))
                or manifest.get("model") != MODEL or manifest.get("reasoning_effort") != EFFORT):
            raise ValueError("export source/runtime binding mismatch")
        seen, count = set(), 0
        for shard in manifest["shards"]:
            path = manifest_path.parent / shard["path"]
            if not path.resolve().is_relative_to(manifest_path.parent) or file_sha(path) != shard["sha256"] or path.stat().st_size != shard["compressed_bytes"]:
                raise ValueError("export shard hash/size mismatch")
            records = 0
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    value = json.loads(line)
                    item_id = value["item_id"]
                    row = self.db.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
                    if (item_id in seen or row is None or row["status"] != "succeeded"
                            or digest(value["result"]) != row["result_sha256"]
                            or value.get("source") != json.loads(row["input_json"])
                            or value.get("batch_id") != row["batch_id"]):
                        raise ValueError("export/database equality or uniqueness mismatch")
                    seen.add(item_id)
                    records += 1
            if records != shard["records"]:
                raise ValueError("export shard count mismatch")
            count += records
        if count != expected_successes or count != manifest["records"]:
            raise ValueError("export total coverage mismatch")

    def export(self, *, shard_size=10000, require_complete=True):
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        content_report = self.audit(require_complete=require_complete)
        directory = self.root / "exports" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        content_audit_path = directory / "CONTENT_AUDIT.json"
        atomic_json(content_audit_path, content_report)
        shards, current, stream, temporary, count = [], 0, None, None, 0
        def finish():
            nonlocal stream
            if stream is None:
                return
            stream.close()
            destination = directory / f"part-{len(shards):05}.jsonl.gz"
            with temporary.open("rb") as raw:
                os.fsync(raw.fileno())
            os.replace(temporary, destination)
            shards.append({"path": destination.name, "split": "review", "records": current,
                           "outputs": current, "compressed_bytes": destination.stat().st_size,
                           "sha256": file_sha(destination)})
            stream = None
        try:
            for row in self.db.execute("SELECT * FROM items WHERE status='succeeded' ORDER BY queue_order,item_id"):
                if stream is None:
                    temporary = directory / f".part-{len(shards):05}.tmp"
                    stream = gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1)
                    current = 0
                # Previously audited canonical JSON is spliced without parsing.
                stream.write('{"item_id":' + canonical(row["item_id"]) + ',"batch_id":' + canonical(row["batch_id"])
                             + ',"source":' + row["input_json"] + ',"result":' + row["result_json"] + '}\n')
                current += 1
                count += 1
                if current == shard_size:
                    finish()
            finish()
        finally:
            if stream is not None:
                stream.close()
        manifest = {"schema_version": SCHEMA, "records": count, "outputs": count, "shards": shards,
                    "source": self.metadata("source"), "model": MODEL, "reasoning_effort": EFFORT,
                    "runtime_binding_sha256": digest(self.metadata("runtime_binding")), "time": self.clock(),
                    "complete": count == self.metadata("source")["count"], "audit_mode": "strict"}
        path = directory / "manifest.json"
        atomic_json(path, manifest)
        # The caller still owns the process flock and this single-writer
        # connection. No inference or mutation occurs between strict content
        # audit and export; only the newly published shards need a second scan.
        self.audit_export(path, count)
        report = {**content_report, "time": self.clock(), "export_checked": True,
                  "content_audit_path": str(content_audit_path), "content_audit_sha256": file_sha(content_audit_path),
                  "export_manifest": str(path), "export_manifest_sha256": file_sha(path)}
        audit_path = directory / "AUDIT.json"
        atomic_json(audit_path, report)
        atomic_json(self.root / "AUDIT.json", report)
        completion = {"complete": manifest["complete"], "audit_mode": "strict", "skipped_gates": [],
                      "manifest": str(path), "manifest_sha256": file_sha(path),
                      "audit": str(audit_path), "audit_sha256": file_sha(audit_path),
                      "records": count, "time": self.clock()}
        with self.db:
            self.put("complete", completion)
        atomic_json(self.root / "EXPORT.json", completion)
        if completion["complete"]:
            atomic_json(self.root / "COMPLETE.json", completion)
        return {**completion, "audit_report": report}


def readonly_status(root):
    root = Path(root).resolve()
    db = sqlite3.connect(f"file:{root / 'state.sqlite'}?mode=ro", uri=True)
    try:
        db.execute("BEGIN")
        metadata = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata")}
        counts = dict(db.execute("SELECT status,count(*) FROM items GROUP BY status"))
        return {"root": str(root), "counts": counts, "total": sum(counts.values()),
                "expected": metadata.get("source", {}).get("count"), "adaptive": metadata.get("adaptive", {}),
                "run": metadata.get("last_run", {}), "completion": metadata.get("complete", {})}
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("init", "run", "status", "audit", "export"):
        command = sub.add_parser(name)
        command.add_argument("--root", required=True)
        if name == "init":
            command.add_argument("--manifest", required=True)
        elif name == "run":
            command.add_argument("--workers", type=int, default=64)
            command.add_argument("--initial-workers", type=int, default=4)
            command.add_argument("--batch-size", type=int, default=8)
            command.add_argument("--max-batches", type=int)
        elif name in ("audit", "export"):
            command.add_argument("--allow-incomplete", action="store_true")
            if name == "export":
                command.add_argument("--shard-size", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.action == "status":
        print(canonical(readonly_status(args.root)))
        return
    with exclusive(args.root):
        farm = Farm(args.root)
        try:
            if args.action == "init":
                result = farm.initialize(args.manifest)
            elif args.action == "run":
                result = farm.run(workers=args.workers, initial_workers=args.initial_workers,
                                  batch_size=args.batch_size, max_batches=args.max_batches)
            elif args.action == "audit":
                export = farm.metadata("complete", {}).get("manifest")
                result = farm.audit(require_complete=not args.allow_incomplete, export_manifest=export)
            else:
                result = farm.export(shard_size=args.shard_size, require_complete=not args.allow_incomplete)
            print(canonical(result))
        finally:
            farm.close()


if __name__ == "__main__":
    main()
