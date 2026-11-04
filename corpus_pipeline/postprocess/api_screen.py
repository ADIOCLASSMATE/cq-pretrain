"""Bounded, resumable blind Qwen review of globally reconciled AI suspects.

This is a screening sidecar, not provenance certification or a training release.
Serving defaults to a primary screening pass; any tentative rejection or
uncertainty awaits an independent secondary review. API defaults to two-model
agreement. Source facts are never sent to models and remain immutable in SQLite. ``annotate`` records the visual recommendation but always leaves the
detector alert pending; it never grants admission or rejection.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sqlite3
import time
from typing import Iterable

from corpus_pipeline.api_review.aggregate import validate_parsed
from corpus_pipeline.production.api_client import QwenAPIClient
from corpus_pipeline.production.client import DynamicClient, ModelStageError, endpoint_lease_valid
from corpus_pipeline.production.native_transport import is_native, native_receipt_errors, TRANSPORT_CONTRACT_FIELDS
from corpus_pipeline.production.common import atomic_json, canonical, digest, file_sha, require_production_registry
from corpus_pipeline.production.image_screening import (
    POLICY_VERSION, REVIEW_STAGE, EVIDENCE_SCOPE, adjudicate_review,
)
from .worker_control import AdmissionControl, MeasuredExecutor, WorkerTelemetry, cpu_limits, timing_summary

VERSION = "postprocess.api_screen.v1"
PROMPT_PATH = Path(__file__).parents[1] / "production" / "image_screening_prompt_v3.txt"
PRIMARY = "qwen3.8-max"
SECONDARY = "qwen3.5-397b-a17b"
ELIGIBLE = "ai_suspect_pending_api"
INPUT_COLUMNS = ("asset_id", "pixel_sha256", "pixel_sha256_conflict", "global_status",
                 "explicit_negative", "valid_protected", "has_invalid_observation",
                 "original_ref_json", "provenance_json")


def validate_response(value):
    errors = validate_parsed(value)
    if errors:
        raise ValueError("invalid_screening_response:" + ",".join(errors))


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def prepare_asset(row):
    """Fail closed before a paid request, including integrity/protection conflicts."""
    if row.get("global_status") != ELIGIBLE:
        return None
    if any(row.get(name) for name in ("pixel_sha256_conflict", "explicit_negative", "valid_protected",
                                      "has_invalid_observation")):
        raise ValueError("suspect_has_unresolved_global_evidence")
    aid = row.get("asset_id")
    if not _sha(aid):
        raise ValueError("invalid_asset_id")
    ref = json.loads(row["original_ref_json"])
    provenance = json.loads(row["provenance_json"])
    if not isinstance(ref, dict) or ref.get("sha256") != aid:
        raise ValueError("asset_and_original_reference_hash_mismatch")
    if not isinstance(provenance, dict):
        raise ValueError("invalid_source_provenance")
    if provenance.get("origin") in {"ai_generated", "ai_edited"}:
        raise ValueError("source_origin_requires_global_rules_not_visual_override")
    if (not isinstance(ref.get("path"), str)
            or type(ref.get("offset")) is not int or ref["offset"] < 0
            or type(ref.get("length")) is not int or not 0 < ref["length"] <= 128 << 20):
        raise ValueError("invalid_original_reference")
    if ref.get("kind", "tar") == "file":
        # Keep the dataset-relative reference intact. The shared resolver checks
        # its pinned root, aliases, containment and complete-file size; the
        # image reader verifies the original SHA256 before sending any bytes.
        from ..production.portable_refs import resolve_ref
        try:
            resolve_ref(ref)
        except OSError as error:
            raise ValueError("invalid_original_reference") from error
    elif ref.get("kind", "tar") != "tar" or Path(ref["path"]).suffix != ".tar":
        raise ValueError("invalid_original_reference")
    return {"asset_id": aid, "sha256": aid, "original_ref": ref}, provenance


def review_context(asset_id):
    # No caption, source name, detector score or prior review can bias either vote.
    return {"asset_id": asset_id, "source_certification": False, "screening_policy": POLICY_VERSION}


def receipt_errors(receipt, asset_id, model, prompt, context, max_tokens, endpoint_identity=None, backend="api"):
    errors = []
    if not isinstance(receipt, dict):
        return ["missing_receipt"]
    identity = receipt.get("identity", {})
    expected = {"stage": REVIEW_STAGE, "role": "verifier", "model": model,
                "assets": [asset_id], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if backend == "serving" else digest(prompt),
                "context_sha256": digest(context), "max_tokens": max_tokens}
    if any(identity.get(k) != v for k, v in expected.items()):
        errors.append("receipt_request_identity_mismatch")
    if endpoint_identity is not None and identity.get("endpoint_identity") != endpoint_identity:
        errors.append("receipt_endpoint_identity_mismatch")
    if receipt.get("response_model") != model or receipt.get("request_context") != context:
        errors.append("receipt_response_model_or_context_mismatch")
    if backend == "serving":
        if is_native(identity):
            errors.extend(native_receipt_errors(receipt))
        else:
            boots = receipt.get("qualified_boots_at_request")
            if not isinstance(boots, list) or receipt.get("response_boot_id") not in boots:
                errors.append("serving_boot_not_qualified")
        if not _sha(receipt.get("endpoint_identity_sha256")):
            errors.append("serving_endpoint_identity_missing")
    check = adjudicate_review(receipt, {"origin": "unknown"}, asset_id=asset_id)
    errors.extend(check.get("validation_errors", []))
    return errors


def combine_reviews(reviews, provenance, *, mode="both"):
    base = {"policy_version": POLICY_VERSION, "evidence_scope": EVIDENCE_SCOPE,
            "source_certification": False, "real_parent_lineage_established": False,
            "origin": provenance.get("origin", "unknown"), "decision": "pending"}
    primary = reviews.get("primary")
    secondary = reviews.get("secondary")
    if mode == "annotate":
        base["detector_alarm_cleared"] = False
        base["model_recommendation"] = primary.get("decision") if primary else None
        if primary:
            return {**base, "reason": "visual_annotation_only_requires_confirmation"}
    if mode in {"cascade", "primary"} and primary and primary.get("decision") == "admit" and secondary is None:
        return {**base, "decision": "admit", "reason": "primary_visual_screen_passed"}
    if mode == "primary" and primary:
        return {**base, "reason": "independent_secondary_review_required"}
    if not primary or not secondary:
        return {**base, "reason": "api_review_failure_pending_retry"}
    if primary["decision"] == secondary["decision"] and primary["decision"] in {"admit", "reject"}:
        return {**base, "decision": primary["decision"], "reason": "two_model_screening_agreement"}
    reason = "reviewer_disagreement" if primary["decision"] != secondary["decision"] else "reviewers_inconclusive"
    return {**base, "reason": reason}


class ShardedQwenClients:
    backend = "api"
    """Share transports/semaphores while placing receipt files in hash shards."""
    def __init__(self, credential_example, root, models, concurrency, timeout):
        self.root = Path(root)
        self.bases = {role: QwenAPIClient(credential_example, self.root / role,
                        generator=model, verifier=model, concurrency=concurrency,
                        timeout=timeout, retries=0) for role, model in models.items()}
        self.shards = {}
        self.endpoint_identity = {role: c.endpoint_identity for role, c in self.bases.items()}

    def client(self, role, asset_id):
        key = (role, asset_id[:2])
        if key not in self.shards:
            client = copy(self.bases[role])
            client.cache = self.root / role / asset_id[:2]
            client.cache.mkdir(parents=True, exist_ok=True)
            client.flights = {}
            self.shards[key] = client
        return self.shards[key]

    async def request(self, role, asset, prompt, context, max_tokens):
        return await self.client(role, asset["asset_id"]).request(
            "verifier", REVIEW_STAGE, context, [asset], prompt, validate_response,
            max_tokens=max_tokens)

    def stats(self):
        return {role: dict(client.stats) for role, client in self.bases.items()}

    async def close(self):
        await asyncio.gather(*(c.close() for c in self.bases.values()))


class ServingUnavailable(RuntimeError):
    """Capacity/readiness interruption; never a model assessment or retry charge."""


class ShardedServingClients:
    """Live, weighted pool of qualified single-endpoint DynamicClient registries.

    Pool files contain {schema_version: corpus.screening_pool.v1,
    registries: [{path: absolute_registry_path, weight: 1, max_inflight: 512}]}.
    max_inflight is optional and limits each replica independently of pool size. Only ready,
    image-capable replicas with the same model contract receive work. Replica
    additions and readiness refreshes do not change the immutable review policy.
    """
    backend = "serving"

    def __init__(self, registries, root, concurrency=128, timeout=120, wait_ready=60, pools=None,
                 admission_control=None, refresh_interval=.5):
        self.root, self.bases, self.shards = Path(root), {}, {}
        self.registries = {role: [value] if isinstance(value, (str, Path)) else list(value)
                           for role, value in registries.items()}
        self.pools = pools or {}
        self.concurrency, self.timeout, self.wait_ready = concurrency, timeout, wait_ready
        self.models, self.contracts, self.endpoint_identity = {}, {}, {}
        self.counters, self.inflight = Counter(), Counter()
        self.cooldown_until = {}
        self.admission = AdmissionControl(concurrency, admission_control)
        self.refresh_interval = refresh_interval
        self.ready_cache, self.refreshed_at = {}, {}
        self.last_entries = {}
        self.pool_members = {}
        self.refresh_locks = {}
        self.control_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="screen-control")
        self.http_body_ms, self.http_header_ms = deque(maxlen=4096), deque(maxlen=4096)
        self.condition = asyncio.Condition()
        self.stop = asyncio.Event()
        for role in set(self.registries) | set(self.pools):
            ready = self.ready_entries(role)
            if not ready:
                raise ValueError("no_qualified_ready_serving_replica:" + role)
            self.endpoint_identity[role] = None

    def entries(self, role):
        values = [{"path": str(path), "weight": 1.0} for path in self.registries.get(role, [])]
        if role in self.pools:
            pool_path = Path(self.pools[role]).resolve()
            data = json.loads(pool_path.read_text())
            if data.get("schema_version") != "corpus.screening_pool.v1" or not isinstance(data.get("registries"), list):
                raise ValueError("invalid_serving_pool_schema")
            values.extend(data["registries"])
        result, seen = [], set()
        for value in values:
            path, weight = Path(value["path"]), value.get("weight", 1)
            if not path.is_absolute() or isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError("invalid_serving_pool_entry")
            path = str(path.resolve())
            if path in seen:
                raise ValueError("duplicate_serving_pool_registry")
            seen.add(path)
            entry = {"path": path, "weight": float(weight), "key": (role, path)}
            if "max_inflight" in value:
                cap = value["max_inflight"]
                if type(cap) is not int or cap <= 0:
                    raise ValueError("invalid_serving_pool_max_inflight")
                entry["max_inflight"] = cap
            result.append(entry)
        return result

    def read_ready_entries(self, role):
        """Only file reads; periodic refresh runs outside the asyncio event loop."""
        ready = []
        self.admission.refresh()
        entries = self.entries(role)
        for entry in entries:
            try:
                registry = require_production_registry(entry["path"])
                endpoints = [value for value in registry["endpoints"] if value["role"] == "verifier"
                             and value.get("ready") and endpoint_lease_valid(value)]
                if len(endpoints) != 1:
                    continue
                endpoint = endpoints[0]
                if ("qwen" not in endpoint["model"].lower() or not endpoint.get("supports_images")
                        or (not is_native(endpoint) and not endpoint.get("expected_boots"))
                        or endpoint.get("backend") != "inspire_serving"):
                    continue
                ready.append({**entry, "endpoint": endpoint})
            except (OSError, ValueError, LookupError, KeyError, TypeError):
                continue
        return ready, {entry["key"] for entry in entries}

    def accept_ready_entries(self, role, scanned):
        candidates, members = scanned
        ready = []
        for entry in candidates:
            key, path, endpoint = entry["key"], entry["path"], entry["endpoint"]
            try:
                contract = {name: endpoint.get(name) for name in
                    ("model", "revision", "precision", "processor_revision", "runtime_revision", "request_overrides", *TRANSPORT_CONTRACT_FIELDS)}
                if role in self.contracts and self.contracts[role] != contract:
                    continue
                if key not in self.bases:
                    self.bases[key] = DynamicClient(path, self.root / role, concurrency=self.concurrency,
                                                   timeout=self.timeout, wait_ready=self.wait_ready)
                    transport = getattr(self.bases[key], "client", None)
                    if transport is not None:
                        transport.event_hooks.setdefault("request", []).append(self.count_http_request)
                        transport.event_hooks.setdefault("response", []).append(self.count_http_response)
                if role not in self.contracts:
                    self.contracts[role] = contract
                    self.models[role] = endpoint["model"]
                self.last_entries[key] = entry
                ready.append(entry)
            except (OSError, ValueError, LookupError, KeyError, TypeError):
                continue
        self.ready_cache[role], self.refreshed_at[role] = ready, time.monotonic()
        self.pool_members[role] = members
        self.counters["readiness_refreshes"] += 1
        return ready

    def ready_entries(self, role):
        # Explicit inspection and initial bootstrap are synchronous; request
        # admission uses one shared periodic asynchronous refresh below.
        return self.accept_ready_entries(role, self.read_ready_entries(role))

    async def cached_ready_entries(self, role):
        if time.monotonic() - self.refreshed_at.get(role, 0) >= self.refresh_interval:
            lock = self.refresh_locks.setdefault(role, asyncio.Lock())
            async with lock:
                if time.monotonic() - self.refreshed_at.get(role, 0) >= self.refresh_interval:
                    try:
                        entries = await asyncio.get_running_loop().run_in_executor(self.control_executor, self.read_ready_entries, role)
                        self.accept_ready_entries(role, entries)
                    except (OSError, ValueError, KeyError, TypeError):
                        # Pool mutation/mount outage pauses admission. It must
                        # not drain retries across every queued image.
                        self.ready_cache[role], self.refreshed_at[role] = [], time.monotonic()
                        self.counters["readiness_refresh_failures"] += 1
        return [entry for entry in self.ready_cache.get(role, [])
                if self.cooldown_until.get(entry["key"], 0) <= time.monotonic()
                and endpoint_lease_valid(entry["endpoint"])]

    async def count_http_request(self, request):
        # Count attempts without retaining URL, credentials, headers or image bytes.
        self.counters["http_attempts"] += 1
        request.extensions["screen_started_monotonic"] = time.monotonic()
        self.counters["http_request_bytes"] += int(request.headers.get("content-length", 0))

    async def count_http_response(self, response):
        started = response.request.extensions.get("screen_started_monotonic", time.monotonic())
        headers_at = response.extensions.get('native_headers_received_at', time.monotonic())
        self.http_header_ms.append((headers_at - started) * 1000)
        try:
            # Keepalive gateways may emit headers/whitespace early. A response
            # hook must read the complete body before recording completion.
            await response.aread()
        except Exception:
            self.counters["http_body_failures"] += 1
            raise
        body_at = response.extensions.get('native_body_received_at', time.monotonic())
        self.http_body_ms.append((body_at - started) * 1000)
        self.counters["http_bodies_completed"] += 1
        self.counters["http_status:" + str(response.status_code)] += 1

    def client(self, role, aid, key):
        shard_key = key, aid[:2]
        if shard_key not in self.shards:
            client = copy(self.bases[key])
            client.cache = self.root / role / aid[:2]
            client.cache.mkdir(parents=True, exist_ok=True)
            client._flights = {}
            self.shards[shard_key] = client
        return self.shards[shard_key]

    async def acquire(self, role):
        deadline = time.monotonic() + self.wait_ready
        async with self.condition:
            while True:
                if self.stop.is_set():
                    raise ServingUnavailable("serving_pool_wait_interrupted")
                ready = await self.cached_ready_entries(role)
                limit = self.admission.limit
                total_weight = sum(entry["weight"] for entry in ready)
                candidates = [entry for entry in ready if self.inflight[entry["key"]]
                    < min(max(1, math.ceil(limit * entry["weight"] / total_weight)),
                          entry.get("max_inflight", self.concurrency))]
                if sum(self.inflight.values()) >= limit:
                    candidates = []
                if candidates:
                    selected = min(candidates, key=lambda entry: (self.inflight[entry["key"]] / entry["weight"], entry["path"]))
                    self.inflight[selected["key"]] += 1
                    return selected["key"]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Ready leases can disappear during preemption. Keep bounded
                    # workers parked, not burning retries across the corpus.
                    self.counters["readiness_wait_windows"] += 1
                    deadline = time.monotonic() + self.wait_ready
                    remaining = self.wait_ready
                try:
                    await asyncio.wait_for(self.condition.wait(), min(.5, remaining))
                except asyncio.TimeoutError:
                    pass

    async def request(self, role, asset, prompt, context, max_tokens):
        key = await self.acquire(role)
        try:
            # Refresh *only the selected registry* before spending work. The
            # shared 0.5s snapshot is scheduling guidance, not an authority to
            # use a withdrawn qualification or changed model contract.
            try:
                require_production_registry(key[1])
                endpoint, _ = self.bases[key].endpoint("verifier")
                if any(endpoint.get(name) != value for name, value in self.contracts[role].items()):
                    raise ValueError("serving_model_contract_changed")
            except (OSError, ValueError, LookupError, KeyError, TypeError):
                raise ServingUnavailable("selected_serving_registry_changed") from None
            client = self.client(role, asset["asset_id"], key)
            self.counters["request_invocations"] += 1
            self.counters["replica:" + digest(key)[:16]] += 1
            receipt = await client.request("verifier", REVIEW_STAGE, context, [asset], prompt,
                                           validate_response, max_tokens=max_tokens)
            if any(receipt["identity"].get(name) != value for name, value in self.contracts[role].items()):
                raise ValueError("serving_receipt_model_contract_mismatch")
            self.counters["cache_hits" if receipt.get("cache_hit") else "successful_responses"] += 1
            for name, value in (receipt.get("usage") or {}).items():
                if name in {"prompt_tokens", "completion_tokens", "total_tokens"} and type(value) is int:
                    self.counters[name] += value
            return receipt
        except ModelStageError as error:
            if error.retryable_after_endpoint_change:
                self.cooldown_until[key] = time.monotonic() + 30
                self.counters["replica_transient_cooldowns"] += 1
            raise
        finally:
            async with self.condition:
                self.inflight[key] -= 1
                self.condition.notify(1)

    def stats(self):
        current = {entry["key"] for entries in self.ready_cache.values() for entry in entries}
        configured = set().union(*self.pool_members.values()) if self.pool_members else set()
        return {**dict(self.counters), "active_requests": sum(self.inflight.values()),
                "known_replicas": len(self.bases), "admission": self.admission.stats(),
                "http_complete_body": timing_summary(self.http_body_ms), "http_headers": timing_summary(self.http_header_ms),
                "replicas": [{"registry_sha256": digest(entry["key"]), "registry_path": entry["path"],
                              "in_pool": entry["key"] in configured, "inflight": self.inflight[entry["key"]],
                              "max_inflight": entry.get("max_inflight", self.concurrency),
                              "ready": entry["key"] in current and endpoint_lease_valid(entry["endpoint"])
                              and self.cooldown_until.get(entry["key"], 0) <= time.monotonic()}
                             for entry in self.last_entries.values()]}

    async def close(self):
        await asyncio.gather(*(client.close() for client in self.bases.values()))
        self.control_executor.shutdown(wait=True)


@contextmanager
def output_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    owner = output / "OWNER.json"
    host = socket.gethostname()
    try:
        fd = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if json.loads(owner.read_text()).get("host") != host:
            raise RuntimeError("screening_output_is_owned_by_another_host")
    else:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical({"host": host}))
    with (output / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("screening_output_has_active_writer") from None
        yield


class Ledger:
    def __init__(self, output, identity):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = self.output / "INPUT.json"
        dbpath = self.output / "index.sqlite"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("immutable_screening_input_or_policy_changed")
            if not dbpath.exists():
                raise ValueError("screening_ledger_missing")
        elif dbpath.exists():
            raise ValueError("screening_manifest_missing")
        self.db = sqlite3.connect(dbpath)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY, input_sha TEXT NOT NULL, input_json TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending', decision TEXT, reason TEXT, decision_json TEXT, attempts INTEGER NOT NULL DEFAULT 0);
          CREATE INDEX IF NOT EXISTS assets_summary ON assets(state,decision,reason);
          CREATE TABLE IF NOT EXISTS requests (
            asset_id TEXT NOT NULL, role TEXT NOT NULL, request_sha TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending',
            receipt_path TEXT, receipt_sha TEXT, last_error TEXT,
            PRIMARY KEY(asset_id,role));
          CREATE TABLE IF NOT EXISTS metrics (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
        """)
        self.db.commit()
        if not manifest.exists():
            atomic_json(manifest, identity)
        self.identity = identity
        self.summary_counts = None

    def count_state(self, state, decision, reason, delta):
        if self.summary_counts is None:
            return
        self.summary_counts["eligible_assets"] += delta
        self.summary_counts["state:" + state] += delta
        if decision:
            self.summary_counts["decision:" + decision] += delta
            self.summary_counts["reason:" + reason] += delta

    def bump(self, name, count=1):
        self.db.execute("INSERT INTO metrics VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (name, count))

    def insert(self, row):
        aid, encoded = row["asset_id"], canonical(row)
        existing = self.db.execute("SELECT input_sha FROM assets WHERE asset_id=?", (aid,)).fetchone()
        checksum = digest(row)
        if existing:
            if existing[0] != checksum:
                raise ValueError("duplicate_asset_has_conflicting_immutable_input")
            return False
        self.db.execute("INSERT INTO assets(asset_id,input_sha,input_json) VALUES(?,?,?)", (aid, checksum, encoded))
        self.count_state("pending", None, None, 1)
        return True

    def finish(self, aid, decision, *, retryable=False):
        state = "retryable" if retryable else "done"
        old = self.db.execute("SELECT state,decision,reason FROM assets WHERE asset_id=?", (aid,)).fetchone()
        if not old:
            raise ValueError("cannot_finish_missing_asset")
        self.db.execute("UPDATE assets SET state=?,decision=?,reason=?,decision_json=?,attempts=attempts+1 WHERE asset_id=?",
                        (state, decision["decision"], decision["reason"], canonical(decision), aid))
        self.db.commit()
        self.count_state(*old, -1)
        self.count_state(state, decision["decision"], decision["reason"], 1)

    def summary(self):
        if self.summary_counts is None:
            self.summary_counts = Counter()
            for state, decision, reason, count in self.db.execute("SELECT state,decision,reason,count(*) FROM assets GROUP BY state,decision,reason"):
                self.count_state(state, decision, reason, count)
        return {"counts": {name: value for name, value in self.summary_counts.items() if value},
                "requests": dict(self.db.execute("SELECT name,value FROM metrics"))}

    def close(self):
        self.db.close()


class Screener:
    def __init__(self, ledger, clients, *, models=None, mode="both", concurrency=128,
                 max_attempts=3, max_tokens=1600, prompt=None, stop=None):
        if mode not in {"both", "cascade", "primary", "annotate"} or not 1 <= concurrency <= 2048 or not 1 <= max_attempts <= 10:
            raise ValueError("invalid_screening_limits_or_mode")
        self.ledger, self.clients = ledger, clients
        self.models = models or {"primary": PRIMARY, "secondary": SECONDARY}
        if ("primary" not in self.models or any("qwen" not in v.lower() for v in self.models.values())
                or mode not in {"primary", "annotate"} and (len(self.models) != 2 or len(set(self.models.values())) != 2)):
            raise ValueError("independent_Qwen_model_configuration_required")
        self.mode, self.concurrency, self.max_attempts, self.max_tokens = mode, concurrency, max_attempts, max_tokens
        self.prompt = prompt if prompt is not None else PROMPT_PATH.read_text()
        self.stop = stop or asyncio.Event()
        if self.clients.backend == "serving":
            self.clients.stop = self.stop
            self.ledger.db.execute("UPDATE requests SET state='deferred', attempts=max(0,attempts-1), "
                                   "last_error='previous_process_interrupted' WHERE state='inflight'")
            self.ledger.db.commit()

    def request_identity(self, role, asset):
        return {"asset_id": asset["asset_id"], "model": self.models[role], "role": role,
                "prompt_sha256": digest(self.prompt), "context": review_context(asset["asset_id"]),
                "max_tokens": self.max_tokens, "policy": POLICY_VERSION,
                "endpoint_identity": self.clients.endpoint_identity.get(role),
                "backend": self.clients.backend, "model_contract": getattr(self.clients, "contracts", {}).get(role)}

    async def review(self, role, asset, provenance):
        aid = asset["asset_id"]
        identity = self.request_identity(role, asset)
        db = self.ledger.db
        request_sha = digest(identity)
        old = db.execute("SELECT request_sha,attempts,state,receipt_path,receipt_sha FROM requests WHERE asset_id=? AND role=?", (aid, role)).fetchone()
        if old and old[0] != request_sha:
            raise ValueError("request_identity_changed")
        if old and old[2] == "ok":
            path = self.ledger.output / old[3]
            if not path.exists() or file_sha(path) != old[4]:
                raise ValueError("saved_review_receipt_hash_mismatch")
            receipt = json.loads(path.read_text())
            self.validate_receipt(role, asset, receipt)
            self.ledger.bump("durable_receipt_hits")
            db.commit()
            return self.adjudicate_receipt(receipt, provenance, aid)
        attempts = old[1] if old else 0
        while attempts < self.max_attempts and not self.stop.is_set():
            attempts += 1
            # Persist before calling: an interrupted request consumes one attempt.
            db.execute("INSERT INTO requests(asset_id,role,request_sha,attempts,state) VALUES(?,?,?,?,'inflight') "
                       "ON CONFLICT(asset_id,role) DO UPDATE SET attempts=excluded.attempts,state='inflight'",
                       (aid, role, request_sha, attempts))
            self.ledger.bump("request_invocations")
            db.commit()
            try:
                receipt = await self.clients.request(role, asset, self.prompt, review_context(aid), self.max_tokens)
                self.validate_receipt(role, asset, receipt)
                path = self.ledger.output / "receipts" / aid[:2] / aid[2:4] / (aid + "." + role + ".json")
                atomic_json(path, receipt)
                db.execute("UPDATE requests SET state='ok',receipt_path=?,receipt_sha=?,last_error=NULL WHERE asset_id=? AND role=?",
                           (str(path.relative_to(self.ledger.output)), file_sha(path), aid, role))
                self.ledger.bump("cached_responses" if receipt.get("cache_hit") else "actual_responses")
                db.commit()
                return self.adjudicate_receipt(receipt, provenance, aid)
            except (Exception,) as error:
                transient = (self.clients.backend == "serving" and
                             (isinstance(error, ServingUnavailable) or
                              isinstance(error, ModelStageError) and error.retryable_after_endpoint_change))
                if transient:
                    db.execute("UPDATE requests SET state='deferred',attempts=max(0,attempts-1),last_error=? "
                               "WHERE asset_id=? AND role=?", (type(error).__name__, aid, role))
                    self.ledger.bump("serving_transient_deferrals")
                    db.commit()
                    return None
                # Do not persist transport exception strings (may contain endpoint secrets).
                db.execute("UPDATE requests SET state='failed',last_error=? WHERE asset_id=? AND role=?", (type(error).__name__, aid, role))
                self.ledger.bump("failed_invocations")
                db.commit()
                if attempts < self.max_attempts and not self.stop.is_set():
                    try:
                        await asyncio.wait_for(self.stop.wait(), timeout=min(30, 2 ** (attempts - 1)))
                    except asyncio.TimeoutError:
                        pass
        return None

    def adjudicate_receipt(self, receipt, provenance, asset_id):
        result = adjudicate_review(receipt, provenance, asset_id=asset_id)
        if self.mode == "annotate":
            # Retain the complete validated description/evidence alongside the
            # immutable full receipt, not just the model's tentative vote.
            result["visual_annotation"] = receipt["parsed"]
        return result

    def finish(self, asset_id, result, *, retryable=False):
        if self.mode == "annotate":
            if result["decision"] != "pending":
                raise ValueError("annotation_mode_cannot_finalize_screening")
            result["detector_alarm_cleared"] = False
            result["source_certification"] = False
            result.setdefault("model_recommendation", None)
        self.ledger.finish(asset_id, result, retryable=retryable)

    def validate_receipt(self, role, asset, receipt):
        errors = receipt_errors(receipt, asset["asset_id"], self.models[role], self.prompt,
                                review_context(asset["asset_id"]), self.max_tokens,
                                self.clients.endpoint_identity.get(role), self.clients.backend)
        if errors:
            raise ValueError("invalid_bound_review:" + ",".join(errors))

    async def process(self, row):
        try:
            prepared = prepare_asset(row)
            if prepared is None:
                raise ValueError("noneligible_asset_entered_review_queue")
            asset, provenance = prepared
        except (ValueError, TypeError, KeyError):
            self.finish(row["asset_id"], {"decision": "pending", "reason": "input_integrity_or_global_evidence_conflict",
                "source_certification": False, "evidence_scope": EVIDENCE_SCOPE})
            return
        reviews = {}
        try:
            primary = await self.review("primary", asset, provenance)
            if primary:
                reviews["primary"] = primary
            if self.mode not in {"primary", "annotate"} and (self.mode == "both" or not primary or primary["decision"] != "admit"):
                secondary = await self.review("secondary", asset, provenance)
                if secondary:
                    reviews["secondary"] = secondary
            result = combine_reviews(reviews, provenance, mode=self.mode)
            result["reviews"] = reviews
            result["input_sha256"] = digest(row)
            result["provenance_sha256"] = digest(provenance)
            retryable = result["reason"] == "api_review_failure_pending_retry"
            if retryable:
                failed = self.ledger.db.execute("SELECT attempts FROM requests WHERE asset_id=? AND state!='ok'", (asset["asset_id"],)).fetchall()
                retryable = not self.stop.is_set() and any(count < self.max_attempts for (count,) in failed)
                result["reason"] = "api_review_interrupted" if self.stop.is_set() else "api_review_failure_exhausted" if not retryable else result["reason"]
                if self.stop.is_set():
                    retryable = True
                elif self.clients.backend == "serving" and self.ledger.db.execute(
                        "SELECT 1 FROM requests WHERE asset_id=? AND state='deferred' LIMIT 1", (asset["asset_id"],)).fetchone():
                    result["reason"] = "serving_temporarily_unavailable"
                    retryable = True
            self.finish(asset["asset_id"], result, retryable=retryable)
        except (ValueError, KeyError, OSError):
            self.finish(asset["asset_id"], {"decision": "pending", "reason": "saved_review_integrity_failure",
                "source_certification": False, "evidence_scope": EVIDENCE_SCOPE})

    async def run(self):
        queue = asyncio.Queue(maxsize=self.concurrency * 2)
        async def worker():
            while True:
                row = await queue.get()
                try:
                    if row is None:
                        return
                    if not self.stop.is_set():
                        await self.process(row)
                finally:
                    queue.task_done()
        async def produce():
            last = 0
            while not self.stop.is_set():
                batch = self.ledger.db.execute("SELECT rowid,input_json FROM assets WHERE rowid>? AND state!='done' ORDER BY rowid LIMIT 512", (last,)).fetchall()
                if not batch:
                    break
                for rowid, encoded in batch:
                    if self.stop.is_set():
                        break
                    await queue.put(json.loads(encoded))
                    last = rowid
            for _ in range(self.concurrency):
                await queue.put(None)
        async with asyncio.TaskGroup() as group:
            group.create_task(produce())
            for _ in range(self.concurrency):
                group.create_task(worker())
        return self.ledger.summary()


def iter_input(paths: Iterable[Path]):
    import pyarrow.parquet as pq
    for path in paths:
        parquet = pq.ParquetFile(path)
        required = {"asset_id", "global_status", "original_ref_json", "provenance_json"}
        if not required <= set(parquet.schema_arrow.names):
            raise ValueError("reconciled_input_columns_missing")
        columns = [name for name in INPUT_COLUMNS if name in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                if row.get("global_status") == ELIGIBLE:
                    yield row


def verify_input_scope(paths, reconciliation_root=None, canary=None, selection_manifest=None):
    """Verify parent COMPLETE -> 256 bucket receipts -> selected api_review files.

    No exists-only or bounded-validation completion grants a full-run scope.
    """
    inputs = [{"path": str(Path(path).resolve()), "sha256": file_sha(path)} for path in paths]
    if selection_manifest is not None:
        import pyarrow.parquet as pq
        selected_path = Path(selection_manifest).resolve(strict=True)
        selection = json.loads(selected_path.read_text())
        if selection.get('schema') != 'corpus.api_review.selection.v1' or selection.get('scope') not in {'validation', 'full'}:
            raise ValueError('invalid_incremental_API_selection_scope')
        def verify_binding(item):
            path = Path(item['path']).resolve(strict=True)
            if path.stat().st_size != item['size_bytes'] or file_sha(path) != item['sha256']:
                raise ValueError('incremental_API_selection_binding_changed')
            return path
        parent_complete = verify_binding(selection['reconciliation_complete'])
        if reconciliation_root is not None and Path(reconciliation_root).resolve() != parent_complete.parent:
            raise ValueError('incremental_API_selection_parent_mismatch')
        parents = [verify_binding(item) for item in selection['source_files']]
        # Recursively enforce the existing COMPLETE -> bucket receipt -> original
        # API input chain; a selected subset cannot authorize an unbound carrier.
        parent_scope = verify_input_scope(parents, parent_complete.parent)
        selected_files = [verify_binding(item) for item in selection['selected_files']]
        if (len(set(selected_files)) != len(selected_files) or len(inputs) != len(selected_files)
                or {item['path'] for item in inputs} != {str(path) for path in selected_files}):
            raise ValueError('incremental_API_input_files_differ_from_selection')
        selected_rows = {}; schema = None
        for path in selected_files:
            reader = pq.ParquetFile(path)
            if schema is None: schema = reader.schema_arrow
            if not reader.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError('incremental_API_selection_schema_mismatch')
            for batch in reader.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    aid = row.get('asset_id')
                    if not _sha(aid) or aid in selected_rows:
                        raise ValueError('incremental_API_selection_duplicate_or_invalid_asset')
                    selected_rows[aid] = digest(row)
        if not selected_rows or len(selected_rows) != selection['selected_assets']:
            raise ValueError('incremental_API_selection_coverage_mismatch')
        pending = verify_binding(selection['pending_snapshot']); pending_ids = set()
        for batch in pq.ParquetFile(pending).iter_batches(columns=['asset_id', 'needs_api']):
            for row in batch.to_pylist():
                if row['needs_api'] is True:
                    if row['asset_id'] in pending_ids: raise ValueError('duplicate_incremental_API_pending_asset')
                    pending_ids.add(row['asset_id'])
        if pending_ids != set(selected_rows):
            raise ValueError('incremental_API_selection_not_equal_pending_snapshot')
        found = set()
        for path in parents:
            reader = pq.ParquetFile(path)
            if not reader.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError('incremental_API_selection_changed_original_schema')
            for batch in reader.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    aid = row['asset_id']
                    if aid not in selected_rows: continue
                    if aid in found or digest(row) != selected_rows[aid]:
                        raise ValueError('incremental_API_selection_changed_original_row')
                    found.add(aid)
        if found != set(selected_rows): raise ValueError('incremental_API_selection_contains_unreconciled_assets')
        if canary is not None and not 1 <= canary <= 10000:
            raise ValueError('canary_must_be_bounded_to_1_10000_assets')
        validation = selection['scope'] == 'validation' or canary is not None
        return {'scope': 'bounded_incremental_validation' if validation else 'selected_globally_reconciled_assets',
                'global_screening_scope_authorized': not validation,
                'reconciliation_complete_sha256': parent_scope['reconciliation_complete_sha256'],
                'selection_manifest': {'path': str(selected_path), 'sha256': file_sha(selected_path)},
                'selected_assets': len(selected_rows), 'canary_limit': canary, 'inputs': inputs}
    if canary is not None:
        if not 1 <= canary <= 10000:
            raise ValueError("canary_must_be_bounded_to_1_10000_assets")
        return {"scope": "bounded_canary_not_global", "canary_limit": canary,
                "global_screening_scope_authorized": False, "inputs": inputs}
    if reconciliation_root is None:
        raise ValueError("full_screening_requires_completed_global_reconciliation")
    root = Path(reconciliation_root).resolve()
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if (complete.get("schema") != "corpus.asset_reconciliation.v1"
            or complete.get("complete") is not True or complete.get("scope") != "full_run"
            or not _sha(complete.get("identity_sha256")) or not _sha(complete.get("input_index_sha256"))):
        raise ValueError("global_reconciliation_incomplete")
    buckets = complete.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != 256 or {b.get("bucket") for b in buckets} != {f"{index:02x}" for index in range(256)}:
        raise ValueError("global_reconciliation_bucket_receipts_missing")
    hashes = {}
    for bucket in buckets:
        receipt_path = (root / bucket["receipt_path"]).resolve()
        if not receipt_path.is_relative_to(root) or file_sha(receipt_path) != bucket["receipt_sha256"]:
            raise ValueError("reconciliation_bucket_receipt_hash_mismatch")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "corpus.asset_reconciliation.bucket.v1"
                or receipt.get("identity_sha256") != complete["identity_sha256"]
                or receipt.get("bucket") != bucket["bucket"]):
            raise ValueError("reconciliation_bucket_identity_mismatch")
        for artifact in receipt.get("artifacts", []):
            target = (root / artifact["path"]).resolve()
            if not target.is_relative_to(root):
                raise ValueError("reconciliation_artifact_outside_root")
            if target.name == "api_review.parquet":
                hashes[str(target)] = artifact["sha256"]
    for item in inputs:
        if hashes.get(item["path"]) != item["sha256"]:
            raise ValueError("api_review_assets_not_bound_by_global_complete")
    return {"scope": "selected_globally_reconciled_buckets", "global_screening_scope_authorized": True,
            "reconciliation_complete_sha256": file_sha(complete_path), "inputs": inputs}


def export_decisions(ledger):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = ledger.output / "decisions.parquet"
    tmp = path.with_suffix(".parquet.partial")
    schema = pa.schema([("asset_id", pa.string()), ("input_sha256", pa.string()),
                        ("decision", pa.string()), ("reason", pa.string()),
                        ("state", pa.string()), ("decision_json", pa.string())])
    cursor = ledger.db.execute("SELECT asset_id,input_sha,state,decision_json FROM assets ORDER BY asset_id")
    try:
        with pq.ParquetWriter(tmp, schema) as writer:
            while batch := cursor.fetchmany(1024):
                values = []
                for aid, checksum, state, encoded in batch:
                    value = json.loads(encoded) if encoded else {"decision": "pending", "reason": "not_yet_reviewed"}
                    values.append({"asset_id": aid, "input_sha256": checksum, "decision": value["decision"],
                                   "reason": value["reason"], "state": state, "decision_json": encoded})
                writer.write_table(pa.Table.from_pylist(values, schema=schema))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": path.name, "sha256": file_sha(path)}


async def execute(args):
    from .review_health import load_credential, preflight, ProgressHealth
    if args.backend == 'serving':
        load_credential(getattr(args, 'api_key_file', None))
    paths = [Path(path).resolve() for path in args.input]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate_input_path")
    scope = verify_input_scope(paths, args.reconciliation_root, args.canary,
                               selection_manifest=getattr(args, 'selection_manifest', None))
    models = {"primary": args.primary, "secondary": args.secondary}
    if args.mode is None:
        args.mode = "primary" if args.backend == "serving" else "both"
    prompt_path = Path(args.prompt).resolve() if args.prompt else PROMPT_PATH
    prompt = prompt_path.read_text()
    output = Path(args.output).resolve()
    stop = asyncio.Event()
    stop_path = Path(args.stop_file).resolve() if getattr(args, 'stop_file', None) else None
    if stop_path is not None and stop_path.exists():
        stop.set()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    started = time.time()
    worker_instance = digest({"host": socket.gethostname(), "pid": os.getpid(), "started_at": started})
    image_threads = getattr(args, "image_threads", None) or cpu_limits()["suggested_image_threads"]
    if not 1 <= image_threads <= cpu_limits()["suggested_image_threads"]:
        raise ValueError("image_threads_exceed_current_CPU_quota_or_16_thread_bound")
    executor = MeasuredExecutor(image_threads)
    loop.set_default_executor(executor)
    telemetry = WorkerTelemetry(executor)
    with output_lock(output):
        if args.backend == "serving":
            if (not args.primary_registry and not args.primary_pool or args.mode not in {"primary", "annotate"}
                    and not args.secondary_registry and not args.secondary_pool):
                raise ValueError("qualified_primary_and_required_secondary_registries_needed")
            registries = {"primary": args.primary_registry or []}
            pools = {"primary": args.primary_pool} if args.primary_pool else {}
            if args.mode not in {"primary", "annotate"}:
                registries["secondary"] = args.secondary_registry or []
                if args.secondary_pool:
                    pools["secondary"] = args.secondary_pool
            clients = ShardedServingClients(registries, output / "cache", args.concurrency, args.timeout, args.wait_ready, pools=pools,
                                            admission_control=getattr(args, "admission_control", None))
            models = clients.models
        else:
            if not args.credential_example:
                raise ValueError("API_backend_requires_explicit_credential_example")
            if args.mode in {"primary", "annotate"}:
                models = {"primary": args.primary}
            clients = ShardedQwenClients(args.credential_example, output / "cache", models, args.concurrency, args.timeout)
        identity = {"version": VERSION, "scope": scope, "models": models, "mode": args.mode,
                    "prompt_sha256": digest(prompt), "prompt_path": str(prompt_path), "max_tokens": args.max_tokens,
                    "max_attempts": args.max_attempts, "endpoint_identity": clients.endpoint_identity,
                    "backend": args.backend, "model_contracts": getattr(clients, "contracts", {}),
                    "code_sha256": file_sha(__file__), "published_training": False,
                    "dependencies": {str(path.relative_to(Path(__file__).parents[1])): file_sha(path) for path in (
                        Path(__file__).parents[1] / "production" / "image_screening.py",
                        Path(__file__).parents[1] / "production" / "client.py",
                        Path(__file__).parents[1] / "production" / "native_transport.py",
                        Path(__file__).parents[1] / "production" / "http_transport.py",
                        Path(__file__).parents[1] / "production" / "api_client.py",
                        Path(__file__).parents[1] / "api_review" / "aggregate.py",
                        Path(__file__).with_name("worker_control.py"),
                        Path(__file__).with_name("review_health.py"))}}
        ledger = None
        async def watch_stop_file():
            while not stop.is_set():
                if stop_path.exists():
                    stop.set()
                    return
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
        stop_watcher = asyncio.create_task(watch_stop_file()) if stop_path is not None else None
        try:
            ledger = Ledger(output, identity)
            count = 0
            imported_path = output / "INPUT_IMPORTED.json"
            imported = json.loads(imported_path.read_text()) if imported_path.exists() else None
            if imported:
                if (imported.get("input_identity_sha256") != digest(identity)
                        or imported.get("eligible_assets") != ledger.db.execute("SELECT count(*) FROM assets").fetchone()[0]):
                    raise ValueError("sealed_screening_import_mismatch")
            else:
                for row in iter_input(paths):
                    if stop.is_set():
                        break
                    ledger.insert(row)
                    count += 1
                    if count % 1000 == 0:
                        ledger.db.commit()
                        if count % 10000 == 0:
                            atomic_json(output / "PROGRESS.json", {"phase": "importing", "imported_rows": count,
                                "updated_at_epoch": time.time(), "published_training": False})
                        await asyncio.sleep(0)
                    if args.canary is not None and count >= args.canary:
                        break
            ledger.db.commit()
            before = ledger.summary()
            if not imported and not stop.is_set():
                atomic_json(imported_path, {"input_identity_sha256": digest(identity),
                                           "eligible_assets": before["counts"].get("eligible_assets", 0)})
            before_done = before["counts"].get("state:done", 0)
            before_responses = before["requests"].get("actual_responses", 0)
            processing_started = time.monotonic()
            worker = Screener(ledger, clients, models=models, mode=args.mode, concurrency=args.concurrency,
                              max_attempts=args.max_attempts, max_tokens=args.max_tokens, prompt=prompt, stop=stop)
            if args.backend == 'serving' and not stop.is_set():
                # No full-queue producer exists until this exact process has
                # received and validated a fresh real-image model response.
                proof = await preflight(worker, process_id=worker_instance)
                atomic_json(output / 'STARTUP_PREFLIGHT.json', {**proof, 'pid': os.getpid(),
                    'implementation': identity['dependencies'], 'time': time.time()})
            health = ProgressHealth(before['requests'], clients.stats())
            health_failure = None
            async def monitor():
                nonlocal health_failure
                while not stop.is_set():
                    current = ledger.summary()
                    health_failure = health.check(current['requests'], clients.stats())
                    if health_failure:
                        atomic_json(output / 'HEALTH_FAILURE.json', {**health_failure,
                            'worker_instance': worker_instance, 'time': time.time()})
                        stop.set()
                        return
                    duration = time.monotonic() - processing_started
                    fresh = current["requests"].get("actual_responses", 0) - before_responses
                    atomic_json(output / "PROGRESS.json", {**current, "phase": "annotating" if args.mode == "annotate" else "screening",
                        "worker_instance": worker_instance,
                        "elapsed_seconds": time.time() - started, "processing_elapsed_seconds": duration,
                        "fresh_actual_responses": fresh, "fresh_valid_responses_per_second": fresh / max(.001, duration),
                        "updated_at_epoch": time.time(), "transport": clients.stats(), "cpu": telemetry.stats(), "published_training": False})
                    await asyncio.sleep(getattr(args, "progress_interval", 10))
            monitor_task = asyncio.create_task(monitor())
            lag_task = asyncio.create_task(telemetry.watch_loop())
            try:
                summary = await worker.run()
            finally:
                monitor_task.cancel()
                lag_task.cancel()
                await asyncio.gather(monitor_task, lag_task, return_exceptions=True)
            processing_elapsed = time.monotonic() - processing_started
            # Stop drains and checkpoints. Exporting the entire unfinished
            # corpus on every handoff wastes CPU and delays safe recovery.
            artifact = None if stop.is_set() else export_decisions(ledger)
            elapsed = time.time() - started
            summary.update(scope=scope, elapsed_seconds=elapsed, updated_at_epoch=time.time(),
                           worker_instance=worker_instance,
                           transport=clients.stats(), cpu=telemetry.stats(), interrupted=stop.is_set(),
                           health_failure=health_failure, published_training=False,
                           processing_elapsed_seconds=processing_elapsed,
                           fresh_actual_responses=summary["requests"].get("actual_responses", 0) - before_responses,
                           fresh_valid_responses_per_second=(summary["requests"].get("actual_responses", 0) - before_responses) / max(.001, processing_elapsed),
                           review_assets_per_second=max(0, summary["counts"].get("state:done", 0) - before_done) / max(elapsed, .001))
            atomic_json(output / "SUMMARY.json", summary)
            atomic_json(output / "PROGRESS.json", summary)
            complete = {"schema_version": VERSION, "screening_pass_complete": not stop.is_set()
                        and summary["counts"].get("state:pending", 0) == 0
                        and summary["counts"].get("state:retryable", 0) == 0,
                        "all_reviews_resolved": summary["counts"].get("decision:pending", 0) == 0,
                        "published_training": False, "scope": scope,
                        "input_sha256": file_sha(output / "INPUT.json"),
                        "summary_sha256": file_sha(output / "SUMMARY.json"), "artifacts": [artifact] if artifact else []}
            atomic_json(output / "COMPLETE.json", complete)
            return summary
        finally:
            if stop_watcher is not None:
                stop_watcher.cancel()
                await asyncio.gather(stop_watcher, return_exceptions=True)
            if ledger is not None:
                ledger.close()
            await clients.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="A reconciled api_review.parquet; repeatable (canary may use selected assets)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("serving", "api"), default="serving")
    parser.add_argument("--credential-example")
    parser.add_argument("--api-key-file", help="Private absolute Serving credential path; value never enters config receipts")
    parser.add_argument("--primary-registry", action="append")
    parser.add_argument("--secondary-registry", action="append")
    parser.add_argument("--primary-pool")
    parser.add_argument("--secondary-pool")
    parser.add_argument("--wait-ready", type=float, default=60)
    parser.add_argument("--reconciliation-root")
    parser.add_argument("--selection-manifest", help="Immutable exact-row incremental API selection manifest")
    parser.add_argument("--stop-file", help="Stop cooperatively when this file exists; unfinished rows remain resumable")
    parser.add_argument("--canary", type=int)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--admission-control", help="Mutable local admission JSON; missing/invalid pauses new requests")
    parser.add_argument("--image-threads", type=int, help="Image preparation workers, at most current CPU quota and 16")
    parser.add_argument("--progress-interval", type=int, choices=range(1, 61), default=10)
    parser.add_argument("--primary", default=PRIMARY)
    parser.add_argument("--secondary", default=SECONDARY)
    parser.add_argument("--mode", choices=("both", "cascade", "primary", "annotate"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--prompt", help="Screening prompt file with the same v3 response protocol; content is bound to this run")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    print(canonical(asyncio.run(execute(args))))


if __name__ == "__main__":
    main()
