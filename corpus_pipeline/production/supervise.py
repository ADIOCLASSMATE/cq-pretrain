"""Persistently wait for verified endpoints, then resume one bounded real canary.

This is a local canary supervisor, not a cluster resource allocator or a full
corpus scheduler. It never retries schema/semantic quality failures forever.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import atomic_json, digest, file_sha, jsonl, require_production_registry
from .client import endpoint_lease_valid


def process_identity(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    return {"pid": pid, "start_ticks": text.rsplit(")", 1)[1].split()[19]}


def ready_generation(registry):
    try:
        value = require_production_registry(registry)
        selected = []
        for role in ("generator", "verifier"):
            entries = [row for row in value["endpoints"] if row["role"] == role and row.get("ready") and endpoint_lease_valid(row)]
            if len(entries) != 1 or not entries[0].get("base_url"):
                return None
            row = entries[0]
            if not row.get("runtime_revision") or str(row["runtime_revision"]).startswith("pending"):
                return None
            selected.append({key: row.get(key) for key in ("role", "model", "revision", "processor_revision",
                            "runtime_revision", "precision", "base_url", "request_overrides", "expected_boots", "incarnation")})
        return {"registry_generation": value["registry_generation"], "endpoint_identity_sha256": digest(selected)}
    except (OSError, KeyError, ValueError):
        return None


def same_qualified_endpoints(first, second):
    """Renewing a readiness lease or registry counter is not a new qualification."""
    return bool(first and second and first.get("endpoint_identity_sha256") == second.get("endpoint_identity_sha256"))


def failure_summary(output):
    summary = json.loads((output / "summary.json").read_text())
    attempt = summary["attempt_id"]
    records = [json.loads(path.read_text()) for path in (output / "attempt_records" / attempt).glob("*.json")]
    failed = [row for row in records if row["status"] == "model_stage_failed"]
    failed += [row for row in jsonl(output / "pair_reviews.jsonl") if row["status"] == "failed"]
    retryable = bool(failed) and not summary["asset_failures"] and all(row.get("retryable_after_endpoint_change") is True for row in failed)
    return {"attempt_id": attempt, "failed_stages": len(failed), "retryable_after_endpoint_change": retryable,
            "summary_sha256": file_sha(output / "summary.json")}


def supervise(args):
    require_production_registry(args.endpoints)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = process_identity(os.getpid())
    identity.update(manifest_sha256=file_sha(args.manifest), code_sha256=file_sha(__file__),
                    started_at_unix=time.time(), scope="persistent_canary_endpoint_waiter")
    child = None
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    blocked_generation = None
    executions = 0
    last_failure = None
    receipt = output / "supervisor.json"
    while not stopping:
        generation = ready_generation(args.endpoints)
        if generation is None or same_qualified_endpoints(generation, blocked_generation):
            atomic_json(receipt, {**identity, "status": "waiting_for_verified_endpoint_generation",
                        "heartbeat_unix": time.time(), "executions": executions,
                        "blocked_generation": blocked_generation, "last_failure": last_failure,
                        "model_generation_completed": False})
            time.sleep(args.poll_seconds)
            continue
        command = [sys.executable, "-m", "corpus_pipeline.production.run", "run",
                   "--manifest", str(Path(args.manifest).resolve()), "--output", str(output),
                   "--endpoints", str(Path(args.endpoints).resolve()), "--wait-endpoints", str(args.wait_endpoints),
                   "--record-concurrency", str(args.record_concurrency), "--model-concurrency", str(args.model_concurrency)]
        executions += 1
        log_path = output / (f"supervisor_child_{executions:03d}_{int(time.time())}.log")
        launched = time.time()
        with log_path.open("a", buffering=1) as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            running_state = {**identity, "status": "running_real_canary", "child": process_identity(child.pid),
                             "heartbeat_unix": time.time(), "executions": executions, "endpoint_generation": generation,
                             "log_path": str(log_path), "command_sha256": digest(command)}
            atomic_json(receipt, running_state)
            while child.poll() is None and not stopping:
                time.sleep(min(args.poll_seconds, 15))
                running_state.update(heartbeat_unix=time.time(), child_running=child.poll() is None)
                atomic_json(receipt, running_state)
            if stopping:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                break
        complete_path = output / "COMPLETE.json"
        summary_path = output / "summary.json"
        if child.returncode != 0 or not complete_path.exists() or complete_path.stat().st_mtime < launched:
            atomic_json(receipt, {**identity, "status": "execution_failed_requires_inspection", "executions": executions,
                        "exit_code": child.returncode, "log_path": str(log_path), "heartbeat_unix": time.time(),
                        "automatic_retry": False})
            return 1
        complete = json.loads(complete_path.read_text())
        if complete["summary_sha256"] != file_sha(summary_path):
            raise ValueError("supervisor received mismatched completion artifacts")
        if complete["canary_completed"]:
            atomic_json(receipt, {**identity, "status": "completed_real_canary", "executions": executions,
                        "completion_sha256": file_sha(complete_path), "heartbeat_unix": time.time(),
                        "log_path": str(log_path), "required_stage_acceptance": complete["required_stage_acceptance"]})
            return 0
        last_failure = failure_summary(output)
        if not last_failure["retryable_after_endpoint_change"]:
            atomic_json(receipt, {**identity, "status": "quality_or_nontransient_failure_requires_inspection",
                        "executions": executions, "last_failure": last_failure,
                        "required_stage_acceptance": complete.get("required_stage_acceptance"),
                        "heartbeat_unix": time.time(), "automatic_retry": False})
            return 2
        blocked_generation = generation
    atomic_json(receipt, {**identity, "status": "stopped", "executions": executions, "heartbeat_unix": time.time()})
    return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoints", default=str(Path(__file__).resolve().parents[1] / "serving/endpoints.json"))
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--wait-endpoints", type=float, default=180)
    parser.add_argument("--record-concurrency", type=int, default=4)
    parser.add_argument("--model-concurrency", type=int, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll seconds must be between 1 and 60")
    return supervise(args)


if __name__ == "__main__":
    sys.exit(main())
