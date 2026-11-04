#!/usr/bin/env python3
"""Push whitelisted UMM metrics to the private experiment Site.

Read SITES_PUSH_TOKEN from the environment, or prompt without echoing. Credentials
are never written to disk. --watch runs independently of the training process;
--export-dir prepares inspectable JSON payloads without contacting a Site.
"""

import argparse
import datetime as dt
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import yaml


def utc_timestamp(seconds):
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def export_attempt(metrics_path):
    directory = metrics_path.parent
    root = directory.parents[2]
    config = yaml.safe_load((directory / "config.yaml").read_text())
    run = read_json(directory / "run.json")
    summary = read_json(directory / "summary.json")
    launch = read_json(directory / "launch_result.json")
    raw = metrics_path.read_bytes()
    records = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue  # A live rank may still be writing this record.
        row = json.loads(line)
        if "train/loss" in row:
            records.append(row)
    if any(a["step"] >= b["step"] for a, b in zip(records, records[1:])):
        raise ValueError(f"Non-increasing steps in {root.name}/{directory.name}")
    if any(row.get("attempt_id") != directory.name for row in records):
        raise ValueError("Metric attempt identity does not match its directory")
    settings = config["training"]
    runtime = run["runtime"]
    dp = int(runtime["topology"]["dp"])
    batch = int(settings["batch_size"])
    accumulation = int(settings["gradient_accumulation_steps"])
    files = [metrics_path] + [directory / name for name in
                             ("config.yaml", "run.json", "summary.json", "launch_result.json")]
    updated = max(path.stat().st_mtime for path in files if path.exists())
    exit_code = launch.get("exit_code", launch.get("returncode"))
    if exit_code is not None and exit_code != 0:
        status = "failed"
    elif summary.get("status") == "completed" and exit_code == 0:
        status = "completed"
    elif summary.get("status") == "completed":
        status = "finishing"
    elif summary.get("status") in ("failed", "error"):
        status = "failed"
    elif records and time.time() - metrics_path.stat().st_mtime < 1800:
        status = "running"
    else:
        status = "unknown"
    checkpoints = []
    for path in sorted((root / "checkpoints").glob("checkpoint-*")):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit() and path.is_dir():
            checkpoints.append({"step": int(suffix), "name": path.name})
    model = config["model"]
    configuration = {key: model.get(key) for key in
                     ("lambda_text", "lambda_video", "freeze_vision", "train_language",
                      "train_wan", "gradient_checkpointing")}
    configuration.update({key: config["optimizer"].get(key) for key in
                          ("backbone_lr", "generator_lr", "warmup_steps", "decay_steps", "min_lr_ratio")})
    configuration.update(qwen_model=Path(model["qwen_path"]).name,
                         wan_model=Path(model["wan_path"]).name,
                         seed=settings.get("seed"), val_every=settings.get("val_every"))
    metadata = {
        "project": "UMM_v3", "expId": root.name, "attemptId": directory.name,
        "status": status, "startedAt": run.get("started_at"),
        "finishedAt": summary.get("finished_at"), "sourceUpdatedAt": utc_timestamp(updated),
        "maxSteps": int(settings["max_train_steps"]), "worldSize": int(runtime["world_size"]),
        "dp": dp, "batchSize": batch, "accumulation": accumulation,
        "globalBatch": dp * batch * accumulation, "precision": settings["mixed_precision"],
        "durationSeconds": summary.get("seconds"), "checkpoints": checkpoints,
        "configuration": configuration,
        "source": {"file": f"{root.name}/logs/train/{directory.name}/metrics.jsonl",
                   "sha256": hashlib.sha256(raw).hexdigest()},
    }
    metrics, samples, elapsed = [], 0, 0.0
    for row in records:
        values = {key: value for key, value in row.items()
                  if (key.startswith("train/") or key in ("grad_norm", "absent_final_microbatch_gradients"))
                  and isinstance(value, (int, float)) and not isinstance(value, bool)}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Nonfinite metric in {root.name} step {row['step']}")
        samples += batch * sum(value for key, value in values.items() if key.endswith("_microbatches"))
        elapsed += row["seconds"]
        metrics.append({"step": row["step"], "timestamp": row["timestamp"],
                        "seconds": row["seconds"], "updates": row["optimizer_updates"],
                        "samplesSeen": int(samples), "elapsedSeconds": elapsed,
                        "values": values, "lr": row.get("lr", [])})
    return metadata, metrics


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a Site credential to a sign-in/other origin.


def request_json(opener, origin, path, token, payload=None):
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if payload is not None else None
    request = urllib.request.Request(origin + path, data=data,
                                     headers={"OAI-Sites-Authorization": "Bearer " + token,
                                              "Content-Type": "application/json", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with opener.open(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not print response bodies or headers: they can contain sign-in data.
            if attempt < 2 and (exc.code == 429 or exc.code >= 500):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Site HTTP {exc.code} on {path.split('?')[0]}") from None
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def prefix_digest(rows, last_step):
    digest = hashlib.sha256()
    for row in rows:
        if row["step"] <= last_step:
            digest.update(json.dumps(row, sort_keys=True, allow_nan=False).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def hidden_credential():
    old = None
    if sys.stdin.isatty():
        import termios
        old = termios.tcgetattr(sys.stdin.fileno())
        new = list(old)
        new[3] &= ~termios.ECHO
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, new)
    print("Ready for Site credential JSON on stdin (input is hidden).", flush=True)
    try:
        return json.loads(sys.stdin.readline())["sitesAuthorization"]
    finally:
        if old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--site-url", default="https://umm-experiments-lpf-20261104.hushedcabin16.chatgpt.site")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--credential-stdin", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")
    url = urllib.parse.urlsplit(args.site_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        parser.error("--site-url must be an HTTPS origin without credentials or a path")
    origin = args.site_url.rstrip("/")
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(args.outputs_dir.glob("*/logs/train/*/metrics.jsonl")):
            run, rows = export_attempt(path)
            target = args.export_dir / f"{run['expId']}__{run['attemptId']}.json"
            target.write_text(json.dumps({"schemaVersion": 1, "run": run, "metrics": rows}, ensure_ascii=False, indent=2) + "\n")
            print(f"Exported {run['expId']}: {len(rows)} records, {run['status']}")
        return
    token = hidden_credential() if args.credential_stdin else os.environ.get("SITES_PUSH_TOKEN")
    if not token:
        if not sys.stdin.isatty():
            parser.error("SITES_PUSH_TOKEN is required for non-interactive pushes")
        token = getpass.getpass("Site service credential (hidden): ")
    opener = urllib.request.build_opener(NoRedirect())
    fingerprints = {}
    verified_prefixes = {}
    failures = 0
    while True:
        try:
            for path in sorted(args.outputs_dir.glob("*/logs/train/*/metrics.jsonl")):
                run, rows = export_attempt(path)
                identity = ":".join(run[key] for key in ("project", "expId", "attemptId"))
                fingerprint = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
                if fingerprints.get(identity) == fingerprint:
                    continue
                current = request_json(opener, origin, "/api/experiments", token)
                remote = next((r for r in current["experiments"] if r["id"] == identity), None)
                last_step = remote["lastStep"] if remote else -1
                verified_step, verified_hash = verified_prefixes.get(identity, (-1, prefix_digest([], -1)))
                if prefix_digest(rows, verified_step) != verified_hash:
                    raise RuntimeError(f"Previously uploaded history changed for {run['expId']}; use a new attempt")
                local_by_step = {row["step"]: row for row in rows}
                known_remote = {row["step"] for row in rows if row["step"] <= verified_step}
                cursor = verified_step
                if remote and (last_step > verified_step or remote["recordedSteps"] != len(known_remote)):
                    while True:
                        query = urllib.parse.urlencode({"run": identity, "after": cursor})
                        page = request_json(opener, origin, "/api/metrics?" + query, token)
                        for existing in page["metrics"]:
                            if local_by_step.get(existing["step"]) != existing:
                                raise RuntimeError(f"Remote history differs for {run['expId']} step {existing['step']}")
                            known_remote.add(existing["step"])
                        cursor = page["nextCursor"]
                        if not page["hasMore"]:
                            break
                pending = [row for row in rows if row["step"] not in known_remote]
                batches = [pending[index:index + 50] for index in range(0, len(pending), 50)] or [[]]
                for chunk in batches:
                    receipt = request_json(opener, origin, "/api/ingest", token,
                                           {"schemaVersion": 1, "run": run, "metrics": chunk})
                if receipt["recordedSteps"] != len(rows):
                    raise RuntimeError(f"Record count mismatch for {run['expId']}: local {len(rows)}, remote {receipt['recordedSteps']}")
                fingerprints[identity] = fingerprint
                verified_prefixes[identity] = (receipt["lastStep"], prefix_digest(rows, receipt["lastStep"]))
                print(f"Synced {run['expId']}: {len(pending)} new, {receipt['recordedSteps']} total, step {receipt['lastStep']}, {run['status']}", flush=True)
            failures = 0
            if not args.watch:
                return
            time.sleep(args.interval)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Sync deferred: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if not args.watch:
                raise SystemExit(1)
            failures += 1
            time.sleep(min(60, args.interval * 2 ** min(failures, 3)))


if __name__ == "__main__":
    main()
