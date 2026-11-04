"""Measure a concurrency stage on the durable production queue, without replay.

This command only updates a local admission file and observes worker progress.
It does not create a worker, contact a model, call a platform, or change labels.
The already running worker keeps every successfully validated annotation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from corpus_pipeline.production.common import atomic_json, canonical, file_sha
from .worker_control import ADMISSION_SCHEMA


def set_limit(path, limit, ceiling=2048):
    if type(limit) is not int or not 0 <= limit <= ceiling <= 2048:
        raise ValueError("invalid_admission_limit")
    atomic_json(path, {"schema": ADMISSION_SCHEMA, "max_inflight": limit,
                       "updated_at_epoch": time.time()})


def delta_summary(first, last):
    if (not first.get("worker_instance") or first.get("worker_instance") != last.get("worker_instance")):
        raise ValueError("worker_restarted_during_stage")
    elapsed = last["updated_at_epoch"] - first["updated_at_epoch"]
    if elapsed <= 0:
        raise ValueError("nonpositive_observation_window")
    requests = {name: last.get("requests", {}).get(name, 0) - first.get("requests", {}).get(name, 0)
                for name in ("actual_responses", "cached_responses", "durable_receipt_hits", "failed_invocations", "serving_transient_deferrals")}
    if any(value < 0 for value in requests.values()):
        raise ValueError("worker_counter_reset_or_wrong_output")
    http_first, http_last = first.get("transport", {}), last.get("transport", {})
    if http_last.get("http_attempts", 0) < http_first.get("http_attempts", 0):
        raise ValueError("worker_restarted_during_stage")
    return {"window_seconds": elapsed, "requests": requests,
            "fresh_valid_images_per_second": requests["actual_responses"] / elapsed,
            "http_attempts": http_last.get("http_attempts", 0) - http_first.get("http_attempts", 0),
            "annotations_completed": last.get("counts", {}).get("state:done", 0) - first.get("counts", {}).get("state:done", 0),
            "cached_and_resumed_responses_in_throughput": False}


def observe_stage(args):
    output, worker = Path(args.output), Path(args.worker_output)
    if output.exists():
        raise ValueError("stage_output_must_be_new")
    if not 1 <= args.seconds <= 3600 or not 0 <= args.warmup <= 600:
        raise ValueError("stage_window_out_of_bounds")
    manifest = worker / "INPUT.json"
    identity = json.loads(manifest.read_text())
    if identity.get("mode") != "annotate" or identity.get("backend") != "serving":
        raise ValueError("stage_requires_existing_serving_annotation_scope")
    if identity.get("scope", {}).get("global_screening_scope_authorized") is not True:
        raise ValueError("stage_requires_sealed_global_queue")
    output.mkdir(parents=True)
    binding = file_sha(manifest)
    atomic_json(output / "STAGE.json", {"schema": "corpus.screening_cpu_stage.v1",
        "worker_output": str(worker.resolve()), "input_sha256": binding, "limit": args.limit,
        "warmup_seconds": args.warmup, "requested_measurement_seconds": args.seconds,
        "code_sha256": file_sha(__file__), "already_committed_annotations_retained": True})
    set_limit(args.control, args.limit)
    # Progress is a durable aggregate snapshot, not a repeated model request.
    started = time.monotonic()
    samples = []
    seen = None
    while time.monotonic() - started < args.warmup + args.seconds:
        if file_sha(manifest) != binding:
            raise ValueError("worker_scope_changed_during_stage")
        try:
            value = json.loads((worker / "PROGRESS.json").read_text())
            stamp = value.get("updated_at_epoch")
            if stamp != seen and value.get("phase") == "annotating":
                if not isinstance(stamp, (int, float)) or not 0 <= time.time() - stamp <= 60:
                    raise ValueError("worker_progress_is_stale")
                value["observer_offset_seconds"] = time.monotonic() - started
                samples.append(value)
                seen = stamp
                atomic_json(output / "OBSERVATIONS.json", samples)
        except FileNotFoundError:
            pass
        time.sleep(min(1, max(.01, args.warmup + args.seconds - (time.monotonic() - started))))
    measured = [value for value in samples if value["observer_offset_seconds"] >= args.warmup
                and value.get("transport", {}).get("admission", {}).get("max_inflight") == args.limit]
    result = {"schema": "corpus.screening_cpu_stage.v1", "limit": args.limit,
              "complete": False, "samples": len(samples), "measured_samples": len(measured),
              "annotations_retained": True, "screening_policy_changed": False,
              "note": "Sequential real-queue windows may differ in image mix and ready replica count; not a controlled model benchmark."}
    if len(measured) >= 2:
        result.update(delta_summary(measured[0], measured[-1]), complete=True)
        result["first_snapshot"] = measured[0]
        result["last_snapshot"] = measured[-1]
        result["ready_replica_counts"] = sorted({sum(entry.get("ready", False) for entry in value["transport"].get("replicas", [])) for value in measured})
    else:
        result["reason"] = "insufficient_fresh_worker_progress_at_requested_admission"
    atomic_json(output / "RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    update = sub.add_parser("set")
    update.add_argument("--control", required=True)
    update.add_argument("--limit", type=int, required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--control", required=True)
    stage.add_argument("--worker-output", required=True)
    stage.add_argument("--output", required=True)
    stage.add_argument("--limit", type=int, required=True)
    stage.add_argument("--warmup", type=int, default=30)
    stage.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args(argv)
    if args.action == "set":
        set_limit(args.control, args.limit)
        print(canonical({"max_inflight": args.limit, "no_model_requests_sent": True}))
    else:
        result = observe_stage(args)
        print(canonical({key: value for key, value in result.items() if key not in {"first_snapshot", "last_snapshot"}}))


if __name__ == "__main__":
    main()
