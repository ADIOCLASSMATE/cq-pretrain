"""Isolated CPU-host, cross-host consistency, and throughput probes; no production writes."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import time

from data.preprocess.manifest import atomic_json, atomic_jsonl, read_jsonl, require_readable, sha256
from data.preprocess.openvidhd import VARIANTS, load_config, now, part_work, reconcile

DATASETS = Path("/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets")
PRODUCTION = DATASETS / "openvidhd_selected"
BASELINE = DATASETS / "openvidhd_selected_pilot10"
PROBE = DATASETS / "openvidhd_selected_parallel_probe_20261030"


def preflight():
    config = load_config(BASELINE)
    report = dict(checked_at=now(), host=platform.node(), architecture=platform.machine(),
                  runtime_matches=True, frozen_code_matches=True, runtime=config["runtime"],
                  logical_cpus=os.cpu_count(), production_touched=False)
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        report[name] = path.read_text().strip()
    first = read_jsonl(BASELINE / "metadata/selected.jsonl")[0]
    source = Path(first["source"]["archive_files"][0])
    with source.open("rb") as handle:
        report["source_zip_signature_readable"] = handle.read(4) == b"PK\x03\x04"
    require_readable(report)
    atomic_json(PROBE / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False))


def lock_check(expected):
    path = PROBE / "shared_lock_probe"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = "available"
        except BlockingIOError:
            state = "blocked"
        report = dict(checked_at=now(), host=platform.node(), expected=expected, observed=state, passed=state == expected)
        atomic_json(PROBE / f"lock_{expected}.json", report)
        print(json.dumps(report))
        if state != expected:
            raise ValueError(f"Shared flock expected {expected}, got {state}")


def mkdir_check(expected):
    claim = PROBE / "mkdir_claim_probe"
    try:
        claim.mkdir()
        state = "available"
    except FileExistsError:
        state = "blocked"
    report = dict(checked_at=now(), host=platform.node(), expected=expected, observed=state, passed=state == expected)
    atomic_json(PROBE / f"mkdir_{expected}.json", report)
    print(json.dumps(report))
    if state != expected:
        raise ValueError(f"Atomic mkdir expected {expected}, got {state}")
    if state == "available":
        claim.rmdir()


def prepare_benchmark():
    output = PROBE / "benchmark64"
    if (output / "metadata/selected.jsonl").exists():
        raise FileExistsError("Benchmark selection already frozen")
    started = {int(p.name.removeprefix("part").split(".")[0])
               for p in (PRODUCTION / "_state").glob("*.status.json")}
    population = []
    with (PRODUCTION / "metadata/selected.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["source"]["part"] not in started:
                population.append(row)
    selected = random.Random(20261030).sample(population, 64)
    selected.sort(key=lambda r: (r["source"]["part"], r["sample_id"]))
    names = {r["sample_id"] for r in selected}
    source_checks = []
    with (PRODUCTION / "metadata/source_integrity.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in names:
                source_checks.append(row)
    if len(source_checks) != 64:
        raise ValueError("Missing source integrity records")
    require_readable(selected)
    atomic_jsonl(output / "metadata/selected.jsonl", selected)
    atomic_jsonl(output / "metadata/source_integrity.jsonl", source_checks)
    report = dict(created_at=now(), seed=20261030, samples=64, population_samples=len(population),
                  selection="Uniform without replacement over samples in not-yet-started parts at snapshot",
                  parts=len({r["source"]["part"] for r in selected}),
                  source_frames=sum(r["source"]["num_frames"] for r in selected),
                  output_frames_per_variant=sum((r["source"]["num_frames"]+1)//2 for r in selected))
    atomic_json(output / "selection.json", report)
    print(json.dumps(report))


def benchmark():
    output = PROBE / "benchmark64"
    config = load_config(BASELINE)
    config["run_id"] = "openvidhd-cpu-benchmark64-20261030"
    selected = read_jsonl(output / "metadata/selected.jsonl")
    source_checks = {r["sample_id"]: r["zip_crc32"]
                     for r in read_jsonl(output / "metadata/source_integrity.jsonl")}
    groups = defaultdict(list)
    for row in selected:
        groups[row["source"]["part"]].append(row)
    if (output / "_state").exists():
        raise FileExistsError("A timing benchmark must start with no committed outputs")
    atomic_json(output / "benchmark_config.json", dict(
        run_id=config["run_id"], workers=16, runtime=config["runtime"],
        transform=config["transform"], reference_pilot=str(BASELINE)))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(part_work, str(output), part, rows, config,
                   {r["sample_id"]: source_checks[r["sample_id"]] for r in rows}, 100)
                   for part, rows in groups.items()]
        for future in as_completed(futures):
            future.result()
    elapsed = time.monotonic() - started
    summary = reconcile(output, config)
    source_frames = sum(r["source"]["num_frames"] for r in selected)
    report = dict(checked_at=now(), host=platform.node(), workers=16,
                  samples=len(selected), source_frames=source_frames,
                  elapsed_seconds=elapsed, source_frames_per_second=source_frames/elapsed,
                  samples_per_hour=len(selected)*3600/elapsed,
                  all_passed=summary["all_candidates_succeeded"],
                  output_files=summary["committed_video_files"], output_bytes=summary["output_bytes"],
                  methodology="Includes process startup, ZIP extraction, dual resize/encode, complete output decode and durable commit; excludes final manifest reconciliation")
    require_readable(report)
    atomic_json(output / "benchmark_report.json", report)
    print(json.dumps(report))
    if not report["all_passed"]:
        raise ValueError("Benchmark output validation failed")


def compare_pilot():
    candidate = PROBE / "pilot10"
    a = json.loads((BASELINE / "validation.json").read_text())
    b = json.loads((candidate / "validation.json").read_text())
    if not (a["all_passed"] and b["all_passed"] and b["verified"] == 20 and b["source_correspondence"]):
        raise ValueError("Both pilots must pass the full source audit")
    expected = {(r["sample_id"], r["variant"]): r for r in read_jsonl(BASELINE / "metadata/integrity.jsonl")}
    actual = {(r["sample_id"], r["variant"]): r for r in read_jsonl(candidate / "metadata/integrity.jsonl")}
    if set(expected) != set(actual):
        raise ValueError("Cross-host sample identity mismatch")
    for key in expected:
        for field in ("source_sha256", "pre_encoder_rgb_sha256", "decoded_rgb_sha256", "output_sha256"):
            if expected[key][field] != actual[key][field]:
                raise ValueError(f"Cross-host difference in {field} for {key}")
    for variant in VARIANTS:
        original = read_jsonl(BASELINE / variant / "samples.jsonl")
        new = read_jsonl(candidate / variant / "samples.jsonl")
        for left, right in zip(original, new, strict=True):
            for field in ("sample_id", "caption", "source", "width", "height", "num_frames", "fps_num", "fps_den", "spatial_transform"):
                if left[field] != right[field]:
                    raise ValueError(f"Cross-host metadata changed: {field}")
            if sha256(candidate / variant / right["video_path"]) != expected[right["sample_id"], variant]["output_sha256"]:
                raise ValueError("Cross-host output file changed")
    report = dict(checked_at=now(), all_passed=True, source_samples=10, output_files=20,
                  all_5120_frames_source_audited=True, captions_and_ids_unchanged=True,
                  encoder_input_pixels_identical=True, decoded_pixels_identical=True,
                  mp4_files_byte_identical=True)
    atomic_json(PROBE / "cross_host_validation.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "prepare-benchmark", "benchmark", "compare-pilot", "lock", "mkdir"])
    parser.add_argument("--expected", choices=["blocked", "available"])
    args = parser.parse_args()
    if args.action == "lock":
        lock_check(args.expected)
    elif args.action == "mkdir":
        mkdir_check(args.expected)
    else:
        {"preflight": preflight, "prepare-benchmark": prepare_benchmark,
         "benchmark": benchmark, "compare-pilot": compare_pilot}[args.action]()
