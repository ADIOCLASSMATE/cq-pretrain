"""Select, process, reconcile, and audit the OpenVidHD stride-2 corpus.

Run from UMM_v3: python -m data.preprocess.openvidhd --help
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
import zipfile

from .archive import SplitFile, extract_video
from .manifest import (append_journal, atomic_json, atomic_jsonl, code_identity,
                       exclusive_run, fingerprint, journal_latest, read_jsonl,
                       runtime_identity, sha256, sync_directory)

ROOT = Path("/inspire/dataset/openvid-1m/v2/OpenVidHD")
CAPTIONS = ROOT.parent / "data/train/OpenVidHD.csv"
ANALYSIS = Path(__file__).resolve().parents[1] / "analysis/results/OpenVidHD"
DESTINATION = Path("/inspire/sj-ssd3/global_user/lipengfei-253107010003/datasets/openvidhd_960x540_stride2")
TRANSFORM = {
    "version": 1, "source_width": 1920, "source_height": 1080,
    "source_min_frames": 161, "caption_filter": "nonempty after strip; stored unchanged",
    "source_fps_filter": None, "start_index": 0, "stride": 2, "truncate": False,
    "width": 960, "height": 540, "fps": "exact source fps / 2",
    "resize": "torch float32 bicubic; align_corners=False; antialias=True; clamp/round/uint8",
    "encoder": "libx264rgb", "crf": 0, "preset": "medium", "encoder_threads": 1,
    "audio": False, "vae": False, "vit": False,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def eligible(metric, caption):
    return (int(metric["width"]) == 1920 and int(metric["height"]) == 1080
            and int(metric["frame_count"]) >= 161 and bool(caption.strip()))


def pilot_rows(rows, archives):
    """Ten distinct cases, deliberately covering rates, parity, length, and split IO."""
    chosen, names, parts = [], set(), set()

    def take(candidates, reason):
        candidates = [r for r in candidates if r["sample_id"] not in names]
        row = min(candidates, key=lambda r: (r["source"]["part"] in parts, r["sample_id"]))
        row = dict(row, pilot_reason=reason)
        chosen.append(row)
        names.add(row["sample_id"])
        parts.add(row["source"]["part"])

    rates = Counter(Fraction(r["source"]["fps_num"], r["source"]["fps_den"]) for r in rows)
    for rate, _ in rates.most_common(5):
        take([r for r in rows if Fraction(r["source"]["fps_num"], r["source"]["fps_den"]) == rate],
             f"common source FPS {rate}")
    for frames in (161, 162):
        take([r for r in rows if r["source"]["num_frames"] == frames], f"boundary source frames {frames}")
    longest = max(r["source"]["num_frames"] for r in rows)
    take([r for r in rows if r["source"]["num_frames"] == longest], "longest eligible source")
    slowest = min(rates)
    take([r for r in rows if Fraction(r["source"]["fps_num"], r["source"]["fps_den"]) == slowest],
         "lowest eligible source FPS")
    common = set(rate for rate, _ in rates.most_common(5))
    take([r for r in rows if len(archives[r["source"]["part"]]) > 1
          and Fraction(r["source"]["fps_num"], r["source"]["fps_den"]) not in common],
         "uncommon FPS and split ZIP")
    return chosen


def select(output, pilot):
    output = Path(output)
    if (output / "processing_config.json").exists():
        raise FileExistsError("Selection already frozen; use process to resume or a new output root")
    census = json.loads((ANALYSIS / "summary.json").read_text())
    if not census["complete"] or Path(census["video_root"]) != ROOT:
        raise ValueError("Expected complete OpenVidHD-only census")
    original_hash = sha256(CAPTIONS)
    if original_hash != census["source_sha256"]:
        raise ValueError("Original annotations changed since census")
    archives = dict(json.loads((ANALYSIS / "archive_manifest.json").read_text())["archives"])
    identities = {}
    for part, paths in archives.items():
        identities[str(part)] = []
        for path in paths:
            path = Path(path)
            if path.parent != ROOT:
                raise ValueError("Archive outside OpenVidHD scope")
            stat = path.stat()
            identities[str(part)].append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    with CAPTIONS.open(newline="", encoding="utf-8") as handle:
        annotations = {}
        for row in csv.DictReader(handle):
            if row["video"] in annotations:
                raise ValueError("Duplicate annotation video ID")
            annotations[row["video"]] = row
    groups = {}
    for group in json.loads((ROOT / "OpenVidHD.json").read_text()):
        for name, videos in group.items():
            for video in videos:
                if video in groups:
                    raise ValueError("Duplicate video in original part index")
                groups[video] = int(name.removeprefix("part"))
    with (ANALYSIS / "inventory.csv").open(newline="") as handle:
        inventory = {}
        for row in csv.DictReader(handle):
            if row["video"] in inventory:
                raise ValueError("Duplicate physical video ID")
            inventory[row["video"]] = row
    rows, seen, population = [], set(), 0
    with (ANALYSIS / "all_video_metrics.csv").open(newline="") as handle:
        for metric in csv.DictReader(handle):
            population += 1
            name = metric["video"]
            if name in seen:
                raise ValueError("Duplicate census ID")
            seen.add(name)
            annotation = annotations.get(name)
            if annotation is None or not eligible(metric, annotation["caption"]):
                continue
            if metric["status"] != "ok" or metric["timing_class"] != "constant" or metric["rotated"] != "False":
                raise ValueError(f"Eligible source needs timing/rotation investigation: {name}")
            if Path(name).name != name:
                raise ValueError("Source ID must be an unchanged basename")
            info = inventory[name]
            part = int(metric["archive"])
            if groups[name] != part or int(info["archive"]) != part or info["member"] != metric["member"]:
                raise ValueError(f"Conflicting archive mappings: {name}")
            if int(float(annotation["frame"])) != int(metric["frame_count"]):
                raise ValueError(f"Annotation/census frame count mismatch: {name}")
            rate = Fraction(int(metric["timescale"]), int(metric["pts_interval_min_ticks"]))
            rows.append(dict(schema_version=1, sample_id=name, video=name,
                video_path=f"videos/part{part}/{name}", caption=annotation["caption"],
                source=dict(dataset="OpenVidHD", part=part, member=info["member"],
                    archive_files=archives[part], bytes=int(info["bytes"]), zip_crc32=int(info["zip_crc32"], 16),
                    width=1920, height=1080, num_frames=int(metric["frame_count"]),
                    fps_num=rate.numerator, fps_den=rate.denominator,
                    annotation=annotation)))
    if population != census["population"]:
        raise ValueError("Incomplete metrics population")
    eligible_count = len(rows)
    total_frames = sum((r["source"]["num_frames"] + 1) // 2 for r in rows)
    if pilot:
        rows = pilot_rows(rows, archives)
    rows.sort(key=lambda r: (r["source"]["part"], r["sample_id"]))
    (output / "metadata").mkdir(parents=True, exist_ok=True)
    for path in (CAPTIONS, ROOT / "OpenVidHD.json"):
        shutil.copyfile(path, output / "metadata" / path.name)
    atomic_jsonl(output / "metadata/selected.jsonl", rows)
    config = dict(schema_version=1, created_at=now(), pilot=pilot, candidates=len(rows),
                  full_eligible_count=eligible_count, full_output_frames=total_frames,
                  transform=TRANSFORM, runtime=runtime_identity(), code_sha256=code_identity(),
                  source_csv_sha256=original_hash, source_index_sha256=sha256(ROOT / "OpenVidHD.json"),
                  metrics_sha256=sha256(ANALYSIS / "all_video_metrics.csv"),
                  inventory_sha256=sha256(ANALYSIS / "inventory.csv"), archive_identities=identities,
                  selected_sha256=sha256(output / "metadata/selected.jsonl"))
    config["fingerprint"] = fingerprint(config)
    atomic_json(output / "processing_config.json", config)
    print(json.dumps({k: config[k] for k in ("pilot", "candidates", "full_eligible_count", "full_output_frames")}), flush=True)


def load_config(output):
    output = Path(output)
    config = json.loads((output / "processing_config.json").read_text())
    identity = dict(config)
    expected = identity.pop("fingerprint")
    if fingerprint(identity) != expected:
        raise ValueError("Configuration fingerprint mismatch")
    if config["code_sha256"] != code_identity() or config["runtime"] != runtime_identity():
        raise ValueError("Code/runtime changed; do not mix preprocessing versions")
    if config["transform"] != TRANSFORM:
        raise ValueError("Transform changed")
    for filename, key in (("selected.jsonl", "selected_sha256"), ("OpenVidHD.csv", "source_csv_sha256"),
                          ("OpenVidHD.json", "source_index_sha256")):
        if sha256(output / "metadata" / filename) != config[key]:
            raise ValueError(f"Frozen metadata changed: {filename}")
    return config


def part_work(output, part, rows, config, min_free_gib):
    import torch
    from .video import transcode
    torch.set_num_threads(1)
    output = Path(output)
    directory = output / "_state"
    directory.mkdir(exist_ok=True)
    journal = directory / f"part{part}.jsonl"
    latest = journal_latest(journal, repair_tail=True)
    names = {row["sample_id"] for row in rows}
    if set(latest) - names:
        raise ValueError("Journal contains records outside selection")
    for saved in latest.values():
        if saved["config_fingerprint"] != config["fingerprint"]:
            raise ValueError("Journal configuration mismatch")
    identities = config["archive_identities"][str(part)]
    for identity in identities:
        stat = Path(identity["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
            raise ValueError("Archive identity changed")
    paths = [i["path"] for i in identities]
    status_path = directory / f"part{part}.status.json"

    def publish(state):
        counts = Counter(r["status"] for r in latest.values())
        atomic_json(status_path, dict(part=part, state=state, updated_at=now(), total=len(rows),
                    success=counts["success"], failed=counts["failure"],
                    pending=len(rows) - len(latest),
                    output_bytes=sum(r["record"]["output_bytes"] for r in latest.values() if r["status"] == "success")))

    scratch = output / "_scratch" / f"part{part}"
    scratch.mkdir(parents=True, exist_ok=True)
    # This part has a single writer under the dataset run lock. Remove only its
    # abandoned extraction directories; original ZIPs and committed MP4s stay intact.
    for abandoned in scratch.glob("video-*"):
        if abandoned.is_dir():
            shutil.rmtree(abandoned)
    publish("running")
    with SplitFile(paths) as joined, zipfile.ZipFile(joined) as archive, journal.open("a", encoding="utf-8") as log:
        rows = sorted(rows, key=lambda r: archive.getinfo(r["source"]["member"]).header_offset)
        for selected in rows:
            if (output / "STOP").exists():
                publish("stopped")
                return
            if shutil.disk_usage(output).free < min_free_gib * 1024 ** 3:
                publish("low_disk")
                raise OSError("Free disk below configured reserve; preserving pending records")
            name = selected["sample_id"]
            destination = output / selected["video_path"]
            saved = latest.get(name)
            if saved and saved["status"] == "success" and destination.exists():
                if sha256(destination) == saved["record"]["output_sha256"]:
                    continue
            # Any existing file without a committed matching success is rebuilt.
            started = time.monotonic()
            temporary = destination.with_name(destination.name + ".partial")
            event = dict(sample_id=name, config_fingerprint=config["fingerprint"], updated_at=now())
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="video-", dir=scratch) as working:
                    original = Path(working) / "source.mp4"
                    original_sha = extract_video(archive, selected, original)
                    record = transcode(original, temporary, selected)
                record["source"]["sha256"] = original_sha
                record["processing"] = dict(TRANSFORM, config_fingerprint=config["fingerprint"])
                record["output_sha256"] = sha256(temporary)
                record["processing_seconds"] = round(time.monotonic() - started, 4)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                sync_directory(destination.parent)
                event.update(status="success", record=record)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                event.update(status="failure", error=f"{type(exc).__name__}: {exc}",
                             traceback=traceback.format_exc())
                print(f"FAIL part{part} {name}: {exc}", flush=True)
            append_journal(log, event)
            latest[name] = event
            publish("running")
    publish("finished")


def reconcile(output, config, *, worker_errors=None):
    output = Path(output)
    selected = read_jsonl(output / "metadata/selected.jsonl")
    rows = {r["sample_id"]: r for r in selected}
    latest = {}
    for part in sorted({r["source"]["part"] for r in selected}):
        for name, event in journal_latest(output / f"_state/part{part}.jsonl").items():
            if name in latest or name not in rows or event["config_fingerprint"] != config["fingerprint"]:
                raise ValueError("Journal identity conflict")
            latest[name] = event
    successes, failures = [], []
    for selected in selected:
        event = latest.get(selected["sample_id"])
        if event is None:
            continue
        if event["status"] == "success":
            record = event["record"]
            if any(record[k] != selected[k] for k in ("sample_id", "video", "video_path", "caption")):
                raise ValueError("Output identity or caption changed")
            path = output / record["video_path"]
            if not path.exists() or path.stat().st_size != record["output_bytes"]:
                raise ValueError(f"Committed output missing/truncated: {path}")
            successes.append(record)
        else:
            failures.append(event)
    pending = len(rows) - len(latest)
    atomic_jsonl(output / "samples.jsonl", successes)
    atomic_jsonl(output / "failures.jsonl", failures)
    summary = dict(updated_at=now(), candidates=len(rows), success=len(successes),
                   failure=len(failures), pending=pending,
                   processing_finished=pending == 0,
                   all_candidates_succeeded=len(successes) == len(rows),
                   output_frames=sum(r["num_frames"] for r in successes),
                   output_bytes=sum(r["output_bytes"] for r in successes),
                   worker_errors=worker_errors or [], config_fingerprint=config["fingerprint"])
    atomic_json(output / "summary.json", summary)
    if not (output / "validation.json").exists():
        atomic_json(output / "commit_validation.json", dict(
            verification_mode="full output decode before every successful atomic commit",
            source_correspondence=False, verified=len(successes),
            all_passed=summary["all_candidates_succeeded"],
            config_fingerprint=config["fingerprint"], updated_at=now()))
    print(json.dumps(summary), flush=True)
    return summary


def check_pilot(path, config):
    path = Path(path)
    pilot = load_config(path)
    validation = json.loads((path / "validation.json").read_text())
    if not (pilot["pilot"] and pilot["candidates"] == 10 and validation["all_passed"]
            and validation["source_correspondence"] and validation["verified"] == 10
            and validation["config_fingerprint"] == pilot["fingerprint"]):
        raise ValueError("Ten-source pilot has not passed the full independent audit")
    for key in ("transform", "runtime", "code_sha256", "source_csv_sha256", "source_index_sha256",
                "metrics_sha256", "inventory_sha256", "archive_identities"):
        if config[key] != pilot[key]:
            raise ValueError(f"Pilot/full configuration mismatch: {key}")


def process(output, workers, min_free_gib, pilot_root):
    output = Path(output)
    config = load_config(output)
    if not config["pilot"]:
        if pilot_root is None:
            raise ValueError("Full processing requires --require-pilot pointing to a passed ten-sample pilot")
        check_pilot(pilot_root, config)
    groups = defaultdict(list)
    for row in read_jsonl(output / "metadata/selected.jsonl"):
        groups[row["source"]["part"]].append(row)
    started = time.monotonic()
    errors = []
    atomic_json(output / "run.json", dict(state="running", started_at=now(), pid=os.getpid(),
                workers=workers, min_free_gib=min_free_gib, pilot_root=pilot_root,
                config_fingerprint=config["fingerprint"]))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(part_work, str(output), part, rows, config, min_free_gib): part
                   for part, rows in groups.items()}
        while futures:
            done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                part = futures.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    errors.append(dict(part=part, error=f"{type(exc).__name__}: {exc}"))
                    print(f"WORKER FAILURE part{part}: {exc}", flush=True)
            statuses = []
            for path in (output / "_state").glob("*.status.json"):
                statuses.append(json.loads(path.read_text()))
            success = sum(s["success"] for s in statuses)
            failed = sum(s["failed"] for s in statuses)
            progress = dict(updated_at=now(), candidates=config["candidates"], success=success,
                            failure=failed, pending=config["candidates"] - success - failed,
                            active_or_queued_parts=len(futures), elapsed_seconds=round(time.monotonic() - started, 2),
                            output_bytes=sum(s["output_bytes"] for s in statuses), worker_errors=errors)
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    summary = reconcile(output, config, worker_errors=errors)
    atomic_json(output / "run.json", dict(state="succeeded" if summary["all_candidates_succeeded"] else "incomplete",
                finished_at=now(), elapsed_seconds=round(time.monotonic() - started, 2), workers=workers,
                config_fingerprint=config["fingerprint"]))
    return 0 if summary["all_candidates_succeeded"] else 2


def verify(output, source_correspondence):
    import torch
    from .video import audit_source_correspondence, verify_output
    torch.set_num_threads(1)
    output = Path(output)
    config = load_config(output)
    summary = reconcile(output, config)
    with (output / "metadata/OpenVidHD.csv").open(newline="", encoding="utf-8") as handle:
        annotations = {r["video"]: r for r in csv.DictReader(handle)}
    results = []
    for record in read_jsonl(output / "samples.jsonl"):
        path = output / record["video_path"]
        try:
            if record["caption"] != annotations[record["sample_id"]]["caption"]:
                raise ValueError("Caption differs from original CSV")
            if record["source"]["annotation"] != annotations[record["sample_id"]]:
                raise ValueError("Original annotation fields differ")
            if sha256(path) != record["output_sha256"]:
                raise ValueError("Output file hash mismatch")
            checks = verify_output(path, record, record["validation"]["rgb_sha256"])
            if source_correspondence:
                with SplitFile(record["source"]["archive_files"]) as joined, zipfile.ZipFile(joined) as archive:
                    with tempfile.TemporaryDirectory(dir=output / "_scratch") as directory:
                        original = Path(directory) / "source.mp4"
                        if extract_video(archive, record, original) != record["source"]["sha256"]:
                            raise ValueError("Original file hash mismatch")
                        checks.update(audit_source_correspondence(original, path, record))
            results.append(dict(sample_id=record["sample_id"], passed=True, **checks))
        except Exception as exc:
            results.append(dict(sample_id=record["sample_id"], passed=False, error=f"{type(exc).__name__}: {exc}"))
    report = dict(updated_at=now(), config_fingerprint=config["fingerprint"],
                  source_correspondence=source_correspondence, verified=len(results),
                  all_passed=summary["all_candidates_succeeded"] and all(r["passed"] for r in results), samples=results)
    atomic_json(output / "validation.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["all_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "process", "finalize", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output", type=Path, default=DESTINATION)
        if command == "select":
            sub.add_argument("--pilot", action="store_true")
        elif command == "process":
            sub.add_argument("--workers", type=int, default=4)
            sub.add_argument("--min-free-gib", type=float, default=100)
            sub.add_argument("--require-pilot")
        elif command == "verify":
            sub.add_argument("--source-correspondence", action="store_true")
    args = parser.parse_args()
    with exclusive_run(args.output):
        if args.command == "select":
            select(args.output, args.pilot)
            return 0
        if args.command == "process":
            if args.workers < 1 or args.min_free_gib < 1:
                parser.error("workers and min-free-gib must be positive")
            return process(args.output, args.workers, args.min_free_gib, args.require_pilot)
        if args.command == "verify":
            return verify(args.output, args.source_correspondence)
        reconcile(args.output, load_config(args.output))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
