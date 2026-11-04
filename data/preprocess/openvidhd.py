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
import zipfile

from .archive import SplitFile, extract_video
from .manifest import (append_journal, atomic_json, atomic_jsonl, code_identity,
                       exclusive_run, journal_latest, read_jsonl,
                       require_readable, runtime_identity, sha256, sync_directory)

ROOT = Path("/inspire/dataset/openvid-1m/v2/OpenVidHD")
CAPTIONS = ROOT.parent / "data/train/OpenVidHD.csv"
ANALYSIS = Path(__file__).resolve().parents[1] / "analysis/results/OpenVidHD"
DESTINATION = Path("/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected")
VARIANTS = {"960x544": {"width": 960, "height": 544}, "832x480": {"width": 832, "height": 480}}
TRANSFORM = {
    "version": 3, "source_width": 1920, "source_height": 1080,
    "source_min_frames": 161, "caption_filter": "nonempty after strip; stored unchanged",
    "source_fps_filter": None, "start_index": 0, "stride": 2, "truncate": False,
    "variants": VARIANTS, "fps": "exact source fps / 2",
    "resize": "independent aspect-preserving cover resize and center crop; torch float32 bicubic; align_corners=False; antialias=True; clamp/round/uint8",
    "encoder": "mpeg4", "codec": "MPEG-4 Part 2", "container": "mp4",
    "pixel_format": "yuv420p", "qscale": 2, "encoder_threads": 1,
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
            rows.append(dict(schema_version=2, sample_id=name, video=name,
                video_path=f"videos/part{part}/{name}", caption=annotation["caption"],
                source=dict(dataset="OpenVidHD", part=part, member=info["member"],
                    archive_files=archives[part], bytes=int(info["bytes"]),
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
    require_readable(rows)
    atomic_jsonl(output / "metadata/selected.jsonl", rows)
    atomic_jsonl(output / "metadata/source_integrity.jsonl", [dict(sample_id=r["sample_id"],
                 zip_crc32=int(inventory[r["sample_id"]]["zip_crc32"], 16)) for r in rows])
    config = dict(schema_version=2, created_at=now(), pilot=pilot, candidates=len(rows),
                  run_id="openvidhd-v3-" + ("pilot-" if pilot else "full-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                  full_eligible_count=eligible_count, full_output_frames=total_frames,
                  transform=TRANSFORM, runtime=runtime_identity(), archive_identities=identities)
    require_readable(config)
    atomic_json(output / "processing_config.json", config)
    atomic_json(output / "metadata/run_integrity.json", dict(code_sha256=code_identity(),
                processing_config_sha256=sha256(output / "processing_config.json"),
                metadata_sha256={name: sha256(output / "metadata" / name) for name in
                    ("selected.jsonl", "source_integrity.jsonl", "OpenVidHD.csv", "OpenVidHD.json")},
                metrics_sha256=sha256(ANALYSIS / "all_video_metrics.csv"),
                inventory_sha256=sha256(ANALYSIS / "inventory.csv")))
    print(json.dumps({k: config[k] for k in ("pilot", "candidates", "full_eligible_count", "full_output_frames")}), flush=True)


def load_config(output):
    output = Path(output)
    config = json.loads((output / "processing_config.json").read_text())
    integrity = json.loads((output / "metadata/run_integrity.json").read_text())
    if sha256(output / "processing_config.json") != integrity["processing_config_sha256"]:
        raise ValueError("Configuration fingerprint mismatch")
    if integrity["code_sha256"] != code_identity() or config["runtime"] != runtime_identity():
        raise ValueError("Code/runtime changed; do not mix preprocessing versions")
    if config["transform"] != TRANSFORM:
        raise ValueError("Transform changed")
    for filename, expected in integrity["metadata_sha256"].items():
        if sha256(output / "metadata" / filename) != expected:
            raise ValueError(f"Frozen metadata changed: {filename}")
    config["_integrity"] = integrity
    return config


def event_counts(rows, latest):
    variants = {}
    for variant in VARIANTS:
        events = [latest[r["sample_id"], variant] for r in rows if (r["sample_id"], variant) in latest]
        successful = [e for e in events if e["status"] == "success"]
        variants[variant] = dict(success=len(successful),
            failure=sum(e["status"] == "failure" for e in events), pending=len(rows) - len(events),
            output_bytes=sum(e["record"]["output_bytes"] for e in successful),
            output_frames=sum(e["record"]["num_frames"] for e in successful))
    paired = failed = 0
    for row in rows:
        states = [latest.get((row["sample_id"], v), {}).get("status") for v in VARIANTS]
        paired += all(s == "success" for s in states)
        failed += "failure" in states
    return dict(success=paired, failure=failed, pending=len(rows) - paired - failed,
                variants=variants, output_bytes=sum(v["output_bytes"] for v in variants.values()),
                committed_video_files=sum(v["success"] for v in variants.values()))


def part_work(output, part, rows, config, source_checks, min_free_gib):
    import torch
    from .video import transcode
    torch.set_num_threads(1)
    output = Path(output)
    directory = output / "_state"
    checks_directory = output / "metadata/integrity_parts"
    directory.mkdir(exist_ok=True)
    checks_directory.mkdir(parents=True, exist_ok=True)
    journal = directory / f"part{part}.jsonl"
    checks_path = checks_directory / f"part{part}.jsonl"
    latest = journal_latest(journal, repair_tail=True, per_variant=True)
    checks = journal_latest(checks_path, repair_tail=True, per_variant=True)
    valid_keys = {(r["sample_id"], v) for r in rows for v in VARIANTS}
    if set(latest) - valid_keys or set(checks) - valid_keys:
        raise ValueError("Journal contains records outside selection")
    if any(e["run_id"] != config["run_id"] for e in [*latest.values(), *checks.values()]):
        raise ValueError("Journal configuration mismatch")
    identities = config["archive_identities"][str(part)]
    for identity in identities:
        stat = Path(identity["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
            raise ValueError("Archive identity changed")
    status_path = directory / f"part{part}.status.json"

    def publish(state):
        atomic_json(status_path, dict(part=part, state=state, updated_at=now(), total=len(rows),
                                      **event_counts(rows, latest)))

    scratch = output / "_scratch" / f"part{part}"
    scratch.mkdir(parents=True, exist_ok=True)
    for abandoned in scratch.glob("video-*"):
        if abandoned.is_dir():
            shutil.rmtree(abandoned)
    publish("running")
    with SplitFile([i["path"] for i in identities]) as joined, zipfile.ZipFile(joined) as archive, \
            journal.open("a", encoding="utf-8") as log, checks_path.open("a", encoding="utf-8") as check_log:
        rows = sorted(rows, key=lambda r: archive.getinfo(r["source"]["member"]).header_offset)
        for selected in rows:
            if (output / "STOP").exists():
                publish("stopped")
                return
            if shutil.disk_usage(output).free < min_free_gib * 1024 ** 3:
                publish("low_disk")
                raise OSError("Free disk below configured reserve; preserving pending records")
            name = selected["sample_id"]
            missing = {}
            for variant in VARIANTS:
                key = name, variant
                destination = output / variant / selected["video_path"]
                saved = latest.get(key)
                if saved and saved["status"] == "success" and key in checks and destination.exists():
                    if sha256(destination) == checks[key]["output_sha256"]:
                        continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                missing[variant] = destination.with_name(destination.name + ".partial")
            if not missing:
                continue
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(prefix="video-", dir=scratch) as working:
                    original = Path(working) / "source.mp4"
                    original_sha = extract_video(archive, selected, original, source_checks[name])
                    records, integrity, failures = transcode(original, missing, selected, VARIANTS)
            except Exception as exc:
                records, integrity = {}, {}
                failures = {variant: f"{type(exc).__name__}: {exc}" for variant in missing}
            for variant, temporary in missing.items():
                key = name, variant
                event = dict(sample_id=name, variant=variant, run_id=config["run_id"], updated_at=now())
                try:
                    if variant in failures:
                        raise ValueError(failures[variant])
                    record = records[variant]
                    record.update(processing_version=TRANSFORM["version"],
                                  processing_seconds=round(time.monotonic() - started, 4))
                    require_readable(record)
                    entry = dict(sample_id=name, variant=variant, run_id=config["run_id"],
                                 source_sha256=original_sha, zip_crc32=source_checks[name],
                                 output_sha256=sha256(temporary), **integrity[variant])
                    destination = output / variant / record["video_path"]
                    with temporary.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination)
                    sync_directory(destination.parent)
                    # Checks commit before the readable success journal. An orphan
                    # file without both committed entries is rebuilt on resume.
                    append_journal(check_log, entry)
                    checks[key] = entry
                    event.update(status="success", record=record)
                except Exception as exc:
                    temporary.unlink(missing_ok=True)
                    event.update(status="failure", error=f"{type(exc).__name__}: {exc}")
                    print(f"FAIL part{part} {name} {variant}: {exc}", flush=True)
                append_journal(log, event)
                latest[key] = event
            publish("running")
    publish("finished")


def reconcile(output, config, *, worker_errors=None):
    output = Path(output)
    selected = read_jsonl(output / "metadata/selected.jsonl")
    valid_keys = {(r["sample_id"], v) for r in selected for v in VARIANTS}
    latest, checks = {}, {}
    for part in sorted({r["source"]["part"] for r in selected}):
        for destination, path in (
            (latest, output / f"_state/part{part}.jsonl"),
            (checks, output / f"metadata/integrity_parts/part{part}.jsonl"),
        ):
            for key, event in journal_latest(path, per_variant=True).items():
                if key in destination or key not in valid_keys or event["run_id"] != config["run_id"]:
                    raise ValueError("Journal identity conflict")
                destination[key] = event
    public = {v: [] for v in VARIANTS}
    paired, failures, integrity = [], [], []
    for row in selected:
        records = {}
        for variant in VARIANTS:
            key = row["sample_id"], variant
            event = latest.get(key)
            if event is None:
                continue
            if event["status"] != "success":
                failures.append(event)
                continue
            record = event["record"]
            if record["variant"] != variant or any(record[k] != value for k, value in VARIANTS[variant].items()):
                raise ValueError("Output variant or dimensions changed")
            if any(record[k] != row[k] for k in ("sample_id", "video", "video_path", "caption")):
                raise ValueError("Output identity or caption changed")
            if any(record["source"][k] != value for k, value in row["source"].items()):
                raise ValueError("Original source metadata changed")
            path = output / variant / record["video_path"]
            if not path.exists() or path.stat().st_size != record["output_bytes"] or key not in checks:
                raise ValueError(f"Committed output missing/truncated or missing integrity: {path}")
            require_readable(record)
            records[variant] = record
            integrity.append(checks[key])
        if len(records) == len(VARIANTS):
            reference = next(iter(records.values()))
            for variant, record in records.items():
                if any(record[k] != reference[k] for k in ("source", "num_frames", "fps_num", "fps_den", "caption")):
                    raise ValueError("Variant temporal/source alignment mismatch")
                public[variant].append(record)
            paired.append(dict(sample_id=row["sample_id"], caption=row["caption"],
                               variants={v: f'{v}/{records[v]["video_path"]}' for v in VARIANTS}))
    for variant, records in public.items():
        atomic_jsonl(output / variant / "samples.jsonl", records)
    require_readable(paired)
    require_readable(failures)
    atomic_jsonl(output / "pairs.jsonl", paired)
    atomic_jsonl(output / "failures.jsonl", failures)
    atomic_jsonl(output / "metadata/integrity.jsonl", integrity)
    counts = event_counts(selected, latest)
    summary = dict(updated_at=now(), candidates=len(selected), **counts,
                   processing_finished=all(v["pending"] == 0 for v in counts["variants"].values()),
                   all_candidates_succeeded=counts["success"] == len(selected),
                   output_frames=sum(v["output_frames"] for v in counts["variants"].values()),
                   worker_errors=worker_errors or [], run_id=config["run_id"])
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "commit_validation.json", dict(
        verification_mode="full output decode before every successful variant commit",
        source_correspondence=False, verified=counts["committed_video_files"],
        all_passed=summary["all_candidates_succeeded"], run_id=config["run_id"], updated_at=now()))
    print(json.dumps(summary), flush=True)
    return summary


def check_pilot(path, config):
    path = Path(path)
    pilot = load_config(path)
    validation = json.loads((path / "validation.json").read_text())
    if not (pilot["pilot"] and pilot["candidates"] == 10 and validation["all_passed"]
            and validation["source_correspondence"] and validation["verified"] == 20
            and validation["run_id"] == pilot["run_id"]):
        raise ValueError("Ten-source, twenty-output pilot has not passed its independent audit")
    for name, expected in pilot["_integrity"]["validation_files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Pilot audit input/report changed: {name}")
    for key in ("transform", "runtime", "archive_identities"):
        if config[key] != pilot[key]:
            raise ValueError(f"Pilot/full configuration mismatch: {key}")
    for key in ("code_sha256", "metrics_sha256", "inventory_sha256"):
        if config["_integrity"][key] != pilot["_integrity"][key]:
            raise ValueError(f"Pilot/full integrity mismatch: {key}")
    for name in ("OpenVidHD.csv", "OpenVidHD.json"):
        if config["_integrity"]["metadata_sha256"][name] != pilot["_integrity"]["metadata_sha256"][name]:
            raise ValueError("Pilot/full original annotation mismatch")
    entries = {(r["sample_id"], r["variant"]): r for r in read_jsonl(path / "metadata/integrity.jsonl")}
    for variant in VARIANTS:
        for record in read_jsonl(path / variant / "samples.jsonl"):
            if sha256(path / variant / record["video_path"]) != entries[record["sample_id"], variant]["output_sha256"]:
                raise ValueError("Pilot MP4 changed since audit")


def process(output, workers, min_free_gib, pilot_root):
    output = Path(output)
    config = load_config(output)
    if not config["pilot"]:
        if pilot_root is None:
            raise ValueError("Full processing requires a passed ten-source pilot")
        check_pilot(pilot_root, config)
    groups = defaultdict(list)
    for row in read_jsonl(output / "metadata/selected.jsonl"):
        groups[row["source"]["part"]].append(row)
    source_checks = {r["sample_id"]: r["zip_crc32"] for r in read_jsonl(output / "metadata/source_integrity.jsonl")}
    started, errors = time.monotonic(), []
    atomic_json(output / "run.json", dict(state="running", started_at=now(), pid=os.getpid(),
                workers=workers, min_free_gib=min_free_gib, pilot_root=pilot_root, run_id=config["run_id"]))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(part_work, str(output), part, rows, config,
                    {r["sample_id"]: source_checks[r["sample_id"]] for r in rows}, min_free_gib): part
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
            statuses = [json.loads(path.read_text()) for path in (output / "_state").glob("*.status.json")]
            success = sum(s["success"] for s in statuses)
            failed = sum(s["failure"] for s in statuses)
            variants = {}
            for variant in VARIANTS:
                good = sum(s["variants"][variant]["success"] for s in statuses)
                bad = sum(s["variants"][variant]["failure"] for s in statuses)
                variants[variant] = dict(success=good, failure=bad, pending=config["candidates"]-good-bad,
                    output_bytes=sum(s["variants"][variant]["output_bytes"] for s in statuses))
            progress = dict(updated_at=now(), candidates=config["candidates"], success=success,
                            failure=failed, pending=config["candidates"]-success-failed, variants=variants,
                            active_or_queued_parts=len(futures), elapsed_seconds=round(time.monotonic()-started, 2),
                            output_bytes=sum(s["output_bytes"] for s in statuses), worker_errors=errors)
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    summary = reconcile(output, config, worker_errors=errors)
    atomic_json(output / "run.json", dict(state="succeeded" if summary["all_candidates_succeeded"] else "incomplete",
                finished_at=now(), elapsed_seconds=round(time.monotonic()-started, 2),
                workers=workers, run_id=config["run_id"]))
    return 0 if summary["all_candidates_succeeded"] else 2


def check_q2(path):
    import re
    import subprocess
    result = subprocess.run(["ffmpeg", "-hide_banner", "-debug", "qp", "-threads", "1",
        "-i", str(path), "-frames:v", "1", "-f", "null", "-"], check=True, capture_output=True, text=True)
    quantizers = []
    for line in result.stderr.splitlines():
        match = re.fullmatch(r"\[mpeg4 @[^\]]+\]\s+((?:\d+\s+){2,}\d+)\s*", line)
        if match:
            quantizers.extend(int(value) for value in match[1].split())
    if not quantizers or set(quantizers) != {2}:
        raise ValueError("First-frame MPEG-4 macroblock quantizers differ from q=2")


def verify(output, source_correspondence):
    import torch
    from .video import audit_source_correspondence, verify_output
    torch.set_num_threads(1)
    output = Path(output)
    config = load_config(output)
    summary = reconcile(output, config)
    with (output / "metadata/OpenVidHD.csv").open(newline="", encoding="utf-8") as handle:
        annotations = {r["video"]: r for r in csv.DictReader(handle)}
    integrity = {(r["sample_id"], r["variant"]): r for r in read_jsonl(output / "metadata/integrity.jsonl")}
    grouped = defaultdict(dict)
    for variant in VARIANTS:
        records = read_jsonl(output / variant / "samples.jsonl")
        require_readable(records)
        for record in records:
            grouped[record["sample_id"]][variant] = record
    results = []
    for name, records in grouped.items():
        try:
            if set(records) != set(VARIANTS):
                raise ValueError("Incomplete variant pair")
            checks = {}
            for variant, record in records.items():
                path = output / variant / record["video_path"]
                entry = integrity[name, variant]
                if record["caption"] != annotations[name]["caption"] or record["source"]["annotation"] != annotations[name]:
                    raise ValueError("Caption/original annotation fields changed")
                if sha256(path) != entry["output_sha256"]:
                    raise ValueError("Output file hash mismatch")
                checks[variant], _ = verify_output(path, record, entry["decoded_rgb_sha256"])
                if config["pilot"]:
                    check_q2(path)
                    checks[variant]["first_frame_q2_passed"] = True
            if source_correspondence:
                first = next(iter(records.values()))
                entries = {variant: integrity[name, variant] for variant in VARIANTS}
                original_entry = next(iter(entries.values()))
                with SplitFile(first["source"]["archive_files"]) as joined, zipfile.ZipFile(joined) as archive:
                    with tempfile.TemporaryDirectory(dir=output / "_scratch") as directory:
                        original = Path(directory) / "source.mp4"
                        digest = extract_video(archive, first, original, original_entry["zip_crc32"])
                        if any(e["source_sha256"] != digest for e in entries.values()):
                            raise ValueError("Original file hash mismatch")
                        audits = audit_source_correspondence(original, output, records, entries)
                        for variant in VARIANTS:
                            checks[variant].update(audits[variant])
            for variant in VARIANTS:
                results.append(dict(sample_id=name, variant=variant, passed=True, **checks[variant]))
        except Exception as exc:
            for variant in records:
                results.append(dict(sample_id=name, variant=variant, passed=False,
                                    error=f"{type(exc).__name__}: {exc}"))
    readable_files = ["processing_config.json", "metadata/selected.jsonl", "pairs.jsonl",
                      "summary.json", "failures.jsonl", *[f"{v}/samples.jsonl" for v in VARIANTS]]
    for name in readable_files:
        data = read_jsonl(output / name) if name.endswith(".jsonl") else json.loads((output / name).read_text())
        require_readable(data)
    if config["pilot"]:
        from data.dataset import read_records
        for variant in VARIANTS:
            loaded = read_records({"path": str(output / variant / "samples.jsonl"), "task": "t2v"})
            if len(loaded) != config["candidates"] or not all(Path(r["videos"][0]["path"]).is_file() for r in loaded):
                raise ValueError("UMM manifest paths did not resolve")
    report = dict(updated_at=now(), run_id=config["run_id"], source_correspondence=source_correspondence,
                  source_samples=len(grouped), verified=len(results), readable_metadata_passed=True,
                  all_passed=summary["all_candidates_succeeded"] and len(results) == 2*config["candidates"]
                    and all(r["passed"] for r in results), samples=results)
    require_readable(report)
    atomic_json(output / "validation.json", report)
    config["_integrity"]["validation_files"] = {name: sha256(output / name) for name in
        ["validation.json", "metadata/integrity.jsonl", "pairs.jsonl", *[f"{v}/samples.jsonl" for v in VARIANTS]]}
    atomic_json(output / "metadata/run_integrity.json", config["_integrity"])
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
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
