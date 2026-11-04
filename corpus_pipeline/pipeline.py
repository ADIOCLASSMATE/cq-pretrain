"""Streaming CLI, immutable configuration, atomic per-unit outputs and resume.

Index files contain references, provenance and ordered text hashes by default,
not copied images. Only committed unit files belong to a run. Reprocessing a
failed unit is idempotent; a changed input/configuration requires a new run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fnmatch
import fcntl
import gzip
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time

from . import __version__
from .adapters import discover_dcvlm, discover_genund, iter_records
from .policy import normalize, evaluate, resolve_source_config, eligible_task_views, CATEGORIES, T2I_CATEGORIES

BASE = Path(__file__).resolve().parent
DEFAULT_DCVLM = "/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z"
DEFAULT_GENUND = "/inspire/dataset/gen-und-mm-pretrain/01"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def run_lock(output):
    with (output / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Run directory already has a writer; use a separate worker output directory") from exc
        yield


def open_state(output):
    db = sqlite3.connect(output / "state.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS units (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL, receipt TEXT NOT NULL)")
    return db


def compact_index(record, include_text=False):
    result = {key: value for key, value in record.items()
              if key not in ("source_config", "metadata", "segments")}
    result["segments"] = []
    for segment in record.get("segments", []):
        span = dict(segment)
        if span.get("kind") == "text":
            text = span.pop("text", "")
            span["text_chars"] = len(text)
            span["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            if include_text:
                span["text"] = text
        result["segments"].append(span)
    result["text_storage"] = "inline" if include_text else "original_record_locator"
    return result


def source_rules(rules, collection, source):
    # Avoid scanning the whole registry for each of hundreds of millions of rows.
    matches = []
    for rule in rules.get("sources", []):
        match = rule.get("match", {})
        c = match.get("collection", match.get("corpus"))
        s = match.get("source", match.get("source_id"))
        if c and c.casefold() != collection.casefold():
            continue
        if s and s.casefold() != source.casefold():
            continue
        if match.get("source_glob") and not fnmatch.fnmatchcase(source.casefold(), match["source_glob"].casefold()):
            continue
        matches.append(rule)
    return {**rules, "sources": matches}


def configure_adapter(unit, registry):
    unit = {**unit, "config": dict(unit.get("config", {}))}
    matching = resolve_source_config(unit["collection"], unit["source"], {"path": unit["path"]}, registry=registry)
    for entry in matching.get("_matched_rules", []):
        config = dict(entry.get("config", {}))
        if "path_prefix_map" in config:
            context = {key: unit["config"].get(key, "") for key in ("collection_root", "source_root", "source_pool")}
            mapping = {old: new.format_map(context) for old, new in config.pop("path_prefix_map").items()}
            unit["config"]["path_prefix_map"] = {**unit["config"].get("path_prefix_map", {}), **mapping}
        unit["config"].update(config)
    return unit


def process_unit(unit, output, rules, max_records, include_text):
    unit_id = str(unit["unit_id"])
    name = hashlib.sha256(unit_id.encode()).hexdigest()
    relative = f"index/{name[:2]}/{name}.jsonl.gz"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    counts, reasons, views, texts = Counter(), Counter(), Counter(), Counter()
    rows = 0
    start = time.monotonic()
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as stream:
                records = iter_records(unit, max_records=max_records)
                try:
                    for envelope in records:
                        config = resolve_source_config(unit["collection"], unit["source"], envelope.get("locator", {}), rules)
                        # Adapter roots are explicitly passed; no cross-partition path assumptions.
                        envelope.setdefault("source_pool", unit.get("config", {}).get("source_pool"))
                        record = evaluate(normalize(envelope, config), rules)
                        row = compact_index(record, include_text)
                        stream.write((dumps(row) + "\n").encode())
                        rows += 1
                        counts[(record["decision"], record["category"])] += 1
                        reasons.update(record["reason_codes"])
                        texts.update([record["provenance"]["text"]["origin"]])
                        if record["decision"] == "accepted":
                            views.update(record.get("eligible_views", []))
                finally:
                    close = getattr(records, "close", None)
                    if close:
                        close()
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dict(unit_id=unit_id, collection=unit["collection"], source=unit["source"],
                input_path=unit["path"], adapter=unit["adapter"], index_path=relative,
                index_sha256=file_digest(path), rows=rows,
                counts=[dict(decision=d, category=c, rows=n) for (d, c), n in sorted(counts.items())],
                reason_counts=dict(reasons), accepted_view_counts=dict(views), text_origin_counts=dict(texts),
                scan_scope="bounded_sample" if max_records is not None else "entire_unit",
                discovery=unit.get("discovery", unit.get("config", {}).get("discovery")),
                seconds=round(time.monotonic() - start, 3), completed_at=utcnow())


def summarize(output, db, run, exhausted=False):
    totals, sources, reasons, views, texts = Counter(), {}, Counter(), Counter(), Counter()
    successful = failed = records = 0
    manifest_temp = output / "manifest.jsonl.partial"
    with manifest_temp.open("w", encoding="utf-8") as manifest:
        for _, _, status, data in db.execute("SELECT * FROM units ORDER BY id"):
            receipt = json.loads(data)
            key = receipt["collection"] + "/" + receipt["source"]
            group = sources.setdefault(key, dict(collection=receipt["collection"], source=receipt["source"],
                                                records=0, units=0, failed_units=0, counts=Counter()))
            if status != "complete":
                failed += 1
                group["failed_units"] += 1
                continue
            manifest.write(dumps(receipt) + "\n")
            successful += 1
            records += receipt["rows"]
            group["units"] += 1
            group["records"] += receipt["rows"]
            for count in receipt["counts"]:
                k = count["decision"] + "/" + count["category"]
                group["counts"][k] += count["rows"]
                totals[k] += count["rows"]
            reasons.update(receipt["reason_counts"])
            views.update(receipt["accepted_view_counts"])
            texts.update(receipt["text_origin_counts"])
        manifest.flush()
        os.fsync(manifest.fileno())
    os.replace(manifest_temp, output / "manifest.jsonl")
    summary = dict(schema_version="corpus-run-summary.v1", run_id=run["run_id"], updated_at=utcnow(),
                   mode=run["mode"], scope="diagnostic_sample_not_population" if run["mode"] == "sample" else "worker_partition",
                   discovery_exhausted=exhausted, successful_units=successful, failed_units=failed,
                   indexed_records=records, counts=dict(totals), reason_counts=dict(reasons),
                   accepted_view_counts=dict(views), text_origin_counts=dict(texts),
                   sources=list(sources.values()),
                   limitations=["No pixel-level AI detector; unknown/mixed provenance is quarantined.",
                                "Source-declared naturalness is not per-image forensic certification.",
                                "Accepted means stage-1 eligible, not fully quality-validated training data.",
                                "Views can overlap; view counts must not be added as unique sample counts.",
                                "No cross-source image deduplication or train/test split decontamination has run."])
    atomic_json(output / "summary.json", summary)
    return summary


def run_command(args):
    output = Path(args.output).resolve()
    root = Path(args.root or (DEFAULT_DCVLM if args.collection == "dcvlm" else DEFAULT_GENUND)).resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside read-only dataset root")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root unavailable in this storage partition: {root}")
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("Invalid worker index/count")
    if min(args.sample_units, args.sample_rows, args.sample_files) < 1:
        raise ValueError("Sample limits must be positive")
    registry = json.loads(Path(args.registry).read_text())
    adapter_registry = json.loads(Path(args.adapters).read_text())
    rules = json.loads(Path(args.rules).read_text())
    if args.policy:
        rules["policy"] = args.policy
        rules["rejected_image_origins"] = (["ai_generated", "ai_edited", "programmatic", "rendered"]
                                            if args.policy == "natural_only" else ["ai_generated", "ai_edited"])
    rules["sources"] = registry["sources"]
    dependencies = {}
    for package in ("pyarrow", "ijson"):
        try:
            dependencies[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            dependencies[package] = None
    configuration = dict(version=__version__, python=platform.python_version(), dependencies=dependencies,
                         collection=args.collection, root=str(root), mode=args.mode,
                         sample_units=args.sample_units, sample_rows=args.sample_rows, sample_files=args.sample_files,
                         worker_index=args.worker_index, worker_count=args.worker_count, include_text=args.include_text,
                         rules=rules, adapters=adapter_registry,
                         code={p.name: file_digest(p) for p in sorted(BASE.glob("*.py")) if not p.name.startswith("test_")})
    if args.collection == "dcvlm":
        configuration["snapshot_files"] = {name: file_digest(root / name)
                                           for name in ("manifest.json", "catalog.json", "COMPLETE.json")}
    run = {**configuration, "run_id": digest(configuration)}
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        previous = output / "run.json"
        if previous.exists():
            if json.loads(previous.read_text())["run_id"] != run["run_id"]:
                raise ValueError("Run configuration or code changed: choose a new output directory")
        else:
            atomic_json(previous, {**run, "started_at": utcnow()})
        db = open_state(output)
        db.execute("CREATE TEMP TABLE seen_units (id TEXT PRIMARY KEY)")
        atomic_json(output / "COMPLETE.json", dict(run_id=run["run_id"], success=False,
                    full_corpus_processed=False, phase="running", started_at=utcnow()))
        selected = Counter()
        cached_rules = {}
        failures = 0
        seen = 0
        complete = False
        try:
            units = (discover_dcvlm(root) if args.collection == "dcvlm" else
                     discover_genund(root, max_files_per_source=args.sample_files if args.mode == "sample" else None))
            for unit in units:
                unit = configure_adapter(unit, adapter_registry)
                source = unit["source"]
                if args.mode == "sample" and selected[source] >= args.sample_units:
                    continue
                selected[source] += 1
                if int(hashlib.sha256(str(unit["unit_id"]).encode()).hexdigest(), 16) % args.worker_count != args.worker_index:
                    continue
                seen += 1
                db.execute("INSERT INTO seen_units VALUES (?)", (str(unit["unit_id"]),))
                fingerprint = digest({"input": unit.get("fingerprint", {}), "config": unit.get("config", {})})
                old = db.execute("SELECT fingerprint,status,receipt FROM units WHERE id=?", (str(unit["unit_id"]),)).fetchone()
                if old:
                    if old[0] != fingerprint:
                        raise RuntimeError(f"Input changed since checkpoint: {unit['path']}")
                    if old[1] == "complete":
                        receipt = json.loads(old[2])
                        if file_digest(output / receipt["index_path"]) != receipt["index_sha256"]:
                            raise RuntimeError("Committed index checksum mismatch: " + receipt["index_path"])
                        continue
                if source not in cached_rules:
                    cached_rules[source] = source_rules(rules, args.collection, source)
                try:
                    receipt = process_unit(unit, output, cached_rules[source],
                                           args.sample_rows if args.mode == "sample" else None, args.include_text)
                    status = "complete"
                except Exception as exc:
                    failures += 1
                    status = "failed"
                    receipt = dict(unit_id=unit["unit_id"], collection=unit["collection"], source=source,
                                   input_path=unit["path"], error_type=type(exc).__name__, error=str(exc), failed_at=utcnow())
                db.execute("INSERT OR REPLACE INTO units VALUES (?,?,?,?)", (str(unit["unit_id"]), fingerprint, status, dumps(receipt)))
                db.commit()
                print(dumps(dict(source=source, status=status, rows=receipt.get("rows", 0), units_seen=seen,
                                 error=receipt.get("error"))), flush=True)
                atomic_json(output / "progress.json", dict(updated_at=utcnow(), units_seen=seen, last_unit=receipt))
            removed = db.execute("SELECT id FROM units WHERE id NOT IN (SELECT id FROM seen_units) LIMIT 1").fetchone()
            if removed:
                raise RuntimeError("Previously indexed unit disappeared from discovery; input snapshot changed: " + removed[0])
            complete = True
        finally:
            summary = summarize(output, db, run, complete)
            atomic_json(output / "COMPLETE.json", dict(run_id=run["run_id"], discovery_exhausted=complete,
                        success=complete and summary["failed_units"] == 0, mode=args.mode,
                        full_corpus_processed=args.mode == "full" and args.worker_count == 1 and complete and summary["failed_units"] == 0,
                        phase="finished" if complete else "interrupted_or_failed", finished_at=utcnow()))
            db.close()
        print(dumps({k: summary[k] for k in ("indexed_records", "successful_units", "failed_units", "counts")}), flush=True)
        return 1 if summary["failed_units"] else 0


def export_command(args):
    root = Path(args.run).resolve()
    completion = json.loads((root / "COMPLETE.json").read_text())
    if not completion.get("success"):
        raise RuntimeError("Cannot export an incomplete or failed run; resolve errors and resume first")
    source_run = json.loads((root / "run.json").read_text())
    training_export = args.decision == "accepted"
    source_collection = str(source_run.get("collection", "")).casefold()
    if training_export and source_collection != "genund" and (
            args.view in T2I_CATEGORIES or args.category in T2I_CATEGORIES):
        raise ValueError("T2I training exports require a genund run; DCVLM retains native tasks. "
                         "Rejected/quarantine audit exports remain available.")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Export output exists; use a new filename")
    if output == root or root in output.parents:
        raise ValueError("Export outside immutable run directory")
    count = 0
    skipped_reasons = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".partial")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as destination:
            with (root / "manifest.jsonl").open() as manifest:
                for line in manifest:
                    unit = json.loads(line)
                    path = root / unit["index_path"]
                    if file_digest(path) != unit["index_sha256"]:
                        raise RuntimeError("Index checksum mismatch")
                    with gzip.open(path, "rt", encoding="utf-8") as source:
                        for line in source:
                            row = json.loads(line)
                            if row["decision"] != args.decision:
                                continue
                            if args.category and row["category"] != args.category:
                                continue
                            collection = str(row.get("collection", source_collection)).casefold()
                            selected_view = args.view or row["category"]
                            if training_export and selected_view in T2I_CATEGORIES and (
                                    source_collection != "genund" or collection != "genund"):
                                skipped_reasons["t2i_training_requires_genund"] += 1
                                continue
                            indexed_views = row.get("eligible_views", [])
                            if training_export:
                                views = eligible_task_views(collection, row["category"], indexed_views)
                                if source_collection != "genund":
                                    views = [view for view in views if view not in T2I_CATEGORIES]
                            else:
                                # Audits include native records even when their
                                # category is disallowed for training. Historical
                                # DCVLM reverse views are not native task evidence.
                                views = ([row["category"]] if collection == "dcvlm"
                                         else list(dict.fromkeys([row["category"], *indexed_views])))
                            if args.view and args.view not in views:
                                continue
                            if training_export and indexed_views != views:
                                row["indexed_eligible_views"] = indexed_views
                                row["eligible_views"] = views
                            row["selected_view"] = selected_view
                            row["export_purpose"] = "training_candidates" if training_export else "audit"
                            row["view_materialized"] = False
                            row["origin_run_id"] = source_run["run_id"]
                            destination.write(dumps(row) + "\n")
                            count += 1
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    atomic_json(Path(str(output) + ".manifest.json"), dict(run_id=source_run["run_id"],
                decision=args.decision, category=args.category, view=args.view, rows=count,
                sha256=file_digest(output), source_run=str(root), generated_at=utcnow(),
                source_mode=source_run["mode"], worker_count=source_run["worker_count"],
                partial_export=not completion.get("full_corpus_processed", False),
                skipped_reason_counts=dict(skipped_reasons),
                export_purpose="training_candidates" if training_export else "audit",
                view_materialized=False, instruction="Retrieve original record then construct selected_view; "
                "T2I training is restricted to genund; DCVLM native tasks and QA direction must be preserved."))
    print(dumps(dict(output=str(output), records=count)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Index immutable input references and apply provenance rules")
    run.add_argument("--collection", choices=("dcvlm", "genund"), required=True)
    run.add_argument("--root")
    run.add_argument("--output", required=True)
    run.add_argument("--mode", choices=("sample", "full"), default="sample")
    run.add_argument("--registry", default=str(BASE / "config/sources.json"))
    run.add_argument("--rules", default=str(BASE / "config/rules.json"))
    run.add_argument("--adapters", default=str(BASE / "config/adapters.json"))
    run.add_argument("--policy", choices=("natural_only", "ai_only"))
    run.add_argument("--sample-units", type=int, default=2, help="Units per source in sample mode")
    run.add_argument("--sample-rows", type=int, default=16, help="Rows per selected unit in sample mode")
    run.add_argument("--sample-files", type=int, default=256, help="Bounded gen-und discovery budget per source")
    run.add_argument("--worker-index", type=int, default=0)
    run.add_argument("--worker-count", type=int, default=1)
    run.add_argument("--include-text", action="store_true", help="Duplicate text payloads into index; default stores hashes and original locators")
    run.set_defaults(func=run_command)
    export = sub.add_parser("export", help="Materialize a filtered reference index without copying image bytes")
    export.add_argument("--run", required=True)
    export.add_argument("--output", required=True, help="New .jsonl.gz destination outside source run")
    export.add_argument("--decision", choices=("accepted", "rejected", "quarantine"), default="accepted")
    export.add_argument("--category", choices=CATEGORIES)
    export.add_argument("--view", choices=CATEGORIES, help="Use eligible task view instead of primary stored category")
    export.set_defaults(func=export_command)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
