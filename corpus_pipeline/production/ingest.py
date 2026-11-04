"""Snapshot-bound DCVLM ingestion with bounded Parquet batches.

This module does not download or extract images and never changes source files.
Catalog/shard manifests are metadata; payload iteration is lazy. Unsupported
collections need an explicit adapter rather than a guessed caption schema.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .common import atomic_json, digest, file_sha
from .records import DEFAULT_ROOT, inspect_asset, load_raw_record, load_record, snapshot

SCHEMA = "production.input.v2"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_shard_manifest(root=DEFAULT_ROOT, *, sources=None, max_shards=None, rows_per_shard=None):
    """Discover published shards; optional bounds are prefix diagnostics, not samples."""
    source_snapshot = snapshot(root)
    if max_shards is not None:
        _positive(max_shards, "max_shards")
    if rows_per_shard is not None:
        _positive(rows_per_shard, "rows_per_shard")
    selected = set(sources or [])
    known = {entry["source"] for entry in source_snapshot["catalog_index"].values()}
    if selected - known:
        raise ValueError("unknown requested DCVLM sources: " + ", ".join(sorted(selected - known)))
    shards = []
    for path, entry in source_snapshot["catalog_index"].items():
        if selected and entry["source"] not in selected:
            continue
        fingerprint = _fingerprint(path)
        if fingerprint["size"] != entry["bytes"]:
            raise ValueError("native Parquet published size mismatch")
        if source_snapshot['manifest']['schema_version'] == 'corpus.native.v2' and file_sha(path) != entry['sha256']:
            raise ValueError('native v2 Parquet published SHA256 mismatch')
        count = min(entry["rows"], rows_per_shard) if rows_per_shard else entry["rows"]
        shards.append({"parquet_path": path, "row_start": 0, "row_stop": count,
                       "source": entry["source"], "source_shard": entry["source_shard"],
                       "collection": entry.get('collection', 'dcvlm'), "parquet_fingerprint": fingerprint})
        if max_shards is not None and len(shards) >= max_shards:
            break
    collections = sorted({shard['collection'] for shard in shards})
    return {"schema_version": SCHEMA, "collection": collections[0] if len(collections) == 1 else 'mixed', "snapshot_root": source_snapshot["root"],
            "snapshot_fingerprint": source_snapshot["fingerprint"], "shards": shards,
            "scope": "bounded_catalog_prefix_not_population_estimate" if max_shards or rows_per_shard else "selected_published_catalog_shards",
            "source_filters": sorted(selected), "all_images_required": True, "min_short_edge": 128,
            "record_count": sum(shard["row_stop"] - shard["row_start"] for shard in shards)}


def iter_records(manifest, source_snapshot, rules, *, batch_size=128, max_records=None):
    """Yield complete records; shard mode keeps at most one Arrow batch in memory.

    The caller owns admission/QC and persistence. A parse failure propagates and
    must never be reported as a completed shard. Legacy explicit-row canaries
    remain supported, and their selected record order is retained.
    """
    _positive(batch_size, "batch_size")
    if max_records is not None:
        _positive(max_records, "max_records")
    v2 = source_snapshot['manifest']['schema_version'] == 'corpus.native.v2'
    if manifest.get("collection", "dcvlm") != "dcvlm" and not (v2 and manifest.get('collection') in ('genund', 'mixed')):
        raise ValueError("unsupported collection: requires explicit source adapter and metadata repair")
    if Path(manifest["snapshot_root"]).resolve() != Path(source_snapshot["root"]).resolve():
        raise ValueError("input snapshot root changed")
    if manifest["snapshot_fingerprint"] != source_snapshot["fingerprint"]:
        raise ValueError("input snapshot fingerprint changed")
    if "records" in manifest and "shards" in manifest:
        raise ValueError("input manifest cannot mix explicit records and shard ranges")
    emitted = 0
    if "records" in manifest:
        if manifest.get("schema_version") not in ("production.canary.v1", "production.canary.v2", SCHEMA):
            raise ValueError("unsupported explicit-record manifest schema")
        for spec in manifest["records"]:
            yield load_record(spec, source_snapshot, rules)
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return
        return
    if manifest.get("schema_version") != SCHEMA or "shards" not in manifest:
        raise ValueError("unsupported input manifest schema")
    import pyarrow.parquet as pq
    seen_ranges = {}
    for shard in manifest["shards"]:
        path = Path(shard["parquet_path"]).resolve()
        entry = source_snapshot["catalog_index"].get(str(path))
        if entry is None or entry["source"] != shard["source"] or entry["source_shard"] != shard["source_shard"]:
            raise ValueError("shard locator/source does not match published snapshot")
        if v2 and (entry['collection'] != shard.get('collection') or file_sha(path) != entry['sha256']):
            raise ValueError('native v2 shard collection/content changed')
        if _fingerprint(path) != shard["parquet_fingerprint"]:
            raise ValueError("native Parquet file changed after manifest creation")
        begin, stop = shard["row_start"], shard["row_stop"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (begin, stop)) or not 0 <= begin < stop <= entry["rows"]:
            raise ValueError("invalid shard row range")
        if any(begin < previous_stop and previous_begin < stop for previous_begin, previous_stop in seen_ranges.setdefault(str(path), [])):
            raise ValueError("overlapping shard ranges duplicate original records")
        seen_ranges[str(path)].append((begin, stop))
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != entry["rows"]:
            raise ValueError("native Parquet row count differs from catalog")
        group_start = 0
        for group in range(parquet.num_row_groups):
            group_stop = group_start + parquet.metadata.row_group(group).num_rows
            if group_stop <= begin:
                group_start = group_stop
                continue
            if group_start >= stop:
                break
            batch_start = group_start
            for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group]):
                low, high = max(begin, batch_start), min(stop, batch_start + batch.num_rows)
                if low < high:
                    for offset, raw in enumerate(batch.slice(low - batch_start, high - low).to_pylist()):
                        spec = {**shard, "parquet_row_index": low + offset,
                                "selection_reason": manifest.get("scope")}
                        yield load_raw_record(raw, spec, source_snapshot, rules, row_group=group)
                        emitted += 1
                        if max_records is not None and emitted >= max_records:
                            return
                batch_start += batch.num_rows
                if batch_start >= stop:
                    break
            group_start = group_stop
        if _fingerprint(path) != shard["parquet_fingerprint"]:
            raise ValueError("native Parquet changed during iteration; discard incomplete run")


def validate_manifest(manifest, rules, *, max_records=64, inspect_images=True):
    """Bounded real-data loader/byte/size verification; never a population estimate."""
    snap = snapshot(manifest["snapshot_root"])
    rows, assets, failures = [], {}, []
    for record in iter_records(manifest, snap, rules, max_records=max_records):
        for ref in record["source_image_refs"]:
            if ref["sha256"] in assets or not inspect_images:
                continue
            try:
                asset = inspect_asset(ref, record["source"], record["domain"])
                assets[ref["sha256"]] = {k: asset[k] for k in ("asset_id", "width", "height", "size_decision", "decode_status")}
            except Exception as error:
                failures.append({"sample_id": record["original_sample_id"], "asset_id": ref["sha256"],
                                 "error_type": type(error).__name__, "message": str(error)})
        rows.append({"sample_id": record["original_sample_id"], "source": record["source"],
                     "source_record_sha256": record["source_record_sha256"], "locator": record["locator"],
                     "category": record["normalized"]["category"], "format": record["raw"]["format"],
                     "image_count": len(record["source_image_refs"]), "caption_contract": record["caption_contract"],
                     "pairing_state": record["pairing_state"], "caption_fields_inspected": record["caption_fields_inspected"],
                     "prepared_adapter_status": record["prepared_adapter_status"]})
    return {"schema_version": "production.input_validation.v1", "input_sha256": digest(manifest),
            "scope": "bounded_selected_real_records_not_quality_or_population_estimate", "records": rows,
            "counts": {"records": len(rows), "sources": dict(Counter(row["source"] for row in rows)),
                       "categories": dict(Counter(row["category"] for row in rows)),
                       "image_occurrences": sum(row["image_count"] for row in rows),
                       "unique_images_inspected": len(assets), "image_failures": len(failures),
                       "size_rejections": sum(asset["size_decision"] != "pass" for asset in assets.values())},
            "assets": list(assets.values()), "failures": failures, "source_files_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover")
    discover.add_argument("--root", default=str(DEFAULT_ROOT))
    discover.add_argument("--source", action="append")
    discover.add_argument("--max-shards", type=int)
    discover.add_argument("--rows-per-shard", type=int)
    discover.add_argument("--output", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--max-records", type=int, default=64)
    validate.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.command == "discover":
        value = build_shard_manifest(args.root, sources=args.source, max_shards=args.max_shards, rows_per_shard=args.rows_per_shard)
    else:
        base = Path(__file__).resolve().parents[1]
        rules = json.loads((base / "config/rules.json").read_text())
        rules["sources"] = json.loads((base / "config/sources.json").read_text())["sources"]
        value = validate_manifest(json.loads(Path(args.manifest).read_text()), rules,
                                  max_records=args.max_records, inspect_images=not args.metadata_only)
    atomic_json(args.output, value)
    print(json.dumps({"output": str(Path(args.output).resolve()), "counts": value.get("counts"),
                      "record_count": value.get("record_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
