"""Full-population OpenVid statistics; no sampling or training-code changes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scan_videos import DEFAULT_RESULTS, DEFAULT_ROOT, run as scan, write_json

DEFAULT_TOKENIZER = Path("/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Qwen3.5-2B/tokenizer.json")
WORD_PATTERN = r"\b\w+(?:['’-]\w+)*\b"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty or non-finite metric")
    names = ["min", "p01", "p05", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .9, .95, .99, 1])
    return {"count": len(values), "mean": float(values.mean()),
            **dict(zip(names, map(float, quantiles)))}


def frequencies(values, name, path):
    counts = pd.Series(values).value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    counts["percent"] = counts["count"] * 100 / counts["count"].sum()
    counts.to_csv(path, index=False)
    return counts


def metadata(root, results, tokenizer_path, source=None):
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    from tokenizers import Tokenizer

    source = source or root / "data/train/OpenVid-1M.csv"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    words = re.compile(WORD_PATTERN)
    results.mkdir(parents=True, exist_ok=True)
    destination = results / "caption_and_csv_metrics.csv"
    temporary = destination.with_suffix(".csv.partial")
    fields = ["video", "csv_fps", "csv_frame_count", "csv_duration_seconds",
              "caption_characters", "caption_words", "caption_tokens"]
    total = 0
    started = time.monotonic()
    with source.open(newline="") as handle, temporary.open("w", newline="") as output:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        batch = []

        def consume(batch):
            nonlocal total
            encoded = tokenizer.encode_batch([row["caption"] for row in batch], add_special_tokens=False)
            for row, encoding in zip(batch, encoded):
                frame_count = float(row["frame"])
                fps, duration = float(row["fps"]), float(row["seconds"])
                if not frame_count.is_integer() or frame_count <= 0 or fps <= 0 or duration <= 0:
                    raise ValueError(f"Invalid numeric metadata: {row['video']}")
                writer.writerow(dict(video=row["video"], csv_fps=fps, csv_frame_count=int(frame_count),
                                     csv_duration_seconds=duration, caption_characters=len(row["caption"]),
                                     caption_words=len(words.findall(row["caption"])),
                                     caption_tokens=len(encoding.ids)))
                total += 1

        for row in reader:
            batch.append(row)
            if len(batch) == 1024:
                consume(batch)
                batch = []
                if total % 16384 == 0:
                    progress = dict(rows=total, elapsed_seconds=round(time.monotonic() - started, 2))
                    write_json(results / "caption_progress.json", progress)
                    print(json.dumps(progress), flush=True)
        if batch:
            consume(batch)
    temporary.replace(destination)
    df = pd.read_csv(destination)
    if df.video.duplicated().any():
        raise ValueError("Duplicate video names in source CSV")
    summary = {"rows": total, "source": str(source), "source_sha256": sha256(source),
               "dataset_name": source.stem,
               "tokenizer": str(tokenizer_path), "tokenizer_sha256": sha256(tokenizer_path),
               "word_regex": WORD_PATTERN,
               "token_definition": "Qwen3.5-2B tokenizer.json; add_special_tokens=False; no chat/video tokens",
               "character_definition": "Python len(caption), Unicode characters including whitespace/punctuation",
               "statistics": {col: describe(df[col]) for col in fields[1:]},
               "elapsed_seconds": time.monotonic() - started}
    frequencies(df.csv_fps, "csv_fps", results / "csv_fps_distribution.csv")
    write_json(results / "metadata_summary.json", summary)
    print(json.dumps({"metadata_complete": total, "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


def aggregate(results, allow_missing_metadata=False):
    source = pd.read_csv(results / "caption_and_csv_metrics.csv")
    manifest = json.loads((results / "archive_manifest.json").read_text())
    rows = []
    for index, _ in manifest["archives"]:
        status = json.loads((results / "video_shards" / f"part_{index:03d}.status.json").read_text())
        if not status["complete"]:
            raise ValueError(f"Incomplete archive {index}")
        shard = results / "video_shards" / f"part_{index:03d}.jsonl"
        with shard.open() as handle:
            rows.extend(json.loads(line) for line in handle)
    video = pd.DataFrame(rows)
    del rows
    errors = video[video.status != "ok"]
    errors.to_csv(results / "video_errors.csv", index=False)
    if len(errors):
        raise ValueError(f"{len(errors)} video metadata errors; see video_errors.csv. No complete charts generated.")
    if video.video.duplicated().any():
        raise ValueError("Duplicate archive video names")
    merged = source.merge(video, on="video", how="outer", validate="one_to_one", indicator=True)
    missing = merged[merged._merge != "both"]
    missing.to_csv(results / "membership_errors.csv", index=False)
    if len(missing) and (not allow_missing_metadata or (missing._merge == "left_only").any()):
        raise ValueError("CSV/archive membership mismatch")
    merged["metadata_present"] = merged._merge == "both"
    merged = merged.drop(columns="_merge")
    # Recompute from the raw measured table fields for every row. This also
    # handles the rare files whose mdhd declaration disagrees with stts.
    merged["media_duration_seconds"] = merged.track_duration_ticks / merged.timescale
    merged["duration_seconds"] = merged.sample_duration_ticks / merged.timescale
    merged["fps"] = merged.frame_count * merged.timescale / merged.sample_duration_ticks
    if "encoded_sample_count" not in merged:
        merged["encoded_sample_count"] = merged.frame_count
    else:
        # Earlier successful rows had already checked stsz == stts.
        merged["encoded_sample_count"] = merged.encoded_sample_count.fillna(merged.frame_count).astype("int64")
    for col in ["width", "height", "frame_count", "timescale"]:
        merged[col] = merged[col].astype("int64")
    merged["resolution"] = merged.width.astype(str) + "x" + merged.height.astype(str)
    merged["short_side"] = merged[["width", "height"]].min(axis=1)
    merged.to_csv(results / "all_video_metrics.csv", index=False)
    merged[merged.timing_class == "invalid_timing"].to_csv(results / "timing_anomalies.csv", index=False)
    n = len(merged)
    known = merged.metadata_present
    resolution = frequencies(merged.resolution, "resolution", results / "resolution_distribution.csv")
    frequencies(merged.height, "height", results / "height_distribution.csv")
    frequencies(merged.short_side, "short_side", results / "short_side_distribution.csv")
    frequencies(merged.fps, "fps", results / "fps_distribution.csv")
    frequencies(merged.frame_count, "frame_count", results / "frame_count_distribution.csv")
    timing = frequencies(merged.timing_class, "timing_class", results / "timing_distribution.csv")
    resolution_targets = {}
    for name, mask in {
        "height_480": merged.height == 480,
        "height_1080": merged.height == 1080,
        "exact_1920x1080": (merged.width == 1920) & (merged.height == 1080),
        "short_side_480": merged.short_side == 480,
        "short_side_1080": merged.short_side == 1080,
        "exact_512x512": (merged.width == 512) & (merged.height == 512),
    }.items():
        resolution_targets[name] = {"count": int(mask.sum()), "percent": float(mask.mean() * 100)}
    deltas = {
        "frame_count_mismatches": int((merged.loc[known, "frame_count"] != merged.loc[known, "csv_frame_count"]).sum()),
        "metadata_comparison_population": int(known.sum()),
        "fps_max_absolute_error": float((merged.fps - merged.csv_fps).abs().max()),
        "fps_error_over_0_001": int(((merged.fps - merged.csv_fps).abs() > .001).sum()),
        "duration_max_absolute_error_seconds": float((merged.duration_seconds - merged.csv_duration_seconds).abs().max()),
        "duration_error_over_0_001": int(((merged.duration_seconds - merged.csv_duration_seconds).abs() > .001).sum()),
        "media_duration_vs_stts_mismatches": int((merged.track_duration_ticks != merged.sample_duration_ticks).sum()),
        "rotated_videos": int(merged.rotated.sum()),
        "invalid_timing_tables": int((merged.timing_class == "invalid_timing").sum()),
        "stsz_stts_frame_count_disagreements": int((merged.encoded_sample_count != merged.frame_count).sum()),
    }
    metadata_summary = json.loads((results / "metadata_summary.json").read_text())
    metrics = ["fps", "frame_count", "encoded_sample_count", "duration_seconds", "caption_characters", "caption_words", "caption_tokens"]
    summary = dict(complete=True, population=n, archives=len(manifest["archives"]),
                   dataset_name=metadata_summary.get("dataset_name", "OpenVid-1M"),
                   video_root=manifest.get("root"),
                   source=metadata_summary["source"], source_sha256=metadata_summary["source_sha256"],
                   parser_sha256=manifest["parser_sha256"], tokenizer=metadata_summary["tokenizer"],
                   repair_parser_sha256=manifest.get("repair_parser_sha256"),
                   tokenizer_sha256=metadata_summary["tokenizer_sha256"],
                   statistics={col: describe(merged[col].dropna()) for col in metrics},
                   caption_population=int(known.sum()), missing_caption_videos=int((~known).sum()),
                   total_duration_hours=float(merged.duration_seconds.sum() / 3600),
                   unique_fps=int(merged.fps.nunique()), unique_resolutions=len(resolution),
                   timing=timing.to_dict("records"), resolution_targets=resolution_targets,
                   consistency= deltas,
                   definitions={
                       "fps": "Number of video samples divided by the complete stts sample-duration sum",
                       "frame_count": "Video frame count declared by stts, reconciled to the dataset CSV; stsz disagreements retained separately",
                       "encoded_sample_count": "Number of encoded samples declared by stsz; differs from stts/CSV in the recorded anomalies",
                       "duration_seconds": "Complete MP4 stts sample-duration sum / timescale; not first-to-last frame span",
                       "media_duration_seconds": "Separately retained MP4 mdhd declared media duration / timescale",
                       "resolution": "Coded sample-entry width x height; rotation reported separately",
                       "constant": "All consecutive presentation timestamps have exactly equal positive intervals",
                       "quantized_uniform": "Unequal intervals, but all PTS within one media tick of endpoint-fitted uniform grid",
                       "variable": "PTS deviate from the uniform grid by more than one media tick",
                       "invalid_timing": "Inconsistent composition/decode timing tables; retained in all statistics, not labeled CFR or VFR",
                       "caption_words": WORD_PATTERN,
                       "caption_characters": metadata_summary["character_definition"],
                       "caption_tokens": metadata_summary["token_definition"],
                       "480p_and_1080p": "Report height=480/1080, exact 1920x1080, and short-side=480/1080 separately",
                   },
                   limitations=["Full census of metadata and complete presentation timing tables; no all-frame RGB decode audit",
                                "Full population of the selected archive directory; no sampling or excluded videos",
                                "Caption statistics use available annotations only; missing annotations are retained and counted separately"])
    write_json(results / "summary.json", summary)
    pd.DataFrame(summary["statistics"]).T.to_csv(results / "numeric_summary.csv", index_label="metric")
    print(json.dumps({"complete": True, "rows": n, "resolution_targets": resolution_targets,
                      "consistency": deltas}, ensure_ascii=False), flush=True)
    return merged, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "metadata", "scan", "aggregate", "plots"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--metadata-csv", type=Path, help="Explicit caption/metadata CSV for the selected video directory")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "metadata"}:
        metadata(args.root, args.results, args.tokenizer, args.metadata_csv)
    if args.stage in {"all", "scan"}:
        scan(args.root, args.results, args.workers)
    if args.stage in {"all", "aggregate"}:
        aggregate(args.results)
    if args.stage in {"all", "plots"}:
        from plot_statistics import make_plots
        make_plots(args.results)


if __name__ == "__main__":
    main()
