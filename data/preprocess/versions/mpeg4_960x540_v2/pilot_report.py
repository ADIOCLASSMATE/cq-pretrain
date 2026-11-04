"""Summarize a completed pilot without extrapolating a representative sample."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil

from .manifest import atomic_json, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    records = read_jsonl(root / "samples.jsonl")
    validation = json.loads((root / "validation.json").read_text())
    config = json.loads((root / "processing_config.json").read_text())
    if not (config["pilot"] and validation["all_passed"] and validation["source_correspondence"]
            and validation["verified"] == len(records) == 10):
        raise ValueError("A complete, independently verified ten-sample pilot is required")
    rows = []
    for record in records:
        source = record["source"]
        rows.append(dict(sample_id=record["sample_id"], part=source["part"],
            reason=record["pilot_reason"], source_frames=source["num_frames"],
            source_fps_exact=str(Fraction(source["fps_num"], source["fps_den"])),
            output_frames=record["num_frames"],
            output_fps_exact=str(Fraction(record["fps_num"], record["fps_den"])),
            width=record["width"], height=record["height"], output_bytes=record["output_bytes"],
            source_bytes=source["bytes"], processing_seconds=record["processing_seconds"],
            caption_unchanged=True, source_correspondence_passed=True))
    with (root / "pilot_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    frames = sum(r["output_frames"] for r in rows)
    size = sum(r["output_bytes"] for r in rows)
    summary = dict(pilot_samples=10, all_passed=True, output_frames=frames, output_bytes=size,
        bytes_per_output_frame=size/frames,
        illustrative_full_bytes=config["full_output_frames"]*size/frames,
        free_disk_bytes_at_report=shutil.disk_usage(root).free,
        estimation_limit="Purposive boundary/coverage sample, not random or representative; storage projection is illustrative only.",
        checks=["960x540", "0,2,4,... source correspondence for all output frames",
                "exact source FPS/2 and original relative timestamps", "ceil(source_frames/2)",
                "MPEG-4 Part 2 q=2, YUV420P, full output decode",
                "independently reconstructed pre-encoder RGB hash and measured lossy error",
                "original filename, part, all annotation fields and caption"],
        config_fingerprint=config["fingerprint"])
    atomic_json(root / "pilot_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
