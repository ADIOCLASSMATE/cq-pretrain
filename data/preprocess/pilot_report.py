"""Summarize the independently audited ten-source / twenty-output pilot."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path

from .manifest import atomic_json, read_jsonl, require_readable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    validation = json.loads((root / "validation.json").read_text())
    config = json.loads((root / "processing_config.json").read_text())
    if not (config["pilot"] and validation["all_passed"] and validation["source_correspondence"]
            and validation["verified"] == 20 and validation["source_samples"] == 10):
        raise ValueError("A complete, independently audited dual-resolution pilot is required")
    audits = {(r["sample_id"], r["variant"]): r for r in validation["samples"]}
    rows, summaries = [], {}
    for variant in config["transform"]["variants"]:
        records = read_jsonl(root / variant / "samples.jsonl")
        for record in records:
            source = record["source"]
            audit = audits[record["sample_id"], variant]
            rows.append(dict(sample_id=record["sample_id"], variant=variant, part=source["part"],
                reason=record["pilot_reason"], source_frames=source["num_frames"],
                source_fps_exact=str(Fraction(source["fps_num"], source["fps_den"])),
                output_frames=record["num_frames"], output_fps_exact=str(Fraction(record["fps_num"], record["fps_den"])),
                width=record["width"], height=record["height"], output_bytes=record["output_bytes"],
                source_bytes=source["bytes"], rgb_sse=audit["rgb_sse"], rgb_sae=audit["rgb_sae"],
                rgb_values=audit["rgb_values"], psnr_rgb_db=audit["psnr_rgb_db"], mae_rgb=audit["mae_rgb"],
                caption_unchanged=True, source_correspondence_passed=True))
        selected = [r for r in rows if r["variant"] == variant]
        frames = sum(r["output_frames"] for r in selected)
        size = sum(r["output_bytes"] for r in selected)
        values = sum(r["rgb_values"] for r in selected)
        sse = sum(r["rgb_sse"] for r in selected)
        summaries[variant] = dict(samples=len(selected), frames=frames, output_bytes=size,
            output_MiB=size/2**20, psnr_rgb_db=10*math.log10(255**2*values/sse) if sse else None,
            mae_rgb=sum(r["rgb_sae"] for r in selected)/values,
            illustrative_full_bytes=config["full_output_frames"]*size/frames)
    with (root / "pilot_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(source_samples=10, output_videos=20, all_passed=True, variants=summaries,
        total_bytes=sum(v["output_bytes"] for v in summaries.values()),
        illustrative_full_bytes=sum(v["illustrative_full_bytes"] for v in summaries.values()),
        estimation_limit="Purposive boundary sample, not random; full-corpus storage projection is illustrative.",
        quality_reference="Each variant's own resized and center-cropped pre-encoder RGB; RGB PSNR/MAE, all frames.",
        run_id=config["run_id"])
    require_readable(summary)
    atomic_json(root / "pilot_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
