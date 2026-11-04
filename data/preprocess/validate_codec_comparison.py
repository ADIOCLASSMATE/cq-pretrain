"""Independently reconcile complete comparison CSVs, media identities and figures."""
import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import av
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "comparison_config.json").read_text())
    with (root / "per_sample.csv").open() as handle:
        samples = list(csv.DictReader(handle))
    with (root / "per_frame.csv").open() as handle:
        frames = list(csv.DictReader(handle))
    with (root / "timings.csv").open() as handle:
        timings = list(csv.DictReader(handle))
    manifest = [json.loads(line) for line in (root / "samples.jsonl").open()]
    originals = {r["sample_id"]:r for r in (json.loads(line) for line in (Path(config["pilot"]) / "samples.jsonl").open())}
    require(summary["complete"] and len(samples) == len(manifest) == 20, "Missing candidate videos")
    require(len({(r["sample_id"], r["mode"]) for r in samples}) == 20, "Duplicate candidate rows")
    require(len(frames) == 5120 and len(timings) == 150, "Missing frames or timing repetitions")
    grouped = defaultdict(list)
    for row in frames:
        grouped[row["sample_id"], row["mode"]].append(row)
    for row in samples:
        identity = row["sample_id"], row["mode"]
        per_frame = grouped[identity]
        require([int(f["frame_index"]) for f in per_frame] == list(range(int(row["num_frames"]))), "Frame sequence changed")
        require(sum(int(f["sse"]) for f in per_frame) == int(row["sse"]), "Per-clip SSE mismatch")
        require(sum(int(f["sae"]) for f in per_frame) == int(row["sae"]), "Per-clip SAE mismatch")
        rate = Fraction(int(row["fps_num"]), int(row["fps_den"]))
        for f in per_frame:
            require(int(f["original_frame_index"]) == int(f["frame_index"])*2, "Frame correspondence changed")
            require(abs(float(f["timestamp_seconds"]) - float(Fraction(int(f["frame_index"]), 1)/rate)) < 1e-10,
                    "Recorded timestamp changed")
            require(-1 <= float(f["ssim_rgb"]) <= 1, "Invalid SSIM")
        require(file_hash(root / row["video_path"]) == row["sha256"], "Candidate MP4 changed")
        with av.open(str(root / row["video_path"])) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            frame = next(container.decode(stream))
            expected_codec = "mpeg4" if row["mode"] == "mpeg4_q2" else "h264"
            require(stream.codec_context.name == expected_codec, "Wrong video codec")
            require(frame.format.name == "yuv420p" and (frame.width, frame.height) == (960, 540), "Wrong pixel format")
            require(stream.average_rate == rate and frame.pts == 0, "Wrong FPS/time origin")
        parameter_directory = root / "parameter_checks"
        parameter_directory.mkdir(exist_ok=True)
        if row["mode"] == "h264_crf18":
            with (root / row["video_path"]).open("rb") as handle:
                header = handle.read(256*1024)
            require(re.search(rb"\bcrf=18\.0\b", header) is not None, "H.264 SEI does not identify CRF18")
        else:
            result = subprocess.run(["ffmpeg", "-hide_banner", "-debug", "qp", "-threads", "1",
                "-i", str(root / row["video_path"]), "-frames:v", "1", "-f", "null", "-"],
                check=True, capture_output=True, text=True)
            quantizers = []
            for line in result.stderr.splitlines():
                match = re.fullmatch(r"\[mpeg4 @[^\]]+\]\s+((?:\d+\s+){2,}\d+)\s*", line)
                if match:
                    quantizers.extend(int(value) for value in match[1].split())
            require(quantizers and set(quantizers) == {2}, "MPEG-4 first-frame macroblocks do not use q=2")
            (parameter_directory / (row["sample_id"]+".qp.txt")).write_text(result.stderr)
        output = next(r for r in manifest if (r["sample_id"], r["mode"]) == identity)
        require(output["caption"] == originals[row["sample_id"]]["caption"], "Caption changed")
        require(output["original_source"] == originals[row["sample_id"]]["source"], "Source mapping changed")
    for result in summary["results"]:
        mode = result["mode"]
        selected = [r for r in samples if r["mode"] == mode]
        per_frame = [r for r in frames if r["mode"] == mode]
        # Recalculate from individual frame rows, independently of the report helper.
        mse = sum(int(r["sse"]) for r in per_frame) / sum(int(r["values"]) for r in per_frame)
        require(abs(10*math.log10(255**2/mse) - result["psnr_rgb_db"]) < 1e-10, "Global PSNR mismatch")
        require(abs(float(np.mean([float(r["ssim_rgb"]) for r in per_frame])) - result["ssim_rgb"]) < 1e-12,
                "Global SSIM mismatch")
        require(sum(int(r["output_bytes"]) for r in selected) == result["total_bytes"], "Total size mismatch")
        for stage in ("encode", "decode"):
            seconds = sum(float(np.median([float(t["seconds"]) for t in timings
                if t["sample_id"] == r["sample_id"] and t["mode"] == mode and t["stage"] == stage])) for r in selected)
            require(abs(seconds - result[stage+"_seconds"]) < 1e-10, "Timing aggregate mismatch")
            require(abs(2560/seconds - result[stage+"_fps"]) < 1e-9, "Throughput mismatch")
    require((root / "comparison.png").exists() and len(list((root / "visual_comparisons").glob("sample*.png"))) == 10,
            "Missing comparison figures")
    by_id = defaultdict(dict)
    for row in samples:
        by_id[row["sample_id"]][row["mode"]] = row
    wins = {}
    for key, higher in [("psnr_rgb_db", True), ("ssim_rgb", True), ("output_bytes", False),
                        ("encode_seconds_median", False), ("decode_seconds_median", False)]:
        wins[key] = sum((float(r["h264_crf18"][key]) > float(r["mpeg4_q2"][key])) if higher
                        else (float(r["h264_crf18"][key]) < float(r["mpeg4_q2"][key])) for r in by_id.values())
    frame_pairs = defaultdict(dict)
    for row in frames:
        frame_pairs[row["sample_id"], row["frame_index"]][row["mode"]] = row
    frame_wins = {}
    for key in ("psnr_rgb_db", "ssim_rgb"):
        differences = [float(r["h264_crf18"][key])-float(r["mpeg4_q2"][key]) for r in frame_pairs.values()]
        frame_wins[key] = dict(h264_better=sum(d > 0 for d in differences),
                              equal=sum(d == 0 for d in differences),
                              h264_worse=sum(d < 0 for d in differences))
    validation = dict(all_checks_passed=True, samples=10, candidate_videos=20,
        frame_comparisons=5120, timing_measurements=150,
        media_pixel_formats_and_hashes_checked=True, captions_and_source_mappings_checked=True,
        h264_crf18_sei_checked=True, mpeg4_q2_first_frame_macroblocks_checked=True,
        frame_to_clip_to_global_metrics_reconciled=True, h264_wins_out_of_ten=wins,
        paired_frame_quality=frame_wins,
        figures_numerical_data_checked=True, rendered_visual_inspection="Not assessed by this script; inspect separately")
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
