"""Census the decoded frame counts recorded in both completed video manifests.

Run with data/analysis/.venv/bin/python data/analysis/analyze_processed_frames.py.
This reads every manifest row; it does not decode or modify the videos again.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(
    "/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected"
)
VARIANTS = ("960x544", "832x480")


def read_frames(path, variant):
    records = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            name = row["sample_id"]
            frames = row["num_frames"]
            source_frames = row["source"]["num_frames"]
            assert name not in records, f"Duplicate sample ID: {name}"
            assert row["variant"] == variant
            assert type(frames) is int and frames >= 81, name
            assert frames == row["validation"]["decoded_frames"], name
            assert source_frames == row["validation"]["source_decoded_frames"], name
            assert frames == (source_frames + 1) // 2, name
            records[name] = (frames, source_frames)
    return records


def percentile(sorted_values, percent):
    """Linear interpolation at (n - 1) * p, equivalent to NumPy's default."""
    position = (len(sorted_values) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
        position - lower
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results/OpenVidHD_processed",
    )
    args = parser.parse_args()
    root, output = args.dataset_root, args.output
    summary = json.loads((root / "summary.json").read_text())
    finished = json.loads((root / "_distributed/finished.json").read_text())
    assert summary["all_candidates_succeeded"] and finished["all_passed"]

    first = read_frames(root / VARIANTS[0] / "samples.jsonl", VARIANTS[0])
    second = read_frames(root / VARIANTS[1] / "samples.jsonl", VARIANTS[1])
    assert first == second, "Sample IDs or source/output frame counts differ"
    values = sorted(frames for frames, _ in first.values())
    count, total = len(values), sum(values)
    assert count == summary["candidates"] == finished["success"]
    for variant in VARIANTS:
        assert count == summary["variants"][variant]["success"]
        assert total == summary["variants"][variant]["output_frames"]
    assert total * len(VARIANTS) == summary["output_frames"]
    histogram = Counter(values)
    bands = [(81, 81), (82, 120), (121, 160), (161, 240),
             (241, 400), (401, 800), (801, None)]
    distribution = []
    for lo, hi in bands:
        n = sum(v for k, v in histogram.items() if k >= lo and (hi is None or k <= hi))
        distribution.append({
            "frames": str(lo) if hi == lo else f"{lo}-{hi}" if hi else f">={lo}",
            "videos": n, "percent": 100 * n / count,
        })
    assert sum(row["videos"] for row in distribution) == count
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "source_manifests": [str(root / v / "samples.jsonl") for v in VARIANTS],
        "scope": "Complete manifest census; one paired source video counted once",
        "measurement": "num_frames matched against full-decode validation.decoded_frames",
        "new_video_decode_performed": False,
        "dataset_finished_at": finished["finished_at"],
        "samples_per_variant": count,
        "frames_per_variant": total,
        "frames_both_variants": total * len(VARIANTS),
        "source_frames_before_stride2": sum(n for _, n in first.values()),
        "mean_frames": total / count,
        "min_frames": values[0], "max_frames": values[-1],
        "percentiles": {str(p): percentile(values, p) for p in (10, 25, 50, 75, 90, 95, 99)},
        "distinct_frame_counts": len(histogram),
        "exactly_81_frames": histogram[81],
        "at_least_81_frames": count,
        "distribution": distribution,
        "checks": {
            "unique_sample_ids": True,
            "all_ids_and_frame_counts_match_across_variants": True,
            "all_frame_counts_match_full_decode_records": True,
            "all_frame_counts_equal_ceil_source_frames_divided_by_2": True,
            "counts_and_totals_match_final_summary": True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "frame_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    with (output / "frame_histogram.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["num_frames", "videos_per_variant", "percent", "cumulative_percent"])
        cumulative = 0
        for frames, videos in sorted(histogram.items()):
            cumulative += videos
            writer.writerow([frames, videos, 100 * videos / count, 100 * cumulative / count])

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    heights = [row["percent"] for row in distribution]
    bars = ax.bar([row["frames"] for row in distribution], heights, color="#3477AE", width=0.7)
    ax.bar_label(bars, labels=[f"{r['percent']:.2f}%\n({r['videos']:,})" for r in distribution], padding=5, fontsize=10)
    ax.set_ylim(0, max(heights) * 1.25)
    ax.set_ylabel("Share of videos (%)")
    ax.set_xlabel("Decoded frames per processed video")
    ax.set_title(f"Processed OpenVidHD frame counts | {count:,} videos per resolution\n"
                 "960x544 and 832x480 have identical frame counts", pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "frame_distribution.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
