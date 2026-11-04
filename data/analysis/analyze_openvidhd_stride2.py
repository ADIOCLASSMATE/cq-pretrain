"""Count 81-frame clips sampled at stride 2, using the full OpenVidHD census."""
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path

from plan_openvidhd_wan480 import sha256


def main():
    results = Path(__file__).resolve().parent / "results/OpenVidHD"
    census = json.loads((results / "summary.json").read_text())
    assert census["complete"]
    assert census["video_root"] == "/inspire/dataset/openvid-1m/v2/OpenVidHD"
    captions = Path(census["source"])
    assert sha256(captions) == census["source_sha256"]
    with captions.open() as handle:
        names = {r["video"] for r in csv.DictReader(handle) if r["caption"].strip()}

    metrics = results / "all_video_metrics.csv"
    original, stride2 = Counter(), Counter()
    total = 0
    with metrics.open() as handle:
        for row in csv.DictReader(handle):
            total += 1
            if (int(row["width"]), int(row["height"])) != (1920, 1080):
                continue
            if row["video"] not in names:
                continue
            assert row["status"] == "ok" and row["timing_class"] == "constant"
            rate = Fraction(int(row["timescale"]), int(row["pts_interval_min_ticks"]))
            frames = int(row["frame_count"])
            if frames >= 81:
                original[rate] += 1
            if frames >= 2 * (81 - 1) + 1:
                stride2[rate] += 1
    assert total == census["population"]

    rows = []
    for rate in sorted(original, key=lambda r: (-stride2[r], r)):
        rows.append(dict(
            source_fps=float(rate), source_fps_exact=str(rate),
            output_fps=float(rate / 2), output_fps_exact=str(rate / 2),
            source_ge81_count=original[rate], source_ge161_count=stride2[rate],
            first_to_last_seconds=float(Fraction(160) / rate),
        ))
    path = results / "native_fps_stride2_81frames_counts.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        scope=census["video_root"], source_metrics_sha256=sha256(metrics),
        source_captions_sha256=census["source_sha256"],
        filters=dict(width=1920, height=1080, caption="nonempty after stripping whitespace",
                     source_min_frames=161, output_frames=81, stride=2),
        sample_indices="start + 2*j, j=0..80", count_unit="source video",
        source_ge81_total=sum(original.values()),
        source_ge161_total=sum(stride2.values()),
        source_29_97_and_30_ge161_count=stride2[Fraction("29.97")] + stride2[Fraction(30)],
        source_23_976_and_24_ge161_count=stride2[Fraction("23.976")] + stride2[Fraction(24)],
        top5=rows[:5],
        validation_scope="Complete metadata census and caption presence; no RGB decoding or clip-caption semantic validation",
    )
    (results / "native_fps_stride2_81frames_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
