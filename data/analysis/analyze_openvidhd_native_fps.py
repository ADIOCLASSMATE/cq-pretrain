"""Full OpenVidHD counts by native FPS for 1920x1080, >=81 frames, caption.

Run from UMM_v3:
    data/analysis/.venv/bin/python data/analysis/analyze_openvidhd_native_fps.py

Uses the complete MP4 census and original caption CSV. Does not decode, convert,
or resample videos, and does not modify training code or configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plan_openvidhd_wan480 import sha256
from plot_statistics import BLUE, GRAY, plt, save, style


BASE = Path(__file__).resolve().parent
STEM = "native_fps_1920x1080_ge81"


def analyze(results: Path):
    census = json.loads((results / "summary.json").read_text())
    assert census["complete"]
    assert census["video_root"] == "/inspire/dataset/openvid-1m/v2/OpenVidHD"
    source = Path(census["source"])
    metrics = results / "all_video_metrics.csv"
    assert sha256(source) == census["source_sha256"]

    captions = pd.read_csv(source, usecols=["video", "caption", "frame", "fps"],
                           keep_default_na=False)
    assert not captions.video.duplicated().any()
    nonempty = captions.caption.str.strip().ne("")
    caption_names = set(captions.loc[nonempty, "video"])
    df = pd.read_csv(metrics, usecols=[
        "video", "width", "height", "frame_count", "fps", "caption_characters",
        "timescale", "sample_duration_ticks", "pts_span_ticks",
        "pts_interval_min_ticks", "pts_interval_max_ticks", "status", "timing_class",
    ])
    assert not df.video.duplicated().any()
    assert len(df) == census["population"]
    assert set(captions.video).issubset(set(df.video))
    has_caption = df.video.isin(caption_names)
    assert has_caption.equals(df.caption_characters.fillna(0).gt(0))

    # Exact mean FPS from integer MP4 timing fields; no float rounding or buckets.
    # For every qualifying video, verify this also equals the constant PTS rate.
    timing_columns = ["frame_count", "timescale", "sample_duration_ticks"]
    unique = df[timing_columns].drop_duplicates()
    lookup = {(n, t, s): Fraction(int(n) * int(t), int(s))
              for n, t, s in unique.itertuples(index=False, name=None)}
    rates = [lookup[key] for key in df[timing_columns].itertuples(index=False, name=None)]
    assert np.allclose([float(r) for r in rates], df.fps, rtol=0, atol=1e-10)

    resolution = df.width.eq(1920) & df.height.eq(1080)
    enough_frames = df.frame_count.ge(81)
    base = resolution & has_caption
    eligible = base & enough_frames
    selected = df.loc[eligible]
    assert selected.status.eq("ok").all()
    assert selected.timing_class.eq("constant").all()
    assert selected.pts_interval_min_ticks.eq(selected.pts_interval_max_ticks).all()
    assert selected.sample_duration_ticks.eq(
        selected.frame_count * selected.pts_interval_min_ticks).all()
    assert selected.pts_span_ticks.eq(
        (selected.frame_count - 1) * selected.pts_interval_min_ticks).all()

    all_counts = Counter(rates)
    base_counts = Counter(r for r, keep in zip(rates, base) if keep)
    eligible_counts = Counter(r for r, keep in zip(rates, eligible) if keep)
    total = int(eligible.sum())

    # Independent reconciliation using annotation frame/FPS fields. Resolution
    # comes from the MP4 census, since the annotation CSV has no resolution field.
    csv_selected = captions.loc[
        nonempty & captions.video.isin(set(df.loc[resolution, "video"]))
        & pd.to_numeric(captions.frame).ge(81)]
    assert set(csv_selected.video) == set(selected.video)
    csv_counts = {Fraction(str(rate)): int(count)
                  for rate, count in csv_selected.fps.value_counts().items()}
    assert csv_counts == dict(eligible_counts)

    rows = []
    for rate in sorted(all_counts, key=lambda r: (-eligible_counts[r], r)):
        rows.append({
            "fps": format(float(rate), ".12g"),
            "fps_exact": str(rate),
            "fps_numerator": rate.numerator,
            "fps_denominator": rate.denominator,
            "all_openvidhd_count": all_counts[rate],
            "resolution_1920x1080_with_caption_count": base_counts[rate],
            "eligible_count": eligible_counts[rate],
            "excluded_under_81_frames_count": base_counts[rate] - eligible_counts[rate],
            "percent_of_eligible": 100 * eligible_counts[rate] / total,
            "percent_of_all_openvidhd": 100 * eligible_counts[rate] / len(df),
        })
    table = pd.DataFrame(rows)
    assert table.eligible_count.sum() == total
    assert table.all_openvidhd_count.sum() == len(df)
    table.to_csv(results / f"{STEM}_counts.csv", index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": census["video_root"],
        "source_metrics": str(metrics.resolve()),
        "source_metrics_sha256": sha256(metrics),
        "source_captions": str(source),
        "source_captions_sha256": census["source_sha256"],
        "filters": {"width": 1920, "height": 1080, "min_frames": 81,
                    "caption": "nonempty after stripping whitespace"},
        "count_unit": "one source video, not the number of 81-frame windows",
        "fps_definition": "frame_count * timescale / sample_duration_ticks; exact rational grouping",
        "funnel": [
            {"stage": "all_openvidhd", "count": len(df)},
            {"stage": "1920x1080", "count": int(resolution.sum())},
            {"stage": "1920x1080_ge81_frames", "count": int((resolution & enough_frames).sum())},
            {"stage": "1920x1080_ge81_frames_with_caption", "count": total},
        ],
        "eligible_count": total,
        "eligible_percent_of_all": 100 * total / len(df),
        "all_openvidhd_distinct_fps": len(all_counts),
        "eligible_distinct_fps": len(eligible_counts),
        "eligible_fps_min": float(min(eligible_counts)),
        "eligible_fps_max": float(max(eligible_counts)),
        "largest_group": rows[0],
        "top5_count": int(table.eligible_count.head(5).sum()),
        "other_eligible_fps_count": len(eligible_counts) - 5,
        "other_eligible_video_count": int(table.eligible_count.iloc[5:].sum()),
        "validation": {
            "caption_csv_rows": len(captions),
            "caption_csv_empty_or_whitespace_count": int((~nonempty).sum()),
            "unique_video_names": True,
            "all_eligible_strict_constant_pts_interval": True,
            "all_eligible_timing_matches_exact_fps": True,
            "annotation_frame_fps_crosscheck": "exact video-set and per-rate count match",
        },
        "limitations": [
            "Reaggregation of complete MP4 metadata/PTS census; no new full RGB decoding",
            "Caption presence checked; semantic alignment to an 81-frame window not evaluated",
        ],
    }
    (results / f"{STEM}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    fps_lines = ["Eligible native FPS, ascending; exact fractions are in the counts CSV."]
    fps_lines.extend(format(float(rate), ".12g") for rate in sorted(eligible_counts))
    (results / f"{STEM}_fps_list.txt").write_text("\n".join(fps_lines) + "\n")
    return table, summary


def plot(table: pd.DataFrame, summary: dict, results: Path):
    style()
    labels = [f"{rate} FPS" for rate in table.fps.head(5)]
    labels.append(f"其余 {summary['other_eligible_fps_count']} 档合计")
    counts = table.eligible_count.head(5).tolist() + [summary["other_eligible_video_count"]]
    assert sum(counts) == summary["eligible_count"]
    fig, ax = plt.subplots(figsize=(11.8, 6.0), layout="constrained")
    ax.barh(labels, counts, color=[BLUE] * 5 + [GRAY], height=.62)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.38)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("符合条件的视频数量（个）")
    ax.grid(axis="x", alpha=.18)
    for i, count in enumerate(counts):
        ax.text(count + max(counts) * .018, i,
                f"{count:,}（{100 * count / summary['eligible_count']:.2f}%）",
                va="center", fontsize=12)
    ax.set_title("1920×1080 · 至少 81 帧 · caption 非空", pad=18)
    fig.suptitle("OpenVidHD 原始帧率分布", fontsize=19)
    fig.supxlabel(
        f"全量筛选：{summary['eligible_count']:,} 个视频 / {summary['eligible_distinct_fps']} 档原始帧率\n"
        "百分比分母为符合条件的视频总数；完整逐档数量与精确 FPS 分数见同名 CSV。",
        fontsize=10, color="#505050")
    save(fig, results, STEM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BASE / "results/OpenVidHD")
    args = parser.parse_args()
    table, summary = analyze(args.results)
    plot(table, summary, args.results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
