"""Check full-population reconciliation and independently verify MP4 parsing."""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import av
import numpy as np
import pandas as pd

from mp4_metadata import SplitFile
from scan_videos import DEFAULT_RESULTS, write_json


def validate(results):
    summary = json.loads((results / "summary.json").read_text())
    df = pd.read_csv(results / "all_video_metrics.csv", low_memory=False)
    n = len(df)
    assert summary["complete"] and n == summary["population"]
    assert df.video.nunique() == n and (df.status == "ok").all()
    known = df.csv_frame_count.notna()
    caption_n = int(known.sum())
    assert caption_n == summary.get("caption_population", n)
    assert int((~known).sum()) == summary.get("missing_caption_videos", 0)
    assert (df.loc[known, "frame_count"] == df.loc[known, "csv_frame_count"]).all()
    assert np.allclose(df.loc[known, "fps"], df.loc[known, "csv_fps"], rtol=0, atol=.001)
    assert np.allclose(df.loc[known, "duration_seconds"], df.loc[known, "csv_duration_seconds"], rtol=0, atol=.001)
    full_checks = []
    for name in ["resolution_distribution", "height_distribution", "short_side_distribution",
                 "fps_distribution", "frame_count_distribution", "timing_distribution"]:
        counts = pd.read_csv(results / f"{name}.csv")
        assert int(counts["count"].sum()) == n
        assert abs(float(counts.percent.sum()) - 100) < 1e-8
        full_checks.append(name)
    for name in ["frame_count", "duration", "caption_characters", "caption_words", "caption_tokens"]:
        counts = pd.read_csv(results / f"{name}_bins.csv")
        assert int(counts["count"].sum()) == (caption_n if name.startswith("caption_") else n)
        full_checks.append(f"{name}_bins")

    # Independent parser checks are validation samples, not the population
    # used for any statistic. Include every repaired special edit-list file.
    names = set(random.Random(20261028).sample(df.video.tolist(), min(64, n)))
    names.update(df.loc[~known, "video"].tolist())
    names.update(df.loc[df.timing_class == "invalid_timing", "video"].tolist())
    variable_names = df.loc[df.timing_class == "variable", "video"].tolist()
    names.update(random.Random(20261028).sample(variable_names, min(32, len(variable_names))))
    names.update(df.loc[df.timing_class == "quantized_uniform", "video"].tolist())
    if (results / "timing_refresh.json").exists():
        refreshed = [r["video"] for r in json.loads((results / "timing_refresh.json").read_text())["records"]]
        names.update(random.Random(20261028).sample(refreshed, min(64, len(refreshed))))
    if (results / "repair_log.json").exists():
        names.update(row["video"] for row in json.loads((results / "repair_log.json").read_text())["records"])
    chosen = df[df.video.isin(names)]
    archives = dict(json.loads((results / "archive_manifest.json").read_text())["archives"])
    records = []
    decoded_anomalies = []
    for index, group in chosen.groupby("archive"):
        with SplitFile(archives[int(index)]) as source, zipfile.ZipFile(source) as archive:
            for row in group.itertuples():
                with archive.open(row.member) as handle, av.open(handle) as container:
                    stream = container.streams.video[0]
                    pts = sorted(p.pts for p in container.demux(stream) if p.pts is not None)
                    intervals = np.diff(pts)
                    assert (stream.width, stream.height, stream.frames) == (row.width, row.height, row.frame_count), row.video
                    assert len(pts) == row.encoded_sample_count, row.video
                    assert abs(float(stream.average_rate) - row.fps) < 1e-6, row.video
                    av_duration = float(stream.duration * stream.time_base)
                    duration_matches = [name for name, value in (
                        ("stts", row.duration_seconds), ("mdhd", row.media_duration_seconds))
                        if abs(av_duration - value) < 1e-6]
                    # FFmpeg can use either the sample-table duration or mdhd
                    # declaration, particularly for the inconsistent files.
                    assert duration_matches, (row.video, av_duration, row.duration_seconds, row.media_duration_seconds)
                    assert float(stream.time_base) == 1 / row.timescale, row.video
                    if row.timing_class != "invalid_timing":
                        assert pts[-1] - pts[0] == row.pts_span_ticks, row.video
                        assert intervals.min() == row.pts_interval_min_ticks, row.video
                        assert intervals.max() == row.pts_interval_max_ticks, row.video
                        grid_error = float(np.abs(np.asarray(pts) - np.linspace(pts[0], pts[-1], len(pts))).max())
                        assert abs(grid_error - row.uniform_grid_max_error_ticks) < 1e-6, row.video
                    records.append({"video": row.video, "archive": int(index), "passed": True,
                                    "timing_class": row.timing_class, "av_duration_matches": duration_matches})
                if row.timing_class == "invalid_timing":
                    with archive.open(row.member) as handle, av.open(handle) as container:
                        decoded_pts = [frame.pts for frame in container.decode(video=0)]
                    decoded_anomalies.append(dict(video=row.video, csv_frame_count=row.csv_frame_count,
                                                  stts_frame_count=row.frame_count,
                                                  stsz_sample_count=row.encoded_sample_count,
                                                  decoded_frame_count=len(decoded_pts),
                                                  decoded_pts=decoded_pts,
                                                  duplicate_decoded_pts=len(decoded_pts) - len(set(decoded_pts))))
    result = dict(full_population=n, full_reconciliation_passed=True,
                  caption_population=caption_n, missing_caption_videos=int((~known).sum()),
                  frequency_and_histogram_checks=full_checks,
                  independent_pyav_checks=len(records), independent_checks_passed=True,
                  measured_invalid_timing_tables=int((df.timing_class == "invalid_timing").sum()),
                  measured_media_duration_disagreements=int((df.track_duration_ticks != df.sample_duration_ticks).sum()),
                  validation_sample_seed=20261028,
                  note="All reported dataset statistics use every video; PyAV checks only validate the metadata parser",
                  independent_records=records)
    write_json(results / "validation.json", result)
    write_json(results / "timing_anomaly_decode.json", {"videos": decoded_anomalies})
    print(json.dumps({k: v for k, v in result.items() if k != "independent_records"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    validate(parser.parse_args().results)
