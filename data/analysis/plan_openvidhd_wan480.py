"""Count fixed-clip candidates from the full OpenVidHD census; no video conversion.

Run with python3 for counts, or the project Python with --check-processor to
exercise the existing Qwen preprocessing/serialization on synthetic RGB frames
and an actual longest-caption candidate. No model weights or accelerator used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def census(results):
    summary = json.loads((results / "summary.json").read_text())
    source = Path(summary["source"])
    assert summary["video_root"] == "/inspire/dataset/openvid-1m/v2/OpenVidHD"
    assert summary["complete"] and sha256(source) == summary["source_sha256"]
    metrics = results / "all_video_metrics.csv"
    names, stages, fps_counts, duration_bands = set(), Counter(), Counter(), Counter()
    words_sum = tokens_sum = duration_false_positives = 0
    max_tokens, longest_video = -1, None
    all_caption_pts5 = all_pts5 = 0
    for row in csv.DictReader(metrics.open()):
        name = row["video"]
        assert name not in names
        names.add(name)
        stages["all_videos"] += 1
        has_caption = float(row["caption_characters"] or 0) > 0
        ticks, scale = int(row["pts_span_ticks"]), int(row["timescale"])
        long_enough = ticks >= 5 * scale
        all_pts5 += long_enough
        all_caption_pts5 += has_caption and long_enough
        if not has_caption:
            continue
        stages["with_caption"] += 1
        if (int(row["width"]), int(row["height"])) != (1920, 1080):
            continue
        stages["with_caption_1920x1080"] += 1
        if row["status"] != "ok" or row["timing_class"] != "constant" or row["rotated"] != "False":
            continue
        stages["valid_cfr_unrotated"] += 1
        duration_false_positives += float(row["duration_seconds"]) >= 5 and not long_enough
        if not long_enough:
            continue
        stages["pts_span_ge_5s"] += 1
        # CFR PTS intervals, not rounded container average FPS.
        if scale < 16 * int(row["pts_interval_max_ticks"]):
            continue
        stages["source_fps_ge_16"] += 1
        fps_counts[row["fps"]] += 1
        span = ticks / scale
        band = "5_to_6" if span < 6 else "6_to_10" if span < 10 else "10_to_20" if span < 20 else "20_plus"
        duration_bands[band] += 1
        words_sum += int(float(row["caption_words"]))
        tokens = int(float(row["caption_tokens"]))
        tokens_sum += tokens
        if tokens > max_tokens:
            max_tokens, longest_video = tokens, name
    assert stages["all_videos"] == summary["population"]
    assert stages["with_caption"] == summary["caption_population"]
    n = stages["source_fps_ge_16"]
    previous, funnel = stages["all_videos"], []
    for stage, count in stages.items():
        funnel.append(dict(stage=stage, count=count, excluded_since_previous=previous-count,
                           percent_of_all=100*count/stages["all_videos"]))
        previous = count
    return dict(
        source_metrics=str(metrics), source_metrics_sha256=sha256(metrics),
        source_captions=str(source), source_captions_sha256=summary["source_sha256"],
        scope=summary["video_root"], population=stages["all_videos"],
        target=dict(width=832, height=480, fps=16, num_frames=81,
                    first_to_last_seconds=5, encoded_duration_seconds=81/16),
        policy=dict(source_resolution=[1920,1080], caption_required=True,
                    strict_cfr=True, min_source_fps=16, min_pts_span_seconds=5,
                    clips_per_source=1, window_start="(pts_span_seconds - 5) / 2",
                    caption_alignment="pending verification or clip-specific recaptioning"),
        funnel=funnel, candidate_count=n, candidate_percent=100*n/stages["all_videos"],
        candidate_content_hours=n*5/3600,
        all_resolutions_pts_span_ge_5=all_pts5,
        all_resolutions_caption_pts_span_ge_5=all_caption_pts5,
        source_duration_ge5_but_pts_span_lt5=duration_false_positives,
        eligible_fps_counts=dict(fps_counts.most_common()),
        eligible_pts_span_bands=dict(duration_bands),
        eligible_caption_words_mean=words_sum/n,
        eligible_caption_tokens_mean=tokens_sum/n,
        eligible_caption_tokens_max=max_tokens,
        longest_caption_video=longest_video,
        limitations=["Full census of metadata and PTS tables, not full RGB decoding",
                     "No semantic caption-window verification performed",
                     "Candidate count precedes decode and caption acceptance and split grouping"],
    )


def processor_check(result):
    sys.path.insert(0, str(BASE.parent.parent))
    import numpy as np
    import torch
    from configs.config import DEFAULTS
    from data.dataset import qwen_video_inputs
    from data.sequence import make_video_sequence
    from training.train import load_processor

    torch.set_num_threads(4)
    config = deepcopy(DEFAULTS)
    processor = load_processor(config)
    settings = dict(num_frames=81, fps=16.0, height=480, width=832,
                    qwen_num_frames=33, qwen_processor={"do_resize": False})
    rgb = torch.zeros(81, 3, 480, 832, dtype=torch.uint8)
    clip = {"timestamps": (np.arange(81) / 16).tolist()}
    pixels, grid, times = qwen_video_inputs(rgb, clip, processor, settings)
    del pixels
    assert grid.tolist() == [17,30,52]
    indices = np.linspace(0,80,33).round().astype(int).tolist()
    assert times == [i/16 for i in indices]
    with Path(result["source_captions"]).open() as f:
        caption = next(r["caption"] for r in csv.DictReader(f)
                       if r["video"] == result["longest_caption_video"])
    text = lambda s: {"type":"text", "text":s}
    visual = {"type":"video", "video_id":"v0"}
    messages = {
        "t2v": [{"role":"user", "content":[text(caption)]},
                {"role":"assistant", "content":[visual]}],
        "v2t": [{"role":"user", "content":[visual,text("Describe this video.")]},
                {"role":"assistant", "content":[text(caption)]}],
    }
    lengths = {task: len(make_video_sequence(msg, processor, grid, times,
                           max_seq_length=8192)["input_ids"])
               for task,msg in messages.items()}
    # Check serialization for all 81 frames without allocating their pixel patches.
    full_grid = torch.tensor([41,30,52])
    full_length = len(make_video_sequence(messages["t2v"], processor, full_grid,
                      clip["timestamps"], max_seq_length=None)["input_ids"])
    assert full_length > 8192
    return dict(settings=settings, frame_indices=indices, timestamps=times,
                grid_thw=grid.tolist(), visual_tokens=int(grid.prod())//4,
                caption_tokens=len(processor.tokenizer.encode(caption,add_special_tokens=False)),
                longest_caption_sample_sequence_lengths=lengths,
                all_81_frames_visual_tokens=int(full_grid.prod())//4,
                all_81_frames_sample_sequence_length=full_length,
                validation_scope="CPU processor and sequence assembly; synthetic RGB plus real caption; no ViT, VAE, DiT or training forward")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=BASE/"results/OpenVidHD")
    p.add_argument("--check-processor", action="store_true")
    args = p.parse_args()
    result = census(args.results)
    if args.check_processor:
        result["processor_check"] = processor_check(result)
    output = args.results / "wan480_81f_plan_statistics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "eligible_fps_counts"}, ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
