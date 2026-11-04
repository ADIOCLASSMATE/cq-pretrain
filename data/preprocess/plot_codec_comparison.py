"""Scientific figures and exact-frame visual inspection for the codec experiment."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

LABELS = {"mpeg4_q2": "MPEG-4 Part 2\nq=2", "h264_crf18": "H.264\nCRF 18 / medium"}
COLORS = {"mpeg4_q2": "#4477AA", "h264_crf18": "#CCAA33"}


def crop_location(pixels, size=192):
    errors = sum(np.abs(pixels[mode].astype(np.int16) - pixels["reference"].astype(np.int16))
                 .sum(axis=2, dtype=np.int32) for mode in LABELS)
    integral = np.pad(errors, ((1, 0), (1, 0))).cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
    height, width = errors.shape
    best = (-1, 0, 0)
    ys = sorted(set(list(range(0, height-size+1, 32)) + [height-size]))
    xs = sorted(set(list(range(0, width-size+1, 32)) + [width-size]))
    for y in ys:
        for x in xs:
            score = integral[y+size, x+size] - integral[y, x+size] - integral[y+size, x] + integral[y, x]
            if score > best[0]:
                best = (score, x, y)
    return best[1], best[2]


def summary_figure(root, summary):
    results = summary["results"]
    modes = [r["mode"] for r in results]
    specifications = [
        ("File size: all ten videos", "total_bytes", 1/2**20, "MiB (lower is smaller)", ".2f"),
        ("Encoding throughput", "encode_fps", 1, "Frames/s (higher is faster)", ".1f"),
        ("Decoding to RGB throughput", "decode_fps", 1, "Frames/s (higher is faster)", ".1f"),
        ("RGB PSNR: pooled squared error", "psnr_rgb_db", 1, "dB (higher is closer to reference)", ".2f"),
        ("RGB mean absolute error", "mae_rgb", 1, "8-bit levels (lower is closer)", ".3f"),
        ("RGB SSIM: all-frame mean", "ssim_rgb", 1, "SSIM (higher is closer; detail scale)", ".6f"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for ax, (title, key, factor, units, formatting) in zip(axes.flat, specifications):
        values = [r[key]*factor for r in results]
        for i, (mode, value) in enumerate(zip(modes, values)):
            if key == "ssim_rgb":
                ax.scatter(value, i, color=COLORS[mode], s=70, marker="o" if i == 0 else "s", zorder=3)
                ax.annotate(format(value, formatting), (value, i), xytext=(0, 12),
                            textcoords="offset points", ha="center", fontsize=10)
            else:
                ax.barh(i, value, color=COLORS[mode], height=0.52)
                ax.text(value + max(values)*0.025, i, format(value, formatting), va="center", fontsize=10)
        ax.set_yticks([0, 1], [LABELS[m] for m in modes])
        ax.set_ylim(1.6, -0.6)
        if key == "ssim_rgb":
            lower = max(0, min(values) - max(0.003, max(values)-min(values))*1.4)
            upper = min(1.0005, max(values) + max(0.003, max(values)-min(values))*1.4)
            ax.set_xlim(lower, upper)
        else:
            ax.set_xlim(0, max(values)*1.25)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel(units, fontsize=10)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("OpenVidHD: same 10 clips, 2,560 frames, 960x540", fontsize=16, y=0.99)
    fig.text(0.02, 0.018,
        "Reference: verified lossless resized RGB. CPU, one codec thread; speed = total frames / sum of per-clip median times (3 repeats).\n"
        "Quality includes YUV420 conversion and compression. Purposive pilot, not a representative full-dataset estimate.", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=2.4, h_pad=3.0)
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)


def sample_figure(root, rows):
    names = list(dict.fromkeys(r["sample_id"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, (key, title, units) in zip(axes, [
        ("psnr_rgb_db", "RGB PSNR by clip", "PSNR (dB), higher is closer"),
        ("ssim_rgb", "RGB SSIM by clip", "SSIM, higher is closer")]):
        for i, name in enumerate(names):
            pair = {r["mode"]:float(r[key]) for r in rows if r["sample_id"] == name}
            values = [pair[m] for m in LABELS]
            ax.plot(values, [i, i], color="#999999", linewidth=1.0, zorder=1)
            for j, mode in enumerate(LABELS):
                ax.scatter(pair[mode], i, color=COLORS[mode], marker="o" if j == 0 else "s", s=40,
                           label=LABELS[mode].replace("\n", " ") if i == 0 else None, zorder=3)
        ax.set_yticks(range(len(names)), [f"P{i+1:02d}" for i in range(len(names))])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel(units)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", fontsize=9)
    fig.text(0.05, 0.02, "P01-P10 follow pilot order. Exact filenames and values: per_sample.csv. Dot plots use detail scales.", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=3)
    fig.savefig(root / "per_sample_quality.png", dpi=180)
    plt.close(fig)


def visual_figures(root, manifest):
    destination = root / "visual_comparisons"
    destination.mkdir(exist_ok=True)
    crops = []
    for item in manifest:
        with np.load(root / item["pixel_data"]) as data:
            pixels = {key:data[key] for key in ("reference", *LABELS)}
        x, y = crop_location(pixels)
        size = 192
        fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                                 gridspec_kw={"hspace": 0.28, "wspace": 0.04})
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.91)
        columns = [("reference", "Reference: lossless RGB"),
                   ("mpeg4_q2", "MPEG-4 Part 2 / q=2"), ("h264_crf18", "H.264 / CRF 18")]
        for column, (mode, title) in enumerate(columns):
            axes[0, column].imshow(pixels[mode], interpolation="nearest")
            axes[0, column].add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="#CCAA33", linewidth=1.2))
            axes[0, column].set_title(title, fontsize=13)
            axes[1, column].imshow(pixels[mode][y:y+size, x:x+size], interpolation="nearest")
            if mode == "reference":
                text = f"Shared crop: x={x}, y={y}, {size}x{size} pixels"
            else:
                quality = item["quality"][mode]
                text = f"Frame PSNR {quality['psnr_rgb_db']:.2f} dB | SSIM {quality['ssim_rgb']:.6f} | MAE {quality['mae_rgb']:.3f}"
            axes[1, column].set_title(text, fontsize=11)
            for row in (0, 1):
                axes[row, column].axis("off")
        fig.suptitle(f"P{item['number']:02d}  {item['sample_id']}  |  frame {item['frame_index']}  |  t={item['timestamp_seconds']:.4f}s", fontsize=13)
        fig.text(0.03, 0.02,
            "Selected frame: maximum RGB MSE across both codecs in this clip. Crop: highest summed absolute error on a shared 32-pixel grid.\n"
            "Bottom row magnifies the same 192x192 region with nearest-pixel display. This is a deliberate worst-error view, not a typical-frame claim.", fontsize=10)
        fig.savefig(destination / f"sample{item['number']:02d}.png", dpi=160)
        plt.close(fig)
        crops.append(dict(number=item["number"], sample_id=item["sample_id"], frame_index=item["frame_index"],
                          x=x, y=y, width=size, height=size))
    (destination / "crop_locations.json").write_text(json.dumps(crops, indent=2) + "\n")


def notes(root, summary, config, rows):
    text = """OpenVidHD codec comparison: methodology and interpretation limits

The original ten independently verified pilot clips are used without temporal
truncation: 2,560 frames at 960x540. The pixel reference is the lossless RGB output
already proven equal to source frames 0,2,4,... after resizing. The evaluation
isolates additional loss from RGB/YUV420 conversion and video encoding; it does
not measure the original source video's loss or the resize operation's loss.

Candidates: MPEG-4 Part 2, yuv420p, qscale=2 (global_quality=2*FF_QP2LAMBDA);
H.264 via libx264, yuv420p, CRF18, medium. Both use the same PyAV/libswscale
defaults for colour conversion, one codec thread and exact rational source FPS/2.
No training code or production preprocessing implementation is changed.

PSNR_RGB = 10 log10(255^2 / pooled RGB MSE). Pool squared errors and pixel counts
before computing dB; do not average per-video PSNR values. MAE is absolute RGB
error averaged over every channel value, in 8-bit [0,255] units.
SSIM_RGB is FFmpeg ssim's All score on gbrp converted losslessly from the exact
PyAV-decoded RGB bytes used by PSNR. It averages all three RGB components; it is
not Y-only SSIM, VMAF or a learned perceptual score. Global SSIM averages frames,
which have equal geometry. Complete per-frame values are retained.
Definitions: https://www.ffmpeg.org/ffmpeg-filters.html#ssim
             https://www.ffmpeg.org/ffmpeg-filters.html#psnr

Timing: three sequential repetitions per clip and codec, with codec order
alternated/rotated. Encode starts from common RGB in memory and includes RGB to
YUV conversion, frame construction, encoder flush and MP4 mux; it excludes ZIP
reading, original-video decode, resizing, quality scoring and SHA256. Decode
includes MP4 open/read, CPU decode and RGB conversion, but no metric calculation
or raw-file writes. OS file cache is normally warm; these are not cold shared-
storage benchmarks. Summed per-video medians give aggregate seconds; throughput
is total frames divided by those seconds. All raw repetitions are in timings.csv.
Byte-identical repeat encoding is not required. Final files and quality use the
last encoding repetition; size ranges and every encoded-file hash are retained.
The shared input RGB digest is checked unchanged after all encoder runs.

All frames in both candidates are checked for 960x540 geometry, exact rational
FPS, frame count, PTS and full video duration. Source IDs and captions are retained.
Video playback files: videos/<codec>/partN/<unchanged-original-filename>.mp4.

Visual comparisons deliberately select the highest RGB MSE frame per clip across
the two candidates, then a common 192x192 crop with maximal summed absolute error
on a 32-pixel grid. They expose local artifacts rather than represent a typical
frame. Exported pixels are rechecked against the measured per-frame SSE/SAE.

These ten clips cover boundary cases and FPS groups; they are not a random sample.
No full-dataset size, throughput, confidence interval or training-quality impact
is established. Higher PSNR/SSIM is closer to the resized pixel reference; these
metrics do not by themselves prove better generated-video quality after training.
Production remains stopped via its STOP marker pending the user's codec decision.
"""
    text += "\nObserved aggregate results:\n" + json.dumps(summary, indent=2) + "\n"
    text += "\nPilot ID mapping:\n"
    for i, name in enumerate(dict.fromkeys(row["sample_id"] for row in rows), start=1):
        text += f"P{i:02d}: {name}\n"
    (root / "methodology.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "comparison_config.json").read_text())
    if not summary["complete"]:
        raise ValueError("Comparison is not complete")
    with (root / "per_sample.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    project = Path(__file__).resolve().parents[2]
    subprocess.run([str(project / ".venv-npu/bin/python"), "-m", "data.preprocess.export_codec_frames",
                    "--output", str(root)], cwd=project, check=True)
    manifest = json.loads((root / "visual_data/manifest.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12})
    summary_figure(root, summary)
    sample_figure(root, rows)
    visual_figures(root, manifest)
    notes(root, summary, config, rows)
    print(json.dumps(dict(figures_complete=True, visual_comparisons=len(manifest))), flush=True)


if __name__ == "__main__":
    main()
