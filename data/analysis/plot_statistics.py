"""Reproducible PNG/SVG/PDF figures from the complete census, including tails."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

BLUE = "#316B9B"
GOLD = "#AA741C"
GRAY = "#6E747A"


def percent_label(value):
    # Preserve small exceptions instead of rounding a non-universal class to 100%.
    return f"{value:.5f}%" if 0 < value < .01 or 99.99 < value < 100 else f"{value:.2f}%"


def style():
    font_path = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "figure.facecolor": "white",
                         "axes.axisbelow": True})


def save(fig, results, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(results / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def count_bars(ax, labels, counts, total, horizontal=False):
    percentages = np.asarray(counts, dtype=float) / total * 100
    if horizontal:
        ax.barh(labels, percentages, color=BLUE)
        ax.invert_yaxis()
        ax.set_xlim(0, max(percentages.max(initial=0) * 1.27, 1e-8))
        ax.set_xlabel("占全部视频的比例（%）")
        for i, (count, value) in enumerate(zip(counts, percentages)):
            ax.text(value + ax.get_xlim()[1] * .012, i, f"{count:,}  ({percent_label(value)})", va="center", fontsize=10)
        ax.grid(axis="x", alpha=.18)
    else:
        ax.bar(labels, percentages, color=BLUE)
        ax.set_ylim(0, max(percentages.max(initial=0) * 1.23, 1))
        ax.set_ylabel("占全部视频的比例（%）")
        for i, (count, value) in enumerate(zip(counts, percentages)):
            ax.text(i, value + ax.get_ylim()[1] * .02, f"{count:,}\n{percent_label(value)}", ha="center", va="bottom", fontsize=10)
        ax.grid(axis="y", alpha=.18)


def histogram(ax, values, width, label, results, name):
    edges = np.arange(0, math.ceil(float(values.max()) / width) * width + width, width)
    counts, _ = np.histogram(values, edges)
    assert int(counts.sum()) == len(values), "Histogram must include every row"
    ax.bar(edges[:-1], counts / len(values) * 100, width=np.diff(edges), align="edge", color=BLUE)
    mean = float(values.mean())
    ax.axvline(mean, color=GOLD, linestyle="--", linewidth=1.6, label=f"均值 {mean:.2f}")
    ax.set_xlim(0, edges[-1])
    ax.set_xlabel(label)
    ax.set_ylabel("占全部视频的比例（%）")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    pd.DataFrame({"lower_inclusive": edges[:-1], "upper_exclusive_except_last": edges[1:],
                  "count": counts, "percent": counts / len(values) * 100}).to_csv(results / f"{name}_bins.csv", index=False)


def make_plots(results):
    results = Path(results)
    summary = json.loads((results / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Plots require a complete census")
    df = pd.read_csv(results / "all_video_metrics.csv", usecols=[
        "fps", "frame_count", "duration_seconds", "width", "height", "timing_class",
        "caption_characters", "caption_words", "caption_tokens"])
    n = len(df)
    assert n == summary["population"]
    dataset = summary.get("dataset_name", "OpenVid-1M")
    style()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    counts = df.fps.value_counts()
    top = counts.head(5)
    labels = [f"{value:g}" for value in top.index] + ["其他 FPS"]
    values = top.tolist() + [int(counts.iloc[5:].sum())]
    count_bars(axes[0, 0], labels, values, n)
    axes[0, 0].set_title("不同视频的平均帧率分布")
    axes[0, 0].set_xlabel("FPS（完整频数表另存 CSV）")
    histogram(axes[0, 1], df.frame_count, 64, "原始帧数（每个区间宽 64 帧）", results, "frame_count")
    axes[0, 1].set_title("原始帧数分布：显示完整范围")
    histogram(axes[1, 0], df.duration_seconds, 2, "视频时长（秒，每个区间宽 2 秒）", results, "duration")
    axes[1, 0].set_title("视频时长分布：显示完整范围")
    timing_labels = {"constant": "严格等间隔", "quantized_uniform": "1 tick 内\n近似等间隔", "variable": "可变间隔", "invalid_timing": "时间表异常"}
    timing_counts = df.timing_class.value_counts()
    count_bars(axes[1, 1], list(timing_labels.values()), [int(timing_counts.get(k, 0)) for k in timing_labels], n)
    axes[1, 1].set_title("单个视频内部的帧呈现时间间隔")
    axes[1, 1].set_xlabel("依据完整 MP4 stts/ctts 时间表")
    fig.suptitle(f"{dataset} 视频帧率、帧数与时长｜全量 {n:,} 条", fontsize=17)
    save(fig, results, "video_timing")

    resolution = pd.read_csv(results / "resolution_distribution.csv")
    top = resolution.head(12)
    names, counts = top.resolution.tolist(), top["count"].tolist()
    remaining = int(resolution.iloc[12:]["count"].sum())
    if remaining:
        names.append(f"其余 {len(resolution)-12} 种尺寸")
        counts.append(remaining)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
    count_bars(axes[0], names, counts, n, horizontal=True)
    axes[0].set_title(f"实际宽×高：共 {len(resolution):,} 种")
    categories = [int((df.height == 480).sum()), int((df.height == 1080).sum()), int((~df.height.isin([480, 1080])).sum())]
    count_bars(axes[1], ["高度 480", "高度 1080", "其他高度"], categories, n)
    axes[1].set_title("480p / 1080p 占比（按高度定义）")
    short_1080 = int((df[["width", "height"]].min(axis=1) == 1080).sum())
    axes[1].set_xlabel(f"按短边=1080统计：{short_1080:,} 条（{percent_label(short_1080 / n * 100)}）")
    fig.suptitle(f"{dataset} 分辨率｜全量 {n:,} 条，无抽样", fontsize=17)
    save(fig, results, "resolution_distribution")

    # Every distinct resolution is also charted; small categories are not lost
    # in the top-category preview or omitted from the downloadable output.
    with PdfPages(results / "all_resolutions.pdf") as pdf:
        for start in range(0, len(resolution), 24):
            page = resolution.iloc[start:start + 24]
            fig, ax = plt.subplots(figsize=(11, max(4, len(page) * .32 + 2)), layout="constrained")
            count_bars(ax, page.resolution.tolist(), page["count"].tolist(), n, horizontal=True)
            ax.set_title(f"{dataset} 全部分辨率 · 第 {start+1}–{start+len(page)} 项 / {len(resolution)} 项\n分母：{n:,} 条视频；各页横轴独立缩放")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    caption_n = int(df.caption_characters.notna().sum())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, column, width, title, unit in zip(
        axes, ["caption_characters", "caption_words", "caption_tokens"], [100, 20, 25],
        ["字符数（含空格和标点）", "英文词数（正则分词）", "Qwen3.5-2B token 数"],
        ["字符数", "词数", "token 数（无模板和特殊 token）"],
    ):
        histogram(ax, df[column].dropna(), width, unit, results, column)
        ax.set_ylabel("占有 caption 视频的比例（%）")
        ax.set_title(title)
    fig.suptitle(f"{dataset} caption 长度｜有标注 {caption_n:,} 条，缺失 {n-caption_n:,} 条", fontsize=17)
    save(fig, results, "caption_lengths")
    print("Saved video_timing, resolution_distribution, caption_lengths as PNG/SVG/PDF; all_resolutions.pdf", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    make_plots(parser.parse_args().results)
