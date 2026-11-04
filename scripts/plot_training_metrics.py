#!/usr/bin/env python3
"""Render a reusable training figure from one RunLogger attempt (CPU only).

Requires matplotlib. No training runtime, model weights, or network connection
is needed. Read one attempt at a time so resumed steps are never silently merged.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def read_metrics(path):
    raw = path.read_bytes()
    rows = []
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        # A live writer may not have finished its final JSONL record yet.
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            print("Skipping unfinished final JSONL line")
            continue
        row = json.loads(line)
        if "train/loss" in row:
            rows.append(row)
    if not rows:
        raise ValueError(f"No complete training records in {path}")
    steps = [int(row["step"]) for row in rows]
    if any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Steps must strictly increase within one attempt")
    attempts = {row.get("attempt_id") for row in rows}
    if len(attempts) != 1:
        raise ValueError("Mixed attempt IDs; plot each attempt separately")
    return raw, rows


def trailing_mean(values, window):
    return [mean(values[max(0, i - window + 1):i + 1])
            for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="UMM_v3 training")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Trailing logged-record count; visual smoothing only")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be positive")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    source = args.metrics.resolve()
    raw, rows = read_metrics(source)
    steps = [int(row["step"]) for row in rows]
    blue, orange = "#205FAD", "#C16A21"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B4BDC8", "text.color": "#1F2937",
        "axes.labelcolor": "#374151", "xtick.color": "#4B5563",
        "ytick.color": "#4B5563", "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(left=.075, right=.975, bottom=.15, top=.83,
                        wspace=.22, hspace=.4)
    fig.suptitle(args.title, x=.075, ha="left", y=.97,
                 fontsize=22, fontweight="bold")
    fig.text(.075, .91,
             f"{len(rows)} logged windows | optimizer steps {steps[0]}-{steps[-1]}"
             f" | {rows[0]['timestamp'][:10]}", fontsize=12)
    fig.text(.075, .875,
             f"Faint lines: raw values   |   Bold lines: trailing {args.smooth}-record mean"
             " (visual smoothing)", fontsize=10, color="#5C6777")

    series = {}

    def draw(ax, key, label, color, linestyle="-", transform=None):
        if not all(key in row for row in rows):
            ax.text(.5, .5, f"Unavailable: {key}", transform=ax.transAxes,
                    ha="center", fontsize=9)
            return
        values = [float(transform(row) if transform else row[key]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite values in {key}; inspect training logs")
        series[key] = values
        ax.plot(steps, values, color=color, alpha=.25, linewidth=1,
                linestyle=linestyle)
        ax.plot(steps, trailing_mean(values, args.smooth), color=color,
                linewidth=2.1, linestyle=linestyle, label=label)

    ax = axes[0, 0]
    ax.set_title("Training loss and global objective", loc="left")
    draw(ax, "train/loss", "Reported train/loss", blue)
    draw(ax, "train/objective_global", "Global normalized objective", orange, "--")
    ax.set_ylabel("Loss (different aggregation definitions)")
    ax.set_ylim(bottom=0)

    ax = axes[0, 1]
    ax.set_title("T2V video loss", loc="left")
    draw(ax, "train/loss_t2v_video", "Video flow MSE", blue)
    ax.set_ylabel("MSE")
    ax.set_ylim(bottom=0)

    ax = axes[1, 0]
    ax.set_title("Text prediction loss", loc="left")
    draw(ax, "train/loss_t2v_text", "T2V text CE", blue)
    draw(ax, "train/loss_v2t_text", "V2T text CE", orange, "--")
    ax.set_ylabel("Cross entropy")
    ax.set_ylim(bottom=0)

    ax = axes[1, 1]
    ax.set_title("Time per optimizer update", loc="left")
    draw(ax, "seconds", "Logged wall time / updates", blue,
         transform=lambda row: row["seconds"] / row["optimizer_updates"])
    if series.get("seconds") and all(value > 0 for value in series["seconds"]):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Seconds / update (log scale)")

    candidates = MaxNLocator(nbins=4, integer=True).tick_values(steps[0], steps[-1])
    ticks = sorted({steps[0], steps[-1]} |
                   {int(step) for step in candidates if steps[0] < step < steps[-1]})
    for ax in axes.flat:
        ax.set_xlabel("Optimizer step")
        ax.set_xticks(ticks)
        if steps[0] < steps[-1]:
            ax.set_xlim(steps[0], steps[-1])
        ax.grid(axis="y", color="#DDE3EA", linewidth=.7)
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.text(.075, .075,
             "Timing includes compilation, I/O and periodic checkpoint overhead;"
             " it is not kernel-only latency.", fontsize=10, color="#5C6777")
    fig.text(.075, .047,
             "Training metrics only; validation loss and generated-video quality"
             " are not measured by these curves.", fontsize=10, color="#5C6777")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        destination = args.output_dir / f"loss_curves.{suffix}"
        fig.savefig(destination, dpi=180, facecolor="white")
        print(destination.resolve())
    plt.close(fig)
    provenance = {
        "source": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "attempt_id": rows[0].get("attempt_id"), "records": len(rows),
        "renderer": {"name": "matplotlib", "version": matplotlib.__version__},
        "first_step": steps[0], "last_step": steps[-1],
        "smoothing": {"method": "trailing arithmetic mean for display only",
                      "records": args.smooth, "partial_initial_window": True},
        "series": {key: {"first": values[0], "last": values[-1]}
                   for key, values in series.items()},
        "time_definition": "seconds / optimizer_updates; includes runtime overhead",
    }
    (args.output_dir / "plot_source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
