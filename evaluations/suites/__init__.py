"""Benchmark evaluation of committed evaluation snapshots (see configs/evaluation.yaml)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("qa", "vqav2", "geneval", "fid")


def settings(path=None):
    value = yaml.safe_load(Path(path or ROOT / "configs/evaluation.yaml").read_text())
    if set(value["tasks"]) - set(TASKS):
        raise ValueError(f"Unknown evaluation tasks: {set(value['tasks']) - set(TASKS)}")
    return value


def task_dir(output, task, step):
    return Path(output) / task / f"step_{step:07d}"
