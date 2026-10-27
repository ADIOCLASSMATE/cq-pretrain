"""Pinned VBench protocol adapter for the common sample/score scheduler.

This defines the integration boundary; official scoring still requires its
separate CUDA installation and has not been ported to Ascend.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .generate import (
    DATA_DIR, FULL_SHA256, SAMPLES, VBENCH_COMMIT,
    load_prompts, make_tasks, prepare_run,
)


def plan_vbench(task, config):
    """Return common sampling requests and unchanged official scorer metadata."""
    from evaluations.sampling import validate_sampling_task, output_types

    validate_sampling_task(task)
    if "mp4" not in output_types(task, "t2v"):
        raise ValueError("VBench sampling requires MP4 outputs for its scoring protocol")
    settings, data = task.get("sample") or {}, task.get("data") or {}
    if set(data) - {"split"}:
        raise ValueError("Pinned VBench data supports split only; prompt metadata is fixed")
    sample_keys = {"mode", "seed", "samples_per_prompt", "flicker_candidates", "generation",
                   "video", "outputs", "collect"}
    if set(settings) - sample_keys:
        raise ValueError(f"Unsupported VBench sample parameters: {sorted(set(settings) - sample_keys)}")
    if settings.get("mode", "t2v") != "t2v":
        raise ValueError("VBench sample.mode must be t2v")
    mode = data.get("split", "subset")
    if mode not in {"subset", "full"}:
        raise ValueError("VBench data.split must be subset or full")
    if settings.get("samples_per_prompt", SAMPLES) != SAMPLES:
        raise ValueError("Pinned VBench requires exactly 5 samples per prompt")
    budget = settings.get("flicker_candidates", 25)
    if type(budget) is not int or not 25 <= budget <= 10000:
        raise ValueError("VBench flicker_candidates must be between 25 and 10000")
    effective = deepcopy(config)
    effective["sampling"].update(settings.get("generation") or {})
    effective["sampling"]["seed"] = (config["sampling"]["seed"] if settings.get("seed") is None else settings["seed"])
    video = settings.get("video")
    if video:
        effective["data"]["video"].update(video)
    rows, prompt_sha = load_prompts(mode)
    planned = make_tasks(rows, effective["sampling"]["seed"], budget)
    requests = [dict(item, task="t2v", id=hashlib.sha256(item["path"].encode()).hexdigest()[:24])
                for item in planned]
    checkpoint = Path(effective["model"]["checkpoint"])
    stat = checkpoint.stat()
    metadata = dict(
        schema_version=1, mode=mode, vbench_commit=VBENCH_COMMIT,
        full_sha256=FULL_SHA256, prompts_sha256=prompt_sha,
        checkpoint=dict(path=str(checkpoint.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns),
        config=effective, seed_rule="sampling.seed + video_index", flicker_candidates=budget,
    )
    return requests, metadata, (DATA_DIR / f"{mode}.json").read_bytes()


def prepare_vbench(directory, metadata, prompts_bytes):
    prepare_run(Path(directory), metadata, prompts_bytes)


def vbench_score_command(task, directory):
    settings = task.get("score") or {}
    score_keys = {"input", "dimensions", "static_filter", "environment", "python", "vbench_root", "dry_run"}
    if set(settings) - score_keys:
        raise ValueError(f"Unsupported VBench score parameters: {sorted(set(settings) - score_keys)}")
    if settings.get("input", "sampled") != "sampled":
        raise ValueError("VBench score.input must be sampled")
    if settings.get("environment", "vbench_cuda") != "vbench_cuda":
        raise ValueError("VBench scoring currently requires environment=vbench_cuda")
    if settings.get("dimensions", "all") != "all" or settings.get("static_filter", True) is not True:
        raise ValueError("Pinned VBench adapter requires all dimensions and the native static filter")
    root = settings.get("vbench_root") or os.environ.get("VBENCH_ROOT")
    if not root:
        raise ValueError("VBench scoring requires score.vbench_root or VBENCH_ROOT")
    executable = settings.get("python") or os.environ.get("VBENCH_PYTHON")
    if not executable:
        raise ValueError("VBench scoring requires score.python or VBENCH_PYTHON in a separate CUDA environment")
    command = [str(executable), "-m", "evaluations.benchmarks.vbench.score",
               "--run-dir", str(Path(directory).resolve()), "--vbench-root", str(Path(root).resolve())]
    if settings.get("dry_run", False):
        command.append("--dry-run")
    return command


def run_vbench_score(task, directory):
    directory = Path(directory).resolve()
    command = vbench_score_command(task, directory)
    project = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project)
    # A score process is independent of the launcher group and loads no NPU
    # plugins. The configured interpreter supplies all CUDA scoring packages.
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "MASTER_ADDR", "MASTER_PORT", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
        env.pop(key, None)
    existing = set((directory / "scores").glob("*/summary.json"))
    log_dir = Path(task["_score_log_dir"]) if task.get("_score_log_dir") else directory / "scores" / f"adapter-{uuid.uuid4().hex}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "console.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", file=sys.stderr, flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"VBench scoring exited {code}; see {log_path}")
    if (task.get("score") or {}).get("dry_run", False):
        return {"status": "dry_run", "command": command}
    summaries = set((directory / "scores").glob("*/summary.json")) - existing
    if len(summaries) != 1:
        raise RuntimeError("VBench did not produce exactly one new summary.json")
    summary = summaries.pop()
    return {"summary": str(summary.relative_to(directory)), "metrics": json.loads(summary.read_text())}
