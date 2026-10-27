"""Score a UMM video run by invoking an unmodified, pinned official VBench checkout."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import zipfile

from evaluations.benchmarks.vbench.generate import (
    FLICKERING, FULL_SHA256, SAMPLES, VBENCH_COMMIT,
    load_prompts, make_tasks, video_is_complete, write_json,
)


def official_checkout(root):
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != VBENCH_COMMIT:
        raise ValueError(f"Use official Vchitect/VBench commit {VBENCH_COMMIT}; found {revision}")
    subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--",
         "evaluate.py", "static_filter.py", "dimension_to_folder.json", "vbench", "scripts"],
        check=True, stdout=subprocess.DEVNULL,
    )
    if hashlib.sha256((root / "vbench/VBench_full_info.json").read_bytes()).hexdigest() != FULL_SHA256:
        raise ValueError("Official full prompt metadata does not match the pinned version")
    folders = json.loads((root / "dimension_to_folder.json").read_text())
    rows, _ = load_prompts("full")
    if set(folders) != {d for row in rows for d in row["dimension"]}:
        raise ValueError("Official dimension mapping is incomplete")
    for row in rows:
        if any(folders[d] != row["dimension"][0] for d in row["dimension"]):
            raise ValueError("Official video-folder mapping differs from bundled prompts")
    return folders


def read_run(directory):
    run = json.loads((directory / "run.json").read_text())
    if (run.get("schema_version") != 1 or run.get("vbench_commit") != VBENCH_COMMIT
            or run.get("full_sha256") != FULL_SHA256 or run.get("mode") not in {"subset", "full"}):
        raise ValueError("Unsupported run metadata or VBench version")
    rows, expected_sha = load_prompts(run["mode"])
    raw = (directory / "prompts.json").read_bytes()
    if (hashlib.sha256(raw).hexdigest() != expected_sha
            or run["prompts_sha256"] != expected_sha or json.loads(raw) != rows):
        raise ValueError("Run prompt list differs from the declared subset/full mode")
    if not 25 <= run["flicker_candidates"] <= 10000:
        raise ValueError("Invalid flickering candidate budget")
    return run, rows


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Official VBench scoring requires a CUDA environment. "
            "Generate videos on NPU, then run vbench_score.sh with VBENCH_PYTHON pointing "
            "to the separate CUDA/VBench environment. NPU scoring has not been ported; "
            "--dry-run can inspect commands without CUDA."
        )


def run_official(command, root, log):
    print(shlex.join(command), flush=True)
    env = dict(os.environ)
    # Keep the official process independent of UMM/MindSpeed imports and launch ranks.
    env["PYTHONPATH"] = str(root)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        env.pop(key, None)
    env["MASTER_ADDR"] = "127.0.0.1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        env["MASTER_PORT"] = str(sock.getsockname()[1])
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Official VBench exited {result.returncode}; log: {log}\n{tail}")


def evaluation_command(root, run_dir, results, dimension, folder):
    videos = (results / "static/filtered_videos" if dimension == FLICKERING
              else run_dir / "videos" / folder)
    return [sys.executable, str(root / "evaluate.py"),
            "--videos_path", str(videos), "--dimension", dimension,
            "--full_json_dir", str(run_dir / "prompts.json"),
            "--output_path", str(results / dimension), "--mode", "vbench_standard"]


def score_run(run_dir, root, *, dry_run=False):
    folders = official_checkout(root)
    run, rows = read_run(run_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results = run_dir / "scores" / timestamp
    bootstrap = [sys.executable, "-c",
                 "from vbench.utils import init_submodules; "
                 "init_submodules(['dynamic_degree'])"]
    # The official static filter loads RAFT but does not download it itself.
    static = [sys.executable, str(root / "static_filter.py"),
              "--videos_path", str(run_dir / "videos" / FLICKERING),
              "--result_path", str(results / "static"),
              "--filter_scope", "all" if run["mode"] == "subset" else FLICKERING]
    commands = [bootstrap, static] + [
        evaluation_command(root, run_dir, results, d, folder) for d, folder in folders.items()
    ]
    aggregate = [sys.executable, str(root / "scripts/cal_final_score.py"),
                 "--zip_file", str(results / "results.zip"),
                 "--model_name", str(results / "aggregate")]
    if run["mode"] == "full":
        commands.append(aggregate)
    if dry_run:
        print(json.dumps({"mode": run["mode"], "prompts": len(rows), "dimensions": list(folders),
                          "results": str(results), "commands": commands}, indent=2))
        return
    require_cuda()
    tasks = make_tasks(rows, run["config"]["sampling"]["seed"], run["flicker_candidates"])
    video = run["config"]["data"]["video"]
    missing = [t["path"] for t in tasks if not video_is_complete(run_dir / t["path"], video)]
    if missing:
        raise ValueError(f"{len(missing)} missing/invalid candidate videos; resume generation first. "
                         f"Examples: {missing[:3]}")
    results.mkdir(parents=True)
    write_json(results / "commands.json", commands)
    run_official(bootstrap, root, results / "prepare_raft.log")
    run_official(static, root, results / "static_filter.log")
    filtered = results / "static/filtered_videos"
    insufficient = [row["prompt_en"] for row in rows if FLICKERING in row["dimension"]
                    and any(not video_is_complete(filtered / f"{row['prompt_en']}-{i}.mp4", video)
                            for i in range(SAMPLES))]
    if insufficient:
        write_json(results / "insufficient_static.json", insufficient)
        raise ValueError(
            f"Fewer than 5 static videos for {len(insufficient)} selected prompts; "
            f"see {results / 'insufficient_static.json'}. Increase --flicker-candidates "
            "when resuming generation, then rerun scoring. No total score was produced."
        )

    summary = {"mode": run["mode"], "vbench_commit": VBENCH_COMMIT,
               "prompts": len(rows), "scoring_videos": len(rows) * SAMPLES,
               "dimensions": {}, "sample_counts": {}}
    result_files = {}
    for dimension, folder in folders.items():
        destination = results / dimension
        destination.mkdir()
        command = evaluation_command(root, run_dir, results, dimension, folder)
        run_official(command, root, destination / "console.log")
        files = list(destination.glob("*_eval_results.json"))
        if len(files) != 1:
            raise RuntimeError(f"Expected exactly one official result for {dimension}: {files}")
        payload = json.loads(files[0].read_text())
        value = payload.get(dimension)
        if (set(payload) != {dimension} or not isinstance(value, list) or len(value) != 2
                or isinstance(value[0], bool) or not isinstance(value[0], (int, float))
                or not math.isfinite(value[0])):
            raise ValueError(f"Missing or non-finite official score for {dimension}")
        # Check the actual list passed to the metric, not just files on disk.
        infos = list(destination.glob("*_full_info.json"))
        if len(infos) != 1:
            raise ValueError(f"Missing official input manifest for {dimension}")
        actual = json.loads(infos[0].read_text())
        expected = {row["prompt_en"] for row in rows if dimension in row["dimension"]}
        base = filtered if dimension == FLICKERING else run_dir / "videos" / folder
        if (len(actual) != len(expected) or {row["prompt_en"] for row in actual} != expected
                or any(set(row["video_list"]) != {
                    str(base / f"{row['prompt_en']}-{i}.mp4") for i in range(SAMPLES)
                } for row in actual)):
            raise ValueError(f"Official scorer did not consume all expected inputs for {dimension}")
        summary["dimensions"][dimension] = value[0]
        summary["sample_counts"][dimension] = {"prompts": len(expected), "videos": len(expected) * SAMPLES}
        result_files[dimension] = files[0]

    if run["mode"] == "full":
        # The upstream aggregator fills absent dimensions with zero; never pass it partial results.
        if set(result_files) != set(folders):
            raise ValueError("All 16 official dimension results are required for the total score")
        with zipfile.ZipFile(results / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for dimension, path in result_files.items():
                archive.write(path, f"{dimension}.json")
        run_official(aggregate, root, results / "aggregate.log")
        totals = dict(re.findall(r"\|\s*(quality|semantic|total) score\|([^|]+)\|",
                                 (results / "aggregate.log").read_text()))
        if set(totals) != {"quality", "semantic", "total"}:
            raise ValueError("Official aggregation did not produce all three scores")
        summary["aggregate"] = {k: float(v) for k, v in totals.items()}
        if not all(math.isfinite(v) for v in summary["aggregate"].values()):
            raise ValueError("Non-finite official aggregate score")
    else:
        summary["note"] = "Fixed UMM diagnostic subset; not a full VBench benchmark result."
    write_json(results / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {results / 'summary.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Output directory from generate.py")
    parser.add_argument("--vbench-root", default=os.environ.get("VBENCH_ROOT"),
                        help="Official Vchitect/VBench checkout at the documented commit")
    parser.add_argument("--dry-run", action="store_true", help="Inspect official commands without CUDA, downloads or scoring")
    args = parser.parse_args(argv)
    if not args.vbench_root:
        parser.error("Pass --vbench-root or set VBENCH_ROOT")
    score_run(Path(args.run_dir).resolve(), Path(args.vbench_root).resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
