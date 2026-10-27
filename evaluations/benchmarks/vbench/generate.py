"""Generate VBench videos with a UMM export; one model replica per device."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
FULL_SHA256 = "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd"
DATA_DIR = Path(__file__).resolve().parent
SAMPLES = 5
FLICKERING = "temporal_flickering"


def load_prompts(mode):
    path = DATA_DIR / f"{mode}.json"
    raw = path.read_bytes()
    if mode == "full" and hashlib.sha256(raw).hexdigest() != FULL_SHA256:
        raise ValueError("full.json differs from the pinned official prompt metadata")
    rows = json.loads(raw)
    if not rows:
        raise ValueError("The prompt list is empty")
    names = set()
    for row in rows:
        prompt, dimensions = row["prompt_en"], row["dimension"]
        if not dimensions or any(not isinstance(d, str) or "/" in d for d in dimensions):
            raise ValueError(f"Invalid dimensions: {dimensions}")
        if (not isinstance(prompt, str) or not prompt or "/" in prompt
                or "\x00" in prompt or len(f"{prompt}-9999.mp4".encode()) > 255):
            raise ValueError(f"Prompt cannot be used as an official filename: {prompt!r}")
        key = (dimensions[0], prompt)
        if key in names:
            raise ValueError(f"Duplicate output filename in prompt list: {key}")
        names.add(key)
    if mode == "subset":
        full, _ = load_prompts("full")
        if any(row not in full for row in rows):
            raise ValueError("Subset must retain complete, unmodified official prompt records")
    return rows, hashlib.sha256(raw).hexdigest()


def make_tasks(rows, seed, flicker_candidates=25):
    tasks = []
    for row in rows:
        folder, prompt = row["dimension"][0], row["prompt_en"]
        count = flicker_candidates if folder == FLICKERING else SAMPLES
        for index in range(count):
            tasks.append({"prompt": prompt, "folder": folder, "index": index,
                          "seed": seed + index,
                          "path": f"videos/{folder}/{prompt}-{index}.mp4"})
    return tasks


def shard_tasks(tasks, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid RANK/WORLD_SIZE")
    # No DistributedSampler padding: each video is owned by exactly one rank.
    return tasks[rank::world_size]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_run(output, run, prompt_bytes):
    # Model configs can contain integer label keys/tuples; compare their on-disk form.
    run = json.loads(json.dumps(run))
    output.mkdir(parents=True, exist_ok=True)
    # Short metadata lock also handles workers arriving before rank zero.
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / "run.json"
        if path.exists():
            previous = json.loads(path.read_text())
            old_budget = previous.pop("flicker_candidates")
            current = {k: v for k, v in run.items() if k != "flicker_candidates"}
            if previous != current or (output / "prompts.json").read_bytes() != prompt_bytes:
                raise ValueError("Output belongs to a different checkpoint/config/prompt list; use a new --output")
            if run["flicker_candidates"] < old_budget:
                raise ValueError(f"Keep --flicker-candidates >= {old_budget} when resuming")
        elif any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("Nonempty output has no run.json; use a new --output")
        prompt_path = output / "prompts.json"
        if not prompt_path.exists():
            prompt_path.write_bytes(prompt_bytes)
        write_json(path, run)


def video_is_complete(path, video):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (stream.width != video["width"] or stream.height != video["height"]
                    or stream.frames != video["num_frames"] or stream.average_rate is None
                    or abs(float(stream.average_rate) - video["fps"]) > 1e-4):
                return False
            return next(container.decode(video=0), None) is not None
    except (av.error.FFmpegError, IndexError, ValueError, EOFError):
        return False


def generate(config, tasks, output, rank, local_rank):
    """Compatibility entry point; model loading and generation use common eval."""
    video = config["data"]["video"]
    pending = [task for task in tasks if not video_is_complete(output / task["path"], video)]
    if not pending:
        print(f"[rank {rank}] all {len(tasks)} assigned videos already complete", flush=True)
        return

    import shutil
    from evaluations.runtime import EvaluationRuntime
    from evaluations.sampling import sample_request

    runtime = EvaluationRuntime(config)
    previous_rank = os.environ.get("LOCAL_RANK")
    os.environ["LOCAL_RANK"] = str(local_rank)
    try:
        device = runtime.device
    finally:
        if previous_rank is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = previous_rank
    model = runtime.get_model()
    print(json.dumps({"rank": rank, "device": str(device), "qwen_gdn": runtime.backend_identity,
                      "model_dtype": str(next(model.parameters()).dtype)}), flush=True)
    settings = {"benchmark": "vbench", "sample": {"mode": "t2v", "outputs": ["mp4"]}}
    staging = output / ".staging"
    staging.mkdir(exist_ok=True)
    with (output / f"generation.rank-{rank}.jsonl").open("a") as log:
        for task in pending:
            started = time.monotonic()
            request = dict(task, task="t2v", id=hashlib.sha256(task["path"].encode()).hexdigest()[:24])
            temporary = staging / uuid.uuid4().hex
            try:
                # Preserve the official names and seed + repetition convention.
                # Validate the encoded video before publishing its final path.
                result = sample_request(runtime, settings, request, temporary)
                artifact = temporary / result["artifacts"]["mp4"]
                target = output / task["path"]
                if not video_is_complete(artifact, video):
                    raise RuntimeError(f"Generated video has unexpected geometry or encoding: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                artifact.replace(target)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
            elapsed = time.monotonic() - started
            log.write(json.dumps({**task, "seconds": elapsed}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"[rank {rank}] {task['path']} seed={task['seed']} {elapsed:.2f}s", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Full weights.pt or weights-ema.pt export")
    parser.add_argument("--mode", choices=("subset", "full"), default="subset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[], help="Existing UMM section.key=value override")
    parser.add_argument("--flicker-candidates", type=int, default=25,
                        help="Candidates per flickering prompt; increase to supplement insufficient static videos")
    parser.add_argument("--max-videos", type=int,
                        help="Visit only the first N planned videos for a smoke run; omit to resume the complete mode")
    parser.add_argument("--dry-run", action="store_true", help="Print counts/configuration without loading models or writing output")
    args = parser.parse_args(argv)
    if args.flicker_candidates < 25 or args.flicker_candidates > 10000:
        parser.error("--flicker-candidates must be between 25 and 10000")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be positive")

    from configs.config import load_config

    config = load_config(args.config, args.set)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error("--checkpoint must be an existing full UMM weight export file")
    config["model"]["checkpoint"] = str(checkpoint)
    rows, prompt_sha = load_prompts(args.mode)
    tasks = make_tasks(rows, config["sampling"]["seed"], args.flicker_candidates)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    selected = tasks if args.max_videos is None else tasks[:args.max_videos]
    assigned = shard_tasks(selected, rank, world)
    output = Path(args.output).resolve()
    if rank == 0:
        print(json.dumps({"mode": args.mode, "prompts": len(rows),
                          "dimensions": len({d for row in rows for d in row["dimension"]}),
                          "candidate_videos": len(tasks), "scoring_videos": len(rows) * SAMPLES,
                          "this_invocation": len(selected), "world_size": world,
                          "video": config["data"]["video"], "sampling": config["sampling"],
                          "output": str(output)}, indent=2), flush=True)
    if args.dry_run:
        return
    stat = checkpoint.stat()
    run = {"schema_version": 1, "mode": args.mode, "vbench_commit": VBENCH_COMMIT,
           "full_sha256": FULL_SHA256, "prompts_sha256": prompt_sha,
           "checkpoint": {"path": str(checkpoint), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
           "config": config, "seed_rule": "sampling.seed + video_index",
           "flicker_candidates": args.flicker_candidates}
    prepare_run(output, run, (DATA_DIR / f"{args.mode}.json").read_bytes())
    generate(config, assigned, output, rank, int(os.environ.get("LOCAL_RANK", "0")))


if __name__ == "__main__":
    main()
