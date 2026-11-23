"""GenEval images: 553 official prompts x 4 seeds (42 + 4 * prompt + sample).

Scoring runs afterwards in evaluations/suites/geneval_detect.py, once the
sampling processes have released the GPUs.
"""

import hashlib
import json
from pathlib import Path

from PIL import Image
import torch.distributed as dist

from evaluations.suites.t2i import Sampler
from utils.checkpoint_catalog import durable_json

PROMPTS, SAMPLES = 553, 4


def run(runtime, out, geneval_root, limit=0):
    if (out / "GENERATED.json").is_file():
        return
    raw = (Path(geneval_root) / "official-geneval/prompts/evaluation_metadata.jsonl").read_bytes()
    prompts = [json.loads(line) for line in raw.splitlines()]
    if len(prompts) != PROMPTS or len({p["prompt"] for p in prompts}) != PROMPTS:
        raise ValueError("Unexpected GenEval prompt file")
    count = limit or PROMPTS
    sampler = Sampler(runtime)
    rank, world = dist.get_rank(), dist.get_world_size()
    mine = list(range(rank, count, world))
    group = max(1, int(runtime.sampling.get("batch", 32)) // SAMPLES)
    for start in range(0, len(mine), group):
        specs = []
        for i in mine[start : start + group]:
            row, folder = prompts[i], out / "images" / f"{i:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            meta = folder / "metadata.jsonl"
            if not meta.is_file():
                temporary = folder / "metadata.jsonl.tmp"
                temporary.write_text(json.dumps(row, ensure_ascii=False) + "\n")
                temporary.replace(meta)
            elif json.loads(meta.read_text()) != row:
                raise ValueError(f"GenEval metadata changed for prompt {i}")
            specs += [
                (row["prompt"], 42 + i * SAMPLES + s, folder / "samples" / f"{s:05d}.png")
                for s in range(SAMPLES)
                if not (folder / "samples" / f"{s:05d}.png").exists()
            ]
        if specs:
            sampler(specs)
        if start % (16 * group) == 0:
            print(f"step {runtime.step} GenEval prompts {start + 1}/{len(mine)} on rank {rank}", flush=True)
    dist.barrier()
    if limit:
        if rank == 0:
            durable_json(out / "SMOKE.json", dict(task="geneval", **runtime.describe(), prompts=count))
        dist.barrier()
        return
    if rank == 0:
        size = runtime.sampling["resolution"]
        for i in range(PROMPTS):
            files = sorted((out / "images" / f"{i:05d}" / "samples").glob("*.png"))
            if [f.name for f in files] != [f"{s:05d}.png" for s in range(SAMPLES)]:
                raise ValueError(f"GenEval prompt {i} is missing images")
            for file in files:
                with Image.open(file) as image:
                    if image.size != (size, size):
                        raise ValueError(f"Unexpected GenEval image size {image.size}")
        s = runtime.sampling
        durable_json(
            out / "GENERATED.json",
            dict(
                task="geneval",
                **runtime.describe(),
                images=PROMPTS * SAMPLES,
                prompts=PROMPTS,
                samples_per_prompt=SAMPLES,
                official_prompt_sha256=hashlib.sha256(raw).hexdigest(),
                resolution=s["resolution"],
                steps=s["steps"],
                guidance=s["guidance"],
                shift=s["shift"],
                seed="42 + 4 * prompt_index + sample",
            ),
        )
        print(f"step {runtime.step} GenEval images complete", flush=True)
    dist.barrier()
