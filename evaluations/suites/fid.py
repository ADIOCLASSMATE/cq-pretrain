"""ImageNet FID-50K images: image i uses class i % 1000 and seed 4200000 + i.

Work is split by class, so the 50 images of a class share one backbone
condition and one DiT batch. fid_score.py extracts Inception features on the
CPU while sampling continues, one finished class at a time.
"""

import hashlib
import json
from pathlib import Path

from PIL import Image
import torch.distributed as dist

from evaluations.suites.t2i import Sampler
from utils.checkpoint_catalog import durable_json

N, CLASSES = 50000, 1000


def run(runtime, out, benchmarks, limit=0):
    if (out / "GENERATED.json").is_file():
        return
    classes = Path(benchmarks) / "fid/classes.json"
    labels = json.loads(classes.read_text())
    if len(labels) != CLASSES:
        raise ValueError("Expected the 1,000 ImageNet classes")
    per_class = N // CLASSES if not limit else max(1, limit // CLASSES)
    units = CLASSES if not limit else min(CLASSES, limit)
    size = runtime.sampling["resolution"]
    sampler = Sampler(runtime)
    rank, world = dist.get_rank(), dist.get_world_size()
    for c in range(rank, units, world):
        marker = out / "ready" / f"c{c:04d}.json"
        if marker.exists():
            continue
        directory = out / "scratch" / f"c{c:04d}"
        indices = [c + CLASSES * j for j in range(per_class)]
        specs = [
            (labels[c]["prompt"], 4200000 + i, directory / f"{i:06d}.png")
            for i in indices
            if not (directory / f"{i:06d}.png").is_file()
        ]
        if specs:
            sampler(specs)
        for i in indices:
            with Image.open(directory / f"{i:06d}.png") as image:
                if image.size != (size, size) or image.mode != "RGB":
                    raise ValueError("Unexpected FID sample")
        durable_json(marker, dict(class_index=c, images=indices, step=runtime.step))
        if c % 100 == rank:
            print(f"step {runtime.step} FID class {c + 1}/{units}", flush=True)
    dist.barrier()
    if limit:
        if rank == 0:
            durable_json(out / "SMOKE.json", dict(task="fid", **runtime.describe(), classes=units, per_class=per_class))
        dist.barrier()
        return
    if rank == 0:
        if len(list((out / "ready").glob("c*.json"))) != CLASSES:
            raise ValueError("FID classes are missing")
        s = runtime.sampling
        durable_json(
            out / "GENERATED.json",
            dict(
                task="fid",
                **runtime.describe(),
                images=N,
                classes=CLASSES,
                samples_per_class=N // CLASSES,
                class_prompt_sha256=hashlib.sha256(classes.read_bytes()).hexdigest(),
                seed="4200000 + sample_index",
                class_of_sample="sample_index % 1000",
                resolution=s["resolution"],
                sampling_steps=s["steps"],
                shift=s["shift"],
                guidance=s["guidance"],
            ),
        )
    dist.barrier()
