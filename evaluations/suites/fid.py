"""ImageNet FID-50K images: 1,000 class prompts x 50 seeds, scored by fid_score.py as they finish."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch.distributed as dist

from utils.checkpoint_catalog import durable_json

N = 50000
CHUNK = 32


def run(runtime, out, benchmarks, limit=0):
    if (out / 'GENERATED.json').is_file():
        return
    classes = Path(benchmarks) / 'fid/classes.json'
    labels = json.loads(classes.read_text())
    if len(labels) != 1000:
        raise ValueError('Expected the 1,000 ImageNet classes')
    size = runtime.sampling['resolution']
    rank, world = dist.get_rank(), dist.get_world_size()
    total = limit or N
    chunks = (total + CHUNK - 1) // CHUNK
    for k in range(rank, chunks, world):
        tag = f'{k:05d}'
        if (out / 'ready' / f'{tag}.json').exists():
            continue
        directory = out / 'scratch' / tag
        start, end = k * CHUNK, min((k + 1) * CHUNK, total)
        specs = [(labels[i % 1000]['prompt'], 4200000 + i, directory / f'{i:06d}.png') for i in range(start, end)
                 if not (directory / f'{i:06d}.png').is_file()]
        if specs:
            runtime.produce_many(specs)
        for i in range(start, end):
            with Image.open(directory / f'{i:06d}.png') as image:
                if image.size != (size, size) or image.mode != 'RGB':
                    raise ValueError('Unexpected FID sample')
                image.verify()
        durable_json(out / 'ready' / f'{tag}.json', dict(chunk=k, start=start, stop=end, step=runtime.step))
        if k % 160 == 0:
            print(f'step {runtime.step} FID chunk {k}/{chunks}', flush=True)
    dist.barrier()
    if limit:
        if rank == 0:
            durable_json(out / 'SMOKE.json', dict(task='fid', **runtime.describe(), images=total))
        dist.barrier()
        return
    if rank == 0:
        if len(list((out / 'ready').glob('*.json'))) != chunks:
            raise ValueError('FID chunks are missing')
        s = runtime.sampling
        durable_json(out / 'GENERATED.json', dict(task='fid', **runtime.describe(), images=N, classes=1000,
            samples_per_class=50, class_prompt_sha256=hashlib.sha256(classes.read_bytes()).hexdigest(),
            seed='4200000 + sample_index', resolution=s['resolution'], sampling_steps=s['steps'],
            shift=s['shift'], guidance=s['guidance'], chunk_size=CHUNK))
    dist.barrier()
