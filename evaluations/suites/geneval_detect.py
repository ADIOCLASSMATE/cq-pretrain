"""Score GenEval images with the pinned official detector (its own Python 3.9 runtime).

Standard library only: runs after the sampling processes have released the GPUs.
Usage: python3 geneval_detect.py OUTPUT GENEVAL_ROOT
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = 'af4902f24d3ca90ebbb446dd9891a59e0f82725f'


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def score(out, root):
    official = root / 'official-geneval'
    head = subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain', '--untracked-files=no'],
                                    text=True).strip()
    if head != COMMIT or dirty:
        raise ValueError('The official GenEval checkout changed')
    clip = Path.home() / '.cache/clip/ViT-L-14.pt'
    clip.parent.mkdir(parents=True, exist_ok=True)
    if not clip.exists():
        try:
            clip.symlink_to(root / 'cache/clip/ViT-L-14.pt')
        except FileExistsError:
            pass
    generation = json.loads((out / 'GENERATED.json').read_text())
    prompts = [json.loads(line) for line in (official / 'prompts/evaluation_metadata.jsonl').read_text().splitlines()]
    devices = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d] or \
        [str(i) for i in range(int(os.environ.get('NPROC_PER_NODE', '1')))]
    workers = len(devices)
    shards = out / 'detector'
    for i in range(len(prompts)):
        folder = shards / f'rank_{i % workers}' / 'images'
        folder.mkdir(parents=True, exist_ok=True)
        link, source = folder / f'{i:05d}', out / 'images' / f'{i:05d}'
        if not link.is_symlink():
            link.symlink_to(source, target_is_directory=True)

    def worker(rank):
        shard = shards / f'rank_{rank}'
        target = shard / 'rows.jsonl'
        if target.is_file():
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank], PYTHONPATH=str(root / 'runtime/mmdetection'),
                   XDG_CACHE_HOME=str(root / 'cache'), TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        lib = root / 'runtime/env/lib'
        kept = [v for v in env.get('LD_LIBRARY_PATH', '').split(':') if v and 'packages' not in v]
        env['LD_LIBRARY_PATH'] = ':'.join([str(lib / 'python3.9/site-packages/torch/lib'), str(lib), *kept])
        for name in ('LD_PRELOAD', 'PYTHONHOME', 'TORCH_DEVICE_BACKEND_AUTOLOAD'):
            env.pop(name, None)
        partial = target.with_suffix('.partial.jsonl')
        partial.unlink(missing_ok=True)
        command = [str(root / 'runtime/env/bin/python'), str(root / 'geneval_worker.py'),
                   str(official / 'evaluation/evaluate_images.py'), str(shard / 'images'),
                   '--outfile', str(partial), '--model-path', str(root / 'detector')]
        with (shard / 'run.log').open('w') as log:
            subprocess.run(command, cwd=official, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sum(1 for _ in partial.open()) != 4 * len(range(rank, len(prompts), workers)):
            raise ValueError(f'Detector shard {rank} is incomplete')
        partial.replace(target)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    expected = {(i, s) for i in range(len(prompts)) for s in range(4)}
    rows, groups = {}, defaultdict(list)
    for rank in range(workers):
        for line in (shards / f'rank_{rank}' / 'rows.jsonl').read_text().splitlines():
            row = json.loads(line)
            image = Path(row['filename'])
            key = (int(image.parent.parent.name), int(image.stem))
            if key not in expected or key in rows or type(row['correct']) is not bool:
                raise ValueError(f'Unexpected detector row {row["filename"]}')
            if json.loads(row['metadata']) != prompts[key[0]] or row['tag'] != prompts[key[0]]['tag']:
                raise ValueError('Detector metadata disagrees with the prompt')
            rows[key] = row
            groups[row['tag']].append(row['correct'])
    if rows.keys() != expected or len(groups) != 6:
        raise ValueError('GenEval detection is incomplete')
    per_image = out / 'per_image.jsonl'
    per_image.write_text(''.join(json.dumps(rows[k]) + '\n' for k in sorted(rows)))
    scores = {tag: sum(v) / len(v) for tag, v in groups.items()}
    atomic(out / 'RESULTS.json', dict(**generation, official_commit=COMMIT, task_scores=scores,
                                      overall=sum(scores.values()) / 6, raw_results=str(per_image),
                                      task_images={tag: len(v) for tag, v in groups.items()}))
    print('GenEval', generation['step'], sum(scores.values()) / 6, flush=True)


if __name__ == '__main__':
    out, root = Path(sys.argv[1]), Path(sys.argv[2])
    if not (out / 'RESULTS.json').is_file():
        score(out, root)
