"""Score GenEval images with the pinned official detector (its own Python 3.9 runtime).

Standard library only: runs after the sampling processes have released the GPUs.
The pinned runtime's torch 1.13 has no kernels for sm_90 and newer, so on
H100/H200 the detector runs on the job's CPUs (see geneval_worker.py); on older
GPUs it uses one GPU per shard as before.
Usage: python3 geneval_detect.py OUTPUT GENEVAL_ROOT
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

COMMIT = 'af4902f24d3ca90ebbb446dd9891a59e0f82725f'


def cpu_quota():
    """CPUs this container may use: the cgroup quota, not the host core count."""
    try:
        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()
        if quota != 'max':
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
        period = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


def memory_quota():
    """Bytes this container may use (cgroup limit), or None when unlimited/unknown."""
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = open(path).read().strip()
        except OSError:
            continue
        if value != 'max' and int(value) < 1 << 60:
            return int(value)
    return None


def detector_env(root):
    env = dict(os.environ, PYTHONPATH=str(root / 'runtime/mmdetection'), XDG_CACHE_HOME=str(root / 'cache'),
               TOKENIZERS_PARALLELISM='false')
    lib = root / 'runtime/env/lib'
    kept = [v for v in env.get('LD_LIBRARY_PATH', '').split(':') if v and 'packages' not in v]
    env['LD_LIBRARY_PATH'] = ':'.join([str(lib / 'python3.9/site-packages/torch/lib'), str(lib), *kept])
    for name in ('LD_PRELOAD', 'PYTHONHOME', 'TORCH_DEVICE_BACKEND_AUTOLOAD'):
        env.pop(name, None)
    return env


def gpu_supported(root, devices):
    """True when the detector's torch build has kernels for this GPU."""
    if not devices:
        return False
    probe = ('import torch,sys;major,minor=torch.cuda.get_device_capability(0);'
             'archs=[a for a in torch.cuda.get_arch_list() if a.startswith("sm_")];'
             'sys.exit(0 if any(int(a[3:-1])==major and int(a[-1])<=minor for a in archs) else 1)')
    env = dict(detector_env(root), CUDA_VISIBLE_DEVICES=devices[0])
    return subprocess.run([str(root / 'runtime/env/bin/python'), '-c', probe], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


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
    on_gpu = gpu_supported(root, devices)
    threads = 2
    # A CPU shard holds the detector and CLIP ViT-L/14 (about 3.5 GiB resident).
    memory = memory_quota()
    by_memory = int(memory * .7 // (4 << 30)) if memory else len(prompts)
    workers = len(devices) if on_gpu else max(1, min(len(prompts), cpu_quota() // threads, by_memory))
    print(f'GenEval detector on {"GPU" if on_gpu else "CPU"}: {workers} shards', flush=True)
    shards = out / 'detector'
    layout = dict(device='gpu' if on_gpu else 'cpu', shards=workers)
    marker = shards / 'LAYOUT.json'
    if shards.exists() and (not marker.is_file() or json.loads(marker.read_text()) != layout):
        # Shards of an attempt with another layout: symlinks and partial rows only.
        shutil.rmtree(shards)
    shards.mkdir(parents=True, exist_ok=True)
    atomic(shards / 'LAYOUT.json', layout)
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
        env = detector_env(root)
        if on_gpu:
            env.update(CUDA_VISIBLE_DEVICES=devices[rank], GENEVAL_DEVICE='cuda', GENEVAL_THREADS='1')
        else:
            env.update(CUDA_VISIBLE_DEVICES='', GENEVAL_DEVICE='cpu', GENEVAL_THREADS=str(threads))
        partial = target.with_suffix('.partial.jsonl')
        partial.unlink(missing_ok=True)
        command = [str(root / 'runtime/env/bin/python'), str(Path(__file__).with_name('geneval_worker.py')),
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
    atomic(out / 'RESULTS.json', dict(**generation, official_commit=COMMIT, detector_device=layout['device'],
                                      task_scores=scores,
                                      overall=sum(scores.values()) / 6, raw_results=str(per_image),
                                      task_images={tag: len(v) for tag, v in groups.items()}))
    print('GenEval', generation['step'], sum(scores.values()) / 6, flush=True)


if __name__ == '__main__':
    out, root = Path(sys.argv[1]), Path(sys.argv[2])
    if not (out / 'RESULTS.json').is_file():
        score(out, root)
