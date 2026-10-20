"""Committed checkpoint discovery and bounded retention; no accelerator imports."""
import json
import os
from pathlib import Path
import re
import shutil


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def shard_names(world, tp):
    if not isinstance(world, int) or not isinstance(tp, int) or tp < 1 or world < tp or world % tp:
        raise ValueError('Invalid checkpoint topology')
    return [f'model_tp_{i:03d}.pt' for i in range(tp)] + [f'rank_{i:05d}.pt' for i in range(world)]


def inspect_checkpoint(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Checkpoint must be a real directory')
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema') not in (3, 4, 5, 6):
        raise ValueError('Unsupported checkpoint schema')
    if (manifest['schema']==6)!=('ema' in manifest):
        raise ValueError('Checkpoint schema and renderer EMA metadata disagree')
    step = manifest['step']
    if not isinstance(step, int) or step < 0 or path.name != f'step_{step:07d}':
        raise ValueError('Checkpoint step/directory mismatch')
    names = shard_names(manifest['world'], manifest['topology']['tp'])
    if manifest['topology']['dp'] * manifest['topology']['tp'] != manifest['world']:
        raise ValueError('Invalid saved DP topology')
    sizes = manifest.get('files')
    if manifest['schema'] >= 5 and (not isinstance(sizes, dict) or set(sizes) != set(names)):
        raise ValueError('Incomplete checkpoint inventory')
    for name in names:
        shard = path / name
        if shard.is_symlink() or not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f'Missing/empty shard: {name}')
        if sizes is not None and shard.stat().st_size != sizes[name]:
            raise ValueError(f'Truncated shard: {name}')
    return manifest


def checkpoint_directories(root):
    root = Path(root)
    return sorted((p for p in root.glob('step_*')
                   if re.fullmatch(r'step_\d{7,}', p.name) and p.is_dir() and not p.is_symlink()),
                  key=lambda p: int(p.name[5:]), reverse=True)


def discover(root):
    committed, rejected = [], []
    for path in checkpoint_directories(root):
        try:
            manifest = inspect_checkpoint(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append(dict(path=str(path), reason=str(error), committed=(path/'manifest.json').exists()))
        else:
            committed.append((path, manifest))
    return committed, rejected


def latest(root):
    committed, rejected = discover(root)
    if not committed and any(item['committed'] for item in rejected):
        raise RuntimeError(f'No intact committed checkpoint; refusing to restart from pretrained weights: {rejected}')
    return (str(committed[0][0]) if committed else None), rejected


def prune(root, keep):
    """Call only with exclusive ownership, after a new checkpoint is committed."""
    if keep < 1:
        raise ValueError('Retain at least one checkpoint')
    committed, _ = discover(root)
    if not committed:
        raise RuntimeError('Refusing checkpoint cleanup without an intact committed checkpoint')
    retained = {p for p, _ in committed[:keep]}
    removed = []
    for path in checkpoint_directories(root):
        if path not in retained:
            shutil.rmtree(path)
            removed.append(str(path))
    fsync_directory(root)
    return removed


def remove_incomplete(root):
    """Remove only uncommitted attempts, while no training writers are running."""
    for path in checkpoint_directories(root):
        if not (path/'manifest.json').exists():
            shutil.rmtree(path)
