"""Durable model-only snapshots for later benchmarks, separate from restart state.

All training ranks call ``save`` at an update boundary. Only DP rank zero writes
each TP model shard. A manifest is the commit marker; this module never prunes.
The caller owns the destination exclusively, just as for restart checkpoints.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch.distributed as dist

from . import parallel as P
from .checkpoint import _save_file
from .checkpoint_catalog import durable_json, fsync_directory


SCHEMA = 'cq_eval_snapshot_v1'
KIND = 'inference_weights'


def inspect_snapshot(path):
    """Validate a committed snapshot without loading its potentially large tensors."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Evaluation snapshot must be a real directory')
    marker = root/'manifest.json'
    if marker.is_symlink():
        raise ValueError('Evaluation snapshot manifest must be a regular file')
    manifest = json.loads(marker.read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('kind') != KIND:
        raise ValueError('Unsupported evaluation snapshot manifest')
    step, tp = manifest['step'], manifest['topology']['tp']
    if type(step) is not int or step < 0 or type(tp) is not int or tp < 1:
        raise ValueError('Invalid evaluation snapshot step or TP size')
    names = {f'model_tp_{index:03d}.pt' for index in range(tp)}
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != names:
        raise ValueError('Incomplete evaluation snapshot inventory')
    for name, size in files.items():
        shard = root/name
        if type(size) is not int or size <= 0 or shard.is_symlink() or not shard.is_file():
            raise ValueError(f'Missing or invalid evaluation shard: {name}')
        if shard.stat().st_size != size:
            raise ValueError(f'Truncated evaluation shard: {name}')
    return manifest


def _error(error):
    return f'{type(error).__name__}: {error}'


def _require_success(error, phase):
    """Propagate ordinary rank-local failures before the next collective phase."""
    errors = [None]*dist.get_world_size()
    dist.all_gather_object(errors, error)
    failures = [f'rank {rank}: {value}' for rank, value in enumerate(errors) if value]
    if failures:
        raise RuntimeError(f'Evaluation snapshot {phase} failed: ' + '; '.join(failures))


def _on_rank_zero(action):
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = {'value': action()}
        except Exception as error:
            result[0] = {'error': _error(error)}
    dist.broadcast_object_list(result, src=0)
    if 'error' in result[0]:
        raise RuntimeError('Evaluation snapshot failed: ' + result[0]['error'])
    return result[0]['value']


def save(path, model, wrapped, step, config, topology, ema=None, provenance=None):
    """Save online backbone plus EMA renderer (when enabled), without training state.

    Model tensors retain their inference dtype: FP32 EMA masters are installed
    into the same model buffers used by validation. Synchronization and forward
    hooks are restored before committing. Committed snapshots are immutable;
    an intact matching snapshot is an idempotent success. An interrupted write
    without a manifest can be retried at the same path.
    """
    begin = time.perf_counter()
    root = Path(path)
    sources = dict(backbone='online', renderer='ema' if ema is not None else 'online')
    if ema is not None:
        sources['ema'] = ema.metadata()
    # Normalize tuples and validate serializability before changing model state.
    specification = json.loads(json.dumps(dict(schema=SCHEMA, kind=KIND, step=step,
        config=config.to_dict(), topology=topology, weight_sources=sources,
        provenance=provenance or {})))
    tp = specification['topology']['tp']

    def prepare():
        if type(step) is not int or step < 0 or type(tp) is not int or tp < 1:
            raise ValueError('Require a nonnegative step and positive TP size')
        if tp != P.tp_size() or dist.get_world_size() % tp:
            raise ValueError('Evaluation snapshot TP does not match the active model')
        if root.is_symlink():
            raise ValueError('Evaluation snapshot must not be a symlink')
        if (root/'manifest.json').exists() or (root/'manifest.json').is_symlink():
            existing = inspect_snapshot(root)
            # DP and attempt provenance can change after a resume; neither alters
            # these replicated model tensors. Never rewrite an existing commit.
            for key in ('step', 'config', 'weight_sources'):
                if existing[key] != specification[key]:
                    raise ValueError(f'Committed evaluation snapshot differs: {key}')
            if existing['topology']['tp'] != tp:
                raise ValueError('Committed evaluation snapshot differs: TP size')
            return existing
        root.mkdir(parents=True, exist_ok=True)
        fsync_directory(root.parent)
        return None

    existing = _on_rank_zero(prepare)
    if existing is not None:
        return dict(path=str(root), step=step, skipped=True, files=existing['files'],
                    weight_sources=existing['weight_sources'], seconds=time.perf_counter()-begin)

    error = None
    hooks_disabled = False
    handles = getattr(wrapped, 'remove_forward_pre_hook_handles', None)
    hooks_active = bool(handles) if handles is not None else wrapped.use_forward_hook
    try:
        sync_error = None
        try:
            wrapped.start_param_sync(force_sync=True)
        except Exception as failure:
            sync_error = _error(failure)
        _require_success(sync_error, 'parameter synchronization')

        hook_error = None
        if hooks_active:
            try:
                hooks_disabled = True
                wrapped.disable_forward_pre_hook(param_sync=False)
            except Exception as failure:
                hook_error = _error(failure)
        _require_success(hook_error, 'hook suspension')

        with ema.average_parameters() if ema is not None else nullcontext():
            write_error = None
            try:
                if P.dp_rank() == 0:
                    rank = P.state().get_tensor_model_parallel_rank()
                    destination = root/f'model_tp_{rank:03d}.pt'
                    if destination.with_name(destination.name+'.tmp').is_symlink():
                        raise ValueError('Evaluation shard temporary file must not be a symlink')
                    # Complete the CPU copy AND serialization while EMA is active;
                    # state_dict contains references to buffers restored on exit.
                    _save_file(model.state_dict(), destination)
            except Exception as failure:
                write_error = _error(failure)
            # Every rank leaves the EMA context together, including on I/O error.
            _require_success(write_error, 'model write')
    except Exception as failure:
        error = _error(failure)
    finally:
        if hooks_disabled:
            try:
                wrapped.enable_forward_pre_hook()
            except Exception as failure:
                error = '; '.join(filter(None, (error, 'hook restore: '+_error(failure))))
    _require_success(error, 'export or model restoration')

    def commit():
        if (root/'manifest.json').exists():
            raise FileExistsError('Evaluation snapshot was committed by another writer')
        files = {}
        for index in range(tp):
            name = f'model_tp_{index:03d}.pt'
            shard = root/name
            if shard.is_symlink() or not shard.is_file() or shard.stat().st_size <= 0:
                raise ValueError(f'Missing or invalid evaluation shard: {name}')
            files[name] = shard.stat().st_size
        manifest = dict(specification, files=files)
        if (root/'manifest.json.tmp').is_symlink():
            raise ValueError('Evaluation manifest temporary file must not be a symlink')
        durable_json(root/'manifest.json', manifest)
        return manifest

    manifest = _on_rank_zero(commit)
    return dict(path=str(root), step=step, skipped=False, files=manifest['files'],
                weight_sources=manifest['weight_sources'], seconds=time.perf_counter()-begin)
