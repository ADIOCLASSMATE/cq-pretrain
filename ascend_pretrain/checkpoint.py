"""Atomic per-rank restart checkpoints, including local FP32 Adam/master shards.

This format deliberately requires the same TP/DP topology on resume. It does not
gather all optimizer shards onto rank zero or silently restart without them.
"""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from . import parallel as P
from .checkpoint_catalog import durable_json, fsync_directory, shard_names


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _optimizers(optimizer):
    return getattr(optimizer, 'chained_optimizers', [optimizer])


def _optimizer_state(opt):
    return dict(inner=opt.optimizer.state_dict(),
                masters=[[p.detach() for p in group['params']] for group in opt.optimizer.param_groups])


def validation_probes(model, optimizer):
    """Small deterministic probes for restart validation, outside speed tests."""
    values=[];layout=[]
    def add(name,tensor):
        if not tensor.numel():return
        flat=tensor.detach().flatten()
        selected=flat[::max(1,flat.numel()//16)][:16].float()
        values.append(selected);layout.append((name,tuple(tensor.shape),selected.numel()))
    for name,parameter in model.named_parameters():add('model/'+name,parameter)
    for i,opt in enumerate(_optimizers(optimizer)):
        for j,group in enumerate(opt.optimizer.param_groups):
            for k,parameter in enumerate(group['params']):
                add(f'master/{i}/{j}/{k}',parameter)
                for name,state in opt.optimizer.state[parameter].items():
                    if isinstance(state,torch.Tensor) and state.numel()>1:add(f'adam/{i}/{j}/{k}/{name}',state)
    return dict(layout=layout,values=torch.cat(values).cpu())


def _save_file(value, path):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(_cpu(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def save(path, model, optimizer, step, config, topology, training_state=None, data_state=None, ema=None):
    root = Path(path)
    if (root / 'manifest.json').exists():
        raise FileExistsError(f'Refusing to overwrite a committed checkpoint: {root}')
    root.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    tp_rank = P.state().get_tensor_model_parallel_rank()
    # BF16 model weights are replicated across DP. Write them exactly once per
    # TP shard, instead of writing 512 copies on a TP1/DP512 checkpoint.
    if P.dp_rank() == 0:
        _save_file(model.state_dict(), root / f'model_tp_{tp_rank:03d}.pt')
    schema=6 if ema is not None else 5
    payload = dict(step=step, config=config.to_dict(), tp_rank=tp_rank,
        topology=topology, cpu_rng=torch.get_rng_state(), npu_rng=torch.npu.get_rng_state(),
        schema=schema, data_state=data_state, optimizer=[_optimizer_state(opt) for opt in _optimizers(optimizer)])
    if ema is not None:payload['ema']=ema.state_dict()
    _save_file(payload, root / f'rank_{rank:05d}.pt')
    dist.barrier()
    if rank == 0:
        files = {name: (root/name).stat().st_size for name in shard_names(dist.get_world_size(), topology['tp'])}
        if any(size <= 0 for size in files.values()):
            raise RuntimeError('Cannot commit an incomplete checkpoint')
        manifest=dict(schema=schema,step=step,world=dist.get_world_size(),
            topology=topology,config=config.to_dict(),training_state=training_state,files=files)
        if ema is not None:manifest['ema']=ema.metadata()
        durable_json(root/'manifest.json',manifest)
    dist.barrier()


def load(path, model, optimizer, config, topology, training_state=None, data_state_out=None, ema=None):
    root = Path(path)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') not in (3, 4, 5, 6):
        raise ValueError('Unsupported checkpoint schema; refusing a partial optimizer restore')
    if (manifest['schema']==6)!=('ema' in manifest):
        raise ValueError('Checkpoint schema and renderer EMA metadata disagree')
    if ('ema' in manifest)!=(ema is not None):
        raise ValueError('Resume must preserve whether renderer EMA is enabled')
    if manifest['world'] != dist.get_world_size() or manifest['topology'] != topology:
        raise ValueError('Resume requires the saved TP/DP topology')
    if manifest['config'] != config.to_dict():
        raise ValueError('Resume model configuration differs from the checkpoint')
    if manifest.get('training_state') != training_state:
        raise ValueError('Resume schedule, batch or data seed differs from the checkpoint')
    saved = torch.load(root / f'rank_{dist.get_rank():05d}.pt', map_location='cpu', weights_only=False)
    if ('ema' in saved)!=('ema' in manifest):
        raise ValueError('Renderer EMA shard and manifest disagree')
    if saved.get('schema')!=manifest['schema']:
        raise ValueError('Checkpoint shard and manifest schemas disagree')
    if data_state_out is not None:
        if saved.get('data_state') is None:
            raise ValueError('Real-data resume requires a checkpoint containing consumed data state')
        data_state_out.update(saved['data_state'])
    if P.dp_rank() == 0:
        state = torch.load(root / f"model_tp_{saved['tp_rank']:03d}.pt", map_location='cpu', weights_only=False)
        model.load_state_dict(state, strict=True)
    group = P.state().get_data_parallel_group()
    source = dist.get_global_rank(group, 0)
    for tensor in model.state_dict().values():
        # MCore linears include a stateless `_extra_state=None` entry.
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Non-tensor model extra state needs an explicit broadcast adapter')
        dist.broadcast(tensor, src=source, group=group)
    if len(_optimizers(optimizer)) != len(saved['optimizer']):
        raise ValueError('Optimizer partition differs from the checkpoint')
    for opt, state in zip(_optimizers(optimizer), saved['optimizer']):
        opt.optimizer.load_state_dict(state['inner'])
        with torch.no_grad():
            for group, masters in zip(opt.optimizer.param_groups, state['masters']):
                if len(group['params']) != len(masters):
                    raise ValueError('Optimizer shard layout differs from checkpoint')
                for parameter, master in zip(group['params'], masters):
                    parameter.copy_(master)
    if ema is not None:
        if 'ema' not in saved:raise ValueError('Checkpoint is missing its renderer EMA shard')
        ema.load_state_dict(saved['ema'])
        if ema.metadata()!=manifest['ema']:raise ValueError('EMA metadata differs across checkpoint shards')
    torch.set_rng_state(saved['cpu_rng'])
    torch.npu.set_rng_state(saved['npu_rng'])
    return saved['step']


def initialization_model_config(config):
    """Weight semantics for an explicit model-only new-stage initialization.

    Packing and DiT activation scheduling carry no learned tensor dimensions.
    They may change for a fresh optimizer/cursor; exact resume above continues
    to compare the entire config and training/data contract without exclusions.
    """
    execution={'checkpoint_backbone','checkpoint_dit','checkpoint_backbone_layers',
        'checkpoint_dit_layers','ce_chunk','keep_attention_activations','preserve_documents',
        'long_document_policy','generation_token_budget','generation_microbatch_images',
        # CQ query count selects how many read-only positions each image emits.
        # It allocates no learned tensor: queries reuse the shared image-mask
        # embedding and the backbone/DiT weights are identical at any Q. A new
        # stage may therefore change Q with a fresh optimizer and data cursor.
        'image_condition_queries'}
    return {k:v for k,v in config.items() if k not in execution}


def initialize_model(path, model, config):
    """Explicit new stage: restore model only, with fresh optimizer and cursor.

    A different corpus/batch/DP world cannot be represented as exact resume.
    Semantic settings must match. TP model slices can be exactly repartitioned;
    this does not relax the strict topology requirement for optimizer resume.
    """
    root=Path(path)
    manifest=json.loads((root/'manifest.json').read_text())
    source_tp=manifest['topology']['tp']
    saved=initialization_model_config(manifest['config'])
    current=initialization_model_config(config.to_dict())
    if saved!=current:
        changed=sorted(k for k in saved.keys()|current.keys() if saved.get(k)!=current.get(k))
        raise ValueError(f'Initialization semantic model config differs from checkpoint: {changed}')
    if P.dp_rank()==0:
        rank=P.state().get_tensor_model_parallel_rank()
        def read(index):
            name=f'model_tp_{index:03d}.pt'
            if (root/name).stat().st_size!=manifest['files'][name]:raise ValueError('Initialization shard size mismatch')
            return torch.load(root/name,map_location='cpu',weights_only=False,mmap=True)
        if source_tp==P.tp_size():state=read(rank)
        else:
            from .checkpoint_reshard import reshard_state
            shards=[read(i) for i in range(source_tp)]
            state=reshard_state(shards,model,P.tp_size(),rank)
            del shards
        model.load_state_dict(state,strict=True)
        del state
    group=P.state().get_data_parallel_group()
    source=dist.get_global_rank(group,0)
    for tensor in model.state_dict().values():
        if isinstance(tensor,torch.Tensor):dist.broadcast(tensor,src=source,group=group)
    return dict(path=str(root),source_step=manifest['step'],source_tp=source_tp,target_tp=P.tp_size(),restored=['model'],new_stage=True,
                optimizer='fresh',data_cursor='new dataset start')
