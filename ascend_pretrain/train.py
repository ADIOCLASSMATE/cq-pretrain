"""Distributed pretraining with local online-tokenized DCVLM or synthetic controls."""
import argparse
from contextlib import nullcontext
import json
import hashlib
import math
import os
from pathlib import Path
import socket
import statistics
import time
import torch
import torch.distributed as dist

from .config import ModelConfig, tiny_config
from .data import fake_batch
from . import parallel as P
from . import checkpoint as checkpoint_io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--vision-assets',help='Local pinned Qwen3.5 vision assets; training never downloads weights')
    parser.add_argument('--text-assets',help='Local pinned Qwen3.5 language checkpoint and tokenizer')
    parser.add_argument('--random-init',action='store_true',help='Explicit architecture control; do not load pretrained language/vision weights')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--method', choices=('CQ_IT','Bi_I+CQ_T'))
    parser.add_argument('--image-condition-queries', type=int)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--sequence-length', type=int, default=8192)
    parser.add_argument('--images', type=int, default=2)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--global-batch', type=int)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--total-steps', type=int, help='Stop at this absolute update count, including resume')
    parser.add_argument('--until-data-end',action='store_true',help='Traverse the remaining finite DCVLM epoch instead of a slot-based step budget')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--moss-ema-decay',type=float,default=0.,
                        help='FP32 distributed optimizer-shard EMA for MOSS; zero disables')
    parser.add_argument('--min-lr', type=float, help='Cosine floor; defaults to one tenth of --lr')
    parser.add_argument('--lr-decay-style', choices=('constant', 'cosine'), default='constant')
    parser.add_argument('--lr-warmup-steps', type=int, default=0)
    parser.add_argument('--lr-decay-steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init-checkpoint',type=Path,help='New-stage model-only initialization; fresh optimizer and data cursor')
    parser.add_argument('--validation-suite',type=Path,help='Fixed local T2I/I2T qualitative suite')
    parser.add_argument('--validate-every',type=int,default=0)
    parser.add_argument('--validate-first',action='store_true')
    parser.add_argument('--validate-final',action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--checkpoint-dir', type=Path, help='Shared checkpoint root, independent of attempt logs')
    parser.add_argument('--keep-checkpoints', type=int, default=3)
    parser.add_argument('--save-first', action='store_true', help='Save after the first successful update')
    parser.add_argument('--save-final', action='store_true', help='Save the last update on normal completion')
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument('--resume')
    resume.add_argument('--auto-resume', action='store_true', help='Resume the newest committed checkpoint in checkpoint-dir')
    parser.add_argument('--validation-probes',action='store_true',help='Write model/Adam/master probes after each update; exclude this run from throughput estimates')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-ranks', default='0', help='Comma separated global ranks')
    parser.add_argument('--bucket-size', type=int, default=40_000_000, help='DDP bucket elements')
    parser.add_argument('--no-param-overlap', action='store_true')
    parser.add_argument('--phase-timing', action='store_true', help='Measure forward/backward/optimizer device time')
    parser.add_argument('--checkpoint-backbone-layers', type=int, default=0)
    parser.add_argument('--checkpoint-dit-layers', type=int, default=0)
    parser.add_argument('--keep-attention-activations',type=int,default=0)
    parser.add_argument('--optimizer-instances', type=int, default=1,
                        help='Replicated optimizer groups for hierarchical DP synchronization')
    parser.add_argument('--ce-chunk', type=int, help='Text output projection/CE chunk size')
    parser.add_argument('--deterministic',action='store_true',help='Use deterministic kernels for exact restart validation')
    recompute = parser.add_mutually_exclusive_group()
    recompute.add_argument('--no-checkpoint', action='store_true')
    recompute.add_argument('--checkpoint', action='store_true', help='Recompute backbone and DiT layers')
    parser.add_argument('--split-reference', action='store_true')
    parser.add_argument('--data-root', '--data-snapshot', dest='data_snapshot', help='Published local DCVLM corpus; tokenize raw text online')
    parser.add_argument('--data-mode', choices=('A','B'), default='A')
    parser.add_argument('--allow-development-data', action='store_true')
    parser.add_argument('--data-workers', type=int, default=2)
    parser.add_argument('--data-prefetch', type=int, default=0, help='CPU packs ahead; zero selects GA+4 (minimum 4)')
    parser.add_argument('--source-root', help='Optional local TAR pool relocation')
    parser.add_argument('--tokenize-rows', type=int, default=32)
    parser.add_argument('--shuffle-block-rows', type=int, default=4)
    parser.add_argument('--image-workers', type=int, default=2)
    parser.add_argument('--image-prefetch', type=int, default=16)
    parser.add_argument('--device-prefetch', type=int, default=2, help='Prepared NPU batches ahead; zero is synchronous control')
    parser.add_argument('--preserve-documents',action='store_true',help='Pack complete documents; budget DiT image/CQ pairs independently')
    parser.add_argument('--long-document-policy',choices=('window','error'),help='Policy only for documents longer than the backbone context')
    parser.add_argument('--dit-microbatch-tokens',type=int,help='Total noisy-image plus CQ tokens per DiT forward/backward')
    parser.add_argument('--dit-microbatch-images',type=int,help='Optional DiT image count limit; zero is unlimited')
    parser.add_argument('--max-images-per-pack', type=int, default=None, help='Legacy fragment packing only; defaults to zero for whole documents')
    parser.add_argument('--pack-padding-multiple', type=int, default=128, help='Trim unused pack tail and round to this multiple; zero disables')
    parser.add_argument('--latent-encoder', help='fake, kl16, or module:factory returning scaled posterior mean/std')
    parser.add_argument('--latent-options', type=json.loads, default={})
    parser.add_argument('--vae-path')
    parser.add_argument('--vae-dtype', choices=('float32','bfloat16'), default='float32')
    parser.add_argument('--vae-chunk', type=int, default=2)
    args=parser.parse_args()
    if args.min_lr is None:args.min_lr=args.lr*.1
    if not math.isfinite(args.lr) or args.lr<=0 or not math.isfinite(args.min_lr) or not 0<=args.min_lr<=args.lr:
        parser.error('Require finite lr > 0 and 0 <= min-lr <= lr')
    if not math.isfinite(args.moss_ema_decay) or not 0<=args.moss_ema_decay<1:
        parser.error('Require finite 0 <= moss-ema-decay < 1')
    return args


def learning_rate(step, args):
    if step < args.lr_warmup_steps:
        return args.lr * (step + 1) / args.lr_warmup_steps
    if args.lr_decay_style == 'constant':
        return args.lr
    progress = min(1., max(0., (step - args.lr_warmup_steps) /
        max(1, args.lr_decay_steps - args.lr_warmup_steps)))
    return args.min_lr + .5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    if args.save_every < 0 or args.keep_checkpoints < 1:
        raise ValueError('Require save-every >= 0 and keep-checkpoints >= 1')
    if args.until_data_end:
        if not args.data_snapshot:raise ValueError('--until-data-end requires a finite --data-root')
        args.total_steps=None
        args.steps=(1<<63)-1
    budget=args.total_steps if args.total_steps is not None else args.steps
    if budget <= args.warmup or args.warmup<0 or args.micro_batch < 1 or args.grad_accum < 1:
        raise ValueError('Require steps > warmup and positive batch/accumulation sizes')
    if args.profile:
        # The profiler swallows callback-construction errors. Reject its
        # unsupported output characters before allocating the training model.
        from torch_npu.utils._path_manager import PathManager
        PathManager._input_path_common_check(str(Path(args.output).resolve()))
    if args.config and args.tiny:raise ValueError('Select either an explicit model config or the tiny validation model')
    cfg = (tiny_config(linear_heads=16,linear_head_dim=128) if args.tiny else
           ModelConfig.load(args.config or Path(__file__).parent/'configs/qwen35_9b_moss.json'))
    if args.vision_assets is not None:cfg.vision_assets=args.vision_assets
    if args.text_assets is not None:cfg.text_assets=args.text_assets
    if args.method is not None: cfg.method=args.method
    if args.image_condition_queries is not None: cfg.image_condition_queries=args.image_condition_queries
    if args.preserve_documents:cfg.preserve_documents=True
    if args.long_document_policy is not None:cfg.long_document_policy=args.long_document_policy
    if args.dit_microbatch_tokens is not None:cfg.generation_token_budget=args.dit_microbatch_tokens
    if args.dit_microbatch_images is not None:cfg.generation_microbatch_images=args.dit_microbatch_images
    if not cfg.preserve_documents and (args.dit_microbatch_tokens is not None or args.dit_microbatch_images is not None):
        raise ValueError('DiT microbatch flags require --preserve-documents or the whole-document config')
    if args.max_images_per_pack is None:args.max_images_per_pack=0 if cfg.preserve_documents else 4
    if cfg.preserve_documents and (not args.data_snapshot or args.max_images_per_pack):
        raise ValueError('Whole-document training needs real data and max-images-per-pack=0; bound DiT separately')
    if args.data_snapshot:
        from .input_pipeline.catalog import Catalog
        from .input_pipeline.packing import token_contract
        catalog = Catalog(args.data_snapshot, args.source_root, args.shuffle_block_rows)
        membership = catalog.check(args.allow_development_data)
        data_tokens = token_contract(catalog.path, cfg)
        if args.micro_batch != 1:
            raise ValueError('Real variable-layout adapter currently requires the requested microbatch=1')
        if args.data_mode == 'A' and not (args.vae_path or args.latent_encoder):
            raise ValueError('Multimodal data requires --latent-encoder (fake for tests) or --vae-path')
    if args.ce_chunk is not None:
        cfg.ce_chunk = args.ce_chunk
    cfg.keep_attention_activations=args.keep_attention_activations
    if args.no_checkpoint:
        cfg.checkpoint_backbone = cfg.checkpoint_dit = False
    if args.checkpoint:
        cfg.checkpoint_backbone = cfg.checkpoint_dit = True
    if args.checkpoint_backbone_layers or args.checkpoint_dit_layers:
        cfg.checkpoint_backbone = cfg.checkpoint_dit = False
        cfg.checkpoint_backbone_layers = args.checkpoint_backbone_layers
        cfg.checkpoint_dit_layers = args.checkpoint_dit_layers
    cfg.validate(args.tp)
    from .qwen35 import asset_manifest,cast_for_training,load_pretrained
    text_manifest=asset_manifest(cfg.text_assets) if cfg.text_assets else None
    vision_manifest=None
    if cfg.vision_encoder=='qwen35':
        from .vision import asset_path, vision_config
        if args.micro_batch!=1:raise ValueError('Native-resolution packing requires micro_batch=1')
        vision_config(cfg.vision_assets)
        vision_manifest=json.loads((asset_path(cfg.vision_assets)/'manifest.json').read_text())
        asset_manifest(cfg.vision_assets)
    if args.deterministic:
        os.environ['HCCL_DETERMINISTIC']='True'
        os.environ['CLOSE_MATMUL_K_SHIFT']='1'
        torch.use_deterministic_algorithms(True)
    complete_pretrained = (cfg.renderer == 'moss' and bool(cfg.text_assets)
        and cfg.vision_encoder == 'qwen35' and not args.random_init)
    skip_initialization = bool(args.resume or args.init_checkpoint) or complete_pretrained
    P.initialize(args.tp, cfg, optimizer_instances=args.optimizer_instances,
                 initialize_weights=not skip_initialization)
    from .model import JointModel
    import torch_npu
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    rank, world = dist.get_rank(), dist.get_world_size()
    profile_ranks={int(r) for r in args.profile_ranks.split(',')}
    if args.profile and (not profile_ranks or min(profile_ranks)<0 or max(profile_ranks)>=world):
        raise ValueError('Profile ranks must be a nonempty subset of the training world')
    dp = P.dp_size()
    if args.global_batch:
        if args.global_batch % (dp * args.micro_batch):
            raise ValueError('global_batch must be divisible by DP * micro_batch')
        args.grad_accum = args.global_batch // (dp * args.micro_batch)
    global_batch = args.micro_batch * dp * args.grad_accum
    if args.data_prefetch==0:args.data_prefetch=max(4,args.grad_accum+4)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    checkpoint_root = args.checkpoint_dir or output
    if args.auto_resume:
        from .checkpoint_catalog import latest, remove_incomplete, prune
        selection = [None]
        if rank == 0:
            try:
                path, rejected = latest(checkpoint_root)
                remove_incomplete(checkpoint_root)
                if path:prune(checkpoint_root, args.keep_checkpoints)
                selection[0] = dict(path=path, skipped=rejected)
            except Exception as error:
                selection[0] = dict(error=str(error))
        dist.broadcast_object_list(selection, src=0)
        if 'error' in selection[0]:raise RuntimeError(selection[0]['error'])
        args.resume = selection[0]['path']
        skip_initialization = bool(args.resume or args.init_checkpoint) or complete_pretrained
        if rank == 0:print(json.dumps(dict(checkpoint_selection=selection[0])), flush=True)
    startup_begin = time.perf_counter()
    if skip_initialization:
        from transformers.modeling_utils import no_init_weights
        construction = no_init_weights()
    else:
        construction = nullcontext()
    with construction:
        model = cast_for_training(JointModel(cfg))
    startup = dict(construct_seconds=time.perf_counter()-startup_begin,
                   skipped_random_initialization=skip_initialization)
    loaded_weights=None
    if not args.resume and not args.init_checkpoint and not args.random_init:
        if cfg.text_assets:loaded_weights=load_pretrained(model,cfg.text_assets)
        elif cfg.vision_encoder=='qwen35':model.patch_embed.load_pretrained(cfg.vision_assets)
        if cfg.renderer=='moss':
            loaded_weights=dict(backbone=loaded_weights,renderer=model.renderer.load_pretrained(cfg.moss_assets))
    model = model.to(device='npu').train()
    if args.init_checkpoint and not args.resume:
        loaded_weights=checkpoint_io.initialize_model(args.init_checkpoint,model,cfg)
    startup['weights_to_device_seconds'] = time.perf_counter()-startup_begin
    if rank == 0:print(json.dumps(dict(startup=startup)),flush=True)
    if args.data_snapshot and args.data_mode == 'B':
        model.patch_embed.requires_grad_(False)
        model.renderer.requires_grad_(False)
    # Count TP shards once and count replicated tensors once.
    unique_parameters = [p for p in model.parameters()
        if getattr(p, 'tensor_model_parallel', False) or P.state().get_tensor_model_parallel_rank() == 0]
    count = torch.tensor([sum(p.numel() for p in unique_parameters),
                          sum(p.numel() for p in unique_parameters if p.requires_grad)],
                         device='npu', dtype=torch.int64)
    if args.tp > 1:
        dist.all_reduce(count, group=P.state().get_tensor_model_parallel_group())
    parameters,trainable_parameters = count.tolist()
    wrapped, optimizer = P.make_optimizer(model, args.lr, bucket_size=args.bucket_size,
                                          overlap_param_gather=not args.no_param_overlap)
    ema=None
    if args.moss_ema_decay:
        if cfg.renderer!='moss' or args.data_mode=='B':raise ValueError('EMA requires a trainable MOSS renderer')
        from .ema import RendererEMA
        ema=RendererEMA(model,optimizer,wrapped,args.moss_ema_decay)
    frozen_in_buckets=[name for name,p in model.named_parameters()
                       if not p.requires_grad and p in wrapped.param_to_bucket_group]
    if frozen_in_buckets:raise RuntimeError(f'Frozen parameters entered gradient buckets: {frozen_in_buckets}')
    training_scope=dict(freeze_vision=cfg.freeze_vision,freeze_vision_merger=cfg.freeze_vision_merger,
                        frozen_parameters_in_gradient_buckets=len(frozen_in_buckets))
    if cfg.vision_encoder=='qwen35':
        tower=model.patch_embed.visual
        training_scope.update(vision_encoder_training=tower.blocks.training,
                              vision_encoder_recompute=tower.recompute)
    topology = dict(tp=args.tp, dp=dp, pp=1, cp=1, optimizer_instances=args.optimizer_instances)
    training_state = {key: getattr(args, key) for key in ('lr', 'min_lr', 'lr_decay_style',
        'lr_warmup_steps', 'lr_decay_steps', 'seed', 'sequence_length', 'images', 'micro_batch', 'grad_accum')}
    training_state['renderer_schedule_sha256']=hashlib.sha256((Path(__file__).parent/'renderer_schedule.py').read_bytes()).hexdigest()
    training_state['initialization_checkpoint']=str(args.init_checkpoint) if args.init_checkpoint else None
    training_state['deterministic']=args.deterministic
    training_state['mc2_row']=os.environ.get('PRETRAIN_MC2_ROW','0')
    training_state['qwen35_assets']=text_manifest
    if ema is not None:
        training_state['moss_ema']=dict(schema=ema.schema,module='renderer',decay=ema.decay,dtype='float32',
            adapter_sha256=hashlib.sha256((Path(__file__).parent/'ema.py').read_bytes()).hexdigest())
    if cfg.renderer == 'moss':
        training_state['moss_assets'] = json.loads((Path(cfg.moss_assets)/'manifest.json').read_text())
        training_state['moss_adapter_sha256'] = hashlib.sha256(b''.join(
            (Path(__file__).parent/name).read_bytes() for name in
            ('moss.py','moss_kernels.py','generation.py','input_pipeline/moss_vae.py'))).hexdigest()
        training_state['moss_pointwise'] = os.environ.get('PRETRAIN_MOSS_FUSED_POINTWISE','0')
    training_state['backbone_adapter_sha256']=hashlib.sha256(b''.join(
        (Path(__file__).parent/name).read_bytes() for name in
        ('qwen35.py','qwen35_layout.py','mainline.py','delta_ascendc.py','model.py','loss.py','kernels.py','parallel.py','mc2.py'))).hexdigest()
    if vision_manifest is not None:
        training_state['vision_assets']=vision_manifest
        training_state['vision_adapter_sha256']=hashlib.sha256(
            (Path(__file__).parent/'vision.py').read_bytes()+(Path(__file__).parent/'mainline.py').read_bytes()).hexdigest()
    resumed_data = {} if args.data_snapshot else None
    if args.data_snapshot:
        from .input_pipeline.tokenizer import tokenizer_metadata
        online_tokenizer=tokenizer_metadata(cfg,catalog.meta['tokenizer'])
        vae_sha=[None]
        if args.data_mode=='A' and args.vae_path and rank==0:
            h=hashlib.sha256()
            with open(args.vae_path,'rb') as handle:
                for block in iter(lambda:handle.read(8<<20),b''):h.update(block)
            vae_sha[0]=h.hexdigest()
        dist.broadcast_object_list(vae_sha,src=0)
        training_state['data'] = dict(snapshot=catalog.meta['identity'], membership=membership['identity'],
            mode=args.data_mode, vae_path=args.vae_path if args.data_mode=='A' else None,
            vae_dtype=args.vae_dtype, vae_sha256=vae_sha[0], tokenizer=online_tokenizer,
            tokenizer_counting_matches_corpus=online_tokenizer==catalog.meta['tokenizer'],
            latent_encoder=args.latent_encoder or ('kl16' if args.vae_path else None),
            latent_options=args.latent_options, max_images_per_pack=args.max_images_per_pack,
            pack_padding_multiple=args.pack_padding_multiple)
        adapter_files=sorted((Path(__file__).parent/'input_pipeline').glob('*.py'))
        training_state['data']['adapter_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in adapter_files)).hexdigest()
    start_step = checkpoint_io.load(args.resume, model, optimizer, cfg, topology, training_state,
                                   data_state_out=resumed_data,ema=ema) if args.resume else 0
    if rank == 0:
        restored=['model','optimizer','FP32 masters','CPU/NPU RNG','consumed data cursor'] if args.resume else []
        if args.resume and ema is not None:restored.append('MOSS FP32 EMA and update counter')
        print(json.dumps(dict(resume=dict(path=args.resume, step=start_step,
            restored=restored))), flush=True)
    if args.total_steps is not None:
        if args.total_steps<1: raise ValueError('total_steps must be positive')
        args.steps=max(0,args.total_steps-start_step)
        if not args.steps:
            if rank==0: print(json.dumps(dict(status='budget_already_complete',step=start_step)),flush=True)
            dist.barrier();dist.destroy_process_group();return
        args.warmup=min(args.warmup,args.steps-1)
    # Different DP replicas see different data. TP ranks must see identical data.
    loader = preparer = ordered_buckets = None
    if args.data_snapshot:
        from .input_pipeline.stream import DocumentStream
        from .input_pipeline.packing import Packer, ConsumedLoader
        from .input_pipeline.vae import DevicePreparer
        from .input_pipeline.device import DeviceLoader
        from .input_pipeline.parallel import OrderedGradientBuckets, OrderedParameterGathers, finish_parameter_gathers
        # Spawned CPU workers must not autoload the NPU backend on import.
        os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
        docs = DocumentStream(args.data_snapshot, args.data_mode, P.dp_rank(), dp, args.seed,
            workers=args.data_workers, prefetch=args.data_prefetch,
            source_root=args.source_root, block_rows=args.shuffle_block_rows, tokenize_rows=args.tokenize_rows,
            image_threads=args.image_workers, image_prefetch=args.image_prefetch,
            state=resumed_data.get('documents') if resumed_data else None,
            allow_development=args.allow_development_data,native_images=cfg.vision_encoder=='qwen35',
            image_size=cfg.image_size,cfg=cfg,sequence_length=args.sequence_length)
        packer = Packer(docs, cfg, args.sequence_length, data_tokens, resumed_data or None,
                        pin_device=f'npu:{int(os.environ.get("LOCAL_RANK",0))}',
                        max_images=args.max_images_per_pack,padding_multiple=args.pack_padding_multiple)
        host_loader = ConsumedLoader(packer, args.data_prefetch, resumed_data or None)
        with torch.random.fork_rng(devices=[torch.npu.current_device()],device_type='npu'):
            preparer = DevicePreparer(cfg, args.vae_path if args.data_mode=='A' else None,
                device=f'npu:{int(os.environ.get("LOCAL_RANK",0))}',
                vae_dtype=getattr(torch,args.vae_dtype), vae_chunk=args.vae_chunk,
                encoder=args.latent_encoder if args.data_mode=='A' else None,encoder_options=args.latent_options)
        loader = DeviceLoader(host_loader, preparer, args.device_prefetch)
        ordered_buckets = OrderedGradientBuckets(wrapped)
        ordered_gathers = OrderedParameterGathers(wrapped)
        from .input_pipeline.runtime import empty_batch
        padding_batch = empty_batch(cfg, data_tokens).to('npu')
    else:
        batch = fake_batch(cfg, args.sequence_length, args.micro_batch, args.images,
                           seed=args.seed + P.dp_rank(), device='npu', dtype=torch.bfloat16)
    environment = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
        device=torch.npu.get_device_name(0), total_memory=torch.npu.get_device_properties(0).total_memory,
        world_size=world, topology=topology, model=cfg.to_dict(), parameters=parameters,
        trainable_parameters=trainable_parameters,frozen_parameters=parameters-trainable_parameters,
        training_scope=training_scope,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, global_batch=global_batch,
        capacity_slots_per_update_upper_bound=global_batch*args.sequence_length,
        training_backend='MindSpeed Core + Megatron Core distributed optimizer',
        attention_backend=f'{cfg.layers-cfg.layers//4} AscendC GatedDeltaNet + {cfg.layers//4} torch_npu gated fusion attention', measured_on='Ascend NPU',
        pretrained_weights=loaded_weights,qwen35_assets=text_manifest,startup=startup)
    if ema is not None:
        environment['moss_ema']=dict(**ema.metadata(),local_storage_bytes=ema.local_bytes,
                                    local_optimizer_shards=len(ema.shadows))
    environment['kernel_options'] = dict(fused_ce=os.environ.get('PRETRAIN_FUSED_CE', '1'),
        compile_modulation=os.environ.get('PRETRAIN_COMPILE_MODULATION', '0'),
        attention_tile=os.environ.get('PRETRAIN_ATTENTION_TILE','0'),
        attention_layout=os.environ.get('PRETRAIN_ATTENTION_LAYOUT', 'SBH'))
    environment['kernel_options']['cpu_noise_cast']=os.environ.get('PRETRAIN_CPU_NOISE_CAST','1')
    environment['kernel_options']['moss_fused_pointwise']=os.environ.get('PRETRAIN_MOSS_FUSED_POINTWISE','0')
    environment['kernel_options']['mc2_row']=os.environ.get('PRETRAIN_MC2_ROW','0')
    if loader:
        environment['data'] = dict(snapshot=catalog.meta, membership=membership,
            online_tokenizer=online_tokenizer,tokenizer_counting_matches_corpus=docs.counting_matches_corpus,
            mode=args.data_mode,
            latent_grid='dynamic' if cfg.renderer=='moss' else [cfg.image_size//cfg.patch_size]*2+[cfg.latent_channels],
            actual_generation_image_tokens='dynamic' if cfg.renderer=='moss' else cfg.image_tokens,
            generation_max_pixels=cfg.generation_max_pixels if cfg.renderer=='moss' else cfg.image_size**2,
            generation_token_budget=cfg.generation_token_budget if cfg.renderer=='moss' else None,
            fake_latents=args.latent_encoder=='fake',
            local_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    source_root = Path(__file__).parent
    environment['source_sha256'] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [*sorted(source_root.glob('*.py')),
                                              *sorted((source_root/'input_pipeline').glob('*.py')),
                                              *sorted((source_root/'vendor').rglob('*.py'))]}
    from . import stack
    environment['stack'] = stack.name()
    environment['dependencies'] = json.loads(stack.lock_path().read_text())
    host = dict(rank=rank, local_rank=int(os.environ.get('LOCAL_RANK', 0)),
        hostname=socket.gethostname(), tp_rank=P.state().get_tensor_model_parallel_rank(),
        dp_rank=P.dp_rank())
    hosts = [None] * world
    dist.all_gather_object(hosts, host)
    environment['rank_placement'] = hosts
    cann_root = Path(os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest'))
    version_file = next((p for p in (cann_root/'x86_64-linux/ascend_toolkit_install.info',
                                    cann_root/'share/info/runtime/version.info',
                                    cann_root/'version.cfg', cann_root/'runtime/version.info',
                                    cann_root/'version.info') if p.is_file()), cann_root/'version.info')
    environment['cann'] = version_file.read_text() if version_file.exists() else 'unknown'
    environment['cann_manifest_path'] = str(version_file.resolve())
    maps = Path('/proc/self/maps')
    environment['cann_runtime_library_paths'] = sorted({line.split()[-1]
        for line in (maps.read_text().splitlines() if maps.exists() else [])
        if any(library in line for library in
               ('libascendcl.so', 'libruntime.so', 'libopapi.so', 'libatb.so'))})
    if rank == 0:
        (output / 'run.json').write_text(json.dumps(environment, indent=2))
        print(json.dumps(environment), flush=True)
    profile = nullcontext()
    profile_this_rank = args.profile and rank in profile_ranks
    if profile_this_rank:
        profile = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=args.warmup, warmup=1, active=2, repeat=1),
            # Record during training; parse after all HCCL work has ended.
            # Synchronous mid-step parsing can make other ranks wait
            # long enough to hit the collective execution timeout.
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(output / 'profile' / f'rank{rank}'),analyse_flag=False),
            record_shapes=True, profile_memory=True, with_stack=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1))
    torch.npu.synchronize()
    completed_step = start_step
    last_saved_step = start_step if args.resume else -1
    def save_checkpoint(step):
        from .checkpoint_catalog import prune, durable_json
        checkpoint_begin = time.perf_counter()
        wrapped.start_param_sync(force_sync=True)
        path = checkpoint_root/f'step_{step:07d}'
        checkpoint_io.save(path, model, optimizer, step, cfg, topology, training_state,
                           data_state=loader.state_dict() if loader else None,ema=ema)
        if rank == 0:
            removed = prune(checkpoint_root, args.keep_checkpoints)
            event = dict(step=step, path=str(path), keep=args.keep_checkpoints,
                         seconds=time.perf_counter()-checkpoint_begin, removed=removed)
            durable_json(checkpoint_root/'latest.json', event)
            with (output/'checkpoints.jsonl').open('a') as handle:handle.write(json.dumps(event)+'\n')
            print(json.dumps(dict(checkpoint_committed=event)), flush=True)
        dist.barrier()
    measurements = []
    validation_events=[]
    def validate(step):
        if not args.validation_suite:return
        from .qualitative import run_suite
        validation_begin=time.perf_counter()
        wrapped.start_param_sync(force_sync=True)
        hooks=wrapped.use_forward_hook
        if hooks:wrapped.disable_forward_pre_hook(param_sync=False)
        weights=dict(backbone='online',renderer='ema' if ema is not None else 'online')
        if ema is not None:weights['ema']=ema.metadata()
        try:
            with ema.average_parameters() if ema is not None else nullcontext():
                event=run_suite(model,args.validation_suite,output/'validation'/f'step_{step:07d}',preparer,
                                weight_sources=weights)
        finally:
            if hooks:wrapped.enable_forward_pre_hook()
        torch.npu.synchronize();dist.barrier()
        event['sampling_seconds']=event['seconds']
        event['seconds']=time.perf_counter()-validation_begin
        event['step']=step
        validation_events.append(event)
        if rank==0:
            from .coordination import atomic_json
            result_path=output/'validation'/f'step_{step:07d}'/'results.json'
            result=json.loads(result_path.read_text());result['end_to_end_seconds']=event['seconds']
            atomic_json(result_path,result)
            with (output/'validation.jsonl').open('a') as handle:handle.write(json.dumps(event)+'\n')
            print(json.dumps(dict(validation=event)),flush=True)
    if args.validate_first:validate(start_step)
    measured_begin = None
    begin = time.perf_counter()
    with profile as profiler:
        for local_step in range(args.steps):
            if local_step == args.warmup:
                torch.npu.synchronize()
                dist.barrier()
                measured_begin = time.perf_counter()
            lr = learning_rate(start_step + local_step, args)
            for opt in checkpoint_io._optimizers(optimizer):
                for group in opt.optimizer.param_groups:
                    group['lr'] = lr * group.get('lr_mult', 1.)
            optimizer.zero_grad()
            wrapped.zero_grad_buffer()
            if ordered_buckets: ordered_buckets.begin()
            torch.npu.reset_peak_memory_stats()
            begin = time.perf_counter()
            phase_events = []
            renderer_forward_events = []
            loss_totals=[torch.zeros((),device='npu',dtype=torch.float32) for _ in range(3)]
            consumed_receipts=[]; data_wait_seconds=0.; data_prepare_seconds=0.
            if loader:
                wait_begin=time.perf_counter()
                micro_batches=loader.next_group(args.grad_accum)
                data_wait_seconds=time.perf_counter()-wait_begin
                counts=torch.tensor([sum(x[0].text_indices.numel() for x in micro_batches if x),
                    sum(len(x[0].image_indices) for x in micro_batches if x),
                    sum(x[1]['body_tokens'] for x in micro_batches if x),
                    sum(x[1]['occupied'] for x in micro_batches if x),
                    sum(x is not None for x in micro_batches),
                    sum(x[1].get('generation_tokens',0) for x in micro_batches if x),
                    sum(d['completed'] for x in micro_batches if x for d in x[1]['documents']),
                    sum(d['completed'] and d.get('direction')=='T2I' for x in micro_batches if x for d in x[1]['documents']),
                    sum(d['completed'] and d.get('direction')=='I2T' for x in micro_batches if x for d in x[1]['documents']),
                    sum(x[0].ids.shape[1]+len(x[0].mainline.query_positions) for x in micro_batches if x),
                    sum(x[0].ids.shape[1]-x[1]['occupied'] for x in micro_batches if x),
                    sum(x[1].get('dit_microbatches',int(bool(x[1]['images']))) for x in micro_batches if x)],device='npu',dtype=torch.int64)
                dist.all_reduce(counts,group=P.state().get_data_parallel_group())
                counts=counts.tolist()
                if counts[4]==0:
                    args.steps=local_step
                    break
                # Each DP replica may have a different number of packs. Its
                # final real microbatch launches the same ordered DP reduction
                # sequence. Only an exhausted replica needs one zero-loss
                # graph; don't run full empty graphs up to the GA upper bound.
                micro_batches=[x for x in micro_batches if x] or [None]
            accumulation_steps=len(micro_batches) if loader else args.grad_accum
            for micro_step in range(accumulation_steps):
                scales=None
                if loader:
                    item=micro_batches[micro_step]
                    prep_begin=time.perf_counter()
                    if item is None:
                        batch=padding_batch; scales=(0.,0.)
                    else:
                        cpu_batch,receipt=item[0],item[1]
                        scales=(dp*cpu_batch.text_indices.numel()/max(1,counts[0]),
                                dp*len(cpu_batch.image_indices)/max(1,counts[1]))
                        batch=loader.acquire(item)
                        consumed_receipts.append(receipt)
                    data_prepare_seconds+=time.perf_counter()-prep_begin
                context = wrapped.no_sync() if micro_step + 1 < accumulation_steps else nullcontext()
                with context:
                    if args.phase_timing:
                        events = [torch.npu.Event(enable_timing=True) for _ in range(3)]
                        renderer_forward_events.append([])
                        events[0].record()
                    deferred=bool(batch.generation is not None and batch.generation.microbatches)
                    losses = wrapped(batch, reference=args.split_reference, loss_scales=scales,defer_renderer=deferred)
                    if ordered_buckets:
                        finish_parameter_gathers(wrapped)
                        if not len(batch.image_indices): ordered_buckets.absent_images()
                        if not batch.text_indices.numel():ordered_buckets.absent_text()
                    if args.phase_timing:
                        events[1].record()
                    if deferred:
                        from .renderer_schedule import backward_deferred_renderer
                        backward_deferred_renderer(model,batch,losses,1. if loader else 1/args.grad_accum,wrapped,
                            renderer_forward_events[-1] if args.phase_timing else None)
                    else:(losses['loss'] / (1 if loader else args.grad_accum)).backward()
                    if args.phase_timing:
                        events[2].record()
                        phase_events.append(events)
                # The objective accumulates every microbatch. Logging only the
                # last scaled loss would shrink it by GA and misweight layouts.
                text_scale,image_scale=scales if scales is not None else (1/args.grad_accum,1/args.grad_accum)
                loss_totals[0].add_(losses['loss'].detach().float()/(1 if loader else args.grad_accum))
                loss_totals[1].add_(losses['text_loss'].detach().float()*text_scale)
                loss_totals[2].add_(losses['image_loss'].detach().float()*image_scale)
            if ordered_buckets: ordered_buckets.finish()
            P.finish_grad_sync(wrapped)
            if args.phase_timing:
                optimizer_events = [torch.npu.Event(enable_timing=True) for _ in range(2)]
                optimizer_events[0].record()
            updated, grad_norm, _ = optimizer.step()
            if ema is not None and updated and math.isfinite(float(grad_norm)):ema.update()
            if args.phase_timing:
                optimizer_events[1].record()
            torch.npu.synchronize()
            data_device_wait_seconds=loader.device_wait_seconds() if loader else 0.
            local_seconds=time.perf_counter()-begin
            if args.phase_timing:
                # DiT forward/backward interleave in the staged schedule. Add
                # its measured forward time to forward_ms, not backward_ms.
                renderer_forward_ms=[sum(a.elapsed_time(b) for a,b in pairs) for pairs in renderer_forward_events]
                pack_forward_ms=[e[0].elapsed_time(e[1])+r for e,r in zip(phase_events,renderer_forward_ms)]
                pack_backward_ms=[e[1].elapsed_time(e[2])-r for e,r in zip(phase_events,renderer_forward_ms)]
            if loader:
                rank_row=dict(step=start_step+local_step+1,seconds=local_seconds,
                    host_wait=data_wait_seconds,prepare_host=data_prepare_seconds,device_wait=data_device_wait_seconds,
                    packs=[dict(length=x[0].ids.shape[1],text=x[0].text_indices.numel(),
                        queries=len(x[0].mainline.query_positions),images=len(x[0].image_indices),
                        vision_patches=sum(x[0].vision.lengths) if x[0].vision else 0,
                        generation_tokens=x[1].get('generation_tokens',0),
                        dit_microbatches=x[1].get('dit_microbatches',1 if len(x[0].image_indices) else 0),
                        generation_grids=x[0].generation.grids if x[0].generation else None) for x in micro_batches if x])
                if args.phase_timing:
                    rank_row.update(forward_ms=sum(pack_forward_ms),backward_ms=sum(pack_backward_ms))
                    for pack,fwd,bwd,rfwd in zip(rank_row['packs'],pack_forward_ms,pack_backward_ms,renderer_forward_ms):
                        pack.update(forward_ms=fwd,backward_ms=bwd,deferred_renderer_forward_ms=rfwd)
                with (output/f'timing_rank_{rank:05d}.jsonl').open('a') as f:f.write(json.dumps(rank_row)+'\n')
            elapsed = torch.tensor(local_seconds, device='npu', dtype=torch.float32)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            seconds = elapsed.item()
            memory = torch.tensor(torch.npu.max_memory_allocated(), device='npu', dtype=torch.int64)
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            values = torch.stack(loss_totals)
            dist.all_reduce(values, group=P.state().get_data_parallel_group())
            values /= dp
            if not torch.isfinite(values).all().item() or not updated or not math.isfinite(float(grad_norm)):
                raise FloatingPointError('Nonfinite loss or skipped optimizer update')
            for receipt in consumed_receipts: loader.acknowledge(receipt)
            completed_step = start_step+local_step+1
            if loader:
                with (output/f'consumed_rank_{rank:05d}.jsonl').open('a') as trace:
                    trace.write(json.dumps(dict(step=start_step+local_step+1,receipts=consumed_receipts))+'\n')
            row = dict(step=start_step+local_step+1, seconds=seconds,
                tokens_per_second=global_batch*args.sequence_length/seconds,
                peak_memory_gib=memory.item()/2**30, loss=values[0].item(),
                text_loss=values[1].item(), image_loss=values[2].item(), grad_norm=float(grad_norm),
                warmup=local_step < args.warmup, lr=lr)
            if ema is not None:row.update(moss_ema_decay=ema.decay,moss_ema_updates=ema.num_updates)
            if loader:
                import resource
                row['capacity_slots_per_second']=row.pop('tokens_per_second')
                row['capacity_slots_per_second']=counts[4]*args.sequence_length/seconds
                resource_values=torch.tensor([data_wait_seconds,data_prepare_seconds,data_device_wait_seconds,
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20],device='npu')
                dist.all_reduce(resource_values,op=dist.ReduceOp.MAX)
                row.update(body_tokens=counts[2],occupied_tokens=counts[3],images_consumed=counts[1],
                           generation_tokens=counts[5],documents_completed=counts[6],
                           t2i_captions_completed=counts[7],i2t_captions_completed=counts[8],
                           active_microbatches=counts[4],data_wait_seconds=resource_values[0].item(),
                           supervised_text_tokens=counts[0],condition_query_tokens=counts[1]*cfg.image_condition_queries,
                           backbone_processed_slots=counts[9],padding_tokens=counts[10],dit_microbatches=counts[11],
                           data_prepare_seconds=resource_values[1].item(),data_device_wait_seconds=resource_values[2].item(),
                           cpu_peak_rss_gib=resource_values[3].item(),
                           images_per_second=counts[1]/seconds,documents_per_second=counts[6]/seconds,
                           body_tokens_per_second=counts[2]/seconds,occupied_tokens_per_second=counts[3]/seconds)
            if args.phase_timing:
                phases = torch.tensor([sum(pack_forward_ms),sum(pack_backward_ms),
                    optimizer_events[0].elapsed_time(optimizer_events[1])], device='npu')
                dist.all_reduce(phases, op=dist.ReduceOp.MAX)
                row.update(zip(('forward_ms','backward_ms','optimizer_ms'), phases.tolist()))
            if rank == 0:
                with (output / 'steps.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
            if local_step >= args.warmup:
                measurements.append(row)
            if args.validation_probes:
                torch.save(checkpoint_io.validation_probes(model,optimizer),
                    output/f"probe_rank_{rank:05d}_step_{row['step']:07d}.pt")
            if (args.save_every and row['step'] % args.save_every == 0) or (args.save_first and row['step'] == 1):
                save_checkpoint(row['step'])
                last_saved_step = row['step']
            if profile_this_rank:
                profiler.step()
            if args.validate_every and completed_step%args.validate_every==0:validate(completed_step)
        if args.validate_final and (not validation_events or validation_events[-1]['step']!=completed_step):validate(completed_step)
        if args.save_final and completed_step > last_saved_step:
            save_checkpoint(completed_step)
        # Includes gradient zeroing, metric collectives and logging/checkpoints.
        torch.npu.synchronize()
        wall_seconds = torch.tensor(time.perf_counter() - (measured_begin or begin), device='npu')
        dist.all_reduce(wall_seconds, op=dist.ReduceOp.MAX)
        wall_seconds = wall_seconds.item()
    if rank == 0:
        if not measurements:
            if not args.until_data_end:
                raise RuntimeError('Dataset ended before any measured updates; reduce warmup or enlarge the explicit development snapshot')
            # A preempted final update can resume directly at EOF. It must
            # complete successfully rather than trigger an endless restart.
            (output/'summary.json').write_text(json.dumps(dict(completed_updates=completed_step,
                status='finite_dataset_complete', measured_updates=0, resumed_from=args.resume), indent=2))
    if rank == 0 and measurements:
        mean_seconds = sum(x['seconds'] for x in measurements)/len(measurements)
        summary = dict(world=world, **topology, method=cfg.method,image_condition_queries=cfg.image_condition_queries,
            completed_updates=completed_step,steps=len(measurements),
            mean_seconds=mean_seconds, tokens_per_second=global_batch*args.sequence_length/mean_seconds,
            tokens_per_second_per_card=global_batch*args.sequence_length/mean_seconds/world,
            peak_memory_gib=max(x['peak_memory_gib'] for x in measurements),
            global_batch=global_batch, sequence_length=args.sequence_length, parameters=parameters,
            trainable_parameters=trainable_parameters,frozen_parameters=parameters-trainable_parameters,
            freeze_vision=cfg.freeze_vision,freeze_vision_merger=cfg.freeze_vision_merger,
            vision_max_pixels=cfg.vision_max_pixels,generation_max_pixels=cfg.generation_max_pixels,
            wall_seconds=wall_seconds,
            wall_tokens_per_second=global_batch*args.sequence_length*len(measurements)/wall_seconds,
            median_seconds=statistics.median(x['seconds'] for x in measurements),
            p95_seconds=sorted(x['seconds'] for x in measurements)[math.ceil(.95*len(measurements))-1],
            bucket_size=args.bucket_size, overlap_param_gather=not args.no_param_overlap,
                micro_batch=args.micro_batch, grad_accum=args.grad_accum, images=args.images,
            checkpoint_backbone=cfg.checkpoint_backbone, checkpoint_dit=cfg.checkpoint_dit,
            checkpoint_backbone_layers=cfg.checkpoint_backbone_layers,
            checkpoint_dit_layers=cfg.checkpoint_dit_layers,
            split_reference=args.split_reference, profiling=args.profile)
        summary['validation_probes']=args.validation_probes
        summary['validation_seconds']=sum(e['seconds'] for e in validation_events if e['step']>start_step)
        summary['qualitative_validation']=bool(args.validation_suite)
        if ema is not None:summary['moss_ema']=dict(**ema.metadata(),local_storage_bytes=ema.local_bytes)
        if args.phase_timing:
            summary.update({key: statistics.mean(x[key] for x in measurements)
                            for key in ('forward_ms','backward_ms','optimizer_ms')})
        if loader:
            summary.pop('images')
            summary['max_images_per_pack']=args.max_images_per_pack
            for old,new in (('tokens_per_second','capacity_slots_per_second'),
                            ('tokens_per_second_per_card','capacity_slots_per_second_per_card'),
                            ('wall_tokens_per_second','wall_capacity_slots_per_second')):
                summary[new]=summary.pop(old)
            summary['data_mode']=args.data_mode
            summary['body_tokens_per_second']=sum(x['body_tokens'] for x in measurements)/wall_seconds
            summary['occupied_tokens_per_second']=sum(x['occupied_tokens'] for x in measurements)/wall_seconds
            summary['images_per_second']=sum(x['images_consumed'] for x in measurements)/wall_seconds
            summary['documents_per_second']=sum(x['documents_completed'] for x in measurements)/wall_seconds
            summary['generation_tokens_per_second']=sum(x['generation_tokens'] for x in measurements)/wall_seconds
            total_seconds=sum(x['seconds'] for x in measurements)
            total_capacity=sum(x['active_microbatches'] for x in measurements)*args.sequence_length
            summary['capacity_slots_per_second']=total_capacity/total_seconds
            summary['capacity_slots_per_second_per_card']=total_capacity/total_seconds/world
            summary['wall_capacity_slots_per_second']=total_capacity/wall_seconds
            for key in ('occupied_tokens','body_tokens','images_consumed','supervised_text_tokens',
                        'condition_query_tokens','backbone_processed_slots','padding_tokens','active_microbatches','dit_microbatches'):
                values=[x[key] for x in measurements]
                summary[key+'_per_update']=dict(mean=statistics.mean(values),min=min(values),max=max(values))
            summary['fake_latents']=args.latent_encoder=='fake'
            for key in ('data_wait_seconds','data_prepare_seconds','data_device_wait_seconds','cpu_peak_rss_gib'):
                summary[key]=statistics.mean(x[key] for x in measurements)
        (output/'summary.json').write_text(json.dumps(summary, indent=2))
    if loader:
        loader.close()
        stats=dict(consumed_state=loader.state_dict(),prefetched_document_stats=docs.stats,
                   packing_stats=packer.stats,preparer_stats=preparer.stats,
                   device_stats=loader.stats,image_stats=docs.image_reader.stats() if docs.image_reader else {})
        (output/f'data_rank_{rank:05d}.json').write_text(json.dumps(stats))
    from .delta_ascendc import provenance
    (output/f'backend_rank_{rank:05d}.json').write_text(json.dumps(provenance(),indent=2))
    dist.barrier()
    dist.destroy_process_group()
    if profile_this_rank:
        from torch_npu.profiler.profiler import analyse
        profile_root=output/'profile'/f'rank{rank}'
        analyse(str(profile_root),max_process_number=4)
        if not list(profile_root.glob('*/ASCEND_PROFILER_OUTPUT/op_statistic.csv')):
            raise RuntimeError(f'Profiling did not export rank {rank}; inspect offline parser errors')


if __name__ == '__main__':
    main()
