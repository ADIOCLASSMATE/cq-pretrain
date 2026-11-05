"""Matched 16-card Qwen CQ / native Gemma E2B training-step comparison.

Synthetic fixed image/text shapes isolate compute. Frozen visual features and
Wan latents are resident before timing; loading, tokenization and I/O are excluded.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
from torch import nn
import torch.distributed as dist
from configs.config import RunConfig
import infra.parallel as P
from models.backbone_qwen35 import QwenBlock, RMSNorm, cast_for_training, unfused_parameters
from scripts.benchmark_gemma import GemmaDecoder, SoftcapCE, attention_tiles

ROOT = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/models')


class Model(nn.Module):
    def __init__(self, cfg, gemma, joint):
        super().__init__(); self.cfg = cfg; self.config = P._config; self.gemma = gemma
        if gemma:
            self.decoder = GemmaDecoder(gemma)
            self.visual_projection = nn.Linear(2048, cfg.width, bias=False)
        else:
            self.embedding = P.make_embedding(cfg.vocab_size, cfg.width)
            self.layers = nn.ModuleList(QwenBlock(cfg, i) for i in range(cfg.layers))
            self.final_norm = RMSNorm(cfg.width, cfg.norm_eps)
        if joint:
            from models.generator_wan21 import Wan21Generator
            self.renderer = Wan21Generator(cfg)

    def forward(self, batch, features, extra):
        plan = batch.mainline
        if self.gemma:
            ids, ropes, tiles = extra
            query = self.decoder(ids, batch.vision.content_indices, self.visual_projection(features),
                batch.ids.shape[1], ropes, tiles, plan.cq_valid, self.cfg.checkpoint_backbone)
            hidden = query.new_zeros(plan.output_size, self.cfg.width).index_copy(0,
                plan.output_positions, query[:, 0])
            weight = self.decoder.embed_tokens.weight
        else:
            from models.cq import backbone
            hidden = backbone(self, batch, vision_features=features)
            weight = self.embedding.weight
        if hasattr(self, 'renderer'):
            image_loss = self.renderer(hidden[plan.condition_indices], batch)
        else: image_loss = hidden.new_zeros(())
        text = hidden[batch.text_indices]; labels = batch.ids.flatten()[batch.text_indices]
        if self.gemma:
            text_loss = SoftcapCE.apply(text, weight, labels, self.cfg.ce_chunk, self.gemma.final_logit_softcapping)
        else:
            from infra.cross_entropy import linear_cross_entropy
            text_loss = linear_cross_entropy(text, weight, labels, self.cfg.ce_chunk)
        return self.cfg.text_weight * text_loss + image_loss

    def load_weights(self):
        from safetensors import safe_open
        if self.gemma:
            root = ROOT/'Gemma4-E2B-Base'; targets = self.decoder.checkpoint_targets()
            with safe_open(str(root/'model.safetensors'), framework='pt') as f, torch.no_grad():
                expected = {k.removeprefix('model.language_model.') for k in f.keys() if k.startswith('model.language_model.')}
                # The official loader also drops these stored but unused weights:
                # shared layers read K/V from the final non-sharing layer.
                unused = {f'layers.{i}.self_attn.{name}.weight'
                          for i in range(self.gemma.num_hidden_layers-self.gemma.num_kv_shared_layers,
                                         self.gemma.num_hidden_layers)
                          for name in ('k_proj', 'v_proj', 'k_norm')}
                assert expected-set(targets) == unused and set(targets) <= expected
                for name, value in targets.items():
                    loaded = P.checkpoint_shard(f.get_tensor('model.language_model.'+name), value)
                    assert value.shape == loaded.shape, name
                    value.copy_(loaded)
            nn.init.normal_(self.visual_projection.weight, std=.02)
        else:
            root = ROOT/'Qwen3.5-2B-Base'
            index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
            targets = {}
            for name, param in self.named_parameters():
                if name.startswith('renderer.'): continue
                for clean, tensor in unfused_parameters(name, param, self.cfg).items():
                    clean = clean.replace('embedding.', 'embed_tokens.').replace('final_norm.', 'norm.')
                    targets['model.language_model.'+clean] = tensor
            assert set(targets) == {k for k in index if k.startswith('model.language_model.')}
            for file in sorted({index[k] for k in targets}):
                with safe_open(str(root/file), framework='pt') as f, torch.no_grad():
                    for name, value in targets.items():
                        if index[name] == file: value.copy_(P.checkpoint_shard(f.get_tensor(name), value))
        if hasattr(self, 'renderer'):
            self.renderer.load_pretrained(self.cfg.wan_assets)


def prepare(cfg, gemma, length, task):
    from data.batch import fake_batch
    from data.planning import build_plan
    data_cfg = replace(cfg, image_size=256, vision_max_pixels=131072,
        generation_max_pixels=131072, checkpoint_backbone=True, checkpoint_dit=True)
    batch = fake_batch(data_cfg, length=length, seed=42+P.dp_rank())
    # Avoid an artificial all-identical-token embedding lookup/scatter workload.
    batch.ids.view(-1)[batch.text_indices] = torch.randint(5, data_cfg.vocab_size,
        (len(batch.text_indices),), generator=torch.Generator().manual_seed(700+P.dp_rank()))
    if task == 'i2t':
        batch.generation_image_indices = torch.empty(0, dtype=torch.long)
        batch.mainline = build_plan(batch, data_cfg)
    extra = None
    if gemma:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
        s = batch.ids.shape[1]; plan = batch.mainline
        query_ids = torch.where(plan.query_image, cfg.image_mask_token_id, cfg.text_mask_token_id)
        content_ids = batch.ids.clone(); content_ids[:, batch.vision.content_indices] = 0
        ids = torch.cat((content_ids, query_ids), 1)
        positions = torch.cat((torch.arange(s), plan.query_positions))[None]
        rotary = Gemma4TextRotaryEmbedding(gemma)
        ropes = {kind: tuple(t.transpose(0, 1).unsqueeze(2).to('npu', torch.bfloat16)
                            for t in rotary(torch.empty(1, len(positions[0]), cfg.width), positions, layer_type=kind))
                 for kind in set(gemma.layer_types)}
        masks = {'full_attention':plan.cq_mask,
                 'sliding_attention':plan.cq_mask | (torch.arange(s) < positions[:, :, None]-511)[:, None]}
        # Maintain the finite sentinel on invalid first-token queries.
        for mask in masks.values(): mask[0, 0, ~plan.cq_valid[0], 0] = False
        tiles = {kind: attention_tiles(mask, 512 if kind == 'sliding_attention' else 2048).bind(mask.to('npu'))
                 for kind, mask in masks.items()}
        extra = ids.to('npu'), ropes, tiles
    features = torch.randn(len(batch.vision.content_indices), 2048,
        generator=torch.Generator().manual_seed(800+P.dp_rank())).to('npu', torch.bfloat16)
    return batch.to('npu'), features, extra


def finish_gradients(wrapped, gemma):
    if not gemma:
        P.finish_grad_sync(wrapped); return
    wrapped.finish_grad_sync()
    # MQA K/V and their norms are replicated; each TP rank sees different Q heads.
    grads = [p.main_grad for name, p in wrapped.module.named_parameters()
             if '.self_attn.' in name and name.endswith(('q_norm.weight', 'k_norm.weight', 'k_proj.weight', 'v_proj.weight'))]
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    merged = _flatten_dense_tensors(grads)
    dist.all_reduce(merged, group=P.state().get_tensor_model_parallel_group())
    for grad, reduced in zip(grads, _unflatten_dense_tensors(merged, grads)): grad.copy_(reduced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen', 'gemma'), required=True)
    parser.add_argument('--task', choices=('i2t', 'joint'), required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--global-batch', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = RunConfig(checkpoint_backbone=True, checkpoint_dit=True, ce_chunk=1024)
    P.initialize(2, cfg, initialize_weights=False)
    if args.global_batch % P.dp_size(): raise ValueError('Global batch must be divisible by DP')
    accumulation = args.global_batch // P.dp_size()
    torch.manual_seed(871); torch.npu.manual_seed_all(871)
    gemma = None
    if args.model == 'gemma':
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        gemma = Gemma4TextConfig(**json.loads((ROOT/'Gemma4-E2B-Base/config.json').read_text())['text_config'])
        cfg = replace(cfg, width=gemma.hidden_size, layers=gemma.num_hidden_layers, vocab_size=gemma.vocab_size)
    from transformers.initialization import no_init_weights
    with no_init_weights(): model = cast_for_training(Model(cfg, gemma, args.task == 'joint'))
    model.load_weights(); model.to('npu'); model.train()
    batch, features, extra = prepare(cfg, gemma, args.length, args.task)
    wrapped, optimizer = P.make_optimizer(model, 1e-5, bucket_size=40_000_000)
    ema = None
    if args.task == 'joint':
        from training.ema import RendererEMA
        ema = RendererEMA(model, optimizer, wrapped, .999)
    parameter_count = sum(p.numel()*(P.tp_size() if getattr(p, 'tensor_model_parallel', False) else 1)
                          for p in model.parameters())
    report = dict(model=args.model, task=args.task, sequence_length=args.length,
        world_size=dist.get_world_size(), tp=2, dp=P.dp_size(), microbatch=1, global_batch=args.global_batch,
        gradient_accumulation=accumulation,
        device=torch.npu.get_device_name(0), mc2_rows=os.environ.get('PRETRAIN_MC2_ROW') == '1',
        trainable_parameters=parameter_count, dtype='bfloat16', activation_checkpointing=True,
        image_condition_queries=cfg.image_condition_queries, visual_query_mode=cfg.visual_query_mode, text_queries=len(batch.text_indices),
        visual_content_tokens=len(batch.vision.content_indices),
        generation_images=batch.generation_image_count, query_tokens=len(batch.mainline.query_positions),
        input='fixed synthetic shapes; cached common visual features and fake VAE latents',
        pretrained_backbone=True, pretrained_wan=args.task == 'joint', steps=[])
    if dist.get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True); print(json.dumps(report), flush=True)
    for step in range(args.steps):
        dist.barrier(); torch.npu.synchronize(); torch.npu.reset_peak_memory_stats()
        begin = time.perf_counter(); optimizer.zero_grad(); wrapped.zero_grad_buffer()
        phases = []; loss_value = torch.zeros((), device='npu')
        for micro in range(accumulation):
            events = [torch.npu.Event(enable_timing=True) for _ in range(3)]
            with wrapped.no_sync() if micro+1 < accumulation else nullcontext():
                events[0].record(); loss = wrapped(batch, features, extra); events[1].record()
                (loss/accumulation).backward(); events[2].record()
            loss_value += loss.detach()/accumulation; phases.append(events)
        finish_gradients(wrapped, gemma)
        opt_events = [torch.npu.Event(enable_timing=True) for _ in range(2)]; opt_events[0].record()
        updated, grad_norm, _ = optimizer.step()
        if ema is not None: ema.update()
        opt_events[1].record()
        torch.npu.synchronize()
        stats = torch.tensor([time.perf_counter()-begin, torch.npu.max_memory_allocated()/2**30,
            sum(a.elapsed_time(b) for a,b,c in phases),
            sum(b.elapsed_time(c) for a,b,c in phases)+phases[-1][2].elapsed_time(opt_events[0]),
            opt_events[0].elapsed_time(opt_events[1])], device='npu')
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        if not updated or not math.isfinite(float(grad_norm)) or not torch.isfinite(loss).item():
            raise FloatingPointError('Invalid optimizer update')
        row = dict(step=step+1, seconds=float(stats[0]), peak_gib=float(stats[1]),
                   loss=float(loss_value), grad_norm=float(grad_norm), warmup=step < args.warmup,
                   forward_ms=float(stats[2]), backward_and_sync_ms=float(stats[3]), optimizer_ms=float(stats[4]))
        report['steps'].append(row)
        if dist.get_rank() == 0:
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
    timed = report['steps'][args.warmup:]
    report.update(median_seconds=statistics.median(r['seconds'] for r in timed),
                  peak_gib=max(r['peak_gib'] for r in timed), complete=True)
    report['content_tokens_per_second'] = args.length * args.global_batch/report['median_seconds']
    if dist.get_rank() == 0: args.output.write_text(json.dumps(report, indent=2)+'\n')
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
