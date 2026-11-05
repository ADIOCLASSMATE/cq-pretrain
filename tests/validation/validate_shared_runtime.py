"""16-rank CQ/Wan update, mixed branches, reentrant GA and exact restart gate."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from configs.config import DEFAULTS
from models.modeling_umm import UMM
from tests.test_modeling_umm_video import configuration
from training.loss import prepare_batch, forward_loss, global_loss_weights
from training.optimizer import apply_trainable_scope
from training.runtime import MCoreRuntime


def make_batch(model, rank, step, micro, *, online=False):
    cfg = model.config.qwen_config
    width = cfg.text_config.hidden_size
    g = torch.Generator().manual_seed(10000 + step * 1000 + rank * 10 + micro)
    # Each rank exercises both branches across GA; target counts differ by rank.
    targets = (rank + micro + step) % 3
    videos = max(1, targets)
    values = [3, 4]
    for video in range(videos):
        for block in range(2):
            values += [7 + block, cfg.vision_start_token_id] + [cfg.video_token_id] * 4 + [cfg.vision_end_token_id]
    values += [12, 13]
    ids = torch.tensor([values])
    labels = torch.full_like(ids, -100)
    labels[:, -2:] = ids[:, -2:]
    result = dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids, dtype=torch.bool),
        mm_token_type_ids=ids.eq(cfg.video_token_id).long() * 2,
        video_grid_thw=torch.tensor([[2, 4, 4]] * videos),
        target_video_ids=torch.arange(targets), video_uncond_mask=torch.tensor([bool((rank + micro + i) % 2) for i in range(videos)]),
        wan_latents=[torch.randn(model.generator.in_dim, 3, 4 + 2 * i, 6, generator=g) for i in range(targets)])
    if online:
        vision = cfg.vision_config
        size = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        result['pixel_values_videos'] = torch.randn(32 * videos, size, generator=g)
    else:
        result['video_features'] = [torch.randn(8, width, generator=g) for _ in range(videos)]
    return result


def assert_tree_equal(left, right, path='state'):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left: assert_tree_equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)): assert_tree_equal(a, b, f'{path}[{i}]')
    else:
        assert left == right, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--full-weights', action='store_true')
    args = parser.parse_args()
    config = deepcopy(DEFAULTS)
    config['distributed'].update(num_processes=16, bucket_size=4_000_000)
    config['training'].update(gradient_accumulation_steps=2, max_train_steps=3, use_ema=False, deterministic=True)
    config['optimizer'].update(warmup_steps=0, decay_steps=0)
    os.environ['HCCL_DETERMINISTIC'] = 'True'
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'
    torch.use_deterministic_algorithms(True)
    runtime = MCoreRuntime(config)
    assert runtime.num_processes == 16
    torch.manual_seed(304)
    if args.full_weights:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        native = Qwen3_5Config.from_pretrained(config['model']['qwen_path'], local_files_only=True)
        model = UMM.from_pretrained_components(config['model']['qwen_path'], config['model']['wan_path'],
            mask_token_id=native.text_config.vocab_size - 2, video_mask_token_id=native.text_config.vocab_size - 1,
            gradient_checkpointing=True, dtype=torch.bfloat16)
    else:
        cfg = configuration(layers=4, freeze=True, checkpointed=True)
        cfg.qwen_config.text_config.linear_key_head_dim = 128
        cfg.qwen_config.text_config.linear_value_head_dim = 128
        model = UMM(cfg)
    apply_trainable_scope(model, config['model'])
    wrapped, optimizer, scheduler = runtime.prepare_model(model)
    wrapped.train()
    if runtime.is_main_process:
        print(f"Model and optimizer ready; full_weights={args.full_weights}", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []

    def update(step):
        torch.manual_seed(100 + step + runtime.process_index)
        batches = [make_batch(model, runtime.process_index, step, i, online=(i == 0)) for i in range(2)]
        weights = global_loss_weights(batches, runtime)
        if runtime.is_main_process:
            print(f"Step {step}: global loss weights ready", flush=True)
        optimizer.zero_grad()
        torch.npu.synchronize()
        start = time.perf_counter()
        losses = []
        for batch, weight in zip(batches, weights):
            prepared = prepare_batch(batch, runtime.device, config)
            with runtime.accumulate(wrapped), runtime.autocast():
                result = forward_loss(wrapped, prepared, config, loss_weights=weight)
                runtime.backward(result['loss'])
            losses.append(float(result['loss'].detach()))
        optimizer.step()
        scheduler.step()
        runtime.synchronize_parameters()
        torch.npu.synchronize()
        assert not runtime.optimizer_step_was_skipped
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert torch.isfinite(torch.tensor(losses)).all()
        # The DP model must agree on every replica after each update.
        for p in (model.backbone.embed_tokens.weight, model.generator.text_embedding[0].weight,
                  model.generator.blocks[0].self_attn.kv.weight, model.generator.head.head.weight):
            other = p.detach().clone()
            dist.broadcast(other, src=0, group=runtime.dp_group)
            assert torch.equal(other, p)
        record = dict(step=step, losses=losses, seconds=time.perf_counter() - start,
                      grad_norm=runtime.last_grad_norm, absent=runtime.last_absent_gradients,
                      memory_gib=torch.npu.max_memory_allocated() / 2**30)
        if runtime.is_main_process:
            print(json.dumps(record), flush=True)
        return record

    records.append(update(1))
    if not args.full_weights:
        runtime.save_state(output / 'step1')
        dist.barrier()
    records.append(update(2))
    if not args.full_weights:
        runtime.save_state(output / 'continuous')
        dist.barrier()
        runtime.load_state(output / 'step1')
        records.append(update(2))
        runtime.save_state(output / 'resumed')
        dist.barrier()
        name = f'optimizer_rank_{runtime.process_index:05d}.pt'
        assert_tree_equal(torch.load(output / 'continuous' / name, weights_only=False, map_location='cpu'),
                          torch.load(output / 'resumed' / name, weights_only=False, map_location='cpu'))
        if runtime.is_main_process:
            assert_tree_equal(torch.load(output / 'continuous/model.pt', weights_only=True, map_location='cpu'),
                              torch.load(output / 'resumed/model.pt', weights_only=True, map_location='cpu'))
    report = dict(passed=True, rank=runtime.process_index, world=runtime.num_processes,
                  full_weights=args.full_weights, records=records, runtime=runtime.runtime_contract())
    (output / f'rank_{runtime.process_index:05d}.json').write_text(json.dumps(report, indent=2))
    dist.barrier()
    if runtime.is_main_process:
        assert len(list(output.glob('rank_*.json'))) == 16
        (output / 'COMPLETE').write_text('16 ranks passed\n')
    runtime.end_training()


if __name__ == '__main__':
    main()
