"""Compare unified evaluation to training validation and native sampling paths.

Run with the same full export/config used by training. torchrun is supported:
all ranks compare their global validation batches; each rank checks both native
sampling paths. This runs Trainer.validate, without an optimizer or training step.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from configs.config import load_config
from data.dataloader import build_validation_loader
from data.dataset import VideoDataset, qwen_video_inputs
from data.packing import collate_video_samples
from data.sequence import make_video_sequence
from evaluations.runtime import EvaluationRuntime
from evaluations.sampling import build_requests, sample_request
from evaluations.scoring import score_loss
from evaluations.validation import evaluation_state, seed_evaluation
from training.loss import to_device
from training.trainer import Trainer


def legacy_video_layout(prompt, processor, config):
    """Frozen ordinary generate.py layout before unifying the entry points."""
    s = config["data"]["video"]
    rgb = torch.zeros(s["num_frames"], 3, s["height"], s["width"], dtype=torch.uint8)
    clip = {"timestamps": [i / s["fps"] for i in range(s["num_frames"])]}
    _, grid, times = qwen_video_inputs(rgb, clip, processor, s)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "video", "video_id": "v0"}]},
    ]
    sequence = make_video_sequence(messages, processor, grid, times,
                                   max_seq_length=config["data"]["max_seq_length"])
    row = dict(sequence, task="t2v", sample_id="generation", clip=clip,
               video_features=None, pixel_values_videos=None, video_grid_thw=grid)
    batch = collate_video_samples([row], pad_token_id=processor.tokenizer.pad_token_id or 0)
    batch["video_observed_mask"].fill_(False)
    return batch


def scalar_output(output):
    return {key: float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else value
            for key, value in output.items()}


def compare_tensor(a, b):
    if a.shape != b.shape:
        raise AssertionError(f"Shapes differ: {a.shape} != {b.shape}")
    error = float((a.float().cpu() - b.float().cpu()).abs().max()) if a.numel() else 0.0
    if not torch.equal(a.cpu(), b.cpu()):
        raise AssertionError(f"Numerical mismatch, max_abs={error}")
    return {"equal": True, "max_abs": error, "shape": list(a.shape)}


def validate_loss_parity(runtime, output):
    import training.trainer as trainer_module

    config, model = runtime.config, runtime.get_model()
    rank = dist.get_rank() if dist.is_initialized() else 0
    loader = build_validation_loader(config, runtime.processor)
    observed, current = [], {}
    def indexed():
        counts = Counter()
        for batch in loader:
            current.update(task=batch["task"], batch_index=counts[batch["task"]],
                           sample_ids=batch.get("sample_ids", []))
            yield batch
            counts[batch["task"]] += 1
    original = trainer_module.forward_loss
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append({**current, **scalar_output(result)})
        return result
    # Use the actual Trainer.validate implementation with the same BF16 model.
    trainer = Trainer.__new__(Trainer)
    trainer.model, trainer.config, trainer.validation_loader = model, config, indexed()
    trainer.schedule, trainer.codec = tuple(config["data"]["schedule"]), runtime.codec
    trainer.accelerator = SimpleNamespace(device=runtime.device, is_mcore=False,
                                         unwrap_model=lambda value: value, autocast=runtime.autocast)
    trainer_module.forward_loss = capture
    try:
        train_metrics = trainer.validate()
    finally:
        trainer_module.forward_loss = original
    train_path = output / f"trainer-batches.rank-{rank}.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in observed))
    task = dict(name="validation", benchmark="loss", score={"save_batches": True})
    standalone = score_loss(runtime, task, output / "standalone", distributed=dist.is_initialized())
    actual = [json.loads(line) for line in (output / "standalone" / f"batches.rank-{rank}.jsonl").read_text().splitlines()]
    if train_metrics != standalone or observed != actual:
        raise AssertionError(f"Training versus standalone loss mismatch on rank {rank}")
    return {"metrics_equal": True, "batches_equal": True, "batches": len(actual), "metrics": standalone}


def validate_sampling_parity(runtime, output):
    config, model, p = runtime.config, runtime.get_model(), runtime.processor
    checks = {}
    for mode in ("t2v", "v2t"):
        task = dict(name=mode, benchmark=mode, data={}, sample={"max_samples": 1,
                    "outputs": ["latent"] if mode == "t2v" else ["text"],
                    "collect": {} if mode == "t2v" else {"token_ids": True, "text_logits": True}})
        request = build_requests(task, config)[0]
        sampling, video = config["sampling"], config["data"]["video"]
        with evaluation_state(model), runtime.autocast():
            seed_evaluation(request["seed"])
            if mode == "t2v":
                # The simple temp_data caption view matches legacy --prompt.
                batch = to_device(legacy_video_layout(request["prompt"], p, config), runtime.device)
                rng = torch.Generator(device=runtime.device).manual_seed(request["seed"])
                expected = model.generate_video(
                    batch, latent_shapes=[(16, (video["num_frames"] - 1) // 4 + 1,
                                          video["height"] // 8, video["width"] // 8)],
                    steps=sampling["steps"], solver=sampling["solver"], cfg=sampling["cfg"],
                    time_shift=sampling["time_shift"], timestep_scale=config["flow"]["timestep_scale"], rng=rng,
                )[0].cpu()
            else:
                row = VideoDataset(request["source"], p, config, prompt_only=True)[request["index"]]
                batch = to_device(collate_video_samples([row], pad_token_id=p.tokenizer.pad_token_id or 0), runtime.device)
                expected = model.generate_text(
                    batch, max_new_tokens=sampling["max_new_tokens"], temperature=sampling["temperature"],
                    top_k=sampling.get("top_k", 0), top_p=sampling.get("top_p", 1.0),
                    eos_token_id=p.tokenizer.eos_token_id, use_cache=False, return_logits=True,
                )
                expected = {key: value.cpu() for key, value in expected.items()}
        result = sample_request(runtime, task, request, output / mode)
        if mode == "t2v":
            actual = torch.load(output / mode / result["artifacts"]["latent"], weights_only=True)
            checks[mode] = compare_tensor(expected, actual["latent"])
        else:
            actual = torch.load(output / mode / result["artifacts"]["text_logits"], weights_only=True)
            checks[mode] = {"tokens": compare_tensor(expected["sequences"][0], actual["token_ids"]),
                            "logits": compare_tensor(expected["logits"][0], actual["logits"])}
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    runtime = EvaluationRuntime(config, args.checkpoint)
    device = runtime.device
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("hccl" if device.type == "npu" else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        loss = validate_loss_parity(runtime, output)
        sampling = validate_sampling_parity(runtime, output / f"sampling-rank-{rank}")
        report = {"rank": rank, "world_size": world, "checkpoint": str(Path(args.checkpoint).resolve()),
                  "dtype": str(next(runtime.model.parameters()).dtype), "gdn": runtime.backend_identity,
                  "loss": loss, "sampling": sampling, "seed_policy": "per_request_v1",
                  "note": "Actual Trainer.validate path; native generation baseline shares explicit generation RNG."}
        (output / f"report.rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if dist.is_initialized():
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
