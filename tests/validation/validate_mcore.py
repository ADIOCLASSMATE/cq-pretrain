"""Four-rank transport/optimizer check against explicit global-batch AdamW.

Run after sourcing scripts/env_mindspeed.sh, using torch.distributed.run.
Exercises divergent branches, accumulated absent gradients, functional conv
weights, FP32 time linears, overlap, and exact local optimizer/master restore.
"""

from copy import deepcopy
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configs.config import DEFAULTS
from training.mcore import MCoreRuntime, _cpu, _optimizers
from training.optimizer import parameter_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = deepcopy(DEFAULTS)
    config["distributed"].update(backend="mindspeed", bucket_size=64)
    config["training"].update(gradient_accumulation_steps=2, max_train_steps=3)
    config["optimizer"].update(
        warmup_steps=0,
        decay_steps=0,
        backbone_lr=1e-3,
        generator_lr=2e-3,
        projector_lr=3e-3,
        embedding_lr=1e-3,
        vision_lr=1e-3,
    )
    runtime = MCoreRuntime(config)
    from models.generator_wan21 import _FP32Linear

    class FunctionalGDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(8, 8, 4, groups=8)

        def forward(self, x):
            return F.conv1d(
                x.transpose(1, 2),
                self.conv1d.weight,
                self.conv1d.bias,
                groups=8,
                padding=3,
            )[..., : x.shape[1]].transpose(1, 2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.ModuleDict(
                dict(proj=nn.Linear(8, 8), linear_attn=FunctionalGDN())
            )
            self.generator = nn.ModuleDict(
                dict(
                    condition_proj=nn.Linear(8, 8),
                    time=nn.Sequential(_FP32Linear(8, 8), nn.SiLU(), _FP32Linear(8, 8)),
                )
            )
            self.lm_head = nn.Linear(8, 2)

        def forward(self, x, video):
            hidden = self.backbone["linear_attn"](self.backbone["proj"](x))
            loss = self.lm_head(hidden).float().square().mean()
            if video:
                value = self.generator["condition_proj"](hidden)
                loss = loss + self.generator["time"](value).square().mean()
            return loss

    torch.manual_seed(510)
    base = Model()
    reference = deepcopy(base).to(device=runtime.device, dtype=torch.bfloat16)
    wrapped, optimizer, scheduler = runtime.prepare_model(base)
    assert len(runtime.functional_hooks) == 1
    master_map = {
        id(p): nn.Parameter(p.detach().float().clone()) for p in reference.parameters()
    }
    reference_groups = parameter_groups(reference, config)
    for group in reference_groups:
        group["params"] = [master_map[id(p)] for p in group["params"]]
    reference_optimizer = torch.optim.AdamW(
        reference_groups,
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
        fused=True,
    )
    report = []
    for step in range(3):
        optimizer.zero_grad()
        accumulated = {
            id(p): torch.zeros_like(p, dtype=torch.float32)
            for p in reference.parameters()
        }
        for micro in range(2):
            seed = 1000 + step * 100 + micro * 10
            for rank in range(runtime.num_processes):
                x = torch.randn(
                    2, 5, 8, generator=torch.Generator().manual_seed(seed + rank)
                ).to(runtime.device, torch.bfloat16)
                video = step != 1 and (rank + micro) % 2 == 0
                reference.zero_grad(set_to_none=True)
                with runtime.autocast():
                    loss = reference(x, video) / (2 * runtime.num_processes)
                loss.backward()
                for p in reference.parameters():
                    if p.grad is not None:
                        accumulated[id(p)].add_(p.grad.float())
                if rank == runtime.process_index:
                    local_x, local_video = x, video
            with runtime.accumulate(wrapped):
                with runtime.autocast():
                    loss = wrapped(local_x, local_video) / 2
                runtime.backward(loss)
        for p in reference.parameters():
            master_map[id(p)].grad = accumulated[id(p)]
        torch.nn.utils.clip_grad_norm_(
            list(master_map.values()), config["training"]["max_grad_norm"]
        )
        reference_optimizer.step()
        with torch.no_grad():
            for p in reference.parameters():
                p.copy_(master_map[id(p)])
        optimizer.step()
        scheduler.step()
        runtime.synchronize_parameters()
        maximum = 0.0
        for (name, actual), (_, expected) in zip(
            base.named_parameters(), reference.named_parameters()
        ):
            maximum = max(
                maximum, float((actual.float() - expected.float()).abs().max())
            )
            torch.testing.assert_close(
                actual, expected, rtol=0.008, atol=0.0003, msg=name
            )
        # Every DP replica must contain the entire same updated BF16 model.
        import torch.distributed as dist

        for p in base.parameters():
            other = p.detach().clone()
            dist.broadcast(other, src=0, group=runtime.dp_group)
            assert torch.equal(p, other)
        report.append(
            dict(
                step=step + 1,
                max_abs_reference=maximum,
                absent_final=runtime.last_absent_gradients,
                grad_norm=runtime.last_grad_norm,
            )
        )
    directory = Path(args.output)
    runtime.save_state(directory / "checkpoint")
    saved_model = _cpu(base.state_dict())
    saved_model = {key: value.clone() for key, value in saved_model.items()}
    original = torch.load(
        directory / "checkpoint" / f"optimizer_rank_{runtime.process_index:05d}.pt",
        weights_only=False,
    )
    with torch.no_grad():
        for p in base.parameters():
            p.add_(1)
        for opt in _optimizers(optimizer.core):
            for group in opt.optimizer.param_groups:
                for p in group["params"]:
                    p.add_(2)
    runtime.load_state(directory / "checkpoint")
    runtime.save_state(directory / "restored")
    restored = torch.load(
        directory / "restored" / f"optimizer_rank_{runtime.process_index:05d}.pt",
        weights_only=False,
    )

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    equal(original, restored)
    equal(saved_model, _cpu(base.state_dict()))
    payload = dict(
        rank=runtime.process_index,
        world=runtime.num_processes,
        updates=report,
        exact_restore=True,
        replicas_equal=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
    )
    (directory / f"rank_{runtime.process_index}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload), flush=True)
    runtime.end_training()


if __name__ == "__main__":
    main()
