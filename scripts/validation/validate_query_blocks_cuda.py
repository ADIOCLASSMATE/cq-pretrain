"""Single-GPU gate for bidirectional visual-query blocks.

Runs one production-shaped GatedDelta layer and one GatedAttention layer on a
packed two-document layout whose visual queries read [prefix ; own block], and
compares the fused BF16 kernels (FLA chunk rule with private per-target
sequences, FlexAttention for content rows plus dense block attention) with the
FP64 CPU reference. It also times each layer with and without the interaction.
No distributed init is required.
"""
import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from configs.config import RunConfig
from data.batch import Batch
from data.batch_layout import build_plan
from models.backbone_qwen35 import GatedAttention, GatedDelta


def relerr(actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def layout(side=14, null=(False, False, True)):
    """Two packed documents, three targets; the last one unconditional."""
    grid = (1, 2 * side, 2 * side)
    n = side * side
    pieces = [(0, [("t", 96), ("i", n), ("t", 80), ("i", n), ("t", 40)]), (1, [("t", 300), ("i", n), ("t", 24)])]
    segments, sigma, rope, spans, text, openers = [], [], [], [], [], []
    position = 0
    for segment, elements in pieces:
        cursor = 0
        start = position
        for kind, length in elements:
            if kind == "t":
                for _ in range(length):
                    segments.append(segment); sigma.append(position); rope.append((cursor,) * 3)
                    if position % 3 == 0:
                        text.append(position)
                    position += 1; cursor += 1
            else:
                spans.append(list(range(position, position + length)))
                openers.append(start + 1)
                for local in range(length):
                    segments.append(segment); sigma.append(position - local)
                    rope.append((cursor, cursor + local // side, cursor + local % side))
                    position += 1
                cursor += side
    s = position
    empty = torch.empty(0)
    return Batch(torch.zeros(1, s, dtype=torch.long), empty, torch.tensor(spans), torch.tensor(text),
                 torch.zeros(1, s, dtype=torch.bool), torch.tensor(sigma)[None], torch.tensor(segments)[None],
                 torch.tensor(rope).float().T[:, :, None].contiguous(), empty, empty,
                 vision=SimpleNamespace(grid_thw=(grid,) * len(spans)),
                 generation_image_indices=torch.arange(len(spans)),
                 generation_unconditional=torch.tensor(null),
                 generation_null_positions=torch.tensor(openers))


def timed(function, repeats=5):
    function(); torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35_2b_wan.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tol", type=float, default=0.035)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    device = "cuda"
    base = RunConfig.load(args.config)
    report = dict(cases=[], timing={})
    for null in ("masked", "bos"):
        cfg = replace(base, query_interaction="bidirectional", null_condition=null)
        batch = layout()
        plan = build_plan(batch, cfg)
        s = batch.ids.shape[1]
        rows = plan.hidden_length if hasattr(plan, "hidden_length") else s + len(plan.query_positions)
        block_rows = torch.cat([torch.arange(a, b) for a, b, _, _ in plan.query_blocks])
        on_device = plan.to(device)
        torch.manual_seed(7)
        for name, module in (("gdn", GatedDelta(cfg)), ("attention", GatedAttention(cfg))):
            reference = copy.deepcopy(module).double()
            layer = module.to(device=device, dtype=torch.bfloat16)
            if name == "gdn":
                layer.A_log.data = reference.A_log.detach().float().to(device)
                layer.norm.weight.data = reference.norm.weight.detach().float().to(device)
            reference.load_state_dict({k: v.detach().cpu().double() for k, v in layer.state_dict().items()})
            x = torch.randn(rows, 1, cfg.width, device=device, dtype=torch.bfloat16, requires_grad=True)
            xr = x.detach().cpu().double().requires_grad_()
            if name == "gdn":
                y, yr = layer(x, on_device.delta_plan), reference(xr, plan.delta_plan)
            else:
                rope = (on_device.rope_cos.to(x.dtype), on_device.rope_sin.to(x.dtype))
                y = layer(x, rope, on_device, s)
                yr = reference(xr, (plan.rope_cos.double(), plan.rope_sin.double()), plan, s, reference=True)
            u = torch.randn_like(y)
            (y[block_rows] * u[block_rows]).sum().backward()
            (yr[block_rows] * u[block_rows].cpu().double()).sum().backward()
            errors = dict(block_output=relerr(y[block_rows], yr[block_rows]), all_output=relerr(y, yr),
                          dx=relerr(x.grad, xr.grad))
            report["cases"].append(dict(layer=name, null=null, rows=rows, content=s,
                                        blocks=[list(b) for b in plan.query_blocks],
                                        events=len(plan.delta_plan.event_q), **errors,
                                        passed=all(v <= args.tol for v in errors.values())))
    # Cost of the interaction on one layer, forward + backward, same hidden rows.
    for interaction in ("none", "bidirectional"):
        cfg = replace(base, query_interaction=interaction)
        batch = layout(side=14, null=(False, False, False))
        plan = build_plan(batch, cfg)
        s = batch.ids.shape[1]
        rows = s + len(plan.query_positions)
        on_device = plan.to(device)
        x = torch.randn(rows, 1, cfg.width, device=device, dtype=torch.bfloat16, requires_grad=True)
        rope = (on_device.rope_cos.to(x.dtype), on_device.rope_sin.to(x.dtype))
        gdn = GatedDelta(cfg).to(device=device, dtype=torch.bfloat16)
        gdn.A_log.data = gdn.A_log.data.float(); gdn.norm.weight.data = gdn.norm.weight.data.float()
        attention = GatedAttention(cfg).to(device=device, dtype=torch.bfloat16)
        report["timing"][interaction] = dict(
            rows=rows, content=s, events=len(plan.delta_plan.event_q),
            gdn_ms=1e3 * timed(lambda: gdn(x, on_device.delta_plan).float().sum().backward()),
            attention_ms=1e3 * timed(lambda: attention(x, rope, on_device, s).float().sum().backward()))
    report["passed"] = all(case["passed"] for case in report["cases"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
