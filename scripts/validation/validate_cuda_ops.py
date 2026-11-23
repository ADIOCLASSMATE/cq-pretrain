"""Check CUDA mask semantics, CE gradients and upstream FP32 wgrad accumulation."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from data.batch import make_mask
from ops.cross_entropy import linear_cross_entropy
from ops.functional import attention, fused_wgrad


def error(actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(77)
    results = {}
    x = torch.randn(73, 96, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(73, 64, device="cuda", dtype=torch.bfloat16)
    dw = torch.randn(64, 96, device="cuda", dtype=torch.float32)
    expected = dw.cpu().double() + dy.cpu().double().T @ x.cpu().double()
    for start, end in ((0, 31), (31, 73)):
        fused_wgrad(x[start:end], dy[start:end], dw)
    results["fused_wgrad_two_chunks"] = error(dw, expected)
    assert results["fused_wgrad_two_chunks"] < 1e-5, results

    for direct in (False, True):
        x = torch.randn(61, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = (torch.randn(320, 96, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
        xr, wr = (t.detach().cpu().double().requires_grad_() for t in (x, w))
        targets = torch.randint(320, (61,), device="cuda")
        if direct:
            w.main_grad = torch.zeros_like(w, dtype=torch.float32)
            w._ce_direct_accumulation = True
            w.grad_added_to_main_grad = False
            w.zero_out_wgrad = True
        y = linear_cross_entropy(x, w, targets, 17)
        yr = F.cross_entropy(F.linear(xr, wr), targets.cpu())
        y.backward()
        yr.backward()
        results[f"ce_direct_{direct}"] = dict(
            loss=error(y, yr), dx=error(x.grad, xr.grad), dw=error(w.main_grad if direct else w.grad, wr.grad)
        )
        if direct:
            assert w.grad_added_to_main_grad

    # Exercise the existing CPU mask planner, including image-like ties,
    # padding and unconditional query rows; the reference is explicit FP64.
    sigma = torch.arange(23).expand(2, -1).clone()
    sigma[:, 4:8] = 4
    segments = torch.zeros_like(sigma)
    segments[:, -2:] = -1
    mask, valid = make_mask(sigma, segments)
    inputs = [
        torch.randn(n, 2, h, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for n, h in ((46, 4), (23, 2), (23, 2))
    ]
    refs = [t.detach().cpu().double().requires_grad_() for t in inputs]
    y = attention(*inputs, mask.cuda(), valid.cuda())
    q, k, v = [t.permute(1, 2, 0, 3) for t in refs]
    k, v = [t.repeat_interleave(2, 1) for t in (k, v)]
    probabilities = ((q @ k.transpose(-1, -2)) / math.sqrt(32)).masked_fill(mask, -torch.inf).softmax(-1)
    yr = (probabilities @ v).permute(2, 0, 1, 3).flatten(-2)
    yr = yr * valid.T[..., None]
    upstream = torch.randn_like(y)
    (y * upstream).sum().backward()
    (yr * upstream.cpu().double()).sum().backward()
    results["attention"] = dict(
        output=error(y, yr), **{name: error(x.grad, ref.grad) for name, x, ref in zip(("dq", "dk", "dv"), inputs, refs)}
    )
    errors = [v for row in results.values() for v in (row.values() if isinstance(row, dict) else [row])]
    passed = all(math.isfinite(v) and v < 0.035 for v in errors)
    report = dict(passed=passed, device=torch.cuda.get_device_name(), tolerance=0.035, errors=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert passed, "CUDA operator gate failed"


if __name__ == "__main__":
    main()
