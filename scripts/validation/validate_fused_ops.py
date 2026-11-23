"""Check the fused SwiGLU and causal convolution against this repo's references.

Both kernels are published upstream; what needs proving is that routing to them
preserves exactly what the reference path computed, including each document
staying independent in the convolution (no token may read across a document
boundary) and every parameter gradient.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.config import tiny_config
from data.batch import fake_batch


def relerr(actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def reference_swiglu(projected):
    gate, up = projected.chunk(2, -1)
    return F.silu(gate) * up


def reference_conv(x, weight, bounds):
    """The loop this replaces: an independent causal conv per packed document."""
    results = []
    for a, b in zip(bounds, bounds[1:]):
        y = F.conv1d(x[:, a:b].transpose(1, 2), weight, padding=weight.shape[-1] - 1, groups=x.shape[-1])[..., : b - a]
        results.append(F.silu(y).transpose(1, 2))
    return torch.cat(results, dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tol", type=float, default=0.02)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    import ops.cuda_fused as fused

    if not fused._OPS_ENABLED:
        raise SystemExit("Set PRETRAIN_CUDA_FUSED_OPS=1 to validate the fused ops")

    dev = "cuda"
    cases = []

    # --- Causal conv on real packed-document boundaries ----------------------
    for length, images in ((2048, 2), (3072, 3)):
        torch.manual_seed(7)
        cfg = tiny_config(linear_heads=16, linear_head_dim=128, qwen_fused_projections=True)
        cfg.text_assets = ".assets/Qwen3.5-2B-Base"
        cfg.vision_assets = ".assets/Qwen3.5-2B-Base"
        cfg.wan_assets = ".assets/Wan2.1-T2V-1.3B"
        batch = fake_batch(cfg, length, 1, images, seed=11)
        plan = batch.mainline.delta_plan
        bounds = plan.content_bounds
        channels = 6144
        kernel = 4
        x = torch.randn(1, int(bounds[-1]), channels, device=dev, dtype=torch.bfloat16) * 0.5
        w = torch.randn(channels, kernel, device=dev, dtype=torch.bfloat16) * 0.3
        x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
        w1, w2 = w.clone().requires_grad_(), w.clone().requires_grad_()
        cu = plan.content_cu.to(dev)
        y1 = fused.causal_conv1d(x1, w1, cu)
        y2 = reference_conv(x2, w2[:, None, :], bounds)
        u = torch.randn_like(y1)
        (y1.float() * u.float()).sum().backward()
        (y2.float() * u.float()).sum().backward()
        errs = dict(output=relerr(y1, y2), dx=relerr(x1.grad, x2.grad), dweight=relerr(w1.grad, w2.grad))

        # Causality / document isolation: perturbing a token must never change
        # any output at or before it, nor any output in another document.
        with torch.no_grad():
            base = fused.causal_conv1d(x, w, cu)
            doc0_end = int(bounds[1])
            probe = x.clone()
            probe[:, doc0_end:] += 3.0  # disturb every later document
            after = fused.causal_conv1d(probe, w, cu)
            cross_doc = float((after[:, :doc0_end] - base[:, :doc0_end]).abs().max())
            probe = x.clone()
            t = doc0_end // 2
            probe[:, t] += 3.0  # disturb one token
            after = fused.causal_conv1d(probe, w, cu)
            backward_leak = float((after[:, :t] - base[:, :t]).abs().max())
            forward_effect = float((after[:, t] - base[:, t]).abs().max())
        isolation = dict(cross_document_leak=cross_doc, acausal_leak=backward_leak, own_token_effect=forward_effect)
        cases.append(
            dict(
                op="causal_conv1d",
                length=length,
                documents=len(bounds) - 1,
                errors=errs,
                isolation=isolation,
                passed=(
                    max(errs.values()) <= a.tol and cross_doc == 0.0 and backward_leak == 0.0 and forward_effect > 0
                ),
            )
        )

    result = dict(
        passed=all(c["passed"] for c in cases), device=torch.cuda.get_device_name(0), tolerance=a.tol, cases=cases
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    for c in cases:
        extra = ""
        if "isolation" in c:
            i = c["isolation"]
            extra = (
                f" cross_doc={i['cross_document_leak']:.1e} acausal={i['acausal_leak']:.1e}"
                f" own={i['own_token_effect']:.2e}"
            )
        errs = " ".join(f"{k}={v:.2e}" for k, v in c["errors"].items())
        print(f"  {c['op']:14s} {errs}{extra} -> {'PASS' if c['passed'] else 'FAIL'}")
    if not result["passed"]:
        raise SystemExit("FUSED OPS GATE FAILED")
    print("FUSED OPS GATE PASSED")


if __name__ == "__main__":
    main()
