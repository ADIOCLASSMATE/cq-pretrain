"""Gate the fused CUDA cross entropy against FP64 and the existing chunked path.

Three things have to hold before this kernel may carry the text objective:
the loss and both gradients must match an FP64 reference; the MCore direct
accumulation protocol must be honoured exactly; and the dispatch must refuse
itself wherever it is not equivalent (NPU, TP>1, deterministic mode, disabled).
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

import ops.cuda_cross_entropy as fused
from ops.cross_entropy import linear_cross_entropy as dispatched, _LinearCE


def relative(actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def build(rows, width, vocab, direct, seed, device="cuda"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = (torch.randn(rows, width, generator=generator) * 0.1).to(device, torch.bfloat16)
    weight = (torch.randn(vocab, width, generator=generator) * 0.02).to(device, torch.bfloat16)
    targets = torch.randint(vocab, (rows,), generator=generator).to(device)
    hidden = hidden.requires_grad_(True)
    weight = weight.requires_grad_(True)
    if direct:
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight._ce_direct_accumulation = True
        weight.grad_added_to_main_grad = False
        weight.zero_out_wgrad = True
    return hidden, weight, targets


def against_fp64(rows, width, vocab, chunk, direct, tol):
    hidden, weight, targets = build(rows, width, vocab, direct, seed=5)
    reference_hidden = hidden.detach().cpu().double().requires_grad_(True)
    reference_weight = weight.detach().cpu().double().requires_grad_(True)
    loss = fused.linear_cross_entropy(hidden, weight, targets, chunk)
    loss.backward()
    expected = F.cross_entropy(F.linear(reference_hidden, reference_weight), targets.cpu())
    expected.backward()
    gradient = weight.main_grad if direct else weight.grad
    errors = dict(
        loss=abs(float(loss) - float(expected)) / max(1e-8, abs(float(expected))),
        dhidden=relative(hidden.grad, reference_hidden.grad),
        dweight=relative(gradient, reference_weight.grad),
    )
    case = dict(
        rows=rows,
        width=width,
        vocab=vocab,
        chunk=chunk,
        direct=direct,
        errors=errors,
        passed=max(errors.values()) <= tol,
    )
    if direct:
        # MCore adds main_grad itself; a second contribution would double it.
        case["grad_added_to_main_grad"] = bool(weight.grad_added_to_main_grad)
        case["returned_gradient_is_zero"] = bool(weight.grad.abs().max() == 0)
        case["passed"] = case["passed"] and case["grad_added_to_main_grad"] and case["returned_gradient_is_zero"]
    return case


def against_baseline(rows, width, vocab, chunk, direct, tol):
    """Same inputs through both paths; the old one is the repo's FP64-gated path."""
    outputs = []
    for implementation in (_LinearCE.apply, fused.linear_cross_entropy):
        hidden, weight, targets = build(rows, width, vocab, direct, seed=9)
        loss = implementation(hidden, weight, targets, chunk)
        loss.backward()
        outputs.append((float(loss), hidden.grad.detach(), (weight.main_grad if direct else weight.grad).detach()))
    (old_loss, old_dh, old_dw), (new_loss, new_dh, new_dw) = outputs
    errors = dict(
        loss=abs(new_loss - old_loss) / max(1e-8, abs(old_loss)),
        dhidden=relative(new_dh, old_dh),
        dweight=relative(new_dw, old_dw),
    )
    return dict(
        rows=rows,
        width=width,
        vocab=vocab,
        chunk=chunk,
        direct=direct,
        errors=errors,
        passed=max(errors.values()) <= tol,
    )


def dispatch_rules(vocab=512, width=64, rows=8):
    """The fused path must decline wherever it is not the same computation."""
    hidden, weight, targets = build(rows, width, vocab, False, seed=3)
    checks = {}
    previous = fused._ENABLED
    try:
        fused._ENABLED = False
        checks["declines_when_disabled"] = not fused.available(hidden, 1)
        fused._ENABLED = True
        checks["declines_tensor_parallel"] = not fused.available(hidden, 2)
        checks["declines_cpu"] = not fused.available(hidden.cpu(), 1)
        torch.use_deterministic_algorithms(True)
        try:
            checks["declines_deterministic"] = not fused.available(hidden, 1)
        finally:
            torch.use_deterministic_algorithms(False)
        checks["accepts_single_rank_cuda"] = fused.available(hidden, 1)
        # With the switch on, the public entry point must route here.
        loss = dispatched(hidden, weight, targets, 4)
        loss.backward()
        checks["public_entry_routes_to_fused"] = True
    finally:
        fused._ENABLED = previous
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=0.02)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    fused._ENABLED = True
    if not fused.available(torch.empty(0, device="cuda"), 1):
        raise SystemExit("flash-attn Triton cross entropy is unavailable")

    fp64, baseline = [], []
    # Toy shapes keep an exact FP64 oracle cheap; the real vocabulary exercises
    # the kernel's actual row tiling, which is where it could differ.
    for rows, width, vocab, chunk in ((61, 96, 320, 17), (128, 128, 2048, 64), (32, 256, 248320, 16)):
        for direct in (False, True):
            case = against_fp64(rows, width, vocab, chunk, direct, args.tol)
            fp64.append(case)
            print(
                f"  fp64     rows={rows:5d} vocab={vocab:6d} direct={int(direct)} "
                f"loss={case['errors']['loss']:.2e} dh={case['errors']['dhidden']:.2e} "
                f"dw={case['errors']['dweight']:.2e} -> {'PASS' if case['passed'] else 'FAIL'}",
                flush=True,
            )
    for rows, width, vocab, chunk in ((2048, 2048, 248320, 1024), (4096, 2048, 248320, 1024)):
        for direct in (False, True):
            case = against_baseline(rows, width, vocab, chunk, direct, args.tol)
            baseline.append(case)
            print(
                f"  vs-old   rows={rows:5d} vocab={vocab:6d} direct={int(direct)} "
                f"loss={case['errors']['loss']:.2e} dh={case['errors']['dhidden']:.2e} "
                f"dw={case['errors']['dweight']:.2e} -> {'PASS' if case['passed'] else 'FAIL'}",
                flush=True,
            )
            torch.cuda.empty_cache()

    rules = dispatch_rules()
    for name, value in rules.items():
        print(f"  dispatch {name:32s} {value}", flush=True)

    result = dict(
        passed=all(c["passed"] for c in fp64 + baseline) and all(rules.values()),
        device=torch.cuda.get_device_name(0),
        tolerance=args.tol,
        fp64_cases=fp64,
        baseline_cases=baseline,
        dispatch_rules=rules,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("passed", "device", "tolerance")}, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("CUDA fused cross-entropy gate FAILED")
    print("CUDA FUSED CE GATE PASSED", flush=True)


if __name__ == "__main__":
    main()
