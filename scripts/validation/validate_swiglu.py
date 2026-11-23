"""Gate the Inductor-fused SwiGLU against this repo's own reference expression.

The reference is the exact code the fused path replaces: chunk the packed
projection and compute silu(gate)*up. Both run on the same strided layout the
model actually produces, so a kernel that mishandles strides fails here.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def reference(projected):
    gate, up = projected.chunk(2, -1)
    return F.silu(gate) * up


def relerr(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tol", type=float, default=0.02)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    import ops.cuda_fused as cuda_fused

    torch.manual_seed(5)
    cases, ok = [], True
    # Shapes from the production profile: [tokens, 1, 2*ffn] with ffn=6144,
    # plus a DiT-sized case and an odd length to exercise dynamic shapes.
    for tokens, ffn in ((16382, 6144), (8192, 6144), (4096, 6144), (777, 6144)):
        x = torch.randn(tokens, 1, 2 * ffn, device="cuda", dtype=torch.bfloat16) * 0.5
        xr = x.detach().clone().requires_grad_()
        xf = x.detach().clone().requires_grad_()
        yr = reference(xr)
        yf = cuda_fused.swiglu(xf)
        u = torch.randn_like(yr)
        (yr * u).sum().backward()
        (yf * u).sum().backward()
        # The fused path must see the same strided views the model produces.
        strided = not xf.chunk(2, -1)[0].is_contiguous()
        e = dict(output=relerr(yf, yr), dx=relerr(xf.grad, xr.grad))
        passed = all(v <= a.tol for v in e.values()) and strided
        ok &= passed
        cases.append(dict(tokens=tokens, ffn=ffn, strided_input=strided, errors=e, passed=passed))
        print(
            f"  swiglu [{tokens},1,{2 * ffn}] strided={strided} "
            f"output={e['output']:.2e} dx={e['dx']:.2e} -> {'PASS' if passed else 'FAIL'}"
        )

    # The model runs this inside torch.utils.checkpoint(use_reentrant=True).
    # Recompute replays the compiled graph under a fresh autograd context; a
    # fused path that only works outside checkpointing would pass every case
    # above and still corrupt training.
    from torch.utils.checkpoint import checkpoint

    def block(fn, x):
        return checkpoint(fn, x, use_reentrant=True)

    x = torch.randn(4096, 1, 12288, device="cuda", dtype=torch.bfloat16) * 0.5
    xr = x.detach().clone().requires_grad_()
    xf = x.detach().clone().requires_grad_()
    yr = block(reference, xr)
    yf = block(cuda_fused.swiglu, xf)
    u = torch.randn_like(yr)
    (yr * u).sum().backward()
    (yf * u).sum().backward()
    e = dict(output=relerr(yf, yr), dx=relerr(xf.grad, xr.grad))
    passed = all(v <= a.tol for v in e.values())
    ok &= passed
    cases.append(dict(tokens=4096, ffn=6144, reentrant_checkpoint=True, errors=e, passed=passed))
    print(
        f"  swiglu under reentrant checkpoint: output={e['output']:.2e} "
        f"dx={e['dx']:.2e} -> {'PASS' if passed else 'FAIL'}"
    )

    # Microbenchmark at the production shape, forward+backward.
    def timed(fn, x):
        for _ in range(3):
            y = fn(x)
            (y * y).sum().backward()
            x.grad = None
        torch.cuda.synchronize()
        s, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(10):
            y = fn(x)
            (y * y).sum().backward()
            x.grad = None
        e2.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e2) / 10

    x = torch.randn(16382, 1, 12288, device="cuda", dtype=torch.bfloat16).requires_grad_()
    tr, tf = timed(reference, x), timed(cuda_fused.swiglu, x)
    print(f"  fwd+bwd [16382,1,12288]: reference {tr:.3f} ms  fused {tf:.3f} ms -> {tr / tf:.2f}x")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            dict(passed=bool(ok), tolerance=a.tol, cases=cases, reference_ms=tr, fused_ms=tf, speedup=tr / tf), indent=2
        )
    )
    print("FUSED SWIGLU GATE " + ("PASSED" if ok else "FAILED"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
