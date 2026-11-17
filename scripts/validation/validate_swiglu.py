"""Gate the Inductor-fused SwiGLU against this repo's reference pair.

The gate and up operands are strided views of one packed projection. A kernel
that assumes contiguity returns wrong values without raising, so correctness
here is checked on genuinely strided inputs, against both an FP64 oracle and
the exact in-dtype expression the model would otherwise run.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ops import cuda_fused


def reference(projected):
    gate, up = projected.chunk(2, -1)
    return F.silu(gate) * up


def relerr(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--tol", type=float, default=0.02)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    if not cuda_fused._SWIGLU_ENABLED:
        raise SystemExit("Set PRETRAIN_CUDA_FUSED_SWIGLU=1 to gate the fused pair")
    torch.manual_seed(5)
    cases = []
    # Production backbone shape (tokens = content + CQ queries) and two others
    # so a tile-size assumption cannot pass by luck.
    for tokens, ffn in ((16382, 6144), (8192, 6144), (4096, 6144)):
        packed = torch.randn(tokens, 1, 2 * ffn, device="cuda", dtype=torch.bfloat16,
                             requires_grad=True)
        ref_in = packed.detach().clone().requires_grad_()
        oracle_in = packed.detach().double().cpu().requires_grad_()

        assert not packed.chunk(2, -1)[0].is_contiguous(), "fixture must be strided"
        used = cuda_fused.swiglu_enabled(packed)
        y = cuda_fused.swiglu(packed)
        y_ref = reference(ref_in)
        y_oracle = reference(oracle_in)

        u = torch.randn_like(y)
        (y * u).sum().backward()
        (y_ref * u).sum().backward()
        (y_oracle * u.detach().double().cpu()).sum().backward()

        row = dict(tokens=tokens, ffn=ffn, fused_used=bool(used),
                   vs_reference=dict(out=relerr(y, y_ref), dx=relerr(packed.grad, ref_in.grad)),
                   vs_fp64_oracle=dict(out=relerr(y.float().cpu(), y_oracle),
                                       dx=relerr(packed.grad.float().cpu(), oracle_in.grad)))
        row["passed"] = used and max(
            list(row["vs_reference"].values()) + list(row["vs_fp64_oracle"].values())) <= a.tol
        cases.append(row)
        print(f"  tokens={tokens:<6} used={used}  vs-ref out={row['vs_reference']['out']:.2e} "
              f"dx={row['vs_reference']['dx']:.2e}  vs-fp64 out={row['vs_fp64_oracle']['out']:.2e} "
              f"dx={row['vs_fp64_oracle']['dx']:.2e}  -> {'PASS' if row['passed'] else 'FAIL'}")

    # Timing on the production shape, forward plus backward.
    packed = torch.randn(16382, 1, 12288, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    u = torch.randn(16382, 1, 6144, device="cuda", dtype=torch.bfloat16)
    timing = {}
    for name, fn in (("reference", reference), ("fused", cuda_fused.swiglu)):
        for _ in range(3):
            packed.grad = None
            (fn(packed) * u).sum().backward()
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(10):
            packed.grad = None
            (fn(packed) * u).sum().backward()
        end.record()
        torch.cuda.synchronize()
        timing[name] = start.elapsed_time(end) / 10
    timing["speedup"] = timing["reference"] / timing["fused"]
    print(f"  fwd+bwd [16382,1,12288]: reference {timing['reference']:.3f} ms  "
          f"fused {timing['fused']:.3f} ms  -> {timing['speedup']:.2f}x")

    result = dict(passed=all(c["passed"] for c in cases), device=torch.cuda.get_device_name(0),
                  tolerance=a.tol, cases=cases, timing=timing)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("FUSED SWIGLU GATE FAILED")
    print("FUSED SWIGLU GATE PASSED")


if __name__ == "__main__":
    main()
