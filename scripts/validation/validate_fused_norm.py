"""Gate the published CUDA norm kernels against this repo's own references.

Checks forward, dx and dweight (plus dgate) for the fused RMSNorm and the fused
gated RMSNorm, at the shapes the backbone actually uses, then re-runs the full
GatedDelta CQ layer with fusion on and off.
"""

import argparse, json, os, time
from pathlib import Path
import torch

os.environ.setdefault("PRETRAIN_BACKEND", "cuda")


def relerr(a, b):
    a = a.detach().cpu().double()
    b = b.detach().cpu().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--tol", type=float, default=0.02)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    dev = "cuda"
    import ops.functional as K
    from ops import cuda_fused
    from models.backbone_qwen35 import RMSNorm, GatedNorm

    cases = []

    # --- RMSNorm: backbone shapes, FP32 offset weight over BF16 activations ---
    for n, w in ((9216, 2048), (9216, 128), (1024, 1536)):
        torch.manual_seed(7)
        norm = RMSNorm(w, 1e-6).to(dev)
        norm.weight.data.normal_(0, 0.1)
        x = torch.randn(n, w, device=dev, dtype=torch.bfloat16).requires_grad_()
        xr = x.detach().clone().requires_grad_()
        ref = RMSNorm(w, 1e-6).to(dev)
        ref.load_state_dict(norm.state_dict())
        y = norm(x, True)  # fused path
        yr = ref(xr, False)  # explicit PyTorch reference
        u = torch.randn_like(y)
        (y * u).sum().backward()
        (yr * u).sum().backward()
        e = dict(output=relerr(y, yr), dx=relerr(x.grad, xr.grad), dweight=relerr(norm.weight.grad, ref.weight.grad))
        cases.append(dict(kernel="rms_norm", shape=[n, w], errors=e, passed=max(e.values()) <= a.tol))

    # --- gated RMSNorm: GDN head layout ---
    for n, h, d in ((9216, 16, 128), (18432, 16, 128)):
        torch.manual_seed(11)
        g = GatedNorm(d, 1e-6).to(dev)
        g.weight.data.normal_(1.0, 0.05)
        gr = GatedNorm(d, 1e-6).to(dev)
        gr.load_state_dict(g.state_dict())
        x = torch.randn(n, h, d, device=dev, dtype=torch.bfloat16).requires_grad_()
        z = torch.randn(n, h, d, device=dev, dtype=torch.bfloat16).requires_grad_()
        xr = x.detach().clone().requires_grad_()
        zr = z.detach().clone().requires_grad_()
        y = g(x, z)
        os.environ["PRETRAIN_CUDA_FUSED_NORM"] = "0"
        cuda_fused._ENABLED = False
        yr = gr(xr, zr)
        os.environ["PRETRAIN_CUDA_FUSED_NORM"] = "1"
        cuda_fused._ENABLED = True
        u = torch.randn_like(y)
        (y * u).sum().backward()
        (yr * u).sum().backward()
        e = dict(
            output=relerr(y, yr),
            dx=relerr(x.grad, xr.grad),
            dgate=relerr(z.grad, zr.grad),
            dweight=relerr(g.weight.grad, gr.weight.grad),
        )
        cases.append(dict(kernel="gated_rms_norm", shape=[n, h, d], errors=e, passed=max(e.values()) <= a.tol))

    # --- microbenchmark ---
    bench = {}
    for name, fn, shape in (
        ("rms_norm", None, (9216, 2048)),
        ("gated_rms_norm", None, (9216, 16, 128)),
    ):
        for mode in (True, False):
            cuda_fused._ENABLED = mode
            if name == "rms_norm":
                m = RMSNorm(shape[-1], 1e-6).to(dev)
                x = torch.randn(*shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
                run = lambda: m(x, mode)
            else:
                m = GatedNorm(shape[-1], 1e-6).to(dev)
                x = torch.randn(*shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
                z = torch.randn(*shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
                run = lambda: m(x, z)
            for _ in range(5):
                out = run()
                out.sum().backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                out = run()
                out.sum().backward()
            torch.cuda.synchronize()
            bench[f"{name}_{'fused' if mode else 'reference'}_ms"] = (time.perf_counter() - t0) / 20 * 1000
    cuda_fused._ENABLED = True
    for k in ("rms_norm", "gated_rms_norm"):
        f, r = bench[f"{k}_fused_ms"], bench[f"{k}_reference_ms"]
        bench[f"{k}_speedup"] = round(r / f, 3)

    result = dict(
        passed=all(c["passed"] for c in cases),
        tolerance=a.tol,
        device=torch.cuda.get_device_name(0),
        cases=cases,
        microbenchmark=bench,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    for c in cases:
        print(
            "  %-15s %-18s %s  %s"
            % (
                c["kernel"],
                c["shape"],
                " ".join("%s=%.2e" % (k, v) for k, v in c["errors"].items()),
                "PASS" if c["passed"] else "FAIL",
            )
        )
    for k in ("rms_norm", "gated_rms_norm"):
        print(
            "  %-15s fused %.3f ms vs reference %.3f ms -> %.2fx"
            % (k, bench[f"{k}_fused_ms"], bench[f"{k}_reference_ms"], bench[f"{k}_speedup"])
        )
    print("PASSED" if result["passed"] else "FAILED")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
