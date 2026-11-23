"""Measure TransformerEngine's own FP8 Linear, not a hand-rolled cast.

TE keeps amax history and fuses the cast where it can, so it should beat the
naive "separate amax pass" measurement. This is the number the decision to
invest in FP8 should rest on.
"""

import argparse
import json
import time
from pathlib import Path

import torch

SHAPES = [
    ("backbone mlp gate_up", 8192, 2048, 12288),
    ("backbone mlp down", 8192, 6144, 2048),
    ("backbone attn q", 8192, 2048, 4096),
    ("gdn in_proj_all", 8192, 2048, 8224),
    ("dit ffn up", 4096, 1536, 8960),
    ("dit ffn down", 4096, 8960, 1536),
]


def bench(fn, n=25, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    print("TE", getattr(__import__("transformer_engine"), "__version__", "?"))
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    rows = []
    for name, M, K, N in SHAPES:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ref = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
        lin = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16)

        def bf16_fwdbwd():
            y = ref(x)
            y.backward(torch.ones_like(y))

        def te_bf16_fwdbwd():
            y = lin(x)
            y.backward(torch.ones_like(y))

        def te_fp8_fwdbwd():
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = lin(x)
            y.backward(torch.ones_like(y))

        t_ref = bench(bf16_fwdbwd)
        t_te_bf16 = bench(te_bf16_fwdbwd)
        try:
            t_te_fp8 = bench(te_fp8_fwdbwd)
            ok = True
        except Exception as exc:
            print(f"  {name}: FP8 path failed: {type(exc).__name__}: {str(exc)[:80]}")
            t_te_fp8, ok = float("nan"), False
        rows.append(
            dict(
                name=name,
                M=M,
                K=K,
                N=N,
                torch_bf16_ms=t_ref,
                te_bf16_ms=t_te_bf16,
                te_fp8_ms=t_te_fp8,
                gain_vs_torch_bf16=t_ref / t_te_fp8 if ok else None,
                gain_vs_te_bf16=t_te_bf16 / t_te_fp8 if ok else None,
            )
        )

    print(f"\n{'shape':24s} {'torch bf16':>11s} {'te bf16':>9s} {'te fp8':>9s} {'vs torch':>9s} {'vs te':>8s}")
    for r in rows:
        g1 = f"{r['gain_vs_torch_bf16']:.2f}x" if r["gain_vs_torch_bf16"] else "n/a"
        g2 = f"{r['gain_vs_te_bf16']:.2f}x" if r["gain_vs_te_bf16"] else "n/a"
        print(
            f"{r['name']:24s} {r['torch_bf16_ms']:11.3f} {r['te_bf16_ms']:9.3f} {r['te_fp8_ms']:9.3f} {g1:>9s} {g2:>8s}"
        )

    good = [r for r in rows if r["gain_vs_torch_bf16"]]
    weighted = (sum(r["torch_bf16_ms"] for r in good) / sum(r["te_fp8_ms"] for r in good)) if good else None
    summary = dict(
        device=torch.cuda.get_device_name(0),
        shapes=rows,
        weighted_gain_vs_torch_bf16=weighted,
        note=(
            "Forward+backward through TE's own FP8 Linear with delayed scaling, "
            "which is how FP8 would actually be adopted. Compared against the "
            "torch BF16 Linear the model uses today. GEMM was 41.8% of kernel "
            "time in the all-on profile."
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2))
    if weighted:
        print(f"\n  weighted TE FP8 vs torch BF16: {weighted:.2f}x")
        print(f"  implied step-level bound if every GEMM converted: {1 / (1 - 0.418 * (1 - 1 / weighted)):.3f}x")


if __name__ == "__main__":
    main()
