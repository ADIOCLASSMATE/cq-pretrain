"""Measure FP8 with the costs real training pays, not just the GEMM.

The earlier ceiling benchmark fed pre-quantized tensors with fixed scales. A
training step cannot: every activation must be cast each time it is produced,
its amax must be reduced for the next step's scale, and the backward pass needs
the transposed operands. This measures a forward linear the way a training loop
would actually run it, so the decision to invest in FP8 rests on the realistic
number rather than the bound.
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
    ("backbone attn o", 8192, 2048, 2048),
    ("gdn in_proj_all", 8192, 2048, 8224),
    ("gdn out_proj", 8192, 2048, 2048),
    ("dit ffn up", 4096, 1536, 8960),
    ("dit ffn down", 4096, 8960, 1536),
]


def bench(fn, n=30, warmup=8):
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
    dev = "cuda"
    fp8 = torch.float8_e4m3fn
    rows = []
    for name, M, K, N in SHAPES:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        sx = torch.ones((), device=dev)
        sw = torch.ones((), device=dev)

        def bf16():
            return torch.nn.functional.linear(x, w)

        # Weight casts amortize over gradient accumulation; activation casts
        # and their amax do not -- they happen every microbatch.
        wq = w.to(fp8)

        def fp8_prequantized():
            return torch._scaled_mm(x.to(fp8), wq.T, scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)

        def fp8_with_amax():
            amax = x.abs().amax()
            scale = (448.0 / amax.clamp_min(1e-12)).float()
            xq = (x * scale).to(fp8)
            return torch._scaled_mm(xq, wq.T, scale_a=1.0 / scale, scale_b=sw, out_dtype=torch.bfloat16)

        t_bf16 = bench(bf16)
        t_pre = bench(fp8_prequantized)
        try:
            t_amax = bench(fp8_with_amax)
        except Exception as exc:  # scale shape rules vary by build
            t_amax = float("nan")
            print(f"  {name}: amax variant unavailable: {type(exc).__name__}")
        rows.append(
            dict(
                name=name,
                M=M,
                K=K,
                N=N,
                bf16_ms=t_bf16,
                fp8_cast_included_ms=t_pre,
                fp8_cast_and_amax_ms=t_amax,
                ratio_cast_included=t_bf16 / t_pre,
                ratio_cast_and_amax=t_bf16 / t_amax if t_amax == t_amax else None,
            )
        )

    print(f"{'shape':24s} {'bf16 ms':>9s} {'+cast':>9s} {'+cast/amax':>11s} {'gain(cast)':>11s} {'gain(amax)':>11s}")
    for r in rows:
        g2 = f"{r['ratio_cast_and_amax']:.2f}x" if r["ratio_cast_and_amax"] else "n/a"
        print(
            f"{r['name']:24s} {r['bf16_ms']:9.3f} {r['fp8_cast_included_ms']:9.3f} "
            f"{r['fp8_cast_and_amax_ms']:11.3f} {r['ratio_cast_included']:10.2f}x {g2:>11s}"
        )

    tot_bf16 = sum(r["bf16_ms"] for r in rows)
    tot_pre = sum(r["fp8_cast_included_ms"] for r in rows)
    tot_amax = sum(r["fp8_cast_and_amax_ms"] for r in rows if r["fp8_cast_and_amax_ms"] == r["fp8_cast_and_amax_ms"])
    summary = dict(
        device=torch.cuda.get_device_name(0),
        weighted_gain_cast_included=tot_bf16 / tot_pre,
        weighted_gain_cast_and_amax=tot_bf16 / tot_amax if tot_amax else None,
        shapes=rows,
        note=(
            "Forward only. The activation cast is charged to FP8 because a training "
            "step produces fresh activations each microbatch; the weight cast is not, "
            "because it amortizes over gradient accumulation. Backward would additionally "
            "need transposed FP8 operands. GEMM was 41.8% of kernel time in the all-on "
            "profile, so the step-level bound is that share times this gain."
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2))
    print(f"\n  weighted, activation cast included: {summary['weighted_gain_cast_included']:.2f}x")
    if summary["weighted_gain_cast_and_amax"]:
        print(f"  weighted, cast + amax included:     {summary['weighted_gain_cast_and_amax']:.2f}x")
    gain = tot_bf16 / tot_amax if tot_amax else tot_bf16 / tot_pre
    print(f"  implied step-level bound if every GEMM converted: {1 / (1 - 0.418 * (1 - 1 / gain)):.3f}x")


if __name__ == "__main__":
    main()
