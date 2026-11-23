"""Measure the FP8 ceiling on this GPU at the shapes this model actually runs.

Adopting FP8 means replacing every tensor-parallel Linear with a
TransformerEngine one, which changes parameter structure, the checkpoint schema
and the gradient protocol. That is only worth considering if the hardware
actually pays for it at our shapes, so measure the raw GEMM first:
``torch._scaled_mm`` is the same cuBLASLt FP8 path TE would reach, without any
refactor. A ratio near 1.0 means the refactor cannot win.
"""

import argparse
import json
import time
from pathlib import Path

import torch


def timed(fn, iterations=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) / iterations * 1e3


def bf16_gemm(m, k, n, device="cuda"):
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    return lambda: torch.nn.functional.linear(a, b)


def fp8_gemm(m, k, n, device="cuda"):
    """Row-major A, column-major B, per-tensor scales: the supported sm89 form."""
    a = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device=device).to(torch.float8_e4m3fn).t()
    scale_a = torch.tensor(1.0, device=device)
    scale_b = torch.tensor(1.0, device=device)

    def run():
        return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)

    run()  # surface unsupported layouts/dtypes immediately
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    args = parser.parse_args()
    tokens = args.tokens
    width, ffn, vocab = 2048, 6144, 248320
    heads, kv_heads, head_dim = 8, 2, 256
    dit_width, dit_ffn = 1536, 8960
    dit_tokens = 4096

    shapes = [
        ("lm_head / CE vocab", tokens, width, vocab),
        ("backbone mlp gate_up", tokens, width, 2 * ffn),
        ("backbone mlp down", tokens, ffn, width),
        ("backbone attn q", tokens, width, 2 * heads * head_dim),
        ("backbone attn kv", tokens, width, 2 * kv_heads * head_dim),
        ("backbone attn o", tokens, heads * head_dim, width),
        ("gdn in_proj_all", tokens, width, 2 * 2048 + 2 * 2048 + 32),
        ("gdn out_proj", tokens, 2048, width),
        ("dit ffn up", dit_tokens, dit_width, dit_ffn),
        ("dit ffn down", dit_tokens, dit_ffn, dit_width),
        ("dit attn qkv", dit_tokens, dit_width, 3 * dit_width),
    ]

    capability = torch.cuda.get_device_capability()
    print(f"device {torch.cuda.get_device_name(0)} sm{capability[0]}{capability[1]}", flush=True)
    print(
        f"{'shape':<24}{'M':>7}{'K':>7}{'N':>8}{'bf16 ms':>10}{'fp8 ms':>10}{'ratio':>8}{'bf16 TF':>9}{'fp8 TF':>9}",
        flush=True,
    )
    results = []
    for name, m, k, n in shapes:
        flops = 2 * m * k * n
        try:
            bf16_ms = timed(bf16_gemm(m, k, n))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        try:
            fp8_ms = timed(fp8_gemm(m, k, n))
            ratio = bf16_ms / fp8_ms
        except Exception as error:
            fp8_ms, ratio = float("nan"), float("nan")
            results.append(
                dict(name=name, m=m, k=k, n=n, bf16_ms=bf16_ms, fp8_error=f"{type(error).__name__}: {error}")
            )
            print(f"{name:<24}{m:>7}{k:>7}{n:>8}{bf16_ms:>10.3f}{'unsupported':>10}", flush=True)
            torch.cuda.empty_cache()
            continue
        results.append(
            dict(
                name=name,
                m=m,
                k=k,
                n=n,
                bf16_ms=bf16_ms,
                fp8_ms=fp8_ms,
                ratio=ratio,
                bf16_tflops=flops / (bf16_ms * 1e-3) / 1e12,
                fp8_tflops=flops / (fp8_ms * 1e-3) / 1e12,
            )
        )
        print(
            f"{name:<24}{m:>7}{k:>7}{n:>8}{bf16_ms:>10.3f}{fp8_ms:>10.3f}{ratio:>8.2f}"
            f"{flops / (bf16_ms * 1e-3) / 1e12:>9.1f}{flops / (fp8_ms * 1e-3) / 1e12:>9.1f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    usable = [r for r in results if r.get("ratio") == r.get("ratio") and "ratio" in r]
    if usable:
        weighted = sum(r["bf16_ms"] for r in usable) / sum(r["fp8_ms"] for r in usable)
        print(f"\nGEMM-time-weighted FP8 speedup across these shapes: {weighted:.2f}x", flush=True)
    else:
        weighted = None
        print("\nNo FP8 GEMM succeeded on this device.", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(0),
                capability=list(capability),
                torch=torch.__version__,
                tokens=tokens,
                weighted_speedup=weighted,
                shapes=results,
            ),
            indent=2,
        )
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
