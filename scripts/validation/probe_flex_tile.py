"""Probe one FlexAttention tile configuration in an isolated process.

L40S (sm89) has ~99 KiB of shared memory per SM, while FlexAttention's default
tiles are sized for much larger budgets; at this backbone's head_dim=256 the
default forward kernel asks for 137 KiB and fails to compile. Shared-memory
feasibility cannot be retried inside one process: Inductor's async compile pool
does not recover after a kernel raises OutOfResources. So each candidate runs in
its own process, is validated against the dense-mask reference, timed, and
reported as a single JSON line. The caller picks the winner and records it.
"""

import argparse
import json
import sys
import time

import torch
from torch.nn import functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block-m", type=int, required=True)
    p.add_argument("--block-n", type=int, required=True)
    p.add_argument("--content", type=int, default=4096)
    p.add_argument("--queries", type=int, default=1024)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--num-stages", type=int, default=0, help="0 keeps the kernel default")
    p.add_argument("--num-warps", type=int, default=0)
    p.add_argument("--iters", type=int, default=6)
    a = p.parse_args()

    report = dict(block_m=a.block_m, block_n=a.block_n, num_stages=a.num_stages, num_warps=a.num_warps, ok=False)
    try:
        # Compile in-process: the subprocess worker pool is what wedges after a
        # failed kernel, and this probe is already isolated.
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = 1

        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        dev = "cuda"
        torch.manual_seed(0)
        s, nq = a.content, a.queries
        rows = s + nq

        # A CQ-shaped layout: two documents, an image block with tied sigma,
        # query rows reading strictly before their own sigma.
        segments = torch.zeros(s, dtype=torch.int32, device=dev)
        segments[s // 2 :] = 1
        sigma = torch.arange(s, dtype=torch.int32, device=dev)
        sigma[100:356] = 100  # tied image block -> bidirectional inside it
        qpos = torch.linspace(1, s - 1, nq).long().to(dev)
        row_sigma = torch.cat((sigma, sigma[qpos]))
        row_segment = torch.cat((segments, segments[qpos]))
        row_is_query = torch.cat(
            (torch.zeros(s, dtype=torch.bool, device=dev), torch.ones(nq, dtype=torch.bool, device=dev))
        )

        def mask_mod(b, h, q_idx, kv_idx):
            same = (row_segment[q_idx] == segments[kv_idx]) & (row_segment[q_idx] >= 0)
            strict = sigma[kv_idx] < row_sigma[q_idx]
            loose = sigma[kv_idx] <= row_sigma[q_idx]
            return same & torch.where(row_is_query[q_idx], strict, loose)

        block_mask = create_block_mask(mask_mod, None, None, rows, s, device=dev, BLOCK_SIZE=(128, 128), _compile=True)

        back = min(a.block_m, a.block_n)
        options = {
            "BLOCK_M": a.block_m,
            "BLOCK_N": a.block_n,
            "BLOCK_M1": back,
            "BLOCK_N1": back,
            "BLOCK_M2": back,
            "BLOCK_N2": back,
        }
        if a.num_stages:
            options["num_stages"] = a.num_stages
        if a.num_warps:
            options["num_warps"] = a.num_warps
        compiled = torch.compile(flex_attention, dynamic=False)

        q = torch.randn(1, a.heads, rows, a.head_dim, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, a.kv_heads, s, a.head_dim, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, a.kv_heads, s, a.head_dim, device=dev, dtype=torch.bfloat16)
        q, k, v = (t.requires_grad_() for t in (q, k, v))

        begin = time.perf_counter()
        out = compiled(q, k, v, block_mask=block_mask, enable_gqa=True, kernel_options=options)
        out.sum().backward()  # force the backward kernel to compile too
        torch.cuda.synchronize()
        report["compile_seconds"] = time.perf_counter() - begin

        # Correctness against the dense-mask reference the repo uses today.
        qi = torch.arange(rows, device=dev)[:, None]
        ki = torch.arange(s, device=dev)[None, :]
        allowed = mask_mod(None, None, qi, ki)
        repeat = a.heads // a.kv_heads
        kd, vd = (t.detach().repeat_interleave(repeat, 1) for t in (k, v))
        ref = F.scaled_dot_product_attention(q.detach(), kd, vd, attn_mask=allowed[None, None])
        err = (out.detach().float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
        report["relative_error"] = float(err)

        def timed(fn):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / a.iters * 1e3

        report["flex_fwd_ms"] = timed(
            lambda: compiled(q, k, v, block_mask=block_mask, enable_gqa=True, kernel_options=options)
        )
        report["dense_fwd_ms"] = timed(lambda: F.scaled_dot_product_attention(q, kd, vd, attn_mask=allowed[None, None]))
        report["speedup_fwd"] = report["dense_fwd_ms"] / max(report["flex_fwd_ms"], 1e-9)

        def timed_bwd(make):
            for _ in range(2):
                make().sum().backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                make().sum().backward()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / a.iters * 1e3

        kg, vg = (t.detach().requires_grad_() for t in (k, v))
        kgr = kg.repeat_interleave(repeat, 1)
        vgr = vg.repeat_interleave(repeat, 1)
        report["flex_fwdbwd_ms"] = timed_bwd(
            lambda: compiled(q, k, v, block_mask=block_mask, enable_gqa=True, kernel_options=options)
        )
        report["dense_fwdbwd_ms"] = timed_bwd(
            lambda: F.scaled_dot_product_attention(q, kgr, vgr, attn_mask=allowed[None, None])
        )
        report["speedup_fwdbwd"] = report["dense_fwdbwd_ms"] / max(report["flex_fwdbwd_ms"], 1e-9)
        report["ok"] = report["relative_error"] < 2e-2
    except Exception as error:  # noqa: BLE001 - the point is to report any failure
        report["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        text = str(error)
        if "shared memory" in text or "OutOfResources" in text:
            report["reason"] = "shared_memory"

    print("RESULT " + json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
