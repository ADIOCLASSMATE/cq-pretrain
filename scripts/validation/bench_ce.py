"""Measure the text cross entropy on the row counts this model actually produces.

The baseline is the repo's existing chunked path, which the CUDA operator gate
already checks against an FP64 reference, so agreement with it is the
correctness signal here; speed and peak memory are the decision.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import torch

from ops.cross_entropy import linear_cross_entropy as baseline_ce


def real_rows(patterns):
    """Supervised text tokens per pack, from completed production runs."""
    counts = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            for line in Path(path).read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for pack in row.get("packs", []):
                    if pack.get("text"):
                        counts.append(int(pack["text"]))
    return counts


def relative(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def make_inputs(rows, width, vocab, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = (torch.randn(rows, width, generator=generator, dtype=torch.float32) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )
    weight = (torch.randn(vocab, width, generator=generator, dtype=torch.float32) * 0.02).to(
        device=device, dtype=torch.bfloat16
    )
    targets = torch.randint(vocab, (rows,), generator=generator).to(device)
    return hidden, weight, targets


def prepare(weight, direct):
    weight = weight.detach().clone().requires_grad_(True)
    if direct:
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight._ce_direct_accumulation = True
        weight.grad_added_to_main_grad = False
        weight.zero_out_wgrad = True
    return weight


def measure(fn, hidden, weight, targets, chunk, direct, iterations):
    """Forward plus backward, as training runs it."""
    h = hidden.detach().clone().requires_grad_(True)
    w = prepare(weight, direct)
    for _ in range(3):  # warmup, including kernel compilation
        loss = fn(h, w, targets, chunk)
        loss.backward()
        h.grad = None
        w.grad = None
        if direct:
            w.main_grad.zero_()
            w.grad_added_to_main_grad = False
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    begin = time.perf_counter()
    for _ in range(iterations):
        loss = fn(h, w, targets, chunk)
        loss.backward()
        if direct:
            w.grad_added_to_main_grad = False
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - begin) / iterations
    peak = torch.cuda.max_memory_allocated() - before
    gradient = w.main_grad if direct else w.grad
    return dict(
        ms=seconds * 1e3,
        peak_gib=peak / 2**30,
        loss=float(loss.detach()),
        dh=h.grad.detach().clone(),
        dw=gradient.detach().clone(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--rows", type=int, nargs="*", default=[])
    parser.add_argument("--timing", nargs="*", default=[".runs/prod-1117/*/timing_rank_*.jsonl"])
    args = parser.parse_args()

    observed = real_rows(args.timing)
    if observed:
        observed.sort()
        quantile = lambda q: observed[min(len(observed) - 1, int(q * len(observed)))]
        profile = dict(packs=len(observed), min=observed[0], p50=quantile(0.5),
                       p90=quantile(0.9), max=observed[-1],
                       mean=sum(observed) / len(observed))
    else:
        profile = {}
    rows_to_test = args.rows or sorted({profile.get("p50", 4096), profile.get("p90", 6144),
                                        profile.get("max", 8192)})
    print(json.dumps(dict(observed_text_rows=profile, benchmarking=rows_to_test), indent=2), flush=True)

    candidates = {"baseline": baseline_ce}
    import ops.cuda_cross_entropy as fused

    fused._ENABLED = True
    if fused.available(torch.empty(0, device="cuda"), 1):
        candidates["flash_triton_ce"] = fused.linear_cross_entropy
    else:
        print("flash-attn Triton CE unavailable", flush=True)
    try:
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )

        liger = LigerFusedLinearCrossEntropyLoss(reduction="mean")
        candidates["liger_flce"] = lambda h, w, t, c: liger(w, h, t)
    except Exception as error:
        print(f"liger unavailable: {type(error).__name__}: {error}", flush=True)

    results = []
    for rows in rows_to_test:
        hidden, weight, targets = make_inputs(rows, args.width, args.vocab, "cuda", 11)
        for direct in (False, True):
            reference = None
            for name, fn in candidates.items():
                if name == "liger_flce" and direct:
                    continue  # liger returns a real grad_weight; direct accumulation is not its contract
                try:
                    value = measure(fn, hidden, weight, targets, args.chunk, direct, args.iterations)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    results.append(dict(rows=rows, direct=direct, name=name, oom=True))
                    print(f"  rows={rows} direct={direct} {name:18s} OOM", flush=True)
                    continue
                entry = dict(rows=rows, direct=direct, name=name, ms=value["ms"],
                             peak_gib=value["peak_gib"], loss=value["loss"])
                if reference is None:
                    reference = value
                else:
                    entry["vs_baseline"] = dict(
                        loss=abs(value["loss"] - reference["loss"]) / max(1e-8, abs(reference["loss"])),
                        dh=relative(value["dh"], reference["dh"]),
                        dw=relative(value["dw"], reference["dw"]),
                    )
                    entry["speedup"] = reference["ms"] / value["ms"]
                results.append(entry)
                suffix = ""
                if "vs_baseline" in entry:
                    worst = max(entry["vs_baseline"].values())
                    suffix = f" speedup={entry['speedup']:.3f}x worst_err={worst:.2e}"
                print(f"  rows={rows} direct={direct} {name:18s} {entry['ms']:8.2f} ms "
                      f"peak={entry['peak_gib']:5.2f} GiB{suffix}", flush=True)
                del value
                torch.cuda.empty_cache()
        del hidden, weight, targets
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(0), width=args.width, vocab=args.vocab,
        chunk=args.chunk, observed_text_rows=profile, results=results), indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
