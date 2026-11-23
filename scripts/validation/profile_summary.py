"""Attribute GPU kernel time in a torch profiler trace to the op that launched it.

Kernel-name categories answer "what kind of work", which is what the first two
L40S profiles were read for. Deciding what to optimize next needs "which part of
the model", so this also walks each kernel back through its correlation id to
the enclosing aten op and its input shapes.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
from pathlib import Path

CATEGORIES = (
    (
        "GDN (FLA Triton)",
        re.compile(r"chunk_gated|fused_recurrent|solve_tril|cumsum|l2norm|wy_repr|dqkwg|dhu|fwd_o|fwd_h"),
    ),
    ("attention", re.compile(r"fmha|flash|attention|_attn|sdpa|flex")),
    ("NCCL collective", re.compile(r"^nccl|ncclDevKernel")),
    ("GEMM", re.compile(r"gemm|cutlass|ampere_|sm\d+_xmma|tensorop|matmul|^ampere")),
    ("norm/reduce", re.compile(r"norm|reduce_kernel|softmax|layer_norm|rms")),
    ("copy/index", re.compile(r"copy|index|gather|scatter|cat_|permute|transpose")),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|pointwise|fill|silu|gelu|add|mul")),
)


def categorize(name):
    # FlexAttention compiles to Inductor template kernels ("tem"); they are
    # this model's CQ attention, not miscellaneous work. Confirmed by A/B:
    # with PRETRAIN_FLEX_ATTENTION=0 these vanish and fmha_cutlass* appear.
    if "triton_tem" in name:
        return "attention"
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def load(root):
    files = [
        f
        for f in glob.glob(os.path.join(root, "**", "*.json*"), recursive=True)
        if "trace" in os.path.basename(f).lower() or f.endswith((".json", ".json.gz"))
    ]
    if not files:
        raise SystemExit(f"no trace found under {root}")
    path = max(files, key=os.path.getsize)
    print(f"trace: {path} {os.path.getsize(path) / 2**20:.1f} MB", flush=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        doc = json.load(handle)
    return doc["traceEvents"] if isinstance(doc, dict) else doc


def shape_tag(args):
    dims = args.get("Input Dims") or args.get("Input dims")
    if not dims:
        return ""
    parts = [",".join(str(x) for x in d) for d in dims if isinstance(d, list) and d]
    return "[" + "][".join(parts[:3]) + "]" if parts else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--output")
    args = parser.parse_args()
    events = load(args.root)

    kernels = []  # (name, dur, correlation)
    launches = {}  # correlation -> (tid, ts)
    cpu_ops = collections.defaultdict(list)  # tid -> [(ts, end, name, shapes)]
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = (event.get("cat") or "").lower()
        detail = event.get("args") or {}
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append((event.get("name", "?"), float(event.get("dur", 0)), detail.get("correlation"), cat))
        elif cat in ("cuda_runtime", "runtime") and detail.get("correlation") is not None:
            launches[detail["correlation"]] = (event.get("tid"), float(event.get("ts", 0)))
        elif cat in ("cpu_op", "user_annotation"):
            ts = float(event.get("ts", 0))
            cpu_ops[event.get("tid")].append(
                (ts, ts + float(event.get("dur", 0)), event.get("name", "?"), shape_tag(detail))
            )

    # One ordered sweep per thread instead of a scan per launch: profiler ops
    # nest, so the innermost enclosing op is whatever is on top of the stack
    # when the launch is reached. A 1.4 GB trace has millions of each.
    attribution = {}
    launches_by_thread = collections.defaultdict(list)
    for correlation, (tid, ts) in launches.items():
        launches_by_thread[tid].append((ts, correlation))
    for tid, items in launches_by_thread.items():
        ops = sorted(cpu_ops.get(tid, ()))
        items.sort()
        stack = []  # (end, name, shapes), innermost last
        index = 0
        for ts, correlation in items:
            while index < len(ops) and ops[index][0] <= ts:
                start, end, name, shapes = ops[index]
                stack.append((end, name, shapes))
                index += 1
            while stack and stack[-1][0] < ts:
                stack.pop()
            attribution[correlation] = (stack[-1][1], stack[-1][2]) if stack else ("<unattributed>", "")

    by_category = collections.Counter()
    by_kernel = collections.Counter()
    by_op = collections.Counter()
    total = 0.0
    for name, dur, correlation, cat in kernels:
        total += dur
        label = "__" + cat if cat != "kernel" else categorize(name)
        by_category[label] += dur
        by_kernel[name] += dur
        if cat == "kernel" and correlation in attribution:
            op, shapes = attribution[correlation]
            by_op[(op, shapes, categorize(name))] += dur

    print(f"\n=== GPU kernel time by category (total {total:.1f} ms) ===")
    for label, value in by_category.most_common():
        print(f"  {label:22s} {value:10.1f} ms  {100 * value / max(total, 1e-9):5.1f}%")

    print(f"\n=== top {args.top} kernels ===")
    for index, (name, value) in enumerate(by_kernel.most_common(args.top), 1):
        print(f"  {index:3d} {value:9.1f} ms {100 * value / max(total, 1e-9):5.1f}%  {name[:96]}")

    if by_op:
        attributed = sum(by_op.values())
        print(
            f"\n=== top {args.top} launching ops ({100 * attributed / max(total, 1e-9):.0f}% of kernel time attributed) ==="
        )
        for index, ((op, shapes, cat), value) in enumerate(by_op.most_common(args.top), 1):
            print(f"  {index:3d} {value:9.1f} ms {100 * value / max(total, 1e-9):5.1f}%  {cat:16s} {op} {shapes[:60]}")
    else:
        print("\n(no correlation ids in this trace; kernel categories only)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(
                dict(
                    total_ms=total,
                    by_category={k: v for k, v in by_category.most_common()},
                    by_kernel={k: v for k, v in by_kernel.most_common(60)},
                    by_op=[dict(op=o, shapes=s, category=c, ms=v) for (o, s, c), v in by_op.most_common(60)],
                ),
                indent=2,
            )
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
