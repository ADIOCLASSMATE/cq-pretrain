"""Compare every saved model/master/Adam/RNG/cursor tensor, not sampled probes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare(left, right):
    left, right = Path(left), Path(right)
    report = dict(
        left=str(left),
        right=str(right),
        exact=True,
        tensors=0,
        elements=0,
        tensor_differences=[],
        other_differences=[],
    )

    def visit(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            report["tensors"] += 1
            report["elements"] += a.numel()
            if a.shape != b.shape or a.dtype != b.dtype:
                report["other_differences"].append(
                    dict(path=path, reason="tensor shape/dtype")
                )
                return
            if torch.equal(a, b):
                return
            maximum, difference2, reference2 = 0.0, 0.0, 0.0
            for x, y in zip(a.reshape(-1).split(1048576), b.reshape(-1).split(1048576)):
                x, y = x.double(), y.double()
                delta = x - y
                maximum = max(maximum, float(delta.abs().max()))
                difference2 += float(delta.square().sum())
                reference2 += float(x.square().sum())
            report["tensor_differences"].append(
                dict(
                    path=path,
                    max_abs=maximum,
                    relative_l2=(difference2 / max(reference2, 1e-60)) ** 0.5,
                )
            )
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                report["other_differences"].append(
                    dict(path=path, reason="numpy state")
                )
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                report["other_differences"].append(dict(path=path, reason="keys"))
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                report["other_differences"].append(dict(path=path, reason="length"))
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}/{index}")
        elif type(a) is not type(b) or a != b:
            report["other_differences"].append(
                dict(path=path, left=str(a)[:200], right=str(b)[:200])
            )

    files = sorted(
        [
            "megatron/model.pt",
            *[
                str(p.relative_to(left))
                for p in left.glob("megatron/optimizer_rank_*.pt")
            ],
            *[p.name for p in left.glob("runtime_rank_*.pt")],
        ]
    )
    for relative in files:
        print(f"Compare {relative}", flush=True)
        a = torch.load(left / relative, map_location="cpu", weights_only=False)
        b = torch.load(right / relative, map_location="cpu", weights_only=False)
        visit(a, b, relative)
        del a, b
    report["exact"] = (
        not report["tensor_differences"] and not report["other_differences"]
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = compare(args.left, args.right)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if not key.endswith("differences")
            }
        ),
        flush=True,
    )
    if args.require_exact and not result["exact"]:
        raise SystemExit("Checkpoint states differ; see the comparison report")
