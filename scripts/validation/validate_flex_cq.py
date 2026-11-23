"""Gate: FlexAttention CQ path must match the dense-mask path it replaces.

This drives the real dispatch in ``ops.attention.attend`` -- the same call the
backbone makes -- so it validates the integration, not just a kernel in
isolation. Layouts come from the production plan builder via ``fake_batch``,
including native-resolution image blocks with tied sigma and, for the Wan
config, unconditional visual queries. Forward and all of dq/dk/dv are compared.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from configs.config import RunConfig
from data.batch import fake_batch
import ops.attention as attention_module
import ops.flex_cq as flex_cq


def relative(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", default=".assets/Qwen3.5-2B-Base")
    p.add_argument("--tol", type=float, default=2e-2)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    cfg = RunConfig.load("configs/qwen35_2b_wan.yaml")
    cfg.text_assets = cfg.vision_assets = a.assets
    dev = "cuda"
    cases = []
    for length, images in ((1536, 1), (2048, 2), (3072, 3)):
        batch = fake_batch(cfg, length, 1, images, seed=7 + images, device="cpu")
        plan = batch.mainline.to(dev)
        predicate = plan.cq_predicate
        if predicate is None:
            raise SystemExit("Plan carries no CQ predicate; batch_layout wiring is missing")
        rows, keys = len(predicate.row_sigma), len(predicate.key_sigma)
        heads, kv_heads, dim = cfg.heads, cfg.kv_heads, cfg.head_dim

        torch.manual_seed(3)
        base = [
            torch.randn(rows, 1, heads, dim, device=dev, dtype=torch.bfloat16),
            torch.randn(keys, 1, kv_heads, dim, device=dev, dtype=torch.bfloat16),
            torch.randn(keys, 1, kv_heads, dim, device=dev, dtype=torch.bfloat16),
        ]
        upstream = torch.randn(rows, 1, heads * dim, device=dev, dtype=torch.bfloat16)

        def run(use_flex):
            q, k, v = (t.detach().clone().requires_grad_() for t in base)
            previous = flex_cq._ENABLED
            flex_cq._ENABLED = use_flex
            # The block mask is cached on the predicate; clear it so each side
            # builds its own and neither can reuse the other's state.
            predicate.block_mask = None
            try:
                took_flex = flex_cq.usable(q, predicate, plan.cq_mask)
                if use_flex and not took_flex:
                    raise SystemExit(
                        "FlexAttention was requested but no tile config is recorded "
                        "for this GPU/head size; see configs/flex_tiles.json"
                    )
                out = attention_module.attend(
                    q,
                    k,
                    v,
                    plan.cq_mask,
                    plan.cq_valid,
                    plan.cq_plan,
                    True,
                    predicate if use_flex else None,
                )
            finally:
                flex_cq._ENABLED = previous
            (out * upstream).sum().backward()
            return out, [t.grad for t in (q, k, v)], took_flex

        dense_out, dense_grads, _ = run(False)
        flex_out, flex_grads, took = run(True)
        errors = dict(output=relative(flex_out, dense_out))
        for name, x, y in zip(("dq", "dk", "dv"), flex_grads, dense_grads):
            errors[name] = relative(x, y)
        passed = took and all(v <= a.tol for v in errors.values())
        cases.append(
            dict(length=length, images=images, rows=rows, keys=keys, used_flex=took, errors=errors, passed=passed)
        )
        print(
            f"  len={length} images={images} rows={rows} flex={took} "
            + " ".join(f"{k}={v:.2e}" for k, v in errors.items())
            + f" -> {'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    major, minor = torch.cuda.get_device_capability()
    result = dict(
        passed=all(c["passed"] for c in cases),
        device=torch.cuda.get_device_name(0),
        capability=f"sm{major}{minor}",
        tolerance=a.tol,
        tiles=flex_cq.tile_options(cfg.head_dim, f"{major}{minor}"),
        worst_error=max(v for c in cases for v in c["errors"].values()),
        cases=cases,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("passed", "worst_error", "tiles")}, indent=2))
    if not result["passed"]:
        raise SystemExit("FLEX CQ GATE FAILED")
    print("FLEX CQ GATE PASSED")


if __name__ == "__main__":
    main()
