"""Adversarial GPU check: unconditional CQ queries must read nothing under Flex.

`visual_uncond_probability` only fires sometimes, so a passing end-to-end gate
does not prove the unconditional rows were ever exercised. These rows carry the
classifier-free-guidance null condition: if FlexAttention let them see any
content, CFG training would be silently wrong while every loss still looked
finite. Here the flags are forced on, and the check is behavioural -- perturb
all content and confirm the unconditional outputs do not move at all.
"""

import argparse
import json
from pathlib import Path

import torch

from configs.config import RunConfig
from data.batch import fake_batch
import ops.attention as attention_module
import ops.flex_cq as flex_cq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", default=".assets/Qwen3.5-2B-Base")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    cfg = RunConfig.load("configs/qwen35_2b_wan.yaml")
    cfg.text_assets = cfg.vision_assets = a.assets
    dev = "cuda"
    cases = []

    for length, images in ((2048, 2), (3072, 3)):
        batch = fake_batch(cfg, length, 1, images, seed=11 + images, device="cpu")
        plan = batch.mainline
        predicate = plan.cq_predicate
        nq = len(plan.query_positions)
        # Force every visual query unconditional, then rebuild the dense mask
        # and the predicate from that same rule so both sides agree.
        from ops.attention_layout import document_cq_mask, cq_mask_vectors

        unconditional = plan.query_image[0].clone()
        if not unconditional.any():
            raise SystemExit("Fixture produced no visual queries to make unconditional")
        mask, valid = document_cq_mask(batch, plan.query_positions, unconditional)
        predicate = cq_mask_vectors(batch, plan.query_positions, valid, unconditional)
        plan.cq_mask, plan.cq_valid, plan.cq_predicate, plan.cq_plan = mask, valid, predicate, None
        plan = plan.to(dev)
        predicate = plan.cq_predicate

        rows, keys = len(predicate.row_sigma), len(predicate.key_sigma)
        heads, kv_heads, dim = cfg.heads, cfg.kv_heads, cfg.head_dim
        torch.manual_seed(5)
        q = torch.randn(rows, 1, heads, dim, device=dev, dtype=torch.bfloat16)
        k = torch.randn(keys, 1, kv_heads, dim, device=dev, dtype=torch.bfloat16)
        v = torch.randn(keys, 1, kv_heads, dim, device=dev, dtype=torch.bfloat16)

        def run(kv_pair, use_flex):
            previous = flex_cq._ENABLED
            flex_cq._ENABLED = use_flex
            predicate.block_mask = None
            try:
                if use_flex and not flex_cq.usable(q, predicate, plan.cq_mask):
                    raise SystemExit("No recorded FlexAttention tile for this GPU/head size")
                with torch.no_grad():
                    return attention_module.attend(
                        q, kv_pair[0], kv_pair[1], plan.cq_mask, plan.cq_valid,
                        plan.cq_plan, True, predicate if use_flex else None)
            finally:
                flex_cq._ENABLED = previous

        flex_out = run((k, v), True)
        dense_out = run((k, v), False)
        # Perturb every content key/value; unconditional rows must not react.
        k2, v2 = k + 3 * torch.randn_like(k), v + 3 * torch.randn_like(v)
        flex_moved = run((k2, v2), True)

        s = predicate.content_length
        uncond_rows = (s + unconditional.nonzero().flatten().to(dev))
        drift = float((flex_moved[uncond_rows] - flex_out[uncond_rows]).abs().max())
        agree = float((flex_out[uncond_rows].float() - dense_out[uncond_rows].float()).abs().max())
        # Conditional rows must react, otherwise the probe proves nothing.
        other = torch.ones(rows, dtype=torch.bool, device=dev)
        other[uncond_rows] = False
        sensitivity = float((flex_moved[other] - flex_out[other]).abs().max())

        passed = drift == 0.0 and sensitivity > 1e-3
        cases.append(dict(length=length, images=images,
                          unconditional_rows=int(unconditional.sum()),
                          unconditional_drift=drift,
                          flex_vs_dense_on_unconditional=agree,
                          conditional_sensitivity=sensitivity, passed=passed))
        print(f"  len={length} img={images} uncond_rows={int(unconditional.sum())} "
              f"drift={drift:.3e} (must be 0) vs_dense={agree:.3e} "
              f"cond_sensitivity={sensitivity:.3e} -> {'PASS' if passed else 'FAIL'}", flush=True)

    result = dict(passed=all(c["passed"] for c in cases), cases=cases,
                  device=torch.cuda.get_device_name(0))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("UNCONDITIONAL-QUERY GATE FAILED")
    print("UNCONDITIONAL-QUERY GATE PASSED")


if __name__ == "__main__":
    main()
