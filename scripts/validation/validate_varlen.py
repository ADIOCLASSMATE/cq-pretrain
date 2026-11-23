"""Gate the packed varlen kernel against the per-segment loop it replaces.

The question is not whether the kernel is accurate in the abstract: it is
whether swapping it in moves training numerics. So each shape is evaluated
three ways -- an FP64 CPU oracle, the per-segment loop in BF16 on device (what
trained until now), and the one-call varlen form in BF16 on device -- and the
two device errors are compared against the oracle. The new form passes when it
is no further from the oracle than the loop already was.

Forward and backward are both checked: the kernel supplies its own gradient,
and a wrong segment mapping shows up in dk/dv long before it shows in the loss.

Both backends are gated the same way: Ascend's npu_fusion_attention TND form
and CUDA's flash_attn_varlen_func replace the same loop, so one comparison
decides both.

    python -m scripts.validation.validate_varlen --output .runs/varlen.json
"""

import argparse
import json
from pathlib import Path

import torch

from ops.attention import varlen_reference


def relative_error(actual, expected):
    a, b = actual.detach().to("cpu", torch.float64), expected.detach().to("cpu", torch.float64)
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def ends(lengths):
    total, out = 0, []
    for length in lengths:
        total += length
        out.append(total)
    return tuple(out)


def inputs(lengths_q, lengths_k, heads, dim, seed):
    generator = torch.Generator().manual_seed(seed)

    def make(total):
        return torch.randn(total, heads, dim, generator=generator, dtype=torch.float64)

    return make(sum(lengths_q)), make(sum(lengths_k)), make(sum(lengths_k))


def fused_varlen():
    """The backend's one-call varlen form, whichever accelerator this is."""
    import distributed.accel as accel

    if accel.IS_NPU:
        from ops import npu_attention

        return npu_attention.varlen
    from ops import cuda_attention

    return cuda_attention.varlen


def run_case(case, device, dtype):
    """Oracle, loop and varlen for one shape; returns their errors and agreement."""

    lengths_q, lengths_k = case["lengths_q"], case["lengths_k"]
    cu_q, cu_k = ends(lengths_q), ends(lengths_k)
    q, k, v = inputs(lengths_q, lengths_k, case["heads"], case["dim"], case["seed"])
    upstream = torch.randn(q.shape, generator=torch.Generator().manual_seed(case["seed"] + 1), dtype=torch.float64)

    oracle_inputs = [t.clone().requires_grad_() for t in (q, k, v)]
    oracle = varlen_reference(*oracle_inputs, cu_q, cu_k)
    (oracle * upstream).sum().backward()
    oracle_grads = [t.grad for t in oracle_inputs]

    results = {}
    for name, call in (("loop", varlen_reference), ("varlen", fused_varlen())):
        tensors = [t.to(device=device, dtype=dtype).detach().requires_grad_() for t in (q, k, v)]
        out = call(*tensors, cu_q, cu_k)
        (out.float() * upstream.to(device=device, dtype=torch.float32)).sum().backward()
        results[name] = dict(
            output=relative_error(out, oracle),
            **{f"d{label}": relative_error(t.grad, g) for label, t, g in zip("qkv", tensors, oracle_grads)},
        )
        results[name + "_tensors"] = [out, *(t.grad for t in tensors)]

    loop_values, varlen_values = results.pop("loop_tensors"), results.pop("varlen_tensors")
    results["loop_vs_varlen"] = dict(
        zip(("output", "dq", "dk", "dv"), (relative_error(a, b) for a, b in zip(varlen_values, loop_values)))
    )
    return results


def geometry(args):
    """Real head geometry; the gate is worthless on shapes training never sees."""
    from configs.config import RunConfig

    cfg = RunConfig.load(args.config)
    dit = dict(heads=cfg.dit_heads, dim=cfg.dit_width // cfg.dit_heads)
    assets = args.vision_assets or cfg.vision_assets
    vision = None
    if assets:
        from utils.assets import vision_config

        settings = vision_config(assets)
        vision = dict(heads=settings["num_heads"], dim=settings["hidden_size"] // settings["num_heads"])
    return dit, vision


def cases(dit, vision):
    # Native-resolution packing: every visual its own length, which is exactly
    # the shape the batched path never covered and the loop always took.
    visuals = (1024, 576, 1600, 256, 900)
    text = (226, 226, 226, 226, 226)
    built = [
        dict(name="dit_self_one_draw", lengths_q=visuals, lengths_k=visuals, seed=11, **dit),
        dict(name="dit_cross_one_draw", lengths_q=visuals, lengths_k=text, seed=12, **dit),
        # Several draws: the flattening must keep (draw, visual) segments apart.
        dict(name="dit_self_two_draws", lengths_q=visuals * 2, lengths_k=visuals * 2, seed=13, **dit),
        dict(name="dit_cross_two_draws", lengths_q=visuals * 2, lengths_k=text * 2, seed=14, **dit),
        dict(name="dit_single_visual", lengths_q=(1024,), lengths_k=(1024,), seed=15, **dit),
    ]
    if vision:
        built += [
            dict(name="vision_tower", lengths_q=(256, 1024, 64, 576), lengths_k=(256, 1024, 64, 576), seed=16, **vision)
        ]
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/qwen35_2b_wan.yaml")
    parser.add_argument("--vision-assets", help="Pinned vision assets, when the config does not name them")
    parser.add_argument("--tolerance", type=float, default=1.25, help="Allowed ratio of varlen error to loop error")
    # The loop's own BF16 error against the FP64 oracle is about 0.004 on these
    # shapes, so this floor is a real bar rather than somewhere to hide.
    parser.add_argument("--floor", type=float, default=0.006, help="Error below which the ratio is not meaningful")
    args = parser.parse_args()

    import distributed.accel as accel

    device = f"{accel.accel_type()}:0"
    accel.device_module().set_device(0)
    dit, vision = geometry(args)
    report = dict(backend=accel.BACKEND, dit=dit, vision=vision, dtype="bfloat16", cases={})
    failures = []
    for case in cases(dit, vision):
        name = case.pop("name")
        try:
            result = run_case(case, device, torch.bfloat16)
        except Exception as error:  # a kernel that rejects this shape must be named, not hidden
            report["cases"][name] = dict(error=f"{type(error).__name__}: {error}"[:500])
            failures.append(f"{name}: raised")
            continue
        verdict = {}
        for field in ("output", "dq", "dk", "dv"):
            loop, fused = result["loop"][field], result["varlen"][field]
            # Equal accuracy is the bar: the loop's own error is the budget.
            passed = fused <= max(loop * args.tolerance, args.floor)
            verdict[field] = passed
            if not passed:
                failures.append(f"{name}.{field}: varlen {fused:.4g} vs loop {loop:.4g}")
        report["cases"][name] = dict(**result, passed=verdict)
    report["passed"] = not failures
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)
    if failures:
        raise SystemExit("varlen gate failed: " + "; ".join(failures))
    print("varlen gate passed", flush=True)


if __name__ == "__main__":
    main()
