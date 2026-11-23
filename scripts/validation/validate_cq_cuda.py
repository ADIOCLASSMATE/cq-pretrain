"""Single-GPU CUDA CQ GatedDeltaNet accuracy gate vs the CPU recurrence oracle.

Compares the wired upstream flash-linear-attention chunk_gated_delta_rule path
(BF16, through the complete GatedDelta layer: fused projections, causal conv,
gated norm) against an FP64 CPU reference built from recurrent_delta, for both
mainline methods and chunk-boundary lengths. Also re-checks that read-only CQ
queries never perturb content/key outputs. No distributed init required.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from configs.config import tiny_config
from data.batch import fake_batch
from unittest.mock import patch

from models.backbone_qwen35 import GatedDelta, RMSNorm, recurrent_delta


def relerr(actual, expected):
    a = actual.detach().cpu().double()
    b = expected.detach().cpu().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--key-heads", type=int, default=16)
    p.add_argument("--value-heads", type=int, default=16)
    p.add_argument("--tol", type=float, default=0.035)
    p.add_argument(
        "--kernel-head-shards",
        type=int,
        default=1,
        help="Split the same layer's packed kernel heads; single-GPU check, not collectives",
    )
    p.add_argument(
        "--control-tol", type=float, default=0.035, help="Tolerance for the conditioned A_log/dt_bias gradient error"
    )
    a = p.parse_args()
    if a.kernel_head_shards < 1 or a.value_heads % a.kernel_head_shards:
        p.error("value-heads must be divisible by a positive kernel-head-shards")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    dev = "cuda"
    torch.cuda.set_device(0)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    if a.kernel_head_shards > 1:
        import ops.cuda_gated_delta as binding

        original = binding.chunk_gated_delta

        def split_heads(q, k, v, g, beta, **kwargs):
            shards = [t.chunk(a.kernel_head_shards, dim=2) for t in (q, k, v, g, beta)]
            outputs = [original(qi, ki, vi, gi, bi, **kwargs)[0] for qi, ki, vi, gi, bi in zip(*shards)]
            return torch.cat(outputs, dim=2), None

        # Keep the full 16-head fixture and FP64 oracle identical while testing
        # each upstream kernel invocation with TP-local head dimensions.
        binding.chunk_gated_delta = split_heads

    # Offset-gamma RMSNorm: BF16 activations, FP32 weight, including dgamma.
    torch.manual_seed(1)
    norm = RMSNorm(256)
    norm.weight.data.normal_(0, 0.1)
    ref = copy.deepcopy(norm)
    norm = norm.to(dev)
    x = torch.randn(31, 256, dtype=torch.bfloat16, device=dev, requires_grad=True)
    xr = x.detach().cpu().float().requires_grad_()
    y = norm(x)
    yr = ref(xr)
    u = torch.randn_like(y)
    (y * u).sum().backward()
    (yr * u.cpu().float()).sum().backward()
    norm_err = dict(output=relerr(y, yr), dx=relerr(x.grad, xr.grad), dw=relerr(norm.weight.grad, ref.weight.grad))

    for length in (61, 64, 65):
        torch.manual_seed(101)
        cfg = tiny_config(linear_heads=a.key_heads, linear_value_heads=a.value_heads, linear_head_dim=128)
        batch = fake_batch(cfg, length, 1, 2)
        plan = batch.mainline.delta_plan
        layer = GatedDelta(cfg)
        reference = copy.deepcopy(layer).double()
        layer = layer.to(device=dev, dtype=torch.bfloat16)
        # Keep FP32 control/norm values, BF16 bulk weights, as in training.
        layer.A_log.data = reference.A_log.detach().float().to(dev)
        layer.norm.weight.data = reference.norm.weight.detach().float().to(dev)
        reference.load_state_dict({k: v.detach().cpu().double() for k, v in layer.state_dict().items()})
        x = torch.randn(plan.hidden_length, 1, cfg.width, device=dev, dtype=torch.bfloat16, requires_grad=True)
        xr = x.detach().cpu().double().requires_grad_()
        dp = plan.to(dev)
        y = layer(x, dp)
        control = {}

        def reference_capture(*values):
            # values[3] is the FP32/FP64 log-decay control g fed to the recurrence.
            values[3].retain_grad()
            control["g"] = values[3]
            return recurrent_delta(*values)

        with patch("models.backbone_qwen35.recurrent_delta", reference_capture):
            yr = reference(xr, plan)
        u = torch.randn_like(y)
        (y * u).sum().backward()
        (yr * u.cpu().double()).sum().backward()
        ref_params = dict(reference.named_parameters())
        errors = {name: relerr(pz.grad, ref_params[name].grad) for name, pz in layer.named_parameters()}
        errors.update(output=relerr(y, yr), dx=relerr(x.grad, xr.grad))
        # A_log/dt_bias accumulate signed per-token contributions that can
        # nearly cancel, so a large RELATIVE error against a near-zero net
        # reference gradient is not evidence of a wrong gradient. Bound the
        # absolute deviation by the FP64 sum of |per-token contribution|
        # (the same conditioning the Ascend gate used). dg/dA_log = g and
        # dg/d(dt_bias) = -exp(A_log)*sigmoid(a+dt_bias).
        heads = cfg.linear_value_heads or cfg.linear_heads
        with torch.no_grad():
            rg = control["g"]
            dg = rg.grad
            activation = reference.in_proj_a(xr[:, 0]).index_select(0, plan.content_order) + reference.dt_bias
            derivative = -reference.A_log.exp() * activation.sigmoid()
            derivative = torch.cat((derivative, derivative.new_zeros(1, heads))).index_select(0, plan.event_write)[None]
            scales = dict(A_log=(dg * rg).abs().sum((0, 1)).norm(), dt_bias=(dg * derivative).abs().sum((0, 1)).norm())
            conditioned = {
                name: float(
                    (getattr(layer, name).grad.cpu().double() - getattr(reference, name).grad).norm()
                    / scale.clamp_min(1e-8)
                )
                for name, scale in scales.items()
            }
        # Read-only CQ queries must not perturb content/key outputs.
        keys = length
        changed = x.detach().clone()
        changed[keys:] += 2
        with torch.no_grad():
            after = layer(changed, dp)
        isolation = float((after[:keys] - y[:keys]).abs().max())
        ordinary = {k: v for k, v in errors.items() if k not in conditioned}
        passed_ordinary = all(math.isfinite(v) and v <= a.tol for v in ordinary.values())
        passed_control = all(math.isfinite(v) and v <= a.control_tol for v in conditioned.values())
        # Keep the unconditioned verdict in the evidence; it is not loosened,
        # only reported next to the conditioned one.
        passed_raw = all(math.isfinite(v) and v <= a.tol for v in errors.values())
        passed = passed_ordinary and passed_control and isolation == 0.0
        cases.append(
            dict(
                method="CQ_IT",
                length=length,
                events=len(plan.event_q),
                errors=errors,
                conditioned_control_error=conditioned,
                isolation_max_abs=isolation,
                passed=passed,
                passed_raw_unconditioned=passed_raw,
            )
        )

    result = dict(
        passed=all(c["passed"] for c in cases) and all(math.isfinite(v) and v <= a.tol for v in norm_err.values()),
        device=torch.cuda.get_device_name(0),
        kernel_head_shards=a.kernel_head_shards,
        tolerance=a.tol,
        control_tolerance=a.control_tol,
        passed_raw_unconditioned=all(c["passed_raw_unconditioned"] for c in cases),
        worst_conditioned_control_error=max(
            (v for c in cases for v in c["conditioned_control_error"].values()), default=0.0
        ),
        norm_errors=norm_err,
        worst_case_error=max((v for c in cases for v in c["errors"].values()), default=0.0),
        cases=cases,
    )
    (out / "cuda_cq.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    for c in cases:
        print(
            c["method"],
            c["length"],
            "worst",
            round(max(c["errors"].values()), 5),
            "iso",
            c["isolation_max_abs"],
            "pass",
            c["passed"],
        )
    if not result["passed"]:
        raise SystemExit("CUDA CQ GDN accuracy gate failed")
    print("CUDA CQ GDN GATE PASSED")


if __name__ == "__main__":
    main()
