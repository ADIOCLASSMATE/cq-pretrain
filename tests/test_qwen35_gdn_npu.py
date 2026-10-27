"""Official MindSpeed scan checks with Qwen3.5-2B's 16 heads / 128 dimensions.

Run in the pinned CANN environment with UMM_VIDEO_DEVICE=npu:<rank>. Tests stay
opt-in on CPU hosts and never accept a native fallback as a successful NPU run.
Cross-backend comparisons allow BF16 kernel rounding; visibility checks are exact.
"""
import os

import pytest
import torch

from models import backbone_qwen35 as cq


_DEVICE = os.environ.get("UMM_VIDEO_DEVICE", "cpu")
pytestmark = pytest.mark.skipif(not _DEVICE.startswith("npu"), reason="requires explicit UMM_VIDEO_DEVICE=npu:<rank>")


@pytest.fixture(scope="module")
def device():
    import torch_npu  # noqa: F401
    torch.npu.set_device(_DEVICE)
    assert cq.resolve_gdn_backend_identity("mindspeed")["module"] == "mindspeed.core.ssm.npu_chunk_gated_delta_rule"
    return torch.device(_DEVICE)


def inputs(length, *, batch=2):
    generator = torch.Generator().manual_seed(1300 + length)
    shape = (batch, length, 16, 128)
    q, k, v = [torch.randn(shape, generator=generator).bfloat16() for _ in range(3)]
    g = -torch.rand(shape[:-1], generator=generator) * 0.2
    beta = torch.rand(shape[:-1], generator=generator).bfloat16()
    return q, k, v, g, beta


def run_scan(values, probe, backend, device):
    values = [value.to(device).detach().requires_grad_() for value in values]
    output, state = cq._cq_select_chunk_gdn(backend)(
        *values[:3], g=values[3], beta=values[4], initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    assert state is None
    gradients = torch.autograd.grad(output, values, probe.to(device))
    return output.detach().float().cpu(), [gradient.detach().float().cpu() for gradient in gradients]


def assert_kernel_close(actual, expected, kind="output"):
    # MindSpeed-MM af74d8df tests/ut_fsdp/models/qwen3_5/
    # test_chunk_gated_delta_rule.py uses atol=rtol=0.01 for BF16 parity.
    # Also require FLA cd61447d tests/ops/test_gdn.py's relative RMS limits:
    # output .005, Q/K/V .008, g/beta .02. Cancellation near zero makes a
    # pointwise relative error alone misleading; both checks are enforced.
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
    limit = {"output": 0.005, "q": 0.008, "k": 0.008, "v": 0.008, "g": 0.02, "beta": 0.02}[kind]
    assert relative_error < limit, (kind, relative_error.item(), limit)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 129])
def test_npu_mindspeed_matches_native_output_and_all_input_gradients(device, length):
    values = inputs(length)
    generator = torch.Generator().manual_seed(2000 + length)
    probe = torch.randn(values[0].shape, generator=generator).bfloat16()
    expected_output, expected_gradients = run_scan(values, probe, "native", "cpu")
    actual_output, actual_gradients = run_scan(values, probe, "mindspeed", device)
    assert_kernel_close(actual_output, expected_output)
    for kind, actual, expected in zip(("q", "k", "v", "g", "beta"), actual_gradients, expected_gradients):
        assert_kernel_close(actual, expected, kind)


def test_npu_readonly_queries_keep_prefix_gradients_without_state_writes(device, monkeypatch):
    import importlib

    upstream = importlib.import_module("mindspeed.core.ssm.npu_chunk_gated_delta_rule")
    original_cumsum = upstream.chunk_local_cumsum
    reverse_scans = []

    def record_reverse_scan(g, *args, **kwargs):
        result = original_cumsum(g, *args, **kwargs)
        if kwargs.get("reverse", False):
            reverse_scans.append((g.detach().cpu(), result.detach().cpu()))
        return result

    monkeypatch.setattr(upstream, "chunk_local_cumsum", record_reverse_scan)
    # Event order: 63 prefix writes, 5 read-only target queries, 61 target writes.
    # Reads straddle a 64-event kernel chunk boundary; all share the same state.
    length, prefix, reads = 129, 63, 5
    values = list(inputs(length, batch=2))
    query_slice = slice(prefix, prefix + reads)
    for value in values[1:]:
        value[:, query_slice] = 0
    generator = torch.Generator().manual_seed(5001)
    probe = torch.zeros_like(values[0])
    probe[:, query_slice] = torch.randn(probe[:, query_slice].shape, generator=generator).bfloat16()
    expected_output, expected_gradients = run_scan(values, probe, "native", "cpu")
    actual_output, actual_gradients = run_scan(values, probe, "mindspeed", device)
    assert_kernel_close(actual_output, expected_output)
    for kind, actual, expected in zip(("q", "k", "v", "g", "beta"), actual_gradients, expected_gradients):
        assert_kernel_close(actual, expected, kind)
    # Loss on reads reaches prefix K/V/g/beta and cannot see target content writes.
    for kind, gradient in zip(("k", "v", "g", "beta"), actual_gradients[1:]):
        assert gradient[:, :prefix].abs().max() > 0
        if kind != "g":
            assert torch.count_nonzero(gradient[:, prefix + reads:]) == 0
    assert torch.count_nonzero(actual_gradients[0][:, :prefix]) == 0
    assert torch.count_nonzero(actual_gradients[0][:, prefix + reads:]) == 0

    # MindSpeed cumsum.py:65-68 computes reverse sum as -cumsum + sum + g.
    # The two FP32 reductions can leave cancellation residuals even when every
    # future input is exactly zero. Inspect the actual unmodified operation:
    # future inputs and a direct FP64 reverse sum must both remain exactly zero.
    # Bound only dg's residual by gamma_(2K+2) * sum(abs(chunk)), accounting for
    # two K-term reductions and their additions; do not relax any other gradient.
    assert len(reverse_scans) == 1
    raw_g, returned_g = reverse_scans[0]
    future = slice(prefix + reads, None)
    assert torch.count_nonzero(raw_g[:, future]) == 0
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    chunk_size = 64
    operations = 2 * chunk_size + 2
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    exact_reverse = torch.zeros_like(raw_g, dtype=torch.float64)
    roundoff_bound = torch.zeros_like(exact_reverse)
    for start in range(0, length, chunk_size):
        block = raw_g[:, start:start + chunk_size].double()
        exact_reverse[:, start:start + chunk_size] = block.flip(1).cumsum(1).flip(1)
        roundoff_bound[:, start:start + chunk_size] = gamma * block.abs().sum(1, keepdim=True)
    assert torch.count_nonzero(exact_reverse[:, future]) == 0
    torch.testing.assert_close(actual_gradients[3], returned_g, rtol=0, atol=0)
    assert bool((returned_g[:, future].double().abs() <= roundoff_bound[:, future]).all())

    def forward(new_values):
        with torch.no_grad():
            output, _ = cq._cq_mindspeed_chunk_gated_delta_rule(
                *[value.to(device) for value in new_values[:3]],
                g=new_values[3].to(device), beta=new_values[4].to(device),
            )
        return output.float().cpu()

    changed_target = [value.clone() for value in values]
    changed_target[1][:, prefix + reads:] *= 4
    changed_target[2][:, prefix + reads:] *= -3
    changed_target[3][:, prefix + reads:] *= 5
    changed_target[4][:, prefix + reads:] = 1 - changed_target[4][:, prefix + reads:]
    target_output = forward(changed_target)
    torch.testing.assert_close(target_output[:, query_slice], actual_output[:, query_slice], rtol=0, atol=0)

    changed_query = [value.clone() for value in values]
    changed_query[0][:, prefix + 1] *= -3
    query_output = forward(changed_query)
    other_positions = torch.arange(length) != prefix + 1
    torch.testing.assert_close(query_output[:, other_positions], actual_output[:, other_positions], rtol=0, atol=0)
    assert not torch.equal(query_output[:, prefix + 1], actual_output[:, prefix + 1])
