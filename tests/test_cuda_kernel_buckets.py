"""Real packs change shape on every step; upstream kernels must not be rebuilt for each one.

FLA's l2norm and gated-norm kernels are compiled and autotuned per row count
(per ceil(rows / 2048)). Unbucketed, the first 64-card run on the real corpus
spent ~270 s per step compiling. These tests pin the row counts that reach the
kernels and the values that come back.
"""

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA kernels")

H, D = 16, 128


def _rand(*shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=g)


def test_gated_delta_l2norm_sees_only_bucketed_row_counts():
    from fla.modules import l2norm as upstream
    from ops import cuda_gated_delta as wired

    seen = []
    forward = upstream.l2norm_fwd

    def spy(x, *args, **kwargs):
        seen.append(x.shape[0])
        return forward(x, *args, **kwargs)

    with patch.object(upstream, "l2norm_fwd", spy):
        for events in (300, 311, 1000):
            q, k, v = (_rand(1, events, H, D, seed=s).to(torch.bfloat16).requires_grad_() for s in (1, 2, 3))
            g = (-torch.rand(1, events, H, device="cuda") * 0.3).requires_grad_()
            beta = torch.rand(1, events, H, device="cuda").requires_grad_()
            cu = torch.tensor([0, events // 3, events], device="cuda")
            out = wired.chunk_gated_delta(q, k, v, g=g, beta=beta, cu_seqlens=cu)[0]
            assert out.shape == (1, events, H, D)
            out.float().square().sum().backward()
            assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v, g, beta))
    assert seen and set(seen) == {wired._L2_ROW_BUCKET}


def test_bucketed_l2norm_matches_the_reference_in_value_and_gradient():
    from ops.cuda_gated_delta import _unit

    x = _rand(1, 311, H, D, seed=7).to(torch.bfloat16).requires_grad_()
    weight = _rand(1, 311, H, D, seed=8)
    y = _unit(x)
    (y.float() * weight).sum().backward()
    ref_in = x.detach().double().requires_grad_()
    ref = ref_in * torch.rsqrt(ref_in.square().sum(-1, keepdim=True) + 1e-6)
    (ref * weight.double()).sum().backward()
    assert y.shape == x.shape and y.dtype == x.dtype
    # BF16 storage of unit-norm rows: 8 significant bits.
    assert float((y.double() - ref).norm() / ref.norm()) < 4e-3
    assert float((x.grad.double() - ref_in.grad).norm() / ref_in.grad.norm()) < 1e-2


def test_gated_norm_pads_long_inputs_and_matches_the_direct_call():
    from fla.modules import fused_norm_gate as upstream
    from ops import cuda_fused as wired

    weight = (torch.rand(D, device="cuda") + 0.5).requires_grad_()
    seen = []
    forward = upstream.layer_norm_gated_fwd

    def spy(x, *args, **kwargs):
        seen.append(x.shape[0])
        return forward(x, *args, **kwargs)

    def run(fn, tokens):
        x = _rand(tokens, H, D, seed=11).to(torch.bfloat16).requires_grad_()
        gate = _rand(tokens, H, D, seed=12).to(torch.bfloat16).requires_grad_()
        weight.grad = None
        out = fn(x, gate)
        (out.float() * _rand(tokens, H, D, seed=13)).sum().backward()
        return out.detach().float(), x.grad.float(), gate.grad.float(), weight.grad.clone()

    def direct(x, gate):
        return upstream.rms_norm_gated(
            x.reshape(-1, D), gate.reshape(-1, D), weight, None, activation="swish", eps=1e-6
        ).reshape(x.shape)

    for tokens in (150, 200, 5):
        expected = run(direct, tokens)
        with patch.object(upstream, "layer_norm_gated_fwd", spy):
            got = run(lambda x, gate: wired.gated_rms_norm(x, gate, weight, 1e-6), tokens)
        for a, b in zip(got, expected):
            assert float((a - b).norm() / b.norm().clamp_min(1e-30)) < 1e-4
    # 150 and 200 tokens are 2400 and 3200 rows: one bucket. 5 tokens stay as they are.
    assert seen == [wired._GATED_ROW_BUCKET, wired._GATED_ROW_BUCKET, 5 * H]
