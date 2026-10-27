"""Opt-in device checks: UMM_WAN_DEVICE=npu:0 (or cuda:0) python -m pytest ...

Uses small models with hardware-sized attention heads. For the full pretrained
network use tests/validation/validate_wan21.py; neither suite needs a backbone or VAE.
"""
from copy import deepcopy
import os

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("diffusers")

from models.generator_wan21 import Wan21Generator, _attention


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("UMM_WAN_DEVICE")
    if not name:
        pytest.skip("Set UMM_WAN_DEVICE to explicitly enable accelerator tests")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401  # Registers the NPU device.
    result = torch.device(name)
    assert result.type in {"cuda", "npu"}, "An actual accelerator is required"
    return result


def small_generator(device):
    model = Wan21Generator(
        condition_dim=24, in_dim=4, out_dim=4, dim=128, ffn_dim=256,
        freq_dim=32, text_dim=32, num_heads=2, num_layers=2,
    )
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.05)
    return model.to(device)


def inputs(device):
    return (
        [torch.randn(4, 3, 8, 12, device=device), torch.randn(4, 5, 6, 8, device=device)],
        torch.tensor([150., 850.], device=device),
        torch.randn(2, 11, 24, device=device),
        torch.tensor([[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                      [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]], dtype=torch.bool, device=device),
    )


def nonzero_finite(value):
    assert value is not None
    assert bool(torch.isfinite(value).all())
    assert value.float().norm().item() > 0


def test_fused_attention_mask_and_backward_match_cpu_oracle(device):
    torch.manual_seed(81)
    cpu = [torch.randn(2, length, 2, 64).bfloat16() for length in (17, 11, 11)]
    actual_inputs = [x.to(device).requires_grad_() for x in cpu]
    oracle_inputs = [x.float().requires_grad_() for x in cpu]
    lengths = torch.tensor([11, 4], device=device)
    actual = _attention(*actual_inputs, lengths, "auto")
    allowed = torch.arange(11)[None] < lengths.cpu()[:, None]
    expected = F.scaled_dot_product_attention(
        *(x.transpose(1, 2) for x in oracle_inputs), attn_mask=allowed[:, None, None],
    ).transpose(1, 2)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0.006, rtol=0.02)
    probe = torch.randn_like(expected)
    (actual.float() * probe.to(device)).sum().backward()
    (expected * probe).sum().backward()
    for x, y in zip(actual_inputs, oracle_inputs):
        torch.testing.assert_close(x.grad.cpu().float(), y.grad, atol=0.012, rtol=0.03)
    for x in actual_inputs[1:]:
        assert torch.count_nonzero(x.grad[1, 4:]).item() == 0


@pytest.mark.parametrize("convert_weights", [False, True])
def test_variable_latents_condition_mask_and_backward(device, convert_weights):
    torch.manual_seed(82)
    model = small_generator(device).train()
    if convert_weights:
        model.bfloat16()
    latents, times, condition, mask = inputs(device)
    latents = [x.requires_grad_() for x in latents]
    condition = condition.masked_fill(~mask[..., None], float("nan")).requires_grad_()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not convert_weights):
        output = model(latents, times, condition, condition_mask=mask)
        with torch.no_grad():
            for i, row in enumerate(output):
                expected, = model([latents[i]], times[i:i + 1], condition[i, mask[i]][None])
                assert row.shape == latents[i].shape and row.dtype == torch.float32
                torch.testing.assert_close(row, expected, atol=0.02, rtol=0.02)
        loss = sum(x.square().mean() for x in output)
    loss.backward()
    for x in latents:
        nonzero_finite(x.grad)
    nonzero_finite(condition.grad[mask])
    assert torch.count_nonzero(condition.grad[~mask]).item() == 0
    for name, p in model.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
    for p in (model.condition_proj.weight, model.patch_embedding.weight,
              model.time_embedding[0].weight, model.blocks[0].self_attn.q.weight,
              model.blocks[0].cross_attn.k.weight):
        nonzero_finite(p.grad)


def test_checkpointing_matches_eager_and_accepts_long_conditions(device):
    torch.manual_seed(83)
    eager = small_generator(device).bfloat16().train()
    checked = deepcopy(eager)
    checked.enable_gradient_checkpointing()
    latents, times, _, _ = inputs(device)
    condition = torch.randn(2, 519, 24, device=device)
    mask = torch.ones(2, 519, dtype=torch.bool, device=device)
    mask[1, 7:] = False
    rows = [condition.clone().requires_grad_() for _ in range(2)]
    predictions = []
    for model, row in zip((eager, checked), rows):
        output = model(latents, times, row, condition_mask=mask)
        sum(x.square().mean() for x in output).backward()
        predictions.append(output)
    for x, y in zip(*predictions):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rows[0].grad, rows[1].grad, atol=1e-5, rtol=1e-5)
    for (name, x), (_, y) in zip(eager.named_parameters(), checked.named_parameters()):
        assert x.grad is not None, name
        torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)
