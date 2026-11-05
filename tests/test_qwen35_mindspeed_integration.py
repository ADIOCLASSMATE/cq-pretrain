"""Opt-in shared CQ layers on AscendC and MindSpeed; no CPU fallback counts."""
from copy import deepcopy
import os

import pytest
import torch

from models.modeling_umm import UMM
from models.qwen35_layers import cast_for_training
from tests.test_modeling_umm_video import configuration, video_batch


@pytest.fixture(params=["ascendc", "mindspeed"])
def network(request, monkeypatch):
    device = os.environ.get("UMM_VIDEO_DEVICE", "")
    if not device.startswith("npu"):
        pytest.skip("Set UMM_VIDEO_DEVICE=npu:<rank>")
    from distributed.megatron import prepare_npu
    prepare_npu()
    torch.manual_seed(721)
    cfg = configuration(layers=4, freeze=True)
    cfg.qwen_config.text_config.linear_key_head_dim = 128
    cfg.qwen_config.text_config.linear_value_head_dim = 128
    cfg.qwen_config.text_config.qwen_gdn_backend = request.param
    result = cast_for_training(UMM(cfg), device).train()
    if request.param == "ascendc":
        import ops.gated_delta_net as provider
        name = "delta"
    else:
        import importlib
        provider = importlib.import_module("mindspeed.core.ssm.npu_chunk_gated_delta_rule")
        name = "chunk_gated_delta_rule"
    original, calls = getattr(provider, name), []
    def audited(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(provider, name, audited)
    yield result
    assert calls, "Requested NPU kernel did not execute"


@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_time_blocks_no_target_leak_and_joint_backward(network, checkpointed):
    device = next(network.parameters()).device
    network.backbone.gradient_checkpointing = checkpointed
    network.generator.cfg.checkpoint_dit = checkpointed
    batch = video_batch(device)
    batch.pop("pixel_values_videos")
    features = torch.randn(8, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    batch["video_features"] = [features]
    noisy = [torch.randn(4, 3, 4, 6, device=device)]
    time = torch.tensor([700.], device=device)
    with torch.autocast("npu", dtype=torch.bfloat16):
        output = network(batch, noisy, time, compute_text_loss=True)
        output["velocity"][0].square().mean().backward(retain_graph=True)
        assert torch.count_nonzero(features.grad) == 0
        network.zero_grad(set_to_none=True)
        features.grad = None
        output["text_loss"].backward()
    assert features.grad.abs().max() > 0
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        other = network(dict(batch, video_features=[features.detach() * 5]), noisy, time)
    torch.testing.assert_close(output["condition"], other["condition"], rtol=0, atol=0)


def test_non_cfg_queries_ignore_every_content_token(network):
    device = next(network.parameters()).device
    batch = video_batch(device)
    batch.pop("pixel_values_videos")
    batch["video_features"] = [torch.randn(8, 32, device=device, dtype=torch.bfloat16)]
    batch["video_uncond_mask"] = torch.ones(1, device=device, dtype=torch.bool)
    prepared = network.prepare_inputs(batch)
    source = prepared["inputs_embeds"].detach().requires_grad_()
    with torch.autocast("npu", dtype=torch.bfloat16):
        output = network.backbone(dict(prepared, inputs_embeds=source))["query_hidden"]
        slots = prepared["video_token_indices"][0]
        (output[:, slots] * torch.randn_like(output[:, slots])).sum().backward()
    # Reverse-prefix sum can have FP32 cancellation residuals; the complete
    # C→Q Jacobian is exactly zero because no read is placed after any C write.
    assert torch.count_nonzero(source.grad) == 0
