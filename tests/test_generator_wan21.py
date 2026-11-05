"""Independent Wan generator checks; no backbone, VAE or UMM composition needed."""
from copy import deepcopy
import json

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("diffusers")

from models.references import modeling_wan21 as native
from models.generator_wan21 import Wan21Generator


def native_state(model, *, gradients=False):
    values = {}
    for name, parameter in model.named_parameters():
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            continue
        name = name.replace('.impl.', '.')
        if '.kv.' in name:
            values.update({name.replace('.kv.', f'.{kind}.'): part for kind, part in zip(('k', 'v'), tensor.chunk(2, 0))})
        else:
            values[name] = tensor
    return values


def architecture(**updates):
    values = dict(model_type="t2v", patch_size=(1, 2, 2), text_len=5,
                  in_dim=4, out_dim=4, dim=24, ffn_dim=48, freq_dim=8,
                  text_dim=10, num_heads=3, num_layers=2,
                  window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6)
    return {**values, **updates}


def generator(**updates):
    model = Wan21Generator(condition_dim=7, attention_backend="sdpa", **architecture(), **updates)
    # The native zero output head deliberately blocks upstream gradients at
    # initialization. Activate it to exercise the entire conditional network.
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.05)
    return model


def inputs():
    return (
        [torch.randn(4, 2, 4, 6), torch.randn(4, 3, 2, 4)],
        torch.tensor([150., 850.]),
        torch.randn(2, 7, 7),
        torch.tensor([[True, False, True, True, False, True, False],
                      [False, True, False, False, True, False, False]]),
    )


def assert_nonzero_finite(tensor):
    assert tensor is not None
    assert torch.isfinite(tensor).all()
    assert tensor.abs().sum() > 0


def test_variable_videos_and_condition_holes_match_independent_calls():
    torch.manual_seed(71)
    model = generator().eval()
    latents, times, condition, mask = inputs()
    with torch.no_grad():
        batched = model(latents, times, condition, condition_mask=mask)
        for i, actual in enumerate(batched):
            expected, = model([latents[i]], times[i:i + 1], condition[i, mask[i]][None])
            assert actual.shape == latents[i].shape
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)

        # Padding must remain irrelevant even when filled with invalid values.
        changed = condition.masked_fill(~mask[..., None], float("nan"))
        padded = model(latents, times, changed, condition_mask=mask)
        for actual, expected in zip(padded, batched):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Changing one sample's condition cannot affect another video.
        changed = condition.clone()
        changed[1] += 100
        separate = model(latents, times, changed, condition_mask=mask)
        torch.testing.assert_close(separate[0], batched[0], atol=0, rtol=0)
        assert not torch.allclose(separate[1], batched[1])


def test_condition_longer_than_legacy_text_len_and_batch():
    model = generator().eval()
    latent = torch.randn(4, 1, 2, 2)
    # There is no 512-token ceiling and the condition need not match the video grid.
    condition = torch.randn(1, 519, 7)
    actual, = model([latent], torch.tensor([500.]), condition)
    assert actual.shape == latent.shape
    assert torch.isfinite(actual).all()


def test_gradients_reach_latents_conditions_and_original_wan_parameters():
    torch.manual_seed(72)
    model = generator().train()
    latents, times, condition, mask = inputs()
    latents = [x.requires_grad_() for x in latents]
    condition = condition.masked_fill(~mask[..., None], float("nan")).requires_grad_()
    output = model(latents, times, condition, condition_mask=mask)
    sum(x.square().mean() for x in output).backward()
    for x in latents:
        assert_nonzero_finite(x.grad)
    assert_nonzero_finite(condition.grad[mask])
    assert torch.count_nonzero(condition.grad[~mask]) == 0
    for name in ("text_embedding.0.weight", "text_embedding.2.weight", "patch_embedding.weight",
                 "time_embedding.0.weight", "time_projection.1.weight",
                 "blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight", "head.head.weight"):
        assert_nonzero_finite(native_state(model, gradients=True)[name])


def test_gradient_checkpointing_matches_eager_outputs_and_gradients():
    torch.manual_seed(73)
    eager = generator().train()
    checked = deepcopy(eager)
    checked.enable_gradient_checkpointing()
    assert checked.gradient_checkpointing
    latents, times, condition, mask = inputs()
    conditions = [condition.clone().requires_grad_() for _ in range(2)]
    predictions = []
    for model, c in zip((eager, checked), conditions):
        output = model(latents, times, c, condition_mask=mask)
        sum(x.square().mean() for x in output).backward()
        predictions.append(output)
    for actual, expected in zip(*predictions):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(conditions[0].grad, conditions[1].grad, rtol=0, atol=0)
    for (name, left), (_, right) in zip(eager.named_parameters(), checked.named_parameters()):
        assert left.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    checked.disable_gradient_checkpointing()
    assert not checked.gradient_checkpointing


@pytest.mark.parametrize("convert_weights", [False, True])
def test_bfloat16_execution_and_backward(convert_weights):
    torch.manual_seed(74)
    model = generator().train()
    if convert_weights:
        model.bfloat16()
    latents, times, condition, mask = inputs()
    condition.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not convert_weights):
        output = model(latents, times, condition, condition_mask=mask)
        loss = sum(row.square().mean() for row in output)
    loss.backward()
    assert all(row.dtype == torch.float32 and torch.isfinite(row).all() for row in output)
    assert_nonzero_finite(condition.grad)
    assert_nonzero_finite(model.time_embedding[0].weight.grad)
    assert_nonzero_finite(model.blocks[0].norm3.weight.grad)


def native_sdpa(q, k, v, k_lens=None, **kwargs):
    """Only replace the unavailable CUDA kernel in the upstream CPU oracle."""
    allowed = None
    if k_lens is not None:
        allowed = torch.arange(k.shape[1], device=q.device)[None] < k_lens.to(q.device)[:, None]
        allowed = allowed[:, None, None]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed,
    ).transpose(1, 2)


def test_matches_native_wan_forward_and_condition_gradients(monkeypatch):
    torch.manual_seed(75)
    source = native.WanModel(**architecture())
    with torch.no_grad():
        source.head.head.weight.normal_(std=0.05)
    adapted = Wan21Generator(condition_dim=10, attention_backend="sdpa", **architecture())
    with torch.no_grad():
        for name, target in native_state(adapted).items():
            target.copy_(source.state_dict()[name])
    monkeypatch.setattr(native, "flash_attention", native_sdpa)
    latents, times, _, _ = inputs()
    context = torch.randn(2, source.text_len, source.text_dim, requires_grad=True)
    other_context = context.detach().clone().requires_grad_()
    seq_len = max(math_prod(x.shape[1:]) // math_prod(source.patch_size) for x in latents)
    expected = source(latents, times, list(context), seq_len)
    actual = adapted(latents, times, other_context)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=5e-6, rtol=5e-5)
    sum(x.square().mean() for x in expected).backward()
    sum(x.square().mean() for x in actual).backward()
    torch.testing.assert_close(context.grad, other_context.grad, atol=2e-7, rtol=2e-4)
    torch.testing.assert_close(source.blocks[0].self_attn.q.weight.grad,
                               adapted.blocks[0].self_attn.q.weight.grad, atol=2e-7, rtol=2e-4)


def math_prod(shape):
    result = 1
    for value in shape:
        result *= value
    return result


def save_original(path, *, sharded=False):
    source = native.WanModel(**architecture())
    with torch.no_grad():
        source.head.head.weight.normal_(std=0.05)
    # Upstream ignore_for_config excludes several constructor fields. Store
    # them explicitly for this non-default tiny checkpoint, as real asset
    # configs/architecture overrides must do for non-default architectures.
    source.register_to_config(**architecture())
    source.save_pretrained(path, max_shard_size="4KB" if sharded else "1GB")
    return source


@pytest.mark.parametrize("sharded", [False, True])
def test_loads_real_native_checkpoint_without_resetting_wan_weights(tmp_path, sharded):
    torch.manual_seed(76)
    source = save_original(tmp_path / "wan", sharded=sharded)
    adapted = Wan21Generator.from_wan_pretrained(
        tmp_path / "wan", condition_dim=7, attention_backend="sdpa",
    )
    original = source.state_dict()
    assert set(native_state(adapted)) == set(original)
    assert adapted.text_embedding[0].weight.shape == (source.dim, 7)
    for name, value in original.items():
        if name != "text_embedding.0.weight":
            torch.testing.assert_close(native_state(adapted)[name], value, rtol=0, atol=0)
    assert all(not parameter.is_meta for parameter in adapted.parameters())
    assert_nonzero_finite(adapted.text_embedding[0].weight)
    assert not adapted.training
    latents, times, condition, mask = inputs()
    assert all(torch.isfinite(x).all() for x in adapted(latents, times, condition, condition_mask=mask))


def test_native_load_bfloat16_and_full_generator_checkpoint_roundtrip(tmp_path):
    save_original(tmp_path / "wan")
    model = Wan21Generator.from_wan_pretrained(
        tmp_path / "wan", condition_dim=7, attention_backend="sdpa", torch_dtype=torch.bfloat16,
        gradient_checkpointing=True,
    )
    assert all(p.dtype == (torch.float32 if name.startswith(("time_embedding.", "time_projection.")) else torch.bfloat16)
               for name, p in model.named_parameters())
    model.save_pretrained(tmp_path / "adapted")
    saved = json.loads((tmp_path / "adapted/config.json").read_text())
    for key in ("condition_dim", "patch_size", "text_dim", "qk_norm", "cross_attn_norm", "attention_backend"):
        assert key in saved
    restored, info = Wan21Generator.from_pretrained(
        tmp_path / "adapted", local_files_only=True, torch_dtype=torch.bfloat16, output_loading_info=True,
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert restored.gradient_checkpointing
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    latents, times, condition, mask = inputs()
    for actual, expected in zip(restored(latents, times, condition, condition_mask=mask),
                                model(latents, times, condition, condition_mask=mask)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_rejects_incomplete_or_incompatible_native_checkpoints(tmp_path, corruption):
    from safetensors.torch import load_file, save_file
    save_original(tmp_path)
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    state = load_file(path)
    key = "blocks.0.self_attn.q.weight"
    if corruption == "missing":
        del state[key]
    elif corruption == "unexpected":
        state["not_a_wan_parameter"] = torch.ones(1)
    else:
        state[key] = state[key][:-1].contiguous()
    save_file(state, path, metadata={"format": "pt"})
    with pytest.raises((ValueError, RuntimeError)):
        Wan21Generator.from_wan_pretrained(tmp_path, condition_dim=7)


def test_complete_checkpoint_cannot_silently_lose_adapter_weight(tmp_path):
    from safetensors.torch import load_file, save_file
    generator().save_pretrained(tmp_path)
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    state = load_file(path)
    del state["text_embedding.0.impl.weight"]
    save_file(state, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Incomplete adapted"):
        Wan21Generator.from_pretrained(tmp_path, local_files_only=True)



def test_rejects_old_three_layer_adapter_checkpoint(tmp_path):
    from safetensors.torch import load_file, save_file
    generator().save_pretrained(tmp_path)
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    state = load_file(path)
    # Even equal input dimensions must not silently accept an old extra layer.
    state["condition_proj.weight"] = torch.eye(7)
    state["condition_proj.bias"] = torch.zeros(7)
    save_file(state, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Incomplete adapted"):
        Wan21Generator.from_pretrained(tmp_path, local_files_only=True)


def test_real_rope_matches_native_complex_rotation_and_extends_axes():
    from ops.wan_layout import wan_rope
    from models.wan_layers import rotary
    torch.manual_seed(77)
    grid = (2, 2, 3)
    x = torch.randn(1, 12, 3, 8, requires_grad=True)
    source = native.WanModel(**architecture())
    expected = native.rope_apply(x, torch.tensor([grid]), source.freqs)
    actual = rotary(x[0], wan_rope(grid, 8))
    torch.testing.assert_close(actual, expected[0], atol=5e-7, rtol=5e-6)
    cos, sin = wan_rope((1025, 1, 1), 8)
    assert cos.shape == (1025, 1, 4)
    assert torch.isfinite(cos).all() and cos.dtype == torch.float32


@pytest.mark.parametrize("bad", ["empty", "channels", "patch", "time", "condition", "mask", "all_masked"])
def test_invalid_inputs_are_rejected(bad):
    model = generator()
    latents, times, condition, mask = inputs()
    if bad == "empty":
        latents = []
    elif bad == "channels":
        latents[0] = latents[0][:3]
    elif bad == "patch":
        latents[0] = latents[0][:, :, :3]
    elif bad == "time":
        times = times[:1]
    elif bad == "condition":
        condition = condition[..., :6]
    elif bad == "mask":
        mask = mask.float()
    else:
        mask[1] = False
    with pytest.raises(ValueError):
        model(latents, times, condition, condition_mask=mask)


def test_unsupported_architectures_and_backend_fail_explicitly():
    with pytest.raises(ValueError, match="T2V"):
        Wan21Generator(condition_dim=7, **architecture(model_type="i2v"))
    with pytest.raises(ValueError, match="global"):
        Wan21Generator(condition_dim=7, **architecture(window_size=(2, 2)))
    with pytest.raises(ValueError, match="attention_backend"):
        Wan21Generator(condition_dim=7, attention_backend="flash", **architecture())


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_sampler_integrates_native_noise_minus_clean_direction(monkeypatch, solver):
    torch.manual_seed(119)
    network = Wan21Generator(condition_dim=12, **architecture())
    noise = [torch.randn(4, 2, 4, 4)]
    clean = torch.randn_like(noise[0])
    velocity = noise[0] - clean
    seen = []
    def constant_velocity(values, timesteps, condition, *args, **kwargs):
        seen.append(timesteps.clone())
        return [velocity.expand_as(x) for x in values]
    monkeypatch.setattr(network, "_predict_videos", constant_velocity)
    result = network.sample(torch.randn(1, 3, 12), initial_noise=noise, steps=4,
                            solver=solver, time_shift=3.0)
    torch.testing.assert_close(result[0], clean, rtol=1e-5, atol=1e-6)
    assert float(seen[0][0]) == 1000.
    assert all(bool(((t >= 0) & (t <= 1000)).all()) for t in seen)
    assert len(seen) == 4 * (2 if solver == "heun" else 1)
