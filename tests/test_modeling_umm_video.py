"""Stage-one video integration; opt-in NPU runs: UMM_VIDEO_DEVICE=npu:0 pytest ...

Native Qwen frame layouts are constructed directly so model tests require no
video decoder/dataset or downloads. The actual native ViT, CQ and Wan run here.
"""
from copy import deepcopy
import os

import pytest
import torch

from models.references import modeling_qwen35 as native
from models.modeling_umm import UMM, UMMConfig
from models.references.modeling_wan21 import WanModel
def build_cq_masks(prepared):
    plan, indices = prepared["_cq_plan"], prepared["_content_positions"]
    b, length = prepared["input_ids"].shape
    n = len(indices)
    allowed = ~plan.cq_mask[0, 0] & plan.cq_valid[0, :, None]
    c = torch.zeros(b * length, b * length, dtype=torch.bool, device=indices.device)
    q = torch.zeros_like(c)
    c[indices[:, None], indices] = allowed[:n]
    q[plan.output_positions[:, None], indices] = allowed[n:]
    return tuple(torch.stack([x[i*length:(i+1)*length, i*length:(i+1)*length] for i in range(b)]) for x in (q, c))


@pytest.fixture(params=["cpu"] + ([os.environ["UMM_VIDEO_DEVICE"]] if os.environ.get("UMM_VIDEO_DEVICE") else []))
def device(request):
    if request.param.startswith("npu"):
        import torch_npu  # noqa: F401 -- register the actual NPU runtime
    return torch.device(request.param)


def configuration(*, layers=8, freeze=False, checkpointed=False):
    qwen = native.Qwen3_5Config(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4, tie_word_embeddings=True,
            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
            rope_parameters=dict(rope_type="default", rope_theta=10000., partial_rotary_factor=0.5,
                                 mrope_section=[1, 1, 0]), pad_token_id=0,
        ),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, num_position_embeddings=16, patch_size=2,
                           temporal_patch_size=2, spatial_merge_size=2),
        video_token_id=58, image_token_id=57, vision_start_token_id=61, vision_end_token_id=62,
    )
    qwen.text_config._attn_implementation = qwen.vision_config._attn_implementation = "sdpa"
    wan = dict(in_dim=4, out_dim=4, dim=32, ffn_dim=64, freq_dim=8, text_dim=48,
               num_heads=4, num_layers=2, text_len=3, gradient_checkpointing=checkpointed)
    return UMMConfig(qwen, 59, 60, wan, freeze_vision=freeze, gradient_checkpointing=checkpointed)


def model(device, **kwargs):
    result = UMM(configuration(**kwargs)).to(device)
    # Native scratch Wan initializes its head to zero; make the gradient path
    # observable just as it is with a trained Wan checkpoint.
    with torch.no_grad():
        result.generator.head.head.weight.normal_(std=0.05)
    return result


def video_batch(device, specs=(((2, 4, 4),),), *, requires_grad=False):
    """specs is rows -> complete videos -> native pre-merger grid (T,H,W)."""
    sequences, grids, target_ids = [], [], []
    for videos in specs:
        ids = [3, 4]
        for t, h, w in videos:
            target_ids.append(len(grids))
            grids.append((t, h, w))
            for frame in range(t):
                ids += [7 + frame, 61] + [58] * (h * w // 4) + [62]
        ids += [12, 13]
        sequences.append(ids)
    length = max(map(len, sequences))
    ids = torch.zeros(len(sequences), length, dtype=torch.long, device=device)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.full_like(ids, -100)
    for row, values in enumerate(sequences):
        ids[row, :len(values)] = torch.tensor(values, device=device)
        valid[row, :len(values)] = True
        labels[row, len(values) - 2:len(values)] = ids[row, len(values) - 2:len(values)]
    grid = torch.tensor(grids, device=device, dtype=torch.long).reshape(-1, 3)
    patches = torch.randn(int(grid.prod(-1).sum()), 24, device=device, requires_grad=requires_grad)
    return dict(input_ids=ids, attention_mask=valid, labels=labels, mm_token_type_ids=ids.eq(58).long() * 2,
                video_grid_thw=grid, pixel_values_videos=patches,
                target_video_ids=torch.tensor(target_ids, device=device, dtype=torch.long))


def test_visual_injection_accepts_custom_autograd_embedding_views():
    class EmbeddingView(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, gradient):
            return gradient

    torch.manual_seed(502)
    network = model("cpu", layers=4)
    batch = video_batch("cpu", (((2, 4, 4),), ((1, 4, 4),)), requires_grad=True)
    expected = network(batch, compute_text_loss=True)["text_loss"].detach()
    handle = network.backbone.embed_tokens.register_forward_hook(
        lambda module, inputs, output: EmbeddingView.apply(output))
    try:
        loss = network(batch, compute_text_loss=True)["text_loss"]
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        for gradient in (batch["pixel_values_videos"].grad, network.backbone.embed_tokens.weight.grad):
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    finally:
        handle.remove()


def test_native_vision_features_positions_and_scatter_match(device):
    torch.manual_seed(101)
    actual = model(device).eval()
    reference = native.Qwen3_5Model(deepcopy(actual.config.qwen_config)).to(device).eval()
    reference.visual.load_state_dict(actual.vision_encoder.visual.state_dict(), strict=True)
    batch = video_batch(device, (((2, 4, 4), (1, 4, 4)), ((2, 4, 8),)))
    with torch.no_grad():
        features = reference.get_video_features(batch["pixel_values_videos"], batch["video_grid_thw"]).pooler_output
        prepared = actual.prepare_inputs(batch)
        positions, _ = reference.get_rope_index(
            batch["input_ids"], batch["mm_token_type_ids"], video_grid_thw=batch["video_grid_thw"],
            attention_mask=batch["attention_mask"],
        )
    torch.testing.assert_close(prepared["position_ids"], positions, rtol=0, atol=0)
    assert prepared["video_batch_indices"].tolist() == [0, 0, 1]
    for i, (row, indices) in enumerate(zip(prepared["video_batch_indices"], prepared["video_token_indices"])):
        torch.testing.assert_close(prepared["inputs_embeds"][row, indices], features[i], rtol=0, atol=0)
        boundaries = prepared["query_read_end"][row, indices]
        assert bool(boundaries.eq(indices[0]).all())
        qmask, cmask = build_cq_masks(prepared)
        assert not bool(qmask[row, indices, int(indices[0]):].any())
        blocks = indices.reshape(int(batch["video_grid_thw"][i, 0]), -1)
        for block in blocks:
            assert cmask[row, block[:, None], block].all()
            assert not cmask[row, block, int(block[-1]) + 1:].any()
    assert actual.lm_head.weight is actual.backbone.embed_tokens.weight


@pytest.mark.parametrize("checkpointed", [False, True])
def test_whole_video_no_leak_train_generation_alignment_and_gradients(device, checkpointed):
    torch.manual_seed(102)
    network = model(device, checkpointed=checkpointed).train()
    batch = video_batch(device, requires_grad=True)
    noisy = [torch.randn(4, 3, 4, 6, device=device)]  # 18 Wan patches vs 8 Qwen tokens
    time = torch.tensor([700.], device=device)
    output = network(batch, noisy, time, return_hidden_states=True)
    prepared = network.prepare_inputs(batch)
    selected = prepared["video_token_indices"][0]
    assert output["query_hidden"].shape == output["content_hidden"].shape == (*batch["input_ids"].shape, 32)
    torch.testing.assert_close(output["condition"][0], output["query_hidden"][0, selected], rtol=0, atol=0)
    valid = batch["labels"].ne(-100)
    torch.testing.assert_close(output["text_logits"], network.lm_head(output["query_hidden"][valid]))
    torch.testing.assert_close(output["text_labels"], batch["labels"][valid])
    assert output["velocity"][0].shape == noisy[0].shape
    assert output["velocity"][0].dtype == torch.float32

    # Flow must depend on query conditioning/backbone, but never target pixels.
    loss = (output["velocity"][0] * torch.randn_like(output["velocity"][0])).sum()
    loss.backward(retain_graph=True)
    pixel_grad = batch["pixel_values_videos"].grad.clone()
    mask_grad = network.backbone.embed_tokens.weight.grad.clone()
    bridge_grad = network.generator.text_embedding[0].weight.grad.clone()
    network.zero_grad(set_to_none=True)
    assert torch.count_nonzero(pixel_grad) == 0
    assert mask_grad[network.config.video_mask_token_id].abs().max() > 0
    assert bridge_grad.abs().max() > 0
    # Later same-position text queries DO read video content and train the ViT.
    text_loss = (output["text_logits"] * torch.randn_like(output["text_logits"])).sum()
    text_loss.backward()
    vision_grad = network.vision_encoder.visual.patch_embed.proj.weight.grad
    assert torch.isfinite(vision_grad).all() and vision_grad.abs().max() > 0

    network.eval()
    changed = {**batch, "pixel_values_videos": torch.randn_like(batch["pixel_values_videos"]) * 5}
    generation = {key: value for key, value in batch.items() if key != "pixel_values_videos"}
    generation["video_observed_mask"] = torch.tensor([False], device=device)
    with torch.no_grad():
        alternative = network(changed, noisy, time, return_hidden_states=True)
        generated = network(generation, noisy, time, return_hidden_states=True)
        for other in (alternative, generated):
            torch.testing.assert_close(other["condition"], output["condition"], rtol=0, atol=0)
            torch.testing.assert_close(other["velocity"][0], output["velocity"][0], rtol=0, atol=0)
            assert not torch.allclose(other["content_hidden"][0, selected], output["content_hidden"][0, selected])
            assert not torch.allclose(other["text_logits"], output["text_logits"])
        condition, mask = network.encode_condition(generation)
        torch.testing.assert_close(condition, generated["condition"], rtol=0, atol=0)
        torch.testing.assert_close(mask, generated["condition_mask"])
        changed_ids = batch["input_ids"].clone()
        changed_ids[:, 0] = 15
        conditioned, _ = network.encode_condition({**generation, "input_ids": changed_ids})
        assert not torch.allclose(conditioned, condition)


def test_query_independence_through_all_video_backbone_layers(device):
    torch.manual_seed(103)
    network = model(device).eval()
    batch = video_batch(device)
    selected = network.prepare_inputs(batch)["video_token_indices"][0]
    changed_position = int(selected[1])
    supplied = {}

    prepared = network.prepare_inputs(batch)
    plan = prepared["_cq_plan"]
    query_index = int((plan.output_positions == changed_position).nonzero()[0])

    def replace_query(module, args):
        hidden, rope, layout, n = args
        if "query" not in supplied:
            supplied["query"] = hidden[n:].detach().clone().requires_grad_()
        return torch.cat((hidden[:n], supplied["query"])), rope, layout, n

    handle = network.backbone.layers[0].register_forward_pre_hook(replace_query)
    try:
        original = network(batch, return_hidden_states=True)
        query = supplied["query"]
        probe = torch.randn_like(original["query_hidden"][:, changed_position])
        grad, = torch.autograd.grad((original["query_hidden"][:, changed_position] * probe).sum(), query)
        others = torch.arange(len(query), device=device) != query_index
        assert torch.count_nonzero(grad[others]) == 0
        assert grad[query_index].abs().max() > 0
        supplied["query"] = query.detach().clone()
        supplied["query"][query_index] *= 5
        with torch.no_grad():
            changed = network(batch, return_hidden_states=True)
        torch.testing.assert_close(changed["content_hidden"], original["content_hidden"], rtol=0, atol=0)
        positions = torch.arange(batch["input_ids"].shape[1], device=device) != changed_position
        torch.testing.assert_close(changed["query_hidden"][:, positions], original["query_hidden"][:, positions], rtol=0, atol=0)
        assert not torch.allclose(changed["query_hidden"][:, changed_position], original["query_hidden"][:, changed_position])
    finally:
        handle.remove()


def test_variable_video_target_order_masks_and_cfg_use_whole_video_ids(device):
    torch.manual_seed(104)
    network = model(device).eval()
    batch = video_batch(device, (((2, 4, 4), (1, 4, 4)), ((2, 4, 8),)))
    batch["target_video_ids"] = torch.tensor([2, 1], device=device)
    batch["video_uncond_mask"] = torch.tensor([False, True, False], device=device)
    noisy = [torch.randn(4, 2, 4, 4, device=device), torch.randn(4, 3, 4, 6, device=device)]
    times = torch.tensor([300., 800.], device=device)
    with torch.no_grad():
        output = network(batch, noisy, times, return_hidden_states=True)
        prepared = network.prepare_inputs(batch)
        qmask, cmask = build_cq_masks(prepared)
        target = prepared["video_token_indices"][1]
        assert not bool(qmask[0, target].any())
        allowed = torch.where(cmask[0, target[0]])[0]
        torch.testing.assert_close(allowed, torch.arange(int(target[-1]) + 1, device=device))
        assert output["condition_mask"].sum(-1).tolist() == [16, 4]
        assert torch.count_nonzero(output["condition"][1, 4:]) == 0
        assert output["target_video_ids"].tolist() == [2, 1]
        for i, video_id in enumerate([2, 1]):
            row = prepared["video_batch_indices"][video_id]
            selected = prepared["video_token_indices"][video_id]
            torch.testing.assert_close(output["condition"][i, :selected.numel()], output["query_hidden"][row, selected])
        # Changing the observed earlier video must not enter the dropped target.
        pixels = batch["pixel_values_videos"].clone()
        pixels[:32] *= 5
        changed = network({**batch, "pixel_values_videos": pixels}, noisy, times)
        torch.testing.assert_close(output["condition"], changed["condition"], rtol=0, atol=0)
        # Conditional mode does read that earlier observed video.
        conditional = {**batch, "video_uncond_mask": torch.zeros(3, device=device, dtype=torch.bool)}
        left, _ = network.encode_condition(conditional)
        right, _ = network.encode_condition({**conditional, "pixel_values_videos": pixels})
        assert not torch.allclose(left[1], right[1])
        # Keep the first observed/context video, mask only the two targets.
        generated = {**conditional, "video_observed_mask": torch.tensor([True, False, False], device=device),
                     "pixel_values_videos": batch["pixel_values_videos"][:32]}
        generated_condition, _ = network.encode_condition(generated)
        torch.testing.assert_close(generated_condition, left, rtol=0, atol=0)
        features = network.vision_encoder(batch["pixel_values_videos"], batch["video_grid_thw"])
        cached = {key: value for key, value in generated.items() if key != "pixel_values_videos"}
        cached["video_features"] = [features[0], None, None]
        cached_condition, _ = network.encode_condition(cached)
        torch.testing.assert_close(cached_condition, generated_condition, rtol=0, atol=0)


def test_packed_video_segments_match_independent_execution(device):
    torch.manual_seed(105)
    network = model(device).eval()
    first, second = video_batch(device), video_batch(device, (((1, 4, 8),),))
    length = first["input_ids"].shape[1]
    packed = {key: torch.cat((first[key], second[key]), dim=1)
              for key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels")}
    packed.update(video_grid_thw=torch.cat((first["video_grid_thw"], second["video_grid_thw"])),
                  pixel_values_videos=torch.cat((first["pixel_values_videos"], second["pixel_values_videos"])),
                  segment_ids=torch.cat((torch.zeros_like(first["input_ids"]), torch.ones_like(second["input_ids"])), 1))
    with torch.no_grad():
        combined = network(packed, return_hidden_states=True)
        for offset, batch in ((0, first), (length, second)):
            output = network(batch, return_hidden_states=True)
            for key in ("content_hidden", "query_hidden"):
                torch.testing.assert_close(combined[key][:, offset:offset + batch["input_ids"].shape[1]], output[key],
                                           rtol=2e-4, atol=3e-6)


def test_plain_text_and_frozen_native_vision(device):
    network = model(device, freeze=True).train()
    assert not network.vision_encoder.training
    assert all(not p.requires_grad for p in network.vision_encoder.parameters())
    ids = torch.tensor([[3, 4, 12]], device=device)
    output = network({"input_ids": ids, "labels": ids}, return_hidden_states=True)
    assert output["text_logits"].shape == (3, 64)
    torch.testing.assert_close(output["text_labels"], ids.flatten())
    changed = network({"input_ids": torch.tensor([[3, 4, 15]], device=device)})
    torch.testing.assert_close(output["text_logits"][-1], changed["text_logits"][0, -1], rtol=0, atol=0)


@pytest.mark.parametrize("converted_weights", [False, True])
def test_bfloat16_joint_forward_backward_and_checkpoint_parity(device, converted_weights):
    torch.manual_seed(107)
    plain = model(device).train()
    if converted_weights:
        plain.bfloat16()
    checked = deepcopy(plain)
    checked.backbone.gradient_checkpointing = True
    checked.generator.enable_gradient_checkpointing()
    batch = video_batch(device)
    noisy = [torch.randn(4, 3, 4, 6, device=device)]
    times = torch.tensor([600.], device=device)
    snapshots = []
    for network in (plain, checked):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not converted_weights):
            output = network(batch, noisy, times)
            # A real training-shaped objective exercises both paths. Target
            # pixels are blocked from flow, while suffix CE trains the ViT.
            loss = output["velocity"][0].square().mean() + torch.nn.functional.cross_entropy(
                output["text_logits"].float(), output["text_labels"],
            )
        loss.backward()
        gradients = {key: p.grad.detach().clone() for key, p in network.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(grad).all() for grad in gradients.values())
        for key in ("generator.text_embedding.0.impl.weight", "backbone.layers.0.linear_attn.in_proj_all.impl.weight",
                    "backbone.vision_encoder.visual.patch_embed.proj.weight"):
            assert gradients[key].abs().max() > 0
        snapshots.append((output, gradients))
    for key in ("condition", "text_logits"):
        torch.testing.assert_close(snapshots[0][0][key], snapshots[1][0][key], rtol=0, atol=0)
    torch.testing.assert_close(snapshots[0][0]["velocity"][0], snapshots[1][0]["velocity"][0], rtol=0, atol=0)
    assert snapshots[0][1].keys() == snapshots[1][1].keys()
    for key, grad in snapshots[0][1].items():
        torch.testing.assert_close(grad, snapshots[1][1][key], rtol=0.02, atol=2e-5)


@pytest.mark.parametrize("fault", ["frame_split_boundary", "metadata_leak_boundary", "missing_pixels", "wrong_grid",
                                    "wrong_mm_types", "wrong_targets", "wrong_positions", "wrong_indices"])
def test_rejects_misaligned_or_leaking_video_inputs(fault):
    network = model(torch.device("cpu")).eval()
    batch = video_batch("cpu")
    prepared = network.prepare_inputs(batch)
    if fault == "frame_split_boundary":
        batch["query_read_end"] = prepared["query_read_end"].clone()
        batch["query_read_end"][0, prepared["video_token_indices"][0][-4:]] += 1
    elif fault == "metadata_leak_boundary":
        batch["query_read_end"] = prepared["query_read_end"].clone()
        batch["query_read_end"][0, prepared["video_token_indices"][0]] = 11
    elif fault == "missing_pixels":
        batch.pop("pixel_values_videos")
    elif fault == "wrong_grid":
        batch["video_grid_thw"] = torch.tensor([[1, 4, 4]])
    elif fault == "wrong_mm_types":
        batch["mm_token_type_ids"] = prepared["token_types"]
    elif fault == "wrong_targets":
        batch["target_video_ids"] = torch.tensor([1])
    elif fault == "wrong_positions":
        batch["position_ids"] = prepared["position_ids"] + 1
    elif fault == "wrong_indices":
        batch["video_token_indices"] = [prepared["video_token_indices"][0].flip(0)]
    else:
        prepared["visual_uncond_mask"][0, prepared["video_token_indices"][0][0]] = True
        with pytest.raises(ValueError, match="complete visual group"):
            network.backbone(prepared)
        return
    with pytest.raises(ValueError):
        network(batch)


def test_native_component_loading_and_state_dict_roundtrip(tmp_path):
    torch.manual_seed(106)
    cfg = configuration(layers=4)
    original = native.Qwen3_5ForConditionalGeneration(cfg.qwen_config).eval()
    original.save_pretrained(tmp_path / "qwen", max_shard_size="30KB")
    wan_config = {k: v for k, v in cfg.wan_config.items() if k != "gradient_checkpointing"}
    wan = WanModel(**wan_config)
    wan.register_to_config(**wan_config)
    with torch.no_grad():
        wan.head.head.weight.normal_(std=0.05)
    wan.save_pretrained(tmp_path / "wan", max_shard_size="30KB")
    loaded = UMM.from_pretrained_components(
        tmp_path / "qwen", tmp_path / "wan", mask_token_id=64, video_mask_token_id=65,
    )
    for name, value in original.model.language_model.state_dict().items():
        actual = loaded.backbone.native_parameters()[name]
        torch.testing.assert_close(actual[:64] if name == "embed_tokens.weight" else actual, value, rtol=0, atol=0)
    for name, value in original.model.visual.state_dict().items():
        torch.testing.assert_close(loaded.vision_encoder.visual.state_dict()[name], value, rtol=0, atol=0)
    from tests.test_generator_wan21 import native_state
    for name, value in wan.state_dict().items():
        if name != "text_embedding.0.weight":
            torch.testing.assert_close(native_state(loaded.generator)[name], value, rtol=0, atol=0)
    assert loaded.backbone.embed_tokens.num_embeddings == 66
    assert loaded.lm_head.weight is loaded.backbone.embed_tokens.weight
    restored = UMM(UMMConfig(**loaded.config.to_dict())).eval()
    restored.load_state_dict(loaded.state_dict(), strict=True)
    batch = video_batch("cpu")
    with torch.no_grad():
        left, mask = loaded.encode_condition(batch)
        right, other_mask = restored.encode_condition(batch)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(mask, other_mask)


@pytest.mark.parametrize("cfg_drop", [False, True])
def test_multimodal_content_matches_every_native_layer_and_cfg_does_not_change_it(device, cfg_drop):
    torch.manual_seed(111)
    network = model(device).eval()
    reference = native.Qwen3_5Model(deepcopy(network.config.qwen_config)).to(device).eval()
    reference.visual.load_state_dict(network.vision_encoder.visual.state_dict())
    # Causal is the native Qwen oracle; visual-block visibility is checked separately.
    network.backbone.cfg.content_visibility = "causal"
    reference.language_model.load_state_dict(network.backbone.native_parameters())
    batch = video_batch(device)
    batch["video_uncond_mask"] = torch.tensor([cfg_drop], device=device)
    captured, expected_layers = [], []
    handles = [layer.register_forward_hook(lambda m, a, out: captured.append(out[:a[3], 0][None]))
               for layer in network.backbone.layers]
    handles += [layer.register_forward_hook(lambda m, a, out: expected_layers.append(out))
                for layer in reference.language_model.layers]
    try:
        with torch.no_grad():
            actual = network(batch, return_hidden_states=True)
            expected = reference(**{k: batch[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids",
                                                          "video_grid_thw", "pixel_values_videos")}, use_cache=False)
        assert len(captured) == len(expected_layers) == 8
        for left, right in zip(captured, expected_layers):
            torch.testing.assert_close(left, right, rtol=2e-4, atol=3e-6)
        torch.testing.assert_close(actual["content_hidden"], expected.last_hidden_state, rtol=2e-4, atol=3e-6)
    finally:
        for handle in handles:
            handle.remove()


def test_video_queries_exclude_all_post_prefix_content_including_frame_characters(device):
    torch.manual_seed(112)
    network = model(device).eval()
    prepared = network.prepare_inputs(video_batch(device))
    slots = prepared["video_token_indices"][0]
    boundary = int(slots[0])
    source = prepared["inputs_embeds"].detach().requires_grad_()
    baseline = network.backbone({**prepared, "inputs_embeds": source}, return_content=True)
    probe = torch.randn_like(baseline["query_hidden"][:, slots])
    gradient, = torch.autograd.grad((baseline["query_hidden"][:, slots] * probe).sum(), source)
    assert torch.count_nonzero(gradient[:, boundary:]) == 0
    assert gradient[:, :boundary].abs().max() > 0
    changed = source.detach().clone()
    changed[:, boundary:] = torch.randn_like(changed[:, boundary:]) * 5
    with torch.no_grad():
        other = network.backbone({**prepared, "inputs_embeds": changed}, return_content=True)
    torch.testing.assert_close(baseline["query_hidden"][:, slots], other["query_hidden"][:, slots], rtol=0, atol=0)
    assert not torch.allclose(baseline["query_hidden"][:, -1], other["query_hidden"][:, -1])
    # Independent reference: one query after only its allowed prefix, retaining
    # that query's original M-RoPE position. No target or inter-frame content.
    with torch.no_grad():
        for slot in (int(slots[0]), int(slots[-1])):
            selected = torch.cat((torch.arange(boundary, device=device), slots.new_tensor([slot])))
            minimal = {k: prepared[k][:, selected] for k in ("input_ids", "token_types")}
            minimal.update(inputs_embeds=source.detach()[:, selected], position_ids=prepared["position_ids"][:, :, selected])
            result = network.backbone(minimal)["query_hidden"][:, -1]
            torch.testing.assert_close(result, baseline["query_hidden"][:, slot], rtol=2e-4, atol=3e-6)


@pytest.mark.parametrize("with_video", [False, True])
def test_uncached_text_decoding_matches_same_position_training(device, with_video):
    torch.manual_seed(113)
    network = model(device).eval()
    prefix = video_batch(device) if with_video else {"input_ids": torch.tensor([[3, 4, 7]], device=device)}
    calls = []
    handle = network.backbone.register_forward_pre_hook(lambda module, args: calls.append(args[0]["input_ids"].shape[1]))
    try:
        decoded = network.generate_text(prefix, max_new_tokens=3, eos_token_id=-1, return_logits=True)
    finally:
        handle.remove()
    length = prefix["input_ids"].shape[1]
    assert calls == [length + 1, length + 2, length + 3]
    full = torch.cat((prefix["input_ids"], decoded["sequences"]), 1)
    labels = full.clone()
    labels[:, :length] = -100
    teacher = {k: v for k, v in prefix.items() if k in ("pixel_values_videos", "video_grid_thw")}
    teacher.update(input_ids=full, labels=labels)
    network.train()
    trained = network(teacher)
    torch.testing.assert_close(trained["text_logits"], decoded["logits"].flatten(0, 1), rtol=2e-4, atol=3e-6)
    torch.testing.assert_close(trained["text_labels"], decoded["sequences"].flatten())
    with pytest.raises(NotImplementedError, match="no-cache"):
        network.generate_text(prefix, max_new_tokens=1, use_cache=True)


def test_uncached_text_generation_handles_different_prompt_lengths(device):
    torch.manual_seed(114)
    network = model(device).eval()
    batch = video_batch(device, (((1, 4, 4),), ((2, 4, 4),)))
    with torch.no_grad():
        output = network.generate_text(batch, max_new_tokens=2, eos_token_id=-1, return_logits=True)
        # The first row has right padding; the second has a longer real prefix.
        first = {k: v[:1] for k, v in batch.items() if k in ("input_ids", "attention_mask", "mm_token_type_ids")}
        first.update(video_grid_thw=batch["video_grid_thw"][:1], pixel_values_videos=batch["pixel_values_videos"][:16])
        independent = network.generate_text(first, max_new_tokens=2, eos_token_id=-1, return_logits=True)
    torch.testing.assert_close(output["sequences"][:1], independent["sequences"])
    torch.testing.assert_close(output["logits"][:1], independent["logits"], rtol=2e-4, atol=3e-6)


@pytest.mark.parametrize("guidance", [1., 3.])
def test_t2v_sampling_uses_one_backbone_pass_and_discards_target_pixels(device, guidance):
    torch.manual_seed(115)
    network = model(device, freeze=True).train()
    batch = video_batch(device)
    noise = [torch.randn(4, 2, 4, 4, device=device)]
    calls = []
    handle = network.backbone.register_forward_pre_hook(lambda module, args: calls.append(args[0]["input_ids"].shape))
    try:
        actual = network.generate_video(batch, initial_noise=noise, steps=2, cfg=guidance)
        assert len(calls) == 1
        assert calls[0][0] == (1 if guidance == 1. else 2)
        changed = {**batch, "pixel_values_videos": torch.randn_like(batch["pixel_values_videos"]) * 7}
        other = network.generate_video(changed, initial_noise=noise, steps=2, cfg=guidance)
        assert len(calls) == 2
    finally:
        handle.remove()
    assert network.training and not network.vision_encoder.training
    assert actual[0].shape == noise[0].shape and torch.isfinite(actual[0]).all()
    torch.testing.assert_close(actual[0], other[0], rtol=0, atol=0)


def test_config_factory_and_no_runtime_reference_imports(tmp_path):
    import subprocess
    import sys
    from models import UMM, UMMConfig
    cfg = configuration(layers=4)
    network = UMM(UMMConfig.from_dict(cfg.to_dict()))
    assert isinstance(network, UMM)
    restored = UMM(cfg.to_dict())
    restored.load_state_dict(network.state_dict(), strict=True)
    script = '''
import importlib.abc, sys
class RejectReferences(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'models.references' or fullname.startswith('models.references.'):
            raise AssertionError('Runtime imported reference: ' + fullname)
sys.meta_path.insert(0, RejectReferences())
from models import UMM, UMMConfig
from models.backbone_qwen35 import Qwen35CQBackbone
from models.vision_encoder import Qwen35VideoEncoder
from models.generator_wan21 import Wan21Generator
assert not any(x.startswith('models.references') for x in sys.modules)
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
