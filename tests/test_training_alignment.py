"""Source regression checks for precision, freezing and validation protocols."""
import ast
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
import yaml

from configs.config import merge_config
from data.dataloader import build_dataloaders, build_validation_loader
from data.manifest import image_validation_indices
from evaluations.validation import validate_losses
from models.modeling_umm import UMM
from training.loss import prepare_batch, prepare_flow
from training.optimizer import apply_trainable_scope, build_optimizer_and_scheduler
from test_data import Tokenizer, assert_batch_equal, config as data_config
from test_model_core import batch, config as tiny_config, reference, activate_head
from test_training import settings, tasks


def source_function(relative_path, name):
    root = os.environ.get("UMM_SOURCE_SNAPSHOT")
    if not root:
        pytest.skip("set UMM_SOURCE_SNAPSHOT for direct source protocol comparisons")
    path = Path(root) / relative_path
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(defaultdict=defaultdict, random=random, math=math)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.float16])
def test_content_jitter_matches_actual_source_before_projection(weight_dtype, cache_dtype):
    config = tiny_config(image_input_noise_strength=.01)
    model = UMM(config).to(dtype=weight_dtype).train()
    original = reference.Qwen3Model(SimpleNamespace(**config.to_dict(),
        layer_types=["full_attention"] * 2, attention_dropout=0, max_position_embeddings=128)).to(dtype=weight_dtype).train()
    original.load_state_dict(model.backbone.state_dict())
    inputs = batch()
    inputs["image_latents"] = inputs["image_latents"].to(cache_dtype)
    clean = inputs["image_latents"].clone()
    cfg = {"model": {**config.to_dict(), "image_uncond_prob": .25, "image_flow_batch_mul": 2}}
    torch.manual_seed(31)
    expected_uncond = torch.rand(1) < .25
    expected = original._build_x0_inputs_embeds(inputs["input_ids"], inputs["token_types"],
        inputs["image_latents"], inputs["image_latent_mask"], image_spans_present=True)
    expected_flow = prepare_flow(inputs, cfg["model"])
    expected_rng = torch.get_rng_state()
    torch.manual_seed(31)
    prepared = prepare_batch(inputs, "cpu", cfg, image_input_dtype=weight_dtype)
    actual = model.backbone.embed_content(prepared)
    actual_flow = prepare_flow(prepared, cfg["model"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(prepared["image_latents"], clean, rtol=0, atol=0)
    torch.testing.assert_close(prepared["image_uncond_rows"], expected_uncond, rtol=0, atol=0)
    for a, b in zip(actual_flow, expected_flow):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert prepared["content_image_latents"].dtype == weight_dtype
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("scope", ["full", "generator"])
def test_trainable_scope_and_actual_update_match_source(tmp_path, scope):
    source_scope = source_function("pretrain/train_selfless_flow.py", "_apply_trainable_scope")
    model = UMM(tiny_config())
    activate_head(model.generator)
    original = torch.nn.Module()
    original.config = model.config
    original.model = deepcopy(model.backbone)
    original.image_flow_condition_proj = deepcopy(model.condition_projector)
    original.image_flow_head = deepcopy(model.generator)
    original.lm_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
    original.lm_head.weight = original.model.embed_tokens.weight
    source_scope(original, SimpleNamespace(training={"trainable_scope": "full" if scope == "full" else "flow_head_only"}))
    def rename(name):
        for old, new in (("model.", "backbone."), ("image_flow_condition_proj.", "condition_projector."), ("image_flow_head.", "generator.")):
            if name.startswith(old):
                return new + name[len(old):]
        return name
    expected = {rename(name) for name, p in original.named_parameters() if p.requires_grad}
    apply_trainable_scope(model, scope)
    assert {name for name, p in model.named_parameters() if p.requires_grad} == expected
    optimizer, _ = build_optimizer_and_scheduler(model, settings(tmp_path))
    assert {name for group in optimizer.param_groups for name in group["param_names"]} == expected
    if scope == "generator":
        before = {name: p.clone() for name, p in model.named_parameters()}
        from training.loss import forward_loss
        forward_loss(model, tasks()[1], settings(tmp_path))["loss"].backward()
        optimizer.step()
        assert all(p.grad is None and torch.equal(p, before[name])
                   for name, p in model.named_parameters() if name.startswith("backbone."))
        assert not torch.equal(model.generator.final_layer.linear.weight, before["generator.final_layer.linear.weight"])
        assert not torch.equal(model.condition_projector.weight, before["condition_projector.weight"])


def test_subset_matches_source_order_with_multidigit_class_indices_and_reordered_cache(tmp_path):
    sample = source_function("utils/training_downstream_validation.py", "stratified_sample")
    classes = [{"synset": f"n{i:08d}", "class_index": i} for i in range(12)]
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"classes": classes}))
    records = [SimpleNamespace(class_index=cls["class_index"], synset=cls["synset"],
        img_id=100 * cls["class_index"] + i, image_id=f"val/img_{cls['class_index']}_{i:03d}")
        for cls in classes for i in range(5)]
    ids = [row.img_id for row in reversed(records)]
    dataset = SimpleNamespace(dataset_split="val", img_ids=ids,
        synsets={row.img_id: row.synset for row in records},
        source_paths_full={row.img_id: f"/{row.image_id}.JPEG" for row in records})
    # Provide only the dataset protocol used by the subset resolver.
    class Cache:
        def __len__(self):
            return len(self.img_ids)
    cache = Cache()
    cache.__dict__.update(vars(dataset))
    expected = sample(records, 24, category=lambda r: r.class_index, identity=lambda r: r.image_id, seed=424242)
    indices = image_validation_indices(cache, classes_path=path)
    assert [ids[index] for index in indices] == [row.img_id for row in expected]
    missing = expected[0].img_id
    cache.img_ids = [img_id for img_id in ids if img_id != missing]
    with pytest.raises(ValueError, match="missing"):
        image_validation_indices(cache, classes_path=path)


@pytest.mark.parametrize("packed", [False, True])
def test_training_and_standalone_share_subset_but_generation_keeps_full_split(tmp_path, packed):
    cfg = data_config(tmp_path)
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps({"classes": [{"synset": "n00000001", "class_index": 0}]}))
    cfg["evaluation"] = dict(seed=424242, loss_image_per_class=2, image_classes=str(classes))
    cfg["data"]["schedule"] = ["t2i", "i2t"]
    cfg["data"]["image"]["packing"] = dict(enabled=packed, nominal_capacity=64)
    _, training_val = build_dataloaders(cfg, Tokenizer())
    standalone = build_validation_loader(cfg, Tokenizer())
    full = build_validation_loader(cfg, Tokenizer(), loss_protocol=False)
    for task in cfg["data"]["schedule"]:
        first, second = training_val.loaders[task], standalone.loaders[task]
        assert first.dataset.indices == second.dataset.indices
        assert len(first.dataset) == 2 and len(full.loaders[task].dataset) == 3
    for actual, expected in zip(training_val, standalone):
        assert_batch_equal(actual, expected)


def test_validation_random_draws_follow_source_task_offsets_not_iteration_order():
    model = UMM(tiny_config())
    batches = [dict(task=task, index=i) for task in ("t2i", "i2t", "text") for i in range(2)]
    seen = {}
    def evaluate(model, batch):
        key = batch["task"], batch["index"]
        values = torch.rand(4)
        seen[key] = values
        return dict(loss=values.mean(), raw_loss=values.mean(), target_count=4)
    validate_losses(model, batches, evaluate, schedule=["text", "t2i", "text", "i2t"])
    for (task, index), actual in seen.items():
        offset = {"text": 0, "t2i": 1000003, "i2t": 2000006}[task]
        expected = torch.rand(4, generator=torch.Generator().manual_seed(424242 + offset + index))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected = dict(seen)
    validate_losses(model, batches[4:] + batches[:4], evaluate, schedule=["text", "t2i", "text", "i2t"])
    for key in expected:
        torch.testing.assert_close(seen[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("prepared_dtype", [torch.float32, torch.bfloat16])
def test_real_training_entrypoint_builds_ema_after_backend_conversion_and_resumes(tmp_path, monkeypatch, prepared_dtype):
    import accelerate.utils
    from training import train
    from test_cli import local_tokenizer
    local_tokenizer(tmp_path / "tokenizer")
    cfg = merge_config(settings(tmp_path / "output"), data_config(tmp_path / "dataset"))
    cfg["model"].update(vocab_size=256, tokenizer_path=str(tmp_path / "tokenizer"),
                        image_mask_token_id=6, eos_token_id=2, pad_token_id=0, from_scratch=True)
    cfg["data"]["schedule"] = ["t2i"]
    cfg["training"].update(max_train_steps=1, gradient_accumulation_steps=1, use_ema=True,
                           ema_shard_chunk_numel=8192)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    build, seed = train.build_accelerator, accelerate.utils.set_seed
    runtimes, seed_policies = [], []
    def prepare_runtime(config):
        runtime = build(config)
        prepare = runtime.prepare
        def convert(*args):
            model, optimizer, scheduler = prepare(*args)
            return model.to(dtype=prepared_dtype), optimizer, scheduler
        runtime.prepare = convert
        runtimes.append(runtime)
        return runtime
    def record_seed(value, device_specific=False):
        seed_policies.append(device_specific)
        return seed(value, device_specific=device_specific)
    monkeypatch.setattr(train, "build_accelerator", prepare_runtime)
    monkeypatch.setattr(accelerate.utils, "set_seed", record_seed)
    train.main(["--config", str(path)])
    output = tmp_path / "output"
    manifest = json.loads((output / "checkpoint-1/ema/ema_manifest.json").read_text())
    assert {item["source_dtype"] for item in manifest["tensors"].values()} == {str(prepared_dtype).removeprefix("torch.")}
    assert {item["ema_dtype"] for item in manifest["tensors"].values()} == {"float32"}
    saved = torch.load(output / "weights-ema.pt", weights_only=True)["model"]
    runtimes[-1].free_memory()
    train.main(["--config", str(path), "--resume", str(output / "checkpoint-1")])
    restored = torch.load(output / "weights-ema.pt", weights_only=True)["model"]
    for key in saved:
        torch.testing.assert_close(restored[key], saved[key], rtol=0, atol=0)
    assert seed_policies == [True, True]
    runtimes[-1].free_memory()
