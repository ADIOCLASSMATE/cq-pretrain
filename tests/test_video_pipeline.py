from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from configs.config import DEFAULTS, load_config, save_config, validate_config
from data.dataset import (
    VideoDataset,
    read_records,
    prepare_raw_video,
    qwen_video_inputs,
    decode_video,
)
from data.sequence import make_video_sequence, native_video_metadata
from data.packing import collate_video_samples
from data.dataloader import build_dataloaders
from data.cache import load_cache
from training.loss import prepare_batch, forward_loss
from training.optimizer import (
    apply_trainable_scope,
    build_optimizer_and_scheduler,
    parameter_role,
)
from training.train import load_processor
from tests.video_fixtures import make_fixture


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(2)
    return make_fixture(tmp_path)


def test_legacy_query_mask_names_keep_ids_after_processor_roundtrip(fixture, tmp_path):
    from transformers import AutoProcessor

    config, processor, _, _, _ = fixture
    tokenizer = processor.tokenizer
    tokenizer.mask_token = "<|text_mask|>"
    vocab = tokenizer.get_vocab()
    legacy = tmp_path / "legacy_processor"
    processor.save_pretrained(legacy)
    # Simulate an existing processor, including its HF special-token metadata.
    originals = {}
    for path in legacy.glob("*.json"):
        text = path.read_text()
        text = text.replace("<|text_mask|>", "<|mdm_mask|>")
        text = text.replace("<|vision_mask|>", "<|video_mask|>")
        path.write_text(text)
        originals[path] = text
    config["model"]["processor_path"] = str(legacy)
    renamed = load_processor(config)
    assert config["model"]["mask_token_id"] == 59
    assert config["model"]["video_mask_token_id"] == 60
    assert renamed.tokenizer.get_vocab() == vocab
    assert renamed.tokenizer.mask_token == "<|text_mask|>"
    assert renamed.tokenizer.encode(
        "<|text_mask|><|vision_mask|>", add_special_tokens=False
    ) == [59, 60]
    assert renamed.tokenizer.decode([59, 60], skip_special_tokens=True) == ""
    assert all(path.read_text() == text for path, text in originals.items())

    saved = tmp_path / "renamed_processor"
    renamed.save_pretrained(saved)
    restored = AutoProcessor.from_pretrained(saved, local_files_only=True)
    assert restored.tokenizer.get_vocab() == vocab
    assert restored.tokenizer.mask_token_id == 59
    assert restored.tokenizer.convert_tokens_to_ids("<|vision_mask|>") == 60
    for path in saved.glob("*.json"):
        assert "<|mdm_mask|>" not in path.read_text()
        assert "<|video_mask|>" not in path.read_text()


def test_conflicting_query_mask_names_are_rejected(fixture, tmp_path):
    config, processor, _, _, _ = fixture
    processor.tokenizer.add_tokens(["<|mdm_mask|>"], special_tokens=True)
    path = tmp_path / "conflicting_processor"
    processor.save_pretrained(path)
    config["model"]["processor_path"] = str(path)
    with pytest.raises(ValueError, match="both legacy and renamed"):
        load_processor(config)


def test_native_processor_parity_and_same_position_labels(fixture):
    config, p, model, _, _ = fixture
    row = read_records(config["data"]["sources"][0])[0]
    rgb, clip = prepare_raw_video(row["videos"][0], config["data"]["video"])
    pixels, grid, times = qwen_video_inputs(rgb, clip, p, config["data"]["video"])
    text = p.apply_chat_template(row["messages"], tokenize=False, enable_thinking=False)
    native = p(
        text=[text],
        videos=[rgb],
        video_metadata=[native_video_metadata(times)],
        do_sample_frames=False,
        return_tensors="pt",
    )
    actual = make_video_sequence(row["messages"], p, grid, times)
    torch.testing.assert_close(
        actual["input_ids"], native["input_ids"][0], rtol=0, atol=0
    )
    torch.testing.assert_close(pixels, native["pixel_values_videos"], rtol=0, atol=0)
    visual = actual["input_ids"].eq(p.video_token_id)
    assert bool(actual["labels"][visual].eq(-100).all())
    torch.testing.assert_close(actual["labels"][~visual], actual["input_ids"][~visual])
    batch = collate_video_samples(
        [VideoDataset(config["data"]["sources"][0], p, config)[0]]
    )
    prepared = model.prepare_inputs(batch)
    assert prepared["inputs_embeds"].shape[:2] == batch["input_ids"].shape
    slots = prepared["video_token_indices"][0]
    assert prepared["query_read_end"][0, slots].unique().tolist() == [int(slots[0])]


@pytest.mark.parametrize("qwen_mode", ["cache", "online"])
@pytest.mark.parametrize("wan_mode", ["cache", "online"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_all_cache_combinations_align(fixture, qwen_mode, wan_mode, batch_size):
    config, p, model, _, latent = fixture
    source = config["data"]["sources"][0]
    cached = VideoDataset(source, p, config)[0]
    config = deepcopy(config)
    config["data"]["representations"] = dict(qwen=qwen_mode, wan=wan_mode)
    actual = VideoDataset(source, p, config)[0]
    assert actual["clip"] == cached["clip"]
    torch.testing.assert_close(actual["input_ids"], cached["input_ids"], atol=0, rtol=0)

    class Codec:
        def encode(self, rgb):
            assert rgb.shape == (3, 5, 16, 16)
            return latent

    old = prepare_batch(collate_video_samples([cached] * batch_size), "cpu", config)
    new = prepare_batch(
        collate_video_samples([actual] * batch_size), "cpu", config, codec=Codec()
    )
    assert "pixel_video_ids" not in new
    with torch.no_grad():
        a = model.prepare_inputs(old)
        b = model.prepare_inputs(new)
    # Caches were encoded one video at a time. Batched ViT matmuls may round
    # differently; the unchanged singleton path still requires exact equality.
    tolerance = dict(rtol=1e-5, atol=1e-7) if batch_size > 1 else dict(rtol=0, atol=0)
    torch.testing.assert_close(a["inputs_embeds"], b["inputs_embeds"], **tolerance)
    torch.testing.assert_close(
        old["wan_latents"][0], new["wan_latents"][0], rtol=0, atol=0
    )


@pytest.mark.parametrize("qwen_mode", ["cache", "online"])
def test_uniform_representations_mixed_tasks_and_gradients(fixture, qwen_mode):
    config, p, model, _, _ = fixture
    source = config["data"]["sources"][0]
    config["data"]["representations"]["qwen"] = qwen_mode
    data = VideoDataset(source, p, config)
    samples = [data[0], data[2], data[1]]
    if qwen_mode == "online":
        samples[1]["pixel_values_videos"] = samples[1]["pixel_values_videos"] * -2
        samples[2]["pixel_values_videos"] = samples[2]["pixel_values_videos"] + 1
    batch = collate_video_samples(samples)
    assert batch["target_video_ids"].tolist() == [0, 2]
    assert "pixel_video_ids" not in batch
    before = deepcopy(batch)
    with torch.no_grad():
        prepared = model.prepare_inputs(batch)
        for i in range(len(samples)):
            expected = samples[i]["video_features"]
            if qwen_mode == "online":
                expected = model.vision_encoder(
                    samples[i]["pixel_values_videos"],
                    samples[i]["video_grid_thw"][None],
                )[0]
            torch.testing.assert_close(
                prepared["inputs_embeds"][i, prepared["video_token_indices"][i]],
                expected,
            )
    apply_trainable_scope(model, config["model"])
    model.train()
    torch.manual_seed(7)
    result = forward_loss(model, prepare_batch(batch, "cpu", config), config)
    result["loss"].backward()
    assert result["video_target_count"] == 2 and result["text_target_count"] > 0
    assert model.generator.condition_proj.weight.grad.abs().sum() > 0
    assert model.backbone.embed_tokens.weight.grad.abs().sum() > 0
    assert all(
        x.grad is None and not x.requires_grad
        for x in model.vision_encoder.parameters()
    )
    if qwen_mode == "cache":
        for old, new in zip(before["video_features"], batch["video_features"]):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
    else:
        assert "video_features" not in batch
        torch.testing.assert_close(
            before["pixel_values_videos"], batch["pixel_values_videos"], atol=0, rtol=0
        )
    optimizer, _ = build_optimizer_and_scheduler(model, config)
    assert set(g["role"] for g in optimizer.param_groups) == {
        "backbone",
        "embedding",
        "projector",
        "generator",
    }
    assert parameter_role("generator.condition_proj.weight") == "projector"


@pytest.mark.parametrize("branch", ["qwen", "wan"])
def test_batch_rejects_mixed_cache_and_online_even_with_prefer_cache(fixture, branch):
    config, p, _, _, _ = fixture
    config["data"]["representations"][branch] = "prefer_cache"
    records = read_records(config["data"]["sources"][0])
    records[1]["videos"][0].pop(f"{branch}_cache")
    data = VideoDataset(config["data"]["sources"][0], p, config, records=records)
    with pytest.raises(ValueError, match=f"{branch.title()} batch cannot mix"):
        collate_video_samples([data[0], data[1]])


def test_prepare_batch_rejects_mixed_wan_without_collation(fixture):
    config, _, _, _, latent = fixture
    batch = dict(
        target_video_ids=torch.tensor([0, 1]),
        wan_latents=[latent, None],
        wan_rgb=[None, torch.zeros(3, 5, 16, 16)],
    )
    with pytest.raises(ValueError, match="Wan batch cannot mix"):
        prepare_batch(batch, "cpu", config)


def test_unfreeze_vision_requires_online_and_preserves_gradient(fixture):
    config, p, model, _, _ = fixture
    config["model"]["freeze_vision"] = False
    with pytest.raises(ValueError, match="Unfrozen"):
        validate_config(config)
    config["data"]["representations"]["qwen"] = "online"
    validate_config(config)
    model.config.freeze_vision = False
    apply_trainable_scope(model, config["model"])
    model.train()
    batch = collate_video_samples(
        [VideoDataset(config["data"]["sources"][0], p, config)[2]]
    )
    forward_loss(model, prepare_batch(batch, "cpu", config), config)["loss"].backward()
    assert any(
        x.grad is not None and x.grad.abs().sum() > 0
        for x in model.vision_encoder.parameters()
    )


def test_preprocessed_video_no_leak_and_generation_alignment(fixture):
    config, p, model, _, _ = fixture
    data = VideoDataset(config["data"]["sources"][0], p, config)
    batch = collate_video_samples([data[0]])
    other = deepcopy(batch)
    other["video_features"][0] = torch.randn_like(other["video_features"][0]) * 100
    prepared = model.prepare_inputs(batch)
    slots = prepared["video_token_indices"][0]
    first = int(slots[0])
    # Change all content, including frame delimiters/timestamps, after the first
    # visual slot at the prepared embedding boundary; leave geometry fixed.
    perturbed = dict(prepared)
    perturbed["inputs_embeds"] = prepared["inputs_embeds"].clone()
    perturbed["inputs_embeds"][:, first:] = (
        torch.randn_like(perturbed["inputs_embeds"][:, first:]) * 10
    )
    with torch.no_grad():
        a = model.backbone(prepared, return_content=True)
        b = model.backbone(perturbed, return_content=True)
        torch.testing.assert_close(
            a["query_hidden"][0, slots], b["query_hidden"][0, slots], rtol=0, atol=0
        )
        assert not torch.equal(a["content_hidden"], b["content_hidden"])
        condition, _ = model.encode_condition(batch)
        other_condition, _ = model.encode_condition(other)
        torch.testing.assert_close(condition, other_condition, atol=0, rtol=0)
        blank = deepcopy(batch)
        blank["video_features"] = [None]
        blank["video_observed_mask"].fill_(False)
        inference_condition, _ = model.encode_condition(blank)
        torch.testing.assert_close(condition, inference_condition, atol=0, rtol=0)
        initial = [torch.randn(16, 2, 2, 2)]
        trained = model.generate_video(batch, initial_noise=initial, steps=2, cfg=1.0)
        inferred = model.generate_video(blank, initial_noise=initial, steps=2, cfg=1.0)
        torch.testing.assert_close(trained[0], inferred[0], atol=0, rtol=0)
    # V2T answer loss is sensitive to the observed video, unlike T2V conditions.
    v2t = collate_video_samples([data[2]])
    changed = deepcopy(v2t)
    changed["video_features"][0] = other["video_features"][0]
    with torch.no_grad():
        assert not torch.equal(model(v2t)["text_logits"], model(changed)["text_logits"])


@pytest.mark.parametrize("mixing", ["balanced", "preserve"])
def test_loader_exact_consumed_resume_and_task_mixing(fixture, mixing):
    config, p, _, _, _ = fixture
    config["data"]["mixing"] = mixing
    validate_config(config)
    loader, _ = build_dataloaders(config, p)
    it = iter(loader)
    batch = next(it)
    loader.mark_consumed(batch)
    state = loader.state_dict()
    expected = next(it)  # prefetched but uncommitted
    assert loader.state_dict() == state
    restored, _ = build_dataloaders(config, p)
    restored.load_state_dict(state)
    actual = next(iter(restored))
    assert (
        actual["sample_ids"] == expected["sample_ids"]
        and actual["task"] == expected["task"]
    )
    torch.testing.assert_close(
        actual["input_ids"], expected["input_ids"], atol=0, rtol=0
    )
    if mixing == "balanced":
        assert (batch["task"], actual["task"]) == ("t2v", "v2t")
    else:
        assert actual["task"] == "mixed"


def test_cache_contracts_and_single_video_errors(fixture, tmp_path):
    config, p, _, rows, _ = fixture
    with pytest.raises(ValueError, match="checkpoint"):
        load_cache(tmp_path / "qwen.pt", "qwen", checkpoint="wrong")
    payload = torch.load(tmp_path / "wan.pt", weights_only=True)
    payload["metadata"]["clip"]["frame_indices"][1] = 0
    torch.save(payload, tmp_path / "badwan.pt")
    rows[0]["videos"][0]["wan_cache"] = str(tmp_path / "badwan.pt")
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="different clips"):
        VideoDataset({"path": str(manifest)}, p, config)[0]
    rows[0]["videos"].append(deepcopy(rows[0]["videos"][0]))
    manifest.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="exactly one video"):
        read_records({"path": str(manifest)})


def test_training_cli_export_and_resume(fixture, tmp_path):
    config, _, _, _, _ = fixture
    path = tmp_path / "run.yaml"
    save_config(config, path)
    command = [sys.executable, "-m", "training.train", "--config", str(path)]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
    )
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    output = Path(config["training"]["output_dir"])
    assert (
        output / "checkpoints" / "checkpoint-2" / "checkpoint_complete.json"
    ).is_file()
    assert (output / "checkpoints/exports/processor" / "tokenizer.json").is_file()
    metrics = [
        json.loads(x)
        for x in next((output / "logs/train").glob("attempt_*/metrics.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert metrics[-1]["step"] == 2 and "train/loss_t2v_video" in metrics[-1]
    resumed_dir = tmp_path / "resumed"
    alternate = command + ["--set", f"training.output_dir={resumed_dir}"]
    stopped = subprocess.run(
        alternate + ["--stop-after-steps", "1"],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    restored = subprocess.run(
        alternate
        + [
            "--auto-resume",
            "--set",
            "training.log_every=10",
            "--set",
            "training.keep_checkpoints=1",
            "--set",
            f"training.log_dir={tmp_path / 'separate-logs'}",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert restored.returncode == 0, restored.stdout + restored.stderr
    expected = torch.load(
        output / "checkpoints" / "exports" / "weights.pt", weights_only=True
    )["model"]
    actual = torch.load(
        resumed_dir / "checkpoints" / "exports" / "weights.pt", weights_only=True
    )["model"]
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    # Exact training-state comparison, including optimizer/scheduler, EMA and consumed data.
    checkpoint = resumed_dir / "checkpoints" / "checkpoint-2"
    assert not (resumed_dir / "checkpoints" / "checkpoint-1").exists()
    assert (resumed_dir / "logs").is_dir()
    attempt = next((tmp_path / "separate-logs/train").glob("attempt_*"))
    info = json.loads((attempt / "run.json").read_text())
    assert info["start_step"] == 1 and info["management_changes"]
    assert json.loads((attempt / "summary.json").read_text())["status"] == "completed"
    metrics = [
        json.loads(line)
        for line in (attempt / "metrics.jsonl").read_text().splitlines()
    ]
    assert (
        metrics[-1]["optimizer_updates"] == 1
    )  # trailing window is flushed despite log_every=10

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    reference = output / "checkpoints" / "checkpoint-2"
    for filename in ("optimizer.bin", "scheduler.bin"):
        equal(
            torch.load(reference / "accelerator" / filename, weights_only=False),
            torch.load(checkpoint / "accelerator" / filename, weights_only=False),
        )
    original_runtime = torch.load(
        reference / "runtime_rank_00000.pt", weights_only=False
    )
    resumed_runtime = torch.load(
        checkpoint / "runtime_rank_00000.pt", weights_only=False
    )
    equal(original_runtime["data_state"], resumed_runtime["data_state"])
    equal(original_runtime["rng"]["torch"], resumed_runtime["rng"]["torch"])
    equal(original_runtime["rng"]["python"], resumed_runtime["rng"]["python"])
    from training.train import load_model

    config["model"]["checkpoint"] = str(
        output / "checkpoints" / "exports" / "weights.pt"
    )
    load_processor(config)
    loaded = load_model(config)
    assert loaded.lm_head.weight is loaded.backbone.embed_tokens.weight
    assert not loaded.generator.gradient_checkpointing
    inference = [
        sys.executable,
        "-m",
        "scripts.generate",
        "--config",
        str(path),
        "--checkpoint",
        str(output / "checkpoints" / "exports" / "weights.pt"),
    ]
    generated = subprocess.run(
        inference
        + [
            "--task",
            "t2v",
            "--prompt",
            "blue block moves",
            "--output",
            str(tmp_path / "generated.pt"),
            "--set",
            "sampling.steps=2",
            "--set",
            "sampling.cfg=1.0",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert torch.load(tmp_path / "generated.pt", weights_only=True)["latent"].shape == (
        16,
        2,
        2,
        2,
    )
    manifest = tmp_path / "v2t.jsonl"
    manifest.write_text(json.dumps(fixture[3][2]) + "\n")
    generated = subprocess.run(
        inference
        + [
            "--task",
            "v2t",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "answer.json"),
            "--set",
            "sampling.max_new_tokens=2",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert isinstance(json.loads((tmp_path / "answer.json").read_text())["text"], str)
    validation_override = "data.validation_sources=" + json.dumps(
        config["data"]["sources"]
    )
    evaluated = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluations.eval",
            "--config",
            str(path),
            "--checkpoint",
            str(output / "checkpoints" / "exports" / "weights.pt"),
            "--set",
            validation_override,
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    assert json.loads(evaluated.stdout)["val/t2v_video_targets"] == 2


def test_cli_precision_override_and_legacy_config_rejection(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    assert load_config(path)["distributed"]["backend"] == "mindspeed"
    assert (
        load_config(path, ["distributed.backend=accelerate", "training.device=cpu",
                           "training.mixed_precision=no"])["training"][
            "mixed_precision"
        ]
        == "no"
    )
    with pytest.raises(ValueError, match="legacy"):
        load_config(path, ["model.backbone_type=cq_it"])


def test_raw_mp4_preserves_presentation_times(tmp_path):
    from scripts.generate import write_video

    video = torch.rand(3, 5, 16, 16) * 2 - 1
    path = tmp_path / "clip.mp4"
    write_video(path, video, 4.0)
    frames, times, _ = decode_video({"path": str(path)})
    assert frames.shape == (5, 16, 16, 3)
    torch.testing.assert_close(
        torch.tensor(times), torch.arange(5, dtype=torch.float64) / 4, rtol=0, atol=0
    )


def test_shell_auto_resume_failure_preserves_metadata_and_cleans_only_checkpoints(
    fixture, tmp_path
):
    import hashlib

    config, _, _, _, _ = fixture
    config["training"].update(
        output_dir=None,
        output_root=str(tmp_path / "experiments"),
        exp_id="isolated",
        save_every=1,
        keep_checkpoints=1,
        use_ema=True,
    )
    config_path = tmp_path / "run.yaml"
    save_config(config, config_path)
    command = ["bash", "scripts/train.sh", "--config", str(config_path)]
    env = dict(
        os.environ,
        PYTHON_BIN=sys.executable,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
    )
    env.pop("ACCELERATE_CONFIG", None)
    output = tmp_path / "experiments/isolated"

    def run(extra, success=True):
        result = subprocess.run(
            command + extra, env=env, text=True, capture_output=True, timeout=120
        )
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result

    run(["--auto-resume", "--stop-after-steps", "1"])
    saved_metadata = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [output / "config.yaml", *list((output / "checkpoints/exports/processor").glob("*"))]
        if p.is_file()
    }
    run(["--auto-resume", "--set", "optimizer.backbone_lr=0.2"], success=False)
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
        for p, digest in saved_metadata.items()
    )
    failed = [
        p
        for p in (output / "logs/train").glob("attempt_*")
        if json.loads((p / "summary.json").read_text())["status"] == "failed"
    ]
    assert len(failed) == 1
    assert "optimizer.backbone_lr" in (failed[0] / "console.log").read_text()
    assert json.loads((failed[0] / "launch_result.json").read_text())["exit_code"] != 0
    # Simulate an incomplete latest write. Recovery quarantines it only after validation succeeds.
    broken = output / "checkpoints/checkpoint-2"
    broken.mkdir()
    (broken / "partial").write_text("interrupted write")
    run(["--auto-resume", "--set", "training.log_every=5"])
    assert not (output / "checkpoints/checkpoint-1").exists()
    assert (output / "checkpoints/checkpoint-2/checkpoint_complete.json").is_file()
    assert list((output / "checkpoints/quarantine").glob("checkpoint-2-*/partial"))
    assert (output / "checkpoints/exports/weights-ema.pt").is_file()
    attempts = list((output / "logs/train").glob("attempt_*"))
    assert len(attempts) == 3 and all((p / "console.log").is_file() for p in attempts)


def test_two_rank_mixed_training_and_checkpoint(fixture, tmp_path):
    config, _, _, _, _ = fixture
    config["data"]["mixing"] = "preserve"
    config["flow"]["repeats"] = 2
    config["training"].update(max_train_steps=1, use_ema=True, ema_decay=0.9)
    validate_config(config)
    path = tmp_path / "distributed.yaml"
    save_config(config, path)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        GLOO_SOCKET_IFNAME="lo",
        TOKENIZERS_PARALLELISM="false",
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "training.train",
        "--config",
        str(path),
    ]
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    output = Path(config["training"]["output_dir"])
    meta = json.loads(
        (output / "checkpoints" / "checkpoint-1" / "metadata.json").read_text()
    )
    assert (
        meta["world_size"] == 2
        and (output / "checkpoints" / "exports" / "weights-ema.pt").is_file()
    )
    # Resume on the same topology verifies per-rank committed cursors and EMA.
    run = subprocess.run(
        command + ["--resume", str(output / "checkpoints" / "checkpoint-1")],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
