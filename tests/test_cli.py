"""Public entrypoints against local tokenizer, cache and real tiny UMM weights."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from test_data import config as data_config
from test_training import settings


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments, expect_success=True):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
                            env=environment, capture_output=True, text=True, timeout=120)
    if expect_success:
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return result


def local_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tokens = ["<pad>", "<unk>", "<eos>", "<|boi|>", "<|eoi|>", "<|mdm_mask|>", "<|img_mask|>"]
    tokens += ["first", "second", "detailed", "caption", "example", "sample"]
    tokens += [f"word{i}" for i in range(100)]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
        pad_token="<pad>", eos_token="<eos>", mask_token="<|mdm_mask|>",
        additional_special_tokens=["<|boi|>", "<|eoi|>", "<|img_mask|>"])
    fast.save_pretrained(path)
    return fast


@pytest.fixture(scope="module")
def local_run(tmp_path_factory):
    from configs.config import merge_config
    directory = tmp_path_factory.mktemp("cli-runtime")
    local_tokenizer(directory / "tokenizer")
    config = merge_config(settings(directory / "output"), data_config(directory / "dataset"))
    config["model"].update(vocab_size=256, tokenizer_path=str(directory / "tokenizer"),
        from_scratch=True, image_mask_token_id=6, eos_token_id=2, pad_token_id=0,
        image_input_noise_strength=.01, image_uncond_prob=.2)
    config["training"].update(max_train_steps=1, use_ema=True, ema_shard_chunk_numel=8192,
                              val_every=1, save_every=1)
    config["optimizer"].update(warmup_steps=0, decay_steps=0)
    probe = directory / "validation.jsonl"
    probe.write_text(json.dumps({"text": "word1 word2 word3 word4"}) + "\n")
    config["data"]["text"]["validation"] = {"jsonl": str(probe), "sequence_length": 8, "batch_size": 1}
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return directory, path, config


@pytest.fixture(scope="module")
def trained_run(local_run):
    directory, path, config = local_run
    run_cli("-m", "training.train", "--config", path)
    return directory, path, config


def test_train_dry_run_validates_without_opening_model_or_data(local_run):
    directory, path, _ = local_run
    result = run_cli("-m", "training.train", "--config", path, "--dry-run",
                     "--set", "model.tokenizer_path=/missing/tokenizer")
    assert json.loads(result.stdout)["model"]["tokenizer_path"] == "/missing/tokenizer"


def test_training_entrypoint_runs_mixed_window_validation_and_exports(trained_run):
    directory, _, _ = trained_run
    output = directory / "output"
    for name in ("config.yaml", "tokenizer/tokenizer.json", "checkpoint-1/checkpoint_complete.json",
                 "checkpoint-1/ema/ema_manifest.json", "weights.pt", "weights-ema.pt"):
        assert (output / name).is_file(), name
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert any("train/loss_t2i" in row and "train/loss_text" in row and "train/loss_i2t" in row for row in rows)
    assert any("val/loss_t2i" in row and "val/loss_text" in row and "val/loss_i2t" in row for row in rows)
    model = torch.load(output / "weights.pt", weights_only=True)
    ema = torch.load(output / "weights-ema.pt", weights_only=True)
    assert model["weights"] == "raw" and ema["weights"] == "ema"
    assert set(model["model"]) == set(ema["model"])


def test_joint_training_loss_evaluation_and_i2t_probe_entrypoints(local_run):
    directory, _, original = local_run
    config = deepcopy(original)
    config["data"]["schedule"] = ["joint"]
    config["data"]["sources"]["joint"] = {"micro_batch_size": 4}
    config["training"].update(output_dir=str(directory / "joint-output"), gradient_accumulation_steps=1)
    path = directory / "joint.yaml"
    path.write_text(yaml.safe_dump(config))
    run_cli("-m", "training.train", "--config", path)
    output = directory / "joint-output"
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    training = next(row for row in rows if "train/loss_joint" in row)
    assert training["train/loss_joint_text"] > 0 and training["train/loss_joint_image"] > 0
    assert training["train/loss_joint"] == pytest.approx(
        config["model"]["lambda_text"] * training["train/loss_joint_text"]
        + config["model"]["lambda_image"] * training["train/loss_joint_image"])
    metrics = directory / "joint-evaluation.json"
    run_cli("-m", "evaluations.eval", "--config", path, "--checkpoint", output / "weights.pt",
            "--task", "loss", "--output", metrics)
    evaluated = json.loads(metrics.read_text())["metrics"]
    assert evaluated["val/joint_text_target_tokens"] > 0 and evaluated["val/joint_image_target_tokens"] > 0
    run_cli("-m", "evaluations.eval", "--config", path, "--checkpoint", output / "weights.pt",
            "--task", "perplexity", "--source", "i2t", "--output", metrics)
    assert json.loads(metrics.read_text())["metrics"]["target_tokens"] > 0


def test_evaluation_loss_uses_normal_validation_loader_and_all_tasks(trained_run):
    directory, path, _ = trained_run
    output = directory / "standalone-eval.json"
    run_cli("-m", "evaluations.eval", "--config", path,
            "--checkpoint", directory / "output" / "weights-ema.pt", "--task", "loss", "--output", output)
    report = json.loads(output.read_text())
    assert report["task"] == "loss"
    for task in ("text", "t2i", "i2t"):
        assert report["metrics"][f"val/{task}_target_tokens"] > 0
    # The raw export is from the same optimizer boundary as in-training val.
    run_cli("-m", "evaluations.eval", "--config", path,
            "--checkpoint", directory / "output" / "weights.pt", "--task", "loss", "--output", output)
    standalone = json.loads(output.read_text())["metrics"]
    records = [json.loads(line) for line in (directory / "output" / "metrics.jsonl").read_text().splitlines()]
    training_val = next(row for row in records if "val/loss" in row)
    assert standalone == {key: value for key, value in training_val.items() if key.startswith("val/")}


def test_validation_retains_training_jitter_cfg_and_no_grad_protocol(tmp_path):
    from evaluations.validation import validate_losses
    from models.modeling_umm import UMM
    from training.loss import prepare_batch, forward_loss
    from test_model_core import activate_head, config as tiny_config
    from test_training import tasks
    config = settings(tmp_path, image_uncond_prob=1.0, image_input_noise_strength=.01)
    model = UMM(tiny_config(image_input_noise_strength=.01))
    activate_head(model.generator)
    model.eval()
    batch = tasks()[1]
    target = batch["image_latents"].clone()
    observations = []

    def validation_forward(current_model, raw_batch):
        assert current_model.training and not torch.is_grad_enabled()
        prepared = prepare_batch(raw_batch, torch.device("cpu"), config)
        assert prepared["image_uncond_rows"].all()
        assert not torch.equal(prepared["content_image_latents"], target)
        torch.testing.assert_close(prepared["image_latents"], target, atol=0, rtol=0)
        result = forward_loss(current_model, prepared, config)
        assert not result["loss"].requires_grad
        observations.append(float(result["raw_loss"]))
        return result

    before = torch.get_rng_state().clone()
    first = validate_losses(model, [batch], validation_forward, schedule=["t2i"], seed=17)
    second = validate_losses(model, [batch], validation_forward, schedule=["t2i"], seed=17)
    assert observations[0] == observations[1]
    assert first["val/loss_t2i"] == pytest.approx(observations[0])
    assert first == second and not model.training
    assert torch.equal(torch.get_rng_state(), before)


def test_generate_cli_produces_joint_latents_for_multiple_prompts(trained_run):
    directory, path, config = trained_run
    output = directory / "generated"
    run_cli(ROOT / "scripts/generate.py", "--config", path,
            "--checkpoint", directory / "output/weights-ema.pt", "--output", output,
            "--prompt", "first example", "--prompt", "second detailed example", "--latents-only",
            "--set", "sampling.steps=2")
    result = torch.load(output / "latents.pt", weights_only=True)
    assert result["latents"].shape == (2, config["model"]["image_tokens_per_img"], config["model"]["image_latent_dim"])
    assert torch.isfinite(result["latents"]).all()
    assert result["prompts"] == ["first example", "second detailed example"]
    metadata = json.loads((output / "generation.json").read_text())
    assert metadata["steps"] == 2 and metadata["solver"] == "heun"


def test_convert_checkpoint_cli_migrates_source_names_and_exports_ema(trained_run):
    directory, path, _ = trained_run
    payload = torch.load(directory / "output/weights.pt", weights_only=True)
    source = {}
    for name, tensor in payload["model"].items():
        if name.startswith("backbone."):
            key = "model." + name.removeprefix("backbone.")
        elif name.startswith("condition_projector."):
            key = "image_flow_condition_proj." + name.removeprefix("condition_projector.")
        elif name.startswith("generator."):
            key = "image_flow_head.net." + name.removeprefix("generator.")
        else:
            key = name
        source[key] = tensor
    source_config = deepcopy(payload["config"])
    source_config["model"]["joint_dit_head_type"] = "z2"
    legacy = directory / "source-z2.pt"
    torch.save({"state_dict": source, "config": source_config}, legacy)
    converted = directory / "converted.pt"
    run_cli(ROOT / "scripts/convert_checkpoint.py", "--input", legacy, "--output", converted)
    converted_state = torch.load(converted, weights_only=True)["model"]
    assert set(converted_state) == set(payload["model"])
    for name, value in converted_state.items():
        torch.testing.assert_close(value, payload["model"][name], rtol=0, atol=0)
    sidecar = json.loads(Path(str(converted) + ".json").read_text())
    assert sidecar["training_state_migrated"] is False
    assert sidecar["reference_commit"].startswith("193ec90")
    exported_ema = directory / "converted-ema.pt"
    run_cli(ROOT / "scripts/convert_checkpoint.py", "--ema-checkpoint", directory / "output/checkpoint-1/ema",
            "--config", path, "--output", exported_ema)
    actual = torch.load(exported_ema, weights_only=True)
    expected = torch.load(directory / "output/weights-ema.pt", weights_only=True)
    assert actual["weights"] == "ema"
    for name, value in actual["model"].items():
        torch.testing.assert_close(value, expected["model"][name], rtol=0, atol=0)


def test_convert_checkpoint_rejects_different_head_before_writing(trained_run):
    directory, _, _ = trained_run
    payload = torch.load(directory / "output/weights.pt", weights_only=True)
    payload["config"]["model"]["joint_dit_head_type"] = "z"
    bad = directory / "different-head.pt"
    torch.save(payload, bad)
    output = directory / "rejected.pt"
    result = run_cli(ROOT / "scripts/convert_checkpoint.py", "--input", bad, "--output", output,
                     expect_success=False)
    assert result.returncode != 0 and "selected Z2 head" in result.stderr
    assert not output.exists()


def test_prepare_latents_cli_produces_readable_scaled_posterior_shards(tmp_path):
    from PIL import Image
    from models.codec import KL16Codec
    from data.cache import load_sharded_posterior
    from data.dataset import ImageNetFlowCacheDataset
    from test_evaluation_runtime import SmallVAE
    module = tmp_path / "codec-source/models"
    module.mkdir(parents=True)
    (module / "vae.py").write_text('''from types import SimpleNamespace
import torch
from torch import nn
class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim, ch_mult, ckpt_path):
        super().__init__()
        self.encoder = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.decoder = nn.ConvTranspose2d(embed_dim, 3, kernel_size=16, stride=16)
        self.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    def encode(self, images):
        mean = self.encoder(images)
        return SimpleNamespace(mean=mean, std=torch.ones_like(mean) * .2)
    def decode(self, latents):
        return self.decoder(latents).tanh()
''')
    vae = SmallVAE()
    checkpoint = tmp_path / "vae.pt"
    torch.save(vae.state_dict(), checkpoint)
    records, pixels = [], []
    for index, color in enumerate(((30, 80, 130), (200, 150, 100), (15, 45, 75))):
        image_path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (32 + 16 * index, 32), color=color).save(image_path)
        records.append(dict(img_id=index + 1, source_path=str(image_path), synset="n00000001", split="val"))
        pixels.append(torch.tensor(color).float().view(3, 1, 1).expand(3, 32, 32) / 255)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    cache = tmp_path / "posterior"
    run_cli(ROOT / "scripts/prepare_latents.py", "--manifest", manifest, "--output", cache,
            "--module-root", module.parent, "--checkpoint", checkpoint,
            "--image-size", 32, "--batch-size", 2, "--shard-size", 2, "--device", "cpu")
    bank = load_sharded_posterior(cache / "index.json")
    assert json.loads((cache / "index.json").read_text())["metadata"]["preprocessing"] == "bicubic_resize_center_crop_v1"
    assert bank["img_ids"].tolist() == [1, 2, 3]
    assert len(list(cache.glob("posterior-*.pt"))) == 2
    actual = torch.stack([bank["posterior_stats"][index] for index in range(3)])
    expected = KL16Codec(vae).encode_posterior(torch.stack(pixels)).half()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    mapping = tmp_path / "synsets.txt"
    mapping.write_text("n00000001 example, sample\n")
    dataset = ImageNetFlowCacheDataset(cache_path=str(cache / "index.json"), manifest_jsonl=str(manifest),
        tokenizer=local_tokenizer(tmp_path / "tokenizer"), boi_token_id=3, eoi_token_id=4,
        mask_token_id=5, eos_token_id=2, image_tokens_per_img=4, image_latent_dim=16,
        synset_mapping_path=str(mapping), conditioning_mode="class")
    assert dataset[0]["image_latents"].shape == (4, 16)
