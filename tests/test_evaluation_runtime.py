from types import SimpleNamespace
import math
import random

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.codec import KL16Codec
from evaluations.generation import canonical_noise, stratified_split_ids, GenerationMetrics, evaluate_generation
from evaluations.metrics import FeatureMoments, InceptionScoreMoments, frechet_distance
from evaluations.validation import evaluation_state, reduce_loss_metrics, validate_losses
from evaluations.understanding import (
    score_batch, evaluate_choices, language_prior_debiased_scores, classification_metrics, retrieval_metrics,
)
from models.modeling_umm import UMM, UMMConfig


class SmallVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 16, kernel_size=16, stride=16)
        self.decoder = nn.ConvTranspose2d(16, 3, kernel_size=16, stride=16)

    def encode(self, images):
        mean = self.encoder(images)
        return SimpleNamespace(mean=mean, std=torch.ones_like(mean) * 0.2)

    def decode(self, latents):
        return self.decoder(latents).tanh()


def small_model():
    return UMM(UMMConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, mask_token_id=26,
        image_mask_token_id=27, boi_token_id=28, eoi_token_id=29, eos_token_id=30, pad_token_id=31,
        image_latent_dim=16, image_tokens_per_img=4, image_flow_width=16, image_flow_depth=1,
        joint_dit_head_dim=8, joint_dit_intermediate=32, image_input_noise_strength=0,
        image_flow_grad_checkpointing=False, gradient_checkpointing=False))


def text_batch():
    return {"input_ids": torch.tensor([[2, 3, 4], [2, 5, 6]]),
            "token_types": torch.zeros(2, 3, dtype=torch.long),
            "segment_ids": torch.zeros(2, 3, dtype=torch.long),
            "labels": torch.tensor([[-100, 3, 4], [-100, -100, 6]]), "task": "text"}


def test_kl16_scaled_posterior_and_decode_layout():
    vae = SmallVAE()
    codec = KL16Codec(vae)
    images = torch.rand(2, 3, 32, 32)
    stats = codec.encode_posterior(images)
    direct = vae.encode(images * 2 - 1)
    expected = torch.cat((direct.mean, direct.std), 1).mul(.2325).permute(0, 2, 3, 1).reshape(2, 4, 32)
    torch.testing.assert_close(stats, expected)
    latent = codec.sample_posterior(stats, noise=torch.ones(2, 4, 16))
    torch.testing.assert_close(latent, stats[..., :16] + stats[..., 16:])
    pixels = codec.decode(latent)
    expected_pixels = (vae.decode(latent.reshape(2, 2, 2, 16).permute(0, 3, 1, 2) / .2325) + 1) / 2
    torch.testing.assert_close(pixels, expected_pixels)
    assert not any(p.requires_grad for p in vae.parameters())
    with pytest.raises(ValueError, match="negative std"):
        codec.sample_posterior(torch.full((1, 4, 32), -1.0))


def test_fid_and_is_sufficient_statistics_match_analytic_values():
    features = torch.tensor([[0., 0.], [2., 0.]], dtype=torch.float64)
    real, fake = FeatureMoments.zeros(2, "cpu"), FeatureMoments.zeros(2, "cpu")
    real.update(features[:1]); real.update(features[1:])
    fake.update(features + torch.tensor([3., 0.]))
    mean, covariance = real.mean_cov()
    torch.testing.assert_close(mean, torch.tensor([1., 0.], dtype=torch.float64))
    torch.testing.assert_close(covariance, torch.diag(torch.tensor([2., 0.], dtype=torch.float64)))
    assert frechet_distance(mean, covariance, *fake.mean_cov()) == pytest.approx(9.0)
    score = InceptionScoreMoments.zeros(2, 2, "cpu")
    score.update(torch.tensor([[30., -30.], [-30., 30.]] * 2), [0, 0, 1, 1])
    mean_is, std, splits = score.compute()
    assert mean_is == pytest.approx(2.0) and std == 0 and splits == pytest.approx([2., 2.])


def test_noise_identity_and_class_stratified_is_are_batch_independent():
    whole = canonical_noise([3, 5, 7], seed=4, tokens=4)
    split = torch.cat([canonical_noise([index], seed=4, tokens=4) for index in [3, 5, 7]])
    torch.testing.assert_close(whole, split, atol=0, rtol=0)
    assert stratified_split_ids([4, 1, 3, 2], ["a", "a", "b", "b"], splits=2) == [1, 0, 1, 0]


def test_validation_keeps_token_mean_and_microbatch_mean_distinct():
    result = reduce_loss_metrics({"text": [26, 10, 3., 2], "t2i": [8, 4, 2., 1]},
                                 schedule=["text", "t2i", "text"])
    assert result["val/loss_text"] == pytest.approx(2.6)
    assert result["val/weighted_contribution_text"] == pytest.approx(1.0)
    assert result["val/loss"] == pytest.approx(5/3)
    with pytest.raises(ValueError, match="cover every"):
        reduce_loss_metrics({"text": [1, 1, 1, 1]}, schedule=["text", "t2i"])


def test_same_position_likelihood_real_cq_model_and_eval_state():
    model, batch = small_model(), text_batch()
    model.train()
    model.generator.eval()  # preserve heterogeneous submodule modes too
    modes = {name: module.training for name, module in model.named_modules()}
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    with evaluation_state(model):
        scored = score_batch(model, batch)
        output = model(batch)
        direct = -F.cross_entropy(output["text_logits"], output["text_labels"], reduction="none")
        torch.testing.assert_close(scored["loglikelihood"], torch.stack((direct[:2].sum(), direct[2])))
        assert scored["token_count"].tolist() == [2, 1]
        _ = torch.randn(8); _ = random.random()
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    choice_batch = {**batch, "question_ids": ["q", "q"], "choice_ids": [0, 1], "normalization_chars": [7, 3]}
    result = evaluate_choices(model, [choice_batch], {"q": 0})
    assert result["questions"] == 1 and result["normalization"] == "original_choice_characters"


def test_language_prior_probability_mean_and_retrieval():
    scores = torch.tensor([[-1., -5.], [-5., -1.]])
    calibrated, prior = language_prior_debiased_scores(scores)
    torch.testing.assert_close(prior, scores.exp().mean(0).log())
    assert classification_metrics(scores, [0, 1])["top1"] == 1.0
    assert all(v == 1.0 for v in retrieval_metrics(scores, ["a", "b"], ["a", "b"]).values())


def test_generation_uses_real_joint_sampler_codec_and_metrics():
    class PixelFeatures(nn.Module):
        def forward(self, pixels):
            features = pixels.float().mean((2, 3))[:, :2]
            return features, features
    model = small_model()
    batch = dict(input_ids=torch.tensor([[2, 28, 27, 27, 27, 27, 29]]).repeat(2, 1),
        token_types=torch.tensor([[0, 2, 1, 1, 1, 1, 2]]).repeat(2, 1),
        image_latents=torch.zeros(2, 7, 16), segment_ids=torch.zeros(2, 7, dtype=torch.long),
        image_span_table=torch.tensor([[0, 0, 2, 6], [1, 0, 2, 6]]), task="t2i",
        sample_indices=[0, 1], is_split_ids=[0, 0], real_images=torch.rand(2, 3, 32, 32))
    metrics = GenerationMetrics(PixelFeatures(), feature_dim=2, classes=2, splits=1)
    result = evaluate_generation(model, [batch], KL16Codec(SmallVAE()), metrics, steps=2)
    assert result["generated_samples"] == 2 and math.isfinite(result["fid"])
    assert result["inception_score_mean"] >= 1
    assert result["weights"] == "provided_model"


def test_evaluation_cli_loads_exported_model_and_scores_targets(tmp_path):
    import json
    import yaml
    from evaluations.eval import main
    from utils.checkpoint import export_weights
    model = small_model()
    export_weights(tmp_path / "model.pt", model=model, config=model.config.to_dict())
    config = {"model": model.config.to_dict(), "data": {"schedule": ["text"]},
              "training": {"cpu": True, "mixed_precision": "no", "gradient_accumulation_steps": 1}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    torch.save({"batches": [text_batch()]}, tmp_path / "batches.pt")
    result = main(["--config", str(tmp_path / "config.yaml"), "--checkpoint", str(tmp_path / "model.pt"),
                   "--batches", str(tmp_path / "batches.pt"), "--task", "perplexity", "--output", str(tmp_path / "metrics.json")])
    assert result["target_tokens"] == 3
    assert json.loads((tmp_path / "metrics.json").read_text())["metrics"] == result


def test_original_imagenet_reference_asset_loads_without_conversion(tmp_path):
    from copy import deepcopy
    from evaluations.metrics import load_reference_statistics
    payload = {"schema": "imagenet_val_inception_feature_moments_v2",
        "stats": {"count": 50000, "sum": torch.tensor([50000., 0.], dtype=torch.float64),
                  "outer_sum": torch.diag(torch.tensor([100000., 0.], dtype=torch.float64))},
        "metadata": {"source": {"classes": 1000, "samples_per_class": 50, "split": "val"},
            "feature": {"feature": 2, "extractor": "torch-fidelity-inception-v3-compat", "accumulation_dtype": "torch.float64"},
            "image_transform": {"resize": 256, "center_crop": 256, "interpolation": "bicubic", "color_mode": "RGB"}}}
    path = tmp_path / "source-reference.pt"
    torch.save(payload, path)
    loaded = load_reference_statistics(path, feature=2)
    torch.testing.assert_close(loaded["mean"], torch.tensor([1., 0.], dtype=torch.float64))
    torch.testing.assert_close(loaded["covariance"], torch.diag(torch.tensor([50000/49999, 0.], dtype=torch.float64)))
    assert loaded["count"] == 50000 and loaded["formal_imagenet_reference"]
    bad = deepcopy(payload)
    bad["metadata"]["image_transform"]["interpolation"] = "bilinear"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="interpolation"):
        load_reference_statistics(path, feature=2)
    bad = deepcopy(payload)
    bad["stats"]["sum"] = bad["stats"]["sum"].float()
    torch.save(bad, path)
    with pytest.raises(ValueError, match="recorded dtype"):
        load_reference_statistics(path, feature=2)


@pytest.mark.parametrize("format", ["explicit", "npz"])
def test_explicit_and_npz_reference_moments_are_not_mislabeled_formal(tmp_path, format):
    import numpy as np
    from evaluations.metrics import load_reference_statistics
    mean, covariance = torch.tensor([1., 2.]), torch.eye(2)
    if format == "npz":
        path = tmp_path / "reference.npz"
        np.savez(path, mu=mean.numpy(), sigma=covariance.numpy())
    else:
        path = tmp_path / "reference.pt"
        torch.save({"mean": mean, "covariance": covariance}, path)
    loaded = load_reference_statistics(path, feature=2)
    torch.testing.assert_close(loaded["mean"], mean.double())
    torch.testing.assert_close(loaded["covariance"], covariance.double())
    assert not loaded["formal_imagenet_reference"] and loaded["count"] is None
