"""Full-image generation and the source FID/Inception Score protocol."""
from __future__ import annotations

import torch

from evaluations.metrics import (
    FeatureMoments, InceptionScoreMoments, extract_inception_features, frechet_distance,
)
from evaluations.validation import evaluation_state


def stratified_split_ids(image_ids, synsets, *, splits=10):
    """Original IS partition: sort stable IDs within each class, then round robin."""
    if splits <= 0 or len(image_ids) < splits or len(image_ids) != len(synsets):
        raise ValueError("invalid stratified IS split dimensions")
    if len(set(map(int, image_ids))) != len(image_ids) or any(not str(s).strip() for s in synsets):
        raise ValueError("IS splitting requires unique image IDs and nonempty synsets")
    groups = {}
    for row, (image, synset) in enumerate(zip(image_ids, synsets)):
        groups.setdefault(str(synset), []).append((int(image), row))
    assignments = [-1] * len(image_ids)
    for members in groups.values():
        for index, (_, row) in enumerate(sorted(members)):
            assignments[row] = index % splits
    if set(assignments) != set(range(splits)):
        raise ValueError("stratified IS would contain empty splits; use fewer splits")
    return assignments


def canonical_noise(indices, *, seed, tokens=256, latent_dim=16):
    """CPU FP32 noise keyed by global sample identity, independent of batching."""
    if not indices or len(set(map(int, indices))) != len(indices) or min(indices) < 0:
        raise ValueError("generation requires distinct nonnegative sample indices")
    return torch.stack([torch.randn(tokens, latent_dim, generator=torch.Generator(device="cpu").manual_seed(
        (int(seed) + int(index)) % (2**63)), dtype=torch.float32) for index in indices])


class GenerationMetrics:
    """Use the original Inception uint8 conversion and unbiased covariance.

    Split IDs are explicit dataset protocol inputs (e.g. source stratified
    synset assignments), never inferred from rank or minibatch number.
    """

    def __init__(self, extractor, *, feature_dim=2048, classes=1008, splits=10, device="cpu",
                 reference_statistics=None):
        self.extractor = extractor
        self.real = FeatureMoments.zeros(feature_dim, device)
        self.fake = FeatureMoments.zeros(feature_dim, device)
        self.inception = InceptionScoreMoments.zeros(splits, classes, device)
        self.reference_statistics = reference_statistics

    @torch.no_grad()
    def update(self, generated, *, split_ids, real_images=None):
        features, logits = extract_inception_features(self.extractor, generated)
        self.fake.update(features)
        self.inception.update(logits, split_ids)
        if real_images is not None:
            if self.reference_statistics is not None:
                raise ValueError("choose reference statistics or reference images, not both")
            real_features, _ = extract_inception_features(self.extractor, real_images)
            self.real.update(real_features)

    def compute(self, *, distributed=True):
        if distributed:
            self.fake.all_reduce_()
            self.inception.all_reduce_()
            if self.reference_statistics is None:
                self.real.all_reduce_()
        real_mean, real_cov = self.reference_statistics if self.reference_statistics is not None else self.real.mean_cov()
        fake_mean, fake_cov = self.fake.mean_cov()
        mean, std, splits = self.inception.compute()
        return {"fid": frechet_distance(real_mean, real_cov, fake_mean, fake_cov),
                "inception_score_mean": mean, "inception_score_std": std,
                "inception_score_splits": splits, "generated_samples": int(self.fake.count)}


def evaluate_generation(model, batches, codec, metrics, *, steps=10, solver="heun", cfg=1.0,
                        cfg_schedule="constant", temperature=1.0, seed=0, ema=None,
                        on_images=None, distributed=True):
    """Batches include sample_indices/is_split_ids and optional real_images.

    Every rank owns distinct sample IDs. Reference moments may be precomputed;
    without them, real_images must cover the intended reference distribution.
    ``on_images(indices, pixels)`` can save a gallery without another sampler.
    """
    from models.sampling import sample_visual_latents

    seen = set()
    with evaluation_state(model, ema=ema):
        for batch in batches:
            indices = [int(index) for index in batch["sample_indices"]]
            if seen.intersection(indices):
                raise ValueError("evaluation encountered duplicate global sample identity")
            seen.update(indices)
            noise = canonical_noise(indices, seed=seed, tokens=model.config.image_tokens_per_img,
                                    latent_dim=model.config.image_latent_dim)
            latents = sample_visual_latents(model, batch, steps=steps, solver=solver, cfg=cfg,
                cfg_schedule=cfg_schedule, temperature=temperature, initial_noise=noise)
            if latents.shape[0] != len(indices):
                raise ValueError("sample identities must cover every generated image span")
            images = codec.decode(latents)
            metrics.update(images, split_ids=batch["is_split_ids"], real_images=batch.get("real_images"))
            if on_images is not None:
                on_images(indices, images)
        result = metrics.compute(distributed=distributed)
    result.update(seed=seed, steps=steps, solver=solver, cfg=cfg, cfg_schedule=cfg_schedule,
                  temperature=temperature, weights="ema" if ema is not None else "provided_model")
    return result
