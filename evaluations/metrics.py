"""Shared Inception feature, FID, and IS utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def load_reference_statistics(path, *, feature=2048, image_size=256):
    """Read original ImageNet sufficient-statistic assets or explicit moments.

    Original ``stats={count,sum,outer_sum}`` files retain their published
    ImageNet-val/extractor/preprocessing validation. Generic mean/covariance
    (or mu/sigma NPZ) inputs are supported but are not labeled formal ImageNet
    references, since they do not prove those data and preprocessing contracts.
    """
    from pathlib import Path
    import torch

    path = Path(path)
    if path.suffix == ".npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as handle:
            payload = {key: torch.from_numpy(handle[key].copy()) for key in handle.files}
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("reference statistics must contain a mapping")
    metadata, count, schema = payload.get("metadata", {}), payload.get("count"), payload.get("schema")
    formal_imagenet = "stats" in payload
    if formal_imagenet:
        if schema not in {"imagenet_inception_feature_moments_v1", "imagenet_val_inception_feature_moments_v2"}:
            raise ValueError(f"unsupported original real-stat schema: {schema!r}")
        if not isinstance(metadata, Mapping):
            raise ValueError("original real statistics require provenance metadata")
        source, extractor, transform = (metadata.get(key, {}) for key in ("source", "feature", "image_transform"))
        expected = {
            "classes": (source.get("classes"), 1000),
            "samples_per_class": (source.get("samples_per_class"), 50),
            "extractor": (extractor.get("extractor"), "torch-fidelity-inception-v3-compat"),
            "resize": (transform.get("resize"), int(image_size)),
            "interpolation": (transform.get("interpolation"), "bicubic"),
            "center_crop": (transform.get("center_crop"), int(image_size)),
            "color_mode": (transform.get("color_mode"), "RGB"),
        }
        if source.get("split") not in {"val", "validation"}:
            raise ValueError("original real statistics must use ImageNet validation")
        for name, (actual, wanted) in expected.items():
            if actual != wanted:
                raise ValueError(f"original real-stat metadata mismatch for {name}: {actual!r} != {wanted!r}")
        if extractor.get("feature") is not None and int(extractor["feature"]) != int(feature):
            raise ValueError("reference Inception feature dimension changed")
        dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(extractor.get("accumulation_dtype"))
        if dtype is None:
            raise ValueError("unsupported reference accumulation dtype")
        stats = payload["stats"]
        count = int(stats["count"])
        if count != 50_000:
            raise ValueError("formal ImageNet reference must contain all 50,000 validation images")
        for name in ("sum", "outer_sum"):
            if not isinstance(stats[name], torch.Tensor) or stats[name].dtype != dtype or not bool(torch.isfinite(stats[name]).all()):
                raise ValueError(f"reference {name} is nonfinite or differs from recorded dtype")
        if stats["sum"].shape != (feature,) or stats["outer_sum"].shape != (feature, feature):
            raise ValueError("reference sufficient-statistic shape differs from requested feature")
        moments = FeatureMoments(torch.tensor(count), stats["sum"].double(), stats["outer_sum"].double())
        mean, covariance = moments.mean_cov()
    else:
        if "mean" in payload and "covariance" in payload:
            mean, covariance = payload["mean"], payload["covariance"]
        elif "mu" in payload and "sigma" in payload:
            mean, covariance = payload["mu"], payload["sigma"]
        else:
            raise ValueError("reference needs original stats or mean/covariance (mu/sigma)")
        mean, covariance = torch.as_tensor(mean).double(), torch.as_tensor(covariance).double()
        if count is not None and int(count) < 2:
            raise ValueError("reference requires at least two samples")
        count = int(count) if count is not None else None
    if mean.shape != (feature,) or covariance.shape != (feature, feature):
        raise ValueError("reference mean/covariance shape differs from requested feature")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(covariance).all()):
        raise ValueError("reference moments contain nonfinite values")
    if not torch.allclose(covariance, covariance.T, rtol=1e-5, atol=1e-8) or bool((covariance.diagonal() < -1e-8).any()):
        raise ValueError("reference covariance must be symmetric with nonnegative variances")
    return {"mean": mean, "covariance": covariance, "count": count,
            "schema": schema or "explicit_reference_moments", "metadata": metadata,
            "formal_imagenet_reference": formal_imagenet}


def build_inception_extractor(
    feature: int,
    weights_path,
    device,
):
    from torchmetrics.image.fid import NoTrainInceptionV3

    class FeaturesAndLogitsInceptionV3(NoTrainInceptionV3):
        def forward(self, images):
            return self._torch_fidelity_forward(images)

    extractor = FeaturesAndLogitsInceptionV3(
        name="inception-v3-compat",
        features_list=[str(int(feature)), "logits_unbiased"],
        feature_extractor_weights_path=(
            str(weights_path) if weights_path is not None else None
        ),
        antialias=True,
    )
    return extractor.to(device).eval()


def metric_accumulation_dtype(device):
    """Choose a distributed-reduction dtype supported by the accelerator."""
    import torch

    # HCCL does not implement float64 all-reduce. Accumulate and reduce the
    # sufficient statistics in fp32 on Ascend, then convert the resulting
    # means/covariances to CPU float64 in ``frechet_distance`` below.
    return (
        torch.float32
        if torch.device(device).type == "npu"
        else torch.float64
    )


def extract_inception_features(extractor, images):
    """Return FID features and unbiased logits in a reducible dtype."""
    import torch

    uint8_images = (
        images.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
    )
    outputs = extractor(uint8_images)
    if isinstance(outputs, torch.Tensor):
        raise RuntimeError(
            "Inception extractor returned one tensor; expected FID features "
            "and logits."
        )
    if isinstance(outputs, Mapping):
        feature_keys = [
            key for key in outputs if key != "logits_unbiased"
        ]
        feature = (
            outputs.get(feature_keys[0])
            if len(feature_keys) == 1
            else None
        )
        logits = outputs.get("logits_unbiased")
    else:
        feature, logits = outputs
    if feature is None or logits is None:
        raise RuntimeError(
            "Inception extractor did not return features and "
            "logits_unbiased."
        )
    dtype = metric_accumulation_dtype(feature.device)
    return (
        feature.reshape(feature.shape[0], -1).to(dtype=dtype),
        logits.reshape(logits.shape[0], -1).to(dtype=dtype),
    )


@dataclass
class FeatureMoments:
    count: Any
    sum: Any
    outer_sum: Any

    @classmethod
    def zeros(cls, dimension: int, device):
        import torch

        dtype = metric_accumulation_dtype(device)

        return cls(
            count=torch.zeros((), dtype=torch.long, device=device),
            sum=torch.zeros(
                int(dimension),
                dtype=dtype,
                device=device,
            ),
            outer_sum=torch.zeros(
                int(dimension),
                int(dimension),
                dtype=dtype,
                device=device,
            ),
        )

    def update(self, features) -> None:
        features = features.to(
            device=self.sum.device,
            dtype=self.sum.dtype,
        )
        self.count += int(features.shape[0])
        self.sum += features.sum(dim=0)
        self.outer_sum += features.T @ features

    def all_reduce_(self) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self.count)
        dist.all_reduce(self.sum)
        dist.all_reduce(self.outer_sum)

    def mean_cov(self):
        count = int(self.count.item())
        if count < 2:
            raise ValueError(
                f"at least two features are required, found {count}"
            )
        mean = self.sum / count
        covariance = (
            self.outer_sum
            - count * mean[:, None] * mean[None, :]
        ) / (count - 1)
        return mean, (covariance + covariance.T) * 0.5


@dataclass
class InceptionScoreMoments:
    count: Any
    probability_sum: Any
    probability_log_probability_sum: Any

    @classmethod
    def zeros(cls, splits: int, classes: int, device):
        import torch

        dtype = metric_accumulation_dtype(device)

        return cls(
            count=torch.zeros(
                int(splits),
                dtype=torch.long,
                device=device,
            ),
            probability_sum=torch.zeros(
                int(splits),
                int(classes),
                dtype=dtype,
                device=device,
            ),
            probability_log_probability_sum=torch.zeros(
                int(splits),
                dtype=dtype,
                device=device,
            ),
        )

    def update(
        self,
        logits,
        split_ids: Sequence[int],
    ) -> None:
        import torch

        probabilities = logits.to(
            device=self.probability_sum.device,
            dtype=self.probability_sum.dtype,
        ).softmax(dim=-1)
        split_ids = torch.as_tensor(
            split_ids,
            device=self.probability_sum.device,
            dtype=torch.long,
        )
        if int(split_ids.numel()) != int(probabilities.shape[0]):
            raise ValueError(
                "Inception Score split ids must match the logits batch: "
                f"{int(split_ids.numel())} != {int(probabilities.shape[0])}"
            )
        if split_ids.numel() and (
            int(split_ids.min().item()) < 0
            or int(split_ids.max().item()) >= int(self.count.numel())
        ):
            raise ValueError(
                "Inception Score split ids must be in "
                f"[0, {int(self.count.numel())}), got "
                f"min={int(split_ids.min().item())}, "
                f"max={int(split_ids.max().item())}"
            )
        p_log_p = (
            probabilities
            * probabilities.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            ).log()
        ).sum(dim=-1)
        for split in split_ids.unique().tolist():
            mask = split_ids == int(split)
            self.count[split] += int(mask.sum().item())
            self.probability_sum[split] += probabilities[mask].sum(dim=0)
            self.probability_log_probability_sum[split] += p_log_p[
                mask
            ].sum()

    def all_reduce_(self) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self.count)
        dist.all_reduce(self.probability_sum)
        dist.all_reduce(self.probability_log_probability_sum)

    def compute(self) -> tuple[float, float, list[float]]:
        import torch

        scores = []
        for split in range(int(self.count.numel())):
            count = int(self.count[split].item())
            if count <= 0:
                raise ValueError(
                    f"Inception Score split {split} is empty"
                )
            marginal = self.probability_sum[split] / count
            expected_p_log_p = (
                self.probability_log_probability_sum[split] / count
            )
            marginal_entropy_term = (
                marginal
                * marginal.clamp_min(
                    torch.finfo(marginal.dtype).tiny
                ).log()
            ).sum()
            scores.append(
                float(
                    torch.exp(
                        expected_p_log_p - marginal_entropy_term
                    )
                )
            )
        values = torch.tensor(scores, dtype=torch.float64)
        return (
            float(values.mean()),
            float(values.std(unbiased=False)),
            scores,
        )


def frechet_distance(
    real_mean,
    real_cov,
    fake_mean,
    fake_cov,
) -> float:
    """Numerically stable FID using symmetric eigendecompositions."""
    import torch

    real_mean = real_mean.detach().cpu().double()
    fake_mean = fake_mean.detach().cpu().double()
    real_cov = (
        real_cov.detach().cpu().double()
        + real_cov.T.detach().cpu().double()
    ) * 0.5
    fake_cov = (
        fake_cov.detach().cpu().double()
        + fake_cov.T.detach().cpu().double()
    ) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(real_cov)
    real_sqrt = (
        eigenvectors
        * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
    ) @ eigenvectors.T
    middle = real_sqrt @ fake_cov @ real_sqrt
    middle = (middle + middle.T) * 0.5
    trace_sqrt_product = (
        torch.linalg.eigvalsh(middle)
        .clamp_min(0)
        .sqrt()
        .sum()
    )
    difference = real_mean - fake_mean
    fid = (
        difference.dot(difference)
        + torch.trace(real_cov)
        + torch.trace(fake_cov)
        - 2.0 * trace_sqrt_product
    )
    return float(fid.clamp_min(0))
