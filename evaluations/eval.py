"""Evaluate exported UMM weights with shared data, loss and sampling pipelines.

Loss/perplexity/generation use the independent configured validation datasets.
Candidate benchmarks additionally accept a prepared batch bundle containing
their frozen candidate identities and answer map, keeping benchmark prompts out
of the model implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from configs.config import load_config
from evaluations.validation import seed_evaluation, validate_losses


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _batches(loader, device, *, task=None, rank=0, world=1, seed=424242):
    # Every process fetches global batches, then selects its deterministic share.
    seed_evaluation(seed)
    iterator = iter(loader)
    index = 0
    while True:
        seed_evaluation(seed + index)
        try:
            batch = next(iterator)
        except StopIteration:
            return
        selected = index % world == rank
        index += 1
        if selected and (task is None or batch["task"] == task):
            yield _to_device(batch, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="raw/EMA weight export or supported model directory")
    parser.add_argument("--task", choices=("loss", "perplexity", "generation", "choices", "classification", "retrieval"), default="loss")
    parser.add_argument("--source", choices=("text", "i2t"), default="text", help="perplexity source")
    parser.add_argument("--batches", help="optional torch bundle with batches and benchmark metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    config["model"]["checkpoint"] = args.checkpoint
    from accelerate import Accelerator
    from training.train import load_model, load_tokenizer
    accelerator = Accelerator(cpu=bool(config["training"].get("cpu", False)),
                              mixed_precision=config["training"]["mixed_precision"])
    bundle = torch.load(args.batches, map_location="cpu", weights_only=True) if args.batches else None
    if bundle is None:
        tokenizer = load_tokenizer(config)
    model = load_model(config).to(accelerator.device)
    if bundle is None:
        from data.dataloader import build_validation_loader
        # Quality probes use a dedicated view even when training used joint
        # caption batches. Loss validation keeps the original task schedule.
        evaluation_config = config
        if args.task in {"generation", "perplexity"}:
            source = "t2i" if args.task == "generation" else args.source
            evaluation_config = {**config, "data": {**config["data"], "schedule": [source]}}
        loader = build_validation_loader(evaluation_config, tokenizer, loss_protocol=args.task == "loss")
        if loader is None:
            raise ValueError("configure independent validation data before evaluation")
    else:
        loader = bundle["batches"]
    settings = config["evaluation"]
    seed = int(settings.get("seed", 424242))
    rank, world = accelerator.process_index, accelerator.num_processes
    options = dict(device=accelerator.device, rank=rank, world=world, seed=seed)

    with accelerator.autocast():
        if args.task == "loss":
            from training.loss import forward_loss, prepare_batch
            from evaluations.validation import loss_validation_options
            # validate_losses owns the per-source rank partition and seeds.
            batches = (_to_device(batch, accelerator.device) for batch in loader)
            result = validate_losses(model, batches, lambda model, batch: forward_loss(model,
                prepare_batch(batch, accelerator.device, config,
                    image_input_dtype=model.backbone.image_token_embedder.z_proj.weight.dtype), config),
                schedule=config["data"]["schedule"], device=accelerator.device,
                **loss_validation_options(config), distributed=world > 1)
        elif args.task == "perplexity":
            from evaluations.understanding import evaluate_perplexity
            result = evaluate_perplexity(model, _batches(loader, task=args.source, **options))
        elif args.task == "generation":
            from models.codec import KL16Codec
            from evaluations.generation import GenerationMetrics, evaluate_generation, stratified_split_ids
            from evaluations.metrics import build_inception_extractor, load_reference_statistics
            reference = None
            reference_path = settings.get("reference_statistics") or settings.get("real_stats_path")
            feature = int(settings.get("inception_feature", settings.get("fid_feature", 2048)))
            reference_info = None
            if reference_path:
                payload = load_reference_statistics(reference_path, feature=feature,
                    image_size=math.isqrt(model.config.image_tokens_per_img) * 16)
                reference = (payload["mean"], payload["covariance"])
                reference_info = {key: payload[key] for key in ("count", "schema", "formal_imagenet_reference")}
                reference_info["path"] = str(reference_path)
            elif bundle is None:
                raise ValueError("generation needs evaluation.real_stats_path/reference_statistics; prepared bundles may instead supply real_images")
            splits = int(settings.get("inception_splits", 10))
            split_lookup = None
            if bundle is None:
                if "t2i" not in loader.loaders:
                    raise ValueError("generation requires T2I validation data")
                dataset = loader.loaders["t2i"].dataset
                ids = [int(value) for value in dataset.img_ids]
                assignments = stratified_split_ids(ids, [dataset.synsets[value] for value in ids], splits=splits)
                split_lookup = dict(zip(ids, assignments))

            def generation_batches():
                for batch in _batches(loader, task="t2i", **options):
                    if split_lookup is not None:
                        batch["is_split_ids"] = [split_lookup[int(value)] for value in batch["img_ids"]]
                    yield batch

            codec = KL16Codec.from_pretrained(**config["codec"], device=accelerator.device)
            extractor = build_inception_extractor(feature,
                settings.get("inception_weights") or settings.get("inception_weights_path"), accelerator.device)
            metrics = GenerationMetrics(extractor, feature_dim=feature, splits=splits,
                                        device=accelerator.device, reference_statistics=reference)
            sampling = {key: value for key, value in config["sampling"].items()
                        if key in {"steps", "solver", "cfg", "cfg_schedule", "temperature", "seed"}}
            result = evaluate_generation(model, generation_batches(), codec, metrics, **sampling)
            if reference_info is not None:
                result["reference_statistics"] = reference_info
                result["formal_imagenet_sample_count"] = bool(reference_info["formal_imagenet_reference"]
                    and result["generated_samples"] == reference_info["count"])
        else:
            if bundle is None:
                raise ValueError("candidate evaluations require --batches with the frozen benchmark candidates")
            from evaluations.understanding import (
                evaluate_choices, score_image_text_candidates, classification_metrics, retrieval_metrics,
            )
            batches = _batches(loader, **options)
            if args.task == "choices":
                result = evaluate_choices(model, batches, bundle["answers"])
            else:
                scores = score_image_text_candidates(model, batches, image_count=bundle["image_count"],
                                                     text_count=bundle["text_count"])
                if args.task == "classification":
                    result = classification_metrics(scores, bundle["labels"])
                else:
                    result = retrieval_metrics(scores, bundle["image_ids"], bundle["text_image_ids"])
    if accelerator.is_main_process:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {"task": args.task, "checkpoint": args.checkpoint, "seed": seed,
                  "world_size": world, "metrics": result}
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    accelerator.wait_for_everyone()
    return result


if __name__ == "__main__":
    main()
