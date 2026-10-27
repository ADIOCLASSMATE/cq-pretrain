#!/usr/bin/env python3
"""Cache native Qwen ViT features and/or normalized Wan latents from one shared clip."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from configs.config import load_config
from data.cache import (
    QWEN_FORMAT,
    WAN_FORMAT,
    checkpoint_id,
    digest,
    load_cache,
    save_cache,
)
from data.dataset import read_records, prepare_raw_video, qwen_video_inputs
from training.train import load_processor, register_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Override sources with this manifest")
    parser.add_argument("--task", choices=["t2v", "v2t"])
    parser.add_argument("--kind", choices=["both", "qwen", "wan"], default="both")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--output", required=True, help="Canonical JSONL containing cache references"
    )
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    if config["model"].get("checkpoint") and args.kind != "wan":
        raise ValueError(
            "Qwen cache preparation uses the native qwen_path weights; use online inputs for a trained ViT"
        )
    device = config["training"]["device"]
    register_device(device)
    sources = config["data"]["sources"]
    if args.manifest:
        sources = [
            {"path": args.manifest, **({"task": args.task} if args.task else {})}
        ]
    if not sources:
        raise ValueError("Provide --manifest or data.sources")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    processor = load_processor(config)
    qwen = codec = None
    kinds = ("qwen", "wan") if args.kind == "both" else (args.kind,)
    ids = {
        "qwen": checkpoint_id(config["model"]["qwen_path"]),
        "wan": checkpoint_id(
            config["codec"].get("path") or config["model"]["wan_path"]
        ),
    }
    settings = config["data"]["video"]
    records = []
    for source in sources:
        for row in read_records(source):
            video = row["videos"][0]
            rgb, clip = prepare_raw_video(video, settings)
            for kind in kinds:
                metadata = dict(clip=clip, preprocessing=settings, checkpoint=ids[kind])
                key = digest(metadata)
                path = Path(args.cache_dir).resolve() / kind / f"{key}.pt"
                if path.exists():
                    load_cache(path, kind, checkpoint=ids[kind], preprocessing=settings)
                elif kind == "qwen":
                    pixels, grid, times = qwen_video_inputs(
                        rgb, clip, processor, settings
                    )
                    if qwen is None:
                        from models.backbone_qwen35 import Qwen35VideoEncoder

                        qwen = (
                            Qwen35VideoEncoder.from_pretrained(
                                config["model"]["qwen_path"],
                                dtype=torch.bfloat16
                                if config["training"]["mixed_precision"] == "bf16"
                                else torch.float32,
                                attn_implementation=config["model"][
                                    "attn_implementation"
                                ],
                            )
                            .to(device)
                            .eval()
                            .requires_grad_(False)
                        )
                    with torch.no_grad():
                        features = (
                            qwen(pixels.to(device), grid[None].to(device))[0]
                            .float()
                            .cpu()
                        )
                    metadata["qwen_timestamps"] = times
                    save_cache(
                        path,
                        dict(
                            format=QWEN_FORMAT,
                            features=features,
                            grid_thw=grid,
                            metadata=metadata,
                        ),
                    )
                else:
                    if codec is None:
                        from data.codec import WanVideoCodec

                        codec = WanVideoCodec.from_pretrained(
                            config["codec"].get("path") or config["model"]["wan_path"],
                            device=device,
                            dtype=getattr(torch, config["codec"]["dtype"]),
                        )
                    latent = codec.encode(
                        rgb.permute(1, 0, 2, 3).float().div(127.5).sub(1)
                    ).cpu()
                    save_cache(
                        path, dict(format=WAN_FORMAT, latent=latent, metadata=metadata)
                    )
                video[f"{kind}_cache"] = str(path)
            records.append(
                {k: row[k] for k in ("sample_id", "task", "messages", "videos")}
            )
            print(f"cached {row['task']} {row['sample_id']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    print(output)


if __name__ == "__main__":
    main()
