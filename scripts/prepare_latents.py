"""Encode a manifest into bounded KL16 posterior shards and a lazy row index.

Each JSONL row has img_id, source_path, synset and split (train or val). Relative
image paths are resolved against the manifest directory. The original manifest
remains the dataset's manifest_jsonl; index.json becomes its cache_path.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.codec import KL16Codec, POSTERIOR_FORMAT, POSTERIOR_LAYOUT, kl16_layout
from data.cache import POSTERIOR_INDEX_SCHEMA


def image_tensor(path, size):
    """Source PIL Resize(size, BICUBIC) + CenterCrop(size), in [0,1].

    Match torchvision's integer short-edge resize and round-to-even crop
    coordinates. KL16Codec performs the source Normalize(0.5, 0.5) once.
    """
    with Image.open(path) as source:
        image = source.convert("RGB")
    if size < 1:
        raise ValueError("image size must be positive")
    width, height = image.size
    resized = (size, int(size * height / width)) if width <= height else (int(size * width / height), size)
    if resized != image.size:
        image = image.resize(resized, Image.Resampling.BICUBIC)
    left, top = int(round((image.width - size) / 2)), int(round((image.height - size) / 2))
    array = np.array(image.crop((left, top, left + size, top + size)), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New cache directory")
    parser.add_argument("--module-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--scaling-factor", type=float, default=.2325)
    args = parser.parse_args(argv)
    _, tokens = kl16_layout(args.image_size)
    if args.batch_size < 1 or args.shard_size < 1:
        parser.error("batch-size and shard-size must be positive")
    manifest, output = Path(args.manifest), Path(args.output)
    codec = KL16Codec.from_pretrained(args.module_root, args.checkpoint, device=args.device,
                                     scaling_factor=args.scaling_factor)
    metadata = dict(format=POSTERIOR_FORMAT, stats_layout=POSTERIOR_LAYOUT,
                    scaling_factor=args.scaling_factor, image_size=args.image_size,
                    preprocessing="bicubic_resize_center_crop_v1")
    output.mkdir(parents=True, exist_ok=False)
    ids, shard_rows, shard_paths, seen = [], [], [], set()
    pending_images, pending_ids, shard_stats, shard_ids = [], [], [], []

    def save_shard():
        if not shard_ids:
            return
        index = len(shard_paths)
        name = f"posterior-{index:05d}.pt"
        torch.save(dict(posterior_stats=torch.stack(shard_stats), img_ids=torch.tensor(shard_ids),
                        metadata=metadata), output / name)
        ids.extend(shard_ids)
        shard_rows.extend((index, row) for row in range(len(shard_ids)))
        shard_paths.append(name)
        shard_stats.clear()
        shard_ids.clear()

    def encode_pending():
        if not pending_ids:
            return
        stats = codec.encode_posterior(torch.stack(pending_images)).cpu().half()
        KL16Codec.validate_posterior(stats)
        for identifier, row in zip(pending_ids, stats):
            shard_stats.append(row.clone())
            shard_ids.append(identifier)
            if len(shard_ids) == args.shard_size:
                save_shard()
        pending_images.clear()
        pending_ids.clear()

    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = int(row["img_id"])
            if identifier in seen or row.get("split") not in {"train", "val"} or not row.get("synset"):
                raise ValueError("Manifest needs unique img_id, synset and train/val split")
            seen.add(identifier)
            path = Path(row["source_path"])
            if not path.is_absolute():
                path = manifest.parent / path
            pending_images.append(image_tensor(path, args.image_size))
            pending_ids.append(identifier)
            if len(pending_ids) == args.batch_size:
                encode_pending()
    encode_pending()
    save_shard()
    if not ids:
        raise ValueError("Manifest is empty")
    torch.save(dict(img_ids=torch.tensor(ids), shard_rows=torch.tensor(shard_rows)), output / "rows.pt")
    (output / "index.json").write_text(json.dumps(dict(schema=POSTERIOR_INDEX_SCHEMA,
        row_index="rows.pt", shards=shard_paths, token_shape=[tokens, 32], metadata=metadata), indent=2) + "\n")


if __name__ == "__main__":
    main()
