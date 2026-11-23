"""How fast can one rank's host pipeline produce packs, and what limits it?

The training loop's `data_wait_seconds` says the loader is behind, but not why.
This runs the real DocumentStream + Packer for one rank with no model, no
device and no collectives, so the only thing being measured is the host: shard
reads, online tokenization, image decode/resize and packing.

It exists because the Ascend step went from 36 s to 29 s of compute and the
loader stopped keeping up, and because `torch.set_num_threads` is never called
anywhere in this repository: every image worker thread calls the torchvision
image processor, which fans each small resize out over the process-wide
intra-op pool. With 16 ranks on a node that is heavy oversubscription, and the
contention does not show up as saturated cores.

    python -m scripts.validation.bench_loader_host --config ... --data-root ... \
        --packs 24 --world 32 --rank 0 --torch-threads 1

Report fields that decide the question:
  packs_per_second          what the training loop can consume
  image_wait_share          fraction of packing time blocked on image decode
  seconds_per_image         decode + resize cost per image, this thread count
"""

import argparse
import json
import os
import time
from pathlib import Path


def build(args, cfg, data, rank, world):
    from data.packing import ConsumedLoader, Packer
    from data.stream import DocumentStream

    docs = DocumentStream(
        args.data_root,
        args.data_mode,
        rank,
        world,
        args.seed,
        epoch=0,
        workers=args.data_workers,
        prefetch=args.data_prefetch,
        source_root=args.source_root,
        block_rows=args.shuffle_block_rows,
        tokenize_rows=args.tokenize_rows,
        image_threads=args.image_workers,
        image_prefetch=args.image_prefetch,
        allow_development=args.allow_development_data,
        cfg=cfg,
        sequence_length=args.sequence_length,
    )
    packer = Packer(
        docs,
        cfg,
        args.sequence_length,
        data.tokens,
        None,
        pin_device=None,
        max_images=args.max_images_per_pack,
        padding_multiple=args.pack_padding_multiple,
    )
    return docs, packer, ConsumedLoader(packer, args.data_prefetch, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=int, default=24, help="Packs to draw after the warm-up ones")
    parser.add_argument("--warmup-packs", type=int, default=8, help="Excluded: they fill the queues")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=0, help="torch.set_num_threads; 0 leaves the default")
    parser.add_argument("--label", default="run")
    parser.add_argument("--output", type=Path)
    known, rest = parser.parse_known_args()

    import torch

    before = torch.get_num_threads()
    if known.torch_threads:
        torch.set_num_threads(known.torch_threads)

    from training import configure
    from training.options import parse_args

    args = parse_args(rest)
    cfg, data, _ = configure.resolve(args)
    if data is None:
        raise SystemExit("This benchmark needs --data-root; it measures the real corpus path")

    docs, packer, loader = build(args, cfg, data, known.rank, known.world)
    for _ in range(known.warmup_packs):
        loader.next()
    images_before = dict(docs.image_reader.counters) if docs.image_reader else {}
    stats_before = dict(docs.stats)
    pack_before = dict(packer.stats)

    begin = time.perf_counter()
    occupied = 0
    for _ in range(known.packs):
        _, receipt = loader.next()
        occupied += receipt["occupied"]
    seconds = time.perf_counter() - begin

    images = dict(docs.image_reader.counters) if docs.image_reader else {}
    decoded = images.get("images", 0) - images_before.get("images", 0)
    waited = docs.stats["image_wait_seconds"] - stats_before["image_wait_seconds"]
    packed = packer.stats["pack_seconds"] - pack_before["pack_seconds"]
    report = dict(
        label=known.label,
        packs=known.packs,
        seconds=round(seconds, 2),
        packs_per_second=round(known.packs / seconds, 3),
        occupied_tokens_per_second=round(occupied / seconds, 1),
        images=decoded,
        images_per_second=round(decoded / seconds, 2),
        seconds_per_image=round(seconds * max(1, args.image_workers) / max(1, decoded), 4),
        image_wait_seconds=round(waited, 2),
        image_wait_share=round(waited / max(1e-9, packed), 3),
        pack_seconds=round(packed, 2),
        parquet_seconds=round(docs.stats["parquet_seconds"] - stats_before["parquet_seconds"], 2),
        tokenize_seconds=round(docs.stats["tokenize_seconds"] - stats_before["tokenize_seconds"], 2),
        plan_seconds=round(packer.stats["plan_seconds"] - pack_before["plan_seconds"], 2),
        settings=dict(
            torch_threads_before=before,
            torch_threads=torch.get_num_threads(),
            data_workers=args.data_workers,
            image_workers=args.image_workers,
            image_prefetch=args.image_prefetch,
            data_prefetch=args.data_prefetch,
            tokenize_rows=args.tokenize_rows,
            omp=os.environ.get("OMP_NUM_THREADS"),
            cpus=os.cpu_count(),
        ),
    )
    docs.close()
    print(json.dumps(report), flush=True)
    if known.output:
        known.output.parent.mkdir(parents=True, exist_ok=True)
        known.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
