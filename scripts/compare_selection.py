"""Digest what the published row preparation selects, for an old/new comparison.

Run once against each data package and diff the outputs. A fix to caption or
upstream binding selection must leave every row that already prepared
successfully byte-identical; only rows that used to raise may change.

    PYTHONPATH=<package>:<source> python compare_selection.py \
        --root <release> --rank 22 --world 32 --block 14246 --blocks 400 --output old.jsonl
"""

import argparse
import hashlib
import json


def digest(row):
    """A stable summary of everything the selection decides for one document."""
    payload = {
        "sample_id": row.get("sample_id"),
        "source": row.get("source"),
        "caption_choice": row.get("caption_choice"),
        "upstream_caption_choice": row.get("upstream_caption_choice"),
        "selected_upstream": row.get("selected_upstream"),
        "segments": [
            {
                "kind": s.get("kind"),
                "text": s.get("text"),
                "image_index": s.get("image_index"),
                "generation_target": s.get("generation_target"),
            }
            for s in row.get("segments", [])
        ],
        "images": [
            {"exact_identity": i.get("exact_identity"), "sha256": i.get("sha256")} for i in row.get("images", [])
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest(), payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=400)
    parser.add_argument("--block-rows", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from data import captions, task_groups
    from data.catalog import permute
    from data.sources import open_catalog, training_rows

    catalog = open_catalog(args.root, None, args.block_rows, 128 << 20)
    units = catalog.meta["units"]
    signature = dict(seed=args.seed, epoch=args.epoch)
    if catalog.meta.get("caption_sampling"):
        signature["caption_sampling"] = captions.CONTRACT
    if catalog.meta.get("task_sampling"):
        signature["task_sampling"] = task_groups.CONTRACT

    rows_seen = errors = 0
    with open(args.output, "w") as handle:
        for local in range(args.block, args.block + args.blocks):
            index = args.rank + local * args.world
            if index >= units:
                break
            shard, start, count = catalog.block(permute(index, units, args.seed, args.epoch))
            for offset in range(count):
                record: dict[str, object] = {"block": local, "offset": offset}
                try:
                    prepared = training_rows(catalog, shard, start + offset, 1, signature)
                    record["digest"], record["payload"] = digest(prepared[0])
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                    errors += 1
                rows_seen += 1
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(dict(rows=rows_seen, errors=errors, output=args.output)), flush=True)


if __name__ == "__main__":
    main()
