"""Explain exactly why one published row fails its upstream occurrence check.

Replays data.captions.select_image_caption for a single row, step by step, so
the failure can be attributed to a specific group member, owner key and image
occurrence instead of inferred. Read-only.

    PYTHONPATH=<source> python triage_one_row.py --root <release> \
        --rank 22 --world 32 --seed 42 --block 14527 --offset 2
"""

import argparse
import hashlib
import json
import traceback


def slots_of(row):
    """The image occurrences of a flat row, as the constraint sees them."""
    from data.loss_permissions import occurrence_images

    out = []
    for occurrence in [s for s in row.get("segments", []) if s["kind"] == "image"]:
        try:
            image, _media = occurrence_images(row, occurrence)
            out.append(
                {
                    "exact_identity": image.get("exact_identity"),
                    "sha256": image.get("sha256"),
                    "generation_target": occurrence.get("generation_target"),
                }
            )
        except Exception as error:
            out.append({"error": f"{type(error).__name__}: {error}"})
    return out


def explain_flat(row, seed, epoch):
    """For a flat row with an upstream binding: published slots vs every context."""
    from data.upstream import origin_key, tie, CONTRACT

    report = {"sample_id": row.get("sample_id"), "source": row.get("source"), "format": row.get("format")}
    report["published_slots"] = slots_of(row)
    binding = row.get("_upstream_binding")
    if not binding:
        report["binding"] = None
        return report
    report["owners"] = binding.get("owners")
    contexts = []
    for key, index in binding["owners"].items():
        context = binding["contexts"][index]
        merged = dict(row)
        merged.update(context)
        contexts.append({"owner": key, "index": index, "slots": slots_of(merged)})
    report["contexts"] = contexts
    # Which candidate and owner the deterministic draw selects.
    encoded = row.get("caption_candidates_json")
    if encoded:
        candidates = json.loads(encoded)
        key = f"caption_group_v1:{int(seed)}:{int(epoch)}:{row['sample_id']}".encode()
        report["candidate_count"] = len(candidates)
        try:
            chosen = candidates[
                int.from_bytes(
                    hashlib.sha256(f"{CONTRACT}:{int(seed)}:{int(epoch)}:{row['sample_id']}".encode()).digest(),
                    "big",
                )
                % len(candidates)
            ]
            raw = chosen["provenance"]
            origins = raw if isinstance(raw, list) else [raw]
            distinct = {origin_key(o): o for o in origins}
            report["drawn_origin_keys"] = sorted(distinct)
            report["drawn_owner"] = tie(sorted(distinct), row["sample_id"], seed, epoch, "caption-origin")
        except Exception:
            report["draw_failed"] = traceback.format_exc(limit=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--block-rows", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()

    import pyarrow.compute as pc

    from data import captions
    from data.catalog import permute
    from data.sources import open_catalog

    catalog = open_catalog(args.root, None, args.block_rows, 128 << 20)
    units = catalog.meta["units"]
    index = args.rank + args.block * args.world
    shard, start, _ = catalog.block(permute(index, units, args.seed, args.epoch))
    row = catalog.read_rows(shard, start + args.offset, 1)[0]

    report = {
        "sample_id": row.get("sample_id"),
        "keys": sorted(k for k in row if not k.startswith("_")),
        "is_image_group": "caption_members" in row,
    }
    if "caption_members" in row:
        members = row["caption_members"]
        active = pc.indices_nonzero(members.field("eligible")).to_numpy()
        key = f"openuni-image-v1:{args.seed}:{args.epoch}:{row['sample_id']}".encode()
        picked = int(active[int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(active)])
        report["eligible_members"] = int(len(active))
        report["picked_member_index"] = picked
        choice = members[picked].as_py()
        report["picked_member"] = {
            "sample_id": choice.get("sample_id"),
            "subset": choice.get("subset"),
            "field": choice.get("field"),
            "has_extra": bool(choice.get("extra")),
        }
        if choice.get("extra"):
            inner = json.loads(choice["extra"])
            inner["caption_candidates_json"] = json.dumps(
                inner.pop("_caption_options_for_training"), ensure_ascii=False
            )
            inner["images"] = [dict(ref, _dataset_roots=row["_dataset_roots"]) for ref in inner["images"]]
            report["inner"] = explain_flat(inner, args.seed, args.epoch)
            try:
                captions.select(dict(inner), args.seed, args.epoch)
                report["inner_select"] = "ok"
            except Exception as error:
                report["inner_select"] = f"{type(error).__name__}: {error}"
    else:
        report["flat"] = explain_flat(row, args.seed, args.epoch)

    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text, flush=True)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
