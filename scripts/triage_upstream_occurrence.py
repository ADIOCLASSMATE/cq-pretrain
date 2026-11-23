"""Find the document that makes a resumed rank fail its upstream occurrence check.

The 32-GPU L40S run restarts from checkpoint 800 and dies at step 824 every
time, on rank 22, in data.upstream.constrain_bound_occurrences. The data cursor
is deterministic, so the offending row is reached again on every attempt.

This walks that rank's own shuffled blocks forward from a saved cursor and runs
exactly the published row preparation (data.sources.training_rows), one block at
a time. A block that raises is then replayed row by row to name the sample and
print what the check compared. Read-only: nothing here writes to the corpus.

    PYTHONPATH=<source> python triage_upstream_occurrence.py \
        --root <release> --rank 22 --world 32 --seed 42 --block 14246 --blocks 600
"""

import argparse
import json
import traceback


def describe(row, sample_id):
    """What the constraint saw, for the row that failed it."""
    from data.loss_permissions import occurrence_images
    from data.upstream import origin_key

    binding = row.get("_upstream_binding") or {}
    published_images = [s for s in row["segments"] if s["kind"] == "image"]
    out = {
        "sample_id": sample_id,
        "source": row.get("source"),
        "format": row.get("format"),
        "published_image_occurrences": len(published_images),
        "owners": list(binding.get("owners", {})),
        "contexts": len(binding.get("contexts", [])),
    }
    published_slots = []
    for occurrence in published_images:
        logical, _ = occurrence_images(row, occurrence)
        published_slots.append({"exact_identity": logical.get("exact_identity"), "sha256": logical.get("sha256")})
    out["published_slots"] = published_slots
    contexts = []
    for key, index in binding.get("owners", {}).items():
        context = binding["contexts"][index]
        bound = dict(row)
        bound.update(context)
        slots = []
        for occurrence in [s for s in context["segments"] if s["kind"] == "image"]:
            try:
                image, _ = occurrence_images(bound, occurrence)
                slots.append({"exact_identity": image.get("exact_identity"), "sha256": image.get("sha256")})
            except Exception as error:  # the context itself may be malformed
                slots.append({"error": f"{type(error).__name__}: {error}"})
        contexts.append({"owner": key, "index": index, "image_occurrences": len(slots), "slots": slots})
    out["bound_contexts"] = contexts
    # Which owner key the deterministic draw actually picks is what decides the
    # failure; a row can be fine for one owner and broken for another.
    try:
        out["origin_keys_ok"] = [origin_key(o) for o in []] or None
    except Exception:
        pass
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--block", type=int, required=True, help="Local block index to start from")
    parser.add_argument("--blocks", type=int, default=600)
    parser.add_argument("--block-rows", type=int, default=4)
    parser.add_argument("--output")
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

    failures = []
    scanned = 0
    for local in range(args.block, args.block + args.blocks):
        index = args.rank + local * args.world
        if index >= units:
            break
        shard, start, count = catalog.block(permute(index, units, args.seed, args.epoch))
        try:
            training_rows(catalog, shard, start, count, signature)
            scanned += count
            continue
        except Exception as error:
            print(f"block {local}: {type(error).__name__}: {error}", flush=True)
        # Narrow to the row. read_rows is the same call training_rows makes.
        for offset in range(count):
            rows = catalog.read_rows(shard, start + offset, 1)
            row = rows[0]
            sample_id = row.get("sample_id")
            try:
                training_rows(catalog, shard, start + offset, 1, signature)
            except Exception as error:
                detail = {"block": local, "row_offset": offset, "error": f"{type(error).__name__}: {error}"}
                try:
                    detail.update(describe(catalog.read_rows(shard, start + offset, 1)[0], sample_id))
                except Exception:
                    detail["describe_failed"] = traceback.format_exc(limit=3)
                failures.append(detail)
                print(json.dumps(detail, ensure_ascii=False)[:4000], flush=True)
        scanned += count
    print(json.dumps(dict(scanned_rows=scanned, failures=len(failures))), flush=True)
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(dict(scanned_rows=scanned, failures=failures), handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
