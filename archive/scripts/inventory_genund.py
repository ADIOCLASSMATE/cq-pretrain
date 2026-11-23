#!/usr/bin/env python3
"""Bounded directory/file inventory, without opening data files or extrapolation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys


DEFAULT_ROOT = "/inspire/dataset/gen-und-mm-pretrain/01"
IGNORED_DIRS = {".git", ".cache", "__pycache__", ".ipynb_checkpoints"}
META = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt", ".caption"}
DATA = META | {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif",
               ".tar", ".tgz", ".gz", ".bz2", ".xz", ".zst", ".zip",
               ".parquet", ".arrow", ".feather", ".h5", ".hdf5", ".lmdb"}


def is_metadata(name):
    return Path(name).suffix.lower() in META and not name.lower().startswith(
        ("readme", "license", "licence", "notice", "citation", "requirements"))


def scan(path, limit, details=True):
    """Inspect at most limit entries, with one name-only lookahead for truncation."""
    result = dict(path=str(path), entries_examined=0, truncated=False, errors=[],
                  direct_subdirectories=[], files=[], symlinks=[], file_count=0,
                  directory_count=0, symlink_count=0, suffix_counts={}, metadata_candidates=[])
    suffixes = Counter()
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= limit:
                    result["truncated"] = True
                    break
                result["entries_examined"] += 1
                try:
                    if entry.is_symlink():
                        result["symlink_count"] += 1
                        if details:
                            result["symlinks"].append(entry.name)
                    elif entry.is_dir(follow_symlinks=False):
                        result["directory_count"] += 1
                        if details:
                            result["direct_subdirectories"].append(entry.name)
                    elif entry.is_file(follow_symlinks=False):
                        suffix = Path(entry.name).suffix.lower() or "<none>"
                        result["file_count"] += 1
                        suffixes[suffix] += 1
                        if details:
                            result["files"].append(dict(name=entry.name, suffix=suffix,
                                                       bytes=entry.stat(follow_symlinks=False).st_size))
                        if is_metadata(entry.name) and len(result["metadata_candidates"]) < 12:
                            result["metadata_candidates"].append(str(path / entry.name))
                except OSError as exc:
                    if len(result["errors"]) < 5:
                        result["errors"].append(f"{entry.name}: {exc}")
    except OSError as exc:
        result["errors"].append(str(exc))
    result["suffix_counts"] = dict(sorted(suffixes.items()))
    result["complete_immediate_count"] = not result["truncated"] and not result["errors"]
    result["direct_subdirectories"].sort()
    result["files"].sort(key=lambda item: item["name"])
    return result


def inventory_dataset(dataset, args):
    cache, visits = {}, Counter()
    rng = random.Random(args.seed)

    def listing(path):
        if path not in cache:
            cache[path] = scan(path, args.max_entries)
        return cache[path]

    direct = listing(dataset)
    normalized = dataset.name.lower().replace("_", "-")
    subset_parents = []
    if "salmon" in normalized or "finevision" in normalized or "llava" in normalized:
        subset_parents.append(dataset)
    if "infinity-mm" in normalized:
        subset_parents.extend(dataset / name for name in direct["direct_subdirectories"]
                              if "stage" in name.lower())
    if normalized in {"openuni", "uniworld-v1-new"} or "llava" in normalized:
        subset_parents.extend(dataset / name for name in direct["direct_subdirectories"]
                              if name.lower() in {"data", "datasets"})
    subset_listings = []
    for parent in dict.fromkeys(subset_parents):
        info = listing(parent)
        subset_listings.append({key: info[key] for key in (
            "path", "direct_subdirectories", "directory_count", "file_count",
            "suffix_counts", "metadata_candidates", "truncated", "errors",
            "complete_immediate_count")})

    sampled_paths, leaves, selected_endpoints = [], {}, set()
    for _ in range(args.sample_paths):
        current, chain, reason = dataset, [], "empty_or_unrecognized_directory"
        endpoint = current
        for depth in range(args.max_depth + 1):
            info = listing(current)
            chain.append(dict(path=str(current), entries_examined=info["entries_examined"],
                              truncated=info["truncated"], errors=info["errors"]))
            directories = [current / name for name in info["direct_subdirectories"]
                           if name not in IGNORED_DIRS]
            files = [current / item["name"] for item in info["files"]
                     if item["suffix"] in DATA and (item["suffix"] not in META or is_metadata(item["name"]))]
            choices = [(path, True) for path in directories] + [(path, False) for path in files]
            if not choices:
                endpoint = current
                break
            rng.shuffle(choices)
            choices.sort(key=lambda item: visits[item[0]])
            endpoint, is_directory = choices[0]
            visits[endpoint] += 1
            if not is_directory:
                reason = "physical_file_selected"
                break
            if depth >= args.max_depth:
                endpoint, reason = current, "depth_limit"
                break
            current = endpoint
        if endpoint in selected_endpoints:
            continue
        selected_endpoints.add(endpoint)
        if current not in leaves:
            count = scan(current, args.leaf_max_entries, details=False)
            for key in ("files", "direct_subdirectories", "symlinks"):
                count.pop(key)
            count["count_scope"] = "Immediate entries in this sampled directory only; no recursive count"
            count["is_leaf_directory"] = count["directory_count"] == 0 and count["complete_immediate_count"]
            leaves[current] = count
        sampled_paths.append(dict(selected_path=str(endpoint), sampled_directory=str(current),
                                  stop_reason=reason, traversal=chain))
    return dict(dataset=dataset.name, path=str(dataset), direct_listing=direct,
                subset_listings=subset_listings, sampled_paths=sampled_paths,
                sampled_directories=list(leaves.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=2000)
    parser.add_argument("--leaf-max-entries", type=int, default=50000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--sample-paths", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not (1 <= args.max_entries <= 2000 and 1 <= args.leaf_max_entries <= 50000
            and args.max_depth >= 0 and 1 <= args.sample_paths <= 3):
        parser.error("Limits: max-entries 1..2000, leaf-max-entries 1..50000, max-depth >=0, sample-paths 1..3")
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("--output must be outside ROOT")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if not root.is_dir():
        parser.error(f"Dataset root is unavailable: {root}")
    root_listing = scan(root, args.max_entries)
    datasets = [root / name for name in root_listing["direct_subdirectories"] if name not in IGNORED_DIRS]
    report = dict(root=str(root), checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Directory/file evidence only; symlinks are not followed. Counts never estimate corpus totals.",
                  limits={key: getattr(args, key) for key in (
                      "max_entries", "leaf_max_entries", "max_depth", "sample_paths", "seed")},
                  root_listing=root_listing, dataset_count=len(datasets), results=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        # Flush each completed dataset so remote progress survives long-running scans.
        prefix = json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False)
        stream.write(prefix[:-1] + ', "results": [\n')
        for index, dataset in enumerate(datasets, 1):
            print(f"[{index}/{len(datasets)}] inventory {dataset.name}", file=sys.stderr, flush=True)
            result = inventory_dataset(dataset, args)
            if index > 1:
                stream.write(',\n')
            json.dump(result, stream, ensure_ascii=False)
            stream.flush()
            print(f"[{index}/{len(datasets)}] done {dataset.name}: {len(result['sampled_paths'])} sampled paths", file=sys.stderr, flush=True)
        stream.write('\n]}\n')
    print(str(output))


if __name__ == "__main__":
    main()
