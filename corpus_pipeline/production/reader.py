"""Read a complete immutable release chain as original records plus sidecars."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .publish import materialize, release_contents, validate_release


def iter_training_records(release_path, *, epoch_seed=0, caption_length=None, revocations=()):
    path = Path(release_path)
    manifest = json.loads((path if path.is_file() else path / "manifest.json").read_text())
    if manifest.get("schema_version") == "production.staged_dataset.v2":
        from .staged_publish import iter_dataset_records
        yield from iter_dataset_records(release_path, epoch_seed=epoch_seed, caption_length=caption_length,
                                        revocations=revocations)
        return
    if manifest.get("schema_version") == "production.staged_reference_release.v1":
        from .staged_publish import iter_staged_records
        yield from iter_staged_records(release_path, epoch_seed=epoch_seed, caption_length=caption_length,
                                       revocations=revocations)
        return
    if revocations:
        raise ValueError("legacy releases do not support staged revocations")
    assets, annotations, views, _ = release_contents(release_path)
    for view in views:
        yield materialize(view, annotations[view["annotation_id"]], assets,
                          epoch_seed=epoch_seed, caption_length=caption_length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--materialize", action="store_true", help="read all original assets for an aggregate dataset")
    args = parser.parse_args()
    path = Path(args.release)
    manifest = json.loads((path if path.is_file() else path / "manifest.json").read_text())
    if manifest.get("schema_version") == "production.staged_dataset.v2":
        from .staged_publish import validate_dataset_release
        result = validate_dataset_release(args.release, materialize=args.materialize)
    elif manifest.get("schema_version") == "production.staged_reference_release.v1":
        from .staged_publish import validate_staged_release
        result = validate_staged_release(args.release)
    else:
        result = validate_release(args.release)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
