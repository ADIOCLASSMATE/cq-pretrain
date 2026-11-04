"""Validate the audited dual-resolution pilot and prove an intact resume skips all files."""
import argparse
import json
from pathlib import Path
import subprocess

from .manifest import atomic_json, read_jsonl, require_readable, sha256
from .openvidhd import VARIANTS, check_pilot, load_config, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    config = load_config(root)
    check_pilot(root, config)
    selected = read_jsonl(root / "metadata/selected.jsonl")
    expected_ids = [r["sample_id"] for r in selected]
    videos = []
    for variant in VARIANTS:
        records = read_jsonl(root / variant / "samples.jsonl")
        if [r["sample_id"] for r in records] != expected_ids:
            raise ValueError("Variant sample IDs/order differ from selection")
        videos.extend(root / variant / r["video_path"] for r in records)

    immutable = [
        root / "processing_config.json", root / "metadata/selected.jsonl",
        root / "metadata/source_integrity.jsonl", root / "metadata/run_integrity.json",
        *[root / name for name in config["_integrity"]["validation_files"]],
        *sorted((root / "_state").glob("*.jsonl")),
        *sorted((root / "metadata/integrity_parts").glob("*.jsonl")),
    ]
    baseline = {}
    for path in [*immutable, *videos]:
        stat = path.stat()
        baseline[str(path.relative_to(root))] = dict(
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(path))
    # Technical checksums stay in this auxiliary file, outside readable manifests.
    atomic_json(root / "metadata/resume_baseline.json", baseline)
    repository = Path(__file__).resolve().parents[2]
    with (root / "logs/resume_validation.log").open("w") as log:
        subprocess.run(["bash", "data/preprocess/run.sh", "process", "--output", str(root),
                        "--workers", "4"], cwd=repository, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    for relative, expected in baseline.items():
        path = root / relative
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Intact resume changed file contents: {relative}")
        if path in videos and path.stat().st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Intact resume rewrote MP4: {relative}")
    check_pilot(root, load_config(root))
    report = dict(updated_at=now(), run_id=config["run_id"], all_passed=True,
                  source_samples=len(selected), verified_video_files=len(videos),
                  variant_ids_and_order_match=True, video_contents_unchanged=True,
                  video_mtimes_unchanged=True, journals_unchanged=True,
                  readable_manifests_unchanged=True, prior_source_audit_still_valid=True)
    require_readable(report)
    atomic_json(root / "resume_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
