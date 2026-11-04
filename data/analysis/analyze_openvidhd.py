"""Full census of the mounted OpenVidHD directory, with explicit scope checks."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from analyze_openvid import DEFAULT_TOKENIZER, aggregate, metadata, sha256
from mp4_metadata import SplitFile
from scan_videos import discover, run as scan, write_json

BASE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("/inspire/dataset/openvid-1m/v2/OpenVidHD")
DEFAULT_CSV = DEFAULT_ROOT.parent / "data/train/OpenVidHD.csv"
DEFAULT_RESULTS = BASE / "results/OpenVidHD"


def inventory(root, source, results):
    """Reconcile every actual archive member against both supplied manifests."""
    archives = discover(root)
    declaration_path = root / "OpenVidHD.json"
    declaration = json.loads(declaration_path.read_text())
    expected = {}
    for item in declaration:
        for key, names in item.items():
            index = int(key.removeprefix("part"))
            if index in expected or len(names) != len(set(names)):
                raise ValueError(f"Duplicate declaration in {key}")
            expected[index] = set(names)
    if set(expected) != {i for i, _ in archives}:
        raise ValueError("Declared and mounted archive groups differ")

    rows, archive_rows, extra_members = [], [], 0
    for index, parts in archives:
        with SplitFile(parts) as handle, zipfile.ZipFile(handle) as archive:
            infos = [i for i in archive.infolist() if not i.is_dir()]
            videos = [i for i in infos if i.filename.lower().endswith(".mp4")]
            names = [Path(i.filename).name for i in videos]
            if len(names) != len(set(names)) or set(names) != expected[index]:
                raise ValueError(f"Archive {index} differs from OpenVidHD.json")
            extra_members += len(infos) - len(videos)
            for info in videos:
                rows.append(dict(video=Path(info.filename).name, archive=index,
                                 member=info.filename, bytes=info.file_size,
                                 compressed_bytes=info.compress_size, zip_crc32=f"{info.CRC:08x}"))
            archive_rows.append(dict(archive=index, videos=len(videos),
                                     parts=len(parts), bytes=sum(Path(p).stat().st_size for p in parts)))
    df = pd.DataFrame(rows)
    df.to_csv(results / "inventory.csv", index=False)
    source_names = pd.read_csv(source, usecols=["video"])
    if df.video.duplicated().any() or source_names.video.duplicated().any():
        raise ValueError("Video names are not globally unique")
    joined = df[["video"]].merge(source_names, on="video", how="outer", validate="one_to_one", indicator=True)
    errors = joined[joined._merge != "both"]
    errors.to_csv(results / "inventory_membership_errors.csv", index=False)
    if (errors._merge == "right_only").any():
        raise ValueError("OpenVidHD.csv refers to missing videos; see inventory_membership_errors.csv")
    missing_caption = errors[errors._merge == "left_only"][["video"]]
    missing_caption.merge(df, on="video", validate="one_to_one").to_csv(results / "missing_caption_videos.csv", index=False)
    result = dict(complete=True, root=str(root), source=str(source),
                  declaration=str(declaration_path), declaration_sha256=sha256(declaration_path),
                  population=len(df), csv_rows=len(source_names), archive_groups=len(archives),
                  archive_files=sum(len(p) for _, p in archives),
                  non_video_archive_members=extra_members,
                  all_archive_members_match_json=True, all_video_names_match_csv=len(errors) == 0,
                  missing_caption_videos=len(missing_caption), all_csv_names_exist_in_archives=True,
                  video_bytes=int(df.bytes.sum()), archives=archive_rows)
    write_json(results / "inventory_summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "archives"}), flush=True)


def validate_scope(results):
    inventory_summary = json.loads((results / "inventory_summary.json").read_text())
    summary = json.loads((results / "summary.json").read_text())
    manifest = json.loads((results / "archive_manifest.json").read_text())
    assert summary["population"] == inventory_summary["population"]
    assert summary["source"] == inventory_summary["source"]
    assert sha256(Path(summary["source"])) == summary["source_sha256"]
    assert summary["video_root"] == inventory_summary["root"]
    root = Path(summary["video_root"])
    assert all(Path(p).parent == root for _, parts in manifest["archives"] for p in parts)
    columns = ["video", "archive", "member", "bytes"]
    expected = pd.read_csv(results / "inventory.csv", usecols=columns)
    actual = pd.read_csv(results / "all_video_metrics.csv", usecols=columns)
    pd.testing.assert_frame_equal(expected.sort_values("video").reset_index(drop=True)[columns],
                                  actual.sort_values("video").reset_index(drop=True)[columns])
    assert summary["caption_population"] == inventory_summary["csv_rows"]
    assert summary["missing_caption_videos"] == inventory_summary["missing_caption_videos"]
    write_json(results / "scope_validation.json", dict(passed=True, population=len(actual),
               root=str(root), video_inventory_matches_metrics=True, source_sha256_matches=True,
               all_archive_parts_inside_selected_directory=True,
               caption_population=summary["caption_population"],
               missing_caption_videos=summary["missing_caption_videos"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["all", "inventory", "metadata", "scan", "aggregate", "plots", "validate"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "inventory"}:
        inventory(args.root, args.metadata_csv, args.results)
    if args.stage in {"all", "metadata"}:
        metadata(args.root, args.results, args.tokenizer, args.metadata_csv)
    if args.stage in {"all", "scan"}:
        scan(args.root, args.results, args.workers)
    if args.stage in {"all", "aggregate"}:
        aggregate(args.results, allow_missing_metadata=True)
    if args.stage in {"all", "plots"}:
        from plot_statistics import make_plots
        make_plots(args.results)
    if args.stage in {"all", "validate"}:
        from validate_results import validate
        validate_scope(args.results)
        validate(args.results)


if __name__ == "__main__":
    main()
