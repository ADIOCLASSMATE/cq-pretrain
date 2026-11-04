"""Copy an exact-frame-count subset without changing or re-encoding its source.

Keep the two public manifests and pairs byte-for-byte at the selected-row level.
Filter source annotations and integrity records to the same IDs. Verify every
copied MP4 against its original checksum and fully decode it with ffprobe.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


VARIANTS = ("960x544", "832x480")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path):
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(path))


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def write_jsonl(path, rows):
    atomic_bytes(path, b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows
    ))


def select_lines(path, predicate):
    records, lines = [], []
    with path.open("rb") as handle:
        for line in handle:
            row = json.loads(line)
            if predicate(row):
                records.append(row)
                lines.append(line)
    return records, b"".join(lines)


def ids_of(rows):
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    return ids


def copy_and_verify(source, output, variant, row, expected, frames):
    relative = Path(variant) / row["video_path"]
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid video path: {relative}")
    src, dst = source / relative, output / relative
    before = snapshot(src)
    if before["sha256"] != expected or before["bytes"] != row["output_bytes"]:
        raise ValueError(f"Source differs from committed integrity: {relative}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    copied = not dst.exists()
    if copied:
        temporary = dst.with_name(dst.name + ".partial")
        shutil.copyfile(src, temporary)
        shutil.copystat(src, temporary)
        temporary.replace(dst)
    src_stat, dst_stat = src.stat(), dst.stat()
    if dst.is_symlink() or (src_stat.st_dev, src_stat.st_ino) == (dst_stat.st_dev, dst_stat.st_ino):
        raise ValueError(f"Destination must be an independent file: {relative}")
    actual = sha256(dst)
    if actual != expected or dst_stat.st_size != row["output_bytes"]:
        raise ValueError(f"Destination checksum/size mismatch: {relative}")
    result = subprocess.run([
        "ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=codec_name,pix_fmt,width,height,nb_read_frames,avg_frame_rate",
        "-of", "json", str(dst),
    ], capture_output=True, text=True, check=True, timeout=180)
    if result.stderr.strip():
        raise ValueError(f"Decode error in {relative}: {result.stderr.strip()}")
    stream, = json.loads(result.stdout)["streams"]
    expected_fields = dict(codec_name="mpeg4", pix_fmt="yuv420p", width=row["width"],
                           height=row["height"], nb_read_frames=str(frames))
    if any(stream[key] != value for key, value in expected_fields.items()):
        raise ValueError(f"Video metadata mismatch: {relative}: {stream}")
    if Fraction(stream["avg_frame_rate"]) != Fraction(row["fps_num"], row["fps_den"]):
        raise ValueError(f"FPS mismatch: {relative}")
    after = src.stat()
    if (after.st_size, after.st_mtime_ns) != (before["bytes"], before["mtime_ns"]):
        raise ValueError(f"Source changed while copying: {relative}")
    return dict(sample_id=row["sample_id"], variant=variant, video_path=str(relative),
                output_sha256=actual, output_bytes=dst_stat.st_size,
                decoded_frames=int(stream["nb_read_frames"]), copied=copied,
                source_mtime_ns=before["mtime_ns"])


def build(args):
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate dataset directories")
    started = time.monotonic()
    source_files = [*(f"{v}/samples.jsonl" for v in VARIANTS), "pairs.jsonl",
                    "metadata/selected.jsonl", "metadata/source_integrity.jsonl",
                    "metadata/integrity.jsonl", "metadata/OpenVidHD.csv",
                    "metadata/OpenVidHD.json", "processing_config.json"]
    before = {rel: snapshot(source / rel) for rel in source_files}
    manifests, contents = {}, {}
    for variant in VARIANTS:
        rows, lines = select_lines(source / variant / "samples.jsonl",
                                   lambda row: row["num_frames"] == args.frames)
        manifests[variant] = rows
        contents[f"{variant}/samples.jsonl"] = lines
    ordered_ids = ids_of(manifests[VARIANTS[0]])
    if not ordered_ids or ordered_ids != ids_of(manifests[VARIANTS[1]]):
        raise ValueError("Variants must contain the same nonempty ordered subset")
    selected_ids = set(ordered_ids)
    for left, right in zip(*manifests.values()):
        for key in ("source", "caption", "num_frames", "fps_num", "fps_den"):
            if left[key] != right[key]:
                raise ValueError(f"Variant alignment mismatch: {left['sample_id']}: {key}")
        for row in (left, right):
            if row["validation"]["decoded_frames"] != args.frames:
                raise ValueError(f"Source validation frame count mismatch: {row['sample_id']}")
    filtered = {}
    for rel in ("pairs.jsonl", "metadata/selected.jsonl", "metadata/source_integrity.jsonl",
                "metadata/integrity.jsonl"):
        rows, lines = select_lines(source / rel, lambda row: row["sample_id"] in selected_ids)
        filtered[rel], contents[rel] = rows, lines
        if rel != "metadata/integrity.jsonl" and ids_of(rows) != ordered_ids:
            raise ValueError(f"Source manifest alignment mismatch: {rel}")
    integrity = {(r["sample_id"], r["variant"]): r for r in filtered["metadata/integrity.jsonl"]}
    if len(integrity) != len(ordered_ids) * 2 or len(integrity) != len(filtered["metadata/integrity.jsonl"]):
        raise ValueError("Incomplete or duplicate source integrity records")
    for index, pair in enumerate(filtered["pairs.jsonl"]):
        for variant in VARIANTS:
            row = manifests[variant][index]
            if pair["variants"][variant] != f'{variant}/{row["video_path"]}' or pair["caption"] != row["caption"]:
                raise ValueError(f"Invalid pair: {pair['sample_id']}")

    required_bytes = sum(row["output_bytes"] for rows in manifests.values() for row in rows)
    print(json.dumps(dict(selected_samples=len(ordered_ids), video_files=len(ordered_ids) * 2,
                          video_bytes=required_bytes, output=str(output))), flush=True)
    if not args.resume:
        if shutil.disk_usage(output.parent).free < required_bytes + 1024 ** 3:
            raise ValueError("Insufficient space for independent copies")
        output.mkdir(exist_ok=False)
    else:
        existing = json.loads((output / "subset_config.json").read_text())
        if existing["source_dataset"] != str(source) or existing["num_frames"] != args.frames:
            raise ValueError("Resume configuration does not match")
    run_id = f"openvidhd-subset-{args.frames}frames-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    write_json(output / "subset_config.json", dict(source_dataset=str(source), num_frames=args.frames,
               selection="existing num_frames equals requested frames in both variants",
               copy_mode="independent byte-identical files", reencode=False, truncate=False,
               candidates=len(ordered_ids), variants=list(VARIANTS), created_at=now(), run_id=run_id))
    write_json(output / "run.json", dict(state="running", started_at=now(), run_id=run_id))

    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for variant, rows in manifests.items():
            for row in rows:
                expected = integrity[row["sample_id"], variant]["output_sha256"]
                future = pool.submit(copy_and_verify, source, output, variant, row, expected, args.frames)
                futures[future] = (row["sample_id"], variant)
        for future in as_completed(futures):
            sample_id, variant = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(dict(sample_id=sample_id, variant=variant, error=str(exc)))
            count = len(completed) + len(failures)
            if count % 100 == 0 or count == len(futures):
                progress = dict(updated_at=now(), candidates=len(ordered_ids),
                                verified_video_files=len(completed), failed_video_files=len(failures),
                                total_video_files=len(futures), pending_video_files=len(futures)-count,
                                elapsed_seconds=round(time.monotonic()-started, 1))
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    write_jsonl(output / "failures.jsonl", failures)
    write_jsonl(output / "metadata/copy_integrity.jsonl",
                sorted(completed, key=lambda row: (row["sample_id"], row["variant"])))
    if failures:
        write_json(output / "run.json", dict(state="failed", finished_at=now(), run_id=run_id))
        raise RuntimeError(f"{len(failures)} copy/verification failures; see failures.jsonl")

    # Publish only complete, verified paired manifests, preserving the source lines.
    for rel, content in contents.items():
        atomic_bytes(output / rel, content)
    with (source / "metadata/OpenVidHD.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        with (output / "metadata/OpenVidHD.csv").open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            csv_ids = []
            for row in reader:
                if row["video"] in selected_ids:
                    writer.writerow(row)
                    csv_ids.append(row["video"])
    if len(csv_ids) != len(selected_ids) or set(csv_ids) != selected_ids:
        raise ValueError("CSV annotation coverage mismatch")
    groups = []
    index_ids = []
    for group in json.loads((source / "metadata/OpenVidHD.json").read_text()):
        selected_group = {}
        for part, names in group.items():
            names = [name for name in names if name in selected_ids]
            if names:
                selected_group[part] = names
                index_ids.extend(names)
        if selected_group:
            groups.append(selected_group)
    if len(index_ids) != len(selected_ids) or set(index_ids) != selected_ids:
        raise ValueError("ZIP index coverage mismatch")
    write_json(output / "metadata/OpenVidHD.json", groups)
    original_config = json.loads((source / "processing_config.json").read_text())
    write_json(output / "metadata/source_processing_config.json", original_config)
    config = dict(schema_version=original_config["schema_version"], created_at=now(), pilot=False,
                  candidates=len(ordered_ids), run_id=run_id, transform=original_config["transform"],
                  source_dataset=str(source), source_run_id=original_config["run_id"],
                  operation="copy exact-frame subset; transform describes upstream video encoding",
                  subset_num_frames=args.frames)
    write_json(output / "processing_config.json", config)

    expected_paths = {f'{variant}/{row["video_path"]}' for variant, rows in manifests.items() for row in rows}
    actual_paths = {str(path.relative_to(output)) for variant in VARIANTS
                    for path in (output / variant / "videos").rglob("*.mp4")}
    if actual_paths != expected_paths:
        raise ValueError("Copied video set differs from manifests")
    after = {rel: snapshot(source / rel) for rel in source_files}
    if before != after:
        raise ValueError("Source dataset metadata changed during subset creation")
    write_json(output / "metadata/subset_integrity.json", dict(
        script_sha256=sha256(Path(__file__)), source_metadata=after,
        source_metadata_unchanged=True,
        output_metadata_sha256={rel: sha256(output / rel) for rel in contents},
    ))
    variants = {variant: dict(success=len(rows), failure=0, pending=0,
                             output_bytes=sum(row["output_bytes"] for row in rows),
                             output_frames=len(rows)*args.frames)
                for variant, rows in manifests.items()}
    summary = dict(updated_at=now(), candidates=len(ordered_ids), success=len(ordered_ids),
                   failure=0, pending=0, variants=variants, output_bytes=required_bytes,
                   committed_video_files=len(completed), processing_finished=True,
                   all_candidates_succeeded=True, output_frames=len(completed)*args.frames,
                   worker_errors=[], run_id=run_id)
    write_json(output / "summary.json", summary)
    validation = dict(updated_at=now(), all_passed=True, verified=len(completed), run_id=run_id,
                      verification_mode="independent copy; committed checksum match; full ffprobe decode",
                      num_frames=args.frames, source_metadata_unchanged=True,
                      original_captions_ids_and_manifest_rows_preserved=True,
                      paired_variants_aligned=True, source_correspondence=False)
    write_json(output / "validation.json", validation)
    write_json(output / "commit_validation.json", validation)
    write_json(output / "run.json", dict(state="succeeded", finished_at=now(), run_id=run_id,
                                         elapsed_seconds=round(time.monotonic()-started, 1)))
    atomic_bytes(output / "README.md", (
        f"# OpenVidHD {args.frames}-frame subset\n\n"
        f"Source: {source}\n\n"
        f"Contains {len(ordered_ids):,} paired samples, with {args.frames} existing frames per video.\n"
        "Videos are independent byte-identical copies: no truncation, resampling, or re-encoding.\n"
        "Both variants retain their original sample_id, filename, caption, FPS, metadata, and order.\n"
        "Each variant's samples.jsonl uses paths relative to that variant directory; pairs.jsonl uses root-relative paths.\n"
        "metadata/OpenVidHD.csv, OpenVidHD.json, selected.jsonl, and integrity records contain only selected IDs.\n"
        "CSV frame/fps/seconds and source.annotation describe the ORIGINAL source video, not the processed MP4.\n"
        "processing_config.json records the upstream transform and this subset operation.\n"
        "The parent processing configuration is retained in metadata/source_processing_config.json.\n"
        "Parent worker state, scratch files, STOP markers, and distributed logs are not part of this new run.\n"
        "Checksums are stored only in separate metadata/*integrity* files.\n"
        "validation.json records full decoding and checksum checks for every copied MP4.\n"
    ).encode())
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
