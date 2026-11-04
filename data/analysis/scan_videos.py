"""Resumable, complete metadata/timing census of OpenVid ZIP members."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

from mp4_metadata import SplitFile, probe

DEFAULT_ROOT = Path("/inspire/dataset/openvid-1m/v2")
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def discover(root):
    groups = {}
    for path in root.glob("OpenVid*part*"):
        match = (re.fullmatch(r"OpenVid_part(\d+)(?:\.zip|_part[a-z]+)", path.name)
                 or re.fullmatch(r"OpenVidHD_part_(\d+)(?:\.zip|_part_[a-z]+)", path.name))
        if match:
            groups.setdefault(int(match[1]), []).append(path)
    result = []
    for index, paths in sorted(groups.items()):
        whole = [p for p in paths if p.suffix == ".zip"]
        parts = whole or sorted(paths)
        result.append((index, [str(p) for p in parts]))
    if not result:
        raise ValueError(f"No OpenVid archives found: {root}")
    return result


def scan_archive(index, parts, directory, parser_hash):
    directory = Path(directory)
    destination = directory / f"part_{index:03d}.jsonl"
    status = directory / f"part_{index:03d}.status.json"
    identity = {"parser_sha256": parser_hash, "parts": [
        {"path": p, "bytes": Path(p).stat().st_size, "mtime_ns": Path(p).stat().st_mtime_ns}
        for p in parts]}
    if destination.exists() and status.exists():
        previous = json.loads(status.read_text())
        if previous.get("identity") == identity and previous.get("complete"):
            return previous
    count = errors = 0
    start = time.monotonic()
    partial = destination.with_suffix(".jsonl.partial")
    with SplitFile(parts) as source, zipfile.ZipFile(source) as archive, partial.open("w") as output:
        members = sorted((i for i in archive.infolist() if i.filename.lower().endswith(".mp4")),
                         key=lambda i: i.header_offset)
        for info in members:
            row = dict(video=Path(info.filename).name, archive=index,
                       member=info.filename, bytes=info.file_size)
            try:
                with archive.open(info) as handle:
                    row.update(probe(handle, info.file_size), status="ok")
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                errors += 1
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            if count % 500 == 0:
                output.flush()
                write_json(status, dict(archive=index, count=count, total=len(members),
                           errors=errors, complete=False, identity=identity,
                           elapsed_seconds=round(time.monotonic() - start, 2)))
    partial.replace(destination)
    result = dict(archive=index, count=count, total=count, errors=errors, complete=True,
                  identity=identity, elapsed_seconds=round(time.monotonic() - start, 2))
    write_json(status, result)
    return result


def run(root, results, workers):
    directory = results / "video_shards"
    directory.mkdir(parents=True, exist_ok=True)
    archives = discover(root)
    parser_hash = hashlib.sha256(Path(__file__).with_name("mp4_metadata.py").read_bytes()).hexdigest()
    manifest_path = results / "archive_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["archives"] != [[i, p] for i, p in archives]:
            raise ValueError("Output directory belongs to a different archive population; use a separate results directory")
    write_json(manifest_path, {"archives": archives, "parser_sha256": parser_hash, "root": str(root)})
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_archive, i, p, str(directory), parser_hash) for i, p in archives}
        while futures:
            done, futures = concurrent.futures.wait(futures, timeout=15,
                                                    return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
            states = [json.loads(p.read_text()) for p in directory.glob("*.status.json")]
            progress = dict(archives_total=len(archives), archives_complete=sum(s["complete"] for s in states),
                            videos_scanned=sum(s["count"] for s in states),
                            errors=sum(s["errors"] for s in states),
                            elapsed_seconds=round(time.monotonic() - start, 2))
            write_json(results / "scan_progress.json", progress)
            print(json.dumps(progress), flush=True)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    run(args.root, args.results, args.workers)
