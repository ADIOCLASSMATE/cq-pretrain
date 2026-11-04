"""Recheck every nonconstant record with FFmpeg-compatible signed CTTS offsets."""
import hashlib
import json
import zipfile
from pathlib import Path

from mp4_metadata import SplitFile, probe
from scan_videos import DEFAULT_RESULTS, write_json


def refresh(results=DEFAULT_RESULTS):
    manifest = json.loads((results / "archive_manifest.json").read_text())
    parser_hash = hashlib.sha256(Path(__file__).with_name("mp4_metadata.py").read_bytes()).hexdigest()
    log = []
    for index, parts in manifest["archives"]:
        path = results / "video_shards" / f"part_{index:03d}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        selected = [r for r in rows if r.get("timing_class") != "constant"]
        if not selected:
            continue
        with SplitFile(parts) as source, zipfile.ZipFile(source) as archive:
            for row in selected:
                old_class = row.get("timing_class")
                with archive.open(row["member"]) as handle:
                    updated = probe(handle, row["bytes"])
                row.update(updated, repair_parser_sha256=parser_hash)
                log.append(dict(video=row["video"], archive=index, before=old_class,
                                after=row["timing_class"]))
        temporary = path.with_suffix(".jsonl.refresh")
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        temporary.replace(path)
        print(f"archive={index} rechecked={len(selected)}", flush=True)
    manifest["repair_parser_sha256"] = parser_hash
    write_json(results / "archive_manifest.json", manifest)
    write_json(results / "timing_refresh.json", dict(parser_sha256=parser_hash, records=log))
    print(f"Rechecked {len(log)} records", flush=True)


if __name__ == "__main__":
    refresh()
