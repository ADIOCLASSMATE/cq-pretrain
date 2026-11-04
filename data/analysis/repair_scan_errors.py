"""Retry only failed records after adding support for leading empty MP4 edits.

Successful initial rows are retained, with both parser identities recorded.
"""
import hashlib
import json
import zipfile
from pathlib import Path

from mp4_metadata import SplitFile, probe
from scan_videos import DEFAULT_RESULTS, write_json


def repair(results=DEFAULT_RESULTS):
    manifest = json.loads((results / "archive_manifest.json").read_text())
    parser_hash = hashlib.sha256(Path(__file__).with_name("mp4_metadata.py").read_bytes()).hexdigest()
    log_path = results / "repair_log.json"
    repaired = json.loads(log_path.read_text())["records"] if log_path.exists() else []
    for index, parts in manifest["archives"]:
        status_path = results / "video_shards" / f"part_{index:03d}.status.json"
        status = json.loads(status_path.read_text())
        if not status["complete"]:
            raise ValueError("Wait for the initial full scan to complete")
        if not status["errors"]:
            continue
        path = results / "video_shards" / f"part_{index:03d}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        with SplitFile(parts) as source, zipfile.ZipFile(source) as archive:
            for row in rows:
                if row["status"] == "ok":
                    continue
                original_error = row["error"]
                try:
                    with archive.open(row["member"]) as handle:
                        result = probe(handle, row["bytes"])
                    row.update(result, status="ok", repair_parser_sha256=parser_hash)
                    row.pop("error")
                    repaired.append(dict(video=row["video"], archive=index,
                                         original_error=original_error, status="ok"))
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    repaired.append(dict(video=row["video"], archive=index,
                                         original_error=original_error, status="error", error=row["error"]))
        temporary = path.with_suffix(".jsonl.repaired")
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        temporary.replace(path)
        status.update(errors=sum(row["status"] != "ok" for row in rows), repair_parser_sha256=parser_hash)
        write_json(status_path, status)
        print(f"archive={index} remaining_errors={status['errors']}", flush=True)
    manifest["repair_parser_sha256"] = parser_hash
    write_json(results / "archive_manifest.json", manifest)
    write_json(log_path, {"parser_sha256": parser_hash, "records": repaired})
    states = [json.loads(p.read_text()) for p in (results / "video_shards").glob("*.status.json")]
    progress = json.loads((results / "scan_progress.json").read_text())
    progress.update(errors=sum(s["errors"] for s in states), error_retry_complete=True)
    write_json(results / "scan_progress.json", progress)
    print(f"Retried {len(repaired)} records", flush=True)


if __name__ == "__main__":
    repair()
