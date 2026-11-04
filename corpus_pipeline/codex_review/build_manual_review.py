"""Package an audited pending queue as an offline human review page.

Embeds exact original JPEG/PNG/WebP bytes after SHA and full decode checks.
Never mutates model results or treats an unreviewed item as a human decision.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import uuid

from PIL import Image, ImageFile

from .farm import atomic_json, canonical, digest, file_sha

MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def require(value, message):
    if not value:
        raise ValueError(message)


def atomic_text(path, text):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_name(path):
    parts = Path(path).parts
    for marker in ("dcvlm_pool_small", "DCVLM", "gen-und"):
        if marker in parts and parts.index(marker) + 1 < len(parts):
            return parts[parts.index(marker) + 1]
    return Path(path).parent.name


def prepare_item(pair):
    number, row = pair
    aid, ref, result = row["item_id"], row["source_ref"], row["result"]
    require(isinstance(aid, str) and re.fullmatch("[0-9a-f]{64}", aid), "invalid item ID")
    require(row["report_category"] == "visual_pending" and row["processing_status"] == "succeeded"
            and row["decision"] == result["decision"] == "pending", "queue contains a non-pending result")
    require(result["item_id"] == aid and digest(result) == row["result_sha256"], "result identity/hash mismatch")
    require(ref["kind"] == "tar" and ref["sha256"] == aid, "unsupported or mismatched source")
    offset, length = ref["offset"], ref["length"]
    require(type(offset) is int and offset >= 0 and type(length) is int and 0 < length <= 128 * 1024**2,
            "invalid source byte range")
    with Path(ref["path"]).open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(offset + length <= before.st_size, "source range outside TAR")
        stream.seek(offset)
        blob = stream.read(length)
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "source changed during read")
    require(len(blob) == length and hashlib.sha256(blob).hexdigest() == aid, "original image SHA mismatch")
    require(not ImageFile.LOAD_TRUNCATED_IMAGES, "unsafe image decoder configuration")
    with Image.open(io.BytesIO(blob)) as picture:
        require(picture.format in MIME and picture.format == ref["format"], "unsupported image format")
        require(picture.width * picture.height <= 100_000_000 and getattr(picture, "n_frames", 1) == 1,
                "oversized or multiframe source")
        require(picture.size == (ref["width"], ref["height"]), "source dimensions changed")
        picture.load()
        width, height, fmt = picture.width, picture.height, picture.format
    return {
        "review_number": number, "item_id": aid,
        "image_data_url": "data:" + MIME[fmt] + ";base64," + base64.b64encode(blob).decode("ascii"),
        "image_path": "", "source_name": source_name(ref["path"]),
        "source_url": ref.get("source_url", ""), "source_path": ref["path"],
        "source_member": ref.get("member", ""), "width": width, "height": height,
        "preview_width": width, "preview_height": height, "image_format": fmt,
        "source_bytes": length, "source_sha256": aid, "image_kind": result["image_kind"],
        "assessment": result["assessment"], "confidence": result["confidence"],
        "quality": result["quality"], "reason": result["reason"], "quality_reason": result["quality_reason"],
        "alternative_explanation": result["alternative_explanation"],
        "visible_evidence": result["visible_evidence"], "prior_summary": row["prior_summary"],
        "result_sha256": row["result_sha256"], "queue_order": row["queue_order"],
        "source_ref": ref, "raw_provenance_reference": row["raw_provenance_reference"],
    }


def build(root, output, workers=8):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    require(1 <= workers <= 16, "workers must be between 1 and 16")
    require(output != root and output.is_relative_to(root), "output must be a dedicated child directory")
    report_path, queue_path = root / "final_report.json", root / "manual_review.jsonl.gz"
    report = json.loads(report_path.read_text())
    complete = json.loads((root / "COMPLETE.json").read_text())
    require(report["complete"] and report["strict_audit_passed"] and complete["complete"], "farm is not complete/audited")
    require(report["audit"]["sha256"] == complete["audit_sha256"]
            and file_sha(complete["audit"]) == complete["audit_sha256"], "audit binding changed")
    queue_id = file_sha(queue_path)
    require(queue_id == report["artifacts"][queue_path.name]["sha256"], "pending queue hash changed")
    with gzip.open(queue_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) == report["counts"]["visual_pending"] and rows, "pending queue count mismatch")
    require(len({row["item_id"] for row in rows}) == len(rows), "duplicate item in pending queue")
    rows.sort(key=lambda row: (row["queue_order"], row["item_id"]))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "MANIFEST.json").exists():
        require(json.loads((output / "MANIFEST.json").read_text())["queue_id"] == queue_id,
                "different queue already occupies this output directory")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        items = list(executor.map(prepare_item, enumerate(rows, 1)))
    data = {"schema_version": "corpus.human_review.package.v1", "queue_id": queue_id,
            "title": "DCVLM 人工复核", "created_at": datetime.now(timezone.utc).isoformat(), "items": items}
    # Data can be embedded inside either a script object or application/json.
    payload = canonical(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template_path = Path(__file__).with_name("manual_review_template.html")
    template = template_path.read_text()
    require(template.count("__REVIEW_DATA_JSON__") == 1, "template must have exactly one data placeholder")
    atomic_text(output / "index.html", template.replace("__REVIEW_DATA_JSON__", payload))
    atomic_text(output / "queue.jsonl", "".join(canonical({k: v for k, v in item.items() if k != "image_data_url"})
                                              + "\n" for item in items))
    manifest = {
        "schema_version": data["schema_version"], "queue_id": queue_id, "records": len(items),
        "created_at": data["created_at"], "source_queue": str(queue_path),
        "source_report": str(report_path), "source_report_sha256": file_sha(report_path),
        "source_complete": complete, "template_sha256": file_sha(template_path),
        "builder_sha256": file_sha(__file__), "source_formats": dict(Counter(x["image_format"] for x in items)),
        "source_bytes": sum(x["source_bytes"] for x in items),
        "images": "exact original bytes; SHA verified and fully decoded; embedded without resizing/recompression",
        "model_results_modified": False, "human_decisions_created": False,
        "files": {name: {"sha256": file_sha(output / name), "bytes": (output / name).stat().st_size}
                  for name in ("index.html", "queue.jsonl")},
    }
    atomic_json(output / "MANIFEST.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(canonical(build(args.root, args.output, args.workers)))


if __name__ == "__main__":
    main()
