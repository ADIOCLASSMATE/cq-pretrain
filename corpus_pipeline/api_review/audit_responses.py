"""Offline audit of saved review responses; never invokes an API or edits input."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpus_pipeline.api_review.strict_json import parse_json_response


def audit(input_path, *, allow_incomplete_tail=False):
    path = Path(input_path)
    data = path.read_bytes()
    rows = []
    statuses = Counter()
    reasons = Counter()
    observed_keys = set()
    duplicate_keys = 0
    incomplete_tail = False
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if allow_incomplete_tail and index == len(lines) and not line.endswith(b"\n"):
                incomplete_tail = True
                break
            rows.append({"line": index, "audit_status": "invalid_journal_json"})
            reasons["invalid_journal_json"] += 1
            continue
        identity = {key: row.get(key) for key in (
            "cache_key", "image_sha256", "review_id", "requested_model", "status", "round")}
        if identity["cache_key"] in observed_keys:
            duplicate_keys += 1
        observed_keys.add(identity["cache_key"])
        statuses[row.get("status", "missing_status")] += 1
        result = {"line": index, **identity}
        text = row.get("text")
        if not isinstance(text, str):
            result["audit_status"] = "no_retained_text"
        elif row.get("text_truncated"):
            result["audit_status"] = "retained_text_truncated"
        else:
            try:
                parsed = parse_json_response(text)
                if parsed != row.get("parsed"):
                    result["audit_status"] = "stored_parsed_mismatch"
                elif row.get("finish_reason") == "max_tokens":
                    result["audit_status"] = "response_truncated"
                else:
                    result["audit_status"] = "strict_json_valid"
            except ValueError as exc:
                result["audit_status"] = "strict_json_invalid"
                # The parser's fixed error strings contain no response text.
                result["reason"] = str(exc).split(": line ")[0]
        reasons[result["audit_status"]] += 1
        rows.append(result)
    invalid_successes = sum(row.get("status") == "ok" and row["audit_status"] != "strict_json_valid"
                            for row in rows)
    summary = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(path.resolve()), "input_sha256": hashlib.sha256(data).hexdigest(),
        "strict_parser_sha256": hashlib.sha256(Path(__file__).with_name("strict_json.py").read_bytes()).hexdigest(),
        "records_audited": len(rows), "original_status_counts": dict(statuses),
        "audit_status_counts": dict(reasons), "invalid_success_count": invalid_successes,
        "duplicate_cache_key_count": duplicate_keys, "incomplete_tail_ignored": incomplete_tail,
        "scope": "JSON integrity only; not semantic schema validation or factual correctness",
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete-tail", action="store_true")
    args = parser.parse_args()
    summary, rows = audit(args.input, allow_incomplete_tail=args.allow_incomplete_tail)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("audit_summary.json", summary), ("audit_rows.json", rows)):
        temporary = output / (name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / name)
    print(json.dumps(summary, ensure_ascii=False))
    return 2 if summary["invalid_success_count"] or summary["audit_status_counts"].get("invalid_journal_json") else 0


if __name__ == "__main__":
    raise SystemExit(main())
