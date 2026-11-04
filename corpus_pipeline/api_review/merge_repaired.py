"""Merge valid API repairs into invalid audit records, preserving immutable inputs."""
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

from corpus_pipeline.api_review.aggregate import validate_parsed
from corpus_pipeline.api_review.strict_json import parse_json_response


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def record_errors(row):
    errors = []
    if row.get("status") != "ok":
        errors.append("api_status_not_ok")
    if row.get("text_truncated"):
        errors.append("retained_text_truncated")
    if row.get("finish_reason") == "max_tokens":
        errors.append("response_truncated")
    try:
        parsed = parse_json_response(row.get("text"))
    except (ValueError, TypeError):
        errors.append("strict_text_json_invalid")
    else:
        try:
            matches = canonical(parsed) == canonical(row.get("parsed"))
        except (TypeError, ValueError):
            matches = False
        if not matches:
            errors.append("stored_parsed_mismatch")
    errors.extend(validate_parsed(row.get("parsed")))
    return sorted(set(errors))


def load_records(path):
    raw = Path(path).read_bytes()
    records, indexed = [], {}
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        try:
            row = parse_json_response(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ValueError(f"Invalid JSONL record at line {line_number}") from None
        image_sha = row.get("image_sha256")
        model = row.get("requested_model")
        round_name = row.get("round")
        if (not isinstance(image_sha, str) or len(image_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in image_sha)):
            raise ValueError(f"Invalid image identity at line {line_number}")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Invalid model identity at line {line_number}")
        if not isinstance(round_name, str) or not round_name.strip():
            raise ValueError(f"Invalid review round at line {line_number}")
        pair = (image_sha, model)
        if pair in indexed:
            raise ValueError(f"Duplicate image/model pair at line {line_number}")
        item = {"row": row, "line": line, "line_number": line_number,
                "pair": pair, "errors": record_errors(row)}
        indexed[pair] = item
        records.append(item)
    return raw, records, indexed


def merge_records(original_path, repairs_path):
    original_bytes, originals, original_index = load_records(original_path)
    repair_paths = list(repairs_path) if isinstance(repairs_path, (list, tuple)) else [repairs_path]
    repairs, repair_inputs, seen_repairs = [], [], set()
    for path in repair_paths:
        repair_bytes, records, _ = load_records(path)
        repair_input = {"path": str(Path(path).resolve()), "sha256": sha256(repair_bytes),
                        "record_count": len(records)}
        repair_inputs.append(repair_input)
        for record in records:
            if record["pair"] in seen_repairs:
                raise ValueError("Duplicate image/model pair across repair inputs")
            seen_repairs.add(record["pair"])
            record["source_path"] = repair_input["path"]
            record["source_sha256"] = repair_input["sha256"]
            repairs.append(record)
    if not originals:
        raise ValueError("Original results are empty")
    replacements = {}
    replacement_audit = []
    for repair in repairs:
        pair = repair["pair"]
        if pair not in original_index:
            raise ValueError("Repair targets an unknown image/model pair")
        original = original_index[pair]
        if repair["row"]["round"] != original["row"]["round"]:
            raise ValueError("Repair round does not match original round")
        if not original["errors"]:
            raise ValueError("Cannot replace an already valid original record")
        if repair["errors"]:
            raise ValueError("Repair is invalid: " + ",".join(repair["errors"]))
        for key in ("cache_key", "prompt_sha256"):
            value = repair["row"].get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError("Repair lacks valid audit identity: " + key)
        replacements[pair] = repair
        old, new = original["row"], repair["row"]
        replacement_audit.append({
            "image_sha256": pair[0], "requested_model": pair[1], "round": old["round"],
            "review_id": old.get("review_id"),
            "original_line": original["line_number"], "repair_line": repair["line_number"],
            "repair_source_path": repair["source_path"], "repair_source_sha256": repair["source_sha256"],
            "original_errors": original["errors"],
            "original_cache_key": old.get("cache_key"), "repair_cache_key": new["cache_key"],
            "original_prompt_sha256": old.get("prompt_sha256"),
            "repair_prompt_sha256": new["prompt_sha256"],
            "original_base_prompt_sha256": old.get("base_prompt_sha256"),
            "repair_base_prompt_sha256": new.get("base_prompt_sha256"),
            "original_record_sha256": sha256(original["line"]),
            "repair_record_sha256": sha256(repair["line"]),
        })
    merged_lines = []
    remaining_invalid = []
    for original in originals:
        selected = replacements.get(original["pair"], original)
        line = selected["line"]
        # All existing bytes of untouched records are preserved; add a missing
        # final newline only if a following record needs JSONL separation.
        merged_lines.append(line)
        if selected["errors"]:
            remaining_invalid.append({
                "image_sha256": original["pair"][0], "requested_model": original["pair"][1],
                "round": original["row"]["round"], "errors": selected["errors"],
            })
    for index in range(len(merged_lines) - 1):
        if not merged_lines[index].endswith(b"\n"):
            merged_lines[index] += b"\n"
    merged_bytes = b"".join(merged_lines)
    audit = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "original_path": str(Path(original_path).resolve()), "original_sha256": sha256(original_bytes),
        "repair_inputs": repair_inputs,
        "repairs_path": repair_inputs[0]["path"] if len(repair_inputs) == 1 else None,
        "repairs_sha256": (repair_inputs[0]["sha256"] if len(repair_inputs) == 1 else
                           sha256(canonical([item["sha256"] for item in repair_inputs]).encode())),
        "repairs_sha256_scope": "single_file_bytes" if len(repair_inputs) == 1 else "ordered_input_sha256_list",
        "merged_sha256": sha256(merged_bytes), "original_count": len(originals),
        "repair_count": len(repairs), "replacement_count": len(replacements),
        "original_invalid_count": sum(bool(item["errors"]) for item in originals),
        "original_error_counts": dict(Counter(error for item in originals for error in item["errors"])),
        "remaining_invalid_count": len(remaining_invalid),
        "all_valid_after_merge": not remaining_invalid,
        "remaining_invalid": remaining_invalid, "replacements": replacement_audit,
        "unchanged_count": len(originals) - len(replacements),
        "inputs_modified": False, "unreplaced_record_bytes_preserved": True,
        "scope": "Response integrity and schema only; not correctness or image provenance",
        "validator_sha256": sha256(Path(__file__).with_name("aggregate.py").read_bytes()),
        "strict_parser_sha256": sha256(Path(__file__).with_name("strict_json.py").read_bytes()),
        "merge_code_sha256": sha256(Path(__file__).read_bytes()),
    }
    return merged_bytes, audit


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--repairs", nargs="+", required=True)
    parser.add_argument("--output", required=True,
                        help="Directory for results.jsonl and merge_audit.json, or explicit .jsonl output path")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.suffix == ".jsonl":
        result_path, audit_path = output, output.with_suffix(".audit.json")
    else:
        result_path, audit_path = output / "results.jsonl", output / "merge_audit.json"
    inputs = {Path(args.original).resolve(), *(Path(path).resolve() for path in args.repairs)}
    if result_path.resolve() in inputs or audit_path.resolve() in inputs:
        parser.error("Outputs must not overwrite original or repair inputs")
    try:
        merged_bytes, audit = merge_records(args.original, args.repairs)
        # Guard concurrent mutation of either immutable input during validation.
        if (sha256(Path(args.original).read_bytes()) != audit["original_sha256"] or any(
                sha256(Path(item["path"]).read_bytes()) != item["sha256"] for item in audit["repair_inputs"])):
            raise ValueError("Input changed while preparing merge")
        atomic_write(result_path, merged_bytes)
        audit["merged_path"] = str(result_path.resolve())
        atomic_write(audit_path, (json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "merge_error", "error_type": type(exc).__name__}, ensure_ascii=False),
              file=sys.stderr)
        return 1
    print(json.dumps({key: audit[key] for key in (
        "original_count", "replacement_count", "remaining_invalid_count", "all_valid_after_merge",
        "original_sha256", "repairs_sha256", "merged_sha256", "merged_path")}, ensure_ascii=False))
    return 0 if audit["all_valid_after_merge"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
