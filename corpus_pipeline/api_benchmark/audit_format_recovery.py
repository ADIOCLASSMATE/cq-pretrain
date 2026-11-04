"""Read-only, bounded format recovery audit for recorded Qwen caption responses.

Never calls an API or rewrites request receipts. Repaired structure is evidence
of recoverable formatting only, not evidence of factual caption quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from corpus_pipeline.api_benchmark.bench import validate_response
from corpus_pipeline.api_review.strict_json import parse_json_response
from corpus_pipeline.production.common import atomic_json, digest, file_sha


VERSION = "caption_format_recovery.v1"
CLASSES = ("direct", "field_move", "trailing_comma", "both", "unrecoverable", "http_failure")


def remove_trailing_commas(text):
    """Only delete commas outside strings before }/], after an existing value.

    Existing string tokens, including escapes and commas, are copied verbatim.
    The result still must pass the existing strict parser; this is not JSON5 or
    partial-object extraction. Empty-value forms such as [,] are not repaired.
    """
    out, removed = [], []
    quoted = escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == ",":
            after, before = index + 1, index - 1
            while after < len(text) and text[after].isspace():
                after += 1
            while before >= 0 and text[before].isspace():
                before -= 1
            if (after < len(text) and text[after] in "}]" and before >= 0
                    and text[before] not in "[{:," ):
                removed.append(index)
                continue
        out.append(char)
    return "".join(out), removed


def audit_response(row, config):
    result = {key: row.get(key) for key in ("request_id", "logical_request_index", "attempt_index", "sample_id",
        "http_status", "stop_reason", "response_model", "status", "effective_success")}
    result["classification"] = "unrecoverable"
    result["operations"] = []
    result["audited_schema_success"] = False
    status = row.get("http_status")
    if not isinstance(status, int) or not 200 <= status < 300:
        result.update(classification="http_failure", reason="http_not_successful_not_examined")
        return result
    expected = {config["model"], *config.get("accepted_response_model_aliases", [])}
    if row.get("response_model") not in expected:
        result["reason"] = "response_model_not_accepted"
        return result
    if row.get("response_text_truncated") is not False or row.get("stop_reason") not in {"end_turn", "stop_sequence"}:
        result["reason"] = "incomplete_truncated_refusal_or_unknown_stop"
        return result
    text, request_id = row.get("response_text"), row.get("request_id")
    if not isinstance(text, str) or not isinstance(request_id, str) or not request_id:
        result["reason"] = "missing_response_or_request_identity"
        return result
    result["response_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    removed = []
    try:
        parsed = parse_json_response(text)
    except json.JSONDecodeError:
        repaired, removed = remove_trailing_commas(text)
        if not removed:
            result["reason"] = "json_syntax_not_a_trailing_comma"
            return result
        try:
            parsed = parse_json_response(repaired)
        except (ValueError, TypeError):
            result["reason"] = "strict_parser_rejects_after_trailing_comma_removal"
            return result
        result["operations"].append({"type": "remove_trailing_comma_outside_string", "character_offsets": removed})
    except (ValueError, TypeError):
        result["reason"] = "strict_parser_rejected_non_syntax_contract"
        return result
    canonical = copy.deepcopy(parsed)
    captions = canonical.get("captions")
    moved = False
    if (set(canonical) == {"request_id", "captions"} and isinstance(captions, dict)
            and set(captions) == {"short", "medium", "long", "uncertainties"}):
        canonical["uncertainties"] = captions.pop("uncertainties")
        moved = True
        result["operations"].append({"type": "move_field", "from": "captions.uncertainties", "to": "uncertainties"})
    try:
        validate_response("caption", canonical, request_id, {})
    except (ValueError, TypeError, KeyError):
        result["reason"] = "canonical_schema_or_request_identity_failed"
        return result
    result.update(classification="both" if moved and removed else "field_move" if moved else "trailing_comma" if removed else "direct",
                  audited_schema_success=True, reason="full_canonical_schema_and_identity_valid",
                  lexical_string_tokens_unchanged=True, canonical_object_sha256=digest(canonical),
                  caption_strings_unchanged=all(parsed["captions"][key] == canonical["captions"][key]
                                               for key in ("short", "medium", "long")))
    return result


def audit_run(run_path):
    config_bytes = run_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("task") != "caption" or not str(config.get("model", "")).lower().startswith("qwen"):
        return None
    path = run_path.parent / "requests.jsonl"
    if not path.exists():
        return None
    snapshot = path.read_bytes()
    complete = snapshot if snapshot.endswith(b"\n") else snapshot[:snapshot.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    records = [audit_response(row, config) for row in rows]
    counts = Counter(record["classification"] for record in records)
    successful = [record for record in records if record["audited_schema_success"]]
    logical_key = lambda record: record.get("logical_request_index", record.get("request_id"))
    unique_successes = len({logical_key(record) for record in successful})
    direct_keys = {logical_key(record) for record in records if record["classification"] == "direct"}
    repaired_keys = {logical_key(record) for record in successful} - direct_keys
    result = {"run_path": str(run_path.parent.resolve()), "run_id": config.get("run_id"), "model": config["model"],
              "concurrency": config.get("concurrency"), "requested": config.get("logical_requests"),
              "completed_http_attempts": len(rows), "observed_logical_requests": len({logical_key(record) for record in records}),
              "run_sha256": hashlib.sha256(config_bytes).hexdigest(), "request_snapshot_sha256": hashlib.sha256(complete).hexdigest(),
              "snapshot_complete_bytes": len(complete), "ignored_incomplete_tail_bytes": len(snapshot) - len(complete),
              "classification_counts": {name: counts[name] for name in CLASSES},
              "http_success_attempts": sum(isinstance(row.get("http_status"), int) and 200 <= row["http_status"] < 300 for row in rows),
              "original_effective_success_attempts": sum(row.get("effective_success") is True for row in rows),
              "audited_success_unique_logical_requests": unique_successes,
              "direct_unique_logical_requests": len(direct_keys), "repaired_only_unique_logical_requests": len(repaired_keys),
              "summary_matches_snapshot": False, "rates": None, "records": records}
    summary_path = path.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        bound = (summary.get("requests_sha256") == result["request_snapshot_sha256"]
                 and summary.get("run_sha256") == result["run_sha256"] and len(snapshot) == len(complete))
        result["summary_matches_snapshot"] = bound
        wall = summary.get("wall_seconds")
        if bound and isinstance(wall, (int, float)) and wall > 0:
            result["rates"] = {"wall_seconds": wall, "http_success_attempts_per_second": result["http_success_attempts"] / wall,
                               "original_effective_success_attempts_per_second": result["original_effective_success_attempts"] / wall,
                               "direct_unique_logical_requests_per_second": len(direct_keys) / wall,
                               "repaired_only_unique_logical_requests_per_second": len(repaired_keys) / wall,
                               "direct_or_repaired_unique_logical_requests_per_second": unique_successes / wall,
                               "quality_pass_rate": None}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error("root must be an existing benchmark directory")
    runs, errors = [], []
    for path in sorted(args.root.rglob("run.json")):
        try:
            result = audit_run(path)
            if result is not None:
                runs.append(result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append({"run_path": str(path.parent.resolve()), "error_type": type(error).__name__})
    output = args.root / "format_recovery_audit.json"
    artifact = {"schema_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "tool_sha256": file_sha(__file__), "root": str(args.root.resolve()), "runs": runs, "errors": errors,
                "original_receipts_modified": False, "api_calls": 0,
                "definitions": {"direct": "Original text passes strict parser and caption schema, request ID, returned model, complete response gates.",
                    "field_move": "Only move nested uncertainties to absent root field under exact allowed key sets; full schema validates.",
                    "trailing_comma": "Only remove string-external trailing comma(s); strict parser and full schema validate.",
                    "both": "Both bounded operations needed; full schema validates.",
                    "unrecoverable": "HTTP success but identity/completeness/parser/schema cannot satisfy the bounded rules.",
                    "http_failure": "HTTP was not successful; response is not examined or repaired."},
                "limitations": "Schema recovery does not establish factual quality or training admission. Original benchmark success is unchanged. Final rates require summary hashes bound to the exact run/request snapshots; running or inconsistent snapshots have no rate estimate."}
    atomic_json(output, artifact)
    print(json.dumps({"output": str(output.resolve()), "runs": len(runs), "errors": len(errors),
                      "results": [{key: run[key] for key in ("run_path", "classification_counts", "rates")} for run in runs]}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
