"""Offline, read-only analysis of completed API benchmark runs.

No provider SDK, credential loader, image reader, or network client is imported.
Only the output analysis JSON is written; run artifacts remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile


DCVLM_CAPTION_RECORDS = 88_908_841
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "production/runs/api_throughput_20261023"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(numerator, denominator):
    return numerator / denominator if denominator and denominator > 0 else None


def percentile(values, quantile):
    values = sorted(value for value in values if number(value))
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def distribution(values):
    values = [value for value in values if number(value)]
    return {"reported_responses": len(values), "mean": ratio(sum(values), len(values)),
            "p50": percentile(values, .5), "p95": percentile(values, .95),
            "max": max(values, default=None), "total": sum(values) if values else None}


def http_ok(row):
    return isinstance(row.get("http_status"), int) and 200 <= row["http_status"] < 300


def logical_key(row, fallback):
    value = row.get("logical_request_index")
    return ("logical", value) if isinstance(value, int) else ("request", row.get("request_id", fallback))


def concurrency_stats(rows, wall):
    """Client request intervals are half-open: an end at t precedes a start at t."""
    events, interval_sum, valid, invalid = [], 0.0, 0, 0
    for row in rows:
        start, end = row.get("start_unix"), row.get("end_unix")
        if not number(start) or not number(end) or end < start:
            invalid += 1
            continue
        valid += 1
        interval_sum += end - start
        if end > start:
            events.extend(((start, 1), (end, -1)))
    active, peak = 0, 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return {"observed_peak_client_inflight": peak, "valid_intervals": valid,
            "invalid_or_missing_intervals": invalid,
            "sum_client_request_interval_seconds": interval_sum,
            "mean_client_inflight_over_api_batch": ratio(interval_sum, wall),
            "definition": "Concurrent client request start/end intervals, including failed attempts; not server-side active inference or configured concurrency."}


def completion_bins(rows, start, end, seconds=10):
    if not number(start) or not number(end) or end <= start:
        return []
    count = max(1, math.ceil((end - start) / seconds))
    if count > 100_000:
        raise ValueError("Implausible batch duration for completion bins")
    bins = [{"index": i, "start_offset_seconds": i * seconds,
             "end_offset_seconds": (i + 1) * seconds,
             "observed_seconds": min(seconds, end - start - i * seconds),
             "completed_attempts": 0, "http_2xx_responses": 0,
             "schema_effective_responses": 0} for i in range(count)]
    for row in rows:
        finish = row.get("end_unix")
        if not number(finish) or finish < start or finish > end:
            continue
        index = min(count - 1, int((finish - start) // seconds))
        target = bins[index]
        target["completed_attempts"] += 1
        target["http_2xx_responses"] += int(http_ok(row))
        target["schema_effective_responses"] += int(row.get("effective_success") is True)
    for item in bins:
        item["attempt_completions_per_second_fixed_bin"] = item["completed_attempts"] / seconds
        item["http_2xx_per_second_fixed_bin"] = item["http_2xx_responses"] / seconds
        item["schema_effective_per_second_fixed_bin"] = item["schema_effective_responses"] / seconds
        item["partial_final_bin"] = item["observed_seconds"] < seconds
    return bins


def usage_value(row, metric):
    value = (row.get("usage_metrics") or {}).get(metric, {})
    value = value.get("value") if isinstance(value, dict) else None
    if number(value):
        return value
    # Legacy top-level counters only. Never infer output text from input text_tokens.
    names = {"billed_output_tokens": ("output_tokens", "completion_tokens"),
             "input_tokens": ("input_tokens", "prompt_tokens")}.get(metric, ())
    for name in names:
        value = (row.get("usage") or {}).get(name)
        if number(value):
            return value
    return None


def extrapolation(task, goodput):
    if task != "caption":
        return {"applicable": False, "reason": "This run does not generate captions."}
    return {
        "applicable": bool(goodput and goodput > 0),
        "basis": "Observed unique logical schema-success responses / entire API batch wall seconds",
        "schema_success_outputs_per_second": goodput,
        "days_per_million_schema_success_outputs": ratio(1_000_000, goodput * 86400) if goodput else None,
        "dcvlm_caption_records": DCVLM_CAPTION_RECORDS,
        "days_for_dcvlm_caption_count_at_this_rate": ratio(DCVLM_CAPTION_RECORDS, goodput * 86400) if goodput else None,
        "quality_qualified": False, "full_pipeline_estimate": False,
        "conditions": [
            "Assumes uninterrupted 24-hour operation at the measured full-batch schema-success rate and unchanged limits, sample mix, output lengths and error behavior.",
            "Three caption lengths are generated in one response, so this denominator is one record rather than three separate requests.",
            "It estimates an output count; it does not guarantee every original record succeeds. Repair, retries, independent review, filtering, deduplication, IO and release work require separate measurement.",
            "Schema success does not establish factual caption quality or training admission; this is not a quality-passed completion estimate.",
        ],
    }


def analyze_run(directory):
    raw = {name: (directory / name).read_bytes() for name in ("summary.json", "run.json", "requests.jsonl")}
    summary, run = json.loads(raw["summary.json"]), json.loads(raw["run.json"])
    if summary.get("schema_version") != "api_benchmark.summary.v1":
        raise ValueError("Unsupported summary schema")
    if summary.get("requests_sha256") != sha(raw["requests.jsonl"]):
        raise ValueError("Completed summary/request hash mismatch; possibly changing run")
    if summary.get("run_sha256") != sha(raw["run.json"]):
        raise ValueError("Completed summary/run hash mismatch")
    rows = [json.loads(line) for line in raw["requests.jsonl"].splitlines() if line.strip()]
    wall = summary.get("wall_seconds")
    if not number(wall) or wall <= 0:
        raise ValueError("Missing or invalid complete API batch wall_seconds")
    start, end = summary.get("start_unix"), summary.get("end_unix")
    if not number(start) or not number(end) or end < start:
        raise ValueError("Missing or invalid complete API batch boundaries")
    effective = [row for row in rows if row.get("effective_success") is True]
    http = [row for row in rows if http_ok(row)]
    schema_valid = [row for row in http if row.get("schema_valid") is True]
    schema_invalid = [row for row in http if row.get("schema_valid") is False]
    logical, effective_logical = {}, {}
    for index, row in enumerate(rows):
        key = logical_key(row, index)
        logical.setdefault(key, row)
        if row.get("effective_success") is True:
            effective_logical.setdefault(key, row)
    timeout = sum("timeout" in str(row.get("error_type", "")).lower() for row in rows)
    rate_limited = sum(row.get("http_status") == 429 for row in rows)
    durations = [row.get("latency_seconds") for row in rows]
    token_stats = {}
    for metric in ("billed_output_tokens", "reported_text_tokens", "reasoning_tokens", "input_tokens", "cache_read_tokens"):
        token_stats[metric] = {
            "http_responses": distribution([usage_value(row, metric) for row in http]),
            "schema_effective_responses": distribution([usage_value(row, metric) for row in effective]),
            "missing_values_are_not_zero": True,
        }
    caption_stats = {}
    for variant in ("short", "medium", "long"):
        texts = [(row.get("parsed") or {}).get("captions", {}).get(variant)
                 for row in effective_logical.values()]
        texts = [text for text in texts if isinstance(text, str)]
        caption_stats[variant] = {"whitespace_words": distribution([len(text.split()) for text in texts]),
                                  "characters": distribution([len(text) for text in texts])}
    complete_caption_sets = [(row.get("parsed") or {}).get("captions", {})
                             for row in effective_logical.values()]
    complete_caption_sets = [captions for captions in complete_caption_sets
                             if all(isinstance(captions.get(key), str) for key in ("short", "medium", "long"))]
    caption_stats["three_variants_total_per_response"] = {
        "whitespace_words": distribution([sum(len(captions[key].split()) for key in ("short", "medium", "long"))
                                           for captions in complete_caption_sets]),
        "characters": distribution([sum(len(captions[key]) for key in ("short", "medium", "long"))
                                     for captions in complete_caption_sets]),
    }
    image_sets = []
    for row in logical.values():
        hashes = tuple(image.get("original_sha256") for image in row.get("image_inputs", []))
        if hashes and all(isinstance(value, str) for value in hashes):
            image_sets.append(hashes)
    sample_ids = [row.get("sample_id") for row in logical.values() if isinstance(row.get("sample_id"), str)]
    config_fields = ("model", "task", "concurrency", "logical_requests", "timeout_seconds", "timeout_semantics",
                     "prompt_sha256", "max_tokens", "thinking", "thinking_budget_tokens", "temperature",
                     "application_retries", "sdk_max_retries", "code_sha256", "manifest_sha256", "offset",
                     "image_preprocessing", "endpoint_sha256", "accepted_response_model_aliases",
                     "provider_prompt_cache_not_disabled", "local_response_cache")
    config = {name: run.get(name) for name in config_fields}
    comparison_config = {key: value for key, value in config.items() if key not in {"model", "concurrency"}}
    goodput = len(effective_logical) / wall
    prep = summary.get("preparation_seconds", run.get("preparation_seconds_excluded_from_api_wall"))
    diagnostics = []
    if len(rows) != summary.get("http_attempts"):
        diagnostics.append("Observed attempt count disagrees with summary")
    if len(effective) != summary.get("effective_successful_requests"):
        diagnostics.append("Observed effective response count disagrees with summary")
    if len(logical) != run.get("logical_requests"):
        diagnostics.append("Some configured logical requests are not represented by completed request rows")
    return {
        "run_path": str(directory), "run_id": run.get("run_id"), "config": config,
        "is_qwen": str(run.get("model", "")).lower().startswith("qwen"),
        "artifacts_sha256": {name: sha(data) for name, data in raw.items()},
        "configuration_match_signature_excluding_model_and_concurrency": sha(json.dumps(comparison_config, sort_keys=True).encode()),
        "configuration_match_is_not_controlled_comparison": True,
        "timing": {"start_unix": start, "end_unix": end, "api_batch_wall_seconds": wall,
                   "api_batch_epoch_span_seconds": end - start,
                   "preparation_seconds": prep,
                   "measured_preparation_plus_api_wall_seconds": wall + prep if number(prep) else None,
                   "whole_batch_includes_timeout_tail_and_queueing": True,
                   "excludes": "API wall excludes image preparation; preparation-plus-API also excludes unmeasured startup/final reporting overhead."},
        "counts": {"configured_logical_requests": run.get("logical_requests"), "observed_logical_requests": len(logical),
                   "attempts": len(rows), "http_2xx_responses": len(http),
                   "schema_valid_http_responses": len(schema_valid), "schema_invalid_http_responses": len(schema_invalid),
                   "schema_unknown_http_responses": len(http) - len(schema_valid) - len(schema_invalid),
                   "schema_effective_responses": len(effective), "schema_effective_logical_requests": len(effective_logical),
                   "http_429": rate_limited, "timeout_attempts": timeout,
                   "status_counts": dict(Counter(row.get("status", "unknown") for row in rows)),
                   "error_type_counts": dict(Counter(row["error_type"] for row in rows if row.get("error_type")))},
        "rates": {"http_429_per_attempt": ratio(rate_limited, len(rows)), "timeout_per_attempt": ratio(timeout, len(rows)),
                  "schema_valid_per_http_2xx": ratio(len(schema_valid), len(http)),
                  "schema_invalid_per_http_2xx": ratio(len(schema_invalid), len(http)),
                  "schema_valid_per_attempt": ratio(len(schema_valid), len(rows)),
                  "schema_effective_per_logical_request": ratio(len(effective_logical), len(logical))},
        "full_batch_throughput": {"primary_schema_effective_logical_requests_per_second": goodput,
                                  "http_2xx_responses_per_second": len(http) / wall,
                                  "schema_effective_responses_per_second": len(effective) / wall,
                                  "all_attempts_per_second": len(rows) / wall,
                                  "schema_effective_per_second_including_measured_preparation": len(effective_logical) / (wall + prep) if number(prep) else None},
        "latency_seconds": {"all_attempts": distribution(durations),
                            "http_2xx_responses": distribution([row.get("latency_seconds") for row in http]),
                            "schema_effective_responses": distribution([row.get("latency_seconds") for row in effective]),
                            "queue_wait_all_attempts": distribution([row.get("queue_wait_seconds") for row in rows])},
        "concurrency": concurrency_stats(rows, wall), "output_token_statistics": token_stats,
        "caption_length_among_schema_effective_logical_requests": caption_stats,
        "caption_words_definition": "Whitespace-separated words, not model tokens; no invalid/missing response is silently counted as zero.",
        "sampling": {"unique_sample_ids": len(set(sample_ids)), "logical_requests_with_sample_id": len(sample_ids),
                     "repeated_sample_id_logical_requests": len(sample_ids) - len(set(sample_ids)),
                     "unique_original_image_sha256": len({value for hashes in image_sets for value in hashes}),
                     "unique_ordered_original_image_sets": len(set(image_sets)),
                     "logical_requests_with_complete_image_hashes": len(image_sets),
                     "repeated_image_set_logical_requests": len(image_sets) - len(set(image_sets)),
                     "retry_attempts_not_additional_unique_inputs": len(rows) - len(logical),
                     "run_reported_repeated_items": run.get("repeated_items")},
        "completion_bins": {"bin_seconds": 10, "not_primary_throughput": True,
                            "definition": "Completion counts in fixed 10-second bins relative to API batch start. Final endpoint is included in final bin; partial final rates still divide by 10. Bursts are not sustainable throughput.",
                            "bins": completion_bins(rows, start, end)},
        "caption_only_conditional_extrapolation": extrapolation(run.get("task"), goodput),
        "effective_definition": summary.get("effective_definition"),
        "quality_pass_inferred": False, "diagnostics": diagnostics,
    }


def analyze_root(root):
    reports, excluded = [], []
    # Snapshot the directory list. Newly completing runs are picked up on rerun.
    for path in sorted(root.rglob("summary.json")):
        directory = path.parent
        if not (directory / "requests.jsonl").is_file():
            continue
        try:
            reports.append(analyze_run(directory))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            excluded.append({"run_path": str(directory), "reason": str(exc)})
    reports.sort(key=lambda value: (value["timing"]["start_unix"], value["run_path"]))
    unfinished = [str(path.parent) for path in sorted(root.rglob("run.json"))
                  if (path.parent / "requests.jsonl").is_file() and not (path.parent / "summary.json").is_file()]
    return {"schema_version": "api_benchmark.offline_analysis.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
            "analyzer_sha256": sha(Path(__file__).read_bytes()), "new_api_requests": 0,
            "completed_runs_analyzed": len(reports), "runs": reports,
            "excluded_completed_runs": excluded, "unfinished_runs_not_scored": unfinished,
            "comparison_cautions": [
                "Different prompt hashes, timeout policies, sample selection, output lengths, processor versions, models or retry policies are not strict controlled comparisons.",
                "Matching recorded configuration still does not control provider load, routing, model revisions or cache behavior. Historical non-Qwen runs are archival observations, not new tests.",
                "Main throughput uses each entire completed API batch including slow/failed timeout tails. Successful-request P95 and 10-second completion bursts cannot replace it.",
                "HTTP 200 and schema success are not factual quality approval. No quality-pass rate or end-to-end training-corpus processing duration is inferred.",
                "Elapsed short benchmark batches do not demonstrate sustained full-corpus service capacity; repeated images do not add independent quality cases.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, help="Defaults to ROOT/analysis.json; only this aggregate artifact is replaced")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "analysis.json").resolve()
    if output.name in {"summary.json", "requests.jsonl", "run.json"}:
        raise SystemExit("Refusing to overwrite source run artifacts")
    report = analyze_root(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".analysis-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "completed_runs_analyzed": len(report["runs"]),
                      "excluded_completed_runs": len(report["excluded_completed_runs"]),
                      "unfinished_runs_not_scored": len(report["unfinished_runs_not_scored"]),
                      "new_api_requests": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
