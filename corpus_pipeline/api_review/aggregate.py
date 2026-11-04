"""Audit-only aggregation of three independent, schema-validated visual reviews."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


IMAGE_FORMS = {
    "photo", "chart", "document_handwriting", "nonphotographic_render",
    "illustration", "screenshot_ui", "medical_scientific", "mixed", "other",
}
ASSESSMENTS = {"likely_ai_generated", "likely_non_ai", "possible_ai_edit", "uncertain"}
CONFIDENCES = {"high", "medium", "low"}
FIELDS = {
    "image_form", "assessment", "confidence", "image_description", "visible_evidence",
    "counterevidence", "reason", "needs_provenance", "edit_lineage",
}
CUE_FIELDS = {"location", "observation", "supports", "alternative_explanation"}
RULE_VERSION = "blind_consensus_candidates_v1_unvalidated"


def validate_parsed(parsed):
    """Return machine-readable errors; invalid JSON never casts an uncertain vote."""
    if not isinstance(parsed, dict):
        return ["parsed_not_object"]
    errors = []
    if set(parsed) != FIELDS:
        errors.append("parsed_fields_mismatch")
    for name, values in (("image_form", IMAGE_FORMS), ("assessment", ASSESSMENTS),
                         ("confidence", CONFIDENCES)):
        if not isinstance(parsed.get(name), str) or parsed[name] not in values:
            errors.append(f"invalid_{name}")
    for name in ("image_description", "reason"):
        if not isinstance(parsed.get(name), str) or not parsed[name].strip():
            errors.append(f"invalid_{name}")
    if type(parsed.get("needs_provenance")) is not bool:
        errors.append("invalid_needs_provenance")
    if parsed.get("edit_lineage") != "not_established_from_pixels":
        errors.append("invalid_edit_lineage")
    cues = parsed.get("visible_evidence")
    if not isinstance(cues, list) or len(cues) > 4:
        errors.append("invalid_visible_evidence")
    else:
        for i, cue in enumerate(cues):
            if not isinstance(cue, dict) or set(cue) != CUE_FIELDS:
                errors.append(f"invalid_cue_fields:{i}")
                continue
            if any(not isinstance(cue[key], str) for key in CUE_FIELDS):
                errors.append(f"invalid_cue_types:{i}")
            if not isinstance(cue["supports"], str) or cue["supports"] not in {"ai", "non_ai", "ambiguous"}:
                errors.append(f"invalid_cue_supports:{i}")
    counter = parsed.get("counterevidence")
    if not isinstance(counter, list) or len(counter) > 3 or any(not isinstance(v, str) for v in counter):
        errors.append("invalid_counterevidence")
    return errors


def is_url(value):
    return isinstance(value, str) and urlparse(value).scheme in {"https", "http"} and bool(urlparse(value).netloc)


def source_rule(sample):
    """Use registered provenance, never infer production history from VLM output."""
    evidence = sample.get("source_evidence")
    registered = (
        sample.get("source_evidence_level") not in {None, "", "unknown", "unverified"}
        and isinstance(evidence, list)
        and any(isinstance(item, dict) and is_url(item.get("url"))
                and isinstance(item.get("claim"), str) and bool(item["claim"].strip())
                for item in evidence)
    )
    origin = sample.get("source_image_origin")
    if origin in {"programmatic", "rendered"} and registered:
        return "protected_source"
    # A dedicated generation metadata record is evidence; a generic caption or URL is not.
    metadata = sample.get("generation_metadata")
    generation_record = isinstance(metadata, dict) and any(
        isinstance(metadata.get(key), str) and metadata[key].strip()
        for key in ("generator", "model", "generated_by")
    )
    if origin == "ai_generated" and (registered or is_url(sample.get("source_evidence_url")) or generation_record):
        return "source_reject"
    return "unresolved_source"


def ai_cue_count(parsed):
    # Repeated wording at the same location is not a second independent observation.
    return len({(cue["location"].strip(), cue["observation"].strip())
                for cue in parsed["visible_evidence"]
                if cue["supports"] == "ai" and cue["location"].strip() and cue["observation"].strip()})


def consensus(votes):
    if all(v["assessment"] == "likely_ai_generated" and v["confidence"] == "high"
           and ai_cue_count(v) >= 2 for v in votes):
        return "unanimous_high_ai_candidate"
    if all(v["assessment"] == "likely_non_ai" and v["confidence"] in {"medium", "high"}
           for v in votes):
        return "unanimous_non_ai_candidate"
    return "needs_adjudication"


def aggregate(samples, results, models):
    if len(models) != 3 or len(set(models)) != 3 or any(not isinstance(m, str) or not m for m in models):
        raise ValueError("Exactly three distinct model names are required")
    sample_map = {}
    for sample in samples:
        sha = sample.get("image_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid sample image_sha256")
        if sha in sample_map:
            raise ValueError(f"Duplicate sample image_sha256: {sha}")
        sample_map[sha] = sample
    if not sample_map:
        raise ValueError("Sample manifest is empty")

    indexed = defaultdict(list)
    unexpected = Counter()
    for row in results:
        # requested_model identifies the independently selected reviewer; a gateway's
        # response model may be a provider alias and is not a fourth vote.
        model = row.get("requested_model", row.get("model"))
        sha = row.get("image_sha256")
        if sha not in sample_map:
            unexpected["unknown_image"] += 1
        elif model not in models:
            unexpected["unexpected_model"] += 1
        else:
            indexed[(sha, model)].append(row)

    decisions = []
    for sha, sample in sample_map.items():
        reviews, errors, rounds = {}, [], set()
        for model in models:
            rows = indexed[(sha, model)]
            if len(rows) != 1:
                error = "missing_result" if not rows else "duplicate_results"
                reviews[model] = {"valid": False, "errors": [error]}
                errors.append({"model": model, "code": error})
                continue
            row = rows[0]
            issues = []
            round_name = row.get("round")
            if not isinstance(round_name, str) or not round_name:
                issues.append("missing_round")
            else:
                rounds.add(round_name)
            if row.get("status") != "ok":
                issues.append(f"api_status:{row.get('status', 'missing')}")
            else:
                issues.extend(validate_parsed(row.get("parsed")))
            reviews[model] = {"valid": not issues, "errors": issues,
                              "api_status": row.get("status"), "round": round_name}
            if not issues:
                reviews[model]["parsed"] = row["parsed"]
            errors.extend({"model": model, "code": issue} for issue in issues)
        if len(rounds) > 1:
            errors.append({"model": None, "code": "mixed_review_rounds"})
        api_status = "api_incomplete" if errors else consensus([reviews[m]["parsed"] for m in models])
        rule = source_rule(sample)
        final_status = rule if rule != "unresolved_source" else api_status
        decisions.append({
            "image_sha256": sha, "image_path": sample.get("image_path"),
            "review_id": sample.get("review_id"), "source": sample.get("source"),
            "domain": sample.get("domain"), "pilot_group": sample.get("pilot_group", "unspecified"),
            "source_image_origin": sample.get("source_image_origin"),
            "source_rule_decision": rule, "status": final_status, "review_status": api_status,
            "reviews": reviews, "errors": errors,
            "audit_only": rule != "unresolved_source",
            "ground_truth": sample.get("ground_truth"), "ground_truth_basis": sample.get("ground_truth_basis"),
            "automatic_deletion": False, "training_eligibility": None,
            "rule_version": RULE_VERSION,
        })

    def summarize(rows):
        per_model = {}
        for model in models:
            valid = [r["reviews"][model]["parsed"] for r in rows if r["reviews"][model]["valid"]]
            per_model[model] = {
                "valid_count": len(valid), "invalid_or_missing_count": len(rows) - len(valid),
                "assessment_counts": dict(Counter(v["assessment"] for v in valid)),
                "confidence_counts": dict(Counter(v["confidence"] for v in valid)),
            }
        return {
            "image_count": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
            "review_status_counts": dict(Counter(r["review_status"] for r in rows)),
            "source_rule_counts": dict(Counter(r["source_rule_decision"] for r in rows)),
            "error_counts": dict(Counter(e["code"] for r in rows for e in r["errors"])),
            "candidate_count": sum(r["status"] in {"unanimous_high_ai_candidate", "unanimous_non_ai_candidate"} for r in rows),
            "per_model": per_model,
        }

    groups = sorted({r["pilot_group"] for r in decisions} | {"detector_alert"})
    controls = [r for r in decisions if r["pilot_group"] == "known_ai_positive_control" and r["ground_truth"] == "ai_generated"]
    stats = {
        "rule_version": RULE_VERSION, "models": models, "automatic_deletion": False,
        "training_eligibility": None,
        "interpretation": "Unvalidated review candidates, not provenance facts or calibrated probabilities; no dataset changes.",
        "overall": summarize(decisions),
        "by_pilot_group": {g: summarize([r for r in decisions if r["pilot_group"] == g]) for g in groups},
        "by_domain": {d: summarize([r for r in decisions if (r["domain"] or "unspecified") == d])
                      for d in sorted({r["domain"] or "unspecified" for r in decisions})},
        "unexpected_result_counts": dict(unexpected),
        "known_ai_positive_control_summary": {
            **summarize(controls),
            "scope": "Convenience controls only; no estimate of general AI detection accuracy or recall.",
        },
        "general_accuracy_precision_recall": None,
    }
    indices = {status: [r["image_sha256"] for r in decisions if r["status"] == status]
               for status in sorted({r["status"] for r in decisions})}
    return decisions, stats, indices


def load_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{number}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--models", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decisions, stats, indices = aggregate(load_jsonl(args.samples), load_jsonl(args.results), args.models)
    stats["created_at"] = datetime.now(timezone.utc).isoformat()
    stats["input_sha256"] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for name, path in (("samples", args.samples), ("results", args.results), ("aggregator", __file__))}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("stats.json", stats), ("indices.json", indices)):
        temporary = output / (name + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / name)
    temporary = output / "decisions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions), encoding="utf-8")
    temporary.replace(output / "decisions.jsonl")
    print(json.dumps(stats["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
