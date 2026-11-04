"""Compile existing evidence into fixed-source profiles; never grant admission.

Usage: python -m corpus_pipeline.prepared.profiles --output-dir PATH
The original ordered source and adapter registries are retained without flattening.
All snapshot/schema validation remains explicit; historical counts are not reused
as current or future dataset counts. This module performs no image/API/model I/O.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path

from corpus_pipeline.policy import evaluate, normalize, resolve_source_config

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "corpus.prepared_profiles.v1"
COLLECTIONS = ("dcvlm", "genund")
NON_T2I_VIEWS = ["single_i2t", "multi_i2t", "interleaved", "pure_text", "mixed_multimodal"]
_LOOKUP_CACHE = {}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _lookups(registry):
    """Compiled registries are immutable during a run; avoid per-row registry scans."""
    key = id(registry)
    cached = _LOOKUP_CACHE.get(key)
    if cached is None or cached[0] is not registry:
        if len(_LOOKUP_CACHE) >= 4:
            _LOOKUP_CACHE.clear()
        cached = (registry,
                  {(group["collection"], group["source"].casefold()) for group in registry["source_groups"]},
                  {profile["profile_id"]: profile for profile in registry["profiles"]})
        _LOOKUP_CACHE[key] = cached
    return cached[1], cached[2]


def _collection_source(rule):
    match = rule.get("match", {})
    return (match.get("collection", match.get("corpus", rule.get("corpus"))),
            match.get("source", match.get("source_id", rule.get("source_id"))))


def _applicable(entry, collection, source):
    """Select potentially applicable adapter entries, retaining their narrow match."""
    if entry.get("sources"):
        return collection == "genund" and source in entry["sources"]
    candidate = deepcopy(entry)
    candidate["match"] = {key: value for key, value in candidate.get("match", {}).items()
                          if key not in ("path_glob", "path_prefix")}
    return bool(resolve_source_config(collection, source, {}, registry=[candidate]).get("_matched_rules"))


def _contracts(collection):
    return {
        "classification": {
            "method": "existing_policy.normalize_and_evaluate_roles_segments",
            "per_record_required": True,
            "directory_is_single_task": False,
            "api_required_for_known_schema": False,
            "qa_is_not_caption": True,
            "dcvlm_direction_conversion_allowed": False,
            "allowed_view_universe": NON_T2I_VIEWS + (["single_t2i", "multi_t2i"] if collection == "genund" else []),
            "t2i_conversion": "validated_single_image_caption_or_prompt_pair_only" if collection == "genund" else "forbidden",
            "native_t2i_requires": "genund_native_role_target_evidence",
            "validation": ["known_adapter_schema", "ordered_segments", "image_reference_binding",
                           "roles_and_completed_rounds", "caption_vs_qa_semantics", "collection_task_restriction"],
        },
        "recaption": {
            "caption_pair": {"action": "add_grounded_caption_sidecars", "preserve_original": True,
                             "t2i_length_views": ["short", "medium", "long"] if collection == "genund" else [],
                             "unique_image_multiplier": 1},
            "visual_qa": {"action": "verify_or_annotate_answer", "generic_caption_replacement": False},
            "multi_i2t": {"action": "context_preserving_turn_validation", "invent_turns": False},
            "multi_t2i": {"action": "annotate_existing_target_sequence_only", "invent_target_images": False,
                          "enabled_collection": collection == "genund"},
            "interleaved": {"action": "optional_image_sidecars_preserve_document_order", "replace_body": False},
            "pure_text": {"action": "skip_visual_recaption"},
            "text_provenance": "retain_original_and_mark_new_ai_annotations",
        },
        "layout": {"automatic_eligibility": False, "route": "selective_asset_level_branch",
                   "requires": ["image_policy_pass", "decode_and_task_resolution_pass", "structure_value",
                                "quality_review", "grounded_layout_validation"],
                   "does_not_change_native_task": True, "homepage_examples_are_not_source_proof": True},
        "size": {"global_short_side_threshold": 128, "threshold_state": "user_required_hard_minimum",
                 "comparison": "original_short_side_greater_than_or_equal",
                 "domain_exceptions": False, "upsampling_can_satisfy_gate": False,
                 "layout_threshold_not_global": True},
    }


def compile_profiles(repo_root=REPO, *, sources_path=None, adapters_path=None, rules_path=None):
    root = Path(repo_root).resolve()
    paths = {
        "sources": Path(sources_path or root / "corpus_pipeline/config/sources.json"),
        "adapters": Path(adapters_path or root / "corpus_pipeline/config/adapters.json"),
        "rules": Path(rules_path or root / "corpus_pipeline/config/rules.json"),
        "dcvlm_report": root / "reports/dcvlm_sources_20261022/dcvlm_sources.json",
        "dcvlm_families": root / "reports/dcvlm_sources_20261022/upstream_source_families.json",
        "genund_inventory": root / "reports/genund_20261021/inventory.json",
        "genund_contracts": root / "corpus_pipeline/prepared/config/genund_source_contracts.json",
    }
    documents = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    original = documents["sources"]
    rules = original["sources"]
    ids = [rule["id"] for rule in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source rule IDs cannot be compiled")
    report = documents["dcvlm_report"]
    dcvlm_sources = {row["source"]: row for row in report["sources"]}
    genund_sources = {row["dataset"]: row for row in documents["genund_inventory"]["results"]}
    genund_contracts = {row["source_id"]: row for row in documents["genund_contracts"]["contracts"]}
    if set(genund_sources) != set(genund_contracts):
        raise ValueError("Prepared genund contracts must exactly cover the observed 21-source inventory")
    families = documents["dcvlm_families"]
    parent_names = {collection: set() for collection in COLLECTIONS}
    for rule in rules:
        collection, source = _collection_source(rule)
        if collection not in COLLECTIONS or not source:
            raise ValueError("Only explicit fixed DCVLM/genund source rules are supported")
        if not any(key in rule.get("match", {}) for key in ("path_glob", "path_prefix")):
            parent_names[collection].add(source)
    missing = {"dcvlm": sorted(set(dcvlm_sources) - parent_names["dcvlm"]),
               "genund": sorted(set(genund_sources) - parent_names["genund"])}
    if any(missing.values()):
        raise ValueError("Observed inventory sources missing from registry: " + canonical(missing))
    profiles = []
    groups = defaultdict(list)
    for index, rule in enumerate(rules):
        collection, source = _collection_source(rule)
        known_formats = deepcopy(rule.get("formats", {}))
        is_parent = not any(key in rule.get("match", {}) for key in ("path_glob", "path_prefix"))
        historical = None
        hints = []
        if collection == "dcvlm" and is_parent and source in dcvlm_sources:
            observed = dcvlm_sources[source]
            historical = {
                "report_ref": "dcvlm_report", "source": source,
                "snapshot_root": report["root"],
                "scope": "historical_published_receipts_not_current_rescan",
                "caption_direction_usable": False,
                "retired_t2i50_snapshot": "t2i50" in report["root"].casefold(),
                "counts_not_inherited_by_new_snapshot": True,
                "documents": observed["exact"]["samples"],
                "native_format_counts": known_formats,
                "sampled_turn_counts": {key: deepcopy(observed.get("sampling", {}).get(key, {}))
                                       for key in ("visual_conversation", "text_conversation", "interleaved")},
                "sample_scope": "stratified_diagnostic_not_population_estimate",
            }
            for family in families.get("interleaved_families", []):
                if source in family["local_sources"]:
                    hints.append({"semantic_family": family["family"], "kind": family["kind"], "url": family["url"]})
            for category, names in families.get("paper_category_mapping", {}).items():
                if source in names:
                    hints.append({"upstream_semantic_category": category, "url": families["paper"],
                                  "local_category_requires_record_validation": True})
        adapter_refs = [entry["id"] for entry in documents["adapters"].get("sources", [])
                        if _applicable(entry, collection, source)]
        pending_refs = [entry["id"] for entry in documents["adapters"].get("pending_schema_overrides", [])
                        if _applicable(entry, collection, source)]
        origin = rule.get("image_origin", "unknown")
        profile = {
            "profile_id": rule["id"], "collection": collection, "source": source,
            "subset": rule.get("subset"), "scope": rule.get("scope", "source"),
            "match": deepcopy(rule.get("match", {})),
            "source_rule_ref": {"index": index, "id": rule["id"], "sha256": digest(rule)},
            "preparation": {
                "source_identity": "inventory_only" if rule.get("inventory_only") else "registered_known_source",
                "image_provenance": "unresolved" if origin in ("unknown", "mixed") else "source_declared_requires_asset_check",
                "schema": "pending_adapter_resolution" if pending_refs else (
                    "historical_format_observed_current_snapshot_requires_validation" if known_formats else "runtime_schema_validation_required"),
                "production_ready": False, "automatic_accept": False,
            },
            "provenance": {"image_origin_prior": origin, "caption_origin_prior": rule.get("caption_origin", "unknown"),
                           "evidence_level": rule.get("evidence_level"), "evidence_ref": f"source_rules[{index}].evidence",
                           "evidence_is_not_per_image_certification": True,
                           "unknown_action": "quarantine", "policy_ref": "original_policy"},
            "native_formats_observed": known_formats,
            "semantic_evidence": hints, "historical_statistics": historical,
            "adapter_rule_refs": adapter_refs, "pending_adapter_refs": pending_refs,
            "source_contract_ref": {"document": "genund_source_contracts", "source_id": source,
                                    "sha256": digest(genund_contracts[source])} if collection == "genund" else None,
            "contracts_ref": collection,
            "snapshot_contract": {"every_new_snapshot_requires_inventory_diff": True,
                                  "every_new_snapshot_requires_schema_validation": True,
                                  "historical_counts_inherited": False, "future_balance200b_autoaccept": False},
        }
        profiles.append(profile)
        groups[(collection, source)].append(profile["profile_id"])
    input_evidence = {}
    for name, path in paths.items():
        try:
            reference = str(path.resolve().relative_to(root))
        except ValueError:
            reference = str(path.resolve())
        input_evidence[name] = {"path": reference, "sha256": file_sha(path),
                                "version": documents[name].get("version")}
    result = {
        "schema_version": SCHEMA, "compiler_version": "2026-10-22.prepared.1",
        "purpose": "fixed_source_preparation_not_training_admission", "input_evidence": input_evidence,
        "source_registry_metadata": {key: deepcopy(value) for key, value in original.items() if key != "sources"},
        "source_rules": deepcopy(rules), "adapter_registry": deepcopy(documents["adapters"]),
        "original_policy": deepcopy(documents["rules"]), "contracts": {c: _contracts(c) for c in COLLECTIONS},
        "genund_source_contracts": deepcopy(documents["genund_contracts"]),
        "profiles": profiles,
        "source_groups": [{"collection": c, "source": s, "profile_ids": values}
                          for (c, s), values in sorted(groups.items())],
        "coverage": {"source_rules_preserved": len(rules), "compiled_profiles": len(profiles),
                     "parent_sources": {key: len(value) for key, value in parent_names.items()},
                     "inventory_sources": {"dcvlm": len(dcvlm_sources), "genund": len(genund_sources)},
                     "missing_inventory_sources": missing,
                     "readiness_counts": dict(Counter(p["preparation"]["schema"] for p in profiles)),
                     "full_dataset_rows_read": False, "api_calls": 0},
        "unregistered_source_action": "quarantine_requires_profile_and_adapter_review",
        "snapshot_validation": "required_even_when_name_matches; no current/future counts inferred",
    }
    result["content_sha256"] = digest(result)
    return result


def match_profiles(registry, collection, source, paths=()):
    """Return all matching profiles, including narrow negative rules, in policy order.

    Inventory-only rules remain in compiled storage but are not executable matches.
    A broad source match never permits discarding narrower image-path evidence.
    """
    config = resolve_source_config(collection, source, {"paths": list(paths)}, registry=registry["source_rules"])
    # _paths deliberately handles strings/lists/dicts; paths is an audit locator.
    _, by_id = _lookups(registry)
    return [by_id[rule["id"]] for rule in config.get("_matched_rules", [])]


def validate_record(registry, envelope):
    """Cheap stage-1 validation; returns candidates, never final release eligibility.

    The caller resolves image references through an existing adapter. This uses the
    unmodified policy for roles, exact path overrides, AI captions, and provenance.
    """
    collection, source = envelope.get("collection"), envelope.get("source")
    known_sources, _ = _lookups(registry)
    known = (collection, str(source).casefold()) in known_sources
    if collection not in COLLECTIONS or not known:
        return {"validation_status": "unregistered_source", "stage1_decision": "quarantine",
                "release_eligibility": None, "candidate_views": [], "reason_codes": ["unregistered_source"]}
    policy = {**registry["original_policy"], "sources": registry["source_rules"]}
    config = resolve_source_config(collection, source, envelope.get("locator", {}), policy)
    record = evaluate(normalize(envelope, config), policy)
    views = list(record["eligible_views"])
    # Existing native T2I remains inspectable. Conversion is only for a validated
    # gen-und caption pair, never arbitrary QA or a multi-image conversation.
    if collection != "genund":
        views = [view for view in views if view not in ("single_t2i", "multi_t2i")]
    elif (record["structure"] != "caption_pair" and record["format"] != "caption"
          and record["category"] not in ("single_t2i", "multi_t2i")):
        views = [view for view in views if view not in ("single_t2i", "multi_t2i")]
    all_matches = resolve_source_config(collection, source,
        {"paths": [envelope.get("locator", {}), *record.get("images", [])]}, policy)
    return {"validation_status": "stage1_checked", "stage1_decision": record["decision"],
            "release_eligibility": None, "native_category": record["category"],
            "structure": record["structure"], "completed_rounds": record["completed_rounds"],
            "candidate_views": views, "reason_codes": record["reason_codes"],
            "provenance": record["provenance"], "sample_id": record["sample_id"],
            "matched_profile_ids": [rule["id"] for rule in all_matches.get("_matched_rules", [])],
            "image_provenance_evidence_retained": True,
            "remaining_release_stages": ["snapshot_validation", "image_quality", "ai_screening",
                                         "deduplication", "annotation_quality", "split_checks", "release_commit"]}


def validate_snapshot(registry, collection, snapshot_id, *, observations=None):
    """Check supplied validation receipts, not the underlying dataset itself.

    An explicit changed snapshot requires schema and inventory evidence. Even when
    receipts are sufficient, this function never inherits historical row counts.
    """
    observed = observations or {}
    issues = []
    if collection not in COLLECTIONS:
        issues.append("unsupported_collection")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        issues.append("missing_snapshot_id")
    if observed.get("schema_validated") is not True or not observed.get("schema_evidence_sha256"):
        issues.append("schema_validation_required")
    if observed.get("inventory_validated") is not True or not observed.get("inventory_evidence_sha256"):
        issues.append("inventory_validation_required")
    names = observed.get("source_names")
    if not isinstance(names, list) or not names:
        issues.append("source_inventory_required")
        names = []
    known = {group["source"].casefold() for group in registry["source_groups"] if group["collection"] == collection}
    new = sorted(str(name) for name in names if str(name).casefold() not in known)
    if new:
        issues.append("unprepared_sources")
    for field in ("schema_evidence_sha256", "inventory_evidence_sha256", "snapshot_sha256"):
        value = observed.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            issues.append("invalid_or_missing_" + field)
    if observed.get("snapshot_id") != snapshot_id:
        issues.append("validation_receipt_snapshot_mismatch")
    return {"snapshot_id": snapshot_id, "collection": collection,
            "status": "requires_validation" if issues else "validation_receipts_present",
            "reason_codes": sorted(set(issues)), "unprepared_sources": new,
            "historical_counts_inherited": False, "production_admission": False,
            "verification_scope": "checks_supplied_receipts_only_no_dataset_io"}


def write_profiles(registry, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "prepared_profiles.json"
    temporary = target.with_suffix(".json.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(registry, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    columns = ["profile_id", "collection", "source", "subset", "scope", "match_json", "image_origin_prior",
               "caption_origin_prior", "source_identity", "schema_state", "automatic_accept", "native_formats_json",
               "adapter_rule_refs", "pending_adapter_refs", "source_rule_sha256"]
    csv_target = output / "prepared_profiles.csv"
    csv_temp = csv_target.with_suffix(".csv.partial")
    with csv_temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for p in registry["profiles"]:
            row = {key: p[key] for key in ("profile_id", "collection", "source", "subset", "scope")}
            row.update(match_json=canonical(p["match"]), image_origin_prior=p["provenance"]["image_origin_prior"],
                       caption_origin_prior=p["provenance"]["caption_origin_prior"],
                       source_identity=p["preparation"]["source_identity"], schema_state=p["preparation"]["schema"],
                       automatic_accept=False, native_formats_json=canonical(p["native_formats_observed"]),
                       adapter_rule_refs=canonical(p["adapter_rule_refs"]), pending_adapter_refs=canonical(p["pending_adapter_refs"]),
                       source_rule_sha256=p["source_rule_ref"]["sha256"])
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(csv_temp, csv_target)
    return {"json": str(target), "csv": str(csv_target), "content_sha256": registry["content_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    registry = compile_profiles(args.repo_root)
    result = write_profiles(registry, args.output_dir)
    print(canonical({**result, "coverage": registry["coverage"]}))


if __name__ == "__main__":
    main()
