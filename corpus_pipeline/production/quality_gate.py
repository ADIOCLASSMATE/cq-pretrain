"""Bound image-quality gate; no source mutation and no AI-origin certification.

The default policy changes only the fixed reviewed subset. Other images explicitly
remain unreviewed and continue their existing gates. An opt-in all_images scope
requires a bound blur receipt for every image; source reputation is no substitute.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from .common import digest

CONFIG_SCHEMA = "corpus.quality_policy.config.v1"
APPLICATION_SCHEMA = "corpus.quality_policy.application.v1"
BINDING_SCHEMA = "production.image_quality_policy.binding.v1"
RECEIPT_SCHEMA = "production.image_quality_gate.v1"
REVIEWED_SCOPE = 688
SCOPES = frozenset({"reviewed_subset", "all_images"})
HARD_REASONS = frozenset({"required_image_quality_excluded", "required_image_policy_ai_excluded"})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(binding):
    require(isinstance(binding, dict) and isinstance(binding.get("path"), str)
            and isinstance(binding.get("sha256"), str), "quality artifact binding required")
    path = Path(binding["path"])
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == binding["sha256"], "quality artifact SHA mismatch: " + str(path))
    if "bytes" in binding:
        require(len(raw) == binding["bytes"], "quality artifact size mismatch")
    return raw


def _json(binding):
    return json.loads(_read(binding))


def _rows(binding):
    raw = _read(binding)
    if str(binding["path"]).endswith(".gz"):
        raw = gzip.decompress(raw)
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(len(rows) == binding.get("records") == REVIEWED_SCOPE, "quality artifact must cover exactly 688 reviewed images")
    return rows


def _reference(path, raw):
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _expected_outcome(row):
    policy = row["policy_input"]
    baseline = row["baseline_screening"]
    if baseline.get("report_category") == "ai_excluded" or baseline.get("human_review", {}).get("decision") == "exclude_ai":
        return "reject", "ai_excluded"
    if policy["blur_status"] == "blur_present":
        return "reject", "quality_excluded"
    if policy["blur_status"] == "uncertain" or policy["image_origin_followup"]:
        return "pending", "visual_pending"
    if policy["context_action"] in {"exclude_pair", "drop_image_keep_text"}:
        return "reject", "quality_excluded"
    if policy["context_action"] in {"needs_context", "retain_caption_prune_invalid_qa"}:
        return "pending", "visual_pending"
    require(policy["context_action"] in {"keep", "flag_low_information"}, "unrecognized quality context action")
    return "admit", "admitted"


def load_bound_quality_policy(binding):
    """Revalidate frozen artifacts independently at publication/read boundaries.

    The mutable CURRENT pointer is verified by load_quality_policy at run start.
    Frozen releases bind its exact digest but read their immutable artifacts, so
    moving CURRENT later does not silently rewrite or invalidate an older release.
    """
    require(isinstance(binding, dict) and binding.get("schema_version") == BINDING_SCHEMA
            and binding.get("enabled") is True and binding.get("require_blur_evidence") is True,
            "enabled quality policy with required blur evidence is mandatory")
    require(binding.get("scope") in SCOPES, "quality policy scope must be frozen in the run binding")
    for key in ("config", "current_overlay", "input_manifest", "quality_decisions"):
        require(isinstance(binding.get(key), dict) and bool(binding[key].get("sha256"))
                and bool(binding[key].get("path")), "missing quality policy binding: " + key)
    config_text = binding.get("config_snapshot_utf8")
    require(isinstance(config_text, str)
            and hashlib.sha256(config_text.encode("utf-8")).hexdigest() == binding["config"]["sha256"],
            "frozen quality config snapshot SHA differs")
    config = json.loads(config_text)
    require(config.get("schema_version") == CONFIG_SCHEMA and config.get("enabled") is True
            and config.get("require_blur_evidence") is True
            and config.get("scope", "reviewed_subset") == binding["scope"],
            "quality scope differs from frozen config snapshot")
    manifest = _json(binding["input_manifest"])
    require(manifest.get("schema_version") == "corpus.quality_policy.bundle.v1"
            and manifest.get("records") == REVIEWED_SCOPE, "quality input manifest schema/scope differs")
    candidates = {row["item_id"]: row for row in _rows(manifest["candidates"])}
    proposed = {row["item_id"]: row for row in _rows(manifest["decisions"])}
    rows = _rows(binding["quality_decisions"])
    require(len(candidates) == len(proposed) == REVIEWED_SCOPE, "duplicate quality candidate/proposal ID")
    result, review_numbers = {}, set()
    for row in rows:
        aid, policy = row.get("item_id"), row.get("policy_input")
        require(aid in candidates and aid in proposed and aid not in result, "unknown or duplicate quality image ID")
        candidate = candidates[aid]
        require(row.get("schema_version") == "corpus.quality_policy.resolved_overlay.v1"
                and row.get("source_certification") is False, "invalid resolved quality schema/certification")
        require(policy == proposed[aid] and policy.get("source_input_sha256") == digest(candidate)
                and policy.get("schema_version") == "corpus.quality_policy.decision.v1"
                and policy.get("authority") == "user_policy_applied_by_model", "quality source input/proposal binding differs")
        require(row.get("source_ref") == policy.get("source_ref") == candidate.get("source_ref")
                and row["source_ref"].get("sha256") == aid, "quality source image SHA/reference differs")
        require(row.get("review_number") == policy.get("review_number") == candidate.get("review_number")
                and type(row["review_number"]) is int and row["review_number"] not in review_numbers,
                "duplicate or mismatched quality review number")
        require(row.get("blur_status") == policy.get("blur_status")
                and row["blur_status"] in {"clear", "blur_present", "uncertain"}
                and type(policy.get("image_origin_followup")) is bool
                and row.get("context_recommendation") == policy.get("context_action")
                and row.get("image_origin_followup") == policy["image_origin_followup"], "quality blur/context evidence differs")
        require(row.get("quality_policy_manifest", {}).get("sha256") == binding["input_manifest"]["sha256"]
                and Path(row["quality_policy_manifest"]["path"]).resolve() == Path(binding["input_manifest"]["path"]).resolve(),
                "quality decision manifest binding differs")
        require(isinstance(row.get("baseline_screening"), dict)
                and row["baseline_screening"].get("item_id") == aid, "quality baseline image differs")
        require((row.get("decision"), row.get("report_category")) == _expected_outcome(row),
                "resolved quality outcome contradicts bound policy")
        result[aid] = row
        review_numbers.add(row["review_number"])
    require(set(result) == set(candidates) == set(proposed)
            and review_numbers == set(range(1, REVIEWED_SCOPE + 1)), "quality reviewed coverage differs")
    return {"binding": deepcopy(binding), "decisions": result}


def load_quality_policy(config_path):
    config_path = Path(config_path).resolve()
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    require(config.get("schema_version") == CONFIG_SCHEMA and config.get("enabled") is True
            and config.get("require_blur_evidence") is True, "enabled quality policy with required blur evidence is mandatory")
    scope = config.get("scope", "reviewed_subset")
    require(scope in SCOPES, "quality policy scope must be reviewed_subset or all_images")
    require(isinstance(config.get("current_overlay"), str) and config["current_overlay"], "quality current_overlay path required")
    pointer_path = (config_path.parent / config["current_overlay"]).resolve()
    pointer_raw = pointer_path.read_bytes()
    pointer = json.loads(pointer_raw)
    require(pointer.get("schema_version") == APPLICATION_SCHEMA
            and pointer.get("revision_sha256") == pointer.get("input_manifest", {}).get("sha256"),
            "quality CURRENT revision binding differs")
    # Verify the publisher completion/report and its exact selected artifact.
    complete, report = _json(pointer["completion"]), _json(pointer["report"])
    require(complete.get("schema_version") == APPLICATION_SCHEMA and complete.get("complete") is True
            and complete.get("revision_sha256") == pointer["revision_sha256"]
            and complete.get("report_sha256") == pointer["report"]["sha256"], "quality application is not complete")
    require(report.get("input_manifest") == pointer["input_manifest"]
            and report.get("revision_sha256") == pointer["revision_sha256"], "quality report/pointer binding differs")
    artifact = report["artifacts"]["quality_decisions.jsonl.gz"]
    require(all(pointer["quality_decisions"].get(key) == value for key, value in artifact.items())
            and Path(pointer["quality_decisions"]["path"]).resolve() == Path(pointer["directory"]).resolve() / "quality_decisions.jsonl.gz",
            "quality decision pointer/report binding differs")
    audit_ref = {"path": str(Path(pointer["directory"]) / "AUDIT.json"), **report["artifacts"]["AUDIT.json"]}
    audit = _json(audit_ref)
    require(audit.get("passed") is True and audit.get("skipped_gates") == []
            and audit.get("policy_records") == REVIEWED_SCOPE, "quality application audit incomplete")
    binding = {"schema_version": BINDING_SCHEMA, "enabled": True, "require_blur_evidence": True, "scope": scope,
               "config": _reference(config_path, config_raw), "current_overlay": _reference(pointer_path, pointer_raw),
               "config_snapshot_utf8": config_raw.decode("utf-8"),
               "input_manifest": pointer["input_manifest"], "quality_decisions": pointer["quality_decisions"]}
    return load_bound_quality_policy(binding)


def binding_from_identity(identity):
    if not isinstance(identity, dict):
        return None
    return identity.get("quality_policy") or binding_from_identity(identity.get("run"))


def quality_receipt(asset_ids, state):
    slots, reasons = [], []
    for index, aid in enumerate(asset_ids):
        row = state["decisions"].get(aid)
        reason, action = None, "continue_existing_ai_and_data_gates"
        if row is None and state["binding"]["scope"] == "all_images":
            reason, action = "image_blur_review_missing", "review_image_before_admission"
        elif row is not None and row["report_category"] == "ai_excluded":
            reason, action = "required_image_policy_ai_excluded", "remove_image_and_repair_references"
        elif row is not None and row["report_category"] == "quality_excluded":
            reason, action = "required_image_quality_excluded", "remove_image_and_repair_references"
        elif row is not None and (row["decision"] != "admit" or row["blur_status"] != "clear"):
            reason, action = "image_quality_review_pending", "review_image_before_admission"
        if reason:
            reasons.append(reason)
        slots.append({"image_index": index, "asset_id": aid, "action": action, "reason_code": reason,
                      "blur_status": row["blur_status"] if row else "unreviewed",
                      "policy_decision": row["decision"] if row else ("pending" if reason else "not_reviewed"),
                      "report_category": row["report_category"] if row else ("visual_pending" if reason else "not_reviewed"),
                      "quality_scope": "reviewed_image" if row else ("required_but_unreviewed" if reason else "out_of_scope"),
                      "decision_sha256": digest(row) if row else None,
                      "review_number": row["review_number"] if row else None,
                      "original_image_preserved": True, "other_text_preserved": True,
                      "repair_applied": False, "source_certification": False})
    return {"schema_version": RECEIPT_SCHEMA, "policy_binding": deepcopy(state["binding"]),
            "reason_codes": sorted(set(reasons)), "slots": slots, "current_view_blocked": bool(reasons),
            "original_record_preserved": True, "source_images_modified": False,
            "image_origin_certified": False, "cleared_images_still_require_ai_screening": True,
            "all_images_still_require_ai_screening": True}


def apply_quality_gate(records, reasons, state):
    receipts = {}
    for record in records:
        sid = record["original_sample_id"]
        receipt = quality_receipt([ref["sha256"] for ref in record["source_image_refs"]], state)
        reasons[sid] = sorted(set(reasons[sid]) | set(receipt["reason_codes"]))
        receipts[sid] = receipt
    return receipts


def validate_quality_admission(asset_ids, admission, identity, *, state=None):
    """Independent publisher check; fabricated clear/missing-slot receipts fail."""
    receipt = admission.get("evidence", {}).get("image_quality")
    binding = binding_from_identity(identity)
    if binding is None and receipt is None:
        return  # Historical releases/explicit legacy test fixtures.
    require(isinstance(receipt, dict), "required image blur receipt missing")
    binding = binding or receipt.get("policy_binding")
    require(receipt.get("policy_binding") == binding, "image quality receipt/run identity mismatch")
    state = state or load_bound_quality_policy(binding)
    require(state["binding"] == binding, "quality validator bound to another revision")
    require(receipt == quality_receipt(asset_ids, state), "image quality receipt contradicts bound image decisions")
    if admission.get("decision") == "admit":
        require(not receipt["reason_codes"], "required image blur evidence has not cleared every slot")
        if asset_ids:
            evidence = admission["evidence"]
            screening = evidence.get("image_screening", [])
            require(evidence.get("image_screening_policy") == "image_screening.v3"
                    and Counter(row.get("asset_id") for row in screening) == Counter(asset_ids)
                    and all(row.get("screening", {}).get("decision") == "admit" for row in screening),
                    "quality clearance cannot replace required AI screening receipts")
    else:
        require(set(receipt["reason_codes"]) <= set(admission.get("reason_codes", [])), "image quality blocking reasons lost")
