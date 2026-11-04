"""Pure, deterministic staged annotation planning; a plan never admits training data.

Source adapters establish task and image/text pairing. This module does not infer
those facts from a model, a directory name, a text length, or SSCD similarity.
Original/source-local text may be retained by the early release policy without
pretending it received per-annotation visual QC. External annotation reuse and
high-resolution coverage require explicit, bound quality evidence.
"""
from __future__ import annotations

from copy import deepcopy
import re

from .common import digest
from .validation import semantic_contract


PHASES = {"early_mixed", "highres_1mp"}
LENGTHS = ("short", "medium", "long")
GENERATION_ACTIONS = {"generate_missing", "regenerate_verified_failure", "complete_highres_caption"}
CAPTION_FIELDS = {
    "caption": ("original", "original", None),
    "recaption": ("original", "existing_annotation", None),
    "re_caption": ("original", "existing_annotation", None),
    "detailed_caption": ("detailed", "existing_annotation", None),
    "caption_internlm2": ("detailed", "existing_annotation", "ai"),
    "caption_internlm2_short": ("short", "existing_annotation", "ai"),
    "caption_florence2": ("detailed", "existing_annotation", "ai"),
    "caption_florence2_short": ("short", "existing_annotation", "ai"),
    "caption_sharecap": ("detailed", "existing_annotation", "ai"),
    "caption_sharecap_short": ("short", "existing_annotation", "ai"),
}


def asset_view_identity(record, image_index):
    """Identity is exact bytes plus declared view, never a nearest neighbour."""
    refs = record.get("source_image_refs", [])
    if isinstance(image_index, bool) or not isinstance(image_index, int) or not 0 <= image_index < len(refs):
        return None
    ref = refs[image_index]
    sha = ref.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha):
        return None
    return digest({"sha256": sha.lower(), "view": ref.get("view", "full_original_exif_oriented"),
                   "crop": ref.get("crop"), "transform": ref.get("transform")})


def _task(record):
    return record["normalized"]["category"], semantic_contract(record)


def _contract_id(record):
    native, semantic = _task(record)
    return digest({"native_task": native, "semantic_task": semantic,
                   "caption_contract": record.get("caption_contract")})


def _origin(record):
    value = record.get("normalized", {}).get("provenance", {}).get("text", {}).get("origin", "unknown")
    return value if value in {"human", "ai", "mixed", "unknown"} else "unknown"


def _native_target_fields(record):
    target = (record.get("caption_contract") or {}).get("target_segment_index")
    if isinstance(target, int) and not isinstance(target, bool):
        return {f"segments[{target}].text"}
    if record["raw"].get("format") == "caption":
        return {f"segments[{index}].text" for index, segment in enumerate(record["raw"].get("segments", []))
                if segment.get("kind") == "text" and segment.get("role") in {None, "assistant"}}
    return set()


def _candidate(record, text, field, *, image_index=0, granularity="original",
               annotation_kind="original", origin=None, source_ref=None, **metadata):
    native, semantic = _task(record)
    value = {"text": text, "image_index": image_index, "granularity": granularity,
             "origin": origin if origin in {"human", "ai", "mixed", "unknown"} else _origin(record),
             "annotation_kind": annotation_kind, "field": field,
             "source_ref": source_ref or {"locator": deepcopy(record.get("locator", {})), "field": field},
             "source_record_sha256": record.get("source_record_sha256"),
             "asset_view_identity": asset_view_identity(record, image_index),
             "native_task": native, "semantic_task": semantic,
             "contract_id": _contract_id(record),
             "language": metadata.get("language", record.get("caption_language", "unknown")),
             "binding_scope": "source_local",
             "quality_evidence": deepcopy(metadata.get("quality_evidence")),
             "risk_flags": list(metadata.get("risk_flags") or [])}
    value["candidate_id"] = digest({key: value[key] for key in (
        "text", "image_index", "granularity", "source_record_sha256", "asset_view_identity", "field", "contract_id")})
    return value


def extract_candidates(record):
    """Recover source-local candidates only for an explicit caption contract.

    Native conversations need an adapter-selected target. Known alternative
    fields remain separate candidates (Megalith's generic field can be a URL).
    A detailed field is not labelled a qualified long description by its name.
    """
    if _task(record)[1] != "caption":
        return []
    raw = record["raw"]
    contract = record.get("caption_contract") or {}
    image_index = contract.get("image_index", 0 if len(record.get("source_image_refs", [])) == 1 else None)
    result = []
    target = contract.get("target_segment_index")
    segments = raw.get("segments", [])
    if isinstance(target, int) and not isinstance(target, bool) and 0 <= target < len(segments):
        segment = segments[target]
        if segment.get("kind") == "text":
            result.append(_candidate(record, segment.get("text"), f"segments[{target}].text", image_index=image_index))
    elif raw.get("format") == "caption":
        for index, segment in enumerate(segments):
            if segment.get("kind") == "text" and segment.get("role") in {None, "assistant"}:
                result.append(_candidate(record, segment.get("text"), f"segments[{index}].text", image_index=image_index))
    if len(record.get("source_image_refs", [])) == 1:
        for field, (granularity, kind, origin) in CAPTION_FIELDS.items():
            if field in raw:
                result.append(_candidate(record, raw[field], field, image_index=image_index,
                                         granularity=granularity, annotation_kind=kind, origin=origin))
    # This field belongs to an adapter, not arbitrary metadata inside raw.
    for index, item in enumerate(record.get("caption_candidates", [])):
        if not isinstance(item, dict):
            continue
        result.append(_candidate(record, item.get("text"), item.get("field", f"caption_candidates[{index}]"),
                                 image_index=item.get("image_index", image_index),
                                 granularity=item.get("granularity", "original"),
                                 annotation_kind=item.get("annotation_kind", "existing_annotation"),
                                 origin=item.get("origin"), source_ref=item.get("source_ref"),
                                 language=item.get("language", "unknown"),
                                 quality_evidence=item.get("quality_evidence"), risk_flags=item.get("risk_flags")))
    seen = set()
    unique = []
    for item in result:
        if item["candidate_id"] not in seen:
            seen.add(item["candidate_id"])
            unique.append(item)
    return unique


def _text_issue(text):
    if text is None or (isinstance(text, str) and not text.strip()):
        return "empty"
    if not isinstance(text, str):
        return "non_string"
    value = text.strip()
    if re.fullmatch(r"https?://[^\s<>]+", value, re.IGNORECASE):
        return "url_only"
    if re.fullmatch(r"(?:/|\./|\.\./)[^\s]+\.(?:jpe?g|png|webp|gif|bmp|tiff?|json|txt|parquet)", value, re.IGNORECASE):
        return "path_only"
    return None


def _evidence_for(candidate, quality_evidence):
    if not isinstance(quality_evidence, dict):
        return candidate.get("quality_evidence") or {}
    entries = quality_evidence.get("candidates")
    if isinstance(entries, dict):
        return entries.get(candidate.get("candidate_id"), candidate.get("quality_evidence") or {})
    return quality_evidence


def _quality(candidate, evidence, record, phase):
    """Caller supplies reviewed artifacts; unbound claims can only raise review."""
    if not isinstance(evidence, dict):
        return "not_reviewed", False, "not_reviewed"
    verdict = evidence.get("verdict")
    aliases = {"pass": "usable", "fail": "needs_correction", "revise": "needs_correction"}
    verdict = aliases.get(verdict, verdict)
    scope = evidence.get("scope", "not_reviewed")
    ids = evidence.get("candidate_ids", [evidence.get("candidate_id")])
    bound = (scope == "per_annotation_visual_review" and evidence.get("verified") is True
             and bool(evidence.get("evidence_ref"))
             and evidence.get("source_record_sha256") == record.get("source_record_sha256")
             and bool(record.get("source_record_sha256"))
             and isinstance(ids, list) and candidate.get("candidate_id") in ids)
    phase_compatible = phase == "early_mixed" or evidence.get("phase") == "highres_1mp"
    if bound and verdict == "usable" and candidate.get("risk_flags"):
        return "suspected", False, scope
    if bound and verdict == "usable":
        return "passed", phase_compatible, scope
    if bound and verdict == "needs_correction":
        return "failed", False, scope
    if verdict in {"needs_correction", "uncertain"} or candidate.get("risk_flags"):
        return "suspected", False, scope
    return "not_reviewed", False, scope


def _compatible(candidate, record):
    native, semantic = _task(record)
    if not isinstance(candidate.get("candidate_id"), str) or not candidate["candidate_id"]:
        return "candidate_identity_missing"
    if candidate.get("native_task") != native or candidate.get("semantic_task") != semantic:
        return "task_mismatch"
    identity = asset_view_identity(record, candidate.get("image_index"))
    if identity is None or candidate.get("asset_view_identity") != identity:
        return "asset_view_mismatch"
    if candidate.get("contract_id") != _contract_id(record):
        return "instruction_contract_mismatch"
    required_language = (record.get("caption_contract") or {}).get("required_language")
    if required_language and candidate.get("language") != required_language:
        return "language_mismatch"
    if candidate.get("binding_scope") == "source_local" and candidate.get("source_record_sha256") != record.get("source_record_sha256"):
        return "source_record_mismatch"
    return None


def _expand_catalog_candidate(record, candidate):
    """Bridge the immutable release catalogue into individual caption views."""
    # Local import keeps the pure planner usable without loading publishing code
    # unless an external signed-by-checksum artifact actually needs validation.
    from .staged_publish import _qualified, _validate_candidate
    _validate_candidate(candidate)
    if (candidate["record_id"] != record.get("original_sample_id")
            or candidate["source_record_sha256"] != record.get("source_record_sha256")
            or candidate["collection"] != record.get("collection", "dcvlm")
            or candidate["native_task"] != _task(record)[0]
            or candidate["asset_ids"] != [ref.get("sha256") for ref in record.get("source_image_refs", [])]):
        raise ValueError("catalog candidate context mismatch")
    provenance = candidate["provenance"]
    receipt = candidate["quality_receipt"]["receipt"]
    generated = candidate.get("generator_receipt")
    annotation = (generated["receipt"].get("parsed", {}) if generated else
                  receipt.get("evidence", {}).get("request_context", {}).get("candidate_annotation", {}))
    concerns = annotation.get("concerns") or []
    risk_flags = list(candidate.get("quality_flags") or [])
    if concerns and "GENERATOR_CONCERNS_RETAINED" not in risk_flags:
        risk_flags.append("GENERATOR_CONCERNS_RETAINED")
    # Highres needs a coherent, fully reviewed triplet artifact. Three partial
    # catalog entries cannot be silently combined into a newly certified set.
    phase = "highres_1mp" if _qualified(candidate, "highres_1mp") else "early_mixed"
    expanded = []
    for length, text in candidate["captions"].items():
        view = _candidate(record, text, "catalog." + candidate["candidate_id"] + "." + length,
                          image_index=0, granularity=length, annotation_kind="existing_annotation",
                          origin="ai" if candidate["origin"] == "generated" else provenance.get("text_origin"),
                          source_ref={"catalog_candidate_id": candidate["candidate_id"], "length": length},
                          language=provenance.get("language", "unknown"))
        view.update(binding_scope="verified_catalog", catalog_candidate_id=candidate["candidate_id"],
                    risk_flags=risk_flags, actual_review_verdict=receipt["verdict"],
                    annotation_concerns=deepcopy(concerns),
                    quality_evidence={"scope": receipt["scope"], "verified": True,
                                      "verdict": receipt["verdict"], "phase": phase,
                                      "evidence_ref": candidate["quality_receipt"]["receipt_sha256"],
                                      "source_record_sha256": candidate["source_record_sha256"],
                                      "candidate_ids": [view["candidate_id"]]})
        expanded.append(view)
    return expanded


def plan_record(record, *, phase="early_mixed", gate_reasons=(), quality_evidence=None,
                candidates=None, in_stage_pool=True):
    """Return an auditable plan without network calls or source mutation.

    ``quality_evidence`` is an explicit trusted input, never a raw dataset field.
    It is either one review or ``{'candidates': {candidate_id: review}}``. Reviews
    bind source_record_sha256 and candidate_ids and provide scope, verified,
    verdict and evidence_ref. Highres reuse additionally requires phase=highres_1mp.
    """
    if phase not in PHASES:
        raise ValueError("unknown annotation phase: " + str(phase))
    if not isinstance(in_stage_pool, bool):
        raise ValueError("in_stage_pool must be boolean")
    native, semantic = _task(record)
    supplied = deepcopy(candidates or [])
    if not isinstance(supplied, list) or any(not isinstance(item, dict) for item in supplied):
        raise ValueError("candidates must be a list of objects")
    expanded, catalog_errors = [], []
    for item in supplied:
        if "captions" in item:
            try:
                expanded.extend(_expand_catalog_candidate(record, item))
            except (KeyError, TypeError, ValueError) as error:
                catalog_errors.append({"candidate_id": item.get("candidate_id"), "error": str(error)})
        else:
            expanded.append(item)
    candidate_list = list({item.get("candidate_id", digest(item)): item
                           for item in extract_candidates(record) + expanded}.values())
    pairing = record.get("pairing_state", "unresolved")
    plan = {"schema_version": "production.annotation_plan.v1", "policy_revision": "staged_recaption.v2.early_uncertainty_retention",
            "phase": phase, "training_phase": phase, "in_stage_pool": in_stage_pool,
            "original_sample_id": record.get("original_sample_id"),
            "source_record_sha256": record.get("source_record_sha256"),
            "caption_origin": _origin(record), "original_usable": False,
            "retained_with_uncertainty": False, "quality_flags": [], "uncertain_candidate_refs": [],
            "source_quality_risk_flags": deepcopy(record.get("caption_risk_flags") or []),
            "original_text_provenance": deepcopy(record.get("normalized", {}).get("provenance", {}).get("text", {})),
            "source_bounds": {"collection": record.get("collection"), "source": record.get("source"),
                              "source_record_sha256": record.get("source_record_sha256"),
                              "locator": deepcopy(record.get("locator", {}))},
            "native_task": native, "semantic_task": semantic, "pairing_state": pairing,
            "caption_contract_id": _contract_id(record), "candidates": candidate_list,
            "selected_candidate_refs": [], "selected_catalog_candidate_ids": [],
            "catalog_errors": catalog_errors, "caption_state": "unknown",
            "qualified_coverage": False, "qualified_lengths_by_image": {},
            "required_lengths": list(LENGTHS) if phase == "highres_1mp" and semantic == "caption" else [],
            "allowed_training_lengths": (record.get("caption_contract") or {}).get("allowed_training_lengths", list(LENGTHS)),
            "training_admission": "requires_separate_existing_gates_and_release_policy",
            "automatic_training_admission": False}

    def finish(action, *reasons, selected=()):
        plan.update(action=action, reason_codes=list(reasons),
                    selected_candidate_refs=[item["candidate_id"] for item in selected],
                    selected_catalog_candidate_ids=sorted({item["catalog_candidate_id"] for item in selected
                                                           if item.get("catalog_candidate_id")}),
                    requires_generation=action in GENERATION_ACTIONS,
                    requires_review=action == "review_caption")
        plan["plan_id"] = digest(plan)
        return plan

    blockers = sorted(set(str(reason) for reason in gate_reasons))
    if blockers:
        return finish("blocked_by_existing_gate", *blockers)
    if record.get("prepared_adapter_status") == "repair_metadata":
        return finish("repair_metadata", "PREPARED_TASK_ADAPTER_UNRESOLVED")
    if catalog_errors:
        return finish("repair_metadata", "CATALOG_CANDIDATE_BINDING_OR_CHECKSUM_INVALID")
    if native == "pure_text":
        return finish("preserve_native_task", "PURE_TEXT_HAS_NO_VISUAL_RECAPTION")
    if pairing != "verified":
        return finish("repair_metadata", "IMAGE_ANNOTATION_BINDING_" + str(pairing).upper())
    if semantic != "caption":
        return finish("preserve_native_task", "NATIVE_NONCAPTION_SUPERVISION_PRESERVED")
    if phase == "highres_1mp" and not in_stage_pool:
        return finish("not_in_stage_pool", "OUTSIDE_FROZEN_HIGHRES_CAPTION_POOL")
    if not record.get("source_image_refs"):
        return finish("repair_metadata", "CAPTION_HAS_NO_BOUND_IMAGE")
    if not record.get("source_record_sha256"):
        return finish("repair_metadata", "SOURCE_RECORD_IDENTITY_MISSING")

    viable, failed, suspect = [], [], []
    for item in candidate_list:
        issue = _text_issue(item.get("text")) or _compatible(item, record)
        item["unusable_reason"] = issue
        evidence = _evidence_for(item, quality_evidence)
        status, qualified, scope = _quality(item, evidence, record, phase)
        if (phase == "early_mixed" and record.get("caption_risk_flags")
                and item.get("binding_scope") == "source_local" and status != "failed"):
            status, qualified = "suspected", False
        item.update(quality_status=status, qualified=qualified, quality_evidence_scope=scope)
        if issue:
            continue
        if status == "failed":
            failed.append(item)
        elif status == "suspected":
            if phase == "early_mixed" and item.get("binding_scope") in {"source_local", "verified_catalog"}:
                # Early general-knowledge training retains source text and
                # validated imported/generated candidates unless a bound review
                # establishes an error. Uncertainty never becomes a QC pass.
                item["retained_with_uncertainty"] = True
                viable.append(item)
            else:
                suspect.append(item)
        elif item.get("binding_scope") == "source_local" or status == "passed":
            viable.append(item)
        else:
            suspect.append(item)

    image_indices = range(len(record["source_image_refs"]))
    qualified_by_image = {str(index): sorted({item["granularity"] for item in viable
                                            if item.get("image_index") == index and item.get("qualified")
                                            and item.get("granularity") in LENGTHS}) for index in image_indices}
    plan["qualified_lengths_by_image"] = qualified_by_image
    allowed_lengths = plan["allowed_training_lengths"]
    native_fields = _native_target_fields(record)
    for item in viable:
        native_original = (item.get("annotation_kind") == "original" and item.get("binding_scope") == "source_local"
                           and item.get("field") in native_fields)
        unrestricted_source_variant = (item.get("binding_scope") == "source_local"
                                       and set(allowed_lengths) == set(LENGTHS)
                                       and item.get("granularity") in {"original", "detailed"})
        item["training_compatible"] = (native_original
                                       or item.get("granularity") in allowed_lengths
                                       or unrestricted_source_variant)
    training_viable = [item for item in viable if item["training_compatible"]]
    all_images_usable = all(any(item.get("image_index") == index for item in training_viable) for index in image_indices)
    originals = [item for item in training_viable if item.get("annotation_kind") == "original"
                 and item.get("binding_scope") == "source_local" and item.get("field") in native_fields]
    plan["original_usable"] = all(any(item.get("image_index") == index for item in originals) for index in image_indices)
    plan["qualified_coverage"] = all(set(values) == set(LENGTHS) for values in qualified_by_image.values())

    if phase == "highres_1mp":
        if plan["qualified_coverage"]:
            plan["caption_state"] = "usable_generated"
            return finish("reuse_annotation", "QUALIFIED_HIGHRES_TRIPLET_COMPLETE", selected=[item for item in viable if item.get("qualified")])
        if record.get("caption_fields_inspected") is not True:
            return finish("repair_metadata", "KNOWN_CAPTION_FIELDS_NOT_FULLY_INSPECTED")
        if suspect or record.get("caption_risk_flags"):
            plan["caption_state"] = "quality_suspected"
            return finish("review_caption", "UNRESOLVED_CAPTION_RISK_BEFORE_HIGHRES_COMPLETION")
        return finish("complete_highres_caption", "QUALIFIED_HIGHRES_TRIPLET_INCOMPLETE", selected=viable)

    if all_images_usable:
        original_covers = plan["original_usable"]
        uncertain = [item for item in training_viable if item.get("retained_with_uncertainty")]
        plan["retained_with_uncertainty"] = bool(uncertain)
        plan["uncertain_candidate_refs"] = [item["candidate_id"] for item in uncertain]
        plan["quality_flags"] = sorted({"CAPTION_QUALITY_UNCERTAIN_RETAINED",
                                        *(flag for item in uncertain for flag in item.get("risk_flags", []))}) if uncertain else []
        if uncertain:
            plan["qualified_coverage"] = False
        plan["caption_state"] = "quality_suspected" if uncertain else "usable_original" if original_covers else "usable_generated"
        return finish("reuse_original" if original_covers else "reuse_annotation",
                      "EARLY_UNCERTAINTY_RETAIN_AND_MARK" if uncertain else "EARLY_TASK_HAS_USABLE_DESCRIPTION",
                      selected=training_viable)
    if suspect:
        plan["caption_state"] = "quality_suspected"
        return finish("review_caption", "CAPTION_QUALITY_REQUIRES_REVIEW")
    if record.get("caption_fields_inspected") is not True:
        return finish("repair_metadata", "KNOWN_CAPTION_FIELDS_NOT_FULLY_INSPECTED")
    if failed:
        plan["caption_state"] = "quality_failure_verified"
        return finish("regenerate_verified_failure", "BOUND_REVIEW_CONFIRMS_CAPTION_FAILURE", selected=viable)
    if viable and not all_images_usable:
        return finish("generate_missing", "CURRENT_TASK_COMPATIBLE_DESCRIPTION_MISSING", selected=training_viable)
    # Identity/contract mismatch is not a missing-caption finding. A foreign
    # candidate must never hide a metadata error or trigger invented supervision.
    if any(item.get("unusable_reason") not in {None, "empty", "url_only", "path_only"} for item in candidate_list):
        return finish("repair_metadata", "CAPTION_CANDIDATE_BINDING_OR_SCHEMA_INVALID")
    plan["caption_state"] = "missing_verified"
    return finish("generate_missing", "NO_USABLE_CAPTION_AFTER_VERIFIED_FIELD_RECOVERY", selected=viable)
