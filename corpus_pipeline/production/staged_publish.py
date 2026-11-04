"""Immutable staged reference releases with explicit retention and candidate receipts.

This publisher never rewrites source records or expands their training exposure.
Model responses, existing annotations, and retained native supervision have distinct
provenance. A release is a frozen snapshot of the supplied batch, not an implicit
merge with an earlier training phase.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import fcntl
import json
import math
from pathlib import Path
import shutil
import sqlite3
import uuid
from urllib.parse import quote

from .common import atomic_json, digest, file_sha, jsonl, write_jsonl
from .records import load_original_record, read_asset_bytes

SCHEMA = "production.staged_reference_release.v1"
DATASET_SCHEMA = "production.staged_dataset.v2"
LENGTHS = ("short", "medium", "long")
PHASES = {"early_mixed", "highres_1mp"}
PENDING_ACTIONS = {"repair_metadata", "review_caption", "blocked_by_existing_gate"}
RETAIN_ACTIONS = {"reuse_original", "preserve_native_task", "not_in_stage_pool", "clean_surface"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _record_key(record):
    return record["original_sample_id"]


def _assets(record):
    return [ref["sha256"] for ref in record["source_image_refs"]]


def _bound_receipt(value, content_sha256):
    payload = {"candidate_content_sha256": content_sha256, "receipt": deepcopy(value)}
    return {**payload, "receipt_sha256": digest(payload)}


def _model_context(record):
    from .run import model_context
    return model_context(record)


def _model_receipt(receipt, assets, role):
    _require(isinstance(receipt, dict) and receipt.get("status") == "ok"
             and receipt.get("real_model_inference") is True, "actual model inference receipt required")
    identity = receipt.get("identity", {})
    _require(identity.get("role") == role and identity.get("assets") == assets
             and bool(identity.get("model")) and receipt.get("cache_key") == digest(identity),
             "model receipt role/assets/cache identity mismatch")
    image_inputs = receipt.get("image_inputs")
    _require(isinstance(image_inputs, list) and len(image_inputs) == len(assets)
             and all(item.get("asset_id") == asset_id and item.get("original_sha256") == asset_id
                     and bool(item.get("sent_sha256")) and item.get("crop") is False
                     and all(isinstance(item.get(field), (list, tuple)) and len(item[field]) == 2
                             and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in item[field])
                             for field in ("original_dimensions", "sent_dimensions"))
                     for item, asset_id in zip(image_inputs, assets)),
             "model receipt missing exact sent-image provenance")
    _require(isinstance(receipt.get("parsed"), dict), "model receipt parsed output required")


def _annotation_matches(annotation, content):
    _require(annotation.get("native_task") == content["native_task"]
             and annotation.get("semantic_task") == "caption", "model annotation task mismatch")
    captions = annotation.get("image_captions")
    _require(isinstance(captions, list) and len(captions) == 1 and captions[0].get("image_index") == 0,
             "model annotation image binding mismatch")
    _require(all(captions[0].get(key) == value for key, value in content["captions"].items()),
             "candidate caption differs from model-reviewed text")


def _verify_model_evidence(content, generator, quality):
    """Check actual request/output bindings, not merely hashes added by publisher."""
    base = content["task_context"]
    _require(base.get("native_task") == content["native_task"] and base.get("prepared_semantic_task") == "caption"
             and base.get("image_count") == 1, "candidate task context mismatch")
    evidence = quality["evidence"]
    _model_receipt(evidence, content["asset_ids"], "verifier")
    from .validation import quality_validator
    quality_validator(evidence["parsed"])
    _require(evidence["parsed"]["verdict"] == quality["verdict"], "quality receipt verdict mismatch")
    if generator is not None:
        _model_receipt(generator, content["asset_ids"], "generator")
        annotation = generator["parsed"]
        _annotation_matches(annotation, content)
        from .validation import annotation_validator
        annotation_validator({"normalized": {"category": content["native_task"]},
                              "source_image_refs": [{"sha256": sha} for sha in content["asset_ids"]],
                              "raw": {"format": base["original_format"], "segments": base["segments"]},
                              "prepared_semantic_task": "caption"})(annotation)
        context = generator.get("request_context", base)
        _require(all(context.get(key) == value for key, value in base.items()), "generator changed original task context")
        _require(digest(context) == generator["identity"].get("context_sha256"), "generator context hash mismatch")
        original_context = {**base, **{key: context[key] for key in ("training_phase", "annotation_quality_policy") if key in context}}
        if quality["verdict"] == "pass" and annotation.get("concerns"):
            _require(original_context.get("training_phase") == "early_mixed",
                     "generation concerns can only be retained in explicit early phase")
    else:
        _require(isinstance(evidence.get("request_context"), dict), "existing annotation review requires explicit request context")
        annotation = evidence["request_context"].get("candidate_annotation", {})
        _annotation_matches(annotation, content)
        original_context = evidence["request_context"].get("original_record", {})
        _require(all(original_context.get(key) == value for key, value in base.items()), "existing review changed original task context")
    changes = {item["segment_index"]: item["text"] for item in annotation.get("assistant_revisions", [])}
    targets = [{"segment_index": index, "text": changes.get(index, segment.get("text")),
                "value_origin": "proposed_revision" if index in changes and changes[index] != segment.get("text") else "retained_original_source_target"}
               for index, segment in enumerate(base["segments"]) if segment["kind"] == "text" and segment.get("role") == "assistant"]
    expected = {"original_record": original_context, "candidate_annotation": annotation,
                "annotation_is_delta": True, "effective_assistant_targets": targets}
    context = evidence.get("request_context", expected)
    _require(all(context.get(key) == value for key, value in expected.items()), "verifier did not review exact candidate/context")
    _require(digest(context) == evidence["identity"].get("context_sha256"), "verifier context hash mismatch")
    # A generic global pass can concern only the original or shortest target.
    # Staged captions require independently recorded checks for every variant.
    if context["original_record"].get("training_phase") in PHASES:
        from .validation import caption_quality_validator
        caption_quality_validator(len(content["asset_ids"]))(evidence["parsed"])


def _quality_metadata(generator, quality):
    parsed = quality["evidence"]["parsed"]
    concerns = generator["parsed"].get("concerns", []) if generator else []
    flags = []
    if quality["verdict"] == "uncertain":
        flags.append("CAPTION_QUALITY_UNCERTAIN")
    if any(check["verdict"] == "uncertain" for check in parsed.get("caption_checks", [])):
        flags.append("CAPTION_VARIANT_UNCERTAIN")
    if concerns:
        flags.append("GENERATOR_CONCERNS_RETAINED")
    failure = quality["verdict"] in {"fail", "revise"} or any(
        check["verdict"] in {"fail", "revise"} for check in parsed.get("caption_checks", []))
    return {"quality_status": quality["verdict"], "quality_flags": flags,
            "retained_with_uncertainty": bool(flags) and not failure}


def build_candidate(record, *, origin, captions, provenance, quality_receipt, generator_receipt=None):
    """Bind supplied factual receipts to one exact record/image/caption payload.

    This function hashes evidence; it does not manufacture a visual quality pass.
    Existing candidates can contain a subset of lengths. New generation requires
    a complete triplet; pass/uncertain/fail receipts remain separate from schema.
    """
    _require(origin in {"existing", "generated"}, "invalid candidate origin")
    _require(isinstance(provenance, dict) and provenance, "candidate provenance required")
    _require(isinstance(captions, dict) and captions and set(captions) <= set(LENGTHS), "invalid caption keys")
    _require(all(isinstance(text, str) and text.strip() and len(text) <= 24000 for text in captions.values()),
             "invalid caption text")
    if origin == "generated":
        _require(set(captions) == set(LENGTHS), "generated caption requires complete triplet")
        _require(isinstance(generator_receipt, dict) and generator_receipt.get("status") == "ok"
                 and bool(generator_receipt.get("identity")) and bool(generator_receipt.get("cache_key")),
                 "generated candidate requires actual generator receipt")
    else:
        _require(generator_receipt is None, "existing annotation must not claim fresh generation")
    _require(isinstance(quality_receipt, dict) and quality_receipt.get("verdict") in {"pass", "uncertain", "fail", "revise"}
             and quality_receipt.get("scope") == "per_annotation_visual_review" and bool(quality_receipt.get("evidence")),
             "candidate requires explicit per-annotation quality evidence")
    _require(len(_assets(record)) == 1, "caption candidate requires one exact image view")
    content = {"record_id": _record_key(record), "collection": record.get("collection", "dcvlm"),
               "source_record_sha256": record["source_record_sha256"], "asset_ids": _assets(record),
               "native_task": record["normalized"]["category"], "origin": origin,
               "captions": deepcopy(captions), "provenance": deepcopy(provenance), "task_context": _model_context(record)}
    _verify_model_evidence(content, generator_receipt, quality_receipt)
    content_sha = digest(content)
    candidate = {**content, "content_sha256": content_sha,
                 "quality_receipt": _bound_receipt(quality_receipt, content_sha),
                 "generator_receipt": _bound_receipt(generator_receipt, content_sha) if generator_receipt else None,
                 **_quality_metadata(generator_receipt, quality_receipt)}
    candidate["candidate_id"] = digest(candidate)
    return candidate


def _validate_candidate(candidate):
    value = deepcopy(candidate)
    candidate_id = value.pop("candidate_id")
    _require(digest(value) == candidate_id, "candidate checksum mismatch")
    content = {key: value[key] for key in ("record_id", "collection", "source_record_sha256", "asset_ids",
                                          "native_task", "origin", "captions", "provenance", "task_context")}
    _require(digest(content) == value["content_sha256"], "candidate content checksum mismatch")
    for name in ("quality_receipt", "generator_receipt"):
        receipt = value[name]
        if receipt is None:
            _require(name == "generator_receipt" and value["origin"] == "existing", "missing model receipt")
            continue
        payload = {key: receipt[key] for key in ("candidate_content_sha256", "receipt")}
        _require(receipt["receipt_sha256"] == digest(payload)
                 and receipt["candidate_content_sha256"] == value["content_sha256"], "receipt checksum/binding mismatch")
    _require(value["origin"] in {"existing", "generated"} and len(value["asset_ids"]) == 1,
             "invalid candidate kind/image count")
    _require(isinstance(value["captions"], dict) and value["captions"] and set(value["captions"]) <= set(LENGTHS)
             and all(isinstance(text, str) and text.strip() and len(text) <= 24000 for text in value["captions"].values()),
             "invalid caption text/keys")
    _require(value["origin"] != "generated" or set(value["captions"]) == set(LENGTHS), "generated caption requires complete triplet")
    quality = value["quality_receipt"]["receipt"]
    _require(quality.get("scope") == "per_annotation_visual_review", "invalid candidate quality scope")
    generator = value["generator_receipt"]["receipt"] if value["generator_receipt"] else None
    _verify_model_evidence(content, generator, quality)
    expected_metadata = _quality_metadata(generator, quality)
    if any(key in value for key in expected_metadata):
        _require(all(value.get(key) == expected for key, expected in expected_metadata.items()), "candidate quality retention metadata mismatch")


def _target(record, semantic_task):
    if semantic_task != "caption":
        return None
    _require(len(record["source_image_refs"]) == 1, "caption replacement requires exactly one image")
    contract = record.get("caption_contract")
    if contract:
        target = contract["target_segment_index"]
        allowed = contract["allowed_training_lengths"]
        _require(isinstance(target, int) and not isinstance(target, bool)
                 and 0 <= target < len(record["raw"]["segments"]), "invalid caption target")
        segment = record["raw"]["segments"][target]
        _require(segment["kind"] == "text" and (segment.get("role") == "assistant"
                 or (record["raw"]["format"] == "caption" and segment.get("role") is None)),
                 "caption contract would alter user/context")
    else:
        _require(record["raw"]["format"] == "caption", "conversation caption requires explicit contract")
        targets = [i for i, segment in enumerate(record["raw"]["segments"]) if segment["kind"] == "text"]
        _require(len(targets) == 1, "ambiguous caption target")
        target, allowed = targets[0], list(LENGTHS)
    _require(isinstance(allowed, list) and allowed and len(allowed) == len(set(allowed))
             and set(allowed) <= set(LENGTHS), "invalid allowed caption lengths")
    return {"target_segment_index": target, "allowed_training_lengths": allowed}


def _revocations(values):
    """Each tombstone is explicit and evidence-bearing, with no mutation of a release."""
    result = []
    for row in values:
        _require(isinstance(row, dict) and row.get("kind") in {"record", "candidate", "asset"}
                 and isinstance(row.get("id"), str) and row["id"] and bool(row.get("reason")), "invalid revocation")
        result.append(deepcopy(row))
    return sorted(result, key=lambda row: (row["kind"], row["id"], digest(row)))


def _revoked(view, candidate_id, revocations):
    keys = {(row["kind"], row["id"]) for row in revocations}
    return (("record", view["original_sample_id"]) in keys
            or (candidate_id is not None and ("candidate", candidate_id) in keys)
            or any(("asset", asset_id) in keys for asset_id in view["asset_ids"]))


def _weight_config(value):
    value = value or {"original": 1, "existing": 1, "generated": 1,
                      "lengths": {"short": .25, "medium": .5, "long": .25}}
    _require(set(value) == {"original", "existing", "generated", "lengths"}
             and set(value["lengths"]) == set(LENGTHS), "invalid selection weights")
    for number in [value[name] for name in ("original", "existing", "generated")] + list(value["lengths"].values()):
        _require(isinstance(number, (float, int)) and not isinstance(number, bool)
                 and math.isfinite(number) and number > 0, "selection weights must be finite positive")
    return deepcopy(value)


def _draw(items, weights, identity):
    total = sum(weights)
    position = (int(digest(identity)[:16], 16) / 16 ** 16) * total
    cumulative = 0
    for item, weight in zip(items, weights):
        cumulative += weight
        if position < cumulative:
            return item
    return items[-1]


def _qualified(candidate, phase):
    """Phase eligibility; early uncertain retention is not highres qualification."""
    quality = candidate["quality_receipt"]["receipt"]
    checks = quality["evidence"]["parsed"].get("caption_checks", [])
    if quality["verdict"] not in {"pass", "uncertain"} or any(check["verdict"] in {"fail", "revise"} for check in checks):
        return False
    if phase != "highres_1mp":
        return True
    generation = candidate["generator_receipt"]
    metadata = _quality_metadata(generation["receipt"] if generation else None, quality)
    if quality["verdict"] != "pass" or metadata["retained_with_uncertainty"]:
        return False
    provenance = candidate["provenance"]
    quality_context = candidate["quality_receipt"]["receipt"]["evidence"].get("request_context", {}).get("original_record", {})
    generation_context = generation["receipt"].get("request_context", {}) if generation else quality_context
    return (set(candidate["captions"]) == set(LENGTHS)
            and (provenance.get("phase") == phase or phase in provenance.get("applicable_phases", []))
            and quality_context.get("training_phase") == phase
            and generation_context.get("training_phase") == phase
            and bool(quality_context.get("annotation_quality_policy")))


def staged_release_contents(path, *, require_complete=True):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    _require(manifest.get("schema_version") == SCHEMA, "not a staged release")
    if require_complete:
        complete = json.loads((path / "COMPLETE.json").read_text())
        _require(complete.get("release_id") == manifest["release_id"]
                 and complete.get("manifest_sha256") == file_sha(path / "manifest.json")
                 and complete.get("reader_validation_sha256") == file_sha(path / "reader_validation.json"),
                 "release completion checksum mismatch")
    for name, expected in manifest["artifacts_sha256"].items():
        _require(file_sha(path / name) == expected, "release artifact checksum mismatch: " + name)
    assets = {row["asset_id"]: row for row in jsonl(path / "assets.jsonl")}
    candidates = {}
    for candidate in jsonl(path / "candidates.jsonl"):
        _validate_candidate(candidate)
        _require(candidate["candidate_id"] not in candidates, "duplicate candidate")
        candidates[candidate["candidate_id"]] = candidate
    views = list(jsonl(path / "views.jsonl"))
    revocations = _revocations(list(jsonl(path / "revocations.jsonl")))
    identity = {"run_identity": manifest["run_identity"], "phase": manifest["phase"],
                "selection_weights": manifest["selection_weights"], "ledger": list(jsonl(path / "decisions.jsonl")),
                "views": views, "candidates": [candidates[key] for key in sorted(candidates)],
                "assets": [assets[key] for key in sorted(assets)], "revocations": revocations}
    _require(digest(identity) == manifest["content_sha256"], "release content identity mismatch")
    from .quality_gate import binding_from_identity, load_bound_quality_policy, validate_quality_admission
    quality_binding = binding_from_identity(manifest["run_identity"])
    quality_state = load_bound_quality_policy(quality_binding) if quality_binding else None
    for row in identity["ledger"]:
        validate_quality_admission(row["asset_ids"], row["admission"], manifest["run_identity"], state=quality_state)
    for view in views:
        validate_quality_admission(view["asset_ids"], view["admission"], manifest["run_identity"], state=quality_state)
    return manifest, assets, candidates, views, revocations


def materialize_staged(view, candidates, assets, *, phase, selection_weights, epoch_seed=0,
                       caption_length=None, revocations=(), load_images=True):
    if _revoked(view, None, revocations):
        return None
    raw = load_original_record(view["locator"])
    _require(raw["sample_id"] == view["original_sample_id"] and digest(raw) == view["source_record_sha256"],
             "published original record changed")
    _require(len(view["asset_ids"]) == raw["image_count"], "published record lost required image")
    loaded = []
    for asset_id in view["asset_ids"]:
        asset = assets[asset_id]
        _require(asset["short_edge"] >= 128 and asset["original_ref"]["sha256"] == asset_id,
                 "published asset violates size/identity gate")
        if load_images:
            loaded.append(read_asset_bytes(asset["original_ref"]))
    available = [candidate_id for candidate_id in view["eligible_candidate_ids"]
                 if not _revoked(view, candidate_id, revocations)]
    allow_original = view["original_eligible"]
    if not available and not allow_original:
        return None
    groups = {}
    if allow_original:
        groups["original"] = [None]
    for candidate_id in available:
        candidate = candidates[candidate_id]
        _require(candidate["record_id"] == view["original_sample_id"]
                 and candidate["source_record_sha256"] == view["source_record_sha256"]
                 and candidate["asset_ids"] == view["asset_ids"]
                 and candidate["native_task"] == view["native_task"]
                 and digest(candidate["task_context"]) == view["task_context_sha256"], "candidate context mismatch")
        _require(_qualified(candidate, phase), "unqualified training candidate")
        groups.setdefault(candidate["origin"], []).append(candidate_id)
    identity = [view["view_id"], epoch_seed]
    origins = sorted(groups)
    origin = _draw(origins, [selection_weights[name] for name in origins], identity + ["origin"])
    choices = sorted(groups[origin]) if origin != "original" else [None]
    selected = _draw(choices, [1] * len(choices), identity + ["candidate"])
    segments = deepcopy(raw["segments"])
    selected_length = None
    candidate_quality = None
    if selected is not None:
        _require(view["semantic_task"] == "caption" and view["caption_target"] is not None,
                 "candidate cannot replace native noncaption task")
        candidate = candidates[selected]
        candidate_quality = _quality_metadata(candidate["generator_receipt"]["receipt"] if candidate["generator_receipt"] else None,
                                              candidate["quality_receipt"]["receipt"])
        allowed = [name for name in view["caption_target"]["allowed_training_lengths"] if name in candidate["captions"]]
        _require(allowed, "candidate has no allowed training length")
        if caption_length is not None:
            _require(caption_length in allowed, "caption length violates native instruction")
            selected_length = caption_length
        else:
            selected_length = _draw(allowed, [selection_weights["lengths"][name] for name in allowed], identity + ["length"])
        segments[view["caption_target"]["target_segment_index"]]["text"] = candidate["captions"][selected_length]
    return {"sample_id": raw["sample_id"], "native_task": view["native_task"], "segments": segments,
            "images": loaded, "source_image_refs": [assets[key]["original_ref"] for key in view["asset_ids"]],
            "selected_candidate_id": selected, "selected_origin": origin, "caption_length": selected_length,
            "text_provenance": (candidates[selected]["provenance"] if selected else view["original_provenance"]),
            "quality_status": (candidate_quality["quality_status"] if selected else
                               "uncertain" if view["original_provenance"].get("retained_with_uncertainty") else "retained_original"),
            "retained_with_uncertainty": (candidate_quality["retained_with_uncertainty"] if selected else
                                          bool(view["original_provenance"].get("retained_with_uncertainty"))),
            "quality_flags": (candidate_quality["quality_flags"] if selected else view["original_provenance"].get("quality_flags", [])),
            "admission": view["admission"], "asset_exposure_multiplier": 1}


def iter_staged_records(path, *, epoch_seed=0, caption_length=None, revocations=(), load_images=True):
    manifest, assets, candidates, views, frozen_revocations = staged_release_contents(path)
    combined = frozen_revocations + _revocations(revocations)
    for view in views:
        value = materialize_staged(view, candidates, assets, phase=manifest["phase"],
                                   selection_weights=manifest["selection_weights"], epoch_seed=epoch_seed,
                                   caption_length=caption_length, revocations=combined, load_images=load_images)
        if value is not None:
            yield value


def validate_staged_release(path, *, require_complete=True):
    manifest, assets, candidates, views, revocations = staged_release_contents(path, require_complete=require_complete)
    count = image_count = 0
    for view in views:
        _require(view["admission"]["decision"] == "admit", "unadmitted published view")
        _require(view["collection"] != "dcvlm" or view["native_task"] not in {"single_t2i", "multi_t2i"},
                 "DCVLM direction changed")
        # Validate every eligible candidate, not only the deterministic draw.
        variants = [{**view, "original_eligible": False, "eligible_candidate_ids": [key]}
                    for key in view["eligible_candidate_ids"]]
        if view["original_eligible"]:
            variants.append({**view, "eligible_candidate_ids": []})
        for variant in variants:
            materialize_staged(variant, candidates, assets, phase=manifest["phase"],
                               selection_weights=manifest["selection_weights"], revocations=revocations,
                               load_images=False)
        result = materialize_staged(view, candidates, assets, phase=manifest["phase"],
                                    selection_weights=manifest["selection_weights"], revocations=revocations)
        if result:
            count += 1
            image_count += len(result["images"])
    return {"success": True, "records_materialized": count, "all_image_slots_read": image_count,
            "candidate_receipts_verified": len(candidates), "source_sha256_verified": True,
            "native_task_preserved": True, "one_training_view_per_record": True,
            "phase": manifest["phase"], "coverage": manifest["coverage"]}


def commit_staged_release(directory, records, assets, plans, *, run_identity, phase, admissions,
                          candidates=(), revocations=(), selection_weights=None, release_id=None):
    """Freeze a batch, including pending denominator rows, after explicit admission.

    A caller-supplied release_id is an immutable label: reusing it with different
    content raises. Omit it to use a content-derived identifier. Late phase output
    never updates or mutates an early phase release.
    """
    _require(phase in PHASES and isinstance(run_identity, dict) and run_identity, "phase/run identity required")
    from .quality_gate import binding_from_identity, load_bound_quality_policy, validate_quality_admission
    quality_binding = binding_from_identity(run_identity)
    quality_state = load_bound_quality_policy(quality_binding) if quality_binding else None
    weights, revoked = _weight_config(selection_weights), _revocations(revocations)
    records = list(records)
    _require(len({_record_key(record) for record in records}) == len(records), "duplicate original record id")
    by_record = {_record_key(record): record for record in records}
    candidate_rows = {}
    for candidate in candidates:
        _validate_candidate(candidate)
        _require(candidate["record_id"] in by_record, "candidate is outside release batch")
        record = by_record[candidate["record_id"]]
        _require(candidate["source_record_sha256"] == record["source_record_sha256"]
                 and candidate["asset_ids"] == _assets(record)
                 and candidate["native_task"] == record["normalized"]["category"]
                 and candidate["task_context"] == _model_context(record), "candidate context mismatch")
        candidate_rows[candidate["candidate_id"]] = candidate
    views, ledger, selected_assets, pool_assets, covered_assets = [], [], set(), set(), set()
    pool_records = covered_records = 0
    for record in sorted(records, key=_record_key):
        key = _record_key(record)
        _require(key in plans and key in admissions, "every input needs plan and explicit admission")
        plan, admission = deepcopy(plans[key]), deepcopy(admissions[key])
        _require(admission.get("decision") in {"admit", "pending", "reject"}
                 and isinstance(admission.get("evidence"), dict) and admission["evidence"], "explicit admission evidence required")
        validate_quality_admission(_assets(record), admission, run_identity, state=quality_state)
        if admission["decision"] == "admit" and admission["evidence"].get("image_screening_policy") == "image_screening.v3":
            screening = admission["evidence"].get("image_screening", [])
            _require(Counter(row.get("asset_id") for row in screening) == Counter(_assets(record))
                     and all(row.get("screening", {}).get("decision") == "admit" for row in screening),
                     "every required image slot needs an admitted screening receipt")
        _require(plan.get("phase", plan.get("training_phase", phase)) == phase, "plan phase mismatch")
        _require(plan.get("native_task", record["normalized"]["category"]) == record["normalized"]["category"], "plan changed native task")
        semantic = plan.get("semantic_task")
        _require(isinstance(semantic, str) and semantic, "plan semantic task required")
        action = plan["action"]
        in_pool = phase == "highres_1mp" and semantic == "caption" and plan.get("in_stage_pool", action != "not_in_stage_pool")
        if in_pool:
            pool_records += 1
            pool_assets.update(_assets(record))
        candidate_ids = []
        target = _target(record, semantic) if semantic == "caption" and action not in PENDING_ACTIONS else None
        for candidate in candidate_rows.values():
            if candidate["record_id"] != key or not _qualified(candidate, phase if in_pool else "early_mixed"):
                continue
            _require(semantic == "caption", "native noncaption supervision cannot receive caption replacement")
            if target and set(candidate["captions"]) & set(target["allowed_training_lengths"]):
                if not in_pool or set(candidate["captions"]) == set(LENGTHS):
                    candidate_ids.append(candidate["candidate_id"])
        original_allowed = (action in RETAIN_ACTIONS or (action in {"reuse_annotation", "complete_highres_caption", "generate_missing"}
                            and plan.get("original_usable") is True)) and not in_pool
        if original_allowed and admission["decision"] == "admit":
            _require(bool(admission["evidence"].get("retention_policy")), "retaining original requires explicit retention policy")
        view = {"view_id": digest([run_identity, phase, record["source_record_sha256"], key]),
                "collection": record.get("collection", "dcvlm"), "source": record["source"],
                "original_sample_id": key, "source_record_sha256": record["source_record_sha256"],
                "locator": deepcopy(record["locator"]), "native_task": record["normalized"]["category"],
                "semantic_task": semantic, "asset_ids": _assets(record), "caption_target": target,
                "task_context_sha256": digest(_model_context(record)),
                "eligible_candidate_ids": sorted(candidate_ids), "original_eligible": original_allowed,
                "original_provenance": {**deepcopy(plan.get("original_text_provenance", {})),
                                        "annotation_kind": "retained_original", "text_origin": plan.get("caption_origin", "unknown"),
                                        "retained_with_uncertainty": bool(plan.get("retained_with_uncertainty") or admission["evidence"].get("retained_with_uncertainty")),
                                        "quality_flags": sorted(set(plan.get("quality_flags", [])) | set(admission["evidence"].get("quality_flags", []))),
                                        "uncertain_candidate_refs": deepcopy(plan.get("uncertain_candidate_refs", [])),
                                        "source_record_sha256": record["source_record_sha256"]},
                "admission": admission, "asset_exposure_multiplier": 1}
        view["eligible_candidate_ids"] = [key for key in view["eligible_candidate_ids"] if not _revoked(view, key, revoked)]
        eligible = (admission["decision"] == "admit" and action not in PENDING_ACTIONS
                    and not _revoked(view, None, revoked)
                    and (original_allowed or bool(view["eligible_candidate_ids"])))
        if in_pool and eligible and view["eligible_candidate_ids"]:
            covered_records += 1
            covered_assets.update(_assets(record))
        ledger.append({"record_id": key, "source_record_sha256": record["source_record_sha256"],
                       "locator": record["locator"], "collection": record.get("collection", "dcvlm"),
                       "source": record["source"], "native_task": record["normalized"]["category"],
                       "asset_ids": _assets(record), "plan": plan, "admission": admission,
                       "in_highres_pool": bool(in_pool), "published": bool(eligible),
                       "pending_reason": None if eligible else "admission_or_plan_or_candidate_requirements_not_met"})
        if eligible:
            for asset_id in _assets(record):
                _require(asset_id in assets and assets[asset_id]["short_edge"] >= 128,
                         "missing or undersized required image")
            views.append(view)
            selected_assets.update(_assets(record))
    coverage = {"highres_pool_records": pool_records, "highres_covered_records": covered_records,
                "highres_pending_records": pool_records - covered_records,
                "highres_unique_assets": len(pool_assets), "highres_covered_unique_assets": len(covered_assets),
                "record_coverage": covered_records / pool_records if pool_records else None,
                "asset_coverage": len(covered_assets) / len(pool_assets) if pool_assets else None,
                "denominator_includes_pending_and_failed": True}
    identity = {"run_identity": run_identity, "phase": phase, "selection_weights": weights,
                "ledger": ledger, "views": views, "candidates": [candidate_rows[key] for key in sorted(candidate_rows)],
                "assets": [assets[key] for key in sorted(selected_assets)], "revocations": revoked}
    content_hash = digest(identity)
    label = release_id or content_hash
    _require(isinstance(label, str) and label and label not in {".", ".."} and Path(label).name == label
             and "/" not in label and "\\" not in label, "invalid release id")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".staged_publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = directory / label
        if target.exists():
            _require(not target.is_symlink(), "release target cannot be symlink")
            previous = json.loads((target / "manifest.json").read_text())
            _require(previous.get("content_sha256") == content_hash, "immutable release cannot be overwritten")
            validation = validate_staged_release(target)
            return {"committed": True, "reused_existing_release": True, "release_id": label,
                    "path": str(target.resolve()), "records": len(views), "coverage": coverage, "reader_validation": validation}
        staging = directory / (".staging_" + uuid.uuid4().hex)
        staging.mkdir()
        try:
            payloads = {"assets.jsonl": identity["assets"], "candidates.jsonl": identity["candidates"],
                        "views.jsonl": views, "decisions.jsonl": ledger, "revocations.jsonl": revoked}
            for filename, rows in payloads.items():
                write_jsonl(staging / filename, rows)
            manifest = {"schema_version": SCHEMA, "release_id": label, "content_sha256": content_hash,
                        "run_identity": run_identity, "phase": phase, "selection_weights": weights,
                        "records": len(views), "input_records": len(records), "pending_or_rejected": len(records) - len(views),
                        "unique_assets": len(selected_assets), "coverage": coverage,
                        "no_source_image_copies": True, "scope": "frozen_supplied_batch",
                        "artifacts_sha256": {name: file_sha(staging / name) for name in payloads}}
            atomic_json(staging / "manifest.json", manifest)
            validation = validate_staged_release(staging, require_complete=False)
            atomic_json(staging / "reader_validation.json", validation)
            atomic_json(staging / "COMPLETE.json", {"published_training": True, "release_id": label,
                        "manifest_sha256": file_sha(staging / "manifest.json"),
                        "reader_validation_sha256": file_sha(staging / "reader_validation.json")})
            staging.replace(target)
            atomic_json(directory / ("current_" + phase + ".json"), {"release_id": label, "path": str(target.resolve())})
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {"committed": True, "reused_existing_release": False, "release_id": label,
                "path": str(target.resolve()), "records": len(views), "coverage": coverage, "reader_validation": validation}


def _dataset_manifest_path(path):
    path = Path(path).resolve()
    return path if path.suffix == ".json" else path / "manifest.json"


def _readonly_index(path):
    return sqlite3.connect("file:" + quote(str(Path(path).resolve())) + "?mode=ro&immutable=1", uri=True)


def _dataset_manifest(path):
    manifest_path = _dataset_manifest_path(path)
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("schema_version") == DATASET_SCHEMA, "not a staged dataset")
    identity = {key: value for key, value in manifest.items() if key != "content_sha256"}
    _require(digest(identity) == manifest.get("content_sha256"), "dataset manifest checksum mismatch")
    _require(file_sha(manifest["index_path"]) == manifest["index_sha256"], "dataset index checksum mismatch")
    if manifest.get("base_release"):
        base = manifest["base_release"]
        _require(file_sha(base["path"]) == base["manifest_sha256"], "dataset base manifest changed")
    for part in manifest["parts"]:
        _require(file_sha(Path(part["path"]) / "manifest.json") == part["manifest_sha256"],
                 "dataset part manifest changed")
    return manifest


def _index_coverage(connection):
    total, published = connection.execute("SELECT COUNT(*), COALESCE(SUM(published),0) FROM records").fetchone()
    pool, covered = connection.execute("SELECT COALESCE(SUM(in_pool),0), COALESCE(SUM(in_pool*published),0) FROM records").fetchone()
    unique_assets = connection.execute("SELECT COUNT(DISTINCT asset_id) FROM record_assets JOIN records USING(collection,record_id) WHERE in_pool=1").fetchone()[0]
    covered_assets = connection.execute("SELECT COUNT(DISTINCT asset_id) FROM record_assets JOIN records USING(collection,record_id) WHERE in_pool=1 AND published=1").fetchone()[0]
    return {"input_records": total, "published_records": published, "pending_or_rejected_records": total - published,
            "highres_pool_records": pool, "highres_covered_records": covered, "highres_pending_records": pool - covered,
            "highres_unique_assets": unique_assets, "highres_covered_unique_assets": covered_assets,
            "record_coverage": covered / pool if pool else None,
            "asset_coverage": covered_assets / unique_assets if unique_assets else None,
            "denominator_includes_pending_and_failed": True}


def build_dataset_release(path, parts, *, phase, run_identity, base_release=None):
    """Freeze a streamed aggregate using a disk-backed logical-record index.

    A new part's pending/rejected record shadows an older admitted record, so an
    unresolved correction cannot silently fall back to stale supervision. The
    previous aggregate and all part releases remain unchanged. Full image reads
    are not repeated here; each part's signed-by-hash validation artifacts and
    receipt bindings are checked. This is integrity, not an external signature.
    """
    _require(phase in PHASES and isinstance(run_identity, dict) and run_identity, "phase/run identity required")
    from .quality_gate import binding_from_identity, load_bound_quality_policy, validate_quality_admission
    quality_binding = binding_from_identity(run_identity)
    quality_state = load_bound_quality_policy(quality_binding) if quality_binding else None
    manifest_path = _dataset_manifest_path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base = None
    ordered_paths = []
    if base_release:
        base_path = _dataset_manifest_path(base_release)
        previous = _dataset_manifest(base_path)
        _require(previous["phase"] == phase, "dataset base must use the same phase")
        base = {"path": str(base_path), "manifest_sha256": file_sha(base_path)}
        ordered_paths.extend(part["path"] for part in previous["parts"])
    ordered_paths.extend(str(Path(part).resolve()) for part in parts)
    _require(ordered_paths, "dataset requires at least one part")
    part_rows = []
    for part_path in ordered_paths:
        manifest, _, _, views, _ = staged_release_contents(part_path)
        _require(manifest["phase"] == phase, "dataset part phase mismatch")
        if quality_state is not None:
            # Old admitted parts cannot silently enter a new required-blur
            # release through --base-release; they need the new bound receipts.
            for view in views:
                validate_quality_admission(view["asset_ids"], view["admission"], run_identity, state=quality_state)
        part_rows.append({"path": part_path, "manifest_sha256": file_sha(Path(part_path) / "manifest.json"),
                          "release_id": manifest["release_id"], "content_sha256": manifest["content_sha256"]})
    request_identity = {"phase": phase, "run_identity": run_identity, "base_release": base, "parts": part_rows}
    request_hash = digest(request_identity)
    with (manifest_path.parent / ".dataset_publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists():
            current = _dataset_manifest(manifest_path)
            _require(current.get("request_sha256") == request_hash, "immutable dataset manifest cannot be overwritten")
            return {"committed": True, "reused_existing_release": True, "path": str(manifest_path),
                    "records": current["coverage"]["published_records"], "coverage": current["coverage"]}
        index_path = manifest_path.parent / (manifest_path.stem + "." + request_hash + ".sqlite")
        temporary = manifest_path.parent / (".dataset_" + uuid.uuid4().hex + ".sqlite")
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE records(collection TEXT NOT NULL,record_id TEXT NOT NULL,part_index INTEGER NOT NULL,
                  published INTEGER NOT NULL,in_pool INTEGER NOT NULL,candidate_ids TEXT NOT NULL,
                  original_eligible INTEGER NOT NULL,PRIMARY KEY(collection,record_id));
                CREATE INDEX records_part ON records(part_index,published);
                CREATE TABLE record_assets(collection TEXT NOT NULL,record_id TEXT NOT NULL,asset_id TEXT NOT NULL,
                  PRIMARY KEY(collection,record_id,asset_id));
                CREATE INDEX assets_id ON record_assets(asset_id);
                CREATE TABLE revocations(kind TEXT NOT NULL,id TEXT NOT NULL,reason TEXT NOT NULL,PRIMARY KEY(kind,id));
            """)
            for part_index, part in enumerate(part_rows):
                directory = Path(part["path"])
                # Per-part maps are bounded by the runner's batch size.
                views = {(view["collection"], view["original_sample_id"]): view for view in jsonl(directory / "views.jsonl")}
                with connection:
                    for decision in jsonl(directory / "decisions.jsonl"):
                        collection, record_id = decision["collection"], decision["record_id"]
                        view = views.get((collection, record_id))
                        _require(bool(view) == bool(decision["published"]), "part decision/view inconsistency")
                        connection.execute("INSERT OR REPLACE INTO records VALUES(?,?,?,?,?,?,?)",
                                           (collection, record_id, part_index, int(decision["published"]),
                                            int(decision["in_highres_pool"]), json.dumps(view["eligible_candidate_ids"] if view else []),
                                            int(view["original_eligible"]) if view else 0))
                        connection.execute("DELETE FROM record_assets WHERE collection=? AND record_id=?", (collection, record_id))
                        connection.executemany("INSERT OR IGNORE INTO record_assets VALUES(?,?,?)",
                                               [(collection, record_id, asset_id) for asset_id in decision["asset_ids"]])
                    for row in jsonl(directory / "revocations.jsonl"):
                        connection.execute("INSERT OR REPLACE INTO revocations VALUES(?,?,?)", (row["kind"], row["id"], row["reason"]))
            with connection:
                connection.execute("UPDATE records SET published=0 WHERE record_id IN(SELECT id FROM revocations WHERE kind='record')")
                connection.execute("""UPDATE records SET published=0 WHERE EXISTS(
                  SELECT 1 FROM record_assets JOIN revocations ON revocations.id=record_assets.asset_id AND revocations.kind='asset'
                  WHERE record_assets.collection=records.collection AND record_assets.record_id=records.record_id)""")
                scan = connection.execute("SELECT collection,record_id,candidate_ids FROM records WHERE published=1 AND original_eligible=0")
                for collection, record_id, candidate_ids in scan:
                    candidates = json.loads(candidate_ids)
                    if all(connection.execute("SELECT 1 FROM revocations WHERE kind='candidate' AND id=?", (key,)).fetchone() for key in candidates):
                        connection.execute("UPDATE records SET published=0 WHERE collection=? AND record_id=?", (collection, record_id))
            coverage = _index_coverage(connection)
            connection.close()
            if index_path.exists():
                _require(not index_path.is_symlink() and file_sha(index_path) == file_sha(temporary), "dataset index identity conflict")
                temporary.unlink()
            else:
                temporary.replace(index_path)
            manifest = {"schema_version": DATASET_SCHEMA, **request_identity, "request_sha256": request_hash,
                        "index_path": str(index_path.resolve()), "index_sha256": file_sha(index_path), "coverage": coverage,
                        "logical_record_operation": "latest_part_by_collection_record_id_including_pending",
                        "source_images_copied": False, "one_view_per_logical_record": True}
            manifest["content_sha256"] = digest(manifest)
            atomic_json(manifest_path, manifest)
            return {"committed": True, "reused_existing_release": False, "path": str(manifest_path),
                    "records": coverage["published_records"], "coverage": coverage}
        finally:
            connection.close()
            temporary.unlink(missing_ok=True)


def iter_dataset_records(path, *, epoch_seed=0, caption_length=None, revocations=(), load_images=True):
    manifest = _dataset_manifest(path)
    from .quality_gate import binding_from_identity, load_bound_quality_policy, validate_quality_admission
    quality_binding = binding_from_identity(manifest["run_identity"])
    quality_state = load_bound_quality_policy(quality_binding) if quality_binding else None
    external = _revocations(revocations)
    connection = _readonly_index(manifest["index_path"])
    try:
        for part_index, part in enumerate(manifest["parts"]):
            if not connection.execute("SELECT 1 FROM records WHERE part_index=? AND published=1 LIMIT 1", (part_index,)).fetchone():
                continue
            part_manifest, assets, candidates, views, _ = staged_release_contents(part["path"])
            for view in views:
                winner = connection.execute("SELECT part_index,published FROM records WHERE collection=? AND record_id=?",
                                            (view["collection"], view["original_sample_id"])).fetchone()
                if winner != (part_index, 1):
                    continue
                if quality_state is not None:
                    validate_quality_admission(view["asset_ids"], view["admission"], manifest["run_identity"], state=quality_state)
                applicable = list(external)
                keys = [("record", view["original_sample_id"])] + [("asset", key) for key in view["asset_ids"]]
                keys += [("candidate", key) for key in view["eligible_candidate_ids"]]
                for kind, key in keys:
                    result = connection.execute("SELECT reason FROM revocations WHERE kind=? AND id=?", (kind, key)).fetchone()
                    if result:
                        applicable.append({"kind": kind, "id": key, "reason": result[0]})
                value = materialize_staged(view, candidates, assets, phase=part_manifest["phase"],
                                           selection_weights=part_manifest["selection_weights"], epoch_seed=epoch_seed,
                                           caption_length=caption_length, revocations=applicable, load_images=load_images)
                if value is not None:
                    yield value
    finally:
        connection.close()


def validate_dataset_release(path, *, materialize=False):
    manifest = _dataset_manifest(path)
    from .quality_gate import binding_from_identity, load_bound_quality_policy, validate_quality_admission
    quality_binding = binding_from_identity(manifest["run_identity"])
    quality_state = load_bound_quality_policy(quality_binding) if quality_binding else None
    connection = _readonly_index(manifest["index_path"])
    try:
        _require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "dataset index corrupt")
        _require(_index_coverage(connection) == manifest["coverage"], "dataset coverage index mismatch")
    finally:
        connection.close()
    # This validates hashes and candidate bindings only; no duplicate image I/O.
    for part in manifest["parts"]:
        _, _, _, views, _ = staged_release_contents(part["path"])
        if quality_state is not None:
            for view in views:
                validate_quality_admission(view["asset_ids"], view["admission"], manifest["run_identity"], state=quality_state)
    count = sum(1 for _ in iter_dataset_records(path)) if materialize else None
    if materialize:
        _require(count == manifest["coverage"]["published_records"], "dataset reader count mismatch")
    return {"success": True, "phase": manifest["phase"], "parts_verified": len(manifest["parts"]),
            "coverage": manifest["coverage"], "records_materialized": count,
            "verification_scope": "original_images_and_records" if materialize else "artifacts_receipts_and_index"}
