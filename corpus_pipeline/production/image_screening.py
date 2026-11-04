"""Turn a bound visual review into a screening decision, never a source fact.

The caller applies source exclusions/protections before invoking the detector.
Mixed and unknown origins are valid inputs here: a passed screen preserves their
original origin instead of relabelling them as verified natural images.
"""
from __future__ import annotations

from corpus_pipeline.api_review.aggregate import validate_parsed

from .common import digest


POLICY_VERSION = "image_screening.v3"
REVIEW_STAGE = "ai_visual_screening.v3"
EVIDENCE_SCOPE = "screening_not_provenance_certificate"


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _receipt_errors(receipt, asset_id):
    """Validate the existing ModelClient receipt and its new prompt semantics.

    This is consistency validation, not a cryptographic attestation that an
    arbitrary caller contacted a model. Production clients create the receipts;
    a bare parsed response or an old provenance-certification prompt is invalid.
    """
    if not isinstance(receipt, dict):
        return ["review_receipt_missing"]
    errors = []
    if receipt.get("status") != "ok" or receipt.get("real_model_inference") is not True:
        errors.append("review_not_successful_actual_inference")
    errors.extend(validate_parsed(receipt.get("parsed")))
    identity = receipt.get("identity")
    context = receipt.get("request_context")
    if not isinstance(identity, dict) or not isinstance(context, dict):
        return errors + ["review_identity_or_context_missing"]
    if (identity.get("stage") != REVIEW_STAGE or identity.get("role") != "verifier"
            or context.get("screening_policy") != POLICY_VERSION
            or context.get("source_certification") is not False):
        errors.append("review_screening_policy_mismatch")
    expected = asset_id if asset_id is not None else context.get("asset_id")
    if (not _sha(expected) or context.get("asset_id") != expected
            or identity.get("assets") != [expected]):
        errors.append("review_asset_binding_mismatch")
    if identity.get("context_sha256") != digest(context):
        errors.append("review_context_checksum_mismatch")
    if receipt.get("cache_key") != digest(identity):
        errors.append("review_identity_checksum_mismatch")
    images = receipt.get("image_inputs")
    if (not isinstance(images, list) or len(images) != 1
            or not isinstance(images[0], dict)):
        errors.append("review_image_input_missing")
    else:
        image = images[0]
        if (image.get("asset_id") != expected or image.get("original_sha256") != expected
                or not _sha(image.get("sent_sha256")) or image.get("crop") is not False):
            errors.append("review_image_input_binding_mismatch")
    return errors


def _cue_count(parsed, supports):
    cues = {(cue["location"].strip().casefold(), cue["observation"].strip().casefold())
            for cue in parsed["visible_evidence"] if cue["supports"] == supports
            and cue["location"].strip() and cue["observation"].strip()}
    # Repeated wording or repeated locations do not meet the two-independent-cue
    # rejection gate. Independence remains a model judgement, not a probability.
    return min(len({location for location, _ in cues}),
               len({observation for _, observation in cues}))


def adjudicate_review(review, provenance, *, asset_id=None):
    """Map an actual v3 review receipt to admit/reject/pending.

    ``review`` may be the ModelClient receipt itself or stages' ``status`` /
    ``receipt`` wrapper. ``provenance`` must be the existing normalized source
    evidence, not model-produced metadata. The result grants only image-screening
    eligibility; the caller still checks size, integrity and the complete record.
    """
    provenance = provenance if isinstance(provenance, dict) else {}
    result = {"policy_version": POLICY_VERSION, "decision": "pending",
              "reason": "ai_visual_review_incomplete", "origin": provenance.get("origin", "unknown"),
              "evidence_scope": EVIDENCE_SCOPE, "source_certification": False,
              "real_parent_lineage_established": False}

    def finish(decision, reason):
        result.update(decision=decision, reason=reason, reason_codes=[reason])
        return result

    # No model output may undo an explicit source exclusion or establish edit
    # lineage. This check also protects callers accidentally routing such assets.
    if result["origin"] == "ai_generated":
        return finish("reject", "source_explicit_ai_generated")
    verified_real_parent = (provenance.get("policy_exception") == "ai_edit_from_natural_source"
                            and provenance.get("edit_source_origin") == "natural"
                            and bool(provenance.get("edit_source_evidence")))
    if result["origin"] == "ai_edited":
        if verified_real_parent:
            result["real_parent_lineage_established"] = True
            return finish("admit", "source_verified_natural_parent_ai_edit")
        return finish("pending", "ai_edit_real_parent_not_established")

    if isinstance(review, dict) and "receipt" in review:
        if review.get("status") != "ok":
            result["validation_errors"] = ["review_wrapper_not_successful"]
            return finish("pending", "ai_visual_review_incomplete")
        receipt = review["receipt"]
    else:
        receipt = review
    errors = _receipt_errors(receipt, asset_id)
    if errors:
        result["validation_errors"] = errors
        return finish("pending", "ai_visual_review_invalid_or_incomplete")
    parsed = receipt["parsed"]
    result.update(review_cache_key=receipt["cache_key"], assessment=parsed["assessment"],
                  confidence=parsed["confidence"], needs_provenance=parsed["needs_provenance"],
                  ai_cue_count=_cue_count(parsed, "ai"), non_ai_cue_count=_cue_count(parsed, "non_ai"))
    if parsed["needs_provenance"]:
        return finish("pending", "ai_visual_review_needs_external_evidence")
    if parsed["assessment"] == "possible_ai_edit":
        return finish("pending", "ai_visual_review_edit_lineage_unresolved")
    if (parsed["assessment"] == "likely_non_ai" and parsed["confidence"] in {"medium", "high"}
            and result["non_ai_cue_count"] >= 1):
        return finish("admit", "ai_visual_review_passed_screening")
    if (parsed["assessment"] == "likely_ai_generated" and parsed["confidence"] == "high"
            and result["ai_cue_count"] >= 2):
        return finish("reject", "ai_visual_review_high_confidence_generated")
    return finish("pending", "ai_visual_review_inconclusive")
