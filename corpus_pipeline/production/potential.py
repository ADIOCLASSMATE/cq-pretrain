"""Cheap resolution and exact-source annotation tags; these never grant admission.

The current corpus is not a frozen future high-resolution pool. Pixel potential
does not certify image quality, caption correctness, or usefulness for layout.
This module performs no image/model/network calls and does not rewrite records.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

from .common import digest

SCHEMA = "production.annotation_potential.v1"
CONTRACT_SCHEMA = "prepared.layout_source_contracts.v1"
MINIMUM_SHORT_EDGE = 128
ROUTES = {"layout_candidate", "not_default_layout", "sample_to_decide"}


@dataclass(frozen=True)
class CompiledSourceContracts:
    """Validated once per run; source lookup is constant-time per record."""
    entries: object
    config_sha256: str


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def compile_source_contracts(source_contracts):
    if isinstance(source_contracts, CompiledSourceContracts):
        return source_contracts
    if not isinstance(source_contracts, dict) or source_contracts.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unsupported layout source contracts schema")
    rows = source_contracts.get("contracts")
    if not isinstance(rows, list):
        raise ValueError("layout source contracts must contain a list")
    indexed = {}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid layout source contract")
        key = (row.get("collection"), row.get("source"))
        if not all(isinstance(value, str) and value for value in key) or key in seen:
            raise ValueError("duplicate or invalid exact source contract")
        seen.add(key)
        if row.get("match") != {"collection": key[0], "source": key[1]}:
            raise ValueError("source contract must match exact collection and source")
        if row.get("tag") not in ROUTES:
            raise ValueError("invalid source layout route")
        if not isinstance(row.get("upstream_annotation_type", []), list) or not isinstance(row.get("evidence", []), list):
            raise ValueError("invalid upstream annotation claim")
        indexed[key] = deepcopy(row)
    return CompiledSourceContracts(MappingProxyType(indexed), digest(source_contracts))


def _source_contract(record, source_contracts):
    compiled = compile_source_contracts(source_contracts)
    return compiled.entries.get((record.get("collection", "dcvlm"), record["source"]))


def _admission(record):
    explicit = record.get("record_admission", record.get("admission"))
    if explicit is not None:
        if not isinstance(explicit, dict) or explicit.get("decision") not in {"admit", "pending", "reject", "not_evaluated"}:
            raise ValueError("invalid explicit record admission")
        return {**deepcopy(explicit), "evidence_scope": "explicit_pipeline_admission_receipt"}
    normalized = record.get("normalized", {})
    source_decision = normalized.get("decision", "unknown")
    # A source's accepted flag alone cannot certify later integrity/AI/dedup gates.
    decision = "reject" if source_decision == "rejected" else "pending" if source_decision in {"quarantine", "quarantined"} else "not_evaluated"
    return {"decision": decision, "source_decision": source_decision,
            "reason_codes": deepcopy(normalized.get("reason_codes", [])),
            "evidence_scope": "source_normalization_only_not_final_pipeline_admission"}


def _local_layout_status(record, asset_ids):
    """Only adapter-bound local references can establish annotation recovery."""
    value = record.get("local_layout_annotation")
    if not isinstance(value, dict):
        return {"status": "not_verified", "annotation_refs": [],
                "reason": "source_page_claim_does_not_establish_local_annotation_recovery"}
    references = value.get("annotation_refs")
    bound = (value.get("status") == "recovered" and value.get("verified") is True
             and value.get("source_record_sha256") == record.get("source_record_sha256")
             and bool(record.get("source_record_sha256")) and value.get("asset_ids") == asset_ids
             and isinstance(references, list) and bool(references)
             and all(isinstance(ref, dict) and bool(ref.get("locator")) and bool(ref.get("sha256")) for ref in references))
    if not bound:
        return {"status": "not_verified", "annotation_refs": [],
                "reason": "local_layout_claim_lacks_exact_record_asset_and_reference_binding"}
    return {"status": "recovered", "annotation_refs": deepcopy(references),
            "coordinate_system": value.get("coordinate_system", "unspecified"),
            "binding_evidence": {"source_record_sha256": record["source_record_sha256"], "asset_ids": asset_ids},
            "reason": "prepared_adapter_bound_local_annotations_not_a_visual_quality_claim"}


def annotate_potential(record, assets, *, source_contracts, pixel_threshold=1048576):
    """Return cheap per-record/per-image potential without selecting a future pool.

    ``assets`` is the inspected-asset mapping keyed by image SHA. Only decoded
    original dimensions are used; model input size, post-resize size and declared
    source metadata cannot turn a small original into an eligible image. A missing
    inspected asset remains unknown instead of silently acquiring zero pixels.

    Callers may attach ``record_admission`` to record to capture the actual final
    gate receipt. Its decision stays independent of resolution/layout potential.
    """
    if not _positive_integer(pixel_threshold):
        raise ValueError("pixel_threshold must be a positive integer")
    if not isinstance(assets, dict):
        raise ValueError("assets must be an inspected-asset mapping")
    contract = _source_contract(record, source_contracts)
    admission = _admission(record)
    per_asset = []
    asset_ids = [ref["sha256"] for ref in record.get("source_image_refs", [])]
    for image_index, asset_id in enumerate(asset_ids):
        asset = assets.get(asset_id)
        if asset is not None and asset.get("asset_id") != asset_id:
            raise ValueError("inspected asset identity mismatch")
        width = asset.get("original_width") if asset else None
        height = asset.get("original_height") if asset else None
        known = _positive_integer(width) and _positive_integer(height)
        pixels = width * height if known else None
        short_edge = min(width, height) if known else None
        eligible = pixels >= pixel_threshold and short_edge >= MINIMUM_SHORT_EDGE if known else None
        if not known:
            tag = "original_dimensions_unknown"
        elif short_edge < MINIMUM_SHORT_EDGE:
            tag = "below_global_minimum_short_edge"
        elif not eligible:
            tag = "below_pixel_threshold"
        else:
            tag = "resolution_only_recaption_candidate"
        per_asset.append({"image_index": image_index, "asset_id": asset_id,
                          "original_dimensions": [width, height] if known else None,
                          "native_pixels": pixels, "original_short_edge": short_edge,
                          "dimension_evidence_scope": "inspected_original_dimensions" if known else "not_available",
                          "resolution_eligible": eligible, "recaption_potential_tag": tag,
                          "visual_quality_verified": False, "caption_quality_verified": False,
                          "automatic_api_required": False, "future_highres_pool_selected": False})
    if contract:
        upstream = {"types": deepcopy(contract.get("upstream_annotation_type", [])),
                    "availability": contract.get("annotation_availability", "unknown"),
                    "evidence": deepcopy(contract.get("evidence", []))}
        route = contract["tag"]
        reason = contract.get("reason", "exact_source_contract")
        domain = contract.get("domain", record.get("domain", "unknown"))
        source_contract_sha = digest(contract)
    else:
        upstream = {"types": [], "availability": "unknown", "evidence": []}
        route, reason = "sample_to_decide", "no_exact_collection_source_layout_contract"
        domain, source_contract_sha = record.get("domain", "unknown"), None
    eligible_count = sum(row["resolution_eligible"] is True for row in per_asset)
    return {"schema_version": SCHEMA, "record_id": record["original_sample_id"],
            "collection": record.get("collection", "dcvlm"), "source": record["source"], "domain": domain,
            "source_record_sha256": record.get("source_record_sha256"),
            "native_task": record.get("normalized", {}).get("category", "unknown"),
            "record_admission": admission, "training_admitted": admission["decision"] == "admit",
            "pixel_threshold": pixel_threshold, "minimum_original_short_edge": MINIMUM_SHORT_EDGE,
            "assets": per_asset, "image_slots": len(per_asset), "resolution_eligible_image_slots": eligible_count,
            "resolution_unknown_image_slots": sum(row["resolution_eligible"] is None for row in per_asset),
            "any_resolution_eligible": bool(eligible_count),
            "all_required_images_resolution_eligible": bool(per_asset) and eligible_count == len(per_asset),
            "layout_source_route": route, "layout_source_reason": reason,
            "layout_source_evidence_scope": "exact_source_contract" if contract else "unknown_source_requires_sampling",
            "layout_source_contract_sha256": source_contract_sha,
            "upstream_annotation_claim": upstream,
            "local_recovered_status": _local_layout_status(record, asset_ids),
            "automatic_api_required": False, "future_highres_pool_selected": False,
            "visual_quality_claim": None, "moss_fewshot_qualified": False,
            "potential_does_not_override_admission": True}
