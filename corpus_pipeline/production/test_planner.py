from copy import deepcopy
import unittest

from .common import digest
from .planner import extract_candidates, plan_record


def caption_record(text="A dog.", *, origin="unknown"):
    raw = {"format": "caption", "segments": [
        {"kind": "image", "image_index": 0}, {"kind": "text", "text": text}],
        "images": [{"sha256": "a" * 64}]}
    return {"collection": "dcvlm", "source": "datacomp_1b", "original_sample_id": "record-a",
            "raw": raw, "normalized": {"category": "single_i2t", "provenance": {"text": {"origin": origin}}},
            "source_record_sha256": digest(raw), "source_image_refs": [{"sha256": "a" * 64}],
            "pairing_state": "verified", "caption_fields_inspected": True,
            "locator": {"path": "/fixture.parquet", "row_index": 0}}


def evidence(record, candidate, verdict="usable", phase="early_mixed", **changes):
    return {"scope": "per_annotation_visual_review", "verified": True, "verdict": verdict,
            "evidence_ref": "/verified-review.json", "source_record_sha256": record["source_record_sha256"],
            "candidate_ids": [candidate["candidate_id"]], "phase": phase, **changes}


def triplet(record, phase="highres_1mp"):
    record["caption_candidates"] = [{"text": f"Description {length}", "granularity": length,
                                      "field": f"qualified.{length}", "origin": "ai"}
                                     for length in ("short", "medium", "long")]
    candidates = extract_candidates(record)
    return {"candidates": {item["candidate_id"]: evidence(record, item, phase=phase)
                            for item in candidates if item["granularity"] != "original"}}


def fixture_catalog_candidate(record, *, phase="early_mixed", origin="existing", verdict="pass", captions=None):
    """Protocol fixture only; never write these simulated receipts to a corpus."""
    from .staged_publish import build_candidate
    from .run import model_context, verification_context
    captions = captions or {"short": "A dog.", "medium": "A dog on grass.", "long": "A dog running through grass."}
    reviewed_captions = {"short": "A dog.", "medium": "A dog on grass.", "long": "A dog running through grass.", **captions}
    context = model_context(record, phase=phase)
    annotation = {"native_task": record["normalized"]["category"], "semantic_task": "caption",
                  "image_captions": [{"image_index": 0, **reviewed_captions}], "assistant_revisions": [],
                  "concerns": [], "layout_candidates": []}
    assets = [ref["sha256"] for ref in record["source_image_refs"]]
    def receipt(role, request_context, parsed):
        identity = {"role": role, "model": "unit-test-fixture", "context_sha256": digest(request_context), "assets": assets}
        return {"status": "ok", "identity": identity, "cache_key": digest(identity), "parsed": parsed,
                "request_context": request_context, "real_model_inference": True, "fixture_only": True,
                "image_inputs": [{"asset_id": sha, "original_sha256": sha, "sent_sha256": sha, "crop": False,
                                  "original_dimensions": [128, 128], "sent_dimensions": [128, 128]} for sha in assets]}
    quality = receipt("verifier", verification_context(record, context, annotation),
                      {"verdict": verdict, "issues": [], "reason": "Unit-test fixture only.",
                       "caption_checks": [{"image_index": 0, "variant": length, "verdict": verdict,
                                           "issues": [], "reason": "Unit-test variant check only."}
                                          for length in ("short", "medium", "long")]})
    return build_candidate(record, origin=origin, captions=captions,
                           provenance={"phase": phase, "annotation_ref": "unit-test-fixture-only"},
                           quality_receipt={"verdict": verdict, "scope": "per_annotation_visual_review", "evidence": quality},
                           generator_receipt=receipt("generator", context, annotation) if origin == "generated" else None)


class PlannerTests(unittest.TestCase):
    def test_short_original_retained_without_fabricating_quality_or_mutating_input(self):
        record = caption_record("Dog.")
        before = deepcopy(record)
        result = plan_record(record)
        self.assertEqual(result["action"], "reuse_original")
        self.assertFalse(result["requires_generation"])
        self.assertFalse(result["qualified_coverage"])
        self.assertFalse(result["automatic_training_admission"])
        self.assertEqual(record, before)

    def test_ai_caption_origin_is_preserved_not_a_regeneration_trigger(self):
        result = plan_record(caption_record(origin="ai"))
        self.assertEqual(result["action"], "reuse_original")
        self.assertEqual(result["candidates"][0]["origin"], "ai")
        self.assertEqual(result["caption_origin"], "ai")
        self.assertEqual(result["original_text_provenance"]["origin"], "ai")
        self.assertTrue(result["original_usable"])

    def test_alternative_original_field_cannot_admit_bad_native_target(self):
        record = caption_record("https://flickr.com/placeholder")
        record["raw"]["caption"] = "A dog."
        result = plan_record(record)
        self.assertEqual(result["action"], "reuse_annotation")
        self.assertFalse(result["original_usable"])

    def test_missing_caption_generates_only_after_complete_field_recovery(self):
        record = caption_record("")
        self.assertEqual(plan_record(record)["action"], "generate_missing")
        record["caption_fields_inspected"] = False
        self.assertEqual(plan_record(record)["action"], "repair_metadata")

    def test_unknown_or_broken_join_cannot_be_repaired_by_generation(self):
        record = caption_record("")
        for state in ("unresolved", "broken", None):
            record["pairing_state"] = state
            self.assertEqual(plan_record(record)["action"], "repair_metadata")

    def test_gate_precedes_caption_generation_and_stage_selection(self):
        for phase in ("early_mixed", "highres_1mp"):
            result = plan_record(caption_record(""), phase=phase, gate_reasons=["ai_generated", "short_edge_lt_128"])
            self.assertEqual(result["action"], "blocked_by_existing_gate")
            self.assertFalse(result["requires_generation"])

    def test_pure_text_never_requires_visual_models(self):
        record = caption_record()
        record["normalized"]["category"] = "pure_text"
        record["source_image_refs"] = []
        record["pairing_state"] = "unresolved"
        self.assertEqual(plan_record(record)["action"], "preserve_native_task")
        self.assertEqual(extract_candidates(record), [])

    def test_qa_dialogue_interleaved_targets_are_not_caption_candidates(self):
        for category in ("single_i2t", "multi_i2t", "multi_t2i", "interleaved"):
            record = caption_record("42")
            record["raw"]["format"] = "conversation"
            record["normalized"]["category"] = category
            for phase in ("early_mixed", "highres_1mp"):
                result = plan_record(record, phase=phase)
                self.assertEqual(result["action"], "preserve_native_task")
                self.assertEqual(result["candidates"], [])

    def test_unknown_caption_source_prompt_requires_adapter_repair(self):
        record = caption_record()
        record["prepared_adapter_status"] = "repair_metadata"
        self.assertEqual(plan_record(record)["action"], "repair_metadata")

    def test_url_placeholder_does_not_hide_megalith_alternative_fields(self):
        record = caption_record("https://flickr.com/photos/123.jpg")
        record["raw"].update(caption_internlm2="A dog playing outdoors.", caption_internlm2_short="A dog.",
                             caption_florence2="A dog on grass.", caption_florence2_short="A brown dog.",
                             caption_sharecap="A dog is running.", caption_sharecap_short="A running dog.")
        result = plan_record(record)
        self.assertEqual(result["action"], "reuse_annotation")
        self.assertEqual(len(result["selected_candidate_refs"]), 6)
        self.assertEqual(result["candidates"][0]["unusable_reason"], "url_only")
        self.assertFalse(result["qualified_coverage"])

    def test_url_in_real_caption_is_not_placeholder(self):
        record = caption_record("A sign reading https://example.com is on a wall.")
        self.assertEqual(plan_record(record)["action"], "reuse_original")

    def test_confirmed_placeholders_generate_missing_without_need_judge(self):
        for value in (None, "", "  ", "https://example.com/image.jpg", "/images/photo.png"):
            result = plan_record(caption_record(value))
            self.assertEqual(result["action"], "generate_missing")
            self.assertFalse(result["requires_review"])

    def test_non_string_caption_is_schema_repair_not_missing_text(self):
        self.assertEqual(plan_record(caption_record({"wrong": "field"}))["action"], "repair_metadata")

    def test_early_uncertain_original_is_retained_and_flagged_without_generation(self):
        record = caption_record()
        candidate = extract_candidates(record)[0]
        result = plan_record(record, quality_evidence=evidence(record, candidate, "uncertain"))
        self.assertEqual(result["action"], "reuse_original")
        self.assertEqual(result["reason_codes"], ["EARLY_UNCERTAINTY_RETAIN_AND_MARK"])
        self.assertFalse(result["requires_generation"])
        self.assertFalse(result["requires_review"])
        self.assertTrue(result["retained_with_uncertainty"])
        self.assertTrue(result["original_usable"])
        self.assertFalse(result["qualified_coverage"])
        self.assertEqual(result["candidates"][0]["quality_status"], "suspected")
        self.assertFalse(result["candidates"][0]["qualified"])

    def test_unbound_failure_cannot_disable_early_original(self):
        record = caption_record()
        candidate = extract_candidates(record)[0]
        for changes in ({"verified": False}, {"source_record_sha256": "wrong"},
                        {"candidate_ids": ["another"]}, {"evidence_ref": None},
                        {"scope": "source_stratum_audit"}):
            result = plan_record(record, quality_evidence=evidence(record, candidate, "needs_correction", **changes))
            self.assertEqual(result["action"], "reuse_original")
            self.assertTrue(result["retained_with_uncertainty"])
            self.assertFalse(result["requires_review"])

    def test_source_risk_signal_does_not_force_early_judge_or_regeneration(self):
        record = caption_record()
        record["caption_risk_flags"] = ["low_similarity", "possible_unverifiable_identity"]
        before = deepcopy(record)
        early = plan_record(record)
        self.assertEqual(early["action"], "reuse_original")
        self.assertTrue(early["retained_with_uncertainty"])
        self.assertFalse(early["requires_review"])
        self.assertEqual(early["source_quality_risk_flags"], record["caption_risk_flags"])
        self.assertEqual(record, before)
        late = plan_record(record, phase="highres_1mp")
        self.assertEqual(late["action"], "review_caption")
        self.assertFalse(late["qualified_coverage"])

    def test_uncertain_original_does_not_change_unknown_pairing_or_image_gate(self):
        record = caption_record()
        candidate = extract_candidates(record)[0]
        review = evidence(record, candidate, "uncertain")
        record["pairing_state"] = "unresolved"
        self.assertEqual(plan_record(record, quality_evidence=review)["action"], "repair_metadata")
        record["pairing_state"] = "verified"
        self.assertEqual(plan_record(record, quality_evidence=review,
                                    gate_reasons=["unknown_image_origin"])["action"], "blocked_by_existing_gate")

    def test_early_uncertain_bound_catalog_is_selected_with_flags(self):
        record = caption_record()
        candidate = fixture_catalog_candidate(record, verdict="uncertain")
        result = plan_record(record, candidates=[candidate])
        self.assertEqual(result["action"], "reuse_original")
        self.assertEqual(result["selected_catalog_candidate_ids"], [candidate["candidate_id"]])
        self.assertTrue(result["retained_with_uncertainty"])
        self.assertFalse(result["qualified_coverage"])
        self.assertFalse(result["requires_review"])

    def test_early_uncertain_catalog_can_fill_missing_native_without_rewriting(self):
        record = caption_record("")
        candidate = fixture_catalog_candidate(record, verdict="uncertain")
        result = plan_record(record, candidates=[candidate])
        self.assertEqual(result["action"], "reuse_annotation")
        self.assertFalse(result["original_usable"])
        self.assertEqual(result["selected_catalog_candidate_ids"], [candidate["candidate_id"]])
        self.assertTrue(result["retained_with_uncertainty"])
        self.assertFalse(result["requires_generation"])
        self.assertFalse(result["qualified_coverage"])

    def test_early_failed_catalog_is_not_reused(self):
        record = caption_record()
        for verdict in ("fail", "revise"):
            candidate = fixture_catalog_candidate(record, verdict=verdict)
            result = plan_record(record, candidates=[candidate])
            self.assertEqual(result["action"], "reuse_original")
            self.assertEqual(result["selected_catalog_candidate_ids"], [])
            self.assertFalse(result["retained_with_uncertainty"])

    def test_early_bound_failure_is_not_softened_by_source_risk_flags(self):
        record = caption_record()
        record["caption_risk_flags"] = ["possible_error"]
        candidate = extract_candidates(record)[0]
        result = plan_record(record, quality_evidence=evidence(record, candidate, "needs_correction"))
        self.assertEqual(result["action"], "regenerate_verified_failure")
        self.assertFalse(result["original_usable"])

    def test_verified_failure_regenerates_without_overwriting_original(self):
        record = caption_record()
        before = deepcopy(record)
        candidate = extract_candidates(record)[0]
        result = plan_record(record, quality_evidence=evidence(record, candidate, "needs_correction"))
        self.assertEqual(result["action"], "regenerate_verified_failure")
        self.assertTrue(result["requires_generation"])
        self.assertEqual(record, before)

    def test_alternative_usable_caption_avoids_regenerating_bad_original(self):
        record = caption_record("Incorrect caption")
        record["raw"]["detailed_caption"] = "A dog."
        original = extract_candidates(record)[0]
        review = {"candidates": {original["candidate_id"]: evidence(record, original, "needs_correction")}}
        result = plan_record(record, quality_evidence=review)
        self.assertEqual(result["action"], "reuse_annotation")
        self.assertNotIn(original["candidate_id"], result["selected_candidate_refs"])

    def test_highres_requires_quality_not_three_populated_strings(self):
        record = caption_record()
        triplet(record)
        result = plan_record(record, phase="highres_1mp")
        self.assertEqual(result["action"], "complete_highres_caption")
        self.assertFalse(result["qualified_coverage"])

    def test_highres_triplet_with_bound_stage_quality_is_reused(self):
        record = caption_record()
        quality = triplet(record)
        result = plan_record(record, phase="highres_1mp", quality_evidence=quality)
        self.assertEqual(result["action"], "reuse_annotation")
        self.assertTrue(result["qualified_coverage"])
        self.assertEqual(len(result["selected_candidate_refs"]), 3)

    def test_early_review_cannot_certify_highres_coverage(self):
        record = caption_record()
        quality = triplet(record, phase="early_mixed")
        result = plan_record(record, phase="highres_1mp", quality_evidence=quality)
        self.assertEqual(result["action"], "complete_highres_caption")
        self.assertFalse(result["qualified_coverage"])

    def test_only_one_qualified_length_does_not_cover_triplet(self):
        record = caption_record()
        record["caption_candidates"] = [{"text": "A dog.", "granularity": "short"}]
        short = extract_candidates(record)[1]
        result = plan_record(record, phase="highres_1mp", quality_evidence={"candidates": {
            short["candidate_id"]: evidence(record, short, phase="highres_1mp")}})
        self.assertFalse(result["qualified_coverage"])
        self.assertEqual(result["action"], "complete_highres_caption")

    def test_outside_frozen_highres_pool_preserves_original_record(self):
        record = caption_record()
        before = deepcopy(record)
        result = plan_record(record, phase="highres_1mp", in_stage_pool=False)
        self.assertEqual(result["action"], "not_in_stage_pool")
        self.assertEqual(record, before)

    def test_caption_direction_is_never_changed(self):
        for category in ("single_i2t", "single_t2i"):
            record = caption_record()
            record["normalized"]["category"] = category
            self.assertEqual(plan_record(record)["native_task"], category)

    def test_short_instruction_never_selects_long_replacement_for_training(self):
        record = caption_record("")
        record["caption_contract"] = {"image_index": 0, "allowed_training_lengths": ["short"]}
        record["caption_candidates"] = [{"text": "A long paragraph.", "granularity": "long"}]
        result = plan_record(record)
        self.assertEqual(result["action"], "generate_missing")
        self.assertEqual(result["selected_candidate_refs"], [])
        quality = triplet(record)
        late = plan_record(record, phase="highres_1mp", quality_evidence=quality)
        self.assertTrue(late["qualified_coverage"])
        self.assertEqual(late["allowed_training_lengths"], ["short"])

    def test_unknown_length_recaption_does_not_satisfy_terse_instruction(self):
        record = caption_record("")
        record["caption_contract"] = {"image_index": 0, "allowed_training_lengths": ["short"]}
        record["raw"]["recaption"] = "A replacement whose length contract is not known."
        self.assertEqual(plan_record(record)["action"], "generate_missing")
        record["caption_contract"]["allowed_training_lengths"] = ["short", "medium", "long"]
        self.assertEqual(plan_record(record)["action"], "reuse_annotation")

    def test_near_duplicate_or_other_view_candidate_cannot_be_reused(self):
        record = caption_record("")
        other = caption_record("A dog.")
        candidate = extract_candidates(other)[0]
        candidate["asset_view_identity"] = "nearest-neighbour"
        result = plan_record(record, candidates=[candidate])
        self.assertEqual(result["action"], "repair_metadata")

    def test_unknown_phase_rejected(self):
        with self.assertRaisesRegex(ValueError, "phase"):
            plan_record(caption_record(), phase="guess")

    def test_release_catalog_bridge_checks_record_and_stage_evidence(self):
        record = caption_record()
        def build(phase):
            return fixture_catalog_candidate(record, phase=phase)
        early = build("early_mixed")
        result = plan_record(record, candidates=[early])
        self.assertEqual(result["action"], "reuse_original")
        self.assertEqual(result["selected_catalog_candidate_ids"], [early["candidate_id"]])
        late = plan_record(record, candidates=[early], phase="highres_1mp")
        self.assertFalse(late["qualified_coverage"])
        self.assertEqual(late["action"], "complete_highres_caption")
        highres = build("highres_1mp")
        late = plan_record(record, candidates=[highres], phase="highres_1mp")
        self.assertTrue(late["qualified_coverage"])
        self.assertEqual(late["selected_catalog_candidate_ids"], [highres["candidate_id"]])
        other = deepcopy(record)
        other["original_sample_id"] = "another-sample"
        self.assertEqual(plan_record(other, candidates=[highres])["action"], "repair_metadata")

    def test_catalog_tampering_cannot_be_silently_ignored_by_usable_original(self):
        record = caption_record()
        candidate = fixture_catalog_candidate(record, captions={"short": "A dog."})
        candidate["captions"]["short"] = "A cat."
        result = plan_record(record, candidates=[candidate])
        self.assertEqual(result["action"], "repair_metadata")
        self.assertEqual(result["reason_codes"], ["CATALOG_CANDIDATE_BINDING_OR_CHECKSUM_INVALID"])

    def test_uncertain_external_catalog_does_not_count_as_qualified(self):
        record = caption_record("")
        candidate = fixture_catalog_candidate(record, phase="highres_1mp", verdict="uncertain", captions={"short": "A dog."})
        result = plan_record(record, candidates=[candidate], phase="highres_1mp")
        self.assertEqual(result["action"], "review_caption")
        self.assertFalse(result["qualified_coverage"])

    def test_partial_catalogs_require_coherent_triplet_review_before_highres_coverage(self):
        record = caption_record()
        candidates = [fixture_catalog_candidate(record, phase="highres_1mp", captions={length: "A dog."})
                      for length in ("short", "medium", "long")]
        result = plan_record(record, candidates=candidates, phase="highres_1mp")
        self.assertEqual(result["action"], "complete_highres_caption")
        self.assertFalse(result["qualified_coverage"])

    def test_provenance_label_alone_cannot_upgrade_early_model_receipt(self):
        record = caption_record()
        candidate = fixture_catalog_candidate(record, phase="early_mixed")
        # Build a well-formed immutable artifact that makes a new metadata claim,
        # while the original generator/verifier request remains an early review.
        from .staged_publish import build_candidate
        relabelled = build_candidate(record, origin="existing", captions=candidate["captions"],
            provenance={"phase": "highres_1mp"}, quality_receipt=candidate["quality_receipt"]["receipt"])
        result = plan_record(record, candidates=[relabelled], phase="highres_1mp")
        self.assertFalse(result["qualified_coverage"])
        self.assertEqual(result["action"], "complete_highres_caption")

    def test_global_pass_without_each_variant_check_cannot_certify_candidate(self):
        from .staged_publish import build_candidate
        record = caption_record()
        candidate = fixture_catalog_candidate(record, phase="highres_1mp")
        quality = deepcopy(candidate["quality_receipt"]["receipt"])
        del quality["evidence"]["parsed"]["caption_checks"]
        with self.assertRaises(ValueError):
            build_candidate(record, origin="existing", captions=candidate["captions"],
                            provenance=candidate["provenance"], quality_receipt=quality)


if __name__ == "__main__":
    unittest.main()
