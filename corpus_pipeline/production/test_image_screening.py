from copy import deepcopy
import unittest

from .common import digest
from .image_screening import adjudicate_review, POLICY_VERSION, REVIEW_STAGE, EVIDENCE_SCOPE


ASSET = "a" * 64


def receipt(assessment="likely_non_ai", confidence="high", needs_provenance=False, cues=None):
    context = {"asset_id": ASSET, "source_certification": False, "screening_policy": POLICY_VERSION}
    identity = {"stage": REVIEW_STAGE, "role": "verifier", "model": "fixture-only",
                "context_sha256": digest(context), "assets": [ASSET]}
    if cues is None:
        cues = [{"location": "foreground hand", "observation": "continuous visible finger joints",
                 "supports": "non_ai", "alternative_explanation": "a generator might also render these"}]
    return {"status": "ok", "identity": identity, "cache_key": digest(identity),
            "request_context": context, "real_model_inference": True, "fixture_only": True,
            "image_inputs": [{"asset_id": ASSET, "original_sha256": ASSET, "sent_sha256": "b" * 64,
                              "crop": False, "original_dimensions": [1024, 1024], "sent_dimensions": [1024, 1024]}],
            "parsed": {"image_form": "photo", "assessment": assessment, "confidence": confidence,
                       "image_description": "A person holding a basket.", "visible_evidence": cues,
                       "counterevidence": ["Visual evidence does not certify origin."],
                       "reason": "Concrete visible relationships support this screening assessment.",
                       "needs_provenance": needs_provenance, "edit_lineage": "not_established_from_pixels"}}


def ai_cues():
    return [{"location": "left hand", "observation": "fingers terminate inside the handle without occlusion",
             "supports": "ai", "alternative_explanation": "unusual pose considered"},
            {"location": "upper background railing", "observation": "rail joins incompatible depths",
             "supports": "ai", "alternative_explanation": "compression considered"}]


class ImageScreeningTests(unittest.TestCase):
    def decide(self, row=None, origin="mixed", **kwargs):
        return adjudicate_review(receipt() if row is None else row, {"origin": origin}, asset_id=ASSET, **kwargs)

    def test_mixed_and_unknown_can_pass_without_relabelling_origin(self):
        for origin in ("mixed", "unknown", "natural"):
            with self.subTest(origin=origin):
                decision = self.decide(origin=origin)
                self.assertEqual(decision["decision"], "admit")
                self.assertEqual(decision["origin"], origin)
                self.assertEqual(decision["evidence_scope"], EVIDENCE_SCOPE)
                self.assertFalse(decision["source_certification"])
                self.assertFalse(decision["real_parent_lineage_established"])

    def test_medium_non_ai_evidence_can_pass_but_low_cannot(self):
        self.assertEqual(self.decide(receipt(confidence="medium"))["decision"], "admit")
        self.assertEqual(self.decide(receipt(confidence="low"))["decision"], "pending")

    def test_concrete_supporting_evidence_required_for_non_ai(self):
        for cues in ([], [{"location": " ", "observation": "looks real", "supports": "non_ai",
                           "alternative_explanation": ""}], ai_cues()):
            with self.subTest(cues=cues):
                self.assertEqual(self.decide(receipt(cues=cues))["decision"], "pending")

    def test_ai_rejection_requires_high_confidence_two_independent_cues(self):
        for confidence, cues, expected in (("high", ai_cues(), "reject"),
                                            ("medium", ai_cues(), "pending"),
                                            ("high", ai_cues()[:1], "pending")):
            with self.subTest(confidence=confidence, count=len(cues)):
                decision = self.decide(receipt("likely_ai_generated", confidence, cues=cues))
                self.assertEqual(decision["decision"], expected)
                self.assertEqual(decision["origin"], "mixed")
                self.assertFalse(decision["source_certification"])

    def test_repeated_locations_or_observations_are_not_independent(self):
        for key in ("location", "observation"):
            cues = ai_cues()
            cues[1][key] = " " + cues[0][key].upper() + " "
            decision = self.decide(receipt("likely_ai_generated", cues=cues))
            self.assertEqual(decision["decision"], "pending")
            self.assertEqual(decision["ai_cue_count"], 1)

    def test_unresolved_external_evidence_never_passes_or_rejects_from_pixels(self):
        for assessment in ("likely_non_ai", "likely_ai_generated", "possible_ai_edit", "uncertain"):
            decision = self.decide(receipt(assessment, needs_provenance=True, cues=ai_cues()))
            self.assertEqual(decision["decision"], "pending")
            self.assertEqual(decision["reason"], "ai_visual_review_needs_external_evidence")

    def test_edit_or_uncertain_visual_assessment_remains_pending(self):
        for assessment in ("possible_ai_edit", "uncertain"):
            self.assertEqual(self.decide(receipt(assessment))["decision"], "pending")

    def test_unknown_or_defer_are_invalid_schema_not_passes(self):
        for assessment in ("unknown", "defer"):
            decision = self.decide(receipt(assessment))
            self.assertEqual(decision["decision"], "pending")
            self.assertIn("invalid_assessment", decision["validation_errors"])

    def test_wrapped_actual_receipt_and_input_immutability(self):
        row = {"status": "ok", "receipt": receipt(), "automatic_policy_override": False}
        before = deepcopy(row)
        self.assertEqual(self.decide(row)["decision"], "admit")
        self.assertEqual(row, before)
        row["status"] = "pending"
        self.assertEqual(self.decide(row)["decision"], "pending")

    def test_missing_failed_or_bare_parsed_response_does_not_decide(self):
        for row in ({}, {"status": "failed"}, receipt()["parsed"], {"status": "ok", "receipt": None}):
            with self.subTest(row=row):
                decision = self.decide(row)
                self.assertEqual(decision["decision"], "pending")
                self.assertTrue(decision["validation_errors"])

    def test_old_prompt_semantics_never_implicitly_upgraded(self):
        for mutation in ("stage", "policy"):
            row = receipt()
            if mutation == "stage":
                row["identity"]["stage"] = "ai_visual_review.v2"
            else:
                del row["request_context"]["screening_policy"]
            row["identity"]["context_sha256"] = digest(row["request_context"])
            row["cache_key"] = digest(row["identity"])
            decision = self.decide(row)
            self.assertEqual(decision["decision"], "pending")
            self.assertIn("review_screening_policy_mismatch", decision["validation_errors"])

    def test_asset_receipt_binding_and_complete_image_are_required(self):
        for mutation in ("asset", "context", "image", "cropped", "cache", "actual", "certified", "missing_image"):
            row = receipt()
            if mutation == "asset":
                row["identity"]["assets"] = ["c" * 64]
            elif mutation == "context":
                row["request_context"]["asset_id"] = "c" * 64
            elif mutation == "image":
                row["image_inputs"][0]["original_sha256"] = "c" * 64
            elif mutation == "cropped":
                row["image_inputs"][0]["crop"] = True
            elif mutation == "cache":
                row["cache_key"] = "c" * 64
            elif mutation == "actual":
                row["real_model_inference"] = False
            elif mutation == "certified":
                row["request_context"]["source_certification"] = True
            else:
                row["image_inputs"] = []
            with self.subTest(mutation=mutation):
                self.assertEqual(self.decide(row)["decision"], "pending")

    def test_source_ai_generation_cannot_be_overruled(self):
        decision = self.decide(origin="ai_generated")
        self.assertEqual(decision["decision"], "reject")
        self.assertEqual(decision["reason"], "source_explicit_ai_generated")

    def test_source_ai_edit_requires_verified_real_parent_not_model_guess(self):
        row = receipt()
        self.assertEqual(self.decide(row, origin="ai_edited")["decision"], "pending")
        provenance = {"origin": "ai_edited", "policy_exception": "ai_edit_from_natural_source",
                      "edit_source_origin": "natural", "edit_source_evidence": [{"status": "natural"}]}
        decision = adjudicate_review(row, provenance, asset_id=ASSET)
        self.assertEqual(decision["decision"], "admit")
        self.assertEqual(decision["origin"], "ai_edited")
        self.assertTrue(decision["real_parent_lineage_established"])
        for field in ("policy_exception", "edit_source_origin", "edit_source_evidence"):
            missing = deepcopy(provenance)
            missing.pop(field)
            self.assertEqual(adjudicate_review(row, missing, asset_id=ASSET)["decision"], "pending")


if __name__ == "__main__":
    unittest.main()
