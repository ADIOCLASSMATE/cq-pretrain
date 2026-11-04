"""Meaningful failure-closed and provenance boundary tests for review aggregation."""
import copy
import unittest

from corpus_pipeline.api_review.aggregate import aggregate, validate_parsed


MODELS = ["qwen", "kimi", "doubao"]


def vote(assessment="likely_ai_generated", confidence="high"):
    return {
        "image_form": "photo", "assessment": assessment, "confidence": confidence,
        "image_description": "Two objects on a table.",
        "visible_evidence": [
            {"location": location, "observation": f"Distorted {location} object edge.",
             "supports": "ai", "alternative_explanation": "Compression may distort the edge."}
            for location in ("left", "right")
        ],
        "counterevidence": ["Low resolution limits judgement."], "reason": "Two visible anomalies.",
        "needs_provenance": True, "edit_lineage": "not_established_from_pixels",
    }


def sample(**kwargs):
    return {"image_sha256": "a" * 64, "source_image_origin": "unknown",
            "pilot_group": "detector_alert", **kwargs}


def results(parsed=None, **kwargs):
    return [{"image_sha256": "a" * 64, "requested_model": model, "status": "ok", "round": "blind",
             "parsed": copy.deepcopy(parsed or vote()), **kwargs} for model in MODELS]


class AggregateTests(unittest.TestCase):
    def test_candidate_requires_all_high_and_two_distinct_cues(self):
        decisions, stats, _ = aggregate([sample()], results(), MODELS)
        self.assertEqual(decisions[0]["status"], "unanimous_high_ai_candidate")
        self.assertFalse(decisions[0]["automatic_deletion"])
        self.assertIsNone(decisions[0]["training_eligibility"])
        self.assertIsNone(stats["general_accuracy_precision_recall"])
        for change in ("medium", "one_cue", "duplicate_cue"):
            rows = results()
            parsed = rows[1]["parsed"]
            if change == "medium":
                parsed["confidence"] = "medium"
            elif change == "one_cue":
                parsed["visible_evidence"] = parsed["visible_evidence"][:1]
            else:
                parsed["visible_evidence"][1] = copy.deepcopy(parsed["visible_evidence"][0])
            self.assertEqual(aggregate([sample()], rows, MODELS)[0][0]["status"], "needs_adjudication")

    def test_failures_missing_duplicates_and_mixed_rounds_are_not_votes(self):
        failed = results()
        failed[0].update(status="timeout", parsed=vote("uncertain"))
        duplicate = results() + [results()[0]]
        mixed = results()
        mixed[2]["round"] = "adjudication"
        malformed = results()
        malformed[1]["parsed"]["confidence"] = 0.9
        for rows in (failed, results()[:2], duplicate, mixed, malformed):
            with self.subTest(rows=rows):
                decision, stats, _ = aggregate([sample()], rows, MODELS)
                self.assertEqual(decision[0]["status"], "api_incomplete")
                self.assertTrue(decision[0]["errors"])
                self.assertEqual(stats["by_pilot_group"]["detector_alert"]["candidate_count"], 0)

    def test_schema_rejects_wrong_types_extra_fields_and_fabricated_lineage(self):
        self.assertEqual(validate_parsed(vote()), [])
        changes = [
            ("needs_provenance", 1), ("image_form", []), ("assessment", "real"),
            ("edit_lineage", "verified_real_parent"), ("counterevidence", ["x"] * 4),
            ("visible_evidence", vote()["visible_evidence"] * 3),
            ("reason", None), ("image_description", ""), ("extra", "unknown"),
        ]
        for field, value in changes:
            parsed = vote()
            parsed[field] = value
            self.assertTrue(validate_parsed(parsed), field)
        malformed = vote()
        malformed["visible_evidence"][0]["supports"] = []
        self.assertTrue(validate_parsed(malformed))

    def test_source_evidence_overrides_votes_but_does_not_hide_api_errors(self):
        protected = sample(source_image_origin="programmatic", source_evidence_level="source_declared",
                           source_evidence=[{"url": "https://example.org/dataset", "claim": "Programmatically rendered."}])
        decisions, _, _ = aggregate([protected], results()[:2], MODELS)
        self.assertEqual(decisions[0]["status"], "protected_source")
        self.assertEqual(decisions[0]["review_status"], "api_incomplete")
        self.assertTrue(decisions[0]["audit_only"])
        for origin in ("natural", "mixed", "unknown"):
            item = {**protected, "source_image_origin": origin}
            self.assertEqual(aggregate([item], results(), MODELS)[0][0]["status"], "unanimous_high_ai_candidate")
        unproven = sample(source_image_origin="rendered")
        self.assertEqual(aggregate([unproven], results(), MODELS)[0][0]["status"], "unanimous_high_ai_candidate")

    def test_ai_controls_remain_source_rejected_even_if_visually_non_ai(self):
        control = sample(source_image_origin="ai_generated", ground_truth="ai_generated",
                         pilot_group="known_ai_positive_control", source_evidence_url="https://example.org/author")
        decisions, stats, indices = aggregate([control], results(vote("likely_non_ai", "medium")), MODELS)
        self.assertEqual(decisions[0]["status"], "source_reject")
        self.assertEqual(decisions[0]["review_status"], "unanimous_non_ai_candidate")
        self.assertEqual(stats["known_ai_positive_control_summary"]["image_count"], 1)
        self.assertEqual(indices["source_reject"], ["a" * 64])
        self.assertEqual(stats["by_pilot_group"]["detector_alert"]["image_count"], 0)

    def test_edit_or_low_confidence_is_always_adjudicated_and_models_are_distinct(self):
        for parsed in (vote("possible_ai_edit"), vote("likely_non_ai", "low"), vote("uncertain")):
            decision = aggregate([sample()], results(parsed), MODELS)[0][0]
            self.assertEqual(decision["status"], "needs_adjudication")
        with self.assertRaises(ValueError):
            aggregate([sample()], results(), ["qwen", "qwen", "doubao"])
        with self.assertRaises(ValueError):
            aggregate([sample(), sample()], results(), MODELS)


if __name__ == "__main__":
    unittest.main()
