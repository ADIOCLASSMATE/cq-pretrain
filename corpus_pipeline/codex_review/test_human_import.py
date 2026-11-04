"""Fixture-only checks for human decisions; no production reads or mutations."""
from copy import deepcopy
import json
import unittest

from .human_import import resolve_row, strict_json, validate_submission


QUEUE_ID = "e" * 64
ITEM_A = "a" * 64
ITEM_B = "b" * 64
QUEUE_ROWS = [
    {"item_id": ITEM_A, "review_number": 1},
    {"item_id": ITEM_B, "review_number": 2},
]
REFERENCE = {"path": "/fixture/submission.json", "sha256": "c" * 64}


def human(decision="keep", item_id=ITEM_A, number=1):
    return {"item_id": item_id, "review_number": number, "decision": decision,
            "note": "人工检查备注", "updated_at": "2026-10-25T02:34:56.789Z"}


def payload(*rows):
    return {"schema_version": "corpus.human_review.decisions.v1",
            "queue_id": QUEUE_ID, "queue_count": len(QUEUE_ROWS),
            "exported_at": "2026-10-25T02:40:00.000Z", "decisions": list(rows)}


def model(item_id=ITEM_A):
    return {"schema_version": "corpus.codex_review.overlay.v1", "item_id": item_id,
            "decision": "pending", "report_category": "visual_pending",
            "rejection_reason": "none", "source_certification": False,
            "result": {"decision": "pending", "assessment": "uncertain",
                       "visible_evidence": [{"location": "subject", "observation": "ambiguous"}]},
            "source_ref": {"path": "/fixture/image.tar", "offset": 12}}


class StrictJSONTests(unittest.TestCase):
    def test_accepts_text_and_bytes_without_changing_unicode(self):
        value = {"note": "人工判定", "decisions": []}
        raw = json.dumps(value, ensure_ascii=False)
        self.assertEqual(strict_json(raw), value)
        self.assertEqual(strict_json(raw.encode("utf-8")), value)

    def test_duplicate_object_keys_rejected_at_every_depth(self):
        for raw in ['{"queue_id":"a","queue_id":"b"}',
                    '{"decisions":[{"decision":"keep","decision":"exclude_ai"}]}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json(raw)

    def test_nonfinite_numbers_rejected(self):
        for value in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                strict_json('{"value":' + value + '}')


class SubmissionTests(unittest.TestCase):
    def validate(self, value):
        return validate_submission(value, QUEUE_ID, QUEUE_ROWS)

    def test_valid_partial_submission_keeps_only_explicit_decisions(self):
        row = human()
        submitted = payload(row)
        before = deepcopy(submitted)
        self.assertEqual(self.validate(submitted), {ITEM_A: row})
        self.assertEqual(submitted, before)
        self.assertNotIn(ITEM_B, self.validate(submitted))

    def test_empty_submission_is_not_implicit_keep(self):
        self.assertEqual(self.validate(payload()), {})

    def test_wrong_queue_identifier_and_count_rejected(self):
        for key, value in [("queue_id", "another-queue"), ("queue_count", 3),
                           ("queue_count", True)]:
            submitted = payload(human())
            submitted[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate(submitted)

    def test_wrong_schema_rejected(self):
        submitted = payload(human())
        submitted["schema_version"] = "unknown.v1"
        with self.assertRaises(ValueError):
            self.validate(submitted)

    def test_duplicate_item_rejected_even_when_decisions_agree(self):
        with self.assertRaises(ValueError):
            self.validate(payload(human(), human()))

    def test_unknown_item_rejected_even_with_valid_review_number(self):
        with self.assertRaises(ValueError):
            self.validate(payload(human(item_id="d" * 64)))

    def test_number_and_id_must_match(self):
        for number in [2, 0, "1", True]:
            with self.subTest(number=number), self.assertRaises(ValueError):
                self.validate(payload(human(number=number)))

    def test_only_ui_decision_enums_accepted(self):
        for decision in ["reject", "admit", "pending", "AI", "", None, 1]:
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                self.validate(payload(human(decision)))
        for decision in ["keep", "exclude_ai", "exclude_quality", "unsure"]:
            with self.subTest(decision=decision):
                self.assertEqual(self.validate(payload(human(decision)))[ITEM_A]["decision"], decision)

    def test_extra_or_missing_fields_rejected(self):
        for location, key in [("payload", "exported_at"), ("row", "updated_at")]:
            for operation in ["add", "remove"]:
                submitted = payload(human())
                target = submitted if location == "payload" else submitted["decisions"][0]
                if operation == "add":
                    target["unexpected"] = "value"
                else:
                    del target[key]
                with self.subTest(location=location, operation=operation), self.assertRaises(ValueError):
                    self.validate(submitted)

    def test_invalid_time_and_nonstring_note_rejected(self):
        for field, value in [("updated_at", "not a timestamp"), ("note", 123),
                             ("note", "x" * 100001)]:
            row = human()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(payload(row))
        submitted = payload(human())
        submitted["exported_at"] = "not a timestamp"
        with self.assertRaises(ValueError):
            self.validate(submitted)


class ResolutionTests(unittest.TestCase):
    def test_each_explicit_human_decision_has_separate_final_category(self):
        expected = {
            "keep": ("admit", "admitted", "none"),
            "exclude_ai": ("reject", "ai_excluded", "ai_generated"),
            "exclude_quality": ("reject", "quality_excluded", "quality_unusable"),
            "unsure": ("pending", "visual_pending", "none"),
        }
        for decision, final in expected.items():
            with self.subTest(decision=decision):
                original = model()
                row = human(decision)
                result = resolve_row(original, row, REFERENCE)
                self.assertEqual(tuple(result[key] for key in
                                       ("decision", "report_category", "rejection_reason")), final)
                self.assertEqual(result["decision_authority"], "human_user")
                self.assertEqual(result["item_id"], ITEM_A)
                self.assertEqual(result["model_screening"], original)
                self.assertEqual(result["human_review"], row)
                self.assertEqual(result["human_submission"], REFERENCE)
                self.assertIs(result["source_certification"], False)
                self.assertTrue(result["schema_version"])

    def test_unreviewed_pending_item_remains_pending_without_human_attribution(self):
        result = resolve_row(model(ITEM_B), None, REFERENCE)
        self.assertEqual(result["decision"], "pending")
        self.assertEqual(result["report_category"], "visual_pending")
        self.assertEqual(result["rejection_reason"], "none")
        self.assertEqual(result["decision_authority"], "codex_model")
        self.assertIsNone(result["human_review"])
        self.assertIsNone(result["human_submission"])

    def test_existing_nonpending_model_categories_preserved_without_human_input(self):
        for decision, category, reason in [("admit", "admitted", "none"),
                                          ("reject", "ai_excluded", "ai_generated"),
                                          ("reject", "quality_excluded", "quality_unusable")]:
            original = model()
            original.update(decision=decision, report_category=category, rejection_reason=reason)
            with self.subTest(category=category):
                result = resolve_row(original, None, REFERENCE)
                self.assertEqual(result["decision"], decision)
                self.assertEqual(result["report_category"], category)
                self.assertEqual(result["rejection_reason"], reason)
                self.assertEqual(result["decision_authority"], "codex_model")

    def test_human_decision_cannot_override_nonpending_model_row(self):
        for decision, category, reason in [("admit", "admitted", "none"),
                                          ("reject", "ai_excluded", "ai_generated"),
                                          ("reject", "quality_excluded", "quality_unusable")]:
            original = model()
            original.update(decision=decision, report_category=category, rejection_reason=reason)
            with self.subTest(category=category), self.assertRaises(ValueError):
                resolve_row(original, human(), REFERENCE)

    def test_resolution_rejects_human_decision_for_a_different_image(self):
        with self.assertRaises(ValueError):
            resolve_row(model(ITEM_A), human(item_id=ITEM_B, number=2), REFERENCE)

    def test_resolution_does_not_mutate_any_input(self):
        original, row, reference = model(), human("exclude_quality"), deepcopy(REFERENCE)
        before = deepcopy((original, row, reference))
        result = resolve_row(original, row, reference)
        self.assertEqual((original, row, reference), before)
        self.assertIsNot(result, original)
        self.assertEqual(result["model_screening"], before[0])

    def test_partial_submission_cannot_silently_admit_omitted_items(self):
        decisions = validate_submission(payload(human()), QUEUE_ID, QUEUE_ROWS)
        output = [resolve_row(model(entry["item_id"]), decisions.get(entry["item_id"]), REFERENCE)
                  for entry in QUEUE_ROWS]
        self.assertEqual([row["decision"] for row in output], ["admit", "pending"])
        self.assertEqual([row["decision_authority"] for row in output], ["human_user", "codex_model"])


if __name__ == "__main__":
    unittest.main()
