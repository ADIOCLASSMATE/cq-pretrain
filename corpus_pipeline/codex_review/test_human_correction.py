"""Fixture-only checks for explicit, parent-bound human corrections."""
from copy import deepcopy
import unittest

from .human_import import validate_correction


ITEM_A = "a" * 64
ITEM_B = "b" * 64
PARENT_SHA = "c" * 64


def fixture():
    previous = {
        ITEM_A: {"item_id": ITEM_A, "review_number": 1, "decision": "keep",
                 "note": "已确认", "updated_at": "2026-10-25T02:00:00Z"},
        ITEM_B: {"item_id": ITEM_B, "review_number": 168, "decision": "unsure",
                 "note": "待用户确认", "updated_at": "2026-10-25T02:00:00Z"},
    }
    revised = deepcopy(previous)
    revised[ITEM_B].update(decision="exclude_quality", note="用户确认因质量排除",
                           updated_at="2026-10-25T03:00:00Z")
    evidence = {
        "schema_version": "corpus.human_review.chat_confirmation.v1",
        "parent_submission_sha256": PARENT_SHA,
        "changes": [{"item_id": ITEM_B, "review_number": 168,
                     "from_decision": "unsure", "to_decision": "exclude_quality",
                     "user_message": "第168张因质量太差排除"}],
    }
    return previous, revised, evidence


class HumanCorrectionTests(unittest.TestCase):
    def test_explicit_parent_bound_correction_passes_without_mutation(self):
        old, new, evidence = fixture()
        before = deepcopy((old, new, evidence))
        result = validate_correction(old, new, evidence, PARENT_SHA)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["item_id"], ITEM_B)
        self.assertEqual((old, new, evidence), before)
        self.assertEqual(new[ITEM_A], old[ITEM_A])

    def test_wrong_parent_or_schema_rejected(self):
        for field, value in [("parent_submission_sha256", "d" * 64),
                             ("schema_version", "unsupported.v1")]:
            old, new, evidence = fixture()
            evidence[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_correction(old, new, evidence, PARENT_SHA)

    def test_item_identity_set_and_fixed_numbers_cannot_change(self):
        for operation in ["remove", "add", "number", "embedded_id"]:
            old, new, evidence = fixture()
            if operation == "remove":
                del new[ITEM_A]
            elif operation == "add":
                new["e" * 64] = deepcopy(new[ITEM_A])
            elif operation == "number":
                new[ITEM_B]["review_number"] = 169
            else:
                new[ITEM_B]["item_id"] = ITEM_A
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                validate_correction(old, new, evidence, PARENT_SHA)

    def test_unlisted_edits_and_extra_fields_rejected(self):
        for operation in ["decision", "note", "time", "extra_field"]:
            old, new, evidence = fixture()
            if operation == "decision":
                new[ITEM_A]["decision"] = "exclude_ai"
            elif operation == "note":
                new[ITEM_A]["note"] = "silent edit"
            elif operation == "time":
                new[ITEM_A]["updated_at"] = "2026-10-25T03:00:00Z"
            else:
                new[ITEM_B]["unexpected"] = "not an allowed edited field"
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                validate_correction(old, new, evidence, PARENT_SHA)

    def test_evidence_must_cover_exact_actual_change_with_nonempty_user_message(self):
        for field, value in [("item_id", ITEM_A), ("review_number", 169),
                             ("from_decision", "keep"), ("to_decision", "exclude_ai"),
                             ("user_message", ""), ("user_message", "   "),
                             ("user_message", None)]:
            old, new, evidence = fixture()
            evidence["changes"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_correction(old, new, evidence, PARENT_SHA)
        old, new, evidence = fixture()
        evidence["changes"].append(deepcopy(evidence["changes"][0]))
        with self.assertRaises(ValueError):
            validate_correction(old, new, evidence, PARENT_SHA)

    def test_no_change_or_missing_evidence_is_not_a_new_correction(self):
        old, new, evidence = fixture()
        with self.assertRaises(ValueError):
            validate_correction(old, deepcopy(old), evidence, PARENT_SHA)
        evidence["changes"] = []
        with self.assertRaises(ValueError):
            validate_correction(old, new, evidence, PARENT_SHA)
        with self.assertRaises(ValueError):
            validate_correction(old, deepcopy(old), evidence, PARENT_SHA)


if __name__ == "__main__":
    unittest.main()
