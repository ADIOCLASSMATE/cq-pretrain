"""Offline tests for auditable replacement of invalid model responses."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from corpus_pipeline.api_review.merge_repaired import merge_records, record_errors


def valid_row(sha="a" * 64, model="test-model"):
    parsed = {
        "image_form": "photo", "assessment": "uncertain", "confidence": "low",
        "image_description": "A small photograph.", "visible_evidence": [{
            "location": "whole image", "observation": "Low resolution",
            "supports": "ambiguous", "alternative_explanation": "Resizing or compression",
        }], "counterevidence": [], "reason": "Insufficient evidence.",
        "needs_provenance": True, "edit_lineage": "not_established_from_pixels",
    }
    return {"image_sha256": sha, "requested_model": model, "round": "adjudicate",
            "status": "ok", "parsed": parsed, "text": json.dumps(parsed),
            "finish_reason": "end_turn", "text_truncated": False,
            "cache_key": "b" * 64, "prompt_sha256": "c" * 64}


def invalid_schema_row():
    row = valid_row()
    del row["parsed"]["visible_evidence"][0]["alternative_explanation"]
    row["text"] = json.dumps(row["parsed"])
    return row


class MergeRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def merge(self, originals, repairs):
        return merge_records(self.write("original.jsonl", originals), self.write("repairs.jsonl", repairs))

    def test_only_invalid_record_replaced_and_inputs_unchanged(self):
        good = valid_row("d" * 64)
        bad = invalid_schema_row()
        repair = valid_row()
        repair["cache_key"], repair["prompt_sha256"] = "e" * 64, "f" * 64
        original_path = self.write("original.jsonl", [good, bad])
        repair_path = self.write("repairs.jsonl", [repair])
        before_original, before_repair = original_path.read_bytes(), repair_path.read_bytes()
        merged, audit = merge_records(original_path, repair_path)
        self.assertEqual(merged.splitlines()[0], before_original.splitlines()[0])
        self.assertEqual([json.loads(line) for line in merged.splitlines()], [good, repair])
        self.assertEqual(original_path.read_bytes(), before_original)
        self.assertEqual(repair_path.read_bytes(), before_repair)
        self.assertEqual(audit["replacement_count"], 1)
        self.assertEqual(audit["remaining_invalid_count"], 0)
        self.assertEqual(audit["replacements"][0]["repair_cache_key"], "e" * 64)
        self.assertEqual(audit["replacements"][0]["original_prompt_sha256"], "c" * 64)

    def test_valid_original_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            self.merge([valid_row()], [valid_row()])

    def test_invalid_repair_rejected(self):
        with self.assertRaises(ValueError):
            self.merge([invalid_schema_row()], [invalid_schema_row()])

    def test_unknown_pair_or_different_round_rejected(self):
        different_image = valid_row("d" * 64)
        different_model = valid_row(model="other-model")
        different_round = valid_row()
        different_round["round"] = "blind"
        for row in (different_image, different_model, different_round):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.merge([invalid_schema_row()], [row])

    def test_duplicate_original_and_repair_pairs_rejected(self):
        for originals, repairs in (([invalid_schema_row()] * 2, [valid_row()]),
                                   ([invalid_schema_row()], [valid_row()] * 2)):
            with self.subTest(originals=len(originals), repairs=len(repairs)), self.assertRaises(ValueError):
                self.merge(originals, repairs)

    def test_ambiguous_text_and_mismatched_parsed_rejected(self):
        conflicting = valid_row()
        conflicting["text"] = "```json\n" + conflicting["text"] + "\n```\n```json\n{}\n```"
        mismatched = valid_row()
        mismatched["parsed"]["assessment"] = "likely_non_ai"
        for row in (conflicting, mismatched):
            with self.subTest(text=row["text"][:12]), self.assertRaises(ValueError):
                self.merge([invalid_schema_row()], [row])

    def test_missing_repairs_remain_explicitly_invalid(self):
        _, audit = self.merge([invalid_schema_row()], [])
        self.assertFalse(audit["all_valid_after_merge"])
        self.assertEqual(audit["remaining_invalid_count"], 1)

    def test_transport_failures_never_count_as_valid(self):
        base = valid_row()
        for updates in ({"status": "api_error"}, {"finish_reason": "max_tokens"}, {"text_truncated": True}):
            row = copy.deepcopy(base)
            row.update(updates)
            with self.subTest(updates=updates):
                self.assertTrue(record_errors(row))

    def test_multiple_repair_files_and_cross_file_duplicates(self):
        first_bad = invalid_schema_row()
        second_bad = invalid_schema_row()
        second_bad["image_sha256"] = "d" * 64
        original = self.write("original.jsonl", [first_bad, second_bad])
        first = self.write("first.jsonl", [valid_row()])
        second = self.write("second.jsonl", [valid_row("d" * 64)])
        _, audit = merge_records(original, [first, second])
        self.assertEqual(audit["replacement_count"], 2)
        self.assertEqual(len(audit["repair_inputs"]), 2)
        self.assertTrue(audit["all_valid_after_merge"])
        with self.assertRaises(ValueError):
            merge_records(original, [first, first])


if __name__ == "__main__":
    unittest.main()
