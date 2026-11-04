import copy
import json
import unittest

from .audit_format_recovery import audit_response, remove_trailing_commas


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.obj = {"request_id": "r1", "captions": {"short": "Short.", "medium": "Medium description.",
                    "long": 'Literal comma sequences ,} and ,] and escaped "quotes" must remain unchanged.'}, "uncertainties": []}
        self.row = {"request_id": "r1", "http_status": 200, "response_model": "qwen-test", "stop_reason": "end_turn",
                    "response_text_truncated": False, "response_text": json.dumps(self.obj)}
        self.config = {"model": "qwen-test", "accepted_response_model_aliases": []}

    def test_lexical_recovery_preserves_every_string_and_exact_field_move(self):
        obj = copy.deepcopy(self.obj)
        obj["captions"]["uncertainties"] = obj.pop("uncertainties")
        text = json.dumps(obj)
        text = text[:-1] + ",}"
        original = copy.deepcopy(self.row)
        original["response_text"] = text
        frozen = copy.deepcopy(original)
        result = audit_response(original, self.config)
        self.assertEqual(result["classification"], "both")
        self.assertTrue(result["caption_strings_unchanged"])
        self.assertEqual(original, frozen)
        repaired, offsets = remove_trailing_commas(text)
        self.assertEqual(len(offsets), 1)
        self.assertEqual(json.loads(repaired)["captions"]["long"], self.obj["captions"]["long"])

    def test_conflict_extra_field_duplicate_key_and_empty_value_are_not_recovered(self):
        cases = []
        obj = copy.deepcopy(self.obj)
        obj["captions"]["uncertainties"] = ["Conflict must not overwrite root."]
        cases.append(json.dumps(obj))
        obj = copy.deepcopy(self.obj)
        obj["extra"] = "Unknown fields must not be discarded."
        cases.append(json.dumps(obj))
        cases.append(json.dumps(self.obj)[:-1] + ', "request_id":"r1"}')
        cases.append('{"request_id":"r1","captions":{},"uncertainties":[,]}')
        for text in cases:
            with self.subTest(text=text):
                row = {**self.row, "response_text": text}
                self.assertEqual(audit_response(row, self.config)["classification"], "unrecoverable")

    def test_model_identity_id_completeness_and_http_failure_cannot_be_repaired(self):
        self.assertEqual(audit_response(self.row, self.config)["classification"], "direct")
        for override in ({"response_model": "other-model"}, {"request_id": "other-id"}, {"stop_reason": "refusal"},
                         {"stop_reason": "max_tokens"}, {"response_text_truncated": True}):
            with self.subTest(override=override):
                self.assertEqual(audit_response({**self.row, **override}, self.config)["classification"], "unrecoverable")
        failed = audit_response({**self.row, "http_status": 429, "response_text": object()}, self.config)
        self.assertEqual(failed["classification"], "http_failure")
        self.assertEqual(failed["operations"], [])


if __name__ == "__main__":
    unittest.main()
