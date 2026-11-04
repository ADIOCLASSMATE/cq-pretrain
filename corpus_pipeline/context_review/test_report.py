"""Independent report safeguards and offline presentation fixtures."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from corpus_pipeline.context_review import report
from corpus_pipeline.context_review.test_worker import item, result


def row(number=1, human="keep", recommendation="keep"):
    candidate = item("sha-placeholder")
    candidate["review_number"] = number
    candidate["source_ref"] = {"path": "/unread/source.tar", "member": "item.webp", "source_url": "https://example.test/source"}
    candidate["human_review"] = {"item_id": candidate["item_id"], "review_number": number, "decision": human,
                                 "note": "original human note", "updated_at": "2026-10-25T00:00:00Z"}
    candidate["source_evidence_sha256"] = "e" * 64
    value = result(candidate["item_id"])
    value["recommendation"] = recommendation
    if recommendation in {"exclude_pair", "drop_image_keep_text"}:
        value.update(training_quality="unusable", image_quality="unusable", alignment="mismatched")
    elif recommendation == "flag_low_information":
        value.update(training_quality="usable_low_information")
    return {"candidate": candidate, "result": value, "source_evidence": {"members": []}, "batch_id": "batch-test"}


class ReportPolicyTests(unittest.TestCase):
    def test_human_ai_exclusion_is_never_overridden_by_new_keep(self):
        source = row(human="exclude_ai")
        before = deepcopy(source)
        overlay = report.make_overlay(source)
        self.assertEqual(overlay["human_review"]["decision"], "exclude_ai")
        self.assertEqual(overlay["context_review"]["recommendation"], "keep")
        self.assertTrue(overlay["human_ai_exclusion_preserved"])
        self.assertFalse(overlay["human_decision_modified"])
        self.assertFalse(overlay["ai_provenance_reassessed"])
        self.assertFalse(overlay["final_training_publication"])
        self.assertEqual(source, before)

    def test_loss_of_subject_is_damage_not_original_pair_mismatch(self):
        source = row(recommendation="exclude_pair")
        self.assertEqual(report.exclusion_cause(source["result"]), "image_information_destroyed")
        source["result"]["image_quality"] = "usable"
        self.assertEqual(report.exclusion_cause(source["result"]), "model_image_text_mismatch")

    def test_cross_summary_separates_human_quality_ai_and_interleaved(self):
        rows = [row(1, "keep", "exclude_pair"), row(2, "exclude_quality", "keep"),
                row(3, "exclude_quality", "flag_low_information"), row(4, "exclude_ai", "keep")]
        rows[0]["candidate"]["task_type"] = "interleaved"
        rows[0]["result"].update(task_type="interleaved", recommendation="drop_image_keep_text")
        summary = report.summarize(rows)
        self.assertEqual(summary["records"], 4)
        self.assertEqual([r["review_number"] for r in summary["human_kept_now_proposed_exclusions"]], [1])
        self.assertEqual([r["review_number"] for r in summary["human_quality_excluded_now_proposed_usable"]], [2, 3])
        self.assertEqual([r["review_number"] for r in summary["human_ai_excluded_now_contextually_usable_but_still_ai_excluded"]], [4])
        self.assertEqual(summary["human_context_cross"]["exclude_quality"], {"keep": 1, "flag_low_information": 1})
        self.assertEqual(summary["exclusion_causes"], {"image_information_destroyed": 1})

    def test_incomplete_and_bad_hash_farm_cannot_publish(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "COMPLETE.json").write_text(json.dumps({"complete": False}))
            with self.assertRaisesRegex(ValueError, "strict_complete"):
                report.load_completed(root)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            (root / "COMPLETE.json").write_text(json.dumps({"complete": True, "audit_mode": "strict", "skipped_gates": [],
                 "records": 688, "manifest": str(manifest), "manifest_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "artifact_hash"):
                report.load_completed(root)


class OfflinePresentationTests(unittest.TestCase):
    def test_html_escapes_all_untrusted_text_and_disallows_script_links(self):
        source = row()
        hostile = '</pre><script>alert("bad")</script><img src=x onerror=alert(1)>'
        source["candidate"]["source_context"]["caption_texts"] = [hostile]
        source["candidate"]["source_context"]["conversation_texts"] = [hostile]
        source["candidate"]["source_context"]["interleaved_segments"] = [
            {"member": 'bad" onmouseover="evil.txt', "sequence_index": 2, "text": hostile}]
        source["candidate"]["source_ref"]["source_url"] = "javascript:alert(1)"
        source["source_evidence"]["members"] = [{"member": "original.txt", "text": hostile}]
        source["result"]["reason"] = hostile
        picture = {"raw": b"actual image bytes", "mime": "image/png", "width": 2, "height": 3, "bytes": 18, "format": "PNG"}
        output = report.render_card(source, picture)
        self.assertNotIn('<script>alert("bad")', output)
        self.assertNotIn('href="javascript:', output)
        self.assertIn("&lt;script&gt;", output)
        self.assertIn("data-src=\"data:image/png;base64,", output)
        self.assertNotIn('<img src="data:', output)
        self.assertIn('bad&quot; onmouseover=&quot;evil.txt', output)

    def test_source_link_allows_only_http_and_https(self):
        for value in ("javascript:alert(1)", "data:text/html,hi", "file:///etc/passwd", "https://", "https://x.test/\nabc", None):
            with self.subTest(value=value):
                self.assertIsNone(report.safe_url(value))
        for value in ("http://example.test/a", "https://example.test/a?x=1&b=2"):
            self.assertEqual(report.safe_url(value), value)

    def test_csv_preserves_chinese_and_quotes_with_formula_protection(self):
        source = row()
        source["result"]["context_interpretation"] = '=HYPERLINK("https://bad.test","危险")'
        source["result"]["reason"] = '原文有逗号, 引号"和\n换行'
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "rows.csv"
            report.write_csv(path, [source])
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["context_interpretation"], "'" + source["result"]["context_interpretation"])
        self.assertEqual(rows[0]["reason"], source["result"]["reason"])

    def test_original_image_payload_is_exact_and_wrong_hash_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            raw_stream = io.BytesIO()
            Image.new("RGB", (7, 9), (20, 90, 50)).save(raw_stream, format="PNG")
            raw = raw_stream.getvalue()
            path = Path(name) / "image.tar"
            path.write_bytes(b"header-prefix" + raw + b"padding")
            candidate = {"item_id": hashlib.sha256(raw).hexdigest(),
                         "source_ref": {"path": str(path), "offset": 13, "length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
            picture = report.original_image(candidate)
            self.assertEqual(picture["raw"], raw)
            self.assertEqual((picture["width"], picture["height"]), (7, 9))
            candidate["source_ref"]["sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "hash_mismatch"):
                report.original_image(candidate)

    def test_json_duplicate_keys_are_rejected(self):
        for raw in ('{"passed":true,"passed":false}', '{"value":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                report.strict_json(raw)


if __name__ == "__main__":
    unittest.main()
