"""Training-context gates and attachment/response binding; no paid inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from corpus_pipeline.codex_review.test_worker import MockCLI
from corpus_pipeline.context_review import worker


def item(item_id="opaque-a", task="caption_pair", status="complete"):
    return {"item_id": item_id, "review_number": 191, "source_name": "datacomp_1b",
            "task_type": task, "source_ref": {},
            "human_review": {"decision": "SECRET_PRIOR_VERDICT"},
            "source_context": {"caption_texts": ["Blue painted concrete wall texture background"],
                "conversation_texts": [], "interleaved_segments": [],
                "metadata_summary": {"url": "https://example.test/texture.jpg"},
                "context_status": status, "context_limitations": []}}


def result(item_id="opaque-a", **changes):
    value = {"item_id": item_id, "task_type": "caption_pair",
             "context_interpretation": "源文字明确在描述蓝色混凝土墙面的纹理背景。",
             "visible_content": "图中有蓝色粗糙墙面及颗粒纹理，没有被要求存在的人物主体。",
             "image_quality": "usable", "alignment": "aligned", "training_quality": "good",
             "recommendation": "keep", "reason": "颜色与材质对应，能提供一般材质和纹理理解监督。",
             "evidence": [{"kind": "visual", "detail": "蓝色墙面颗粒纹理可辨认。"},
                          {"kind": "source_text", "detail": "原文说明是蓝色混凝土墙面纹理。"}],
             "human_review_not_used": True}
    value.update(changes)
    return value


class PromptTests(unittest.TestCase):
    def test_prior_labels_never_enter_whitelisted_context(self):
        source = item()
        source["source_context"]["prior_review"] = "SECRET_MODEL_VERDICT"
        before = deepcopy(source)
        prompt = worker.build_prompt([source])
        self.assertNotIn("SECRET_PRIOR_VERDICT", prompt)
        self.assertNotIn("SECRET_MODEL_VERDICT", prompt)
        self.assertIn("Blue painted concrete", prompt)
        self.assertEqual(before, source)

    def test_explicit_truncation_and_original_source_unchanged(self):
        source = item()
        source["source_context"]["caption_texts"] = ["a" * (worker.TEXT_BUDGET_PER_ITEM + 1000), "b" * 100]
        before = deepcopy(source)
        data = json.loads(worker.build_prompt([source])[len(worker.PROMPT):])["items"][0]
        self.assertTrue(data["source_context"]["prompt_context_truncated"])
        self.assertEqual(sum(map(len, data["source_context"]["caption_texts"])), worker.TEXT_BUDGET_PER_ITEM)
        self.assertEqual(source, before)

    def test_attachment_mapping_and_interleaved_positions(self):
        first, second = item(), item("opaque-b", "interleaved")
        second["source_context"]["interleaved_segments"] = [
            {"member": "record-0.txt", "sequence_index": 0, "kind": "text", "text": "这是一段正文。"}]
        second["source_context"]["metadata_summary"]["record_images"] = [
            {"member": "record-1.webp", "sequence_index": 1, "is_target": True, "displayed": True},
            {"member": "record-3.webp", "sequence_index": 3, "is_target": False, "displayed": False}]
        data = json.loads(worker.build_prompt([first, second])[len(worker.PROMPT):])["items"]
        self.assertEqual([(r["attachment_number"], r["item_id"]) for r in data], [(1, "opaque-a"), (2, "opaque-b")])
        self.assertFalse(data[1]["source_context"]["metadata_summary"]["record_images"][1]["displayed"])

    def test_source_errors_batch_bounds_and_duplicate_ids_rejected(self):
        for sources in ([], [item()] * 2, [item(str(i)) for i in range(5)],
                        [item(status="extraction_error")], [item(status="invented")]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                worker.build_prompt(sources)


class DecisionTests(unittest.TestCase):
    def test_texture_and_sku_retention_are_distinct_valid_outcomes(self):
        worker.validate_result(result(), "opaque-a", item())
        worker.validate_result(result(training_quality="usable_low_information", recommendation="flag_low_information",
                                     alignment="weakly_related"), "opaque-a", item())

    def test_retention_needs_source_and_visual_evidence(self):
        for kind in ("visual", "source_text"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                worker.validate_result(result(evidence=[e for e in result()["evidence"] if e["kind"] == kind]), "opaque-a", item())

    def test_blurred_portrait_can_be_unusable_despite_unassessable_alignment(self):
        reject = result(image_quality="unusable", alignment="unassessable", training_quality="unusable",
                        recommendation="exclude_pair")
        worker.validate_result(reject, "opaque-a", item(status="partial"))

    def test_interleaved_damage_drops_target_slot_without_certifying_whole_record(self):
        value = result(task_type="interleaved", image_quality="unusable", alignment="unassessable",
                       training_quality="unusable", recommendation="drop_image_keep_text",
                       reason="只删除当前损坏图像槽；现有正文可候选保留，但未验证其他图像或整条记录。")
        worker.validate_result(value, "opaque-a", item(task="interleaved"))
        value["recommendation"] = "exclude_pair"
        with self.assertRaisesRegex(ValueError, "scope_mismatch"):
            worker.validate_result(value, "opaque-a", item(task="interleaved"))

    def test_partial_and_missing_context_never_implicitly_pass(self):
        for status in ("partial", "missing"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                worker.validate_result(result(), "opaque-a", item(status=status))
            worker.validate_result(result(alignment="unassessable", training_quality="needs_context",
                                          recommendation="needs_context"), "opaque-a", item(status=status))

    def test_truncated_unknown_and_empty_text_are_incomplete(self):
        variants = [item(task="unknown"), item(), item()]
        variants[1]["source_context"]["caption_texts"] = ["x" * (worker.TEXT_BUDGET_PER_ITEM + 1000)]
        variants[2]["source_context"]["caption_texts"] = []
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.validate_result(result(task_type=value["task_type"]), "opaque-a", value)

    def test_clear_mismatch_allowed_but_low_information_never_sufficient_for_exclusion(self):
        worker.validate_result(result(alignment="mismatched", training_quality="unusable", recommendation="exclude_pair"), "opaque-a", item())
        with self.assertRaisesRegex(ValueError, "damage_or_mismatch"):
            worker.validate_result(result(alignment="weakly_related", training_quality="unusable", recommendation="exclude_pair"), "opaque-a", item())

    def test_task_evidence_human_flag_and_quality_combinations_enforced(self):
        bads = [result(task_type="conversation"), result(human_review_not_used=1),
                result(recommendation="exclude_pair"), result(image_quality="unusable"),
                result(evidence=result()["evidence"] + result()["evidence"][:1]),
                result(reason="placeholder answer"), result(extra="invented")]
        for value in bads:
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.validate_result(value, "opaque-a", item())


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="context-worker-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.next_id = 0

    def prepare_image(self, source, directory, index, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{index:02d}.png"
        Image.new("RGB", (16, 16), (80, 100, 140)).save(path)
        with Image.open(path) as image:
            image.load()
        return {"path": str(path), "sent_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": 16, "height": 16, "decoded": True}

    def run_fake(self, values, sources, **kwargs):
        self.next_id += 1
        fake = MockCLI({"items": values}, **kwargs)
        with mock.patch("corpus_pipeline.codex_review.inputs.prepare_image", side_effect=self.prepare_image), \
             mock.patch.object(worker.subprocess, "Popen", side_effect=fake):
            outcome = worker.run_batch(sources, self.root / str(self.next_id), {"codex_cli_version": "test"})
        return outcome, fake

    def test_fixed_runtime_real_attachments_and_raw_context_provenance(self):
        sources = [item()]
        outcome, fake = self.run_fake([result()], sources)
        self.assertEqual(outcome["results"], [result()])
        self.assertEqual(outcome["errors"], {})
        self.assertEqual(fake.stdin_bytes.decode(), worker.build_prompt(sources))
        self.assertIn("gpt-5.6-sol", fake.command)
        self.assertIn('model_reasoning_effort="low"', fake.command)
        self.assertIn("--ignore-user-config", fake.command)
        self.assertEqual(fake.command.count("--image"), 1)
        self.assertFalse(outcome["metadata"]["prior_human_or_model_labels_in_prompt"])
        for artifact in outcome["metadata"]["artifacts"].values():
            self.assertEqual(worker.file_sha(artifact["path"]), artifact["sha256"])

    def test_bad_context_result_retries_only_invalid_sibling(self):
        outcome, _ = self.run_fake([result(), result("opaque-b")], [item(), item("opaque-b", status="partial")])
        self.assertEqual([r["item_id"] for r in outcome["results"]], ["opaque-a"])
        self.assertEqual(set(outcome["errors"]), {"opaque-b"})
        self.assertEqual(outcome["error_type"], "partial_response")

    def test_order_binding_and_tool_use_fail_closed(self):
        outcome, _ = self.run_fake([result("opaque-b"), result()], [item(), item("opaque-b")])
        self.assertEqual(outcome["results"], [])
        outcome, _ = self.run_fake([result()], [item()], tool="web_search")
        self.assertEqual(outcome["results"], [])
        self.assertEqual(outcome["metadata"]["error"], "unexpected_tool_use")


if __name__ == "__main__":
    unittest.main()
