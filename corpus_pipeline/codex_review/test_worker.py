"""Worker contract tests: real PNG attachments, mocked (never paid) CLI calls."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

from . import worker


def result(item_id="opaque-a", **overrides):
    value = {
        "item_id": item_id,
        "decision": "admit",
        "assessment": "likely_non_ai",
        "confidence": "medium",
        "image_kind": "photograph",
        "quality": "usable",
        "quality_reason": "主体清晰可见，具备基本视觉语义信息。",
        "rejection_reason": "none",
        "visible_evidence": [{"location": "左侧树叶", "observation": "叶脉与遮挡连续一致",
                              "supports": "non_ai"}],
        "alternative_explanation": "常规摄影可以合理解释这些细节。",
        "needs_external_evidence": False,
        "reason": "可见结构自然连贯，未见足够的生成异常。",
    }
    value.update(overrides)
    return value


def rejection(item_id="opaque-a", **overrides):
    value = result(item_id, decision="reject", assessment="likely_ai_generated", confidence="high",
                   rejection_reason="ai_generated",
                   visible_evidence=[
                       {"location": "左侧手部", "observation": "手指在掌心分叉且没有合理遮挡",
                        "supports": "ai"},
                       {"location": "右侧栏杆", "observation": "连续栏杆悬空合并到人物衣袖",
                        "supports": "ai"},
                   ])
    value.update(overrides)
    return value


def quality_rejection(item_id="opaque-a", **overrides):
    value = result(item_id, decision="reject", assessment="uncertain", confidence="medium",
                   quality="unusable", rejection_reason="quality_unusable",
                   quality_reason="整幅图像被严重色块破坏，主体轮廓及任何可用内容均无法识别。",
                   visible_evidence=[
                       {"location": "整幅画面", "observation": "大块解码色带覆盖全部主体，未留下可辨认结构",
                        "supports": "quality"},
                   ])
    value.update(overrides)
    return value


class SemanticGateTests(unittest.TestCase):
    def test_admission_requires_concrete_non_ai_evidence(self):
        worker.validate_result(result(), "opaque-a")
        for cues in [[], [{"location": "图片中央", "observation": "整体看起来不太确定",
                           "supports": "ambiguous"}]]:
            with self.subTest(cues=cues), self.assertRaisesRegex(ValueError, "admission_gate"):
                worker.validate_result(result(visible_evidence=cues), "opaque-a")

    def test_rejection_requires_high_confidence_likely_ai(self):
        worker.validate_result(rejection(), "opaque-a")
        for changes in [{"confidence": "medium"}, {"assessment": "likely_non_ai"},
                        {"visible_evidence": rejection()["visible_evidence"][:1]}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                worker.validate_result(rejection(**changes), "opaque-a")

    def test_rejection_requires_independent_locations_and_observations(self):
        for repeated_field in ["location", "observation"]:
            bad = rejection()
            bad["visible_evidence"][1][repeated_field] = bad["visible_evidence"][0][repeated_field]
            with self.subTest(repeated_field=repeated_field), self.assertRaisesRegex(ValueError, "rejection_gate"):
                worker.validate_result(bad, "opaque-a")

    def test_repeated_cue_rejected_even_for_pending(self):
        cue = result()["visible_evidence"][0]
        bad = result(decision="pending", visible_evidence=[cue, dict(cue)])
        with self.assertRaisesRegex(ValueError, "duplicate_cue"):
            worker.validate_result(bad, "opaque-a")

    def test_possible_edit_and_uncertain_cannot_be_automatically_admitted_or_rejected(self):
        for assessment in ["possible_ai_edit", "uncertain"]:
            for decision in ["admit", "reject"]:
                with self.subTest(assessment=assessment, decision=decision), self.assertRaisesRegex(ValueError, "requires_pending"):
                    factory = result if decision == "admit" else rejection
                    worker.validate_result(factory(assessment=assessment), "opaque-a")
            worker.validate_result(result(assessment=assessment, decision="pending", visible_evidence=[]), "opaque-a")

    def test_low_confidence_and_external_evidence_require_pending(self):
        for extra in [{"confidence": "low"}, {"needs_external_evidence": True}]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "requires_pending"):
                worker.validate_result(result(**extra), "opaque-a")
            worker.validate_result(result(decision="pending", **extra), "opaque-a")

    def test_nonphotographic_class_is_not_an_ai_rejection_reason(self):
        for kind in ["chart", "document", "handwriting", "conventional_render", "illustration", "composite"]:
            with self.subTest(kind=kind):
                worker.validate_result(result(image_kind=kind), "opaque-a")

    def test_strict_fields_ids_types_and_placeholders(self):
        variants = []
        missing = result(); del missing["confidence"]; variants.append(missing)
        extra = result(); extra["provenance"] = "invented"; variants.append(extra)
        variants += [result(item_id="wrong"), result(needs_external_evidence=1),
                     result(reason="placeholder description"), result(alternative_explanation="TBD later"),
                     result(reason="短"), result(visible_evidence="looks real"),
                     result(image_kind="certified_real")]
        for value in variants:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                worker.validate_result(value, "opaque-a")


class QualityGateTests(unittest.TestCase):
    def test_degraded_but_usable_is_not_automatically_rejected(self):
        worker.validate_result(result(quality="degraded_usable",
                                      quality_reason="有压缩痕迹，但主体与上下文仍清楚可辨。"), "opaque-a")

    def test_quality_rejection_can_be_independent_of_unknown_ai_provenance(self):
        for assessment in ["uncertain", "possible_ai_edit", "likely_non_ai", "likely_ai_generated"]:
            for confidence in ["medium", "high"]:
                for external in [False, True]:
                    with self.subTest(assessment=assessment, confidence=confidence, external=external):
                        worker.validate_result(quality_rejection(assessment=assessment, confidence=confidence,
                                                               needs_external_evidence=external), "opaque-a")

    def test_quality_rejection_requires_unusable_medium_or_high_and_visible_quality_evidence(self):
        variants = [quality_rejection(confidence="low"), quality_rejection(quality="usable"),
                    quality_rejection(quality="degraded_usable"), quality_rejection(visible_evidence=[]),
                    quality_rejection(visible_evidence=[
                        {"location": "左侧手部", "observation": "手指分叉且没有合理遮挡", "supports": "ai"}]),
                    quality_rejection(visible_evidence=[
                        {"location": "图像中心", "observation": "无法确定图像的生成来源", "supports": "ambiguous"}])]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.validate_result(value, "opaque-a")

    def test_quality_unusable_cannot_pass_or_stay_semantically_pending(self):
        for decision in ["admit", "pending"]:
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                worker.validate_result(result(decision=decision, quality="unusable"), "opaque-a")

    def test_nonrejected_item_must_not_claim_a_rejection_reason(self):
        for decision in ["admit", "pending"]:
            for reason in ["ai_generated", "quality_unusable"]:
                with self.subTest(decision=decision, reason=reason), self.assertRaises(ValueError):
                    worker.validate_result(result(decision=decision, rejection_reason=reason), "opaque-a")

    def test_rejected_item_must_name_an_allowed_rejection_reason(self):
        for reason in ["none", "low_resolution", "technical_failure", "unsupported_format"]:
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                worker.validate_result(rejection(rejection_reason=reason), "opaque-a")

    def test_quality_route_does_not_bypass_ai_rejection_gate(self):
        value = quality_rejection(rejection_reason="ai_generated")
        with self.assertRaises(ValueError):
            worker.validate_result(value, "opaque-a")

    def test_unusable_quality_is_primary_reason_even_with_strong_ai_evidence(self):
        value = rejection(quality="unusable", quality_reason="画面主体被严重色块破坏，缺少可用视觉信息。")
        value["visible_evidence"].append(
            {"location": "画面中心", "observation": "主体的大部分被失真色块覆盖", "supports": "quality"})
        with self.assertRaisesRegex(ValueError, "separate_quality_reason"):
            worker.validate_result(value, "opaque-a")
        value["rejection_reason"] = "quality_unusable"
        worker.validate_result(value, "opaque-a")
        value = rejection(quality="unusable", confidence="medium")
        with self.assertRaises(ValueError):
            worker.validate_result(value, "opaque-a")

    def test_quality_reason_is_required_and_not_a_placeholder(self):
        variants = [quality_rejection(quality_reason=""), quality_rejection(quality_reason="坏"),
                    quality_rejection(quality_reason="placeholder reason"), quality_rejection(quality_reason=None),
                    result(quality_reason=""), result(quality="unknown_quality")]
        missing = result(); del missing["quality_reason"]; variants.append(missing)
        for value in variants:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                worker.validate_result(value, "opaque-a")


class MockCLI:
    """Popen-shaped fake that writes the same evidence files as a fresh CLI."""

    def __init__(self, raw, *, event_raw=None, exit_code=0, stderr="", completed=True,
                 thread=True, tool=None, timeout=False, stubborn=False, malformed_events=False):
        self.raw = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
        self.event_raw = self.raw if event_raw is None else event_raw
        self.exit_code = exit_code
        self.stderr_text = stderr
        self.completed = completed
        self.thread = thread
        self.tool = tool
        self.timeout = timeout
        self.stubborn = stubborn
        self.malformed_events = malformed_events
        self.returncode = None
        self.pid = 987654321
        self.wait_calls = 0
        self.command = None
        self.stdin_bytes = None

    def __call__(self, command, *, stdin, stdout, stderr, start_new_session):
        assert stdin == subprocess.PIPE and start_new_session is True
        self.command = command
        self.stdout, self.stderr = stdout, stderr
        return self

    def communicate(self, data, timeout):
        self.stdin_bytes = data
        path = Path(self.command[self.command.index("--output-last-message") + 1])
        path.write_text(self.raw, encoding="utf-8")
        events = []
        if self.thread:
            events.append({"type": "thread.started", "thread_id": "test-fresh-thread"})
        if self.tool:
            events.append({"type": "item.completed", "item": {"type": self.tool}})
        events.append({"type": "item.completed", "item": {"type": "agent_message", "text": self.event_raw}})
        if self.completed:
            events.append({"type": "turn.completed", "usage": {"input_tokens": 128, "output_tokens": 64}})
        self.stdout.write(("\n".join(json.dumps(v, ensure_ascii=False) for v in events) + "\n").encode())
        if self.malformed_events:
            self.stdout.write(b"{broken-event\n")
        self.stdout.flush()
        self.stderr.write(self.stderr_text.encode()); self.stderr.flush()
        if self.timeout:
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.returncode = self.exit_code
        return None, None

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.stubborn and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.returncode = -signal.SIGKILL if self.stubborn else -signal.SIGTERM
        return self.returncode


class EventProvenanceTests(unittest.TestCase):
    def parse(self, events):
        with tempfile.TemporaryDirectory(prefix="codex-event-tests-") as temporary:
            path = Path(temporary) / "events.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                for event in events:
                    output.write(json.dumps(event) + "\n")
            return worker.parse_events(path)

    def test_multiple_threads_cannot_be_collapsed_into_one_provenance_record(self):
        for second_id in ["first-thread", "different-thread"]:
            with self.subTest(second_id=second_id), self.assertRaises(ValueError):
                self.parse([{"type": "thread.started", "thread_id": "first-thread"},
                            {"type": "thread.started", "thread_id": second_id},
                            {"type": "turn.completed", "usage": {"input_tokens": 1}}])

    def test_multiple_turns_cannot_be_silently_overwritten(self):
        with self.assertRaises(ValueError):
            self.parse([{"type": "thread.started", "thread_id": "first-thread"},
                        {"type": "turn.completed", "usage": {"input_tokens": 1}},
                        {"type": "turn.completed", "usage": {"input_tokens": 2}}])

    def test_usage_must_have_original_structured_record(self):
        for usage in [None, "unknown", 42, []]:
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                self.parse([{"type": "thread.started", "thread_id": "first-thread"},
                            {"type": "turn.completed", "usage": usage}])

    def test_last_agent_message_matches_final_not_intermediate_message(self):
        parsed = self.parse([
            {"type": "thread.started", "thread_id": "first-thread"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "intermediate"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "final raw JSON"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}},
        ])
        self.assertEqual(parsed["last_agent_message"], "final raw JSON")
        self.assertEqual(parsed["usage"], {"input_tokens": 1, "output_tokens": 2})
        self.assertEqual(parsed["thread_id"], "first-thread")


class WorkerBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="codex-worker-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "real_source.png"
        Image.new("RGB", (19, 23), (32, 85, 141)).save(self.source)
        self.batch_counter = 0

    def prepare_image(self, item, output_dir, index, **kwargs):
        if item.get("decode_error"):
            raise ValueError("test corrupt source")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{index:02d}.png"
        with Image.open(self.source) as image:
            image.load()
            image.convert("RGB").save(path)
        return {"path": str(path), "sent_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": 19, "height": 23, "decoded": True}

    def run_fake(self, fake, items=None):
        self.batch_counter += 1
        batch = self.root / f"batch-{self.batch_counter}"
        module = types.ModuleType("corpus_pipeline.codex_review.inputs")
        module.prepare_image = self.prepare_image
        with mock.patch.dict(sys.modules, {module.__name__: module}), \
             mock.patch.object(worker.subprocess, "Popen", side_effect=fake) as popen, \
             mock.patch.object(worker.os, "killpg") as killpg:
            output = worker.run_batch(items or [{"item_id": "opaque-a"}], batch,
                                      {"timeout_seconds": 0.1, "codex_cli_version": "test-version"})
        return output, batch, popen, killpg

    def assert_evidence_retained(self, output, batch):
        metadata = json.loads((batch / "metadata.json").read_text())
        self.assertEqual(metadata, output["metadata"])
        self.assertEqual(metadata["model"], "gpt-5.6-sol")
        self.assertEqual(metadata["reasoning_effort"], "low")
        self.assertEqual(metadata["codex_cli_version"], "test-version")
        self.assertEqual(metadata["error_type"], output["error_type"])
        self.assertEqual(metadata["transport_ok"], output["transport_ok"])
        for artifact in metadata["artifacts"].values():
            path = Path(artifact["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
        with gzip.open(batch / "events.jsonl.gz", "rb") as handle:
            handle.read()
        self.assertFalse((batch / "events.jsonl").exists())
        for attachment in metadata["attachments"]:
            path = Path(attachment["path"])
            with Image.open(path) as image:
                image.load(); self.assertEqual(image.size, (19, 23))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), attachment["sha256"])

    def test_success_has_real_attachment_and_fixed_cli_runtime(self):
        fake = MockCLI({"items": [result()]})
        output, batch, popen, _ = self.run_fake(fake)
        self.assertEqual(output["results"], [result()]); self.assertEqual(output["errors"], {})
        self.assertTrue(output["transport_ok"]); self.assertIsNone(output["error_type"])
        self.assertEqual(popen.call_count, 1)
        cmd = fake.command
        self.assertEqual(cmd[cmd.index("--model") + 1], "gpt-5.6-sol")
        self.assertIn('model_reasoning_effort="low"', cmd)
        self.assertIn("--ephemeral", cmd); self.assertIn("--ignore-user-config", cmd)
        self.assertEqual(cmd[cmd.index("--sandbox") + 1], "read-only")
        self.assertEqual(cmd.count("--image"), 1)
        self.assertIn("1. opaque-a", fake.stdin_bytes.decode())
        self.assert_evidence_retained(output, batch)

    def test_partial_response_keeps_valid_ids_and_only_retries_missing(self):
        fake = MockCLI({"items": [result("opaque-a"), result("opaque-c")]})
        items = [{"item_id": "opaque-a"}, {"item_id": "opaque-b"}, {"item_id": "opaque-c"}]
        output, _, _, _ = self.run_fake(fake, items)
        self.assertEqual([r["item_id"] for r in output["results"]], ["opaque-a", "opaque-c"])
        self.assertEqual(set(output["errors"]), {"opaque-b"})
        self.assertTrue(output["transport_ok"])
        self.assertEqual(output["error_type"], "partial_response")

    def test_semantically_bad_item_does_not_discard_good_sibling(self):
        fake = MockCLI({"items": [result("opaque-a"), result("opaque-b", assessment="possible_ai_edit")]})
        output, _, _, _ = self.run_fake(fake, [{"item_id": "opaque-a"}, {"item_id": "opaque-b"}])
        self.assertEqual([r["item_id"] for r in output["results"]], ["opaque-a"])
        self.assertEqual(set(output["errors"]), {"opaque-b"})
        self.assertIn("edit_or_uncertain", output["errors"]["opaque-b"])

    def test_quality_rejection_and_ai_rejection_keep_distinct_reasons_in_batch(self):
        values = [quality_rejection("opaque-a", needs_external_evidence=True), rejection("opaque-b")]
        output, batch, _, _ = self.run_fake(MockCLI({"items": values}),
                                          [{"item_id": "opaque-a"}, {"item_id": "opaque-b"}])
        self.assertEqual(output["results"], values)
        self.assertEqual(output["errors"], {})
        self.assertEqual([value["rejection_reason"] for value in output["results"]],
                         ["quality_unusable", "ai_generated"])
        self.assert_evidence_retained(output, batch)

    def test_wrong_order_rejects_batch_mapping(self):
        fake = MockCLI({"items": [result("opaque-b"), result("opaque-a")]})
        output, batch, _, _ = self.run_fake(fake, [{"item_id": "opaque-a"}, {"item_id": "opaque-b"}])
        self.assertEqual(output["results"], [])
        self.assertEqual(set(output["errors"]), {"opaque-a", "opaque-b"})
        self.assertIn("order_mismatch", output["metadata"]["error"])
        self.assert_evidence_retained(output, batch)

    def test_unknown_or_duplicate_id_rejects_all_results(self):
        for raw in [[result(), result("stranger")], [result(), result()]]:
            with self.subTest(raw=raw):
                output, _, _, _ = self.run_fake(MockCLI({"items": raw}))
                self.assertEqual(output["results"], [])
                self.assertIn("duplicate_or_unexpected_id", output["metadata"]["error"])

    def test_non_object_extra_result_is_not_silently_ignored(self):
        output, _, _, _ = self.run_fake(MockCLI({"items": [result(), "foreign scalar"]}))
        self.assertEqual(output["results"], [])
        self.assertEqual(output["error_type"], "validation_error")

    def test_last_message_must_match_last_event(self):
        fake = MockCLI({"items": [result()]}, event_raw=json.dumps({"items": [rejection()]}))
        output, batch, _, _ = self.run_fake(fake)
        self.assertEqual(output["results"], [])
        self.assertIn("last_message_event_mismatch", output["metadata"]["error"])
        self.assert_evidence_retained(output, batch)

    def test_tool_use_cannot_be_accepted_as_visual_only_review(self):
        for tool in ["command_execution", "mcp_tool_call", "web_search", "file_change"]:
            with self.subTest(tool=tool):
                output, _, _, _ = self.run_fake(MockCLI({"items": [result()]}, tool=tool))
                self.assertEqual(output["results"], [])
                self.assertEqual(output["metadata"]["error"], "unexpected_tool_use")

    def test_missing_thread_or_completion_cannot_commit(self):
        for kwargs in [{"thread": False}, {"completed": False}]:
            with self.subTest(kwargs=kwargs):
                output, batch, _, _ = self.run_fake(MockCLI({"items": [result()]}, **kwargs))
                self.assertEqual(output["results"], []); self.assertFalse(output["transport_ok"])
                self.assert_evidence_retained(output, batch)

    def test_malformed_events_prevent_commit_and_preserve_raw_bytes(self):
        output, batch, _, _ = self.run_fake(MockCLI({"items": [result()]}, malformed_events=True))
        self.assertEqual(output["results"], []); self.assertFalse(output["transport_ok"])
        self.assertEqual(output["error_type"], "validation_error")
        with gzip.open(batch / "events.jsonl.gz", "rb") as handle:
            self.assertIn(b"{broken-event", handle.read())
        self.assert_evidence_retained(output, batch)

    def test_network_exit_is_retryable_and_preserves_stderr(self):
        fake = MockCLI({"items": []}, exit_code=1, completed=False, stderr="HTTP 429 rate limit exceeded")
        output, batch, _, _ = self.run_fake(fake)
        self.assertEqual(output["error_type"], "transport_error")
        self.assertEqual(output["results"], []); self.assertEqual(output["metadata"]["exit_code"], 1)
        self.assertIn("429", (batch / "stderr.log").read_text())
        self.assert_evidence_retained(output, batch)

    def test_non_network_exit_is_not_mislabeled_transport_outage(self):
        fake = MockCLI({"items": []}, exit_code=2, completed=False, stderr="invalid CLI argument")
        output, _, _, _ = self.run_fake(fake)
        self.assertEqual(output["error_type"], "codex_exit")
        self.assertFalse(output["transport_ok"])

    def test_timeout_terminates_process_group_and_preserves_evidence(self):
        fake = MockCLI({"items": []}, completed=False, timeout=True)
        output, batch, _, killpg = self.run_fake(fake)
        self.assertEqual(output["error_type"], "timeout")
        self.assertEqual(output["results"], [])
        killpg.assert_called_once_with(fake.pid, signal.SIGTERM)
        self.assert_evidence_retained(output, batch)

    def test_unresponsive_timeout_escalates_to_kill(self):
        fake = MockCLI({"items": []}, completed=False, timeout=True, stubborn=True)
        output, _, _, killpg = self.run_fake(fake)
        self.assertEqual(output["error_type"], "timeout")
        self.assertEqual(killpg.call_args_list, [mock.call(fake.pid, signal.SIGTERM), mock.call(fake.pid, signal.SIGKILL)])

    def test_complete_raw_evidence_can_be_salvaged_after_nonzero_exit(self):
        output, batch, _, _ = self.run_fake(MockCLI({"items": [result()]}, exit_code=1))
        self.assertEqual(output["results"], [result()])
        self.assertEqual(output["error_type"], "codex_exit")
        self.assertEqual(output["metadata"]["exit_code"], 1)
        self.assertTrue(output["metadata"]["salvaged_valid_response"])
        self.assert_evidence_retained(output, batch)

    def test_decode_failure_never_attaches_or_fabricates_missing_item(self):
        items = [{"item_id": "opaque-a", "decode_error": True}, {"item_id": "opaque-b"}]
        fake = MockCLI({"items": [result("opaque-b")]})
        output, batch, _, _ = self.run_fake(fake, items)
        self.assertEqual([r["item_id"] for r in output["results"]], ["opaque-b"])
        self.assertEqual(output["metadata"]["attached_item_ids"], ["opaque-b"])
        self.assertEqual(fake.command.count("--image"), 1)
        self.assertIn("1. opaque-b", fake.stdin_bytes.decode())
        self.assertEqual(set(output["errors"]), {"opaque-a"})
        self.assertTrue(output["errors"]["opaque-a"].startswith("decode_error:"))
        self.assert_evidence_retained(output, batch)

    def test_all_decode_failures_make_no_cli_call_but_retain_audit_artifacts(self):
        fake = MockCLI({"items": []})
        output, batch, popen, _ = self.run_fake(fake, [{"item_id": "opaque-a", "decode_error": True}])
        popen.assert_not_called()
        self.assertEqual(output["results"], []); self.assertEqual(output["error_type"], "decode_error")
        self.assertFalse(output["transport_ok"])
        self.assert_evidence_retained(output, batch)

    def test_existing_batch_directory_cannot_overwrite_prior_evidence(self):
        fake = MockCLI({"items": [result()]})
        _, batch, _, _ = self.run_fake(fake)
        original = (batch / "metadata.json").read_bytes()
        module = types.ModuleType("corpus_pipeline.codex_review.inputs")
        module.prepare_image = self.prepare_image
        with mock.patch.dict(sys.modules, {module.__name__: module}), self.assertRaisesRegex(ValueError, "attempt_evidence"):
            worker.run_batch([{"item_id": "opaque-a"}], batch, {})
        self.assertEqual(original, (batch / "metadata.json").read_bytes())

    def test_precreated_claim_file_is_preserved_while_worker_adds_evidence(self):
        batch = self.root / "batch-1"
        batch.mkdir()
        claim = b'{"items":[{"item_id":"opaque-a"}]}\n'
        (batch / "input.json").write_bytes(claim)
        output, actual_batch, _, _ = self.run_fake(MockCLI({"items": [result()]}))
        self.assertEqual(actual_batch, batch)
        self.assertEqual(output["results"], [result()])
        self.assertEqual((batch / "input.json").read_bytes(), claim)
        self.assert_evidence_retained(output, batch)


if __name__ == "__main__":
    unittest.main()
