"""Control-flow regressions; these fake model receipts are never corpus evidence."""
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from .common import digest, write_jsonl
from .planner import plan_record
from .staged_publish import build_candidate
from .staged_run import process_batch
from .test_planner import caption_record, fixture_catalog_candidate
from .state import State


class FakeClient:
    def __init__(self, review_verdict="pass", generation_verdict="pass", fail=False, omit_caption_checks=False):
        self.calls = []
        self.review_verdict = review_verdict
        self.generation_verdict = generation_verdict
        self.fail = fail
        self.omit_caption_checks = omit_caption_checks

    async def request(self, role, stage, context, assets, prompt, validator, **kwargs):
        self.calls.append((role, stage, deepcopy(context)))
        await asyncio.sleep(0)
        if self.fail:
            raise RuntimeError("fake failure for control-flow test")
        if stage == "typed_annotation.v1":
            value = {"native_task": context["native_task"], "semantic_task": context["prepared_semantic_task"],
                     "image_captions": [{"image_index": 0, "short": "A dog.", "medium": "A dog on grass.",
                                         "long": "A brown dog running on green grass."}],
                     "assistant_revisions": [], "concerns": [], "layout_candidates": []}
        else:
            value = {"verdict": self.review_verdict if stage == "original_supervision_review.v2" else self.generation_verdict,
                     "issues": [], "reason": "Unit-test fixture verdict; not real quality evidence"}
            if (context.get("candidate_annotation", {}).get("semantic_task") == "caption"
                    and context.get("original_record", {}).get("training_phase")
                    and not self.omit_caption_checks):
                value["caption_checks"] = [{"image_index": 0, "variant": length,
                                            "verdict": self.generation_verdict, "issues": [],
                                            "reason": "Unit-test variant check only."}
                                           for length in ("short", "medium", "long")]
        validator(value)
        asset_ids = [asset["sha256"] for asset in assets]
        identity = {"role": role, "model": "fake-test-only", "context_sha256": digest(context), "assets": asset_ids}
        # Simulates the transport receipt shape inside this temp-only control-flow
        # test; it is not an actual model result or corpus validation evidence.
        return {"status": "ok", "parsed": value, "identity": identity, "request_context": context,
                "cache_key": digest(identity), "real_model_inference": True, "fixture_only": True,
                "image_inputs": [{"asset_id": sha, "original_sha256": sha, "sent_sha256": sha, "crop": False,
                                  "original_dimensions": [128, 128], "sent_dimensions": [128, 128]} for sha in asset_ids]}


def fixture_record(text="A dog.", category="single_i2t"):
    record = caption_record(text)
    record["domain"] = "photo"
    record["normalized"].update(category=category, decision="accepted", reason_codes=[])
    if category != "single_i2t":
        record["raw"]["format"] = "conversation" if category != "pure_text" else "text"
    if category == "pure_text":
        record["source_image_refs"] = []
        record["raw"]["segments"] = [{"kind": "text", "text": "Native text."}]
    record["source_record_sha256"] = digest(record["raw"])
    return record


def imported_candidate(record, phase="early_mixed", origin="existing"):
    return fixture_catalog_candidate(record, phase=phase, origin=origin)


class StagedRunTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, record, *, phase="early_mixed", audit=0, imported=(), reasons=(), client=None):
        client = client or FakeClient()
        captured = {}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(io_workers=1, cached_detector_scores="unused", dedup_history=None,
                output=directory, record_concurrency=2, phase=phase, audit_fraction=audit,
                layout="off", publish_dir=None)
            assets = {ref["sha256"]: {"asset_id": ref["sha256"], "sha256": ref["sha256"], "short_edge": 128}
                      for ref in record["source_image_refs"]}
            async def inspect(*_args, **_kwargs):
                return assets, {}, {record["original_sample_id"]: list(reasons)}
            async def ai(records, assets, reasons, client, output, **kwargs):
                rows = [{"asset_id": ref["sha256"], "record_id": r["original_sample_id"],
                         "screening": {"decision": "admit", "reason": "fixture_only"}}
                        for r in records if not reasons[r["original_sample_id"]] for ref in r["source_image_refs"]]
                write_jsonl(output / "ai_evidence.jsonl", rows)
                return rows
            async def dedup(*_args, **_kwargs):
                return {"assets": [], "relations": [], "receipt": {"input_assets": len(assets)}}
            def publish(directory, records, assets, plans, **kwargs):
                captured.update(plans=deepcopy(plans), **deepcopy(kwargs))
                for decision in kwargs["admissions"].values():
                    self.assertTrue(decision["evidence"], "publisher requires evidence for pending too")
                return {"committed": True, "records": sum(a["decision"] == "admit" for a in kwargs["admissions"].values())}
            state = SimpleNamespace(candidates=lambda sid: list(imported), rejected_asset=lambda sha: False,
                                    screening_rejections=lambda: [])
            with patch("corpus_pipeline.production.staged_run.inspect_records", inspect), \
                 patch("corpus_pipeline.production.staged_run.ai_stage", ai), \
                 patch("corpus_pipeline.production.staged_run.dedup_stage", dedup), \
                 patch("corpus_pipeline.production.staged_run.commit_staged_release", publish):
                summary = await process_batch([record], state, client, args, Path(directory) / "batch", {"test": "control-flow"})
        return summary, captured, client

    async def test_usable_native_caption_avoids_model_calls(self):
        summary, captured, client = await self.exercise(fixture_record())
        self.assertEqual(client.calls, [])
        self.assertEqual(summary["admission_counts"], {"admit": 1})
        self.assertEqual(summary["action_counts"], {"reuse_original": 1})

    async def test_native_qa_is_preserved_without_generation(self):
        record = fixture_record("42", "multi_i2t")
        original = deepcopy(record)
        summary, _, client = await self.exercise(record)
        self.assertEqual(client.calls, [])
        self.assertEqual(summary["action_counts"], {"preserve_native_task": 1})
        self.assertEqual(record, original)

    async def test_pure_text_skips_visual_review_even_at_full_audit(self):
        _, _, client = await self.exercise(fixture_record(category="pure_text"), audit=1)
        self.assertEqual(client.calls, [])

    async def test_confirmed_missing_skips_need_judge_and_generates(self):
        summary, captured, client = await self.exercise(fixture_record(""), audit=1)
        self.assertEqual([call[1] for call in client.calls], ["typed_annotation.v1", "typed_quality.v1"])
        self.assertEqual(summary["new_candidates"], 1)
        self.assertEqual(summary["admission_counts"], {"admit": 1})
        self.assertEqual(len(captured["candidates"]), 1)

    async def test_metadata_pending_is_recorded_without_model_or_batch_crash(self):
        record = fixture_record("")
        record["caption_fields_inspected"] = False
        summary, captured, client = await self.exercise(record, audit=1)
        self.assertEqual(client.calls, [])
        self.assertEqual(summary["admission_counts"], {"pending": 1})
        self.assertTrue(captured["admissions"][record["original_sample_id"]]["evidence"])

    async def test_early_original_review_uncertain_retains_native_with_flags(self):
        record = fixture_record()
        summary, captured, client = await self.exercise(record, audit=1, client=FakeClient(review_verdict="uncertain"))
        self.assertEqual(summary["admission_counts"], {"admit": 1})
        self.assertEqual([call[1] for call in client.calls], ["original_supervision_review.v2"])
        self.assertTrue(captured["plans"][record["original_sample_id"]]["retained_with_uncertainty"])
        self.assertFalse(captured["plans"][record["original_sample_id"]]["qualified_coverage"])
        self.assertEqual(captured["candidates"], [])

    async def test_model_failure_records_pending_instead_of_claiming_coverage(self):
        summary, captured, _ = await self.exercise(fixture_record(), phase="highres_1mp", client=FakeClient(fail=True))
        self.assertEqual(summary["admission_counts"], {"pending": 1})
        self.assertEqual(captured["candidates"], [])

    async def test_existing_image_gates_block_every_model_call(self):
        summary, _, client = await self.exercise(fixture_record(""), audit=1, reasons=["required_image_short_edge_lt_128"])
        self.assertEqual(client.calls, [])
        self.assertEqual(summary["admission_counts"], {"reject": 1})

    async def test_original_review_does_not_certify_unseen_import_as_highres(self):
        record = fixture_record()
        imported = imported_candidate(record)
        summary, captured, client = await self.exercise(record, phase="highres_1mp", audit=1, imported=[imported])
        self.assertIn("typed_annotation.v1", [call[1] for call in client.calls])
        self.assertEqual(summary["new_candidates"], 1)
        self.assertEqual(captured["plans"][record["original_sample_id"]]["action"], "complete_highres_caption")

    async def test_failed_original_does_not_invalidate_separately_qualified_import(self):
        record = fixture_record("A wrong original caption.")
        imported = imported_candidate(record)
        summary, captured, client = await self.exercise(record, audit=1, imported=[imported], client=FakeClient(review_verdict="fail"))
        self.assertEqual([call[1] for call in client.calls], ["original_supervision_review.v2"])
        self.assertEqual(summary["new_candidates"], 0)
        self.assertEqual(captured["plans"][record["original_sample_id"]]["action"], "reuse_annotation")

    async def test_imported_generation_not_counted_as_fresh_generation(self):
        record = fixture_record()
        imported = imported_candidate(record, origin="generated")
        summary, _, client = await self.exercise(record, imported=[imported])
        self.assertEqual(client.calls, [])
        self.assertEqual(summary["new_candidates"], 0)

    async def test_early_generation_uncertain_is_admitted_with_actual_verdict(self):
        summary, captured, _ = await self.exercise(fixture_record(""), client=FakeClient(generation_verdict="uncertain"))
        self.assertEqual(summary["admission_counts"], {"admit": 1})
        self.assertEqual(captured["candidates"][0]["quality_receipt"]["receipt"]["verdict"], "uncertain")
        self.assertEqual(summary["retained_uncertainty_records"], 1)

    async def test_highres_generation_uncertain_stays_pending(self):
        summary, captured, _ = await self.exercise(fixture_record(""), phase="highres_1mp", client=FakeClient(generation_verdict="uncertain"))
        self.assertEqual(summary["admission_counts"], {"pending": 1})
        self.assertEqual(captured["candidates"], [])

    async def test_early_generation_confirmed_failure_stays_pending(self):
        summary, captured, _ = await self.exercise(fixture_record(""), client=FakeClient(generation_verdict="fail"))
        self.assertEqual(summary["admission_counts"], {"pending": 1})
        self.assertEqual(captured["candidates"], [])

    async def test_optional_audit_failure_keeps_usable_original_with_flag(self):
        summary, captured, _ = await self.exercise(fixture_record(), audit=1, client=FakeClient(fail=True))
        self.assertEqual(summary["admission_counts"], {"admit": 1})
        self.assertEqual(summary["retained_uncertainty_records"], 1)
        self.assertEqual(captured["candidates"], [])

    async def test_early_uncertain_qa_keeps_native_target(self):
        record = fixture_record("42", "multi_i2t")
        summary, captured, _ = await self.exercise(record, audit=1, client=FakeClient(review_verdict="uncertain"))
        self.assertEqual(summary["admission_counts"], {"admit": 1})
        self.assertEqual(summary["retained_uncertainty_records"], 1)
        self.assertEqual(captured["candidates"], [])

    async def test_source_potential_tags_do_not_require_model(self):
        summary, _, client = await self.exercise(fixture_record())
        self.assertEqual(summary["layout_source_counts"], {"sample_to_decide": 1})
        self.assertFalse(summary["future_highres_pool_automatically_selected"])
        self.assertEqual(client.calls, [])

    async def test_incomplete_variant_review_is_pending_not_triplet_pass(self):
        summary, captured, _ = await self.exercise(fixture_record(""), phase="highres_1mp",
                                                   client=FakeClient(omit_caption_checks=True))
        self.assertEqual(summary["admission_counts"], {"pending": 1})
        self.assertEqual(captured["candidates"], [])

    async def test_caption_missing_gate_does_not_remove_unknown_image_gate(self):
        from .stages import inspect_records
        record = fixture_record("")
        record["normalized"].update(decision="quarantine", reason_codes=["missing_text", "unknown_image_origin"])
        state = SimpleNamespace(rejected_asset=lambda sha: False)
        def inspect(ref, source, domain):
            return {"asset_id": ref["sha256"], "sha256": ref["sha256"], "size_decision": "pass"}
        with patch("corpus_pipeline.production.stages.inspect_asset", inspect):
            _, _, reasons = await inspect_records([record], state, workers=1)
        self.assertEqual(reasons[record["original_sample_id"]], ["unknown_image_origin"])
        record["caption_fields_inspected"] = False
        with patch("corpus_pipeline.production.stages.inspect_asset", inspect):
            _, _, reasons = await inspect_records([record], state, workers=1)
        self.assertEqual(reasons[record["original_sample_id"]], ["missing_text", "unknown_image_origin"])

    async def test_api_transport_singleflight_and_concurrency_without_network(self):
        from .api_client import QwenAPIClient
        class Transport:
            def __init__(self):
                self.active = self.peak = self.calls = 0
            async def request(self, **kwargs):
                self.calls += 1
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(.001)
                self.active -= 1
                return {"status": "response_received", "response_model": kwargs["model"],
                        "stop_reason": "end_turn", "response_text": '{"ok":true}'}
            async def close(self):
                pass
        transport = Transport()
        def valid(value):
            self.assertEqual(value, {"ok": True})
        with tempfile.TemporaryDirectory() as directory, \
             patch("corpus_pipeline.production.api_client.credentials_from_example", return_value=SimpleNamespace(base_url="https://fixture.invalid")), \
             patch("corpus_pipeline.production.api_client.BenchClient", return_value=transport):
            client = QwenAPIClient("unused", directory, concurrency=2, retries=0)
            try:
                same = await asyncio.gather(*(client.request("verifier", "fixture", {"id": 0}, [], "fixture", valid) for _ in range(8)))
                self.assertEqual(transport.calls, 1)
                self.assertEqual(len({row["cache_key"] for row in same}), 1)
                await asyncio.gather(*(client.request("verifier", "fixture", {"id": index}, [], "fixture", valid) for index in range(1, 7)))
                self.assertEqual(transport.calls, 7)
                self.assertEqual(transport.peak, 2)
                reused = await client.request("verifier", "fixture", {"id": 0}, [], "fixture", valid)
                self.assertTrue(reused["cache_hit"])
                self.assertEqual(transport.calls, 7)
            finally:
                await client.close()


class StateTests(unittest.TestCase):
    def test_duplicate_image_slots_cannot_hide_later_negative_provenance(self):
        record = fixture_record()
        record["source_image_refs"] *= 2
        record["normalized"]["provenance"]["images"] = [{"origin": "natural"}, {"origin": "ai_generated"}]
        with tempfile.TemporaryDirectory() as directory:
            state = State(Path(directory) / "state.sqlite", {"test": "negative_provenance"})
            try:
                state.ingest([record])
                self.assertTrue(state.rejected_asset("a" * 64))
            finally:
                state.close()


if __name__ == "__main__":
    unittest.main()
