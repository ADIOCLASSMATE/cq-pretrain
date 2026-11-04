"""Routing regressions; fixtures verify decisions, not visual detector accuracy."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, AsyncMock

from .stages import inspect_records, ai_stage
from .staged_run import admission
from .state import State
from .test_staged_run import fixture_record
from .test_image_screening import ASSET, receipt, ai_cues


def record(origin="mixed", sid="source-fixture"):
    row = fixture_record()
    row["original_sample_id"] = sid
    row["normalized"].update(decision="quarantine" if origin in {"mixed", "unknown"} else "accepted",
        reason_codes=["image_provenance_" + origin] if origin in {"mixed", "unknown"} else [],
        provenance={"images": [{"origin": origin}], "text": {"origin": "unknown"}})
    return row


ASSETS = {ASSET: {"asset_id": ASSET, "sha256": ASSET, "size_decision": "pass"}}


def predictions(alert=False):
    return {ASSET: {"scores": {name: {"status": "ok", "prob": .99 if alert else .1}
                               for name in ("cf384", "unifd")}}}


class SourceScreeningTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_and_mixed_route_without_mutating_source_evidence(self):
        for origin in ("mixed", "unknown"):
            row = record(origin)
            original = deepcopy(row)
            with patch("corpus_pipeline.production.stages.inspect_asset", return_value=ASSETS[ASSET]):
                _, _, reasons = await inspect_records([row], SimpleNamespace(rejected_asset=lambda _: False), workers=1)
            self.assertEqual(reasons[row["original_sample_id"]], [])
            self.assertEqual(row, original)

    async def test_only_origin_uncertainty_is_removed(self):
        row = record()
        row["normalized"]["reason_codes"] += ["ai_edit_source_unknown", "unresolved_image_reference"]
        with patch("corpus_pipeline.production.stages.inspect_asset", return_value=ASSETS[ASSET]):
            _, _, reasons = await inspect_records([row], SimpleNamespace(rejected_asset=lambda _: True), workers=1)
        self.assertEqual(set(reasons[row["original_sample_id"]]),
                         {"ai_edit_source_unknown", "unresolved_image_reference", "cross_source_explicit_negative_evidence"})

    async def exercise(self, rows, *, alert=False, response=None, state=None, scores=None):
        client = SimpleNamespace(request=AsyncMock(return_value=response or receipt()))
        reasons = {r["original_sample_id"]: [] for r in rows}
        with tempfile.TemporaryDirectory() as temp, \
             patch("corpus_pipeline.production.stages.detector_scores", new_callable=AsyncMock,
                   return_value=predictions(alert) if scores is None else scores) as detectors:
            evidence = await ai_stage(rows, ASSETS, reasons, client, Path(temp), cached_scores="fixture",
                                      workers=1, state=state)
        return evidence, reasons, client, detectors

    async def test_no_alert_admits_mixed_without_api_or_origin_relabel(self):
        row = record()
        evidence, reasons, client, detectors = await self.exercise([row])
        self.assertEqual(admission(row, reasons[row["original_sample_id"]], evidence)["decision"], "admit")
        self.assertEqual(evidence[0]["origin"], "mixed")
        self.assertEqual(evidence[0]["screening"]["reason"], "detectors_no_alert")
        self.assertEqual(detectors.await_count, 1)
        self.assertEqual(client.request.await_count, 0)

    async def test_alert_reviewer_can_pass_without_certifying_source(self):
        evidence, reasons, client, _ = await self.exercise([record("unknown")], alert=True)
        self.assertEqual(evidence[0]["screening"]["decision"], "admit")
        self.assertFalse(evidence[0]["screening"]["source_certification"])
        self.assertEqual(client.request.await_count, 1)
        self.assertEqual(reasons["source-fixture"], [])

    async def test_one_rejected_asset_rejects_every_same_batch_record(self):
        rows = [record(sid="one"), record("unknown", "two")]
        evidence, reasons, client, _ = await self.exercise(rows, alert=True,
            response=receipt("likely_ai_generated", cues=ai_cues()))
        self.assertEqual(client.request.await_count, 1)
        for row in rows:
            sid = row["original_sample_id"]
            self.assertEqual(reasons[sid], ["ai_screening_rejected_generated"])
            self.assertEqual(admission(row, reasons[sid], [e for e in evidence if e["record_id"] == sid])["decision"], "reject")

    async def test_uncertain_or_detector_failure_stays_pending(self):
        for response, scores in ((receipt("uncertain", needs_provenance=True), None), (receipt(), {})):
            evidence, reasons, _, _ = await self.exercise([record()], alert=True, response=response, scores=scores)
            self.assertEqual(admission(record(), reasons["source-fixture"], evidence)["decision"], "pending")

    async def test_protected_exact_asset_in_later_source_skips_detector(self):
        first, other = record(), record("programmatic", "later-source")
        with tempfile.TemporaryDirectory() as temp:
            state = State(Path(temp) / "state.sqlite", {"test": "protection"})
            try:
                state.ingest([first, other])
                evidence, reasons, client, detectors = await self.exercise([first], state=state)
                self.assertEqual(evidence[0]["origin"], "mixed")
                self.assertTrue(evidence[0]["detector_skipped"])
                self.assertEqual(reasons["source-fixture"], [])
                self.assertEqual(detectors.await_count, 0)
                self.assertEqual(client.request.await_count, 0)
            finally:
                state.close()

    async def test_rejection_is_reused_across_batches_without_new_models(self):
        first, other = record(), record("unknown", "second-batch")
        with tempfile.TemporaryDirectory() as temp:
            state = State(Path(temp) / "state.sqlite", {"test": "cache"})
            try:
                state.ingest([first, other])
                await self.exercise([first], alert=True, response=receipt("likely_ai_generated", cues=ai_cues()), state=state)
                evidence, reasons, client, detectors = await self.exercise([other], state=state)
                self.assertEqual(reasons["second-batch"], ["ai_screening_rejected_generated"])
                self.assertEqual(state.screening_rejections(), [ASSET])
                self.assertEqual(detectors.await_count, 0)
                self.assertEqual(client.request.await_count, 0)
                self.assertTrue(evidence[0]["asset_screening_cache_reused"])
            finally:
                state.close()

    def test_missing_required_screening_slot_cannot_admit(self):
        row = record()
        self.assertEqual(admission(row, [], []) ["decision"], "pending")
        row["source_image_refs"] *= 2
        one = {"asset_id": ASSET, "screening": {"decision": "admit"}}
        self.assertEqual(admission(row, [], [one])["decision"], "pending")


if __name__ == "__main__":
    unittest.main()
