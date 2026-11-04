"""Screening ledger regressions; all provenance/receipts below are fixtures."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from .common import digest
from .state import State


SHA = "a" * 64
SECOND_SHA = "b" * 64


def record(sid, origin="unknown", *, sha=SHA, provenance=None):
    evidence = {"origin": origin, "evidence": [{"fixture_only": True, "status": origin}],
                **deepcopy(provenance or {})}
    return {"original_sample_id": sid, "collection": "dcvlm", "source": "fixture_source",
            "source_record_sha256": digest(sid), "source_image_refs": [{"sha256": sha}],
            "normalized": {"provenance": {"images": [evidence]}}}


def outcome(decision, tag="first"):
    return {"decision": decision, "reason": "fixture_" + tag,
            "source_certification": False,
            "receipt": {"fixture_only": True, "id": tag}}


class ScreeningStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.sqlite"
        self.identity = {"test": "fixture_only_screening_state", "policy": "image_screening.v3"}
        self.state = State(self.path, self.identity)
        self.addCleanup(lambda: self.state.close())

    def reopen(self):
        self.state.close()
        self.state = State(self.path, self.identity)

    def test_later_source_protection_is_visible_before_processing_first_batch(self):
        unknown = record("first-unknown")
        rendered = record("last-rendered", "rendered")
        before = deepcopy([unknown, rendered])
        self.state.ingest([unknown, rendered])
        first = next(self.state.batches(1))[0]
        protection = self.state.protected_asset(first["source_image_refs"][0]["sha256"])
        self.assertEqual(protection["asset_id"], SHA)
        self.assertEqual(protection["sources"][0]["record_id"], "last-rendered")
        self.assertEqual(protection["sources"][0]["provenance"], rendered["normalized"]["provenance"]["images"][0])
        self.assertEqual(first["normalized"]["provenance"]["images"][0]["origin"], "unknown")
        self.assertEqual([unknown, rendered], before)
        self.assertIsNone(self.state.protected_asset(SECOND_SHA))
        self.reopen()
        self.assertEqual(self.state.protected_asset(SHA), protection)

    def test_all_protected_slots_and_source_records_are_preserved(self):
        duplicated = record("repeated", "programmatic")
        duplicated["source_image_refs"].append({"sha256": SHA})
        duplicated["normalized"]["provenance"]["images"].append({"origin": "rendered"})
        self.state.ingest([duplicated, record("other", "rendered")])
        sources = self.state.protected_asset(SHA)["sources"]
        self.assertEqual([(s["record_id"], s["image_index"]) for s in sources],
                         [("other", 0), ("repeated", 0), ("repeated", 1)])

    def test_explicit_ai_negative_overrides_protection_even_same_record(self):
        value = record("contradictory", "rendered")
        value["source_image_refs"].append({"sha256": SHA})
        value["normalized"]["provenance"]["images"].append({"origin": "ai_generated"})
        self.state.ingest([value])
        self.assertTrue(self.state.rejected_asset(SHA))
        self.assertIsNone(self.state.protected_asset(SHA))
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM source_protections").fetchone()[0], 1)

    def test_natural_edit_requires_exception_parent_and_lineage_evidence(self):
        valid = {"policy_exception": "ai_edit_from_natural_source", "edit_source_origin": "natural",
                 "edit_source_evidence": [{"fixture_only": True, "status": "natural"}]}
        records = [record("valid", "ai_edited", provenance=valid)]
        for index, missing in enumerate(valid, 1):
            evidence = {key: value for key, value in valid.items() if key != missing}
            records.append(record("missing-" + missing, "ai_edited", sha=str(index) * 64, provenance=evidence))
        self.state.ingest(records)
        self.assertIsNotNone(self.state.protected_asset(SHA))
        for index in range(1, 4):
            self.assertIsNone(self.state.protected_asset(str(index) * 64))

    def test_natural_and_unknown_alone_are_not_detector_protections(self):
        self.state.ingest([record("natural", "natural"), record("unknown", sha=SECOND_SHA)])
        self.assertIsNone(self.state.protected_asset(SHA))
        self.assertIsNone(self.state.protected_asset(SECOND_SHA))

    def test_screening_is_durable_idempotent_and_separate_from_origins(self):
        original = record("unknown")
        self.state.ingest([original])
        value = outcome("reject")
        before = deepcopy(value)
        saved = self.state.save_screening_outcome(SHA, value)
        self.assertEqual(value, before)
        self.assertEqual(saved, {"asset_id": SHA, **value})
        self.assertEqual(self.state.save_screening_outcome(SHA, value), saved)
        self.assertEqual(self.state.save_screening_outcome(SHA, saved), saved)
        self.assertEqual(self.state.screening_rejections(), [SHA])
        self.assertFalse(self.state.rejected_asset(SHA))
        self.assertEqual(next(self.state.batches(1))[0], original)
        self.reopen()
        self.assertEqual(self.state.screening_outcome(SHA), saved)
        saved["receipt"]["id"] = "caller mutation"
        self.assertEqual(self.state.screening_outcome(SHA)["receipt"]["id"], "first")

    def test_different_same_decision_receipts_preserve_evidence(self):
        self.state.ingest([record("unknown")])
        self.state.save_screening_outcome(SHA, outcome("admit", "one"))
        merged = self.state.save_screening_outcome(SHA, outcome("admit", "two"))
        self.assertEqual(merged["decision"], "admit")
        self.assertEqual({item["receipt"]["id"] for item in merged["evidence_history"]}, {"one", "two"})
        self.assertEqual(self.state.save_screening_outcome(SHA, merged), merged)
        self.assertEqual(self.state.screening_rejections(), [])

    def test_conflicting_decisions_remain_pending_and_keep_all_receipts(self):
        self.state.ingest([record("unknown")])
        self.state.save_screening_outcome(SHA, outcome("reject", "negative"))
        merged = self.state.save_screening_outcome(SHA, outcome("admit", "positive"))
        self.assertEqual(merged["decision"], "pending")
        self.assertEqual(merged["reason"], "image_screening_outcome_conflict")
        self.assertEqual({item["decision"] for item in merged["outcomes"]}, {"admit", "reject"})
        self.assertEqual(self.state.screening_rejections(), [])
        repeated = self.state.save_screening_outcome(SHA, outcome("reject", "negative"))
        self.assertEqual(repeated, merged)
        self.assertEqual(self.state.save_screening_outcome(SHA, merged), merged)
        self.reopen()
        self.assertEqual(self.state.screening_outcome(SHA), merged)

    def test_pending_cannot_be_silently_overwritten_by_later_pass(self):
        self.state.ingest([record("unknown")])
        self.state.save_screening_outcome(SHA, outcome("pending", "ambiguous"))
        merged = self.state.save_screening_outcome(SHA, outcome("admit", "later"))
        self.assertEqual(merged["decision"], "pending")
        self.assertEqual(merged["reason"], "image_screening_outcome_conflict")

    def test_binding_unknown_assets_and_invalid_decisions_fail_closed(self):
        self.state.ingest([record("unknown")])
        for sha, value in [("invalid", outcome("admit")), (SECOND_SHA, outcome("reject")),
                           (SHA, {"decision": "pass"}), (SHA, {**outcome("admit"), "asset_id": SECOND_SHA})]:
            with self.subTest(sha=sha, value=value), self.assertRaises(ValueError):
                self.state.save_screening_outcome(sha, value)
        self.assertIsNone(self.state.screening_outcome(SHA))

    def test_incomplete_ingestion_rebuilds_protections_before_reuse(self):
        def broken():
            for index in range(1000):
                yield record(str(index), "rendered")
            raise RuntimeError("fixture iterator failed after committed block")
        with self.assertRaises(RuntimeError):
            self.state.ingest(broken())
        self.assertIsNone(self.state.get("ingested"))
        self.state.ingest([record("recovered", sha=SECOND_SHA)])
        self.assertIsNone(self.state.protected_asset(SHA))
        self.assertEqual(len(next(self.state.batches(10))), 1)

    def test_screening_checksum_tampering_is_detected(self):
        self.state.ingest([record("unknown")])
        self.state.save_screening_outcome(SHA, outcome("reject"))
        with self.state.db:
            self.state.db.execute("UPDATE image_screening_outcomes SET decision='admit' WHERE sha256=?", (SHA,))
        with self.assertRaises(ValueError):
            self.state.screening_outcome(SHA)

    def test_protection_cannot_be_moved_to_different_asset_without_detection(self):
        self.state.ingest([record("rendered", "rendered"), record("unknown", sha=SECOND_SHA)])
        with self.state.db:
            self.state.db.execute("UPDATE source_protections SET sha256=? WHERE sha256=?", (SECOND_SHA, SHA))
        with self.assertRaises(ValueError):
            self.state.protected_asset(SECOND_SHA)


if __name__ == "__main__":
    unittest.main()
