"""Temp-only gate evidence tests; no fixture is a corpus quality claim."""
import gzip
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from .common import atomic_json, digest, file_sha, write_jsonl
from .quality_gate import (apply_quality_gate, load_quality_policy, load_bound_quality_policy, quality_receipt,
                           validate_quality_admission)
from .staged_run import process_batch
from . import test_staged_publish as publishing_fixtures
from .test_staged_run import fixture_record, FakeClient


def policy_fixture(root, first_asset=None, first_outcome="clear", scope="reviewed_subset"):
    """A complete isolated 688-ID fixture matching the immutable bundle protocol."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    candidates, policies = [], []
    for i in range(1, 689):
        aid = first_asset if i == 1 and first_asset else digest({"fixture_only_image": i})
        source = {"sha256": aid, "path": "/fixture-only/source.tar", "member": f"{i}.jpg", "offset": i * 512, "length": 123}
        candidate = {"item_id": aid, "review_number": i, "source_ref": source}
        candidates.append(candidate)
        policies.append({"schema_version": "corpus.quality_policy.decision.v1", "item_id": aid,
            "review_number": i, "source_ref": source, "source_input_sha256": digest(candidate),
            "blur_status": first_outcome if i == 1 and first_outcome in {"blur_present", "uncertain"} else "clear",
            "context_action": "keep", "image_origin_followup": False, "reason": "Test fixture only.",
            "authority": "user_policy_applied_by_model"})
    write_jsonl(root / "candidates.jsonl", candidates)
    write_jsonl(root / "proposals.jsonl", policies)
    def ref(path, records=None):
        result = {"path": str(path), "sha256": file_sha(path)}
        if records is not None:
            result["records"] = records
        return result
    manifest = {"schema_version": "corpus.quality_policy.bundle.v1", "records": 688,
                "candidates": ref(root / "candidates.jsonl", 688), "decisions": ref(root / "proposals.jsonl", 688)}
    atomic_json(root / "manifest.json", manifest)
    manifest_ref = ref(root / "manifest.json")
    resolved = []
    for policy in policies:
        first = policy["review_number"] == 1
        blur = policy["blur_status"]
        ai = first and first_outcome == "ai"
        outcome = ("reject", "ai_excluded") if ai else (("reject", "quality_excluded") if blur == "blur_present"
                  else (("pending", "visual_pending") if blur == "uncertain" else ("admit", "admitted")))
        resolved.append({"schema_version": "corpus.quality_policy.resolved_overlay.v1",
            "item_id": policy["item_id"], "review_number": policy["review_number"], "source_ref": policy["source_ref"],
            "policy_input": policy, "quality_policy_manifest": manifest_ref, "blur_status": blur,
            "context_recommendation": policy["context_action"], "image_origin_followup": False,
            "decision": outcome[0], "report_category": outcome[1], "source_certification": False,
            "baseline_screening": {"item_id": policy["item_id"], "report_category": "ai_excluded" if ai else "admitted",
                                   "human_review": {"decision": "exclude_ai" if ai else "keep"}}})
    with gzip.open(root / "quality_decisions.jsonl.gz", "wt") as stream:
        for row in resolved:
            stream.write(json.dumps(row) + "\n")
    atomic_json(root / "AUDIT.json", {"passed": True, "skipped_gates": [], "policy_records": 688})
    report = {"revision_sha256": manifest_ref["sha256"], "input_manifest": manifest_ref,
              "artifacts": {"quality_decisions.jsonl.gz": {"sha256": file_sha(root / "quality_decisions.jsonl.gz"), "records": 688},
                            "AUDIT.json": {"sha256": file_sha(root / "AUDIT.json")}}}
    atomic_json(root / "REPORT.json", report)
    atomic_json(root / "COMPLETE.json", {"schema_version": "corpus.quality_policy.application.v1", "complete": True,
        "revision_sha256": manifest_ref["sha256"], "report_sha256": file_sha(root / "REPORT.json")})
    pointer = {"schema_version": "corpus.quality_policy.application.v1", "directory": str(root),
        "revision_sha256": manifest_ref["sha256"], "input_manifest": manifest_ref,
        "quality_decisions": ref(root / "quality_decisions.jsonl.gz", 688),
        "report": ref(root / "REPORT.json"), "completion": ref(root / "COMPLETE.json")}
    atomic_json(root / "CURRENT.json", pointer)
    atomic_json(root / "config.json", {"schema_version": "corpus.quality_policy.config.v1", "enabled": True,
                "require_blur_evidence": True, "scope": scope,
                "current_overlay": "CURRENT.json", "description": "fixture only"})
    return root / "config.json", candidates


def rewrite_resolved_fixture(root, transform):
    """Rebind an intentionally incorrect publisher output to test semantic checks."""
    path = root / "quality_decisions.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    transform(rows)
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    report = json.loads((root / "REPORT.json").read_text())
    report["artifacts"][path.name]["sha256"] = file_sha(path)
    atomic_json(root / "REPORT.json", report)
    complete = json.loads((root / "COMPLETE.json").read_text())
    complete["report_sha256"] = file_sha(root / "REPORT.json")
    atomic_json(root / "COMPLETE.json", complete)
    pointer = json.loads((root / "CURRENT.json").read_text())
    for key, filename in (("report", "REPORT.json"), ("completion", "COMPLETE.json"), ("quality_decisions", path.name)):
        pointer[key]["sha256"] = file_sha(root / filename)
    atomic_json(root / "CURRENT.json", pointer)


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_bound_load_covers_exact_reviewed_subset(self):
        config, candidates = policy_fixture(self.root)
        state = load_quality_policy(config)
        self.assertEqual(len(state["decisions"]), 688)
        self.assertEqual(state["binding"]["config"]["sha256"], file_sha(config))
        self.assertIn(candidates[0]["item_id"], state["decisions"])

    def test_disabled_or_optional_blur_config_cannot_bypass_cli_gate(self):
        config, _ = policy_fixture(self.root)
        original = json.loads(config.read_text())
        for field in ("enabled", "require_blur_evidence"):
            atomic_json(config, {**original, field: False})
            with self.assertRaisesRegex(ValueError, "mandatory"):
                load_quality_policy(config)

    def test_pointer_revision_or_bound_artifact_tamper_fails(self):
        config, _ = policy_fixture(self.root)
        pointer_path = self.root / "CURRENT.json"
        pointer = json.loads(pointer_path.read_text())
        atomic_json(pointer_path, {**pointer, "revision_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "revision binding"):
            load_quality_policy(config)
        atomic_json(pointer_path, pointer)
        with (self.root / "proposals.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            load_quality_policy(config)

    def test_all_images_scope_unknown_and_uncertain_images_stay_pending(self):
        config, candidates = policy_fixture(self.root, first_outcome="uncertain", scope="all_images")
        state = load_quality_policy(config)
        receipt = quality_receipt([candidates[0]["item_id"], "f" * 64], state)
        self.assertEqual(receipt["reason_codes"], ["image_blur_review_missing", "image_quality_review_pending"])
        self.assertFalse(receipt["image_origin_certified"])

    def test_default_scope_unknown_is_explicitly_unreviewed_and_still_requires_ai(self):
        config, _ = policy_fixture(self.root)
        state = load_quality_policy(config)
        receipt = quality_receipt(["f" * 64], state)
        self.assertEqual(state["binding"]["scope"], "reviewed_subset")
        self.assertEqual(receipt["reason_codes"], [])
        self.assertEqual(receipt["slots"][0]["blur_status"], "unreviewed")
        self.assertEqual(receipt["slots"][0]["policy_decision"], "not_reviewed")
        self.assertEqual(receipt["slots"][0]["quality_scope"], "out_of_scope")
        self.assertEqual(receipt["slots"][0]["action"], "continue_existing_ai_and_data_gates")
        admission = {"decision": "admit", "reason_codes": [], "evidence": {"image_quality": receipt}}
        with self.assertRaisesRegex(ValueError, "cannot replace required AI screening"):
            validate_quality_admission(["f" * 64], admission, {"quality_policy": state["binding"]})

    def test_scope_cannot_change_without_changing_frozen_run_binding(self):
        config, _ = policy_fixture(self.root)
        state = load_quality_policy(config)
        receipt = quality_receipt(["f" * 64], state)
        changed = deepcopy(state["binding"])
        changed["scope"] = "all_images"
        with self.assertRaisesRegex(ValueError, "receipt/run identity mismatch"):
            validate_quality_admission(["f" * 64], {"decision": "admit", "reason_codes": [],
                "evidence": {"image_quality": receipt}}, {"quality_policy": changed})
        with self.assertRaisesRegex(ValueError, "scope differs from frozen config"):
            load_bound_quality_policy(changed)

    def test_checksum_valid_publisher_mistakes_still_fail_semantic_and_coverage_checks(self):
        cases = [
            (lambda rows: rows[0].update(decision="admit", report_category="admitted"), "contradicts bound policy"),
            (lambda rows: rows.pop(), "exactly 688"),
            (lambda rows: rows.__setitem__(1, deepcopy(rows[0])), "duplicate quality image"),
            (lambda rows: rows[0]["source_ref"].update(sha256="f" * 64), "reference differs"),
        ]
        for i, (transform, expected_error) in enumerate(cases):
            with self.subTest(case=i):
                directory = self.root / str(i)
                config, _ = policy_fixture(directory, first_outcome="blur_present")
                rewrite_resolved_fixture(directory, transform)
                with self.assertRaisesRegex(ValueError, expected_error):
                    load_quality_policy(config)

    def test_bad_slot_does_not_mutate_record_or_other_text(self):
        config, candidates = policy_fixture(self.root, first_outcome="blur_present")
        state = load_quality_policy(config)
        record = {"original_sample_id": "sequence", "source_image_refs": [candidates[1]["source_ref"], {"sha256": "f" * 64}, candidates[0]["source_ref"]],
                  "raw": {"segments": [{"kind": "text", "text": "Keep the context."}]}}
        before = deepcopy(record)
        reasons = {"sequence": []}
        receipt = apply_quality_gate([record], reasons, state)["sequence"]
        self.assertEqual(reasons["sequence"], ["required_image_quality_excluded"])
        self.assertEqual(receipt["slots"][1]["quality_scope"], "out_of_scope")
        self.assertEqual(receipt["slots"][2]["action"], "remove_image_and_repair_references")
        self.assertFalse(receipt["slots"][2]["repair_applied"])
        self.assertEqual(record, before)

    def test_known_ai_is_distinct_from_quality_exclusion(self):
        config, candidates = policy_fixture(self.root, first_outcome="ai")
        receipt = quality_receipt([candidates[0]["item_id"]], load_quality_policy(config))
        self.assertEqual(receipt["reason_codes"], ["required_image_policy_ai_excluded"])
        self.assertEqual(receipt["slots"][0]["blur_status"], "clear")

    def test_publisher_rejects_missing_forged_or_incomplete_receipts(self):
        config, candidates = policy_fixture(self.root, first_outcome="blur_present")
        state = load_quality_policy(config)
        identity = {"run": {"quality_policy": state["binding"]}}
        ids = [candidates[0]["item_id"]]
        admission = {"decision": "admit", "reason_codes": [], "evidence": {}}
        with self.assertRaisesRegex(ValueError, "receipt missing"):
            validate_quality_admission(ids, admission, identity)
        admission["evidence"]["image_quality"] = quality_receipt(ids, state)
        with self.assertRaisesRegex(ValueError, "not cleared every slot"):
            validate_quality_admission(ids, admission, identity)
        admission["evidence"]["image_quality"]["slots"][0]["blur_status"] = "clear"
        with self.assertRaisesRegex(ValueError, "contradicts"):
            validate_quality_admission(ids, admission, identity)
        admission["evidence"]["image_quality"] = quality_receipt([], state)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            validate_quality_admission(ids, admission, identity)

    def test_pure_text_has_zero_required_visual_slots(self):
        config, _ = policy_fixture(self.root)
        state = load_quality_policy(config)
        admission = {"decision": "admit", "reason_codes": [], "evidence": {"image_quality": quality_receipt([], state)}}
        validate_quality_admission([], admission, {"quality_policy": state["binding"]})

    def test_real_publisher_boundary_requires_receipt_and_reader_revalidates(self):
        fixture = publishing_fixtures.StagedReleaseTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        config, _ = policy_fixture(self.root, first_asset=fixture.ref["sha256"])
        state = load_quality_policy(config)
        record = fixture.record()
        kwargs = {"run_identity": {"quality_policy": state["binding"]}, "phase": "early_mixed"}
        with self.assertRaisesRegex(ValueError, "receipt missing"):
            publishing_fixtures.commit_staged_release(fixture.root / "blocked", [record], fixture.assets,
                {"record": fixture.plan()}, admissions={"record": fixture.admission()}, **kwargs)
        decision = fixture.admission()
        decision["evidence"]["image_quality"] = quality_receipt([fixture.ref["sha256"]], state)
        with self.assertRaisesRegex(ValueError, "cannot replace required AI screening"):
            publishing_fixtures.commit_staged_release(fixture.root / "blocked", [record], fixture.assets,
                {"record": fixture.plan()}, admissions={"record": decision}, **kwargs)
        decision["evidence"].update(image_screening_policy="image_screening.v3", image_screening=[
            {"asset_id": fixture.ref["sha256"], "screening": {"decision": "admit", "reason": "fixture_only"}}])
        release = publishing_fixtures.commit_staged_release(fixture.root / "releases", [record], fixture.assets,
            {"record": fixture.plan()}, admissions={"record": decision}, **kwargs)
        self.assertEqual(publishing_fixtures.validate_staged_release(release["path"])["records_materialized"], 1)
        # CURRENT changes must not change a frozen release's historical policy.
        (self.root / "CURRENT.json").write_text("{}")
        (self.root / "config.json").write_text("{}")
        self.assertEqual(publishing_fixtures.validate_staged_release(release["path"])["records_materialized"], 1)
        with (self.root / "quality_decisions.jsonl.gz").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            publishing_fixtures.validate_staged_release(release["path"])

    def test_aggregate_cannot_import_legacy_admitted_images_without_blur_receipts(self):
        fixture = publishing_fixtures.StagedReleaseTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        record = fixture.record()
        old_release = fixture.publish([record])
        config, _ = policy_fixture(self.root, first_asset=fixture.ref["sha256"])
        state = load_quality_policy(config)
        with self.assertRaisesRegex(ValueError, "receipt missing"):
            publishing_fixtures.build_dataset_release(self.root / "new.json", [old_release["path"]],
                phase="early_mixed", run_identity={"quality_policy": state["binding"]})


class QualityRunTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, outcome, ai_reject=False, scope="reviewed_subset"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = fixture_record()
            config, _ = policy_fixture(root / "policy", first_asset=record["source_image_refs"][0]["sha256"], first_outcome=outcome, scope=scope)
            policy = load_quality_policy(config)
            if outcome == "unknown":
                record["source_image_refs"][0]["sha256"] = "f" * 64
            original = deepcopy(record)
            args = SimpleNamespace(io_workers=1, cached_detector_scores="unused", dedup_history=None,
                output=directory, record_concurrency=2, phase="early_mixed", audit_fraction=0,
                layout="off", publish_dir=None, quality_policy_state=policy)
            assets = {ref["sha256"]: {"asset_id": ref["sha256"], "sha256": ref["sha256"], "short_edge": 128}
                      for ref in record["source_image_refs"]}
            seen, captured = [], {}
            async def inspect(*args, **kwargs):
                return assets, {}, {record["original_sample_id"]: []}
            async def ai(records, assets, reasons, client, output, **kwargs):
                seen.extend(r["original_sample_id"] for r in records if not reasons[r["original_sample_id"]])
                rows = []
                for r in records:
                    if reasons[r["original_sample_id"]]:
                        continue
                    if ai_reject:
                        reasons[r["original_sample_id"]].append("ai_screening_rejected_generated")
                    rows.extend({"asset_id": ref["sha256"], "record_id": r["original_sample_id"],
                                 "screening": {"decision": "reject" if ai_reject else "admit", "reason": "fixture_only"}}
                                for ref in r["source_image_refs"])
                write_jsonl(output / "ai_evidence.jsonl", rows)
                return rows
            async def dedup(*args, **kwargs):
                return {"assets": [], "relations": [], "receipt": {"input_assets": len(assets)}}
            def publish(directory, records, assets, plans, **kwargs):
                captured.update(deepcopy(kwargs))
                return {"committed": True}
            state = SimpleNamespace(candidates=lambda sid: [], rejected_asset=lambda sha: False, screening_rejections=lambda: [])
            with patch("corpus_pipeline.production.staged_run.inspect_records", inspect), \
                 patch("corpus_pipeline.production.staged_run.ai_stage", ai), \
                 patch("corpus_pipeline.production.staged_run.dedup_stage", dedup), \
                 patch("corpus_pipeline.production.staged_run.commit_staged_release", publish):
                summary = await process_batch([record], state, FakeClient(), args, root / "batch",
                                               {"run": {"quality_policy": policy["binding"]}, "batch": 0})
            self.assertEqual(record, original)
            return summary, captured, seen

    async def test_all_images_missing_blur_blocks_before_ai_and_keeps_pending(self):
        summary, captured, seen = await self.exercise("unknown", scope="all_images")
        self.assertEqual(seen, [])
        self.assertEqual(summary["admission_counts"], {"pending": 1})
        self.assertEqual(summary["reason_counts"], {"image_blur_review_missing": 1})

    async def test_default_out_of_scope_unknown_enters_original_ai_stage(self):
        summary, captured, seen = await self.exercise("unknown", ai_reject=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(summary["admission_counts"], {"reject": 1})
        self.assertEqual(summary["reason_counts"], {"ai_screening_rejected_generated": 1})
        decision = next(iter(captured["admissions"].values()))
        self.assertEqual(decision["evidence"]["image_quality"]["slots"][0]["policy_decision"], "not_reviewed")

    async def test_known_blur_rejects_with_evidenced_asset_revocation(self):
        summary, captured, seen = await self.exercise("blur_present")
        self.assertEqual(seen, [])
        self.assertEqual(summary["admission_counts"], {"reject": 1})
        self.assertEqual(captured["revocations"][0]["reason"], "required_image_quality_excluded")
        self.assertTrue(captured["revocations"][0]["decision_sha256"])

    async def test_clear_still_enters_ai_and_ai_can_reject(self):
        summary, captured, seen = await self.exercise("clear", ai_reject=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(summary["admission_counts"], {"reject": 1})
        self.assertEqual(summary["reason_counts"], {"ai_screening_rejected_generated": 1})
        decision = next(iter(captured["admissions"].values()))
        self.assertTrue(decision["evidence"]["image_quality"]["cleared_images_still_require_ai_screening"])


if __name__ == "__main__":
    unittest.main()
