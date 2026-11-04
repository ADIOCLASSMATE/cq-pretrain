"""Release boundary tests; original records are fixtures, not claimed corpus runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import digest
from .reader import iter_training_records
from .staged_publish import build_candidate, commit_staged_release, iter_staged_records, validate_staged_release
from .staged_publish import build_dataset_release, iter_dataset_records, validate_dataset_release


class StagedReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        blob = b"fixture-image-byte-range"
        asset_id = hashlib.sha256(blob).hexdigest()
        path = self.root / "source.tar"
        path.write_bytes(blob)
        self.ref = {"path": str(path), "offset": 0, "length": len(blob), "sha256": asset_id}
        self.assets = {asset_id: {"asset_id": asset_id, "short_edge": 128, "original_ref": self.ref}}
        self.raw_by_path = {}
        self.reader_patch = patch("corpus_pipeline.production.staged_publish.load_original_record", side_effect=self.read)
        self.reader_patch.start()

    def tearDown(self):
        self.reader_patch.stop()
        self.temp.cleanup()

    def read(self, locator):
        return deepcopy(self.raw_by_path[locator["path"]])

    def record(self, key="record", task="single_i2t", semantic="caption", short_only=False):
        is_text = task == "pure_text"
        segments = ([] if is_text else [{"kind": "image", "image_index": 0, "role": "user"}])
        if semantic == "caption" and not short_only:
            segments.append({"kind": "text", "text": "A correct short original."})
            format_name = "caption"
        else:
            segments.extend([{"kind": "text", "role": "user", "text": "Answer briefly."},
                             {"kind": "text", "role": "assistant", "text": "42"}])
            format_name = "conversation"
        raw = {"sample_id": key, "format": format_name, "segments": segments, "image_count": 0 if is_text else 1}
        path = str(self.root / (key + ".parquet"))
        self.raw_by_path[path] = raw
        record = {"original_sample_id": key, "source": "fixture", "source_record_sha256": digest(raw),
                  "raw": deepcopy(raw), "normalized": {"category": task}, "locator": {"path": path, "row_index": 0},
                  "source_image_refs": [] if is_text else [self.ref]}
        if short_only:
            record["caption_contract"] = {"target_segment_index": 2, "allowed_training_lengths": ["short"]}
            record["prepared_semantic_task"] = "caption"
        return record

    def plan(self, action="reuse_original", semantic="caption", phase="early_mixed"):
        return {"action": action, "semantic_task": semantic, "phase": phase}

    def admission(self, decision="admit"):
        return {"decision": decision, "reason_codes": ["fixture"],
                "evidence": {"retention_policy": "native_integrity_v1", "quality_scope": "not_individually_visual_reviewed"}}

    def candidate(self, record, origin="generated", verdict="pass", phase="early_mixed", captions=None, concerns=None):
        from .run import model_context, verification_context
        captions = captions or {"short": "A short caption.", "medium": "A medium caption.", "long": "A long caption."}
        context = model_context(record, phase=phase)
        annotation = {"native_task": record["normalized"]["category"], "semantic_task": "caption",
                      "image_captions": [{"image_index": 0, **captions}], "assistant_revisions": [], "concerns": concerns or [], "layout_candidates": []}
        quality_context = verification_context(record, context, annotation)
        inputs = [{"asset_id": self.ref["sha256"], "original_sha256": self.ref["sha256"], "sent_sha256": self.ref["sha256"],
                   "crop": False, "original_dimensions": [128, 128], "sent_dimensions": [128, 128]}]
        def receipt(role, request_context, parsed):
            identity = {"role": role, "model": "fixture-" + role, "context_sha256": digest(request_context), "assets": [self.ref["sha256"]]}
            return {"status": "ok", "identity": identity, "cache_key": digest(identity), "parsed": parsed,
                    "request_context": request_context, "real_model_inference": True, "image_inputs": inputs}
        quality = receipt("verifier", quality_context, {"verdict": verdict, "issues": [], "reason": "Fixture verification.",
                          "caption_checks": [{"image_index": 0, "variant": variant, "verdict": verdict,
                                              "issues": [], "reason": "Fixture review of " + variant + " caption."}
                                             for variant in ("short", "medium", "long")]})
        generator = receipt("generator", context, annotation)
        return build_candidate(record, origin=origin,
                               captions=captions,
                               provenance={"phase": phase, "model": "fixture", "text_origin": "ai"},
                               quality_receipt={"verdict": verdict, "scope": "per_annotation_visual_review",
                                                "evidence": quality},
                               generator_receipt=generator
                               if origin == "generated" else None)

    def publish(self, records, plans=None, admissions=None, **kwargs):
        return commit_staged_release(self.root / "releases", records, self.assets,
                                     plans or {r["original_sample_id"]: self.plan() for r in records},
                                     admissions=admissions or {r["original_sample_id"]: self.admission() for r in records},
                                     run_identity={"run": "fixture"}, phase=kwargs.pop("phase", "early_mixed"), **kwargs)

    def test_original_and_native_tasks_are_byte_semantically_preserved_without_model_receipts(self):
        records = [self.record(), self.record("qa", semantic="qa"),
                   self.record("text", task="pure_text", semantic="pure_text"),
                   self.record("dialogue", task="multi_i2t", semantic="dialogue")]
        plans = {r["original_sample_id"]: self.plan("reuse_original" if i == 0 else "preserve_native_task",
                                                   ["caption", "qa", "pure_text", "dialogue"][i]) for i, r in enumerate(records)}
        result = self.publish(records, plans=plans)
        values = {row["sample_id"]: row for row in iter_training_records(result["path"])}
        self.assertEqual(len(values), len(records))
        for record in records:
            value = values[record["original_sample_id"]]
            self.assertEqual(value["segments"], record["raw"]["segments"])
            self.assertIsNone(value["selected_candidate_id"])
            self.assertEqual(value["selected_origin"], "original")
            self.assertNotIn("ai_generated_text", value)
        self.assertEqual((Path(result["path"]) / "candidates.jsonl").read_text(), "")

    def test_native_optional_review_uncertainty_survives_training_reader(self):
        record = self.record("qa", semantic="qa")
        admission = self.admission()
        admission["evidence"].update(retained_with_uncertainty=True,
                                   quality_flags=["native_supervision_review_uncertain"])
        result = self.publish([record], plans={"qa": self.plan("preserve_native_task", "qa")},
                              admissions={"qa": admission})
        value = next(iter_training_records(result["path"]))
        self.assertEqual(value["segments"], record["raw"]["segments"])
        self.assertTrue(value["retained_with_uncertainty"])
        self.assertEqual(value["quality_status"], "uncertain")
        self.assertEqual(value["quality_flags"], ["native_supervision_review_uncertain"])

    def test_missing_retention_evidence_is_not_a_fake_quality_pass(self):
        record = self.record()
        with self.assertRaisesRegex(ValueError, "retention policy"):
            self.publish([record], admissions={"record": {"decision": "admit", "evidence": {"visual": "unknown"}}})

    def test_new_screening_policy_cannot_publish_missing_image_receipt(self):
        record = self.record()
        admission = self.admission()
        admission["evidence"].update(image_screening_policy="image_screening.v3", image_screening=[])
        with self.assertRaisesRegex(ValueError, "every required image slot"):
            self.publish([record], admissions={"record": admission})

    def test_retained_ai_caption_keeps_actual_source_provenance_without_new_generator(self):
        record = self.record()
        provenance = {"origin": "ai", "ai_generated": True, "evidence": [{"kind": "source_metadata", "generator": "upstream"}]}
        result = self.publish([record], plans={"record": {**self.plan(), "caption_origin": "ai", "original_text_provenance": provenance}})
        row = next(iter_staged_records(result["path"]))
        self.assertEqual(row["text_provenance"]["origin"], "ai")
        self.assertIs(row["text_provenance"]["ai_generated"], True)
        self.assertEqual(row["text_provenance"]["evidence"], provenance["evidence"])
        self.assertEqual(row["selected_origin"], "original")
        self.assertIsNone(row["selected_candidate_id"])

    def test_generated_requires_complete_captions_and_real_receipt_shape(self):
        record = self.record()
        with self.assertRaisesRegex(ValueError, "complete triplet"):
            self.candidate(record, captions={"short": "Only short."})
        with self.assertRaisesRegex(ValueError, "generator receipt"):
            build_candidate(record, origin="generated", captions={name: name for name in ("short", "medium", "long")},
                            provenance={"model": "fixture"}, quality_receipt={})

    def test_uncertain_candidate_keeps_highres_denominator_pending(self):
        first, second = self.record("first"), self.record("second")
        candidates = [self.candidate(first, phase="highres_1mp"), self.candidate(second, verdict="uncertain", phase="highres_1mp")]
        result = self.publish([first, second], phase="highres_1mp", candidates=candidates,
                              plans={key: self.plan("complete_highres_caption", phase="highres_1mp") for key in ("first", "second")})
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["coverage"]["highres_pool_records"], 2)
        self.assertEqual(result["coverage"]["highres_pending_records"], 1)
        self.assertEqual(result["coverage"]["record_coverage"], .5)
        self.assertEqual(len(list(iter_staged_records(result["path"]))), 1)

    def test_early_triplet_is_not_implicitly_qualified_for_highres(self):
        record = self.record()
        result = self.publish([record], phase="highres_1mp", candidates=[self.candidate(record)],
                              plans={"record": self.plan("complete_highres_caption", phase="highres_1mp")})
        self.assertEqual(result["records"], 0)
        self.assertEqual(result["coverage"]["highres_pending_records"], 1)

    def test_caption_instruction_and_original_user_turn_are_immutable(self):
        record = self.record(short_only=True)
        result = self.publish([record], candidates=[self.candidate(record)],
                              plans={"record": self.plan("regenerate_verified_failure")})
        value = next(iter_staged_records(result["path"]))
        self.assertEqual(value["caption_length"], "short")
        self.assertEqual(value["segments"][:2], record["raw"]["segments"][:2])
        with self.assertRaisesRegex(ValueError, "violates native instruction"):
            list(iter_staged_records(result["path"], caption_length="long"))

    def test_caption_candidate_cannot_rewrite_dialogue_or_qa(self):
        record = self.record(semantic="qa")
        with self.assertRaisesRegex(ValueError, "context mismatch|noncaption"):
            self.publish([record], candidates=[self.candidate(record)], plans={"record": self.plan("preserve_native_task", "qa")})

    def test_phase_release_is_immutable_and_idempotent(self):
        record = self.record()
        first = self.publish([record], release_id="frozen")
        second = self.publish([record], release_id="frozen")
        self.assertEqual(first["path"], second["path"])
        self.assertTrue(second["reused_existing_release"])
        with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
            self.publish([record], release_id="frozen", admissions={"record": self.admission("pending")})

    def test_candidates_do_not_multiply_record_exposure_and_choices_are_deterministic(self):
        record = self.record()
        candidates = [self.candidate(record), self.candidate(record, "existing")]
        result = self.publish([record], candidates=candidates)
        first = list(iter_staged_records(result["path"], epoch_seed=7))
        second = list(iter_staged_records(result["path"], epoch_seed=7))
        self.assertEqual(len(first), 1)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["asset_exposure_multiplier"], 1)

    def test_runtime_revocation_removes_native_or_generated_without_mutation(self):
        record = self.record()
        candidate = self.candidate(record)
        result = self.publish([record], candidates=[candidate], plans={"record": self.plan("regenerate_verified_failure")})
        original_sha = hashlib.sha256((Path(result["path"]) / "manifest.json").read_bytes()).hexdigest()
        self.assertEqual(list(iter_staged_records(result["path"], revocations=[
            {"kind": "candidate", "id": candidate["candidate_id"], "reason": "later discovered hallucination"}])), [])
        self.assertEqual(list(iter_staged_records(result["path"], revocations=[
            {"kind": "asset", "id": self.ref["sha256"], "reason": "source withdrawn"}])), [])
        self.assertEqual(hashlib.sha256((Path(result["path"]) / "manifest.json").read_bytes()).hexdigest(), original_sha)

    def test_receipt_tampering_and_context_swaps_are_rejected(self):
        record = self.record()
        candidate = self.candidate(record)
        candidate["quality_receipt"]["receipt"]["verdict"] = "uncertain"
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.publish([record], candidates=[candidate])
        record2 = self.record("other")
        candidate = self.candidate(record2)
        with self.assertRaisesRegex(ValueError, "outside release"):
            self.publish([record], candidates=[candidate])

    def test_original_modification_is_detected_on_read(self):
        record = self.record()
        result = self.publish([record])
        self.raw_by_path[record["locator"]["path"]]["segments"][-1]["text"] = "Changed"
        with self.assertRaisesRegex(ValueError, "original record changed"):
            validate_staged_release(result["path"])

    def test_manifest_and_artifact_tampering_is_detected(self):
        record = self.record()
        result = self.publish([record])
        manifest_path = Path(result["path"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["phase"] = "highres_1mp"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "completion checksum"):
            list(iter_staged_records(result["path"]))

    def test_model_receipts_must_bind_actual_caption_context_assets_and_phase(self):
        record = self.record()
        candidate = self.candidate(record)
        quality = candidate["quality_receipt"]["receipt"]
        generator = candidate["generator_receipt"]["receipt"]
        with self.assertRaisesRegex(ValueError, "differs from model-reviewed"):
            build_candidate(record, origin="generated", captions={**candidate["captions"], "long": "Unreviewed altered long caption."},
                            provenance=candidate["provenance"], quality_receipt=quality, generator_receipt=generator)
        bad = deepcopy(generator)
        bad["identity"]["assets"] = ["wrong-image"]
        bad["cache_key"] = digest(bad["identity"])
        with self.assertRaisesRegex(ValueError, "role/assets/cache"):
            build_candidate(record, origin="generated", captions=candidate["captions"], provenance=candidate["provenance"],
                            quality_receipt=quality, generator_receipt=bad)
        relabel = build_candidate(record, origin="generated", captions=candidate["captions"],
                                  provenance={"phase": "highres_1mp"}, quality_receipt=quality, generator_receipt=generator)
        result = self.publish([record], phase="highres_1mp", candidates=[relabel],
                              plans={"record": self.plan("complete_highres_caption", phase="highres_1mp")})
        self.assertEqual(result["records"], 0)

    def test_global_pass_without_all_variant_checks_cannot_qualify_staged_captions(self):
        for phase in ("early_mixed", "highres_1mp"):
            record = self.record()
            candidate = self.candidate(record, phase=phase)
            for missing, short_only in ((True, False), (False, True)):
                quality = deepcopy(candidate["quality_receipt"]["receipt"])
                if missing:
                    del quality["evidence"]["parsed"]["caption_checks"]
                if short_only:
                    quality["evidence"]["parsed"]["caption_checks"] = quality["evidence"]["parsed"]["caption_checks"][:1]
                quality["evidence"]["parsed"]["reason"] = "The original short caption matches the image."
                with self.assertRaises(ValueError):
                    build_candidate(record, origin="generated", captions=candidate["captions"],
                                    provenance=candidate["provenance"], quality_receipt=quality,
                                    generator_receipt=candidate["generator_receipt"]["receipt"])

    def test_early_uncertain_candidate_is_retained_with_actual_verdict_but_never_highres_coverage(self):
        for origin in ("generated", "existing"):
            record = self.record(origin)
            candidate = self.candidate(record, origin=origin, verdict="uncertain", phase="highres_1mp")
            self.assertEqual(candidate["quality_status"], "uncertain")
            self.assertTrue(candidate["retained_with_uncertainty"])
            early = self.publish([record], candidates=[candidate], plans={origin: self.plan("reuse_annotation")})
            row = next(iter_staged_records(early["path"]))
            self.assertEqual(row["quality_status"], "uncertain")
            self.assertTrue(row["retained_with_uncertainty"])
            self.assertEqual(row["selected_origin"], origin)
            late = self.publish([record], phase="highres_1mp", candidates=[candidate],
                                plans={origin: self.plan("reuse_annotation", phase="highres_1mp")})
            self.assertEqual(late["records"], 0)
            self.assertEqual(late["coverage"]["highres_pending_records"], 1)

    def test_early_pass_with_generator_concerns_retains_receipt_and_flags(self):
        record = self.record()
        candidate = self.candidate(record, concerns=["The material is not clear."])
        self.assertEqual(candidate["quality_status"], "pass")
        self.assertEqual(candidate["generator_receipt"]["receipt"]["parsed"]["concerns"], ["The material is not clear."])
        result = self.publish([record], candidates=[candidate], plans={"record": self.plan("generate_missing")})
        row = next(iter_staged_records(result["path"]))
        self.assertTrue(row["retained_with_uncertainty"])
        self.assertIn("GENERATOR_CONCERNS_RETAINED", row["quality_flags"])
        with self.assertRaisesRegex(ValueError, "only be retained in explicit early"):
            self.candidate(record, phase="highres_1mp", concerns=["The material is not clear."])

    def test_early_uncertainty_does_not_admit_known_failed_variant(self):
        record = self.record()
        candidate = self.candidate(record, verdict="uncertain")
        quality = deepcopy(candidate["quality_receipt"]["receipt"])
        quality["evidence"]["parsed"]["caption_checks"][-1]["verdict"] = "fail"
        candidate = build_candidate(record, origin="generated", captions=candidate["captions"],
                                    provenance=candidate["provenance"], quality_receipt=quality,
                                    generator_receipt=candidate["generator_receipt"]["receipt"])
        result = self.publish([record], candidates=[candidate], plans={"record": self.plan("generate_missing")})
        self.assertEqual(result["records"], 0)

    def test_original_uncertainty_flags_survive_native_materialization(self):
        record = self.record()
        plan = {**self.plan(), "retained_with_uncertainty": True, "quality_flags": ["SOURCE_CAPTION_QUALITY_UNCERTAIN_RETAINED"],
                "uncertain_candidate_refs": ["original-candidate"]}
        result = self.publish([record], plans={"record": plan})
        row = next(iter_staged_records(result["path"]))
        self.assertEqual(row["segments"], record["raw"]["segments"])
        self.assertTrue(row["retained_with_uncertainty"])
        self.assertEqual(row["quality_flags"], plan["quality_flags"])
        self.assertEqual(row["text_provenance"]["uncertain_candidate_refs"], ["original-candidate"])

    def test_aggregate_streams_one_latest_record_and_pending_shadows_older_admission(self):
        first, second = self.record("first"), self.record("second")
        part1 = self.publish([first, second])
        dataset1 = build_dataset_release(self.root / "dataset1.json", [part1["path"]], phase="early_mixed", run_identity={"run": "one"})
        self.assertEqual(len(list(iter_training_records(dataset1["path"]))), 2)
        part2 = self.publish([first], admissions={"first": self.admission("pending")})
        dataset2 = build_dataset_release(self.root / "dataset2.json", [part2["path"]], phase="early_mixed",
                                         run_identity={"run": "two"}, base_release=dataset1["path"])
        self.assertEqual([row["sample_id"] for row in iter_dataset_records(dataset2["path"])], ["second"])
        self.assertEqual(dataset2["coverage"]["input_records"], 2)
        self.assertEqual(dataset2["coverage"]["pending_or_rejected_records"], 1)
        self.assertEqual(len(list(iter_training_records(dataset1["path"]))), 2)
        self.assertTrue(validate_dataset_release(dataset2["path"], materialize=True)["success"])

    def test_aggregate_is_immutable_idempotent_and_phase_specific(self):
        record = self.record()
        part = self.publish([record])
        args = {"phase": "early_mixed", "run_identity": {"run": "same"}}
        first = build_dataset_release(self.root / "dataset.json", [part["path"]], **args)
        second = build_dataset_release(self.root / "dataset.json", [part["path"]], **args)
        self.assertTrue(second["reused_existing_release"])
        with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
            build_dataset_release(self.root / "dataset.json", [part["path"]], phase="early_mixed", run_identity={"run": "new"})
        with self.assertRaisesRegex(ValueError, "same phase"):
            build_dataset_release(self.root / "late.json", [], phase="highres_1mp", run_identity={"run": "late"}, base_release=first["path"])

    def test_aggregate_revocation_part_removes_historical_record(self):
        record = self.record()
        first = self.publish([record])
        revoke = self.publish([], revocations=[{"kind": "record", "id": "record", "reason": "review withdrawal"}])
        result = build_dataset_release(self.root / "dataset.json", [first["path"], revoke["path"]],
                                      phase="early_mixed", run_identity={"run": "revoked"})
        self.assertEqual(result["records"], 0)
        self.assertEqual(result["coverage"]["input_records"], 1)
        self.assertEqual(list(iter_dataset_records(result["path"])), [])
        self.assertEqual(len(list(iter_staged_records(first["path"]))), 1)

    def test_aggregate_highres_pending_denominator_and_index_checksum(self):
        first, second = self.record("first"), self.record("second")
        part1 = self.publish([first], phase="highres_1mp", candidates=[self.candidate(first, phase="highres_1mp")],
                             plans={"first": self.plan("complete_highres_caption", phase="highres_1mp")})
        part2 = self.publish([second], phase="highres_1mp", candidates=[self.candidate(second, verdict="uncertain", phase="highres_1mp")],
                             plans={"second": self.plan("complete_highres_caption", phase="highres_1mp")})
        result = build_dataset_release(self.root / "dataset.json", [part1["path"], part2["path"]],
                                      phase="highres_1mp", run_identity={"run": "late"})
        self.assertEqual(result["coverage"]["record_coverage"], .5)
        self.assertEqual(validate_dataset_release(result["path"], materialize=True)["records_materialized"], 1)
        manifest = json.loads(Path(result["path"]).read_text())
        with Path(manifest["index_path"]).open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "index checksum"):
            list(iter_dataset_records(result["path"]))


if __name__ == "__main__":
    unittest.main()
