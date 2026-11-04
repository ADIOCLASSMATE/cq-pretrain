"""Offline policy tests; tiny model outputs are fixtures, never corpus evidence."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .frontpass import BASE, POLICY, checked_inference, optimized_configuration, prediction_errors, process_task, record_decision, record_groups, route, run_worker, validate_image_slots
from corpus_pipeline.production.common import digest, file_sha
from corpus_pipeline.production.records import snapshot, read_asset_bytes


THRESHOLDS = {"cf384": .9, "unifd": .95}


def prediction(cf=.1, unifd=.2, status="ok"):
    return {"status": status, "scores": {"cf384": {"status": "ok", "prob": cf},
                                         "unifd": {"status": "ok", "prob": unifd}},
            "sscd": {"status": "ok"}, "original_width": 1024, "original_height": 1024}


def record(origins=("unknown",), category="single_i2t", reason_codes=(), decision="quarantine"):
    return {"source_image_refs": [{"sha256": str(i).zfill(64)} for i in range(len(origins))],
            "normalized": {"category": category, "decision": decision, "reason_codes": list(reason_codes),
                           "provenance": {"images": [{"image_index": i, "origin": origin}
                                                      for i, origin in enumerate(origins)]}}}


def native_fixture(directory):
    root = Path(directory)
    pool = root / "source_pool" / "unit_native"
    pool.mkdir(parents=True)
    (root / "data").mkdir()
    tar_path = pool / "000000.tar"
    images = {}
    with tarfile.open(tar_path, "w") as archive:
        for name, size, color in (("shared", (1024, 1024), "white"), ("alert", (192, 256), "red"),
                                  ("generated", (192, 256), "blue"), ("small", (127, 500), "green"),
                                  ("rendered", (256, 256), "black")):
            buffer = io.BytesIO()
            Image.new("RGB", size, color).save(buffer, format="PNG")
            blob = buffer.getvalue()
            member = name + ".png"
            info = tarfile.TarInfo(member)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
            images[name] = {"member": member, "length": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                            "width": size[0], "height": size[1]}
    with tarfile.open(tar_path, "r") as archive:
        for image in images.values():
            image["offset"] = archive.getmember(image["member"]).offset_data

    def text(value, role=None, turn=None):
        return {"kind": "text", "text": value, "role": role, "turn": turn,
                "image_index": None, "source_member": "annotation.txt"}

    def image(name, index=0, role=None, turn=None):
        return {"kind": "image", "text": None, "role": role, "turn": turn,
                "image_index": index, "source_member": images[name]["member"]}

    definitions = [
        ("caption", ["shared"], [image("shared"), text("Original AI-written caption.")], "programmatic", "ai"),
        ("interleaved", ["shared", "alert"], [text("Before"), image("shared"), text("Between"),
                                                image("alert", 1), text("After")], "mixed", None),
        ("conversation", ["shared"], [image("shared", role="user", turn=0), text("How many?", "user", 0),
                                         text("One.", "assistant", 1), text("What color?", "user", 2),
                                         text("White.", "assistant", 3)], None, None),
        ("text", [], [text("Original AI-written text must be retained.")], None, "ai"),
        ("caption", ["generated"], [image("generated"), text("Known generated image.")], "ai_generated", None),
        ("caption", ["small"], [image("small"), text("Small original image.")], "unknown", None),
        ("caption", ["rendered"], [image("rendered"), text("Traditional rendering.")], "rendered", None),
    ]
    rows = []
    for i, (fmt, names, segments, image_origin, caption_origin) in enumerate(definitions):
        rows.append({"sample_id": str(i).zfill(64), "source": "unit_native", "source_shard": "unit_native/000000.tar",
                     "source_key": str(i), "format": fmt, "modality": "multimodal" if names else "text",
                     "image_count": len(names), "segments": segments, "images": [images[name] for name in names],
                     "image_origin": image_origin, "caption_origin": caption_origin})
    parquet = root / "data/000000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet, row_group_size=2)
    (root / "catalog.json").write_text(json.dumps([{"path": "data/000000.parquet", "rows": len(rows),
        "bytes": parquet.stat().st_size, "source": "unit_native", "source_shard": "unit_native/000000.tar"}]))
    (root / "manifest.json").write_text(json.dumps({"schema_version": "dcvlm.pretrain.v1", "phase": "complete",
                                                   "source_pool": str(root / "source_pool")}))
    (root / "COMPLETE.json").write_text(json.dumps({"validation_passed": True,
                                                   "catalog_sha256": file_sha(root / "catalog.json")}))
    task = {"parquet_path": str(parquet), "row_start": 0, "row_end": len(rows), "source": "unit_native",
            "source_shard": "unit_native/000000.tar", "parquet_fingerprint": {
                "size": parquet.stat().st_size, "mtime_ns": parquet.stat().st_mtime_ns}}
    return rows, images, task


class FixtureResidentWorker:
    identity_sha256 = "unit-test-only-no-real-model"

    def __init__(self, alert_sha):
        self.alert_sha, self.calls = alert_sha, []

    def iter_results(self, inputs, *, models):
        inputs = list(inputs)
        self.calls.append((tuple(models), [item["sha256"] for item in inputs]))
        for asset in inputs:
            raw = read_asset_bytes(asset["original_ref"])
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
            row = {"image_sha256": asset["sha256"], "status": "ok", "scores": {},
                   "original_width": width, "original_height": height, "pixel_sha256": "fixture-pixel-" + asset["sha256"]}
            if min(width, height) < 128:
                row.update(status="too_small", size_status="rejected_too_small")
                yield row
                continue
            for model in models:
                if model == "sscd":
                    vector = np.arange(1, 513, dtype=np.float32)
                    row["sscd"] = {"status": "ok", "embedding": vector / np.linalg.norm(vector)}
                else:
                    row["scores"][model] = {"status": "ok", "prob": .999 if asset["sha256"] == self.alert_sha else .01}
            yield row


class FrontpassTests(unittest.TestCase):
    def test_record_aggregation_bounds_keeps_order_and_oversized_record_whole(self):
        rows = [record(()) for _ in range(7)]
        for index, row in enumerate(rows):
            row['index'] = index
        rows[3]['large_text'] = 'x' * 5000
        groups = list(record_groups(iter(rows), {}, aggregation={
            'target_images': 128, 'max_records': 2, 'max_record_bytes': 1500}))
        self.assertEqual([r['index'] for group in groups for r in group], list(range(7)))
        self.assertTrue(all(len(group) <= 2 for group in groups))
        self.assertEqual(next(group for group in groups if rows[3] in group), [rows[3]])

    def test_record_aggregation_counts_distinct_required_images_not_text_or_known_generation(self):
        rows = [record(('unknown',)), record(('unknown',)), record(('ai_generated',)), record(())]
        rows.append(record(('unknown',)))
        rows[-1]['source_image_refs'][0]['sha256'] = 'new'
        groups = list(record_groups(iter(rows), {}, aggregation={
            'target_images': 2, 'max_records': 10, 'max_record_bytes': 100000}))
        self.assertEqual([len(group) for group in groups], [5])

    def test_record_aggregation_rechecks_task_cache_after_yield(self):
        rows = [record(('unknown',)) for _ in range(4)]
        seen = {}
        groups = record_groups(iter(rows), seen, aggregation={
            'target_images': 1, 'max_records': 4, 'max_record_bytes': 100000})
        first = next(groups)
        self.assertEqual(len(first), 1)
        seen[first[0]['source_image_refs'][0]['sha256']] = {'status': 'ok'}
        self.assertEqual(len(next(groups)), 3)

    def test_streaming_worker_reorder_drop_and_extra_output_are_rejected(self):
        wanted = [{'sha256': 'a'}, {'sha256': 'b'}]
        for batches in ([], [([wanted[1]], [{**prediction(), 'image_sha256': 'b'}])],
                        [([wanted[0]], [{**prediction(), 'image_sha256': 'a'}])]):
            closed = []
            def stream(*args, **kwargs):
                try:
                    yield from batches
                finally:
                    closed.append(True)
            worker = SimpleNamespace(batch_size=2, iter_batches=stream)
            with self.assertRaisesRegex(RuntimeError, 'image coverage'):
                list(checked_inference(worker, wanted, models=('sscd',)))
            self.assertTrue(closed)

    def test_optimized_qualification_requires_all_exact_code_and_execution_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(optimized_configuration(root, dtype='float32', batch_size=32, threads=4))
            receipt = {'success': True, 'dtype': 'float32', 'batch_size': 32, 'threads': 4,
                       'base_inference_code_sha256': file_sha(Path(__file__).with_name('inference.py')),
                       'optimized_inference_code_sha256': file_sha(Path(__file__).with_name('optimized_inference.py')),
                       'frontpass_code_sha256': file_sha(Path(__file__).with_name('frontpass.py')),
                       'aggregation': {'target_images': 128, 'max_records': 1024, 'max_record_bytes': 64 << 20}}
            path = root / 'OPTIMIZED_QUALIFIED.json'
            path.write_text(json.dumps(receipt))
            self.assertEqual(optimized_configuration(root, dtype='float32', batch_size=32, threads=4), receipt['aggregation'])
            for key in ('success', 'base_inference_code_sha256', 'optimized_inference_code_sha256', 'frontpass_code_sha256', 'batch_size', 'threads'):
                bad = {**receipt, key: 'wrong'}
                path.write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'qualification'):
                    optimized_configuration(root, dtype='float32', batch_size=32, threads=4)

    def test_aggregated_frontpass_preserves_native_records_decisions_and_late_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / 'dataset'
            _, images, task = native_fixture(dataset)
            contracts = SimpleNamespace(entries={})
            all_outputs = []
            for name, aggregation in [('reference', None), ('aggregated', {
                    'target_images': 2, 'max_records': 3, 'max_record_bytes': 100000})]:
                claim = {'task': task, 'task_id': name, 'token': 'fixture'}
                worker = FixtureResidentWorker(images['alert']['sha256'])
                summary, _ = process_task(Path(directory) / 'run', claim, worker, snapshot(dataset), {},
                                          contracts, THRESHOLDS, record_batch=1, aggregation=aggregation)
                out = Path(directory) / 'run/artifacts' / name / 'fixture'
                records = pq.read_table(out / 'records.parquet').to_pylist()
                image_rows = pq.read_table(out / 'images.parquet').to_pylist()
                all_outputs.append((records, [(r['asset_id'], r['status'], r['original_width'],
                    r['original_height'], r['pixel_sha256']) for r in image_rows]))
                self.assertEqual(summary['counts']['records'], 7)
                self.assertEqual(summary['api_calls'], 0)
            self.assertEqual(*all_outputs)

    def test_model_forward_failure_aborts_task_instead_of_becoming_completed_pending(self):
        for failed_model in ("cf384", "unifd", "sscd"):
            row = {**prediction(status="incomplete"), "image_sha256": "a"}
            failure = {"status": "error", "error_type": "RuntimeError", "error": "device failed"}
            if failed_model == "sscd":
                row["sscd"] = failure
            else:
                row["scores"][failed_model] = failure
            worker = SimpleNamespace(batch_size=2, iter_results=lambda *args, **kwargs: iter([row]))
            with self.subTest(model=failed_model), self.assertRaisesRegex(RuntimeError, "model forward failed"):
                list(checked_inference(worker, [{"sha256": "a"}], models=("cf384", "unifd", "sscd")))

    def test_missing_entire_storage_batch_aborts_but_individual_bad_images_remain_pending(self):
        def missing(sha):
            return {"status": "error", "image_sha256": sha, "error_stage": "decode", "error_type": "FileNotFoundError"}
        inputs = [{"sha256": "a"}, {"sha256": "b"}]
        worker = SimpleNamespace(batch_size=2, iter_results=lambda *args, **kwargs: iter([missing("a"), missing("b")]))
        with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
            list(checked_inference(worker, inputs, models=("sscd",)))
        worker.iter_results = lambda *args, **kwargs: iter([missing("a"), {**prediction(), "image_sha256": "b"}])
        rows = list(checked_inference(worker, inputs, models=("sscd",)))
        self.assertEqual(rows[0]["error_type"], "FileNotFoundError")
        worker.iter_results = lambda *args, **kwargs: iter([missing("a")])
        self.assertEqual(len(list(checked_inference(worker, inputs[:1], models=("sscd",)))), 1)

    def test_worker_missing_or_reordered_outputs_cannot_complete_task(self):
        for rows in ([], [{**prediction(), "image_sha256": "wrong"}]):
            worker = SimpleNamespace(batch_size=2, iter_results=lambda *args, **kwargs: iter(rows))
            with self.assertRaisesRegex(RuntimeError, "image coverage"):
                list(checked_inference(worker, [{"sha256": "a"}], models=("sscd",)))

    def worker_loop(self, claims, states, *, max_tasks=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snap = {"root": directory, "fingerprint": {"test": "immutable"}}
            plan = {"policy": POLICY, "snapshot_root": directory, "snapshot_fingerprint": snap["fingerprint"],
                    "rules_sha256": digest({}), "thresholds_sha256": file_sha(BASE / "detector_calibration/thresholds.json"),
                    "layout_contracts_sha256": file_sha(BASE / "prepared/config/layout_source_contracts.json")}
            (root / "plan.json").write_text(json.dumps(plan))
            resident = MagicMock()
            resident.__enter__.return_value = SimpleNamespace(identity={"fixture": True}, identity_sha256="fixture")
            with patch("corpus_pipeline.cluster.frontpass.load_rules", return_value={}), \
                 patch("corpus_pipeline.cluster.frontpass.snapshot", return_value=snap), \
                 patch("corpus_pipeline.cluster.inference.ResidentInferenceWorker", return_value=resident), \
                 patch("corpus_pipeline.cluster.queue.claim", side_effect=claims) as claim, \
                 patch("corpus_pipeline.cluster.queue.status", side_effect=states) as status, \
                 patch("corpus_pipeline.cluster.queue.complete") as complete, \
                 patch("corpus_pipeline.cluster.queue.fail") as fail, \
                 patch("corpus_pipeline.cluster.frontpass.process_task", return_value=({"counts": {"records": 2}, "elapsed_seconds": 1}, [])), \
                 patch("corpus_pipeline.cluster.frontpass.time.sleep") as sleep, \
                 patch("corpus_pipeline.cluster.frontpass.random.uniform", return_value=32):
                run_worker(root, device="cpu", max_tasks=max_tasks, queue_backend='local')
                return claim.call_count, status.call_count, sleep.call_args_list, complete.call_count, fail.call_count

    def test_idle_worker_waits_for_pending_or_running_then_only_exits_on_complete(self):
        for active in ("pending", "running"):
            counts = {"pending": 0, "running": 0, "failed": 0, "completed": 0}
            counts[active] = 1
            result = self.worker_loop([None, None], [
                {"all_tasks_completed": False, "counts": counts},
                {"all_tasks_completed": True, "counts": {"pending": 0, "running": 0, "failed": 0, "completed": 1}}])
            self.assertEqual(result[:2], (2, 2))
            self.assertEqual(len(result[2]), 1)
            self.assertEqual(result[2][0].args, (32,))

    def test_terminal_failed_queue_cannot_look_successfully_finished(self):
        with self.assertRaisesRegex(RuntimeError, "terminal failed"):
            self.worker_loop([None], [{"all_tasks_completed": False,
                                      "counts": {"pending": 0, "running": 0, "failed": 1, "completed": 10}}])

    def test_active_worker_does_not_scan_global_status_between_tasks(self):
        claim = {"task_id": "unit-task", "token": "attempt"}
        result = self.worker_loop([claim], [], max_tasks=1)
        self.assertEqual(result, (1, 0, [], 1, 0))

    def test_idle_worker_can_claim_recovered_tail_task_without_early_success(self):
        result = self.worker_loop([None, {"task_id": "unit-task", "token": "reclaimed"}], [
            {"all_tasks_completed": False, "counts": {"pending": 0, "running": 1, "failed": 0, "completed": 10}}], max_tasks=1)
        self.assertEqual(result[:2], (2, 1))
        self.assertEqual(result[3:], (1, 0))

    def test_unknown_and_mixed_low_scores_pass_high_scores_only_queue(self):
        for origin in ("unknown", "mixed", "natural"):
            self.assertEqual(route({"origin": origin}, prediction(), THRESHOLDS), "no_detector_alert")
            self.assertEqual(route({"origin": origin}, prediction(cf=.99), THRESHOLDS), "ai_suspect_pending_api")
            self.assertEqual(route({"origin": origin}, prediction(unifd=.99), THRESHOLDS), "ai_suspect_pending_api")

    def test_source_protection_and_known_generation_take_precedence(self):
        for origin in ("rendered", "programmatic"):
            self.assertEqual(route({"origin": origin}, prediction(cf=1, unifd=1), THRESHOLDS), "protected_source")
        self.assertEqual(route({"origin": "ai_generated"}, {}, THRESHOLDS), "rejected_known_ai_source")
        provenance = {"origin": "ai_edited", "policy_exception": "ai_edit_from_natural_source", "edit_source_origin": "natural"}
        self.assertEqual(route(provenance, prediction(cf=1), THRESHOLDS), "protected_source")
        self.assertEqual(route({"origin": "ai_edited"}, prediction(), THRESHOLDS), "pending_edit_lineage")

    def test_protected_occurrence_ignores_unrelated_detector_failure_but_not_sscd_failure(self):
        row = prediction(status="incomplete")
        row["scores"]["cf384"] = {"status": "error"}
        self.assertEqual(route({"origin": "programmatic"}, row, THRESHOLDS), "protected_source")
        self.assertEqual(route({"origin": "unknown"}, row, THRESHOLDS), "pending_inference_or_integrity_error")
        row["sscd"] = {"status": "error"}
        self.assertEqual(route({"origin": "programmatic"}, row, THRESHOLDS), "pending_inference_or_integrity_error")

    def test_missing_nonfinite_or_invalid_scores_never_pass(self):
        for value in (None, float("nan"), float("inf"), -.1, 1.1, True, "0.1"):
            with self.subTest(value=value):
                self.assertEqual(route({"origin": "unknown"}, prediction(cf=value), THRESHOLDS),
                                 "pending_inference_or_integrity_error")

    def test_nested_model_errors_preserved_without_embedding_payloads(self):
        row = prediction(status="incomplete")
        row["scores"]["cf384"] = {"status": "error", "error_type": "RuntimeError", "error": "fixture model failure"}
        row["sscd"]["embedding"] = np.zeros(512, dtype=np.float32)
        self.assertEqual(prediction_errors(row), {"models": {"cf384": row["scores"]["cf384"]}})
        row["sscd"] = {"status": "error", "error_type": "ValueError", "error": "bad descriptor", "embedding": np.zeros(512)}
        errors = prediction_errors(row)
        self.assertEqual(errors["models"]["sscd"]["error"], "bad descriptor")
        self.assertNotIn("embedding", errors["models"]["sscd"])
        self.assertEqual(prediction_errors({"status": "error", "error_stage": "decode", "error": "bad bytes"}),
                         {"error_stage": "decode", "error": "bad bytes"})

    def test_small_or_corrupt_image_cannot_use_source_protection(self):
        self.assertEqual(route({"origin": "programmatic"}, {"status": "too_small"}, THRESHOLDS), "rejected_too_small")
        self.assertEqual(route({"origin": "programmatic"}, {"status": "error", "error_stage": "decode"}, THRESHOLDS),
                         "pending_inference_or_integrity_error")

    def test_record_all_images_required_and_source_unknown_is_not_a_blocker(self):
        value = record(("mixed", "unknown"), reason_codes=("image_provenance_mixed", "image_provenance_unknown"))
        self.assertEqual(record_decision(value, ["no_detector_alert", "protected_source"]),
                         ("frontpass_pass_pending_global_dedup", []))
        self.assertEqual(record_decision(value, ["no_detector_alert", "ai_suspect_pending_api"])[0], "pending")
        self.assertEqual(record_decision(value, ["no_detector_alert"])[0], "pending")
        self.assertEqual(record_decision(value, ["no_detector_alert", "rejected_too_small"])[0], "rejected")
        value["normalized"]["reason_codes"].append("image_reference_unresolved")
        self.assertEqual(record_decision(value, ["no_detector_alert", "no_detector_alert"])[0], "pending")

    def test_provenance_slot_coverage_must_be_exact(self):
        value = record(("unknown", "mixed"))
        validate_image_slots(value)
        for mutation in ("missing", "duplicate", "boolean_index"):
            malformed = deepcopy(value)
            images = malformed["normalized"]["provenance"]["images"]
            if mutation == "missing":
                images.pop()
            elif mutation == "duplicate":
                images[1]["image_index"] = 0
            else:
                images[1]["image_index"] = True
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "slot coverage"):
                validate_image_slots(malformed)

    def test_native_snapshot_all_slots_types_ai_text_and_pending_queues_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            originals, images, task = native_fixture(dataset)
            source_before = {str(p): file_sha(p) for p in dataset.rglob("*") if p.is_file()}
            worker = FixtureResidentWorker(images["alert"]["sha256"])
            claim = {"task": task, "task_id": "fixture-task", "token": "fixture-attempt"}
            run = Path(directory) / "run"
            contracts = SimpleNamespace(entries={("dcvlm", "unit_native"): {"tag": "sample_to_decide"}})
            with patch("corpus_pipeline.production.api_client.QwenAPIClient.request", side_effect=AssertionError("API forbidden")) as api:
                summary, artifacts = process_task(run, claim, worker, snapshot(dataset), {}, contracts, THRESHOLDS, record_batch=1)
                api.assert_not_called()
            output = run / "artifacts/fixture-task/fixture-attempt"
            records = pq.read_table(output / "records.parquet").to_pylist()
            occurrences = pq.read_table(output / "images.parquet").to_pylist()
            suspects = pq.read_table(output / "suspects.parquet").to_pylist()
            self.assertEqual(len(records), 7)
            self.assertEqual(len(occurrences), 7)
            self.assertEqual([row["category"] for row in records],
                             ["single_i2t", "interleaved", "multi_i2t", "pure_text", "single_i2t", "single_i2t", "single_i2t"])
            self.assertEqual([records[i]["text_origin"] for i in (0, 3)], ["ai", "ai"])
            self.assertTrue(all(row["text_retained"] for row in records))
            self.assertEqual(records[1]["image_ids"], [images["shared"]["sha256"], images["alert"]["sha256"]])
            self.assertEqual(records[1]["status"], "pending")
            self.assertEqual(records[2]["status"], "frontpass_pass_pending_global_dedup")
            self.assertEqual(records[3]["status"], "frontpass_pass_pending_global_dedup")
            self.assertEqual(records[4]["status"], "rejected")
            self.assertEqual(records[5]["status"], "rejected")
            self.assertEqual(suspects[0]["status"], "ai_suspect_pending_api")
            self.assertEqual(len(suspects), 1)
            self.assertEqual(len(records[3]["image_ids"]), 0)
            self.assertTrue(records[0]["recaption_resolution_potential"])
            self.assertEqual(summary["api_calls"], 0)
            self.assertFalse(summary["training_release"])
            self.assertFalse(summary["global_dedup_complete"])
            self.assertTrue(all(file_sha(item["path"]) == item["sha256"] for item in artifacts))
            self.assertEqual(source_before, {str(p): file_sha(p) for p in dataset.rglob("*") if p.is_file()})
            self.assertEqual(pq.read_table(task["parquet_path"]).to_pylist(), originals)
            shared_calls = [models for models, shas in worker.calls if images["shared"]["sha256"] in shas]
            self.assertEqual(shared_calls, [("sscd",), ("cf384", "unifd")])
            self.assertTrue(all(images["generated"]["sha256"] not in shas for _, shas in worker.calls))
            # A shared image has one SSCD descriptor despite multiple records and
            # later detector processing; distinct supervision records remain.
            shared_rows = [r for r in occurrences if r["asset_id"] == images["shared"]["sha256"]]
            self.assertEqual({r["sscd_byte_offset"] for r in shared_rows}, {0})
            self.assertEqual((output / "sscd.f32").stat().st_size, 3 * 512 * 4)


if __name__ == "__main__":
    unittest.main()
