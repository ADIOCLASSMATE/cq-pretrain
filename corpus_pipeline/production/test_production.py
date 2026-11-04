from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from .client import image_payload, parse_json
from .records import inspect_asset, read_asset_bytes, prepared_caption_contract
from .validation import annotation_validator, layout_validator, quality_validator
from .common import digest
from .publish import commit_release, validate_release, release_contents
from .ai import route_asset, frozen_detector_identity, equivalent_paid_result
from .client import DynamicClient, endpoint_lease_valid, ModelStageError
from .supervise import ready_generation, failure_summary, same_qualified_endpoints


def asset_fixture(directory, size=(128, 140), rgba=False):
    buffer = io.BytesIO()
    Image.new("RGBA" if rgba else "RGB", size, (255, 0, 0, 0) if rgba else "white").save(buffer, "PNG")
    blob = buffer.getvalue()
    path = Path(directory) / "test.tar"
    path.write_bytes(b"prefix" + blob + b"suffix")
    return {"path": str(path), "offset": 6, "length": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}, blob


class ProductionTests(unittest.TestCase):
    def test_manual_review_can_only_quarantine_and_preserves_original_model_result(self):
        from .review import apply_admission_override
        original = {"publishable": True, "reason_codes": [], "annotation": {"cache_key": "executed-original"}}
        rejected = apply_admission_override(original, {"action": "quarantine", "reason_code": "unverifiable_identity"})
        self.assertFalse(rejected["publishable"])
        self.assertTrue(original["publishable"])
        self.assertEqual(rejected["annotation"], original["annotation"])
        with self.assertRaisesRegex(ValueError, "cannot approve"):
            apply_admission_override(original, {"action": "accept"})

    def test_production_rejects_qualification_only_registry(self):
        from .common import require_production_registry
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps({"qualification_only": True, "endpoints": []}))
            with self.assertRaisesRegex(ValueError, "qualification-only"):
                require_production_registry(path)
            self.assertIsNone(ready_generation(path))

    def test_dialogue_indices_count_images_and_empty_delta_keeps_original_answer(self):
        from .run import model_context, verification_context
        record = {"source_image_refs": [{}], "normalized": {"category": "multi_i2t"}, "raw": {
            "format": "conversation", "segments": [{"kind": "image", "role": "user"},
                {"kind": "text", "role": "user", "text": "How many?"},
                {"kind": "text", "role": "assistant", "text": "2"},
                {"kind": "text", "role": "user", "text": "And adding two?"},
                {"kind": "text", "role": "assistant", "text": "4"}]}}
        context = model_context(record)
        self.assertEqual(context["editable_assistant_segment_indices"], [2, 4])
        self.assertEqual(context["segments"][2]["segment_index"], 2)
        self.assertNotIn("segment_index", record["raw"]["segments"][2])
        verified = verification_context(record, context, {"assistant_revisions": []})
        self.assertEqual([target["text"] for target in verified["effective_assistant_targets"]], ["2", "4"])
        self.assertTrue(all(target["value_origin"] == "retained_original_source_target" for target in verified["effective_assistant_targets"]))

    def test_supervisor_lease_renewal_does_not_retry_but_new_boot_can(self):
        import time
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "endpoints.json"
            entries = [{"role": role, "ready": True, "model": role, "base_url": "https://fixture.invalid/v1",
                        "backend": "inspire_serving", "runtime_revision": "verified-v1", "expected_boots": [role + "-a"],
                        "incarnation": role + "-a", "ready_expires_at": time.time() + 30} for role in ("generator", "verifier")]
            path.write_text(json.dumps({"registry_generation": 1, "endpoints": entries}))
            first = ready_generation(path)
            for entry in entries:
                entry["ready_expires_at"] += 10
            path.write_text(json.dumps({"registry_generation": 100, "endpoints": entries}))
            renewed = ready_generation(path)
            self.assertTrue(same_qualified_endpoints(first, renewed))
            entries[0].update(incarnation="generator-b", expected_boots=["generator-b"])
            path.write_text(json.dumps({"registry_generation": 101, "endpoints": entries}))
            self.assertFalse(same_qualified_endpoints(first, ready_generation(path)))

    def test_native_serving_requires_fresh_readiness_lease_and_qualified_boot(self):
        import time
        endpoint = {"backend": "inspire_serving", "expected_boots": ["boot-a"], "ready_expires_at": time.time() + 30}
        self.assertTrue(endpoint_lease_valid(endpoint))
        self.assertFalse(endpoint_lease_valid({**endpoint, "ready_expires_at": time.time() - 1}))
        self.assertFalse(endpoint_lease_valid({**endpoint, "expected_boots": []}))
        self.assertFalse(endpoint_lease_valid({**endpoint, "ready_expires_at": float("inf")}))

    def test_paid_reuse_requires_same_request_and_actual_sent_pixels(self):
        identity = {"image_sha256": "a", "model": "kimi-k3", "prompt_sha256": "prompt", "max_tokens": 5000,
                    "thinking_budget": 2400, "preprocessing": "fixture", "processing_code_identity": {"transport": "new"}}
        metadata = {"sent_sha256": "pixels", "crop": False}
        previous = {"status": "ok", "source_certification": False, "automatic_policy_override": False,
                    "identity": {**identity, "processing_code_identity": {"transport": "old"}},
                    "sent_sha256": "pixels", "image_preprocessing": metadata,
                    "parsed": {"image_form": "photo", "assessment": "uncertain", "confidence": "low",
                               "image_description": "Fixture photo", "visible_evidence": [], "counterevidence": [],
                               "reason": "Fixture unknown origin", "needs_provenance": True,
                               "edit_lineage": "not_established_from_pixels"}}
        self.assertTrue(equivalent_paid_result(previous, identity, metadata))
        self.assertFalse(equivalent_paid_result(previous, {**identity, "prompt_sha256": "different"}, metadata))
        self.assertFalse(equivalent_paid_result(previous, identity, {**metadata, "sent_sha256": "different"}))

    def test_supervisor_requires_both_verified_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "endpoints.json"
            entries = [{"role": role, "ready": True, "base_url": "https://fixture.invalid/v1",
                        "runtime_revision": "verified-v1", "model": role} for role in ("generator", "verifier")]
            path.write_text(json.dumps({"registry_generation": 1, "endpoints": entries[:1]}))
            self.assertIsNone(ready_generation(path))
            entries[1]["runtime_revision"] = "pending-container-version-report"
            path.write_text(json.dumps({"registry_generation": 2, "endpoints": entries}))
            self.assertIsNone(ready_generation(path))
            entries[1]["runtime_revision"] = "verified-v1"
            path.write_text(json.dumps({"registry_generation": 3, "endpoints": entries}))
            self.assertEqual(ready_generation(path)["registry_generation"], 3)

    def test_supervisor_does_not_loop_schema_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "attempt_records/a").mkdir(parents=True)
            (root / "summary.json").write_text(json.dumps({"attempt_id": "a", "asset_failures": 0}))
            (root / "pair_reviews.jsonl").write_text("")
            result = root / "attempt_records/a/record.json"
            result.write_text(json.dumps({"status": "model_stage_failed", "retryable_after_endpoint_change": False}))
            self.assertFalse(failure_summary(root)["retryable_after_endpoint_change"])
            result.write_text(json.dumps({"status": "model_stage_failed", "retryable_after_endpoint_change": True}))
            self.assertTrue(failure_summary(root)["retryable_after_endpoint_change"])

    def test_size_gate_is_global_and_boundary_inclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, _ = asset_fixture(directory, (128, 140))
            self.assertEqual(inspect_asset(ref, "source", "ocr")["size_decision"], "pass")
            ref, _ = asset_fixture(directory, (127, 999))
            self.assertEqual(inspect_asset(ref, "source", "ocr")["size_decision"], "reject_short_edge_lt_128")

    def test_original_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, _ = asset_fixture(directory)
            ref["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                read_asset_bytes(ref)

    def test_image_payload_keeps_original_png_and_white_composites_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, blob = asset_fixture(directory)
            asset = inspect_asset(ref, "source", "document")
            _, meta = image_payload(asset)
            self.assertEqual(meta["sent_sha256"], hashlib.sha256(blob).hexdigest())
            self.assertEqual(meta["transform"], "original_bytes_unchanged")
            ref, _ = asset_fixture(directory, rgba=True)
            asset = inspect_asset(ref, "source", "document")
            payload, meta = image_payload(asset)
            import base64
            result = Image.open(io.BytesIO(base64.b64decode(payload["image_url"]["url"].split(",", 1)[1])))
            self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(meta["alpha_background"], "white")

    def test_qa_annotation_cannot_change_user_segment(self):
        record = {"source_image_refs": [{}], "normalized": {"category": "single_i2t"},
                  "raw": {"format": "conversation", "segments": [
                      {"kind": "text", "role": "user"}, {"kind": "text", "role": "assistant"}]}}
        value = {"native_task": "single_i2t", "semantic_task": "qa", "image_captions": [],
                 "assistant_revisions": [{"segment_index": 0, "text": "changed", "reason": "test"}],
                 "concerns": [], "layout_candidates": []}
        with self.assertRaisesRegex(ValueError, "user"):
            annotation_validator(record)(value)
        value["assistant_revisions"][0]["segment_index"] = 1
        annotation_validator(record)(value)
        value["semantic_task"] = "caption"
        with self.assertRaisesRegex(ValueError, "semantic task"):
            annotation_validator(record)(value)
        value["semantic_task"] = "qa"
        value["native_task"] = "single_t2i"
        with self.assertRaisesRegex(ValueError, "native task"):
            annotation_validator(record)(value)

    def test_multiturn_and_pure_text_structure_preserved(self):
        record = {"source_image_refs": [], "normalized": {"category": "pure_text"},
                  "raw": {"format": "text_conversation", "segments": []}}
        value = {"native_task": "pure_text", "semantic_task": "pure_text", "image_captions": [],
                 "assistant_revisions": [], "concerns": [], "layout_candidates": []}
        annotation_validator(record)(value)
        record["normalized"]["category"] = "multi_i2t"; value["native_task"] = "multi_i2t"
        with self.assertRaisesRegex(ValueError, "semantic task"):
            annotation_validator(record)(value)

    def test_external_detector_cache_requires_frozen_artifact_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = Path(directory) / "scores.jsonl"
            scores.write_text('{"image_sha256":"fixture","scores":{}}\n')
            with self.assertRaisesRegex(ValueError, "unrecognized frozen"):
                frozen_detector_identity(scores)

    def test_caption_conversation_requires_exact_registered_source_and_instruction(self):
        raw = {"source": "coco_caption", "image_count": 1, "segments": [
            {"kind": "text", "role": "user", "text": "Write a terse but informative summary of the picture."},
            {"kind": "text", "role": "assistant", "text": "A visible cat."}]}
        contract = prepared_caption_contract(raw, "single_i2t")
        self.assertEqual(contract["allowed_training_lengths"], ["short"])
        self.assertEqual(contract["target_segment_index"], 1)
        raw["segments"][0]["text"] = "How many cats are visible?"
        self.assertIsNone(prepared_caption_contract(raw, "single_i2t"))
        raw["segments"][0]["text"] = "Summarize the visual content of the image."
        raw["source"] = "unknown"
        self.assertIsNone(prepared_caption_contract(raw, "single_i2t"))

    def test_layout_coordinates_and_major_quality_error_cannot_pass(self):
        value = {"image_index": 0, "coordinate_system": "normalized_0_1000", "summary": "A box",
                 "elements": [{"id": "e1", "kind": "region", "text": None, "bbox": [0, 0, 1000, 1000], "description": "region"}]}
        layout_validator(0)(value)
        value["elements"][0]["bbox"] = [0, 0, 1001, 1000]
        with self.assertRaisesRegex(ValueError, "bounds"):
            layout_validator(0)(value)
        with self.assertRaisesRegex(ValueError, "major"):
            quality_validator({"verdict": "pass", "issues": [{"field": "OCR", "severity": "major", "reason": "wrong text"}], "reason": "test"})

    def test_response_parser_rejects_nonfinite_and_multiple_objects(self):
        self.assertEqual(parse_json('```json\n{"a":1}\n```'), {"a": 1})
        with self.assertRaises(ValueError):
            parse_json('{"a":NaN}')
        with self.assertRaises(ValueError):
            parse_json('{"a":1}{"b":2}')

    def test_production_detector_route_does_not_invent_evaluation_split(self):
        asset = {"sha256": "a" * 64, "asset_id": "a" * 64}
        prediction = {"scores": {"cf384": {"status": "ok", "prob": .99}, "unifd": {"status": "ok", "prob": .1}}}
        result = route_asset(asset, {"origin": "unknown"}, prediction)
        self.assertTrue(result["requires_paid_review"])
        self.assertIsNone(result["review"]["split"])
        self.assertEqual(result["review"]["scope"], "production_canary_not_calibration_or_holdout")
        self.assertFalse(route_asset(asset, {"origin": "programmatic"}, prediction)["requires_paid_review"])

    def test_incremental_release_reads_parent_chain_and_resume_is_idempotent(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raws = [{"sample_id": name, "format": "text_conversation", "image_count": 0,
                     "segments": [{"kind": "text", "role": "assistant", "text": name}]} for name in ("first", "second")]
            parquet = root / "fixture.parquet"
            pq.write_table(pa.Table.from_pylist(raws), parquet)
            annotation = {"native_task": "pure_text", "semantic_task": "pure_text", "image_captions": [],
                          "assistant_revisions": [], "concerns": [], "layout_candidates": []}
            processed = {raw["sample_id"]: {"publishable": True, "annotation": {"parsed": annotation,
                         "cache_key": raw["sample_id"], "identity": {"fixture_only": True}},
                         "quality": {"parsed": {"verdict": "pass", "issues": [], "reason": "fixture"},
                                     "identity": {"fixture_only": True}, "cache_key": raw["sample_id"] + ".qc"}}
                         for raw in raws}
            records = [{"original_sample_id": raw["sample_id"], "source": "fixture", "source_image_refs": [],
                        "source_record_sha256": digest(raw), "locator": {"path": str(parquet), "row_index": index},
                        "normalized": {"category": "pure_text"}} for index, raw in enumerate(raws)]
            first = commit_release(root / "releases", [records[0]], {}, processed, {"assets": []}, run_identity={"batch": 1})
            second = commit_release(root / "releases", [records[1]], {}, processed, {"assets": []}, run_identity={"batch": 2})
            checked = validate_release(second["path"])
            self.assertEqual(checked["records_materialized"], 2)
            self.assertEqual(checked["release_chain_length"], 2)
            again = commit_release(root / "releases", [records[1]], {}, processed, {"assets": []}, run_identity={"batch": 2})
            self.assertTrue(again["reused_existing_release"])
            self.assertEqual(again["release_id"], second["release_id"])


class DynamicClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_visual_review_requires_new_annotation_and_independent_qc(self):
        from .run import annotate_record
        record = {"source_image_refs": [], "normalized": {"category": "pure_text"},
                  "raw": {"format": "text_conversation", "segments": []},
                  "quality_review_request": {"issues": ["fixture factual concern"]}}
        calls = []
        original = {"parsed": {"concerns": [], "layout_candidates": [], "assistant_revisions": []}, "cache_key": "original"}
        revised = {"parsed": {"concerns": [], "layout_candidates": [], "assistant_revisions": []}, "cache_key": "revised"}
        class FixtureClient:
            async def request(self, role, stage, *args, **kwargs):
                calls.append((role, stage))
                if stage == "typed_annotation.v1": return original
                if stage == "typed_annotation_adjudication.v1": return revised
                return {"parsed": {"verdict": "uncertain" if stage == "typed_adjudication_quality.v1" else "pass"}}
        result = await annotate_record(record, {}, FixtureClient(), allow_layout=False)
        self.assertFalse(result["annotation_quality_pass"])
        self.assertEqual(result["pre_adjudication_annotation"]["cache_key"], "original")
        self.assertEqual(result["annotation"]["cache_key"], "revised")
        self.assertIn(("verifier", "typed_adjudication_quality.v1"), calls)

    async def test_qualified_processor_override_sent_and_cache_bound_without_altering_original_size(self):
        import httpx
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "endpoints.json"
            ref, _ = asset_fixture(directory, (128, 140))
            asset = inspect_asset(ref, "source", "document")
            entry = {"role": "generator", "model": "fixture", "protocol": "openai", "ready": True,
                     "base_url": "https://fixture.invalid/v1", "supports_images": True,
                     "request_overrides": {"mm_processor_kwargs": {"min_pixels": 1048576, "max_pixels": 1048576}}}
            atomic_json(registry, {"endpoints": [entry]})
            requests = []
            def respond(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={"model": "fixture", "choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}]})
            client = DynamicClient(registry, root / "cache", wait_ready=1)
            await client.client.aclose()
            client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            first = await client.request("generator", "fixture", {}, [asset], "Return JSON", lambda value: None)
            self.assertEqual(requests[0]["mm_processor_kwargs"], {"min_pixels": 1048576, "max_pixels": 1048576})
            self.assertEqual(tuple(first["image_inputs"][0]["original_dimensions"]), (128, 140))
            self.assertEqual(tuple(first["image_inputs"][0]["sent_dimensions"]), (128, 140))
            entry["request_overrides"]["mm_processor_kwargs"]["min_pixels"] = 262144
            atomic_json(registry, {"endpoints": [entry]})
            second = await client.request("generator", "fixture", {}, [asset], "Return JSON", lambda value: None)
            self.assertEqual(len(requests), 2)
            self.assertNotEqual(first["cache_key"], second["cache_key"])
            self.assertFalse(second["cache_hit"])
            await client.close()

    async def test_serving_inf_api_key_uses_bearer_without_changing_model_identity(self):
        import os
        import time
        import httpx
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "endpoints.json"
            atomic_json(registry, {"registry_generation": 7, "endpoints": [{"role": "generator", "model": "fixture",
                "protocol": "openai", "ready": True, "base_url": "https://serving.invalid/v1", "api_key_env": "INF_API_KEY",
                "backend": "inspire_serving", "expected_boots": ["boot-a"], "ready_expires_at": time.time() + 60,
                "revision": "weight", "precision": "BF16", "processor_revision": "processor", "runtime_revision": "runtime"}]})
            def respond(request):
                self.assertEqual(request.headers["Authorization"], "Bearer fixture-serving-key")
                return httpx.Response(200, headers={"X-UMM-Boot-ID": "boot-a"}, json={"model": "fixture", "choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}]})
            client = DynamicClient(registry, root / "cache", wait_ready=1)
            await client.client.aclose()
            client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            with patch.dict(os.environ, {"INF_API_KEY": "fixture-serving-key"}):
                result = await client.request("generator", "fixture", {}, [], "Return JSON", lambda value: None)
            self.assertEqual(result["identity"]["precision"], "BF16")
            self.assertEqual(result["response_boot_id"], "boot-a")
            self.assertNotIn("fixture-serving-key", json.dumps(result))
            await client.close()

    async def test_unqualified_response_boot_is_transient_and_never_success_cached(self):
        import httpx
        import time
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "endpoints.json"
            atomic_json(registry, {"endpoints": [{"role": "generator", "model": "fixture", "protocol": "openai",
                "ready": True, "base_url": "https://fixture.invalid/v1", "backend": "inspire_serving",
                "expected_boots": ["boot-a"], "ready_expires_at": time.time() + 60}]})
            client = DynamicClient(registry, root / "cache", wait_ready=1)
            await client.client.aclose()
            client.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,
                headers={"X-UMM-Boot": "unqualified-new-boot"}, json={"model": "fixture", "choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}]})))
            with patch("corpus_pipeline.production.client.asyncio.sleep", new=__import__('unittest.mock', fromlist=['AsyncMock']).AsyncMock()):
                with self.assertRaises(ModelStageError) as caught:
                    await client.request("generator", "fixture", {}, [], "Return JSON", lambda value: None)
            self.assertTrue(caught.exception.retryable_after_endpoint_change)
            self.assertTrue(all(path.name.endswith(".failure.json") for path in (root / "cache").glob("*.json")))
            await client.close()

    async def test_identical_concurrent_requests_share_one_http_and_complete_result(self):
        import asyncio
        import httpx
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "endpoints.json"
            atomic_json(registry, {"endpoints": [{"role": "generator", "model": "fixture", "protocol": "openai",
                "ready": True, "base_url": "https://fixture.invalid/v1"}]})
            calls = []
            async def respond(request):
                calls.append(request)
                await asyncio.sleep(.02)
                return httpx.Response(200, json={"model": "fixture", "choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}]})
            client = DynamicClient(registry, root / "cache", wait_ready=1)
            await client.client.aclose()
            client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            def validate(value):
                self.assertEqual(value, {"valid": True})
            results = await asyncio.gather(*(client.request("generator", "fixture", {}, [], "Return JSON", validate) for _ in range(16)))
            self.assertEqual(len(calls), 1)
            self.assertTrue(all(result == results[0] for result in results))
            cached = json.loads(next((root / "cache").glob("*.json")).read_text())
            self.assertEqual(cached["parsed"], {"valid": True})
            await client.close()

    async def test_optional_layout_failure_preserves_native_annotation(self):
        from .run import annotate_record
        record = {"source_image_refs": [{"sha256": "a"}], "normalized": {"category": "single_i2t"},
                  "raw": {"format": "conversation", "segments": []}}
        class FixtureClient:
            async def request(self, role, stage, *args, **kwargs):
                if stage == "typed_annotation.v1":
                    return {"parsed": {"concerns": [], "layout_candidates": [{"eligible": True, "image_index": 0, "reason": "fixture"}]}}
                if stage == "typed_quality.v1":
                    return {"parsed": {"verdict": "pass"}}
                raise RuntimeError("fixture optional layout outage")
        result = await annotate_record(record, {"a": {}}, FixtureClient())
        self.assertTrue(result["annotation_quality_pass"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["layouts"][0]["status"], "failed")

    async def test_wrong_response_model_is_not_cached_as_success(self):
        import httpx
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "endpoints.json"
            atomic_json(registry, {"endpoints": [{"role": "generator", "model": "expected", "protocol": "openai",
                "ready": True, "base_url": "https://fixture.invalid/v1"}]})
            client = DynamicClient(registry, root / "cache", wait_ready=1)
            await client.client.aclose()
            client.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,
                json={"model": "wrong", "choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}]})))
            with patch("corpus_pipeline.production.client.asyncio.sleep", new=__import__('unittest.mock', fromlist=['AsyncMock']).AsyncMock()):
                with self.assertRaisesRegex(RuntimeError, "model_stage_failed"):
                    await client.request("generator", "fixture", {}, [], "Return JSON", lambda value: None)
            self.assertTrue(all(path.name.endswith(".failure.json") for path in (root / "cache").glob("*.json")))
            await client.close()
    async def test_retry_reads_replaced_endpoint_and_valid_success_is_cached(self):
        import httpx
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "endpoints.json"
            entry = {"role": "generator", "model": "fixture", "protocol": "openai", "ready": True,
                     "revision": "fixture-v1", "base_url": "https://first.invalid/v1", "supports_images": True,
                     "processor_revision": "p1", "runtime_revision": "r1"}
            atomic_json(registry, {"registry_generation": 1, "endpoints": [entry]})
            urls = []
            def respond(request):
                urls.append(str(request.url))
                if len(urls) == 1:
                    entry["base_url"] = "https://second.invalid/v1"
                    atomic_json(registry, {"registry_generation": 2, "endpoints": [entry]})
                    return httpx.Response(503, json={"error": "fixture outage"})
                return httpx.Response(200, json={"choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}], "model": "fixture"})
            client = DynamicClient(registry, root / "cache", wait_ready=2)
            await client.client.aclose()
            client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            def validate(value):
                self.assertEqual(value, {"valid": True})
            with patch("corpus_pipeline.production.client.asyncio.sleep", new=__import__('unittest.mock', fromlist=['AsyncMock']).AsyncMock()):
                result = await client.request("generator", "fixture", {}, [], "Return JSON", validate)
            self.assertEqual(len(urls), 2)
            self.assertIn("second.invalid", urls[1])
            self.assertEqual(result["registry_generation"], 2)
            cached = await client.request("generator", "fixture", {}, [], "Return JSON", validate)
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(len(urls), 2)
            await client.close()

    async def test_truncated_response_increases_budget_instead_of_repeating(self):
        import httpx
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "endpoints.json"
            atomic_json(registry, {"registry_generation": 1, "endpoints": [{"role": "generator", "model": "fixture",
                "protocol": "openai", "ready": True, "revision": "fixture-v1", "base_url": "https://fixture.invalid/v1"}]})
            budgets = []
            def respond(request):
                budgets.append(json.loads(request.content)["max_tokens"])
                return httpx.Response(200, json={"choices": [{"message": {"content": '{"valid":true}'},
                    "finish_reason": "length" if len(budgets) == 1 else "stop"}], "model": "fixture"})
            client = DynamicClient(registry, root / "cache", wait_ready=2)
            await client.client.aclose()
            client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            with patch("corpus_pipeline.production.client.asyncio.sleep", new=__import__('unittest.mock', fromlist=['AsyncMock']).AsyncMock()):
                result = await client.request("generator", "fixture", {}, [], "Return JSON", lambda value: None, max_tokens=100)
            self.assertEqual(budgets, [100, 200])
            self.assertEqual(result["identity"]["max_tokens"], 200)
            await client.close()


if __name__ == "__main__":
    unittest.main()
