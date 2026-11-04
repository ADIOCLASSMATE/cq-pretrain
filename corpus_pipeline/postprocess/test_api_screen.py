from copy import deepcopy
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import time

import pyarrow as pa
import pyarrow.parquet as pq

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from corpus_pipeline.production.test_image_screening import receipt, ai_cues
from .api_screen import (
    PRIMARY, SECONDARY, POLICY_VERSION, REVIEW_STAGE, Ledger, Screener,
    combine_reviews, iter_input, prepare_asset, receipt_errors, review_context,
    verify_input_scope, export_decisions, ShardedServingClients, ServingUnavailable, ModelStageError, execute,
)

AID = "a" * 64
PROMPT = "fixture screening prompt"
MODELS = {"primary": PRIMARY, "secondary": SECONDARY}


def sample(aid=AID):
    return {"asset_id": aid, "pixel_sha256": "b" * 64, "pixel_sha256_conflict": False,
            "global_status": "ai_suspect_pending_api", "explicit_negative": False,
            "valid_protected": False, "has_invalid_observation": False,
            "original_ref_json": json.dumps({"path": "/fixture/not_read.tar", "offset": 512,
                                              "length": 2048, "sha256": aid}),
            "provenance_json": json.dumps({"origin": "unknown", "evidence": ["immutable source fact"]})}


def bound_receipt(aid=AID, model=PRIMARY, decision="admit", prompt=PROMPT, context=None, backend="api"):
    if decision == "reject":
        row = receipt("likely_ai_generated", cues=ai_cues())
    elif decision == "pending":
        row = receipt("uncertain", needs_provenance=True)
    else:
        row = receipt()
    context = context or review_context(aid)
    row["request_context"] = context
    row["identity"] = {"stage": REVIEW_STAGE, "role": "verifier", "model": model,
                       "context_sha256": digest(context), "assets": [aid], "max_tokens": 1600,
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if backend == "serving" else digest(prompt),
                       "endpoint_identity": "c" * 64}
    row["cache_key"] = digest(row["identity"])
    row["image_inputs"][0].update(asset_id=aid, original_sha256=aid)
    row["response_model"] = model
    if backend == "serving":
        row.update(response_boot_id="boot1", qualified_boots_at_request=["boot1"], endpoint_identity_sha256="c" * 64)
    return row


class FakeClients:
    backend = "api"
    endpoint_identity = {"primary": "c" * 64, "secondary": "c" * 64}

    def __init__(self, decisions=None, fail=0, mutate=None):
        self.calls = []
        self.decisions = decisions or {"primary": "admit", "secondary": "admit"}
        self.fail, self.mutate = fail, mutate
        self.active = self.peak = 0

    async def request(self, role, asset, prompt, context, max_tokens):
        self.calls.append((role, deepcopy(asset), prompt, deepcopy(context)))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(.001)
        self.active -= 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("fixture transport unavailable")
        value = bound_receipt(asset["asset_id"], MODELS[role], self.decisions[role], prompt, context)
        if self.mutate:
            self.mutate(value)
        return value


class StopFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_file_before_import_and_during_request_never_completes(self):
        for initially_present in [True, False]:
            with self.subTest(initially_present=initially_present), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);stop_path=root/'STOP';source=root/'input.parquet'
                pq.write_table(pa.Table.from_pylist([sample()]),source)
                prompt=root/'prompt.txt';prompt.write_text(PROMPT)
                if initially_present:stop_path.touch()
                class Client(FakeClients):
                    closed=False
                    def stats(self):return {}
                    async def close(self):self.closed=True
                    async def request(self,*args,**kwargs):
                        stop_path.touch()
                        await asyncio.sleep(1.1)
                        return await super().request(*args,**kwargs)
                client=Client()
                args=SimpleNamespace(input=[str(source)],reconciliation_root=None,canary=1,
                    primary=PRIMARY,secondary=SECONDARY,mode='primary',backend='api',prompt=str(prompt),
                    output=str(root/'out'),stop_file=str(stop_path),image_threads=1,credential_example='unused',
                    concurrency=1,timeout=5,max_attempts=1,max_tokens=1600,progress_interval=1)
                with patch('corpus_pipeline.postprocess.api_screen.ShardedQwenClients',return_value=client):
                    result=await execute(args)
                self.assertTrue(result['interrupted']);self.assertTrue(client.closed)
                self.assertFalse(json.loads((root/'out/COMPLETE.json').read_text())['screening_pass_complete'])
                if initially_present:
                    self.assertEqual(client.calls,[])
                    self.assertFalse((root/'out/INPUT_IMPORTED.json').exists())
                else:self.assertEqual(len(client.calls),1)


class ScreeningTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.identity = {"fixture": True, "policy": POLICY_VERSION}
        self.ledger = Ledger(self.root / "out", self.identity)

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def worker(self, client, **kwargs):
        return Screener(self.ledger, client, concurrency=3, max_attempts=1, prompt=PROMPT, **kwargs)

    def result(self, aid=AID):
        encoded = self.ledger.db.execute("SELECT decision_json FROM assets WHERE asset_id=?", (aid,)).fetchone()[0]
        return json.loads(encoded)

    async def test_identical_duplicates_only_call_each_model_once(self):
        row = sample()
        self.assertTrue(self.ledger.insert(row))
        self.assertFalse(self.ledger.insert(row))
        self.ledger.db.commit()
        client = FakeClients()
        result = await self.worker(client).run()
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["counts"]["decision:admit"], 1)
        self.assertEqual(self.result()["origin"], "unknown")
        self.assertFalse(self.result()["source_certification"])
        for _, _, _, context in client.calls:
            self.assertEqual(set(context), {"asset_id", "source_certification", "screening_policy"})
            self.assertNotIn("evidence", context)
        self.assertEqual(row, sample())

    async def test_custom_prompt_is_used_and_receipt_bound(self):
        self.ledger.insert(sample())
        client = FakeClients()
        custom = "fixture v4 wording preserving the v3 response schema"
        worker = Screener(self.ledger, client, concurrency=2, max_attempts=1, prompt=custom)
        await worker.run()
        self.assertEqual(self.result()["decision"], "admit")
        self.assertTrue(all(call[2] == custom for call in client.calls))
        receipt_path = next((self.root / "out" / "receipts").rglob("*.json"))
        value = json.loads(receipt_path.read_text())
        self.assertEqual(value["identity"]["prompt_sha256"], digest(custom))
        self.assertEqual(value["identity"]["stage"], REVIEW_STAGE)

    async def test_independent_model_disagreement_pending(self):
        self.ledger.insert(sample())
        await self.worker(FakeClients({"primary": "reject", "secondary": "admit"})).run()
        self.assertEqual(self.result()["decision"], "pending")
        self.assertEqual(self.result()["reason"], "reviewer_disagreement")

    async def test_two_valid_rejections_required(self):
        self.ledger.insert(sample())
        await self.worker(FakeClients({"primary": "reject", "secondary": "reject"})).run()
        self.assertEqual(self.result()["decision"], "reject")
        self.assertEqual(len(self.result()["reviews"]), 2)

    async def test_annotate_known_ai_model_admit_cannot_clear_detector_alarm(self):
        # A known-generated control can receive a wrong non-AI model assessment.
        # The control label is test knowledge, never leaked into model context.
        self.ledger.insert(sample())
        client = FakeClients({"primary": "admit"})
        await self.worker(client, mode="annotate", models={"primary": PRIMARY}).run()
        value = self.result()
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(value["decision"], "pending")
        self.assertEqual(value["reason"], "visual_annotation_only_requires_confirmation")
        self.assertEqual(value["model_recommendation"], "admit")
        self.assertFalse(value["detector_alarm_cleared"])
        self.assertFalse(value["source_certification"])
        self.assertEqual(value["origin"], "unknown")
        self.assertEqual(value["reviews"]["primary"]["decision"], "admit")
        annotation = value["reviews"]["primary"]["visual_annotation"]
        self.assertEqual(annotation["assessment"], "likely_non_ai")
        self.assertTrue(annotation["visible_evidence"])

    async def test_annotate_reject_and_uncertain_remain_pending(self):
        for index, recommendation in enumerate(("reject", "pending")):
            aid = hashlib.sha256(str(index).encode()).hexdigest()
            row = sample(aid)
            self.ledger.insert(row)
            await self.worker(FakeClients({"primary": recommendation}), mode="annotate", models={"primary": PRIMARY}).process(row)
            value = self.result(aid)
            self.assertEqual(value["decision"], "pending")
            self.assertEqual(value["model_recommendation"], recommendation)
            self.assertFalse(value["detector_alarm_cleared"])

    async def test_annotate_invalid_response_is_not_a_completed_visual_review(self):
        self.ledger.insert(sample())
        await self.worker(FakeClients(fail=1), mode="annotate", models={"primary": PRIMARY}).run()
        value = self.result()
        self.assertEqual(value["decision"], "pending")
        self.assertEqual(value["reason"], "api_review_failure_exhausted")
        self.assertIsNone(value["model_recommendation"])
        self.assertFalse(value["detector_alarm_cleared"])

    async def test_primary_only_cannot_reject(self):
        self.ledger.insert(sample())
        client = FakeClients({"primary": "reject"})
        await self.worker(client, mode="primary", models={"primary": PRIMARY}).run()
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.result()["decision"], "pending")
        self.assertEqual(self.result()["reason"], "independent_secondary_review_required")

    async def test_primary_only_can_screen_admit_without_source_change(self):
        self.ledger.insert(sample())
        await self.worker(FakeClients(), mode="primary", models={"primary": PRIMARY}).run()
        self.assertEqual(self.result()["decision"], "admit")
        self.assertEqual(self.result()["origin"], "unknown")

    async def test_pending_integrity_never_calls_model(self):
        row = sample()
        row["has_invalid_observation"] = True
        self.ledger.insert(row)
        client = FakeClients()
        await self.worker(client).run()
        self.assertFalse(client.calls)
        self.assertEqual(self.result()["reason"], "input_integrity_or_global_evidence_conflict")

    async def test_response_validation_failure_is_not_uncertain_vote(self):
        self.ledger.insert(sample())
        client = FakeClients(mutate=lambda value: value["parsed"].update(confidence="certain"))
        await self.worker(client).run()
        self.assertEqual(self.result()["reason"], "api_review_failure_exhausted")
        self.assertEqual(self.result()["reviews"], {})

    async def test_bound_receipt_corruption_cannot_be_reused(self):
        self.ledger.insert(sample())
        client = FakeClients()
        worker = self.worker(client)
        asset, provenance = prepare_asset(sample())
        await worker.review("primary", asset, provenance)
        path = next((self.root / "out" / "receipts").rglob("*.json"))
        path.write_text("{}")
        await worker.run()
        self.assertEqual(self.result()["reason"], "saved_review_integrity_failure")
        self.assertEqual(len(client.calls), 1)

    async def test_resume_does_not_retry_exhausted_or_completed(self):
        self.ledger.insert(sample())
        client = FakeClients(fail=2)
        await self.worker(client).run()
        self.assertEqual(self.result()["reason"], "api_review_failure_exhausted")
        await self.worker(client).run()
        self.assertEqual(len(client.calls), 2)

    async def test_resume_partial_success_reuses_valid_receipt(self):
        self.ledger.insert(sample())
        client = FakeClients()
        worker = self.worker(client)
        asset, provenance = prepare_asset(sample())
        await worker.review("primary", asset, provenance)
        await worker.run()
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(self.ledger.summary()["requests"]["durable_receipt_hits"], 1)

    async def test_pool_is_bounded_and_does_not_spawn_per_asset(self):
        for index in range(15):
            self.ledger.insert(sample(hashlib.sha256(str(index).encode()).hexdigest()))
        client = FakeClients()
        result = await self.worker(client).run()
        self.assertLessEqual(client.peak, 3)
        self.assertEqual(result["counts"]["state:done"], 15)
        output = export_decisions(self.ledger)
        self.assertEqual(pq.read_table(self.root / "out" / output["path"]).num_rows, 15)

    async def test_stop_leaves_unstarted_rows_pending(self):
        self.ledger.insert(sample())
        stop = asyncio.Event()
        stop.set()
        client = FakeClients()
        result = await self.worker(client, stop=stop).run()
        self.assertFalse(client.calls)
        self.assertEqual(result["counts"]["state:pending"], 1)

    async def test_serving_transient_failure_preserves_retry_budget_and_resumes(self):
        class TransientServing(FakeClients):
            backend = "serving"
            endpoint_identity = {"primary": None}
            def __init__(self):
                super().__init__()
                self.unavailable = True
            async def request(self, role, asset, prompt, context, max_tokens):
                self.calls.append(role)
                if self.unavailable:
                    raise ModelStageError(REVIEW_STAGE, [{"type": "EndpointNotReady"}], retryable_after_endpoint_change=True)
                return bound_receipt(asset["asset_id"], PRIMARY, prompt=prompt, context=context, backend="serving")
        self.ledger.insert(sample())
        client = TransientServing()
        worker = self.worker(client, mode="primary", models={"primary": PRIMARY})
        await worker.run()
        self.assertEqual(self.result()["reason"], "serving_temporarily_unavailable")
        self.assertEqual(self.ledger.db.execute("SELECT attempts,state FROM requests").fetchone(), (0,"deferred"))
        client.unavailable = False
        await worker.run()
        self.assertEqual(self.result()["decision"], "admit")
        self.assertEqual(len(client.calls), 2)

    async def test_serving_restart_refunds_interrupted_inflight_attempt(self):
        client = FakeClients()
        client.backend = "serving"
        self.ledger.insert(sample())
        self.ledger.db.execute("INSERT INTO requests(asset_id,role,request_sha,attempts,state) VALUES(?,?,?,1,'inflight')",
                               (AID, "primary", "fixture_hash"))
        self.ledger.db.commit()
        self.worker(client, mode="primary", models={"primary": PRIMARY})
        self.assertEqual(self.ledger.db.execute("SELECT attempts,state FROM requests").fetchone(), (0,"deferred"))

    def test_hash_identity_mismatch_rejected_before_model(self):
        row = sample()
        ref = json.loads(row["original_ref_json"])
        ref["sha256"] = "b" * 64
        row["original_ref_json"] = json.dumps(ref)
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            prepare_asset(row)

    def test_conflicting_duplicate_source_evidence_fails_closed(self):
        self.ledger.insert(sample())
        other = sample()
        other["provenance_json"] = json.dumps({"origin": "natural"})
        with self.assertRaisesRegex(ValueError, "conflicting_immutable_input"):
            self.ledger.insert(other)

    def test_resume_requires_same_manifest_and_existing_ledger(self):
        with self.assertRaisesRegex(ValueError, "input_or_policy_changed"):
            Ledger(self.root / "out", {"different": True})

    def test_receipt_checks_prompt_model_asset_and_serving_boot(self):
        good = bound_receipt()
        self.assertEqual(receipt_errors(good, AID, PRIMARY, PROMPT, review_context(AID), 1600, "c" * 64), [])
        for field in ("model", "prompt_sha256", "assets"):
            bad = deepcopy(good)
            bad["identity"][field] = "wrong"
            self.assertTrue(receipt_errors(bad, AID, PRIMARY, PROMPT, review_context(AID), 1600))
        serving = bound_receipt(backend="serving")
        self.assertEqual(receipt_errors(serving, AID, PRIMARY, PROMPT, review_context(AID), 1600, backend="serving"), [])
        serving["response_boot_id"] = "unqualified"
        self.assertIn("serving_boot_not_qualified", receipt_errors(serving, AID, PRIMARY, PROMPT, review_context(AID), 1600, backend="serving"))


class InputTests(unittest.TestCase):
    def test_pinned_loose_image_survives_root_move_and_checks_original_bytes(self):
        from corpus_pipeline.production.records import read_asset_bytes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original'
            original.mkdir()
            blob = b'original image fixture'
            aid = hashlib.sha256(blob).hexdigest()
            (original / 'image.jpg').write_bytes(blob)
            dataset = 'inspire/fixture@' + '1' * 64
            mapping = root / 'roots.json'
            mapping.write_text(json.dumps({dataset: str(original)}))
            ref = {'kind': 'file', 'path': 'image.jpg', 'member': 'image.jpg',
                   'dataset_id': dataset, 'offset': 0, 'offset_data': 0,
                   'length': len(blob), 'size': len(blob), 'sha256': aid}
            row = {**sample(aid), 'original_ref_json': json.dumps(ref)}
            with patch.dict('os.environ', {'CORPUS_DATASET_ROOTS': str(mapping)}):
                asset, _ = prepare_asset(row)
                self.assertEqual(asset['original_ref'], ref)
                self.assertEqual(read_asset_bytes(asset['original_ref']), blob)
                moved = root / 'moved'
                original.rename(moved)
                mapping.write_text(json.dumps({dataset: str(moved)}))
                relocated, _ = prepare_asset(row)
                self.assertEqual(relocated, asset)
                self.assertEqual(read_asset_bytes(relocated['original_ref']), blob)
                for change in ({'dataset_id': None}, {'path': '../image.jpg'},
                               {'offset': 1}, {'member': 'other.jpg'}, {'size': len(blob) + 1}):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        prepare_asset({**row, 'original_ref_json': json.dumps({**ref, **change})})
                (moved / 'image.jpg').write_bytes(b'x' * len(blob))
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    read_asset_bytes(prepare_asset(row)[0]['original_ref'])

    def test_incremental_selection_validates_exact_rows_and_preserves_validation_scope(self):
        from corpus_pipeline.unified.screening import build_decisions
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'buckets/00/api_review.parquet';path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([sample(),sample('c'*64)]),path)
            buckets=[];identity='d'*64
            for number in range(256):
                code=f'{number:02x}';receipt=root/'buckets'/code/'COMPLETE.json'
                artifacts=[{'path':str(path.relative_to(root)),'sha256':file_sha(path),'size_bytes':path.stat().st_size}] if number==0 else []
                atomic_json(receipt,{'schema':'corpus.asset_reconciliation.bucket.v1','identity_sha256':identity,
                                    'bucket':code,'artifacts':artifacts})
                buckets.append({'bucket':code,'receipt_path':str(receipt.relative_to(root)),'receipt_sha256':file_sha(receipt)})
            atomic_json(root/'COMPLETE.json',{'schema':'corpus.asset_reconciliation.v1','complete':True,
                'scope':'full_run','identity_sha256':identity,'input_index_sha256':'e'*64,'buckets':buckets,
                'counts':{'api_review_assets':2}})
            pending=build_decisions(root,root/'screening',scope='validation')
            selection=pending['api_selection'];marker=Path(selection['manifest']['path'])
            files=[Path(item['path']) for item in selection['files']]
            scope=verify_input_scope(files,root,selection_manifest=marker)
            self.assertFalse(scope['global_screening_scope_authorized'])
            self.assertEqual(scope['reconciliation_complete_sha256'],file_sha(root/'COMPLETE.json'))
            self.assertEqual(scope['selected_assets'],2)
            rows=pq.read_table(files[0]).to_pylist();rows[0]['original_ref_json']='{"different":"range"}'
            pq.write_table(pa.Table.from_pylist(rows),files[0])
            changed=json.loads(marker.read_text());changed['selected_files'][0].update(
                sha256=file_sha(files[0]),size_bytes=files[0].stat().st_size)
            atomic_json(marker,changed)
            with self.assertRaisesRegex(ValueError,'changed_original_row'):
                verify_input_scope(files,root,selection_manifest=marker)

    def test_canary_is_bounded_not_globally_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.parquet"
            rows = [sample(), {**sample("b" * 64), "global_status": "pending_integrity"}]
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.assertEqual(len(list(iter_input([path]))), 1)
            scope = verify_input_scope([path], canary=8)
            self.assertFalse(scope["global_screening_scope_authorized"])
            with self.assertRaisesRegex(ValueError, "completed_global_reconciliation"):
                verify_input_scope([path])

    def test_full_scope_verifies_all_bucket_receipts_and_selected_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "buckets" / "00" / "api_review.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([sample()]), path)
            identity = "d" * 64
            buckets = []
            for index in range(256):
                bucket = f"{index:02x}"
                receipt_path = root / "buckets" / bucket / "COMPLETE.json"
                artifacts = [{"path": str(path.relative_to(root)), "sha256": file_sha(path), "size_bytes": path.stat().st_size}] if index == 0 else []
                atomic_json(receipt_path, {"schema": "corpus.asset_reconciliation.bucket.v1",
                    "identity_sha256": identity, "bucket": bucket, "artifacts": artifacts})
                buckets.append({"bucket": bucket, "receipt_path": str(receipt_path.relative_to(root)),
                                "receipt_sha256": file_sha(receipt_path)})
            atomic_json(root / "COMPLETE.json", {"schema": "corpus.asset_reconciliation.v1", "complete": True,
                "scope": "full_run", "identity_sha256": identity, "input_index_sha256": "e" * 64, "buckets": buckets})
            self.assertTrue(verify_input_scope([path], root)["global_screening_scope_authorized"])
            (root / buckets[255]["receipt_path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "receipt_hash_mismatch"):
                verify_input_scope([path], root)


class FakeDynamic:
    calls = []

    def __init__(self, registry, cache, **kwargs):
        self.registry, self.cache = Path(registry), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def endpoint(self, role):
        data = json.loads(self.registry.read_text())
        values = [entry for entry in data["endpoints"] if entry["role"] == role and entry.get("ready")
                  and entry.get("ready_expires_at", 0) > time.time()]
        if len(values) != 1:
            raise LookupError("not ready")
        return values[0], data.get("registry_generation")

    async def request(self, role, stage, context, assets, prompt, validator, *, max_tokens):
        self.calls.append(self.registry.name)
        await asyncio.sleep(.005)
        endpoint, _ = self.endpoint(role)
        value = bound_receipt(assets[0]["asset_id"], endpoint["model"], prompt=prompt, context=context, backend="serving")
        value["identity"].update({name: endpoint.get(name) for name in
            ("revision", "precision", "processor_revision", "runtime_revision", "request_overrides")})
        value["cache_key"] = digest(value["identity"])
        return value

    async def close(self):
        pass


class ServingPoolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        FakeDynamic.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def registry(self, name, *, ready=True, revision="rev1", qualification_only=False):
        path = self.root / name
        atomic_json(path, {"qualification_only": qualification_only, "registry_generation": "fixture",
            "endpoints": [{"role": "verifier", "ready": ready, "backend": "inspire_serving",
                "ready_expires_at": time.time() + 1000, "supports_images": True,
                "model": PRIMARY, "revision": revision, "expected_boots": ["boot1"]}]})
        return path

    async def test_routes_only_to_ready_verified_replicas_and_dynamic_pool_growth(self):
        first, second = self.registry("one.json"), self.registry("two.json", ready=False)
        pool_path = self.root / "pool.json"
        atomic_json(pool_path, {"schema_version": "corpus.screening_pool.v1", "registries": [{"path": str(first), "weight": 1}]})
        with patch("corpus_pipeline.postprocess.api_screen.DynamicClient", FakeDynamic):
            client = ShardedServingClients({}, self.root / "cache", concurrency=4, pools={"primary": pool_path}, refresh_interval=0)
            asset, _ = prepare_asset(sample())
            await client.request("primary", asset, PROMPT, review_context(AID), 1600)
            atomic_json(pool_path, {"schema_version": "corpus.screening_pool.v1", "registries": [
                {"path": str(first), "weight": 1}, {"path": str(second), "weight": 1}]})
            await client.request("primary", asset, PROMPT, review_context(AID), 1600)
            self.assertEqual(set(FakeDynamic.calls), {"one.json"})
            self.registry("two.json", ready=True)
            await asyncio.gather(*(client.request("primary", asset, PROMPT, review_context(AID), 1600) for _ in range(4)))
            self.assertIn("two.json", FakeDynamic.calls)
            self.assertEqual(sum(client.inflight.values()), 0)
            await client.close()

    async def test_weighted_capacity_and_total_budget(self):
        paths = [self.registry("one.json"), self.registry("two.json")]
        pool = self.root / "pool.json"
        atomic_json(pool, {"schema_version": "corpus.screening_pool.v1", "registries": [
            {"path": str(paths[0]), "weight": 1}, {"path": str(paths[1]), "weight": 3}]})
        with patch("corpus_pipeline.postprocess.api_screen.DynamicClient", FakeDynamic):
            client = ShardedServingClients({}, self.root / "cache", concurrency=4, wait_ready=.01, pools={"primary": pool})
            keys = [await client.acquire("primary") for _ in range(4)]
            self.assertEqual([key[1] for key in keys].count(str(paths[0])), 1)
            self.assertEqual([key[1] for key in keys].count(str(paths[1])), 3)
            waiting = asyncio.create_task(client.acquire("primary"))
            await asyncio.sleep(.03)
            self.assertFalse(waiting.done())
            client.stop.set()
            with self.assertRaises(ServingUnavailable):
                await waiting
            self.assertEqual(sum(client.inflight.values()), 4)
            await client.close()

    async def test_contract_mismatch_and_qualification_only_are_not_scheduled(self):
        paths = [self.registry("one.json"), self.registry("other_revision.json", revision="rev2"),
                 self.registry("qualification.json", qualification_only=True)]
        with patch("corpus_pipeline.postprocess.api_screen.DynamicClient", FakeDynamic):
            client = ShardedServingClients({"primary": paths}, self.root / "cache")
            self.assertEqual(len(client.ready_entries("primary")), 1)
            await client.close()


if __name__ == "__main__":
    unittest.main()
