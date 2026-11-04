import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import httpx

from corpus_pipeline.production.common import atomic_json
from .api_screen import ShardedServingClients, ServingUnavailable, Ledger, execute
from .test_api_screen import FakeDynamic, PRIMARY, sample
from .test_api_screen import PROMPT, AID
from .api_screen import prepare_asset, review_context
from .worker_control import AdmissionControl, ADMISSION_SCHEMA, MeasuredExecutor, cpu_limits
from .cpu_stage import delta_summary
from .screen_supervisor import build_command, checked_config


class Controls(unittest.TestCase):
    def test_stage_throughput_excludes_cache_history_and_rejects_restart(self):
        first = {"worker_instance": "same", "updated_at_epoch": 100,
                 "requests": {"actual_responses": 1000, "cached_responses": 10},
                 "transport": {"http_attempts": 1000}}
        last = {"worker_instance": "same", "updated_at_epoch": 110,
                "requests": {"actual_responses": 1040, "cached_responses": 400},
                "transport": {"http_attempts": 1042}}
        self.assertEqual(delta_summary(first, last)["fresh_valid_images_per_second"], 4)
        last["worker_instance"] = "restarted"
        with self.assertRaisesRegex(ValueError, "restarted"):
            delta_summary(first, last)

    def test_supervisor_forwards_cpu_parameters_without_resolving_venv(self):
        config = checked_config({name: "/fixture/" + name for name in
            ("reconciliation_root", "pool", "output", "python", "workdir", "api_key_file")}
            | {"mode": "annotate", "image_threads": 16, "progress_interval": 10,
               "admission_control": "/fixture/admission.json", "concurrency": 2048})
        command = build_command(config, [])
        self.assertEqual(command[0], "/fixture/python")
        self.assertIn("--admission-control", command)
        self.assertEqual(command[command.index("--concurrency") + 1], "2048")
        self.assertEqual(command[command.index("--image-threads") + 1], "16")

    def test_cgroup_quota_limits_threads_not_machine_affinity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "cpu.max").write_text("1600000 100000")
            (path / "memory.max").write_text("274877906944")
            with patch("os.sched_getaffinity", return_value=set(range(192))):
                value = cpu_limits(path)
            self.assertEqual(value["suggested_image_threads"], 16)
            self.assertEqual(value["cpu_quota"], 16)

    def test_admission_missing_malformed_or_excessive_is_paused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"
            control = AdmissionControl(2048, path)
            for value in (None, {}, {"schema": ADMISSION_SCHEMA, "max_inflight": True},
                          {"schema": ADMISSION_SCHEMA, "max_inflight": 2049}):
                if value is not None:
                    atomic_json(path, value)
                control.refresh()
                self.assertEqual(control.limit, 0)
                self.assertEqual(control.status, "invalid_control_paused")
            atomic_json(path, {"schema": ADMISSION_SCHEMA, "max_inflight": 2048})
            control.refresh()
            self.assertEqual(control.limit, 2048)

    def test_incremental_summary_matches_disk_after_retry_resume_and_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(directory, {"fixture": True})
            self.assertEqual(ledger.summary()["counts"], {})
            for number in range(20):
                row = sample(hashlib.sha256(str(number).encode()).hexdigest())
                ledger.insert(row)
                ledger.insert(row)
                if number % 2 == 0:
                    ledger.finish(row["asset_id"], {"decision": "pending", "reason": "retry"}, retryable=True)
                    ledger.finish(row["asset_id"], {"decision": "pending", "reason": "annotation"})
            ledger.db.commit()
            fast = deepcopy(ledger.summary())
            self.assertEqual(fast["counts"]["eligible_assets"], 20)
            self.assertEqual(fast["counts"]["state:done"], 10)
            ledger.summary_counts = None
            self.assertEqual(ledger.summary(), fast)
            ledger.close()
            resumed = Ledger(directory, {"fixture": True})
            self.assertEqual(resumed.summary(), fast)
            resumed.close()


class AsyncControls(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def pool(self, replicas, cap=32, control=128, ceiling=2048):
        registries = []
        for index in range(replicas):
            path = self.root / f"registry-{index}.json"
            atomic_json(path, {"endpoints": [{"role": "verifier", "ready": True,
                "ready_expires_at": time.time() + 600, "backend": "inspire_serving", "supports_images": True,
                "model": PRIMARY, "expected_boots": ["boot1"], "revision": "same-revision"}]})
            registries.append({"path": str(path), "weight": 1, "max_inflight": cap})
        self.pool_path = self.root / "pool.json"
        self.control_path = self.root / "admission.json"
        atomic_json(self.pool_path, {"schema_version": "corpus.screening_pool.v1", "registries": registries})
        self.set_limit(control)
        with patch("corpus_pipeline.postprocess.api_screen.DynamicClient", FakeDynamic):
            return ShardedServingClients({}, self.root / "cache", concurrency=ceiling,
                pools={"primary": self.pool_path}, admission_control=self.control_path, refresh_interval=.02)

    def set_limit(self, limit):
        atomic_json(self.control_path, {"schema": ADMISSION_SCHEMA, "max_inflight": limit})

    async def test_64_replicas_share_cached_scan_and_distribute_128_without_duplicates(self):
        client = self.pool(64)
        with patch.object(client, "read_ready_entries", wraps=client.read_ready_entries) as reads:
            client.refresh_interval = 60
            keys = await asyncio.gather(*(client.acquire("primary") for _ in range(128)))
            self.assertEqual(reads.call_count, 0)
            self.assertEqual(len(set(keys)), 64)
            self.assertEqual(max(client.inflight.values()), 2)
            self.assertEqual(sum(client.inflight.values()), 128)
        await client.close()

    async def test_dynamic_increase_reduction_and_preemption_cap(self):
        client = self.pool(2, cap=2, control=1, ceiling=8)
        await client.acquire("primary")
        pending = asyncio.create_task(client.acquire("primary"))
        await asyncio.sleep(.03)
        self.assertFalse(pending.done())
        self.set_limit(4)
        await asyncio.wait_for(pending, 1)
        await client.acquire("primary")
        await client.acquire("primary")
        self.assertEqual(sorted(client.inflight.values()), [2, 2])
        self.set_limit(1)
        await asyncio.sleep(.03)
        pending = asyncio.create_task(client.acquire("primary"))
        await asyncio.sleep(.03)
        self.assertFalse(pending.done())
        client.stop.set()
        with self.assertRaises(ServingUnavailable):
            await pending
        await client.close()

    async def test_expired_lease_is_rejected_even_before_refresh(self):
        client = self.pool(1)
        client.refresh_interval = 60
        client.ready_cache["primary"][0]["endpoint"]["ready_expires_at"] = time.time() - 1
        pending = asyncio.create_task(client.acquire("primary"))
        await asyncio.sleep(.02)
        self.assertFalse(pending.done())
        client.stop.set()
        with self.assertRaises(ServingUnavailable):
            await pending
        await client.close()

    async def test_removed_replica_remains_visible_until_inflight_has_drained(self):
        client = self.pool(2)
        key = await client.acquire("primary")
        value = json.loads(self.pool_path.read_text())
        value["registries"] = [entry for entry in value["registries"] if entry["path"] != key[1]]
        atomic_json(self.pool_path, value)
        client.ready_entries("primary")
        removed = next(entry for entry in client.stats()["replicas"] if entry["registry_path"] == key[1])
        self.assertFalse(removed["in_pool"])
        self.assertFalse(removed["ready"])
        self.assertEqual(removed["inflight"], 1)
        client.inflight[key] -= 1
        drained = next(entry for entry in client.stats()["replicas"] if entry["registry_path"] == key[1])
        self.assertEqual(drained["inflight"], 0)
        await client.close()

    async def test_invalid_pool_refresh_pauses_without_charging_model_attempts(self):
        client = self.pool(1)
        self.pool_path.write_text("malformed")
        await asyncio.sleep(.03)
        pending = asyncio.create_task(client.acquire("primary"))
        await asyncio.sleep(.03)
        self.assertFalse(pending.done())
        self.assertGreater(client.counters["readiness_refresh_failures"], 0)
        self.assertEqual(client.counters["request_invocations"], 0)
        client.stop.set()
        with self.assertRaises(ServingUnavailable):
            await pending
        await client.close()

    async def test_cached_readiness_cannot_send_to_withdrawn_qualification_or_changed_model(self):
        for changed in ("qualification", "model"):
            client = self.pool(1)
            client.refresh_interval = 60
            path = self.root / "registry-0.json"
            value = json.loads(path.read_text())
            if changed == "qualification":
                value["qualification_only"] = True
            else:
                value["endpoints"][0]["model"] = "Qwen3.6-changed"
            atomic_json(path, value)
            asset, _ = prepare_asset(sample())
            calls = len(FakeDynamic.calls)
            with self.assertRaises(ServingUnavailable):
                await client.request("primary", asset, PROMPT, review_context(AID), 1600)
            self.assertEqual(len(FakeDynamic.calls), calls)
            self.assertEqual(sum(client.inflight.values()), 0)
            await client.close()

    async def test_header_heartbeat_does_not_count_complete_body(self):
        client = self.pool(1)
        gate = asyncio.Event()
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b" "
                await gate.wait()
                yield b'{"choices":[]}'
        def respond(request):
            return httpx.Response(200, stream=Stream())
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), event_hooks={
                "request": [client.count_http_request], "response": [client.count_http_response]}) as http:
            task = asyncio.create_task(http.post("https://fixture.invalid/v1", json={}))
            await asyncio.sleep(.02)
            self.assertEqual(client.counters["http_attempts"], 1)
            self.assertEqual(client.counters["http_bodies_completed"], 0)
            self.assertEqual(len(client.http_header_ms), 1)
            gate.set()
            await task
            self.assertEqual(client.counters["http_bodies_completed"], 1)
            self.assertGreater(client.http_body_ms[0], client.http_header_ms[0])
        await client.close()

    async def test_measured_executor_bounds_real_to_thread_workers(self):
        executor = MeasuredExecutor(2)
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        def image_payload(value):
            time.sleep(.005)
            return value * 2
        values = await asyncio.gather(*(asyncio.to_thread(image_payload, index) for index in range(12)))
        self.assertEqual(values, [index * 2 for index in range(12)])
        stats = executor.stats()
        self.assertEqual(stats["successful"], 12)
        self.assertEqual(stats["queued"], 0)
        self.assertLessEqual(stats["peak_running"], 2)
        executor.shutdown(wait=True)

    async def test_execute_resumes_sealed_import_without_rescanning_or_cross_model_reuse(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        client = self.pool(1, control=2, ceiling=2)
        await client.close()
        source = self.root / "input.parquet"
        pq.write_table(pa.Table.from_pylist([sample()]), source)
        prompt = self.root / "prompt.txt"
        prompt.write_text("fixture screening prompt")
        args = SimpleNamespace(input=[str(source)], reconciliation_root=None, canary=1,
            primary=PRIMARY, secondary="qwen-other", mode="annotate", prompt=str(prompt),
            output=str(self.root / "annotations"), backend="serving", primary_registry=[],
            primary_pool=str(self.pool_path), secondary_registry=[], secondary_pool=None,
            concurrency=2, timeout=1, wait_ready=1, max_tokens=1600, max_attempts=1,
            admission_control=str(self.control_path), image_threads=1, progress_interval=1)
        with patch("corpus_pipeline.postprocess.api_screen.DynamicClient", FakeDynamic):
            first = await execute(args)
            self.assertEqual(first["counts"]["state:done"], 1)
            self.assertEqual(first["counts"]["decision:pending"], 1)
            calls = len(FakeDynamic.calls)
            with patch("corpus_pipeline.postprocess.api_screen.iter_input", side_effect=AssertionError("must skip sealed import")):
                second = await execute(args)
            self.assertEqual(second["fresh_actual_responses"], 0)
            self.assertEqual(len(FakeDynamic.calls), calls)
            target = self.root / "registry-0.json"
            data = json.loads(target.read_text())
            data["endpoints"][0]["model"] = "Qwen3.6-35B-A3B-W8A8"
            atomic_json(target, data)
            with self.assertRaisesRegex(ValueError, "immutable_screening_input_or_policy_changed"):
                await execute(args)


if __name__ == "__main__":
    unittest.main()
