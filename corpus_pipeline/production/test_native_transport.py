import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, AsyncMock

import httpx

from .client import DynamicClient, endpoint_lease_valid, ModelStageError
from .common import atomic_json, digest
from .native_transport import (PLATFORM_SCOPE, assemble_openai_sse, native_receipt_errors, platform_snapshot)
from corpus_pipeline.postprocess.api_screen import receipt_errors, review_context, ShardedServingClients
from corpus_pipeline.postprocess.screen_supervisor import ready_pool
from corpus_pipeline.postprocess.test_api_screen import bound_receipt, AID, PROMPT


def endpoint(**overrides):
    return {"role": "verifier", "model": "qwen3.6", "protocol": "openai", "backend": "inspire_serving",
        "ready": True, "base_url": "https://fixture.invalid/v1", "supports_images": True,
        "transport_mode": "native_openai", "attestation_scope": PLATFORM_SCOPE, "native_stream": True,
        "platform_instance_id": "platform-instance-1", "incarnation": "incarnation-1", "launch_contract_sha256": "c" * 64,
        "ready_expires_at": time.time() + 60, "revision": "weights", "runtime_revision": "official-command-image",
        "processor_revision": "processor", "precision": "W8A8_DYNAMIC", "max_output_tokens": 3000, **overrides}


def sse(content='{"valid":true}', model="qwen3.6", finish="stop"):
    values = [
        {"model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": content[:4]}, "finish_reason": None}]},
        {"model": model, "choices": [{"index": 0, "delta": {"content": content[4:]}, "finish_reason": None}]},
        {"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        {"model": model, "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}},
    ]
    return "".join("data: " + json.dumps(value) + "\n\n" for value in values) + "data: [DONE]\n\n"


class NativeProtocolTests(unittest.TestCase):
    def test_native_mode_must_be_explicit_and_complete_never_header_downgrade(self):
        value = endpoint()
        self.assertTrue(endpoint_lease_valid(value))
        for field in ("transport_mode", "attestation_scope", "platform_instance_id", "incarnation", "launch_contract_sha256"):
            changed = deepcopy(value)
            del changed[field]
            self.assertFalse(endpoint_lease_valid(changed), field)
        self.assertFalse(endpoint_lease_valid(endpoint(expected_boots=["pretend-boot"])))
        self.assertFalse(endpoint_lease_valid(endpoint(ready_expires_at=time.time() - 1)))

    def test_standard_sse_assembles_usage_and_requires_terminal_event(self):
        result = assemble_openai_sse(sse(), "qwen3.6")
        self.assertEqual(result["choices"][0]["message"]["content"], '{"valid":true}')
        self.assertEqual(result["usage"]["total_tokens"], 16)
        for body in (sse().replace("data: [DONE]\n\n", ""), sse(model="wrong"),
                     'data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n',
                     sse() + 'data: {}\n\n', sse(finish=None)):
            with self.assertRaises((ValueError, KeyError)):
                assemble_openai_sse(body, "qwen3.6")

    def test_native_receipt_is_honest_about_absent_response_boot(self):
        value = bound_receipt(model="qwen3.6", backend="serving")
        value["identity"].update(transport_mode="native_openai", attestation_scope=PLATFORM_SCOPE, native_stream=True)
        value.update(response_boot_id=None, qualified_boots_at_request=None, response_boot_attested=False,
                     attestation_scope=PLATFORM_SCOPE)
        value["platform_identity_at_request"] = platform_snapshot(endpoint(), time.time())
        value["platform_identity_at_response"] = deepcopy(value["platform_identity_at_request"])
        value["cache_key"] = digest(value["identity"])
        self.assertEqual(receipt_errors(value, AID, "qwen3.6", PROMPT, review_context(AID), 1600, backend="serving"), [])
        for change in ({"response_boot_attested": True}, {"response_boot_id": "invented"},
                       {"qualified_boots_at_request": ["invented"]}, {"attestation_scope": "response_boot"}):
            self.assertTrue(native_receipt_errors({**value, **change}))
        value["platform_identity_at_response"]["incarnation"] = "new"
        self.assertTrue(native_receipt_errors(value))


class NativeClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry = self.root / "registry.json"
        self.entry = endpoint()
        self.publish()

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self):
        atomic_json(self.registry, {"registry_generation": time.time(), "endpoints": [self.entry]})

    async def client(self, respond):
        client = DynamicClient(self.registry, self.root / "cache", wait_ready=.001)
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        return client

    async def request(self, client):
        return await client.request("verifier", "fixture", {}, [], "Return JSON", lambda value: None, max_tokens=3000)

    async def test_native_standard_sse_without_boot_header_and_lease_renewal(self):
        bodies = []
        def respond(request):
            bodies.append(json.loads(request.content))
            self.entry["ready_expires_at"] = time.time() + 120
            self.publish()
            return httpx.Response(200, content=sse(), headers={"content-type": "text/event-stream"})
        client = await self.client(respond)
        value = await self.request(client)
        self.assertTrue(bodies[0]["stream"])
        self.assertTrue(bodies[0]["stream_options"]["include_usage"])
        self.assertIs(value["response_boot_attested"], False)
        self.assertIsNone(value["response_boot_id"])
        self.assertEqual(value["attestation_scope"], PLATFORM_SCOPE)
        self.assertEqual(native_receipt_errors(value), [])
        self.assertTrue((await self.request(client))["cache_hit"])
        self.assertEqual(len(bodies), 1)
        await client.close()

    async def test_native_nonstream_supported_but_untrusted_header_not_promoted(self):
        self.entry["native_stream"] = False
        self.publish()
        client = await self.client(lambda request: httpx.Response(200, headers={"x-umm-boot": "not-an-attestation"},
            json={"model": "qwen3.6", "choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}]}))
        result = await self.request(client)
        self.assertIsNone(result["response_boot_id"])
        self.assertIs(result["response_boot_attested"], False)
        await client.close()

    async def test_incarnation_change_during_response_is_retryable_and_never_cached_success(self):
        index = 0
        def respond(request):
            nonlocal index
            index += 1
            self.entry["incarnation"] = "changed-" + str(index)
            self.publish()
            return httpx.Response(200, content=sse())
        client = await self.client(respond)
        with patch("corpus_pipeline.production.client.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(ModelStageError) as caught:
                await self.request(client)
        self.assertTrue(caught.exception.retryable_after_endpoint_change)
        self.assertTrue(all(path.name.endswith(".failure.json") for path in (self.root / "cache").glob("*.json")))
        await client.close()

    async def test_expired_lease_after_response_is_not_accepted(self):
        def respond(request):
            self.entry["ready_expires_at"] = time.time() - 1
            self.publish()
            return httpx.Response(200, content=sse())
        client = await self.client(respond)
        with patch("corpus_pipeline.production.client.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(ModelStageError) as caught:
                await self.request(client)
        self.assertTrue(caught.exception.retryable_after_endpoint_change)
        await client.close()

    async def test_bad_sse_or_missing_body_model_is_not_accepted(self):
        for stream, body in ((True, sse().replace("[DONE]", "")),
                             (False, json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}))):
            self.entry["native_stream"] = stream
            self.publish()
            client = await self.client(lambda request: httpx.Response(200, content=body))
            with patch("corpus_pipeline.production.client.asyncio.sleep", new=AsyncMock()):
                with self.assertRaises(ModelStageError) as caught:
                    await self.request(client)
            self.assertFalse(caught.exception.retryable_after_endpoint_change)
            await client.close()

    async def test_pool_and_supervisor_accept_native_without_boot_headers(self):
        pool = self.root / "pool.json"
        atomic_json(pool, {"schema_version": "corpus.screening_pool.v1", "registries": [
            {"path": str(self.registry), "weight": 1, "max_inflight": 32}]})
        self.assertEqual(ready_pool(pool)["ready_replicas"], 1)
        clients = ShardedServingClients({}, self.root / "pool-cache", pools={"primary": pool})
        self.assertEqual(clients.models["primary"], "qwen3.6")
        self.assertEqual(clients.contracts["primary"]["transport_mode"], "native_openai")
        await clients.close()

    async def test_transport_change_cannot_reuse_header_attested_cache(self):
        calls = []
        def respond(request):
            calls.append(json.loads(request.content))
            if calls[-1].get("stream"):
                return httpx.Response(200, content=sse())
            return httpx.Response(200, headers={"X-UMM-Boot": "old-boot"}, json={"model": "qwen3.6",
                "choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}]})
        native = deepcopy(self.entry)
        self.entry.update(transport_mode="boot_header", expected_boots=["old-boot"], native_stream=False)
        self.entry.pop("attestation_scope")
        self.publish()
        client = await self.client(respond)
        old = await self.request(client)
        self.entry = native
        self.publish()
        new = await self.request(client)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(old["cache_key"], new["cache_key"])
        self.assertFalse(new["cache_hit"])
        self.assertFalse(new["response_boot_attested"])
        await client.close()

    async def test_native_whole_request_timeout_cancels_stream_and_is_retryable(self):
        closed = []
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"data: "
                await asyncio.Event().wait()
            async def aclose(self):
                closed.append(True)
        client = await self.client(lambda request: httpx.Response(200, stream=SlowStream()))
        client.timeout = .01
        with patch("corpus_pipeline.production.client.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(ModelStageError) as caught:
                await self.request(client)
        self.assertTrue(caught.exception.retryable_after_endpoint_change)
        self.assertEqual(len(closed), 4)
        await client.close()


if __name__ == "__main__":
    unittest.main()
