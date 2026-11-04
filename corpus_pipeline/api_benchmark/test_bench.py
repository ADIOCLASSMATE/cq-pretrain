import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from corpus_pipeline.api_review.client import Credentials
from .bench import BenchClient, aggregate, usage_metrics, validate_response, safe_retry_after, retry_delay


class ValidationTests(unittest.TestCase):
    def test_correct_request_id_does_not_substitute_for_visual_truth(self):
        truth = {"images": [{"index": 0, "number": "01234", "color": "orange"}, {"index": 1, "number": "56789", "color": "green"}]}
        parsed = {"request_id": "fresh-a", "images": [{**row} for row in truth["images"]]}
        self.assertTrue(validate_response("vision_probe", parsed, "fresh-a", {"truth": truth})["vision_truth_passed"])
        parsed["images"][0]["number"] = "1234"
        self.assertFalse(validate_response("vision_probe", parsed, "fresh-a", {"truth": truth})["vision_truth_passed"])
        with self.assertRaises(ValueError):
            validate_response("vision_probe", parsed, "different-request", {"truth": truth})

    def test_reasoning_zero_text_and_billed_output_remain_distinct(self):
        usage = usage_metrics({"input_tokens": 50, "output_tokens": 386, "billing_usage": {"openai_usage": {
            "completion_tokens": 386, "completion_tokens_details": {"reasoning_tokens": 319, "text_tokens": 0}}}})
        self.assertEqual(usage["reported_text_tokens"]["value"], 0)
        self.assertEqual(usage["reasoning_tokens"]["value"], 319)
        self.assertEqual(usage["billed_output_tokens"]["value"], 386)
        self.assertEqual(usage["estimated_visible_output_tokens"], 67)
        summary = aggregate([
            {"status": "ok", "http_status": 200, "effective_success": True, "latency_seconds": 2, "usage_metrics": usage},
            {"status": "api_error", "http_status": 429, "effective_success": False, "latency_seconds": .5, "error_type": "RateLimitError"},
            {"status": "schema_error", "http_status": 200, "effective_success": False, "latency_seconds": 1, "error_type": "ValueError"},
        ], 4, 3)
        self.assertEqual(summary["effective_requests_per_second"], .25)
        self.assertAlmostEqual(summary["effective_failure_rate"], 2 / 3)
        self.assertEqual(summary["http_429_count"], 1)
        self.assertEqual(summary["tokens"]["billed_output_tokens"]["per_wall_second"], 96.5)
        self.assertEqual(summary["tokens"]["reported_text_tokens"]["per_wall_second"], 0)
        self.assertIsNone(safe_retry_after("do not persist arbitrary secret header text"))

    def test_output_counters_do_not_use_input_text_and_retry_after_is_honored(self):
        usage = usage_metrics({"text_tokens": 999, "prompt_tokens_details": {"text_tokens": 500},
            "billing_usage": {"openai_usage": {"completion_tokens_details": {"text_tokens": 0}}}})
        self.assertEqual(usage["reported_text_tokens"]["value"], 0)
        self.assertEqual(usage["reported_text_tokens"]["usage_path"], "billing_usage.openai_usage.completion_tokens_details.text_tokens")
        self.assertIsNone(usage_metrics({"prompt_tokens_details": {"text_tokens": 500}})["reported_text_tokens"]["value"])
        self.assertEqual(retry_delay(270, 0), 270)
        self.assertEqual(retry_delay("1970-01-01T00:02:00+00:00", 0, now=5), 115)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_raw_parse_records_real_roundtrip_and_explicit_disabled(self):
        captured = []
        async def handler(request):
            captured.append(json.loads(request.content))
            await asyncio.sleep(.01)
            return httpx.Response(200, json={"id": "msg_fixture", "type": "message", "role": "assistant", "model": "fixture",
                "content": [{"type": "thinking", "thinking": "private reasoning content", "signature": "fixture"},
                            {"type": "text", "text": '{"request_id":"fixture"}'}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 30, "output_tokens": 10, "billing_usage": {"openai_usage": {"completion_tokens_details": {"reasoning_tokens": 3}}}}})
        client = BenchClient(Credentials("https://fixture.invalid", "fixture-secret"), timeout=2, concurrency=1,
                             transport=httpx.MockTransport(handler))
        row = await client.request(model="fixture", content=[{"type": "text", "text": "test"}], max_tokens=128,
                                   thinking="disabled", thinking_budget=0)
        await client.close()
        self.assertEqual(row["http_status"], 200)
        self.assertEqual(row["status"], "response_received")
        self.assertGreaterEqual(row["latency_seconds"], .009)
        self.assertGreater(row["end_unix"], row["start_unix"])
        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})
        self.assertEqual(row["usage_metrics"]["reasoning_tokens"]["value"], 3)
        self.assertNotIn("fixture-secret", json.dumps(row))
        self.assertEqual(row["thinking_blocks_count"], 1)
        self.assertNotIn("private reasoning content", json.dumps(row))

    async def test_whole_request_deadline_ends_an_open_transport_and_parse_errors_keep_status(self):
        async def slow(request):
            await asyncio.sleep(10)
            raise AssertionError("wall deadline should cancel transport")
        client = BenchClient(Credentials("https://fixture.invalid", "fixture-secret"), timeout=.02, concurrency=1,
                             transport=httpx.MockTransport(slow))
        row = await client.request(model="fixture", content=[{"type": "text", "text": "test"}], max_tokens=128,
                                   thinking="disabled", thinking_budget=0)
        self.assertEqual(row["error_type"], "TimeoutError")
        self.assertLess(row["latency_seconds"], .5)
        self.assertGreaterEqual(row["latency_seconds"], .015)
        def bad_parse():
            raise ValueError("sensitive raw response must never be logged")
        raw = SimpleNamespace(status_code=200, headers={}, parse=bad_parse)
        with patch.object(client.client.messages.with_raw_response, "create", new=AsyncMock(return_value=raw)):
            row = await client.request(model="fixture", content=[{"type": "text", "text": "test"}], max_tokens=128,
                                       thinking="disabled", thinking_budget=0)
        await client.close()
        self.assertEqual(row["http_status"], 200)
        self.assertEqual(row["status"], "client_error")
        self.assertEqual(row["error_type"], "ValueError")
        self.assertNotIn("sensitive raw response", json.dumps(row))

    async def test_rate_limit_is_not_retried_or_hidden_and_error_body_not_logged(self):
        calls = []
        def handler(request):
            calls.append(1)
            return httpx.Response(429, headers={"retry-after": "7"}, json={"type": "error", "error": {
                "type": "rate_limit_error", "message": "fixture-secret and full raw exception must not persist"}})
        client = BenchClient(Credentials("https://fixture.invalid", "fixture-secret"), timeout=2, concurrency=1,
                             transport=httpx.MockTransport(handler))
        row = await client.request(model="fixture", content=[{"type": "text", "text": "test"}], max_tokens=128,
                                   thinking="omit", thinking_budget=0)
        await client.close()
        self.assertEqual(len(calls), 1)
        self.assertEqual(row["http_status"], 429)
        self.assertEqual(row["retry_after"], 7)
        self.assertNotIn("fixture-secret", json.dumps(row))
        self.assertNotIn("full raw exception", json.dumps(row))


if __name__ == "__main__":
    unittest.main()
