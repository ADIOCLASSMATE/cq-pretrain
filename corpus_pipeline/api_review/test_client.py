"""Offline regression tests for review integrity; no real API calls are made."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
from PIL import Image

from corpus_pipeline.api_review import client, run_review


class StrictResponseParsingTests(unittest.TestCase):
    def test_valid_object_and_single_fence(self):
        expected = {"decision": "uncertain", "evidence": {"visible": []}}
        for text in (json.dumps(expected), "```json\n" + json.dumps(expected) + "\n```",
                     "Result:\n" + json.dumps(expected)):
            with self.subTest(text=text):
                self.assertEqual(client.parse_json_response(text), expected)

    def test_conflicting_complete_objects_are_rejected(self):
        texts = [
            '{"decision":"keep"} {"decision":"exclude"}',
            '```json\n{"decision":"keep"}\n```\n```json\n{"decision":"exclude"}\n```',
        ]
        for text in texts:
            with self.subTest(text=text), self.assertRaises(ValueError):
                client.parse_json_response(text)

    def test_malformed_outer_object_cannot_promote_nested_object(self):
        with self.assertRaises(ValueError):
            client.parse_json_response('{"decision":"keep", "evidence":{"confidence":0.9}')

    def test_array_cannot_promote_nested_object(self):
        with self.assertRaises(ValueError):
            client.parse_json_response('[{"decision":"keep"}]')

    def test_duplicate_keys_are_rejected_at_any_depth(self):
        for text in ('{"decision":"keep","decision":"exclude"}',
                     '{"evidence":{"confidence":0.1,"confidence":0.99}}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                client.parse_json_response(text)

    def test_nonfinite_constants_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                client.parse_json_response('{"confidence":' + value + '}')


class CredentialAndImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_example_is_never_executed_and_env_is_supported(self):
        marker = self.root / "must_not_exist"
        example = self.root / "credentials.py"
        example.write_text(
            "import os\nopen(" + repr(str(marker)) + ", 'w').write('executed')\n"
            "client=Anthropic(base_url='https://example.invalid', api_key=os.getenv('REVIEW_TEST_KEY'))\n"
        )
        with patch.dict(os.environ, {"REVIEW_TEST_KEY": "test-only-secret"}):
            credentials = client.credentials_from_example(example)
        self.assertEqual(credentials.api_key, "test-only-secret")
        self.assertNotIn("test-only-secret", repr(credentials))
        self.assertFalse(marker.exists())

    def test_arbitrary_credential_expression_is_rejected(self):
        example = self.root / "credentials.py"
        example.write_text("client=Anthropic(base_url='https://example.invalid', api_key=fetch_secret())")
        with self.assertRaises(ValueError):
            client.credentials_from_example(example)

    def test_original_bytes_hash_and_mismatch(self):
        image_path = self.root / "image.jpg"
        Image.new("RGB", (64, 32), "red").save(image_path)
        sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        _, metadata = client.prepare_image(image_path, sha)
        self.assertEqual(metadata["original_sha256"], metadata["sent_sha256"])
        self.assertEqual(metadata["transform_reasons"], [])
        with self.assertRaises(ValueError):
            client.prepare_image(image_path, "0" * 64)

    def test_transparency_is_composited_and_tiny_image_is_padded(self):
        image_path = self.root / "image.png"
        Image.new("RGBA", (2, 4), (255, 0, 0, 128)).save(image_path)
        _, metadata = client.prepare_image(image_path, hashlib.sha256(image_path.read_bytes()).hexdigest())
        self.assertEqual((metadata["sent_width"], metadata["sent_height"]), (16, 16))
        self.assertIn("transparent_composited_on_white", metadata["transform_reasons"])
        self.assertIn("small_dimension_white_padding", metadata["transform_reasons"])

    def test_large_image_is_resized_and_hash_is_recorded(self):
        image_path = self.root / "image.png"
        Image.new("RGB", (5000, 40), "white").save(image_path)
        _, metadata = client.prepare_image(image_path, hashlib.sha256(image_path.read_bytes()).hexdigest())
        self.assertEqual(metadata["sent_width"], 2048)
        self.assertLessEqual(metadata["sent_bytes"], 4 * 1024 * 1024)
        self.assertNotEqual(metadata["original_sha256"], metadata["sent_sha256"])

    def test_optional_small_detail_preserves_original_and_enlarges_whole_image(self):
        image_path = self.root / "tiny.png"
        original = Image.new("RGB", (26, 44), "white")
        original.putpixel((0, 0), (0, 0, 0))
        original.save(image_path)
        sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        payload, metadata = client.prepare_image(image_path, sha, small_image_detail=True)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 2)
        self.assertEqual(base64.b64decode(payload[0]["source"]["data"]), image_path.read_bytes())
        detail_bytes = base64.b64decode(payload[1]["source"]["data"])
        detail = Image.open(io.BytesIO(detail_bytes))
        expected = original.resize((round(26 * 512 / 44), 512), Image.Resampling.NEAREST)
        self.assertEqual(detail.tobytes(), expected.tobytes())
        self.assertFalse(metadata["detail_view"]["cropped"])
        self.assertEqual(metadata["detail_view"]["sent_sha256"], hashlib.sha256(detail_bytes).hexdigest())
        default_payload, default_metadata = client.prepare_image(image_path, sha)
        self.assertIsInstance(default_payload, dict)
        self.assertFalse(default_metadata["detail_view"]["enabled"])

    def test_optional_detail_is_not_added_to_large_image(self):
        image_path = self.root / "normal.png"
        Image.new("RGB", (256, 128), "white").save(image_path)
        sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        payload, metadata = client.prepare_image(image_path, sha, small_image_detail=True)
        self.assertIsInstance(payload, dict)
        self.assertFalse(metadata["detail_view"]["eligible"])
        self.assertFalse(metadata["detail_view"]["enabled"])


class ClientResponseTests(unittest.TestCase):
    def make_client(self, create):
        review_client = client.ReviewClient.__new__(client.ReviewClient)
        review_client._secret = "test-only-secret"
        review_client._client = SimpleNamespace(messages=SimpleNamespace(create=create))
        return review_client

    def invoke(self, review_client):
        return asyncio.run(review_client.review(model="test-model", image_payload={"type": "image"},
                                               prompt="Classify image", max_tokens=4000, thinking_budget=1600))

    def response(self, text='{"decision":"uncertain"}', stop_reason="end_turn"):
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking="PRIVATE_THINKING_MUST_NOT_BE_SAVED"),
                     SimpleNamespace(type="text", text=text)],
            model="test-model", stop_reason=stop_reason,
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 10, "output_tokens": 20}),
        )

    def test_thinking_is_omitted_and_echoed_key_is_redacted(self):
        create = AsyncMock(return_value=self.response('{"note":"test-only-secret"}'))
        result = self.invoke(self.make_client(create))
        encoded = json.dumps(result)
        self.assertNotIn("PRIVATE_THINKING", encoded)
        self.assertNotIn("test-only-secret", encoded)
        self.assertEqual(result["parsed"]["note"], "<redacted>")
        sent = create.call_args.kwargs
        self.assertEqual([part["type"] for part in sent["messages"][0]["content"]], ["image", "text"])

    def test_max_tokens_and_unparseable_output_are_not_success(self):
        for response in (self.response(stop_reason="max_tokens"), self.response("not json")):
            with self.subTest(response=response.stop_reason):
                result = self.invoke(self.make_client(AsyncMock(return_value=response)))
                self.assertNotEqual(result["status"], "ok")

    def test_two_image_payload_order_precedes_text(self):
        create = AsyncMock(return_value=self.response())
        review_client = self.make_client(create)
        images = [{"type": "image", "test_order": "original"},
                  {"type": "image", "test_order": "nearest_detail"}]
        result = asyncio.run(review_client.review(model="test-model", image_payload=images,
            prompt="Review both views", max_tokens=4000, thinking_budget=1600))
        self.assertEqual(result["status"], "ok")
        sent = create.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(sent[:2], images)
        self.assertEqual(sent[2]["type"], "text")

    def test_error_does_not_serialize_request_or_secret(self):
        request = httpx.Request("POST", "https://example.invalid", headers={"x-api-key": "test-only-secret"})
        response = httpx.Response(400, request=request)
        exception = anthropic.BadRequestError("test-only-secret request dump", response=response,
                                              body={"error": "test-only-secret"})
        result = self.invoke(self.make_client(AsyncMock(side_effect=exception)))
        self.assertEqual(result["status"], "api_error")
        self.assertEqual(result["http_status"], 400)
        self.assertNotIn("test-only-secret", json.dumps(result))

    def test_retries_are_bounded_and_only_transient_errors_retry(self):
        request = httpx.Request("POST", "https://example.invalid")
        transient = anthropic.RateLimitError("rate limit", response=httpx.Response(429, request=request), body={})
        create = AsyncMock(side_effect=transient)
        with patch.object(client.asyncio, "sleep", new=AsyncMock()):
            result = self.invoke(self.make_client(create))
        self.assertEqual(create.call_count, 3)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["status"], "api_error")


class RunnerIntegrityTests(unittest.TestCase):
    def test_conflicting_duplicate_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_path = root / "image.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            common = {"image_sha256": sha, "image_path": str(image_path)}
            rows = [{**common, "review_context": {"source": "A"}},
                    {**common, "review_context": {"source": "B"}}]
            manifest = root / "samples.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(ValueError):
                run_review.load_samples(manifest, "review_context")
            self.assertEqual(len(run_review.load_samples(manifest)), 1)

    def test_blind_context_cache_and_interrupted_journal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_path = root / "image.jpg"
            Image.new("RGB", (64, 32), "red").save(image_path)
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            example = root / "credentials.py"
            example.write_text("client=Anthropic(base_url='https://example.invalid', api_key='test-only-secret')")
            prompt = root / "prompt.txt"
            prompt.write_text("Review the image.")
            manifest = root / "samples.jsonl"
            manifest.write_text(json.dumps({
                "image_sha256": sha, "image_path": str(image_path), "review_id": "X",
                "source": "DO_NOT_SEND_THIS_SOURCE", "review_context": {"note": "EXPLICIT_CONTEXT"},
            }) + "\n")
            requests = []

            class FakeClient:
                def __init__(self, *args, **kwargs):
                    pass

                async def close(self):
                    pass

                async def review(self, **kwargs):
                    requests.append(kwargs)
                    return {"status": "ok", "parsed": {"decision": "uncertain"},
                            "text": '{"decision":"uncertain"}', "usage": {}}

            args = argparse.Namespace(
                credential_example=str(example), samples=str(manifest), output=str(root / "blind"),
                models=["test-model"], prompt_file=str(prompt), concurrency=2, max_tokens=4000,
                thinking_budget=1600, timeout=180, round="blind", context_field=None,
            )
            with patch.object(run_review, "ReviewClient", FakeClient):
                self.assertEqual(asyncio.run(run_review.run(args)), 0)
                args.output = str(root / "blind")
                args.context_field = None
                args.round = "blind"
                example.write_text("client=Anthropic(base_url='https://other-route.invalid', api_key='test-only-secret')")
                with self.assertRaises(ValueError):
                    asyncio.run(run_review.run(args))
                example.write_text("client=Anthropic(base_url='https://example.invalid', api_key='test-only-secret')")
                self.assertEqual(asyncio.run(run_review.run(args)), 0)
                self.assertEqual(len(requests), 1)
                self.assertNotIn("EXPLICIT_CONTEXT", requests[0]["prompt"])
                self.assertNotIn("DO_NOT_SEND_THIS_SOURCE", requests[0]["prompt"])
                args.output = str(root / "context")
                args.context_field = "review_context"
                args.round = "adjudicate"
                self.assertEqual(asyncio.run(run_review.run(args)), 0)
            self.assertEqual(len(requests), 2)
            self.assertIn("EXPLICIT_CONTEXT", requests[1]["prompt"])
            self.assertNotIn("DO_NOT_SEND_THIS_SOURCE", requests[1]["prompt"])
            blind_row = json.loads((root / "blind/results.jsonl").read_text())
            context_row = json.loads((root / "context/results.jsonl").read_text())
            self.assertNotEqual(blind_row["cache_key"], context_row["cache_key"])
            journal = root / "blind/responses.journal.jsonl"
            with journal.open("ab") as stream:
                stream.write(b'{"interrupted')
            self.assertEqual(len(run_review.load_journal(journal)), 1)
            self.assertTrue(journal.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
