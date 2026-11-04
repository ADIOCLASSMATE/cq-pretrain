"""Offline integration checks for benchmark dispatch, with no API requests."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from corpus_pipeline.api_review.client import Credentials
from . import bench


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_dispatch_has_fresh_ids_and_explicit_pool_reuse(self):
        calls, prepared = [], []
        state = {"active": 0, "peak": 0, "closed": False}

        class FakeClient:
            def __init__(self, credentials, *, timeout, concurrency):
                pass

            async def close(self):
                state["closed"] = True

            async def request(self, *, content, **kwargs):
                request_id = re.search(r"REQUEST_ID: (\S+)", content[-1]["text"]).group(1)
                calls.append(request_id)
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                start = time.time()
                await asyncio.sleep(.005)
                state["active"] -= 1
                return {"status": "response_received", "http_status": 200,
                        "start_unix": start, "latency_seconds": time.time() - start,
                        "stop_reason": "end_turn", "response_text_truncated": False,
                        "response_model": state.get("response_model", "fixture"),
                        "response_model_matches_request": True,
                        "response_text": json.dumps({"request_id": request_id,
                            "captions": {"short": "A test image.", "medium": "A simple test image.",
                                         "long": "A simple test image with limited visual details."},
                            "uncertainties": ["Fixture, not a quality assessment."]})}

        def prepare(path, sha):
            prepared.append(path)
            return {"type": "text", "text": "offline fixture image"}, {"original_sha256": sha}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": "api_benchmark.manifest.v1", "items": [
                {"sample_id": str(index), "images": [{"path": str(root / f"{index}.png"), "sha256": "a" * 64}]}
                for index in range(3)]}))
            args = SimpleNamespace(manifest=manifest, task="caption", offset=1, requests=5,
                output=root / "output", credential_example="not-executed.py", prompt_file=None,
                model="fixture", concurrency=2, retries=0, thinking="disabled", thinking_budget=1024,
                max_tokens=3000, temperature=None, timeout=2)
            with patch.object(bench, "BenchClient", FakeClient), patch.object(bench, "prepare_image", prepare), \
                 patch.object(bench, "credentials_from_example", return_value=Credentials("https://fixture.invalid", "fixture-secret")), \
                 redirect_stdout(io.StringIO()):
                code = await bench.run(args)
            rows = [json.loads(line) for line in (args.output / "requests.jsonl").read_text().splitlines()]
            config = json.loads((args.output / "run.json").read_text())
            summary = json.loads((args.output / "summary.json").read_text())
            ordered = sorted(rows, key=lambda row: row["logical_request_index"])
            self.assertEqual(code, 0)
            self.assertEqual(state["peak"], 2)
            self.assertTrue(state["closed"])
            self.assertEqual(len(calls), 5)
            self.assertEqual(len(set(calls)), 5)
            self.assertEqual(len(prepared), 3)  # Input bytes may be prepared once; answers cannot be cached.
            self.assertEqual([row["sample_id"] for row in ordered], ["1", "2", "0", "1", "2"])
            self.assertEqual([row["manifest_item_occurrence"] for row in ordered], [0, 0, 0, 1, 1])
            self.assertEqual(config["repeated_items"], 2)
            self.assertTrue(config["selection_wraps"])
            self.assertFalse(config["local_response_cache"])
            self.assertEqual(summary["effective_successful_requests"], 5)
            self.assertEqual(summary["effective_failure_rate"], 0)
            self.assertGreater(summary["wall_seconds"], .01)
            self.assertNotIn("fixture-secret", json.dumps([config, summary, rows]))
            state["response_model"] = "unconfigured-alias"
            args.output = root / "wrong_model"
            with patch.object(bench, "BenchClient", FakeClient), patch.object(bench, "prepare_image", prepare), \
                 patch.object(bench, "credentials_from_example", return_value=Credentials("https://fixture.invalid", "fixture-secret")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(await bench.run(args), 1)
            wrong = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(wrong["unaccepted_response_model_count"], 5)
            self.assertEqual(wrong["effective_successful_requests"], 0)
            args.output = root / "configured_alias"
            args.response_model_alias = ["unconfigured-alias"]
            with patch.object(bench, "BenchClient", FakeClient), patch.object(bench, "prepare_image", prepare), \
                 patch.object(bench, "credentials_from_example", return_value=Credentials("https://fixture.invalid", "fixture-secret")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(await bench.run(args), 0)


if __name__ == "__main__":
    unittest.main()
