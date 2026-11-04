"""Paid Qwen transport implementing the same audited request interface as Serving.

Credentials are parsed, never executed or written to receipts. This backend is
an explicit option, never an automatic fallback from an unavailable deployment.
"""
from __future__ import annotations
import asyncio
from collections import Counter
from pathlib import Path
import json

from corpus_pipeline.api_benchmark.bench import BenchClient, retry_delay
from corpus_pipeline.api_review.client import credentials_from_example
from .client import image_payload, parse_json, ModelStageError
from .common import atomic_json, canonical, digest, file_sha


class QwenAPIClient:
    def __init__(self, credential_example, cache_dir, *, generator="qwen3.8-max",
                 verifier="qwen3.8-flash", concurrency=16, timeout=120, retries=2):
        if any(not model.lower().startswith("qwen") for model in (generator, verifier)):
            raise ValueError("This explicitly selected API backend accepts Qwen models only")
        self.models = {"generator": generator, "verifier": verifier}
        credentials = credentials_from_example(credential_example)
        self.endpoint_identity = digest(credentials.base_url)
        self.transport = BenchClient(credentials, timeout=timeout, concurrency=concurrency)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.flights = {}
        self.stats = Counter()

    async def close(self):
        await self.transport.close()

    async def request(self, role, stage, context, assets, prompt, validator, *, max_tokens=3000):
        identity = {"stage": stage, "role": role, "model": self.models[role],
                    "revision": "provider_alias_unpinned", "precision": "provider_undisclosed",
                    "endpoint_identity": self.endpoint_identity,
                    "prompt_sha256": digest(prompt), "context_sha256": digest(context),
                    "assets": [asset["sha256"] for asset in assets], "max_tokens": max_tokens,
                    "preprocessor": "original_bytes_or_full_image_lossless_white_alpha_max2048.v2",
                    "thinking": "disabled", "temperature": 0.1,
                    "processing_code_identity": {p.name: file_sha(p) for p in
                        (Path(__file__), Path(__file__).with_name("client.py"), Path(__file__).with_name("validation.py"))}}
        key = digest(identity)
        if key not in self.flights:
            task = asyncio.create_task(self._request(key, identity, context, assets, prompt, validator, max_tokens))
            self.flights[key] = task
        task = self.flights[key]
        try:
            result = await asyncio.shield(task)
            validator(result["parsed"])
            return result
        finally:
            if task.done():
                self.flights.pop(key, None)

    async def _request(self, key, identity, context, assets, prompt, validator, max_tokens):
        path = self.cache / (key + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("identity") == identity and cached.get("status") == "ok":
                validator(cached["parsed"])
                self.stats["cache_hits"] += 1
                return {**cached, "cache_hit": True}
        errors = []
        async with self.semaphore:
            prepared = await asyncio.gather(*(asyncio.to_thread(image_payload, asset) for asset in assets))
            content = [{"type": "text", "text": prompt + "\nINPUT_JSON:\n" + canonical(context)}]
            for i, (payload, metadata) in enumerate(prepared):
                header, encoded = payload["image_url"]["url"].split(",", 1)
                content.extend([{"type": "text", "text": f"Input image index {i}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": header[5:].split(";", 1)[0], "data": encoded}}])
            for attempt in range(self.retries + 1):
                self.stats["http_attempts"] += 1
                self.stats["stage:" + identity["stage"]] += 1
                row = await self.transport.request(model=identity["model"], content=content, max_tokens=max_tokens,
                    thinking="disabled", thinking_budget=0, temperature=0.1)
                try:
                    if row["status"] != "response_received":
                        raise ValueError("transport_not_successful")
                    if row.get("response_model") != identity["model"]:
                        raise ValueError("unaccepted_response_model")
                    if row.get("stop_reason") in {"max_tokens", "refusal"} or row.get("response_text_truncated"):
                        raise ValueError("incomplete_response")
                    parsed = parse_json(row["response_text"])
                    validator(parsed)
                    result = {"status": "ok", "identity": identity, "cache_key": key, "parsed": parsed, "request_context": context,
                              "text": row["response_text"], "usage": row.get("usage"),
                              "response_model": row.get("response_model"), "image_inputs": [m for _, m in prepared],
                              "cache_hit": False, "real_model_inference": True, "attempt": attempt + 1,
                              "provider_alias_revision_not_pinned": True}
                    atomic_json(path, result)
                    self.stats["successful_responses"] += 1
                    return result
                except (ValueError, KeyError, TypeError) as error:
                    errors.append({"type": type(error).__name__, "status_code": row.get("http_status"),
                                   "reason": str(error)[:200]})
                    atomic_json(self.cache / (key + ".failure.json"), {"status": "failed", "identity": identity,
                        "errors": errors, "response_text": row.get("response_text"), "attempt": attempt + 1})
                    status = row.get("http_status")
                    if isinstance(status, int) and 400 <= status < 500 and status not in {408, 429}:
                        break
                    if attempt < self.retries:
                        await asyncio.sleep(retry_delay(row.get("retry_after"), attempt))
        self.stats["failed_requests"] += 1
        raise ModelStageError(identity["stage"], errors)


class DeferredClient:
    """Plan-only model execution: pending is explicit and cannot be published."""
    stats = {}

    async def request(self, role, stage, *args, **kwargs):
        raise ModelStageError(stage, [{"type": "ModelExecutionDeferred", "role": role}])

    async def close(self):
        pass
