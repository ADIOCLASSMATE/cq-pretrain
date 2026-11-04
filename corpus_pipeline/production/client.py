"""Bounded dynamic-endpoint client; only validated successful results are cached."""
from __future__ import annotations
import asyncio
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import httpx
from PIL import Image, ImageOps

from .common import atomic_json, canonical, digest, file_sha
from .records import read_asset_bytes
from .http_transport import BufferedTransport, versions as transport_versions
from .native_transport import (
    is_native, native_endpoint_valid, platform_snapshot, native_receipt_errors,
    assemble_openai_sse, PLATFORM_SCOPE, TRANSPORT_CONTRACT_FIELDS, TRANSPORT_IDENTITY_FIELDS,
)


class ModelStageError(RuntimeError):
    def __init__(self, stage, errors, *, retryable_after_endpoint_change=False):
        self.stage = stage
        self.errors = errors
        self.retryable_after_endpoint_change = retryable_after_endpoint_change
        self.failure_kind = "transient_transport_or_unavailable" if retryable_after_endpoint_change else "model_output_or_configuration"
        super().__init__("model_stage_failed:" + stage + ":" + canonical(errors))


class EndpointStateChanged(RuntimeError):
    """A transport incarnation changed after readiness was checked."""


def endpoint_lease_valid(endpoint):
    expiry = endpoint.get("ready_expires_at")
    if endpoint.get("backend") == "inspire_serving":
        if is_native(endpoint):
            if not native_endpoint_valid(endpoint):
                return False
        else:
            if endpoint.get("transport_mode") not in (None, "boot_header"):
                return False
            boots = endpoint.get("expected_boots")
            if not isinstance(boots, list) or not boots or any(not isinstance(value, str) or not value for value in boots):
                return False
        if not isinstance(expiry, (int, float)) or isinstance(expiry, bool):
            return False
    return expiry is None or isinstance(expiry, (int, float)) and not isinstance(expiry, bool) and math.isfinite(expiry) and expiry > time.time()


def parse_json(text):
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def request_overrides(endpoint):
    """Apply the same qualified preprocessing and decoding settings as the probe."""
    allowed = {"chat_template_kwargs", "reasoning_effort", "temperature", "top_p", "top_k", "presence_penalty",
               "mm_processor_kwargs"}
    overrides = {key: value for key, value in endpoint.get("request_overrides", {}).items() if key in allowed}
    processor = overrides.get("mm_processor_kwargs")
    if processor is not None:
        if not isinstance(processor, dict) or set(processor) - {"min_pixels", "max_pixels"}:
            raise ValueError("unsupported qualified image processor override")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in processor.values()):
            raise ValueError("qualified image pixel bounds must be positive integers")
        if processor.get("min_pixels", 1) > processor.get("max_pixels", float("inf")):
            raise ValueError("qualified image minimum pixels exceeds maximum")
    return overrides


def image_payload(asset, max_side=2048):
    raw = read_asset_bytes(asset["original_ref"])
    with Image.open(io.BytesIO(raw)) as original:
        before = original.size
        orientation = original.getexif().get(274, 1)
        has_alpha = original.mode in ("RGBA", "LA", "P") or "transparency" in original.info
        if original.format in ("JPEG", "PNG") and not has_alpha and orientation in (None, 1) and max(before) <= max_side and len(raw) <= 8 << 20:
            value, size, media = raw, before, "image/jpeg" if original.format == "JPEG" else "image/png"
            transformation = "original_bytes_unchanged"
        else:
            original.load()
            picture = ImageOps.exif_transpose(original)
            if has_alpha:
                rgba = picture.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                picture = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                picture = picture.convert("RGB")
            picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            picture.save(output, format="PNG")
            value, size, media = output.getvalue(), picture.size, "image/png"
            transformation = "full_image_exif_white_alpha_resize_if_needed_lossless_png"
    return {"type": "image_url", "image_url": {"url": "data:" + media + ";base64," + base64.b64encode(value).decode()}}, {
        "asset_id": asset["asset_id"], "original_sha256": asset["sha256"], "sent_sha256": hashlib.sha256(value).hexdigest(),
        "original_dimensions": before, "sent_dimensions": size, "crop": False, "media_type": media,
        "transform": transformation, "alpha_background": "white" if has_alpha else None}


class DynamicClient:
    def __init__(self, registry, cache_dir, *, concurrency=6, timeout=300, wait_ready=3600):
        self.registry = Path(registry)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.wait_ready = wait_ready
        self._flights = {}
        self.processing_code_identity = {name: file_sha(Path(__file__).with_name(name))
                                         for name in ("client.py", "validation.py", "native_transport.py", "http_transport.py")}
        self.processing_code_identity['http_dependencies'] = transport_versions()
        self.client = BufferedTransport(httpx.AsyncClient(timeout=timeout, trust_env=False,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)), concurrency, timeout)

    async def close(self):
        await self.client.aclose()

    def endpoint(self, role):
        registry = json.loads(self.registry.read_text())
        items = [entry for entry in registry["endpoints"] if entry["role"] == role and entry.get("ready") and endpoint_lease_valid(entry)]
        if len(items) != 1:
            raise LookupError("exactly one ready endpoint required for " + role)
        endpoint = items[0]
        if endpoint["protocol"] != "openai":
            raise ValueError("unsupported local model protocol")
        return endpoint, registry.get("registry_generation")

    async def request(self, role, stage, context, assets, prompt, validator, *, max_tokens=3000):
        try:
            endpoint, _ = self.endpoint(role)
            serving_identity = {name: endpoint.get(name) for name in ("model", "revision", "processor_revision",
                                "runtime_revision", "precision", "request_overrides", "max_output_tokens", *TRANSPORT_CONTRACT_FIELDS)}
        except (OSError, ValueError, LookupError, KeyError):
            serving_identity = {"pending_verified_endpoint": role}
        key = digest({"role": role, "stage": stage, "context": context, "images": [a["sha256"] for a in assets],
                      "prompt": prompt, "max_tokens": max_tokens, "serving_identity": serving_identity,
                      "processing_code_identity": self.processing_code_identity})
        task = self._flights.get(key)
        if task is None:
            task = asyncio.create_task(self._request(role, stage, context, assets, prompt, validator, max_tokens=max_tokens))
            self._flights[key] = task
            def finished(completed):
                if self._flights.get(key) is completed:
                    self._flights.pop(key, None)
                if not completed.cancelled():
                    completed.exception()  # Retrieve failures even if every awaiting caller was cancelled.
            task.add_done_callback(finished)
        result = await asyncio.shield(task)
        validator(result["parsed"])
        return result

    async def _request(self, role, stage, context, assets, prompt, validator, *, max_tokens=3000):
        deadline = time.monotonic() + self.wait_ready
        errors = []
        token_budget = max_tokens
        for attempt in range(4):
            while True:
                try:
                    endpoint, generation = self.endpoint(role)
                    break
                except (OSError, ValueError, LookupError, KeyError):
                    if time.monotonic() >= deadline:
                        raise ModelStageError(stage, [{"type": "EndpointNotReady", "role": role}],
                                              retryable_after_endpoint_change=True) from None
                    await asyncio.sleep(3)
            if assets and not endpoint.get("supports_images"):
                raise ValueError("endpoint does not advertise image support")
            if assets and len(assets) > endpoint.get("max_images", 32):
                raise ValueError("complete record exceeds endpoint image capacity; no images dropped")
            overrides = request_overrides(endpoint)
            token_budget = min(token_budget, endpoint.get("max_output_tokens", 12000))
            identity = {"stage": stage, "role": role, "model": endpoint["model"], "revision": endpoint.get("revision"),
                        "precision": endpoint.get("precision"), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "context_sha256": digest(context), "assets": [a["sha256"] for a in assets],
                        "preprocessor": "original_bytes_or_full_image_lossless_white_alpha_max2048.v2", "max_tokens": token_budget,
                        "processor_revision": endpoint.get("processor_revision"), "runtime_revision": endpoint.get("runtime_revision"),
                        "request_overrides": overrides, "processing_code_identity": self.processing_code_identity}
            identity.update({name: endpoint.get(name) for name in TRANSPORT_CONTRACT_FIELDS})
            key = digest(identity)
            target = self.cache / (key + ".json")
            if target.exists():
                cached = json.loads(target.read_text())
                if cached.get("status") == "ok" and cached.get("identity") == identity:
                    validator(cached["parsed"])
                    if is_native(endpoint) and native_receipt_errors(cached):
                        raise ValueError("cached_native_platform_receipt_invalid")
                    return {**cached, "cache_hit": True}
            async with self.semaphore:
                response_text = None
                finish_reason = None
                try:
                    prepared = await asyncio.gather(*(asyncio.to_thread(image_payload, asset) for asset in assets))
                    content = [{"type": "text", "text": prompt + "\nINPUT_JSON:\n" + canonical(context)}]
                    for index, (payload, _) in enumerate(prepared):
                        content.extend([{"type": "text", "text": f"Input image index {index}:"}, payload])
                    try:
                        current, _ = self.endpoint(role)
                    except (OSError, ValueError, KeyError, LookupError):
                        raise EndpointStateChanged("endpoint readiness expired before request") from None
                    transport_fields = TRANSPORT_IDENTITY_FIELDS
                    if any(current.get(name) != endpoint.get(name) for name in transport_fields):
                        raise EndpointStateChanged("endpoint identity changed before request")
                    native = is_native(endpoint)
                    native_before = platform_snapshot(current, time.time()) if native else None
                    headers = {}
                    key_env = endpoint.get("api_key_env")
                    if key_env:
                        token = os.environ.get(key_env)
                        if not token:
                            raise RuntimeError("required_endpoint_credential_unavailable")
                        headers["Authorization"] = "Bearer " + token
                    body = {"model": endpoint["model"], "messages": [{"role": "user", "content": content}],
                            "max_tokens": token_budget, "temperature": 0.1, **overrides}
                    if native and endpoint.get("native_stream"):
                        body.update(stream=True, stream_options={"include_usage": True})
                    if native:
                        # HTTPX read timeout resets per received SSE chunk. Keep
                        # the whole native request bounded for graceful drain.
                        async with asyncio.timeout(self.timeout):
                            response = await self.client.post(endpoint["base_url"].rstrip("/") + "/chat/completions", headers=headers, json=body)
                    else:
                        response = await self.client.post(endpoint["base_url"].rstrip("/") + "/chat/completions", headers=headers, json=body)
                    response.raise_for_status()
                    native_after = None
                    if native:
                        try:
                            after, _ = self.endpoint(role)
                        except (OSError, ValueError, KeyError, LookupError):
                            raise EndpointStateChanged("platform lease unavailable after native response") from None
                        if any(after.get(name) != current.get(name) for name in transport_fields):
                            raise EndpointStateChanged("platform identity changed during native request")
                        native_after = platform_snapshot(after, time.time())
                    response_boot = None if native else response.headers.get("x-umm-boot") or response.headers.get("x-umm-boot-id")
                    if not native and endpoint.get("expected_boots") and response_boot not in endpoint["expected_boots"]:
                        raise EndpointStateChanged("response came from an unqualified model boot")
                    result = assemble_openai_sse(response.text, endpoint["model"]) if native and endpoint.get("native_stream") else response.json()
                    if (native and result.get("model") != endpoint["model"]
                            or result.get("model") and result["model"] != endpoint["model"]):
                        raise ValueError("response model identity differs from selected endpoint")
                    choice = result["choices"][0]
                    response_text = choice["message"].get("content")
                    finish_reason = choice.get("finish_reason")
                    if choice.get("finish_reason") == "length":
                        ceiling = min(12000, endpoint.get("max_output_tokens", 12000))
                        if token_budget >= ceiling:
                            raise RuntimeError("truncated_at_maximum_output_budget")
                        token_budget = min(token_budget * 2, ceiling)
                        raise ValueError("truncated_model_response")
                    text = choice["message"]["content"]
                    parsed = parse_json(text)
                    validator(parsed)
                    row = {"status": "ok", "identity": identity, "cache_key": key, "parsed": parsed, "request_context": context,
                           "text": text, "usage": result.get("usage"), "response_model": result.get("model"),
                           "registry_generation": generation, "endpoint_identity_sha256": digest(endpoint["base_url"]),
                           "response_boot_id": response_boot, "qualified_boots_at_request": endpoint.get("expected_boots"),
                           "readiness_lease_until_at_request": current.get("ready_expires_at"),
                           "image_inputs": [metadata for _, metadata in prepared], "attempt": attempt + 1,
                           "cache_hit": False, "real_model_inference": True}
                    if native:
                        row.update(response_boot_attested=False, attestation_scope=PLATFORM_SCOPE,
                                   platform_identity_at_request=native_before, platform_identity_at_response=native_after,
                                   native_stream_chunks=result.get("native_stream_chunks"))
                        if native_receipt_errors(row):
                            raise EndpointStateChanged("native platform receipt consistency check failed")
                    atomic_json(target, row)
                    return row
                except Exception as error:
                    # Do not persist HTTP bodies, headers, tokens, or base64 requests.
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    transient = isinstance(error, (httpx.TransportError, EndpointStateChanged, TimeoutError)) or isinstance(status, int) and (status == 429 or status >= 500)
                    errors.append({"type": type(error).__name__, "status_code": status, "transient": transient})
                    atomic_json(self.cache / (key + ".failure.json"), {"status": "failed", "identity": identity,
                        "attempts": errors, "validation_error": str(error)[:300] if isinstance(error, ValueError) else None,
                        "response_text": response_text[:20000] if isinstance(response_text, str) else None,
                        "finish_reason": finish_reason})
                    if isinstance(error, RuntimeError) and str(error) == "truncated_at_maximum_output_budget":
                        break
            await asyncio.sleep(min(2 ** attempt, 8))
        raise ModelStageError(stage, errors, retryable_after_endpoint_change=bool(errors) and all(error["transient"] for error in errors))
