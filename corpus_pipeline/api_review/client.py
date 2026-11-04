"""Conservative VLM review client; credentials never enter result artifacts."""
from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import io
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import httpx
from PIL import Image, ImageOps

from corpus_pipeline.api_review.strict_json import parse_json_response


PREPROCESSING = {
    "version": "vlm_original_or_png_v2",
    "original_max_bytes": 4 * 1024 * 1024,
    "original_max_edge": 4096,
    "converted_max_edge": 2048,
    "minimum_edge": 16,
    "transparent_background": "white",
    "animated": "first_frame",
    "large_png": "iteratively_shrink_until_within_byte_limit",
    "small_image_detail": {"eligibility_max_edge_exclusive": 256, "target_max_edge": 512,
                           "interpolation": "NEAREST", "scope": "complete_original_image"},
}


@dataclass(frozen=True, repr=False)
class Credentials:
    base_url: str
    api_key: str

    def __repr__(self) -> str:
        return "Credentials(<redacted>)"


def credentials_from_example(path: str | Path) -> Credentials:
    """Read supported literal/environment AST nodes, never execute the example."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments[node.target.id] = node.value

    def evaluate(node: ast.AST, active: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, type(None))):
            return node.value
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in active:
            return evaluate(assignments[node.id], active | {node.id})
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for item in node.values:
                value = evaluate(item, active)
                if value:
                    return value
            return value
        if isinstance(node, ast.Call):
            function = ast.unparse(node.func)
            if function in ("os.getenv", "os.environ.get", "getenv") and not node.keywords:
                if 1 <= len(node.args) <= 2:
                    key = evaluate(node.args[0], active)
                    default = evaluate(node.args[1], active) if len(node.args) == 2 else None
                    if isinstance(key, str):
                        return os.environ.get(key, default)
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            key = evaluate(node.slice, active)
            if isinstance(key, str):
                return os.environ.get(key)
        raise ValueError("Unsupported credential expression; use literals or environment lookups")

    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else ""
        )
        if name in ("Anthropic", "AsyncAnthropic"):
            kwargs = {k.arg: k.value for k in node.keywords}
            if "base_url" in kwargs and "api_key" in kwargs:
                candidates.append(kwargs)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one Anthropic constructor with base_url and api_key")
    base_url = evaluate(candidates[0]["base_url"])
    api_key = evaluate(candidates[0]["api_key"])
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValueError("Credential base_url is missing or invalid")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Credential API key is missing; configure the example's environment variable")
    return Credentials(base_url, api_key)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def prepare_image(image_path: str | Path, expected_sha256: str, *,
                  small_image_detail: bool = False) -> tuple[dict | list[dict], dict]:
    original = Path(image_path).read_bytes()
    sha = hashlib.sha256(original).hexdigest()
    if sha != expected_sha256:
        raise ValueError("Image SHA256 does not match manifest")
    image = Image.open(io.BytesIO(original))
    image.load()
    original_size, original_mode, original_format = image.size, image.mode, image.format
    animated = bool(getattr(image, "is_animated", False))
    transparent = "A" in image.getbands() or "transparency" in image.info
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
    reasons = []
    if len(original) > PREPROCESSING["original_max_bytes"]:
        reasons.append("original_over_4MiB")
    if max(image.size) > PREPROCESSING["original_max_edge"]:
        reasons.append("original_long_edge_over_4096")
    if min(image.size) < PREPROCESSING["minimum_edge"]:
        reasons.append("small_dimension_white_padding")
    if transparent:
        reasons.append("transparent_composited_on_white")
    if animated:
        reasons.append("animated_first_frame")
    if original_format not in mime:
        reasons.append("unsupported_original_format")
    sent, media_type, sent_size = original, mime.get(original_format), original_size
    orientation_changed = False
    shrink_steps = 0
    if reasons:
        orientation = image.getexif().get(274, 1)
        image = ImageOps.exif_transpose(image)
        orientation_changed = orientation not in (None, 1)
        if transparent:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        if len(original) > PREPROCESSING["original_max_bytes"] or max(original_size) > 4096:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        if min(image.size) < 16:
            width, height = max(16, image.width), max(16, image.height)
            padded = Image.new("RGB", (width, height), "white")
            padded.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            image = padded
        while True:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            sent = buf.getvalue()
            if len(sent) <= PREPROCESSING["original_max_bytes"]:
                break
            width, height = max(16, int(image.width * .85)), max(16, int(image.height * .85))
            if (width, height) == image.size:
                raise ValueError("Could not satisfy image payload byte limit")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            shrink_steps += 1
        sent_size, media_type = image.size, "image/png"
    metadata = {
        "preprocessing": PREPROCESSING, "original_sha256": sha,
        "sent_sha256": hashlib.sha256(sent).hexdigest(),
        "original_bytes": len(original), "sent_bytes": len(sent),
        "original_width": original_size[0], "original_height": original_size[1],
        "sent_width": sent_size[0], "sent_height": sent_size[1],
        "original_mode": original_mode, "original_format": original_format,
        "sent_media_type": media_type, "transform_reasons": reasons,
        "converted_exif_orientation_applied": orientation_changed,
        "additional_png_shrink_steps": shrink_steps,
    }
    payload = {"type": "image", "source": {"type": "base64", "media_type": media_type,
                "data": base64.b64encode(sent).decode("ascii")}}
    detail_eligible = max(original_size) < 256
    metadata["detail_view"] = {"requested": small_image_detail, "eligible": detail_eligible,
                               "enabled": False}
    if small_image_detail and detail_eligible:
        detail = ImageOps.exif_transpose(Image.open(io.BytesIO(original))).convert("RGBA")
        background = Image.new("RGBA", detail.size, "white")
        background.alpha_composite(detail)
        detail = background.convert("RGB")
        scale = 512 / max(detail.size)
        target_size = (max(1, round(detail.width * scale)), max(1, round(detail.height * scale)))
        detail = detail.resize(target_size, Image.Resampling.NEAREST)
        detail_buffer = io.BytesIO()
        detail.save(detail_buffer, format="PNG")
        detail_bytes = detail_buffer.getvalue()
        detail_payload = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
            "data": base64.b64encode(detail_bytes).decode("ascii")}}
        payload = [payload, detail_payload]
        metadata["detail_view"].update({
            "enabled": True, "image_order_zero_based": 1,
            "original_sha256": sha, "sent_sha256": hashlib.sha256(detail_bytes).hexdigest(),
            "sent_width": detail.width, "sent_height": detail.height,
            "sent_bytes": len(detail_bytes), "sent_media_type": "image/png",
            "transform": "complete_original_image_nearest_enlargement_to_max_edge_512",
            "cropped": False, "interpolation": "NEAREST", "transparent_background": "white",
        })
    return payload, metadata


def safe_error(exc: Exception) -> dict:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None
    kind = type(exc).__name__
    descriptions = {
        "AuthenticationError": "API authentication failed",
        "PermissionDeniedError": "API permission denied",
        "RateLimitError": "API rate limit reached",
        "BadRequestError": "API rejected request parameters or model capability",
        "NotFoundError": "API model or endpoint not found",
        "APITimeoutError": "API request timed out",
        "APIConnectionError": "API connection failed",
        "ValueError": "Local validation failed",
        "FileNotFoundError": "Local input file missing",
        "UnidentifiedImageError": "Local image could not be decoded",
    }
    return {"error_type": kind, "http_status": status,
            "error_summary": descriptions.get(kind, "Request or local processing failed")}


class ReviewClient:
    def __init__(self, credentials: Credentials, timeout: float = 180):
        self._secret = credentials.api_key
        self._http = httpx.AsyncClient(trust_env=False, timeout=timeout)
        self._client = anthropic.AsyncAnthropic(
            base_url=credentials.base_url, api_key=credentials.api_key,
            http_client=self._http, max_retries=0, timeout=timeout,
        )

    async def close(self):
        await self._client.close()

    async def review(self, *, model: str, image_payload: dict | list[dict], prompt: str,
                     max_tokens: int, thinking_budget: int) -> dict:
        image_parts = image_payload if isinstance(image_payload, list) else [image_payload]
        if not image_parts or any(part.get("type") != "image" for part in image_parts):
            raise ValueError("Expected one or more image content blocks")
        kwargs = {"model": model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": [*image_parts,
                      {"type": "text", "text": prompt}]}]}
        if thinking_budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        for attempt in range(3):
            try:
                response = await self._client.messages.create(**kwargs)
                text = "\n".join(block.text for block in response.content
                                 if getattr(block, "type", None) == "text")
                text = text.replace(self._secret, "<redacted>")
                result = {"status": "ok", "text": text[:20000],
                          "text_truncated": len(text) > 20000,
                          "response_model": response.model,
                          "finish_reason": response.stop_reason,
                          "usage": response.usage.model_dump(mode="json"),
                          "attempts": attempt + 1}
                try:
                    result["parsed"] = parse_json_response(text)
                    if response.stop_reason == "max_tokens":
                        result["status"] = "truncated_response"
                except ValueError:
                    result["status"] = "format_error"
                    result["error_summary"] = "No unambiguous JSON object in text response"
                return result
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = status == 429 or (isinstance(status, int) and status >= 500)
                retryable = retryable or isinstance(exc, (anthropic.APITimeoutError, httpx.TimeoutException))
                if retryable and attempt < 2:
                    await asyncio.sleep(min(20, 2 ** (attempt + 1)) + random.random())
                    continue
                return {"status": "api_error", **safe_error(exc), "attempts": attempt + 1}
        raise AssertionError("Unreachable")
