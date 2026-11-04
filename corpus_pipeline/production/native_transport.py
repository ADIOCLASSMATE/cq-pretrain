"""Client-side official OpenAI transport, with explicitly limited attestation.

No server wrapper or custom response headers are used by native_openai. The
external platform observer supplies a short registry lease. Matching identities
before/after a request are platform/registry evidence, never a responding
process's boot attestation or evidence of an image's provenance.
"""
from __future__ import annotations

import json
import math
from .common import digest

NATIVE_MODE = "native_openai"
PLATFORM_SCOPE = "platform_instance_and_registry"
TRANSPORT_CONTRACT_FIELDS = ("transport_mode", "attestation_scope", "native_stream")
TRANSPORT_IDENTITY_FIELDS = ("base_url", "api_key_env", "incarnation", "expected_boots", "model", "revision",
    "processor_revision", "runtime_revision", "precision", "request_overrides", "max_output_tokens",
    "transport_mode", "attestation_scope", "native_stream", "platform_instance_id", "launch_contract_sha256")


def is_native(endpoint):
    return endpoint.get("transport_mode") == NATIVE_MODE


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def native_endpoint_valid(endpoint):
    return (is_native(endpoint) and endpoint.get("attestation_scope") == PLATFORM_SCOPE
            and isinstance(endpoint.get("platform_instance_id"), str) and bool(endpoint["platform_instance_id"])
            and isinstance(endpoint.get("incarnation"), str) and bool(endpoint["incarnation"])
            and sha(endpoint.get("launch_contract_sha256"))
            and endpoint.get("expected_boots") in (None, [])
            and type(endpoint.get("native_stream", False)) is bool)


def platform_snapshot(endpoint, checked_at):
    if not native_endpoint_valid(endpoint):
        raise ValueError("invalid_native_platform_identity")
    return {"platform_instance_id": endpoint["platform_instance_id"], "incarnation": endpoint["incarnation"],
            "launch_contract_sha256": endpoint["launch_contract_sha256"],
            "transport_identity_sha256": digest({name: endpoint.get(name) for name in TRANSPORT_IDENTITY_FIELDS}),
            "checked_at_epoch": checked_at, "ready_expires_at": endpoint.get("ready_expires_at")}


def native_receipt_errors(receipt):
    """Check receipt consistency, not cryptographic proof of a process identity."""
    errors = []
    identity = receipt.get("identity", {})
    if (identity.get("transport_mode") != NATIVE_MODE or identity.get("attestation_scope") != PLATFORM_SCOPE
            or receipt.get("attestation_scope") != PLATFORM_SCOPE or receipt.get("response_boot_attested") is not False
            or receipt.get("response_boot_id") is not None or receipt.get("qualified_boots_at_request") not in (None, [])):
        errors.append("native_response_attestation_scope_mismatch")
    before, after = receipt.get("platform_identity_at_request"), receipt.get("platform_identity_at_response")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return errors + ["native_platform_observations_missing"]
    for name in ("platform_instance_id", "incarnation", "launch_contract_sha256", "transport_identity_sha256"):
        if not isinstance(before.get(name), str) or not before[name] or before[name] != after.get(name):
            errors.append("native_platform_identity_changed:" + name)
    if not all(sha(before.get(name)) for name in ("launch_contract_sha256", "transport_identity_sha256")):
        errors.append("native_platform_identity_digest_invalid")
    for value in (before, after):
        check, expiry = value.get("checked_at_epoch"), value.get("ready_expires_at")
        if (type(check) not in (int, float) or type(expiry) not in (int, float)
                or not math.isfinite(check) or not math.isfinite(expiry) or check >= expiry):
            errors.append("native_platform_lease_invalid_at_observation")
    if (isinstance(before.get("checked_at_epoch"), (int, float)) and isinstance(after.get("checked_at_epoch"), (int, float))
            and before["checked_at_epoch"] > after["checked_at_epoch"]):
        errors.append("native_platform_observation_order_invalid")
    return errors


def assemble_openai_sse(text, model):
    """Assemble standard OpenAI SSE after full transport consumption.

    The native server streams tokens through the platform ingress; we do not
    inject heartbeats or synthesize a successful terminal event. A truncated
    stream, missing [DONE], wrong model, or error envelope fails closed.
    """
    events, data = [], []
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line:
            if data:
                events.append("\n".join(data))
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line.startswith(":") or line.startswith(("event:", "id:", "retry:")):
            continue
        else:
            raise ValueError("invalid_openai_sse_line")
    if data:
        events.append("\n".join(data))
    fragments, done, finish, usage, chunk_count = [], False, None, None, 0
    for raw in events:
        if done:
            raise ValueError("openai_sse_event_after_done")
        if raw == "[DONE]":
            done = True
            continue
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("error") or value.get("model") != model:
            raise ValueError("openai_sse_error_or_model_mismatch")
        chunk_count += 1
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("openai_sse_invalid_choices")
        if value.get("usage") is not None:
            usage = value["usage"]
        if not choices:
            continue
        choice = choices[0]
        if choice.get("index", 0) != 0 or not isinstance(choice.get("delta"), dict):
            raise ValueError("openai_sse_invalid_delta")
        content = choice["delta"].get("content")
        if content is not None:
            if not isinstance(content, str) or finish is not None and content:
                raise ValueError("openai_sse_invalid_content")
            fragments.append(content)
        if choice.get("finish_reason") is not None:
            if finish is not None:
                raise ValueError("openai_sse_duplicate_finish")
            finish = choice["finish_reason"]
    if not done or not chunk_count or finish not in {"stop", "length"}:
        raise ValueError("openai_sse_incomplete_or_unsupported_finish")
    return {"model": model, "choices": [{"message": {"content": "".join(fragments)}, "finish_reason": finish}],
            "usage": usage, "native_stream_chunks": chunk_count}
