"""Process-local review admission and progress health; no historical success credit."""
from __future__ import annotations
import os
from pathlib import Path
import time


def load_credential(path=None, *, environ=None):
    """Load before constructing clients, importing rows or waiting for readiness."""
    environ = os.environ if environ is None else environ
    if path is not None:
        file = Path(path)
        if not file.is_absolute() or not file.is_file() or file.stat().st_size > 16384:
            raise ValueError('invalid_serving_credential_file')
        token = file.read_text().strip()
        if not token or any(ch.isspace() for ch in token):
            raise ValueError('invalid_serving_credential')
        environ['INF_API_KEY'] = token
    if not environ.get('INF_API_KEY'):
        raise ValueError('required_serving_credential_unavailable')


class ProgressHealth:
    """Circuit breaker for both startup and later zero-return failure windows.

    Waiting for queued/preempted replicas alone does not trip the breaker.
    At least 64 real failed HTTP attempts or failed invocations must accumulate
    without a valid response over 120 seconds. Counters are process-local deltas.
    """
    def __init__(self, counters, transport, *, now=None, seconds=120, attempts=64):
        self.seconds, self.attempts = seconds, attempts
        self.reset(counters, transport, time.monotonic() if now is None else now)

    def reset(self, counters, transport, now):
        self.since = now
        self.valid = counters.get('actual_responses', 0) + counters.get('cached_responses', 0)
        self.failed = counters.get('failed_invocations', 0)
        self.http = transport.get('http_attempts', 0)

    def check(self, counters, transport, *, now=None):
        now = time.monotonic() if now is None else now
        valid = counters.get('actual_responses', 0) + counters.get('cached_responses', 0)
        if valid != self.valid:
            self.reset(counters, transport, now)
            return None
        failed = counters.get('failed_invocations', 0) - self.failed
        sent = transport.get('http_attempts', 0) - self.http
        if now - self.since >= self.seconds and max(failed, sent) >= self.attempts:
            return {'reason': 'no_valid_review_response', 'seconds_without_progress': now - self.since,
                    'failed_invocations': failed, 'http_attempts': sent, 'complete': False}
        return None


async def preflight(worker, *, process_id):
    """Same process, clients, credentials, image preparation and receipt validator.

    A fresh context prevents a cached response from authorizing a new consumer.
    No source facts or captions are sent, and this probe never decides a row.
    """
    from .api_screen import prepare_asset, review_context, receipt_errors
    rows = worker.ledger.db.execute("SELECT input_json FROM assets WHERE state!='done' ORDER BY asset_id")
    import json
    asset = None
    for (encoded,) in rows:
        try:
            prepared = prepare_asset(json.loads(encoded))
        except (ValueError, KeyError, TypeError, OSError):
            continue
        if prepared is not None:
            asset = prepared[0]
            break
    if asset is None:
        # Integrity handling still belongs to the worker. An empty queue needs
        # no model; invalid-only input cannot grant a successful preflight.
        return {'required': False, 'reason': 'no_eligible_pending_image'}
    context = {**review_context(asset['asset_id']), 'startup_probe': process_id}
    proofs = {}
    for role, model in worker.models.items():
        result = await worker.clients.request(role, asset, worker.prompt, context, worker.max_tokens)
        errors = receipt_errors(result, asset['asset_id'], model, worker.prompt, context,
                                worker.max_tokens, worker.clients.endpoint_identity.get(role), 'serving')
        if errors or result.get('cache_hit') or result.get('real_model_inference') is not True:
            raise ValueError('same_process_real_image_preflight_failed')
        proofs[role] = result
    return {'required': True, 'complete': True, 'process_id': process_id, 'receipts': proofs}
