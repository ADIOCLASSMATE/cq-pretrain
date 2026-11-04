"""Platform-authenticated Serving proxy; private engine credential stays inside."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
import httpx
import uvicorn

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads(Path(os.environ.get('CORPUS_CONFIG', ROOT / 'config.json')).read_text())
ENGINE_KEY = Path(CONFIG['engine_key_path']).read_text().strip()
LEADER = 'http://' + os.environ['QWEN_LEADER_IP'] + ':9000'
REPLICA = os.environ['QWEN_REPLICA_ID']
BOOT_ID = os.environ['QWEN_BOOT_ID']
KEEPALIVE_ENABLED = CONFIG.get('enable_json_keepalive', False)
if not isinstance(KEEPALIVE_ENABLED, bool):
    raise ValueError('enable_json_keepalive must be boolean')


def wants_json_keepalive(path, method, body):
    if method != 'POST' or path not in ('v1/chat/completions', 'v1/completions'):
        return False
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeError):
        return False
    return isinstance(parsed, dict) and parsed.get('stream', False) is False


class JSONHeartbeatResponse(Response):
    """One bounded JSON response with whitespace while inference is pending.

    Owns the ASGI disconnect listener before upstream response headers arrive.
    Response start is deferred until completion or the first heartbeat. Once
    started, failures use an OpenAI-style error object and cannot change HTTP
    status. No partial model JSON is exposed before successful body validation.
    """
    media_type = 'application/json'

    def __init__(self, upstream_factory, *, headers, interval=10., timeout=900.,
                 max_response_bytes=8 * 1024 * 1024):
        if interval <= 0 or timeout <= interval or max_response_bytes <= 0:
            raise ValueError('Invalid bounded JSON heartbeat configuration')
        super().__init__(content=None, headers={**headers, 'X-Accel-Buffering': 'no',
                                                'Cache-Control': 'no-store'})
        self.raw_headers = [(key, value) for key, value in self.raw_headers
                            if key != b'content-length']
        self.upstream_factory = upstream_factory
        self.interval = interval
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    @staticmethod
    def error_body(code, status=503):
        # Never send exception text, credentials, addresses, or an upstream trace.
        return json.dumps({'error': {'message': 'Model request failed; retry later',
                                    'type': 'upstream_error', 'code': code,
                                    'upstream_status': status}}, separators=(',', ':')).encode()

    async def fetch(self):
        upstream = None
        try:
            upstream = await self.upstream_factory()
            body = bytearray()
            async for chunk in upstream.aiter_bytes():
                if len(body) + len(chunk) > self.max_response_bytes:
                    return 502, self.error_body('gateway_response_too_large', 502)
                body.extend(chunk)
            if upstream.status_code >= 400:
                return upstream.status_code, self.error_body('gateway_upstream_status', upstream.status_code)
            try:
                parsed = json.loads(body)
            except (ValueError, UnicodeError):
                return 502, self.error_body('gateway_invalid_json', 502)
            if not isinstance(parsed, dict):
                return 502, self.error_body('gateway_invalid_json_object', 502)
            return upstream.status_code, bytes(body)
        finally:
            if upstream is not None:
                await upstream.aclose()

    async def fetch_bounded(self):
        try:
            return await asyncio.wait_for(self.fetch(), timeout=self.timeout)
        except asyncio.TimeoutError:
            return 504, self.error_body('gateway_deadline_exceeded', 504)
        except httpx.HTTPError:
            return 503, self.error_body('gateway_transport_error', 503)
        # Cancellation intentionally propagates to httpx and closes its socket.

    async def send_response(self, work, send):
        started = False
        while True:
            await asyncio.wait({work}, timeout=self.interval)
            if work.done():
                status, body = work.result()
                if not started:
                    await send({'type': 'http.response.start', 'status': status,
                                'headers': self.raw_headers})
                await send({'type': 'http.response.body', 'body': body, 'more_body': False})
                return
            if not started:
                await send({'type': 'http.response.start', 'status': 200,
                            'headers': self.raw_headers})
                started = True
            await send({'type': 'http.response.body', 'body': b' \n', 'more_body': True})

    @staticmethod
    async def wait_disconnect(receive):
        # The route has already consumed request.body(); this is the sole reader.
        while True:
            if (await receive())['type'] == 'http.disconnect':
                return

    async def __call__(self, scope, receive, send):
        work = asyncio.create_task(self.fetch_bounded())
        delivery = asyncio.create_task(self.send_response(work, send))
        disconnect = asyncio.create_task(self.wait_disconnect(receive))
        tasks = (work, delivery, disconnect)
        try:
            finished, _ = await asyncio.wait({delivery, disconnect},
                                             return_when=asyncio.FIRST_COMPLETED)
            if delivery in finished:
                delivery.result()
        except OSError:
            # ASGI 2.4 servers may signal a closed connection from send().
            pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(900, connect=10),
                                 limits=httpx.Limits(max_connections=512, max_keepalive_connections=128)) as client:
        app.state.http = client
        yield


app = FastAPI(lifespan=lifespan)


@app.get('/')
async def live():
    # Container liveness is separate from the model readiness endpoint below.
    return {'service': CONFIG['model_name'], 'replica': REPLICA, 'readiness_endpoint': '/health'}


@app.get('/health')
async def health(request: Request):
    try:
        response = await request.app.state.http.get(LEADER + '/health',
            headers={'Authorization': 'Bearer ' + ENGINE_KEY}, timeout=3)
        code = 200 if response.status_code == 200 else 503
    except httpx.HTTPError:
        code = 503
    return JSONResponse({'model_ready': code == 200, 'replica': REPLICA, 'boot_id': BOOT_ID}, status_code=code,
                        headers={'X-UMM-Boot': BOOT_ID})


@app.api_route('/{path:path}', methods=['GET', 'POST'])
async def forward(path: str, request: Request):
    # The Inspire public endpoint authenticates INF_API_KEY before forwarding.
    # Keep the private engine credential separate; never require the old Job key.
    if path not in ('v1/models', 'v1/chat/completions', 'v1/completions', 'tokenize', 'v1/tokenize'):
        return JSONResponse({'error': {'message': 'Unsupported route'}}, status_code=404)
    route = 'tokenize' if path == 'v1/tokenize' else path
    body = await request.body()
    headers = {'Authorization': 'Bearer ' + ENGINE_KEY}
    if request.headers.get('content-type'):
        headers['Content-Type'] = request.headers['content-type']
    if KEEPALIVE_ENABLED and wants_json_keepalive(path, request.method, body):
        async def open_upstream():
            return await request.app.state.http.send(request.app.state.http.build_request(
                request.method, LEADER + '/' + route, content=body, headers=headers), stream=True)
        return JSONHeartbeatResponse(open_upstream,
            headers={'X-UMM-Replica': REPLICA, 'X-UMM-Boot': BOOT_ID})
    try:
        upstream = await request.app.state.http.send(request.app.state.http.build_request(
            request.method, LEADER + '/' + route, content=body, headers=headers), stream=True)
    except httpx.HTTPError:
        return JSONResponse({'error': {'message': 'Model replica unavailable; retry later'}}, status_code=503,
                            headers={'Retry-After': '10', 'X-UMM-Replica': REPLICA})

    async def chunks():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    outgoing = {k: v for k, v in upstream.headers.items() if k.lower() in ('content-type', 'cache-control')}
    outgoing['X-UMM-Replica'] = REPLICA
    outgoing['X-UMM-Boot'] = BOOT_ID
    return StreamingResponse(chunks(), status_code=upstream.status_code, headers=outgoing)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, access_log=False)
