"""Platform-authenticated Serving proxy; private engine credential stays inside."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads(Path(os.environ.get('CORPUS_CONFIG', ROOT / 'config.json')).read_text())
ENGINE_KEY = Path(CONFIG['engine_key_path']).read_text().strip()
LEADER = 'http://' + os.environ['QWEN_LEADER_IP'] + ':9000'
REPLICA = os.environ['QWEN_REPLICA_ID']
BOOT_ID = os.environ['QWEN_BOOT_ID']


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
