"""Platform-authenticated native Serving gateway; legacy Job routing retained."""
from __future__ import annotations

import asyncio
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
CONFIG = json.loads((ROOT / 'config.json').read_text())
PUBLIC_HASH = Path(CONFIG['client_key_hash_path']).read_text().strip()
ENGINE_KEY = Path(CONFIG['engine_key_path']).read_text().strip()
DEPLOYMENT = os.environ['CORPUS_DEPLOYMENT_ID']
BOOT_ID = os.environ.get('CORPUS_BOOT_ID')
BACKEND = os.environ.get('CORPUS_BACKEND', 'inspire_serving')
REPLICAS = int(os.environ.get('CORPUS_LOCAL_REPLICAS', '1'))
assert (BACKEND, REPLICAS) in {('inspire_serving', 1), ('inspire_job', 4)}
ENGINES = ['http://127.0.0.1:' + str(9000 + i) for i in range(REPLICAS)]
INFLIGHT = [0] * len(ENGINES)
READY = [False] * len(ENGINES)
LOCK = asyncio.Lock()


async def check(client, index):
    try:
        reply = await client.get(ENGINES[index] + '/health', timeout=3)
        READY[index] = reply.status_code == 200
    except httpx.HTTPError:
        READY[index] = False


async def watch(client):
    while True:
        await asyncio.gather(*(check(client, i) for i in range(len(ENGINES))))
        await asyncio.sleep(3)


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(900, connect=10),
                                limits=httpx.Limits(max_connections=512, max_keepalive_connections=128)) as client:
        app.state.http = client
        watcher = asyncio.create_task(watch(client))
        try:
            yield
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.get('/')
async def live():
    return {'service': CONFIG['model_name'], 'deployment': DEPLOYMENT, 'boot_id': BOOT_ID,
            'readiness_endpoint': '/health'}


@app.get('/health')
async def health():
    # Every allocated replica must be ready before this deployment is healthy.
    return JSONResponse({'model_ready': all(READY), 'replicas_ready': sum(READY),
                         'replicas_total': len(ENGINES), 'deployment': DEPLOYMENT, 'boot_id': BOOT_ID},
                        status_code=200 if all(READY) else 503)


@app.api_route('/{path:path}', methods=['GET', 'POST'])
async def forward(path: str, request: Request):
    auth = request.headers.get('authorization', '')
    token = auth[7:] if auth.startswith('Bearer ') else ''
    # Native Serving authentication is enforced by the platform ingress with
    # INF_API_KEY. A second incompatible client Bearer key would reject valid
    # platform requests. The local engine still requires its private key.
    if BACKEND == 'inspire_job' and (not token or not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), PUBLIC_HASH)):
        return JSONResponse({'error': {'message': 'Valid API key required'}}, status_code=401)
    if path not in ('v1/models', 'v1/chat/completions', 'v1/completions', 'tokenize', 'v1/tokenize'):
        return JSONResponse({'error': {'message': 'Unsupported route'}}, status_code=404)
    async with LOCK:
        ready = [i for i in range(len(ENGINES)) if READY[i]]
        if not ready:
            return JSONResponse({'error': {'message': 'Gemma replicas are loading or unavailable'}},
                                status_code=503, headers={'Retry-After': '10'})
        index = min(ready, key=lambda i: INFLIGHT[i])
        INFLIGHT[index] += 1
    headers = {'Authorization': 'Bearer ' + ENGINE_KEY}
    if request.headers.get('content-type'):
        headers['Content-Type'] = request.headers['content-type']
    route = 'tokenize' if path == 'v1/tokenize' else path
    try:
        body = await request.body()
        upstream = await request.app.state.http.send(request.app.state.http.build_request(
            request.method, ENGINES[index] + '/' + route, content=body, headers=headers), stream=True)
    except httpx.HTTPError:
        async with LOCK:
            INFLIGHT[index] -= 1
        return JSONResponse({'error': {'message': 'Gemma replica temporarily unavailable'}},
                            status_code=503, headers={'Retry-After': '10'})
    except BaseException:
        # Client disconnects and task cancellation can happen during body reads.
        # Release the lease even when no upstream streaming response was created.
        async with LOCK:
            INFLIGHT[index] -= 1
        raise

    async def chunks():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            async with LOCK:
                INFLIGHT[index] -= 1

    outgoing = {key: value for key, value in upstream.headers.items()
                if key.lower() in ('content-type', 'cache-control')}
    outgoing['X-UMM-Replica'] = DEPLOYMENT + '-tp4-' + str(index)
    if BOOT_ID:
        outgoing['X-UMM-Boot-ID'] = BOOT_ID
    return StreamingResponse(chunks(), status_code=upstream.status_code, headers=outgoing)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, access_log=False)
