"""Buffered HTTP transport; HTTPX owns serialization and response semantics.

Receive tiny SSE chunks with aiohttp's C parser. The model client still verifies
the complete stream, image hashes, model identity and platform lease. There is
no patch installer, retry, redirect, proxy fallback or credential logging here.
"""
import asyncio
import importlib.metadata
import ssl
import time

import aiohttp
import certifi
import httpx


class BufferedTransport:
    def __init__(self, original, concurrency, timeout):
        self.original = original
        self.event_hooks = original.event_hooks
        self.timeout = timeout
        self.concurrency = concurrency
        self.session = None

    def _session(self):
        if self.session is not None:
            return self.session
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.concurrency, limit_per_host=self.concurrency,
                ssl=ssl.create_default_context(cafile=certifi.where())),
            timeout=aiohttp.ClientTimeout(total=self.timeout, connect=self.timeout, sock_read=self.timeout),
            trust_env=False, auto_decompress=False, cookie_jar=aiohttp.DummyCookieJar())
        return self.session

    async def post(self, url, *, headers, json):
        # HTTPX's unchanged serializer/default headers define exactly the same
        # wire request. Neither credentials nor image bytes are logged here.
        request = self.original.build_request('POST', url, headers=headers, json=json)
        for hook in self.event_hooks.get('request', []):
            await hook(request)
        started = time.monotonic()
        try:
            async with self._session().post(str(request.url), data=request.content,
                    headers=list(request.headers.multi_items()), allow_redirects=False) as incoming:
                headers_at = time.monotonic()
                body = await incoming.read()
                response = httpx.Response(incoming.status,
                    headers=[(k,v) for k,v in incoming.raw_headers], content=body, request=request,
                    extensions={'native_headers_received_at': headers_at,
                                'native_body_received_at': time.monotonic(),
                                'native_transport_started_at': started})
        except asyncio.TimeoutError as error:
            raise httpx.ReadTimeout('native HTTP receive deadline exceeded', request=request) from error
        except aiohttp.ClientError as error:
            raise httpx.TransportError('native HTTP transport failure: ' + type(error).__name__) from error
        self.original.cookies.extract_cookies(response)
        for hook in self.event_hooks.get('response', []):
            await hook(response)
        return response

    async def aclose(self):
        if self.session is not None:
            await self.session.close()
        await self.original.aclose()


def versions():
    return {name: importlib.metadata.version(name) for name in ('aiohttp','httpx','certifi')}
