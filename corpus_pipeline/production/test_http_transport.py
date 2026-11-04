import asyncio
import gzip
import json
import unittest
from aiohttp import web
import httpx
from .http_transport import BufferedTransport

class Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received=[]
        async def handle(request):
            self.received.append({'body':await request.read(),'auth':request.headers.get('Authorization')})
            if request.path=='/cookie':
                response=web.Response(text=request.headers.get('Cookie',''))
                response.set_cookie('route','fixture')
                return response
            if request.path=='/timeout':
                await asyncio.sleep(.2)
                return web.Response(text='late')
            if request.path=='/unavailable':
                return web.Response(status=503,text='temporary')
            content=b'data: {"token":"same"}\n\ndata: [DONE]\n\n'
            if request.path=='/gzip':
                return web.Response(body=gzip.compress(content),headers={'Content-Encoding':'gzip'})
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            await response.write(content[:12]);await asyncio.sleep(.03)
            await response.write(content[12:]);await response.write_eof()
            return response
        app=web.Application();app.router.add_post('/{path:.*}',handle)
        self.runner=web.AppRunner(app);await self.runner.setup()
        self.site=web.TCPSite(self.runner,'127.0.0.1',0);await self.site.start()
        port=self.site._server.sockets[0].getsockname()[1]
        self.base=f'http://127.0.0.1:{port}'
        self.original=httpx.AsyncClient(trust_env=False)
        self.transport=BufferedTransport(self.original,8,1)

    async def asyncTearDown(self):
        await self.transport.aclose();await self.runner.cleanup()

    async def test_exact_wire_body_and_complete_sse_before_response_hook(self):
        payload={'messages':[{'content':'中文图像'}],'stream':True}
        expected=self.original.build_request('POST',self.base+'/sse',json=payload).content
        hooks=[]
        async def request_hook(request):hooks.append('request')
        async def response_hook(response):
            hooks.append('response');self.assertTrue((await response.aread()).endswith(b'[DONE]\n\n'))
        self.transport.event_hooks.update(request=[request_hook],response=[response_hook])
        result=await self.transport.post(self.base+'/sse',headers={'Authorization':'Bearer fixture'},json=payload)
        self.assertEqual(self.received[0],{'body':expected,'auth':'Bearer fixture'})
        self.assertEqual(hooks,['request','response'])
        self.assertGreater(result.extensions['native_body_received_at'],result.extensions['native_headers_received_at'])

    async def test_gzip_decoded_once(self):
        result=await self.transport.post(self.base+'/gzip',headers={},json={})
        self.assertEqual(result.content,b'data: {"token":"same"}\n\ndata: [DONE]\n\n')

    async def test_retryable_http_status_retained(self):
        result=await self.transport.post(self.base+'/unavailable',headers={},json={})
        with self.assertRaises(httpx.HTTPStatusError):result.raise_for_status()
        self.assertEqual(result.status_code,503)

    async def test_original_cookie_policy_is_preserved(self):
        await self.transport.post(self.base+'/cookie',headers={},json={})
        response=await self.transport.post(self.base+'/cookie',headers={},json={})
        self.assertEqual(response.text,'route=fixture')

    async def test_deadline_remains_httpx_transport_error(self):
        await self.transport.aclose()
        self.original=httpx.AsyncClient(trust_env=False)
        self.transport=BufferedTransport(self.original,8,.025)
        with self.assertRaises(httpx.TransportError):
            await self.transport.post(self.base+'/timeout',headers={},json={})

if __name__=='__main__':unittest.main()
