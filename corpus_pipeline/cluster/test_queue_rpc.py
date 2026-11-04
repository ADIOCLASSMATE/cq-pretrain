"""Real loopback HTTP tests; all queue data stays in temporary directories."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import urlsplit

from corpus_pipeline.cluster import queue, queue_rpc as rpc


class QueueRPCTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        queue.init_queue(self.root, [dict(task_id=f't{i:02d}', parquet_path='/fixture.parquet',
                                         row_start=i, row_end=i + 1) for i in range(32)],
                         lease_seconds=120)
        self.server = rpc.QueueServer(self.root, bind='127.0.0.1', port=0,
                                      advertise_host='127.0.0.1')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.temp.cleanup()

    def raw(self, body, *, authorized=True, method='POST', path='/rpc', headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=5)
        supplied = {'Content-Type': 'application/json'}
        if authorized:
            supplied['Authorization'] = 'Bearer ' + self.server.token
        supplied.update(headers or {})
        encoded = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=encoded, headers=supplied)
        response = conn.getresponse()
        status, result = response.status, json.loads(response.read())
        conn.close()
        return status, result

    def output(self, lease, content=b'actual artifact bytes'):
        path = Path(lease['artifact_dir']) / 'result.jsonl'
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        return {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest()}

    def test_authentication_and_private_config(self):
        self.assertEqual((self.root / 'queue_rpc.json').stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.raw(None, authorized=False, method='GET', path='/health')[0], 401)
        self.assertEqual(self.raw({'method': 'claim', 'params': {'worker_id': 'x'}}, authorized=False)[0], 401)
        self.assertTrue(rpc.health(self.root)['ok'])
        self.assertEqual(rpc.status(self.root)['counts']['pending'], 32)

    def test_concurrent_claims_unique_and_local_metadata_serialized(self):
        with ThreadPoolExecutor(max_workers=16) as pool:
            claims = list(pool.map(lambda i: rpc.claim(self.root, f'w{i}'), range(32)))
        self.assertEqual(len({c['task_id'] for c in claims}), 32)
        self.assertEqual(len({c['token'] for c in claims}), 32)
        self.assertIsNone(rpc.claim(self.root, 'empty'))
        self.assertEqual(rpc.status(self.root)['counts']['running'], 32)

    def test_server_artifact_validation_and_idempotent_completion(self):
        lease = rpc.claim(self.root, 'worker')
        artifact = self.output(lease)
        with self.assertRaises(rpc.ArtifactError):
            rpc.complete(self.root, lease['task_id'], lease['token'],
                         artifacts=[{**artifact, 'sha256': '0' * 64}],
                         processed_records=1, processed_images=1)
        first = rpc.complete(self.root, lease['task_id'], lease['token'], artifacts=[artifact],
                             processed_records=1, processed_images=1)
        second = rpc.complete(self.root, lease['task_id'], lease['token'], artifacts=[artifact],
                              processed_records=1, processed_images=1)
        self.assertEqual(first, second)
        self.assertEqual(rpc.status(self.root)['processed_images'], 1)

    def test_targeted_claim_uses_only_server_frozen_task(self):
        target = rpc.claim_task(self.root, 'cpu-text', 't31')
        self.assertEqual(target['task']['row_start'], 31)
        self.assertEqual(target['task']['parquet_path'], '/fixture.parquet')
        self.assertIsNone(rpc.claim_task(self.root, 'second', 't31'))
        self.assertEqual(rpc.claim(self.root, 'ordinary')['task_id'], 't00')
        for params in ({'worker_id': 'x', 'task_id': 't01', 'parquet_path': '/untrusted'},
                       {'worker_id': 'x', 'task_id': '../escape'},
                       {'worker_id': 'x', 'task_id': 'absent'}):
            self.assertEqual(self.raw({'method': 'claim_task', 'params': params})[0], 400)
        with self.assertRaises(ValueError):
            rpc.claim_task(self.root, 'x', 't02', now=1)

    def test_expired_token_cannot_heartbeat_complete_or_fail(self):
        old = rpc.claim(self.root, 'preempted')
        artifact = self.output(old)
        with queue._locked(self.root):
            state_path = queue._state_path(self.root, old['task_id'])
            state = queue._read(state_path)
            state['lease_expires_at'] = time.time() - 1
            queue._atomic(state_path, state)
        newer = rpc.claim(self.root, 'replacement')
        self.assertEqual(newer['task_id'], old['task_id'])
        self.assertNotEqual(newer['token'], old['token'])
        for call in [lambda: rpc.heartbeat(self.root, old['task_id'], old['token']),
                     lambda: rpc.fail(self.root, old['task_id'], old['token'], 'late'),
                     lambda: rpc.complete(self.root, old['task_id'], old['token'], artifacts=[artifact],
                                          processed_records=1, processed_images=1)]:
            with self.assertRaises(rpc.LeaseLost):
                call()
        self.assertEqual(rpc.heartbeat(self.root, newer['task_id'], newer['token'])['task_id'], old['task_id'])

    def test_reject_root_time_path_and_unknown_method_injection(self):
        cases = [('status', {'root': '/tmp/another'}), ('status', {'now': 1}),
                 ('_atomic', {}), ('heartbeat', {'task_id': '../escape', 'token': 'x'})]
        for method, params in cases:
            status, result = self.raw({'method': method, 'params': params})
            self.assertEqual(status, 400)
            self.assertNotIn(self.server.token, json.dumps(result))
        with self.assertRaises(ValueError):
            rpc.claim(self.root, 'worker', now=1)
        status, _ = self.raw(None, headers={'Content-Length': str(rpc.MAX_BODY + 1)})
        self.assertEqual(status, 413)

    def test_live_and_stale_owner_refuse_split_brain(self):
        with self.assertRaises(FileExistsError):
            rpc.QueueServer(self.root, bind='127.0.0.1', port=0, advertise_host='127.0.0.1')
        with tempfile.TemporaryDirectory() as other:
            other = Path(other)
            queue.init_queue(other, [{'task_id': 'test'}])
            (other / '.queue_rpc_owner').mkdir()
            with self.assertRaises(FileExistsError):
                rpc.QueueServer(other, bind='127.0.0.1', port=0, advertise_host='127.0.0.1')

    def test_transport_failure_does_not_retry_claim(self):
        with patch('urllib.request.OpenerDirector.open', side_effect=URLError('fixture')) as send:
            with self.assertRaises(rpc.QueueTransportError):
                rpc.claim(self.root, 'worker')
            self.assertEqual(send.call_count, 1)

    def test_official_https_path_and_query_preserved(self):
        base = 'https://gateway.sii.edu.cn/notebook/FIXTURE_TOKEN/proxy/18783/?access=FIXTURE_QUERY'
        for suffix in ['rpc', 'health']:
            parsed = urlsplit(rpc._operation_url(base, suffix))
            self.assertEqual(parsed.path, '/notebook/FIXTURE_TOKEN/proxy/18783/' + suffix)
            self.assertEqual(parsed.query, 'access=FIXTURE_QUERY')
        for endpoint in ['https://sii.edu.cn.evil.example/SECRET',
                         'https://user:SECRET@gateway.sii.edu.cn/proxy',
                         'https://evil.example/SECRET',
                         'https://gateway.sii.edu.cn:8443/SECRET',
                         'http://gateway.sii.edu.cn/SECRET']:
            with self.assertRaises(rpc.QueueRPCError) as error:
                rpc._operation_url(endpoint, 'health')
            self.assertNotIn('SECRET', str(error.exception))
            self.assertNotIn(endpoint, str(error.exception))

    def test_route_update_preserves_server_identity_and_private_mode(self):
        before = rpc._config(self.root)
        result = rpc.reconfigure_endpoint(self.root, before['endpoint'])
        self.assertTrue(result['ok'])
        after = rpc._config(self.root)
        self.assertEqual(after, before)
        self.assertEqual((self.root / 'queue_rpc.json').stat().st_mode & 0o777, 0o600)
        self.assertNotIn(before['token'], json.dumps(result))

    def test_proxy_route_file_must_be_private_and_official_https(self):
        path = self.root / 'route.private.json'
        path.write_text(json.dumps({'data': {'url': 'https://evil.example/SECRET'}}))
        path.chmod(0o644)
        with self.assertRaises(rpc.QueueRPCError):
            rpc.configure_from_private_file(self.root, path)
        path.chmod(0o600)
        with self.assertRaises(rpc.QueueRPCError) as error:
            rpc.configure_from_private_file(self.root, path)
        self.assertNotIn('SECRET', str(error.exception))

    def test_redirect_never_reaches_second_server(self):
        # The actual loopback HTTP handler issues a redirect; urllib must not
        # follow it with the bearer token, even when destination is loopback.
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        contacted = []
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                contacted.append(self.path)
                self.send_response(302)
                self.send_header('Location', self.server.target)
                self.send_header('Content-Length', '0')
                self.end_headers()
        redirect = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
        redirect.target = f'http://127.0.0.1:{self.server.server_address[1]}/health'
        thread = threading.Thread(target=redirect.serve_forever, daemon=True)
        thread.start()
        try:
            candidate = {**rpc._config(self.root), 'endpoint': f'http://127.0.0.1:{redirect.server_address[1]}'}
            with self.assertRaises(rpc.QueueTransportError):
                rpc._request_config(candidate, timeout=5)
            self.assertEqual(contacted, ['/health'])
        finally:
            redirect.shutdown(); redirect.server_close(); thread.join(5)


if __name__ == '__main__':
    unittest.main()
