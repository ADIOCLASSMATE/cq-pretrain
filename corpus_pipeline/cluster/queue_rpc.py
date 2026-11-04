"""Single-control-host queue RPC. Workers must never call queue.py directly.

Server: python -m corpus_pipeline.cluster.queue_rpc serve --root RUN --port 18783
The private RUN/queue_rpc.json contains the endpoint and bearer token. It must
not be printed. A crashed server leaves .queue_rpc_owner behind; there is no
automatic stale-owner takeover. Verify the old process is dead before manually
recovering its ownership directory. No RPC operation retries automatically.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import socket
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import uuid

from corpus_pipeline.cluster import queue as local

SCHEMA = 'corpus_queue_rpc.v1'
MAX_BODY = 4 * 1024 * 1024
MAX_RESPONSE = 16 * 1024 * 1024
LeaseLost = local.LeaseLost
ArtifactError = local.ArtifactError
QueueError = local.QueueError


class QueueRPCError(QueueError):
    pass


class QueueTransportError(QueueRPCError):
    """Delivery/response uncertain. Do not blindly retry a claim."""


METHODS = {
    'claim': ({'worker_id'}, {'worker_id'}),
    'claim_task': ({'worker_id', 'task_id'}, {'worker_id', 'task_id'}),
    'heartbeat': ({'task_id', 'token'}, {'task_id', 'token'}),
    'complete': ({'task_id', 'token', 'artifacts', 'processed_records', 'processed_images', 'metadata'},
                 {'task_id', 'token', 'artifacts', 'processed_records', 'processed_images'}),
    'fail': ({'task_id', 'token', 'error', 'retryable'}, {'task_id', 'token', 'error'}),
    'status': (set(), set()),
}
ERRORS = {'LeaseLost': LeaseLost, 'ArtifactError': ArtifactError,
          'QueueError': QueueError, 'ValueError': ValueError,
          'TypeError': TypeError, 'QueueRPCError': QueueRPCError}
SAFE_MESSAGES = {
    'LeaseLost': 'Lease is expired, superseded, or no longer running.',
    'ArtifactError': 'Server artifact path, bytes, or checksum validation failed.',
    'QueueError': 'Queue operation rejected; state or completion contract invalid.',
    'ValueError': 'Invalid RPC parameters.', 'TypeError': 'Invalid RPC parameter types.',
    'QueueRPCError': 'Queue RPC request rejected.',
    'InternalError': 'Queue server operation failed; inspect control-host evidence.',
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode()


def _private_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(_json(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        local._sync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def routed_ip():
    # UDP connect selects a local route; no datagram is transmitted.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(('1.1.1.1', 53))
        result = sock.getsockname()[0]
    if ipaddress.ip_address(result).is_loopback or result == '0.0.0.0':
        raise QueueRPCError('No routable control-host address; set --advertise-host')
    return result


def _validate_request(body):
    if not isinstance(body, dict) or set(body) != {'method', 'params'}:
        raise ValueError('Invalid request envelope')
    method, params = body['method'], body['params']
    if not isinstance(method, str) or method not in METHODS or not isinstance(params, dict):
        raise ValueError('Unknown method or invalid parameters')
    allowed, required = METHODS[method]
    if not required <= set(params) <= allowed:
        raise ValueError('Missing or unexpected parameters')
    # The service exclusively owns both queue root and clock. Neither can be
    # supplied by workers; artifact paths are fenced by local.complete itself.
    if 'task_id' in params:
        local._task_id(params['task_id'])
    if 'token' in params and (not isinstance(params['token'], str) or len(params['token']) > 256):
        raise ValueError('Invalid lease token')
    if 'retryable' in params and not isinstance(params['retryable'], bool):
        raise ValueError('retryable must be boolean')
    return method, params


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *args):
        # Default logs can contain untrusted request targets. Keep them silent.
        pass

    def _send(self, status, value):
        body = _json(value)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Outcome is already committed; client must reconcile.

    def _authorized(self):
        actual = self.headers.get('Authorization', '').encode('utf-8')
        expected = ('Bearer ' + self.server.token).encode()
        if not hmac.compare_digest(actual, expected):
            self._send(401, {'ok': False, 'error_type': 'QueueRPCError',
                             'message': 'Authentication required.'})
            return False
        incarnation = self.headers.get('X-Queue-Server-ID')
        if incarnation and not hmac.compare_digest(incarnation.encode(), self.server.server_id.encode()):
            self._send(409, {'ok': False, 'error_type': 'QueueRPCError',
                             'message': 'Queue server identity changed; reload configuration.'})
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if self.path != '/health':
            self._send(404, {'ok': False, 'error_type': 'QueueRPCError', 'message': 'Unknown route.'})
            return
        self._send(200, {'ok': True, 'schema': SCHEMA, 'server_id': self.server.server_id,
                         'queue_manifest_sha256': self.server.manifest_sha256,
                         'time': time.time(), 'pid': os.getpid()})

    def do_POST(self):
        if not self._authorized():
            return
        if self.path != '/rpc':
            self._send(404, {'ok': False, 'error_type': 'QueueRPCError', 'message': 'Unknown route.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '-1'))
        except ValueError:
            length = -1
        if self.headers.get('Transfer-Encoding') or not 0 <= length <= MAX_BODY:
            self._send(413, {'ok': False, 'error_type': 'QueueRPCError', 'message': 'Invalid or oversized body.'})
            return
        if self.headers.get_content_type() != 'application/json':
            self._send(415, {'ok': False, 'error_type': 'QueueRPCError', 'message': 'JSON required.'})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete body')
            body = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            method, params = _validate_request(body)
            result = getattr(local, method)(self.server.queue_root, **params)
            self._send(200, {'ok': True, 'result': result, 'server_id': self.server.server_id})
        except Exception as exc:
            error_type = type(exc).__name__
            if error_type not in ERRORS:
                error_type = 'InternalError'
            self._send(409 if error_type in {'LeaseLost', 'ArtifactError', 'QueueError'} else 400,
                       {'ok': False, 'error_type': error_type,
                        'message': SAFE_MESSAGES[error_type]})


class QueueServer(ThreadingHTTPServer):
    # server_close waits until active completions have finished before releasing
    # ownership. queue.py's flock then serializes metadata on this single host.
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True

    def __init__(self, root, *, bind='0.0.0.0', port=18783, advertise_host=None):
        self.queue_root = Path(root).resolve()
        config = local._config(self.queue_root)
        self.manifest_sha256 = config['manifest_sha256']
        self.server_id = uuid.uuid4().hex
        self.token = secrets.token_urlsafe(48)
        self.owner_dir = self.queue_root / '.queue_rpc_owner'
        self.owner_dir.mkdir(mode=0o700)  # Atomic, no stale-owner auto-recovery.
        self._owned = True
        try:
            _private_json(self.owner_dir / 'owner.json', {
                'server_id': self.server_id, 'hostname': socket.gethostname(),
                'pid': os.getpid(), 'started_at': time.time()})
            super().__init__((bind, port), Handler)
            host = advertise_host or routed_ip()
            ipaddress.IPv4Address(host)
            endpoint = f'http://{host}:{self.server_address[1]}'
            self.config_path = self.queue_root / 'queue_rpc.json'
            _private_json(self.config_path, {
                'schema': SCHEMA, 'root': str(self.queue_root), 'endpoint': endpoint,
                'token': self.token, 'server_id': self.server_id,
                'queue_manifest_sha256': self.manifest_sha256,
                'hostname': socket.gethostname(), 'pid': os.getpid(), 'created_at': time.time()})
        except BaseException:
            if hasattr(self, 'socket'):
                super().server_close()
            self._release_owner()
            raise

    def _release_owner(self):
        if not self._owned:
            return
        owner = self.owner_dir / 'owner.json'
        if owner.exists() and json.loads(owner.read_text()).get('server_id') == self.server_id:
            owner.unlink()
            self.owner_dir.rmdir()
        self._owned = False

    def server_close(self):
        super().server_close()
        self._release_owner()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _endpoint_parts(endpoint):
    """Only direct IPv4 HTTP or the platform's authenticated HTTPS gateway.

    A gateway path/query may contain an access credential. Never include any
    supplied endpoint or parser exception text in a diagnostic.
    """
    try:
        if not isinstance(endpoint, str) or any(ch.isspace() for ch in endpoint):
            raise ValueError()
        parsed = urlsplit(endpoint)
        if parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise ValueError()
        if parsed.scheme == 'http':
            ipaddress.IPv4Address(parsed.hostname)
            if not parsed.port or parsed.path or parsed.query:
                raise ValueError()
        elif parsed.scheme == 'https':
            host = parsed.hostname or ''
            if (not host.endswith('.sii.edu.cn') or host == '.sii.edu.cn'
                    or parsed.port not in (None, 443) or '\\' in endpoint):
                raise ValueError()
        else:
            raise ValueError()
        return parsed
    except (ValueError, TypeError, AttributeError):
        raise QueueRPCError('Endpoint is not an approved direct IP or official HTTPS Notebook route') from None


def _operation_url(endpoint, suffix):
    if suffix not in ('health', 'rpc'):
        raise QueueRPCError('Unknown RPC operation path')
    parsed = _endpoint_parts(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc,
                       parsed.path.rstrip('/') + '/' + suffix, parsed.query, ''))


def _config(root):
    root = Path(root).resolve()
    path = root / 'queue_rpc.json'
    config = json.loads(path.read_text())
    if path.stat().st_mode & 0o077:
        raise QueueRPCError('Queue RPC config must have mode 0600')
    if config.get('schema') != SCHEMA or config.get('root') != str(root):
        raise QueueRPCError('Queue RPC root or schema mismatch')
    _endpoint_parts(config.get('endpoint'))
    if not isinstance(config.get('token'), str) or len(config['token']) < 32:
        raise QueueRPCError('Incomplete queue RPC configuration')
    return config


def _request(root, *, method=None, params=None, timeout=120):
    config = _config(root)
    return _request_config(config, method=method, params=params, timeout=timeout)


def _request_config(config, *, method=None, params=None, timeout=120):
    data = None if method is None else _json({'method': method, 'params': params or {}})
    if data is not None and len(data) > MAX_BODY:
        raise QueueRPCError('RPC request exceeds 4 MiB')
    request = Request(_operation_url(config['endpoint'], 'health' if method is None else 'rpc'),
                      data=data, headers={'Authorization': 'Bearer ' + config['token'],
                                          'X-Queue-Server-ID': config['server_id'],
                                          'Content-Type': 'application/json'})
    # No proxies, redirects, session pooling or implicit retries.
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as exc:
            response = exc
        with response:
            raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise QueueTransportError('RPC response exceeds limit; outcome uncertain')
        result = json.loads(raw)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise QueueTransportError('Queue RPC delivery or response failed; outcome may be committed. No retry sent.') from None
    if not isinstance(result, dict):
        raise QueueTransportError('Invalid RPC response; outcome uncertain')
    if result.get('ok') is not True:
        kind = result.get('error_type')
        raise ERRORS.get(kind, QueueRPCError)(SAFE_MESSAGES.get(kind, SAFE_MESSAGES['InternalError']))
    if result.get('server_id') != config['server_id']:
        raise QueueTransportError('Queue server identity changed; outcome uncertain')
    return result if method is None else result['result']


def _server_clock(now):
    if now is not None:
        raise ValueError('Remote callers cannot override the queue server clock')


def claim(root, worker_id, *, now=None):
    _server_clock(now)
    return _request(root, method='claim', params={'worker_id': worker_id})


def claim_task(root, worker_id, task_id, *, now=None):
    _server_clock(now)
    return _request(root, method='claim_task', params={'worker_id': worker_id, 'task_id': task_id})


def heartbeat(root, task_id, token, *, now=None):
    _server_clock(now)
    return _request(root, method='heartbeat', params={'task_id': task_id, 'token': token}, timeout=30)


def complete(root, task_id, token, *, artifacts, processed_records, processed_images, metadata=None, now=None):
    _server_clock(now)
    return _request(root, method='complete', params={'task_id': task_id, 'token': token,
        'artifacts': artifacts, 'processed_records': processed_records,
        'processed_images': processed_images, 'metadata': metadata})


def fail(root, task_id, token, error, *, retryable=True, now=None):
    _server_clock(now)
    return _request(root, method='fail', params={'task_id': task_id, 'token': token,
                                               'error': error, 'retryable': retryable})


def status(root, *, now=None):
    _server_clock(now)
    return _request(root, method='status', params={})


def health(root):
    return _request(root, timeout=30)


def reconfigure_endpoint(root, endpoint):
    """Validate new route against the same live server before atomic publication.

    Only the owning control host can refresh. All fields except endpoint are
    preserved, including server_id and bearer token. No server restart needed.
    """
    _endpoint_parts(endpoint)
    root = Path(root).resolve()
    with (root / '.queue_rpc_route.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        before = _config(root)
        owner = json.loads((root / '.queue_rpc_owner/owner.json').read_text())
        if owner.get('hostname') != socket.gethostname() or owner.get('server_id') != before['server_id']:
            raise QueueRPCError('Route refresh must run on the owning queue control host')
        candidate = {**before, 'endpoint': endpoint}
        result = _request_config(candidate, timeout=30)
        if result.get('queue_manifest_sha256') != before['queue_manifest_sha256']:
            raise QueueRPCError('New route points to a different queue')
        if _config(root) != before:
            raise QueueRPCError('Queue configuration changed during route validation')
        _private_json(root / 'queue_rpc.json', candidate)
    return {'ok': True, 'route_updated': True, 'server_id': before['server_id'],
            'time': time.time(), 'credentials_printed': False}


def configure_from_private_file(root, route_file):
    path = Path(route_file)
    if path.stat().st_mode & 0o077:
        raise QueueRPCError('Proxy route file must have mode 0600')
    try:
        endpoint = json.loads(path.read_text())['data']['url']
    except (ValueError, KeyError, TypeError):
        raise QueueRPCError('Invalid private proxy route file') from None
    if _endpoint_parts(endpoint).scheme != 'https':
        raise QueueRPCError('Private Notebook route must use official HTTPS')
    return reconfigure_endpoint(root, endpoint)


def refresh_notebook_route(root, *, account='wjx-ascend', workspace='昇腾卡公共空间',
                           notebook='cpu-wjx-ascend', port=18783):
    """Read a fresh official proxy URL through the SDK's authenticated adapter."""
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    logging.getLogger('inspire').setLevel(logging.CRITICAL)
    from inspire.sdk import InspireClient
    from inspire.platform.web.browser_api import resolve_notebook_port_forward_url
    try:
        with InspireClient(account=account) as client:
            target = client.notebooks.get(notebook, workspace=workspace)
            if target.status != 'RUNNING' or target.resource.get('gpu_count') != 0:
                raise QueueRPCError('Expected running CPU control Notebook')
            with client._transport.scope(timeout=90):
                endpoint = resolve_notebook_port_forward_url(
                    target.ref.key, port=port, service_path='', session=client._transport.session,
                    timeout=60, refresh=True)
            if _endpoint_parts(endpoint).scheme != 'https':
                raise QueueRPCError('Notebook route must use official HTTPS')
            return reconfigure_endpoint(root, endpoint)
    except Exception:
        # SDK/network exceptions can embed the credential-bearing proxy URL.
        raise QueueRPCError('Authenticated Notebook route refresh failed; existing route preserved') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['serve', 'health', 'status', 'set-route', 'refresh-route'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--port', type=int, default=18783)
    parser.add_argument('--advertise-host')
    parser.add_argument('--route-file', type=Path)
    parser.add_argument('--account', default='wjx-ascend')
    parser.add_argument('--workspace', default='昇腾卡公共空间')
    parser.add_argument('--notebook', default='cpu-wjx-ascend')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--interval', type=int, default=1200)
    args = parser.parse_args()
    if args.command == 'set-route':
        if not args.route_file:
            parser.error('--route-file is required')
        print(json.dumps(configure_from_private_file(args.root, args.route_file)))
        return
    if args.command == 'refresh-route':
        if args.interval < 60:
            parser.error('Refresh interval must be at least 60 seconds')
        while True:
            try:
                result = refresh_notebook_route(args.root, account=args.account, workspace=args.workspace,
                                                notebook=args.notebook, port=args.port)
            except Exception:
                if not args.loop:
                    raise QueueRPCError('Notebook route refresh failed; existing route preserved') from None
                result = {'ok': False, 'route_updated': False, 'time': time.time()}
            print(json.dumps(result), flush=True)
            if not args.loop:
                break
            time.sleep(args.interval)
        return
    if args.command != 'serve':
        print(json.dumps(health(args.root) if args.command == 'health' else status(args.root)))
        return
    server = QueueServer(args.root, port=args.port, advertise_host=args.advertise_host)
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps({'status': 'serving', 'server_id': server.server_id,
                      'config_path': str(server.config_path), 'pid': os.getpid()}), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
