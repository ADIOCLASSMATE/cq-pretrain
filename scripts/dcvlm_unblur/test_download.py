"""A broken connection must retain its bytes; bad archive hashes must never pass."""
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import requests

import worker


def main():
    data = b'a' * 65536 + b'b' * 65536
    ranges = []
    class Response:
        status_code = 206
        def __init__(self, start, end):
            self.start, self.end = start, end
            self.headers = {'Content-Range': f'bytes {start}-{end}/{len(data)}'}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def raise_for_status(self):
            pass
        def iter_content(self, size):
            if len(ranges) == 1:
                yield data[:65536]
                raise requests.ConnectionError('injected interrupted transfer')
            yield data[self.start:self.end + 1]
    class Session:
        def get(self, url, *, headers, **kwargs):
            start, end = map(int, headers['Range'].removeprefix('bytes=').split('-'))
            ranges.append((start, end))
            return Response(start, end)
    job = dict(repo='test', revision='pinned', path='fixture.tar', size=len(data),
               sha256=hashlib.sha256(data).hexdigest())
    with tempfile.TemporaryDirectory(dir=worker.RUN) as folder:
        directory = Path(folder)
        with patch.object(worker, 'session', return_value=Session()), patch.object(worker.time, 'sleep'), ThreadPoolExecutor(2) as pool:
            result = worker.download_tar(job, directory, pool)
            assert result.read_bytes() == data
            assert ranges == [(0, 131071), (65536, 131071)], ranges
            try:
                worker.download_tar(dict(job, sha256='0' * 64), directory, pool)
            except ValueError:
                pass
            else:
                raise AssertionError('Corrupt upstream archive was accepted')
            assert not (directory / 'parts').exists() and not result.exists()
    print('PASS: interrupted range resume preserves bytes; corrupt source cache rejected')


if __name__ == '__main__':
    main()
