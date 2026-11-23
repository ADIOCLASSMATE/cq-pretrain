"""Respect a shared Retry-After and distinguish unchanged original image bytes."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import worker


def main():
    now, calls = [1000.0], []
    class Response:
        url = 'https://cdn.example.test/pinned-object'
        headers = {'Retry-After': '120'}
        def __init__(self, status):
            self.status_code = status
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            assert self.status_code == 206
    class Session:
        def get(self, *args, **kwargs):
            calls.append(now[0])
            return Response(429 if len(calls) == 1 else 206)
    def sleep(seconds):
        now[0] += seconds
    with tempfile.TemporaryDirectory(dir=worker.RUN) as folder:
        run = Path(folder)
        with patch.object(worker, 'RUN', run), patch.object(worker, 'session', return_value=Session()), \
             patch.object(worker.time, 'time', side_effect=lambda: now[0]), patch.object(worker.time, 'sleep', side_effect=sleep):
            result = worker.source_url(dict(repo='test', revision='pinned', path='sample.tar'))
            assert result == Response.url and calls == [1000.0, 1120.0], calls
            assert json.loads((run / 'HF_RESOLVE_GATE.json').read_text())['next_allowed'] == 1120.5
            row = dict(baseline_sha256='a' * 64, status='missing_face_metadata', metadata={'sha256': 'a' * 64})
            assert worker.recover(row, run)['status'] == 'already_original_bytes'
            row['metadata']['sha256'] = 'b' * 64
            assert worker.recover(row, run)['status'] == 'missing_face_metadata'
    print('PASS: shared Retry-After honored; only exact original bytes classified unchanged')


if __name__ == '__main__':
    main()
