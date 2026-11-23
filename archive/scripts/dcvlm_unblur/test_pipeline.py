"""Metadata completion must not count as completed image recovery."""
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import worker


def main():
    rows = [dict(baseline_sha256='a' * 64, status='missing_face_metadata', metadata={'sha256': 'a' * 64}),
            dict(baseline_sha256='b' * 64, status='missing_face_metadata', metadata={'sha256': 'c' * 64})]
    job = dict(path='pool/000000.tar', wanted=[r['baseline_sha256'] for r in rows])
    with tempfile.TemporaryDirectory(dir=worker.RUN) as folder:
        run = Path(folder)
        with patch.object(worker, 'RUN', run), patch.object(worker, 'collect', return_value=rows), \
             patch.object(worker.time, 'sleep'), ThreadPoolExecutor(2) as downloads:
            worker.process(job, downloads, None, metadata_only=True)
            directory = run / 'upstream/000000.tar'
            assert (directory / 'METADATA_COMPLETE.json').exists()
            assert not (directory / 'DONE.json').exists()
            worker.process(job, downloads, None)
            result = json.loads((directory / 'DONE.json').read_text())
            assert result['counts'] == {'already_original_bytes': 1, 'missing_face_metadata': 1}
            assert (directory / 'METADATA_RECHECKED.json').exists()
            before = (directory / 'results.jsonl').read_bytes()
            worker.process(job, downloads, None)
            assert (directory / 'results.jsonl').read_bytes() == before
    print('PASS: metadata phase cannot mark recovery complete; recovery resumes idempotently')


if __name__ == '__main__':
    main()
