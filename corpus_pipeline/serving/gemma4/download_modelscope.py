"""Resume direct, parallel downloads from Google's ModelScope mirror.

Require sizes and LFS hashes to agree with the locked official HF revision;
verify every final file against the mirror's SHA256 before publishing readiness.
"""
from __future__ import annotations

import os
for _key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    os.environ.pop(_key, None)

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Gemma4-31B-IT-842da379')


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk-mib', type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.chunk_mib <= 512:
        raise ValueError('invalid concurrency or chunk size')
    manifest = json.loads((ROOT / 'download_manifest.json').read_text())
    started = time.monotonic()
    client = httpx.Client(trust_env=False, follow_redirects=True,
                         timeout=httpx.Timeout(90, connect=20),
                         limits=httpx.Limits(max_connections=args.workers + 2,
                                             max_keepalive_connections=args.workers + 2))
    response = client.get('https://www.modelscope.cn/api/v1/models/google/gemma-4-31B-it/repo/files',
                          params={'Revision': 'master', 'Recursive': 'true'})
    response.raise_for_status()
    mirror = {f['Path']: f for f in response.json()['Data']['Files']}
    expected = []
    for item in manifest['files']:
        match = mirror[item['name']]
        if match['Size'] != item['size']:
            raise ValueError('mirror_size_does_not_match_hf: ' + item['name'])
        if item['expected_lfs_sha256'] and item['expected_lfs_sha256'] != match['Sha256']:
            raise ValueError('mirror_lfs_hash_does_not_match_hf: ' + item['name'])
        # Existing small HF downloads serve as a second check for non-LFS metadata.
        hf_copy = ROOT / 'weights' / item['name']
        if hf_copy.is_file() and checksum(hf_copy) != match['Sha256']:
            raise ValueError('mirror_metadata_does_not_match_hf: ' + item['name'])
        expected.append({**item, 'sha256': match['Sha256'], 'mirror_file_revision': match['Revision']})
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / 'modelscope_manifest.json', {
        **manifest, 'files': expected, 'mirror': 'google/gemma-4-31B-it',
        'destination': str(args.output), 'proxy_environment_removed': True, 'http_client_trust_env': False,
    })
    completed = []
    for item in expected:
        name, size, digest = item['name'], item['size'], item['sha256']
        final = args.output / name
        if final.exists():
            if final.stat().st_size != size or checksum(final) != digest:
                raise ValueError('existing_final_file_invalid: ' + name)
            completed.append(item)
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        part = final.with_suffix(final.suffix + '.part')
        state_path = ROOT / (name.replace('/', '_') + '.chunks.json')
        chunk_size = args.chunk_mib * 1024 * 1024
        identity = {'name': name, 'size': size, 'sha256': digest, 'chunk_size': chunk_size}
        done = set()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['identity'] != identity or not part.is_file() or part.stat().st_size != size:
                raise ValueError('resume_identity_mismatch: ' + name)
            done = set(state['completed_chunks'])
        fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, size)
        count = (size + chunk_size - 1) // chunk_size
        url = ('https://www.modelscope.cn/models/google/gemma-4-31B-it/resolve/'
               + quote(item['mirror_file_revision'], safe='') + '/' + quote(name, safe='/'))
        print(json.dumps({'event': 'file_start', 'file': name, 'bytes': size,
                          'chunks': count, 'resume_chunks': len(done), 'direct_https': True}), flush=True)

        def fetch(index):
            begin = index * chunk_size
            end = min(size, begin + chunk_size) - 1
            last_error = None
            for attempt in range(4):
                try:
                    with client.stream('GET', url, headers={'Range': f'bytes={begin}-{end}'}) as reply:
                        whole_file = begin == 0 and end + 1 == size
                        if reply.status_code != 206 and not (whole_file and reply.status_code == 200):
                            raise ValueError('range_response_required')
                        if reply.status_code == 206 and reply.headers.get('content-range') != f'bytes {begin}-{end}/{size}':
                            raise ValueError('content_range_mismatch')
                        position = begin
                        for chunk in reply.iter_bytes(chunk_size=1024 * 1024):
                            if position + len(chunk) > end + 1:
                                raise ValueError('range_overflow')
                            view = memoryview(chunk)
                            while view:
                                written = os.pwrite(fd, view, position)
                                if written <= 0:
                                    raise OSError('short_pwrite')
                                position += written
                                view = view[written:]
                        if position != end + 1:
                            raise ValueError('truncated_range')
                    return index
                except Exception as error:
                    last_error = type(error).__name__ + (':' + str(error) if isinstance(error, ValueError) else '')
                    if attempt < 3:
                        time.sleep(min(2 ** attempt, 8))
            raise RuntimeError(f'chunk_failed:{index}:{last_error}')

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(fetch, index): index for index in range(count) if index not in done}
                for future in as_completed(futures):
                    done.add(future.result())
                    os.fsync(fd)
                    atomic_json(state_path, {'identity': identity, 'completed_chunks': sorted(done)})
                    if len(done) % 8 == 0 or len(done) == count:
                        print(json.dumps({'event': 'progress', 'file': name, 'chunks_done': len(done),
                                          'chunks_total': count, 'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)
        finally:
            os.close(fd)
        if checksum(part) != digest:
            raise ValueError('final_sha256_mismatch: ' + name)
        part.replace(final)
        completed.append(item)
        print(json.dumps({'event': 'file_verified', 'file': name, 'bytes': size}), flush=True)
    client.close()
    receipt = {
        **manifest, 'files': completed, 'mirror': 'google/gemma-4-31B-it',
        'destination': str(args.output), 'all_required_files_verified': True,
        'proxy_environment_removed': True, 'http_client_trust_env': False,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': round(time.monotonic() - started, 3),
    }
    atomic_json(ROOT / 'WEIGHTS_READY.json', receipt)
    print(json.dumps({'event': 'weights_ready', 'path': str(args.output)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'event': 'download_error', 'error_type': type(error).__name__,
                          'error_code': str(error) if isinstance(error, (ValueError, RuntimeError)) else None,
                          'http_status': getattr(getattr(error, 'response', None), 'status_code', None)}), flush=True)
        raise SystemExit(1)
