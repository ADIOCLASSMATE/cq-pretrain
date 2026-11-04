"""Download and verify the official Gemma4 31B checkpoint over direct HTTPS."""
from __future__ import annotations

import os

# Remove every common proxy spelling before importing network clients.
for _key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    os.environ.pop(_key, None)
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx
from huggingface_hub import HfApi, set_client_factory, snapshot_download

MODEL_ID = 'google/gemma-4-31B-it'
REVISION = '842da3794eaa0b77d5f08bae87a17459d91ff475'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'weights')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_client_factory(lambda: httpx.Client(trust_env=False, timeout=300, follow_redirects=True))
    started = time.monotonic()
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    required = [s for s in info.siblings if s.rfilename.endswith(('.json', '.safetensors', '.jinja'))]
    metadata = {'model_id': MODEL_ID, 'revision': info.sha, 'gated': info.gated,
                'proxy_environment_removed': True, 'http_client_trust_env': False,
                'files': [{'name': s.rfilename, 'size': s.size,
                           'expected_lfs_sha256': getattr(s.lfs, 'sha256', None)} for s in required]}
    (args.output.parent / 'download_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'event': 'download_start', 'model_id': MODEL_ID, 'revision': info.sha,
                      'required_bytes': sum(s.size or 0 for s in required), 'direct_https': True}), flush=True)
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.output,
                      allow_patterns=[s.rfilename for s in required], max_workers=4)
    verified = []
    for item in metadata['files']:
        path = args.output / item['name']
        if path.stat().st_size != item['size']:
            raise ValueError('downloaded_size_mismatch: ' + item['name'])
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(chunk)
        checksum = digest.hexdigest()
        if item['expected_lfs_sha256'] and checksum != item['expected_lfs_sha256']:
            raise ValueError('downloaded_hash_mismatch: ' + item['name'])
        verified.append({**item, 'sha256': checksum})
        print(json.dumps({'event': 'file_verified', 'name': item['name'], 'bytes': item['size']}), flush=True)
    receipt = {**metadata, 'files': verified, 'all_required_files_verified': True,
               'completed_at_utc': datetime.now(timezone.utc).isoformat(),
               'duration_seconds': round(time.monotonic() - started, 3)}
    path = args.output.parent / 'WEIGHTS_READY.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps({'event': 'weights_ready', 'receipt': str(path)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Avoid logging signed redirect URLs or credentials from client exceptions.
        print(json.dumps({'event': 'download_error', 'error_type': type(error).__name__,
                          'http_status': getattr(getattr(error, 'response', None), 'status_code', None)}), flush=True)
        raise SystemExit(1)
