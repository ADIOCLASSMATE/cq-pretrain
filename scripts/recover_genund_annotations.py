"""Recover pinned public annotation metadata into a separate derived directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import requests


def atomic(path, value):
    path = Path(path); tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)); os.replace(tmp, path)


def get(url, **kwargs):
    last = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 60), **kwargs); response.raise_for_status(); return response
        except requests.RequestException as exc:
            last = exc
            if attempt < 2: time.sleep(2)
    raise last


def recover(repo, root, download):
    out = root/repo.split('/')[-1]; out.mkdir(parents=True, exist_ok=True)
    try:
        info = get('https://huggingface.co/api/datasets/'+repo, params={'blobs': 'true'}).json()
        atomic(out/'hub_info.json', info)
        revision = info['sha']
        readme = get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md').text
        (out/'README.md').write_text(readme)
        paths = [x['rfilename'] for x in info.get('siblings', [])]
        result = {'repo': repo, 'revision': revision, 'files': len(paths), 'sample_paths': paths[:30],
                  'metadata_paths': [p for p in paths if p.endswith(('.json','.jsonl','.csv','.tsv'))], 'complete': True}
        if download and repo == 'long-xing1/ScaleCap-450k':
            item = next(x for x in info['siblings'] if x['rfilename'] == 'ScaleCap450k.json')
            target = out/'ScaleCap450k.json'; temporary = target.with_suffix('.json.partial')
            expected_size = item.get('size'); expected_sha = (item.get('lfs') or {}).get('sha256')
            if not target.exists():
                start = temporary.stat().st_size if temporary.exists() else 0
                headers = {'Range': f'bytes={start}-'} if start else {}
                response = get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/ScaleCap450k.json', headers=headers, stream=True)
                if start and response.status_code != 206: start = 0
                with temporary.open('ab' if start else 'wb') as stream:
                    for block in response.iter_content(8 << 20):
                        if block: stream.write(block)
                        atomic(out/'PROGRESS.json', {'bytes': stream.tell(), 'expected_bytes': expected_size, 'updated_at': time.time()})
                if expected_size is not None and temporary.stat().st_size != expected_size: raise ValueError('annotation length mismatch')
                os.replace(temporary, target)
            checksum = hashlib.sha256()
            with target.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''): checksum.update(block)
            if expected_sha is not None and checksum.hexdigest() != expected_sha: raise ValueError('annotation SHA mismatch')
            result['annotation'] = {'path': str(target), 'bytes': target.stat().st_size, 'sha256': checksum.hexdigest(),
                                    'expected_sha256': expected_sha}
        atomic(out/'COMPLETE.json', result); return result
    except Exception as exc:
        result = {'repo': repo, 'complete': False, 'error_type': type(exc).__name__, 'error': str(exc)[:500]}
        atomic(out/'ERROR.json', result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); parser.add_argument('--download-scalecap', action='store_true')
    args = parser.parse_args(); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda repo: recover(repo, root, args.download_scalecap),
                               ['HuggingFaceM4/FineVision', 'long-xing1/ScaleCap-450k']))
    atomic(root/'RECOVERY.json', result); print(json.dumps(result))


if __name__ == '__main__': main()
