"""Fetch pinned author checkpoints, with direct connections and SHA-256 verification."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import requests

CF_REV = 'ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4'
UNIFD_REV = '030495aea3300a8b54c0ec37ec7fe1dd7e63c619'
FC_SHA256 = '477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847'
HF_FILES = {
    'cf224': {'repo': 'OwensLab/commfor-model-224', 'revision': '26afc31e6b40c312c3fd42c05a758be62446215b', 'sha256': 'a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0', 'size': 86678644},
    'cf384': {'repo': 'OwensLab/commfor-model-384', 'revision': '6076002bf0d9dd37537f965ee2f06f826c333b61', 'sha256': 'b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387', 'size': 87262324},
    'clip_vitl14': {'repo': 'openai/clip-vit-large-patch14', 'revision': '32bd64288804d66eefd0ccbe215aa642df71cc41', 'sha256': 'a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb', 'size': 1710540580},
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def download(url, path, expected=None, size=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not expected or sha256(path) == expected):
        return {'path': str(path), 'url': url, 'sha256': sha256(path), 'bytes': path.stat().st_size}
    session = requests.Session(); session.trust_env = False
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(4):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={start}-'} if start else {}
            with session.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
                r.raise_for_status()
                if start and r.status_code == 206:
                    if not r.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise ValueError('Invalid range response')
                    mode = 'ab'
                else: mode = 'wb'
                with part.open(mode) as f:
                    for chunk in r.iter_content(8 << 20):
                        if chunk: f.write(chunk)
            if size is not None and part.stat().st_size != size: raise ValueError('download size mismatch')
            digest = sha256(part)
            if expected and digest != expected: raise ValueError('download SHA-256 mismatch')
            part.replace(path)
            print(json.dumps({'downloaded': str(path), 'sha256': digest, 'bytes': path.stat().st_size}), flush=True)
            return {'path': str(path), 'url': url, 'sha256': digest, 'bytes': path.stat().st_size}
        except Exception:
            if attempt == 3: raise
            time.sleep(attempt + 1)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hf-endpoint', default='https://huggingface.co')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    specs = []
    for name, info in HF_FILES.items():
        base = f'{args.hf_endpoint}/{info["repo"]}/resolve/{info["revision"]}'
        specs.extend([(name+'/weights', base+'/model.safetensors?download=true', args.output/name/'model.safetensors', info['sha256'], info['size']), (name+'/config', base+'/config.json', args.output/name/'config.json', None, None)])
    source_files = {
        'community-forensics': ('JeongsooP/Community-Forensics', CF_REV, ['models.py','dataloader.py','dataprocessor_hf.py','custom_transforms.py','utils.py','eval.py','README.md','LICENSE']),
        'unifd': ('WisconsinAIVision/UniversalFakeDetect', UNIFD_REV, ['validate.py','models/clip_models.py','models/clip/model.py','models/clip/clip.py','README.md','LICENSE','pretrained_weights/fc_weights.pth']),
    }
    for name, (repo, revision, files) in source_files.items():
        for file in files:
            expected = FC_SHA256 if file == 'pretrained_weights/fc_weights.pth' else None
            specs.append((name+'/'+file, f'https://raw.githubusercontent.com/{repo}/{revision}/{file}', args.output/'official_source'/name/file, expected, None))
    manifest = {'network': {'proxy': False, 'hf_download_endpoint': args.hf_endpoint, 'mirror_note': 'Pinned author repository revisions; weights checked against published LFS SHA-256. No model code imported from hub.'}, 'models': HF_FILES, 'source_revisions': {'community-forensics': CF_REV, 'unifd': UNIFD_REV}, 'files': {}}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        pending = {pool.submit(download, url, path, expected, size): name for name,url,path,expected,size in specs}
        for future in concurrent.futures.as_completed(pending):
            manifest['files'][pending[future]] = future.result()
            tmp=args.output/'manifest.partial.json';tmp.write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('All pinned model artifacts ready', flush=True)

if __name__ == '__main__': main()
