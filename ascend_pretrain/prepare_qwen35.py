"""Download a pinned official Qwen3.5-0.8B checkpoint for offline CQ training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import shutil
import requests

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_upstream(root,metadata):
    if metadata.get('sha')!=REVISION:raise ValueError('Local download metadata has a different revision')
    for info in metadata['siblings']:
        path=root/info['rfilename']
        if not path.is_file():continue
        digest=hashlib.sha256() if 'lfs' in info else hashlib.sha1()
        if 'lfs' not in info:digest.update(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8<<20),b''):digest.update(block)
        expected=info['lfs']['sha256'] if 'lfs' in info else info['blobId']
        if digest.hexdigest()!=expected:raise ValueError(f'Official Git/LFS hash mismatch: {path.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='assets/Qwen3.5-0.8B')
    parser.add_argument('--source',type=Path,help='Existing local copy of the pinned official checkpoint')
    args = parser.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    manifest = root/'manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data['revision'] != REVISION or any(sha256(root/k) != v for k,v in data['files'].items()):
            raise ValueError('Pinned Qwen assets changed')
        source=args.source or (Path(data['local_source']) if data.get('local_source') else None)
        if source:
            verify_upstream(root,json.loads((source/'.hfd/repo_metadata.json').read_text()))
        print(json.dumps(dict(status='verified', root=str(root))), flush=True); return
    if args.source:
        source=args.source.resolve()
        config=json.loads((source/'config.json').read_text())
        pinned=json.loads((Path(__file__).resolve().parents[1]/'assets/Qwen3.5-0.8B-vision/config.json').read_text())
        if config!=pinned:raise ValueError('Local checkpoint config differs from the pinned official revision')
        index=json.loads((source/'model.safetensors.index.json').read_text())
        names=['config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json',
               'model.safetensors.index.json','LICENSE',*sorted(set(index['weight_map'].values()))]
        files={}
        for name in names:
            shutil.copyfile(source/name,root/name);files[name]=sha256(root/name)
        verify_upstream(root,json.loads((source/'.hfd/repo_metadata.json').read_text()))
        manifest.write_text(json.dumps(dict(model_id=MODEL_ID,revision=REVISION,
            files=files,local_source=str(source)),indent=2)+'\n')
        print(json.dumps(dict(status='prepared_local',root=str(root),files=files)),flush=True);return

    def get(url, **kwargs):
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=(15,60), **kwargs)
                response.raise_for_status(); return response
            except requests.RequestException:
                if attempt == 4: raise
                time.sleep(1+attempt)

    meta = get(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true').json()
    siblings = {item['rfilename']: item for item in meta['siblings']}
    names = ['config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json',
             'generation_config.json','model.safetensors.index.json','LICENSE']
    names += sorted(name for name in siblings if name.endswith('.safetensors'))
    names=[name for name in names if name in siblings]
    base = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/'

    def download(name):
        info = siblings[name]; expected = info.get('lfs',{}).get('sha256')
        path = root/name
        if path.exists() and path.stat().st_size == info['size'] and (not expected or sha256(path)==expected):
            return name, sha256(path)
        temporary = path.with_suffix(path.suffix+'.partial')
        with get(base+name+'?download=true',stream=True) as response, temporary.open('wb') as f:
            for chunk in response.iter_content(1<<20):
                f.write(chunk)
        digest = sha256(temporary)
        if temporary.stat().st_size != info['size'] or (expected and digest != expected):
            raise ValueError(f'Download size/checksum mismatch: {name}')
        temporary.replace(path)
        print(json.dumps(dict(file=name,bytes=path.stat().st_size,sha256=digest)),flush=True)
        return name,digest

    with ThreadPoolExecutor(3) as pool:
        files = dict(pool.map(download,names))
    data = dict(model_id=MODEL_ID,revision=REVISION,files=files)
    manifest.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(status='prepared',root=str(root))),flush=True)


if __name__ == '__main__':main()
