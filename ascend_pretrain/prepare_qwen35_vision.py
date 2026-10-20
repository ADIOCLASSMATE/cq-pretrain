"""Extract pinned official vision weights; no language weights or runtime downloads.

Use HTTP byte ranges to fetch only the visual tensors from the official
safetensors shard. Every range is validated; unsupported range servers fail
instead of silently downloading the complete language checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .vision import MODEL_ID, REVISION, asset_path, vision_config, QwenVisionTower


def sha256(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='assets/Qwen3.5-0.8B-vision')
    parser.add_argument('--checkpoint',type=Path,help='Optional existing official safetensors shard')
    args=parser.parse_args();root=asset_path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text())
        if (manifest['model_id'],manifest['revision'])!=(MODEL_ID,REVISION):
            raise ValueError('Refusing to replace a different asset identity')
        for name,digest in manifest['files'].items():
            if sha256(root/name)!=digest:raise ValueError(f'Asset checksum differs: {name}')
        print(json.dumps(dict(status='already_verified',path=str(root))),flush=True);return
    base=f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/'
    session=requests.Session()
    def get(name,start=None,end=None):
        for attempt in range(4):
            try:
                headers={} if start is None else {'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
                query='' if start is None else f'?download=true&range={start}-{end}'
                with session.get(base+name+query,headers=headers,timeout=(20,60),stream=True) as response:
                    response.raise_for_status()
                    if start is not None:
                        expected=f'bytes {start}-{end}/'
                        if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(expected):
                            raise ValueError('Checkpoint server did not honor the requested byte range')
                    body=response.content
                if start is not None and len(body)!=end-start+1:raise ValueError('Short checkpoint byte range')
                return body
            except (requests.RequestException,ValueError):
                if attempt==3:raise
                time.sleep(1+attempt)
    for name in ('config.json','preprocessor_config.json','LICENSE'):
        (root/name).write_bytes(get(name))
    index=json.loads(get('model.safetensors.index.json'))
    prefix='model.visual.'
    names=sorted(k for k in index['weight_map'] if k.startswith(prefix))
    if not names:raise ValueError('The checkpoint contains no official model.visual tensors')
    shards={index['weight_map'][k] for k in names}
    if len(shards)!=1:raise ValueError('Expected the pinned single-shard checkpoint')
    shard=next(iter(shards))
    if args.checkpoint:
        with safe_open(str(args.checkpoint),framework='pt',device='cpu') as source:
            weights={k.removeprefix(prefix):source.get_tensor(k) for k in names}
        header_sha=None
    else:
        size=struct.unpack('<Q',get(shard,0,7))[0]
        if size>16<<20:raise ValueError('Unexpectedly large safetensors header')
        header=get(shard,8,7+size);header_sha=hashlib.sha256(header).hexdigest()
        metadata=json.loads(header);offset=8+size
        ordered=sorted(names,key=lambda k:metadata[k]['data_offsets'][0])
        groups=[]
        for name in ordered:
            start,end=metadata[name]['data_offsets']
            if groups and start-groups[-1][1]<=65536 and end-groups[-1][0]<=32<<20:
                groups[-1][1]=end;groups[-1][2].append(name)
            else:groups.append([start,end,[name]])
        weights={};types={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32}
        for i,(start,end,keys) in enumerate(groups):
            payload=get(shard,offset+start,offset+end-1)
            for name in keys:
                info=metadata[name];a,b=info['data_offsets']
                weights[name.removeprefix(prefix)]=torch.frombuffer(
                    bytearray(payload[a-start:b-start]),dtype=types[info['dtype']]).reshape(info['shape'])
            print(json.dumps(dict(range=i+1,ranges=len(groups),tensors=len(weights))),flush=True)
    # A strict, allocation-free shape/key check before publishing assets.
    with torch.device('meta'):tower=QwenVisionTower(vision_config(root))
    expected=tower.state_dict()
    if set(expected)!=set(weights) or any(expected[k].shape!=weights[k].shape for k in expected):
        raise ValueError('Official vision tensors do not match the implemented tower exactly')
    temporary=root/'vision.safetensors.partial'
    save_file(weights,str(temporary),metadata={'model_id':MODEL_ID,'revision':REVISION})
    temporary.replace(root/'vision.safetensors')
    manifest=dict(model_id=MODEL_ID,revision=REVISION,source_shard=shard,
        source_header_sha256=header_sha,tensors=len(weights),parameters=sum(t.numel() for t in weights.values()),
        files={name:sha256(root/name) for name in ('config.json','preprocessor_config.json','LICENSE','vision.safetensors')})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='prepared',path=str(root),**manifest)),flush=True)


if __name__=='__main__':main()
