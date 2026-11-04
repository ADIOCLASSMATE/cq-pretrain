"""Compare every model, FP32 optimizer/master shard, RNG and consumed cursor."""
import argparse
import json
from pathlib import Path
import torch


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or not torch.equal(a,b):
            delta=(a.float()-b.float()).abs().max().item() if isinstance(b,torch.Tensor) and a.shape==b.shape and a.numel() else None
            raise AssertionError(f'{path}: tensors differ, max_abs={delta}')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(f'{path}: keys differ')
        for key in a:equal(a[key],b[key],f'{path}.{key}')
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):raise AssertionError(f'{path}: sequence differs')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{path}[{i}]')
    elif a!=b:raise AssertionError(f'{path}: {a!r} != {b!r}')


def compare(first,second,output):
    first,second=Path(first),Path(second)
    manifests=[json.loads((p/'manifest.json').read_text()) for p in (first,second)]
    equal(*manifests)
    files=sorted(first.glob('*.pt'))
    if {p.name for p in files}!={p.name for p in second.glob('*.pt')}:raise AssertionError('Checkpoint shards differ')
    for path in files:
        equal(torch.load(path,map_location='cpu',weights_only=False),
              torch.load(second/path.name,map_location='cpu',weights_only=False),path.name)
    result=dict(passed=True,bitwise_identical=True,world=manifests[0]['world'],files=len(files),
                checked=['model','FP32 Adam/master shards','CPU/NPU RNG','consumed data state'])
    Path(output).write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',required=True);p.add_argument('--second',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(compare(a.first,a.second,a.output)))
