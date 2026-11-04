"""Compare NPU threshold kernels on a real SSCD score tile; probe only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch_npu


def main():
    p=argparse.ArgumentParser(); p.add_argument('--vectors',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.npu.set_device(0); torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.matmul.allow_hf32=False
    raw=np.memmap(a.vectors,dtype=np.float32,mode='r').reshape(-1,512)
    q=torch.from_numpy(np.array(raw[:8192])).to('npu',dtype=torch.float16)
    d=torch.from_numpy(np.array(raw[65536:131072])).to('npu',dtype=torch.float16)
    scores=q@d.T; torch.npu.synchronize()
    scalar=torch.tensor(.74,dtype=scores.dtype,device='npu')
    tensor=torch.full((1,),.74,dtype=scores.dtype,device='npu')
    cases=[('ge_scalar',lambda:scores>=.74),('ge_device_scalar',lambda:scores>=scalar),
           ('ge_device_tensor',lambda:scores>=tensor),('gt_scalar',lambda:scores>.74),
           ('ge_flat_scalar',lambda:scores.flatten()>=.74),
           ('amax_rows',lambda:torch.amax(scores,dim=1)),
           ('ge_flat_device_tensor',lambda:scores.flatten()>=tensor)]
    for name,fn in cases:
        for repeat in range(3):
            torch.npu.synchronize(); begin=time.perf_counter(); value=fn(); torch.npu.synchronize(); elapsed=time.perf_counter()-begin
            record={'method':name,'repeat':repeat,'seconds':elapsed,'result_shape':list(value.shape),'score_format':torch_npu.get_npu_format(scores)}
            if name=='amax_rows':record['positive_rows']=int((value>=scalar).sum().item())
            else:record['candidate_count']=int(value.sum().item())
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
            print(json.dumps(record),flush=True);del value
    (out/'COMPLETE').touch()


if __name__=='__main__':main()
