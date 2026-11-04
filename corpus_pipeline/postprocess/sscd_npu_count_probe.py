"""Bounded timing of the dense-candidate safety count on the deployed backend."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch_npu


def main():
    p=argparse.ArgumentParser();p.add_argument('--vectors',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.matmul.allow_hf32=False
    torch.npu.set_option({'ACL_OP_SELECT_IMPL_MODE':'high_precision','ACL_OPTYPELIST_FOR_IMPLMODE':'MatMul'})
    raw=np.memmap(a.vectors,dtype=np.float32,mode='r').reshape(-1,512)
    qh=np.array(raw[:8192]);dh=np.array(raw[65536:131072]);q=torch.from_numpy(qh).to('npu');d=torch.from_numpy(dh).to('npu')
    records=[]
    def emit(v):
        records.append(v);print(json.dumps(v),flush=True)
    for repeat in range(4):
        torch.npu.synchronize();start=time.perf_counter();scores=q@d.T;torch.npu.synchronize();s1=time.perf_counter()
        mask=scores>=torch.tensor(.75-2e-6,device='npu',dtype=torch.float32);torch.npu.synchronize();s2=time.perf_counter()
        count=int(mask.sum().item());torch.npu.synchronize();s3=time.perf_counter()
        positions=mask.flatten().nonzero().flatten().cpu().numpy();s4=time.perf_counter()
        rows=positions//65536;cols=positions%65536
        values=(qh[rows].astype(np.float64)*dh[cols].astype(np.float64)).sum(1);s5=time.perf_counter()
        emit(dict(kind='production_chain',repeat=repeat,matmul=s1-start,threshold=s2-s1,sum_item=s3-s2,
                  nonzero_egress=s4-s3,cpu_float64_rerank=s5-s4,total=s5-start,candidates=count,
                  full_square_128_cards_hours=90535667**2/(8192*65536/(s5-start)*128*3600)))
        del scores,mask
    for dense in (False,True):
        mask=torch.ones((8192,65536),device='npu',dtype=torch.bool) if dense else q@d.T>=torch.tensor(.75,device='npu')
        for name,fn in [('sum_default',lambda:mask.sum().item()),('sum_int32',lambda:mask.sum(dtype=torch.int32).item()),('count_nonzero',lambda:torch.count_nonzero(mask).item())]:
            for repeat in range(3):
                torch.npu.synchronize();start=time.perf_counter();n=fn();torch.npu.synchronize();secs=time.perf_counter()-start
                emit(dict(kind='count',method=name,dense=dense,repeat=repeat,seconds=secs,count=n))
        del mask
    Path(a.output).write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
