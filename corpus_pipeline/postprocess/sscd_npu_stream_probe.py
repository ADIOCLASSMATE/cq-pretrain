"""Bounded resident-database NPU probe with real shared-filesystem query reads."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch_npu


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    p.add_argument('--queries',type=int,default=24);a=p.parse_args()
    root=Path(a.index);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.matmul.allow_hf32=False;sync=torch.npu.synchronize
    def emit(**v):
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n');f.flush()
        print(json.dumps(v),flush=True)
    blocks=json.loads((root/'catalog'/'COMPLETE.json').read_text())['blocks']
    def read_rows(block,limit=None):
        meta=json.loads((root/block['receipt_path']).read_text())
        path=next(x['path'] for x in meta['artifacts'] if x['path'].endswith('/vectors.f32'))
        view=np.memmap(root/path,dtype=np.float32,mode='r').reshape(-1,512)
        return np.array(view if limit is None else view[:limit],copy=True)
    begin=time.perf_counter();dbhost=np.concatenate([read_rows(b) for b in blocks[:4]],axis=0);read_seconds=time.perf_counter()-begin
    begin=time.perf_counter();db32=torch.from_numpy(dbhost).to('npu');db16=db32.half();sync();transfer_seconds=time.perf_counter()-begin
    emit(kind='database',rows=len(dbhost),bytes=dbhost.nbytes,read_seconds=read_seconds,
         copy_and_cast_seconds=transfer_seconds,allocated_gib=torch.npu.memory_allocated()/2**30)
    cutoff=torch.tensor(.74,device='npu',dtype=torch.float16);cutoff32=torch.tensor(.75,device='npu',dtype=torch.float32)
    records=[]
    for number in range(a.queries):
        block=blocks[8+number*17];start=time.perf_counter();qh=read_rows(block,8192);s1=time.perf_counter()
        q32=torch.from_numpy(qh).to('npu');q16=q32.half();sync();s2=time.perf_counter()
        exact_kept=0;candidates=0;serialized=[]
        for offset in range(0,len(dbhost),65536):
            scores=q16 @ db16[offset:offset+65536].T
            mask=scores>=cutoff
            positions=mask.flatten().nonzero().flatten();cols=positions%mask.shape[1]+offset;rows=positions//mask.shape[1]
            candidates+=len(positions)
            for j in range(0,len(rows),4096):
                rr=rows[j:j+4096];cc=cols[j:j+4096]
                exact=(q32.index_select(0,rr)*db32.index_select(0,cc)).sum(1)
                accept=exact>=cutoff32
                values=torch.stack([rr[accept].to(torch.float32),cc[accept].to(torch.float32),exact[accept]],dim=1).cpu().numpy()
                exact_kept+=len(values);serialized.append(values)
            del scores,mask,positions,rows,cols
        sync();s3=time.perf_counter()
        # Real metadata output I/O, isolated from production search artifacts.
        np.save(out/('edges_%03d.npy'%number),np.concatenate(serialized) if serialized else np.zeros((0,3),np.float32))
        s4=time.perf_counter();pairs=len(qh)*len(dbhost);seconds=s4-start
        row=dict(kind='query',number=number,queries=len(qh),database_rows=len(dbhost),read_seconds=s1-start,
                 h2d_cast_seconds=s2-s1,search_exact_rerank_seconds=s3-s2,write_seconds=s4-s3,total_seconds=seconds,
                 candidates=candidates,accepted=exact_kept,pairs_per_second=pairs/seconds,
                 full_square_128_cards_hours=90535667**2/(pairs/seconds*128*3600),
                 memory_gib=torch.npu.memory_allocated()/2**30,bytes_read=qh.nbytes,
                 ids_are_probe_local_not_production_ids=True)
        emit(**row);records.append(row)
    # Independent pure FP32 control: optional exact math backend avoids the
    # empirical low-precision candidate margin but may be much slower.
    for repeat in range(4):
        sync();start=time.perf_counter();scores=q32@db32[:65536].T;sync();s1=time.perf_counter()
        pos=(scores>=cutoff32).flatten().nonzero().flatten();sync();s2=time.perf_counter()
        emit(kind='pure_fp32_control',repeat=repeat,matmul_seconds=s1-start,threshold_nonzero_seconds=s2-s1,
             pairs=len(qh)*65536,total_seconds=s2-start,full_square_128_cards_hours=90535667**2/((len(qh)*65536)/(s2-start)*128*3600),candidates=len(pos))
        del scores,pos
    (out/'COMPLETE.json').write_text(json.dumps({'probe_only':True,'queries':records},indent=2)+'\n')


if __name__=='__main__':main()
