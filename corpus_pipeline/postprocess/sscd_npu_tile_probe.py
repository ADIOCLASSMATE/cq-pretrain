"""Bounded real-vector NPU matrix/threshold/rerank probe, never production output."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True); p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--tiles', default='8192x65536,32768x32768,65536x65536')
    args = p.parse_args(); root = Path(args.index); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu
    torch.set_num_threads(4); torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.matmul.allow_hf32 = False
    threshold = .75; margin = .01
    sync = torch.npu.synchronize
    def emit(kind, **kw):
        value = dict(kind=kind, at=time.time(), **kw)
        with (out/'events.jsonl').open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(value), flush=True)
        return value
    total = json.loads((root/'catalog'/'COMPLETE.json').read_text())
    blocks = total['blocks']
    # One SHA bucket is effectively shuffled by content hash. Two nonoverlapping
    # offsets from the same real block allow deliberate self-match accuracy checks.
    block = blocks[0]; receipt = root/block['receipt_path']; meta=json.loads(receipt.read_text())
    vectors = next(x for x in meta['artifacts'] if x['path'].endswith('/vectors.f32'))
    path = root/vectors['path']; payload_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if payload_sha != vectors['sha256']: raise ValueError('Packed input checksum mismatch')
    tiles=[tuple(map(int,x.split('x'))) for x in args.tiles.split(',')]
    maxrows=max(max(x) for x in tiles)
    start=time.perf_counter(); raw=np.memmap(path,dtype=np.float32,mode='r').reshape(-1,512)
    qhost=np.array(raw[:maxrows],copy=True); dhost=np.array(raw[:maxrows],copy=True)
    host_read=time.perf_counter()-start
    emit('environment',host=socket.gethostname(),torch=torch.__version__,torch_npu=torch_npu.__version__,
         device=torch.npu.get_device_name(0),visible_devices=torch.npu.device_count(),
         input_path=str(path),input_sha256=payload_sha,rows=len(raw),host_copy_seconds=host_read,
         threshold=threshold,margin=margin,full_catalog_vectors=total.get('total_vectors',90535667),
         source='real packed DCVLM SSCD FP32 vectors',probe_only=True,allow_hf32=False)
    start=time.perf_counter(); q32=torch.from_numpy(qhost).to('npu'); d32=torch.from_numpy(dhost).to('npu'); sync()
    emit('transfer',dtype='float32',bytes=qhost.nbytes+dhost.nbytes,seconds=time.perf_counter()-start)
    # Full CPU FP32 reference over a small deliberately overlapping submatrix.
    ref=qhost[:1024] @ dhost[:2048].T
    precision=[]; records=[]
    for dtype in (torch.float16,torch.bfloat16):
        name=str(dtype).split('.')[-1]
        start=time.perf_counter(); q=q32.to(dtype); d=d32.to(dtype); sync()
        device_threshold=torch.tensor(threshold-margin,dtype=dtype,device='npu')
        exact_threshold=torch.tensor(threshold,dtype=torch.float32,device='npu')
        emit('device_cast',dtype=name,seconds=time.perf_counter()-start)
        approx=(q[:1024] @ d[:2048].T).float().cpu().numpy()
        err=np.abs(approx-ref); exact_positive=ref>=threshold; candidate=approx>=threshold-margin
        precision.append(emit('accuracy',dtype=name,pairs=ref.size,max_abs_error=float(err.max()),
            p999_abs_error=float(np.quantile(err,.999)),reference_positive_pairs=int(exact_positive.sum()),
            missed_positive_pairs_with_margin=int((exact_positive & ~candidate).sum()),
            measured_margin_not_global_mathematical_guarantee=True))
        for m,n in tiles:
            try:
                for repeat in range(args.repeats):
                    sync(); wall=time.perf_counter(); s0=wall
                    scores=q[:m] @ d[:n].T; sync(); s1=time.perf_counter()
                    mask=scores >= device_threshold; sync(); s2=time.perf_counter()
                    indices=mask.nonzero(); sync(); s3=time.perf_counter()
                    rows=indices[:,0]; cols=indices[:,1]
                    del scores,mask
                    kept=0; max_candidate_abs_error=0.; candidate_count=len(indices)
                    # Exact candidate rerank uses FP32 multiplication and sum.
                    for offset in range(0,candidate_count,4096):
                        rr=rows[offset:offset+4096]; cc=cols[offset:offset+4096]
                        exact=(q32.index_select(0,rr)*d32.index_select(0,cc)).sum(dim=1)
                        kept+=int((exact>=exact_threshold).sum().item())
                    sync(); s4=time.perf_counter()
                    # Exercise candidate ID egress, rather than reporting matmul alone.
                    ids=indices.cpu().numpy(); sync(); s5=time.perf_counter()
                    elapsed=s5-wall; pairs=m*n
                    rec=emit('tile',dtype=name,m=m,n=n,repeat=repeat,cold_iteration=repeat==0,
                        matmul_seconds=s1-s0,threshold_seconds=s2-s1,nonzero_seconds=s3-s2,
                        fp32_rerank_seconds=s4-s3,ids_to_host_seconds=s5-s4,total_seconds=elapsed,
                        candidates=candidate_count,exact_positive_pairs=kept,includes_self_matches=True,
                        pairs_per_second=pairs/elapsed,matmul_tflops=2*pairs*512/(s1-s0)/1e12,
                        full_pipeline_tflops=2*pairs*512/elapsed/1e12,
                        memory_allocated_gib=torch.npu.memory_allocated()/2**30,
                        estimated_full_square_128_cards_hours=90535667**2/(pairs/elapsed*128*3600),
                        extrapolation_excludes_full_corpus_io_scheduling_and_distribution=True)
                    records.append(rec)
                    del indices,rows,cols,ids
                gc.collect(); torch.npu.empty_cache()
            except Exception as exc:
                emit('tile_error',dtype=name,m=m,n=n,error_type=type(exc).__name__,error=str(exc)[:1200])
                gc.collect(); torch.npu.empty_cache()
        del q,d; gc.collect(); torch.npu.empty_cache()
    summary={'probe_only':True,'precision':precision,'tiles':records,'completed_at':time.time()}
    (out/'COMPLETE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
