"""Read-only serial configuration grid on a dedicated NPU; no queue access."""
from __future__ import annotations
import argparse, copy, json, os, signal, socket, subprocess, threading, time
from pathlib import Path
import numpy as np
from corpus_pipeline.production.common import atomic_json, file_sha
from .prepare_profile import sample_assets


def compare(reference, candidate):
    if len(reference)!=len(candidate): raise AssertionError('coverage changed')
    maximum={'cf384':0.,'unifd':0.}; cosine=1.; flips=0
    for a,b in zip(reference,candidate):
        ignored={'decode_seconds','scores','sscd'}
        if {k:v for k,v in a.items() if k not in ignored}!={k:v for k,v in b.items() if k not in ignored}: raise AssertionError('metadata/status changed')
        if a['status'] not in ('ok','too_small'): raise AssertionError('failed inference')
        if a['status']=='too_small': continue
        for name,threshold in [('cf384',.90),('unifd',.95)]:
            x,y=a['scores'][name],b['scores'][name]
            if x['status']!='ok' or y['status']!='ok' or not np.isfinite([x['prob'],y['prob']]).all(): raise AssertionError('failed detector')
            maximum[name]=max(maximum[name],abs(x['prob']-y['prob']))
            flips+=int((x['prob']>=threshold)!=(y['prob']>=threshold))
        x,y=a['sscd']['embedding'].astype('float64'),b['sscd']['embedding'].astype('float64')
        c=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
        if not np.isfinite(c): raise AssertionError('invalid SSCD')
        cosine=min(cosine,c)
    if max(maximum.values())>.001 or cosine<.999 or flips: raise AssertionError('qualification threshold exceeded')
    return dict(images=len(reference),max_probability_error=maximum,min_sscd_cosine=cosine,decision_flips=flips,metadata_equal=True)


def configurations():
    values=[(32,4,4,1,False),(32,8,4,2,False),(32,16,4,2,False),
            (32,8,4,2,True),(32,16,4,2,True),(64,8,4,2,True),
            (64,16,1,2,True),(128,8,4,2,True),(128,16,1,2,True),
            (256,8,4,2,True),(256,16,1,2,True)]
    return [dict(batch_size=b,decode_workers=d,cpu_threads=c,prefetch_batches=p,fast_prepare=f) for b,d,c,p,f in values]


def snapshot_smi():
    try: return subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=5).stdout[-12000:]
    except Exception as e:return type(e).__name__


def run_case(worker,assets,config,output,label,reference=None,prepared=None):
    worker.configure(**config)
    # Warm the exact forward shape before measuring. Every timed full-path case
    # rereads and decodes the original compressed bytes; no tensor cache.
    warm=list(worker.iter_results(assets[:config['batch_size']]))
    if any(x['status'] not in ('ok','too_small') for x in warm):raise RuntimeError('warmup failure')
    del warm
    before=copy.deepcopy(worker.stats)
    worker.synchronize(); worker.torch.npu.reset_peak_memory_stats(worker.device)
    smi=[]; done=threading.Event()
    def monitor():
        while not done.is_set():
            smi.append(dict(at=time.time(),info=snapshot_smi()))
            done.wait(1)
    t=threading.Thread(target=monitor,daemon=True);t.start()
    started=time.perf_counter(); cpu=time.process_time()
    try:
        if prepared is None: values=list(worker.iter_results(assets))
        else:
            values=[]
            for batch in prepared: values.extend(worker._infer_prepared(copy.deepcopy(batch),worker.names))
        worker.synchronize()
        elapsed=time.perf_counter()-started; cpu=time.process_time()-cpu
    finally:done.set(); t.join(timeout=7)
    if any(x['status'] not in ('ok','too_small') for x in values):
        atomic_json(output/(label+'-FAILURE.json'),[{k:v for k,v in x.items() if k!='sscd'} for x in values if x['status'] not in ('ok','too_small')]);raise RuntimeError('incomplete inference')
    delta={k:worker.stats[k]-before[k] for k in ('decode_seconds','preparation_wait_seconds','inference_seconds','stream_seconds')}
    row=dict(label=label,config=config,images=len(values),seconds=elapsed,images_per_second=len(values)/elapsed,process_cpu_cores=cpu/elapsed,
        timing=delta,model_seconds={k:worker.stats['model_seconds'][k]-before['model_seconds'][k] for k in worker.names},
        peak_npu_allocated_bytes=worker.torch.npu.max_memory_allocated(worker.device),peak_npu_reserved_bytes=worker.torch.npu.max_memory_reserved(worker.device),
        execution_identity=worker.execution_identity,comparison=compare(reference,values) if reference is not None else None)
    atomic_json(output/(label+'-SMI.json'),smi);atomic_json(output/(label+'.json'),row)
    print(json.dumps({k:v for k,v in row.items() if k!='execution_identity'}),flush=True)
    return row, values


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=512);p.add_argument('--device',default='npu:0');p.add_argument('--configs',type=Path);p.add_argument('--finalist',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('probe deadline')));signal.alarm(1500)
    from .tuned_inference import TunedResidentInferenceWorker
    assets=sample_assets(args.root,args.count);atomic_json(args.output/'ASSETS.json',assets)
    configs=json.loads(args.configs.read_text()) if args.configs else configurations()
    rows=[]
    with TunedResidentInferenceWorker(device=args.device,dtype='float32',**configs[0]) as worker:
        atomic_json(args.output/'IDENTITY.json',worker.identity)
        reference=None
        for i,config in enumerate(configs):
            row,values=run_case(worker,assets,config,args.output,f'grid-{i:02d}',reference)
            if reference is None:reference=values
            rows.append(row);atomic_json(args.output/'GRID.json',rows)
        if not args.configs:
            winner=max(rows[3:],key=lambda x:x['images_per_second'])['config']
            # Two more alternating trials on 1024 real originals; loading excluded.
            final_assets=sample_assets(args.root,1024);final_reference=None;final=[]
            for i,config in enumerate([configs[0],winner,winner,configs[0]]):
                row,values=run_case(worker,final_assets,config,args.output,f'final-{i:02d}',final_reference)
                if final_reference is None:final_reference=values
                final.append(row)
            atomic_json(args.output/'FINALISTS.json',final)
            # Preparation-free upper bound using the same qualified FP32 arithmetic.
            worker.configure(**winner)
            prepped=[list(worker.pool.map(lambda a:worker._prepare(a,worker.names),final_assets[i:i+winner['batch_size']])) for i in range(0,256,winner['batch_size'])]
            row,_=run_case(worker,final_assets[:256],winner,args.output,'model-only-ceiling',prepared=prepped)
            atomic_json(args.output/'SELECTED.json',winner)
        atomic_json(args.output/'COMPLETE.json',dict(success=True,hostname=socket.gethostname(),images=args.count,scope='read_only_real_AS100M_configuration_grid',queue_mutations=0,finished_at=time.time(),code_sha256={x:file_sha(Path(__file__).with_name(x)) for x in ('inference.py','optimized_inference.py','tuned_inference.py','fast_prepare.py')},cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip()))
    signal.alarm(0)
if __name__=='__main__':main()
