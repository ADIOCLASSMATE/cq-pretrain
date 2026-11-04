"""Leased, bounded workers shared by both collections; no platform allocation here."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid

from corpus_pipeline.cluster import frontpass
from corpus_pipeline.production.records import snapshot
from corpus_pipeline.production.common import digest,file_sha
from corpus_pipeline.production.potential import compile_source_contracts
from .config import load,require


def frontpass_worker(config, *, device='cpu', queue_backend='local', max_tasks=None, delegate=None):
    from .workflow import verify_execution
    cfg=verify_execution(config);root=Path(cfg['output'])/'frontpass'
    plan=json.loads((root/'plan.json').read_text());rules=frontpass.rules_for_plan(plan)
    snap=snapshot(plan['snapshot_root'])
    require(plan['snapshot_fingerprint']==snap['fingerprint'] and plan['rules_sha256']==digest(rules),
            'Worker input/rules differ from frozen frontpass plan')
    require(plan['thresholds_sha256']==file_sha(frontpass.BASE/'detector_calibration/thresholds.json') and
            plan['layout_contracts_sha256']==file_sha(frontpass.BASE/'prepared/config/layout_source_contracts.json'),
            'Worker thresholds/source contracts changed')
    if queue_backend=='rpc':
        from corpus_pipeline.cluster import queue_rpc as queue
        queue.health(root)
    elif queue_backend=='local':
        require(device=='cpu' and cfg['scope']=='validation','Distributed workers must use the control-host queue RPC')
        from corpus_pipeline.cluster import queue
    else:raise ValueError('Unknown queue transport')
    contracts=compile_source_contracts(json.loads((frontpass.BASE/'prepared/config/layout_source_contracts.json').read_text()))
    thresholds=json.loads((frontpass.BASE/'detector_calibration/thresholds.json').read_text())['review_thresholds']
    cache_enabled=cfg['cache']['enabled']
    aggregation={'target_images':cfg['compute']['batch_size']*4,'max_records':8192,'max_record_bytes':64*1024*1024}
    owned=delegate is None
    if owned:
        from corpus_pipeline.cluster.inference import ResidentInferenceWorker
        compute=cfg['compute']
        worker_class=ResidentInferenceWorker
        if device!='cpu' and compute['inference_engine']=='optimized':
            aggregation=frontpass.optimized_configuration(root,dtype=compute['dtype'],batch_size=compute['batch_size'],threads=compute['threads'])
            require(aggregation is not None,'Optimized NPU execution requires a current qualified receipt')
            from corpus_pipeline.cluster.optimized_inference import OptimizedResidentInferenceWorker
            worker_class=OptimizedResidentInferenceWorker
        model_paths={key:compute[key] for key in ('models_dir','sscd_model') if compute.get(key)}
        delegate=worker_class(device=device,dtype=compute['dtype'],
            batch_size=compute['batch_size'],threads=compute['threads'],**model_paths)
        delegate.__enter__()
    if cache_enabled:
        from .cache import CachedInferenceWorker
        worker=CachedInferenceWorker(delegate,cfg['cache']['directory'],histories=cfg['cache']['histories'])
    else:
        worker=delegate
    worker_id=f'{socket.gethostname()}-{device.replace(":","_")}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    from corpus_pipeline.production.common import atomic_json
    worker_root=root/'workers'/worker_id
    atomic_json(worker_root/'model_identity.json',worker.identity)
    atomic_json(worker_root/'execution_identity.json',getattr(worker,'execution_identity',{}))
    done=0
    try:
        while max_tasks is None or done<max_tasks:
            if (Path(cfg['output'])/'STOP').exists():break
            claim=queue.claim(root,worker_id)
            if claim is None:
                state=queue.status(root)
                if state['all_tasks_completed']:break
                if state['counts'].get('failed'):raise RuntimeError('Terminal failed frontpass task')
                if queue_backend=='local':break
                time.sleep(15);continue
            stop=threading.Event();errors=[]
            def heartbeat():
                while not stop.wait(30):
                    try:queue.heartbeat(root,claim['task_id'],claim['token'])
                    except Exception as exc:errors.append(str(exc));return
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
            try:
                summary,artifacts=frontpass.process_task(root,claim,worker,snap,rules,contracts,thresholds,aggregation=aggregation)
                require(not errors,'Worker lost lease before publication')
                queue.complete(root,claim['task_id'],claim['token'],artifacts=artifacts,
                    processed_records=summary['counts']['records'],processed_images=summary['counts'].get('image_occurrences',0))
                done+=1
            except BaseException as exc:
                try:queue.fail(root,claim['task_id'],claim['token'],str(exc),retryable=True)
                except Exception:pass
                raise
            finally:stop.set();thread.join(timeout=2)
        if cache_enabled:
            worker.seal_cache()
            cache_statistics={'enabled':True,**dict(worker.cache_stats)}
        else:
            cache_statistics={'enabled':False,'status':'disabled','reads':0,'writes':0,'seals':0}
        atomic_json(worker_root/'FINISHED.json',{'complete':True,'tasks_completed':done,'cache':cache_statistics})
    finally:
        if callable(getattr(worker,'close',None)):worker.close()
    return {'tasks_completed_this_process':done,'worker_id':worker_id}


def run(config,stage,*,device='cpu',queue_backend='local',max_tasks=None):
    if stage=='frontpass':return frontpass_worker(config,device=device,queue_backend=queue_backend,max_tasks=max_tasks)
    from .workflow import verify_execution
    cfg=verify_execution(config);root=Path(cfg['output'])
    if (root/'STOP').exists():return {'complete':False,'phase':'stopped'}
    if stage=='sscd_search':
        from corpus_pipeline.postprocess import sscd_exact_npu as search
        area=root/'sscd_search'
        if not (area/'calibration/COMPLETE.json').exists():
            search.calibrate(area,root/'sscd_index/calibration/unified_reference/evidence.npz',device=device)
        return search.worker(area,worker_id=f'{socket.gethostname()}-{device}-{os.getpid()}',device=device,max_tasks=max_tasks)
    if stage=='sscd_catalog':
        from corpus_pipeline.postprocess.sscd_parallel_catalog import worker
        return worker(root/'sscd_index',workers=cfg['compute']['io_workers'],max_tasks=max_tasks)
    raise ValueError('Not a distributable worker stage: '+stage)


def node_worker(config,stage,*,queue_backend='rpc'):
    """Delegate per-card processes to the preemption-safe allocation supervisor."""
    from .pool import run as supervise
    return supervise(config,stage,queue_backend=queue_backend)
