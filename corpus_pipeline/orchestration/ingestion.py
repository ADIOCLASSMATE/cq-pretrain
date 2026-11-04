"""Registered source plans and bounded CPU workers; completion is terminal."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from corpus_pipeline.coordination import locked
from corpus_pipeline.unified.parallel_ingest import registered
from .config import SOURCES, check_config, environment, immutable, read, save, sha
from .normalization import native_ready


def register(cfg):
    """Register exact adapter plans; incomplete copy/recovery remains a blocker.

    Annotation recovery/byte binding is performed by source adapters. Neither
    filename resemblance nor a guessed caption/image ordering is admissible.
    """
    plans = cfg.get('source_plans', {})
    if set(plans) != set(SOURCES):
        raise ValueError('Every declared execution source needs an explicit source plan')
    run = Path(cfg['run'])
    for name, value in plans.items():
        path = Path(value).resolve(strict=True)
        plan = read(path)
        sealed_loose=plan.get('identity',{}).get('contract')=='genund.loose_original_supervision.v1'
        if (plan.get('complete') is not True and not sealed_loose) or not plan.get('units'):
            raise ValueError('Source plan is incomplete: '+name)
        for unit in plan['units']:
            if unit.get('sha256') and sha(unit['path'])!=unit['sha256']:
                raise ValueError('Source plan work unit changed')
        registration = {'complete':True, 'source':name, 'plan':str(path), 'plan_sha256':sha(path),
                        'image_payloads_persisted':bool(plan.get('image_payloads_persisted',False))}
        immutable(run/'ingest_registry'/(name+'.json'), registration)
    entries = registered(run)
    if {row['source'] for row,_ in entries} != set(SOURCES):
        raise ValueError('Registered source coverage differs')
    immutable(run/'ops/INGEST_CONTRACT.json', {'schema':'corpus.ingest_contract.v1',
        'sources':list(SOURCES), 'claim_protocol':'atomic_mkdir',
        'implementation_sha256':sha(Path(cfg['runtime'])/'corpus_pipeline/unified/parallel_ingest.py'),
        'registrations':{row['source']:row['plan_sha256'] for row,_ in entries}})
    return {'registered_sources':len(entries)}


def all_complete(run):
    return all(native_ready(Path(run)/'normalized_sources'/name) for name in SOURCES)


def supervise(cfg, name, workers):
    if not 1<=workers<=min(32,os.cpu_count() or 1):
        raise ValueError('CPU worker count exceeds local capacity')
    run=Path(cfg['run']); area=run/'ingest_workers';area.mkdir(parents=True,exist_ok=True)
    assignment=read(run/'ops/INGEST_WORKERS.json',{}).get('workers',[])
    names=[name+'-'+str(i) for i in range(workers)]
    if not set(names)<=set(assignment):raise ValueError('Unregistered CPU worker partition')
    stop=False; children={}; exits={}; log_handles=[]
    def stopping(*_):
        nonlocal stop
        stop=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stopping)
    with locked(area/(name+'.supervisor.lock'),nonblocking=True):
        try:
            while not stop and not (run/'ops/STOP_INGEST').exists():
                if all_complete(run):
                    save(area/(name+'.supervisor.json'),{'pid':os.getpid(),'complete':True,'status':'complete','time':time.time()})
                    return 0
                for worker in names:
                    child=children.get(worker)
                    if child is not None and child.poll() is None:continue
                    if child is not None:
                        exits[worker]=exits.get(worker,0)+1
                        if exits[worker]>=3:raise RuntimeError('Ingestion worker repeatedly exited before sealed completion')
                    log=(area/(worker+'.log')).open('ab');log_handles.append(log)
                    children[worker]=subprocess.Popen([cfg['python'],'-m','corpus_pipeline.unified.parallel_ingest',
                        '--run',str(run),'--worker',worker],cwd=cfg['runtime'],env=environment(cfg),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                save(area/(name+'.supervisor.json'),{'pid':os.getpid(),'status':'running','time':time.time(),
                    'workers':{key:child.pid for key,child in children.items()}})
                time.sleep(10)
        finally:
            for child in children.values():
                if child.poll() is None:child.terminate()
            for child in children.values():
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            for log in log_handles:log.close()
    return 75


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    return supervise(check_config(args.config),args.name,args.workers)


if __name__=='__main__':raise SystemExit(main())
