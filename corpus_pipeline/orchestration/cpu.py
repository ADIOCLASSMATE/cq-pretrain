"""Stage-aware lifecycle for explicitly registered CPU Notebooks.

Retirement is durable and bound to sealed source receipts. A waiting review
consumer keeps its CPU; completed ingestion never resurrects retired workers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time

from corpus_pipeline.coordination import locked
from .config import SOURCES, check_config, read, save, sha
from .normalization import native_ready


def completion_proofs(run):
    run=Path(run);proofs=[]
    for source in SOURCES:
        root=run/'normalized_sources'/source
        if not native_ready(root):return None
        path=root/'COMPLETE.json'
        proofs.append({'path':str(path),'sha256':sha(path)})
    return proofs


def release_complete(run):
    run=Path(run);value=read(run/'COMPLETE.json')
    return bool(value and value.get('complete') is True and (run/'joint_release/COMPLETE.json').exists()
                and value.get('joint_release_complete_sha256')==sha(run/'joint_release/COMPLETE.json'))


def validate_workers(cfg):
    workers=cfg.get('cpu_workers',[])
    names=[row['name'] for row in workers]
    if len(names)!=len(set(names)) or cfg['review_notebook'] not in names:
        raise ValueError('CPU registry needs unique names and the retained review consumer')
    for row in workers:
        if not row.get('notebook_id') or not row.get('workspace_id'):
            raise ValueError('CPU registry requires exact accepted resource and workspace IDs')
        if not set(row.get('roles',[]))<= {'ingest','review'} or not row.get('roles'):
            raise ValueError('Explicit supported CPU stage roles required')
        if ('review' in row['roles']) != (row['name']==cfg['review_notebook']):
            raise ValueError('Review consumer identity differs from CPU role assignment')
    return workers


def policy(cfg):
    workers=validate_workers(cfg)
    proofs=completion_proofs(cfg['run'])
    complete=release_complete(cfg['run'])
    stop=(Path(cfg['run'])/'STOP').exists()
    retired={row['name'] for row in workers if complete or stop or (proofs is not None and 'review' not in row['roles'])}
    path=Path(cfg['run'])/'ops/CPU_RETIREMENTS.json';old=read(path)
    value={'schema':'corpus.cpu_retirement.v1','names':sorted(retired),'normalized_complete':proofs,
           'review_notebook':cfg['review_notebook'],'release_complete':complete,'stop_requested':stop}
    if old:
        if not set(old['names'])<=retired:raise ValueError('Retirement cannot be reversed by missing completion evidence')
        for item in old.get('normalized_complete') or []:
            if sha(item['path'])!=item['sha256']:raise ValueError('Retirement completion evidence changed')
    save(path,value)
    return workers,retired,proofs


def step(cfg, client):
    from inspire.sdk.models_notebooks import NotebookRef
    from corpus_pipeline.cluster.notebook_exec import execute
    workers,retired,proofs=policy(cfg);states={};run=Path(cfg['run'])
    for row in workers:
        ref=client.notebooks._make_ref(NotebookRef,row['name'],row['notebook_id'],row['workspace_id'])
        detail=client.notebooks.get(ref,workspace=cfg['workspace'])
        view=detail.to_dict()
        if (detail.ref.key!=row['notebook_id'] or detail.name!=row['name']
                or view['project']!=cfg['platform']['project'] or view['workspace']!=cfg['workspace']
                or int(detail.resource['gpu_count'])!=0 or int(detail.priority)!=cfg['cpu_priority']):
            raise ValueError('CPU Notebook ownership or resource binding differs')
        states[row['name']]=detail.status
        if row['name'] in retired:
            if detail.status not in ('STOPPED','FAILED','DELETED'):
                # Always re-observe next cycle; accepted Stop alone is not release.
                client.notebooks.stop(ref,workspace=cfg['workspace'])
            continue
        if (run/'STOP').exists():continue
        if detail.status=='STOPPED':
            client.notebooks.start(ref,workspace=cfg['workspace'])
            continue
        if detail.status!='RUNNING' or 'ingest' not in row['roles'] or proofs is not None:continue
        heartbeat=run/'ingest_workers'/(row['name']+'.supervisor.json')
        old=read(heartbeat,{})
        if old.get('complete') is True:continue
        if time.time()-old.get('time',0)<240:continue
        # The durable runner adopts an existing launch, including a lost launch
        # response. Never spawn an unowned nohup process on heartbeat timeout.
        runner=[str(run/'ops/cpu_python.sh'),'-m','corpus_pipeline.resource_manager.command_runner',
                '--root',str(run/'cpu_processes'/row['name']),'--resource-key',row['notebook_id']]
        attempt_path=run/'cpu_processes'/row['name']/'ADMISSION.json'
        admission=read(attempt_path,{'attempt':1,'not_before':0})
        if admission['attempt']>3 or time.time()<admission['not_before']:continue
        operation='ingest-'+hashlib.sha256((cfg['config_path']+row['notebook_id']).encode()).hexdigest()[:24]+'-'+str(admission['attempt'])
        save(attempt_path,{**admission,'operation_id':operation})
        command=[str(run/'ops/cpu_python.sh'),'-m','corpus_pipeline.orchestration.ingestion',
                 '--config',cfg['config_path'],'--name',row['name'],'--workers',str(row.get('workers',12))]
        result=execute(shlex.join([*runner,'launch','--operation-id',operation,'--',*command]),
            notebook=row['name'],account=cfg['account'],workspace=cfg['workspace'],
            notebook_id=row['notebook_id'],workspace_id=row['workspace_id'],timeout=60,max_output_bytes=8192)
        save(run/'ops/cpu_observations'/(row['name']+'.json'),result)
        if result.get('completed') and result.get('returncode')==0:
            replies=[]
            for line in result.get('output','').splitlines():
                try:value=json.loads(line)
                except ValueError:continue
                if isinstance(value,dict) and value.get('launch_operation_id')==operation:replies.append(value)
            if len(replies)==1 and replies[0].get('completed') is True and not replies[0].get('running'):
                save(attempt_path,{'attempt':admission['attempt']+1,'not_before':time.time()+60,
                    'previous_operation':operation,'previous_exit_code':replies[0].get('exit_code')})
    result={'time':time.time(),'states':states,'retired':sorted(retired),
            'verified_released':all(states.get(name) in ('STOPPED','FAILED','DELETED') for name in retired)}
    save(run/'ops/CPU_POOL_STATUS.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--once',action='store_true');args=parser.parse_args();cfg=check_config(args.config)
    from corpus_pipeline.serving.platform_client import InspireClient
    with locked(Path(cfg['run'])/'ops/cpu_controller.lock',nonblocking=True), InspireClient(account=cfg['account']) as client:
        while True:
            try:
                result=step(cfg,client)
                if args.once or ((release_complete(cfg['run']) or (Path(cfg['run'])/'STOP').exists())
                                and result['verified_released']):return
            except Exception as error:
                save(Path(cfg['run'])/'ops/CPU_CONTROLLER_ERROR.json',{'error_type':type(error).__name__,'time':time.time()})
                if args.once:raise
            time.sleep(30)


if __name__=='__main__':main()
