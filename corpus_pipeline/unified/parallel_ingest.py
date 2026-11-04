"""Recoverable shared-filesystem ingestion of explicitly registered source plans.

Workers claim one immutable unit with atomic mkdir. Complete source snapshots become
available independently; a missing source plan can never be mistaken for success.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback
import shutil
import uuid
import importlib

from corpus_pipeline.production.common import atomic_json, file_sha, digest
from .ingest import _process_unit

SOURCES = ('OpenUni', 'ScaleCap450k_image', 'DenseFusion-1M',
           'pixmo-ask-model-anything', 'FineVision', 'BLIP3o-Pretrain-Long-Caption',
           'DenseFusion-4V-100k')
RULES = {'task_routing': 'corpus.task_policy.v1', 't2i_uncertain': 'exclude'}
ADAPTERS = {'genund_loose', 'genund_openuni', 'genund_scalecap', 'genund_finevision'}
_ADAPTER_HASHES = {}


def current_adapter(unit):
    """A run cannot silently mix adapter versions between committed units."""
    name=('corpus_pipeline.unified.genund_loose' if unit['adapter']=='genund_loose'
          else 'corpus_pipeline.production.genund_collections')
    module=importlib.import_module(name)
    checksum=file_sha(module.__file__)
    if name in _ADAPTER_HASHES and _ADAPTER_HASHES[name]!=checksum:
        raise ValueError('Ingestion adapter changed; start a reviewed new runtime snapshot')
    _ADAPTER_HASHES[name]=checksum
    return {'module':name,'sha256':checksum,'time':time.time()}


@contextmanager
def directory_claim(path, owner):
    """mkdir is atomic across mounts; flock can be local on this platform.

    An interrupted process can recover its own claim on the same host. Claims
    on another host require an externally verified stopped-instance recovery.
    No elapsed-time-only takeover can overlap a slow but still live writer.
    """
    path=Path(path); token=uuid.uuid4().hex
    try:
        path.mkdir()
    except FileExistsError:
        try:
            old=json.loads((path/'owner.json').read_text())
            if old['owner']==owner and old['host']==socket.gethostname():
                alive=True
                try: os.kill(old['pid'],0)
                except ProcessLookupError: alive=False
                if not alive:
                    stale=path.with_name(path.name+'.retired.'+token)
                    path.rename(stale); shutil.rmtree(stale)
                    path.mkdir()
                else:
                    yield False; return
            else:
                yield False; return
        except (FileNotFoundError,FileExistsError,KeyError,json.JSONDecodeError):
            yield False; return
    atomic_json(path/'owner.json',{'owner':owner,'host':socket.gethostname(),'pid':os.getpid(),
                                   'token':token,'created_at':time.time()})
    try: yield True
    finally:
        if json.loads((path/'owner.json').read_text()).get('token')!=token:
            raise RuntimeError('distributed claim ownership changed')
        (path/'owner.json').unlink(); path.rmdir()


def registered(run):
    """Read only atomic, explicitly sealed plans, never infer subset coverage."""
    result = []
    for registration in sorted((run/'ingest_registry').glob('*.json')):
        value = json.loads(registration.read_text())
        if value.get('complete') is not True or value.get('source') not in SOURCES:
            raise ValueError('invalid sealed source registration: '+str(registration))
        plan_path = Path(value['plan'])
        if file_sha(plan_path) != value['plan_sha256']:
            raise ValueError('registered plan changed: '+str(plan_path))
        plan = json.loads(plan_path.read_text())
        units = plan['units']
        if not units or len({u['unit_id'] for u in units}) != len(units):
            raise ValueError('empty or duplicate plan units')
        if any(u['adapter'] not in ADAPTERS or u.get('collection','genund') != 'genund' for u in units):
            raise ValueError('unsupported parallel ingestion adapter')
        result.append((value, units))
    if len({r['source'] for r,_ in result}) != len(result):
        raise ValueError('duplicate registered source')
    return result


def finalize_source(run, registration, units, owner='finalizer'):
    out = run/'normalized_sources'/registration['source']
    out.mkdir(parents=True, exist_ok=True)
    with directory_claim(out/'.finalize.claim',owner) as acquired:
        if not acquired: return None
        if (out/'COMPLETE.json').exists(): return json.loads((out/'COMPLETE.json').read_text())
        paths = [out/'units'/u['unit_id']/'COMPLETE.json' for u in units]
        if not all(p.is_file() for p in paths): return None
        catalog=[]; receipts=[]; rows=slots=0
        formats=Counter(); sources=Counter(); statuses=Counter()
        for unit,path in zip(units,paths):
            receipt=json.loads(path.read_text())
            if receipt.get('unit') != unit or receipt.get('complete') is not True or receipt.get('source_exhausted') is not True:
                raise ValueError('incomplete or mismatched source receipt')
            for item in receipt['catalog']:
                part=out/item['path']
                if part.stat().st_size != item['bytes'] or file_sha(part) != item['sha256']:
                    raise ValueError('committed normalized shard changed')
            rows+=receipt['records']; slots+=receipt['image_slots']; catalog.extend(receipt['catalog'])
            formats.update(receipt['formats']); sources.update(receipt['sources']); statuses.update(receipt['adapter_statuses'])
            receipts.append({'path':str(path.relative_to(out)),'sha256':file_sha(path)})
        if not rows: raise ValueError('empty normalized source')
        plan={'schema_version':'corpus.parallel_ingest.v1','registration':registration,
              'units':units,'rules':RULES,'source_files_modified':False}
        atomic_json(out/'PLAN.json',plan)
        stats={'samples':rows,'image_slots':slots,'formats':dict(formats),'sources':dict(sources),
               'adapter_statuses':dict(statuses),'input_units_planned':len(units),'input_units_processed':len(units)}
        manifest={'schema_version':'corpus.native.v2','phase':'complete','collections':['genund'],
                  'allowed_image_roots':sorted({u['root'] for u in units}),
                  'portable_datasets':sorted({v for u in units for v in [u['dataset_id'],*u.get('portable_datasets',[])]}),
                  'scope':'full_declared_inputs','declared_subset':registration['source'],
                  'source_files_modified':False,'image_payloads_copied':bool(registration.get('image_payloads_persisted',False)),
                  'recovered_asset_roots':sorted({u['recovered_root'] for u in units if u.get('recovered_root')}),
                  'enumeration_scope':'registered_mounted_subset','plan_sha256':file_sha(out/'PLAN.json'),
                  'unit_receipts':receipts,'statistics':stats}
        atomic_json(out/'catalog.json',catalog); atomic_json(out/'manifest.json',manifest)
        result={'schema_version':'corpus.native.v2','complete':True,'validation_passed':True,
                'validation_scope':'original_structure_and_byte_bindings_not_image_decode_or_quality',
                'catalog_sha256':file_sha(out/'catalog.json'),'manifest_sha256':file_sha(out/'manifest.json'),
                'plan_sha256':file_sha(out/'PLAN.json'),'scope':'full_declared_inputs','statistics':stats}
        atomic_json(out/'COMPLETE.json',result)
        return result


def worker(run, name, *, once=False, retry_limit=3):
    run=Path(run).resolve(); state=run/'ingest_workers'; state.mkdir(exist_ok=True)
    heartbeat=state/(name+'.json'); rng=random.Random(name); completed=0; started=time.time()
    assignment=run/'ops/INGEST_WORKERS.json'
    workers=json.loads(assignment.read_text())['workers'] if assignment.exists() else [name]
    if name not in workers or len(workers)!=len(set(workers)):
        raise ValueError('worker is not in the immutable ingestion partition')
    position=workers.index(name)
    while not (run/'ops'/'STOP_INGEST').exists():
        plans=registered(run); candidates=[]
        for reg,units in plans:
            out=run/'normalized_sources'/reg['source']
            if (out/'COMPLETE.json').exists(): continue
            for unit in units:
                if int(unit['unit_id'][:16],16)%len(workers)!=position: continue
                if not (out/'units'/unit['unit_id']/'COMPLETE.json').exists(): candidates.append((reg,unit))
        rng.shuffle(candidates); did_work=False
        for reg,unit in candidates:
            out=run/'normalized_sources'/reg['source']; locks=out/'claims'; locks.mkdir(parents=True,exist_ok=True)
            with directory_claim(locks/(unit['unit_id']+'.claim'),name) as acquired:
                if not acquired: continue
                if (out/'units'/unit['unit_id']/'COMPLETE.json').exists(): continue
                failure=locks/(unit['unit_id']+'.error.json')
                previous=json.loads(failure.read_text()) if failure.exists() else {'attempts':0}
                if previous['attempts']>=retry_limit: continue
                status={'worker':name,'host':socket.gethostname(),'pid':os.getpid(),'time':time.time(),
                        'source':reg['source'],'unit_id':unit['unit_id'],'status':'running','completed_units':completed}
                atomic_json(heartbeat,status); begin=time.time()
                try:
                    version=current_adapter(unit)
                    atomic_json(locks/(unit['unit_id']+'.adapter.json'),version)
                    receipt=_process_unit(out,unit,RULES,None,8192)
                    completed+=1
                    atomic_json(locks/(unit['unit_id']+'.timing.json'),
                                {'seconds':time.time()-begin,'records':receipt['records'],'worker':name})
                    if failure.exists(): failure.unlink()
                except Exception as exc:
                    atomic_json(failure,{'attempts':previous['attempts']+1,'time':time.time(),
                                         'error':str(exc),'traceback':traceback.format_exc(),'worker':name})
                atomic_json(heartbeat,{**status,'time':time.time(),'status':'idle','completed_units':completed})
                did_work=True
            break
        if not did_work:
            for reg,units in plans: finalize_source(run,reg,units,name)
            done=[s for s in SOURCES if (run/'normalized_sources'/s/'COMPLETE.json').exists()]
            complete=len(done)==len(SOURCES)
            atomic_json(heartbeat,{'worker':name,'pid':os.getpid(),'host':socket.gethostname(),
                                  'time':time.time(),'status':'complete' if complete else 'waiting',
                                  'complete':complete,'completed_units':completed,'complete_sources':done})
            if len(done)==len(SOURCES) or once: return
            time.sleep(15+rng.random()*10)
        elif once: return


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True); parser.add_argument('--worker',required=True)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(); worker(args.run,args.worker,once=args.once)


if __name__=='__main__': main()
