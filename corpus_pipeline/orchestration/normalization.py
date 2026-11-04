"""Adopt sealed native sources, preserving every artifact and byte binding."""
import os
from pathlib import Path
from .config import read,sha,immutable

def native_ready(source):
    complete=read(Path(source)/'COMPLETE.json')
    if complete is None:return False
    if complete.get('complete') is not True or complete.get('validation_passed') is not True:return False
    for name in ('catalog','manifest'):
        if sha(Path(source)/(name+'.json'))!=complete.get(name+'_sha256'):raise ValueError('Changed normalized '+name)
    manifest=read(Path(source)/'manifest.json')
    if manifest.get('schema_version')!='corpus.native.v2' or manifest.get('phase')!='complete':raise ValueError('Not a sealed native v2 source')
    if complete.get('scope') not in ('full','full_declared_inputs'):raise ValueError('Bounded normalized source cannot enter full production')
    return True


def ingest_integrity_verified(cfg):
    """Exact registered plans and the immutable distributed-claim implementation."""
    from .config import SOURCES
    from corpus_pipeline.unified.parallel_ingest import registered
    path=Path(cfg['run'])/'ops/INGEST_CONTRACT.json'
    proof=read(path)
    if proof is None:return False
    rows=registered(Path(cfg['run']))
    if (proof.get('schema')!='corpus.ingest_contract.v1' or proof.get('claim_protocol')!='atomic_mkdir'
            or proof.get('sources')!=list(SOURCES)
            or proof.get('registrations')!={row['source']:row['plan_sha256'] for row,_ in rows}
            or set(proof['registrations'])!=set(SOURCES)
            or proof.get('implementation_sha256')!=sha(Path(cfg['runtime'])/'corpus_pipeline/unified/parallel_ingest.py')):
        raise ValueError('Ingestion plan or implementation binding changed')
    return True


def adopt_native(config):
    """Reuse sealed metadata with hard links; never copy image payloads."""
    import pyarrow.parquet as pq
    from corpus_pipeline.unified import workflow,release
    from corpus_pipeline.production.records import snapshot
    from corpus_pipeline.production.task_policy import receipt_policy
    cfg,root=workflow.initialize(config)
    old=workflow.validated_receipt(root,'normalize')
    if old:return old
    source=Path(cfg['inputs'][0]['root']).resolve()
    if not native_ready(source):raise ValueError('Source normalization is unfinished')
    snap=snapshot(source);native=root/'native';native.mkdir(parents=True,exist_ok=True)
    count=slots=0;artifacts=[]
    for original,entry in snap['catalog_index'].items():
        if entry['collection']!='genund':raise ValueError('Source cannot relabel another collection')
        original=Path(original);relative=Path(entry['path']);target=native/relative
        if sha(original)!=entry['sha256']:raise ValueError('Native Parquet SHA mismatch: '+str(original))
        metadata=pq.ParquetFile(original).metadata
        if metadata.num_rows!=entry['rows']:raise ValueError('Native row count mismatch')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(original,target)
        if not os.path.samefile(original,target):raise ValueError('Adopted file is not the sealed native inode')
        count+=entry['rows'];slots+=entry.get('image_slots',0);artifacts.append(target)
    complete=read(source/'COMPLETE.json')
    if count!=complete['statistics']['samples']:raise ValueError('Incomplete native source coverage')
    source_manifest=read(source/'manifest.json')
    extra=[]
    if complete.get('plan_sha256'):
        if sha(source/'PLAN.json')!=complete['plan_sha256'] or source_manifest.get('plan_sha256')!=complete['plan_sha256']:
            raise ValueError('Native source plan changed')
        extra.append(('PLAN.json',complete['plan_sha256']))
    for item in source_manifest.get('unit_receipts',[]):
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe native unit receipt path')
        extra.append((str(relative),item['sha256']))
    for relative,expected_sha in extra:
        original=source/relative;target=native/relative
        if sha(original)!=expected_sha:raise ValueError('Native unit/plan receipt changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(original,target)
        if not os.path.samefile(original,target):raise ValueError('Adopted native receipt is not its bound inode')
        artifacts.append(target)
    for name in ('manifest.json','catalog.json','COMPLETE.json'):
        target=native/name
        if target.exists():
            if sha(target)!=sha(source/name):raise ValueError('Adopted native metadata changed')
        else:
            temporary=target.with_name('.'+target.name+'.adopt')
            temporary.write_bytes((source/name).read_bytes());os.replace(temporary,target)
        artifacts.append(target)
    adoption={'schema':'corpus.native_hardlink_adoption.v1','complete':True,'source':str(source),
        'source_complete_sha256':sha(source/'COMPLETE.json'),'files':len(snap['catalog_index']),
        'records':count,'image_payloads_copied':False,'source_files_modified':False}
    immutable(root/'NATIVE_ADOPTION.json',adoption);artifacts.append(root/'NATIVE_ADOPTION.json')
    return release.seal_receipt(root/'receipts/normalize.json',stage='normalize',collections=workflow.collection_bindings(root),
        coverage={'unit':'records','expected':count,'processed':count,'missing':0,'pending_infra':0},
        inputs=[root/'EXECUTION.json',source/'COMPLETE.json'],artifacts=artifacts,dependencies={},
        statistics=complete['statistics'],semantics={'scope':'full','required_collections':['genund'],
        'task_direction_preserved':True,'ordered_supervision_preserved':True,'sscd_candidates_automatically_deleted':False},
        policy_config=receipt_policy(cfg))
