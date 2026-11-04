"""One resumable DAG for the joint corpus; proven production backends underneath.

Only the coordinator mutates stage state. Accelerator workers commit leased
parts, never release training data. A stopped or incomplete stage cannot publish.
"""
from __future__ import annotations
from collections import Counter,OrderedDict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from .config import load, require, STAGES, DEPENDENCIES
from . import release

BASE=Path(__file__).resolve().parents[1]
_VERIFIED=OrderedDict()
_VERIFIED_LOCK=threading.Lock()


def verify_artifact(item):
    """Avoid rehashing immutable terabyte stages every 30-second watch cycle.

    Each process verifies bytes at first use and after any filesystem identity
    change. The publisher independently performs the final full hash audit.
    """
    p=Path(item['path']);st=p.stat()
    key=(item['sha256'],item['size_bytes'],st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    with _VERIFIED_LOCK:
        cached=_VERIFIED.get(str(p))==key
    if not cached:
        require(st.st_size==item['size_bytes'] and file_sha(p)==item['sha256'],'Changed stage artifact: '+str(p))
        after=p.stat()
        require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                'Stage artifact changed during verification')
    with _VERIFIED_LOCK:
        _VERIFIED[str(p)]=key
        _VERIFIED.move_to_end(str(p))
        while len(_VERIFIED)>200000:_VERIFIED.popitem(last=False)


def verify_artifacts(items, workers=8):
    """Bounded parallel hashing; no persistent trust cache or skipped bytes."""
    require(type(workers) is int and 1<=workers<=16,'Artifact verification workers must be in [1,16]')
    unique={}
    for item in items:
        previous=unique.setdefault(item['path'],item)
        require((previous['sha256'],previous['size_bytes'])==(item['sha256'],item['size_bytes']),
                'Conflicting artifact bindings')
    values=list(unique.values())
    if len(values)<2:
        for item in values:verify_artifact(item)
        return
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='verify-artifact') as pool:
        for start in range(0,len(values),256):
            for _ in pool.map(verify_artifact,values[start:start+256]):pass


class Pending(RuntimeError):
    """Work has a durable checkpoint and awaits workers or external evidence."""


def read(path):return json.loads(Path(path).read_text())


def implementation():
    files={}
    for directory in ('unified','production','cluster','postprocess','dedup','orchestration','resource_manager'):
        for path in sorted((BASE/directory).glob('*.py')):
            if not path.name.startswith('test_'):files[str(path.relative_to(BASE))]=file_sha(path)
    for name in ('policy.py','config/rules.json','config/sources.json','detector_calibration/thresholds.json',
                 'prepared/config/layout_source_contracts.json','production/image_screening_prompt_v4.txt'):
        files[name]=file_sha(BASE/name)
    return files


def preflight(config):
    cfg=load(config);missing=[];unsupported=[]
    for item in cfg['inputs']:
        if not Path(item['root']).is_dir():missing.append(item['root'])
        elif item['adapter']=='dcvlm_baseline_wds':
            control=Path(item['download_control']);manifest_path=control/'manifest.json'
            if not manifest_path.is_file():missing.append(str(manifest_path))
            else:
                manifest=read(manifest_path)
                if (manifest.get('repo')!='mlfoundations/dcvlm-baseline-200b' or
                        Path(manifest.get('destination','')).resolve()!=Path(item['root']).resolve()):
                    unsupported.append({'collection':'dcvlm','reason':'baseline_download_manifest_root_or_repo_mismatch'})
        elif item['adapter']=='genund_caption_tar':
            from corpus_pipeline.production.genund_ingest import SUPPORTED_SOURCES
            names=[item['source']] if item.get('source') else [p.name for p in Path(item['root']).iterdir() if p.is_dir()]
            unsupported.extend({'collection':'genund','source':name,'reason':'explicit_source_adapter_required'}
                               for name in names if name not in SUPPORTED_SOURCES)
    for path in cfg['cache']['histories']+cfg['screening']['history_decision_manifests']:
        if not Path(path).exists():missing.append(path)
    return {'ready':not missing and not unsupported,'scope':cfg['scope'],'missing':missing,'unsupported_inputs':unsupported,
            'collections':sorted({x['collection'] for x in cfg['inputs']}),
            'required_collections':cfg['required_collections'],
            'stages':list(STAGES),'output':cfg['output'],'resources':cfg['resources'],
            'policy':cfg['policy'],'config_sha256':file_sha(config),
            'formal_run_started':False}


def initialize(config):
    cfg=load(config);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    identity={'schema':'corpus.unified.execution.v1','config':cfg,'code':implementation()}
    path=root/'EXECUTION.json'
    if path.exists():require(read(path)==identity,'Run configuration/code changed; use a new output snapshot')
    else:
        require(not (root/'receipts').exists(),'Unbound existing pipeline output')
        atomic_json(path,identity)
    atomic_json(root/'CONFIG.json',cfg)
    return cfg,root


def collection_bindings(root):
    entries=read(root/'native/catalog.json');result={}
    for name in sorted({e['collection'] for e in entries}):
        result[name]=digest({'snapshot':file_sha(root/'native/COMPLETE.json'),
                            'collection':name,'files':[e for e in entries if e['collection']==name]})
    return result


def verify_execution(config):
    cfg=load(config);root=Path(cfg['output']);value=read(root/'EXECUTION.json')
    require(value['config']==cfg and value['code']==implementation(),'Worker code/config differ from frozen execution')
    return cfg


def validated_receipt(root,stage):
    path=root/'receipts'/f'{stage}.json'
    if not path.exists():return None
    value=read(path)
    require(value.get('complete') is True and value['stage']==stage,'Invalid stage receipt')
    verify_artifacts(value['inputs']+value['artifacts'])
    require(set(value['dependencies'])==set(DEPENDENCIES[stage]),'Missing required dependency edge')
    for name,sha in value['dependencies'].items():
        require(file_sha(root/'receipts'/f'{name}.json')==sha,'Changed parent stage receipt')
    return value


def task_ids(root,cfg):
    if cfg['scope']=='full':return None
    return [x['task']['task_id'] for x in read(root/'frontpass/frontpass_index.json')['parts']]


def finalized_files(directory):
    """Bind final outputs, not mutable progress, locks, spill or abandoned attempts.

    Backend COMPLETE/manifest receipts bind their larger payloads. Those payloads
    are also bound here whenever listed as artifacts, files, or block receipts.
    """
    directory=Path(directory);found={};visited=set()
    def add(path,follow=True):
        path=path.resolve()
        if not path.is_file() or path in visited:return
        visited.add(path);found[str(path)]=path
        if path.suffix!='.json' or not follow:return
        value=read(path)
        def walk(obj,required=False):
            if isinstance(obj,list):
                for item in obj:walk(item,required)
            elif isinstance(obj,dict):
                if 'path' in obj and 'sha256' in obj:
                    candidate=Path(obj['path'])
                    if not candidate.is_absolute():
                        possibilities=[directory/candidate,path.parent/candidate]
                        candidate=next((p for p in possibilities if p.is_file()),possibilities[0])
                    if candidate.is_relative_to(directory.resolve()) and (required or candidate.is_file()):
                        require(candidate.is_file() and file_sha(candidate)==obj['sha256'],'Backend payload changed: '+str(candidate))
                        add(candidate)
                if 'receipt_path' in obj:
                    candidate=directory/obj['receipt_path']
                    require(file_sha(candidate)==obj['receipt_sha256'],'Backend block receipt changed');add(candidate)
                for key,child in obj.items():
                    if isinstance(child,(dict,list)):walk(child,key in {'artifacts','files','native_metadata_bindings'})
        walk(value)
    for name in ('COMPLETE.json','manifest.json','catalog.json','catalog/COMPLETE.json','frontpass_index.json','FRONTPASS_COMPLETE.json'):
        add(directory/name)
    marker=directory/'COMPLETE.json'
    if marker.exists():
        complete=read(marker)
        if complete.get('schema')=='corpus.sscd_exact_npu.v1':
            from corpus_pipeline.postprocess.sscd_exact_npu import pair_tasks,task_directory
            plan_path=directory/'PLAN.json';plan=read(plan_path)
            require(file_sha(plan_path)==complete['plan_sha256'],'Search plan changed')
            add(plan_path,follow=False);calibration=directory/'calibration/COMPLETE.json'
            require(file_sha(calibration)==complete['calibration_sha256'],'Search calibration changed');add(calibration)
            hashes=hashlib.sha256();pairs=0;tasks=0
            for task in pair_tasks(plan['blocks']):
                receipt=task_directory(directory,task)/'COMPLETE.json'
                hashes.update(file_sha(receipt).encode());add(receipt);pairs+=task['pairs'];tasks+=1
            require(hashes.hexdigest()==complete['task_receipts_sha256'] and pairs==complete['completed_pairs'] and
                    tasks==complete['completed_tasks'],'Search task coverage changed')
        elif complete.get('schema')=='corpus.sscd_graph.v1':
            plan_path=directory/'PLAN.json'
            require(file_sha(plan_path)==complete['plan_sha256'],'Graph plan changed');add(plan_path,follow=False)
            add(directory/'packed/COMPLETE.json')
            for phase,sha in complete['phases'].items():
                require(phase in {'adjacency','components','representatives'},'Unknown graph phase')
                path=directory/phase/'COMPLETE.json';require(file_sha(path)==sha,'Graph phase changed');add(path)
    require(bool(found),'No finalized backend artifacts: '+str(directory))
    return sorted(found.values())


def record_statistics(root,complete):
    import duckdb
    import pyarrow as pa
    from corpus_pipeline.postprocess.reconcile import quoted
    catalog=read(root/'native/catalog.json');manifest=read(root/'training_candidate_index/manifest.json')
    lookup=pa.Table.from_pylist([{'original_parquet_path':str(root/'native'/x['path']),
                                'collection':x['collection']} for x in catalog])
    files=[str(root/'training_candidate_index'/a['path']) for b in manifest['buckets']
           for a in b['artifacts'] if a['path'].endswith('.parquet')]
    db=duckdb.connect();db.execute('SET threads=2');db.execute("SET memory_limit='8GB'")
    try:
        db.register('sources',lookup)
        paths='['+','.join(quoted(p) for p in files)+']'
        rows=db.execute('SELECT collection,decision,count(*) FROM read_parquet('+paths+
                        ',hive_partitioning=false) LEFT JOIN sources USING(original_parquet_path) GROUP BY 1,2').fetchall()
        result={name:{'records':0,'counts':dict.fromkeys(('keep','exclude','quarantine'),0)} for name in collection_bindings(root)}
        for name,decision,count in rows:
            require(name in result,'Record lost its original collection binding')
            result[name]['records']+=count;result[name]['counts'][decision]+=count
        require(sum(x['records'] for x in result.values())==complete['records'],'Collection row coverage mismatch')
        return {'records':complete['records'],'counts':{k:complete['counts'].get(k,0) for k in ('keep','exclude','quarantine')},
                'per_collection':result,'review_pending':read(root/'screening/COMPLETE.json')['review_uncertain']}
    finally:db.close()


def execute_stage(cfg,root,stage,*,local=False,delegate=None):
    """Return actual coverage, statistics, directories whose output is sealed."""
    compute=cfg['compute'];bounded=cfg['scope']=='validation'
    from corpus_pipeline.production.task_policy import run_rules, receipt_policy, PROFILE
    from corpus_pipeline.cluster.frontpass import load_rules
    routing_rules=run_rules(cfg,load_rules()) if cfg['policy'].get('profile')==PROFILE else None
    unit='records';count=None;statistics={};outputs=[];reader=None
    if stage=='normalize':
        baseline=[item for item in cfg['inputs'] if item['adapter']=='dcvlm_baseline_wds']
        if baseline:
            from . import baseline_index
            source=baseline[0]
            index=Path(source.get('index_output',root/'native'))
            baseline_index.prepare(source['download_control'],index,
                                   rows_per_file=compute.get('index_rows_per_file',compute['rows_per_task']))
            progress=baseline_index.status(source['download_control'],index)
            if progress['phase'] not in ('ready_to_finalize','complete'):
                raise Pending('Portable baseline indexing: '+str(progress['units_complete'])+'/'+
                              str(progress['units_total'])+' TAR units; run the CPU index supervisor')
            compact_needed=index.resolve()!=(root/'native').resolve()
            if progress['phase']=='complete' and compact_needed:
                # status binds the published receipt; compaction independently
                # verifies every source Parquet before publishing its own output.
                result=read(index/'COMPLETE.json')
            else:
                result=baseline_index.finalize(source['download_control'],index)
            if compact_needed:
                from . import native_compact
                result=native_compact.run(index,root/'native',workers=compute['io_workers'])
        else:
            from .ingest import build
            result=build(cfg['inputs'],root/'native',rules=routing_rules,
                         max_records=cfg.get('max_records'),rows_per_file=compute['rows_per_task'])
        count=result['statistics']['samples'];outputs=[root/'native'];statistics=result['statistics']
    elif stage=='frontpass':
        from corpus_pipeline.cluster import frontpass,queue,monitor
        area=root/'frontpass'
        if not (area/'plan.json').exists():
            plan=frontpass.prepare(area,dataset=root/'native',rows_per_task=compute['rows_per_task'],rules=routing_rules)
            plan.update(scope='unified_full' if not bounded else 'unified_validation',old_result_reuse=cfg['cache']['enabled'],
                        execution_sha256=file_sha(root/'EXECUTION.json'))
            atomic_json(area/'plan.json',plan)
        require(read(area/'plan.json')['execution_sha256']==file_sha(root/'EXECUTION.json'),'Unbound frontpass execution')
        qualification=compute.get('optimized_qualification')
        if qualification:
            value=read(qualification);target=area/'OPTIMIZED_QUALIFIED.json'
            if target.exists():require(read(target)==value,'Optimized qualification changed')
            else:atomic_json(target,value)
            frontpass.optimized_configuration(area,dtype=compute['dtype'],batch_size=compute['batch_size'],threads=compute['threads'])
        progress=queue.status(area)
        if not progress['all_tasks_completed']:
            if not local:raise Pending('frontpass tasks prepared; start queue RPC and the capped frontpass fleet')
            require(bounded,'CPU execution is available only for bounded validation')
            from .worker import frontpass_worker
            frontpass_worker(root/'CONFIG.json',device='cpu',queue_backend='local',delegate=delegate)
            progress=queue.status(area)
        require(progress['all_tasks_completed'],'Frontpass still has uncommitted tasks')
        if not (area/'FRONTPASS_COMPLETE.json').exists():monitor.finalize(area,progress)
        result=read(area/'FRONTPASS_COMPLETE.json');count=result['counts']['records'];statistics=result['counts'];outputs=[area]
    elif stage=='reconcile':
        from corpus_pipeline.postprocess.reconcile import run
        result=run(root/'frontpass',root/'reconciliation',threads=compute['threads'],memory_limit=compute['memory_limit'],
                   selected_task_ids=task_ids(root,cfg))
        from .history import register_history
        register_history(root/'frontpass',root/'reconciliation',root/'image_history')
        unit='assets';count=result['counts']['assets'];statistics=result['counts'];outputs=[root/'reconciliation',root/'image_history']
    elif stage=='screening':
        from .screening import build_decisions
        options=cfg['screening'];rules=[]
        for rule in options['quality_decisions']:
            if isinstance(rule,str):
                value=read(rule);rules.extend(value if isinstance(value,list) else [value])
            else:rules.append(rule)
        def assemble():return build_decisions(root/'reconciliation',root/'screening',
            review_root=root/'api_review',quality_decisions=rules,
            policy_config=receipt_policy(cfg),
            history_decision_manifests=options['history_decision_manifests'])
        result=assemble()
        if not result['complete']:
            if options['backend']=='history_only':raise Pending('Missing or conflicting review evidence; see screening/PENDING.json')
            if result.get('conflicts'):raise Pending('Contradictory evidence requires review; see screening/conflicts.json')
            from corpus_pipeline.postprocess.api_screen import main
            selection=result['api_selection']
            require(selection['assets']>0,'Missing review evidence has no actionable API selection')
            args=['--selection-manifest',str(selection['manifest']['path']),
                  '--reconciliation-root',str(root/'reconciliation'),'--output',str(root/'api_review'),
                  '--backend',options['backend'],'--primary',options['primary'],'--mode','annotate',
                  '--concurrency',str(options['concurrency']),'--image-threads',str(min(compute['io_workers'],16)),
                  '--stop-file',str(root/'STOP'),'--prompt',options['prompt'],
                  '--timeout',str(options['timeout']),'--max-tokens',str(options['max_tokens'])]
            for item in selection['files']:args+=['--input',str(item['path'])]
            for key in ('credential_example','primary_pool','api_key_file'):
                if options.get(key):args+=['--'+key.replace('_','-'),str(options[key])]
            for path in options.get('primary_registries',[]):args+=['--primary-registry',str(path)]
            main(args)
            if (root/'STOP').exists():raise Pending('API processing stopped with durable partial results')
            result=assemble()
            if not result['complete']:raise Pending('Review has unfinished work; see screening/PENDING.json')
        unit='assets';count=result['records'];statistics={k:result[k] for k in ('records','review_candidates','review_uncertain','counts')};outputs=[root/'screening']
    elif stage=='sscd_catalog':
        from corpus_pipeline.postprocess import sscd_parallel_catalog as catalog,sscd_exact_cache as cache
        catalog.prepare(root/'reconciliation',root/'sscd_index',block_rows=cfg['sscd']['block_rows'],allow_bounded=bounded)
        catalog.worker(root/'sscd_index',workers=compute['io_workers']);result=catalog.finalize(root/'sscd_index')
        cache.prepare(root/'sscd_index',root/'sscd_cache',allow_bounded=bounded)
        cache.worker(root/'sscd_cache',threads=compute['threads']);cache.finalize(root/'sscd_cache')
        unit='vectors';count=result['vectors'];statistics={'vectors':count};outputs=[root/'sscd_index',root/'sscd_cache']
    elif stage=='sscd_search':
        from .reference import build
        from corpus_pipeline.postprocess import sscd_exact_npu as search
        options=cfg['sscd']
        reference=build(root/'sscd_index',threshold=options['threshold'],query_count=options['query_count'],
                        top_k=options['reference_top_k'],threads=compute['threads'],allow_bounded=bounded)
        search.prepare(root/'sscd_index',root/'sscd_cache',root/'sscd_search',threshold=options['threshold'],
                       backend_qualification=options['backend_qualification'],query_tile=options['query_tile'],
                       database_tile=options['database_tile'],candidate_batch=options['candidate_batch'],
                       min_free_gib=options['min_free_gib'],allow_bounded=bounded)
        progress=search.status(root/'sscd_search')
        if progress['completed_tasks']!=progress['total_tasks']:
            if not local:raise Pending('SSCD catalog and search plan ready; start the capped SSCD fleet')
            require(bounded,'Local CPU SSCD search is bounded-validation only')
            search.calibrate(root/'sscd_search',reference,device='cpu')
            search.worker(root/'sscd_search',worker_id='unified-cpu-validation',device='cpu')
        result=search.finalize(root/'sscd_search');unit='vector_pairs';count=result['completed_pairs'];statistics=result;outputs=[root/'sscd_search']
    elif stage=='assets':
        from corpus_pipeline.dedup.asset_release import build
        result=build(root/'reconciliation',root/'screening/COMPLETE.json',root/'asset_map',
                     threads=compute['threads'],memory=compute['memory_limit'])
        unit='assets';count=result['counts']['assets'];statistics=result['counts'];outputs=[root/'asset_map']
    elif stage=='graph':
        from corpus_pipeline.dedup import graph_policy,sscd_graph,asset_similarity_index
        graph_policy.build(root/'sscd_index',root/'asset_map/COMPLETE.json',root/'graph_policy',threads=compute['threads'])
        area=root/'sscd_graph';sscd_graph.prepare(root/'sscd_search',area,allow_bounded=bounded)
        sscd_graph.pack(area,workers=compute['threads']);sscd_graph.structure(area)
        sscd_graph.representatives(area,eligible=root/'graph_policy/eligible.u1',protected=root/'graph_policy/protected.u1',
                                   priority=root/'graph_policy/priority.u8');sscd_graph.finalize(area)
        result=asset_similarity_index.build(root/'asset_map/COMPLETE.json',root/'sscd_index',area,root/'asset_similarity_index',
                                            threads=compute['threads'],memory=compute['memory_limit'])
        unit='assets';count=result['assets'];statistics={'assets':count,'sscd_vectors':result['sscd_vectors'],'advisory_only':True}
        outputs=[root/'graph_policy',area,root/'asset_similarity_index']
    elif stage=='records':
        from corpus_pipeline.dedup.training_release import build
        quality=[{**x,'path':str(root/'screening'/x['path'])} for x in read(root/'screening/COMPLETE.json').get('record_quality_files',[])]
        result=build(root/'frontpass',root/'asset_similarity_index/COMPLETE.json',root/'training_candidate_index',
                     threads=compute['threads'],memory_limit=compute['memory_limit'],task_ids=task_ids(root,cfg),record_quality_files=quality)
        count=result['records'];statistics=record_statistics(root,result);outputs=[root/'training_candidate_index']
    elif stage=='source_view':
        from corpus_pipeline.dedup.source_view import build
        area=root/'training_candidate_source_view';result=build(root/'training_candidate_index',area,
            threads=compute['threads'],memory_limit=compute['memory_limit'])
        count=result['records'];statistics=result['counts'];outputs=[area];reader={'kind':'native_source_view','path':str(area)}
        if cfg['policy'].get('profile')==PROFILE:
            from corpus_pipeline.dedup.asset_supervision_view import build as build_asset_view
            asset_area=root/'training_asset_supervision_view'
            asset_result=build_asset_view(root/'training_candidate_index',asset_area,
                threads=compute['threads'],memory_limit=compute['memory_limit'])
            outputs.append(asset_area)
            reader['asset_supervision_view']=str(asset_area)
    else:raise ValueError('Unknown stage '+stage)
    require(type(count) is int and count>=0,'Backend did not report authoritative coverage')
    artifacts=sorted({path for output in outputs for path in finalized_files(output)})
    dependencies={parent:file_sha(root/'receipts'/f'{parent}.json') for parent in DEPENDENCIES[stage]}
    return release.seal_receipt(root/'receipts'/f'{stage}.json',stage=stage,collections=collection_bindings(root),
        coverage={'unit':unit,'expected':count,'processed':count,'missing':0,'pending_infra':0},
        inputs=[root/'EXECUTION.json'],artifacts=artifacts,dependencies=dependencies,statistics=statistics,
        semantics={'scope':cfg['scope'],'required_collections':cfg['required_collections'],
                   'task_direction_preserved':True,'ordered_supervision_preserved':True,
                   'sscd_candidates_automatically_deleted':False},reader=reader,policy_config=receipt_policy(cfg))


def run(config,*,local=False,stop_after=None,delegate=None):
    cfg=load(config);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'.coordinator.lock').open('a') as lock:
        fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,root=initialize(config)
        for stage in STAGES:
            if (root/'STOP').exists():
                result={'complete':False,'phase':'stopped','next_stage':stage}
                atomic_json(root/'STATUS.json',result);return result
            if validated_receipt(root,stage) is None:
                atomic_json(root/'STATUS.json',{'complete':False,'phase':'running','stage':stage,'updated_at':time.time()})
                try:execute_stage(cfg,root,stage,local=local,delegate=delegate)
                except Pending as exc:
                    value={'complete':False,'phase':'waiting','stage':stage,'reason':str(exc),'updated_at':time.time()}
                    atomic_json(root/'STATUS.json',value);return value
                except Exception as exc:
                    atomic_json(root/'STATUS.json',{'complete':False,'phase':'failed','stage':stage,
                        'error_type':type(exc).__name__,'error':str(exc),'updated_at':time.time()});raise
            if stop_after==stage:return {'complete':False,'phase':'stage_complete','stage':stage}
        if (root/'STOP').exists():return {'complete':False,'phase':'stopped','next_stage':'publish'}
        from corpus_pipeline.production.task_policy import receipt_policy
        result=release.finalize(root,{stage:root/'receipts'/f'{stage}.json' for stage in STAGES},
                                policy_config=receipt_policy(cfg),
                                expected_collections=collection_bindings(root),required_collections=cfg['required_collections'])
        atomic_json(root/'STATUS.json',{'complete':True,'phase':'published','scope':cfg['scope'],**result})
        return result


def status(config):
    cfg=load(config);root=Path(cfg['output']);path=root/'STATUS.json'
    result=read(path) if path.exists() else {'complete':False,'phase':'not_started'}
    result['receipts_present']=[stage for stage in STAGES if (root/'receipts'/f'{stage}.json').exists()]
    result['scope']=cfg['scope'];result['required_collections']=cfg['required_collections'];result['output']=str(root)
    result['stop_requested']=(root/'STOP').exists()
    if result.get('stage')=='normalize' and any(x['adapter']=='dcvlm_baseline_wds' for x in cfg['inputs']):
        source=next(x for x in cfg['inputs'] if x['adapter']=='dcvlm_baseline_wds')
        index=Path(source.get('index_output',root/'native'))
        if (index/'PLAN.json').exists():
            from . import baseline_index
            result['progress']=baseline_index.status(source['download_control'],index)
    if result.get('stage')=='frontpass' and (root/'frontpass/queue.json').exists():
        from corpus_pipeline.cluster.queue import status as queue_status
        result['progress']=queue_status(root/'frontpass')
    if result.get('stage')=='sscd_search' and (root/'sscd_search/PLAN.json').exists():
        from corpus_pipeline.postprocess.sscd_exact_npu import status as search_status
        result['progress']=search_status(root/'sscd_search')
    return result
