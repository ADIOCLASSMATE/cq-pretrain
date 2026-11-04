#!/usr/bin/env python3
"""Explicit exact-only publication using existing verified identity/record rules.

No approximate search or graph receipts are manufactured. Frontpass embeddings
remain bound to their real artifacts for a later optional advisory search.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from corpus_pipeline.production.common import atomic_json,digest,file_sha
from corpus_pipeline.unified import workflow,release
from corpus_pipeline.production.task_policy import receipt_policy
from corpus_pipeline.unified.policy import policy_hash
from corpus_pipeline.unified.config import require

SCHEMA='corpus.exact_only_release.v1'
MODE='exact_identity_ordered_supervision_v1'
UPSTREAM=('normalize','frontpass','reconcile','screening','assets')
SELECTION_GATE='corpus.strict_task_selection.v1'

def read(path):return json.loads(Path(path).read_text())
def bound(path):return release.artifact(path)
def immutable(path,value):
    path=Path(path)
    if path.exists():require(read(path)==value,'Exact release identity changed: '+str(path))
    else:atomic_json(path,value)

def strict_selection_check(candidate_root,screening,*,threads=2,memory_limit='4GB'):
    """Require resolved, explicitly permitted T2I before publishing any view.

    Candidate indices intentionally retain unresolved records for inspection.
    This publication contract is stricter. Historical frontpass pending alerts
    may be resolved by screening, so the joined final state is authoritative.
    I2T continues to bypass AI screening while always disabling image loss.
    """
    import duckdb
    from corpus_pipeline.dedup.training_release import parquet_sql,validate_artifacts
    candidate_root=Path(candidate_root);complete=read(candidate_root/'COMPLETE.json')
    require(complete.get('complete') is True and complete.get('manifest_sha256')==file_sha(candidate_root/'manifest.json'),
        'Strict selection requires a complete bound candidate index')
    manifest=read(candidate_root/'manifest.json');paths=[]
    for bucket in manifest['buckets']:
        files=[item for item in bucket['artifacts'] if Path(item['path']).name=='keep.parquet']
        require(len(files)==1,'Strict selection requires one selected file per bucket')
        validate_artifacts(candidate_root,{'artifacts':files});paths.append(candidate_root/files[0]['path'])
    require(bool(paths),'Strict selection requires candidate partitions')
    con=duckdb.connect()
    try:
        con.execute('SET threads='+str(int(threads)));con.execute("SET memory_limit='"+memory_limit.replace("'","''")+"'")
        con.execute('CREATE VIEW selected AS SELECT * FROM '+parquet_sql(paths,union_by_name=True))
        required={'decision','processing_policy','task_route','image_loss_allowed','unresolved_screening',
            'integrity_quarantine','preexcluded','status','reason_codes','screening_evidence'}
        columns={row[0] for row in con.execute('DESCRIBE selected').fetchall()}
        require(required<=columns,'Strict selection missing final task or screening columns')
        values=con.execute('''SELECT count(*),count(*) FILTER(WHERE task_route='i2t'),
            count(*) FILTER(WHERE task_route='t2i'),
            count(*) FILTER(WHERE decision IS DISTINCT FROM 'keep' OR
                processing_policy IS DISTINCT FROM 'corpus.task_policy.v1' OR
                task_route IS NULL OR task_route NOT IN ('i2t','t2i','mixed','text','unknown') OR
                image_loss_allowed IS NULL OR (image_loss_allowed AND task_route<>'t2i')),
            count(*) FILTER(WHERE task_route='t2i' AND (
                image_loss_allowed IS DISTINCT FROM true OR unresolved_screening IS DISTINCT FROM false OR
                integrity_quarantine IS DISTINCT FROM false OR preexcluded IS DISTINCT FROM false OR
                status IS NULL OR status NOT IN ('frontpass_pass_pending_global_dedup','pending') OR
                (status='pending' AND (reason_codes IS NULL OR len(reason_codes)=0 OR
                    len(list_filter(reason_codes,r->r NOT IN
                        ('ai_suspect_pending_api','pending_edit_lineage','pending_inference_or_integrity_error')))>0)) OR
                coalesce(len(list_filter(screening_evidence,e->e.action IS NULL OR e.action<>'retain_annotated')),0)>0))
            FROM selected''').fetchone()
    finally:
        con.close()
    result=dict(zip(('selected_records','i2t_records','t2i_records','invalid_task_permissions','unresolved_t2i_records'),values))
    require(result['selected_records']==manifest['counts'].get('keep',0),'Strict selected coverage mismatch')
    require(result['invalid_task_permissions']==0,'Strict selection rejects invalid task/image-loss permissions')
    require(result['unresolved_t2i_records']==0,'Strict selection rejects unresolved or unpermitted T2I records')
    if result['t2i_records']:
        require(screening.get('complete') is True and screening.get('pending_infra')==0 and
            screening.get('review_uncertain')==0 and screening.get('counts',{}).get('pending',0)==0,
            'Strict selection requires completed non-pending T2I screening')
    return {'schema':SELECTION_GATE,'complete':True,**result,
        'candidate_manifest_sha256':file_sha(candidate_root/'manifest.json'),
        'screening_complete_sha256':digest(screening),'i2t_image_loss_allowed':False,
        'pending_t2i_selected':False,'same_image_different_supervision_preserved':True}

def verify(root,verify_payloads=True):
    root=Path(root);pointer=read(root/'EXACT_CURRENT.json')
    require(pointer.get('schema')==SCHEMA and pointer.get('complete') is True,'Not an exact-only publication')
    for key in ('manifest','completion'):
        item=pointer[key];require(file_sha(item['path'])==item['sha256'],'Changed exact release '+key)
    manifest=read(pointer['manifest']['path']);done=read(pointer['completion']['path'])
    require(manifest.get('complete') is True and done.get('complete') is True and
        manifest.get('schema')==done.get('schema')==SCHEMA and
        manifest.get('dedup_mode')==MODE and manifest.get('advisory_search_performed') is False and
        done['manifest_sha256']==pointer['manifest']['sha256'],'Incorrect exact-only semantics')
    require(set(manifest['upstream'])==set(UPSTREAM),'Missing exact release prerequisites')
    gate=manifest.get('selection_gate');require(isinstance(gate,dict),'Missing strict task selection gate')
    workflow.verify_artifact(gate);selection=read(gate['path'])
    require(selection.get('schema')==SELECTION_GATE and selection.get('complete') is True and
        selection.get('invalid_task_permissions')==selection.get('unresolved_t2i_records')==0 and
        selection.get('selected_records')==manifest['counts'].get('keep',0),
        'Invalid strict task selection gate')
    if verify_payloads:
        for item in [*manifest['upstream'].values(),*manifest['outputs']]:
            workflow.verify_artifact(item)
        for item in manifest['upstream'].values():
            value=read(item['path'])
            for artifact in value['inputs']+value['artifacts']:workflow.verify_artifact(artifact)
        for name in ('records','source_view'):
            value=read(Path(pointer['manifest']['path']).parent/(name+'.receipt.json'))
            for item in value['inputs']+value['artifacts']:workflow.verify_artifact(item)
    return manifest

def publish(config,*,allow_validation=False):
    from corpus_pipeline.dedup import training_release,source_view,asset_supervision_view
    cfg=workflow.verify_execution(config);root=Path(cfg['output']);area=root/'exact_release'
    require(cfg['scope']=='full' or allow_validation,'Exact full publication requires full source scope')
    require(cfg.get('sscd',{}).get('dedup_mode')==MODE,'Explicit exact-only opt-in required')
    require(cfg['policy'].get('profile')=='task_aware_v1' and cfg['policy'].get('t2i_uncertain')=='exclude',
        'Exact-only publication requires strict task-aware I2T/T2I policy')
    require(not (root/'STOP').exists(),'Stopped source cannot publish')
    for name in UPSTREAM[:-1]:require(workflow.validated_receipt(root,name) is not None,'Unfinished prerequisite '+name)
    if workflow.validated_receipt(root,'assets') is None:workflow.execute_stage(cfg,root,'assets')
    if (root/'EXACT_CURRENT.json').exists():return verify(root)
    policy=receipt_policy(cfg);bindings=workflow.collection_bindings(root)
    normalize=read(root/'receipts/normalize.json')
    if cfg['scope']=='full':release._verify_full_native_snapshot(normalize,bindings)
    compute=cfg['compute'];screen=read(root/'screening/COMPLETE.json')
    quality=[{**x,'path':str(root/'screening'/x['path'])} for x in screen.get('record_quality_files',[])]
    candidate=training_release.build(root/'frontpass',root/'asset_map/COMPLETE.json',root/'training_candidate_index',
        threads=compute['threads'],memory_limit=compute['memory_limit'],task_ids=workflow.task_ids(root,cfg),record_quality_files=quality)
    rows=candidate['records'];expected=normalize['coverage']['expected']
    require(rows==expected and sum(candidate['counts'].values())==rows,'Exact record coverage differs from native source')
    selection=strict_selection_check(root/'training_candidate_index',screen,
        threads=compute['threads'],memory_limit=compute['memory_limit'])
    immutable(area/'selection_gate.json',selection)
    statistics=workflow.record_statistics(root,candidate)
    semantics={'scope':cfg['scope'],'required_collections':cfg['required_collections'],'task_direction_preserved':True,
        'ordered_supervision_preserved':True,'sscd_candidates_automatically_deleted':False,
        'dedup_mode':MODE,'advisory_search_performed':False,'frontpass_embeddings_retained':True}
    coverage={'unit':'records','expected':rows,'processed':rows,'missing':0,'pending_infra':0}
    records=release.seal_receipt(area/'records.receipt.json',stage='records',collections=bindings,coverage=coverage,
        inputs=[root/'EXECUTION.json',root/'receipts/frontpass.json',root/'receipts/assets.json'],
        artifacts=workflow.finalized_files(root/'training_candidate_index')+[area/'selection_gate.json'],
        dependencies={n:file_sha(root/'receipts'/(n+'.json')) for n in ('frontpass','assets')},
        statistics=statistics,semantics=semantics,policy_config=policy)
    source_view.build(root/'training_candidate_index',root/'training_candidate_source_view',threads=compute['threads'],memory_limit=compute['memory_limit'])
    asset_supervision_view.build(root/'training_candidate_index',root/'training_asset_supervision_view',threads=compute['threads'],memory_limit=compute['memory_limit'])
    counts={k:candidate['counts'].get(k,0) for k in ('keep','exclude','quarantine')}
    reader=release._verify_source_view(root/'training_candidate_source_view',rows,counts)
    reader.update(kind='native_source_view',path=str(root/'training_candidate_source_view'),
        asset_supervision_view=str(root/'training_asset_supervision_view'))
    outputs=workflow.finalized_files(root/'training_candidate_source_view')+workflow.finalized_files(root/'training_asset_supervision_view')
    release.seal_receipt(area/'source_view.receipt.json',stage='source_view',collections=bindings,coverage=coverage,
        inputs=[area/'records.receipt.json'],artifacts=outputs,dependencies={'records':file_sha(area/'records.receipt.json')},
        statistics=counts,semantics=semantics,reader=reader,policy_config=policy)
    upstream={name:bound(root/'receipts'/(name+'.json')) for name in UPSTREAM}
    payloads=[bound(area/'records.receipt.json'),bound(area/'source_view.receipt.json'),bound(area/'selection_gate.json')]
    manifest={'schema':SCHEMA,'complete':True,'scope':cfg['scope'],'dedup_mode':MODE,'records':rows,'counts':counts,
        'per_collection':statistics['per_collection'],'collections':bindings,'policy':policy,'policy_sha256':policy_hash(policy),
        'advisory_search_performed':False,'sscd_candidates_used_for_deletion':False,'frontpass_embeddings_retained':True,
        'source_files_modified':False,'image_payloads_copied':False,'upstream':upstream,'outputs':payloads,'reader':reader,
        'selection_gate':bound(area/'selection_gate.json'),
        'implementation_sha256':file_sha(__file__),'limitations':['Approximate SSCD candidate search and graph were not run; exact identity and ordered supervision determine deletion.']}
    immutable(area/'manifest.json',manifest)
    done={'schema':SCHEMA,'complete':True,'scope':cfg['scope'],'manifest_sha256':file_sha(area/'manifest.json'),
        'records':rows,'counts':counts,'advisory_search_performed':False}
    immutable(area/'COMPLETE.json',done)
    immutable(root/'EXACT_CURRENT.json',{'schema':SCHEMA,'complete':True,'scope':cfg['scope'],'manifest':bound(area/'manifest.json'),
        'completion':bound(area/'COMPLETE.json'),'release_id':digest(manifest)})
    return verify(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    value=publish(args.config);print(json.dumps({k:value[k] for k in ('schema','complete','records','counts','advisory_search_performed')}))
