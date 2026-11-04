#!/usr/bin/env python3
"""Run-owned controller: sealed sources -> existing unified DAG -> exact union.

This script does not schedule platform resources itself. The existing resource
manager and baseline supervisor own all creation, retries and verified release.
Only one source supervisor may be active, so source and accelerator phases share
one explicitly authorized card ceiling. Missing native receipts never count as processed sources.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import shlex, signal, socket, subprocess, sys, time

from .config import (SCHEMA, SOURCES, LOGICAL_SOURCES, SUPPLEMENTAL_PARENTS,
    PROXY_KEYS, read, sha, save, immutable, check_config, environment)
from .normalization import native_ready, ingest_integrity_verified, adopt_native
SCRIPT = Path(__file__).resolve()
MODULE = 'corpus_pipeline.orchestration.controller'
EXACT_RELEASE = SCRIPT.parents[1]/'unified/exact_release.py'

def card_budget(rows,ceiling=512):
    # Enough parallelism for <=~20 min frontpass at conservative 50 img/s/card.
    demand=max(1,(rows+59999)//60000)
    return min(ceiling,1<<(demand-1).bit_length())

def replace_paths(value,mapping):
    if isinstance(value,dict):return {k:replace_paths(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [replace_paths(v,mapping) for v in value]
    if isinstance(value,str):
        for key,val in mapping.items():value=value.replace(key,val)
    return value

def prepare_source(cfg,name):
    from corpus_pipeline.unified import config as pipeline_config,fleet
    from corpus_pipeline.unified.baseline_pipeline_supervisor import checked_ops
    run=Path(cfg['run']);source=run/'normalized_sources'/name;root=run/'sources'/name
    if not ingest_integrity_verified(cfg):raise ValueError('Ingestion distributed lock audit is not verified')
    if not native_ready(source):raise ValueError('Cannot prepare incomplete source')
    rows=read(source/'COMPLETE.json')['statistics']['samples'];cards=card_budget(rows,cfg['max_cards'])
    if (root/'ops.json').exists():
        checked_ops(root/'ops.json')
        existing=read(root/'pipeline.json')
        if existing['inputs']!=[{'collection':'genund','adapter':'genund_native','root':str(source)}]:
            raise ValueError('Existing source input binding changed')
        return {'source':name,'rows':rows,'frontpass_cards':existing['resources']['frontpass_max_cards'],
                'pipeline':str(root/'pipeline.json'),'ops':str(root/'ops.json')}
    pipeline={'schema':'corpus.unified.pipeline.v1','scope':'full','required_collections':['genund'],
        'inputs':[{'collection':'genund','adapter':'genund_native','root':str(source)}],'output':str(root),
        'policy':{'profile':'task_aware_v1','t2i_uncertain':'exclude'},
        'compute':{'threads':4,'io_workers':8,'memory_limit':'32GB','rows_per_task':8192,'batch_size':32,
          'dtype':'float32','inference_engine':'optimized','optimized_qualification':cfg['frontpass_qualification'],
          'models_dir':cfg['repository']+'/corpus_pipeline/models/ai_detectors',
          'sscd_model':cfg['repository']+'/corpus_pipeline/models/sscd/sscd_disc_mixup.torchscript.pt'},
        'cache':{'directory':str(root/'image_cache'),'histories':[],'enabled':False},
        'resources':{'max_cards':cfg['max_cards'],'frontpass_max_cards':cards,'review_max_cards':cfg['max_cards'],'sscd_max_cards':cards,'prefer_occupied_nodes':True},
        'screening':{'backend':'serving','primary':'qwen3.6','concurrency':1024,'history_decision_manifests':[],
          'quality_decisions':[],'prompt':cfg['runtime']+'/corpus_pipeline/production/image_screening_prompt_v4.txt',
          'primary_pool':str(root/'review_pool/pool.json'),'api_key_file':cfg['api_key_file']},
        'sscd':{'backend_qualification':cfg['sscd_qualification'],'dedup_mode':cfg['dedup_mode']}}
    pipeline=pipeline_config.checked(pipeline);config_path=root/'pipeline.json';immutable(config_path,pipeline)
    template_path=root/'fleet-template.json'
    # Resume keeps its immutable platform identity; newly admitted sources use the single-card template.
    template=read(template_path) or read(run/'ops/fleet_template.json');template['workload']['max_cards']=cards
    template['state_dir']=str(root/'fleet/template/controller')
    template['adapter_config']['registry_dir']=str(root/'fleet/template/adapter')
    template['adapter_config']['worker_state_dir']=str(root/'fleet/remote_workers')
    immutable(template_path,template)
    fleet.build(config_path,template_path)
    review_template=read(cfg['review_template'])
    if review_template is None:raise ValueError('Review pool template is not prepared')
    review=replace_paths(review_template,{'{source_run}':str(root),'{run}':str(run),'{runtime}':cfg['runtime'],'{source_slug}':dict(zip(SOURCES,('openuni','scalecap','densefusion','pixmo','finevision','bliplong','dense4v')))[name]})
    immutable(root/'review_pool/config.json',review)
    env={k:environment(cfg)[k] for k in ('PYTHONPATH','CORPUS_DATASET_ROOTS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCH_DEVICE_BACKEND_AUTOLOAD','PYTHONUNBUFFERED')}
    def command(argv):return {'argv':argv,'cwd':cfg['runtime'],'env':env}
    def helper(action,python=None):return command([python or cfg['python'],'-m',MODULE,action,'--config',cfg['config_path'],'--source',name])
    def work(stage=None):
        args=[cfg['python'],'-m','corpus_pipeline.unified','run','--config',str(config_path),'--watch']
        if stage:args+=['--stage',stage]
        return command(args)
    port=cfg['queue_port_base']+SOURCES.index(name)
    commands={'normalize':helper('adopt'),'queue_rpc':command([cfg['python'],'-m','corpus_pipeline.cluster.queue_rpc','serve','--root',str(root/'frontpass'),'--port',str(port)]),
        'frontpass_workflow':work('frontpass'),'screening_workflow':helper('screening-remote',cfg['sdk_python']),
        'sscd_workflow':work('sscd_search'),'publish_workflow':command([cfg['python'],'-m','corpus_pipeline.unified.exact_release','--config',str(config_path)]),
        'review_start':helper('review-start'),'review_ready':helper('review-ready',cfg['sdk_python']),
        'review_stop':helper('review-stop',cfg['sdk_python']),'review_released':helper('review-released',cfg['sdk_python']),
        'index_stop':helper('index-stop'),'index_released':helper('index-released')}
    for stage,stem in [('frontpass','frontpass'),('sscd_search','sscd')]:
        manager=str(root/'fleet'/(stage+'.json'))
        commands[stem+'_manager']=command([cfg['sdk_python'],'-m','corpus_pipeline.orchestration.fleet','run','--config',manager])
        commands[stem+'_stop']=command([cfg['sdk_python'],'-m','corpus_pipeline.resource_manager','stop','--config',manager])
        commands[stem+'_released']=command([cfg['sdk_python'],'-m','corpus_pipeline.unified.baseline_fleet_ops','released','--manager-config',manager])
    # Baseline supervisor requires each operation to name its source run explicitly.
    for entry in commands.values():
        if MODULE in entry['argv']:entry['argv']+=['--source-run',str(root)]
    ops={'schema':'corpus.baseline_pipeline_supervisor.v1','run_root':str(root),'control':str(root/'orchestrator'),
        'repository':cfg['runtime'],'deadline_utc':cfg['deadline_utc'],'queue_port':port,'poll_seconds':10,
        'retry_seconds':15,'max_retry_seconds':300,'max_failures':30,'stop_grace_seconds':120,'commands':commands,
        'receipts':{s:str(root/'receipts'/(s+'.json')) for s in ('normalize','frontpass','screening','sscd_search')},
        'gates':{'frontpass_plan':{'checks':[{'path':str(root/'frontpass/plan.json'),'equals':{'scope':'unified_full','policy':'dcvlm.offline_frontpass.v1'}}]},
          'sscd_plan':{'checks':[{'path':str(root/'sscd_search/PLAN.json'),'equals':{'identity.schema':'corpus.sscd_exact_npu.v1','identity.allow_bounded':False}}]},
          **{name:{'command':name} for name in ('frontpass_released','sscd_released','index_released','review_ready')}}}
    ops['receipts']['publication']=str(root/'CURRENT.json');immutable(root/'ops.json',ops);checked_ops(root/'ops.json')
    return {'source':name,'rows':rows,'frontpass_cards':cards,'pipeline':str(config_path),'ops':str(root/'ops.json')}

def review_inputs_prepared(root):
    """A warm model alone must not race the owner still building its API queue."""
    root=Path(root);reconciliation=read(root/'reconciliation/COMPLETE.json')
    pending=read(root/'screening/PENDING.json')
    if reconciliation is None or pending is None:return False
    if reconciliation.get('complete') is not True or not (root/'receipts/reconcile.json').exists():return False
    if pending.get('identity',{}).get('reconciliation_complete_sha256')!=sha(root/'reconciliation/COMPLETE.json'):
        raise ValueError('Prepared review inputs do not match sealed reconciliation')
    return pending.get('api_selection') is not None

def review_action(cfg,name,action):
    root=Path(cfg['run'])/'sources'/name;config=root/'pipeline.json'
    marker=root/'REVIEW_NOT_REQUIRED.json'
    if action=='review-start':
        frontpass=read(root/'frontpass/FRONTPASS_COMPLETE.json',{})
        if frontpass.get('counts',{}).get('image:ai_suspect_pending_api',0)>0:
            subprocess.run([cfg['sdk_python'],'-m','corpus_pipeline.orchestration.review',
                '--config',cfg['config_path'],'--source',name],env=environment(cfg),check=True)
        # A resumed screening worker already owns the workflow lock. Its sealed
        # reconciliation and bound API selection are enough to resume the pool.
        if not review_inputs_prepared(root):
            from corpus_pipeline.unified import workflow
            from corpus_pipeline.unified.screening import build_decisions
            from corpus_pipeline.production.task_policy import receipt_policy
            workflow.run(config,stop_after='reconcile')
            pipeline=workflow.load(config)
            result=build_decisions(root/'reconciliation',root/'screening',review_root=root/'api_review',policy_config=receipt_policy(pipeline))
            if result['complete']:
                # A warm pool may prove unnecessary after exact-source reduction.
                # Release it before the no-review marker can bypass release gates.
                if (root/'review_pool/RUN_PROCESS.json').exists():
                    command=[cfg['sdk_python'],'-m','corpus_pipeline.serving.review_control']
                    subprocess.run([*command,'stop','--config',str(root/'review_pool/config.json')],env=environment(cfg),check=True)
                    while True:
                        checked=subprocess.run([*command,'released','--config',str(root/'review_pool/config.json')],env=environment(cfg),check=True,capture_output=True,text=True)
                        if json.loads(checked.stdout).get('released') is True:break
                        time.sleep(5)
                immutable(marker,{'complete':True,'reason':'no_unresolved_review_candidates',
                    'screening_complete_sha256':sha(root/'screening/COMPLETE.json')})
                return 0
        # Exec keeps existing supervisor ownership and stop semantics intact.
        child=subprocess.Popen([cfg['sdk_python'],'-m','corpus_pipeline.serving.budget_pool','guard','--config',str(root/'review_pool/config.json')],env=environment(cfg),start_new_session=True)
        def terminate(*_):
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,terminate)
        return child.wait()
    if marker.exists():
        value={'released':True,'reserved':0,'reason':'no_review_resources_started'} if action!='review-ready' else {'ready':False,'ready_replicas':0,'reason':'no_review_required'}
        print(json.dumps(value));return 0
    if action=='review-ready' and not review_inputs_prepared(root):
        print(json.dumps({'ready':False,'ready_replicas':0,'reason':'CPU_review_inputs_still_preparing'}));return 0
    action={'review-ready':'ready','review-stop':'stop','review-released':'released'}[action]
    os.execv(cfg['sdk_python'],[cfg['sdk_python'],'-m','corpus_pipeline.serving.review_control',action,'--config',str(root/'review_pool/config.json')])

def screening_remote(cfg,name):
    root=Path(cfg['run'])/'sources'/name;config=root/'pipeline.json'
    if (root/'REVIEW_NOT_REQUIRED.json').exists():
        result=subprocess.run([cfg['python'],'-m','corpus_pipeline.unified','run','--config',str(config),'--stage','screening'],cwd=cfg['runtime'],env=environment(cfg))
        return result.returncode
    from corpus_pipeline.cluster.notebook_exec import execute
    notebook=cfg['review_notebook']
    registered=next(row for row in cfg['cpu_workers'] if row['name']==notebook)
    remote_python=str(Path(cfg['run'])/'ops/cpu_python.sh')
    attempt_path=root/'REMOTE_SCREENING_ATTEMPT.json'
    attempt=read(attempt_path,{'attempt':1})['attempt']
    if type(attempt) is not int or attempt<1:raise ValueError('Invalid remote screening attempt')
    operation='screen-'+hashlib.sha256(str(root).encode()).hexdigest()[:16]+'-'+str(attempt)
    save(attempt_path,{'attempt':attempt,'operation':operation})
    runner=[remote_python,'-m','corpus_pipeline.resource_manager.command_runner','--root',str(root/'remote_screening'),'--resource-key',notebook]
    command=[remote_python,'-m','corpus_pipeline.unified','run','--config',str(config),'--stage','screening','--watch']
    launched=False
    while not (root/'receipts/screening.json').exists():
        if (root/'STOP').exists() or (Path(cfg['run'])/'STOP').exists():return 75
        args=runner+(['launch','--operation-id',operation,'--',*command] if not launched else ['status'])
        result=execute('cd '+shlex.quote(cfg['runtime'])+' && '+shlex.join(args),account=cfg['account'],workspace=cfg['workspace'],notebook=notebook,
            notebook_id=registered['notebook_id'],workspace_id=registered['workspace_id'],timeout=45,max_output_bytes=8192)
        if result.get('completed') and result.get('returncode')==0:
            candidates=[]
            for line in result.get('output','').splitlines():
                try:value=json.loads(line)
                except ValueError:continue
                if isinstance(value,dict) and 'running' in value:candidates.append(value)
            if len(candidates)!=1:raise ValueError('Ambiguous remote screening process receipt')
            value=candidates[0]
            if value.get('launch_operation_id')!=operation:raise ValueError('Remote screening operation ownership differs')
            save(root/'REMOTE_SCREENING_STATUS.json',{**value,'observed_at':time.time(),'notebook':notebook})
            launched=True
            if value.get('completed') and not (root/'receipts/screening.json').exists():
                if value.get('running'):raise ValueError('Completed remote screening still reports running')
                save(attempt_path,{'attempt':attempt+1,'previous_operation':operation,'previous_exit_code':value.get('exit_code')})
                raise RuntimeError('Remote screening ended without sealed receipt: '+str(value.get('exit_code')))
        else:save(root/'REMOTE_SCREENING_STATUS.json',{'observed_at':time.time(),'notebook':notebook,'transport_error':result.get('error_type'),'complete':False})
        time.sleep(20)
    return 0

def source_supervise(cfg,name):
    from corpus_pipeline.unified.baseline_pipeline_supervisor import Supervisor,checked_ops
    from corpus_pipeline.unified import baseline_supervisor as process
    root=Path(cfg['run'])/'sources'/name
    class SourceSupervisor(Supervisor):
        def __init__(self,config):
            super().__init__(config)
            if self.state['phase']=='publish' and (root/'EXACT_CURRENT.json').exists():self.state['phase']='exact_publish'
        def step(self,now=None):
            now=time.time() if now is None else now
            if self.state['phase']=='review_start' and (root/'REVIEW_NOT_REQUIRED.json').exists():self.state['phase']='screening'
            if (Path(cfg['run'])/'STOP').exists():self.stop_requested=True
            if self.state['phase']=='sscd':self.state['phase']='exact_publish'
            if self.state['phase']=='exact_publish':
                if self.stop_requested or (root/'STOP').exists():
                    done=self.stopping(now);self.save(now);return done
                if (root/'EXACT_CURRENT.json').exists():
                    from corpus_pipeline.unified import exact_release
                    exact_release.verify(root)
                    released=self.gate('frontpass_released',now) and self.gate('index_released',now) and self.released_review(now)
                    if released and self.stop(tuple(self.config['commands']),now):
                        self.state['phase']='complete';self.save(now)
                        save(self.control/'COMPLETE.json',{'complete':True,'time':now,'dedup_mode':cfg['dedup_mode'],
                            'advisory_search_performed':False,'publication':str(root/'EXACT_CURRENT.json'),
                            'publication_sha256':sha(root/'EXACT_CURRENT.json'),'deadline_met':now<=cfg['deadline_epoch']})
                        return True
                else:self.ensure('publish_workflow',now,repeat=True)
                self.save(now);return False
            return super().step(now)
    supervisor=SourceSupervisor(checked_ops(root/'ops.json'))
    with process.exclusive(supervisor.control/'controller.lock'):
        supervisor.bind()
        for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,lambda *_:setattr(supervisor,'stop_requested',True))
        while not supervisor.step():time.sleep(10)
    return 0 if supervisor.state['phase']=='complete' else 2

def union(cfg):
    from corpus_pipeline.unified.baseline_union import export_candidate_index,merge_selected_indices,verify_reference_release
    run=Path(cfg['run']);indices=[];recons=[cfg['baseline_reconciliation']]
    for name in SOURCES:
        root=run/'sources'/name
        proof=read(root/'orchestrator/COMPLETE.json')
        if not proof or proof.get('complete') is not True:raise ValueError('Cannot union unfinished source: '+name)
    for name in SOURCES:
        root=run/'sources'/name
        export_candidate_index(root/'training_candidate_index',root/'selected_reference_index')
        indices.append(root/'selected_reference_index');recons.append(root/'reconciliation')
    result=merge_selected_indices(cfg['baseline_index'],indices,run/'joint_release',reconciliations=recons,threads=4,memory_limit='8GB')
    verify_reference_release(run/'joint_release',verify_payloads=True)
    save(run/'COMPLETE.json',{'schema':SCHEMA,'complete':True,'time':time.time(),'deadline_met':time.time()<=cfg['deadline_epoch'],
        'sources':list(LOGICAL_SOURCES),'execution_sources':list(SOURCES),'supplemental_logical_parents':SUPPLEMENTAL_PARENTS,'baseline_selected_records':read(Path(cfg['baseline_index'])/'COMPLETE.json')['records'],'records':result['records'],
        'joint_release_complete_sha256':sha(run/'joint_release/COMPLETE.json')})
    return result

def controller(cfg):
    from corpus_pipeline.unified import baseline_supervisor as process
    run=Path(cfg['run']);area=run/'ops/stage_control';area.mkdir(parents=True,exist_ok=True)
    state=read(area/'STATE.json',{'sources':{},'phase':'waiting_normalized_sources'})
    binding={'schema':SCHEMA,'config_sha256':sha(cfg['config_path']),'script_sha256':sha(SCRIPT),'exact_release_sha256':sha(EXACT_RELEASE),'host':socket.gethostname()}
    immutable(area/'INPUT.json',binding)
    with process.exclusive(area/'controller.lock'):
        while True:
            ensure_cpu_controller(cfg)
            state.update(updated_at=time.time(),controller=process.process_identity(os.getpid()),deadline_utc=cfg['deadline_utc'],deadline_seconds_remaining=cfg['deadline_epoch']-time.time())
            active=[]
            for name,row in state['sources'].items():
                argv=[cfg['python'],'-m',MODULE,'source-supervise','--config',cfg['config_path'],'--source',name]
                matches=process.matching_processes(argv,cfg['runtime'])
                if len(matches)>1:raise RuntimeError('Multiple owned source supervisors')
                if matches:row['identity']=matches[0];active.append(name)
                complete=read(run/'sources'/name/'orchestrator/COMPLETE.json')
                row['complete']=bool(complete and complete.get('complete') is True)
            if len(active)>1:raise RuntimeError('Global source stage ownership conflict')
            if (run/'STOP').exists():
                for name in active:
                    (run/'sources'/name/'STOP').touch()
                state['phase']='stopping' if active else 'stopped';save(area/'STATE.json',state)
                if not active:return 2
            elif active:
                state.update(phase='processing_source',active_source=active[0])
            elif not ingest_integrity_verified(cfg):
                state.update(phase='waiting_ingest_integrity_audit',active_source=None)
            elif all(state['sources'].get(n,{}).get('complete') for n in SOURCES):
                state['phase']='exact_union';save(area/'STATE.json',state);union(cfg)
                state['phase']='complete';state['complete']=True;save(area/'STATE.json',state);return 0
            else:
                # Ready sources can proceed without waiting for earlier incomplete collections.
                unfinished=[n for n in SOURCES if n in state['sources'] and not state['sources'][n].get('complete')]
                selected=unfinished[0] if unfinished else next((n for n in SOURCES if not state['sources'].get(n,{}).get('complete') and native_ready(run/'normalized_sources'/n)),None)
                if selected:
                    row=state['sources'].setdefault(selected,{})
                    if row.get('attempts',0)>=3:
                        state.update(phase='needs_repair',active_source=selected)
                    else:
                        details=prepare_source(cfg,selected);row.update(details,attempts=row.get('attempts',0)+1)
                        save(area/'STATE.json',state)
                        argv=[cfg['python'],'-m',MODULE,'source-supervise','--config',cfg['config_path'],'--source',selected]
                        with (area/(selected+'.log')).open('ab',buffering=0) as log:
                            child=subprocess.Popen(argv,cwd=cfg['runtime'],env=environment(cfg),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                        row['pid']=child.pid;state.update(phase='processing_source',active_source=selected)
                else:state.update(phase='waiting_normalized_sources',active_source=None)
            save(area/'STATE.json',state);time.sleep(15)


def ensure_cpu_controller(cfg):
    """Adopt one exact controller; a persistent owned CPU lifecycle survives resumes."""
    from corpus_pipeline.unified import baseline_supervisor as process
    argv=[cfg['sdk_python'],'-m','corpus_pipeline.orchestration.cpu','--config',cfg['config_path']]
    matches=process.matching_processes(argv,cfg['runtime'])
    if len(matches)>1:raise RuntimeError('Multiple CPU lifecycle owners')
    if matches:return
    area=Path(cfg['run'])/'ops'
    with (area/'cpu_controller.log').open('ab') as log:
        child=subprocess.Popen(argv,cwd=cfg['runtime'],env=environment(cfg),stdin=subprocess.DEVNULL,
            stdout=log,stderr=log,start_new_session=True)
    save(area/'CPU_CONTROLLER_PROCESS.json',{'pid':child.pid,'command':argv,'time':time.time()})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','adopt','source-supervise','screening-remote','review-start','review-ready','review-stop','review-released','index-stop','index-released','union'))
    parser.add_argument('--config',required=True);parser.add_argument('--source',choices=SOURCES);parser.add_argument('--source-run')
    args=parser.parse_args();cfg=check_config(args.config)
    sys.path.insert(0,cfg['runtime'])
    for name in PROXY_KEYS:os.environ.pop(name,None)
    os.environ.update(environment(cfg));os.chdir(cfg['runtime'])
    if args.action=='union':union(cfg);return 0
    if args.source is None:parser.error('--source required')
    root=Path(cfg['run'])/'sources'/args.source
    if args.source_run is not None and Path(args.source_run).resolve()!=root.resolve():raise ValueError('Wrong source run binding')
    if args.action=='prepare':print(json.dumps(prepare_source(cfg,args.source)));return 0
    if args.action in ('adopt','source-supervise') and not ingest_integrity_verified(cfg):
        raise ValueError('Ingestion distributed lock audit is not verified')
    if args.action=='adopt':adopt_native(root/'pipeline.json');return 0
    if args.action=='source-supervise':return source_supervise(cfg,args.source)
    if args.action=='screening-remote':return screening_remote(cfg,args.source)
    if args.action.startswith('review-'):return review_action(cfg,args.source,args.action)
    if args.action in ('index-stop','index-released'):
        # This source supervisor owns no ingestion instances. Its native input was sealed before adoption.
        if not native_ready(Path(cfg['run'])/'normalized_sources'/args.source):raise ValueError('Source became incomplete')
        print(json.dumps({'released':True,'reserved':0,'reason':'sealed_native_input_no_index_workers_owned'}));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'complete':False,'error_type':type(error).__name__}),file=sys.stderr);raise
