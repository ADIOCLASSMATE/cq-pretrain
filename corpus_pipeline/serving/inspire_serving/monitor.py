"""Native Serving observer, qualification and endpoint publisher.

This controller never creates Jobs, speculative copies, or placement constraints.
Platform native scheduling/restarts handle PENDING replicas. Only a confirmed
preempted, STOPPED LOW service may be started in place, at most once per event.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from corpus_pipeline.serving.platform_client import InspireClient

ROOT = Path(__file__).resolve().parent
MONITOR_CODE_REVISION = 'sha256:' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RUNTIME = ROOT/'runtime'
REGISTRY = ROOT.parent/'endpoints.json'
WORKSPACE = '昇腾卡公共空间'
PROJECT = '多模态大模型新架构评测探索与scaling-law'
PYTHON = ROOT.parents[2]/'.venv/bin/python'
LOCK = threading.Lock()
CANDIDATES = {}
PROCESSES = {}
PACKAGES = ('vllm','vllm-ascend','torch','torch-npu','transformers')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.chmod(0o600)
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid':int(pid), 'starttime_ticks':int(fields[19]),
                'command':[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]}
    except (OSError, ValueError, IndexError):
        return None


def import_handoffs():
    for tier, priority in [('high',6),('low',3)]:
        path = RUNTIME/f'gemma-{tier}.json'
        source = ROOT.parent/f'gemma4/serving_{tier}_handoff_20261023.json'
        if not path.exists() and source.exists():
            state = load(source)
            state.update(name=state['service_name'], priority=priority, cards=4,
                         expected_nodes=1, expected_replicas=1, runtime_layout='fixed',
                         request_overrides={'chat_template_kwargs':{'enable_thinking':False}},
                         payload=load(state['create_kwargs_path']), auto_recovery_enabled=tier=='low')
            save(path,state)


def assert_owned(state, detail):
    view = detail.to_dict()
    assert detail.ref.key == state['service_id'], 'Service identity changed'
    assert detail.name == state['name'] and view['project'] == PROJECT
    assert type(state['priority']) is int and state['tier'] in ('high', 'low')
    assert state['priority'] in ((4, 6) if state['tier']=='high' else (1, 3))
    assert view['priority'] == state['priority']
    assert view['replicas'] == 1 and view['nodes_per_replica'] == state['expected_nodes']
    raw = detail.raw
    for key in ('workspace_id','project_id','model_id','model_version','mirror_id','command','port','logic_compute_group_id'):
        assert raw[key] == state['payload'][key], f'Native Serving field changed: {key}'
    for key in ('gpu_count','cpu_count','memory_size_gib'):
        assert raw['resource_spec_price'][key] == state['payload']['resource_spec_price'][key], f'Native Serving resource changed: {key}'
    assert raw['resource_spec_price']['gpu_info']['gpu_type']==state['payload']['resource_spec_price']['gpu_type']


def instance_signature(instances):
    rows=[]
    for instance in instances:
        raw = instance.raw
        assert raw.get('started_at') and raw.get('created_at'), 'Missing platform incarnation evidence'
        rows.append({k:raw.get(k) for k in ('name','node','created_at','started_at','restarts','term')})
    return sorted(rows, key=lambda x:x['name'])


def attest(state, instances):
    """No stale shared-file success can qualify a new container incarnation."""
    if len(instances) != state['expected_nodes'] or any(x.status!='INSTANCE_RUNNING' for x in instances):
        raise ValueError('Native instances are not all running')
    platform = instance_signature(instances)
    if state['runtime_layout']=='prefixed_groups':
        paths = sorted(Path(state['runtime_root']).glob(state['deployment_id']+'-*'),
                       key=lambda x:(x/'node0.json').stat().st_mtime if (x/'node0.json').exists() else 0,reverse=True)
        paths = [x for x in paths if (x/'node0.json').exists() and time.time()-load(x/'node0.json')['time']<20]
        assert len(paths)==1, 'Expected one fresh replica runtime directory'
        runtime=paths[0]
    else:
        runtime=Path(state['runtime_dir'])
    reports=[]; beats=[]
    for node in range(state['expected_nodes']):
        beat=load(runtime/f'node{node}.json'); report=load(runtime/f'versions-node{node}.json')
        assert 0 <= time.time()-beat['time'] < 20, 'Stale runtime heartbeat'
        assert beat['deployment_id']==report['deployment_id']==runtime.name
        matching=[x for x in instances if x.rank==node]
        assert len(matching)==1
        actual=matching[0].raw
        # Multi-node native Serving uses host networking/hostname; single-node
        # Gemma uses its pod hostname. Both must match this exact platform rank.
        assert beat['hostname'] in (actual['name'].split('/')[-1],actual['node']), 'Heartbeat is not current platform rank/node'
        assert all(report.get(k)==state[k] for k in ('model','revision','processor_revision','precision'))
        assert all(isinstance(report.get(k),str) and report[k] for k in PACKAGES)
        if beat.get('boot_id'):
            assert report.get('boot_id')==beat['boot_id']
        else:
            raise ValueError('Runtime boot UUID missing')
        if state['role']=='verifier':
            assert report.get('local_replicas')==1 and report.get('tensor_parallel')==4
        reports.append(report);beats.append(beat)
    if state['role']=='generator':
        for rank in range(state['cards']):
            hccl=load(runtime/f'hccl-rank-{rank:02}.json')
            assert hccl.get('correctness_passed') and hccl.get('world_size')==state['cards']
            assert hccl.get('boot_id')==beats[rank//16]['boot_id'], 'HCCL receipt belongs to an old boot'
    # Runtime cache identity contains actual versions, not transient pod IDs.
    version_identity=[{k:r[k] for k in (*PACKAGES,'model','revision','processor_revision','precision','processor_options','runtime_config','serving_code_revision','communication') if k in r} for r in reports]
    incarnation=digest({'platform':platform,'boots':[b.get('boot_id') for b in beats], 'runtime':str(runtime)})
    if not all(r.get('serving_code_revision') for r in reports):
        # A currently observed source hash is not proof of loaded code. Scope
        # successful cache entries to this boot until startup records the hash.
        version_identity.append({'legacy_boot_scope':incarnation})
    return {'incarnation':incarnation,'runtime_dir':str(runtime), 'platform_instances':platform,
            'runtime_revision':'runtime-versions-sha256:'+digest({'reports':version_identity,'request_overrides':state['request_overrides']}),
            'expected_boots':[b['boot_id'] for b in beats if b.get('boot_id')],
            'reports':reports}


def health(state, attestation):
    token=Path(state['api_key_file']).read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req=urllib.request.Request(state['endpoint'].rstrip('/')+'/health',headers={'Authorization':'Bearer '+token})
    try:
        with opener.open(req,timeout=4) as response:
            value=json.load(response)
        if state['role']=='generator':
            return value.get('model_ready') is True and value.get('boot_id') in attestation['expected_boots']
        return value.get('boot_id') in attestation['expected_boots'] and bool(value.get('model_ready') or value.get('healthy_replicas')==1 or value.get('ready') is True)
    except Exception:
        return False


def probe(state, attestation):
    incarnation=attestation['incarnation']
    base=state['endpoint'].rstrip('/')+'/v1'
    attempt=state.get('probe_attempt',1)
    out=RUNTIME/'probes'/state['name']/incarnation/f'attempt{attempt:02}'
    result=out/'result.json'; receipt=out/'process.json'
    report=load(result) if result.exists() else {}
    if report.get('finished_at'):
        boot_bound=report.get('expected_boots')==attestation['expected_boots'] and all(x.get('boot_id') in attestation['expected_boots'] for x in report.get('requests',[])) and report.get('request_overrides')==state['request_overrides']
        if report.get('passed') and report.get('base_url')==base and report.get('revision')==state['revision'] and boot_bound:
            state['qualification_receipt']=str(result)
            state['probe_status']='PASSED'
            return True
        if report.get('passed') and not boot_bound:
            state['probe_attempt']=attempt+1
            state['probe_status']='REQUALIFY_BOOT_BINDING'
            return False
        transient=report.get('error_type') in ('ReadTimeout','ConnectTimeout','ConnectError','ReadError','RemoteProtocolError','ProbeProcessExited','ProbeWallClockTimeout') or report.get('error_status') in (429,500,502,503,504)
        if transient and attempt<5 and time.time()>report['finished_at']+min(300,30*2**(attempt-1)):
            state['probe_attempt']=attempt+1
        state['probe_status']='BACKOFF' if transient and attempt<5 else 'FAILED_REQUIRES_DIAGNOSIS'
        return False
    if receipt.exists():
        old=load(receipt);live=identity(old['pid']);proc=PROCESSES.get(str(out))
        same=live and all(live.get(k)==old.get(k) for k in ('pid','starttime_ticks','command'))
        expired=time.time()-old['created_at']>1800
        if same and not expired and (proc is None or proc.poll() is None):
            return False
        if same and expired:
            os.killpg(old['pid'],signal.SIGTERM)
        report.update(passed=False,finished_at=time.time(),error_type='ProbeWallClockTimeout' if expired else 'ProbeProcessExited')
        save(result,report)
        return False
    out.mkdir(parents=True,exist_ok=True)
    command=[str(PYTHON),str(ROOT.parent/'vision_probe.py'),'--base-url',base,'--model',state['model'],
             '--revision',state['revision'],'--output',str(out),'--api-key-env','INF_API_KEY','--expected-replicas','1']
    command.extend(['--request-overrides-json',json.dumps(state['request_overrides'])])
    for boot in attestation['expected_boots']:
        command.extend(['--expected-boot',boot])
    env=dict(os.environ);env['INF_API_KEY']=Path(state['api_key_file']).read_text().strip()
    for key in list(env):
        if key.lower().endswith('_proxy') and key.lower()!='no_proxy':
            env.pop(key,None)
    with (out/'stdout.log').open('a') as output:
        process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    PROCESSES[str(out)]=process
    save(receipt,{**identity(process.pid),'created_at':time.time(),'incarnation':incarnation})
    state['probe_status']='RUNNING'
    return False


def recovery_decision(state, status, events):
    if state['tier']!='low' or not state.get('auto_recovery_enabled') or status!='STOPPED':
        return None
    # Do not infer preemption from HTTP failure, queueing, FAILED, or old events.
    relevant=[]
    for event in events:
        reason=str(event.get('reason','')).lower();message=str(event.get('message','')).lower()
        if 'preempt' not in reason or not any(token in message for token in ('preempted by','was preempted','被抢占')):
            continue
        try:
            timestamp=float(event.get('last_timestamp') or event.get('created_at') or 0)/1000
        except (TypeError,ValueError):
            continue
        # Require a new victim event after the latest verified RUNNING sample,
        # associated with this service's own instance and a current stop transition.
        if not math.isfinite(timestamp) or not 0<=time.time()-timestamp<=60 or timestamp < state.get('last_running_at',float('inf')):
            continue
        if state['service_id'] not in str(event.get('object_id','')):
            continue
        if not state.get('stopped_transition_at') or abs(timestamp-state['stopped_transition_at'])>60:
            continue
        relevant.append(event)
    if not relevant:
        return None
    event=digest(relevant[-1])
    if event==state.get('restart_event') or not state.get('was_running'):
        return None
    return event


def quality_if_needed(state,path,attestation):
    """Run an isolated regression/QC worker before publishing a new precision."""
    baseline=RUNTIME/'BF16_BASELINE_READY.json'
    worker_path=ROOT/'quantization_quality.py'
    if not baseline.exists() or not worker_path.exists():
        state['probe_status']='WAIT_FROZEN_BF16_BASELINE_AND_QUALITY_WORKER'
        return False
    incarnation=attestation['incarnation']
    output=RUNTIME/f'quant-quality-{incarnation}.json'
    process_receipt=output.with_suffix('.process.json')
    expected_hash=hashlib.sha256(baseline.read_bytes()).hexdigest()
    if output.exists():
        proof=load(output)
        binding=('model','revision','processor_revision','precision','incarnation','runtime_revision')
        frozen=load(baseline)
        required=frozen.get('required_strata',[])
        required_expected=['photo_caption','chart_qa','document_ocr','rendered_qa','multi_turn_or_multi_image']
        baseline_ids={case['sample_id'] for case in frozen.get('cases',[])}
        case_ids={case.get('sample_id') for case in proof.get('cases',[])}
        valid=(proof.get('schema_version')=='corpus_pipeline.quantization_quality.v1'
               and proof.get('qualification_only') is True and proof.get('published_training') is False
               and proof.get('baseline_sha256')==expected_hash and all(proof.get(k)==state[k] for k in binding)
               and set(proof.get('expected_boots',[]))==set(state['expected_boots'])
               and required==required_expected and required==proof.get('required_strata')
               and baseline_ids and proof.get('coverage')==frozen.get('coverage')
               and all(type(proof.get('coverage',{}).get(domain)) is int and proof['coverage'][domain]>=1 for domain in required)
               and proof.get('worker_sha256')==hashlib.sha256(worker_path.read_bytes()).hexdigest()
               and proof.get('candidate_service_id')==state['service_id']
               and proof.get('all_baseline_cases_completed') is True and case_ids==baseline_ids
               and len(proof.get('cases',[]))==len(baseline_ids)
               and all(case.get('passed') is True for case in proof.get('cases',[])))
        if valid and proof.get('passed') is True:
            state['quality_receipt']=str(output);state['quality_gate_passed']=True
            return True
        if proof.get('finished_at') or proof.get('status') in ('failed','error','passed','incomplete','timeout'):
            state['probe_status']='QUANTIZED_DATA_QUALITY_FAILED_OR_UNCERTAIN'
            return False
    if process_receipt.exists():
        receipt=load(process_receipt);current=identity(receipt['pid'])
        same=current and all(current[k]==receipt[k] for k in ('pid','starttime_ticks','command'))
        if same and time.time()-receipt['created_at']<1860:
            state['probe_status']='QUANTIZED_QUALITY_RUNNING'
            return False
        if same:
            os.killpg(receipt['pid'],signal.SIGTERM)
        save(output,{'schema_version':'corpus_pipeline.quantization_quality.v1','passed':False,'status':'error',
                     'retryable':True,'finished_at':time.time(),'error':'Quality worker exited without complete receipt or exceeded wall limit',
                     'qualification_only':True,'published_training':False})
        state['probe_status']='QUANTIZED_QUALITY_WORKER_EXITED_REQUIRES_DIAGNOSIS'
        return False
    command=[str(PYTHON),'-I',str(worker_path),'--candidate-state',str(path),'--baseline-manifest',str(baseline),
             '--registry',str(REGISTRY),'--output',str(output),'--concurrency','2','--request-timeout','600','--max-duration','1800']
    save(path,state)
    env=dict(os.environ)
    env[state.get('api_key_env','INF_API_KEY')]=Path(state['api_key_file']).read_text().strip()
    for name in list(env):
        if name.lower().endswith('_proxy') and name.lower()!='no_proxy':
            env.pop(name,None)
    with output.with_suffix('.log').open('a') as log:
        process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    PROCESSES[str(output)]=process
    save(process_receipt,{**identity(process.pid),'created_at':time.time(),'incarnation':incarnation,'baseline_sha256':expected_hash})
    state['probe_status']='QUANTIZED_QUALITY_RUNNING'
    return False


def choose(candidates, allow_low=False):
    fresh=lambda x:x.get('ready') and x.get('ready_expires_at',float('inf'))>time.time()
    high=next((x for x in candidates if x['tier']=='high' and fresh(x)),None)
    low=next((x for x in candidates if x['tier']=='low' and fresh(x)),None)
    fallback_tier='low' if allow_low else 'high'
    return (low if allow_low and low else high or low or next((x for x in candidates if x['tier']==fallback_tier),candidates[0]))


def publish_all():
    # Caller holds LOCK. Publisher never writes credentials.
    endpoints=[]
    for role in ('generator','verifier'):
        choices=[v for v in CANDIDATES.values() if v['role']==role]
        if not choices:
            continue
        x=dict(choose(choices,(RUNTIME/'HIGH_ACCEPTANCE_COMPLETE').exists()))
        if x.get('ready_expires_at',0)<=time.time():
            x['ready']=False
        endpoints.append({k:x[k] for k in ('role','model','base_url','protocol','api_key_env','revision','precision','ready','supports_images','max_images','service_name','service_id','backend','tier','deployment_id','processor_revision','runtime_revision','max_output_tokens','request_overrides','incarnation','qualification_receipt','expected_boots','ready_expires_at')})
    with (REGISTRY.with_suffix('.lock')).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        old=load(REGISTRY) if REGISTRY.exists() else {'registry_generation':0,'endpoints':[]}
        semantic=lambda rows:[{k:v for k,v in row.items() if k!='ready_expires_at'} for row in rows]
        if old['endpoints']!=endpoints:
            generation=old['registry_generation']+int(semantic(old['endpoints'])!=semantic(endpoints))
            save(REGISTRY,{'schema_version':'corpus_pipeline.endpoints.v1','registry_generation':generation,
                           'endpoints':endpoints,'updated_at':time.time(),'publisher':'native_serving_monitor'})


def candidate(state, ready):
    return {'role':state['role'],'model':state['model'],'base_url':state.get('endpoint','').rstrip('/')+'/v1',
        'protocol':'openai','api_key_env':'INF_API_KEY','revision':state['revision'],'precision':state['precision'],
        'ready':ready,'supports_images':True,'max_images':state['max_images'],'service_name':state['name'],
        'service_id':state['service_id'],'backend':'inspire_serving','tier':state['tier'],
        'deployment_id':state['deployment_id'],'processor_revision':state['processor_revision'],
        'runtime_revision':state.get('runtime_revision','pending-container-version-report'),'max_output_tokens':12000,
        'request_overrides':state['request_overrides'],'incarnation':state.get('incarnation'),
        'qualification_receipt':state.get('qualification_receipt'),'expected_boots':state.get('expected_boots',[]),
        'ready_expires_at':time.time()+30 if ready else 0}


def worker(path):
    # A long-lived thread owns its own SDK client for its full lifetime.
    with InspireClient(account='wjx-ascend') as client:
        state=load(path); ref=None; events=[]; last_events=0; backoff=2
        while not (RUNTIME/'WATCHDOG_DISABLED').exists():
            started=time.time(); ready=False
            try:
                detail=client.servings.get(ref or state['name'],workspace=WORKSPACE)
                assert_owned(state,detail);ref=detail.ref
                instances=client.servings.instances(ref,workspace=WORKSPACE)
                status=detail.status
                previous=state.get('platform_status')
                state.update(platform_status=status,endpoint=detail.to_dict()['endpoint'],last_observed_at=time.time(),observation_error=None)
                if status=='RUNNING':
                    state['was_running']=True
                    state['last_running_at']=time.time()
                    state['stopped_transition_at']=None
                elif status=='STOPPED' and previous=='RUNNING':
                    state['stopped_transition_at']=time.time()
                if time.time()-last_events>10:
                    events=list(client.servings.events(ref,workspace=WORKSPACE,limit=30).items)
                    last_events=time.time()
                save(RUNTIME/f'{path.stem}-live.json',{'time':time.time(),'detail':detail.to_dict(),'instances':[vars(x) for x in instances],'events':events})
                try:
                    evidence=attest(state,instances)
                    if state.get('incarnation')!=evidence['incarnation']:
                        state.update(incarnation=evidence['incarnation'],probe_attempt=1,qualification_receipt=None,canary_passed=False)
                        with LOCK:
                            CANDIDATES[path.stem]=candidate(state,False);publish_all()
                    state['runtime_revision']=evidence['runtime_revision']
                    state['expected_boots']=evidence['expected_boots']
                    save(RUNTIME/f'{path.stem}-attestation.json',evidence)
                    if status=='RUNNING' and health(state,evidence):
                        ready=probe(state,evidence)
                        state['image_canary_passed']=ready
                        if ready and state.get('quality_gate_required'):
                            ready=quality_if_needed(state,path,evidence)
                        state['readiness_reason']='QUALIFIED' if ready else state.get('probe_status','QUALIFYING')
                    else:
                        state['readiness_reason']='MODEL_LOADING_OR_HTTP_UNAVAILABLE'
                except (AssertionError,OSError,ValueError,KeyError) as exc:
                    state['readiness_reason']=str(exc)[:400]
                event=recovery_decision(state,status,events)
                if event:
                    state['restart_event']=event
                    state['restart_intent_at']=time.time()
                    state['recovery_pending']={'victim_event_digest':event,'service_id':state['service_id'],'confirmed_at':time.time()}
                    state['capacity_samples']=[]
                    save(path,state)
                if state.get('recovery_pending'):
                    from downsize import recover_low
                    if recover_low(client,state,path,save,assert_owned):
                        ref=None
                        ready=False
                backoff=2
            except Exception as exc:
                state['observation_error']={'type':type(exc).__name__,'message':str(exc)[:800],'time':time.time()}
                backoff=min(30,backoff*2)
            state.update(canary_passed=ready,heartbeat_at=time.time(),monitor_pid=os.getpid(),thread_id=threading.get_ident())
            save(path,state)
            with LOCK:
                CANDIDATES[path.stem]=candidate(state,ready);publish_all()
            time.sleep(max(.2,backoff-(time.time()-started)))


def main():
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import_handoffs()
        paths=[RUNTIME/f'{model}-{tier}.json' for model in ('qwen','gemma') for tier in ('high','low')]
        states=[load(p) for p in paths]
        assert sum(s['cards'] for s in states)<=72
        assert len({s['service_id'] for s in states})==4
        with LOCK:
            for p,s in zip(paths,states):
                CANDIDATES[p.stem]=candidate(s,False)
            publish_all()
        threads=[threading.Thread(target=worker,args=(p,),name=p.stem,daemon=True) for p in paths]
        for thread in threads:
            thread.start()
        while not (RUNTIME/'WATCHDOG_DISABLED').exists():
            if not all(t.is_alive() for t in threads):
                raise RuntimeError('Serving worker died; watchdog must restart monitor')
            save(RUNTIME/'monitor-heartbeat.json',{**identity(os.getpid()),'time':time.time(),'workers':[t.name for t in threads],'maximum_owned_cards':72,'loaded_monitor_code_revision':MONITOR_CODE_REVISION})
            time.sleep(2)


if __name__=='__main__':
    main()
