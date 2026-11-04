"""Single-copy, owned GPUJob recovery; transport failures never imply preemption.

Run with the inspire-skill Python. Model/API probes run in the project venv.
Mutations are restricted to exact job IDs recorded in this task's state files.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
SKILL = Path('/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/.codex/skills/inspire-wjx-ascend/scripts')
sys.path.insert(0, str(SKILL))
from fast_job_control import API, TERMINAL, QUEUED, locked, save, state_of, placement, preemption_kind, validate_payload as strict_validate_payload
from node_policy import apply_fixed_exclusions
from watchdog import process_identity

ACCOUNT = 'wjx-ascend'
WORKSPACE = '昇腾卡公共空间'
GROUP = '910B资源'
CLIENT_PYTHON = ROOT.parents[1] / '.venv/bin/python'
REGISTRY = ROOT / 'endpoints.json'
PROBES = {}
APIS = {}
REGISTRY_THREAD_LOCK = threading.Lock()
OBSERVATIONS = {}


def log(kind, **values):
    print(json.dumps({'time': time.time(), 'kind': kind, **values}, ensure_ascii=False), flush=True)


def load(path):
    return json.loads(Path(path).read_text())


def api_for_current_thread():
    identity = threading.get_ident()
    if identity not in APIS:
        APIS[identity] = API(ACCOUNT, WORKSPACE, GROUP, node_cache=ROOT/'runtime/shared-nodes.json', cache_interval=2)
    return APIS[identity]


def publish(role, endpoint):
    with REGISTRY_THREAD_LOCK, locked(REGISTRY):
        value = load(REGISTRY) if REGISTRY.exists() else {'schema_version': 'corpus_pipeline.endpoints.v1', 'registry_generation': 0, 'endpoints': []}
        before = value['endpoints']
        after = sorted([x for x in before if x['role'] != role] + [endpoint], key=lambda x: x['role'])
        if after != before:
            value.update(endpoints=after, registry_generation=value['registry_generation'] + 1, updated_at=time.time())
            save(REGISTRY, value)


def decision(status, seen_running, confirmed_preemption, http_ready=None):
    # http_ready deliberately does not select any mutation.
    if status == 'RUNNING':
        return 'observe'
    if confirmed_preemption and (status in TERMINAL or (seen_running and status in QUEUED)):
        return 'recover'
    if status in TERMINAL:
        return 'diagnose'
    return 'observe'


def validate_payload(payload, row):
    # Live GetJob 2026-10-22 normalizes image_type and drops envs; retain the
    # exact original payload while checking the actual immutable image URL.
    # New generations encode the nonsensitive deployment ID in command.
    normalized = copy.deepcopy(row)
    expected = payload.get('framework_config', [])
    actual = normalized.get('framework_config', [])
    if len(expected) == len(actual) == 1 and expected[0].get('image') == actual[0].get('image'):
        if expected[0].get('image_type') == 'SOURCE_PRIVATE' and actual[0].get('image_type') == 'SOURCE_PUBLIC':
            actual[0]['image_type'] = 'SOURCE_PRIVATE'
    if not normalized.get('envs') and payload.get('envs'):
        if all(x.get('name') == 'CORPUS_DEPLOYMENT_ID' for x in payload['envs']):
            normalized['envs'] = copy.deepcopy(payload['envs'])
    strict_validate_payload(payload, normalized)


def assert_owned(state, rows):
    if state['job_id'] not in state['owned_jobs']:
        raise ValueError('Active job is outside this task ownership set')
    row = rows[state['job_id']]
    validate_payload(state['payload'], row)
    if str(row.get('priority_level') or '').upper() not in ('LOW', ''):
        raise ValueError('Refusing to control a non-LOW job')
    other_active = [key for key, value in rows.items() if key != state['job_id'] and state_of(value) not in TERMINAL]
    if other_active:
        raise ValueError('Another owned generation is still active; no overlapping create')


def fresh_gateway(state):
    path = Path(state['heartbeat_root']) / state['deployment_id'] / 'node0.json'
    try:
        beat = load(path)
        if beat['deployment_id'] != state['deployment_id'] or not 0 <= time.time()-beat['time'] < 20:
            return None
        ip = beat['ip']
        import ipaddress
        if not ipaddress.ip_address(ip).is_private:
            raise ValueError('Unexpected nonprivate backend address')
        return f'http://{ip}:8000/v1'
    except (OSError, ValueError, KeyError):
        return None


def health(base_url):
    if not base_url:
        return False
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base_url.removesuffix('/v1') + '/health', timeout=3) as response:
            return response.status == 200 and json.load(response).get('model_ready') is True
    except Exception:
        return False


def endpoint(state, url, ready):
    return {'role': state['role'], 'model': state['model'], 'base_url': url or '',
            'protocol': 'openai', 'api_key_env': 'CORPUS_INFERENCE_API_KEY',
            'revision': state['revision'], 'precision': state['precision'],
            'ready': ready, 'supports_images': True, 'max_images': state['max_images'],
            'service_name': state['payload']['name'], 'backend': 'inspire_gpu_job',
            'deployment_id': state['deployment_id'], 'job_id': state['job_id'],
            'processor_revision': state.get('processor_revision', state['revision']),
            'runtime_revision': state.get('runtime_revision', 'pending-container-version-report'),
            'max_output_tokens': 12000,
            'request_overrides': state.get('request_overrides', {})}


def attest_runtime(state):
    reports = []
    count = sum(int(x['instance_count']) for x in state['payload']['framework_config'])
    for node in range(count):
        path = Path(state['heartbeat_root']) / state['deployment_id'] / f'versions-node{node}.json'
        try:
            report = load(path)
        except (OSError, ValueError):
            return False
        if any(report.get(key) != state[key] for key in ('deployment_id','model','precision','revision','processor_revision')):
            return False
        if not all(isinstance(report.get(key), str) and report[key] for key in ('vllm','vllm-ascend','torch','torch-npu','transformers')):
            return False
        if any(report.get(key) != state[key] for key in ('local_replicas','tensor_parallel') if key in state):
            return False
        reports.append(report)
    state['runtime_revision'] = 'runtime-versions-sha256:' + hashlib.sha256(json.dumps(reports, sort_keys=True).encode()).hexdigest()
    return True


def inspect_probe_process(receipt, process=None):
    live = process_identity(receipt.get('pid', -1))
    same = bool(live and live['starttime_ticks'] == receipt.get('starttime_ticks') and
                live['command'] == [x.encode() for x in receipt['command']])
    if (process is not None and process.poll() is not None) or not same:
        return 'ProbeProcessExited', False
    if time.time()-receipt['created_at'] > 1800:
        return 'ProbeWallClockTimeout', True
    return None, True


def probe_if_needed(state, url):
    attempt = state.get('probe_attempt', 1)
    key = state['deployment_id'] + f'-attempt{attempt:02}'
    output = ROOT / 'runtime' / 'probes' / key
    result = output / 'result.json'
    if result.exists():
        report = load(result)
        if report.get('passed') and report.get('base_url') == url and report.get('revision') == state['revision']:
            return True
        if report.get('finished_at'):
            transient = report.get('error_type') in ('ReadTimeout','ConnectTimeout','ConnectError','ReadError','RemoteProtocolError','ProbeProcessExited','ProbeWallClockTimeout') or report.get('error_status') in (429,500,502,503,504)
            if transient and attempt < 5:
                retry_at = report['finished_at'] + min(300, 30 * 2**(attempt-1))
                state['probe_status'] = 'TRANSPORT_RETRY_BACKOFF_NO_WORKLOAD_RESTART'
                state['probe_retry_after'] = retry_at
                if time.time() >= retry_at:
                    state['probe_attempt'] = attempt + 1
                return False
            state['probe_status'] = 'FAILED_REQUIRES_DIAGNOSIS'
            return False
    # A controller restart may leave an earlier probe running. Verify its identity,
    # detect dead children even when they left an unfinished result, and bound runtime.
    receipt = output / 'process.json'
    if receipt.exists():
        old = load(receipt)
        failure, still_same = inspect_probe_process(old, PROBES.get(key))
        if failure is None:
            return False
        if still_same and failure == 'ProbeWallClockTimeout':
            try:
                os.killpg(old['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
        report = load(result) if result.exists() else {}
        report.update(passed=False, finished_at=time.time(), error_type=failure,
                      error='Probe subprocess ended without a final receipt or exceeded 1800 seconds; workload unchanged')
        save(result, report)
        log('probe_process_failed_no_workload_restart', deployment_id=key, error_type=failure)
        return False
    if key not in PROBES:
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(name, None)
        env['CORPUS_INFERENCE_API_KEY'] = (ROOT / 'private/client.key').read_text().strip()
        command = [str(CLIENT_PYTHON), str(ROOT / 'vision_probe.py'), '--base-url', url,
                   '--model', state['model'], '--revision', state['revision'], '--output', str(output),
                   '--expected-replicas', str(state.get('local_replicas', 1))]
        with (output / 'probe.log').open('a') as stream:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        PROBES[key] = process
        identity = process_identity(process.pid)
        save(receipt, {'pid': process.pid, 'starttime_ticks': identity['starttime_ticks'] if identity else None,
                       'command': command, 'created_at': time.time()})
        log('probe_started', deployment_id=key, pid=process.pid)
    return False


def reconcile_create(api, state, path):
    rows = api.named_records(state['payload']['name'])
    if len(rows) != 1:
        state['phase'] = 'CREATE_UNCERTAIN_RECONCILE_ONLY'
        save(path, state)
        return False
    row = rows[0]
    validate_payload(state['payload'], row)
    state['job_id'] = row['job_id']
    if state['job_id'] not in state['owned_jobs']:
        state['owned_jobs'].append(state['job_id'])
    state['phase'] = 'OBSERVING'
    save(path, state)
    return True


def profiles_for(state):
    profiles = load(ROOT / 'profiles.json').get(state['role'], [])
    if profiles:
        return profiles
    return [{'profile_id': state.get('profile_id', state['role']), 'cards':state['budget_cards'],
             'payload_path':state['original_payload_path'], **{key:state[key] for key in
             ('launch_command','deployment_prefix','name_prefix','heartbeat_root','model','revision','precision','max_images')}}]


def import_gemma_handoff():
    """Adopt only the one explicitly prepared Gemma Job; never creates a workload."""
    target = ROOT / 'runtime/gemma-state.json'
    handoff = ROOT / 'gemma4/job_handoff.json'
    if target.exists() or not handoff.exists():
        return
    info = load(handoff)
    if info.get('role') != 'verifier' or info.get('backend') != 'inspire_job' or info.get('budget_cards') != 16:
        raise ValueError('Unexpected Gemma handoff scope')
    payload = load(info['original_payload_path'])
    qwen = load(ROOT / 'runtime/qwen-state.json')
    if any(payload.get(key) != qwen['payload'].get(key) for key in ('workspace_id','project_id','logic_compute_group_id')):
        raise ValueError('Gemma project/workspace mismatch')
    if payload['task_priority'] != 3 or sum(x['gpu_count']*x['instance_count'] for x in payload['framework_config']) != 16:
        raise ValueError('Gemma LOW/16-card contract mismatch')
    api = API(ACCOUNT, WORKSPACE, GROUP)
    try:
        row = api.records([info['job_id']])[info['job_id']]
        validate_payload(payload, row)
    finally:
        api.close()
    original = ROOT / 'runtime/gemma-submitted-payload-r01.json'
    save(original, payload)
    state = {**info, 'schema_version':'corpus_pipeline.qwen_supervisor.v1', 'payload':payload,
             'owned_jobs':[info['job_id']], 'phase':'OBSERVING', 'disabled':False,
             'profile_id':'gemma_bf16_16_tp4x4', 'original_payload_path':str(original),
             'canary_passed':False, 'allow_initial_capacity_qualification':False}
    profiles = load(ROOT / 'profiles.json')
    profiles['verifier'] = [{'profile_id':state['profile_id'],'cards':16,'payload_path':str(original),
         'qualification_required':True, 'local_replicas':4,
         'minimum_live_job_quota_cards':16,
         **{key:state[key] for key in ('launch_command','deployment_prefix','name_prefix','heartbeat_root',
             'model','revision','precision','max_images','processor_revision')}}]
    save(ROOT / 'profiles.json', profiles)
    save(target, state)
    log('adopted_owned_gemma_job', job_id=info['job_id'], cards=16)


def choose_profile(state, nodes, requested=None):
    """Prefer the largest authorized profile that fits complete live idle nodes.

    A smaller precision profile is deployed for qualification, never marked ready
    merely because it fits. Its new model/revision must pass its own image probe.
    """
    for profile in profiles_for(state):
        if requested and profile['profile_id'] != requested:
            continue
        if profile['cards'] > state['budget_cards']:
            continue
        payload = apply_fixed_exclusions(load(profile['payload_path']), ACCOUNT)
        candidate = placement(payload, nodes)
        if candidate is not None:
            chosen, prepared = candidate
            return profile, chosen, prepared
    return None


def consider_initial_qualification(api, state, instances):
    if not state.get('allow_initial_capacity_qualification') or state.get('seen_running'):
        return False
    if state['platform_status'] not in QUEUED or any(x.get('node') for x in instances):
        return False
    if time.time()-state.get('last_capacity_check', 0) < 15:
        return False
    state['last_capacity_check'] = time.time()
    candidate = choose_profile(state, api.nodes())
    if candidate is None or candidate[0]['profile_id'] == state['profile_id']:
        state.pop('capacity_candidate', None)
        return False
    profile = candidate[0]
    previous = state.get('capacity_candidate', {})
    if previous.get('profile_id') != profile['profile_id']:
        state['capacity_candidate'] = {'profile_id':profile['profile_id'], 'first_observed':time.time()}
        return False
    if time.time()-previous['first_observed'] < 15:
        return False
    state['planned_transition'] = {'reason':'two_live_capacity_snapshots_before_first_run',
         'profile_id':profile['profile_id'], 'qualification_required':True}
    return True


def recover(api, state, path, rows):
    current = rows[state['job_id']]
    status = state_of(current)
    nodes = api.nodes()
    # The complement of fully idle nodes excludes every currently occupied node,
    # including high-priority occupants even where their priority isn't exposed.
    requested = (state.get('planned_transition') or {}).get('profile_id')
    candidate = choose_profile(state, nodes, requested)
    if candidate is None:
        state['phase'] = 'WAITING_IDLE_NODES_AFTER_PREEMPTION'
        state['fallback_status'] = 'NO_COMPLETE_IDLE_NODE_FOR_AUTHORIZED_PROFILES'
        return
    profile, chosen, payload = candidate
    if status not in TERMINAL:
        # Re-read immediately before mutation; observed RUNNING always wins.
        current = api.records([state['job_id']])[state['job_id']]
        if state_of(current) not in QUEUED:
            return
        validate_payload(state['payload'], current)
        if not state.get('confirmed_preemption'):
            if not state.get('planned_transition') or state.get('seen_running'):
                return
            if any(x.get('node') for x in api.instances(state['job_id'])):
                return
        state['phase'] = 'STOP_INTENT_CONFIRMED_TRANSITION'
        save(path, state)
        api.stop(state['job_id'])
        log('stop_own_queue_for_confirmed_transition', job_id=state['job_id'], profile=profile['profile_id'])
        return
    # All generations must be confirmed terminal before allocating another one.
    if any(state_of(row) not in TERMINAL for row in rows.values()):
        return
    generation = state['generation'] + 1
    deployment = profile['deployment_prefix'] + f'-r{generation:02}'
    payload['name'] = profile['name_prefix'] + f'-r{generation:02}'
    payload['envs'] = []
    payload['command'] = 'bash -c ' + shlex.quote('env CORPUS_DEPLOYMENT_ID=' + shlex.quote(deployment) + ' ' + profile['launch_command'])
    cards = sum(int(x['gpu_count'])*int(x['instance_count']) for x in payload['framework_config'])
    if cards > state['budget_cards'] or payload['task_priority'] != 3:
        raise ValueError('Budget/LOW contract violated')
    if api.named_records(payload['name']):
        raise ValueError('Next generation name already exists; reconcile required')
    usage = api.client.resources.usage(WORKSPACE, project='多模态大模型新架构评测探索与scaling-law', group=GROUP, details=True).to_dict()
    if usage.get('truncated'):
        raise ValueError('Incomplete project allocation observation')
    states = [load(p) for p in (ROOT/'runtime').glob('*-state.json')]
    controlled = [x for x in states if x.get('schema_version') == 'corpus_pipeline.qwen_supervisor.v1']
    names = {x['payload']['name'] for x in controlled}
    outside_allocated = sum(float(x['gpus']) for x in usage['items'] if x['task'] not in names)
    own_ceiling = sum(x['budget_cards'] for x in controlled)
    if own_ceiling > 48 or outside_allocated + own_ceiling > 256:
        state['phase'] = 'WAITING_PROJECT_256_CARD_CEILING'
        return
    # The quota lookup can be slower than a placement snapshot; refresh placement.
    refreshed = choose_profile(state, api.nodes(), profile['profile_id'])
    if refreshed is None:
        state['phase'] = 'WAITING_IDLE_NODES_AFTER_PREEMPTION'
        return
    chosen = refreshed[1]
    payload['exclude_nodes'] = refreshed[2]['exclude_nodes']
    state.update(generation=generation, deployment_id=deployment, payload=payload,
                 phase='CREATE_INTENT', operation_id='corpus-recover-'+uuid.uuid4().hex,
                 seen_running=False, confirmed_preemption=False, canary_passed=False)
    for key in ('profile_id','launch_command','deployment_prefix','name_prefix','heartbeat_root',
                'model','revision','precision','max_images','processor_revision'):
        state[key] = profile[key]
    state.pop('planned_transition', None)
    state.pop('capacity_candidate', None)
    state.pop('probe_status', None)
    state.pop('probe_attempt', None)
    state.pop('probe_retry_after', None)
    state.pop('runtime_revision', None)
    state.pop('runtime_attested', None)
    save(ROOT / 'runtime' / f"{state['role']}-payload-r{generation:02}.json", payload)
    save(path, state)
    job_id = api.create(payload, state['operation_id'])
    state['job_id'] = job_id
    state['owned_jobs'].append(job_id)
    state['phase'] = 'OBSERVING'
    save(path, state)
    log('created_single_successor', job_id=job_id, nodes=chosen, cards=cards)


def tick(api, path):
    state = load(path)
    # Provenance is read from the prepared profile, and must match runtime reports.
    for profile in profiles_for(state):
        if profile['profile_id'] == state.get('profile_id') and profile.get('processor_revision'):
            state['processor_revision'] = profile['processor_revision']
    if state.get('disabled'):
        publish(state['role'], endpoint(state, '', False))
        return
    if state['phase'].startswith('CREATE_') and not reconcile_create(api, state, path):
        return
    rows = api.records(state['owned_jobs'])
    assert_owned(state, rows)
    row = rows[state['job_id']]
    status = state_of(row)
    state.update(platform_status=status, last_platform_observed_at=time.time())
    if status == 'RUNNING':
        state['seen_running'] = True
    instances = api.instances(state['job_id'])
    events = []
    if time.time()-state.get('last_events_check',0) >= 5 or status != state.get('previous_platform_status'):
        events = api.events('job', [state['job_id']])
        ids = [x.get('name') for x in instances]
        if all(ids) and ids:
            events += api.events('instance', ids)
        state['last_events_check'] = time.time()
    state['previous_platform_status'] = status
    observed = [x for x in events if preemption_kind(x) == 'preemption_observed']
    if observed:
        state['confirmed_preemption'] = True
        save(ROOT / 'runtime' / f"preemption-{state['deployment_id']}.json", observed)
    url = fresh_gateway(state) if status == 'RUNNING' else None
    state['runtime_attested'] = attest_runtime(state) if status == 'RUNNING' else False
    healthy = health(url)
    ready = healthy and state['runtime_attested'] and probe_if_needed(state, url)
    state['canary_passed'] = bool(ready)
    state['endpoint_health'] = 'READY' if ready else ('PROBING' if healthy else 'UNAVAILABLE')
    publish(state['role'], endpoint(state, url, bool(ready)))
    action = decision(status, state.get('seen_running', False), state.get('confirmed_preemption', False), healthy)
    if status != 'RUNNING' and (state.get('planned_transition') or consider_initial_qualification(api, state, instances)):
        action = 'recover'
    if action == 'recover':
        recover(api, state, path, rows)
    elif action == 'diagnose':
        state['phase'] = 'TERMINAL_REQUIRES_DIAGNOSIS'
    else:
        state['phase'] = 'OBSERVING'
    save(path, state)
    save(ROOT / 'runtime' / f"{state['role']}-observation.json", {'time': time.time(), 'job': row, 'instances': instances, 'events': events})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    with locked(ROOT / 'runtime/supervisor'):
        def worker(path):
            try:
                tick(api_for_current_thread(), path)
                OBSERVATIONS[path.name] = {'last_success_at':time.time(),'last_error_type':None,'healthy':True}
            except Exception as exc:
                # Unknown observations cannot mutate workloads or infer termination.
                log('observation_error_no_mutation', state=path.name, error_type=type(exc).__name__, error=str(exc)[:500])
                OBSERVATIONS[path.name] = {**OBSERVATIONS.get(path.name,{}), 'last_failure_at':time.time(),
                                          'last_error_type':type(exc).__name__,'healthy':False}
                state = load(path)
                if state.get('job_id'):
                    publish(state['role'], endpoint(state, '', False))
                # Limit API outage / throttling pressure without blocking the other role.
                time.sleep(min(30, max(5, args.interval*3)))
        with ThreadPoolExecutor(max_workers=2) as executor:
            active = {}
            while True:
                try:
                    import_gemma_handoff()
                except Exception as exc:
                    log('gemma_handoff_refused', error_type=type(exc).__name__, error=str(exc)[:500])
                for path in sorted((ROOT / 'runtime').glob('*-state.json')):
                    if load(path).get('schema_version') != 'corpus_pipeline.qwen_supervisor.v1':
                        continue
                    if path not in active or active[path].done():
                        active[path] = executor.submit(worker, path)
                save(ROOT / 'runtime/supervisor-heartbeat.json', {'pid': os.getpid(), 'time': time.time(),
                     'interval': args.interval, 'node_exclusion_policy': 'fixed exclusions + all occupied nodes on recovery',
                     'http_timeout_triggers_restart': False, 'maximum_owned_cards': 48,
                     'observations':{name:{**value,'healthy':value['healthy'] and time.time()-value.get('last_success_at',0)<15}
                                     for name,value in list(OBSERVATIONS.items())},
                     'role_workers_inflight':sum(not f.done() for f in active.values())})
                if args.once:
                    for future in active.values():
                        future.result()
                    break
                time.sleep(args.interval)


if __name__ == '__main__':
    main()
