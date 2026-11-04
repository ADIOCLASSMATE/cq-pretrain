"""Single-allocation HIGH native Serving laboratory; never restarts resources."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import threading
import time

from inspire.sdk import InspireClient
from inspire.sdk.models_serving import ServingCreateSpec
from corpus_pipeline.serving import ai_review_pool as pool

native = pool.native
observable_project_budget = pool.observable_project_budget
ROOT = Path(__file__).resolve().parent
LOW = ROOT.parent / 'ai_review_20261024'
PROFILE_LIMITS = {'s32':(32,4096), 's64':(64,4096), 's128':(128,4096),
                  's64t2048':(64,2048), 's128t2048':(128,2048),
                  's64t8192':(64,8192), 's128t8192':(128,8192),
                  's32k':(32,4096), 's64k':(64,4096), 's128k':(128,4096),
                  's64kh':(64,4096), 's128kh':(128,4096),
                  's64kt8192':(64,8192), 's128kt8192':(128,8192),
                  's64km':(64,4096), 's128km':(128,4096)}
PROFILES = tuple(PROFILE_LIMITS)
CARDS = 32
MAX_COMBINED_CARDS = 128
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TERMINAL_INSTANCES = {'INSTANCE_STOPPED', 'INSTANCE_FAILED', 'INSTANCE_SUCCEEDED',
                      'INSTANCE_DELETED', 'INSTANCE_TERMINATED', 'INSTANCE_COMPLETED'}
RUNTIME_FILES = ('serve.py', 'gateway.py', 'launch.sh', 'hccl_probe.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paths(profile):
    require(profile in PROFILES, 'Unsupported lab profile')
    return ROOT / 'states' / (profile + '.json'), ROOT / 'intents' / (profile + '.json'), ROOT / 'observations' / profile


@contextmanager
def locked(path, *, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def persist(path, state):
    # Transport addresses only enter the private endpoint registry, not state/status.
    native.save(path, {k: v for k, v in state.items() if k != 'endpoint'})


def summary(state):
    return {k: state.get(k) for k in ('profile', 'name', 'service_id', 'phase', 'cards', 'priority',
        'platform_status', 'canary_passed', 'release_verified', 'readiness_reason', 'observation_error', 'heartbeat_at')}


def binding(profile):
    require(profile in PROFILE_LIMITS, 'Unsupported lab profile')
    sequences, tokens = PROFILE_LIMITS[profile]
    path = ROOT / 'profiles' / (profile + '.json')
    config = native.load(path)
    baseline = native.load(LOW / 'qwen-01.json')
    manifest_path = ROOT / 'RUNTIME_MANIFEST.json'
    manifest = native.load(manifest_path)
    files = manifest.get('files', {})
    require(set('runtime/' + name for name in RUNTIME_FILES) <= set(files), 'Runtime manifest is incomplete')
    require('profiles/' + profile + '.json' in files, 'Profile absent from runtime manifest')
    for relative, expected in files.items():
        current = (ROOT / relative).resolve()
        require(current.is_relative_to(ROOT.resolve()) and file_sha(current) == expected, 'Manifest file changed')
    require(config['model_name'] == baseline['model'] == 'Qwen3.5-397B-A17B', 'Wrong model')
    require(config['precision'] == baseline['precision'] == 'BF16', 'Precision change forbidden')
    require(config['revision'] == baseline['revision'] and config['processor_revision'] == baseline['processor_revision'],
            'Model or processor revision changed')
    require(config['nodes'] == 2 and config['data_parallel_size'] == 4, 'Topology change forbidden')
    require(config['epoch'] == 'infra-20261024' and config['max_num_seqs'] == sequences, 'Profile mismatch')
    require(config['max_model_len'] == 65536 and config['max_images'] == 16, 'Input contract changed')
    require(config['max_num_batched_tokens'] == tokens, 'Unexpected token batch configuration')
    require(type(config.get('enable_json_keepalive', False)) is bool
            and config.get('enable_json_keepalive', False) == ('k' in profile), 'Unexpected transport profile')
    require(type(config.get('enable_mc2_hierarchy_comm', False)) is bool
            and config.get('enable_mc2_hierarchy_comm', False) == profile.endswith('h'), 'Unexpected MoE profile')
    require(type(config.get('num_speculative_tokens', 0)) is int
            and config.get('num_speculative_tokens', 0) == int(profile.endswith('m')), 'Unexpected MTP profile')
    require(config['cudagraph_capture_sizes'] == [(1+config.get('num_speculative_tokens',0))*2**n
            for n in range(sequences.bit_length())],
            'Unexpected graph capture sizes')
    runtime_hashes = {name: files['runtime/' + name] for name in RUNTIME_FILES}
    return config, baseline, {'profile_sha256': file_sha(path), 'runtime_manifest_sha256': file_sha(manifest_path),
        'runtime_files': runtime_hashes, 'lab_code_sha256': CODE_SHA256,
        'native_monitor_sha256': file_sha(Path(native.__file__)),
        'serving_code_revision': 'sha256:' + hashlib.sha256(json.dumps(runtime_hashes, sort_keys=True).encode()).hexdigest()}


def validate_state(state):
    require(state.get('profile') in PROFILES, 'Unknown managed profile')
    require(state.get('scope') == str(ROOT.resolve()), 'State belongs to another lab')
    require(state.get('tier') == 'high' and state.get('priority') == 6 and state.get('cards') == CARDS,
            'Unqualified HIGH allocation')
    require(state.get('expected_nodes') == 2 and state.get('expected_replicas') == 1
            and state.get('precision') == 'BF16' and state.get('auto_recovery_enabled') is False,
            'Unqualified topology or recovery policy')
    expected = 'corpus-qwen397b-infra-high-' + state['profile'] + '-20261024-' + state['binding']['profile_sha256'][:8]
    require(state['name'] == state['service_name'] == state['operation_id'] == expected, 'Managed name changed')


def load_state(profile):
    path, intent, _ = paths(profile)
    state = native.load(path if path.exists() else intent)
    validate_state(state)
    if intent.exists():
        original = native.load(intent)
        require(all(state.get(k) == original.get(k) for k in
                    ('scope', 'profile', 'name', 'operation_id', 'payload', 'binding', 'deployment_id')), 'Intent identity changed')
    return state


def reconcile(client, profile, state):
    if state.get('service_id'):
        detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
        native.assert_owned(state, detail)
        return state
    matches = [row for row in client.servings.iter(workspace=native.WORKSPACE) if row.name == state['name']]
    require(len(matches) == 1, 'Create intent unresolved; no retransmission')
    detail = client.servings.get(matches[0].ref, workspace=native.WORKSPACE)
    candidate = {**state, 'service_id': detail.ref.key, 'phase': 'OBSERVING'}
    native.assert_owned(candidate, detail)
    persist(paths(profile)[0], candidate)
    return candidate


def stopped(client, state):
    detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
    native.assert_owned(state, detail)
    instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
    statuses = [instance.status for instance in instances]
    return detail, statuses, detail.status == 'STOPPED' and set(statuses) <= TERMINAL_INSTANCES


def low_released(client):
    require((LOW / 'DISABLED').exists(), 'LOW pool must first be disabled')
    receipt = native.load(LOW / 'STOPPED.json')
    require(receipt.get('complete') is True, 'LOW release is not verified')
    rows = [native.load(path) for path in sorted(LOW.glob('qwen-[0-9][0-9].json'))]
    pool.validate_budget(rows)
    require(rows and all(row.get('service_id') for row in rows), 'LOW has unknown or missing reservations')
    require({row['service_id'] for row in rows} == {row['service_id'] for row in receipt['services']},
            'LOW release receipt coverage changed')
    for row in rows:
        require(stopped(client, row)[2], 'LOW allocation is not fully stopped now')
    return {'low_states': len(rows), 'low_reserved_cards': 0, 'release_receipt_sha256': file_sha(LOW / 'STOPPED.json')}


def lab_reserved(client):
    profiles = {p.stem for directory in ('states', 'intents') for p in (ROOT / directory).glob('*.json')}
    rows = [load_state(profile) for profile in sorted(profiles)]
    require(len({r['name'] for r in rows}) == len(rows), 'Duplicate lab name')
    ids = [r['service_id'] for r in rows if r.get('service_id')]
    require(len(ids) == len(set(ids)), 'Duplicate lab service ID')
    reserved = 0
    for row in rows:
        released = False
        if row.get('release_verified') is True and row.get('service_id'):
            receipt_path = paths(row['profile'])[2] / 'STOPPED.json'
            receipt = native.load(receipt_path)
            require(file_sha(receipt_path) == row.get('release_receipt_sha256') and receipt.get('complete') is True
                    and receipt.get('service_id') == row['service_id'] and receipt.get('binding') == row['binding'],
                    'Lab release receipt is missing or changed')
            released = stopped(client, row)[2]
        if not released:
            reserved += CARDS
    return reserved


def create(profile):
    path, intent, observation = paths(profile)
    with locked(ROOT / 'lifecycle.lock'):
        require(not (ROOT / 'PRODUCTION_HANDOFF.json').exists(), 'HIGH lab budget handed off to production')
        config, baseline, bound = binding(profile)
        with InspireClient(account='wjx-ascend') as client:
            if path.exists() or intent.exists():
                state = load_state(profile)
                require(state['binding'] == bound, 'Existing profile/code binding changed')
                return summary(reconcile(client, profile, state))
            require(not (observation / 'DISABLED').exists(), 'Profile stop requested')
            with locked(LOW / 'provision.lock'), locked(LOW / 'lifecycle.lock'):
                low = low_released(client)
                reserved = lab_reserved(client)
                require(reserved == 0 and reserved + CARDS + low['low_reserved_cards'] <= MAX_COMBINED_CARDS,
                        'Another lab allocation or unknown create still reserves cards')
                name = 'corpus-qwen397b-infra-high-' + profile + '-20261024-' + bound['profile_sha256'][:8]
                deployment = 'infra-high-' + profile + '-20261024-' + bound['profile_sha256'][:8]
                command = shlex.join(['env', 'TZ=Asia/Shanghai', 'CORPUS_CONFIG=' + str((ROOT / 'profiles' / (profile+'.json')).resolve()),
                    'CORPUS_DEPLOYMENT_ID=' + deployment, 'bash', str((ROOT / 'runtime/launch.sh').resolve())])
                spec = ServingCreateSpec(name=name, model='corpus-qwen397b-bf16-ref-20261023', model_version=1,
                    command=command, port=8000, workspace=native.WORKSPACE, project=native.PROJECT,
                    group='910B资源', quota='16,128,1024', image='vllm-ascend:v0.23.0rc1', replicas=1,
                    nodes_per_replica=2, shm_gib=128, priority=6, auto_scaling=False, public_path_readonly=True,
                    description='Isolated HIGH32 BF16 native Serving infrastructure benchmark; no automatic restart')
                plan = client.servings.plan(spec)
                payload = plan.create_kwargs
                require(plan.priority == 6 and payload['task_priority'] == 6 and payload['replicas'] == 1
                        and payload['node_num_per_replica'] == 2, 'Resolved HIGH topology changed')
                require(all(payload['resource_spec_price'][k] == v for k, v in
                    {'gpu_count':16, 'cpu_count':128, 'memory_size_gib':1024}.items()), 'Resolved quota changed')
                require(not any(row.name == name for row in client.servings.iter(workspace=native.WORKSPACE)),
                        'Unowned same-name service exists')
                # Recheck after planning; unknown reservations never disappear on a missing list entry.
                low = low_released(client)
                require(lab_reserved(client) == 0, 'Lab budget changed during planning')
                budget = observable_project_budget(client, CARDS)
                require(binding(profile)[2] == bound, 'Execution files changed during planning')
                native.save(observation / 'budget-before-create.json', {'low': low, 'account': budget,
                    'lab_reserved_before_create': 0, 'requested_cards': CARDS, 'combined_ceiling': MAX_COMBINED_CARDS})
                state = {key: baseline[key] for key in ('schema_version', 'role', 'model', 'revision', 'processor_revision',
                    'precision', 'max_images', 'api_key_file', 'api_key_env', 'request_overrides')}
                state.update(scope=str(ROOT.resolve()), profile=profile, tier='high', priority=6, cards=CARDS,
                    role='generator', expected_nodes=2, expected_replicas=1, runtime_layout='prefixed_groups',
                    runtime_root=str((ROOT / 'runtime/runtime/infra-20261024').resolve()), auto_recovery_enabled=False,
                    name=name, service_name=name, operation_id=name, deployment_id=deployment, payload=payload,
                    phase='CREATE_INTENT', created_at=time.time(), canary_passed=False, release_verified=False,
                    binding=bound, config_snapshot=config, probe_attempt=1)
                validate_state(state)
                persist(intent, state)  # Immutable reservation before the only Create transmission.
                persist(path, state)
                require(not (observation / 'DISABLED').exists() and (LOW / 'DISABLED').exists(), 'Stop requested before Create')
                result = client.servings.create(spec, operation_id=name)
                state.update(service_id=result.ref.key, phase='OBSERVING')
                persist(path, state)
                return summary(state)


def publish(profile, state, ready):
    _, _, directory = paths(profile)
    target = directory / 'registry.json'
    previous = native.load(target) if target.exists() else {'registry_generation': 0}
    candidate_state = dict(state)
    if not candidate_state.get('endpoint') and previous.get('endpoints'):
        candidate_state['endpoint'] = previous['endpoints'][0]['base_url'].removesuffix('/v1')
    row = native.candidate(candidate_state, bool(ready))
    row['role'] = 'verifier'
    native.save(target, {'schema_version': 'corpus_pipeline.endpoints.v1', 'registry_generation': previous['registry_generation'] + 1,
        'endpoints': [row], 'updated_at': time.time(), 'publisher': 'infra_high_lab'})


def observe(profile, *, once=False, interval=8, stop_event=None):
    require(0 < interval <= 60, 'Observation interval must be in (0,60]')
    path, _, directory = paths(profile)
    stop_event = stop_event or threading.Event()
    native.RUNTIME = ROOT
    with locked(ROOT / ('observe-' + profile + '.lock'), nonblocking=True):
        while not stop_event.is_set():
            with locked(ROOT / 'lifecycle.lock'):
                state = load_state(profile); ready = False
                try:
                    require(binding(profile)[2] == state['binding'], 'Profile/code changed after Create')
                    with InspireClient(account='wjx-ascend') as client:
                        state = reconcile(client, profile, state)
                        detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
                        native.assert_owned(state, detail)
                        state.update(platform_status=detail.status, last_observed_at=time.time(), observation_error=None,
                                     endpoint=detail.to_dict()['endpoint'])
                        if not (directory / 'DISABLED').exists() and detail.status == 'RUNNING':
                            evidence = native.attest(state, client.servings.instances(detail.ref, workspace=native.WORKSPACE))
                            expected = {k:v for k,v in state['config_snapshot'].items() if 'key' not in k and k!='processor_files'}
                            require(all(report.get('runtime_config') == expected and report.get('serving_code_revision') ==
                                state['binding']['serving_code_revision'] for report in evidence['reports']), 'Loaded profile/runtime differs')
                            if state.get('incarnation') != evidence['incarnation']:
                                state.update(incarnation=evidence['incarnation'], qualification_receipt=None, probe_attempt=1)
                            state.update(runtime_revision=evidence['runtime_revision'], expected_boots=evidence['expected_boots'])
                            native.save(directory / 'attestation.json', evidence)
                            if native.health(state, evidence):
                                ready = native.probe(state, evidence)
                            state['readiness_reason'] = 'QUALIFIED' if ready else state.get('probe_status', 'LOADING_OR_PROBING')
                        else:
                            state['readiness_reason'] = 'DISABLED' if (directory / 'DISABLED').exists() else 'NOT_RUNNING'
                except Exception as error:
                    state['observation_error'] = {'type': type(error).__name__, 'time': time.time()}
                    state['readiness_reason'] = 'OBSERVATION_OR_QUALIFICATION_FAILED'
                ready = ready and not (directory / 'DISABLED').exists()
                state.update(canary_passed=ready, heartbeat_at=time.time(), monitor_pid=os.getpid())
                persist(path, state)
                if state.get('service_id'):
                    publish(profile, state, ready)
                native.save(directory / 'STATUS.json', summary(state))
            if once or (directory / 'DISABLED').exists():
                return summary(state)
            stop_event.wait(interval)
        # An observer stop only revokes its lease; it never stops or restarts Serving.
        with locked(ROOT / 'lifecycle.lock'):
            state = load_state(profile)
            if state.get('service_id'):
                publish(profile, state, False)
        return summary(state)


def stop(profile, *, timeout=300):
    require(0 <= timeout <= 3600, 'Invalid stop deadline')
    path, _, directory = paths(profile)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'DISABLED').write_text('Explicit HIGH lab stop requested\n')
    with locked(ROOT / 'lifecycle.lock'):
        state = load_state(profile)
        with InspireClient(account='wjx-ascend') as client:
            state = reconcile(client, profile, state)
            detail, instances, released = stopped(client, state)
            state['endpoint'] = detail.to_dict()['endpoint']
            publish(profile, state, False)
            if detail.status != 'STOPPED':
                native.save(directory / 'STOP_INTENT.json', {'service_id': state['service_id'], 'time': time.time()})
                client.servings.stop(detail.ref, workspace=native.WORKSPACE)
            deadline = time.monotonic() + timeout
            while True:
                detail, instances, released = stopped(client, state)
                receipt = {'complete': bool(released), 'time': time.time(), 'service_id': state['service_id'],
                    'name': state['name'], 'status': detail.status, 'instance_statuses': instances,
                    'binding': state['binding']}
                native.save(directory / 'STOPPED.json', receipt)
                if released:
                    state.update(platform_status='STOPPED', phase='RELEASED', release_verified=True, canary_passed=False,
                                 release_receipt_sha256=file_sha(directory / 'STOPPED.json'))
                    persist(path, state)
                    return summary(state)
                if time.monotonic() >= deadline:
                    raise RuntimeError('Final HIGH lab resource release is not verified')
                time.sleep(min(5, max(0, deadline-time.monotonic())))


def main():
    pool.configure_management_proxy_bypass()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('create', 'observe', 'stop'))
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=8)
    args = parser.parse_args()
    try:
        if args.action == 'observe':
            event = threading.Event()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: event.set())
            result = observe(args.profile, once=args.once, interval=args.interval, stop_event=event)
        else:
            result = globals()[args.action](args.profile)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        # SDK errors can include transport credentials/URLs; expose the type only.
        print(json.dumps({'action':args.action, 'profile':args.profile, 'failed':True, 'error_type':type(error).__name__}))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
