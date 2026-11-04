"""Bounded native Serving pool using the qualified Qwen BF16 deployment.

Four independently qualified 32-card replicas; no training Jobs, speculative
replacements, precision changes, or automatic restarts after manual stops.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

NATIVE = Path(__file__).resolve().parent / 'inspire_serving'
sys.path.insert(0, str(NATIVE))
import monitor as native
from downsize import observable_project_budget
from inspire.sdk import InspireClient
from inspire.sdk.models_serving import ServingCreateSpec

ROOT = Path(__file__).resolve().parent / 'ai_review_20261024'
SDK_PYTHON = '/root/.local/share/uv/tools/inspire-skill/bin/python'
MAX_CARDS = 128
CARDS = 32
LOCK = threading.Lock()
POOL_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def configure_management_proxy_bypass(environ=None):
    """Keep existing bypasses and route the two verified platform hosts directly.

    Requests prefers lowercase no_proxy when both forms exist. Synchronize the
    union before clients or guardian children start; leave other proxy settings
    intact, including those needed for paid APIs and unrelated services.
    """
    environ = os.environ if environ is None else environ
    items = []
    seen = set()
    for raw in (environ.get('NO_PROXY', ''), environ.get('no_proxy', ''),
                'qz.sii.edu.cn,keycloak-inspire-prod.sii.edu.cn'):
        for value in raw.split(','):
            item = value.strip()
            if item and item.casefold() not in seen:
                items.append(item)
                seen.add(item.casefold())
    environ['NO_PROXY'] = environ['no_proxy'] = ','.join(items)


def states(root=ROOT):
    return sorted(root.glob('qwen-[0-9][0-9].json'))


def validate_budget(rows, extra=0):
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Duplicate service name')
    ids = [r['service_id'] for r in rows if r.get('service_id')]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate service identity')
    if any(r['cards'] != CARDS or r['expected_nodes'] != 2 or r['expected_replicas'] != 1
           or r['priority'] != 3 or r['precision'] != 'BF16' for r in rows):
        raise ValueError('Unqualified topology or priority')
    # Unknown create responses and STOPPED reservations still consume a slot.
    total = sum(r['cards'] for r in rows) + extra
    if total > MAX_CARDS or extra < 0:
        raise ValueError('128-card pool budget exceeded')
    return total


def pool_index():
    native.save(ROOT / 'pool.json', {
        'schema_version': 'corpus.screening_pool.v1', 'max_cards': MAX_CARDS,
        'cards_per_replica': CARDS,
        'registries': [{'path': str(ROOT / f'registry-{i:02}.json'), 'weight': 1}
                       for i in range(1, 5)]})


def publish(path, state, ready):
    row = native.candidate(state, ready)
    row['role'] = 'verifier'  # Native attestation retains generator's HCCL checks.
    number = path.stem.split('-')[1]
    target = ROOT / f'registry-{number}.json'
    old = native.load(target) if target.exists() else {'registry_generation': 0}
    semantic = lambda v: {k: x for k, x in v.items() if k != 'ready_expires_at'}
    changed = semantic(old.get('endpoints', [{}])[0]) != semantic(row)
    native.save(target, {'schema_version': 'corpus_pipeline.endpoints.v1',
                        'registry_generation': old['registry_generation'] + int(changed),
                        'endpoints': [row], 'updated_at': time.time(),
                        'publisher': 'ai_review_pool'})


def reconcile_create(client, path, state):
    matches = [r for r in client.servings.iter(workspace=native.WORKSPACE)
               if r.name == state['name']]
    if len(matches) != 1:
        raise RuntimeError('Create intent unresolved; no automatic duplicate create')
    detail = client.servings.get(matches[0].ref, workspace=native.WORKSPACE)
    candidate = {**state, 'service_id': detail.ref.key}
    native.assert_owned(candidate, detail)
    native.save(path, {**candidate, 'phase': 'OBSERVING'})
    return candidate


def provision(count):
    if count < 1 or count > MAX_CARDS // CARDS:
        raise ValueError('Replica count must be between 1 and 4')
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'DISABLED').exists():
        raise RuntimeError('Pool is disabled')
    with (ROOT / 'provision.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = [native.load(p) for p in states()]
        validate_budget(current)
        if not current:
            raise ValueError('Explicit original-service adoption receipt required')
        with InspireClient(account='wjx-ascend') as client:
            for p in states():
                state = native.load(p)
                if not state.get('service_id'):
                    reconcile_create(client, p, state)
            for number in range(2, count + 1):
                if (ROOT / 'DISABLED').exists():
                    raise RuntimeError('Pool disabled during provisioning')
                path = ROOT / f'qwen-{number:02}.json'
                if path.exists():
                    continue
                base = native.load(ROOT / 'qwen-01.json')
                if not base.get('canary_passed') or not 0 <= time.time() - base.get('heartbeat_at', 0) < 30:
                    raise RuntimeError('First replica must pass current-boot image qualification before expansion')
                observed = client.servings.get(base['name'], workspace=native.WORKSPACE)
                native.assert_owned(base, observed)
                evidence = native.attest(base, client.servings.instances(observed.ref, workspace=native.WORKSPACE))
                if (observed.status != 'RUNNING' or evidence['incarnation'] != base.get('incarnation')
                        or not native.health(base, evidence)):
                    raise RuntimeError('First replica current-boot qualification expired')
                validate_budget([native.load(p) for p in states()], CARDS)
                budget = observable_project_budget(client, CARDS)
                native.save(ROOT / f'budget-before-{number:02}.json', budget)
                name = f'corpus-ai-review-qwen397b-low-20261024-{number:02}'
                deployment = f'ai-review-qwen397b-20261024-{number:02}'
                spec = ServingCreateSpec(name=name, model='corpus-qwen397b-bf16-ref-20261023',
                    model_version=1,
                    command=f'env TZ=Asia/Shanghai CORPUS_DEPLOYMENT_ID={deployment} bash {NATIVE}/qwen/launch.sh',
                    port=8000, workspace=native.WORKSPACE, project=native.PROJECT,
                    group='910B资源', quota='16,128,1024', image='vllm-ascend:v0.23.0rc1',
                    replicas=1, nodes_per_replica=2, shm_gib=128, priority=3,
                    auto_scaling=False, public_path_readonly=True,
                    description='DCVLM AI image review; Qwen397B BF16 32 cards; bounded 128-card native Serving pool')
                plan = client.servings.plan(spec)
                assert plan.priority == 3
                assert not any(r.name == name for r in client.servings.iter(workspace=native.WORKSPACE))
                native.save(ROOT / f'plan-{number:02}.json', plan.to_dict())
                state = {k: base[k] for k in ('schema_version', 'role', 'tier', 'priority', 'cards',
                         'model', 'revision', 'processor_revision', 'precision', 'runtime_root', 'runtime_layout',
                         'max_images', 'expected_nodes', 'expected_replicas', 'api_key_file', 'api_key_env',
                         'request_overrides', 'auto_recovery_enabled')}
                state.update(name=name, service_name=name, operation_id=name, phase='CREATE_INTENT',
                    payload=plan.create_kwargs, deployment_id=deployment, created_at=time.time(),
                    canary_passed=False, probe_attempt=1)
                native.save(path, state)  # Reserve before the only create transmission.
                if (ROOT / 'DISABLED').exists():
                    raise RuntimeError('Pool disabled; persisted reservation not sent')
                result = client.servings.create(spec, operation_id=name)
                state.update(service_id=result.ref.key, phase='OBSERVING')
                native.save(path, state)
                print(json.dumps({'created': name, 'cards': CARDS, 'reserved_cards': number * CARDS}), flush=True)
        pool_index()


def worker(path):
    with InspireClient(account='wjx-ascend') as client:
        state = native.load(path)
        previous_events = 0
        events = []
        while not (ROOT / 'DISABLED').exists():
            ready = False
            started = time.time()
            try:
                if not state.get('service_id'):
                    # A monitor may see the intent before Create returns. Keep
                    # this slot reserved and the other observers alive.
                    state = reconcile_create(client, path, state)
                detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
                native.assert_owned(state, detail)
                instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
                previous = state.get('platform_status')
                state.update(platform_status=detail.status, endpoint=detail.to_dict()['endpoint'],
                             last_observed_at=time.time(), observation_error=None)
                if detail.status == 'RUNNING':
                    state.update(was_running=True, last_running_at=time.time(), stopped_transition_at=None)
                elif detail.status == 'STOPPED' and previous == 'RUNNING':
                    state['stopped_transition_at'] = time.time()
                if time.time() - previous_events > 20:
                    events = list(client.servings.events(detail.ref, workspace=native.WORKSPACE, limit=30).items)
                    previous_events = time.time()
                native.save(ROOT / f'{path.stem}-live.json', {'time': time.time(),
                    'detail': detail.to_dict(), 'instances': [vars(x) for x in instances], 'events': events})
                try:
                    evidence = native.attest(state, instances)
                    if state.get('incarnation') != evidence['incarnation']:
                        state.update(incarnation=evidence['incarnation'], qualification_receipt=None,
                                     canary_passed=False, probe_attempt=1)
                        publish(path, state, False)
                    state.update(runtime_revision=evidence['runtime_revision'], expected_boots=evidence['expected_boots'])
                    native.save(ROOT / f'{path.stem}-attestation.json', evidence)
                    if detail.status == 'RUNNING' and native.health(state, evidence):
                        ready = native.probe(state, evidence)
                        state['readiness_reason'] = 'QUALIFIED' if ready else state.get('probe_status', 'QUALIFYING')
                    else:
                        state['readiness_reason'] = 'MODEL_LOADING_OR_HTTP_UNAVAILABLE'
                except (AssertionError, OSError, ValueError, KeyError) as error:
                    state['readiness_reason'] = str(error)[:350]
                event = native.recovery_decision(state, detail.status, events)
                if event:
                    # stop() runs in another process. The file lock makes an
                    # in-flight Start finish before Stop observes and releases it.
                    with LOCK, (ROOT / 'lifecycle.lock').open('a') as lifecycle:
                        fcntl.flock(lifecycle, fcntl.LOCK_EX)
                        ready = False
                        if not (ROOT / 'DISABLED').exists():
                            validate_budget([native.load(p) for p in states()])
                            budget = observable_project_budget(client, CARDS)
                            if not (ROOT / 'DISABLED').exists():
                                state.update(restart_event=event, restart_intent_at=time.time())
                                native.save(path, state)
                                native.save(ROOT / f'{path.stem}-restart-budget.json', budget)
                                # Same ID only; stop() will wait for this call.
                                client.servings.start(detail.ref, workspace=native.WORKSPACE)
            except Exception as error:
                state['observation_error'] = {'type': type(error).__name__, 'time': time.time()}
                if type(error).__name__ == 'AuthenticationError':
                    # Re-open this thread's SDK session; model runtime is intact.
                    client.__exit__(None, None, None)
                    client = InspireClient(account='wjx-ascend')
                    client.__enter__()
            if (ROOT / 'DISABLED').exists():
                ready = False
            state.update(canary_passed=ready, heartbeat_at=time.time(), monitor_pid=os.getpid())
            native.save(path, state)
            if state.get('service_id'):
                publish(path, state, ready)
            time.sleep(max(0.2, 8 - (time.time() - started)))
        if state.get('service_id'):
            publish(path, state, False)
        client.close()


def monitor():
    native.RUNTIME = ROOT
    with (ROOT / 'monitor.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        threads = {}
        pool_index()
        while not (ROOT / 'DISABLED').exists():
            paths = states()
            rows = [native.load(p) for p in paths]
            reserved = validate_budget(rows)
            for path in paths:
                if path not in threads:
                    threads[path] = threading.Thread(target=worker, args=(path,), daemon=True, name=path.stem)
                    threads[path].start()
            if any(not t.is_alive() for t in threads.values()):
                raise RuntimeError('Serving observer exited; guardian must restore it')
            native.save(ROOT / 'STATUS.json', {**native.identity(os.getpid()), 'time': time.time(),
                'max_cards': MAX_CARDS, 'reserved_cards': reserved,
                'loaded_code_sha256': POOL_CODE_SHA256,
                'services': [{k: r.get(k) for k in ('name', 'service_id', 'platform_status',
                    'cards', 'canary_passed', 'readiness_reason', 'heartbeat_at', 'observation_error')} for r in rows]})
            time.sleep(5)


def guard():
    command = [SDK_PYTHON, '-m', 'corpus_pipeline.serving.ai_review_pool', 'monitor']
    with (ROOT / 'guardian.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (ROOT / 'DISABLED').exists():
            receipt = native.load(ROOT / 'monitor-process.json') if (ROOT / 'monitor-process.json').exists() else {}
            live = native.identity(receipt.get('pid', -1))
            if not live or live['command'] != command or live['starttime_ticks'] != receipt.get('starttime_ticks'):
                with (ROOT / 'monitor.log').open('a') as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
                native.save(ROOT / 'monitor-process.json', {**native.identity(proc.pid), 'time': time.time()})
            native.save(ROOT / 'guardian-heartbeat.json', {**native.identity(os.getpid()), 'time': time.time()})
            time.sleep(5)


def stop():
    """Release only this pool after review completion or an explicit user stop."""
    (ROOT / 'DISABLED').write_text('AI review pool stop requested\n')
    with (ROOT / 'provision.lock').open('a') as handle, \
            (ROOT / 'lifecycle.lock').open('a') as lifecycle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        fcntl.flock(lifecycle, fcntl.LOCK_EX)
        rows = [native.load(p) for p in states()]
        validate_budget(rows)
        with InspireClient(account='wjx-ascend') as client:
            for path, state in zip(states(), rows):
                if not state.get('service_id'):
                    state = reconcile_create(client, path, state)
                detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
                native.assert_owned(state, detail)
                publish(path, state, False)
                if detail.status != 'STOPPED':
                    native.save(ROOT / f'{path.stem}-stop-intent.json',
                                {'time': time.time(), 'service_id': state['service_id']})
                    client.servings.stop(detail.ref, workspace=native.WORKSPACE)
            rows = [native.load(p) for p in states()]
            deadline = time.monotonic() + 300
            while True:
                statuses = []
                for state in rows:
                    detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
                    native.assert_owned(state, detail)
                    instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
                    statuses.append({'service_id': detail.ref.key, 'name': detail.name,
                        'status': detail.status, 'instance_statuses': [v.status for v in instances]})
                terminal = {'INSTANCE_STOPPED', 'INSTANCE_FAILED', 'INSTANCE_SUCCEEDED',
                            'INSTANCE_DELETED', 'INSTANCE_TERMINATED', 'INSTANCE_COMPLETED'}
                complete = all(v['status'] == 'STOPPED' and set(v['instance_statuses']) <= terminal
                               for v in statuses)
                native.save(ROOT / 'STOPPED.json', {'time': time.time(), 'complete': complete, 'services': statuses})
                if complete:
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError('Pool stop accepted but final resource release not yet verified')
                time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('provision', 'monitor', 'guard', 'stop'))
    parser.add_argument('--replicas', type=int, default=4)
    args = parser.parse_args()
    configure_management_proxy_bypass()
    ROOT.mkdir(exist_ok=True)
    if args.action == 'provision':
        provision(args.replicas)
    else:
        globals()[args.action]()


if __name__ == '__main__':
    main()
