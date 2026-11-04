"""Resume only accepted LOW fragment Notebooks and their idempotent workers.

Run on the stable CPU controller with the Inspire SDK interpreter. The allocator
adds validated LOW Notebooks within an explicit per-run card cap and the live
256-card account budget. Lifecycle writes
only target accepted IDs; no resources are deleted. Unknown observations and
uncertain starts fail closed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import signal
import socket
import tempfile
import threading
import time

from corpus_pipeline.production.common import atomic_json, file_sha

REPO = Path(__file__).resolve().parents[2]
ACCOUNT = 'wjx-ascend'
WORKSPACE = '昇腾卡公共空间'
PROJECT = '多模态大模型新架构评测探索与scaling-law'
GROUP = '910B资源'
EXCLUDED = {f'infra-gpu-npu-{n}.host.shzhisuan.com' for n in ('047', '248', '295')}
PENDING_LOST_FIT_GRACE = 90
PENDING_MAX_WAIT = 240
RETIRE_COOLDOWN_BASE = 120
RETIRE_COOLDOWN_CAP = 600


def load_submissions(execution):
    execution = Path(execution).resolve()
    submitted = json.loads((execution / 'SUBMITTED.json').read_text())
    notebooks = submitted['notebooks']
    if not notebooks or len({r['notebook_id'] for r in notebooks}) != len(notebooks):
        raise ValueError('Accepted Notebook identities must be nonempty and unique')
    if len({r['name'] for r in notebooks}) != len(notebooks):
        raise ValueError('Accepted Notebook names must be unique')
    if sum(r['devices'] for r in notebooks) != submitted['total_cards'] or submitted['total_cards'] > 8:
        raise ValueError('Accepted fragment allocation exceeds its eight-card contract')
    result = []
    for row in notebooks:
        if row.get('priority') != 3 or row['devices'] not in (1, 2, 4, 8):
            raise ValueError('Only explicitly accepted LOW fragment Notebooks are supported')
        intent = json.loads((execution / (row['name'] + '-INTENT.json')).read_text())
        payload = intent['create_kwargs']
        accepted = json.loads((execution / (row['name'] + '-ACCEPTED.json')).read_text())
        if accepted['notebook_id'] != row['notebook_id'] or accepted['name'] != row['name']:
            raise ValueError('Submitted identity differs from the accepted receipt')
        if (payload['name'] != row['name'] or payload['gpu_count'] != row['devices']
                or payload['task_priority'] != 3 or payload['project_name'] != PROJECT
                or payload['node_id'] != row['node'] or row['node'] in EXCLUDED
                or payload['is_publicpath_readonly'] is not True):
            raise ValueError('Submitted Notebook payload differs from the approved contract')
        result.append({**row, 'workspace_id': payload['workspace_id'],
                       'project_id': payload['project_id'],
                       'group_id': payload['logic_compute_group_id'],
                       'cpu': payload['cpu_count'], 'memory': payload['memory_size']})
    if len({r['workspace_id'] for r in result}) != 1 or len({r['project_id'] for r in result}) != 1:
        raise ValueError('Accepted Notebooks must share the approved workspace and project')
    return result


def completed(root, queue_status=None):
    """Only the verified full frontpass sentinel authorizes resource cleanup."""
    path = root / 'FRONTPASS_COMPLETE.json'
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    plan = json.loads((root / 'plan.json').read_text())
    if (value.get('complete') is not True or value.get('scope') != 'offline_frontpass_only'
            or value.get('counts', {}).get('records') != plan['record_count']
            or value.get('snapshot_fingerprint') != plan['snapshot_fingerprint']
            or value.get('index_sha256') != file_sha(root / 'frontpass_index.json')):
        raise ValueError('Frontpass completion evidence is inconsistent')
    if queue_status is not None and not (
            queue_status.get('all_tasks_completed') is True
            and queue_status.get('task_count') == plan['task_count']
            and queue_status.get('counts', {}).get('completed') == plan['task_count']
            and not any(queue_status.get('counts', {}).get(k, 0) for k in ('pending', 'running', 'failed'))
            and queue_status.get('processed_records') == plan['record_count']):
        raise ValueError('Published frontpass sentinel disagrees with the full queue')
    return True


def worker_command(root, item, action):
    parts = ['bash', str(REPO / 'corpus_pipeline/cluster/npu_env.sh'), '-m',
             'corpus_pipeline.cluster.notebook_worker', action,
             '--root', str(root), '--name', item['name']]
    if action == 'launch':
        parts.extend(['--devices', str(item['devices'])])
    return 'cd ' + shlex.quote(str(REPO)) + ' && ' + shlex.join(parts)


def parse_worker_receipt(receipt, item):
    if receipt.get('completed') is not True or receipt.get('returncode') != 0:
        raise RuntimeError('Worker observation did not complete')
    # Environment startup can emit harmless notices before the JSON line.
    lines = receipt.get('output', '').splitlines()
    candidates = []
    for line in lines:
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get('name') == item['name']:
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError('Expected one unambiguous worker status object')
    value = candidates[0]
    if value.get('status') not in ('running', 'exited', 'unknown') or not value.get('hostname'):
        raise ValueError('Unrecognized worker observation')
    if value.get('devices') not in (None, item['devices']):
        raise ValueError('Worker device contract changed')
    # Keep only known non-secret status fields, never arbitrary terminal output.
    return {k: value[k] for k in ('name', 'hostname', 'status', 'lock_busy', 'attempt',
            'devices', 'pid', 'phase', 'returncode', 'ended_at', 'error_type',
            'launched', 'command_sha256', 'node_command_sha256') if k in value}


def validate_live(item, value):
    if (value.get('notebook_id') != item['notebook_id'] or value.get('name') != item['name']
            or value.get('workspace') != WORKSPACE or value.get('project') != PROJECT
            or value.get('compute_group') != GROUP
            or str(value.get('priority')) != '3' or value.get('priority_level') != 'LOW'
            or value.get('resource', {}).get('gpu_count') != item['devices']):
        raise ValueError('Live Notebook identity or LOW resource contract changed')
    node = (value.get('node') or {}).get('name')
    if node and (node in EXCLUDED or node != item['node']):
        raise ValueError('Live Notebook placement escaped the approved node')


class Platform:
    def __init__(self, client, managed, allocator=None, states=None):
        self.client = client
        self.managed = managed
        self.allocator = allocator
        self.states = states if states is not None else {}

    def ref(self, item):
        from inspire.sdk.models_notebooks import NotebookRef
        return NotebookRef(name=item['name'], account=ACCOUNT, base_url=self.client.base_url,
                           key=item['notebook_id'], workspace_id=item['workspace_id'])

    def observe(self, item):
        with self.client._transport.scope(timeout=30):
            value = self.client.notebooks.get(self.ref(item), workspace=WORKSPACE)
        return value.to_dict() | {'notebook_id': value.ref.key}

    def events(self, item):
        from corpus_pipeline.cluster.watch_jobs import event_view
        with self.client._transport.scope(timeout=20):
            events = self.client.notebooks.events(self.ref(item), workspace=WORKSPACE, limit=8)
        return [event_view(row) | {'origin': 'inspire_sdk_notebook_events_authority_unconfirmed',
                'event_id': row.get('event_id') or row.get('id'),
                'bottleneck_independently_verified': False} for row in events.items]

    def budget(self, item):
        from corpus_pipeline.serving.inspire_serving.downsize import observable_project_budget
        from .fragment_allocator import verified_retired
        managed_budget = self.allocator.managed_budget(self.managed, self.states) if self.allocator else None
        if (managed_budget and managed_budget['max_cards'] is not None
                and managed_budget['reserved_total_cards'] > managed_budget['max_cards']):
            raise ValueError('Restart would exceed the configured pipeline card cap')
        value = observable_project_budget(self.client, 0)
        reservation = self.allocator.reserved_budget(value, self.managed, self.states) if self.allocator else None
        if reservation:
            managed_budget = reservation['managed_budget']
            if (managed_budget['max_cards'] is not None
                    and managed_budget['reserved_total_cards'] > managed_budget['max_cards']):
                raise ValueError('Restart would exceed the configured pipeline card cap')
        # Existing accepted IDs already present in live active usage are not
        # counted a second time. STOPPED entries normally are absent.
        additional = reservation['managed_missing_from_live_cards'] if reservation else sum(row['devices'] for row in self.managed
                         if row['notebook_id'] not in value['records']
                         and not verified_retired(row, self.states.get(row['notebook_id'], {})))
        uncertain = reservation['uncertain_cards'] if reservation else 0
        total = value['observable_requested_cards'] + additional + uncertain
        if total > 256:
            raise ValueError('Restart would exceed the observable 256-card ceiling')
        return {'observable_requested_cards': value['observable_requested_cards'],
                'additional_restart_cards': additional, 'total_with_restart': total,
                'unconfirmed_create_reserved_cards': uncertain,
                'managed_budget': managed_budget,
                'checked_at': time.time()}

    def pending_fit(self, item):
        if self.allocator is None:
            raise RuntimeError('Pending retirement requires the configured allocator')
        return self.allocator.pending_fit(item)

    def retirement_cooldown(self, item):
        losses = sum(1 for key, state in self.states.items()
                     if key != item['notebook_id']
                     and (state.get('retirement') or {}).get('node') == item['node']
                     and (state.get('retirement') or {}).get('stop_attempted_at') is not None)
        return min(RETIRE_COOLDOWN_CAP, RETIRE_COOLDOWN_BASE * 2 ** min(losses, 3))

    def start(self, item):
        with self.client._transport.scope(timeout=30):
            self.client.notebooks.start(self.ref(item), workspace=WORKSPACE)

    def stop(self, item):
        with self.client._transport.scope(timeout=30):
            self.client.notebooks.stop(self.ref(item), workspace=WORKSPACE)

    def worker(self, root, item, action):
        from corpus_pipeline.cluster.notebook_exec import execute
        result = execute(worker_command(root, item, action), notebook=item['name'],
                         notebook_id=item['notebook_id'], workspace_id=item['workspace_id'],
                         account=ACCOUNT, workspace=WORKSPACE, timeout=40, max_output_bytes=16384)
        return parse_worker_receipt(result, item)

    def job_ref(self, item):
        from inspire.sdk.models import JobRef
        return JobRef(name=item['name'], account=ACCOUNT, base_url=self.client.base_url,
                      key=item['job_id'], workspace_id=item['workspace_id'])

    def observe_job(self, item):
        with self.client._transport.scope(timeout=30):
            value = self.client.jobs.get(self.job_ref(item), workspace=WORKSPACE)
        if value.ref.key != item['job_id'] or value.name != item['name'] or value.project != PROJECT:
            raise ValueError('Completion cleanup Job identity differs from accepted receipt')
        return {'status': value.status,
                'native_retry': value.raw.get('auto_fault_tolerance') is True}

    def stop_job(self, item):
        with self.client._transport.scope(timeout=30):
            self.client.jobs.stop(self.job_ref(item), workspace=WORKSPACE)


def handle_retirement(item, state, platform, save, now):
    """Observe an irreversible retirement intent; never replay an uncertain stop."""
    retirement = state['retirement']
    if (retirement.get('notebook_id') != item['notebook_id']
            or retirement.get('node') != item['node']):
        raise ValueError('Retirement identity differs from accepted Notebook')
    status = state['platform_status']
    if status == 'STOPPED':
        if retirement.get('phase') != 'verified_stopped':
            retirement.update(phase='verified_stopped', verified_stopped_at=now,
                              cooldown_until=now + retirement['cooldown_seconds'])
        state.update(phase='retired_stopped', released=True, platform_action=None)
        if state.get('action_required', '').startswith('retirement_'):
            state.pop('action_required', None)
        save(); return
    if retirement.get('phase') == 'verified_stopped':
        # A later external start must restore the budget reservation. Do not
        # race the external actor by stopping a now RUNNING notebook here.
        state.update(phase='retired_became_active', released=False,
                     action_required='retirement_previously_stopped_now_active')
        save(); return
    if retirement.get('stop_attempted_at') is not None:
        state.update(phase='retirement_wait_for_stopped', released=False)
        if now - retirement['stop_attempted_at'] >= 300:
            state['action_required'] = 'retirement_stop_remains_unconfirmed'
        save(); return
    # This second exact-ID GET occurs after the durable intent and immediately
    # before stop. It catches allocation races during the capacity query/save.
    fresh = platform.observe(item)
    validate_live(item, fresh)
    state.update(platform_status=fresh['status'], observed_at=now, node=fresh.get('node'))
    if fresh['status'] == 'RUNNING':
        retirement.update(phase='cancelled_running_race', cancelled_at=now)
        state.setdefault('retirement_history', []).append(retirement)
        state.pop('retirement')
        state.update(phase='retirement_cancelled_running_race', pending_since=None)
        save(); return
    if fresh['status'] == 'STOPPED':
        handle_retirement(item, state, platform, save, now)
        return
    if fresh['status'] != 'PENDING':
        state['phase'] = 'retirement_wait_for_pending_or_stopped'
        save(); return
    retirement.update(phase='stop_sending', stop_attempted_at=now, stop_uncertain=True)
    state['phase'] = 'retirement_stop_intent'
    save()
    # This boundary is single-send. Exceptions or a process crash retain the
    # exact reservation until a future read proves STOPPED.
    platform.stop(item)
    retirement.update(phase='stop_requested', stop_uncertain=False)
    state['phase'] = 'retirement_stop_requested'
    save()


def maybe_retire_pending(item, state, platform, save, now):
    if state.get('pending_since') is None:
        since = max(item.get('accepted_at', now), state.get('last_platform_restart', 0))
        state['pending_since'] = min(since, now)
    waited = now - state['pending_since']
    if waited < PENDING_LOST_FIT_GRACE:
        return False
    reason = None
    try:
        evidence = platform.pending_fit(item)
        state['pending_placement_check'] = evidence
        if evidence.get('fits') is False:
            reason = 'fresh_node_no_longer_fits'
    except Exception as exc:
        state['pending_placement_check'] = {'error_type': type(exc).__name__, 'observed_at': now}
    if waited >= PENDING_MAX_WAIT:
        reason = reason or 'pending_deadline_exceeded'
    if reason is None:
        return False
    cooldown = platform.retirement_cooldown(item)
    state['retirement'] = {'notebook_id': item['notebook_id'], 'node': item['node'],
        'requested_at': now, 'reason': reason, 'phase': 'intent',
        'pending_seconds': waited, 'cooldown_seconds': cooldown}
    state.update(phase='retirement_intent', released=False)
    save()
    handle_retirement(item, state, platform, save, now)
    return True


def step(root, item, state, platform, *, is_complete, save, now=None):
    """One fenced observation/action; injected platform permits offline tests."""
    now = time.time() if now is None else now
    state.setdefault('consecutive_worker_errors', 0)
    live = platform.observe(item)
    validate_live(item, live)
    status = live['status']
    previous_status = state.get('platform_status')
    state.update(platform_status=status, observed_at=now, node=live.get('node'))
    if state.get('retirement'):
        handle_retirement(item, state, platform, save, now)
        return
    if status == 'RUNNING':
        state['pending_since'] = None
    elif status == 'PENDING' and previous_status == 'RUNNING':
        # A preempted previously running container starts a fresh scheduling
        # grace period rather than inheriting its original creation age.
        state['pending_since'] = now
    pending = state.get('platform_action')
    if pending:
        resolved = ((pending['action'] == 'start' and status in ('RUNNING', 'PENDING', 'CREATING'))
                    or (pending['action'] == 'stop' and status in ('STOPPED', 'FAILED')))
        if resolved:
            state['platform_action'] = None
            if state.get('action_required') == 'platform_' + pending['action'] + '_remains_uncertain':
                state.pop('action_required', None)
        elif pending.get('uncertain'):
            state['phase'] = 'observe_uncertain_' + pending['action']
            if now - pending['time'] > 300:
                state['action_required'] = 'platform_' + pending['action'] + '_remains_uncertain'
            save(); return
        elif now - pending['time'] < 300:
            state['phase'] = 'await_' + pending['action']
            save(); return
        else:
            state['platform_action'] = None
    if is_complete:
        if status in ('STOPPED', 'FAILED'):
            state.update(phase='complete_released', released=True)
        elif status in ('RUNNING', 'PENDING', 'CREATING'):
            state['platform_action'] = {'action': 'stop', 'time': now, 'uncertain': True}
            save()
            platform.stop(item)
            state['platform_action']['uncertain'] = False
            state['phase'] = 'complete_stop_requested'
        else:
            state['phase'] = 'completion_wait_for_known_platform_state'
        save(); return
    if state.get('action_required'):
        state['phase'] = 'action_required'
        save(); return
    if status in ('STOPPED', 'FAILED'):
        state['restart_budget'] = platform.budget(item)
        state['platform_action'] = {'action': 'start', 'time': now, 'uncertain': True}
        save()
        platform.start(item)
        state['platform_action']['uncertain'] = False
        state.update(phase='start_requested', last_platform_restart=now, pending_since=now)
        save(); return
    if status != 'RUNNING':
        if status == 'PENDING' and maybe_retire_pending(item, state, platform, save, now):
            return
        state['phase'] = 'wait_for_running'
        if hasattr(platform, 'events'):
            try:
                state['latest_events'] = platform.events(item)
            except Exception as exc:
                state['events_error_type'] = type(exc).__name__
        save(); return
    worker = platform.worker(root, item, 'status')
    state['worker'] = worker
    previous_host = state.get('worker_hostname')
    host_changed = previous_host is not None and previous_host != worker['hostname']
    if host_changed:
        state.update(consecutive_worker_errors=0, last_failed_attempt=None,
                     running_since=None, phase='container_replaced')
    state['worker_hostname'] = worker['hostname']
    if worker['status'] == 'running':
        if (state.get('running_attempt') != worker.get('attempt')
                or not isinstance(state.get('running_since'), (int, float))):
            state.update(running_attempt=worker.get('attempt'), running_since=now)
        # Sustained healthy process observation breaks a sequence of startup
        # crashes; merely launching a fresh process never resets the counter.
        if now - state.get('running_since', now) >= 600:
            state['consecutive_worker_errors'] = 0
        state['phase'] = 'worker_running'
        save(); return
    state['running_since'] = None
    if worker.get('lock_busy') is not False:
        state['phase'] = 'worker_identity_unknown_lock_busy'
        save(); return
    attempt = worker.get('attempt')
    failure_key = (worker['hostname'] + ':' + str(attempt)) if attempt is not None else None
    if (worker['status'] == 'exited' and worker.get('returncode') not in (0, None)
            and failure_key and failure_key != state.get('last_failed_attempt') and not host_changed):
        state['consecutive_worker_errors'] += 1
        state['last_failed_attempt'] = failure_key
    if state['consecutive_worker_errors'] >= 5:
        state.update(action_required='five_consecutive_worker_errors', phase='action_required')
        save(); return
    # The remote launcher proves exclusive ownership using its container-local
    # flock and PID/start-time/boot-ID identity before it starts anything.
    state.update(phase='worker_launch_intent', last_launch_intent=now)
    save()
    launched = platform.worker(root, item, 'launch')
    state.update(worker=launched, phase='worker_launch_observed')
    save()


def release_jobs(items, runtime, platform, save, *, now=None):
    """Only the four accepted frontpass Jobs are eligible for final cleanup."""
    now = time.time() if now is None else now
    states = runtime.setdefault('completion_jobs', {})
    for item in items:
        state = states.setdefault(item['job_id'], {'name': item['name']})
        try:
            live = platform.observe_job(item)
            state.update(status=live['status'], observed_at=now)
            terminal = live['status'] in ('STOPPED', 'CANCELLED', 'SUCCEEDED', 'COMPLETED', 'DELETED')
            terminal |= live['status'] == 'FAILED' and not live['native_retry']
            if terminal:
                state.update(released=True, stop_pending=None); save(); continue
            if live['status'] not in ('QUEUING', 'RUNNING', 'PENDING', 'CREATING', 'FAILED', 'RETRYING', 'RESTARTING'):
                state['phase'] = 'unknown_wait'; save(); continue
            pending = state.get('stop_pending')
            if pending and (pending['uncertain'] or now - pending['time'] < 300):
                state['phase'] = 'observe_stop_result'; save(); continue
            state['stop_pending'] = {'time': now, 'uncertain': True}
            save()
            platform.stop_job(item)
            state['stop_pending']['uncertain'] = False
            state['phase'] = 'stop_requested'
            save()
        except Exception as exc:
            state['observation_error_type'] = type(exc).__name__; save()
    return all(states.get(item['job_id'], {}).get('released') for item in items)


def merge_managed(initial, additional):
    indexed = {item['notebook_id']: item for item in initial}
    names = {item['name']: item['notebook_id'] for item in initial}
    for item in additional:
        if item['notebook_id'] in indexed and indexed[item['notebook_id']] != item:
            raise ValueError('Persistent Notebook registration changed')
        if names.get(item['name'], item['notebook_id']) != item['notebook_id']:
            raise ValueError('Two accepted Notebook IDs share a name')
        indexed[item['notebook_id']] = item
        names[item['name']] = item['notebook_id']
    return list(indexed.values())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--execution', '--submission', required=True, type=Path)
    parser.add_argument('--interval', type=int, default=60, help='Management interval; allocation polls every 10 seconds')
    parser.add_argument('--auto-scale', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-cards', type=int,
                        help='Pipeline card cap, including pending and unconfirmed requests; required with --auto-scale')
    parser.add_argument('--consolidate', action=argparse.BooleanOptionalAction, default=True,
                        help='Consolidate sparse owned nodes when automatic allocation is enabled')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.interval < 60:
        parser.error('Polling interval must be at least 60 seconds')
    if args.max_cards is not None and not 1 <= args.max_cards <= 256:
        parser.error('--max-cards must be between 1 and 256')
    if args.auto_scale and args.max_cards is None:
        parser.error('--max-cards is required when automatic allocation is enabled')
    return args


def main():
    args = parse_args()
    root, execution = args.root.resolve(), args.execution.resolve()
    if execution.name == 'SUBMITTED.json':
        execution = execution.parent
    items = load_submissions(execution)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    logging.getLogger('inspire').setLevel(logging.CRITICAL)
    # This lock is deliberately on container-local /tmp. Shared storage does
    # not implement cross-host flock; this controller only runs on its stable
    # CPU host and records that host in every persisted observation.
    key = hashlib.sha256(str(root).encode()).hexdigest()[:24]
    lock = open(Path(tempfile.gettempdir()) / ('corpus-fragment-controller-' + key + '.lock'), 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from corpus_pipeline.cluster.text_worker import verify_control_host
    verify_control_host(root)
    path = execution / 'CONTROLLER_RUNTIME.json'
    runtime = json.loads(path.read_text()) if path.exists() else {'notebooks': {}}
    if runtime.get('hostname') not in (None, socket.gethostname()):
        raise RuntimeError('Controller owner host changed; reconcile previous owner before takeover')
    runtime.update(schema='fragment_notebook_controller.v1', pid=os.getpid(),
                   hostname=socket.gethostname(), interval_seconds=args.interval,
                   execution=str(execution), root=str(root), script_sha256=file_sha(__file__),
                   creates_resources=args.auto_scale, max_cards=args.max_cards,
                   consolidates_nodes=args.auto_scale and args.consolidate,
                   deletes_resources=False)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    from inspire.sdk import InspireClient
    def save():
        runtime['updated_at'] = time.time()
        atomic_json(path, runtime)
    with InspireClient(account=ACCOUNT) as client:
        from corpus_pipeline.cluster.fragment_allocator import Allocator
        from corpus_pipeline.cluster.fragment_consolidation import Consolidator
        from corpus_pipeline.cluster.watch_jobs import load_submissions as load_jobs
        from corpus_pipeline.cluster.queue_rpc import status as queue_status
        cleanup_jobs = load_jobs(root)
        allocator = Allocator(client, root, execution, max_cards=args.max_cards)
        items[:] = merge_managed(items, allocator.load_managed())
        runtime['managed_budget'] = allocator.managed_budget(items, runtime['notebooks'])
        save()
        platform = Platform(client, items, allocator, runtime['notebooks'])
        consolidator = Consolidator(root)
        next_management = next_allocation = 0
        allocation_errors = 0
        done = False
        queue_exhausted = False
        while not stop.is_set():
            started = time.time()
            if started >= next_management:
                try:
                    current = queue_status(root)
                    done = completed(root, current)
                    queue_exhausted = (current.get('all_tasks_terminal') is True
                                       and bool(current.get('counts', {}).get('failed')))
                    runtime['queue_observation'] = {k: current.get(k) for k in (
                        'task_count', 'counts', 'all_tasks_completed', 'processed_records', 'processed_images')}
                except Exception as exc:
                    runtime.update(health='completion_observation_error', error_type=type(exc).__name__)
                    save()
                    if args.once: break
                    stop.wait(10); continue
                for item in items:
                    if stop.is_set():
                        break
                    state = runtime['notebooks'].setdefault(item['notebook_id'], {'name': item['name']})
                    before = {k: state.get(k) for k in ('phase', 'platform_status', 'action_required')}
                    try:
                        step(root, item, state, platform, is_complete=done, save=save)
                        state.pop('observation_error_type', None)
                    except Exception as exc:
                        state.update(observation_error_type=type(exc).__name__, last_error_at=time.time())
                    after = {k: state.get(k) for k in ('phase', 'platform_status', 'action_required')}
                    if before != after:
                        with (execution / 'CONTROLLER_CHANGES.jsonl').open('a') as stream:
                            stream.write(json.dumps({'time': time.time(), 'notebook_id': item['notebook_id'],
                                'name': item['name'], 'before': before, 'after': after}, ensure_ascii=False) + '\n')
                            stream.flush(); os.fsync(stream.fileno())
                    save()
                next_management = time.time() + args.interval
            if not stop.is_set() and time.time() >= next_allocation:
                try:
                    previous_count = len(items)
                    consolidation = consolidator.advance(
                        items, runtime['notebooks'], platform, allocator, save,
                        is_complete=done or queue_exhausted,
                        enabled=args.auto_scale and args.consolidate,
                        should_stop=stop.is_set)
                    runtime['consolidation'] = consolidation
                    additional = allocator.discover_and_allocate(
                        items, runtime['notebooks'],
                        allow_create=(args.auto_scale and not done and not queue_exhausted
                                      and consolidation['allow_normal_allocation']),
                        should_stop=stop.is_set)
                    items[:] = merge_managed(items, allocator.load_managed() + additional)
                    allocation_errors = 0
                    runtime['allocation'] = {'healthy': True, 'last_success': time.time(),
                                             'managed_notebooks': len(items)}
                    runtime['managed_budget'] = allocator.managed_budget(items, runtime['notebooks'])
                    if additional or len(items) != previous_count: next_management = 0
                except Exception as exc:
                    allocation_errors += 1
                    runtime['allocation'] = {'healthy': False, 'error_type': type(exc).__name__,
                                             'consecutive_errors': allocation_errors, 'time': time.time()}
                    try:
                        known = merge_managed(items, allocator.load_managed())
                        if len(known) != len(items): next_management = 0
                        items[:] = known
                    except Exception as registry_error:
                        runtime['allocation']['registry_error_type'] = type(registry_error).__name__
                next_allocation = time.time() + min(300, 10 * 2 ** min(allocation_errors, 5))
                save()
            runtime['health'] = 'action_required' if any(s.get('action_required') for s in runtime['notebooks'].values()) else 'observing'
            runtime['frontpass_complete'] = done
            jobs_released = release_jobs(cleanup_jobs, runtime, platform, save) if done else False
            save()
            if (done and jobs_released and allocator.reserved_unconfirmed_cards() == 0
                    and all(runtime['notebooks'].get(i['notebook_id'], {}).get('released') for i in items)):
                runtime['health'] = 'complete_released'; save(); break
            if args.once: break
            stop.wait(10)


if __name__ == '__main__':
    main()
