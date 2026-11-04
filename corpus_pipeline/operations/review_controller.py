"""Operational successor for a frozen review pool; no model or data mutation.

Run this file with the frozen runtime on PYTHONPATH. The original configuration,
reservation ledger, backend functions, model command and 30-second readiness
lease stay intact. Only scheduling and redundant SDK name/event lookups change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from corpus_pipeline.serving import budget_pool as original

SCHEMA = 'corpus.review_operational_policy.v1'
SOURCE = Path(__file__).resolve()
SOURCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
BUDGET_SCHEMA = 'corpus.fresh_requested_card_budget.v1'
BUDGET_TERMINAL = {'STOPPED', 'FAILED', 'DELETED', 'CANCELLED', 'SUCCEEDED', 'COMPLETED'}


def fresh_pages(service, workspace, counters, kind):
    """Finish the live SDK cursor chain, refusing ambiguous/truncated scans."""
    cursor, cursors, identities = None, set(), set()
    for _ in range(10000):
        page = service.list(workspace, limit=100, cursor=cursor)
        counters[kind + '_pages'] = counters.get(kind + '_pages', 0) + 1
        original.require(page.items or page.next_cursor is None, 'Empty nonterminal budget page')
        for row in page.items:
            key = row.ref.key
            original.require(isinstance(key, str) and key and key not in identities,
                             'Missing or duplicate budget resource identity')
            identities.add(key)
            yield row
        if page.next_cursor is None:
            return
        original.require(isinstance(page.next_cursor, str) and page.next_cursor
                         and page.next_cursor not in cursors, 'Budget cursor repeated')
        cursors.add(page.next_cursor)
        cursor = page.next_cursor
    raise ValueError('Budget pagination exceeded bounded full scan')


def budget_integer(value, name, minimum=0):
    original.require(type(value) is int and value >= minimum, 'Invalid budget ' + name)
    return value


def budget_owner(raw):
    creator = raw.get('created_by')
    owner = raw.get('user_id') or (creator.get('id') if isinstance(creator, dict) else None)
    return owner if isinstance(owner, str) and owner else None


def serving_requested_cards(raw):
    resource = raw.get('resource_spec_price')
    original.require(isinstance(resource, dict), 'Serving budget topology missing')
    return (budget_integer(resource.get('gpu_count'), 'gpu_count')
            * budget_integer(raw.get('replicas'), 'replicas', 1)
            * budget_integer(raw.get('node_num_per_replica'), 'node_num_per_replica', 1))


def requested_card_budget(client, requested, user_id, workspace='昇腾卡公共空间'):
    """Fresh requested-card accounting; Serving allocation is explicitly unmeasured.

    ListServings already requests my_serving=True and its fresh rows contain the
    admission topology. Missing/invalid fields require a fresh exact-ID detail;
    failures never become zero cards. Unknown states count their full request.
    No readiness evidence comes from this budget-only function.
    """
    budget_integer(requested, 'additional cards')
    original.require(isinstance(user_id, str) and user_id, 'Current budget owner missing')
    records, counters = {}, {'serving_detail_fallbacks': 0, 'serving_instances_queries': 0}
    for service in fresh_pages(client.servings, workspace, counters, 'serving'):
        if service.status in BUDGET_TERMINAL:
            continue
        raw = service.raw
        owner = budget_owner(raw)
        if owner and owner != user_id:
            continue
        try:
            cards = serving_requested_cards(raw)
        except (ValueError, TypeError, AttributeError):
            cards = None
        if owner is None or cards is None:
            detail = client.servings.get(service.ref, workspace=workspace)
            counters['serving_detail_fallbacks'] += 1
            original.require(detail.ref.key == service.ref.key, 'Serving budget fallback identity changed')
            raw, owner = detail.raw, budget_owner(detail.raw)
            original.require(owner is not None, 'Missing Serving budget ownership evidence')
            if owner != user_id:
                continue
            cards = serving_requested_cards(raw)
        records[service.ref.key] = {'kind': 'serving', 'name': service.name,
            'cards': cards, 'requested_cards': cards, 'allocated_cards': None,
            'allocated_cards_upper_bound': cards, 'allocation_measurement': 'conservative_upper_bound',
            'status': service.status, 'budget_evidence': 'fresh_personal_list_or_exact_detail'}
    # Preserve the existing account-wide Notebook and Job coverage and actual
    # placement observations. Only Serving's redundant detail/instances fanout
    # is removed. Lists are never cached by this operational budget function.
    for notebook in fresh_pages(client.notebooks, workspace, counters, 'notebook'):
        if notebook.status in BUDGET_TERMINAL:
            continue
        original.require(notebook.ref.key not in records, 'Cross-kind budget identity conflict')
        cards = budget_integer(notebook.resource.get('gpu_count'), 'notebook gpu_count')
        allocated = cards if notebook.status == 'RUNNING' or notebook.node.get('name') else 0
        records[notebook.ref.key] = {'kind': 'notebook', 'name': notebook.name,
            'cards': cards, 'requested_cards': cards, 'allocated_cards': allocated,
            'allocated_cards_upper_bound': cards, 'allocation_measurement': 'live_status_and_placement',
            'status': notebook.status}
    for job in fresh_pages(client.jobs, workspace, counters, 'job'):
        if job.status in BUDGET_TERMINAL:
            continue
        original.require(job.ref.key not in records, 'Cross-kind budget identity conflict')
        detail = client.jobs.get(job.ref, workspace=workspace)
        original.require(detail.ref.key == job.ref.key, 'Job budget fallback identity changed')
        if detail.status in BUDGET_TERMINAL:
            continue
        owner = budget_owner(detail.raw)
        original.require(owner is not None, 'Missing Job budget ownership evidence')
        if owner != user_id:
            continue
        configs = detail.raw.get('framework_config')
        original.require(isinstance(configs, list) and configs, 'Job budget resource contract missing')
        gpu_counts = {budget_integer(x.get('gpu_count'), 'job gpu_count') for x in configs}
        original.require(len(gpu_counts) == 1, 'Heterogeneous Job needs explicit allocation accounting')
        cards = sum(budget_integer(x.get('gpu_count'), 'job gpu_count')
                    * budget_integer(x.get('instance_count'), 'job instance_count', 1) for x in configs)
        instances = client.jobs.instances(detail.ref, workspace=workspace)
        allocated = next(iter(gpu_counts)) * sum(bool(x.node) for x in instances)
        records[job.ref.key] = {'kind': 'job', 'name': job.name, 'cards': cards,
            'requested_cards': cards, 'allocated_cards': allocated, 'allocated_cards_upper_bound': cards,
            'allocation_measurement': 'live_instances', 'status': detail.status}
    used = sum(row['requested_cards'] for row in records.values())
    original.require(used + requested <= 256, 'Observable account Ascend GPU ceiling would be exceeded')
    return {'schema': BUDGET_SCHEMA, 'time': time.time(), 'records': records,
        'observable_requested_cards': used, 'observable_allocated_cards': None,
        'observable_allocated_cards_upper_bound': used, 'allocation_measurement': 'not_fully_measured',
        'requested_cards': requested, 'observable_total_with_request': used + requested,
        'ceiling': 256, 'scan_counts': counters,
        'scope': 'current account visible active Ascend workloads across projects; unknown and queued states fully reserved'}


class FreshBudgetScanner:
    def __init__(self, receipt_path):
        self.receipt_path = Path(receipt_path)
        self.last = None

    def __call__(self, client, requested):
        from inspire.platform.web.browser_api.jobs import get_current_user
        with client._transport.scope(timeout=20):
            user_id = get_current_user(session=client._transport.session)['id']
        result = requested_card_budget(client, requested, user_id)
        result.update(operational_source_sha256=SOURCE_SHA,
                      readiness_detail_and_instances_unchanged=True)
        original.atomic_json(self.receipt_path, result)
        self.last = result
        return result


class ServingReads:
    """Fresh detail/instances, stable validated references, bounded event cache.

    A reference is reused only after this very SDK client resolved the name in
    the requested workspace. The SDK still validates its account and base URL.
    Event cache is only used while the latest fresh detail says RUNNING; stopped
    or pending services always fetch current events before recovery decisions.
    """
    def __init__(self, raw, events_ttl=60):
        self.raw = raw
        self.events_ttl = events_ttl
        self.refs = {}
        self.workspaces = {}
        self.statuses = {}
        self.events_cache = {}
        self.counts = {'fresh_details': 0, 'fresh_instances': 0,
                       'fresh_events': 0, 'cached_events': 0, 'reference_hits': 0}

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _kwargs(self, ref, kwargs):
        kwargs = dict(kwargs)
        workspace = kwargs.get('workspace')
        key = getattr(ref, 'key', None)
        if key in self.workspaces and workspace == self.workspaces[key]:
            kwargs.pop('workspace', None)
        return kwargs

    def get(self, ref, **kwargs):
        selector = (kwargs.get('workspace'), ref) if isinstance(ref, str) else None
        resolved = self.refs.get(selector, ref)
        if resolved is not ref:
            self.counts['reference_hits'] += 1
        result = self.raw.get(resolved, **self._kwargs(resolved, kwargs))
        self.counts['fresh_details'] += 1
        if selector:
            self.refs[selector] = result.ref
            self.workspaces[result.ref.key] = selector[0]
        self.statuses[result.ref.key] = result.status
        if result.status != 'RUNNING':
            self.events_cache.pop(result.ref.key, None)
        return result

    def instances(self, ref, **kwargs):
        result = self.raw.instances(ref, **self._kwargs(ref, kwargs))
        self.counts['fresh_instances'] += 1
        return result

    def events(self, ref, **kwargs):
        key = getattr(ref, 'key', None)
        signature = json.dumps(kwargs, sort_keys=True, default=str)
        cached = self.events_cache.get(key)
        now = time.monotonic()
        if (self.statuses.get(key) == 'RUNNING' and cached is not None
                and cached[1] == signature and now - cached[0] < self.events_ttl):
            self.counts['cached_events'] += 1
            return cached[2]
        result = self.raw.events(ref, **self._kwargs(ref, kwargs))
        self.counts['fresh_events'] += 1
        if self.statuses.get(key) == 'RUNNING':
            self.events_cache[key] = (now, signature, result)
        return result


class ClientFacade:
    def __init__(self, raw, events_ttl):
        self.raw = raw
        self.servings = ServingReads(raw.servings, events_ttl)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ThreadClients:
    """One SDK client per observer thread; clients never cross thread boundaries."""
    def __init__(self, factory, events_ttl=60):
        self.factory = factory
        self.events_ttl = events_ttl
        self.local = threading.local()
        self.lock = threading.Lock()
        self.clients = []

    def __call__(self, *args, **kwargs):
        signature = (args, tuple(sorted(kwargs.items())))
        clients = getattr(self.local, 'clients', None)
        if clients is None:
            clients = self.local.clients = {}
        if signature not in clients:
            raw = self.factory(*args, **kwargs)
            raw.__enter__()
            facade = ClientFacade(raw, self.events_ttl)
            clients[signature] = facade
            with self.lock:
                self.clients.append(facade)
        return clients[signature]

    def statistics(self):
        result = {}
        with self.lock:
            for client in self.clients:
                for name, value in client.servings.counts.items():
                    result[name] = result.get(name, 0) + value
        return result


def checked_policy(path, config):
    raw = Path(path).read_bytes()
    if len(raw) > 16384:
        raise ValueError('Operational policy too large')
    value = json.loads(raw)
    original.require(value.get('schema') == SCHEMA, 'Wrong operational policy schema')
    original.require(value.get('config_sha256') == config['sha256'], 'Policy binds wrong pool config')
    for key, low, high in (
        ('max_cards', config['cards_per_replica'], config['max_cards']),
        ('target_replicas', 0, config['replicas']),
    ):
        original.require(type(value.get(key)) is int and low <= value[key] <= high, 'Invalid ' + key)
    original.require(value['target_replicas'] * config['cards_per_replica'] <= value['max_cards'],
                     'Operational target exceeds card ceiling')
    for key, low, high in (('poll_seconds', 8, 20), ('provision_seconds', 3, 60),
                           ('observer_start_interval', .1, .5), ('events_ttl_seconds', 30, 120)):
        original.require(type(value.get(key)) in (int, float) and low <= value[key] <= high,
                         'Invalid ' + key)
    return value, hashlib.sha256(raw).hexdigest()


class OperationalPool(original.BudgetPool):
    def __init__(self, config_path, policy_path, backend=None):
        super().__init__(config_path, backend)
        self.policy_path = Path(policy_path).resolve()
        self.original_limits = dict(self.config, sha256=self.config_sha)
        self.policy = None
        self.policy_sha = None
        self.policy_error = None
        self.client_factory = None
        self.budget_scanner = None
        self._observer_slots = threading.BoundedSemaphore(2)
        self.refresh_policy(required=True)
        if backend is None:
            # The frozen adapter's functions and binding remain exactly intact.
            # Its module-local factory is the sole SDK transport optimization.
            namespace = self.backend.observe.__func__.__globals__
            self.client_factory = ThreadClients(namespace['InspireClient'], self.policy['events_ttl_seconds'])
            namespace['InspireClient'] = self.client_factory
            original.require(original.audited_unsent_backend(self.backend, self.config['backend_file']) is not None,
                             'Operational wrapper must retain the audited backend identity')
            self.budget_scanner = FreshBudgetScanner(self.policy_path.parent / 'REVIEW_BUDGET_SCAN.json')
            # Both the frozen prepare and create functions use this same
            # imported pool module. Their model/ownership/single-send contracts
            # stay intact; only the admission-accounting implementation changes.
            namespace['pool'].observable_project_budget = self.budget_scanner

    def refresh_policy(self, required=False):
        try:
            policy, sha = checked_policy(self.policy_path, self.original_limits)
            self.policy, self.policy_sha, self.policy_error = policy, sha, None
            # These are scheduling settings only; self.config_sha remains the
            # immutable original file SHA used by every existing reservation.
            self.config['poll_seconds'] = policy['poll_seconds']
            self.config['provision_seconds'] = policy['provision_seconds']
            self._observer_start_interval = policy['observer_start_interval']
            if self.client_factory:
                self.client_factory.events_ttl = policy['events_ttl_seconds']
                with self.client_factory.lock:
                    for client in self.client_factory.clients:
                        client.servings.events_ttl = policy['events_ttl_seconds']
        except Exception as exc:
            self.policy_error = type(exc).__name__
            if required:
                raise

    def target(self):
        if self.policy_error:
            return 0
        return min(super().target(), self.policy['target_replicas'])

    def provision_one(self):
        self.refresh_policy()
        # Target bounds all new slots; lowering the policy never drops an
        # existing reservation or stops a running model.
        return super().provision_one()

    def status(self):
        result = super().status()
        result['operational_controller'] = {
            'source': str(SOURCE), 'source_sha256': SOURCE_SHA,
            'policy': str(self.policy_path), 'policy_sha256': self.policy_sha,
            'policy_error': self.policy_error,
            'effective_target_replicas': self.target(),
            'operational_card_ceiling': self.policy['max_cards'],
            'poll_seconds': self.config['poll_seconds'],
            'provision_seconds': self.config['provision_seconds'],
            'observer_concurrency': 2,
            'observer_start_interval': self._observer_start_interval,
            'ready_lease_seconds_unchanged': 30,
            'sdk_reads': self.client_factory.statistics() if self.client_factory else {},
            'budget_scan_schema': BUDGET_SCHEMA,
            'budget_scan': ({k: self.budget_scanner.last[k] for k in
                ('time', 'observable_requested_cards', 'observable_total_with_request', 'scan_counts')}
                if self.budget_scanner and self.budget_scanner.last else None),
        }
        return result

    def guard(self):
        command = [sys.executable, str(SOURCE), 'run', '--config', str(self.config_path),
                   '--policy', str(self.policy_path)]
        receipt_path = self.root / 'RUN_PROCESS.json'
        with original.locked(self.root / 'guard.lock', nonblocking=True):
            while not self.exit.is_set():
                stopped = self.root / 'STOPPED.json'
                if self.disabled() and stopped.exists() and original.read(stopped).get('complete'):
                    return
                receipt = original.read(receipt_path) if receipt_path.exists() else {}
                actual = original.process_identity(receipt.get('pid', -1))
                if not actual or actual['command'] != command or actual != receipt.get('identity'):
                    original.require(hashlib.sha256(self.config_path.read_bytes()).hexdigest() == self.config_sha,
                                     'Frozen config changed')
                    original.require(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA,
                                     'Operational source changed; migrate explicitly')
                    with (self.root / 'run.log').open('a') as stream:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True, cwd=os.getcwd())
                    identity = original.process_identity(child.pid)
                    original.require(identity is not None, 'Operational runner exited before identity capture')
                    original.atomic_json(receipt_path, {'pid': child.pid, 'identity': identity,
                        'time': time.time(), 'config_sha256': self.config_sha,
                        'operational_source_sha256': SOURCE_SHA})
                original.atomic_json(self.root / 'GUARD_HEARTBEAT.json',
                    {'identity': original.process_identity(os.getpid()), 'time': time.time(),
                     'operational_source_sha256': SOURCE_SHA})
                self.exit.wait(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'guard', 'status', 'validate'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--policy', required=True)
    args = parser.parse_args()
    pool = OperationalPool(args.config, args.policy)
    if args.action in ('run', 'guard'):
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: pool.exit.set())
        getattr(pool, args.action)()
    elif args.action == 'validate':
        # Local checks only: no observe/prepare/create or HTTP model call.
        rows = pool.active_rows()
        print(json.dumps({'valid': True, 'reserved_slots': len(rows), 'reserved_cards': sum(x['cards'] for x in rows),
            'config_sha256': pool.config_sha, 'policy_sha256': pool.policy_sha,
            'source_sha256': SOURCE_SHA, 'target_replicas': pool.target()}))
    else:
        print(json.dumps(pool.status()))


if __name__ == '__main__':
    main()
