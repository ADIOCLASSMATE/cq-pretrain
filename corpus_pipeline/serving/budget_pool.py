"""Budgeted native Serving lifecycle, independent of model and data queue.

A model backend owns immutable runtime binding, native resource identity and
per-boot readiness. This module owns the durable card reservations, exclusive
lifecycle, discovery and bounded concurrent observation. Unknown operations
retain their reservations and are reconciled, never blindly duplicated.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time

from corpus_pipeline.resource_manager.engine import atomic_json
from corpus_pipeline.serving import retirement
from corpus_pipeline.coordination import locked, available_lock

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def process_identity(pid):
    try:
        base = Path('/proc') / str(int(pid))
        stat = (base / 'stat').read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': int(pid), 'starttime_ticks': stat[19],
                'command': [v.decode() for v in (base / 'cmdline').read_bytes().split(b'\0') if v]}
    except (OSError, ValueError, IndexError):
        return None


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def audited_unsent_backend(backend, source):
    """Only the reviewed native adapter guarantees a pre-transmission marker.

    Absence of a marker means nothing for other backends. The approved adapter
    inherits the exact lifecycle create/reconcile implementations; subclasses,
    replacements and instance-level overrides do not opt in implicitly.
    """
    approved = Path(__file__).resolve().parent / 'qwen36_w8a8_20261024/official_v023/backend.py'
    if source is None or Path(source).resolve() != approved:
        return None
    try:
        from corpus_pipeline.serving.qwen36_w8a8_20261024 import backend as lifecycle
    except ImportError:
        return None
    cls = type(backend)
    if cls.__name__ != 'Backend' or cls.__bases__ != (lifecycle.Backend,):
        return None
    for name in ('create', 'reconcile'):
        if getattr(getattr(backend, name, None), '__func__', None) is not getattr(lifecycle.Backend, name):
            return None
    for name in ('prepare', 'binding', 'observe', 'publish'):
        function = getattr(getattr(backend, name, None), '__func__', None)
        if function is None or Path(function.__code__.co_filename).resolve() != approved:
            return None
    return lifecycle


def is_transport_error(error):
    try:
        from inspire.platform.errors import TransportError
    except ImportError:
        return False
    return isinstance(error, TransportError)


def safe_error_chain(error):
    """Diagnostic codes only: exception messages/args may contain credentials."""
    chain, seen = [], set()
    while error is not None and len(chain) < 4 and id(error) not in seen:
        seen.add(id(error))
        row = {'type': type(error).__name__}
        for name in ('status', 'status_code', 'errno'):
            value = getattr(error, name, None)
            if type(value) is int:
                row[name] = value
        chain.append(row)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return chain


def observation_interval(state, live_seconds=8):
    status = state.get('platform_status')
    if status == 'RUNNING': return live_seconds
    if status == 'DEPLOYING': return 20
    if status in ('FAILED', 'SUCCEEDED', 'DELETED'): return 120
    return 60


def reservation_total(rows, external_cards=0, *, extra=0, ceiling=128):
    """STOPPED/preempted/unknown slots stay reserved until deliberately retired."""
    require(type(external_cards) is int and external_cards >= 0, 'Invalid external reservation')
    require(type(extra) is int and extra >= 0, 'Invalid additional reservation')
    names = [row['name'] for row in rows]
    ids = [row['service_id'] for row in rows if row.get('service_id')]
    require(len(names) == len(set(names)) and len(ids) == len(set(ids)), 'Duplicate resource identity')
    require(all(type(row['cards']) is int and row['cards'] > 0 for row in rows), 'Invalid cards')
    total = sum(row['cards'] for row in rows) + external_cards + extra
    require(total <= ceiling, 'Serving card budget exceeded')
    return total


class BudgetPool:
    def __init__(self, config_path, backend=None):
        self.config_path = Path(config_path).resolve()
        self.config = read(self.config_path)
        cfg = self.config
        require(cfg.get('schema_version') == 'corpus.native_budget_pool.v1', 'Unknown pool config')
        self.config_sha = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.root = Path(cfg['root'])
        require(self.root.is_absolute(), 'Absolute pool root required')
        for key in ('max_cards', 'cards_per_replica', 'replicas', 'per_replica_max_inflight'):
            require(type(cfg[key]) is int and cfg[key] > 0, 'Invalid ' + key)
        require(cfg['replicas'] * cfg['cards_per_replica'] <= cfg['max_cards'], 'Replica target exceeds budget')
        require(type(cfg['priority']) is int and cfg['priority'] in (1, 3),
                'Production pool requires LOW priority 1 or legacy 3')
        require(cfg.get('poll_seconds', 8) >= 2, 'Excessive platform polling')
        self.root.mkdir(parents=True, exist_ok=True)
        if backend is None:
            source = Path(cfg['backend_file'])
            require(source.is_absolute(), 'Absolute backend path required')
            spec = importlib.util.spec_from_file_location('corpus_serving_model_backend', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            backend = module.Backend(**cfg.get('backend_kwargs', {}))
        self.backend = backend
        self.exit = threading.Event()
        self.workers = {}
        self._observer_slots = threading.BoundedSemaphore(8)
        self._waiting_observers = threading.BoundedSemaphore(1)
        self._observation_role = threading.local()
        self._observer_start_lock = threading.Lock()
        self._observer_start_interval = 0.35
        self._observer_last_start = float('-inf')

    def disabled(self):
        return (self.root / 'DISABLED').exists()

    def target(self):
        path = self.root / 'TARGET.json'
        target = read(path)['replicas'] if path.exists() else self.config['replicas']
        require(type(target) is int and 0 <= target <= self.config['replicas'], 'Invalid replica target')
        return target

    def _backoff_state(self):
        path = self.root / 'PROVISION_BACKOFF.json'
        if not path.exists():
            return {'schema_version': 'corpus.provision_backoff.v1', 'root': str(self.root),
                    'config_sha256': self.config_sha, 'consecutive_429': 0, 'rate_limit_events': 0,
                    'not_before_epoch': 0}
        state = read(path)
        require(state.get('schema_version') == 'corpus.provision_backoff.v1'
                and state.get('root') == str(self.root) and state.get('config_sha256') == self.config_sha,
                'Provision backoff scope changed')
        for key in ('consecutive_429', 'rate_limit_events'):
            require(type(state.get(key)) is int and state[key] >= 0, 'Invalid provision backoff counter')
        deadline = state.get('not_before_epoch')
        require(type(deadline) in (int, float) and 0 <= deadline < float('inf'),
                'Invalid provision backoff deadline')
        return state

    def provisioning_delayed(self):
        # Call again after acquiring provision.lock; a prior contender may have
        # established a cooldown while this observer was waiting for its permit.
        return self._backoff_state()['not_before_epoch'] > time.time()

    def record_rate_limit(self, error):
        chain = safe_error_chain(error)
        if not any(row.get('status') == 429 or row.get('status_code') == 429 for row in chain):
            return False
        with locked(self.root / 'provision_backoff.lock'):
            state = self._backoff_state()
            now = time.time()
            state['rate_limit_events'] += 1
            state['last_rate_limit_at'] = now
            state['last_cause_chain'] = chain
            if state['not_before_epoch'] <= now:
                # A synchronized burst counts as one backoff step. Further
                # failures after that cooldown grow the delay, at most 15 min.
                state['consecutive_429'] += 1
                base = min(900, 60 * 2 ** min(state['consecutive_429'] - 1, 4))
                delay = min(900, base + random.uniform(0, base * 0.2))
                state.update(not_before_epoch=now + delay, cooldown_seconds=delay)
            atomic_json(self.root / 'PROVISION_BACKOFF.json', state)
        return True

    @contextmanager
    def provisioning_attempt(self):
        """Record rate limiting before releasing the allocation lock."""
        try:
            yield
        except Exception as error:
            self.record_rate_limit(error)
            raise

    def provisioning_succeeded(self):
        path = self.root / 'PROVISION_BACKOFF.json'
        if not path.exists():
            return
        with locked(self.root / 'provision_backoff.lock'):
            state = self._backoff_state()
            now = time.time()
            # A concurrent observer may have found a newer 429 while Create
            # was in flight. A successful reply must not erase that cooldown.
            if state['not_before_epoch'] > now:
                return
            state.update(consecutive_429=0, not_before_epoch=0, last_success_at=now)
            atomic_json(path, state)

    def ledger(self):
        path = self.root / 'RESERVATIONS.json'
        marker = self.root / 'STATE_REQUIRED.json'
        if marker.exists():
            require(read(marker)['config_sha256'] == self.config_sha, 'Pool config identity changed')
            require(path.exists(), 'Required reservation ledger missing; refusing empty budget')
        if path.exists():
            value = read(path)
            require(value['config_sha256'] == self.config_sha and value['root'] == str(self.root),
                    'Pool reservation scope changed')
            return value
        return {'config_sha256': self.config_sha, 'root': str(self.root), 'slots': {}, 'created_at': time.time()}

    def save_ledger(self, ledger):
        # A crash between marker and ledger intentionally fails closed.
        atomic_json(self.root / 'STATE_REQUIRED.json', {'config_sha256': self.config_sha})
        atomic_json(self.root / 'RESERVATIONS.json', ledger)

    def directory(self, number):
        return self.root / 'slots' / f'{int(number):02}'

    def state(self, number, row=None):
        row = row or self.ledger()['slots'][str(number)]
        path = self.directory(number) / 'state.json'
        # A reserved but never-sent allocation is retained even after a crash.
        state = read(path) if path.exists() else dict(row['intent'])
        require(state['name'] == row['name'] and state['cards'] == row['cards']
                and state.get('pool_config_sha256') == self.config_sha,
                'Slot reservation identity changed')
        require(state['cards'] == self.config['cards_per_replica'] and state['priority'] == self.config['priority'],
                'Slot topology or priority changed')
        return state

    def save_state(self, number, state):
        # Endpoints are only published through the model backend's private registry.
        atomic_json(self.directory(number) / 'state.json',
                    {key: value for key, value in state.items() if key not in ('endpoint', 'detail')})

    def retired(self, number, row=None, state=None):
        row = row or self.ledger()['slots'][str(number)]
        committed = row.get('retirement')
        if not committed:
            return False
        state = state or self.state(number, row)
        require(all(state.get(key) == row['intent'].get(key)
                    for key in ('name','cards','pool_scope','pool_slot','binding')),
                'Retirement original reservation binding changed')
        directory = self.directory(number)
        evidence = directory / 'retirement'
        path = evidence / 'COMMITTED.json'
        require(committed.get('receipt') == str(path)
                and committed.get('sha256') == hashlib.sha256(path.read_bytes()).hexdigest(),
                'Retirement ledger commitment changed')
        return retirement.verify(state, read(path),
            (evidence / 'REQUEST.json').read_bytes(), (evidence / 'STOPPED_SNAPSHOT.json').read_bytes(),
            read(directory / 'STOPPED.json'), disabled=(directory / 'DISABLED').exists())

    def accounting(self, ledger=None):
        ledger = ledger or self.ledger()
        rows = [self.state(n, row) for n, row in ledger['slots'].items()]
        for n, state in zip(ledger['slots'], rows):
            require(state.get('pool_slot') == int(n), 'Reservation slot identity changed')
        retired = {state['pool_slot'] for (n, row), state in zip(ledger['slots'].items(), rows)
                   if self.retired(n, row, state)}
        counts = retirement.summary(rows, retired, ceiling=self.config['max_cards'])
        return rows, retired, counts

    def active_rows(self, ledger=None):
        rows, retired, _ = self.accounting(ledger)
        return [state for state in rows if state['pool_slot'] not in retired]

    def unsent_allowed(self, number):
        rows, retired, _ = self.accounting()
        sent = {state['pool_slot'] for state in rows if state.get('service_id')
                or (self.directory(state['pool_slot']) / 'CREATE_TRANSMISSION.json').exists()}
        return int(number) in retirement.eligible_unsent(rows, retired, sent, self.target())

    def retire(self, number):
        """Explicitly retire a released slot without deleting its cleanup ledger.

        No Stop/Start/Create is issued here. A fresh owned native STOPPED and
        terminal-instance observation must precede the budget exemption.
        """
        with locked(self.root / 'provision.lock'), locked(self.directory(number) / 'lifecycle.lock'):
            require(not self.disabled(), 'Cannot retire while whole-pool cleanup is active')
            ledger = self.ledger()
            row = ledger['slots'][str(number)]
            state = self.state(number, row)
            if self.retired(number, row, state):
                return read(self.directory(number) / 'retirement/COMMITTED.json')
            directory = self.directory(number)
            require((directory / 'DISABLED').exists() and state.get('service_id'),
                    'Explicitly disabled, known native resource required')
            require(retirement.terminal(state, read(directory / 'STOPPED.json')),
                    'Existing native release evidence required')
            lifecycle = audited_unsent_backend(self.backend, self.config.get('backend_file'))
            require(lifecycle is not None, 'Only audited native adapter may retire reservations')
            require(self.backend.binding()[1] == state['binding'], 'Retirement runtime binding changed')
            evidence = directory / 'retirement'
            request_path = evidence / 'REQUEST.json'
            if not request_path.exists():
                atomic_json(request_path, {'schema_version':'corpus.slot_retirement_request.v1',
                    'retire':True, **{k:state[k] for k in retirement.IDENTITY}, 'requested_at':time.time()})
            with lifecycle.InspireClient(account='wjx-ascend') as client:
                state, detail = self.backend.reconcile(client, state)
                instances = client.servings.instances(detail.ref, workspace=lifecycle.native.WORKSPACE)
                statuses = [instance.status for instance in instances]
                require(detail.status == 'STOPPED' and bool(statuses)
                        and all(value in retirement.TERMINAL for value in statuses),
                        'Retirement resource is not freshly terminal')
                observed = {'ownership_verified':True, 'service_id':state['service_id'],
                    'status':detail.status, 'instance_statuses':statuses, 'observed_at':time.time()}
            state.update(platform_status='STOPPED', phase='RELEASED', release_verified=True,
                         canary_passed=False, instance_statuses=statuses,
                         last_observed_at=observed['observed_at'])
            self.save_state(number, state)
            atomic_json(directory / 'STOPPED.json', {'complete':True, 'service_id':state['service_id'],
                'status':'STOPPED', 'instance_statuses':statuses, 'time':observed['observed_at']})
            snapshot_path = evidence / 'STOPPED_SNAPSHOT.json'
            committed_path = evidence / 'COMMITTED.json'
            # A crash before ledger commit can safely re-use the hash-bound
            # completed proof. Never overwrite evidence of an existing commit.
            if committed_path.exists():
                proof = read(committed_path)
                retirement.verify(state, proof, request_path.read_bytes(), snapshot_path.read_bytes(),
                                  read(directory / 'STOPPED.json'), disabled=True)
            else:
                atomic_json(snapshot_path, read(directory / 'STOPPED.json'))
                proof = retirement.commit(state, request_path.read_bytes(), snapshot_path.read_bytes(),
                                          observed, disabled=True, now=time.time())
                atomic_json(committed_path, proof)
            self.backend.publish(directory, state, False)
            row['retirement'] = {'receipt':str(committed_path),
                'sha256':hashlib.sha256(committed_path.read_bytes()).hexdigest()}
            self.save_ledger(ledger)
            self.publish_index(ledger)
            return proof

    def external_cards(self):
        total = 0
        for external in self.config.get('external_reservations', []):
            path = Path(external['state'])
            require(path.is_absolute() and type(external['cards']) is int and external['cards'] > 0,
                    'Invalid external reservation')
            total += external['cards']  # Missing/ambiguous evidence always holds the reservation.
            if not path.exists():
                continue
            state = read(path)
            require(state['cards'] == external['cards'], 'External resource size changed')
            if external.get('service_id'):
                require(state.get('service_id') == external['service_id'], 'External resource identity changed')
            if (path.parent / 'DISABLED').exists() and state.get('release_verified') is True:
                if state.get('phase') == 'RELEASED' and self.backend.released(state):
                    total -= external['cards']
        return total

    def publish_index(self, ledger=None):
        ledger = ledger or self.ledger()
        limit = self.config['per_replica_max_inflight']
        override = self.root / 'REQUEST_LIMIT.json'
        if override.exists():
            requested = read(override)['max_inflight']
            require(type(requested) is int and 1 <= requested <= limit, 'Invalid request limit')
            limit = requested
        registries = [{'path': str(self.directory(n) / 'registry.json'), 'weight': 1, 'max_inflight': limit}
                      for n in sorted(ledger['slots'], key=int) if not self.retired(n, ledger['slots'][n])]
        for ext in self.config.get('external_reservations', []):
            if ext.get('drain_marker') and Path(ext['drain_marker']).exists():
                continue  # Stop new traffic without releasing its card reservation.
            registries.append({'path': ext['registry'], 'weight': 1, 'max_inflight': limit})
        atomic_json(self.root / 'pool.json', {'schema_version': 'corpus.screening_pool.v1',
            'max_cards': self.config['max_cards'], 'cards_per_replica': self.config['cards_per_replica'],
            'config_sha256': self.config_sha, 'registries': registries})

    def provision_one(self):
        with locked(self.root / 'provision.lock'), self.provisioning_attempt():
            require(not self.disabled(), 'Pool disabled')
            if self.provisioning_delayed():
                return False
            ledger = self.ledger()
            rows = self.active_rows(ledger)
            target = self.target()
            if len(rows) >= target:
                return False  # No allocation: do not repeatedly query released external HIGH resources.
            external = self.external_cards()
            used = reservation_total(rows, external, ceiling=self.config['max_cards'])
            if used + self.config['cards_per_replica'] > self.config['max_cards']:
                return False
            number = max(map(int, ledger['slots']), default=0) + 1
            directory = self.directory(number)
            name = self.config['name_prefix'] + f'-{number:02}'
            state = self.backend.prepare(directory, number, name, self.config['priority'])
            require(state['cards'] == self.config['cards_per_replica'] and state['name'] == name
                    and state['priority'] == self.config['priority'], 'Prepared topology changed')
            state.update(pool_config_sha256=self.config_sha, pool_scope=str(self.root), pool_slot=number)
            require(not self.disabled(), 'Pool disabled during planning')
            if self.provisioning_delayed():
                return False  # A ready observer may have seen 429 during the preflight reads.
            reservation_total(rows, external, extra=state['cards'], ceiling=self.config['max_cards'])
            ledger['slots'][str(number)] = {'name': name, 'cards': state['cards'], 'intent': state,
                                           'reserved_at': time.time()}
            self.save_ledger(ledger)
            atomic_json(directory / 'CREATE_INTENT.json', state)
            self.save_state(number, state)
            self.publish_index(ledger)
            with locked(directory / 'lifecycle.lock'):
                require(not self.disabled(), 'Pool disabled before Create; reservation retained')
                if self.provisioning_delayed():
                    return False
                try:
                    state = self.backend.create(directory, state)
                except Exception as error:
                    stamp = time.time()
                    state.update(pool_error_type=type(error).__name__, pool_error_stage='CREATE_UNKNOWN',
                                 pool_error_at=stamp, pool_last_error={'type': type(error).__name__,
                                 'stage': 'CREATE_UNKNOWN', 'at': stamp, 'cause_chain': safe_error_chain(error)})
                    self.save_state(number, state)
                    raise
                self.save_state(number, state)
                self.provisioning_succeeded()
            return True

    def recover_unsent(self, number, state):
        """Retry a proven *unsent* intent in its existing reserved slot only.

        Caller holds this slot's lifecycle lock. The native create implementation
        rechecks the account-wide 256-card guard, immutable binding and DISABLED,
        then durably writes CREATE_TRANSMISSION before its single SDK write.
        """
        directory = self.directory(number)
        marker = directory / 'CREATE_TRANSMISSION.json'
        if state.get('service_id') or marker.exists() or self.disabled():
            return state
        lifecycle = audited_unsent_backend(self.backend, self.config.get('backend_file'))
        if lifecycle is None:
            return state
        with available_lock(self.root / 'provision.lock') as acquired, self.provisioning_attempt():
            if not acquired:
                return None  # Defer this slot without holding an observer permit waiting for allocation.
            if self.provisioning_delayed():
                return None  # Proven-unsent slots must not enumerate names during cooldown either.
            state = self.state(number)
            if (state.get('service_id') or marker.exists() or self.disabled()
                    or (directory / 'DISABLED').exists() or not self.unsent_allowed(number)):
                return state
            ledger = self.ledger()
            rows = self.active_rows(ledger)
            reservation_total(rows, self.external_cards(), ceiling=self.config['max_cards'])
            intent = read(directory / 'CREATE_INTENT.json')
            require(all(intent.get(key) == state.get(key) for key in
                        ('name', 'cards', 'operation_id', 'binding')), 'Unsent Create intent changed')
            self.backend.validate(state)
            require(self.backend.binding()[1] == state['binding'], 'Unsent runtime binding changed')
            with lifecycle.InspireClient(account='wjx-ascend') as client:
                matches = [row for row in client.servings.iter(workspace=lifecycle.native.WORKSPACE)
                           if row.name == state['name']]
            if matches:
                # The normal exact-identity reconciliation handles any existing
                # resource, including ambiguous or incompatible matches.
                return state
            if (self.disabled() or (directory / 'DISABLED').exists() or marker.exists()
                    or not self.unsent_allowed(number)):
                return state
            if self.provisioning_delayed():
                return None
            state['pool_unsent_recovery_at'] = time.time()
            self.save_state(number, state)
            state = self.backend.create(directory, state)
            self.save_state(number, state)
            self.provisioning_succeeded()
            return state

    def preserve_transport_lease(self, directory, state, error):
        """A transient read failure may consume, but never renew, a valid lease."""
        if (not is_transport_error(error) or self.disabled() or (directory / 'DISABLED').exists()
                or state.get('platform_status') != 'RUNNING' or not state.get('service_id')
                or not state.get('incarnation') or not state.get('deployment_id')):
            return False
        try:
            entries = read(directory / 'registry.json')['endpoints']
            now = time.time()
            return bool(entries) and all(
                row.get('ready') is True and type(row.get('ready_expires_at')) in (int, float)
                and row['ready_expires_at'] > now
                and row.get('service_id') == state['service_id']
                and row.get('deployment_id') == state.get('deployment_id')
                and row.get('incarnation') == state.get('incarnation')
                for row in entries)
        except (OSError, ValueError, KeyError, TypeError):
            return False

    @contextmanager
    def observation_permit(self):
        """Bound management bursts before taking any resource lifecycle lock.

        The start limiter also staggers a restarted pool's existing observers.
        Waiting never holds provision/lifecycle locks and shutdown interrupts it.
        """
        acquired = False
        semaphore = self._observer_slots if getattr(self._observation_role, 'live', False) else self._waiting_observers
        while not self.exit.is_set():
            if semaphore.acquire(timeout=0.1):
                acquired = True
                break
        if not acquired:
            yield False
            return
        try:
            permitted = False
            while not self.exit.is_set():
                with self._observer_start_lock:
                    now = time.monotonic()
                    delay = self._observer_last_start + self._observer_start_interval - now
                    if delay <= 0:
                        self._observer_last_start = now
                        permitted = True
                        break
                if self.exit.wait(delay):
                    break
            yield permitted
        finally:
            semaphore.release()

    def observe_one(self, number):
        self._observation_role.live = self.state(number).get('platform_status') == 'RUNNING'
        with self.observation_permit() as permitted:
            if permitted:
                return self._observe_one(number)
        return None

    def _observe_one(self, number):
        directory = self.directory(number)
        with available_lock(directory / 'lifecycle.lock') as acquired:
            if not acquired:
                return None  # Create/Stop owns this slot; preserve its state and registry untouched.
            state = self.state(number)
            if self.retired(number, state=state):
                return state  # Explicitly retired resources cannot be restarted or requalified.
            stage = 'UNSENT_RECOVERY'
            try:
                state = self.recover_unsent(number, state)
                if state is None:
                    return None
                stage = 'OBSERVE'
                state = self.backend.observe(directory, state)
                event = self.backend.can_restart(state)
                if event and not self.disabled():
                    # The same reservation and native service ID are reused.
                    stage = 'RESTART'
                    with available_lock(self.root / 'provision.lock') as acquired, self.provisioning_attempt():
                        if acquired and not self.disabled() and not self.provisioning_delayed():
                            rows = self.active_rows()
                            reservation_total(rows, self.external_cards(), ceiling=self.config['max_cards'])
                            if not self.provisioning_delayed():
                                state = self.backend.restart(directory, state, event)
                    stage = 'OBSERVE'
                state.pop('pool_error_type', None)
                state.pop('pool_error_stage', None)
                state.pop('pool_registry_lease_preserved', None)
            except Exception as error:
                stamp = time.time()
                if stage == 'OBSERVE':
                    self.record_rate_limit(error)
                keep = stage == 'OBSERVE' and self.preserve_transport_lease(directory, state, error)
                state.update(pool_error_type=type(error).__name__, pool_error_stage=stage,
                             pool_error_at=stamp, pool_registry_lease_preserved=keep,
                             pool_last_error={'type': type(error).__name__, 'stage': stage, 'at': stamp,
                                              'cause_chain': safe_error_chain(error)})
                if not keep:
                    self.backend.publish(directory, state, False)
            if self.disabled():
                self.backend.publish(directory, state, False)
            state['pool_observed_at'] = time.time()
            self.save_state(number, state)
        return state

    def worker(self, number):
        while not self.exit.is_set() and not self.disabled():
            started = time.monotonic()
            try:
                self.observe_one(number)
            except Exception as error:
                atomic_json(self.directory(number) / 'OBSERVER_ERROR.json',
                            {'error_type': type(error).__name__, 'time': time.time()})
            interval = observation_interval(self.state(number), self.config.get('poll_seconds', 8))
            self.exit.wait(max(1, interval - (time.monotonic() - started)))

    def start_observers(self, ledger):
        for number in ledger['slots']:
            if number not in self.workers or not self.workers[number].is_alive():
                worker = threading.Thread(target=self.worker, args=(number,), daemon=True)
                self.workers[number] = worker
                worker.start()

    def status(self):
        ledger = self.ledger()
        rows, retired, counts = self.accounting(ledger)
        backoff = self._backoff_state()
        fields = ('name', 'service_id', 'phase', 'platform_status', 'release_verified',
                  'pool_error_type', 'pool_error_stage', 'pool_error_at', 'pool_last_error',
                  'pool_registry_lease_preserved', 'pool_observed_at', 'readiness_reason', 'canary_passed')
        return {'time': time.time(), 'pid': os.getpid(), 'config_sha256': self.config_sha,
                'loaded_code_sha256': CODE_SHA256,
                'observation_policy': {'live_permits': 8, 'waiting_permits': 1,
                    'pending_poll_seconds': 60, 'terminal_poll_seconds': 120,
                    'readiness_lease_extended': False},
                'max_cards': self.config['max_cards'], 'slot_reserved_cards': counts['active_reserved_cards'],
                'reservation_accounting':counts, 'target_replicas':self.target(),
                'reservation_accounting_scope':'owned slots only; external reservations checked live before allocation',
                'external_reserved_cards_max': sum(x['cards'] for x in self.config.get('external_reservations', [])),
                'provision_not_before_epoch': backoff['not_before_epoch'],
                'provision_consecutive_429': backoff['consecutive_429'],
                'disabled': self.disabled(), 'services': [
                    {**{k:r.get(k) for k in fields}, 'retired':r['pool_slot'] in retired} for r in rows]}

    def cancel_never_sent(self, number, state):
        """Cancel only a reviewed adapter's demonstrably never-sent intent.

        Caller owns cleanup.lock and the slot lifecycle lock after the DISABLED
        provision barrier. This is not a platform STOPPED observation, and its
        original reservation remains in the disabled pool ledger.
        """
        directory = self.directory(number)
        marker = directory / 'CREATE_TRANSMISSION.json'
        if state.get('service_id') or marker.exists():
            return None
        lifecycle = audited_unsent_backend(self.backend, self.config.get('backend_file'))
        if lifecycle is None:
            return None
        require(self.disabled(), 'Never-sent cancellation requires a disabled pool')
        intent_path = directory / 'CREATE_INTENT.json'
        intent_bytes = intent_path.read_bytes()
        intent = json.loads(intent_bytes)
        reserved = self.ledger()['slots'][str(number)]['intent']
        require(not intent.get('service_id') and not reserved.get('service_id'),
                'Never-sent intent contains a service identity')
        # An approved control-only pool-config migration can leave the original
        # Create intent intact. Keep its exact hash; all native create bindings
        # must still agree, and the live reservation must bind current config.
        keys = ('name', 'cards', 'priority', 'operation_id', 'binding', 'pool_scope', 'pool_slot')
        require(all(intent.get(key) == state.get(key) == reserved.get(key) for key in keys),
                'Never-sent cancellation binding changed')
        require(state.get('pool_config_sha256') == reserved.get('pool_config_sha256') == self.config_sha,
                'Never-sent reservation config changed')
        self.backend.validate(state)
        require(self.backend.binding()[1] == state['binding'], 'Never-sent runtime binding changed')
        with lifecycle.InspireClient(account='wjx-ascend') as client:
            matches = [row for row in client.servings.iter(workspace=lifecycle.native.WORKSPACE)
                       if row.name == state['name']]
        if matches:
            return None  # Existing resources still require normal exact identity reconciliation/Stop.
        require(self.disabled() and not marker.exists() and not self.state(number).get('service_id'),
                'Never-sent cancellation changed during inspection')
        require(intent_path.read_bytes() == intent_bytes, 'Never-sent intent changed during inspection')
        proof = {'schema_version': 'corpus.cancelled_never_sent.v1', 'cancelled_never_sent': True,
                 'pool_root': str(self.root), 'pool_config_sha256': self.config_sha, 'slot': int(number),
                 'name': state['name'], 'operation_id': state['operation_id'], 'binding': state['binding'],
                 'create_intent_sha256': hashlib.sha256(intent_bytes).hexdigest(),
                 'original_intent_pool_config_sha256': intent.get('pool_config_sha256'),
                 'transmission_marker_absent': True, 'live_exact_name_matches': 0,
                 'platform_stop_claimed': False, 'time': time.time()}
        atomic_json(directory / 'CANCELLED_NEVER_SENT.json', proof)
        state = {**state, 'phase': 'CANCELLED_UNSENT', 'cancelled_never_sent': True,
                 'release_verified': True, 'canary_passed': False,
                 'readiness_reason': 'CANCELLED_NEVER_SENT', 'cancelled_at': proof['time']}
        self.backend.publish(directory, state, False)
        return state

    def stop_once(self):
        (self.root / 'DISABLED').write_text('Explicit pool stop requested\n')
        self.exit.set()
        with locked(self.root / 'cleanup.lock'):
            return self._stop_once()

    def _stop_once(self):
        with locked(self.root / 'provision.lock'):
            numbers = list(self.ledger()['slots'])
        results = []
        # Native stop calls are independent. Preserve a separate slot lock for
        # each in-flight observer and continue cleanup if one identity fails.
        from concurrent.futures import ThreadPoolExecutor
        start_lock = threading.Lock()
        last_start = [float('-inf')]
        cleanup_wait = threading.Event()  # exit is already set; cleanup must still run.
        def stop_slot(number):
            with start_lock:
                delay = last_start[0] + 0.35 - time.monotonic()
                if delay > 0:
                    cleanup_wait.wait(delay)
                last_start[0] = time.monotonic()
            with locked(self.directory(number) / 'lifecycle.lock'):
                try:
                    previous = self.state(number)
                    state = self.cancel_never_sent(number, previous)
                    if state is None:
                        state = self.backend.stop(self.directory(number), previous)
                    self.save_state(number, state)
                    return {'slot': number, 'released': state.get('release_verified') is True,
                            'cancelled_never_sent': state.get('cancelled_never_sent') is True}
                except Exception as error:
                    return {'slot': number, 'released': False, 'error_type': type(error).__name__,
                            'cause_chain': safe_error_chain(error)}
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(stop_slot, numbers))
        result = {'complete': all(row['released'] for row in results), 'slots': results, 'time': time.time(),
                  'external_reservations_owned': False}
        atomic_json(self.root / 'STOPPED.json', result)
        return result

    def run(self):
        with locked(self.root / 'run.lock', nonblocking=True):
            while not self.exit.is_set():
                if self.disabled():
                    if self.stop_once()['complete']:
                        return
                    self.exit.clear()
                    self.exit.wait(5)
                    continue
                # Resume existing leases before potentially slow fresh budget
                # scans/planning/Create. New reservations are started below.
                self.start_observers(self.ledger())
                error = None
                try:
                    self.provision_one()
                except Exception as exception:
                    error = type(exception).__name__
                ledger = self.ledger()
                self.publish_index(ledger)
                self.start_observers(ledger)
                atomic_json(self.root / 'STATUS.json', {**self.status(), 'provision_error_type': error})
                self.exit.wait(self.config.get('provision_seconds', 3))

    def guard(self):
        command = [sys.executable, '-m', 'corpus_pipeline.serving.budget_pool', 'run',
                   '--config', str(self.config_path)]
        receipt_path = self.root / 'RUN_PROCESS.json'
        with locked(self.root / 'guard.lock', nonblocking=True):
            while not self.exit.is_set():
                stopped = self.root / 'STOPPED.json'
                if self.disabled() and stopped.exists() and read(stopped).get('complete'):
                    return
                receipt = read(receipt_path) if receipt_path.exists() else {}
                actual = process_identity(receipt.get('pid', -1))
                if not actual or actual['command'] != command or actual != receipt.get('identity'):
                    require(hashlib.sha256(self.config_path.read_bytes()).hexdigest() == self.config_sha,
                            'Guard config changed')
                    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == CODE_SHA256,
                            'Guard code changed; reload explicitly')
                    with (self.root / 'run.log').open('a') as stream:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            cwd=Path(__file__).resolve().parents[2])
                    identity = process_identity(child.pid)
                    require(identity is not None, 'Pool runner exited before identity capture')
                    atomic_json(receipt_path, {'pid': child.pid, 'identity': identity,
                                              'time': time.time(), 'config_sha256': self.config_sha})
                atomic_json(self.root / 'GUARD_HEARTBEAT.json',
                            {'identity': process_identity(os.getpid()), 'time': time.time()})
                self.exit.wait(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'guard', 'stop', 'status', 'index', 'retire'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--slot', type=int)
    args = parser.parse_args()
    pool = BudgetPool(args.config)
    if args.action in ('run', 'guard'):
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: pool.exit.set())
        getattr(pool, args.action)()
    elif args.action == 'stop':
        print(json.dumps(pool.stop_once(), ensure_ascii=False))
    elif args.action == 'index':
        pool.publish_index()
    elif args.action == 'retire':
        require(args.slot is not None and args.slot > 0, 'Retire requires a positive --slot')
        print(json.dumps(pool.retire(args.slot), ensure_ascii=False))
    else:
        print(json.dumps(pool.status(), ensure_ascii=False))


if __name__ == '__main__':
    main()
