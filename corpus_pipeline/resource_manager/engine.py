"""Durable, platform-neutral controller for bounded accelerator workloads.

The adapter owns platform identity validation and API timeouts. This engine
owns single-send intents, reservations, argv lifecycle, budgets and recovery.
No corpus-specific modules or dataset assumptions are used here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Protocol
import uuid

from .models import WorkloadSpec, NodeCapacity, Quota, Placement, Reservation
from .planning import plan_allocations, plan_consolidation, available_nodes


class Adapter(Protocol):
    capabilities: dict
    def snapshot(self) -> dict: ...
    def create(self, intent: dict) -> str: ...
    def reconcile(self, intent: dict) -> str | None: ...
    def observe(self, resource_id: str) -> dict: ...
    def stop(self, resource_id: str, operation_id: str) -> None: ...
    def launch(self, resource_id: str, command: list[str], operation_id: str) -> None: ...
    def health(self, resource_id: str, readiness_command: list[str] | None) -> dict: ...
    def drain(self, resource_id: str, command: list[str], operation_id: str) -> None: ...
    def completion(self, resource_id: str, command: list[str]) -> bool: ...


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def spec_identity(spec):
    return {key: value for key, value in asdict(spec).items()
            if key not in ('max_cards', 'poll_seconds')}


def summary(state):
    rows = list(state.get('resources', {}).values())
    reserved = [row for row in rows if not (row.get('retired_verified') and row.get('state') == 'STOPPED')]
    return {'scope': state.get('scope'), 'profile': state.get('profile'),
            'phase': state.get('phase', 'new'), 'max_cards': state.get('max_cards'),
            'reserved_cards': sum(row['intent']['placement']['quota']['cards'] for row in reserved),
            'resources': len(reserved), 'running': sum(row.get('state') == 'RUNNING' for row in reserved),
            'ready': sum(row.get('ready') is True for row in reserved),
            'unknown_creates': sum(not row.get('resource_id') for row in reserved),
            'action_required': [key for key, row in state.get('resources', {}).items()
                                if row.get('action_required')],
            'updated_at': state.get('updated_at'), 'migration': state.get('migration')}


class Controller:
    def __init__(self, adapter: Adapter, spec: WorkloadSpec, state_dir, *,
                 clock=time.time, max_startup_failures=5, migration_cooldown=300):
        if (spec.workload_kind == 'service' and spec.migration_policy != 'disabled'
                and (spec.migration_policy != 'drain' or not spec.drain_command
                     or spec.min_ready_replicas < 1)):
            raise ValueError('Service migration requires explicit drain and at least one ready replica')
        self.adapter, self.spec = adapter, spec
        self.directory = Path(state_dir).resolve()
        self.path = self.directory / 'STATE.json'
        self.required_path = self.directory / 'STATE_REQUIRED.json'
        self.clock = clock
        self.max_startup_failures = max_startup_failures
        self.migration_cooldown = migration_cooldown
        self._locked = False
        self.state = None

    def _load(self):
        if self.required_path.exists():
            required = json.loads(self.required_path.read_text())
            if (required.get('state_dir') != str(self.directory)
                    or required.get('identity_sha256') != _digest(spec_identity(self.spec))):
                raise ValueError('Required state marker belongs to another workload')
            if not self.path.exists():
                raise RuntimeError('Required durable STATE.json is missing; refusing an empty budget')
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if (state.get('schema') != 'resource_manager.v1'
                    or state.get('identity_sha256') != _digest(spec_identity(self.spec))):
                raise ValueError('Workload identity changed; use a separate state directory')
            for key, row in state['resources'].items():
                if (row['intent']['operation_id'] != key
                        or row.get('intent_sha256') != _digest(row['intent'])
                        or row['intent']['scope'] != self.spec.scope
                        or row['intent']['profile'] != self.spec.profile):
                    raise ValueError('Durable resource intent identity changed')
            return state
        return {'schema': 'resource_manager.v1', 'scope': self.spec.scope,
                'profile': self.spec.profile, 'identity_sha256': _digest(spec_identity(self.spec)),
                'phase': 'running', 'resources': {}, 'created_at': self.clock(),
                'max_cards': self.spec.max_cards}

    def _save(self):
        # The marker is durable first: a crash before the first state write
        # fails closed on restart. No lifecycle request is sent until both
        # writes return, so initialization cannot lose an accepted reservation.
        if not self.required_path.exists():
            atomic_json(self.required_path, {'state_dir': str(self.directory),
                'identity_sha256': _digest(spec_identity(self.spec))})
        self.state.update(updated_at=self.clock(), max_cards=self.spec.max_cards)
        atomic_json(self.path, self.state)

    @contextmanager
    def locked(self):
        if self._locked:
            yield; return
        self.directory.mkdir(parents=True, exist_ok=True)
        owner_dir = self.directory / '.owner'
        try:
            owner_dir.mkdir()
            atomic_json(owner_dir / 'HOST.json', {'hostname': socket.gethostname()})
        except FileExistsError:
            owner = json.loads((owner_dir / 'HOST.json').read_text())
            if owner.get('hostname') != socket.gethostname():
                raise RuntimeError('Controller state belongs to another host; explicit takeover required')
        with (self.directory / '.controller.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._locked = True
            try:
                yield
            finally:
                self._locked = False
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _snapshot(self):
        snapshot = self.adapter.snapshot()
        if (type(snapshot.get('global_remaining_cards')) is not int
                or snapshot['global_remaining_cards'] < 0
                or not isinstance(snapshot.get('resources'), dict)):
            raise ValueError('Snapshot must give nonnegative global headroom and visible resource IDs')
        return snapshot

    def _reservations(self, snapshot):
        result = []
        for key, row in self.state['resources'].items():
            intent, resource_id = row['intent'], row.get('resource_id')
            quota = intent['placement']['quota']
            visible = snapshot['resources'].get(resource_id, {})
            # A live active observation overrides a stale STOPPED retirement.
            state = visible.get('state', row.get('state', 'UNKNOWN')).upper()
            if row.get('action_required') or row.get('observation_error_type'):
                state = 'UNKNOWN'  # Reserve capacity without making it a migration victim.
            result.append(Reservation(
                id=key, node=intent['placement']['node'], **quota,
                scope=self.spec.scope, profile=self.spec.profile, state=state,
                retired_verified=row.get('retired_verified', False),
                capacity_accounted=visible.get('capacity_accounted') is True,
                ready=row.get('ready') is True and state == 'RUNNING',
                operation_id=key, resource_id=resource_id))
        return result

    def _inputs(self, snapshot):
        nodes = [row if isinstance(row, NodeCapacity) else NodeCapacity(**row) for row in snapshot['nodes']]
        quotas = [row if isinstance(row, Quota) else Quota(**row) for row in snapshot['quotas']]
        reservations = self._reservations(snapshot)
        missing = sum(row.cards for row in reservations
                      if not row.is_retired and row.resource_id not in snapshot['resources'])
        return nodes, quotas, reservations, max(0, snapshot['global_remaining_cards'] - missing)

    def plan(self):
        self.state = self._load()
        snapshot = self._snapshot()
        nodes, quotas, reservations, remaining = self._inputs(snapshot)
        return asdict(plan_allocations(self.spec, nodes, quotas, reservations,
                                       global_remaining_cards=remaining))

    def _accept(self, row, identifier):
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Adapter did not return an exact resource ID')
        if any(other is not row and other.get('resource_id') == identifier
               for other in self.state['resources'].values()):
            raise ValueError('Resource ID is already bound to another intent')
        row.update(resource_id=identifier, state='PENDING', accepted_at=self.clock())
        self._save()

    def _create(self, placement, *, migration=None):
        operation_id = uuid.uuid4().hex
        intent = {'operation_id': operation_id, 'name': 'rm-' + operation_id[:20],
                  'scope': self.spec.scope, 'profile': self.spec.profile,
                  'placement': asdict(placement), 'command': list(self.spec.command),
                  'created_at': self.clock(), 'spec': asdict(self.spec)}
        row = {'intent': intent, 'intent_sha256': _digest(intent), 'state': 'UNKNOWN',
               'retired_verified': False, 'ready': False, 'create_sent_at': self.clock()}
        self.state['resources'][operation_id] = row
        if migration is not None:
            migration['replacement_ids'].append(operation_id)
        else:
            self.state['replacement_cards'] = max(0, self.state.get('replacement_cards', 0) - placement.cards)
        self._save()  # Durable before the only Create send, including a crash window.
        try:
            self._accept(row, self.adapter.create(intent))
        except Exception as error:
            row['create_error_type'] = type(error).__name__
            self._save()
        return operation_id

    def _operation(self, row, kind, callback):
        if row.get(kind):
            return False
        operation = {'operation_id': uuid.uuid4().hex, 'phase': 'intent', 'time': self.clock()}
        row[kind] = operation
        self._save()
        try:
            callback(operation['operation_id'])
        except Exception as error:
            operation.update(phase='unknown', error_type=type(error).__name__)
        else:
            operation['phase'] = 'sent'
        self._save()
        return True

    def _register_replacement(self, row):
        if not row.get('replacement_registered') and not row.get('worker_completed'):
            row['replacement_registered'] = True
            self.state['replacement_cards'] = self.state.get('replacement_cards', 0) + row['intent']['placement']['quota']['cards']

    def _observe(self, row):
        if not row.get('resource_id'):
            identifier = self.adapter.reconcile(row['intent'])
            if identifier:
                self._accept(row, identifier)
            else:
                return
        observed = self.adapter.observe(row['resource_id'])
        state = str(observed['state']).upper()
        if state not in ('PENDING', 'RUNNING', 'STOPPED', 'FAILED', 'UNKNOWN', 'CREATING'):
            raise ValueError('Adapter returned an unsupported resource state')
        previous = row.get('state')
        row.update(state=state, observed_at=self.clock(), ready=False)
        if state == 'STOPPED':
            newly_stopped = not row.get('retired_verified')
            row['retired_verified'] = True
            # A fresh owned platform observation supersedes a stale launch
            # transport warning. Other identity/action failures stay visible.
            if row.get('action_required') == 'launch_result_unconfirmed':
                row.pop('action_required')
            if newly_stopped and not row.get('stop') and not row.get('worker_completed'):
                self.state['backoff_until'] = max(self.state.get('backoff_until', 0), self.clock() + 10)
                row['retirement_reason'] = 'preemption_or_external_stop'
                if self.state['phase'] != 'stopping':
                    self._register_replacement(row)
            return
        if row.get('retired_verified'):
            row['action_required'] = 'previously_retired_resource_became_active'
            return
        if state == 'FAILED':
            if self.state['phase'] != 'stopping':
                self._register_replacement(row)
            self._operation(row, 'stop', lambda op: self.adapter.stop(row['resource_id'], op))
            return
        if state != 'RUNNING':
            return
        health = self.adapter.health(row['resource_id'],
                                     list(self.spec.readiness_command) if self.spec.readiness_command else None)
        runtime_id = health.get('runtime_id')
        previous_runtime = row.get('runtime_id')
        if runtime_id and previous_runtime and runtime_id != previous_runtime and not row.get('stop'):
            # Exact resource identity survived, but its container did not.
            row.pop('launch', None); row.pop('action_required', None)
            row.update(startup_failures=0, worker_completed=False)
        if runtime_id:
            row['runtime_id'] = runtime_id
        row['health'] = {key: health.get(key) for key in (
            'running', 'ready', 'completed', 'exit_code', 'launch_operation_id',
            'drained', 'drain_operation_id', 'runtime_id', 'process_gone')}
        launch = row.get('launch')
        matches = (launch is not None and health.get('launch_operation_id') == launch['operation_id'])
        matches |= bool(self.adapter.capabilities.get('native_command') and launch)
        if matches and health.get('running') is True:
            row['ready'] = (health.get('ready') is True and
                            (self.spec.workload_kind != 'service' or self.spec.readiness_command is not None))
        if self.state['phase'] == 'stopping' or row.get('stop') or row.get('drain') or row.get('action_required'):
            return
        if matches and (health.get('completed') is True or health.get('process_gone') is True):
            if health.get('completed') is True and health.get('exit_code') == 0 and self.spec.workload_kind == 'batch':
                row['worker_completed'] = True
                return
            if row.get('last_failed_launch') != launch['operation_id']:
                failures = row.get('startup_failures', 0) + 1
                row.update(startup_failures=failures, last_failed_launch=launch['operation_id'],
                           retry_at=self.clock() + min(300, 10 * 2 ** (failures - 1)))
                row.pop('launch', None)
                if failures >= self.max_startup_failures:
                    row['action_required'] = 'startup_failure_limit'
            return
        if launch and (not matches or (not health.get('running') and not health.get('completed'))) and self.clock() - launch['time'] >= 300:
            row['action_required'] = 'launch_result_unconfirmed'
            return
        if not launch and self.clock() >= row.get('retry_at', 0):
            self._operation(row, 'launch', lambda op: self.adapter.launch(
                row['resource_id'], list(self.spec.command), op))

    def _stop_row(self, row, *, use_drain=False):
        if not row.get('resource_id') or (row.get('retired_verified') and row.get('state') == 'STOPPED'):
            return
        if use_drain:
            if not self.adapter.capabilities.get('drain') or not self.spec.drain_command:
                raise ValueError('Adapter cannot implement the requested explicit drain')
            self._operation(row, 'drain', lambda op: self.adapter.drain(
                row['resource_id'], list(self.spec.drain_command), op))
            health = self.adapter.health(row['resource_id'],
                                         list(self.spec.readiness_command) if self.spec.readiness_command else None)
            if not (health.get('drained') is True
                    and health.get('drain_operation_id') == row['drain']['operation_id']):
                return
        previous = row.get('stop')
        if (previous and previous.get('phase') == 'unknown'
                and self.adapter.capabilities.get('retry_owned_stop')
                and not row.get('observation_error_type')
                and row.get('state') in ('RUNNING', 'PENDING', 'FAILED', 'CREATING')
                and self.clock() - previous.get('last_retry_at', previous['time']) >= 30):
            # Reuse the original stop intent only. The adapter revalidates the
            # exact resource on every call. No new Create/Start is authorized.
            previous.update(last_retry_at=self.clock(), retries=previous.get('retries', 0) + 1)
            self._save()
            try:
                self.adapter.stop(row['resource_id'], previous['operation_id'])
            except Exception as error:
                previous['error_type'] = type(error).__name__
            else:
                previous['phase'] = 'sent'
            self._save()
        else:
            self._operation(row, 'stop', lambda op: self.adapter.stop(row['resource_id'], op))

    def _migration(self, snapshot):
        migration = self.state.get('migration')
        if migration and migration['phase'] in ('completed', 'aborted'):
            if self.clock() < migration['cooldown_until']:
                return False
            migration = None
        if not migration:
            if (self.spec.migration_policy == 'disabled'
                    or not self.adapter.capabilities.get('node_placement')
                    or (self.spec.workload_kind == 'service' and not self.adapter.capabilities.get('drain'))):
                return False
            nodes, quotas, reservations, remaining = self._inputs(snapshot)
            proposal = plan_consolidation(self.spec, nodes, quotas, reservations,
                                          global_remaining_cards=remaining)
            if proposal is None:
                return False
            migration = asdict(proposal) | {'phase': 'stopping', 'created_at': self.clock(),
                                            'replacement_ids': []}
            self.state['migration'] = migration
            self._save()
        if self.clock() - migration['created_at'] > 900:
            migration.update(phase='aborted', reason='deadline_keep_uncertain_reservations',
                             cooldown_until=self.clock() + self.migration_cooldown)
            self._save(); return False
        sources = [self.state['resources'][key] for key in migration['source_ids']]
        if any(row.get('action_required') or row.get('observation_error_type') for row in sources):
            migration.update(phase='aborted', reason='source_requires_reconciliation',
                             cooldown_until=self.clock() + self.migration_cooldown)
            self._save(); return False
        if not migration['replacement_ids']:
            nodes, _, reservations, _ = self._inputs(snapshot)
            free = {node.name: [node.gpu, node.cpu, node.memory] for node in available_nodes(nodes, reservations)}
            source_ids = set(migration['source_ids'])
            fits = True
            for target in migration['placements']:
                capacity = free.get(target['node'])
                need = [target['quota'][key] for key in ('cards', 'cpu', 'memory')]
                if capacity is None or any(a < b for a, b in zip(capacity, need)):
                    fits = False; break
                free[target['node']] = [a - b for a, b in zip(capacity, need)]
            if not fits:
                migration.update(phase='aborted', reason='target_capacity_changed',
                                 cooldown_until=self.clock() + self.migration_cooldown)
                self._save(); return False
        if any(not (row.get('retired_verified') and row.get('state') == 'STOPPED') for row in sources):
            if self.spec.workload_kind == 'service':
                # Observations above are fresh each poll. Check the whole
                # source group, not one drain at a time, against min-ready.
                source_ids = set(migration['source_ids'])
                remaining_ready = sum(row.get('ready') is True for key, row in self.state['resources'].items()
                                      if key not in source_ids)
                if remaining_ready < self.spec.min_ready_replicas:
                    migration.update(phase='aborted', reason='minimum_ready_changed',
                                     cooldown_until=self.clock() + self.migration_cooldown)
                    self._save(); return False
            for row in sources:
                if not (row.get('retired_verified') and row.get('state') == 'STOPPED'):
                    self._stop_row(row, use_drain=self.spec.migration_policy == 'drain')
                    return True
        # Replacement decisions reuse current planner feasibility, but are
        # constrained to the durable target list and exact quotas.
        for index, raw in enumerate(migration['placements']):
            if index < len(migration['replacement_ids']):
                row = self.state['resources'][migration['replacement_ids'][index]]
                if not row.get('resource_id'):
                    return True
                continue
            fresh = self._snapshot()
            nodes, quotas, reservations, remaining = self._inputs(fresh)
            placement = Placement(raw['node'], Quota(**raw['quota']))
            target = next((node for node in nodes if node.name == placement.node), None)
            # Ask the normal planner for exactly this request. Unknown and
            # pending allocations still reduce residual resources and budget.
            target_spec = replace(self.spec, max_cards=sum(r.cards for r in reservations if not r.is_retired)
                                  + placement.cards, allowed_cards=(placement.cards,), replica_cards=self.spec.replica_cards,
                                  min_ready_replicas=0)
            allowed = plan_allocations(target_spec, [target] if target else [], [placement.quota], reservations,
                                       global_remaining_cards=remaining)
            if (not allowed.placements or allowed.placements[0] != placement
                    or sum(r.cards for r in reservations if not r.is_retired) + placement.cards > self.spec.max_cards):
                migration.update(phase='aborted', reason='target_unavailable',
                                 cooldown_until=self.clock() + self.migration_cooldown)
                self._save(); return False
            self._create(placement, migration=migration)
            self._save(); return True
        migration.update(phase='completed', cooldown_until=self.clock() + self.migration_cooldown)
        self._save(); return False

    def poll(self, *, should_stop=lambda: False):
        with self.locked():
            # A failed load must never leave the prior poll's state eligible
            # for the run loop's error-reporting write.
            self.state = None
            self.state = self._load()
            if self.state['phase'] == 'completed':
                return summary(self.state)
            if (self.directory / 'STOP_REQUEST.json').exists():
                request = json.loads((self.directory / 'STOP_REQUEST.json').read_text())
                if request.get('scope') != self.spec.scope or request.get('profile') != self.spec.profile:
                    raise ValueError('Stop request belongs to another workload')
                self.state.update(phase='stopping', finish_reason='operator_stop')
            for row in self.state['resources'].values():
                if should_stop():
                    self._save(); return summary(self.state)
                try:
                    self._observe(row)
                    row.pop('observation_error_type', None)
                except Exception as error:
                    row.update(observation_error_type=type(error).__name__, ready=False)
                self._save()
            rows = list(self.state['resources'].values())
            active = [row for row in rows if not (row.get('retired_verified') and row.get('state') == 'STOPPED')]
            if self.spec.workload_kind == 'batch' and self.state['phase'] not in ('stopping', 'completed'):
                if self.spec.completion_command:
                    for row in active:
                        if row.get('resource_id') and row.get('state') == 'RUNNING':
                            if self.adapter.completion(row['resource_id'], list(self.spec.completion_command)):
                                self.state.update(phase='stopping', finish_reason='completion_probe')
                                break
                elif any(row.get('worker_completed') for row in rows):
                    self.state['phase'] = 'finishing'
                    if all(row.get('worker_completed') for row in active) and not self.state.get('replacement_cards'):
                        self.state.update(phase='stopping', finish_reason='all_started_workers_succeeded')
            if self.state['phase'] == 'stopping':
                for row in active:
                    if should_stop():
                        break
                    self._stop_row(row, use_drain=bool(self.spec.workload_kind == 'service'
                                                       and self.spec.drain_command))
                if not active:
                    self.state['phase'] = 'completed'
                self._save(); return summary(self.state)
            if any(row.get('observation_error_type') for row in rows):
                self._save(); return summary(self.state)
            if should_stop() or (self.state['phase'] == 'finishing' and not self.state.get('replacement_cards')):
                self._save(); return summary(self.state)
            snapshot = self._snapshot()
            if self.state['phase'] != 'finishing' and self._migration(snapshot):
                self._save(); return summary(self.state)
            if self.clock() < self.state.get('backoff_until', 0):
                self._save(); return summary(self.state)
            # Re-plan after every Create: a seven-card node becomes 4+2+1
            # while accepted but unallocated requests consume residual CPU/RAM.
            for _ in range(self.spec.max_cards):
                if should_stop():
                    break
                snapshot = self._snapshot()
                nodes, quotas, reservations, remaining = self._inputs(snapshot)
                allocation_spec = self.spec
                if self.state['phase'] == 'finishing':
                    recovery_limit = sum(r.cards for r in reservations if not r.is_retired) + self.state.get('replacement_cards', 0)
                    allocation_spec = replace(self.spec, max_cards=min(self.spec.max_cards, recovery_limit))
                proposal = plan_allocations(allocation_spec, nodes, quotas, reservations,
                                            global_remaining_cards=remaining)
                if not proposal.placements:
                    break
                key = self._create(proposal.placements[0])
                if not self.state['resources'][key].get('resource_id'):
                    break  # Unknown Create stops this burst, retaining its reservation.
            self._save(); return summary(self.state)

    def run(self, *, should_stop=lambda: False, once=False):
        with self.locked():
            while not should_stop():
                try:
                    result = self.poll(should_stop=should_stop)
                except Exception as error:
                    if once:
                        raise
                    if self.state is None:
                        raise  # Loading/validating the ledger failed; do not repair it from memory.
                    self.state['last_poll_error_type'] = type(error).__name__
                    self._save()
                    result = summary(self.state)
                if once or result['phase'] == 'completed':
                    return result
                deadline = time.monotonic() + self.spec.poll_seconds
                while not should_stop() and time.monotonic() < deadline:
                    time.sleep(min(.25, max(0, deadline - time.monotonic())))
            return summary(self._load())


def request_stop(state_dir, *, scope, profile):
    atomic_json(Path(state_dir) / 'STOP_REQUEST.json',
                {'scope': scope, 'profile': profile, 'requested_at': time.time()})
