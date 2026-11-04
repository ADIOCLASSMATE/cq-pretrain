"""Compile bounded resource-manager fleets; generation never starts resources.

StageAdapter delegates lifecycle/packing/preemption recovery to the existing
manager, adding a durable inter-stage budget fence. The remote worker must call
preflight before loading models or claiming tasks. Platform paths are explicit;
this module neither uploads code nor assumes storage is shared across partitions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import socket

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from corpus_pipeline.resource_manager.models import WorkloadSpec
from .config import load, require
from corpus_pipeline.coordination import locked

STAGES = {'frontpass': 'frontpass_max_cards', 'sscd_search': 'sscd_max_cards'}
SCHEMA = 'corpus.unified.fleet.v1'
ADAPTER = 'corpus_pipeline.resource_manager.inspire_adapter:InspireAdapter'


def _stop_paths(cfg, stage):
    root = Path(cfg['output'])
    return (root / 'STOP', root / stage / 'STOP')


def _is_stopped(cfg, stage):
    return any(path.exists() for path in _stop_paths(cfg, stage))


def request_stop(config_path):
    """Request business/controller shutdown without claiming resources released."""
    from corpus_pipeline.resource_manager.engine import request_stop as stop_manager
    config_path = Path(config_path).resolve(strict=True)
    cfg = load(config_path); root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    guard = root / 'fleet/guard'; guard.mkdir(parents=True, exist_ok=True)
    requests = []
    with locked(guard / '.stage.lock'):
        markers = [root / name for name in ('STOP', 'frontpass/STOP', 'sscd_search/STOP', 'sscd_index/catalog_parallel/STOP')]
        for marker in markers:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        identity_path = root / 'fleet/IDENTITY.json'
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            require(identity['config_path'] == str(config_path), 'Fleet belongs to another configuration path')
            for stage in STAGES:
                path = Path(identity['manager_directory']) / (stage + '.json')
                if not path.exists():
                    continue
                value = json.loads(path.read_text()); contract = value['adapter_config']['contract']
                require(value['adapter'] == 'corpus_pipeline.unified.fleet:StageAdapter' and
                        contract['identity_sha256'] == file_sha(identity_path) and contract['stage'] == stage and
                        value['state_dir'] == contract['state_directory'] and
                        value['workload']['scope'] == contract['scope'] and value['workload']['profile'] == contract['profile'],
                        'Stop target differs from bound manager identity')
                stop_manager(value['state_dir'], scope=contract['scope'], profile=contract['profile'])
                requests.append({'stage': stage, 'state_dir': value['state_dir'], 'stop_requested': True})
    return {'stop_requested': True, 'resources_released_verified': False, 'root': str(root),
            'business_stop_markers': [str(path) for path in markers], 'manager_stop_requests': requests,
            'next_action': 'Keep or resume each owning manager until phase=completed and reserved_cards=0; release external review pools separately.'}


def _factory(name, config):
    module, separator, attr = name.partition(':')
    require(bool(separator and module and attr), 'Explicit module:factory required')
    return getattr(importlib.import_module(module), attr)(config)


def _absolute(value, name):
    require(isinstance(value, str) and Path(value).is_absolute(), 'Absolute path required: ' + name)
    return Path(value)


def _runtime_bindings(repository):
    from corpus_pipeline.orchestration.prepare import source_files, PACKAGE
    # Use the same explicit source inventory as snapshots. Never crawl model
    # weights, millions of review receipts or abandoned experiment directories.
    return [{'relative_path':'corpus_pipeline/'+str(path.relative_to(PACKAGE)),
             'sha256':file_sha(repository/'corpus_pipeline'/path.relative_to(PACKAGE))}
            for path in source_files()]


def build(config_path, template_path, output=None):
    """Generate frontpass/SSCD configs from an explicit existing Notebook template.

    Returns manager-config paths and identity. No SDK/client is instantiated.
    Changing a bound configuration requires a fresh pipeline output directory.
    """
    config_path, template_path = Path(config_path).resolve(strict=True), Path(template_path).resolve(strict=True)
    cfg = load(config_path)
    template = json.loads(template_path.read_text())
    require(template.get('adapter') == ADAPTER, 'Fleet requires the existing explicit Inspire Notebook adapter')
    adapter = deepcopy(template['adapter_config'])
    require(adapter.get('backend', 'notebook') == 'notebook', 'Batch fleet requires Notebook; Serving is a separate stage')
    for name in ('account', 'workspace', 'project', 'group', 'image'):
        value = adapter.get(name)
        require(isinstance(value, str) and value.strip() and 'REPLACE_' not in value, 'Unresolved platform template: ' + name)
    require(adapter.get('priority') in (1, 3, 4, 6), 'Explicit LOW(1/3)/HIGH(4/6) priority required')
    for name in ('repository', 'remote_python', 'worker_state_dir', 'registry_dir'):
        _absolute(adapter.get(name), name)
    resources = cfg['resources']
    require(1 <= resources['max_cards'] <= 512, 'This fleet total budget must not exceed 512 cards')
    require(resources.get('prefer_occupied_nodes') is True, 'This fleet preserves occupied-node packing')
    require(type(adapter.get('global_max_cards', 128)) is int and adapter.get('global_max_cards', 128) > 0,
            'Invalid platform account ceiling')
    ceiling = min(resources['max_cards'], adapter.get('global_max_cards', 128))
    defaults = WorkloadSpec(**template['workload'])
    directory = Path(output).resolve() if output is not None else Path(cfg['output']) / 'fleet'
    canonical_root = Path(cfg['output']) / 'fleet'
    local_repository = Path(__file__).resolve().parents[2]
    runtime = _runtime_bindings(local_repository)
    # One canonical fence location cannot be bypassed by choosing another config directory.
    identity = {'schema': SCHEMA, 'config_path': str(config_path), 'config_sha256': file_sha(config_path),
                'template_path': str(template_path), 'template_sha256': file_sha(template_path),
                'code_sha256': file_sha(__file__), 'total_max_cards': resources['max_cards'],
                'runtime_sha256': digest(runtime),
                'account_ceiling': ceiling, 'stage_budgets': {stage: resources[key] for stage, key in STAGES.items()},
                'remote_repository': adapter['repository'], 'remote_python': adapter['remote_python'],
                'manager_directory': str(directory), 'guard_directory': str(canonical_root / 'guard')}
    canonical_root.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = canonical_root / 'IDENTITY.json'
    if identity_path.exists():
        require(json.loads(identity_path.read_text()) == identity, 'Fleet identity changed; do not discard existing resource ledgers')
    else:
        atomic_json(identity_path, identity)
    identity_sha = file_sha(identity_path)
    sentinel = canonical_root / 'SHARED_PATH_SENTINEL.json'
    sentinel_value = {'schema': SCHEMA, 'identity_sha256': identity_sha, 'output': cfg['output']}
    if sentinel.exists():
        require(json.loads(sentinel.read_text()) == sentinel_value, 'Shared-path sentinel changed')
    else:
        atomic_json(sentinel, sentinel_value)
    preflight_value = {'schema': SCHEMA, 'identity': identity, 'identity_sha256': identity_sha,
                       'runtime': runtime, 'shared_sentinel': {'path': str(sentinel), 'sha256': file_sha(sentinel)},
                       'dataset_roots': sorted({item['root'] for item in cfg['inputs']}),
                       'cache_directory': cfg['cache']['directory'],
                       'mount_contract': 'same absolute paths and bound bytes; shared output storage required'}
    preflight_path = canonical_root / 'PREFLIGHT.json'
    if preflight_path.exists():
        require(json.loads(preflight_path.read_text()) == preflight_value, 'Fleet runtime changed; existing deployment identity is immutable')
    else:
        atomic_json(preflight_path, preflight_value)
    manager_configs = {}
    for stage, budget_key in STAGES.items():
        budget = resources[budget_key]
        allowed = [cards for cards in defaults.allowed_cards if cards <= budget]
        require(bool(allowed), 'Template has no legal allocation size within stage budget')
        inner = deepcopy(adapter)
        inner['global_max_cards'] = ceiling
        inner['backend'] = 'notebook'
        inner['registry_dir'] = str(directory / stage / 'adapter')
        inner['worker_state_dir'] = str(Path(adapter['worker_state_dir']) / identity_sha[:16] / stage)
        python = adapter['remote_python']
        workload = {'scope': 'unified-' + identity_sha[:16], 'profile': stage, 'max_cards': budget,
                    'command': [python, '-m', 'corpus_pipeline.unified', 'node-worker', '--config', str(config_path), '--stage', stage],
                    'allowed_cards': allowed, 'replica_cards': None, 'migration_policy': 'restartable',
                    'workload_kind': 'batch', 'min_ready_replicas': 0, 'readiness_command': None,
                    'drain_command': None, 'completion_command': [python, '-m', 'corpus_pipeline.unified.fleet',
                        'completion', '--config', str(config_path), '--stage', stage], 'poll_seconds': defaults.poll_seconds}
        WorkloadSpec(**workload)
        contract = {'schema': SCHEMA, 'stage': stage, 'identity_path': str(identity_path),
                    'identity_sha256': identity_sha, 'guard_directory': str(canonical_root / 'guard'),
                    'state_directory': str(directory / stage / 'controller'), 'config_path': str(config_path),
                    'config_sha256': identity['config_sha256'], 'total_max_cards': resources['max_cards'],
                    'stage_max_cards': budget, 'account_ceiling': ceiling,
                    'scope': workload['scope'], 'profile': stage}
        result = {'adapter': 'corpus_pipeline.unified.fleet:StageAdapter',
                  'state_dir': contract['state_directory'], 'workload': workload,
                  'adapter_config': {'inner_adapter': ADAPTER, 'inner_config': inner, 'contract': contract},
                  'unified_preflight': str(preflight_path), 'unified_preflight_sha256': file_sha(preflight_path)}
        path = directory / (stage + '.json')
        if path.exists():
            require(json.loads(path.read_text()) == result, 'Generated manager configuration changed')
        else:
            atomic_json(path, result)
        manager_configs[stage] = {'path': str(path), 'sha256': file_sha(path)}
    return {'schema': SCHEMA, 'generated_only': True, 'resources_started': False,
            'identity_path': str(identity_path), 'identity_sha256': identity_sha,
            'preflight_path': str(preflight_path), 'manager_configs': manager_configs}


def _controller_running(directory):
    path = Path(directory) / '.controller.lock'
    if not path.exists():
        return False
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
        return False


def _retired(active):
    """Absence never releases an intent; only verified retirement can do so."""
    directory = Path(active['state_directory'])
    if _controller_running(directory):
        return False
    path = directory / 'STATE.json'
    if not path.exists():
        # A read-only resource-manager plan can claim a stage before any intent.
        # The engine's durable marker is written before any lifecycle operation.
        return not (directory / 'STATE_REQUIRED.json').exists()
    state = json.loads(path.read_text())
    require(state.get('scope') == active['scope'] and state.get('profile') == active['profile'],
            'Previous stage controller identity changed')
    require(state.get('schema') == 'resource_manager.v1' and isinstance(state.get('resources'), dict) and
            (directory / 'STATE_REQUIRED.json').exists(), 'Previous stage controller ledger is incomplete')
    return state.get('phase') == 'completed' and all(
        row.get('state') == 'STOPPED' and row.get('retired_verified') is True
        for row in state.get('resources', {}).values())


class StageAdapter:
    """Budget fence around the existing adapter, not a second resource scheduler."""
    def __init__(self, config):
        self.config = deepcopy(config)
        self.contract = self.config['contract']
        self._validate()
        self.inner = _factory(config['inner_adapter'], config['inner_config'])
        self.capabilities = dict(self.inner.capabilities)

    def _validate(self):
        value = self.contract
        require(value.get('schema') == SCHEMA and value.get('stage') in STAGES, 'Invalid fleet stage contract')
        require(file_sha(value['identity_path']) == value['identity_sha256'], 'Fleet identity SHA mismatch')
        require(file_sha(value['config_path']) == value['config_sha256'], 'Unified config changed after fleet preparation')
        identity = json.loads(Path(value['identity_path']).read_text())
        require(identity['code_sha256'] == file_sha(__file__), 'Fleet controller implementation changed')
        require(identity['config_path'] == value['config_path'] and identity['config_sha256'] == value['config_sha256'] and
                identity['guard_directory'] == value['guard_directory'] and
                identity['stage_budgets'][value['stage']] == value['stage_max_cards'] and
                identity['total_max_cards'] == value['total_max_cards'] and
                identity['account_ceiling'] == value['account_ceiling'], 'Fleet contract differs from bound identity')
        require(value['profile'] == value['stage'] and value['scope'] == 'unified-' + value['identity_sha256'][:16] and
                value['state_directory'] == str(Path(identity['manager_directory']) / value['stage'] / 'controller'),
                'Stage controller binding changed')
        require(0 < value['stage_max_cards'] <= value['total_max_cards'] <= 512, 'Fleet resource ceiling invalid')
        require(self.config['inner_adapter'] == ADAPTER, 'Unregistered inner platform adapter')
        require(self.config['inner_config']['global_max_cards'] == value['account_ceiling'], 'Account budget cannot bypass fleet ceiling')
        require(file_sha(identity['template_path']) == identity['template_sha256'], 'Bound platform template changed')
        expected = json.loads(Path(identity['template_path']).read_text())['adapter_config']
        expected.update(global_max_cards=value['account_ceiling'], backend='notebook',
                        registry_dir=str(Path(identity['manager_directory']) / value['stage'] / 'adapter'),
                        worker_state_dir=str(Path(expected['worker_state_dir']) / value['identity_sha256'][:16] / value['stage']))
        require(expected == self.config['inner_config'], 'Generated platform adapter configuration changed')

    @contextmanager
    def _fence(self):
        self._validate()
        root = Path(self.contract['guard_directory'])
        root.mkdir(parents=True, exist_ok=True)
        with locked(root / '.stage.lock'):
            path = root / 'ACTIVE.json'
            required = root / 'ACTIVE_REQUIRED.json'
            marker = {'identity_sha256': self.contract['identity_sha256'], 'guard_directory': str(root)}
            if required.exists():
                require(json.loads(required.read_text()) == marker and path.exists(),
                        'Required fleet stage ledger is missing or changed; refusing an empty budget')
            else:
                require(not path.exists(), 'Fleet stage ledger lacks its durable required marker')
                atomic_json(required, marker)
            active = json.loads(path.read_text()) if path.exists() else None
            if active is not None and active != self.contract:
                require(active.get('identity_sha256') == self.contract['identity_sha256'], 'Another pipeline identity owns the fleet fence')
                require(_retired(active), 'Previous stage still owns active/pending/unknown resource reservations')
            if active != self.contract:
                atomic_json(path, self.contract)
            yield

    def snapshot(self):
        with self._fence():
            stopped = self._propagate_stop()
            business_done = self._business_done_seen()
            if business_done:
                # Do not start a new consolidation while surviving pools seal.
                self.capabilities['node_placement'] = False
            result = self.inner.snapshot()
            return {**result, 'global_remaining_cards': 0} if stopped or business_done else result

    def _business_done_seen(self):
        path = Path(self.contract['state_directory']) / 'POOL_COMPLETION_ACKS.json'
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        require(value['identity_sha256'] == self.contract['identity_sha256'] and
                value['stage'] == self.contract['stage'], 'Pool completion ledger belongs to another stage')
        # Every ACK includes a successful global completion probe. That business
        # fact is monotonic for this immutable execution, even after preemption.
        return bool(value['resources'])

    def _propagate_stop(self):
        cfg = load(self.contract['config_path'])
        if not _is_stopped(cfg, self.contract['stage']):
            return False
        from corpus_pipeline.resource_manager.engine import request_stop as stop_manager
        stop_manager(self.contract['state_directory'], scope=self.contract['scope'], profile=self.contract['profile'])
        return True

    def create(self, intent):
        with self._fence():
            require(not self._propagate_stop(), 'Pipeline/stage STOP forbids new allocations')
            require(intent['scope'] == self.contract['scope'] and intent['profile'] == self.contract['profile'],
                    'Allocation belongs to another stage')
            require(intent['spec']['max_cards'] <= self.contract['stage_max_cards'] and
                    intent['placement']['quota']['cards'] <= self.contract['stage_max_cards'], 'Allocation bypasses stage ceiling')
            expected_command = [self.config['inner_config']['remote_python'], '-m', 'corpus_pipeline.unified',
                                'node-worker', '--config', self.contract['config_path'], '--stage', self.contract['stage']]
            require(list(intent['command']) == expected_command, 'Allocation command differs from bound unified stage')
            state = json.loads((Path(self.contract['state_directory']) / 'STATE.json').read_text())
            saved = state.get('resources', {}).get(intent['operation_id'], {}).get('intent')
            require(saved is not None and digest(saved) == digest(intent), 'Allocation intent is not durable in bound controller ledger')
            return self.inner.create(intent)

    def launch(self, identifier, command, operation_id):
        with self._fence():
            require(not self._propagate_stop(), 'Pipeline/stage STOP forbids new worker launches')
            return self.inner.launch(identifier, command, operation_id)

    def reconcile(self, intent):
        return self.inner.reconcile(intent)

    def observe(self, identifier):
        return self.inner.observe(identifier)

    def stop(self, identifier, operation_id):
        return self.inner.stop(identifier, operation_id)

    def health(self, identifier, readiness_command=None):
        return self.inner.health(identifier, readiness_command)

    def drain(self, identifier, command, operation_id):
        return self.inner.drain(identifier, command, operation_id)

    def completion(self, identifier, command):
        # The adapter's remote probe process does not inherit the worker's env.
        # Resolve the exact allocation from the owned ledger; never scan stale
        # pool files or infer allocation identity from a node name.
        state = json.loads((Path(self.contract['state_directory']) / 'STATE.json').read_text())
        matches = [row['intent'] for row in state['resources'].values() if row.get('resource_id') == identifier]
        require(len(matches) == 1, 'Completion probe resource is not uniquely bound to this controller')
        intent = matches[0]
        require(intent['scope'] == self.contract['scope'] and intent['profile'] == self.contract['profile'],
                'Completion probe resource belongs to another workload')
        completed = self.inner.completion(identifier, ['env',
            'RESOURCE_MANAGER_ALLOCATION_ID=' + intent['operation_id'],
            'RESOURCE_MANAGER_CARDS=' + str(intent['placement']['quota']['cards']), *command])
        if not completed:
            return False
        # The generic engine treats ANY successful completion probe as global
        # permission to stop ALL allocations. Record per-allocation proofs and
        # return true only after every launched, still-active pool has sealed.
        directory = Path(self.contract['state_directory'])
        with locked(directory / '.pool_completion.lock'):
            state = json.loads((directory / 'STATE.json').read_text())
            path = directory / 'POOL_COMPLETION_ACKS.json'
            acknowledgments = json.loads(path.read_text()) if path.exists() else {
                'identity_sha256': self.contract['identity_sha256'], 'stage': self.contract['stage'], 'resources': {}}
            require(acknowledgments['identity_sha256'] == self.contract['identity_sha256'] and
                    acknowledgments['stage'] == self.contract['stage'], 'Pool completion ledger belongs to another stage')

            def binding(row):
                return {'allocation': row['intent']['operation_id'], 'resource_id': row.get('resource_id'),
                        'launch_operation_id': (row.get('launch') or {}).get('operation_id'),
                        'runtime_id': row.get('runtime_id')}

            current = next(row for row in state['resources'].values() if row.get('resource_id') == identifier)
            require(current.get('launch') and current.get('runtime_id'), 'Pool completion requires a bound launched runtime')
            acknowledgments['resources'][identifier] = binding(current)
            atomic_json(path, acknowledgments)
            for row in state['resources'].values():
                if row.get('retired_verified') is True and row.get('state') == 'STOPPED':
                    continue
                if not row.get('launch'):
                    continue  # No command has been sent, hence no local cache to seal.
                if acknowledgments['resources'].get(row.get('resource_id')) != binding(row):
                    return False
            return True


def preflight(config_path, stage, *, require_contract=True):
    """Check remote path/content identity before any task or accelerator work."""
    require(stage in STAGES, 'Unsupported fleet worker stage')
    path = Path(config_path).resolve(strict=True)
    cfg = load(path)
    require(not _is_stopped(cfg, stage), 'Pipeline/stage STOP forbids worker startup')
    contract_path = Path(cfg['output']) / 'fleet/PREFLIGHT.json'
    if not contract_path.exists() and not require_contract:
        return {'passed': True, 'fleet_contract_present': False, 'scope': 'manual_nonfleet_worker'}
    contract = json.loads(contract_path.read_text())
    identity = contract['identity']
    require(str(path) == identity['config_path'] and file_sha(path) == identity['config_sha256'],
            'Remote config path/content differs from controller')
    identity_path = Path(cfg['output']) / 'fleet/IDENTITY.json'
    require(file_sha(identity_path) == contract['identity_sha256'] and json.loads(identity_path.read_text()) == identity,
            'Remote fleet identity differs')
    require(digest(contract['runtime']) == identity['runtime_sha256'], 'Remote runtime binding list changed')
    require(contract['dataset_roots'] == sorted({item['root'] for item in cfg['inputs']}) and
            contract['cache_directory'] == cfg['cache']['directory'], 'Remote mount contract changed')
    require(identity['stage_budgets'][stage] == cfg['resources'][STAGES[stage]], 'Remote stage budget changed')
    require(Path(__file__).resolve().parents[2] == Path(identity['remote_repository']).resolve(),
            'Remote imported repository does not match deployment template')
    for item in contract['runtime']:
        target = Path(identity['remote_repository']) / item['relative_path']
        require(file_sha(target) == item['sha256'], 'Remote runtime differs: ' + item['relative_path'])
    sentinel = contract['shared_sentinel']
    require(file_sha(sentinel['path']) == sentinel['sha256'], 'Remote shared output path is missing or differs')
    for root in contract['dataset_roots']:
        require(Path(root).is_dir() and os.access(root, os.R_OK | os.X_OK), 'Dataset mount missing/unreadable: ' + root)
    output = Path(cfg['output'])
    require(output.is_dir() and os.access(output, os.R_OK | os.W_OK | os.X_OK), 'Shared output is not writable')
    cache = Path(contract['cache_directory'])
    cache_parent = cache
    while not cache_parent.exists() and cache_parent.parent != cache_parent:
        cache_parent = cache_parent.parent
    require(os.access(cache_parent, os.R_OK | os.W_OK | os.X_OK), 'Shared cache parent is not writable')
    cards = int(os.environ.get('RESOURCE_MANAGER_CARDS', '1'))
    require(1 <= cards <= identity['stage_budgets'][stage] <= identity['total_max_cards'] <= 512, 'Remote allocation exceeds budget')
    return {'passed': True, 'fleet_contract_present': True, 'stage': stage, 'cards': cards,
            'config_sha256': identity['config_sha256'], 'identity_sha256': contract['identity_sha256'],
            'runtime_files_checked': len(contract['runtime']), 'mount_contract': contract['mount_contract']}


def completion(config_path, stage):
    """Global completion only; local worker exit is never sufficient evidence."""
    require(stage in STAGES, 'Unsupported fleet completion stage')
    cfg = load(config_path)
    if _is_stopped(cfg, stage):
        return False
    root = Path(cfg['output'])
    allocation = os.environ.get('RESOURCE_MANAGER_ALLOCATION_ID')
    if allocation is not None:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', allocation):
            return False
        pool_path = root / 'fleet/pools' / stage / (allocation + '.json')
        if not pool_path.exists():
            return False
        pool = json.loads(pool_path.read_text()); binding = pool.get('binding', {})
        runtime_id = socket.gethostname() + ':' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        cards = int(os.environ.get('RESOURCE_MANAGER_CARDS', '0'))
        if not (pool.get('status') == 'complete' and not pool.get('fatal') and
                pool.get('allocation') == binding.get('allocation') == allocation and
                pool.get('stage') == binding.get('stage') == stage and
                pool.get('cards') == binding.get('cards') == cards and 1 <= cards <= cfg['resources'][STAGES[stage]] and
                pool.get('host') == socket.gethostname() and pool.get('runtime_id') == runtime_id and
                pool.get('active_children') == {} and binding.get('schema') == 'corpus.unified.worker_pool.v1' and
                binding.get('config_path') == str(Path(config_path).resolve()) and
                binding.get('config_sha256') == file_sha(config_path) and
                binding.get('execution_sha256') == file_sha(root / 'EXECUTION.json')):
            return False
    if stage == 'frontpass':
        from corpus_pipeline.cluster.queue_rpc import status
        value = status(root / 'frontpass')
        return value.get('all_tasks_completed') is True and value.get('counts', {}).get('failed', 0) == 0
    area = root / 'sscd_search'
    marker = area / 'COMPLETE.json'
    if not marker.exists():
        return False
    value = json.loads(marker.read_text())
    plan = json.loads((area / 'PLAN.json').read_text())
    calibration = json.loads((area / 'calibration/COMPLETE.json').read_text())
    return (value.get('complete') is True and value.get('all_pairs_covered') is True and
            calibration.get('passed') is True and
            value.get('plan_sha256') == file_sha(area / 'PLAN.json') and
            value.get('calibration_sha256') == file_sha(area / 'calibration/COMPLETE.json') and
            value.get('completed_tasks') == value.get('total_tasks') == plan['total_tasks'] and
            value.get('completed_pairs') == value.get('total_pairs') == plan['total_pairs'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'preflight', 'completion', 'stop'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--template')
    parser.add_argument('--output')
    parser.add_argument('--stage', choices=tuple(STAGES))
    args = parser.parse_args(argv)
    if args.action == 'build':
        require(bool(args.template), '--template required for fleet build')
        result = build(args.config, args.template, args.output)
    elif args.action == 'preflight':
        result = preflight(args.config, args.stage)
    elif args.action == 'stop':
        result = request_stop(args.config)
    else:
        passed = completion(args.config, args.stage)
        print(json.dumps({'complete': passed, 'stage': args.stage}))
        return 0 if passed else 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
