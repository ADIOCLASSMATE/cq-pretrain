"""Configuration-bound LOW native Serving pool with an exclusive HIGH handoff.

One process owns one native probe scope. Four independently qualified 32-card
replicas; unknown Create outcomes reserve capacity and are never retransmitted.
No command in this module changes precision, creates Jobs, or replaces an ID.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from inspire.sdk import InspireClient
from inspire.sdk.models_serving import ServingCreateSpec
from corpus_pipeline.serving import ai_review_pool as legacy

native = legacy.native
observable_project_budget = legacy.observable_project_budget
SDK_PYTHON = legacy.SDK_PYTHON
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RUNTIME_FILES = ('serve.py', 'gateway.py', 'launch.sh', 'hccl_probe.py')
TERMINAL = {'INSTANCE_STOPPED', 'INSTANCE_FAILED', 'INSTANCE_SUCCEEDED',
            'INSTANCE_DELETED', 'INSTANCE_TERMINATED', 'INSTANCE_COMPLETED'}
_NATIVE_SCOPE = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def locked(path, *, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def persist(path, state):
    native.save(path, {key: value for key, value in state.items() if key != 'endpoint'})


def absolute(value):
    path = Path(value)
    require(path.is_absolute(), 'Configuration paths must be absolute')
    return path.resolve()


@dataclass(frozen=True)
class PoolSpec:
    config_path: Path
    config_sha256: str
    root: Path
    lab: Path
    runtime: Path
    profile_path: Path
    manifest_path: Path
    baseline_path: Path
    gate_path: Path
    retired_roots: tuple[Path, ...]
    name_prefix: str
    replicas: int
    raw: dict

    @classmethod
    def load(cls, path):
        path = absolute(path)
        raw = native.load(path)
        require(raw.get('schema') == 'corpus.native_serving_pool.v1', 'Unsupported pool config schema')
        require(raw.get('cards_per_replica', 32) == 32 and raw.get('max_cards', 128) == 128
                and raw.get('priority', 3) == 3, 'Pool must remain LOW BF16, 4 x 32, max 128 cards')
        replicas = raw.get('replicas', 4)
        require(type(replicas) is int and 1 <= replicas <= 4, 'Replica count must be 1..4')
        if 'per_replica_max_inflight' in raw:
            cap = raw['per_replica_max_inflight']
            require(type(cap) is int and cap > 0, 'per_replica_max_inflight must be a positive integer')
        if 'bootstrap_from_qualified_winner' in raw:
            require(type(raw['bootstrap_from_qualified_winner']) is bool, 'bootstrap_from_qualified_winner must be boolean')
        if raw.get('bootstrap_from_qualified_winner', False):
            require(raw.get('winner_receipt_path') and raw.get('winner_receipt_sha256'), 'Bootstrap requires an explicit winner receipt')
        prefix = raw['name_prefix']
        require(re.fullmatch(r'[a-z0-9][a-z0-9-]{3,60}', prefix), 'Invalid unique service name prefix')
        root, lab, runtime = (absolute(raw[key]) for key in ('scope_root', 'retired_lab_root', 'runtime_dir'))
        retired = tuple(absolute(p) for p in raw['retired_pool_roots'])
        require(retired and len(set(retired)) == len(retired), 'Retired LOW pool coverage is required')
        require(all(root != old and not root.is_relative_to(old) and not old.is_relative_to(root)
                    for old in (lab, *retired)), 'New scope must be separate from retired pools and HIGH lab')
        require(runtime == lab / 'runtime', 'Runtime must be the frozen lab runtime')
        profile = absolute(raw['profile_path']); manifest = absolute(raw['runtime_manifest_path'])
        baseline = absolute(raw['baseline_state_path']); gate = absolute(raw['high_gate_receipt_path'])
        require(profile.parent == lab / 'profiles' and manifest == lab / 'RUNTIME_MANIFEST.json',
                'Profile and manifest must belong to the retired lab')
        require(gate == lab / 'HANDOFF_GATE_READY.json', 'Unexpected HIGH gate receipt path')
        require(any(baseline.parent == old for old in retired), 'Baseline must belong to a retired LOW pool')
        require(bool(raw.get('winner_receipt_path')) == bool(raw.get('winner_receipt_sha256')),
                'Winner receipt path/hash must be supplied together')
        return cls(path, sha(path), root, lab, runtime, profile, manifest, baseline,
                   gate, retired, prefix, replicas, raw)


class NativeServingPool:
    def __init__(self, spec):
        self.spec = spec
        self.root = spec.root
        self.root.mkdir(parents=True, exist_ok=True)

    def disabled(self):
        return (self.root / 'DISABLED').exists()

    def runtime_scope(self):
        global _NATIVE_SCOPE
        require(_NATIVE_SCOPE in (None, self.root), 'Native probes require one pool scope per process')
        _NATIVE_SCOPE = self.root
        native.RUNTIME = self.root

    def command(self, action):
        return [SDK_PYTHON, '-m', 'corpus_pipeline.serving.configured_pool', action,
                '--config', str(self.spec.config_path)]

    def binding(self):
        spec = self.spec
        require(sha(spec.config_path) == spec.config_sha256, 'Pool configuration changed')
        require(sha(Path(__file__)) == CODE_SHA256, 'Controller source changed after process start')
        require(sha(spec.manifest_path) == spec.raw['runtime_manifest_sha256'], 'Runtime manifest changed')
        require(sha(spec.profile_path) == spec.raw['profile_sha256'], 'Winner profile changed')
        files = native.load(spec.manifest_path)['files']
        required = {'runtime/' + name for name in RUNTIME_FILES} | {str(spec.profile_path.relative_to(spec.lab))}
        require(required <= set(files), 'Runtime manifest is incomplete')
        for relative, expected in files.items():
            path = (spec.lab / relative).resolve()
            require(path.is_relative_to(spec.lab) and sha(path) == expected, 'Manifest-bound file changed')
        config, base = native.load(spec.profile_path), native.load(spec.baseline_path)
        require(config['model_name'] == base['model'] == 'Qwen3.5-397B-A17B'
                and config['precision'] == base['precision'] == 'BF16', 'Model or precision changed')
        require(all(config[k] == base[k] for k in ('revision', 'processor_revision')), 'Model revisions changed')
        require(config['nodes'] == 2 and config['data_parallel_size'] == 4
                and config['max_model_len'] == 65536 and config['max_images'] == base['max_images'] == 16, 'Input/topology contract changed')
        epoch = config['epoch']
        require(isinstance(epoch, str) and re.fullmatch(r'[a-zA-Z0-9_-]+', epoch), 'Unsafe runtime epoch')
        runtime_hashes = {name: files['runtime/' + name] for name in RUNTIME_FILES}
        bound = {'config_sha256': spec.config_sha256, 'profile_sha256': sha(spec.profile_path),
                 'runtime_manifest_sha256': sha(spec.manifest_path), 'runtime_files': runtime_hashes,
                 'controller_code_sha256': CODE_SHA256, 'native_monitor_sha256': sha(Path(native.__file__)),
                 'serving_code_revision': 'sha256:' + hashlib.sha256(json.dumps(runtime_hashes, sort_keys=True).encode()).hexdigest()}
        if spec.raw.get('winner_receipt_path'):
            require(sha(absolute(spec.raw['winner_receipt_path'])) == spec.raw['winner_receipt_sha256'], 'Winner receipt changed')
            bound['winner_receipt_sha256'] = spec.raw['winner_receipt_sha256']
        return config, base, bound

    def qualified_winner(self, config, base, bound):
        """Validate an explicit root-approved HIGH proof; never confer LOW readiness.

        Called once per provision transaction under all lifecycle locks. The
        winner receipt hash remains in binding() and is rechecked before every
        Create. Each LOW boot still runs loaded_evidence/health/native.probe.
        """
        from corpus_pipeline.production.common import digest
        from corpus_pipeline.production.image_screening import _receipt_errors

        winner_path = absolute(self.spec.raw['winner_receipt_path'])
        require(sha(winner_path) == self.spec.raw['winner_receipt_sha256'], 'Winner receipt changed')
        winner = native.load(winner_path)
        require(winner.get('schema') == 'corpus.high_lab.bootstrap_winner.v1'
                and winner.get('issuer') == 'root' and winner.get('approved') is True
                and winner.get('purpose') == 'bootstrap_low_parallel_submission_only', 'Winner is not explicitly approved for bootstrap')
        require(winner.get('controller_code_sha256') == CODE_SHA256, 'Winner controller code differs')
        profile = self.spec.profile_path.stem
        require(winner.get('profile') == profile, 'Winner is a different profile')
        for key in ('profile_sha256', 'runtime_manifest_sha256', 'runtime_files', 'serving_code_revision'):
            require(winner.get(key) == bound[key], 'Winner profile/runtime binding differs')
        corpus = Path(__file__).resolve().parents[1]
        expected_code = {
            'high_lab.py': self.spec.lab / 'high_lab.py',
            'client_benchmark.py': self.spec.lab / 'client_benchmark.py',
            'production/client.py': corpus / 'production/client.py',
            'production/validation.py': corpus / 'production/validation.py',
            'production/image_screening.py': corpus / 'production/image_screening.py',
            'production/common.py': corpus / 'production/common.py',
            'api_review/aggregate.py': corpus / 'api_review/aggregate.py',
        }
        require(winner.get('code_sha256') == {key: sha(path) for key, path in expected_code.items()},
                'Winner validation/client/HIGH code changed')
        artifacts = winner.get('artifacts', {})
        require(set(artifacts) == {'high_state', 'attestation', 'canary', 'benchmark_complete',
                                  'benchmark_result', 'input', 'prompt'}, 'Winner artifacts incomplete')
        paths = {}
        for key, record in artifacts.items():
            path = absolute(record['path'])
            permitted = (path.parent == corpus / 'production' and
                         path.name in ('image_screening_prompt_v3.txt', 'image_screening_prompt_v4.txt')) if key == 'prompt' else path.is_relative_to(self.spec.lab)
            require(permitted and sha(path) == record['sha256'], 'Winner artifact changed or escaped scope')
            paths[key] = path
        require(paths['high_state'] == self.spec.lab / 'states' / (profile + '.json')
                and paths['attestation'] == self.spec.lab / 'observations' / profile / 'attestation.json',
                'Winner state/attestation not from exact HIGH profile')
        high = native.load(paths['high_state']); attest = native.load(paths['attestation'])
        require(high.get('profile') == profile and high.get('scope') == str(self.spec.lab)
                and high.get('tier') == 'high' and high.get('priority') == 6 and high.get('cards') == 32
                and high.get('release_verified') is True and high.get('service_id') == winner.get('service_id')
                and high.get('config_snapshot') == config, 'Winner HIGH identity/profile/release differs')
        for key in ('profile_sha256', 'runtime_manifest_sha256', 'runtime_files', 'serving_code_revision'):
            require(high['binding'].get(key) == bound[key], 'Winner HIGH loaded binding differs')
        require(high['binding'].get('lab_code_sha256') == winner['code_sha256']['high_lab.py'], 'Winner HIGH code differs')
        require(all(high.get(key) == base.get(key) for key in
                    ('model', 'revision', 'processor_revision', 'precision', 'request_overrides')), 'Winner model/input contract differs')
        require(paths['canary'] == absolute(high['qualification_receipt'])
                and paths['canary'].is_relative_to(self.spec.lab / 'probes'), 'Winner canary does not belong to HIGH')
        boots = attest.get('expected_boots')
        require(isinstance(boots, list) and len(boots) == len(set(boots)) == 2
                and boots == high.get('expected_boots') and attest.get('incarnation') == high.get('incarnation')
                and attest.get('runtime_revision') == high.get('runtime_revision'), 'Winner boot attestation differs')
        expected_config = {key: value for key, value in config.items() if 'key' not in key and key != 'processor_files'}
        reports = attest.get('reports', [])
        require(len(reports) == 2 and all(row.get('runtime_config') == expected_config and
                    row.get('serving_code_revision') == bound['serving_code_revision'] and
                    all(row.get(key) == base.get(key) for key in ('model', 'revision', 'processor_revision', 'precision'))
                    for row in reports), 'Winner loaded runtime/model differs')
        canary = native.load(paths['canary']); requests = canary.get('requests', [])
        require(canary.get('passed') is True and canary.get('authentication_passed') is True
                and canary.get('finished_at') and canary.get('expected_boots') == boots
                and canary.get('model') == base['model'] and canary.get('revision') == base['revision']
                and canary.get('request_overrides') == base['request_overrides']
                and len(requests) == 13 and all(row.get('passed') is True and row.get('expected')
                    and row.get('actual') == row['expected'] and row.get('boot_id') in boots for row in requests),
                'Winner 13-request current-boot canary did not pass')
        result_path = paths['benchmark_result']; complete_path = paths['benchmark_complete']
        require(result_path.name == 'RESULT.json' and complete_path == result_path.parent / 'COMPLETE.json'
                and result_path.is_relative_to(self.spec.lab / 'trials'), 'Winner benchmark artifact scope differs')
        complete = native.load(complete_path); result = native.load(result_path)
        count = result.get('count'); results = result.get('results', [])
        require(type(count) is int and 512 <= count <= 2048 and len(results) == count
                and result.get('schema') == 'corpus.client_infra_benchmark.v1'
                and result.get('scope') == 'isolated_native_load_test_not_training_decisions'
                and result.get('successful') == complete.get('successful') == count
                and result.get('failed') == complete.get('failed') == 0
                and result.get('cache_hits') == 0 and complete.get('completed') is True
                and complete.get('published_training') is False
                and complete.get('summary_sha256') == sha(result_path), 'Winner benchmark is incomplete or has failures')
        require(result.get('input_sha256') == artifacts['input']['sha256']
                and result.get('prompt_sha256') == digest(paths['prompt'].read_text())
                and result.get('model') == base['model']
                and result.get('code_sha256') == winner['code_sha256']['client_benchmark.py'], 'Winner benchmark code/input differs')
        concurrency = result.get('concurrency')
        require(type(concurrency) is int and concurrency > 0
                and type(winner.get('max_tokens')) is int and winner['max_tokens'] > 0,
                'Winner request limits invalid')
        if 'per_replica_max_inflight' in self.spec.raw:
            require(self.spec.raw['per_replica_max_inflight'] <= concurrency, 'Pool cap exceeds tested per-replica concurrency')
        hashes = winner.get('benchmark_receipt_sha256', {})
        keys = [row.get('receipt_cache_key') for row in results]
        assets = [row.get('asset_id') for row in results]
        require(len(set(keys)) == len(set(assets)) == count and set(hashes) == set(keys)
                and all(isinstance(key, str) and re.fullmatch(r'[0-9a-f]{64}', key) for key in keys),
                'Winner benchmark receipt coverage is incomplete')
        for row in results:
            key = row['receipt_cache_key']; receipt_path = result_path.parent / 'cache' / (key + '.json')
            require(row.get('status') == 'ok' and row.get('cache_hit') is False
                    and sha(receipt_path) == hashes[key], 'Winner benchmark response failed or changed')
            receipt = native.load(receipt_path); identity = receipt.get('identity', {})
            require(not _receipt_errors(receipt, row['asset_id']) and receipt.get('cache_hit') is False
                    and receipt.get('cache_key') == key and receipt.get('qualified_boots_at_request') == boots
                    and receipt.get('response_boot_id') in boots, 'Winner response is not a valid actual boot-bound image inference')
            require(all(identity.get(k) == base.get(k) for k in ('model', 'revision', 'processor_revision', 'precision', 'request_overrides'))
                    and identity.get('runtime_revision') == attest['runtime_revision']
                    and identity.get('prompt_sha256') == hashlib.sha256(paths['prompt'].read_text().encode()).hexdigest()
                    and identity.get('max_tokens') == winner['max_tokens']
                    and identity.get('processing_code_identity') == {
                        'client.py': winner['code_sha256']['production/client.py'],
                        'validation.py': winner['code_sha256']['production/validation.py']},
                    'Winner response used a different execution contract')
        proof = {'schema': 'corpus.low_pool.bootstrap_validation.v1', 'validated_at': time.time(),
                 'winner_receipt_sha256': self.spec.raw['winner_receipt_sha256'],
                 'controller_code_sha256': CODE_SHA256, 'profile_sha256': bound['profile_sha256'],
                 'runtime_manifest_sha256': bound['runtime_manifest_sha256'], 'high_service_id': high['service_id'],
                 'canary_requests': 13, 'valid_real_benchmark_results': count,
                 'low_readiness_inherited': False}
        native.save(self.root / 'BOOTSTRAP_VALIDATION.json', proof)
        return proof

    def paths(self, number):
        require(type(number) is int and 1 <= number <= self.spec.replicas, 'Slot exceeds configured budget')
        name = f'qwen-{number:02}.json'
        return self.root / 'states' / name, self.root / 'intents' / name

    def numbers(self):
        values = {p.stem for directory in ('states', 'intents') for p in (self.root / directory).glob('*.json')}
        require(all(re.fullmatch(r'qwen-0[1-4]', value) for value in values), 'Unexpected managed reservation file')
        return sorted(int(value.split('-')[1]) for value in values)

    def validate(self, state, number):
        expected = self.spec.name_prefix + f'-{number:02}'
        require(state.get('scope') == str(self.root) and state.get('slot') == number
                and state.get('config_sha256') == self.spec.config_sha256, 'State belongs to another scope/config')
        require(state.get('name') == state.get('service_name') == state.get('operation_id') == expected,
                'Managed service identity changed')
        require(state.get('tier') == 'low' and state.get('priority') == 3 and state.get('cards') == 32
                and state.get('expected_nodes') == 2 and state.get('expected_replicas') == 1
                and state.get('precision') == 'BF16' and state.get('auto_recovery_enabled') is True,
                'Managed resource contract changed')
        self.paths(number)

    def load(self, number):
        path, intent = self.paths(number)
        require(intent.exists(), 'Immutable reservation missing')
        original = native.load(intent)
        state = native.load(path) if path.exists() else original
        self.validate(original, number); self.validate(state, number)
        require(all(state.get(k) == original.get(k) for k in ('scope', 'slot', 'config_sha256', 'name',
                'operation_id', 'payload', 'binding', 'deployment_id', 'config_snapshot')), 'Reservation identity changed')
        return state

    def rows(self):
        rows = [self.load(n) for n in self.numbers()]
        require(len(rows) <= self.spec.replicas and len(rows) * 32 <= 128, '128-card reservation budget exceeded')
        ids = [row['service_id'] for row in rows if row.get('service_id')]
        require(len(ids) == len(set(ids)), 'Duplicate managed resource ID')
        return rows

    def reconcile(self, client, number, state):
        if not state.get('service_id'):
            matches = [r for r in client.servings.iter(workspace=native.WORKSPACE) if r.name == state['name']]
            require(len(matches) == 1, 'Create outcome unresolved; reservation retained, no retransmission')
            detail = client.servings.get(matches[0].ref, workspace=native.WORKSPACE)
            state = {**state, 'service_id': detail.ref.key, 'phase': 'OBSERVING'}
        else:
            detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
        native.assert_owned(state, detail)
        persist(self.paths(number)[0], state)
        return state

    def terminal(self, client, state):
        require(state.get('service_id'), 'Unresolved resource identity')
        detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
        native.assert_owned(state, detail)
        statuses = [v.status for v in client.servings.instances(detail.ref, workspace=native.WORKSPACE)]
        return {'service_id': detail.ref.key, 'name': detail.name, 'status': detail.status,
                'instance_statuses': statuses, 'terminal': detail.status == 'STOPPED' and set(statuses) <= TERMINAL}

    def gate_proof(self):
        proof = native.load(self.spec.gate_path)
        entry = self.spec.lab / 'high_lab.py'
        require(proof.get('schema') == 'corpus.high_lab.handoff_gate.v1'
                and proof.get('enabled') is True and proof.get('tests_passed') is True
                and proof.get('gate_inside_lifecycle_lock') is True
                and proof.get('gate_marker') == 'PRODUCTION_HANDOFF.json'
                and proof.get('entry_path') == str(entry)
                and proof.get('entry_sha256') == proof.get('tested_entry_sha256') == sha(entry)
                and 'PRODUCTION_HANDOFF.json' in entry.read_text(), 'HIGH handoff gate is absent, changed, or untested')
        return {'gate_receipt_sha256': sha(self.spec.gate_path), 'entry_sha256': sha(entry)}

    def predecessors_released(self, client):
        """Called with lab lifecycle lock; historical unknown intents never free cards."""
        proof = self.gate_proof()
        released = []
        with ExitStack() as stack:
            for root in sorted(self.spec.retired_roots):
                stack.enter_context(locked(root / 'provision.lock'))
                stack.enter_context(locked(root / 'lifecycle.lock'))
                require((root / 'DISABLED').exists(), 'Retired LOW pool is not disabled')
                receipt = native.load(root / 'STOPPED.json')
                require(receipt.get('complete') is True, 'Retired LOW release is not verified')
                rows = [native.load(path) for path in sorted(root.glob('qwen-[0-9][0-9].json'))]
                legacy.validate_budget(rows)
                require(rows and all(row.get('service_id') for row in rows), 'Retired LOW contains unresolved reservations')
                require({r['service_id'] for r in rows} == {r['service_id'] for r in receipt['services']}, 'LOW release coverage changed')
                for row in rows:
                    require(self.terminal(client, row)['terminal'], 'Retired LOW is not fully stopped now')
                released.append({'root': str(root), 'receipt_sha256': sha(root / 'STOPPED.json')})
        lab = self.spec.lab
        profiles = {p.stem for directory in ('states', 'intents') for p in (lab / directory).glob('*.json')}
        high_ids = set()
        for profile in sorted(profiles):
            intent_path = lab / 'intents' / (profile + '.json')
            state_path = lab / 'states' / (profile + '.json')
            require(intent_path.exists() and state_path.exists(), 'HIGH contains unresolved intent')
            original, row = native.load(intent_path), native.load(state_path)
            require(all(row.get(k) == original.get(k) for k in ('scope', 'profile', 'name', 'operation_id', 'payload', 'binding', 'deployment_id')),
                    'HIGH immutable identity changed')
            require(row.get('scope') == str(lab) and row.get('tier') == 'high' and row.get('priority') == 6
                    and row.get('cards') == 32 and row.get('release_verified') is True and row.get('service_id'), 'HIGH is unresolved or unreleased')
            require(row['service_id'] not in high_ids, 'Duplicate HIGH ID'); high_ids.add(row['service_id'])
            receipt_path = lab / 'observations' / profile / 'STOPPED.json'
            receipt = native.load(receipt_path)
            require(sha(receipt_path) == row.get('release_receipt_sha256') and receipt.get('complete') is True
                    and receipt.get('service_id') == row['service_id'] and receipt.get('binding') == row['binding'], 'HIGH release receipt changed')
            require(self.terminal(client, row)['terminal'], 'HIGH allocation is not fully stopped now')
            released.append({'profile': profile, 'service_id': row['service_id'], 'receipt_sha256': sha(receipt_path)})
        return {**proof, 'predecessors': released}

    def handoff(self, client):
        evidence = self.predecessors_released(client)
        path = self.spec.lab / 'PRODUCTION_HANDOFF.json'
        identity = {'scope_root': str(self.root), 'config_sha256': self.spec.config_sha256,
                    'controller_code_sha256': CODE_SHA256, 'guard_command': self.command('guard')}
        if path.exists():
            old = native.load(path)
            require(all(old.get(k) == value for k, value in identity.items()), 'HIGH lab already handed to another pool/config')
            require(old.get('gate_receipt_sha256') == evidence['gate_receipt_sha256'], 'HIGH gate proof changed after handoff')
        else:
            require(not self.disabled(), 'Pool is disabled')
            native.save(path, {'schema': 'corpus.high_lab.production_handoff.v1', 'created_at': time.time(),
                               **identity, **evidence})
        return evidence

    def budget(self, client, extra=0):
        rows = self.rows()
        require(type(extra) is int and extra >= 0 and len(rows) * 32 + extra <= 128, '128-card pool cap exceeded')
        live = observable_project_budget(client, 0)
        records = live['records']
        missing = sum(32 for row in rows if not row.get('service_id') or row['service_id'] not in records)
        used = sum(int(row['cards']) for row in records.values())
        require(used + missing + extra <= 256, 'Account budget including unresolved reservations exceeds 256')
        return {**live, 'pool_reserved_cards': len(rows) * 32, 'unreflected_reserved_cards': missing,
                'requested_extra_cards': extra, 'conservative_total_cards': used + missing + extra}

    def loaded_evidence(self, state, instances):
        evidence = native.attest(state, instances)
        expected = {k: v for k, v in state['config_snapshot'].items() if 'key' not in k and k != 'processor_files'}
        reports = evidence['reports']
        require(len(reports) == 2 and all(r.get('runtime_config') == expected and
                r.get('serving_code_revision') == state['binding']['serving_code_revision'] for r in reports), 'Current boot loaded a different profile/runtime')
        return evidence

    def first_qualified(self, client, bound):
        state = self.load(1)
        require(state['binding'] == bound and state.get('canary_passed') is True and state.get('qualification_receipt')
                and 0 <= time.time() - state.get('heartbeat_at', 0) < 30, 'First replica needs fresh current-boot image qualification')
        detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
        native.assert_owned(state, detail)
        state['endpoint'] = detail.to_dict()['endpoint']
        evidence = self.loaded_evidence(state, client.servings.instances(detail.ref, workspace=native.WORKSPACE))
        require(detail.status == 'RUNNING' and evidence['incarnation'] == state.get('incarnation')
                and native.health(state, evidence), 'First replica current-boot qualification expired')

    def pool_index(self):
        limits = ({'max_inflight': self.spec.raw['per_replica_max_inflight']}
                  if 'per_replica_max_inflight' in self.spec.raw else {})
        native.save(self.root / 'pool.json', {'schema_version': 'corpus.screening_pool.v1', 'max_cards': 128,
            'cards_per_replica': 32, 'config_sha256': self.spec.config_sha256,
            'registries': [{'path': str(self.root / f'registry-{n:02}.json'), 'weight': 1, **limits}
                           for n in range(1, self.spec.replicas + 1)]})

    def provision(self, target=None):
        target = self.spec.replicas if target is None else target
        require(type(target) is int and 1 <= target <= self.spec.replicas, 'Requested replicas exceed config')
        with locked(self.root / 'provision.lock'), locked(self.root / 'lifecycle.lock'), locked(self.spec.lab / 'lifecycle.lock'):
            require(not self.disabled(), 'Pool is disabled')
            config, base, bound = self.binding()
            bootstrap = (self.qualified_winner(config, base, bound)
                         if self.spec.raw.get('bootstrap_from_qualified_winner', False) else None)
            with InspireClient(account='wjx-ascend') as client:
                self.handoff(client)
                for number in self.numbers():
                    self.reconcile(client, number, self.load(number))
                for number in range(1, target + 1):
                    if number in self.numbers():
                        continue
                    if number > 1 and not bootstrap:
                        self.first_qualified(client, bound)
                    require(not self.disabled(), 'Pool disabled during provisioning')
                    self.budget(client, 32)
                    name = self.spec.name_prefix + f'-{number:02}'
                    deployment = name + '-' + self.spec.config_sha256[:8]
                    command = shlex.join(['env', 'TZ=Asia/Shanghai', 'CORPUS_CONFIG=' + str(self.spec.profile_path),
                        'CORPUS_DEPLOYMENT_ID=' + deployment, 'bash', str(self.spec.runtime / 'launch.sh')])
                    spec = ServingCreateSpec(name=name, model='corpus-qwen397b-bf16-ref-20261023', model_version=1,
                        command=command, port=8000, workspace=native.WORKSPACE, project=native.PROJECT,
                        group='910B资源', quota='16,128,1024', image='vllm-ascend:v0.23.0rc1', replicas=1,
                        nodes_per_replica=2, shm_gib=128, priority=3, auto_scaling=False, public_path_readonly=True,
                        description='Configuration-bound BF16 native review pool; 32 cards per replica; exclusive 128-card budget')
                    plan = client.servings.plan(spec); payload = plan.create_kwargs
                    require(plan.priority == 3 and payload['task_priority'] == 3 and payload['replicas'] == 1
                            and payload['node_num_per_replica'] == 2, 'Resolved LOW topology changed')
                    require(all(payload['resource_spec_price'][k] == v for k, v in
                            {'gpu_count': 16, 'cpu_count': 128, 'memory_size_gib': 1024}.items()), 'Resolved quota changed')
                    require(not any(r.name == name for r in client.servings.iter(workspace=native.WORKSPACE)), 'Unowned same-name resource exists')
                    self.handoff(client)
                    budget = self.budget(client, 32)
                    if number > 1 and not bootstrap:
                        self.first_qualified(client, bound)
                    require(self.binding()[2] == bound, 'Binding changed during planning')
                    require(not self.disabled(), 'Pool disabled before Create intent')
                    state = {k: base[k] for k in ('schema_version', 'model', 'revision', 'processor_revision',
                             'precision', 'max_images', 'api_key_file', 'api_key_env', 'request_overrides')}
                    state.update(scope=str(self.root), slot=number, config_sha256=self.spec.config_sha256,
                        role='generator', tier='low', priority=3, cards=32, expected_nodes=2, expected_replicas=1,
                        auto_recovery_enabled=True, runtime_layout='prefixed_groups',
                        runtime_root=str(self.spec.runtime / 'runtime' / config['epoch']), name=name, service_name=name,
                        operation_id=name, deployment_id=deployment, payload=payload, binding=bound, config_snapshot=config,
                        phase='CREATE_INTENT', created_at=time.time(), canary_passed=False, release_verified=False, probe_attempt=1)
                    path, intent = self.paths(number)
                    self.validate(state, number)
                    native.save(self.root / f'budget-before-{number:02}.json', budget)
                    persist(intent, state); persist(path, state)
                    require(not self.disabled(), 'Pool disabled; unsent reservation retained')
                    result = client.servings.create(spec, operation_id=name)
                    state.update(service_id=result.ref.key, phase='OBSERVING')
                    persist(path, state)
                    self.pool_index()
                    # Return normally while the first boot is awaiting its image probe.
                    if number == 1 and not bootstrap:
                        break
                self.pool_index()
                return {'reserved_cards': len(self.rows()) * 32, 'replicas': len(self.rows())}

    def publish(self, number, state, ready):
        target = self.root / f'registry-{number:02}.json'
        old = native.load(target) if target.exists() else {'registry_generation': 0, 'endpoints': []}
        value = dict(state)
        if not value.get('endpoint') and old['endpoints']:
            value['endpoint'] = old['endpoints'][0]['base_url'].removesuffix('/v1')
        row = native.candidate(value, bool(ready and not self.disabled()))
        row['role'] = 'verifier'
        native.save(target, {'schema_version': 'corpus_pipeline.endpoints.v1', 'registry_generation': old['registry_generation'] + 1,
            'endpoints': [row], 'updated_at': time.time(), 'publisher': 'configured_native_serving_pool'})

    def observe_once(self, number):
        self.runtime_scope()
        with locked(self.root / 'lifecycle.lock'):
            state = self.load(number); ready = False
            try:
                require(not self.disabled(), 'Pool is disabled')
                require(self.binding()[2] == state['binding'], 'Execution binding changed after Create')
                with InspireClient(account='wjx-ascend') as client:
                    state = self.reconcile(client, number, state)
                    detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
                    native.assert_owned(state, detail)
                    instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
                    previous = state.get('platform_status')
                    state.update(platform_status=detail.status, endpoint=detail.to_dict()['endpoint'],
                                 last_observed_at=time.time(), observation_error=None)
                    if detail.status == 'RUNNING':
                        state.update(was_running=True, last_running_at=time.time(), stopped_transition_at=None)
                        evidence = self.loaded_evidence(state, instances)
                        if state.get('incarnation') != evidence['incarnation']:
                            state.update(incarnation=evidence['incarnation'], canary_passed=False,
                                         qualification_receipt=None, probe_attempt=1)
                            self.publish(number, state, False)
                        state.update(runtime_revision=evidence['runtime_revision'], expected_boots=evidence['expected_boots'])
                        native.save(self.root / f'attestation-{number:02}.json', evidence)
                        if native.health(state, evidence):
                            ready = bool(native.probe(state, evidence))
                        state['readiness_reason'] = 'QUALIFIED' if ready else state.get('probe_status', 'QUALIFYING')
                    else:
                        state['readiness_reason'] = 'PLATFORM_' + detail.status
                        if detail.status == 'STOPPED' and previous == 'RUNNING':
                            state['stopped_transition_at'] = time.time()
                        events = list(client.servings.events(detail.ref, workspace=native.WORKSPACE, limit=30).items)
                        event = native.recovery_decision(state, detail.status, events)
                        if event:
                            with locked(self.spec.lab / 'lifecycle.lock'):
                                require(not self.disabled(), 'Pool disabled before recovery')
                                self.handoff(client)
                                budget = self.budget(client)
                                require(not self.disabled(), 'Pool disabled during recovery checks')
                                # A lost Start reply cannot repeat this victim event.
                                state.update(restart_event=event, restart_intent_at=time.time(), canary_passed=False,
                                             qualification_receipt=None, probe_attempt=1)
                                persist(self.paths(number)[0], state)
                                native.save(self.root / f'restart-budget-{number:02}.json', budget)
                                client.servings.start(detail.ref, workspace=native.WORKSPACE)
            except Exception as error:
                state['observation_error'] = {'type': type(error).__name__, 'time': time.time()}
                state['readiness_reason'] = 'OBSERVATION_OR_QUALIFICATION_FAILED'
            state.update(canary_passed=bool(ready and not self.disabled()), heartbeat_at=time.time(), monitor_pid=os.getpid())
            persist(self.paths(number)[0], state)
            if state.get('service_id'):
                self.publish(number, state, state['canary_passed'])
            return {k: state.get(k) for k in ('slot', 'service_id', 'platform_status', 'canary_passed', 'heartbeat_at', 'readiness_reason', 'observation_error')}

    def run(self, *, interval=5):
        self.runtime_scope()
        with locked(self.root / 'run.lock', nonblocking=True):
            self.pool_index()
            while not self.disabled():
                started = time.monotonic(); observations = []; error = None
                for number in self.numbers():
                    observations.append(self.observe_once(number))
                try:
                    rows = self.rows()
                    if len(rows) < self.spec.replicas or any(not row.get('service_id') for row in rows):
                        self.provision()
                except Exception as exc:
                    error = {'type': type(exc).__name__, 'time': time.time()}
                native.save(self.root / 'STATUS.json', {**native.identity(os.getpid()), 'time': time.time(),
                    'config_sha256': self.spec.config_sha256, 'loaded_code_sha256': CODE_SHA256,
                    'reserved_cards': len(self.rows()) * 32, 'max_cards': 128, 'services': observations, 'provision_error': error})
                time.sleep(max(0.2, interval - (time.monotonic() - started)))

    def guard(self):
        command = self.command('run')
        with locked(self.root / 'guardian.lock', nonblocking=True):
            while not self.disabled():
                receipt_path = self.root / 'run-process.json'
                receipt = native.load(receipt_path) if receipt_path.exists() else {}
                live = native.identity(receipt.get('pid', -1))
                if not live or live['command'] != command or live['starttime_ticks'] != receipt.get('starttime_ticks'):
                    require(sha(self.spec.config_path) == self.spec.config_sha256, 'Guardian configuration changed')
                    if self.disabled():
                        break
                    with (self.root / 'run.log').open('a') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, cwd=Path(__file__).resolve().parents[2])
                    identity = native.identity(child.pid)
                    require(identity is not None, 'Pool runner exited before identity capture')
                    native.save(receipt_path, {**identity, 'time': time.time(), 'config_sha256': self.spec.config_sha256})
                native.save(self.root / 'guardian-heartbeat.json', {**native.identity(os.getpid()), 'time': time.time(),
                    'config_sha256': self.spec.config_sha256, 'loaded_code_sha256': CODE_SHA256})
                time.sleep(5)

    def stop(self, *, timeout=300):
        (self.root / 'DISABLED').write_text('Configured native Serving pool stop requested\n')
        with locked(self.root / 'provision.lock'), locked(self.root / 'lifecycle.lock'):
            # A broken or unknown slot must not prevent cleanup of other owned IDs.
            native.save(self.root / 'STOPPED.json', {'complete': False, 'time': time.time(), 'config_sha256': self.spec.config_sha256})
            with InspireClient(account='wjx-ascend') as client:
                deadline = time.monotonic() + timeout
                while True:
                    statuses = []; rows = []; errors = []
                    for number in self.numbers():
                        state = None
                        try:
                            state = self.reconcile(client, number, self.load(number))
                            self.publish(number, state, False)
                            detail = client.servings.get(state['name'], workspace=native.WORKSPACE)
                            native.assert_owned(state, detail)
                            if detail.status != 'STOPPED':
                                intent_path = self.root / f'stop-intent-{number:02}.json'
                                if intent_path.exists():
                                    intent = native.load(intent_path)
                                    require(intent.get('service_id') == state['service_id']
                                            and intent.get('config_sha256') == self.spec.config_sha256,
                                            'Stop intent ownership changed')
                                else:
                                    intent = {'time': time.time(), 'service_id': state['service_id'],
                                              'config_sha256': self.spec.config_sha256, 'phase': 'SENDING'}
                                    native.save(intent_path, intent)
                                    try:
                                        client.servings.stop(detail.ref, workspace=native.WORKSPACE)
                                    except Exception as error:
                                        intent.update(phase='UNKNOWN', error_type=type(error).__name__)
                                        errors.append({'slot': number, 'operation': 'stop', 'type': type(error).__name__})
                                    else:
                                        intent['phase'] = 'ACCEPTED'
                                    native.save(intent_path, intent)
                                # An existing SENDING/UNKNOWN/ACCEPTED intent is observed,
                                # never blindly transmitted again after a lost reply.
                            statuses.append({'slot': number, **self.terminal(client, state)})
                            rows.append(state)
                        except Exception as error:
                            errors.append({'slot': number, 'operation': 'reconcile_or_observe', 'type': type(error).__name__})
                            statuses.append({'slot': number, 'service_id': state.get('service_id') if state else None,
                                             'terminal': False, 'error_type': type(error).__name__})
                    complete = all(row['terminal'] for row in statuses)
                    receipt = {'complete': complete, 'time': time.time(), 'scope_root': str(self.root),
                               'config_sha256': self.spec.config_sha256, 'services': statuses, 'errors': errors}
                    native.save(self.root / 'STOPPED.json', receipt)
                    if complete:
                        for state in rows:
                            state.update(release_verified=True, release_receipt_sha256=sha(self.root / 'STOPPED.json'),
                                         platform_status='STOPPED', phase='RELEASED', canary_passed=False)
                            persist(self.paths(state['slot'])[0], state)
                        return receipt
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Stop requested but every reserved resource release is not verified')
                    time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('provision', 'run', 'guard', 'stop'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--replicas', type=int)
    args = parser.parse_args()
    legacy.configure_management_proxy_bypass()
    pool = NativeServingPool(PoolSpec.load(args.config))
    try:
        result = pool.provision(args.replicas) if args.action == 'provision' else getattr(pool, args.action)()
    except Exception as error:
        # SDK exception messages can contain authenticated transport addresses.
        print(json.dumps({'ok': False, 'action': args.action, 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
    if result is not None:
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
