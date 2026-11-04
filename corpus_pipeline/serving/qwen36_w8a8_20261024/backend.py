"""Reusable native Serving adapter; caller owns budgets, locks and scheduling.

This adapter never auto-creates replacements or restarts during observe().
Each slot has independent files, with no mutable monitor globals.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

from corpus_pipeline.serving.platform_client import InspireClient
from inspire.sdk.models_serving import ServingCreateSpec
from corpus_pipeline.serving import account_budget as pool
from corpus_pipeline.serving.inspire_serving import monitor as native
from corpus_pipeline.serving.qwen36_w8a8_20261024.runtime.contract import (
    CODE_FILES, check, code_revision, digest, file_sha, validate_assets, validate_config)

SCOPE = Path(__file__).resolve().parent
MODEL_REGISTRY_NAME = 'corpus-qwen36-35ba3b-w8a8-1a118d71'
IMAGE = 'vllm-ascend:v0.30.0rc2'
TERMINAL = {'INSTANCE_STOPPED', 'INSTANCE_FAILED', 'INSTANCE_SUCCEEDED', 'INSTANCE_DELETED',
            'INSTANCE_TERMINATED', 'INSTANCE_COMPLETED'}


def persist(path, state):
    native.save(path, {k: v for k, v in state.items() if k != 'endpoint'})


def disabled(slot_dir, state):
    return ((Path(slot_dir) / 'DISABLED').exists()
            or bool(state.get('pool_scope') and (Path(state['pool_scope']) / 'DISABLED').exists()))


def prepare_runtime():
    """Freeze runtime configuration only after the verified asset is complete."""
    receipt_path = SCOPE / 'weights/WEIGHTS_READY.json'
    ready = native.load(receipt_path)
    check(ready.get('complete') is True and ready.get('precision') == 'W8A8_DYNAMIC', 'Weights are not ready')
    private = SCOPE.parent / 'private'
    config = {'model_name': 'Qwen3.6-35B-A3B-W8A8', 'model_path': ready['model_path'],
        'revision': ready['revision'], 'precision': 'W8A8_DYNAMIC', 'quantization': 'ascend', 'dtype': 'bfloat16',
        'processor_files': ready['processor_files'], 'processor_revision': ready['processor_revision'],
        'weights_ready_path': str(receipt_path), 'weights_ready_sha256': file_sha(receipt_path),
        'nodes': 1, 'tensor_parallel_size': 2, 'data_parallel_size': 1, 'enable_expert_parallel': True,
        'max_num_seqs': 128, 'max_num_batched_tokens': 4096, 'max_model_len': 16384,
        'gpu_memory_utilization': 0.90, 'num_speculative_tokens': 3, 'enable_json_keepalive': True,
        'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        'max_images': 16, 'max_pixels': 1048576, 'engine_key_path': str(private / 'engine.key')}
    validate_assets(config)
    path = SCOPE / 'runtime/config.json'
    if path.exists():
        check(native.load(path) == config, 'Existing runtime configuration differs')
    else:
        native.save(path, config)
    code, hashes = code_revision(path.parent)
    manifest = {'schema': 'corpus.qwen36.runtime.v1', 'config_sha256': file_sha(path),
        'runtime_files': hashes, 'serving_code_revision': code, 'weights_ready_sha256': file_sha(receipt_path)}
    target = SCOPE / 'RUNTIME_MANIFEST.json'
    if target.exists():
        check(native.load(target) == manifest, 'Frozen runtime manifest differs')
    else:
        native.save(target, manifest)
    return config


class Backend:
    def __init__(self, runtime_dir=None, config_path=None, account_card_ceiling=256):
        pool.configure_management_proxy_bypass()
        self.runtime_dir = Path(runtime_dir or SCOPE / 'runtime').resolve()
        self.config_path = Path(config_path or self.runtime_dir / 'config.json').resolve()
        self.account_card_ceiling = account_card_ceiling

    def binding(self):
        config = native.load(self.config_path); validate_config(config)
        code, files = code_revision(self.runtime_dir)
        check(file_sha(config['weights_ready_path']) == config['weights_ready_sha256'], 'Weight receipt changed')
        return config, {'config_sha256': file_sha(self.config_path), 'runtime_files': files,
            'serving_code_revision': code, 'weights_ready_sha256': config['weights_ready_sha256'],
            'backend_sha256': file_sha(__file__), 'smoke_sha256': file_sha(SCOPE / 'smoke.py')}

    @staticmethod
    def validate(state):
        check(state.get('cards') == 2 and state.get('expected_nodes') == 1 and state.get('expected_replicas') == 1,
              'Expected one native two-card allocation')
        check(type(state.get('priority')) is int and state.get('tier') in ('high', 'low')
              and state['priority'] in ((4, 6) if state['tier'] == 'high' else (1, 3)),
              'Tier or priority changed')
        check(state.get('precision') == 'W8A8_DYNAMIC' and state.get('role') == 'verifier'
              and state.get('name') == state.get('service_name') == state.get('operation_id'), 'State identity changed')

    def prepare(self, slot_dir, slot, name, priority):
        slot_dir = Path(slot_dir).resolve()
        check(type(priority) is int and priority in (1, 3, 4, 6)
              and name.replace('-', '').isalnum(), 'Invalid allocation name or priority')
        config, binding = self.binding(); validate_assets(config)
        registration = native.load(SCOPE / 'MODEL_REGISTERED.json')
        check(registration['model_path'] == config['model_path'] and registration['revision'] == config['revision'],
              'Registered model asset changed')
        fields = dict(name=name, model=MODEL_REGISTRY_NAME, model_version=1,
            command=shlex.join(['env', 'TZ=Asia/Shanghai', 'CORPUS_CONFIG=' + str(self.config_path),
                'CORPUS_DEPLOYMENT_ID=' + name, 'bash', str(self.runtime_dir / 'launch.sh')]),
            port=8000, workspace=native.WORKSPACE, project=native.PROJECT, group='910B资源', quota='2,16,128',
            image=IMAGE, replicas=1, nodes_per_replica=1, shm_gib=128, priority=priority,
            auto_scaling=False, public_path_readonly=True,
            description='Owned Qwen3.6 Ascend W8A8 TP2 DP1 EP2; basic current-boot image smoke; no quality matrix')
        with InspireClient(account='wjx-ascend') as client:
            plan = client.servings.plan(ServingCreateSpec(**fields))
            pool.observable_project_budget(client, 2, ceiling=self.account_card_ceiling)
        body = plan.create_kwargs
        check(body['model_id'] == registration['model_id'] and plan.priority == priority
              and body['task_priority'] == priority and body['replicas'] == 1 and body['node_num_per_replica'] == 1,
              'Resolved model or topology changed')
        check(all(body['resource_spec_price'][key] == value for key, value in
                  {'gpu_count': 2, 'cpu_count': 16, 'memory_size_gib': 128}.items()), 'Resolved quota changed')
        state = {'schema_version': 'corpus_pipeline.native_serving.v1', 'scope': str(slot_dir), 'slot': slot,
            'role': 'verifier', 'tier': 'high' if priority in (4, 6) else 'low', 'priority': priority, 'cards': 2,
            'name': name, 'service_name': name, 'operation_id': name, 'deployment_id': name,
            'create_spec': fields, 'payload': body, 'binding': binding, 'config_snapshot': config,
            'model': config['model_name'], 'revision': config['revision'], 'precision': config['precision'],
            'processor_revision': config['processor_revision'], 'expected_nodes': 1, 'expected_replicas': 1,
            'runtime_root': str(self.runtime_dir / 'runs' / name), 'max_images': config['max_images'],
            'api_key_file': str(SCOPE.parent / 'private/inspire-serving.key'), 'api_key_env': 'INF_API_KEY',
            'auto_recovery_enabled': priority in (1, 3), 'phase': 'PREPARED', 'created_at': time.time(),
            'canary_passed': False, 'release_verified': False, 'qualification_kind': 'basic_smoke',
            'request_overrides': {'chat_template_kwargs': {'enable_thinking': False},
                                  'mm_processor_kwargs': {'max_pixels': config['max_pixels']}}}
        self.validate(state)
        return state

    def create(self, slot_dir, state):
        self.validate(state)
        check(self.binding()[1] == state['binding'], 'Runtime changed before Create')
        slot_dir = Path(slot_dir)
        check((slot_dir / 'CREATE_INTENT.json').exists() and (slot_dir / 'state.json').exists(),
              'Caller must persist Create reservation before transmission')
        marker = slot_dir / 'CREATE_TRANSMISSION.json'
        check(not marker.exists(), 'Create transmission already attempted; reconcile only')
        with InspireClient(account='wjx-ascend') as client:
            pool.observable_project_budget(client, 2, ceiling=self.account_card_ceiling)
            check(not disabled(slot_dir, state), 'Slot or pool disabled before Create')
            native.save(marker, {'operation_id': state['operation_id'], 'time': time.time(), 'binding': state['binding']})
            result = client.servings.create(ServingCreateSpec(**state['create_spec']), operation_id=state['operation_id'])
        return {**state, 'service_id': result.ref.key, 'phase': 'OBSERVING'}

    def reconcile(self, client, state):
        self.validate(state)
        if state.get('service_id'):
            from inspire.sdk.models_serving import ServingRef
            ref = client.servings._make_ref(ServingRef, state['name'], state['service_id'], state['payload']['workspace_id'])
            detail = client.servings.get(ref, workspace=native.WORKSPACE)
            native.assert_owned(state, detail)
            return state, detail
        rows = [row for row in client.servings.iter(workspace=native.WORKSPACE) if row.name == state['name']]
        check(len(rows) == 1, 'Create intent unresolved; never retransmit')
        detail = client.servings.get(rows[0].ref, workspace=native.WORKSPACE)
        candidate = {**state, 'service_id': detail.ref.key, 'phase': 'OBSERVING'}
        native.assert_owned(candidate, detail)
        return candidate, detail

    def attest(self, state, instances):
        check(self.binding()[1] == state['binding'], 'Runtime changed after Create')
        check(len(instances) == 1 and instances[0].status == 'INSTANCE_RUNNING', 'Instance is not running')
        platform = native.instance_signature(instances)
        candidates = []
        for directory in Path(state['runtime_root']).iterdir():
            path = directory / 'node0.json'
            if path.exists():
                beat = native.load(path)
                if 0 <= time.time()-beat['time'] < 20:
                    candidates.append((directory, beat))
        check(len(candidates) == 1, 'Expected one fresh boot')
        directory, beat = candidates[0]; report = native.load(directory / 'versions-node0.json')
        raw = instances[0].raw
        check(beat['hostname'] in (raw['name'].split('/')[-1], raw['node']), 'Heartbeat is not the live instance')
        check(beat['deployment_id'] == report['deployment_id'] == state['deployment_id']
              and beat['boot_id'] == report['boot_id'] == directory.name, 'Boot identity changed')
        check(all(report[key] == state[key] for key in ('model', 'revision', 'precision', 'processor_revision')),
              'Loaded model identity changed')
        expected = {k: v for k, v in state['config_snapshot'].items() if 'key' not in k and k != 'processor_files'}
        check(report['runtime_config'] == expected and report['serving_code_revision'] == state['binding']['serving_code_revision']
              and report['visible_npu_count'] == 2 and report['tensor_parallel'] == 2
              and report['expert_parallel'] is True and report['data_parallel'] == 1, 'Loaded runtime contract differs')
        check(all(isinstance(report.get(key), str) and report[key] for key in native.PACKAGES), 'Package versions missing')
        identity = {k: report[k] for k in (*native.PACKAGES, 'model', 'revision', 'precision', 'processor_revision',
                    'runtime_config', 'serving_code_revision', 'processor_options', 'communication')}
        return {'incarnation': digest({'platform': platform, 'boot': beat['boot_id'], 'runtime_dir': str(directory)}),
            'expected_boots': [beat['boot_id']], 'runtime_revision': 'runtime-versions-sha256:' + digest(identity),
            'runtime_dir': str(directory), 'platform_instances': platform, 'reports': [report]}

    def smoke(self, slot_dir, state, evidence):
        directory = Path(slot_dir) / 'smoke' / evidence['incarnation']
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / 'RESULT.json'; process_path = directory / 'PROCESS.json'
        if output.exists():
            result = native.load(output)
            check(all(result.get(key) == state.get(key) for key in
                      ('model', 'revision', 'precision', 'processor_revision', 'runtime_revision', 'incarnation', 'expected_boots')),
                  'Smoke receipt identity changed')
            check(result.get('qualification_kind') == 'basic_smoke' and result.get('request_count') == 1,
                  'Unexpected smoke contract')
            state['qualification_receipt'] = str(output)
            state['smoke_status'] = 'PASSED' if result.get('passed') is True else 'FAILED'
            return result.get('passed') is True
        if process_path.exists():
            process = native.load(process_path); live = native.identity(process['pid'])
            state['smoke_status'] = 'RUNNING' if live and all(live.get(k) == process.get(k)
                for k in ('pid', 'command', 'starttime_ticks')) else 'PROCESS_EXITED_WITHOUT_RECEIPT'
            return False
        if (directory / 'SMOKE_INTENT.json').exists():
            state['smoke_status'] = 'UNKNOWN_PROCESS_AFTER_INTENT'
            return False
        manifest = SCOPE.parents[1] / 'model_comparison_20261023/manifest_all140.jsonl'
        with manifest.open() as stream:
            sample = json.loads(next(stream))
        request = {key: state[key] for key in ('model', 'revision', 'precision', 'processor_revision',
            'runtime_revision', 'incarnation', 'expected_boots', 'api_key_file', 'request_overrides')}
        request.update(base_url=state['endpoint'].rstrip('/') + '/v1', asset={key: sample[key]
            for key in ('asset_id', 'sha256', 'original_ref')})
        native.save(directory / 'INPUT.json', request)
        native.save(directory / 'SMOKE_INTENT.json', {'incarnation': state['incarnation'], 'time': time.time(),
                    'input_sha256': file_sha(directory / 'INPUT.json')})
        command = [str(SCOPE.parents[2] / '.venv/bin/python'), '-m',
            'corpus_pipeline.serving.qwen36_w8a8_20261024.smoke', '--input', str(directory / 'INPUT.json')]
        with (directory / 'stdout.log').open('ab') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=SCOPE.parents[2])
        native.save(process_path, {**native.identity(child.pid), 'created_at': time.time()})
        state['smoke_status'] = 'RUNNING'
        return False

    def publish(self, slot_dir, state, ready):
        path = Path(slot_dir) / 'registry.json'
        old = native.load(path) if path.exists() else {'registry_generation': 0, 'endpoints': []}
        state = dict(state)
        if not state.get('endpoint') and old['endpoints']:
            state['endpoint'] = old['endpoints'][0]['base_url'].removesuffix('/v1')
        rows = []
        if state.get('service_id') and state.get('endpoint'):
            row = native.candidate(state, bool(ready))
            row.update(max_output_tokens=3000, qualification_kind='basic_smoke')
            rows.append(row)
        native.save(path, {'schema_version': 'corpus_pipeline.endpoints.v1',
            'registry_generation': old['registry_generation'] + 1, 'updated_at': time.time(),
            'endpoints': rows, 'publisher': 'qwen36_w8a8_native_backend'})

    def observe(self, slot_dir, state):
        state = dict(state); ready = False
        with InspireClient(account='wjx-ascend') as client:
            state, detail = self.reconcile(client, state)
            instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
            previous = state.get('platform_status')
            state.update(platform_status=detail.status, endpoint=detail.to_dict()['endpoint'],
                instance_statuses=[x.status for x in instances], last_observed_at=time.time(), observation_error_type=None)
            if detail.status == 'RUNNING':
                state.update(was_running=True, last_running_at=time.time(), stopped_transition_at=None)
            elif detail.status == 'STOPPED' and state.get('was_running') and previous != 'STOPPED':
                state['stopped_transition_at'] = time.time()
            state['events'] = list(client.servings.events(detail.ref, workspace=native.WORKSPACE, limit=30).items)
            if detail.status == 'RUNNING' and not disabled(slot_dir, state):
                try:
                    evidence = self.attest(state, instances)
                    if state.get('incarnation') != evidence['incarnation']:
                        state.update(incarnation=evidence['incarnation'], qualification_receipt=None, canary_passed=False)
                    state.update(expected_boots=evidence['expected_boots'], runtime_revision=evidence['runtime_revision'])
                    native.save(Path(slot_dir) / 'attestation.json', evidence)
                    if native.health(state, evidence):
                        ready = self.smoke(slot_dir, state, evidence)
                    state['readiness_reason'] = 'BASIC_SMOKE_PASSED' if ready else state.get('smoke_status', 'LOADING')
                except (OSError, KeyError, ValueError, AssertionError) as error:
                    state.update(readiness_reason='LOADING_OR_ATTESTATION_FAILED', observation_error_type=type(error).__name__)
            else:
                state['readiness_reason'] = 'DISABLED_OR_NOT_RUNNING'
        ready = ready and not disabled(slot_dir, state)
        state.update(canary_passed=ready, heartbeat_at=time.time())
        if ready:
            state.update(phase='QUALIFIED', first_qualified_at=state.get('first_qualified_at', time.time()))
        self.publish(slot_dir, state, ready)
        return state

    @staticmethod
    def can_restart(state):
        # A fresh observation is required, but an explicit victim event does not
        # expire while the controller is offline. Older boot events are fenced
        # by the last verified RUNNING sample and the durable restart event.
        scope = Path(state['scope'])
        if (state.get('tier') != 'low' or not state.get('auto_recovery_enabled')
                or state.get('platform_status') != 'STOPPED' or not state.get('was_running')
                or not state.get('stopped_transition_at') or disabled(scope, state)
                or (scope / 'STOP_INTENT.json').exists()
                or not 0 <= time.time() - state.get('last_observed_at', 0) <= 60):
            return None
        relevant = []
        for event in state.get('events', []):
            reason = str(event.get('reason', '')).lower(); message = str(event.get('message', '')).lower()
            if 'preempt' not in reason or not any(x in message for x in ('preempted by', 'was preempted', '被抢占')):
                continue
            try:
                timestamp = float(event.get('last_timestamp') or event.get('created_at') or 0) / 1000
            except (TypeError, ValueError):
                continue
            if (math.isfinite(timestamp) and state.get('last_running_at', float('inf')) <= timestamp
                    <= state['last_observed_at'] and state['service_id'] in str(event.get('object_id', ''))):
                relevant.append((timestamp, event))
        if not relevant:
            return None
        event = native.digest(max(relevant, key=lambda x: x[0])[1])
        return event if event != state.get('restart_event') else None

    def restart(self, slot_dir, state, event):
        slot_dir = Path(slot_dir)
        check(not disabled(slot_dir, state) and state.get('tier') == 'low', 'Restart disabled')
        check(isinstance(event, str) and len(event) == 64 and all(c in '0123456789abcdef' for c in event), 'Invalid restart event')
        path = slot_dir / 'restart_events' / (event + '.json')
        if path.exists():
            return state
        check(self.can_restart(state) == event, 'Restart event is not fresh and owned')
        with InspireClient(account='wjx-ascend') as client:
            state, detail = self.reconcile(client, state)
            instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
            check(detail.status == 'STOPPED' and all(x.status in TERMINAL for x in instances), 'Restart source not terminal')
            native.save(path, {'service_id': state['service_id'], 'event': event, 'time': time.time()})
            state.update(restart_event=event, canary_passed=False, qualification_receipt=None, phase='RESTART_INTENT')
            persist(slot_dir / 'state.json', state)
            self.publish(slot_dir, state, False)
            check(not disabled(slot_dir, state), 'Stop requested before restart')
            client.servings.start(detail.ref, workspace=native.WORKSPACE)
        return {**state, 'phase': 'OBSERVING'}

    def released(self, state):
        with InspireClient(account='wjx-ascend') as client:
            state, detail = self.reconcile(client, state)
            instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
            return detail.status == 'STOPPED' and all(x.status in TERMINAL for x in instances)

    def stop(self, slot_dir, state):
        slot_dir = Path(slot_dir); (slot_dir / 'DISABLED').write_text('Explicit owned stop requested\n')
        self.publish(slot_dir, state, False)
        with InspireClient(account='wjx-ascend') as client:
            state, detail = self.reconcile(client, state)
            path = slot_dir / 'STOP_INTENT.json'
            if path.exists():
                check(native.load(path)['service_id'] == state['service_id'], 'Stop intent identity changed')
            if detail.status != 'STOPPED' and not path.exists():
                native.save(path, {'service_id': state['service_id'], 'time': time.time()})
                client.servings.stop(detail.ref, workspace=native.WORKSPACE)
            detail = client.servings.get(state['name'], workspace=native.WORKSPACE); native.assert_owned(state, detail)
            instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
            released = detail.status == 'STOPPED' and all(x.status in TERMINAL for x in instances)
        state.update(canary_passed=False, release_verified=released, platform_status=detail.status,
            instance_statuses=[x.status for x in instances], phase='RELEASED' if released else 'STOPPING')
        native.save(slot_dir / 'STOPPED.json', {'complete': released, 'service_id': state['service_id'],
            'status': detail.status, 'instance_statuses': state['instance_statuses'], 'time': time.time()})
        return state
