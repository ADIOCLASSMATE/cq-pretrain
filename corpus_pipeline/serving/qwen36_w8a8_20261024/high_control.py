"""One owned two-card HIGH native Serving; no automatic restart or expansion."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
from inspire.sdk import InspireClient
from inspire.platform.web import browser_api
from corpus_pipeline.serving.qwen36_w8a8_20261024.backend import (
    Backend, SCOPE, MODEL_REGISTRY_NAME, TERMINAL, native, pool, prepare_runtime, persist)
from corpus_pipeline.serving.qwen36_w8a8_20261024.runtime.contract import check, file_sha, code_revision, validate_assets

HIGH = SCOPE / 'high'
NAME = 'corpus-qwen36-w8a8-high-20261024'


@contextmanager
def locked(path, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def paths_in(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and key in ('source_path', 'model_source_path', 'path', 'model_path'):
                yield item
            yield from paths_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from paths_in(item)


def register_model(config):
    """Register once; an unknown reply is reconciled by exact model identity."""
    with locked(SCOPE / 'model_registry.lock'):
        receipt_path = SCOPE / 'MODEL_REGISTERED.json'
        intent_path = SCOPE / 'MODEL_REGISTER_INTENT.json'
        with InspireClient(account='wjx-ascend') as client:
            candidates = [x for x in client.models.list(workspace=native.WORKSPACE,
                project=native.PROJECT, keyword=MODEL_REGISTRY_NAME, limit=100).items if x.name == MODEL_REGISTRY_NAME]
            check(len(candidates) <= 1, 'Ambiguous model registration')
            if not candidates:
                check(not intent_path.exists() and not receipt_path.exists(), 'Unknown registration; reconcile only')
                intent = {'name': MODEL_REGISTRY_NAME, 'operation_id': MODEL_REGISTRY_NAME,
                    'model_path': config['model_path'], 'revision': config['revision'],
                    'weights_ready_sha256': config['weights_ready_sha256'], 'time': time.time()}
                native.save(intent_path, intent)
                handle = client.models.register(MODEL_REGISTRY_NAME, source_path=config['model_path'],
                    workspace=native.WORKSPACE, project=native.PROJECT,
                    description='Verified Ascend W8A8_DYNAMIC Qwen3.6 TP2 multimodal weights; pinned revision ' + config['revision'],
                    operation_id=MODEL_REGISTRY_NAME)
                model = client.models.get(handle.ref, workspace=native.WORKSPACE, project=native.PROJECT)
            else:
                check(intent_path.exists(), 'Existing registry name was not created by this scope')
                intent = native.load(intent_path)
                check(intent['model_path'] == config['model_path'] and intent['revision'] == config['revision']
                      and intent['weights_ready_sha256'] == config['weights_ready_sha256'], 'Registration intent changed')
                model = candidates[0]
            with client._transport.scope(timeout=30):
                records = browser_api.list_model_version_records(model.ref.key,
                    session=client._transport.session, workspace_id=model.ref.workspace_id)
            check(config['model_path'] in set(paths_in(records)), 'Registered model source path differs')
            if receipt_path.exists():
                old = native.load(receipt_path)
                check(old['model_id'] == model.ref.key and old['model_path'] == config['model_path'], 'Model receipt changed')
            receipt = {'complete': True, 'name': MODEL_REGISTRY_NAME, 'model_id': model.ref.key,
                'model_path': config['model_path'], 'revision': config['revision'], 'model_version': 1,
                'weights_ready_sha256': config['weights_ready_sha256'], 'status': model.status, 'time': time.time()}
            native.save(receipt_path, receipt)
            return receipt


def safe(state):
    return {k: state.get(k) for k in ('service_id', 'phase', 'platform_status', 'canary_passed',
        'readiness_reason', 'observation_error_type', 'heartbeat_at', 'expected_boots', 'release_verified')}


def repair_start(kind='bf16'):
    """One explicitly authorized same-ID retry per preserved startup failure."""
    check(kind in ('bf16', 'capture'), 'Unknown bounded startup repair')
    with locked(HIGH / 'lifecycle.lock'):
        backend = Backend(); state = native.load(HIGH / 'state.json')
        backend.validate(state)
        check(state.get('name') == NAME and state['tier'] == 'high', 'Unexpected HIGH owner')
        marker = HIGH / ('BF16_REPAIR_START_INTENT.json' if kind == 'bf16' else 'CAPTURE_REPAIR_START_INTENT.json')
        if marker.exists():
            check(native.load(marker)['service_id'] == state['service_id'], 'Repair identity changed')
            return safe(state)  # Unknown Start is reconciled, never retransmitted.
        snapshot = SCOPE / ('attempts/01-fp16/SNAPSHOT.json' if kind == 'bf16'
                            else 'attempts/02-bf16-stream-limit/SNAPSHOT.json')
        old = native.load(snapshot)
        check(old['complete'] is True and old['release_verified'] is True
              and old['service_id'] == state['service_id'], 'Prior failure snapshot missing')
        for relative, expected in old['files'].items():
            check(file_sha(snapshot.parent / relative) == expected, 'Prior failure snapshot changed')
        check((HIGH / 'DISABLED').exists() and state.get('release_verified') is True
              and backend.released(state), 'Existing HIGH must be freshly released')
        stop_stamp = (HIGH / 'DISABLED').stat().st_mtime_ns
        config = native.load(backend.config_path)
        if kind == 'bf16':
            check(config['dtype'] == 'float16', 'Expected preserved FP16 failure before repair')
            config['dtype'] = 'bfloat16'
            changes = {'dtype':['float16','bfloat16']}
            basis = 'CANN grouped_matmul A2/A3 INT8 per-token table: BF16 scale/BF16 output supported'
            source = 'https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md'
        else:
            check(config['dtype'] == 'bfloat16' and 'cudagraph_capture_sizes' not in config,
                  'Expected preserved unbounded capture failure before repair')
            config['cudagraph_capture_sizes'] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            changes = {'cudagraph_capture_sizes':[None,config['cudagraph_capture_sizes']]}
            basis = 'Resolved engine default has 51 shapes through512; bound to10 shapes, retaining MTP3 times128 sequences'
            source = 'https://docs.vllm.ai/en/stable/api/vllm/config/vllm/'
        validate_assets(config)
        previous_binding = state['binding']
        native.save(backend.config_path, config)
        code, files = code_revision(backend.runtime_dir)
        native.save(SCOPE / 'RUNTIME_MANIFEST.json', {'schema':'corpus.qwen36.runtime.v1',
            'config_sha256':file_sha(backend.config_path), 'runtime_files':files,
            'serving_code_revision':code, 'weights_ready_sha256':config['weights_ready_sha256']})
        config, binding = backend.binding()
        with InspireClient(account='wjx-ascend') as client:
            state, detail = backend.reconcile(client, state)
            instances = client.servings.instances(detail.ref, workspace=native.WORKSPACE)
            check(detail.status == 'STOPPED' and all(x.status in TERMINAL for x in instances), 'HIGH is not terminal')
            budget = pool.observable_project_budget(client, 2)
            check(sum(row['cards'] for row in budget['records'].values()
                      if row['name'].startswith('corpus-')) + 2 <= 128, 'Task card ceiling exceeded')
            check((HIGH / 'DISABLED').stat().st_mtime_ns == stop_stamp, 'New stop arrived during repair')
            for name in ('STOP_INTENT.json','STOPPED.json','DISABLED'):
                (HIGH / name).unlink(missing_ok=True)
            native.save(marker, {'service_id':state['service_id'], 'time':time.time(),
                'snapshot_sha256':file_sha(snapshot), 'previous_binding':previous_binding, 'binding':binding,
                'change':changes, 'quantization_unchanged':'W8A8_DYNAMIC', 'basis':basis, 'source':source})
            state.update(binding=binding, config_snapshot=config, previous_boots=state.get('expected_boots', []),
                expected_boots=[], incarnation=None, runtime_revision=None, qualification_receipt=None,
                canary_passed=False, release_verified=False, phase='REPAIR_START_INTENT',
                smoke_status=None, observation_error_type=None, restarted_at=time.time())
            persist(HIGH / 'state.json', state); backend.publish(HIGH, state, False)
            # The marker above fences a lost reply. This is the original ID,
            # never a new allocation, and current-boot smoke must run again.
            check(not (HIGH / 'DISABLED').exists(), 'New stop arrived before repaired Start')
            client.servings.start(detail.ref, workspace=native.WORKSPACE)
        state['phase'] = 'OBSERVING'; persist(HIGH / 'state.json', state)
        return safe(state)


def create():
    with locked(HIGH / 'lifecycle.lock'):
        check(not (HIGH / 'DISABLED').exists(), 'HIGH explicitly disabled')
        backend = Backend()
        if (HIGH / 'CREATE_INTENT.json').exists():
            state = native.load(HIGH / 'state.json'); backend.validate(state)
            return safe(state)  # Never resend an ambiguous Create.
        config = prepare_runtime()
        register_model(config)
        with InspireClient(account='wjx-ascend') as client:
            budget = pool.observable_project_budget(client, 2)
            # This first HIGH is intentionally the only new resource. The pool
            # subsequently reserves its two cards until fresh release evidence.
            task_cards = sum(row['cards'] for row in budget['records'].values()
                             if row['name'].startswith('corpus-'))
            check(task_cards + 2 <= 128, 'Task card ceiling exceeded')
            native.save(HIGH / 'BUDGET_BEFORE_CREATE.json', budget)
        state = backend.prepare(HIGH, 0, NAME, 6)
        persist(HIGH / 'state.json', state)
        native.save(HIGH / 'CREATE_INTENT.json', {'operation_id': NAME, 'cards': 2,
            'binding': state['binding'], 'time': time.time()})
        try:
            state = backend.create(HIGH, state)
        except Exception as error:
            state.update(phase='CREATE_UNKNOWN', observation_error_type=type(error).__name__)
            persist(HIGH / 'state.json', state)
            raise
        persist(HIGH / 'state.json', state)
        return safe(state)


def observe(once=False, event=None):
    event = event or threading.Event()
    backend = Backend()
    with locked(HIGH / 'observer.lock', nonblocking=True):
        native.save(HIGH / 'OBSERVER.json', {**native.identity(os.getpid()), 'time': time.time()})
        while not event.is_set() and not (HIGH / 'DISABLED').exists():
            with locked(HIGH / 'lifecycle.lock'):
                state = native.load(HIGH / 'state.json')
                try:
                    state = backend.observe(HIGH, state)
                except Exception as error:
                    state.update(canary_passed=False, readiness_reason='OBSERVATION_FAILED',
                        observation_error_type=type(error).__name__, heartbeat_at=time.time())
                    backend.publish(HIGH, state, False)
                persist(HIGH / 'state.json', state)
                native.save(HIGH / 'STATUS.json', safe(state))
            if once:
                return safe(state)
            event.wait(8)
    return {'observer_stopped': True}


def stop():
    HIGH.mkdir(parents=True, exist_ok=True)
    (HIGH / 'DISABLED').write_text('Owned HIGH stopped; no automatic recovery\n')
    with locked(HIGH / 'lifecycle.lock'):
        backend = Backend(); state = native.load(HIGH / 'state.json')
        state = backend.stop(HIGH, state)
        persist(HIGH / 'state.json', state)
        return safe(state)


def main():
    pool.configure_management_proxy_bypass()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'create', 'repair-start', 'capture-repair-start', 'observe', 'stop'])
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: event.set())
    try:
        if args.action == 'prepare':
            config = prepare_runtime(); receipt = register_model(config)
            result = {'model_registered': receipt['complete'], 'model_id': receipt['model_id'], 'cards': 2}
        elif args.action == 'create':
            result = create()
        elif args.action == 'repair-start':
            result = repair_start()
        elif args.action == 'capture-repair-start':
            result = repair_start('capture')
        elif args.action == 'observe':
            result = observe(args.once, event)
        else:
            result = stop()
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'success': False, 'error_type': type(error).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
