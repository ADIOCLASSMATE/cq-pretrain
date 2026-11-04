"""CPU front pass for documented whole sources with zero native images.

No detector or tensor framework is imported. The source list only selects work;
every actual row still passes the native adapter and explicit zero-image checks.
Outputs and completion receipts use the normal frontpass/queue contracts.

Production use on the existing control CPU (no RPC server restart needed):
  python -m corpus_pipeline.cluster.text_worker --root RUN --queue-backend local
Multiple independent processes safely compete for the same eligible task IDs.
An unexpected image releases the complete task for a normal worker, persistently
excludes it from this CPU path, then exits nonzero. Do not automatically restart
an errored worker without inspecting its worker receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from itertools import islice
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from corpus_pipeline.production.ingest import iter_records
from corpus_pipeline.production.potential import compile_source_contracts
from corpus_pipeline.production.records import snapshot
from . import queue as local
from .frontpass import (BASE, POLICY, ParquetRows, load_rules, record_decision,
                        schemas, validate_image_slots)


class UnexpectedImage(ValueError):
    """Metadata selection did not prove this actual record is text-only."""


def verify_control_host(root):
    """Local flock is allowed only alongside the live RPC owner on this host."""
    root = Path(root).resolve()
    owner = json.loads((root / '.queue_rpc_owner' / 'owner.json').read_text())
    config = json.loads((root / 'queue_rpc.json').read_text())
    if (owner.get('hostname') != socket.gethostname()
            or config.get('hostname') != owner['hostname']
            or config.get('server_id') != owner.get('server_id')
            or config.get('pid') != owner.get('pid')
            or config.get('root') != str(root)
            or config.get('queue_manifest_sha256') != local._config(root)['manifest_sha256']):
        raise RuntimeError('local text worker requires the current queue control host')
    pid = owner.get('pid')
    if type(pid) is not int or pid <= 0:
        raise RuntimeError('queue owner has no valid process ID')
    try:
        os.kill(pid, 0)
    except OSError as error:
        raise RuntimeError('queue control process is not alive') from error
    return owner['server_id']


class ControlHostQueue:
    """Recheck host ownership before every local queue operation."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.server_id = verify_control_host(self.root)

    def __getattr__(self, name):
        if name not in {'claim_task', 'heartbeat', 'complete', 'fail'}:
            raise AttributeError(name)
        def operation(root, *args, **kwargs):
            if (Path(root).resolve() != self.root
                    or verify_control_host(self.root) != self.server_id):
                raise RuntimeError('queue control ownership changed')
            return getattr(local, name)(self.root, *args, **kwargs)
        return operation


def select_tasks(root, source_plan, snap, plan):
    """Bind metadata selection to the immutable run, snapshot and task catalog."""
    root = Path(root).resolve()
    if (source_plan.get('schema_version') != 'dcvlm_text_only_source_plan.v1'
            or Path(source_plan['dataset_root']).resolve() != Path(snap['root']).resolve()):
        raise ValueError('unsupported or mismatched text source plan')
    evidence = source_plan['evidence_sha256']
    if (any(evidence[name] != snap['fingerprint'][name]
            for name in ('manifest.json', 'catalog.json'))
            or evidence['plan.json'] != file_sha(root / 'plan.json')):
        raise ValueError('text source plan evidence changed')
    sources = source_plan['sources']
    names = {item['source'] for item in sources}
    if (len(names) != len(sources) or names != set(source_plan['source_names'])
            or len(names) != source_plan['source_count']
            or source_plan.get('image_occurrences') != 0
            or any(item['manifest_image_occurrences'] != 0 for item in sources)):
        raise ValueError('text source selection is not whole-source zero-image metadata')
    declared = {row[0]: row for row in snap['manifest']['statistics']['sources']}
    for item in sources:
        actual = declared[item['source']]
        if actual[3] != 0 or actual[1] != item['manifest_record_count']:
            raise ValueError('source selection disagrees with native manifest')
    config = local._config(root)
    if config['task_count'] != plan['task_count']:
        raise ValueError('queue and run task counts differ')
    selected, totals = [], Counter()
    for task_id, (ordinal, expected_digest) in local.catalog_index(root).items():
        item = local._manifest(root, config, ordinal)
        if item['task_id'] != task_id or item['task_sha256'] != expected_digest:
            raise ValueError('frozen task identity changed during text selection')
        task = item['task']
        if task['source'] in names:
            selected.append(task_id)
            totals[task['source']] += task['row_end'] - task['row_start']
    if (len(selected) != source_plan['task_count']
            or sum(totals.values()) != source_plan['record_count']
            or any(totals[item['source']] != item['manifest_record_count'] for item in sources)):
        raise ValueError('frozen tasks do not cover the complete selected text sources')
    return selected


def assert_text_only(record):
    raw = record['raw']
    if (raw.get('image_count') != 0 or raw.get('images')
            or any(segment.get('kind') == 'image' for segment in raw['segments'])
            or record['source_image_refs']
            or record['normalized']['provenance']['images']):
        raise UnexpectedImage('native row contains image evidence; normal image worker required')
    validate_image_slots(record)


def process_task(root, claim, snap, rules, contracts, *, record_batch=128):
    """Commit nothing until all actual rows are checked and coverage matches."""
    if type(record_batch) is not int or record_batch < 1:
        raise ValueError('record_batch must be positive')
    root = Path(root).resolve()
    task = claim['task']
    out = root / 'artifacts' / claim['task_id'] / claim['token']
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 'production.input.v2', 'collection': 'dcvlm',
                'snapshot_root': snap['root'], 'snapshot_fingerprint': snap['fingerprint'],
                'shards': [{**task, 'row_stop': task['row_end']}], 'scope': POLICY}
    iterator = iter(iter_records(manifest, snap, rules, batch_size=record_batch))
    record_schema, image_schema = schemas()
    counts, categories = Counter(), Counter()
    started = time.time()
    with ExitStack() as stack:
        records_out = ParquetRows(out / 'records.parquet', record_schema)
        stack.callback(records_out.close)
        for filename in ('images.parquet', 'suspects.parquet'):
            empty = ParquetRows(out / filename, image_schema)
            stack.callback(empty.close)
        with (out / 'sscd.f32').open('wb') as vectors:
            while records := list(islice(iterator, record_batch)):
                rows = []
                for record in records:
                    assert_text_only(record)
                    decision, reasons = record_decision(record, [])
                    contract = contracts.entries.get(('dcvlm', record['source']), {})
                    normalized = record['normalized']
                    category = normalized['category']
                    rows.append({'sample_id': record['original_sample_id'],
                        'row_index': record['locator']['row_index'],
                        'source_record_sha256': record['source_record_sha256'],
                        'category': category, 'text_origin': normalized['provenance']['text']['origin'],
                        'text_retained': True, 'image_ids': [], 'image_statuses': [],
                        'status': decision, 'reason_codes': reasons,
                        'layout_source_tag': contract.get('tag', 'sample_to_decide'),
                        'recaption_resolution_potential': False})
                    counts['records'] += 1
                    counts['record:' + decision] += 1
                    categories[category] += 1
                records_out.write(rows)
            vectors.flush()
            os.fsync(vectors.fileno())
    if counts['records'] != task['row_end'] - task['row_start']:
        raise ValueError('incomplete text task row coverage')
    identity = {'kind': 'cpu_text_only_no_models', 'code_sha256': file_sha(Path(__file__)),
                'models_loaded': [], 'native_adapter': 'dcvlm.native.v1'}
    summary = {'policy': POLICY, 'task': task, 'token': claim['token'], 'counts': dict(counts),
               'categories': dict(categories), 'elapsed_seconds': time.time() - started,
               'model_identity_sha256': digest(identity), 'execution_identity': identity,
               'models_loaded': [], 'api_calls': 0, 'old_results_used': False, 'all_actual_rows_zero_images': True,
               'global_dedup_complete': False, 'training_release': False}
    atomic_json(out / 'summary.json', summary)
    artifacts = [{'path': str(path), 'sha256': file_sha(path)}
                 for path in sorted(out.iterdir()) if path.is_file()]
    return summary, artifacts


def run_worker(root, *, source_plan=None, queue_backend='local', max_tasks=None,
               record_batch=128):
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    root = Path(root).resolve()
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError('max_tasks must be positive')
    if queue_backend == 'local':
        queue = ControlHostQueue(root)
    elif queue_backend == 'rpc':
        from . import queue_rpc as queue
        queue.health(root)
    else:
        raise ValueError('invalid queue backend')
    plan = json.loads((root / 'plan.json').read_text())
    rules, snap = load_rules(), snapshot(plan['snapshot_root'])
    if (plan['policy'] != POLICY or plan['snapshot_fingerprint'] != snap['fingerprint']
            or plan['rules_sha256'] != digest(rules)
            or plan['thresholds_sha256'] != file_sha(BASE / 'detector_calibration/thresholds.json')
            or plan['layout_contracts_sha256'] != file_sha(BASE / 'prepared/config/layout_source_contracts.json')):
        raise ValueError('run/input/rule identity changed')
    contracts = compile_source_contracts(json.loads((BASE / 'prepared/config/layout_source_contracts.json').read_text()))
    source_plan = Path(source_plan) if source_plan else root / 'text_only_sources.json'
    selected = select_tasks(root, json.loads(source_plan.read_text()), snap, plan)
    worker_id = f'{socket.gethostname()}-text-cpu-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    worker_root = root / 'workers' / worker_id
    worker_root.mkdir(parents=True)
    atomic_json(worker_root / 'execution.json', {'kind': 'cpu_text_only_no_models',
        'queue_backend': queue_backend, 'source_plan_sha256': file_sha(source_plan),
        'selected_task_count': len(selected), 'api_calls': 0, 'auto_restart': False})
    done = 0
    for task_id in selected:
        if max_tasks is not None and done >= max_tasks:
            break
        # A task found to contain images must only re-enter the ordinary worker.
        exclusion = root / 'text_cpu_exclusions' / (task_id + '.json')
        if exclusion.exists():
            continue
        claim = queue.claim_task(root, worker_id, task_id)
        if claim is None:
            continue
        stop, heartbeat_errors = threading.Event(), []
        def beat():
            while not stop.wait(30):
                try:
                    queue.heartbeat(root, claim['task_id'], claim['token'])
                except Exception as error:
                    heartbeat_errors.append(type(error).__name__)
                    return
        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            # Check after claiming too: another CPU process may have just
            # released this task and published an exclusion.
            if exclusion.exists():
                raise UnexpectedImage('task previously excluded from CPU text path')
            summary, artifacts = process_task(root, claim, snap, rules, contracts,
                                              record_batch=record_batch)
            if heartbeat_errors:
                raise RuntimeError('text worker heartbeat failed: ' + heartbeat_errors[0])
            queue.complete(root, task_id, claim['token'], artifacts=artifacts,
                           processed_records=summary['counts']['records'], processed_images=0)
            done += 1
            print(json.dumps({'worker': worker_id, 'task': task_id, **summary['counts'],
                              'elapsed_seconds': summary['elapsed_seconds']}), flush=True)
        except Exception as error:
            if not isinstance(error, local.LeaseLost):
                atomic_json(exclusion, {'task_id': task_id, 'token': claim['token'],
                    'worker_id': worker_id, 'error_type': type(error).__name__,
                    'reason': str(error), 'normal_worker_required': True})
            atomic_json(worker_root / 'FAILED.json', {'task_id': task_id,
                'error_type': type(error).__name__, 'auto_restart': False,
                'normal_worker_required': not isinstance(error, local.LeaseLost), 'at': time.time()})
            try:
                queue.fail(root, task_id, claim['token'], type(error).__name__ + ': ' + str(error),
                           retryable=True)
            except local.LeaseLost:
                pass  # The old attempt cannot publish; the current owner proceeds.
            raise
        finally:
            stop.set()
            thread.join(timeout=2)
    result = {'worker_id': worker_id, 'completed_tasks': done, 'api_calls': 0,
              'full_run_complete': False, 'training_release': False}
    atomic_json(worker_root / 'EXIT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-plan', type=Path)
    parser.add_argument('--queue-backend', choices=('local', 'rpc'), default='local')
    parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--record-batch', type=int, default=128)
    print(json.dumps(run_worker(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
