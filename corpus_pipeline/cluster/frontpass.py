"""Offline, restartable DCVLM front pass; never calls a model API.

The output is a screening index, not a final training release. Suspects and
inference failures remain explicitly pending. Original records stay immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from itertools import islice, tee
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import socket
import threading
import time
import traceback
import uuid

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from corpus_pipeline.production.records import DEFAULT_ROOT, snapshot
from corpus_pipeline.production.ingest import iter_records
from corpus_pipeline.production.potential import compile_source_contracts

BASE = Path(__file__).resolve().parents[1]
POLICY = 'dcvlm.offline_frontpass.v1'
SCREENABLE = {'image_provenance_mixed', 'image_provenance_unknown'}
TASK_POLICY = 'corpus.task_policy.v1'


def load_rules():
    rules = json.loads((BASE / 'config/rules.json').read_text())
    rules['sources'] = json.loads((BASE / 'config/sources.json').read_text())['sources']
    return rules


def rules_for_plan(plan):
    """Load the immutable per-run rules; legacy plans retain their old loader."""
    rules = deepcopy(plan['rules']) if 'rules' in plan else load_rules()
    if not isinstance(rules, dict) or digest(rules) != plan['rules_sha256']:
        raise ValueError('run/input/rule identity changed')
    if rules.get('task_routing') not in (None, TASK_POLICY):
        raise ValueError('unsupported task routing policy')
    return rules


def task_metadata(record, rules=None):
    normalized = record['normalized']
    enabled = (rules.get('task_routing') == TASK_POLICY if rules is not None else
               normalized.get('processing_policy') == TASK_POLICY)
    if not enabled:
        return {}
    route_name = normalized.get('task_route')
    allowed = normalized.get('image_loss_allowed')
    if (normalized.get('processing_policy') != TASK_POLICY
            or route_name not in {'i2t', 't2i', 'mixed', 'text', 'unknown'}
            or type(allowed) is not bool or (route_name == 'i2t' and allowed is not False)):
        raise ValueError('missing or inconsistent normalized task policy')
    return {'processing_policy': TASK_POLICY, 'task_route': route_name,
            'image_loss_allowed': allowed}


def prepare(root, *, dataset=DEFAULT_ROOT, rows_per_task=8192, max_shards=None, rules=None):
    """Only enumerate catalog metadata, never materialize all native rows."""
    from .queue import init_queue
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if rows_per_task < 1:
        raise ValueError('rows_per_task must be positive')
    frozen_rules = deepcopy(rules) if rules is not None else load_rules()
    if not isinstance(frozen_rules, dict) or frozen_rules.get('task_routing') not in (None, TASK_POLICY):
        raise ValueError('unsupported task routing policy')
    snap = snapshot(dataset)
    tasks = []
    for i, (path, entry) in enumerate(snap['catalog_index'].items()):
        if max_shards is not None and i >= max_shards:
            break
        stat = Path(path).stat()
        if stat.st_size != entry['bytes']:
            raise ValueError('catalog Parquet size mismatch')
        for start in range(0, entry['rows'], rows_per_task):
            tasks.append({'task_id': f's{i:05d}-r{start:08d}',
                          'parquet_path': path, 'row_start': start,
                          'row_end': min(start + rows_per_task, entry['rows']),
                          'source': entry['source'], 'source_shard': entry['source_shard'],
                          'collection': entry.get('collection', 'dcvlm'),
                          'parquet_fingerprint': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}})
    now = datetime.now(timezone.utc)
    plan = {'policy': POLICY, 'snapshot_root': snap['root'], 'snapshot_fingerprint': snap['fingerprint'],
            'task_count': len(tasks), 'record_count': sum(t['row_end'] - t['row_start'] for t in tasks),
            'created_at': now.isoformat(), 'deadline_at': (now + timedelta(hours=24)).isoformat(),
            'scope': 'full_native_dcvlm' if max_shards is None else 'bounded_real_validation',
            'api_enabled': False, 'recaption_enabled': False, 'layout_generation_enabled': False,
            'minimum_short_edge': 128, 'rules_sha256': digest(frozen_rules),
            'thresholds_sha256': file_sha(BASE / 'detector_calibration/thresholds.json'),
            'layout_contracts_sha256': file_sha(BASE / 'prepared/config/layout_source_contracts.json'),
            'old_result_reuse': False, 'dedup_stage': 'extract_only_global_search_not_yet_run',
            'training_release': False}
    if rules is not None:
        plan['rules'] = frozen_rules
    if frozen_rules.get('task_routing') == TASK_POLICY:
        plan.update(processing_policy=TASK_POLICY, i2t_minimum_short_edge=1)
    if (root / 'plan.json').exists():
        raise ValueError('run already planned; resume workers rather than replacing its identity')
    atomic_json(root / 'plan.json', plan)
    init_queue(root, tasks, lease_seconds=900, max_attempts=5, deadline_at=plan['deadline_at'])
    return plan


def protected(provenance):
    return provenance['origin'] in {'programmatic', 'rendered'} or provenance.get('policy_exception') == 'ai_edit_from_natural_source'


def route(provenance, prediction, thresholds, *, task_route=None):
    """No detector score can reject an image or certify natural provenance."""
    if task_route == 'i2t':
        if (prediction.get('error_stage') != 'decode'
                and prediction.get('sscd', {}).get('status') == 'ok'):
            return 'i2t_dedup_only'
        return 'pending_inference_or_integrity_error'
    if provenance['origin'] == 'ai_generated':
        return 'rejected_known_ai_source'
    if provenance['origin'] == 'ai_edited' and provenance.get('edit_source_origin') not in {'natural', 'unknown', 'mixed', None}:
        return 'rejected_ai_edit_nonreal_parent'
    if prediction.get('status') == 'too_small':
        return 'rejected_too_small'
    if task_route is not None and any(type(prediction.get(key)) is int and prediction[key] < 128
                                      for key in ('original_width', 'original_height')):
        # A shared I2T occurrence may have valid SSCD evidence below the T2I
        # size floor. That does not waive the generation quality requirement.
        return 'rejected_too_small'
    if protected(provenance):
        # A shared image can have an unprotected occurrence whose detector
        # failed. That irrelevant detector failure must not remove a verified
        # source protection when this same original image and SSCD passed.
        if (prediction.get('status') in {'ok', 'incomplete'}
                and prediction.get('error_stage') != 'decode'
                and prediction.get('sscd', {}).get('status') == 'ok'):
            return 'protected_source'
        return 'pending_inference_or_integrity_error'
    if prediction.get('status') != 'ok':
        return 'pending_inference_or_integrity_error'
    if provenance['origin'] == 'ai_edited':
        return 'pending_edit_lineage'
    scores = prediction.get('scores', {})
    if any(scores.get(m, {}).get('status') != 'ok' for m in ('cf384', 'unifd')):
        return 'pending_inference_or_integrity_error'
    if any(not isinstance(scores[m].get('prob'), Real) or isinstance(scores[m]['prob'], bool)
           or not math.isfinite(scores[m]['prob']) or not 0 <= scores[m]['prob'] <= 1
           for m in ('cf384', 'unifd')):
        return 'pending_inference_or_integrity_error'
    if any(scores[m]['prob'] >= thresholds[m] for m in ('cf384', 'unifd')):
        return 'ai_suspect_pending_api'
    return 'no_detector_alert'


def record_decision(record, image_statuses):
    normalized = record['normalized']
    metadata = task_metadata(record)
    if 'i2t_dedup_only' in image_statuses and metadata.get('task_route') != 'i2t':
        raise ValueError('I2T dedup status requires the bound I2T task policy')
    reasons = set(normalized['reason_codes']) - SCREENABLE
    if normalized['decision'] == 'rejected' or any(s.startswith('rejected_') for s in image_statuses):
        return 'rejected', sorted(reasons | {s for s in image_statuses if s.startswith('rejected_')})
    if len(image_statuses) != len(record['source_image_refs']):
        return 'pending', sorted(reasons | {'required_image_status_coverage_incomplete'})
    if reasons or any(s.startswith('pending_') or s == 'ai_suspect_pending_api' for s in image_statuses):
        return 'pending', sorted(reasons | {s for s in image_statuses if s.startswith('pending_') or s == 'ai_suspect_pending_api'})
    if metadata.get('task_route') == 'i2t':
        return 'i2t_dedup_only', []
    return 'frontpass_pass_pending_global_dedup', []


def validate_image_slots(record):
    refs = record['source_image_refs']
    images = record['normalized']['provenance']['images']
    if (len(refs) != len(images) or any(type(image.get('image_index')) is not int
            or image['image_index'] != index for index, image in enumerate(images))):
        raise ValueError('required image provenance slot coverage mismatch')


def prediction_errors(prediction):
    """Keep per-model failures from the resident worker, without vector payloads."""
    errors = {key: value for key, value in prediction.items() if key.startswith('error')}
    stages = {**prediction.get('scores', {})}
    if 'sscd' in prediction:
        stages['sscd'] = prediction['sscd']
    failed = {name: {key: value for key, value in result.items()
                     if key == 'status' or key.startswith('error')}
              for name, result in stages.items() if result.get('status') != 'ok'}
    if failed:
        errors['models'] = failed
    return errors


def checked_inference(worker, inputs, *, models):
    """Infrastructure/model failures retry the task instead of completing it.

    Individual bad image bytes remain per-image pending evidence. A full batch
    of missing paths indicates unavailable storage; a single missing image alone
    is insufficient to make that diagnosis.
    """
    iterator, expected = tee(iter(inputs))
    batch_size = getattr(worker, 'batch_size', 32)
    def serial_batches():
        while batch := list(islice(iterator, batch_size)):
            yield batch, list(worker.iter_results(batch, models=models))
    batches = (worker.iter_batches(iterator, models=models)
               if callable(getattr(worker, 'iter_batches', None)) else serial_batches())
    try:
        for batch, predictions in batches:
            wanted = list(islice(expected, len(batch)))
            if (not batch or len(batch) > batch_size or len(predictions) != len(batch)
                    or [asset['sha256'] for asset in batch] != [asset['sha256'] for asset in wanted]
                    or [p.get('image_sha256') for p in predictions] != [asset['sha256'] for asset in wanted]):
                raise RuntimeError('resident worker returned incomplete or reordered image coverage')
            for prediction in predictions:
                results = prediction.get('scores', {})
                for model in models:
                    result = prediction.get('sscd', {}) if model == 'sscd' else results.get(model, {})
                    if result.get('status') == 'error':
                        raise RuntimeError('resident model forward failed: ' + model + ':' + result.get('error_type', 'unknown'))
            if len(predictions) > 1 and all(prediction.get('status') == 'error'
                    and prediction.get('error_type') == 'FileNotFoundError' for prediction in predictions):
                raise RuntimeError('image storage unavailable: every path in inference batch is missing')
            yield from predictions
        if next(expected, None) is not None:
            raise RuntimeError('resident worker returned incomplete or reordered image coverage')
    finally:
        close = getattr(batches, 'close', None)
        if close:
            close()


def record_groups(iterator, seen, *, record_batch=128, aggregation=None):
    """Aggregate images without reordering or splitting native records.

    max_record_bytes bounds serialized record payload, not Python allocator RSS.
    A native record exceeding the byte cap is yielded alone and never truncated.
    Only image bytes for current + one future inference batch are decoded.
    """
    if aggregation is None:
        while records := list(islice(iterator, record_batch)):
            yield records
        return
    required = {'target_images', 'max_records', 'max_record_bytes'}
    if set(aggregation) != required or any(type(aggregation[k]) is not int or aggregation[k] < 1 for k in required):
        raise ValueError('invalid bounded record aggregation settings')
    records, new_images, byte_count = [], set(), 0
    for record in iterator:
        validate_image_slots(record)
        size = len(json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        if records and byte_count + size > aggregation['max_record_bytes']:
            yield records
            records, new_images, byte_count = [], set(), 0
        records.append(record)
        byte_count += size
        for ref, provenance in zip(record['source_image_refs'], record['normalized']['provenance']['images']):
            if ref['sha256'] not in seen and (provenance['origin'] != 'ai_generated'
                    or task_metadata(record).get('task_route') == 'i2t'):
                new_images.add(ref['sha256'])
        if (len(records) >= aggregation['max_records'] or byte_count >= aggregation['max_record_bytes']
                or len(new_images) >= aggregation['target_images']):
            yield records
            records, new_images, byte_count = [], set(), 0
    if records:
        yield records


def stats_delta(before, after):
    """Numeric counters only; stage times can overlap and must not be summed."""
    return {key: stats_delta(before.get(key, {}), value) if isinstance(value, dict)
            else value - before.get(key, 0)
            for key, value in after.items() if isinstance(value, (dict, int, float))}


def optimized_configuration(root, *, dtype, batch_size, threads):
    path = Path(root) / 'OPTIMIZED_QUALIFIED.json'
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    checks = {'success': True, 'dtype': dtype, 'batch_size': batch_size, 'threads': threads,
              'base_inference_code_sha256': file_sha(Path(__file__).with_name('inference.py')),
              'optimized_inference_code_sha256': file_sha(Path(__file__).with_name('optimized_inference.py')),
              'frontpass_code_sha256': file_sha(__file__)}
    if any(receipt.get(key) != value for key, value in checks.items()):
        raise RuntimeError('optimized inference qualification does not match this execution')
    aggregation = receipt.get('aggregation')
    if (not isinstance(aggregation, dict) or set(aggregation) != {'target_images', 'max_records', 'max_record_bytes'}
            or any(type(v) is not int or v < 1 for v in aggregation.values())):
        raise RuntimeError('optimized inference qualification has invalid aggregation settings')
    return aggregation


class ParquetRows:
    def __init__(self, path, schema):
        import pyarrow.parquet as pq
        self.schema = schema
        self.writer = pq.ParquetWriter(str(path), schema, compression='zstd')
    def write(self, rows):
        import pyarrow as pa
        if rows:
            self.writer.write_table(pa.Table.from_pylist(rows, schema=self.schema))
    def close(self):
        self.writer.close()


def schemas(task_aware=False):
    import pyarrow as pa
    records = pa.schema([('sample_id', pa.string()), ('row_index', pa.int64()),
        ('source_record_sha256', pa.string()), ('category', pa.string()), ('text_origin', pa.string()),
        ('text_retained', pa.bool_()), ('image_ids', pa.list_(pa.string())), ('image_statuses', pa.list_(pa.string())),
        ('status', pa.string()), ('reason_codes', pa.list_(pa.string())), ('layout_source_tag', pa.string()),
        ('recaption_resolution_potential', pa.bool_())])
    images = pa.schema([('asset_id', pa.string()), ('sample_id', pa.string()), ('image_index', pa.int32()),
        ('original_ref_json', pa.string()), ('provenance_json', pa.string()), ('status', pa.string()),
        ('original_width', pa.int32()), ('original_height', pa.int32()), ('pixel_sha256', pa.string()),
        ('cf384_prob', pa.float32()), ('unifd_prob', pa.float32()), ('sscd_byte_offset', pa.int64()),
        ('resolution_potential', pa.bool_()), ('error_json', pa.string())])
    if task_aware:
        fields = [('processing_policy', pa.string()), ('task_route', pa.string()),
                  ('image_loss_allowed', pa.bool_())]
        for name, typ in fields:
            records = records.append(pa.field(name, typ))
            images = images.append(pa.field(name, typ))
    return records, images


def process_task(root, claim, worker, snap, rules, contracts, thresholds, *, record_batch=128, aggregation=None):
    """One independent attempt. Native source and every image slot stay bound."""
    import numpy as np
    root = Path(root).resolve()
    task = claim['task']
    out = root / 'artifacts' / claim['task_id'] / claim['token']
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 'production.input.v2', 'collection': task.get('collection', 'dcvlm'),
                'snapshot_root': snap['root'], 'snapshot_fingerprint': snap['fingerprint'],
                'shards': [{**task, 'row_stop': task['row_end']}], 'scope': POLICY}
    iterator = iter(iter_records(manifest, snap, rules, batch_size=record_batch))
    task_aware = rules.get('task_routing') == TASK_POLICY
    record_schema, image_schema = schemas(task_aware)
    records_out = ParquetRows(out / 'records.parquet', record_schema)
    images_out = ParquetRows(out / 'images.parquet', image_schema)
    suspects_out = ParquetRows(out / 'suspects.parquet', image_schema)
    counts, categories = Counter(), Counter()
    seen = {}  # Bounded by one task; cache never crosses an old run.
    baseline_stats = deepcopy(getattr(worker, 'stats', {}))
    group_histogram = Counter()
    started = time.time()
    try:
        with (out / 'sscd.f32').open('wb') as vectors:
            for records in record_groups(iterator, seen, record_batch=record_batch, aggregation=aggregation):
                group_histogram[str(len(records))] += 1
                ordinary, source_protected, i2t_only = {}, {}, {}
                for record in records:
                    validate_image_slots(record)
                    metadata = task_metadata(record, rules)
                    i2t = metadata.get('task_route') == 'i2t'
                    for ref, provenance in zip(record['source_image_refs'], record['normalized']['provenance']['images']):
                        sha = ref['sha256']
                        if i2t:
                            i2t_only[sha] = {'sha256': sha, 'original_ref': ref,
                                             'allow_small_for_dedup': True}
                            continue
                        if sha in seen or provenance['origin'] == 'ai_generated':
                            continue
                        target = source_protected if protected(provenance) else ordinary
                        target[sha] = {'sha256': sha, 'original_ref': ref}
                # If a hash also has protected evidence in this batch, preserve
                # both source associations. Global reconciliation happens later.
                for sha in ordinary:
                    source_protected.pop(sha, None)
                for inputs, models in ((ordinary, ('cf384', 'unifd', 'sscd')), (source_protected, ('sscd',))):
                    for prediction in checked_inference(worker, inputs.values(), models=models):
                        sha = prediction['image_sha256']
                        sscd = prediction.get('sscd', {})
                        if sscd.get('status') == 'ok':
                            vector = sscd.pop('embedding')
                            prediction['sscd_byte_offset'] = vectors.tell()
                            vectors.write(np.asarray(vector, dtype='<f4').tobytes())
                        seen[sha] = prediction
                        counts['unique_images_inferred_within_task'] += 1
                # SSCD-only I2T does not run a detector and accepts readable
                # small images. A T2I too-small result has no descriptor, so
                # shared images may need this separate bounded decode.
                missing_i2t = {sha: asset for sha, asset in i2t_only.items()
                               if seen.get(sha, {}).get('sscd', {}).get('status') != 'ok'}
                for prediction in checked_inference(worker, missing_i2t.values(), models=('sscd',)):
                    sha = prediction['image_sha256']; previous = seen.get(sha, {})
                    sscd = prediction.get('sscd', {})
                    if sscd.get('status') == 'ok':
                        vector = sscd.pop('embedding')
                        prediction['sscd_byte_offset'] = vectors.tell()
                        vectors.write(np.asarray(vector, dtype='<f4').tobytes())
                    if previous.get('scores'):
                        prediction['scores'] = previous['scores']
                    seen[sha] = prediction
                    counts['unique_images_inferred_within_task'] += int(not previous)
                record_rows, image_rows, suspect_rows = [], [], []
                for record in records:
                    statuses, potentials, ids = [], [], []
                    metadata = task_metadata(record, rules)
                    i2t = metadata.get('task_route') == 'i2t'
                    for index, (ref, provenance) in enumerate(zip(record['source_image_refs'], record['normalized']['provenance']['images'])):
                        sha = ref['sha256']; prediction = seen.get(sha, {})
                        # A protected-only previous occurrence needs detector
                        # evidence before an unprotected occurrence can pass.
                        if not i2t and not protected(provenance) and provenance['origin'] not in {'ai_generated'} and prediction.get('status') == 'ok' and not prediction.get('scores'):
                            extra = next(checked_inference(worker, [{'sha256': sha, 'original_ref': ref}], models=('cf384', 'unifd')))
                            prediction = {**prediction, 'scores': extra.get('scores', {}), 'status': extra['status'],
                                          **{key: value for key, value in extra.items() if key.startswith('error')}}
                            seen[sha] = prediction
                        status = route(provenance, prediction, thresholds, task_route=metadata.get('task_route'))
                        width, height = prediction.get('original_width'), prediction.get('original_height')
                        potential = bool(width and height and min(width, height) >= 128 and width * height >= 1048576)
                        row = {'asset_id': sha, 'sample_id': record['original_sample_id'], 'image_index': index,
                            'original_ref_json': json.dumps(ref, separators=(',', ':')),
                            'provenance_json': json.dumps({**provenance,
                                'collection':record.get('collection',record.get('normalized',{}).get('collection','dcvlm')),
                                'source':record['source'], **metadata}, ensure_ascii=False, separators=(',', ':')),
                            'status': status, 'original_width': width, 'original_height': height,
                            'pixel_sha256': prediction.get('pixel_sha256'),
                            'cf384_prob': None if i2t else prediction.get('scores', {}).get('cf384', {}).get('prob'),
                            'unifd_prob': None if i2t else prediction.get('scores', {}).get('unifd', {}).get('prob'),
                            'sscd_byte_offset': prediction.get('sscd_byte_offset'), 'resolution_potential': potential,
                            'error_json': json.dumps(prediction_errors(prediction)), **metadata}
                        image_rows.append(row)
                        if status == 'ai_suspect_pending_api' or status.startswith('pending_'):
                            suspect_rows.append(row)
                        statuses.append(status); ids.append(sha); potentials.append(potential)
                        counts['image_occurrences'] += 1; counts['image:' + status] += 1
                    decision, reasons = record_decision(record, statuses)
                    collection=record.get('collection',record.get('normalized',{}).get('collection','dcvlm'))
                    contract = contracts.entries.get((collection, record['source']), {})
                    category = record['normalized']['category']
                    record_rows.append({'sample_id': record['original_sample_id'],
                        'row_index': record['locator']['row_index'], 'source_record_sha256': record['source_record_sha256'],
                        'category': category, 'text_origin': record['normalized']['provenance']['text']['origin'],
                        'text_retained': True, 'image_ids': ids, 'image_statuses': statuses,
                        'status': decision, 'reason_codes': reasons,
                        'layout_source_tag': contract.get('tag', 'sample_to_decide'),
                        'recaption_resolution_potential': any(potentials), **metadata})
                    counts['records'] += 1; counts['record:' + decision] += 1; categories[category] += 1
                records_out.write(record_rows); images_out.write(image_rows); suspects_out.write(suspect_rows)
            vectors.flush(); os.fsync(vectors.fileno())
    finally:
        records_out.close(); images_out.close(); suspects_out.close()
    if counts['records'] != task['row_end'] - task['row_start']:
        raise ValueError('incomplete task row coverage')
    summary = {'policy': POLICY, 'task': task, 'token': claim['token'], 'counts': dict(counts),
               'categories': dict(categories), 'elapsed_seconds': time.time() - started,
               'model_identity_sha256': worker.identity_sha256, 'api_calls': 0,
               'old_results_used': bool(getattr(worker,'stats',{}).get('evidence_cache_hits',0)-baseline_stats.get('evidence_cache_hits',0)),
               'global_dedup_complete': False, 'training_release': False}
    if task_aware:
        summary.update(processing_policy=TASK_POLICY, rules_sha256=digest(rules))
    summary['performance'] = {'worker_counters': stats_delta(baseline_stats, getattr(worker, 'stats', {})),
                              'record_group_histogram': dict(group_histogram), 'aggregation': aggregation,
                              'stage_times_can_overlap': True}
    if getattr(worker, 'execution_identity', None):
        summary['execution_identity'] = worker.execution_identity
    atomic_json(out / 'summary.json', summary)
    artifacts = [{'path': str(p), 'sha256': file_sha(p)} for p in sorted(out.iterdir()) if p.is_file()]
    return summary, artifacts


def run_worker(root, *, device, dtype='float32', batch_size=32, threads=4, max_tasks=None, queue_backend='rpc'):
    from .inference import ResidentInferenceWorker
    if queue_backend == 'rpc':
        from . import queue_rpc as queue
        queue.health(root)
    elif queue_backend == 'local' and device == 'cpu':
        from . import queue
    else:
        raise ValueError('production workers require the central queue RPC')
    root = Path(root).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    rules = rules_for_plan(plan); snap = snapshot(plan['snapshot_root'])
    if plan['policy'] != POLICY or plan['snapshot_fingerprint'] != snap['fingerprint'] or plan['rules_sha256'] != digest(rules):
        raise ValueError('run/input/rule identity changed')
    if plan['thresholds_sha256'] != file_sha(BASE / 'detector_calibration/thresholds.json') or plan['layout_contracts_sha256'] != file_sha(BASE / 'prepared/config/layout_source_contracts.json'):
        raise ValueError('run threshold/layout identity changed')
    contracts = compile_source_contracts(json.loads((BASE / 'prepared/config/layout_source_contracts.json').read_text()))
    thresholds = json.loads((BASE / 'detector_calibration/thresholds.json').read_text())['review_thresholds']
    worker_id = f'{socket.gethostname()}-{device.replace(":", "_")}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    worker_root = root / 'workers' / worker_id
    worker_root.mkdir(parents=True)
    aggregation = optimized_configuration(root, dtype=dtype, batch_size=batch_size, threads=threads)
    worker_class = ResidentInferenceWorker
    if aggregation is not None:
        from .optimized_inference import OptimizedResidentInferenceWorker
        worker_class = OptimizedResidentInferenceWorker
    with worker_class(device=device, dtype=dtype, batch_size=batch_size, threads=threads) as worker:
        atomic_json(worker_root / 'model_identity.json', worker.identity)
        if getattr(worker, 'execution_identity', None):
            atomic_json(worker_root / 'execution_identity.json', worker.execution_identity)
        done = 0
        while max_tasks is None or done < max_tasks:
            claim = queue.claim(root, worker_id)
            if claim is None:
                state = queue.status(root)
                atomic_json(worker_root / 'last_queue_status.json', state)
                if state['all_tasks_completed']:
                    break
                if state['counts']['pending'] or state['counts']['running']:
                    # Other workers may still own the tail tasks. Stay alive
                    # so a preempted owner's expired lease can be reclaimed.
                    # The global status scan only runs when claim found no work.
                    time.sleep(random.uniform(30, 45))
                    continue
                if state['counts']['failed']:
                    raise RuntimeError('queue ended with terminal failed tasks: ' + str(state['counts']['failed']))
                raise RuntimeError('queue has no claimable work but is not complete')
            stop = threading.Event(); heartbeat_errors = []
            def beat():
                while not stop.wait(30):
                    try:
                        queue.heartbeat(root, claim['task_id'], claim['token'])
                    except Exception as error:
                        heartbeat_errors.append(repr(error)); return
            thread = threading.Thread(target=beat, daemon=True); thread.start()
            try:
                summary, artifacts = process_task(root, claim, worker, snap, rules, contracts, thresholds,
                                                  aggregation=aggregation)
                if heartbeat_errors:
                    raise RuntimeError('lost lease: ' + heartbeat_errors[0])
                queue.complete(root, claim['task_id'], claim['token'], artifacts=artifacts,
                               processed_records=summary['counts']['records'],
                               processed_images=summary['counts'].get('image_occurrences', 0))
                done += 1
                print(json.dumps({'worker': worker_id, 'task': claim['task_id'], **summary['counts'],
                                  'elapsed_seconds': summary['elapsed_seconds']}), flush=True)
            except Exception as error:
                traceback.print_exc()
                queue.fail(root, claim['task_id'], claim['token'], repr(error), retryable=True)
                raise
            finally:
                stop.set(); thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('prepare'); init.add_argument('--root', required=True, type=Path)
    init.add_argument('--dataset', type=Path, default=DEFAULT_ROOT)
    init.add_argument('--rows-per-task', type=int, default=8192); init.add_argument('--max-shards', type=int)
    worker = sub.add_parser('worker'); worker.add_argument('--root', required=True, type=Path)
    worker.add_argument('--device', required=True); worker.add_argument('--dtype', default='float32')
    worker.add_argument('--batch-size', type=int, default=32); worker.add_argument('--threads', type=int, default=4)
    worker.add_argument('--max-tasks', type=int)
    worker.add_argument('--queue-backend', choices=['rpc', 'local'], default='rpc')
    args = parser.parse_args(); params = vars(args); command = params.pop('command')
    result = prepare(**params) if command == 'prepare' else run_worker(**params)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
