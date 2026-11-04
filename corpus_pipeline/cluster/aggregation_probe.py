"""Bounded aggregation profiling against existing real-task baseline artifacts.

One resident candidate is reconfigured serially. Historical baseline throughput
is descriptive, not a paired benchmark or production qualification. This module
never claims queue work, pauses workers, publishes a qualification, or calls APIs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import islice
import json
import os
from pathlib import Path
import signal
import socket
import time

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from .tuned_frontpass_probe import select_aggregation


SOURCES = ('as100m', 'anyword3m', 'allava')
MAJOR_SOURCES = ('datacomp_1b', 'relaion-2b-en-vol-1-unpacked-TEST',
                 'relaion-2b-en-vol-2-unpacked-TEST')
ARTIFACT_FILES = ('records.parquet', 'images.parquet', 'sscd.f32')
COMPUTATION_FILES = ('inference.py', 'optimized_inference.py', 'frontpass.py',
                     'tuned_inference.py', 'fast_prepare.py', 'optimization_probe.py')
BASELINE_AGGREGATION = {'target_images': 128, 'max_records': 1024, 'max_record_bytes': 64 << 20}


def profiles(source, configuration):
    """Fixed bounded grid; all candidates retain the chosen fast preparation."""
    if source not in SOURCES + MAJOR_SOURCES:
        raise ValueError('Unsupported diagnostic source: ' + source)
    result = []
    if source in MAJOR_SOURCES:
        settings = [(32, 8, 2, 2), (64, 16, 1, 2), (32, 4, 4, 2),
                    (128, 8, 1, 2), (256, 8, 1, 2)]
        if source == 'datacomp_1b':
            settings.append((512, 8, 1, 1))
        for batch, decode, cpu, prefetch in settings:
            cfg = {**configuration, 'batch_size': batch, 'decode_workers': decode,
                   'cpu_threads': cpu, 'prefetch_batches': prefetch}
            result.append({'name': f'b{batch}-d{decode}-c{cpu}-p{prefetch}-a1024',
                           'configuration': cfg, 'aggregation': select_aggregation(cfg, 1024)})
        return result
    if source != 'anyword3m':
        result.append({'name': 'selected-a1024', 'configuration': dict(configuration),
                       'aggregation': select_aggregation(configuration, 1024)})
    alternatives = []
    if source == 'as100m':
        alternatives = [(32, 8, 2)]
    elif source == 'anyword3m':
        alternatives = [(32, 8, 2), (64, 16, 1), (32, 4, 4)]
    for batch, decode, cpu in alternatives:
        config = {**configuration, 'batch_size': batch, 'decode_workers': decode,
                  'cpu_threads': cpu, 'prefetch_batches': 2}
        result.append({'name': f'b{batch}-d{decode}-c{cpu}-p2-a1024',
                       'configuration': config, 'aggregation': select_aggregation(config, 1024)})
    return result


def reference_cases(reference, selected):
    """Bind exact task IDs/source/row coverage before comparing artifact rows."""
    if reference.get('success') is not True or reference.get('dtype') != 'float32':
        raise ValueError('A successful FP32 frontpass reference is required')
    result = {}
    for label, task in selected:
        matches = [row for row in reference.get('results', [])
                   if row.get('case') == label and row.get('repetition') == 0
                   and row.get('source') == task['source']]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous reference case: ' + label)
        row = matches[0]
        baseline = row['timings']['baseline']
        expected_rows = task['row_end'] - task['row_start']
        if (baseline['counts']['records'] != expected_rows
                or baseline['elapsed_seconds'] <= 0
                or row['comparison']['record_outcomes_equal'] is not True):
            raise ValueError('Reference row coverage/timing is invalid: ' + label)
        result[label] = baseline
    return result


def all_code_hashes():
    return {name: file_sha(Path(__file__).with_name(name))
            for name in COMPUTATION_FILES + ('aggregation_probe.py',)}


def selected_cases(root, snap):
    result = []
    for source in MAJOR_SOURCES:
        path, entry = next((p, e) for p, e in sorted(snap['catalog_index'].items())
                           if e['source'] == source)
        stat = Path(path).stat()
        result.append(('major-' + source, {'parquet_path': path, 'row_start': 0,
            'row_end': min(1024, entry['rows']), 'source': source, 'source_shard': entry['source_shard'],
            'parquet_fingerprint': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}))
    for task_id, source, bound in (('s00030-r00000000', 'as100m', 2048),
                                  ('s00016-r00000000', 'anyword3m', 1024),
                                  ('s00007-r00000000', 'allava', 8192)):
        task = deepcopy(json.loads((root / 'states' / (task_id + '.json')).read_text())['task'])
        if task['task_id'] != task_id or task['source'] != source:
            raise ValueError('Native task/source mismatch: ' + task_id)
        task['row_end'] = min(task['row_end'], task['row_start'] + bound)
        result.append((task_id, task))
    return result


def warm_inputs_for(task, count, snap, rules):
    """Read only enough image references within the complete bounded task.

    Records can be text-only or contain several images: a record count is not
    an image count. Keep the task's row bound, then lazily cap flattened refs.
    """
    from corpus_pipeline.production.ingest import iter_records
    if type(count) is not int or count < 1:
        raise ValueError('Positive warmup image count required')
    manifest = {'schema_version': 'production.input.v2', 'collection': 'dcvlm',
        'snapshot_root': snap['root'], 'snapshot_fingerprint': snap['fingerprint'],
        'shards': [{**task, 'row_stop': task['row_end']}]}
    refs = (ref for record in iter_records(manifest, snap, rules)
            for ref in record['source_image_refs'])
    inputs = [{'sha256': ref['sha256'], 'original_ref': ref} for ref in islice(refs, count)]
    if len(inputs) < count:
        raise ValueError('Insufficient images for a complete warmup batch: ' + task['source'])
    return inputs


def run(root, output, reference_root, config, device):
    import torch
    from .tuned_inference import TunedResidentInferenceWorker
    from .optimized_inference import OptimizedResidentInferenceWorker
    from .frontpass import BASE, POLICY, load_rules, process_task
    from .optimization_probe import compare_outputs
    from corpus_pipeline.production.records import snapshot
    from corpus_pipeline.production.potential import compile_source_contracts

    original_hashes = all_code_hashes()
    reference_path = reference_root / 'COMPLETE.json'
    reference_sha = file_sha(reference_path)
    reference = json.loads(reference_path.read_text())
    plan_sha = file_sha(root / 'plan.json')
    plan = json.loads((root / 'plan.json').read_text())
    snap, rules = snapshot(plan['snapshot_root']), load_rules()
    if (plan['policy'] != POLICY or plan['snapshot_fingerprint'] != snap['fingerprint']
            or plan['rules_sha256'] != digest(rules)
            or reference.get('plan_sha256') != plan_sha
            or reference.get('snapshot_fingerprint') != snap['fingerprint']
            or any(reference.get('code_sha256', {}).get(name) != original_hashes[name]
                   for name in COMPUTATION_FILES)):
        raise ValueError('Frozen plan/source/rules/reference/code identity mismatch')
    for key, path in [('thresholds_sha256', BASE / 'detector_calibration/thresholds.json'),
                      ('layout_contracts_sha256', BASE / 'prepared/config/layout_source_contracts.json')]:
        if plan[key] != file_sha(path):
            raise ValueError('Frozen policy input changed: ' + key)
    contracts = compile_source_contracts(json.loads((BASE / 'prepared/config/layout_source_contracts.json').read_text()))
    thresholds = json.loads((BASE / 'detector_calibration/thresholds.json').read_text())['review_thresholds']
    selected = selected_cases(root, snap)
    historical = [(label, task) for label, task in selected if task['source'] in SOURCES]
    stored_cases = dict(json.loads((reference_root / 'CASES.json').read_text()))
    if any(stored_cases.get(label) != task for label, task in historical):
        raise ValueError('Reference task metadata/row bounds differ')
    baselines = reference_cases(reference, historical)
    grids = {task['source']: profiles(task['source'], config) for _, task in selected}
    reference_hashes = {}
    for label, _ in historical:
        baseline_dir = reference_root / 'artifacts' / (label + '-r0') / 'baseline'
        reference_hashes[label] = {name: file_sha(baseline_dir / name) for name in ARTIFACT_FILES}
    atomic_json(output / 'INPUT.json', {'root': str(root), 'reference_root': str(reference_root),
        'reference_complete_sha256': reference_sha, 'reference_artifact_sha256': reference_hashes,
        'plan_sha256': plan_sha, 'snapshot_fingerprint': snap['fingerprint'],
        'code_sha256': original_hashes, 'profiles': grids, 'cases': selected,
        'device': device, 'dtype': 'float32', 'hostname': socket.gethostname(),
        'scope': 'profiling_only_historical_baseline_not_production_qualification'})
    warm_inputs = warm_inputs_for(next(task for _, task in selected if task['source'] == 'as100m'), 32, snap, rules)
    torch.set_num_interop_threads(1)
    results = []
    started = time.perf_counter()
    baseline_paths = {label: reference_root / 'artifacts' / (label + '-r0') / 'baseline'
                      for label, _ in historical}
    # Fresh major-source baselines are measured in this diagnostic run. Release
    # their worker before allocating the single reusable tuned candidate.
    with OptimizedResidentInferenceWorker(device=device, dtype='float32', batch_size=32, threads=4) as baseline:
        if baseline.identity != reference['model_identity']:
            raise ValueError('Incumbent model identity differs from the reference')
        values = baseline.infer_batch(warm_inputs[:32])
        if any(row['status'] not in ('ok', 'too_small') for row in values):
            raise RuntimeError('Major-source baseline warmup failed')
        del values
        baseline.synchronize()
        for label, task in selected:
            if task['source'] not in MAJOR_SOURCES:
                continue
            claim = {'task_id': label + '-r0', 'token': 'baseline', 'task': task}
            summary, artifacts = process_task(output, claim, baseline, snap, rules, contracts,
                thresholds, aggregation=BASELINE_AGGREGATION)
            baseline.synchronize()
            baselines[label] = {'elapsed_seconds': summary['elapsed_seconds'], 'counts': summary['counts'],
                'performance': summary.get('performance'), 'artifacts': artifacts,
                'images_per_second': summary['counts'].get('image_occurrences', 0) / summary['elapsed_seconds']}
            baseline_paths[label] = output / 'artifacts' / claim['task_id'] / 'baseline'
            reference_hashes[label] = {name: file_sha(baseline_paths[label] / name) for name in ARTIFACT_FILES}
            atomic_json(output / 'BASELINES.json', baselines)
            print(json.dumps({'source': task['source'], 'mode': 'fresh_baseline',
                              'images_per_second': baselines[label]['images_per_second']}), flush=True)
    if str(device).startswith('npu'):
        torch.npu.empty_cache()
    with TunedResidentInferenceWorker(device=device, dtype='float32', **config) as worker:
        if worker.identity != reference['model_identity']:
            raise ValueError('Candidate model identity differs from the reference')
        for label, task in selected:
            baseline_dir = baseline_paths[label]
            warm_inputs = warm_inputs_for(task, max(p['configuration']['batch_size']
                                                   for p in grids[task['source']]), snap, rules)
            for profile in grids[task['source']]:
                worker.configure(**profile['configuration'])
                values = worker.infer_batch(warm_inputs[:worker.batch_size])
                if any(row['status'] not in ('ok', 'too_small') for row in values):
                    raise RuntimeError('Candidate profile warmup failed')
                del values
                worker.synchronize()
                if str(worker.device).startswith('npu'):
                    torch.npu.reset_peak_memory_stats(worker.device)
                # Keep the reference task ID for exact record bindings; only
                # the output attempt token differs between independent profiles.
                claim = {'task_id': label + '-r0', 'token': profile['name'], 'task': task}
                cpu_started = time.process_time()
                summary, artifacts = process_task(output, claim, worker, snap, rules, contracts,
                    thresholds, aggregation=profile['aggregation'])
                elapsed_cpu = time.process_time() - cpu_started
                worker.synchronize()
                candidate_dir = output / 'artifacts' / claim['task_id'] / claim['token']
                comparison = compare_outputs(baseline_dir, candidate_dir)
                elapsed = summary['elapsed_seconds']
                timing = {'elapsed_seconds': elapsed, 'process_cpu_seconds': elapsed_cpu,
                    'counts': summary['counts'], 'performance': summary.get('performance'),
                    'artifacts': artifacts,
                    'images_per_second': summary['counts'].get('image_occurrences', 0) / elapsed,
                    'records_per_second': summary['counts']['records'] / elapsed}
                if str(worker.device).startswith('npu'):
                    timing.update(peak_npu_allocated_bytes=torch.npu.max_memory_allocated(worker.device),
                                  peak_npu_reserved_bytes=torch.npu.max_memory_reserved(worker.device))
                row = {'case': label, 'source': task['source'], **profile,
                    'comparison': comparison, 'timing': timing,
                    'execution_identity': deepcopy(worker.execution_identity),
                    'baseline_timing': baselines[label],
                    'baseline_measurement': 'fresh_same_run' if task['source'] in MAJOR_SOURCES else 'historical',
                    'ratio_to_baseline': baselines[label]['elapsed_seconds'] / elapsed}
                results.append(row)
                atomic_json(output / 'PARTIAL.json', results)
                print(json.dumps({k: row[k] for k in ('case', 'source', 'name', 'configuration',
                    'aggregation', 'comparison', 'baseline_measurement', 'ratio_to_baseline')}
                    | {'images_per_second': timing['images_per_second']}), flush=True)
            atomic_json(output / (task['source'] + '.COMPLETE.json'),
                {'success': True, 'results': [r for r in results if r['source'] == task['source']],
                 'scope': 'profiling_only_not_production_qualification'})
        identity = deepcopy(worker.identity)
    if (all_code_hashes() != original_hashes or file_sha(reference_path) != reference_sha
            or file_sha(root / 'plan.json') != plan_sha):
        raise RuntimeError('Code/reference/plan changed during profiling')
    for label, hashes in reference_hashes.items():
        baseline_dir = baseline_paths[label]
        if hashes != {name: file_sha(baseline_dir / name) for name in ARTIFACT_FILES}:
            raise RuntimeError('Reference artifacts changed during profiling')
    receipt = {'success': True, 'results': results, 'model_identity': identity,
        'elapsed_seconds': time.perf_counter() - started, 'code_sha256': original_hashes,
        'plan_sha256': plan_sha, 'reference_complete_sha256': reference_sha,
        'reference_root': str(reference_root), 'reference_artifact_sha256': reference_hashes,
        'hostname': socket.gethostname(), 'device': device, 'dtype': 'float32',
        'scope': 'profiling_only_historical_baseline_not_production_qualification',
        'production_queue_mutations': 0, 'production_paused': False, 'api_calls': 0,
        'production_qualification': False, 'completed_at': time.time()}
    atomic_json(output / 'COMPLETE.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'output', 'reference-root', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--timeout-seconds', type=int, default=900)
    args = parser.parse_args()
    root, output, reference_root = args.root.resolve(), args.output.resolve(), args.reference_root.resolve()
    if os.path.lexists(args.output):
        parser.error('output already exists; choose a new diagnostic directory')
    if not 30 <= args.timeout_seconds <= 900:
        parser.error('timeout must be between 30 and 900 seconds')
    config = json.loads(args.config.read_text())
    for source in SOURCES + MAJOR_SOURCES:
        profiles(source, config)
    output.mkdir(parents=True, exist_ok=False)
    def timeout(*_):
        raise TimeoutError('Bounded aggregation profiling deadline')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        result = run(root, output, reference_root, config, args.device)
    except BaseException as error:
        atomic_json(output / 'FAILURE.json', {'success': False, 'error_type': type(error).__name__,
            'error': str(error)[:1000], 'at': time.time(), 'production_queue_mutations': 0})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    print(json.dumps({'success': result['success'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
