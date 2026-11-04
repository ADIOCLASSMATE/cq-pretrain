"""Dedicated-NPU comparison of real frontpass tasks; never touches queue state.

Writes an exclusively created diagnostic directory. No production pause, task
claim, API request, source change, or production qualification file is performed.
The incumbent optimized worker and candidate retain identical FP32 model math.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path
import signal
import socket
import time

from corpus_pipeline.production.common import atomic_json, digest, file_sha


BASELINE_AGGREGATION = {'target_images': 128, 'max_records': 1024, 'max_record_bytes': 64 << 20}
CONFIG_FIELDS = {'batch_size', 'decode_workers', 'cpu_threads', 'prefetch_batches', 'fast_prepare'}
CODE_FILES = ('inference.py', 'optimized_inference.py', 'fast_prepare.py',
              'tuned_inference.py', 'frontpass.py', 'optimization_probe.py', 'tuned_frontpass_probe.py',
              'tuned_frontpass.py', 'tuned_configuration.py', 'node_worker.py')


def select_aggregation(config, target_images=None):
    from .tuned_inference import TunedResidentInferenceWorker
    if set(config) != CONFIG_FIELDS:
        raise ValueError('config JSON must contain exactly: ' + ', '.join(sorted(CONFIG_FIELDS)))
    TunedResidentInferenceWorker._validate_configuration(config['batch_size'], config['decode_workers'],
        config['cpu_threads'], config['prefetch_batches'])
    if type(config['fast_prepare']) is not bool:
        raise ValueError('fast_prepare must be boolean')
    required = config['batch_size'] * (config['prefetch_batches'] + 1)
    if target_images is None:
        target_images = next((n for n in (256, 512, 1024) if n >= required), None)
    if target_images not in (256, 512, 1024) or target_images < required:
        raise ValueError('target-images must be 256/512/1024 and >= batch_size*(prefetch_batches+1)')
    return {**BASELINE_AGGREGATION, 'target_images': target_images}


def cases(root, snap):
    result = []
    for task_id, bound, source in (('s00030-r00000000', 2048, 'as100m'),
                                   ('s00016-r00000000', 1024, 'anyword3m'),
                                   ('s00007-r00000000', 8192, 'allava')):
        state = json.loads((root / 'states' / (task_id + '.json')).read_text())
        task = deepcopy(state['task'])
        if task['task_id'] != task_id or task['source'] != source:
            raise ValueError('Native task identity/source mismatch: ' + task_id)
        task['row_end'] = min(task['row_end'], task['row_start'] + bound)
        result.append((task_id, task))
    for source in ('datacomp_1b', 'relaion-2b-en-vol-1-unpacked-TEST',
                   'relaion-2b-en-vol-2-unpacked-TEST',
                   'chartqa', 'clevr_math', 'docvqa', 'iam', 'coco_caption'):
        path, entry = next((p, e) for p, e in snap['catalog_index'].items() if e['source'] == source)
        stat = Path(path).stat()
        primary = source in ('datacomp_1b', 'relaion-2b-en-vol-1-unpacked-TEST',
                             'relaion-2b-en-vol-2-unpacked-TEST')
        bound = 1024 if primary else 16
        if entry['rows'] < bound:
            raise ValueError('Source shard lacks required validation rows: ' + source)
        result.append((('primary-' if primary else 'diverse-') + source, {'parquet_path': path, 'row_start': 0,
            'row_end': bound, 'source': source, 'source_shard': entry['source_shard'],
            'parquet_fingerprint': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}))
    return result


def code_hashes():
    return {name: file_sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def run(root, output, device, config, candidate_aggregation, source_profiles):
    import torch
    from .optimized_inference import OptimizedResidentInferenceWorker
    from .tuned_inference import TunedResidentInferenceWorker
    from .frontpass import BASE, POLICY, load_rules, process_task
    from .optimization_probe import compare_outputs
    from .tuned_configuration import select_source_profile, _validate_source_profiles
    from corpus_pipeline.production.records import snapshot
    from corpus_pipeline.production.potential import compile_source_contracts
    from corpus_pipeline.production.ingest import iter_records

    _validate_source_profiles(source_profiles)
    profiles = {'configuration': config, 'aggregation': candidate_aggregation,
                'source_profiles': source_profiles}
    original_hashes = code_hashes()
    plan = json.loads((root / 'plan.json').read_text())
    snap, rules = snapshot(plan['snapshot_root']), load_rules()
    if plan['policy'] != POLICY or plan['snapshot_fingerprint'] != snap['fingerprint'] or plan['rules_sha256'] != digest(rules):
        raise ValueError('Frozen source/plan/rules identity mismatch')
    for key, path in [('thresholds_sha256', BASE / 'detector_calibration/thresholds.json'),
                       ('layout_contracts_sha256', BASE / 'prepared/config/layout_source_contracts.json')]:
        if plan[key] != file_sha(path):
            raise ValueError('Frozen policy input changed: ' + key)
    contracts = compile_source_contracts(json.loads((BASE / 'prepared/config/layout_source_contracts.json').read_text()))
    thresholds = json.loads((BASE / 'detector_calibration/thresholds.json').read_text())['review_thresholds']
    selected = cases(root, snap)
    atomic_json(output / 'CASES.json', selected)
    atomic_json(output / 'INPUT.json', {'root': str(root), 'plan_sha256': file_sha(root / 'plan.json'),
        'code_sha256': original_hashes, 'candidate_configuration': config,
        'baseline_aggregation': BASELINE_AGGREGATION, 'candidate_aggregation': candidate_aggregation,
        'source_profiles': source_profiles,
        'device': device, 'dtype': 'float32', 'hostname': socket.gethostname()})
    torch.set_num_interop_threads(1)
    results = []
    with OptimizedResidentInferenceWorker(device=device, dtype='float32', batch_size=32, threads=4) as baseline, \
         TunedResidentInferenceWorker(device=device, dtype='float32', **config) as candidate:
        if baseline.identity != candidate.identity or baseline.identity_sha256 != candidate.identity_sha256:
            raise AssertionError('Candidate changed qualified model identity')
        warm_task = selected[0][1]
        warm_manifest = {'schema_version': 'production.input.v2', 'collection': 'dcvlm',
            'snapshot_root': snap['root'], 'snapshot_fingerprint': snap['fingerprint'],
            'shards': [{**warm_task, 'row_stop': warm_task['row_start'] + max(32, config['batch_size'])}]}
        warm_inputs = [{'sha256': ref['sha256'], 'original_ref': ref}
                       for record in iter_records(warm_manifest, snap, rules)
                       for ref in record['source_image_refs']]
        for worker, cpu_threads in ((baseline, 4), (candidate, config['cpu_threads'])):
            torch.set_num_threads(cpu_threads)
            values = worker.infer_batch(warm_inputs[:worker.batch_size])
            if any(row['status'] not in ('ok', 'too_small') for row in values):
                raise RuntimeError('Dedicated validation warmup failed')
            del values
        for repetition in range(2):
            for label, task in selected:
                if repetition and task['source'] != 'as100m':
                    continue
                profile = select_source_profile(profiles, task['source'])
                actual_config, actual_aggregation = profile['configuration'], profile['aggregation']
                if any(candidate.execution_identity.get(k) != v for k, v in actual_config.items()):
                    candidate.configure(**actual_config)
                    # Compile/warm the new batch shape outside the timed stream,
                    # using this source's native images, with no queue mutations.
                    source_warm = {**warm_manifest, 'shards': [{**task,
                        'row_stop': task['row_end']}]}
                    source_inputs = list(islice(({'sha256': ref['sha256'], 'original_ref': ref}
                        for record in iter_records(source_warm, snap, rules)
                        for ref in record['source_image_refs']), candidate.batch_size))
                    values = candidate.infer_batch(source_inputs[:candidate.batch_size])
                    if any(row['status'] not in ('ok', 'too_small') for row in values):
                        raise RuntimeError('Source profile warmup failed: ' + task['source'])
                    del values, source_inputs
                outputs, timings = {}, {}
                modes = [('baseline', baseline, BASELINE_AGGREGATION, 4),
                         ('candidate', candidate, actual_aggregation, actual_config['cpu_threads'])]
                if repetition:
                    modes.reverse()
                for mode, worker, aggregation, cpu_threads in modes:
                    # This setting is process-global. Change it only between
                    # fully drained serial streams, never while another forwards.
                    torch.set_num_threads(cpu_threads)
                    worker.synchronize()
                    if str(worker.device).startswith('npu'):
                        torch.npu.reset_peak_memory_stats(worker.device)
                    claim = {'task_id': f'{label}-r{repetition}', 'token': mode, 'task': task}
                    cpu_started = time.process_time()
                    summary, artifacts = process_task(output, claim, worker, snap, rules, contracts,
                        thresholds, aggregation=aggregation)
                    elapsed_cpu = time.process_time() - cpu_started
                    worker.synchronize()
                    outputs[mode] = output / 'artifacts' / claim['task_id'] / mode
                    timings[mode] = {'elapsed_seconds': summary['elapsed_seconds'],
                        'process_cpu_seconds': elapsed_cpu, 'counts': summary['counts'],
                        'performance': summary.get('performance'), 'artifacts': artifacts,
                        'images_per_second': summary['counts'].get('image_occurrences', 0) / summary['elapsed_seconds'],
                        'records_per_second': summary['counts']['records'] / summary['elapsed_seconds']}
                    if str(worker.device).startswith('npu'):
                        timings[mode].update(peak_npu_allocated_bytes=torch.npu.max_memory_allocated(worker.device),
                                             peak_npu_reserved_bytes=torch.npu.max_memory_reserved(worker.device))
                comparison = compare_outputs(outputs['baseline'], outputs['candidate'])
                row = {'case': label, 'source': task['source'], 'repetition': repetition,
                       'candidate_configuration': actual_config, 'candidate_aggregation': actual_aggregation,
                       'comparison': comparison, 'timings': timings,
                       'speedup': timings['baseline']['elapsed_seconds'] / timings['candidate']['elapsed_seconds']}
                results.append(row)
                atomic_json(output / 'PARTIAL.json', results)
                print(json.dumps({k: row[k] for k in ('case', 'source', 'repetition', 'speedup', 'comparison')}), flush=True)
        identity = deepcopy(baseline.identity)
        if any(candidate.execution_identity.get(k) != v for k, v in config.items()):
            candidate.configure(**config)
        execution_identity = deepcopy(candidate.execution_identity)
    final_hashes = code_hashes()
    if final_hashes != original_hashes:
        raise RuntimeError('Implementation changed while validating; diagnostic cannot qualify execution')
    receipt = {'success': True, 'hostname': socket.gethostname(), 'device': device, 'dtype': 'float32',
        'plan_sha256': file_sha(root / 'plan.json'), 'snapshot_fingerprint': snap['fingerprint'],
        'candidate_configuration': config, 'baseline_configuration': {'batch_size': 32, 'threads': 4},
        'baseline_aggregation': BASELINE_AGGREGATION, 'candidate_aggregation': candidate_aggregation,
        'source_profiles': source_profiles,
        'results': results, 'model_identity': identity, 'candidate_execution_identity': execution_identity,
        'code_sha256': original_hashes, 'validated_at': time.time(),
        'scope': 'bounded_real_frontpass_numerical_and_throughput_comparison_not_AI_accuracy',
        'production_queue_mutations': 0, 'production_paused': False, 'api_calls': 0, 'training_release': False,
        'cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip()}
    atomic_json(output / 'COMPLETE.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-profiles', type=Path)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--target-images', type=int, choices=(256, 512, 1024))
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        parser.error('output already exists; choose a new diagnostic directory')
    config = json.loads(args.config.read_text())
    aggregation = select_aggregation(config, args.target_images)
    source_profiles = json.loads(args.source_profiles.read_text()) if args.source_profiles else {}
    if args.timeout_seconds < 30:
        parser.error('timeout must be at least 30 seconds')
    output.mkdir(parents=True, exist_ok=False)
    def timeout(*_):
        raise TimeoutError('Dedicated frontpass probe deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        result = run(root, output, args.device, config, aggregation, source_profiles)
    except BaseException as error:
        atomic_json(output / 'FAILURE.json', {'success': False, 'error_type': type(error).__name__,
            'error': str(error)[:1000], 'at': time.time(), 'production_queue_mutations': 0})
        raise
    finally:
        signal.alarm(0)
    print(json.dumps({'success': result['success'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
