"""Bounded real-data comparison; never claims or publishes production tasks.

Optionally pauses the exact local Notebook's existing frontpass children while
comparing two implementations on its assigned NPU. Identity is checked before
each signal; finally, SIGTERM and a hard time budget all resume those children.
The completed production queue and raw dataset remain untouched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np

from corpus_pipeline.production.common import atomic_json, file_sha

AGGREGATION = {'target_images': 128, 'max_records': 1024, 'max_record_bytes': 64 << 20}


def _same_process(expected):
    from .notebook_worker import _process
    actual = _process(expected['pid'])
    return actual is not None and all(actual[k] == expected[k] for k in (
        'pid', 'start_ticks', 'boot_id', 'command_sha256'))


def _resume_watchdog(payload):
    """Separate process: resume even if the benchmarking process is SIGKILLed."""
    output = Path(payload['output'])
    atomic_json(output / 'WATCHDOG_READY.json', {'pid': os.getpid(), 'at': time.time()})
    while not (output / 'PRODUCTION_RESUME.json').is_file():
        parent_alive = _same_process(payload['parent'])
        if not parent_alive or time.time() >= payload['deadline']:
            resumed = []
            for child in payload['children']:
                if _same_process(child):
                    try:
                        os.kill(child['pid'], signal.SIGCONT)
                        resumed.append(child['pid'])
                    except ProcessLookupError:
                        pass
            atomic_json(output / 'WATCHDOG_INTERVENTION.json', {
                'at': time.time(), 'pids': resumed,
                'reason': 'deadline' if parent_alive else 'probe_process_exited'})
            if parent_alive and _same_process(payload['parent']):
                try:
                    os.kill(payload['parent']['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return
        time.sleep(.25)


@contextmanager
def pause_production(root, name, output, seconds=240):
    from .notebook_worker import _location, _read_attempt, _identity_matches, _process
    _, base = _location(root, name)
    directory, intent = _read_attempt(base)
    if intent is None:
        raise RuntimeError('No accepted local production supervisor')
    owner = json.loads((directory / 'OWNER.json').read_text())
    if not _identity_matches(owner, intent['supervisor_command']):
        raise RuntimeError('Production supervisor identity is not live')
    node = _process(owner['node_pid'])
    if node is None or 'corpus_pipeline.cluster.node_worker' not in node['argv']:
        raise RuntimeError('Production node identity is not live')
    node_fields = (Path('/proc') / str(node['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    if int(node_fields[1]) != owner['pid'] or node['argv'][1:] != intent['node_command'][2:]:
        raise RuntimeError('Production node is not the accepted supervisor child')
    children = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[1]) != node['pid']:
                continue
            child = _process(int(path.name))
            if child and 'corpus_pipeline.cluster.frontpass' in child['argv']:
                index = child['argv'].index('--root')
                if Path(child['argv'][index + 1]).resolve() != root:
                    raise RuntimeError('Unexpected production queue owner')
                children.append(child)
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not children or len(children) != intent['devices']:
        raise RuntimeError('Production child count differs from Notebook contract')
    for lease_path in (root / 'claims').glob('*/lease.json'):
        try:
            lease = json.loads(lease_path.read_text())
        except FileNotFoundError:
            continue
        if any(f'-{child["pid"]}-' in lease['worker_id'] and
               lease['worker_id'].startswith(socket.gethostname() + '-') for child in children):
            if lease['lease_expires_at'] - time.time() < seconds + 60:
                raise RuntimeError('Remaining task lease is too short for a bounded pause')
    paused = []
    previous = {}
    watchdog = None

    def interrupt(signum, _frame):
        raise TimeoutError('bounded validation interrupted: ' + str(signum))

    try:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            previous[signum] = signal.signal(signum, interrupt)
        payload = {'parent': _process(os.getpid()), 'children': children,
                   'output': str(output), 'deadline': time.time() + seconds}
        watchdog = subprocess.Popen([sys.executable, '-c',
            'import json,sys; from corpus_pipeline.cluster.optimization_probe import _resume_watchdog; '
            '_resume_watchdog(json.loads(sys.argv[1]))', json.dumps(payload)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        ready_deadline = time.monotonic() + 5
        while not (output / 'WATCHDOG_READY.json').is_file():
            if watchdog.poll() is not None or time.monotonic() >= ready_deadline:
                raise RuntimeError('Independent production resume watchdog did not start')
            time.sleep(.05)
        signal.alarm(seconds)
        for child in children:
            if not _same_process(child):
                raise RuntimeError('Production child changed before pause')
            os.kill(child['pid'], signal.SIGSTOP)
            paused.append(child)
        atomic_json(output / 'PRODUCTION_PAUSE.json', {
            'notebook': name, 'hostname': socket.gethostname(), 'started_at': time.time(),
            'max_seconds': seconds,
            'children': [{k: c[k] for k in ('pid', 'start_ticks', 'boot_id', 'command_sha256')} for c in paused]})
        yield
    finally:
        resumed = []
        for child in paused:
            if _same_process(child):
                try:
                    os.kill(child['pid'], signal.SIGCONT)
                    resumed.append(child['pid'])
                except ProcessLookupError:
                    pass
        signal.alarm(0)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        atomic_json(output / 'PRODUCTION_RESUME.json', {'at': time.time(), 'pids': resumed})
        if watchdog is not None:
            try:
                watchdog.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass  # Independent watchdog remains bounded by its deadline.


def compare_outputs(left, right):
    """Exact policy/record coverage plus tolerant scores and vector comparison."""
    import pyarrow.parquet as pq
    a = pq.read_table(left / 'records.parquet').to_pylist()
    b = pq.read_table(right / 'records.parquet').to_pylist()
    if a != b:
        raise AssertionError('Original record bindings or policy outcomes changed')
    aa = pq.read_table(left / 'images.parquet').to_pylist()
    bb = pq.read_table(right / 'images.parquet').to_pylist()
    if len(aa) != len(bb):
        raise AssertionError('Image coverage changed')
    va = np.fromfile(left / 'sscd.f32', dtype='<f4').reshape(-1, 512)
    vb = np.fromfile(right / 'sscd.f32', dtype='<f4').reshape(-1, 512)
    errors = {'cf384_prob': 0., 'unifd_prob': 0.}
    cosine = 1.
    for x, y in zip(aa, bb):
        for key in x:
            if key in errors:
                if (x[key] is None) != (y[key] is None):
                    raise AssertionError('Detector score missingness changed')
                if x[key] is not None:
                    if not np.isfinite([x[key], y[key]]).all():
                        raise AssertionError('Nonfinite detector score')
                    errors[key] = max(errors[key], abs(x[key] - y[key]))
            elif key != 'sscd_byte_offset' and x[key] != y[key]:
                raise AssertionError('Image binding, provenance or policy changed: ' + key)
        if (x['sscd_byte_offset'] is None) != (y['sscd_byte_offset'] is None):
            raise AssertionError('SSCD coverage changed')
        if x['sscd_byte_offset'] is not None:
            vx, vy = va[x['sscd_byte_offset'] // 2048], vb[y['sscd_byte_offset'] // 2048]
            if not np.isfinite(vx).all() or not np.isfinite(vy).all():
                raise AssertionError('Nonfinite descriptor')
            value = float(np.dot(vx.astype('float64'), vy) / (np.linalg.norm(vx) * np.linalg.norm(vy)))
            if not np.isfinite(value):
                raise AssertionError('Invalid descriptor norm/cosine')
            cosine = min(cosine, value)
    if max(errors.values()) > .001 or cosine < .999:
        raise AssertionError('Numerical equivalence exceeded qualification tolerance')
    return {'records': len(a), 'image_occurrences': len(aa), 'record_outcomes_equal': True,
            'image_metadata_and_outcomes_equal': True, 'max_probability_error': errors,
            'minimum_sscd_cosine': cosine, 'threshold_decision_disagreements': 0}


def cases(root, snap):
    result = []
    for task_id, bound in (('s00007-r00000000', None), ('s00016-r00000000', 1024)):
        state = json.loads((root / 'states' / (task_id + '.json')).read_text())
        task = copy.deepcopy(state['task'])
        if bound:
            task['row_end'] = min(task['row_end'], task['row_start'] + bound)
        result.append((task_id, task))
    for source in ('chartqa', 'clevr_math', 'docvqa', 'iam', 'coco_caption'):
        path, entry = next((p, e) for p, e in snap['catalog_index'].items() if e['source'] == source)
        stat = Path(path).stat()
        result.append(('diverse-' + source, {'parquet_path': path, 'row_start': 0,
            'row_end': min(16, entry['rows']), 'source': source, 'source_shard': entry['source_shard'],
            'parquet_fingerprint': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}))
    return result


def run(root, output, device):
    from .inference import ResidentInferenceWorker
    from .optimized_inference import OptimizedResidentInferenceWorker
    from .frontpass import BASE, load_rules, process_task
    from corpus_pipeline.production.records import snapshot
    from corpus_pipeline.production.potential import compile_source_contracts
    from corpus_pipeline.production.ingest import iter_records
    plan = json.loads((root / 'plan.json').read_text())
    snap = snapshot(plan['snapshot_root'])
    rules = load_rules()
    contracts = compile_source_contracts(json.loads((BASE / 'prepared/config/layout_source_contracts.json').read_text()))
    thresholds = json.loads((BASE / 'detector_calibration/thresholds.json').read_text())['review_thresholds']
    selected = cases(root, snap)
    atomic_json(output / 'CASES.json', selected)
    worker_args = {'device': device, 'dtype': 'float32', 'batch_size': 32, 'threads': 4}
    results = []
    with ResidentInferenceWorker(**worker_args) as baseline, OptimizedResidentInferenceWorker(**worker_args) as optimized:
        # Warm the same full image batch for both before timed tasks.
        manifest = {'schema_version': 'production.input.v2', 'collection': 'dcvlm',
            'snapshot_root': snap['root'], 'snapshot_fingerprint': snap['fingerprint'],
            'shards': [{**selected[1][1], 'row_stop': selected[1][1]['row_start'] + 32}]}
        inputs = [{'sha256': ref['sha256'], 'original_ref': ref}
                  for record in iter_records(manifest, snap, rules)
                  for ref in record['source_image_refs']][:32]
        for worker in (baseline, optimized):
            predictions = worker.infer_batch(inputs)
            if any(p['status'] not in ('ok', 'too_small') for p in predictions):
                raise RuntimeError('Validation warmup failed')
        for repetition in range(2):
            for label, task in selected:
                if repetition and label.startswith('diverse-'):
                    continue
                outputs, timings = {}, {}
                # Reverse order on second repetition to limit cache/order bias.
                order = [('baseline', baseline, None), ('optimized', optimized, AGGREGATION)]
                if repetition:
                    order.reverse()
                for mode, worker, aggregation in order:
                    claim = {'task_id': f'{label}-r{repetition}', 'token': mode, 'task': task}
                    summary, _ = process_task(output, claim, worker, snap, rules, contracts, thresholds,
                                              aggregation=aggregation)
                    outputs[mode] = output / 'artifacts' / claim['task_id'] / mode
                    timings[mode] = {'elapsed_seconds': summary['elapsed_seconds'],
                        'counts': summary['counts'], 'performance': summary.get('performance')}
                comparison = compare_outputs(outputs['baseline'], outputs['optimized'])
                row = {'case': label, 'source': task['source'], 'repetition': repetition,
                       'comparison': comparison, 'timings': timings,
                       'speedup': timings['baseline']['elapsed_seconds'] / timings['optimized']['elapsed_seconds']}
                results.append(row)
                atomic_json(output / 'PARTIAL.json', results)
                print(json.dumps({k: row[k] for k in ('case', 'repetition', 'speedup', 'comparison')}), flush=True)
        base_identity, execution = baseline.identity, optimized.execution_identity
    receipt = {'success': True, 'dtype': 'float32', 'batch_size': 32, 'threads': 4,
        **{k: execution[k] for k in ('base_inference_code_sha256', 'optimized_inference_code_sha256')},
        'frontpass_code_sha256': file_sha(Path(__file__).with_name('frontpass.py')),
        'aggregation': AGGREGATION, 'results': results, 'base_model_identity': base_identity,
        'validation_code_sha256': file_sha(__file__), 'validated_at': time.time(),
        'scope': 'bounded_real_dcvlm_numerical_and_throughput_comparison_not_AI_accuracy',
        'api_calls': 0, 'production_queue_mutations': 0, 'training_release': False}
    atomic_json(output / 'COMPLETE.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--pause-production-name', required=True)
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with pause_production(args.root, args.pause_production_name, args.output):
        result = run(args.root, args.output, args.device)
    if (args.output / 'WATCHDOG_INTERVENTION.json').exists():
        result['success'] = False
        result['interrupted_by_resume_watchdog'] = True
        atomic_json(args.output / 'COMPLETE.json', result)
        raise RuntimeError('Qualification invalid because the production resume watchdog intervened')
    print(json.dumps({'success': result['success'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
