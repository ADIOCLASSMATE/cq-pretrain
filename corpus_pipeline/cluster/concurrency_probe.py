"""Bounded same-NPU process comparison on fixed real AS100M images.

Every scenario processes the same total image occurrences. Independent resident
children load, validate CPU inputs and warm up before a timestamp barrier. Only
the steady inference interval contributes to throughput. No production queue,
platform API, task leases or dataset contents are modified.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

from corpus_pipeline.production.common import atomic_json, file_sha


def split_indices(count, workers):
    if type(count) is not int or type(workers) is not int or count < workers or workers < 1:
        raise ValueError('nonempty disjoint worker assignments required')
    return [list(range(rank, count, workers)) for rank in range(workers)]


def child_parent_guard(expected_parent):
    """Linux child exits if its own benchmarking parent disappears."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'cannot install benchmark parent-death guard')
    if os.getppid() != expected_parent:
        raise RuntimeError('benchmark parent exited before child startup')


def host_snapshot(pids=(), *, include_npu=False):
    result = {'at': time.time(), 'monotonic': time.monotonic(), 'hostname': socket.gethostname()}
    for name in ('cpu.max', 'cpu.stat', 'memory.current', 'memory.max'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            result['cgroup_' + name] = path.read_text().strip()
    result['processes'] = {}
    for pid in pids:
        try:
            fields = dict(line.split(':', 1) for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines() if ':' in line)
            result['processes'][str(pid)] = {key: fields.get(key, '').strip()
                for key in ('State', 'Threads', 'VmRSS', 'VmHWM')}
        except FileNotFoundError:
            result['processes'][str(pid)] = {'exited': True}
    if include_npu:
        try:
            value = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=5)
            result['npu_smi'] = {'returncode': value.returncode, 'stdout': value.stdout[-32000:], 'stderr': value.stderr[-2000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result['npu_smi'] = {'error': type(exc).__name__}
    return result


def tensor_audit(worker, assets):
    from .inference import ResidentInferenceWorker, MODEL_NAMES
    rows = []
    for asset in assets:
        actual, tensors = worker._prepare(asset, MODEL_NAMES)
        expected, reference = ResidentInferenceWorker._prepare(worker, asset, MODEL_NAMES)
        actual.pop('decode_seconds', None); expected.pop('decode_seconds', None)
        if actual != expected or actual['status'] not in ('prepared', 'too_small'):
            raise AssertionError('fast/reference prepared metadata mismatch')
        hashes = {}
        if tensors is not None:
            for name in MODEL_NAMES:
                if not worker.torch.equal(tensors[name], reference[name]):
                    raise AssertionError('fast/reference input tensor is not bitwise equal: ' + name)
                hashes[name] = hashlib.sha256(tensors[name].numpy().tobytes()).hexdigest()
        rows.append({'sha256': asset['sha256'], 'status': actual['status'],
                     'pixel_sha256': actual.get('pixel_sha256'), 'tensor_sha256': hashes})
    return rows


def save_predictions(path, indices, assets, predictions):
    if len(indices) != len(predictions):
        raise AssertionError('worker output count differs from assignment')
    count = len(indices)
    vectors = np.full((count, 512), np.nan, dtype=np.float32)
    scores = np.full((count, 2), np.nan, dtype=np.float32)
    logits = np.full((count, 2), np.nan, dtype=np.float32)
    for row, (index, prediction) in enumerate(zip(indices, predictions)):
        if prediction.get('image_sha256') != assets[index]['sha256'] or prediction['status'] not in ('ok', 'too_small'):
            raise AssertionError('worker output error or reordered image identity')
        if prediction['status'] == 'ok':
            vectors[row] = prediction['sscd']['embedding']
            for column, name in enumerate(('cf384', 'unifd')):
                scores[row, column] = prediction['scores'][name]['prob']
                logits[row, column] = prediction['scores'][name]['logit']
            if not np.isfinite(vectors[row]).all() or not np.isfinite(scores[row]).all() or not np.isfinite(logits[row]).all():
                raise AssertionError('nonfinite model output')
    np.savez(path, indices=np.asarray(indices, dtype=np.int64),
        sha256=np.asarray([assets[i]['sha256'] for i in indices], dtype='U64'),
        status=np.asarray([p['status'] for p in predictions], dtype='U32'),
        pixel_sha256=np.asarray([p.get('pixel_sha256', '') for p in predictions], dtype='U64'),
        width=np.asarray([p['original_width'] for p in predictions], dtype=np.int32),
        height=np.asarray([p['original_height'] for p in predictions], dtype=np.int32),
        probabilities=scores, logits=logits, sscd=vectors)


def _child(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    child_parent_guard(spec['parent_pid'])
    output = Path(spec['output'])
    os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
    from .tuned_inference import TunedResidentInferenceWorker
    from .frontpass import stats_delta
    from copy import deepcopy
    assets = json.loads(Path(spec['samples']).read_text())
    assignments = spec['indices']
    started_loading = time.monotonic()
    try:
        with TunedResidentInferenceWorker(**spec['worker_config']) as worker:
            # Identical shared canary in all processes, independent of assignment.
            audit = tensor_audit(worker, assets[:spec['audit_images']])
            atomic_json(output / 'INPUT_AUDIT.json', audit)
            warmup = assets[:worker.batch_size]
            warm = worker.infer_batch(warmup)
            if any(p['status'] not in ('ok', 'too_small') for p in warm):
                raise AssertionError('model warmup failed')
            worker.synchronize()
            worker.torch.npu.reset_peak_memory_stats(worker.device)
            atomic_json(output / 'READY.json', {
                'pid': os.getpid(), 'load_audit_warmup_seconds': time.monotonic() - started_loading,
                'model_identity_sha256': worker.identity_sha256,
                'model_identity': worker.identity, 'execution_identity': worker.execution_identity,
                'input_audit_sha256': file_sha(output / 'INPUT_AUDIT.json'),
                'memory_allocated_bytes': worker.torch.npu.memory_allocated(worker.device),
                'memory_reserved_bytes': worker.torch.npu.memory_reserved(worker.device)})
            barrier_path = Path(spec['barrier'])
            ready_deadline = time.monotonic() + spec['barrier_timeout']
            while not barrier_path.is_file():
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError('parent did not release inference barrier')
                time.sleep(.02)
            barrier = json.loads(barrier_path.read_text())
            target = barrier['start_monotonic']
            while time.monotonic() < target:
                time.sleep(min(.01, max(0., target - time.monotonic())))
            baseline = deepcopy(worker.stats)
            cpu_started, started = time.process_time(), time.monotonic()
            predictions = list(worker.iter_results([assets[index] for index in assignments]))
            worker.synchronize()
            ended, cpu_ended = time.monotonic(), time.process_time()
            save_predictions(output / 'PREDICTIONS.npz', assignments, assets, predictions)
            result = {'success': True, 'rank': spec['rank'], 'pid': os.getpid(),
                'images': len(assignments), 'model_inferred_images': sum(p['status'] == 'ok' for p in predictions),
                'started_monotonic': started, 'ended_monotonic': ended,
                'barrier_start_lag_seconds': started - target, 'elapsed_seconds': ended - started,
                'process_cpu_seconds': cpu_ended - cpu_started,
                'memory_allocated_bytes': worker.torch.npu.memory_allocated(worker.device),
                'memory_reserved_bytes': worker.torch.npu.memory_reserved(worker.device),
                'peak_memory_allocated_bytes': worker.torch.npu.max_memory_allocated(worker.device),
                'peak_memory_reserved_bytes': worker.torch.npu.max_memory_reserved(worker.device),
                'performance': stats_delta(baseline, worker.stats),
                'predictions_sha256': file_sha(output / 'PREDICTIONS.npz')}
            atomic_json(output / 'DONE.json', result)
    except BaseException as exc:
        atomic_json(output / 'ERROR.json', {'success': False, 'type': type(exc).__name__, 'message': str(exc)[:1000]})
        traceback.print_exc()
        raise


def _stop_children(children, grace=2.):
    """Only process groups created by this diagnostic can be signaled."""
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace
    for child in children:
        try:
            child.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for child in children:
        child.wait(timeout=5)


def _wait_stage(children, directories, filename, deadline, *, require_exit=False, on_poll=None):
    while True:
        for child, directory in zip(children, directories):
            if (directory / 'ERROR.json').is_file():
                raise RuntimeError('benchmark child reported failure: ' + str(directory))
            if child.poll() is not None and (child.returncode != 0 or not (directory / filename).is_file()):
                raise RuntimeError('benchmark child exited before successful ' + filename + ': ' + str(child.returncode))
        if all((directory / filename).is_file() for directory in directories):
            if not require_exit or all(child.poll() == 0 for child in children):
                return [json.loads((directory / filename).read_text()) for directory in directories]
        if time.monotonic() >= deadline:
            raise TimeoutError('benchmark stage timed out: ' + filename)
        if on_poll:
            on_poll()
        time.sleep(.05)


def run_scenario(output, samples_path, count, workers, worker_config, *, startup_timeout=300.,
                 run_timeout=300., audit_images=16, command_builder=None, sample_resources=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assignments = split_indices(count, workers)
    children, files, directories = [], [], []
    barrier = output / 'START.json'
    resource_rows, last_sample = [], -float('inf')
    def resources():
        nonlocal last_sample
        if sample_resources and time.monotonic() - last_sample >= 2.:
            last_sample = time.monotonic()
            resource_rows.append(host_snapshot([child.pid for child in children], include_npu=True))
            atomic_json(output / 'RESOURCES.json', resource_rows)
    try:
        for rank, indices in enumerate(assignments):
            directory = output / f'worker-{rank:02d}'
            directory.mkdir()
            directories.append(directory)
            spec = {'parent_pid': os.getpid(), 'rank': rank, 'output': str(directory.resolve()),
                'samples': str(Path(samples_path).resolve()), 'indices': indices,
                'barrier': str(barrier.resolve()), 'barrier_timeout': startup_timeout + run_timeout,
                'audit_images': audit_images, 'worker_config': worker_config}
            atomic_json(directory / 'SPEC.json', spec)
            command = (command_builder(directory / 'SPEC.json') if command_builder else
                [sys.executable, '-m', 'corpus_pipeline.cluster.concurrency_probe', '--child-spec', str(directory / 'SPEC.json')])
            stream = (directory / 'process.log').open('wb')
            files.append(stream)
            children.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True))
        atomic_json(output / 'CHILDREN.json', {'pids': [child.pid for child in children]})
        ready = _wait_stage(children, directories, 'READY.json', time.monotonic() + startup_timeout, on_poll=resources)
        if len({value['model_identity_sha256'] for value in ready}) != 1:
            raise AssertionError('resident workers have differing model identities')
        audits = [json.loads((directory / 'INPUT_AUDIT.json').read_text()) for directory in directories]
        if any(value != audits[0] for value in audits[1:]):
            raise AssertionError('shared canary CPU input tensors differ across worker processes')
        start = time.monotonic() + .5
        atomic_json(barrier, {'start_monotonic': start, 'workers': workers, 'total_images': count})
        done = _wait_stage(children, directories, 'DONE.json', start + run_timeout,
                           require_exit=True, on_poll=resources)
        if (any(value.get('success') is not True for value in done)
                or sum(value['images'] for value in done) != count
                or any(not 0 <= value['barrier_start_lag_seconds'] < 5 for value in done)):
            raise AssertionError('worker completion/coverage/barrier invariant failed')
        ended = max(value['ended_monotonic'] for value in done)
        elapsed = ended - start
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise AssertionError('invalid steady benchmark duration')
        result = {'success': True, 'workers': workers, 'total_images': count,
            'worker_config': worker_config, 'steady_seconds': elapsed,
            'aggregate_images_per_second': count / elapsed,
            'total_process_cpu_seconds': sum(value['process_cpu_seconds'] for value in done),
            'worker_results': done, 'worker_ready': ready,
            'timing_scope': 'barrier_release_to_last_device_synchronize_excludes_model_load_audit_warmup_and_output_write',
            'memory_scope': 'per_process_allocator_peaks_plus_sampled_shared_device_npu_smi',
            'production_queue_mutations': 0, 'api_calls': 0}
        atomic_json(output / 'SCENARIO.json', result)
        return result
    finally:
        _stop_children(children)
        for stream in files:
            stream.close()
        atomic_json(output / 'CLEANUP.json', {'at': time.time(), 'children': [
            {'pid': child.pid, 'returncode': child.returncode} for child in children]})


def load_predictions(scenario, count):
    parts = []
    for path in sorted(Path(scenario).glob('worker-*/PREDICTIONS.npz')):
        with np.load(path, allow_pickle=False) as value:
            parts.append({key: value[key].copy() for key in value.files})
    if not parts:
        raise AssertionError('missing prediction evidence')
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(result['indices'])
    result = {key: value[order] for key, value in result.items()}
    if not np.array_equal(result['indices'], np.arange(count)):
        raise AssertionError('prediction evidence has missing or duplicate original occurrences')
    return result


def compare_predictions(reference, actual, thresholds=(.9, .95)):
    for key in ('indices', 'sha256', 'status', 'pixel_sha256', 'width', 'height'):
        if not np.array_equal(reference[key], actual[key]):
            raise AssertionError('prediction metadata changed: ' + key)
    mask = reference['status'] == 'ok'
    if not np.all(np.isin(reference['status'], ['ok', 'too_small'])):
        raise AssertionError('unexpected image status in reference')
    count = int(mask.sum())
    score_error, logit_error, min_cosine = 0., 0., 1.
    if count:
        expected, observed = reference['probabilities'][mask], actual['probabilities'][mask]
        a, b = reference['sscd'][mask].astype(np.float64), actual['sscd'][mask].astype(np.float64)
        for value in (expected, observed, a, b, reference['logits'][mask], actual['logits'][mask]):
            if not np.isfinite(value).all():
                raise AssertionError('nonfinite numerical evidence')
        score_error = float(np.max(np.abs(expected - observed)))
        logit_error = float(np.max(np.abs(reference['logits'][mask] - actual['logits'][mask])))
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        if (norms < 1e-8).any():
            raise AssertionError('invalid SSCD descriptor norm')
        min_cosine = float(np.min(np.sum(a * b, axis=1) / norms))
        if score_error > .001 or min_cosine < .999:
            raise AssertionError('same-card concurrency numerical equivalence failed')
        if not np.array_equal((expected >= thresholds).any(axis=1), (observed >= thresholds).any(axis=1)):
            raise AssertionError('same-card concurrency changed detector threshold decisions')
    return {'images': len(mask), 'model_inferred_images': count,
            'maximum_probability_error': score_error, 'maximum_logit_error': logit_error,
            'minimum_sscd_cosine': min_cosine, 'threshold_decision_disagreements': 0,
            'pixel_and_original_occurrence_metadata_equal': True}


def run(root, output, *, count=1024, worker_counts=(1, 2, 3), batch_size=64,
        decode_workers=4, cpu_threads=1, prefetch_batches=2, startup_timeout=300., run_timeout=300.):
    if type(count) is not int or not 64 <= count <= 4096 or count % 2:
        raise ValueError('count must be even in [64,4096]')
    if not worker_counts or worker_counts[0] != 1 or len(set(worker_counts)) != len(worker_counts) or any(n not in (1, 2, 3) for n in worker_counts):
        raise ValueError('worker_counts must begin with baseline1 and uniquely select1/2/3')
    if startup_timeout <= 0 or run_timeout <= 0:
        raise ValueError('positive bounded timeouts required')
    from .prepare_profile import sample_assets
    from .tuned_inference import TunedResidentInferenceWorker
    from corpus_pipeline.production.records import snapshot
    TunedResidentInferenceWorker._validate_configuration(batch_size, decode_workers, cpu_threads, prefetch_batches)
    plan = json.loads((root / 'plan.json').read_text())
    snap = snapshot(plan['snapshot_root'])
    if snap['fingerprint'] != plan['snapshot_fingerprint']:
        raise AssertionError('bound DCVLM snapshot changed')
    assets = sample_assets(root, count)
    if len(assets) != count:
        raise AssertionError('real AS100M sample coverage mismatch')
    samples = output / 'SAMPLES.json'
    atomic_json(samples, assets)
    config = {'device': 'npu:0', 'dtype': 'float32', 'batch_size': batch_size,
              'decode_workers': decode_workers, 'cpu_threads': cpu_threads,
              'prefetch_batches': prefetch_batches, 'fast_prepare': True}
    thresholds_path = Path(__file__).resolve().parents[1] / 'detector_calibration/thresholds.json'
    threshold_values = json.loads(thresholds_path.read_text())['review_thresholds']
    thresholds = [threshold_values['cf384'], threshold_values['unifd']]
    results, baseline = [], None
    for workers in worker_counts:
        directory = output / f'processes-{workers}'
        result = run_scenario(directory, samples, count, workers, config,
                              startup_timeout=startup_timeout, run_timeout=run_timeout)
        predictions = load_predictions(directory, count)
        if baseline is None:
            baseline = predictions
        result['comparison_to_one_process'] = compare_predictions(baseline, predictions, thresholds)
        if results:
            for key in ('input_audit_sha256', 'model_identity_sha256'):
                if result['worker_ready'][0][key] != results[0]['worker_ready'][0][key]:
                    raise AssertionError('model or shared canary input audit changed between scenarios: ' + key)
        results.append(result)
        atomic_json(output / 'PARTIAL.json', results)
        print(json.dumps({'workers': workers, 'images': count,
                          'aggregate_images_per_second': result['aggregate_images_per_second'],
                          'comparison': result['comparison_to_one_process']}), flush=True)
    receipt = {'success': True, 'hostname': socket.gethostname(), 'results': results,
        'fixed_images_per_scenario': count, 'samples_sha256': file_sha(samples),
        'thresholds_sha256': file_sha(thresholds_path),
        'code_sha256': {name: file_sha(Path(__file__).with_name(name)) for name in
            ('inference.py', 'optimized_inference.py', 'tuned_inference.py', 'fast_prepare.py', 'concurrency_probe.py')},
        'scope': 'bounded_real_as100m_same_npu_process_concurrency_not_whole_corpus_throughput',
        'model_identity_unchanged': True, 'cold_cache_claimed': False,
        'production_queue_mutations': 0, 'api_calls': 0, 'training_release': False}
    atomic_json(output / 'COMPLETE.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child-spec', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--count', type=int, default=1024)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--decode-workers', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--prefetch-batches', type=int, default=2)
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--run-timeout', type=float, default=300)
    args = parser.parse_args()
    if args.child_spec:
        _child(args.child_spec)
        return
    if args.root is None or args.output is None:
        parser.error('--root and --output are required')
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    def interrupt(signum, _frame):
        raise InterruptedError('benchmark interrupted by signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    try:
        run(args.root, args.output, count=args.count, worker_counts=tuple(args.workers),
            batch_size=args.batch_size, decode_workers=args.decode_workers, cpu_threads=args.cpu_threads,
            prefetch_batches=args.prefetch_batches, startup_timeout=args.startup_timeout, run_timeout=args.run_timeout)
    except BaseException as exc:
        atomic_json(args.output / 'ERROR.json', {'success': False, 'type': type(exc).__name__, 'message': str(exc)[:2000]})
        raise


if __name__ == '__main__':
    main()
