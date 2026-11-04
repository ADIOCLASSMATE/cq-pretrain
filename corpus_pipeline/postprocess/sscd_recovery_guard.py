"""Independent, bounded SSCD coordinator recovery and honest fleet telemetry.

This sidecar never edits frozen runtime/configuration or bypasses a failed recall
gate. A failed stage needs repair; only an unexpectedly missing coordinator is
automatically relaunched. The coordinator's existing monitor then resumes its
owned workers. No platform resources are created by this module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import uuid

VERSION = 'corpus.sscd_recovery_guard.v1'
MODULE = 'corpus_pipeline.postprocess.sscd_cluster_runner'
PARALLEL = {'catalog', 'build', 'search'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.lockf(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False; return
        try:
            yield True
        finally:
            fcntl.lockf(stream, fcntl.LOCK_UN)


def local_process(progress, pid, required_tokens):
    """No remote PID guesses and no signalling potentially reused PIDs."""
    if progress.get('host') != socket.gethostname():
        return 'foreign_host'
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if progress.get('boot_id') and progress['boot_id'] != boot:
        return 'dead'
    if type(pid) is not int or pid <= 0:
        return 'dead'
    try:
        base = Path('/proc') / str(pid)
        stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return 'dead'
        argv = (base / 'cmdline').read_bytes().decode(errors='replace').split('\0')
    except FileNotFoundError:
        return 'dead'
    except (PermissionError, OSError):
        return 'unknown'
    # Reused PIDs are conservatively held for manual inspection, never killed.
    return 'alive' if all(token in argv for token in required_tokens) else 'unknown'


def worker_counts(rows, phase, now, stale_seconds=180):
    counts = {'active_stage_workers': 0, 'idle_workers': 0, 'stale_workers': 0,
              'failed_workers': 0, 'observed_workers': len(rows)}
    for row in rows:
        if now - row.get('updated_at', 0) > stale_seconds:
            counts['stale_workers'] += 1
        elif row.get('fatal') or row.get('status') == 'failed':
            counts['failed_workers'] += 1
        elif (phase in PARALLEL and row.get('phase') == phase
              and row.get('status') == 'running' and row.get('child_pid')):
            counts['active_stage_workers'] += 1
        else:
            counts['idle_workers'] += 1
    counts['active_is_heartbeat_based_not_measured_throughput'] = True
    return counts


def classify(progress, *, now, process_status, child_status, stopped=False, stale_seconds=180):
    phase = progress.get('phase')
    if stopped or phase == 'stopped':
        return 'stopped', 'explicit_stop_preserved', False
    if phase == 'complete':
        return 'complete', 'controller_reports_complete', False
    error = str(progress.get('error', '')).lower()
    if phase == 'await_calibration' or (phase == 'failed' and 'recall gate not met' in error):
        return 'await_calibration', 'full_catalog_recall_gate_failed', False
    if phase == 'failed':
        return 'await_repair', 'explicit_stage_failure_not_retried', False
    if process_status == 'alive':
        if now - progress.get('updated_at', 0) > stale_seconds:
            return 'await_diagnosis', 'live_coordinator_heartbeat_stale', False
        return 'running', 'coordinator_live', False
    if process_status != 'dead':
        return 'await_diagnosis', 'coordinator_identity_not_verifiable', False
    if child_status != 'dead':
        return 'await_diagnosis', 'stage_child_may_still_be_running', False
    if now - progress.get('updated_at', 0) <= stale_seconds:
        return 'observing', 'wait_for_stale_heartbeat_before_recovery', False
    return 'recoverable', 'unexpected_coordinator_loss', True


class Guard:
    def __init__(self, config, *, max_attempts=3, cooldown_seconds=300, launcher=None):
        self.cp = Path(config).absolute(); self.c = read(self.cp)
        self.digest = sha(self.cp); self.root = Path(self.c['root']); self.controller = self.root / 'controller'
        self.d = self.root / 'recovery_guard' / self.digest[:16]
        if not 1 <= max_attempts <= 5 or cooldown_seconds < 60:
            raise ValueError('Recovery requires a bounded attempt count and cooldown')
        if self.c.get('coordinator_host') != socket.gethostname():
            raise ValueError('Guard must run on the explicitly configured coordinator host')
        self.maximum = max_attempts; self.cooldown = cooldown_seconds
        self.launcher = launcher or self.launch_coordinator
        self.stop = threading.Event()

    def validate(self):
        if sha(self.cp) != self.digest:
            raise ValueError('Cluster configuration changed')
        for relative, expected in self.c['runtime_sha256'].items():
            if sha(Path(self.c['runtime']) / relative) != expected:
                raise ValueError('Frozen runtime changed')
        bound = self.controller / 'INPUT.json'
        if not bound.exists() or read(bound).get('config_sha256') != self.digest:
            raise ValueError('Coordinator input binding missing or different')

    def launch_coordinator(self):
        # A direct process launch has no platform side effects. The existing
        # frozen coordinator checks its singleton lock and resource ownership.
        argv = [self.c['python'], '-m', MODULE, 'run', '--config', str(self.cp)]
        env = dict(os.environ, PYTHONPATH=self.c['runtime'], PYTHONUNBUFFERED='1')
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = str(self.c['threads'])
        with (self.d / 'coordinator.log').open('ab', buffering=0) as stream:
            child = subprocess.Popen(argv, cwd=self.c['runtime'], env=env,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                     start_new_session=True, close_fds=True)
        return {'submitted_pid': child.pid}

    def tick(self, now=None):
        now = time.time() if now is None else now
        with lock(self.d / 'TICK.lock') as acquired:
            if not acquired:
                return {'state': 'another_tick_active'}
            self.validate()
            progress = read(self.controller / 'PROGRESS.json')
            state_path = self.d / 'STATE.json'
            state = read(state_path) if state_path.exists() else {'attempts': {}}
            co = local_process(progress, progress.get('pid'), [MODULE, str(self.cp), 'run'])
            child = local_process(progress, progress.get('child_pid'), ['--output', self.c['output']])
            stopped = (self.controller / 'STOP').exists() or (self.d / 'STOP').exists()
            status, reason, recover = classify(progress, now=now, process_status=co,
                                               child_status=child, stopped=stopped)
            phase = progress.get('phase', 'unknown')
            rows = []
            for path in (self.controller / 'workers').glob('*.json'):
                row = read(path)
                if row.get('worker_id'):
                    rows.append(row)
            counts = worker_counts(rows, phase, now)
            value = {'schema': VERSION, 'updated_at': now, 'phase': phase, 'state': status,
                     'reason': reason, 'coordinator_process': co, 'stage_child_process': child,
                     'config_sha256': self.digest, 'workers': counts, 'actions': [],
                     'recall_thresholds_unchanged': True, 'platform_resources_created': 0,
                     'automatic_recovery_attempt_limit_per_phase': self.maximum}
            attempt = state['attempts'].get(phase, {'count': 0, 'last_at': 0})
            if recover:
                if attempt['count'] >= self.maximum:
                    value.update(state='await_repair', reason='bounded_recovery_attempts_exhausted')
                elif now - attempt['last_at'] < self.cooldown:
                    value.update(state='observing', reason='recovery_cooldown')
                else:
                    with lock(self.controller / 'COORDINATOR.lock') as available:
                        if not available:
                            value.update(state='observing', reason='coordinator_lock_still_held')
                        else:
                            # Persist before submission: an interrupted launch consumes
                            # an attempt instead of retrying without a bound.
                            attempt = {'count': attempt['count'] + 1, 'last_at': now}
                            state['attempts'][phase] = attempt; save(state_path, state)
                    if available:
                        receipt = self.launcher()
                        value['actions'].append(dict(action='relaunch_coordinator', **receipt))
                        value.update(state='recovery_requested', reason='checkpoint_resume_requested')
            value['attempts_for_phase'] = attempt['count']
            alarm = value['state'] in ('await_calibration', 'await_repair', 'await_diagnosis')
            if counts['failed_workers'] or (phase in PARALLEL and counts['active_stage_workers'] == 0):
                alarm = True
            save(self.d / 'ALARM.json', {'schema': VERSION, 'active': alarm, 'at': now,
                                        'state': value['state'], 'reason': value['reason'],
                                        'workers': counts, 'requires_external_notification_transport': True})
            save(self.d / 'PROGRESS.json', value)
            return value

    def run(self):
        with lock(self.d / 'RUN.lock') as acquired:
            if not acquired:
                return {'state': 'already_running'}
            signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
            signal.signal(signal.SIGINT, lambda *_: self.stop.set())
            while not self.stop.is_set():
                try:
                    result = self.tick()
                    if result['state'] in ('complete', 'stopped'):
                        return result
                except Exception as exc:
                    # Do not include arbitrary platform output, command text, or secrets.
                    save(self.d / 'ALARM.json', {'schema': VERSION, 'active': True, 'at': time.time(),
                                                'state': 'guard_error', 'error_type': type(exc).__name__})
                self.stop.wait(30)
            return {'state': 'guard_stopped'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('once', 'run', 'status'))
    parser.add_argument('--cluster-config', required=True)
    parser.add_argument('--max-attempts', type=int, default=3)
    args = parser.parse_args(argv); guard = Guard(args.cluster_config, max_attempts=args.max_attempts)
    result = guard.tick() if args.action == 'once' else guard.run() if args.action == 'run' else read(guard.d / 'PROGRESS.json')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
