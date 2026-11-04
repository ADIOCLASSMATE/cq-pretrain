"""Persistent local CPU supervision for the manifest-bound baseline index.

Run from an explicitly selected (preferably frozen) repository. This module owns
only its local index workers; it never calls the download fleet or platform API.
The index engine independently holds a POSIX lockf lock for each worker slot.
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
import sys
import tempfile
import time

MODULE = 'corpus_pipeline.unified.baseline_index'
SCHEMA = 'corpus.baseline_index.supervisor.v1'
IDENTITY_KEYS = ('pid', 'host', 'boot_id', 'start_ticks', 'argv', 'cwd')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive(path):
    """POSIX record lock; unlike flock this also fences the shared filesystem."""
    with Path(path).open('a+') as stream:
        fcntl.lockf(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def process_identity(pid):
    try:
        if type(pid) is not int or pid <= 0:
            return None
        base = Path('/proc') / str(pid)
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': pid, 'host': socket.gethostname(),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'start_ticks': int(fields[19]),
                'argv': [os.fsdecode(x) for x in (base / 'cmdline').read_bytes().split(b'\0') if x],
                'cwd': str((base / 'cwd').resolve(strict=True))}
    except (OSError, ValueError, IndexError):
        return None


def same_process(identity, command, repository):
    if not isinstance(identity, dict) or identity.get('argv') != command or identity.get('cwd') != repository:
        return False
    current = process_identity(identity.get('pid'))
    return current is not None and all(current.get(k) == identity.get(k) for k in IDENTITY_KEYS)


def matching_processes(command, repository):
    """Recover the Popen-to-ledger crash window, restricted to this exact run."""
    found = []
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            identity = process_identity(int(path.name))
            if identity and identity['argv'] == command and identity['cwd'] == repository:
                found.append(identity)
    return found


def signal_owned(identity, command, repository, number):
    # Signal the bound PID only, never a process group or any downloader process.
    if not same_process(identity, command, repository):
        return False
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            fd = os.pidfd_open(identity['pid'])
            try:
                if not same_process(identity, command, repository):
                    return False
                signal.pidfd_send_signal(fd, number)
            finally:
                os.close(fd)
        else:
            os.kill(identity['pid'], number)
        return True
    except ProcessLookupError:
        return False


def command_for(config, action, slot=None):
    argv = [config['python'], '-m', MODULE, action,
            '--download-control', config['download_control'], '--output', config['output']]
    if action == 'worker':
        argv += ['--worker', str(slot), '--watch']
    return argv


def child_environment(config):
    env = dict(os.environ)
    env.update(PYTHONPATH=config['repository'], PYTHONUNBUFFERED='1',
               CORPUS_DATASET_ROOTS=config['source_roots'], OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    return env


def checked_config(args):
    config = vars(args).copy()
    config.pop('command', None)
    for key in ('repository', 'download_control', 'output', 'source_roots', 'control'):
        if key in ('source_roots', 'control') and not config.get(key):
            name = 'source_roots.json' if key == 'source_roots' else 'index_control'
            config[key] = str(Path(config['output']).parent / name)
        path = Path(config[key])
        if not path.is_absolute():
            raise ValueError('Absolute path required: ' + key)
        config[key] = str(path.resolve())
    # Keep the venv interpreter symlink, since resolving it changes its runtime.
    python = Path(config['python'])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('An explicit absolute executable --python is required')
    config['python'] = str(python)
    if not Path(config['source_roots']).is_file():
        raise ValueError('Source root mapping is missing')
    if not Path(config['download_control']).is_dir():
        raise ValueError('Download control directory is missing')
    for module in ('baseline_index.py', 'baseline_supervisor.py'):
        if not (Path(config['repository']) / 'corpus_pipeline/unified' / module).is_file():
            raise ValueError('Runtime module missing: ' + module)
    control, output, download = (Path(config[k]) for k in ('control', 'output', 'download_control'))
    if control.is_relative_to(output) or output.is_relative_to(control):
        raise ValueError('Supervisor control/logs must be outside the index output')
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in ((control, download), (output, download))):
        raise ValueError('Index state must not overlap download control')
    roots = read_json(config['source_roots'])
    if not isinstance(roots, dict) or not roots or any(
            not isinstance(root, str) or not Path(root).is_absolute() for root in roots.values()):
        raise ValueError('Source roots must map dataset IDs to absolute paths')
    for raw in roots.values():
        raw = Path(raw).resolve()
        if any(path.is_relative_to(raw) or raw.is_relative_to(path) for path in (control, output)):
            raise ValueError('Index state must not overlap raw dataset roots')
    for key, low, high in (('workers', 1, 16), ('poll_seconds', 1, 60),
                           ('status_seconds', 1, 300), ('stop_grace_seconds', 1, 300),
                           ('command_timeout_seconds', 1, 300)):
        if not low <= config[key] <= high:
            raise ValueError('Invalid supervisor limit: ' + key)
    return config


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.control = Path(config['control'])
        self.control.mkdir(parents=True, exist_ok=True)
        self.children = {}
        self.finalizer_child = None
        self.stop_requested = False
        self.state = read_json(self.control / 'STATE.json', {})
        self.state.setdefault('workers', {})
        self.state.setdefault('phase', 'starting')
        self.state.setdefault('last_index_status', None)
        self.state.setdefault('last_status_time', 0)
        self.stopping_since = None

    def bind(self):
        expected = Path(self.config['repository']) / 'corpus_pipeline/unified/baseline_supervisor.py'
        if expected.resolve() != Path(__file__).resolve():
            raise ValueError('Launch the supervisor from its explicitly configured repository')
        hashes = {}
        for name in ('baseline_index.py', 'baseline_supervisor.py'):
            path = Path(self.config['repository']) / 'corpus_pipeline/unified' / name
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        binding = {'schema': SCHEMA, 'config': self.config, 'host': socket.gethostname(),
                   'runtime_sha256': hashes,
                   'source_roots_sha256': hashlib.sha256(Path(self.config['source_roots']).read_bytes()).hexdigest()}
        existing = read_json(self.control / 'INPUT.json')
        if existing is not None and existing != binding:
            raise ValueError('Supervisor runtime/configuration changed; use a new control directory')
        if existing is None:
            atomic_json(self.control / 'INPUT.json', binding)
        self.state['supervisor'] = process_identity(os.getpid())
        # Always refresh authoritative engine status after restarting supervision.
        self.state['last_status_time'] = 0

    def save(self):
        self.state.update(schema=SCHEMA, time=time.time(), pid=os.getpid(), host=socket.gethostname(),
                          requested_workers=self.config['workers'])
        atomic_json(self.control / 'STATE.json', self.state)
        atomic_json(self.control / 'STATUS.json', self.state)

    def invoke(self, action):
        result = subprocess.run(command_for(self.config, action), cwd=self.config['repository'],
            env=child_environment(self.config), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, timeout=self.config['command_timeout_seconds'])
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            raise RuntimeError(f'{action} returned non-JSON output (exit {result.returncode})') from None
        if not isinstance(payload, dict):
            raise RuntimeError(action + ' returned a non-object status')
        return result.returncode, payload

    def refresh(self, now, force=False):
        if not force and now - self.state['last_status_time'] < self.config['status_seconds']:
            return True
        try:
            code, payload = self.invoke('status')
            if code:
                raise RuntimeError(f'index status failed (exit {code})')
            self.state.update(last_index_status=payload, last_status_time=now, last_status_error=None)
            return True
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self.state['last_status_error'] = str(exc)
            self.state['last_status_attempt'] = now
            return False

    def reconcile(self, slot, now):
        key = str(slot)
        row = self.state['workers'].setdefault(key, {'slot': slot, 'attempts': 0, 'failures': 0, 'next_start': 0})
        child = self.children.get(slot)
        if child is not None:
            code = child.poll()  # Reap our own exited child; adopted processes have no Popen handle.
            if code is not None:
                row['exit_code'] = code
                self.children.pop(slot)
        command = command_for(self.config, 'worker', slot)
        alive = same_process(row.get('identity'), command, self.config['repository'])
        if not alive:
            matches = matching_processes(command, self.config['repository'])
            if len(matches) > 1:
                row.update(phase='identity_conflict', matching_pids=[x['pid'] for x in matches])
                raise RuntimeError(f'Multiple exact index worker processes for slot {slot}; no signals sent')
            if matches:
                row.update(identity=matches[0], adopted_at=now, phase='running')
                alive = True
            elif row.get('phase') in ('running', 'launching', 'stopping'):
                runtime = now - row.get('started_at', now)
                failures = (0 if runtime >= 300 else row['failures']) + 1
                row.update(phase='backoff', failures=failures, exited_at=now,
                           next_start=now + min(300, 5 * 2 ** min(failures - 1, 6)))
        heartbeat = Path(self.config['output']) / 'state/workers' / f'{slot}.json'
        try:
            row['last_status'] = read_json(heartbeat)
        except (OSError, ValueError):
            row['heartbeat_error'] = 'Worker heartbeat is temporarily unreadable'
        heartbeat = row.get('last_status') or {}
        row['heartbeat_age_seconds'] = max(0, now - heartbeat['time']) if isinstance(heartbeat.get('time'), (int, float)) else None
        row['heartbeat_matches_process'] = alive and all(heartbeat.get(k) == row['identity'].get(k)
            for k in ('pid', 'host', 'boot_id', 'start_ticks'))
        row['alive'] = alive
        if alive:
            row['phase'] = 'running'
        return alive

    def launch(self, slot, now):
        row = self.state['workers'][str(slot)]
        command = command_for(self.config, 'worker', slot)
        # Durable intent plus exact argv recovery covers a supervisor crash during Popen.
        row.update(phase='launching', expected_argv=command, started_at=now, attempts=row['attempts'] + 1)
        self.save()
        log_path = self.control / 'logs' / f'worker-{slot}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=self.config['repository'], env=child_environment(self.config),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        self.children[slot] = child
        deadline = time.monotonic() + 2
        identity = None
        while child.poll() is None and time.monotonic() < deadline:
            candidate = process_identity(child.pid)
            if candidate and candidate['argv'] == command and candidate['cwd'] == self.config['repository']:
                identity = candidate
                break
            time.sleep(.01)
        row.update(identity=identity, alive=identity is not None, phase='running' if identity else 'launching', log=str(log_path))
        self.save()

    def drain(self, now):
        if self.stopping_since is None:
            self.stopping_since = now
        alive = False
        for slot in range(self.config['workers']):
            row = self.state['workers'][str(slot)]
            if not row['alive']:
                continue
            alive = True
            command = command_for(self.config, 'worker', slot)
            if not row.get('term_sent_at'):
                if signal_owned(row.get('identity'), command, self.config['repository'], signal.SIGTERM):
                    row['term_sent_at'] = now
            elif now - row['term_sent_at'] >= self.config['stop_grace_seconds']:
                if signal_owned(row.get('identity'), command, self.config['repository'], signal.SIGKILL):
                    row['kill_sent_at'] = now
        return not alive

    def observe_finalizer(self):
        row = self.state.get('finalizer', {})
        if not row:
            return False
        if self.finalizer_child is not None:
            code = self.finalizer_child.poll()
            if code is not None:
                row['exit_code'] = code
                self.finalizer_child = None
        command = command_for(self.config, 'finalize')
        alive = same_process(row.get('identity'), command, self.config['repository'])
        if not alive and row.get('phase') in ('running', 'launching'):
            matches = matching_processes(command, self.config['repository'])
            if len(matches) > 1:
                raise RuntimeError('Multiple exact finalizers; no signals sent')
            if matches:
                row['identity'] = matches[0]
                alive = True
        row['alive'] = alive
        return alive

    def finalize_step(self, now):
        row = self.state.setdefault('finalizer', {'attempts': 0, 'next_start': 0})
        if self.observe_finalizer():
            self.state['phase'] = 'finalizing'
            return False
        if row.get('phase') in ('running', 'launching'):
            try:
                result = read_json(self.control / 'finalize.stdout.json', {})
                receipt = read_json(Path(self.config['output']) / 'COMPLETE.json', {})
            except (OSError, ValueError):
                result, receipt = {}, {}
            valid = (isinstance(result, dict) and isinstance(receipt, dict)
                     and result == receipt and result.get('complete') is True
                     and result.get('validation_passed') is True)
            # An adopted child cannot be waitpid'd. Its complete JSON stdout is
            # written only after the engine has verified and published receipt.
            if row.get('exit_code') in (None, 0) and valid:
                self.state['phase'] = 'complete'
                row.update(phase='complete', finished_at=now)
                atomic_json(self.control / 'COMPLETE.json', {'schema': SCHEMA, 'time': now,
                    'output': self.config['output'], 'finalize': result})
                return True
            row.update(phase='backoff', next_start=now + 30, result=result, exited_at=now)
            self.state['phase'] = 'waiting_for_inputs' if row.get('exit_code') == 2 else 'finalize_failed'
            self.state['last_status_time'] = 0
            return False
        if now < row['next_start']:
            return False
        command = command_for(self.config, 'finalize')
        row.update(phase='launching', attempts=row['attempts'] + 1, expected_argv=command, started_at=now,
                   identity=None, exit_code=None)
        row.pop('term_sent_at', None)
        self.save()
        with (self.control / 'finalize.stdout.json').open('wb', buffering=0) as out, \
                (self.control / 'finalize.stderr.log').open('ab', buffering=0) as err:
            child = subprocess.Popen(command, cwd=self.config['repository'], env=child_environment(self.config),
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True, close_fds=True)
        self.finalizer_child = child
        deadline = time.monotonic() + 2
        while child.poll() is None and time.monotonic() < deadline:
            identity = process_identity(child.pid)
            if identity and identity['argv'] == command and identity['cwd'] == self.config['repository']:
                row['identity'] = identity
                break
            time.sleep(.01)
        row['phase'] = 'running'
        self.state['phase'] = 'finalizing'
        return False

    def step(self, now=None):
        now = time.time() if now is None else now
        previous = {slot: self.state['workers'].get(str(slot), {}).get('alive', False)
                    for slot in range(self.config['workers'])}
        alive = {slot: self.reconcile(slot, now) for slot in range(self.config['workers'])}
        finalizer_alive = self.observe_finalizer()
        if self.stop_requested or (self.control / 'STOP').exists():
            self.state['phase'] = 'stopping'
            drained = self.drain(now)
            if finalizer_alive:
                row = self.state['finalizer']
                number = signal.SIGTERM if not row.get('term_sent_at') else signal.SIGKILL
                if number == signal.SIGTERM or now - row['term_sent_at'] >= self.config['stop_grace_seconds']:
                    if signal_owned(row['identity'], command_for(self.config, 'finalize'),
                                    self.config['repository'], number):
                        row.setdefault('term_sent_at', now)
            if drained and not finalizer_alive:
                self.state['phase'] = 'stopped'
                self.save()
                return True
            self.save()
            return False
        if self.state.get('finalizer', {}).get('phase') in ('running', 'launching'):
            finished = self.finalize_step(now)
            self.save()
            return finished
        refreshed = self.refresh(now, force=any(previous[s] and not alive[s] for s in alive))
        index_status = self.state.get('last_index_status') or {}
        ready = refreshed and index_status.get('phase') in ('ready_to_finalize', 'complete', 'completed')
        if ready:
            self.state['phase'] = 'draining'
            if self.drain(now):
                finished = self.finalize_step(now)
                self.save()
                return finished
        elif refreshed:
            self.state['phase'] = 'indexing'
            for slot, running in alive.items():
                row = self.state['workers'][str(slot)]
                if not running and now >= row['next_start']:
                    row.pop('term_sent_at', None)
                    row.pop('kill_sent_at', None)
                    self.launch(slot, now)
        else:
            self.state['phase'] = 'status_unavailable'
        self.save()
        return False

    def run(self):
        with exclusive(self.control / 'supervisor.lock'):
            self.bind()
            previous_handlers = {}
            def stopping(number, frame):
                self.stop_requested = True
            for number in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[number] = signal.signal(number, stopping)
            try:
                while True:
                    try:
                        if self.step():
                            return 0
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        self.state.update(phase='supervisor_error', last_error=str(exc))
                        self.save()
                    # Short sleeps make SIGTERM responsive without busy polling.
                    until = time.monotonic() + (1 if self.stop_requested else self.config['poll_seconds'])
                    while time.monotonic() < until:
                        time.sleep(min(.25, max(0, until - time.monotonic())))
                        if self.stop_requested:
                            break
            finally:
                for number, handler in previous_handlers.items():
                    signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    for flag in ('download-control', 'output', 'repository', 'python'):
        run.add_argument('--' + flag, required=True)
    run.add_argument('--source-roots')
    run.add_argument('--control')
    run.add_argument('--workers', type=int, default=8)
    run.add_argument('--poll-seconds', type=int, default=15)
    run.add_argument('--status-seconds', type=int, default=60)
    run.add_argument('--stop-grace-seconds', type=int, default=60)
    run.add_argument('--command-timeout-seconds', type=int, default=120)
    for action in ('status', 'stop'):
        sub.add_parser(action).add_argument('--control', required=True)
    args = parser.parse_args(argv)
    if args.command == 'status':
        print(json.dumps(read_json(Path(args.control) / 'STATUS.json', {}), sort_keys=True))
        return 0
    if args.command == 'stop':
        control = Path(args.control)
        if not (control / 'INPUT.json').is_file():
            parser.error('No bound supervisor exists in this control directory')
        atomic_json(control / 'STOP', {'time': time.time(), 'requested_by_pid': os.getpid()})
        return 0
    try:
        return Supervisor(checked_config(args)).run()
    except (OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
