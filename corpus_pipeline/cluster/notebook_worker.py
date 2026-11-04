"""Idempotent, detached offline worker launcher inside an existing Notebook.

This module never calls the platform or creates/stops compute resources. Run
``launch --root RUN --devices N --name NOTEBOOK`` in a temporary terminal.
``status --root RUN --name NOTEBOOK`` returns running/exited/unknown. Each
container holds a local /tmp flock for the complete supervisor/node lifetime;
the shared filesystem is used only for receipts, not cross-host locking.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

from corpus_pipeline.production.common import atomic_json, digest

MODULE = 'corpus_pipeline.cluster.notebook_worker'
SCHEMA = 'corpus_notebook_worker.v1'
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\Z')
_ATTEMPT = re.compile(r'[0-9a-f]{32}\Z')
REPO = Path(__file__).resolve().parents[2]


def _location(root, name):
    root = Path(root).resolve()
    hostname = socket.gethostname()
    if not root.is_dir() or not _NAME.fullmatch(name) or not _NAME.fullmatch(hostname):
        raise ValueError('existing run root and safe notebook/host names required')
    return root, root / 'notebooks' / name / hostname


def _lock_path(root):
    key = hashlib.sha256(str(root).encode()).hexdigest()
    return Path('/tmp') / f'corpus-notebook-worker-{os.getuid()}-{key}.lock'


def _open_lock(root):
    # Never unlink: replacing the inode would permit duplicate supervisors.
    return os.open(_lock_path(root), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)


def _acquire(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _process(pid):
    if type(pid) is not int or pid <= 0:
        return None
    try:
        path = Path('/proc') / str(pid)
        # Fields after the final ')' start at stat field 3. starttime is 22.
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        raw = (path / 'cmdline').read_bytes()
        argv = [item.decode(errors='surrogateescape') for item in raw.rstrip(b'\0').split(b'\0')]
        return {'pid': pid, 'start_ticks': int(fields[19]),
                'boot_id': _boot_id(), 'command_sha256': digest(argv), 'argv': argv}
    except (OSError, ValueError, IndexError):
        return None


def _identity_matches(owner, expected):
    actual = _process(owner.get('pid'))
    if actual is None or any(actual.get(key) != owner.get(key)
                             for key in ('pid', 'start_ticks', 'boot_id', 'command_sha256')):
        return False
    return (actual['argv'] == expected
            and len(expected) > 3 and expected[1:4] == ['-m', MODULE, '_supervise'])


def _read_attempt(base):
    try:
        current = json.loads((base / 'current.json').read_text())
        attempt = current['attempt']
        if not isinstance(attempt, str) or not _ATTEMPT.fullmatch(attempt):
            return None, None
        directory = base / attempt
        intent = json.loads((directory / 'LAUNCH.json').read_text())
        if intent.get('schema') != SCHEMA or intent.get('attempt') != attempt:
            return None, None
        return directory, intent
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def _status(root, name, *, lock_busy):
    root, base = _location(root, name)
    result = {'schema': SCHEMA, 'name': name, 'hostname': socket.gethostname(),
              'status': 'unknown', 'lock_busy': lock_busy}
    directory, intent = _read_attempt(base)
    if intent is None:
        return result
    result.update(attempt=intent['attempt'], devices=intent['devices'],
                  command_sha256=intent['supervisor_command_sha256'],
                  node_command_sha256=intent['node_command_sha256'])
    try:
        owner = json.loads((directory / 'OWNER.json').read_text())
    except (OSError, ValueError):
        owner = {}
    if _identity_matches(owner, intent['supervisor_command']):
        result.update(status='running', pid=owner['pid'],
                      phase='node_started' if owner.get('node_pid') else 'supervisor_started')
        return result
    try:
        exited = json.loads((directory / 'EXIT.json').read_text())
        if (exited.get('attempt') == intent['attempt']
                and exited.get('command_sha256') == intent['supervisor_command_sha256']):
            result.update(status='exited', returncode=exited['returncode'],
                          ended_at=exited['ended_at'], error_type=exited.get('error_type'))
    except (OSError, ValueError, KeyError):
        pass
    return result


def status(root, name):
    root, _ = _location(root, name)
    fd = _open_lock(root)
    try:
        return _status(root, name, lock_busy=not _acquire(fd))
    finally:
        os.close(fd)


def _node_command(root, devices):
    return ['bash', str(Path(__file__).with_name('npu_env.sh')), '-m',
            'corpus_pipeline.cluster.node_worker', '--root', str(root),
            '--devices', str(devices), '--dtype', 'float32', '--batch-size', '32', '--threads', '4']


def launch(root, devices, name):
    root, base = _location(root, name)
    if type(devices) is not int or not 1 <= devices <= 16:
        raise ValueError('devices must be an integer in 1..16')
    if not (root / 'QUALIFIED.json').is_file() or not (root / 'queue_rpc.json').is_file():
        raise ValueError('qualified inference receipt and queue configuration required')
    fd = _open_lock(root)
    try:
        if not _acquire(fd):
            return {**_status(root, name, lock_busy=True), 'launched': False}
        previous = _status(root, name, lock_busy=False)
        if previous['status'] == 'running':
            # Defensive: never duplicate a live verified supervisor even if a
            # future implementation accidentally fails to retain its lock.
            return {**previous, 'launched': False}
        attempt = uuid.uuid4().hex
        directory = base / attempt
        directory.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-m', MODULE, '_supervise', '--root', str(root),
                   '--devices', str(devices), '--name', name, '--attempt', attempt,
                   '--lock-fd', str(fd)]
        node_command = _node_command(root, devices)
        intent = {'schema': SCHEMA, 'attempt': attempt, 'root': str(root),
                  'name': name, 'hostname': socket.gethostname(), 'devices': devices,
                  'created_at': time.time(), 'supervisor_command': command,
                  'supervisor_command_sha256': digest(command), 'node_command': node_command,
                  'node_command_sha256': digest(node_command)}
        atomic_json(directory / 'LAUNCH.json', intent)
        atomic_json(base / 'current.json', {'schema': SCHEMA, 'attempt': attempt})
        try:
            with (directory / 'supervisor.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
            atomic_json(directory / 'SPAWN.json', {'pid': child.pid, 'at': time.time(),
                                                   'command_sha256': digest(command)})
        except Exception as error:
            atomic_json(directory / 'EXIT.json', {'attempt': attempt,
                'command_sha256': digest(command), 'returncode': 125, 'ended_at': time.time(),
                'error_type': type(error).__name__})
            raise
        # The child retains this same open-file-description lock. Closing our
        # descriptor below must NOT explicitly call LOCK_UN.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = _status(root, name, lock_busy=True)
            if result['status'] in {'running', 'exited'}:
                return {**result, 'launched': True}
            if child.poll() is not None:
                break
            time.sleep(.02)
        return {**_status(root, name, lock_busy=child.poll() is None), 'launched': True}
    finally:
        os.close(fd)


def supervise(root, devices, name, attempt, lock_fd):
    """Internal child: keep the lock, supervise node_worker, publish exit."""
    root, base = _location(root, name)
    if not _ATTEMPT.fullmatch(attempt):
        raise ValueError('invalid launch attempt')
    directory = base / attempt
    intent = json.loads((directory / 'LAUNCH.json').read_text())
    own = _process(os.getpid())
    lock_stat, fd_stat = _lock_path(root).stat(), os.fstat(lock_fd)
    if (lock_stat.st_dev != fd_stat.st_dev or lock_stat.st_ino != fd_stat.st_ino
            or not _acquire(lock_fd) or own is None
            or intent['root'] != str(root) or intent['name'] != name
            or intent['hostname'] != socket.gethostname() or intent['devices'] != devices
            or intent['attempt'] != attempt
            or own['argv'] != intent['supervisor_command']
            or digest(intent['supervisor_command']) != intent['supervisor_command_sha256']
            or intent['node_command'] != _node_command(root, devices)
            or digest(intent['node_command']) != intent['node_command_sha256']):
        raise RuntimeError('supervisor launch/lock identity mismatch')
    owner = {key: value for key, value in own.items() if key != 'argv'}
    owner.update(attempt=attempt, started_at=time.time(), hostname=socket.gethostname())
    child, stopping, termination_started = None, False, None
    def stop(*_):
        nonlocal stopping, termination_started
        stopping = True
        termination_started = termination_started or time.monotonic()
        if child is not None and child.poll() is None:
            child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    atomic_json(directory / 'OWNER.json', owner)
    code, error_type = 125, None
    try:
        with (directory / 'node_worker.log').open('ab', buffering=0) as log:
            # Node inherits the lock too: if this supervisor is SIGKILLed, a
            # surviving node still prevents a second launch in this container.
            child = subprocess.Popen(intent['node_command'], cwd=REPO,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock_fd,))
            owner['node_pid'] = child.pid
            atomic_json(directory / 'OWNER.json', owner)
            if stopping:
                stop()
            while child.poll() is None:
                if stopping and time.monotonic() - termination_started > 90:
                    os.killpg(child.pid, signal.SIGKILL)
                time.sleep(.2)
            code = child.wait()
            if stopping and code == 0:
                code = 143
    except Exception as error:
        error_type = type(error).__name__
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    finally:
        atomic_json(directory / 'EXIT.json', {'attempt': attempt,
            'command_sha256': intent['supervisor_command_sha256'], 'returncode': code,
            'ended_at': time.time(), 'error_type': error_type, 'termination_requested': stopping})
        os.close(lock_fd)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('launch', 'status', '_supervise'):
        child = commands.add_parser(command)
        child.add_argument('--root', type=Path, required=True)
        child.add_argument('--name', required=True)
        if command != 'status':
            child.add_argument('--devices', type=int, required=True)
        if command == '_supervise':
            child.add_argument('--attempt', required=True)
            child.add_argument('--lock-fd', type=int, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == '_supervise':
        return supervise(**args)
    result = launch(**args) if command == 'launch' else status(**args)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
