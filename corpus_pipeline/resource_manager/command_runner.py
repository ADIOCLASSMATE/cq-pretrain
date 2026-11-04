"""Idempotent argv runner used inside an allocated container.

Launch and drain use independent supervisor processes and durable operation
IDs. Commands never pass through an implicit shell. Status proves process
identity with host, boot ID and Linux start ticks before reporting running.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import threading
import uuid


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _runtime_id():
    return socket.gethostname() + ':' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _identity(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if value[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': int(value[19]), 'runtime_id': _runtime_id()}
    except (OSError, ValueError, IndexError):
        return None


def _alive(identity):
    return bool(identity and _identity(identity.get('pid')) == identity)


def _argv(command):
    if (not isinstance(command, (list, tuple)) or not command
            or any(not isinstance(part, str) or '\x00' in part for part in command) or not command[0]):
        raise ValueError('Command must be a nonempty argv')
    return list(command)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,180}', value):
        raise ValueError('Invalid resource or operation identity')
    return value


def resource_directory(root, resource_key):
    return Path(root).resolve() / _name(resource_key)


def _current(directory, kind):
    pointer = directory / ('CURRENT_' + kind.upper() + '.json')
    if not pointer.exists():
        return None
    current = json.loads(pointer.read_text())
    return json.loads((directory / 'operations' / (_name(current['operation_id']) + '.json')).read_text())


def probe(command, timeout=10):
    try:
        return subprocess.run(_argv(command), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=timeout, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def status(root, resource_key, readiness=None):
    directory = resource_directory(root, resource_key)
    current, drained = _current(directory, 'launch'), _current(directory, 'drain')
    running = bool(current and (_alive(current.get('child')) or _alive(current.get('supervisor'))))
    completed = bool(current and current.get('phase') == 'completed')
    return {'running': running, 'ready': running and (probe(readiness) if readiness else True),
            'completed': completed, 'exit_code': current.get('exit_code') if completed else None,
            'launch_operation_id': current.get('operation_id') if current else None,
            'runtime_id': _runtime_id(),
            'process_gone': bool(current and current.get('phase') == 'running'
                                 and current.get('runtime_id') == _runtime_id()
                                 and current.get('supervisor') and current.get('child') and not running),
            'drained': bool(drained and drained.get('phase') == 'completed' and drained.get('exit_code') == 0),
            'drain_operation_id': drained.get('operation_id') if drained else None}


def launch(root, resource_key, operation_id, command, *, kind='launch'):
    operation_id = _name(operation_id)
    if kind not in ('launch', 'drain'):
        raise ValueError('Unsupported runner operation')
    command = _argv(command)
    directory = resource_directory(root, resource_key)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('.' + kind + '.submit.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / 'operations' / (operation_id + '.json')
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['command'] != command or existing['kind'] != kind:
                raise ValueError('Operation ID is already bound to a different command')
            return status(root, resource_key)
        previous = _current(directory, kind)
        replaced_dead_operation = None
        if previous and previous.get('phase') != 'completed':
            same_runtime = previous.get('runtime_id') == _runtime_id()
            previous_alive = (_alive(previous.get('child'))
                              or _alive(previous.get('supervisor')))
            proven_gone = bool(same_runtime and previous.get('phase') == 'running'
                               and previous.get('child') and previous.get('supervisor')
                               and not previous_alive)
            if previous_alive or (same_runtime and not proven_gone):
                raise RuntimeError('Previous operation is running or has an unknown result')
            if proven_gone:
                # A killed supervisor cannot seal its operation. Match the
                # process_gone status contract, but also require its runtime
                # lock to be free before admitting the manager's next ID.
                with (directory / ('.' + kind + '.runtime.lock')).open('a') as runtime_lock:
                    try:
                        fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise RuntimeError('Previous operation runtime lock is still held') from error
                replaced_dead_operation = previous['operation_id']
        value = {'operation_id': operation_id, 'kind': kind, 'command': command,
                 'phase': 'intent', 'runtime_id': _runtime_id(), 'created_at': time.time()}
        if replaced_dead_operation is not None:
            value['replaces_proven_dead_operation_id'] = replaced_dead_operation
        _save(path, value)
        _save(directory / ('CURRENT_' + kind.upper() + '.json'), {'operation_id': operation_id})
        # The supervisor owns the operation from here. An uncertain spawn is
        # observed by its operation ID and is never issued a second time.
        argv = [sys.executable, '-m', 'corpus_pipeline.resource_manager.command_runner',
                '--root', str(Path(root).resolve()), '--resource-key', resource_key,
                '_supervise', '--operation-id', operation_id]
        supervisor = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        # Long-lived callers reap their own child; the CLI may exit immediately
        # and the detached supervisor continues independently.
        threading.Thread(target=supervisor.wait, daemon=True).start()
        return status(root, resource_key)


def supervise(root, resource_key, operation_id):
    directory = resource_directory(root, resource_key)
    path = directory / 'operations' / (_name(operation_id) + '.json')
    value = json.loads(path.read_text())
    with (directory / ('.' + value['kind'] + '.runtime.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if value['phase'] != 'intent' or value['runtime_id'] != _runtime_id():
            raise RuntimeError('Supervisor intent is no longer eligible')
        value.update(phase='running', supervisor=_identity(os.getpid()), started_at=time.time())
        _save(path, value)
        try:
            with (directory / 'operations' / (operation_id + '.log')).open('ab', buffering=0) as log:
                process = subprocess.Popen(value['command'], stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=log, close_fds=True)
                value['child'] = _identity(process.pid)
                _save(path, value)
                result = process.wait()
        except Exception as error:
            result = 127
            value['error_type'] = type(error).__name__
        value.update(phase='completed', exit_code=result, completed_at=time.time())
        _save(path, value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--resource-key', required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('launch', 'drain'):
        child = sub.add_parser(action)
        child.add_argument('--operation-id', required=True)
        child.add_argument('command', nargs=argparse.REMAINDER)
    status_parser = sub.add_parser('status')
    status_parser.add_argument('--readiness', '--readiness-json')
    child = sub.add_parser('probe'); child.add_argument('command', nargs=argparse.REMAINDER)
    child = sub.add_parser('_supervise'); child.add_argument('--operation-id', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.action == '_supervise':
        supervise(args.root, args.resource_key, args.operation_id)
        return
    command = getattr(args, 'command', [])
    if command[:1] == ['--']:
        command = command[1:]
    if args.action in ('launch', 'drain'):
        result = launch(args.root, args.resource_key, args.operation_id, command, kind=args.action)
    elif args.action == 'status':
        result = status(args.root, args.resource_key,
                        json.loads(args.readiness) if args.readiness else None)
    else:
        result = status(args.root, args.resource_key) | {'probe_success': probe(command, timeout=30)}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
