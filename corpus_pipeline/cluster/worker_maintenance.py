"""Stop one identity-verified local Notebook worker for a qualified rollout.

Does not stop its Notebook or alter the queue. The control host may retry the
returned leases only after this receipt proves their old processes have exited.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import time

from .notebook_worker import _location, _read_attempt, _identity_matches, _process, _node_command
from corpus_pipeline.production.common import atomic_json, digest

PROC = Path('/proc')


def validate_intent(intent, owner, root, name):
    if (intent.get('root') != str(root) or intent.get('name') != name
            or intent.get('hostname') != socket.gethostname()
            or owner.get('hostname') != socket.gethostname()
            or owner.get('attempt') != intent.get('attempt')
            or type(intent.get('devices')) is not int or not 1 <= intent['devices'] <= 16
            or digest(intent['supervisor_command']) != intent.get('supervisor_command_sha256')
            or digest(intent['node_command']) != intent.get('node_command_sha256')
            or intent['node_command'] != _node_command(root, intent['devices'])):
        raise RuntimeError('Accepted attempt identity or command contract changed')


def lease_belongs(lease, child, hostname):
    """Match the complete frontpass worker ID grammar, never a PID substring."""
    argv = child['argv']
    device = argv[argv.index('--device') + 1].replace(':', '_')
    pattern = re.escape(f'{hostname}-{device}-{child["pid"]}-') + r'[0-9a-f]{8}'
    return re.fullmatch(pattern, lease.get('worker_id', '')) is not None


def same(identity):
    actual = _process(identity['pid'])
    return actual is not None and all(actual[k] == identity[k] for k in (
        'pid', 'start_ticks', 'boot_id', 'command_sha256'))


def replacement_mode(root):
    """Validate the exact activation path the unchanged node launcher will use."""
    if os.path.lexists(Path(root) / 'TUNED_QUALIFIED.json'):
        from .tuned_configuration import configuration
        if configuration(root) is None:
            raise RuntimeError('Replacement tuned execution is not qualified')
        return 'tuned'
    from .frontpass import optimized_configuration
    if optimized_configuration(root, dtype='float32', batch_size=32, threads=4) is None:
        raise RuntimeError('Replacement optimized execution is not qualified')
    return 'optimized'


def stop(root, name, output):
    root = Path(root).resolve()
    # Never disrupt an existing worker until its replacement is qualified.
    replacement = replacement_mode(root)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _, base = _location(root, name)
    directory, intent = _read_attempt(base)
    if intent is None:
        raise RuntimeError('No accepted local worker attempt')
    owner = json.loads((directory / 'OWNER.json').read_text())
    validate_intent(intent, owner, root, name)
    if not _identity_matches(owner, intent['supervisor_command']):
        raise RuntimeError('Supervisor is not the accepted live process')
    supervisor = _process(owner['pid'])
    node = _process(owner['node_pid'])
    # npu_env.sh ends with exec <qualified python> "$@", so its bash/script
    # prefix is replaced by the supervisor's same Python executable.
    expected_node = [intent['supervisor_command'][0], *intent['node_command'][2:]]
    if supervisor is None or node is None or node['argv'] != expected_node:
        raise RuntimeError('Node worker identity is not live')
    node_fields = (PROC / str(node['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    if int(node_fields[1]) != supervisor['pid']:
        raise RuntimeError('Node worker no longer belongs to accepted supervisor')
    children = []
    devices = set()
    modules = set()
    expected_children = {}
    for index in range(intent['devices']):
        for module in ('corpus_pipeline.cluster.frontpass', 'corpus_pipeline.cluster.tuned_frontpass'):
            expected = [expected_node[0], '-m', module, 'worker',
                '--root', str(root), '--device', f'npu:{index}',
                '--dtype', 'float32', '--batch-size', '32', '--threads', '4']
            expected_children[tuple(expected)] = (index, module)
    for path in PROC.iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[1]) != node['pid']:
                continue
            child = _process(int(path.name))
            if child:
                match = expected_children.get(tuple(child['argv']))
                if match is None or match[0] in devices:
                    raise RuntimeError('Unexpected child under the accepted node worker')
                index, module = match
                devices.add(index); modules.add(module); children.append(child)
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(children) != intent['devices']:
        raise RuntimeError('Production child count differs from accepted resource contract')
    if len(modules) != 1:
        raise RuntimeError('Mixed worker entrypoints do not match one accepted node launch')
    identities = [supervisor, node, *children]
    receipt = {'name': name, 'root': str(root), 'hostname': socket.gethostname(),
        'attempt': intent['attempt'], 'intent_at': time.time(), 'reason': 'qualified_optimization_rollout',
        'current_worker_module': next(iter(modules)), 'replacement_mode': replacement,
        'processes': [{k: c[k] for k in ('pid', 'start_ticks', 'boot_id', 'command_sha256')} for c in identities]}
    atomic_json(output / 'STOP_INTENT.json', receipt)
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Race-safe pidfd signaling is unavailable; no worker was stopped')
    pidfd = os.pidfd_open(supervisor['pid'], 0)
    try:
        if replacement_mode(root) != replacement:
            raise RuntimeError('Replacement execution changed before stop')
        if not all(same(child) for child in identities):
            raise RuntimeError('Accepted worker changed before stop')
        # This FD identifies the original process even if its numeric PID is
        # reused between the final identity check and signal dispatch.
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
    finally:
        os.close(pidfd)
    deadline = time.monotonic() + 110
    while any(same(child) for child in identities):
        if time.monotonic() > deadline:
            raise RuntimeError('Some old worker processes remain; queue retry not authorized')
        time.sleep(.2)
    leases = []
    for path in (root / 'claims').glob('*/lease.json'):
        try:
            lease = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        if any(lease_belongs(lease, child, socket.gethostname()) for child in children):
            leases.append({k: lease[k] for k in ('task_id', 'token', 'worker_id')})
    receipt.update(stopped_at=time.time(), all_original_processes_exited=True, retry_candidates=leases)
    atomic_json(output / 'STOPPED.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stop(args.root, args.name, args.output)), flush=True)


if __name__ == '__main__':
    main()
