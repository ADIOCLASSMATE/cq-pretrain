"""Capture a running Gemma Serving's current boot without restarting it.

Run inside that Serving. Only an environment allowlist is read into the receipt;
authentication values never leave their files or enter the output. This does not
change versions-node0.json or the consumer's current cache identity.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ('config.json', 'gateway.py', 'launch.sh', 'serve.py')
ENV_ALLOWLIST = (
    'CORPUS_DEPLOYMENT_ID', 'CORPUS_BOOT_ID', 'CORPUS_BACKEND',
    'CORPUS_LOCAL_REPLICAS', 'HCCL_INTRA_ROCE_ENABLE', 'HCCL_SOCKET_IFNAME',
    'HCCL_IF_IP', 'GLOO_SOCKET_IFNAME', 'TP_SOCKET_IFNAME', 'VLLM_HOST_IP',
    'ASCEND_RT_VISIBLE_DEVICES', 'MASTER_PORT', 'VLLM_PORT',
)


def process(pid):
    directory = Path('/proc') / str(pid)
    argv = directory.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    # These services pass keys by environment, never argv. Refuse an unexpected
    # command-line credential instead of persisting a possible secret.
    assert not any('api-key' in arg.lower() or 'authorization' in arg.lower() for arg in argv)
    environ = {}
    for item in directory.joinpath('environ').read_bytes().split(b'\0'):
        key, _, value = item.partition(b'=')
        name = key.decode(errors='replace')
        if name in ENV_ALLOWLIST:
            environ[name] = value.decode(errors='replace')
    # comm can contain spaces; fields after its final ')' start with stat field3.
    stat_tail = directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'starttime_ticks': int(stat_tail[19]),
            'argv': argv, 'environment_allowlist': environ}


def main():
    deployment = os.environ['CORPUS_DEPLOYMENT_ID']
    assert deployment.replace('-', '').isalnum()
    run = ROOT / 'runtime' / deployment
    before = json.loads((run / 'node0.json').read_text())
    versions = json.loads((run / 'versions-node0.json').read_text())
    assert before['hostname'] == socket.gethostname(), 'Wrong container'
    assert time.time() - before['time'] < 15, 'Stale heartbeat'
    assert before['boot_id'] == versions['boot_id']
    supervisor = process(before['pid'])
    assert str(ROOT / 'serve.py') in supervisor['argv']
    child_ids = Path(f"/proc/{before['pid']}/task/{before['pid']}/children").read_text().split()
    children = [process(int(pid)) for pid in child_ids]
    assert len(children) == 2, 'Expected one gateway and one TP4 engine parent'
    assert all(child['environment_allowlist'].get('CORPUS_BOOT_ID') == before['boot_id']
               for child in children), 'Child boot differs from heartbeat'
    engine_launch = json.loads((run / 'launch-replica0.json').read_text())['argv']
    assert any(child['argv'][-(len(engine_launch) - 1):] == engine_launch[1:]
               for child in children), 'Actual engine argv differs from launch receipt'
    assert any(str(ROOT / 'gateway.py') in child['argv'] for child in children)
    source_files = []
    for name in SOURCE_FILES:
        path = ROOT / name
        stat = path.stat()
        assert stat.st_mtime <= before['boot_started_at'], 'Source changed after boot: ' + name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.stat().st_mtime_ns == stat.st_mtime_ns, 'Source changed during capture'
        source_files.append({'name': name, 'sha256': digest,
                             'mtime': stat.st_mtime, 'size': stat.st_size})
    source_manifest = [{'name': x['name'], 'sha256': x['sha256']} for x in source_files]
    code_revision = 'sha256:' + hashlib.sha256(json.dumps(
        source_manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    after = json.loads((run / 'node0.json').read_text())
    assert (before['boot_id'], before['pid']) == (after['boot_id'], after['pid'])
    receipt = {'captured_at': datetime.now(timezone(timedelta(hours=8), 'Asia/Shanghai')).isoformat(),
               'deployment_id': deployment, 'boot_id': before['boot_id'],
               'boot_started_at': before['boot_started_at'],
               'serving_code_revision': code_revision, 'source_files': source_files,
               'supervisor': supervisor, 'children': children,
               'effective_launch_argv': engine_launch,
               'runtime_versions': versions, 'heartbeat_before': before,
               'heartbeat_after': after, 'versions_file_modified': False,
               'qualification_changed': False,
               'evidence_limit': 'Current mounted source content predates this boot; not a signed image attestation.'}
    output = run / ('live-attestation-' + before['boot_id'] + '.json')
    with output.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': str(output), 'boot_id': before['boot_id'],
                      'serving_code_revision': code_revision}))


if __name__ == '__main__':
    main()
