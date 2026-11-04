"""Gemma4 BF16 supervisor: temporary TP4 Serving or four TP4 Job replicas.

Model files are never fetched here. Only the verified, pinned shared asset is used.
"""
from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config.json').read_text())
GROUP = os.environ['CORPUS_DEPLOYMENT_ID']
assert GROUP.replace('-', '').isalnum(), 'Invalid deployment id'
BACKEND = os.environ.get('CORPUS_BACKEND', 'inspire_serving')
REPLICAS = int(os.environ.get('CORPUS_LOCAL_REPLICAS', '1'))
assert (BACKEND, REPLICAS) in {('inspire_serving', 1), ('inspire_job', 4)}
RUN = ROOT / 'runtime' / GROUP
RUN.mkdir(parents=True, exist_ok=True)
BOOT_ID = uuid.uuid4().hex
BOOT_STARTED_AT = time.time()
os.environ['CORPUS_BOOT_ID'] = BOOT_ID
children = []


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def spawn(args, name, env=None):
    with (RUN / (name + '.log')).open('a', buffering=1) as log:
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
    children.append(child)
    return child


def stop(*_):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    raise SystemExit(0)


def heartbeat():
    route = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    route.connect(('1.1.1.1', 80))
    ip = route.getsockname()[0]
    route.close()
    while True:
        write_json(RUN / 'node0.json', {'time': time.time(), 'ip': ip,
                   'hostname': socket.gethostname(), 'deployment_id': GROUP,
                   'backend': BACKEND, 'local_replicas': REPLICAS, 'pid': os.getpid(),
                   'boot_id': BOOT_ID, 'boot_started_at': BOOT_STARTED_AT})
        time.sleep(3)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
threading.Thread(target=heartbeat, daemon=True).start()

# Queuing may finish before the direct downloader. Wait for its atomic receipt,
# never accept the presence of a preallocated .part file as complete weights.
deadline = time.monotonic() + 7200
while not (ROOT / 'WEIGHTS_READY.json').is_file():
    if time.monotonic() > deadline:
        raise RuntimeError('Verified model assets did not become ready within two hours')
    print(json.dumps({'stage': 'waiting_for_verified_weights', 'deployment_id': GROUP}), flush=True)
    time.sleep(30)

receipt = json.loads((ROOT / 'WEIGHTS_READY.json').read_text())
assert receipt['revision'] == CONFIG['revision'] and receipt['all_required_files_verified']
assert receipt['destination'] == CONFIG['model_path']
model = Path(CONFIG['model_path'])
assert all((model / entry['name']).stat().st_size == entry['size'] for entry in receipt['files'])
metadata = json.loads((model / 'config.json').read_text())
assert metadata['architectures'] == ['Gemma4ForConditionalGeneration']
assert not metadata.get('quantization_config') and not metadata['text_config'].get('quantization_config')
processor_provenance = json.loads((ROOT / 'processor_provenance.json').read_text())
processor_files = []
for item in processor_provenance['files']:
    digest = hashlib.sha256((model / item['name']).read_bytes()).hexdigest()
    assert digest == item['sha256'], 'Processor asset content changed: ' + item['name']
    processor_files.append({'name': item['name'], 'sha256': digest})
processor_revision = 'sha256:' + hashlib.sha256(json.dumps(
    sorted(processor_files, key=lambda item: item['name']), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert processor_revision == CONFIG['processor_revision'] == processor_provenance['processor_revision']
write_json(RUN / 'processor-provenance.json', processor_provenance)

versions = {}
for package in ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers'):
    versions[package] = importlib.metadata.version(package)
import torch
import torch_npu
assert torch.npu.device_count() == REPLICAS * 4, 'Visible NPU count differs from requested allocation'
versions['visible_npu_count'] = torch.npu.device_count()
versions['npu_name'] = torch.npu.get_device_name(0)
versions.update(deployment_id=GROUP, model=CONFIG['model_name'], precision='BF16',
                revision=CONFIG['revision'], processor_revision=processor_revision,
                local_replicas=REPLICAS, tensor_parallel=4, boot_id=BOOT_ID,
                boot_started_at=BOOT_STARTED_AT,
                processor_options={'max_soft_tokens': CONFIG['max_soft_tokens']},
                max_images=CONFIG['max_images'], max_model_len=CONFIG['max_model_len'],
                communication={'HCCL_INTRA_ROCE_ENABLE': os.environ.get('HCCL_INTRA_ROCE_ENABLE'),
                               'HCCL_SOCKET_IFNAME': os.environ.get('HCCL_SOCKET_IFNAME'),
                               'HCCL_IF_IP': os.environ.get('HCCL_IF_IP')})
write_json(RUN / 'versions-node0.json', versions)
with (RUN / 'npu-smi.txt').open('w') as output:
    subprocess.run(['npu-smi', 'info'], stdout=output, stderr=subprocess.STDOUT, check=False)

try:
    front = spawn([sys.executable, str(ROOT / 'gateway.py')], 'gateway')
    for replica in range(REPLICAS):
        args = ['vllm', 'serve', str(model), '--host', '0.0.0.0', '--port',
                str(9000 + replica),
                '--served-model-name', CONFIG['model_name'], '--tensor-parallel-size', '4',
                '--dtype', 'bfloat16', '--max-model-len', str(CONFIG['max_model_len']),
                '--max-num-seqs', str(CONFIG['max_num_seqs']), '--max-num-batched-tokens', '4096',
                '--gpu-memory-utilization', '0.90', '--enforce-eager',
                '--reasoning-parser', 'gemma4', '--mm-processor-kwargs',
                json.dumps({'max_soft_tokens': CONFIG['max_soft_tokens']}),
                '--limit-mm-per-prompt', json.dumps({'image': CONFIG['max_images'], 'video': 0, 'audio': 0}),
                '--no-enable-log-requests', '--disable-uvicorn-access-log', '--seed', '1024']
        env = os.environ.copy()
        if BACKEND == 'inspire_job':
            env['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(4 * replica, 4 * replica + 4))
        env['MASTER_PORT'] = str(29500 + replica)
        env['VLLM_PORT'] = str(29600 + replica)
        env['VLLM_API_KEY'] = Path(CONFIG['engine_key_path']).read_text().strip()
        write_json(RUN / f'launch-replica{replica}.json', {
            'argv': args, 'visible_devices': env.get('ASCEND_RT_VISIBLE_DEVICES', 'container_all'),
            'versions': versions, 'time': time.time(), 'revision': CONFIG['revision'],
            'precision': 'bfloat16', 'quantized': False})
        spawn(args, f'engine-replica{replica}', env)
    print(json.dumps({'stage': 'loading_model', 'deployment_id': GROUP, 'versions': versions,
                      'replicas': REPLICAS, 'tensor_parallel': 4}), flush=True)
    while all(child.poll() is None for child in children):
        time.sleep(2)
    raise RuntimeError('Gemma runtime subprocess exited: ' + str([p.poll() for p in children]))
finally:
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
