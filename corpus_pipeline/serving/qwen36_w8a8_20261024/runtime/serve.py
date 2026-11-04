"""One native Serving allocation: Qwen3.6 W8A8, two NPUs, TP2/DP1/EP2."""
from __future__ import annotations
import importlib.metadata
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

from contract import code_revision, engine_argv, validate_assets

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get('CORPUS_CONFIG', ROOT / 'config.json'))
CONFIG = json.loads(CONFIG_PATH.read_text())
DEPLOYMENT = os.environ['CORPUS_DEPLOYMENT_ID']
assert DEPLOYMENT.replace('-', '').isalnum(), 'Invalid deployment name'
assert int(os.environ.get('LWS_GROUP_SIZE', '1')) == 1, 'Only one node per replica is supported'
BOOT = uuid.uuid4().hex
RUN = ROOT / 'runs' / DEPLOYMENT / BOOT
RUN.mkdir(parents=True, exist_ok=False)
os.environ.update(QWEN_BOOT_ID=BOOT, QWEN_REPLICA_ID=DEPLOYMENT, QWEN_LEADER_IP='127.0.0.1')
children = []


def save(name, value):
    path = RUN / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def heartbeat():
    while True:
        save('node0.json', {'time': time.time(), 'hostname': socket.gethostname(), 'deployment_id': DEPLOYMENT,
             'boot_id': BOOT, 'node': 0, 'pid': os.getpid(), 'runtime_dir': str(RUN)})
        time.sleep(3)


def spawn(argv, name, env=None):
    with (RUN / (name + '.log')).open('a', buffering=1) as output:
        process = subprocess.Popen(argv, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    children.append(process)
    return process


def stop(*_):
    for process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
threading.Thread(target=heartbeat, daemon=True).start()
assets = validate_assets(CONFIG)
versions = {name: importlib.metadata.version(name)
            for name in ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers')}
import torch
import torch_npu
assert torch.npu.device_count() == 2, 'Expected exactly two allocated NPUs'
code, hashes = code_revision(ROOT)
argv = engine_argv(CONFIG)
versions.update(model=CONFIG['model_name'], revision=CONFIG['revision'], precision=CONFIG['precision'],
    processor_revision=CONFIG['processor_revision'], deployment_id=DEPLOYMENT, boot_id=BOOT,
    visible_npu_count=2, tensor_parallel=2, data_parallel=1, expert_parallel=True,
    serving_code_revision=code, source_sha256=hashes, asset_validation=assets,
    runtime_config={k: v for k, v in CONFIG.items() if 'key' not in k and k != 'processor_files'},
    processor_options={'max_pixels': CONFIG['max_pixels']},
    communication={k: os.environ.get(k) for k in ('HCCL_INTRA_ROCE_ENABLE', 'HCCL_OP_EXPANSION_MODE', 'HCCL_BUFFSIZE')})
save('versions-node0.json', versions)
save('launch.json', {'argv': argv, 'precision': CONFIG['precision'], 'quantization': 'ascend',
     'boot_id': BOOT, 'created_at': time.time(), 'source_sha256': hashes})
try:
    env = os.environ.copy()
    env['VLLM_API_KEY'] = Path(CONFIG['engine_key_path']).read_text().strip()
    spawn(argv, 'engine', env)
    spawn([sys.executable, str(ROOT / 'gateway.py')], 'gateway')
    print(json.dumps({'stage': 'loading_model', 'boot_id': BOOT, 'runtime_dir': str(RUN),
                      'model': CONFIG['model_name'], 'precision': CONFIG['precision']}), flush=True)
    while all(process.poll() is None for process in children):
        time.sleep(2)
    raise RuntimeError('Native model or gateway process exited')
finally:
    for process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
