"""One owned GPUJob replica = two 16-NPU nodes = TP8 / DP4 / EP32, BF16."""
from __future__ import annotations

import hashlib
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

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads(Path(os.environ.get('CORPUS_CONFIG', ROOT / 'config.json')).read_text())
NODE = int(os.environ['PET_NODE_RANK'])
NODES = int(os.environ['PET_NNODES'])
assert NODES == CONFIG.get('nodes', 2) and 0 <= NODE < NODES, 'Profile/node topology mismatch'
GROUP = os.environ.get('CORPUS_DEPLOYMENT_ID', 'qwen397b-bf16-20261022-r01')
assert GROUP.replace('-', '').isalnum()
RUN = ROOT / 'runtime' / CONFIG['epoch'] / GROUP
RUN.mkdir(parents=True, exist_ok=True)
os.environ['QWEN_RUN_DIR'] = str(RUN)
os.environ['QWEN_REPLICA_ID'] = GROUP
os.environ['QWEN_EXPECTED_CARDS'] = str(NODES * 16)
children = []


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def heartbeat():
    while True:
        write_json(RUN / f'node{NODE}.json', {'time': time.time(), 'node': NODE,
                   'ip': os.environ['HCCL_IF_IP'], 'hostname': socket.gethostname(),
                   'epoch': CONFIG['epoch'], 'deployment_id': GROUP, 'pid': os.getpid()})
        time.sleep(3)


def spawn(args, name, env=None):
    log = (RUN / f'{name}-node{NODE}.log').open('a', buffering=1)
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    log.close()
    children.append(proc)
    return proc


def stop(*_):
    for proc in children:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
threading.Thread(target=heartbeat, daemon=True).start()
started = time.time()
while True:
    try:
        record = json.loads((RUN / 'node0.json').read_text())
        if time.time() - record['time'] < 15:
            leader_ip = record['ip']
            break
    except (OSError, ValueError, KeyError):
        pass
    if time.time() - started > 600:
        raise SystemExit('Replica leader rendezvous timed out')
    time.sleep(1)
os.environ['QWEN_LEADER_IP'] = leader_ip
front = spawn([sys.executable, str(ROOT / 'gateway.py')], 'gateway')

versions = {p: importlib.metadata.version(p) for p in ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers')}
processor_hashes = {name: hashlib.sha256((Path(CONFIG['model_path']) / name).read_bytes()).hexdigest()
                    for name in CONFIG['processor_files']}
assert processor_hashes == CONFIG['processor_files'], 'Processor assets changed'
processor_revision = 'sha256:' + hashlib.sha256(json.dumps(processor_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert processor_revision == CONFIG['processor_revision']
versions.update(deployment_id=GROUP, model=CONFIG['model_name'], precision=CONFIG.get('precision', 'BF16'),
                revision=CONFIG['revision'], processor_revision=processor_revision, node=NODE)
write_json(RUN / f'versions-node{NODE}.json', versions)
print(json.dumps({'replica': GROUP, 'node': NODE, 'stage': 'hccl', 'versions': versions}), flush=True)
probe = spawn(['torchrun', f'--nnodes={NODES}', '--nproc-per-node=16', f'--node-rank={NODE}',
               '--master-addr=' + leader_ip, '--master-port=29877', str(ROOT / 'hccl_probe.py')], 'hccl')
try:
    if probe.wait(timeout=660) != 0:
        raise RuntimeError('Profile HCCL correctness probe failed')
    model = Path(CONFIG['model_path'])
    metadata = json.loads((model / 'config.json').read_text())
    assert metadata['architectures'] == ['Qwen3_5MoeForConditionalGeneration']
    assert metadata.get('vision_config') and metadata['text_config']['dtype'] == 'bfloat16'
    quantized = CONFIG.get('precision') == 'W8A8_DYNAMIC'
    if quantized:
        description = json.loads((model / 'quant_model_description.json').read_text())
        assert 'W8A8_DYNAMIC' in description.values()
    else:
        assert not metadata.get('quantization_config') and not metadata['text_config'].get('quantization_config')
    index = json.loads((model / ('quant_model_weights.safetensors.index.json' if quantized else 'model.safetensors.index.json')).read_text())
    assert all((model / name).is_file() for name in set(index['weight_map'].values()))

    args = ['vllm', 'serve', str(model), '--host', '0.0.0.0', '--port', '9000',
            '--served-model-name', CONFIG['model_name'], '--dtype', 'bfloat16',
            '--distributed-executor-backend', 'mp', '--tensor-parallel-size', '8',
            '--data-parallel-size', str(CONFIG.get('data_parallel_size', 4)), '--data-parallel-size-local', '2',
            '--data-parallel-start-rank', str(NODE * 2), '--data-parallel-address', leader_ip,
            '--data-parallel-rpc-port', '13389', '--enable-expert-parallel',
            '--max-model-len', str(CONFIG['max_model_len']), '--max-num-seqs', str(CONFIG['max_num_seqs']),
            '--max-num-batched-tokens', '4096', '--gpu-memory-utilization', '0.90',
            '--limit-mm-per-prompt', json.dumps({'image': CONFIG['max_images'], 'video': 0}),
            '--mm-processor-kwargs', json.dumps({'max_pixels': 1048576}),
            '--reasoning-parser', 'qwen3', '--no-enable-prefix-caching', '--enable-chunked-prefill',
            '--compilation-config', json.dumps({'cudagraph_mode': 'FULL_DECODE_ONLY',
                                              'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32]}),
            '--additional-config', json.dumps({'multistream_overlap_shared_expert': True}),
            '--seed', '1024', '--trust-remote-code', '--safetensors-load-strategy', 'lazy',
            '--no-enable-log-requests', '--disable-uvicorn-access-log']
    if quantized:
        args += ['--quantization', 'ascend']
    args += ['--headless'] if NODE else ['--api-server-count', '1']
    write_json(RUN / f'launch-node{NODE}.json', {'argv': args, 'replica': GROUP, 'node': NODE,
                                              'versions': versions, 'time': time.time()})
    env = os.environ.copy()
    env['VLLM_API_KEY'] = Path(CONFIG['engine_key_path']).read_text().strip()
    engine = spawn(args, 'engine', env)
    print(json.dumps({'stage': 'loading_model', 'replica': GROUP, 'node': NODE}), flush=True)
    while engine.poll() is None:
        if front.poll() is not None:
            raise RuntimeError('HTTP gateway exited')
        time.sleep(2)
    raise RuntimeError('Model engine exited with code ' + str(engine.returncode))
finally:
    for proc in children:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
