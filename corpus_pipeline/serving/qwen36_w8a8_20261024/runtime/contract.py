"""Pure runtime contract for one Qwen3.6 Ascend W8A8 TP2/EP allocation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

CODE_FILES = ('serve.py', 'gateway.py', 'launch.sh', 'contract.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    check(config['model_name'] == 'Qwen3.6-35B-A3B-W8A8', 'Wrong model')
    check(config['precision'] == 'W8A8_DYNAMIC' and config['quantization'] == 'ascend', 'Wrong quantization')
    check(config['nodes'] == 1 and config['tensor_parallel_size'] == 2
          and config['data_parallel_size'] == 1 and config['enable_expert_parallel'] is True, 'Wrong topology')
    check(config['dtype'] == 'bfloat16', 'Nonquantized dtype must use the validated CANN W8A8 scale/output pair')
    check(config['max_num_seqs'] == 128 and config['max_num_batched_tokens'] == 4096, 'Unexpected scheduling profile')
    check(config['max_model_len'] == 16384 and config['gpu_memory_utilization'] == 0.90, 'Unexpected memory profile')
    check(config['num_speculative_tokens'] == 3 and config['enable_json_keepalive'] is True, 'Unexpected recipe')
    check(config['cudagraph_capture_sizes'] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
          'Expected bounded graph shapes through 128 sequences times four speculative tokens')
    check(config['max_images'] == 16 and config['max_pixels'] == 1048576, 'Image input contract changed')


def validate_assets(config):
    """Read the verified receipt, rehash metadata, and check every shard size.

    Large weight SHA verification belongs to the completed download receipt;
    startup does not rescan 40 GB for each of the eventual read-only replicas.
    """
    validate_config(config)
    receipt_path = Path(config['weights_ready_path'])
    check(file_sha(receipt_path) == config['weights_ready_sha256'], 'Asset receipt changed')
    receipt = json.loads(receipt_path.read_text())
    check(receipt.get('complete') is True and receipt.get('precision') == 'W8A8_DYNAMIC'
          and receipt.get('quantization') == 'ascend', 'Verified W8A8 asset is not ready')
    model = Path(config['model_path']).resolve()
    check(Path(receipt['model_path']).resolve() == model and receipt['revision'] == config['revision'], 'Asset identity changed')
    for entry in receipt['files']:
        path = (model / entry['name']).resolve()
        check(path.is_relative_to(model) and entry.get('verified') is True
              and path.is_file() and path.stat().st_size == entry['size_bytes'], 'Asset file absent or changed')
        if not path.name.endswith('.safetensors'):
            check(file_sha(path) == entry['sha256'], 'Asset metadata changed')
    processor = {name: file_sha(model / name) for name in config['processor_files']}
    check(processor == config['processor_files'] and 'sha256:' + digest(processor) == config['processor_revision'],
          'Processor content changed')
    metadata = json.loads((model / 'config.json').read_text())
    check(metadata.get('architectures') == ['Qwen3_5MoeForConditionalGeneration']
          and bool(metadata.get('vision_config')), 'Wrong multimodal architecture')
    description = json.loads((model / 'quant_model_description.json').read_text())
    quantized = sum(value == 'W8A8_DYNAMIC' for value in description.values())
    check(quantized > 0, 'Real Ascend W8A8 tensor description missing')
    index = json.loads((model / 'quant_model_weights.safetensors.index.json').read_text())
    expected_files = {entry['name'] for entry in receipt['files']}
    check(set(index['weight_map'].values()) <= expected_files, 'Weight index references an unverified shard')
    return {'quantized_description_entries': quantized, 'verified_file_count': len(receipt['files']),
            'weight_shards': len(set(index['weight_map'].values())), 'weights_ready_sha256': config['weights_ready_sha256']}


def code_revision(root):
    hashes = {name: file_sha(Path(root) / name) for name in CODE_FILES}
    return 'sha256:' + digest(hashes), hashes


def engine_argv(config):
    validate_config(config)
    return ['vllm', 'serve', config['model_path'], '--host', '127.0.0.1', '--port', '9000',
        '--served-model-name', config['model_name'], '--tensor-parallel-size', '2', '--data-parallel-size', '1',
        '--enable-expert-parallel', '--quantization', 'ascend', '--dtype', config['dtype'],
        '--max-model-len', str(config['max_model_len']), '--max-num-seqs', str(config['max_num_seqs']),
        '--max-num-batched-tokens', str(config['max_num_batched_tokens']), '--gpu-memory-utilization', '0.90',
        '--trust-remote-code', '--safetensors-load-strategy', 'lazy', '--enable-prefix-caching',
        '--speculative-config', json.dumps({'method': 'qwen3_5_mtp', 'num_speculative_tokens': 3}),
        '--compilation-config', json.dumps({'cudagraph_mode': 'FULL_AND_PIECEWISE',
            'cudagraph_capture_sizes': config['cudagraph_capture_sizes'], 'max_cudagraph_capture_size': 512}),
        '--additional-config', json.dumps({'enable_cpu_binding': True, 'multistream_overlap_shared_expert': True}),
        '--mm-processor-kwargs', json.dumps({'max_pixels': config['max_pixels']}),
        '--limit-mm-per-prompt', json.dumps({'image': config['max_images'], 'video': 0, 'audio': 0}),
        '--reasoning-parser', 'qwen3', '--default-chat-template-kwargs', json.dumps({'enable_thinking': False}),
        '--no-enable-log-requests', '--disable-uvicorn-access-log', '--seed', '1024']
