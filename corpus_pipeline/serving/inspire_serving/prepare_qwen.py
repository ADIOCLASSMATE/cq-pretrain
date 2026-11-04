"""Create only the authorized native Serving HIGH32 + LOW32 pair."""
import argparse
import json
import os
from pathlib import Path
import time

from inspire.sdk import InspireClient
from inspire.sdk.models_serving import ServingCreateSpec

ROOT = Path(__file__).resolve().parent
WORKSPACE = '昇腾卡公共空间'
PROJECT = '多模态大模型新架构评测探索与scaling-law'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.chmod(temp, 0o600)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', choices=['high', 'low'])
    args = parser.parse_args()
    cfg = json.loads((ROOT/'qwen/config.json').read_text())
    assert (ROOT.parent/'runtime/WATCHDOG_DISABLED').exists()
    assert all(json.loads((ROOT.parent/'runtime'/x).read_text())['disabled'] for x in ['qwen-state.json', 'gemma-state.json'])
    with InspireClient(account='wjx-ascend') as client:
        for old in ['corpus-qwen397b-bf16-p3-20261022-r01', 'corpus-gemma4-bf16-p3-20261022-r01']:
            assert client.jobs.get(old, workspace=WORKSPACE).status == 'CANCELLED'
        for tier, priority in [('high', 6), ('low', 3)]:
            name = f'corpus-qwen-serving-{tier}-20261023-r01'
            deployment = f'qwen-serving-{tier}-20261023-r01'
            spec = ServingCreateSpec(name=name, model='corpus-qwen397b-bf16-ref-20261023', model_version=1,
                command=f'env TZ=Asia/Shanghai CORPUS_DEPLOYMENT_ID={deployment} bash {ROOT}/qwen/launch.sh',
                port=8000, workspace=WORKSPACE, project=PROJECT, group='910B资源', quota='16,128,1024',
                image='vllm-ascend:v0.23.0rc1', replicas=1, nodes_per_replica=2, shm_gib=128,
                priority=priority, auto_scaling=False, public_path_readonly=True,
                description=f'Owned corpus Qwen397B BF16 native Serving {tier.upper()}; 2x16 cards TP8 DP4 EP32; platform authentication; visual qualification required')
            plan = client.servings.plan(spec)
            assert plan.priority == priority
            save(ROOT/f'runtime/qwen-{tier}-plan.json', plan.to_dict())
            save(ROOT/f'runtime/qwen-{tier}-payload.json', {**plan.create_kwargs, 'inference_serving_type': 'CUSTOM'})
            if args.submit != tier:
                continue
            state_path = ROOT/f'runtime/qwen-{tier}.json'
            if state_path.exists():
                raise RuntimeError('Prior create intent exists: reconcile exact service identity; never blind-resubmit')
            assert not any(x.name == name for x in client.servings.list(workspace=WORKSPACE).items)
            state = {'schema_version': 'corpus_pipeline.native_serving.v1', 'role': 'generator', 'tier': tier,
                'name': name, 'service_name': name, 'operation_id': name, 'phase': 'CREATE_INTENT',
                'payload': plan.create_kwargs, 'priority': priority, 'cards': 32,
                'model': cfg['model_name'], 'revision': cfg['revision'], 'processor_revision': cfg['processor_revision'],
                'precision': 'BF16', 'deployment_id': deployment,
                'runtime_root': str(ROOT/'qwen/runtime/serving-20261023'), 'runtime_layout': 'prefixed_groups',
                'max_images': 16, 'expected_nodes': 2, 'expected_replicas': 1,
                'api_key_file': str(ROOT.parent/'private/inspire-serving.key'), 'api_key_env': 'INF_API_KEY',
                'created_at': time.time(), 'canary_passed': False,
                'request_overrides': {'chat_template_kwargs': {'enable_thinking': False},'mm_processor_kwargs': {'min_pixels':1048576,'max_pixels':1048576}},
                'auto_recovery_enabled': tier == 'low'}
            save(state_path, state)
            handle = client.servings.create(spec, operation_id=name)
            state['service_id'] = handle.ref.key
            state['phase'] = 'OBSERVING'
            save(state_path, state)
            detail = client.servings.get(handle.ref, workspace=WORKSPACE)
            state['endpoint'] = detail.to_dict().get('endpoint', '')
            save(state_path, state)
            print(json.dumps({'created': name, 'service_id': handle.ref.key, 'priority': priority, 'cards':32}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
