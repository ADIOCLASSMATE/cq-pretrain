"""Create exactly the authorized HIGH4 + LOW4 native Serving pair.

Use the inspire-skill Python. A dry plan is always saved before submission.
No Job or LOW Job binder is created. All downloads stay outside the runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from inspire.sdk import InspireClient
from inspire.sdk.models_serving import ServingCreateSpec

ROOT = Path(__file__).resolve().parent
WORKSPACE = '昇腾卡公共空间'
ACCOUNT = 'wjx-ascend'
STAMP = '20261023'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', choices=['high', 'low'])
    args = parser.parse_args()
    ready = json.loads((ROOT / 'WEIGHTS_READY.json').read_text())
    config = json.loads((ROOT / 'config.json').read_text())
    assert ready['all_required_files_verified'] and ready['revision'] == config['revision']
    with InspireClient(account=ACCOUNT) as client:
        for tier, priority in [('high', 6), ('low', 3)]:
            name = f'corpus-gemma4-serving-{tier}-{STAMP}-r01'
            deployment = f'gemma4-serving-{tier}-{STAMP}-r01'
            spec = ServingCreateSpec(
                name=name, model='Gemma4-31B-IT-842da379', model_version=1,
                command=f'env TZ=Asia/Shanghai CORPUS_DEPLOYMENT_ID={deployment} CORPUS_BACKEND=inspire_serving CORPUS_LOCAL_REPLICAS=1 bash {ROOT / "launch.sh"}',
                port=8000, workspace=WORKSPACE,
                project='多模态大模型新架构评测探索与scaling-law', group='910B资源',
                quota='4,32,256', image='vllm-ascend:v0.30.0rc2',
                replicas=1, nodes_per_replica=1, shm_gib=128, priority=priority,
                auto_scaling=False, public_path_readonly=True,
                description=f'Owned corpus Gemma4 BF16 native Serving {tier.upper()}; TP4 one replica; platform INF_API_KEY auth; visual canary required',
            )
            plan = client.servings.plan(spec)
            assert plan.priority == priority
            save(ROOT / f'serving_{tier}_plan_{STAMP}.json', plan.to_dict())
            save(ROOT / f'serving_{tier}_kwargs_{STAMP}.json', plan.create_kwargs)
            # Explicitly mirror the exact current CreateServingConsole body for review.
            payload = {key: value for key, value in plan.create_kwargs.items() if value is not None}
            payload['inference_serving_type'] = 'CUSTOM'
            save(ROOT / f'serving_{tier}_payload_{STAMP}.json', payload)
            if args.submit != tier:
                continue
            handoff_path = ROOT / f'serving_{tier}_handoff_{STAMP}.json'
            if handoff_path.exists():
                raise RuntimeError('Existing Serving identity; inspect it instead of submitting again')
            for old in ['corpus-gemma4-bf16-p3-20261022-r01', 'corpus-qwen397b-bf16-p3-20261022-r01']:
                assert client.jobs.get(old, workspace=WORKSPACE).status == 'CANCELLED', 'Old deployment Job must be stopped'
            for old in ['gemma-state.json', 'qwen-state.json']:
                assert json.loads((ROOT.parent / 'runtime' / old).read_text())['disabled'], 'Old Job controller state must be disabled'
            existing = client.servings.list(workspace=WORKSPACE)
            assert not any(item.name == name for item in existing.items), 'Named Serving already exists'
            operation = f'corpus-gemma4-serving-{tier}-{STAMP}-r01'
            save(ROOT / f'serving_{tier}_intent_{STAMP}.json', {
                'name': name, 'operation_id': operation, 'requested_priority': priority,
                'time_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'reserved_pair_cards': 8, 'combined_qwen_gemma_cards': 72,
            })
            handle = client.servings.create(spec, operation_id=operation)
            detail = client.servings.get(handle.ref, workspace=WORKSPACE)
            save(handoff_path, {
                'backend': 'inspire_serving', 'role': 'verifier', 'tier': tier,
                'service_id': handle.ref.key, 'service_name': name, 'deployment_id': deployment,
                'payload_path': str(ROOT / f'serving_{tier}_payload_{STAMP}.json'),
                'create_kwargs_path': str(ROOT / f'serving_{tier}_kwargs_{STAMP}.json'),
                'model': config['model_name'], 'revision': config['revision'],
                'processor_revision': config['processor_revision'], 'precision': 'BF16',
                'max_images': config['max_images'], 'local_replicas': 1, 'tensor_parallel': 4,
                'requested_priority': priority, 'workspace': WORKSPACE, 'account': ACCOUNT,
                'heartbeat_root': str(ROOT / 'runtime'), 'api_key_env': 'INF_API_KEY',
                'runtime_root': str(ROOT / 'runtime'), 'runtime_dir': str(ROOT / 'runtime' / deployment),
                'endpoint': detail.to_dict().get('endpoint', ''),
                'api_key_file': str(ROOT.parent / 'private/inspire-serving.key'),
                'auth': 'platform_authorization_bearer', 'canary_passed': False,
                'created_at_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'detail': detail.to_dict(), 'operation_id': operation,
            })
            print(json.dumps({'created': name, 'service_id': handle.ref.key, 'priority': priority,
                              'handoff': str(handoff_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
