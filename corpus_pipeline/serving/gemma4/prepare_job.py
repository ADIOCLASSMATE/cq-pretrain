"""Prepare, then explicitly submit, the owned 16-NPU Gemma4 deployment.

Use the inspire-skill Python. Submission requires the temporary Serving object
to be confirmed STOPPED and every downloaded model file to be verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from inspire.sdk import InspireClient
from inspire.sdk.models import JobCreateSpec

ROOT = Path(__file__).resolve().parent
SKILL = Path('/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/.codex/skills/inspire-wjx-ascend/scripts')
sys.path.insert(0, str(SKILL))
from fast_job_control import API
from node_policy import apply_fixed_exclusions

NAME = 'corpus-gemma4-bf16-p3-20261022-r01'
DEPLOYMENT = 'gemma4-31b-20261022-r01'
WORKSPACE = '昇腾卡公共空间'
ACCOUNT = 'wjx-ascend'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    raise SystemExit('Retired: model inference must use native Serving. Use prepare_servings.py; do not create a deployment Job.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if (ROOT / 'job_handoff.json').exists():
        raise SystemExit('Existing job handoff: never submit a duplicate deployment')
    command = ('env CORPUS_DEPLOYMENT_ID=' + DEPLOYMENT + ' CORPUS_BACKEND=inspire_job '
               'CORPUS_LOCAL_REPLICAS=4 bash ' + str(ROOT / 'launch.sh'))
    spec = JobCreateSpec(
        name=NAME, workspace=WORKSPACE, project='多模态大模型新架构评测探索与scaling-law',
        group='910B资源', quota='16,128,1024', image='vllm-ascend:v0.30.0rc2',
        command=command, nodes=1, shm_gib=128, priority=3, auto_fault_tolerance=False,
        public_path_readonly=True, enable_notification=False,
        envs={'CORPUS_DEPLOYMENT_ID': DEPLOYMENT},
        exclude_nodes=['infra-gpu-npu-248.host.shzhisuan.com'],
        description='corpus_pipeline owned Gemma4-31B BF16; 4 isolated TP4 replicas use all 16 NPUs; recovery by corpus supervisor',
    )
    with InspireClient(account=ACCOUNT) as client:
        plan = client.jobs.plan(spec)
        payload = apply_fixed_exclusions(plan.create_kwargs, ACCOUNT)
        save(ROOT / 'job_plan.json', plan.to_dict())
        save(ROOT / 'job_original_payload.json', payload)
    if not args.submit:
        print(json.dumps({'prepared': True, 'submitted': False, 'name': NAME, 'cards': 16}))
        return
    ready = json.loads((ROOT / 'WEIGHTS_READY.json').read_text())
    assert ready['all_required_files_verified']
    status = subprocess.run(['inspire', '--json', 'serving', 'status',
                             'corpus-gemma4-31b-p3-canary-1022', '--workspace', WORKSPACE,
                             '--account', ACCOUNT], check=True, text=True, capture_output=True)
    details = json.loads(status.stdout)
    assert details['success'] and details['data']['status'] == 'STOPPED', 'Confirm temporary 4-card Serving stopped first'
    save(ROOT / 'serving_confirmed_stopped.json', details)
    api = API(ACCOUNT, WORKSPACE, '910B资源')
    try:
        matches = api.named_records(NAME)
        if matches:
            raise RuntimeError('Matching job already exists; reconcile its exact identity instead of creating again')
        operation = 'corpus-gemma4-20261022-r01'
        save(ROOT / 'job_create_intent.json', {'operation_id': operation, 'name': NAME,
                                              'time': time.time(), 'payload_path': str(ROOT / 'job_original_payload.json')})
        identifier = api.create(payload, operation)
        handoff = {
            'backend': 'inspire_job', 'job_id': identifier, 'name': NAME,
            'original_payload_path': str(ROOT / 'job_original_payload.json'),
            'payload_path': str(ROOT / 'job_original_payload.json'),
            'deployment_id': DEPLOYMENT, 'deployment_prefix': 'gemma4-31b-20261022',
            'name_prefix': 'corpus-gemma4-bf16-p3-20261022',
            'heartbeat_root': str(ROOT / 'runtime'),
            'launch_command': 'env CORPUS_BACKEND=inspire_job CORPUS_LOCAL_REPLICAS=4 bash ' + str(ROOT / 'launch.sh'),
            'model': 'Gemma4-31B-IT', 'revision': ready['revision'], 'precision': 'BF16',
            'processor_revision': json.loads((ROOT / 'processor_provenance.json').read_text())['processor_revision'],
            'max_images': 8, 'local_replicas': 4, 'tensor_parallel': 4,
            'budget_cards': 16, 'priority': 3, 'role': 'verifier', 'generation': 1,
            'client_api_key_path': str(ROOT.parent / 'private/client.key'),
            'request_overrides': {'chat_template_kwargs': {'enable_thinking': False}},
            'canary_passed': False, 'operation_id': operation, 'created_at': time.time(),
        }
        save(ROOT / 'job_handoff.json', handoff)
        print(json.dumps(handoff, ensure_ascii=False))
    finally:
        api.close()


if __name__ == '__main__':
    main()
