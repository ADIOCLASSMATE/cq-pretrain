"""Plan four independent LOW 16-NPU Jobs; submit only with --execute.

Run with the Inspire SDK Python. Every Job shares one durable work queue and
starts 16 workers. This launcher never resubmits an uncertain Create, nor resumes
a partially submitted batch automatically. Inspect its receipts to reconcile it.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / 'corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z'
SKILL = Path('/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/.codex/skills/inspire-wjx-ascend/scripts')
ACCOUNT = 'wjx-ascend'
WORKSPACE = '昇腾卡公共空间'
PROJECT = '多模态大模型新架构评测探索与scaling-law'
GROUP = '910B资源'
IMAGE_NAME = 'dev-wjx-ascend:v-1.4'
IMAGE_URL = 'docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.4'
INSPIRE_EXCLUSION = 'infra-gpu-npu-248.host.shzhisuan.com'
JOB_COUNT = 4
CARDS_PER_JOB = 16
TOTAL_CARDS = JOB_COUNT * CARDS_PER_JOB


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x') as stream:
        os.chmod(path, 0o600)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def qualification(root, required):
    path = root / 'QUALIFIED.json'
    if not path.exists():
        if required:
            raise RuntimeError('QUALIFIED.json missing; no submission allowed')
        return {'success': False, 'reason': 'not_yet_qualified'}
    receipt = json.loads(path.read_text())
    success = receipt.get('success') is True and receipt.get('dtype') == 'float32'
    if required and not success:
        raise RuntimeError('float32 NPU qualification has not passed')
    return {'success': success, 'path': str(path), 'sha256': file_sha(path),
            'dtype': receipt.get('dtype')}


def worker_command(root):
    return shlex.join(['bash', str(REPO / 'corpus_pipeline/cluster/npu_env.sh'),
                       '-m', 'corpus_pipeline.cluster.node_worker', '--root', str(root),
                       '--devices', '16', '--dtype', 'float32', '--batch-size', '32',
                       '--threads', '4'])


def validate_payload(payload, expected_command):
    configs = payload.get('framework_config') or []
    if len(configs) != 1:
        raise ValueError('Expected one homogeneous framework config')
    cfg = configs[0]
    required = {'gpu_count': 16, 'instance_count': 1, 'cpu': 128,
                'mem_gi': 1024, 'shm_gi': 128, 'image': IMAGE_URL}
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError('Unexpected resolved image or resource contract')
    if payload.get('task_priority') != 3 or payload.get('is_publicpath_readonly') is not True:
        raise ValueError('LOW priority and readonly project public paths required')
    if (payload.get('auto_fault_tolerance') is not True
            or payload.get('fault_tolerance_max_retry') != 100
            or payload.get('fault_tolerance_retry_interval_sec') != 60):
        raise ValueError('Expected native fault tolerance 100 retries at 60 seconds')
    if INSPIRE_EXCLUSION not in payload.get('exclude_nodes', []):
        raise ValueError('INSPIRE.md exclusion 248 omitted')
    # SDK wraps the exact command in bash -c; reject any changed worker options.
    if shlex.split(payload['command']) != ['bash', '-c', expected_command]:
        raise ValueError('Unexpected worker command')


def live_plans(client, root):
    from inspire.sdk.models import JobCreateSpec
    from node_policy import apply_fixed_exclusions, require_fixed_exclusions
    from corpus_pipeline.serving.inspire_serving.downsize import observable_project_budget
    from inspire.platform.web.browser_api.availability.api import list_node_dimension

    image = client.images.get(IMAGE_NAME, workspace=WORKSPACE)
    if image.status != 'SUCCESS' or image.url != IMAGE_URL:
        raise RuntimeError('Image must be live SUCCESS with the expected registry URL')
    quotas = client.jobs.quotas(WORKSPACE, group=GROUP, limit=100)
    quota_rows = [row.to_dict() for row in quotas.items]
    if not any(row.get('quota') == '16,128,1024' for row in quota_rows):
        raise RuntimeError('Required 16,128,1024 live quota unavailable')
    availability = [row.to_dict() for row in client.resources.availability(WORKSPACE, group=GROUP)]
    matching = [row for row in availability if row.get('compute_group') == GROUP]
    if not matching or any(row.get('gpus_per_node') != 16 for row in matching):
        raise RuntimeError('Availability and selected per-node quota disagree')
    budget = observable_project_budget(client, TOTAL_CARDS)
    if budget['observable_total_with_request'] > 256:
        raise RuntimeError('Visible queued plus allocated requests exceed the 256-card ceiling')
    command = worker_command(root)
    plans = []
    for index in range(JOB_COUNT):
        name = 'corpus-frontpass-' + root.name.removeprefix('dcvlm_frontpass_').lower() + f'-n{index + 1:02d}'
        spec = JobCreateSpec(
            name=name, workspace=WORKSPACE, project=PROJECT, group=GROUP,
            quota='16,128,1024', image=IMAGE_URL, command=command, nodes=1,
            shm_gib=128, priority=3, public_path_readonly=True, enable_notification=False,
            auto_fault_tolerance=True, fault_tolerance_max_retry=100,
            fault_tolerance_retry_interval_sec=60, exclude_nodes=[INSPIRE_EXCLUSION],
        )
        initial = client.jobs.plan(spec)
        merged = apply_fixed_exclusions(initial.create_kwargs, ACCOUNT)
        spec = replace(spec, exclude_nodes=sorted(merged.get('exclude_nodes', [])))
        plan = client.jobs.plan(spec)
        if plan.project.name != PROJECT or plan.group.name != GROUP or plan.workspace.name != WORKSPACE:
            raise RuntimeError('Resolved project/workspace/group mismatch')
        validate_payload(plan.create_kwargs, command)
        require_fixed_exclusions(plan.create_kwargs, ACCOUNT)
        plans.append(plan)
    with client._transport.scope(timeout=30):
        rows = list_node_dimension(plans[0].create_kwargs['logic_compute_group_id'],
                                  workspace_id=plans[0].create_kwargs['workspace_id'],
                                  session=client._transport.session)
    nodes = {'time': time.time(), 'inventory_count': len(rows),
             'rows': [{'name': row.get('name'), 'status': row.get('status'),
                       'gpu': row.get('gpu'), 'cordon_type': row.get('cordon_type')}
                      for row in rows]}
    return plans, {'time': time.time(), 'image': image.to_dict(), 'quotas': quota_rows,
                   'availability': availability, 'nodes': nodes, 'budget': budget}


def send_exact(client, payload, operation_id):
    """Use SDK single-send transport with the saved, exact planned payload.

    Public jobs.create(spec) replans; using its own SDK adapter here guarantees
    that the bytes represented in submitted-payload.json are the submitted body.
    operation_id is an audit marker, NOT a platform idempotency key.
    """
    from inspire.platform.web.browser_api.jobs import create_training_job
    with client._transport.scope(timeout=120):
        session = client._transport.session
        with client._transport.single_send(operation_id, create=True, inspect='jobs'):
            result = create_training_job(payload=payload, session=session)
    job_id = result.get('job_id') or result.get('id')
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError('Create response did not confirm a Job ID; reconcile, never retry blindly')
    return job_id


def dispatch_batch(client, plans, artifact_dir, initial_budget, qualified):
    from node_policy import require_fixed_exclusions
    from job_controller import bind_after_submit
    from corpus_pipeline.serving.inspire_serving.downsize import observable_project_budget
    accepted = []
    for index, plan in enumerate(plans):
        job_dir = artifact_dir / f'job-{index + 1:02d}'
        job_dir.mkdir(mode=0o700)
        payload = plan.create_kwargs
        require_fixed_exclusions(payload, ACCOUNT)
        if qualified.get('path') and file_sha(qualified['path']) != qualified['sha256']:
            raise RuntimeError('Qualification changed after planning; stop before Create')
        fresh = observable_project_budget(client, 0)
        # Accepted Jobs might not yet be visible; never subtract them prematurely.
        conservative_used = max(fresh['observable_requested_cards'],
                                initial_budget['observable_requested_cards'] + len(accepted) * 16)
        remaining_cards = TOTAL_CARDS - len(accepted) * 16
        if conservative_used + remaining_cards > 256:
            raise RuntimeError('Live budget changed; remaining submission stopped')
        save_new(job_dir / 'budget-before-create.json', {
            'live': fresh, 'conservative_current_cards': conservative_used,
            'remaining_new_cards': remaining_cards,
            'conservative_total_with_remaining': conservative_used + remaining_cards})
        save_new(job_dir / 'submitted-payload.json', payload)
        operation_id = uuid.uuid4().hex
        save_new(job_dir / 'CREATE_INTENT.json', {
            'time': time.time(), 'operation_id': operation_id, 'name': plan.name,
            'payload_sha256': file_sha(job_dir / 'submitted-payload.json'),
            'qualification': qualified, 'retry_allowed': False})
        try:
            job_id = send_exact(client, payload, operation_id)
        except Exception as exc:
            save_new(job_dir / 'CREATE_UNCONFIRMED.json', {
                'time': time.time(), 'operation_id': operation_id,
                'error_type': type(exc).__name__, 'automatic_retry_allowed': False,
                'action': 'Inspect live platform and this exact intent; do not repeat Create.'})
            raise RuntimeError('Create not confirmed; inspect CREATE_UNCONFIRMED.json') from None
        receipt = {'time': time.time(), 'job_id': job_id, 'name': plan.name,
                   'operation_id': operation_id, 'cards': 16,
                   'payload_sha256': file_sha(job_dir / 'submitted-payload.json')}
        save_new(job_dir / 'ACCEPTED.json', receipt)
        accepted.append(receipt)
        hook = bind_after_submit(job_id, payload, job_dir, account=ACCOUNT,
                                 workspace=WORKSPACE, group=GROUP)
        if hook.get('enabled') is not False or hook.get('reason') != 'below_64_gpus':
            raise RuntimeError('Unexpected controller hook for independent 16-card Job')
    save_new(artifact_dir / 'SUBMITTED.json', {'jobs': accepted, 'total_cards': TOTAL_CARDS,
                                             'status': 'accepted_not_yet_running'})
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error('Shared queue run directory does not exist')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    sys.path.insert(0, str(SKILL))
    from inspire.sdk import InspireClient
    artifacts = root / 'job_submission'
    artifacts.mkdir(mode=0o700, exist_ok=True)
    with (artifacts / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.execute and (artifacts / 'execution').exists():
            parser.error('Execution already attempted; inspect receipts and reconcile, never repeat automatically')
        qualified = qualification(root, args.execute)
        with InspireClient(account=ACCOUNT) as client:
            plans, live = live_plans(client, root)
            if args.execute and qualification(root, True) != qualified:
                raise RuntimeError('Qualification changed during live planning')
            tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
            out = artifacts / ('execution' if args.execute else 'plan-' + tag)
            out.mkdir(mode=0o700)
            save_new(out / 'live.json', live)
            save_new(out / 'PLAN.json', {
                'execute': args.execute, 'account': ACCOUNT, 'root': str(root),
                'jobs': [plan.to_dict() for plan in plans], 'total_cards': TOTAL_CARDS,
                'qualification': qualified, 'script_sha256': file_sha(__file__),
                'worker_command': worker_command(root),
                'source_hashes': {str(p.relative_to(REPO)): file_sha(p) for p in [
                    REPO / 'corpus_pipeline/cluster/npu_env.sh',
                    REPO / 'corpus_pipeline/cluster/node_worker.py']}})
            for index, plan in enumerate(plans):
                save_new(out / f'planned-payload-{index + 1:02d}.json', plan.create_kwargs)
            result = dispatch_batch(client, plans, out, live['budget'], qualified) if args.execute else []
            print(json.dumps({'artifact_dir': str(out), 'execute': args.execute,
                              'planned_cards': TOTAL_CARDS, 'budget': live['budget'],
                              'qualification': qualified, 'accepted': result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
