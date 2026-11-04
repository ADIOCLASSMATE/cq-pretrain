"""Operational successor for a frozen baseline pipeline; no data-code changes.

Inherited attempts retain their exact old runner identity and result directory.
Only subsequent attempts use the successor ops. This lets a coordinator handoff
preserve an active reconciliation/API worker without restarting or duplicating it.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_handoff(config):
    handoff = config['operational_handoff']
    original_path = Path(handoff['predecessor_ops'])
    if sha(original_path) != handoff['predecessor_ops_sha256']:
        raise ValueError('Predecessor ops changed')
    original = read(original_path)
    for key in ('run_root', 'repository', 'queue_port', 'receipts', 'gates'):
        if config[key] != original[key]:
            raise ValueError('Handoff changed data ownership/gates: ' + key)
    changed = set(handoff['changed_commands'])
    if changed - {'review_start'}:
        raise ValueError('Only the operational review controller can be replaced')
    for name, command in original['commands'].items():
        if name not in changed and command != config['commands'][name]:
            raise ValueError('Inherited workload changed: ' + name)
    if config['control'] == original['control']:
        raise ValueError('Successor must have a new control directory')
    if sha(handoff['launcher']) != handoff['launcher_sha256']:
        raise ValueError('Operational launcher changed')
    for name, attempt in handoff['inherited_attempts'].items():
        if name not in original['commands'] or type(attempt) is not int or attempt < 0:
            raise ValueError('Invalid inherited attempt')
    return original


def install_routes(module, config, original):
    handoff = config['operational_handoff']

    def inherited(name, attempt):
        return 0 < attempt <= handoff['inherited_attempts'].get(name, 0)

    def prior_route(cfg, ops_path, name, attempt, seen=None):
        seen = set() if seen is None else seen
        if ops_path in seen:
            raise ValueError('Cyclic operational handoff')
        seen.add(ops_path)
        parent = cfg.get('operational_handoff')
        if parent and 0 < attempt <= parent['inherited_attempts'].get(name, 0):
            path = parent['predecessor_ops']
            if sha(path) != parent['predecessor_ops_sha256']:
                raise ValueError('Inherited operational ancestor changed')
            return prior_route(read(path), path, name, attempt, seen)
        prefix = ([sys.executable, parent['launcher']] if parent else
                  [sys.executable, '-m', module.MODULE])
        command = prefix + ['execute', '--ops', ops_path, '--name', name, '--attempt', str(attempt)]
        return command, Path(cfg['control']) / 'commands' / name / str(attempt)

    def runner_command(cfg, name, attempt):
        if inherited(name, attempt):
            return prior_route(original, handoff['predecessor_ops'], name, attempt)[0]
        return [sys.executable, handoff['launcher'], 'execute', '--ops',
                cfg['ops_path'], '--name', name, '--attempt', str(attempt)]

    def command_area(cfg, name, attempt):
        if inherited(name, attempt):
            return prior_route(original, handoff['predecessor_ops'], name, attempt)[1]
        return Path(cfg['control']) / 'commands' / name / str(attempt)

    module.runner_command = runner_command
    module.command_area = command_area


def estimate_policy(policy, pending, completed, ready, samples, now):
    """Use successful completed assets, never issued HTTP requests, for ETA."""
    deadline = datetime.fromisoformat(policy['deadline_utc'].replace('Z', '+00:00')).timestamp()
    review_seconds = max(1, deadline - now - policy['reserve_after_review_seconds'])
    per_replica = policy['fallback_assets_per_replica_second']
    measured = None
    samples = [row for row in samples if now - row['time'] <= 900]
    samples.append({'time': now, 'completed': completed, 'ready': ready})
    if len(samples) >= 2:
        first = samples[0]
        elapsed = now - first['time']
        if elapsed >= 300 and completed >= first['completed']:
            measured = (completed - first['completed']) / elapsed
            # Zero progress must request the maximum, not fall back to an
            # optimistic historical speed; unhealthy services remain accounted.
            mean_ready = max(1, sum(x['ready'] for x in samples) / len(samples))
            per_replica = min(per_replica, measured / mean_ready)
    required = (math.ceil(pending * policy['throughput_margin'] /
                         max(1e-9, review_seconds * per_replica)) if pending else 0)
    target = min(policy['max_replicas'], max(policy['minimum_replicas'], required))
    predicted = pending / max(1e-9, ready * per_replica)
    return {'target_replicas': target, 'required_replicas': required,
            'ready_replicas': ready, 'remaining_assets': pending,
            'measured_assets_per_second': measured,
            'conservative_assets_per_replica_second': per_replica,
            'review_seconds_available': review_seconds,
            'estimated_review_seconds_at_current_ready': predicted,
            'capacity_risk': required > policy['max_replicas'],
            'deadline_overdue': now > deadline}, samples


def review_completion(progress):
    counts = progress.get('counts', {})
    valid = counts.get('reason:visual_annotation_only_requires_confirmation', 0)
    blocked = max(0, counts.get('state:done', 0) - valid)
    return valid, blocked


def update_policy(config, state, atomic_json):
    policy_path = Path(config['deadline_resource_policy'])
    policy = read(policy_path)
    root = Path(config['run_root'])
    if (policy['run_root'] != str(root) or policy['deadline_utc'] != config['deadline_utc']
            or policy['pool_config_sha256'] != sha(root / 'review_pool/config.json')):
        raise ValueError('Deadline resource policy binding changed')
    pool_config = read(root / 'review_pool/config.json')
    if not 1 <= policy['minimum_replicas'] <= policy['max_replicas'] <= pool_config['replicas']:
        raise ValueError('Resource target exceeds bound operational pool capacity')
    output = Path(config['control']) / 'RESOURCE_STATUS.json'
    previous = read(output, {})
    now = time.time()
    if now - previous.get('time', 0) < 60:
        return
    pool = read(root / 'review_pool/STATUS.json', {})
    fresh = now - pool.get('time', 0) < 120
    ready = sum(x.get('phase') == 'QUALIFIED' and x.get('platform_status') == 'RUNNING'
                and x.get('canary_passed') is True and not x.get('retired')
                for x in pool.get('services', [])) if fresh else 0
    progress = read(root / 'api_review/PROGRESS.json', {})
    selection = read(root / 'screening/PENDING.json', {}).get('api_selection', {})
    imported = read(root / 'api_review/INPUT_IMPORTED.json', {})
    total = selection.get('assets', imported.get('eligible_assets', policy['estimated_new_review_assets']))
    completed, blocked = review_completion(progress)
    pending = max(0, total - completed)
    samples = previous.get('samples', []) if progress.get('phase') in ('annotating', 'screening') else []
    result, samples = estimate_policy(policy, pending, completed, ready, samples, now)
    active = state.get('phase') in ('review_start', 'screening')
    if active and not (root / 'review_pool/DISABLED').exists():
        atomic_json(root / 'review_pool/TARGET.json', {
            'replicas': result['target_replicas'], 'time': now,
            'policy_sha256': sha(policy_path), 'deadline_utc': policy['deadline_utc'],
            'owner': str(config['control'])})
    result.update(time=now, phase=state.get('phase'), active=active,
                  deadline_utc=policy['deadline_utc'], policy_sha256=sha(policy_path),
                  pool_status_fresh=fresh, pool_reserved_cards=pool.get('slot_reserved_cards'),
                  valid_review_assets=completed, completion_blocked_assets=blocked,
                  model_request_concurrency=1024, samples=samples,
                  completion_requires_validated_CURRENT=True)
    atomic_json(output, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'execute', 'status'))
    parser.add_argument('--ops', required=True)
    parser.add_argument('--name')
    parser.add_argument('--attempt', type=int)
    args = parser.parse_args(argv)
    raw = read(args.ops)
    # Only operational code is new. All business workflows and validation use
    # the same frozen source and execution identity as the running predecessor.
    sys.path.insert(0, raw['repository'])
    from corpus_pipeline.unified import baseline_pipeline_supervisor as module
    config = module.checked_ops(args.ops)
    original = checked_handoff(config)
    install_routes(module, config, original)
    if args.action == 'execute':
        return module.execute(config, args.name, args.attempt)
    if args.action == 'status':
        print(json.dumps(read(Path(config['control']) / 'STATUS.json', {})))
        return 0
    supervisor = module.Supervisor(config)
    with module.process.exclusive(supervisor.control / 'controller.lock'):
        predecessor = config['operational_handoff']['predecessor_identity']
        if module.process.same_process(predecessor, predecessor['argv'], predecessor['cwd']):
            raise ValueError('Predecessor supervisor is still alive')
        supervisor.bind()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: setattr(supervisor, 'stop_requested', True))
        while True:
            complete = supervisor.step()
            try:
                update_policy(config, supervisor.state, module.atomic_json)
            except Exception as exc:
                module.atomic_json(supervisor.control / 'RESOURCE_POLICY_ERROR.json',
                                   {'time': time.time(), 'error_type': type(exc).__name__, 'error': str(exc)})
            if complete:
                break
            time.sleep(config['poll_seconds'])
    return 0 if supervisor.state['phase'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
