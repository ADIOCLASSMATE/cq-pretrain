"""Read-only observer of the four exact accepted Job IDs; never mutates Jobs.

Polling retains native platform retries. Missing API observations remain unknown,
not deleted/cancelled. No name lookup, workload create, stop or delete is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import socket
import sys
import threading
import time

from corpus_pipeline.production.common import atomic_json, file_sha

ACCOUNT = 'wjx-ascend'
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/dcvlm_frontpass_20261022T2129Z'
URL = re.compile(r'https?://[^\s\"\x27<>]+', re.I)
SENSITIVE = re.compile(r'(?i)(authorization\s*[:=]\s*bearer\s+|(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*)[^\s,;]+')


def sanitized(value, limit=8000):
    value = URL.sub('[redacted-url]', str(value))
    value = SENSITIVE.sub(r'\1[redacted]', value)
    return value[-limit:]


def load_submissions(root):
    execution = root / 'job_submission/execution'
    submitted = json.loads((execution / 'SUBMITTED.json').read_text())
    jobs = submitted['jobs']
    if len(jobs) != 4 or len({row['job_id'] for row in jobs}) != 4 or submitted['total_cards'] != 64:
        raise ValueError('Expected four distinct accepted 16-card Jobs')
    result = []
    for index, item in enumerate(jobs):
        directory = execution / f'job-{index + 1:02d}'
        accepted = json.loads((directory / 'ACCEPTED.json').read_text())
        payload_path = directory / 'submitted-payload.json'
        if accepted['job_id'] != item['job_id'] or file_sha(payload_path) != item['payload_sha256']:
            raise ValueError('Accepted Job identity or payload hash mismatch')
        payload = json.loads(payload_path.read_text())
        if payload['name'] != item['name'] or item['cards'] != 16:
            raise ValueError('Accepted Job name or card contract mismatch')
        result.append({**item, 'workspace_id': payload['workspace_id'],
                       'submitted_priority': payload['task_priority'],
                       'submitted_native_retry': {
                           'enabled': payload.get('auto_fault_tolerance'),
                           'max_retry': payload.get('fault_tolerance_max_retry'),
                           'interval_seconds': payload.get('fault_tolerance_retry_interval_sec')}})
    if len({item['workspace_id'] for item in result}) != 1:
        raise ValueError('Accepted Jobs do not share the intended workspace')
    return result


def event_view(event):
    allowed = ('type', 'reason', 'message', 'count', 'first_timestamp', 'last_timestamp',
               'event_time', 'created_at', 'firstTimestamp', 'lastTimestamp', 'object_name')
    return {key: sanitized(event[key], 2000) if isinstance(event[key], str) else event[key]
            for key in allowed if key in event}


def observe(client, jobs):
    from inspire.platform.web.browser_api.jobs import (
        list_jobs_by_ids, get_job_detail_v2, list_job_instances, list_job_events_by_ids)
    from inspire.services.job.job_events import event_sort_key
    from inspire.services.job.job_output import public_job_status
    from inspire.services.job.job_status import normalize_status
    from inspire.sdk.models import JobRef

    ids = [item['job_id'] for item in jobs]
    workspace_id = jobs[0]['workspace_id']
    with client._transport.scope(timeout=90):
        rows = list_jobs_by_ids(ids, workspace_id=workspace_id, session=client._transport.session)
    try:
        with client._transport.scope(timeout=60):
            events, missing_events = list_job_events_by_ids(ids, session=client._transport.session)
        event_error = None
    except Exception as exc:
        events, missing_events, event_error = {}, ids, type(exc).__name__
    result = []
    for accepted in jobs:
        job_id = accepted['job_id']
        row = rows.get(job_id)
        observation_error = None
        if row is None:
            try:
                with client._transport.scope(timeout=35):
                    row = get_job_detail_v2(job_id, session=client._transport.session)
            except Exception as exc:
                observation_error = type(exc).__name__
        view = public_job_status(row or {}, fallback_name=accepted['name'])
        record = {
            'job_id': job_id, 'name': accepted['name'], 'cards': accepted['cards'],
            'observed': bool(row), 'status': normalize_status(row.get('status')) if row else 'UNKNOWN_NOT_OBSERVED',
            'raw_status': row.get('status') if row else None,
            'submitted_priority': accepted['submitted_priority'],
            'live_priority': view.get('priority'), 'live_priority_level': view.get('priority_level'),
            'native_retry_submitted': accepted['submitted_native_retry'],
            'native_retry_live': {key: row.get(key) for key in (
                'auto_fault_tolerance', 'fault_tolerance_max_retry', 'fault_tolerance_retry_interval_sec')} if row else {},
            'observation_error_type': observation_error,
            'events_observed': job_id not in missing_events,
            'events_error_type': event_error,
            'latest_events': [event_view(item) for item in sorted(events.get(job_id, []), key=event_sort_key)[-12:]],
        }
        try:
            with client._transport.scope(timeout=35):
                instances, total = list_job_instances(job_id, limit=100, session=client._transport.session)
            if len(instances) < total:
                raise RuntimeError('Instance list unexpectedly truncated')
            record['instances'] = [{key: item.get(key) for key in (
                'name', 'instance_status', 'instance_type', 'node', 'node_status',
                'created_at', 'started_at', 'finished_at', 'running_round')}
                for item in instances]
            record['instances_observed'] = True
        except Exception as exc:
            record.update(instances=[], instances_observed=False, instances_error_type=type(exc).__name__)
        if record['status'] == 'RUNNING':
            try:
                ref = JobRef(name=accepted['name'], account=client.account, base_url=client.base_url,
                             key=job_id, workspace_id=workspace_id)
                logs = client.jobs.logs(ref, tail=40, limit=40, max_chars=8000, window='30m')
                text = sanitized(logs.text)
                record['logs'] = {'text': text, 'truncated': logs.truncated,
                                  'total': logs.total, 'time': time.time(),
                                  'contains_error_marker': bool(re.search(r'\b(error|traceback|failed|exception)\b', text, re.I))}
                record['recommendation'] = ('Inspect worker logs and native retry evidence.' if record['logs']['contains_error_marker']
                                            else 'Confirm worker model initialization and queue progress; keep native retries.')
            except Exception as exc:
                record['logs'] = {'error_type': type(exc).__name__}
        result.append(record)
    return {'schema': 'dcvlm_job_observation.v1', 'observed_at': time.time(),
            'observed_at_shanghai': datetime.now(timezone.utc).astimezone(
                timezone(timedelta(hours=8))).isoformat(),
            'account': ACCOUNT, 'lookup': 'official ListJobs exact job_ids, missing fallback GetJob exact ID',
            'read_only': True, 'job_mutations': 0, 'native_retry_managed_by_platform': True,
            'jobs': result, 'all_job_ids_observed': all(item['observed'] for item in result)}


def transition_key(row):
    return {'status': row['status'], 'observed': row['observed'],
            'instances': [{key: item.get(key) for key in ('name', 'instance_status', 'node', 'running_round')}
                          for item in row.get('instances', [])]}


def save_observation(root, value):
    path = root / 'JOB_STATUS.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    old = {row['job_id']: row for row in previous.get('jobs', [])}
    changes = []
    for row in value['jobs']:
        before = transition_key(old[row['job_id']]) if row['job_id'] in old else None
        after = transition_key(row)
        if before != after:
            changes.append({'time': value['observed_at'], 'job_id': row['job_id'],
                            'name': row['name'], 'before': before, 'after': after,
                            'recommendation': row.get('recommendation')})
    if changes:
        with (root / 'JOB_STATUS_CHANGES.jsonl').open('a') as stream:
            for change in changes:
                stream.write(json.dumps(change, ensure_ascii=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
    atomic_json(path, value)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('Interval must be at least 10 seconds')
    root = args.root.resolve()
    jobs = load_submissions(root)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    logging.getLogger('inspire').setLevel(logging.CRITICAL)
    from inspire.sdk import InspireClient
    owner = root / '.job_watch_owner'
    owner.mkdir(mode=0o700)  # No stale-owner takeover, no cross-host flock reliance.
    atomic_json(owner / 'owner.json', {'pid': os.getpid(), 'hostname': socket.gethostname(),
                                     'started_at': time.time()})
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        with InspireClient(account=ACCOUNT) as client:
            while not stop.is_set():
                started = time.time()
                try:
                    value = observe(client, jobs)
                    changes = save_observation(root, value)
                    runtime = {'healthy': True, 'last_success': time.time(), 'changes': len(changes)}
                    print(json.dumps({'time': time.time(), 'changes': changes,
                                      'statuses': {r['name']: r['status'] for r in value['jobs']}}), flush=True)
                except Exception as exc:
                    runtime = {'healthy': False, 'time': time.time(), 'error_type': type(exc).__name__}
                    print(json.dumps(runtime), flush=True)
                atomic_json(root / 'JOB_WATCH_RUNTIME.json', {**runtime, 'pid': os.getpid(),
                    'hostname': socket.gethostname(), 'interval_seconds': args.interval,
                    'read_only': True, 'script_sha256': file_sha(__file__)})
                if args.once:
                    break
                stop.wait(max(0, args.interval - (time.time() - started)))
    finally:
        (owner / 'owner.json').unlink()
        owner.rmdir()


if __name__ == '__main__':
    main()
