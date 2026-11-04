"""Persist progress/ETA and publish a completed offline index, never a training release."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from .queue import status, _digest as queue_digest
from corpus_pipeline.production.common import atomic_json, file_sha


def image_reference_basis(plan):
    """Read the frozen snapshot's image denominator without changing plan.json.

    A full-snapshot count cannot be applied to a bounded shard validation run.
    The manifest hash and record count both have to match the frozen plan.
    """
    if plan.get('scope') != 'full_native_dcvlm':
        return {'total': None, 'scope': 'image_denominator_unknown_for_bounded_run'}
    path = Path(plan['snapshot_root']) / 'manifest.json'
    expected = plan['snapshot_fingerprint']['manifest.json']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('ETA snapshot manifest differs from frozen plan')
    manifest = json.loads(raw)
    if manifest.get('phase') != 'complete' or manifest.get('schema_version') != 'dcvlm.pretrain.v1':
        raise ValueError('ETA requires a completed native DCVLM manifest')
    statistics = manifest['statistics']
    if statistics['samples'] != plan['record_count']:
        raise ValueError('ETA manifest record denominator differs from frozen plan')
    count = statistics['image_occurrences']
    if type(count) is not int or count < 0:
        raise ValueError('ETA image reference denominator must be a nonnegative integer')
    return {'total': count, 'scope': 'full_native_snapshot_image_occurrences',
            'manifest_path': str(path), 'manifest_sha256': expected,
            'unit': 'image_references_not_global_unique_images'}


def estimate_eta(current, plan, *, records_per_s, images_per_s, image_basis):
    """Require measurable progress in every unfinished channel before an ETA.

    Taking the maximum prevents fast CPU text work from hiding stalled image
    work. This remains conditional on sustained observed rates, not an SLA or a
    prediction for later global deduplication, API review, or training release.
    """
    total_images = image_basis['total']
    remaining_records = max(0, plan['record_count'] - current['processed_records'])
    remaining_images = (max(0, total_images - current['processed_images'])
                        if total_images is not None else None)
    def channel_seconds(remaining, rate):
        if remaining == 0:
            return 0.0
        if remaining is None or not math.isfinite(rate) or rate <= 0:
            return None
        return remaining / rate
    record_eta = channel_seconds(remaining_records, records_per_s)
    image_eta = channel_seconds(remaining_images, images_per_s)
    reasons = []
    if record_eta is None:
        reasons.append('record_channel_has_no_positive_observed_rate')
    if remaining_images is None:
        reasons.append('image_reference_denominator_unknown')
    elif image_eta is None:
        reasons.append('image_channel_has_no_positive_observed_rate')
    completed = current.get('all_tasks_completed') is True
    if completed and remaining_records == 0 and remaining_images in (0, None):
        # A bounded run can finish by authoritative task completion even though
        # its image denominator was unavailable for extrapolation beforehand.
        eta, reason, reasons = 0.0, 'all_queue_tasks_completed', []
    elif reasons:
        eta, reason = None, 'waiting_for_all_unfinished_channels'
    elif remaining_records == remaining_images == 0 and not completed:
        eta, reason = None, 'waiting_for_authoritative_task_completion'
    else:
        eta = max(record_eta, image_eta)
        reason = 'maximum_of_record_and_image_channel_estimates'
    deadline = current.get('seconds_to_deadline')
    return {'remaining_records': remaining_records,
            'remaining_image_references': remaining_images,
            'image_references_total': total_images,
            'estimated_records_seconds_remaining': record_eta,
            'estimated_image_seconds_remaining': image_eta,
            'estimated_seconds_remaining': eta,
            'deadline_projected_feasible': (eta <= deadline) if eta is not None and deadline is not None else None,
            'eta_reason': reason, 'eta_unavailable_reasons': reasons,
            'eta_scope': 'offline_frontpass_only_conditional_on_sustained_rolling_committed_rates',
            'eta_excludes': ['global_source_reconciliation', 'global_sscd_search', 'API_review', 'training_release'],
            'eta_image_reference_basis': image_basis}


def finalize(root, progress):
    root = Path(root).resolve()
    if not progress['all_tasks_completed']:
        raise ValueError('cannot finalize an incomplete queue')
    plan = json.loads((root / 'plan.json').read_text())
    if progress['processed_records'] != plan['record_count']:
        raise ValueError('completed record count does not match frozen catalog')
    counts, categories, parts = Counter(), Counter(), []
    for path in sorted((root / 'states').glob('*.json')):
        state = json.loads(path.read_text()); receipt = state['receipt']
        if state.get('status') != 'completed' or receipt.get('status') != 'completed':
            raise ValueError('noncompleted state in completed queue')
        if state['task_sha256'] != queue_digest(state['task']) or receipt['task_sha256'] != state['task_sha256']:
            raise ValueError('task identity changed after queue completion')
        intent = {key: receipt[key] for key in ('task_id', 'token', 'task_sha256', 'artifacts',
                                               'processed_records', 'processed_images', 'metadata')}
        if receipt['intent_sha256'] != queue_digest(intent):
            raise ValueError('completion intent checksum changed')
        bound = {Path(item['path']).name: item for item in receipt['artifacts']}
        summary_path = Path(bound['summary.json']['path'])
        if not summary_path.is_absolute():
            summary_path = root / summary_path
        if file_sha(summary_path) != bound['summary.json']['sha256']:
            raise ValueError('committed summary has changed')
        summary = json.loads(summary_path.read_text())
        if summary['task'] != state['task'] or summary['token'] != receipt['token']:
            raise ValueError('summary task/attempt does not match the committed lease')
        if summary['counts'].get('records', 0) != receipt['processed_records']:
            raise ValueError('summary per-task record coverage differs from completion receipt')
        if summary['counts'].get('image_occurrences', 0) != receipt['processed_images']:
            raise ValueError('summary per-task image count differs from completion receipt')
        counts.update(summary['counts']); categories.update(summary['categories'])
        parts.append({'task': summary['task'], 'artifacts': receipt['artifacts'],
                      'model_identity_sha256': summary['model_identity_sha256']})
    if len(parts) != plan['task_count'] or counts['records'] != plan['record_count']:
        raise ValueError('incomplete final part coverage')
    result = {'schema': 'dcvlm.offline_frontpass_release.v1', 'complete': True,
              'scope': 'offline_frontpass_only', 'training_release': False,
              'snapshot_root': plan['snapshot_root'], 'snapshot_fingerprint': plan['snapshot_fingerprint'],
              'counts': dict(counts), 'categories': dict(categories), 'parts': parts,
              'api_calls': 0, 'source_records_rewritten': 0, 'suspects_are_deleted': False,
              'global_exact_source_reconciliation_complete': False,
              'global_sscd_search_complete': False,
              'image_counts_unit': 'image_references_not_global_unique_images',
              'required_next_steps': ['global_asset_source_reconciliation', 'global_sscd_candidate_search',
                                      'separate_suspect_queue_API_review_when_requested'],
              'finished_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(root / 'frontpass_index.json', result)
    atomic_json(root / 'FRONTPASS_COMPLETE.json', {k: v for k, v in result.items() if k != 'parts'} |
                {'index_sha256': file_sha(root / 'frontpass_index.json')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    with (root / '.monitor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        history = []
        plan = json.loads((root / 'plan.json').read_text())
        image_basis = image_reference_basis(plan)
        while True:
            current = status(root); now = time.time()
            history.append((now, current['processed_records'], current['processed_images']))
            history = [entry for entry in history if entry[0] >= now - 1800]
            baseline = history[0]; elapsed = now - baseline[0]
            records_per_s = (current['processed_records'] - baseline[1]) / elapsed if elapsed > 0 else 0
            images_per_s = (current['processed_images'] - baseline[2]) / elapsed if elapsed > 0 else 0
            current.update(rolling_window_seconds=elapsed, rolling_records_per_second=records_per_s,
                           rolling_image_references_per_second=images_per_s,
                           api_calls=0, training_release=False)
            current.update(estimate_eta(current, plan, records_per_s=records_per_s,
                                        images_per_s=images_per_s, image_basis=image_basis))
            atomic_json(root / 'PROGRESS.json', current)
            print(json.dumps(current), flush=True)
            if current['all_tasks_completed']:
                finalize(root, current); break
            if current['all_tasks_terminal'] and current['counts']['failed']:
                atomic_json(root / 'FRONTPASS_FAILED.json', current); raise RuntimeError('exhausted shard retries')
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
