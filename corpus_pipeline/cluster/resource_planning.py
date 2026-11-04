"""Map the existing frontpass receipts onto the reusable resource planner.

Dataset completion, qualified worker commands and platform receipts remain in
their existing adapters. No resource lifecycle or queue writes occur here.
"""
from __future__ import annotations

import time

from corpus_pipeline.resource_manager.models import NodeCapacity, Quota, Reservation, WorkloadSpec


def make_spec(root, max_cards):
    return WorkloadSpec(scope=str(root), profile='frontpass', max_cards=max_cards,
        command=('frontpass-worker',), allowed_cards=(1, 2, 4, 8, 16),
        migration_policy='restartable')


def reservations(managed, states, uncertain=(), now=None, *, root='frontpass'):
    now = time.time() if now is None else now
    result = []
    for item in managed:
        state = states.get(item['notebook_id'], {})
        retirement = state.get('retirement') or {}
        retired = (retirement.get('phase') == 'verified_stopped'
                   and retirement.get('notebook_id') == item['notebook_id']
                   and retirement.get('node') == item['node']
                   and isinstance(retirement.get('verified_stopped_at'), (int, float)))
        fresh = -5 <= now - state.get('observed_at', 0) <= 120
        actual = state.get('platform_status', 'UNKNOWN')
        # Legacy receipts predating quota fields used eight CPUs/64 GiB per card.
        # Production load_managed restores exact quotas from immutable intents;
        # this fallback exists only for minimal historical planning callers.
        result.append(Reservation(id=item['notebook_id'], resource_id=item['notebook_id'],
            operation_id=item.get('operation_id'), node=item['node'], cards=item['devices'],
            cpu=item.get('cpu', item['devices'] * 8), memory=item.get('memory', item['devices'] * 64),
            scope=str(root), profile='frontpass', state=actual, retired_verified=retired,
            capacity_accounted=fresh and actual == 'RUNNING', ready=fresh and actual == 'RUNNING'
            and not state.get('retirement') and not state.get('platform_action') and not state.get('action_required')))
    for index, intent in enumerate(uncertain):
        payload = intent['create_kwargs']
        identifier = intent.get('operation_id', 'legacy-unknown-' + str(index))
        result.append(Reservation(id='intent:' + identifier, operation_id=identifier,
            node=payload['node_id'], cards=payload['gpu_count'], cpu=payload['cpu_count'],
            memory=payload['memory_size'], scope=str(root), profile='frontpass', state='UNKNOWN'))
    return result


def capacities(rawnodes, excluded=()):
    excluded = set(excluded)
    result = []
    for row in rawnodes:
        gpu_type = row.get('gpu_type') or (row.get('gpu_info') or {}).get('gpu_type')
        result.append(NodeCapacity(name=row['name'],
            gpu=max(0, int(float(row.get('gpu', {}).get('available', 0)))),
            cpu=max(0, float(row.get('cpu', {}).get('available', 0))),
            memory=max(0, float(row.get('memory', {}).get('available', 0))),
            eligible=(row['name'] not in excluded and row.get('status') == 'Ready'
                      and row.get('cordon_type') in ('', None) and gpu_type == 'ASCEND_910B_64G')))
    return result


def quotas(options):
    return [Quota(cards=row['gpu'], cpu=row['cpu'], memory=row['memory']) for row in options]
