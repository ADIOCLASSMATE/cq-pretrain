"""Consolidate only this run's accepted LOW Notebooks, without temporary cards.

A durable single plan stops complete source-node groups before requesting a
same-size bundle of legal instances on an already occupied target. Unknown stop/Create
results never permit a replay. Queue leases provide worker recovery; this
module does not edit the queue or inspect/control unrelated workloads.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

from corpus_pipeline.production.common import atomic_json, digest
from corpus_pipeline.resource_manager.planning import available_nodes, plan_consolidation as common_consolidation
from .fragment_allocator import LEGAL_CARDS, EXCLUDED, verified_retired, validate_max_cards
from .fragment_controller import validate_live
from . import resource_planning

FRESH_SECONDS = 120
COOLDOWN_SECONDS = 300
PLAN_TIMEOUT_SECONDS = 900
TERMINAL_PHASES = {'completed', 'aborted'}
IDENTITY_FIELDS = ('notebook_id', 'name', 'node', 'devices', 'priority',
                   'workspace_id', 'project_id', 'group_id', 'cpu', 'memory')


def _identity(item):
    return {key: item[key] for key in IDENTITY_FIELDS}


def _record_observation(state, live, now):
    # Preserve controller.step's preemption grace when this module observes a
    # transition first; otherwise a historical accepted_at could retire a
    # freshly preempted target immediately on the next management cycle.
    previous = state.get('platform_status')
    status = live['status']
    state.update(platform_status=status, observed_at=now, node=live.get('node'))
    if status == 'RUNNING':
        state['pending_since'] = None
    elif status == 'PENDING' and previous == 'RUNNING':
        state['pending_since'] = now


def _eligible(item, state, now):
    age = now - state.get('observed_at', 0)
    return (item.get('priority') == 3 and item.get('devices') in LEGAL_CARDS
            and state.get('platform_status') == 'RUNNING'
            and -5 <= age <= FRESH_SECONDS
            and not state.get('retirement') and not state.get('platform_action')
            and not state.get('action_required'))


def plan_consolidation(managed, states, nodes, options, max_cards, *, now=None):
    """Return a pure proposal that removes at least one complete owned node.

    Source groups include *every* non-retired accepted ID on a node. A group
    with pending, missing or uncertain state cannot be selected. The generic
    planner may split the exact moved total into several legal target quotas.
    """
    validate_max_cards(max_cards, required=True)
    now = time.time() if now is None else now
    indexed = {}
    for item in managed:
        identity = _identity(item)
        key = item['notebook_id']
        if key in indexed and indexed[key] != identity:
            raise ValueError('Consolidation managed identity changed')
        indexed[key] = identity
    groups = {}
    for item in indexed.values():
        if not verified_retired(item, states.get(item['notebook_id'], {})):
            groups.setdefault(item['node'], []).append(item)
    total = sum(item['devices'] for rows in groups.values() for item in rows)
    if total > max_cards or len(groups) < 2:
        return None
    eligible = {node: sorted(rows, key=lambda row: row['notebook_id'])
                for node, rows in groups.items()
                if all(_eligible(row, states.get(row['notebook_id'], {}), now) for row in rows)}
    planning_states = {key: dict(value) for key, value in states.items()}
    for key, item in indexed.items():
        if not verified_retired(item, states.get(key, {})) and not _eligible(item, states.get(key, {}), now):
            planning_states.setdefault(key, {})['platform_status'] = 'UNKNOWN'
    spec = resource_planning.make_spec('frontpass', max_cards)
    reservations = resource_planning.reservations(indexed.values(), planning_states, now=now, root='frontpass')
    quota_options = resource_planning.quotas(options)
    candidates = []
    for target in nodes:
        target_name = target['name']
        if target_name not in eligible:
            continue
        result = common_consolidation(spec, resource_planning.capacities([target], excluded=EXCLUDED),
                                      quota_options, reservations)
        if result is None or any(p.node != target_name for p in result.placements):
            continue
        target_quotas = []
        for placement in result.placements:
            matching = [option for option in options if (
                option['gpu'], option['cpu'], option['memory']) == (
                placement.quota.cards, placement.quota.cpu, placement.quota.memory)]
            if not matching:
                raise ValueError('Common planner selected an unavailable quota')
            target_quotas.append(dict(min(matching, key=lambda option: option['quota'])))
        source_nodes = result.source_nodes
        proposal = {'target_node': target_name, 'target_quotas': target_quotas,
                    'target_members': eligible[target_name],
                    'sources': [indexed[key] for key in result.source_ids],
                    'source_nodes': list(source_nodes), 'moved_cards': result.moved_cards,
                    'owned_nodes_before': len(groups), 'owned_nodes_after': result.nodes_after,
                    'max_cards': max_cards,
                    'target_node_snapshot': {key: target.get(key) for key in
                        ('name', 'status', 'cordon_type', 'gpu', 'cpu', 'memory')},
                    'placement_observed_at': now}
        target_cards = sum(row['devices'] for row in eligible[target_name])
        score = (-len(source_nodes), -target_cards, result.moved_cards,
                 len(target_quotas), target_name, source_nodes)
        candidates.append((score, proposal))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _target_quotas(contract):
    """Read old v1 single-quota plans without changing their hash or tag."""
    bundle = contract.get('target_quotas')
    if bundle is None:
        bundle = [contract['target_quota']]
    if (not bundle or any(quota['gpu'] not in LEGAL_CARDS for quota in bundle)
            or sum(quota['gpu'] for quota in bundle) != contract['moved_cards']):
        raise ValueError('Consolidation quota bundle differs from moved-card contract')
    return bundle


def _allocation_tag(contract, index):
    base = 'consolidation:' + contract['operation_id']
    return base + ':' + str(index) if 'target_quotas' in contract else base


class Consolidator:
    """One controller-owned, resumable plan. One stop or Create per advance.

    Call after normal per-Notebook management and before normal allocation.
    While ``allow_normal_allocation`` is false the controller should only call
    this planner and continue its usual exact-ID observations/worker recovery.
    """
    def __init__(self, root, *, cooldown_seconds=COOLDOWN_SECONDS,
                 timeout_seconds=PLAN_TIMEOUT_SECONDS):
        self.root = Path(root).resolve()
        self.path = self.root / 'FRAGMENT_CONSOLIDATION.json'
        self.directory = self.root / 'fragment_consolidation'
        self.cooldown_seconds = cooldown_seconds
        self.timeout_seconds = timeout_seconds

    def _load(self):
        if not self.path.exists():
            return None
        plan = json.loads(self.path.read_text())
        contract = plan['contract']
        if (plan.get('schema') != 'corpus_fragment_consolidation.v1'
                or contract.get('root') != str(self.root)
                or plan.get('contract_sha256') != digest(contract)):
            raise ValueError('Consolidation contract identity changed')
        return plan

    def _save(self, plan):
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, plan)
        atomic_json(self.directory / (plan['contract']['operation_id'] + '.json'), plan)

    def _result(self, plan=None, *, phase=None):
        actual = phase or (plan['phase'] if plan else 'idle')
        active = plan is not None and plan['phase'] not in TERMINAL_PHASES
        return {'active': active, 'allow_normal_allocation': not active,
                'phase': actual,
                'operation_id': plan['contract']['operation_id'] if plan else None,
                'moved_cards': plan['contract']['moved_cards'] if plan else 0}

    def _finish(self, plan, phase, reason, now):
        plan.update(phase=phase, reason=reason, finished_at=now,
                    cooldown_until=now + self.cooldown_seconds)
        self._save(plan)
        return self._result(plan)

    def _restore_retirement(self, item, entry, state, contract, save):
        """Recover a controller-state write lost after the plan's durable send intent."""
        existing = state.get('retirement')
        if existing and existing.get('consolidation_id') != contract['operation_id']:
            raise ValueError('Source retirement belongs to another operation')
        if not existing:
            state['retirement'] = {
                'notebook_id': item['notebook_id'], 'node': item['node'],
                'requested_at': entry['stop_attempted_at'], 'reason': 'node_consolidation',
                'phase': 'stop_sending', 'stop_attempted_at': entry['stop_attempted_at'],
                'stop_uncertain': True, 'cooldown_seconds': self.cooldown_seconds,
                'consolidation_id': contract['operation_id']}
            state.update(phase='retirement_stop_intent', released=False)
            save()

    def _target_fits(self, contract, managed, states, platform, allocator, save, now, start_index=0):
        current = {item['notebook_id']: _identity(item) for item in [*managed, *allocator.load_managed()]}
        for member in contract['target_members']:
            if current.get(member['notebook_id']) != member:
                return False
            live = platform.observe(member)
            validate_live(member, live)
            state = states.setdefault(member['notebook_id'], {'name': member['name']})
            _record_observation(state, live, now)
            save()
            if not _eligible(member, state, now):
                return False
        original_members = {member['notebook_id'] for member in contract['target_members']}
        for key, item in current.items():
            if (item['node'] == contract['target_node'] and key not in original_members
                    and not verified_retired(item, states.get(key, {}))):
                # Accepted replacements may already be RUNNING; only this
                # fresh exact observation permits avoiding a second capacity
                # subtraction. PENDING replacements still reserve their quota.
                live = platform.observe(item)
                validate_live(item, live)
                state = states.setdefault(key, {'name': item['name']})
                _record_observation(state, live, now)
                save()
        context, nodes = allocator.live.discover()
        bundle = _target_quotas(contract)[start_index:]
        if any(quota not in context['options'] for quota in bundle):
            return False
        free = available_nodes(resource_planning.capacities(nodes, excluded=EXCLUDED),
            resource_planning.reservations(current.values(), states, now=now, root=str(self.root)))
        node = next((row for row in free if row.name == contract['target_node']), None)
        return bool(node and node.eligible and node.gpu >= sum(q['gpu'] for q in bundle)
                    and node.cpu >= sum(q['cpu'] for q in bundle)
                    and node.memory >= sum(q['memory'] for q in bundle))

    def advance(self, managed, states, platform, allocator, save, *,
                is_complete=False, enabled=True, should_stop=None, now=None):
        now = time.time() if now is None else now
        should_stop = should_stop or (lambda: False)
        plan = self._load()
        if should_stop():
            return self._result(plan, phase='controller_stopping')
        if plan and plan['phase'] in TERMINAL_PHASES:
            if now < plan['cooldown_until']:
                return self._result(plan, phase='cooldown')
            plan = None
        if plan is None:
            if is_complete or not enabled or allocator.max_cards is None:
                return self._result()
            budget = allocator.managed_budget(managed, states)
            if budget['uncertain_cards'] or budget['over_cap']:
                return self._result(phase='reservation_unavailable')
            context, nodes = allocator.live.discover()
            proposal = plan_consolidation(managed, states, nodes, context['options'],
                                          allocator.max_cards, now=now)
            if proposal is None or should_stop():
                return self._result()
            contract = proposal | {'root': str(self.root), 'created_at': now,
                                   'operation_id': uuid.uuid4().hex}
            plan = {'schema': 'corpus_fragment_consolidation.v1', 'contract': contract,
                    'contract_sha256': digest(contract), 'phase': 'planned', 'sources': {}}
            self._save(plan)
        contract = plan['contract']
        bundle = _target_quotas(contract)
        indexed = {item['notebook_id']: item for item in managed}
        for item in contract['sources']:
            if item['notebook_id'] not in indexed or _identity(indexed[item['notebook_id']]) != item:
                raise ValueError('Consolidation source identity no longer matches accepted ID')
            entry = plan['sources'].get(item['notebook_id'], {})
            if entry.get('stop_attempted_at') is not None:
                state = states.setdefault(item['notebook_id'], {'name': item['name']})
                self._restore_retirement(item, entry, state, contract, save)
        if is_complete or not enabled or allocator.max_cards is None:
            return self._finish(plan, 'aborted', 'disabled_or_work_complete', now)
        if allocator.max_cards != contract['max_cards']:
            return self._finish(plan, 'aborted', 'configured_cap_changed', now)
        if now - contract['created_at'] > self.timeout_seconds:
            return self._finish(plan, 'aborted', 'plan_deadline_retaining_uncertain_reservations', now)

        # Confirm every source as STOPPED. Attempted stops are observed once
        # and never resent, including after a process crash before SDK return.
        for item in contract['sources']:
            key = item['notebook_id']
            entry = plan['sources'].get(key, {})
            state = states.setdefault(key, {'name': item['name']})
            if entry.get('stop_attempted_at') is not None:
                live = platform.observe(item)
                validate_live(item, live)
                _record_observation(state, live, now)
                if live['status'] == 'STOPPED':
                    retirement = state['retirement']
                    retirement.update(phase='verified_stopped', verified_stopped_at=now,
                                      cooldown_until=now + self.cooldown_seconds)
                    state.update(phase='retired_stopped', released=True, platform_action=None)
                    if str(state.get('action_required', '')).startswith('retirement_'):
                        state.pop('action_required', None)
                    entry.update(phase='verified_stopped', verified_stopped_at=now)
                    save(); self._save(plan)
                    continue
                plan['phase'] = 'waiting_source_stopped'
                save(); self._save(plan)
                return self._result(plan)
            if should_stop():
                return self._result(plan, phase='controller_stopping')
            if not self._target_fits(contract, managed, states, platform, allocator, save, now):
                return self._finish(plan, 'aborted', 'target_capacity_or_ownership_changed', now)
            live = platform.observe(item)
            validate_live(item, live)
            _record_observation(state, live, now)
            if live['status'] != 'RUNNING' or not _eligible(item, state, now):
                save()
                return self._finish(plan, 'aborted', 'source_not_running_or_has_other_action', now)
            if should_stop():
                save()
                return self._result(plan, phase='controller_stopping')
            entry = {'phase': 'stop_sending', 'stop_attempted_at': now}
            plan['sources'][key] = entry
            plan['phase'] = 'stopping_sources'
            self._save(plan)
            self._restore_retirement(item, entry, state, contract, save)
            try:
                platform.stop(item)
            except Exception as error:
                entry['stop_error_type'] = type(error).__name__
                plan['phase'] = 'waiting_source_stopped'
                self._save(plan)
                return self._result(plan)
            entry['phase'] = 'stop_requested'
            state['retirement'].update(phase='stop_requested', stop_uncertain=False)
            save(); self._save(plan)
            return self._result(plan)

        # All source reservations are now released. Each exact quota gets one
        # durable tag; only an accepted result advances to the next placement.
        if should_stop():
            return self._result(plan, phase='controller_stopping')
        def record_accepted(index, tag, tag_status):
            identifier = tag_status['notebook_id']
            accepted = next((row for row in allocator.load_managed()
                             if row['notebook_id'] == identifier), None)
            quota = bundle[index]
            if (not accepted or accepted.get('allocation_tag') != tag
                    or accepted['node'] != contract['target_node']
                    or (accepted['devices'], accepted['cpu'], accepted['memory']) != (
                        quota['gpu'], quota['cpu'], quota['memory'])):
                raise ValueError('Replacement does not match exact consolidation contract')
            entry = {'allocation_tag': tag, 'notebook_id': identifier, 'quota': quota}
            replacements = plan.setdefault('replacements', {})
            if str(index) in replacements and replacements[str(index)] != entry:
                raise ValueError('Replacement result changed after acceptance')
            if any(key != str(index) and other['notebook_id'] == identifier
                   for key, other in replacements.items()):
                raise ValueError('Two replacement quotas resolved to one Notebook')
            replacements[str(index)] = entry
            plan['replacement_notebook_ids'] = [replacements[str(i)]['notebook_id']
                                                for i in range(len(bundle)) if str(i) in replacements]
            if len(bundle) == 1:
                plan['replacement_notebook_id'] = identifier
            self._save(plan)

        for index, quota in enumerate(bundle):
            tag = _allocation_tag(contract, index)
            tag_status = allocator.allocation_tag_status(tag)
            if tag_status['status'] == 'accepted':
                record_accepted(index, tag, tag_status)
                continue
            if str(index) in plan.get('replacements', {}):
                raise ValueError('Accepted replacement lost its durable allocation status')
            if tag_status['status'] == 'not_sent':
                if not self._target_fits(contract, managed, states, platform, allocator, save, now,
                                         start_index=index):
                    return self._finish(plan, 'aborted', 'target_capacity_lost_after_stops', now)
                if should_stop():
                    return self._result(plan, phase='controller_stopping')
            elif tag_status['status'] == 'rejected':
                return self._finish(plan, 'aborted', 'replacement_unavailable_allow_normal_refill', now)
            elif tag_status['status'] != 'pending':
                raise ValueError('Unknown consolidation allocation status')
            plan.update(phase='replacement_pending', allocation_tag=tag, replacement_index=index)
            self._save(plan)
            try:
                allocator.discover_and_allocate(
                    managed, states, target_node=contract['target_node'], target_cards=quota['gpu'],
                    target_quota=quota, allocation_tag=tag, should_stop=should_stop)
            except Exception as error:
                plan['allocation_error_type'] = type(error).__name__
                self._save(plan)
            tag_status = allocator.allocation_tag_status(tag)
            if tag_status['status'] == 'accepted':
                record_accepted(index, tag, tag_status)
                if index == len(bundle) - 1:
                    return self._finish(plan, 'completed', 'same_card_replacement_bundle_accepted', now)
                return self._result(plan)
            if tag_status['status'] in ('not_sent', 'rejected'):
                return self._finish(plan, 'aborted', 'replacement_unavailable_allow_normal_refill', now)
            if tag_status['status'] != 'pending':
                raise ValueError('Unknown consolidation allocation status')
            return self._result(plan)
        return self._finish(plan, 'completed', 'same_card_replacement_bundle_accepted', now)
