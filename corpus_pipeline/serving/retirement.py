"""Proof validation and accounting for permanently retired native Serving slots.

The caller must hold provision.lock then slot lifecycle.lock, verify ownership
with the native backend, persist immutable evidence, and retain the whole ledger.
Only a verified receipt may exempt a reservation; a stopped state never suffices.
"""
import hashlib
import json

TERMINAL = {'INSTANCE_STOPPED', 'INSTANCE_FAILED', 'INSTANCE_SUCCEEDED',
            'INSTANCE_DELETED', 'INSTANCE_TERMINATED', 'INSTANCE_COMPLETED'}
IDENTITY = ('pool_scope', 'pool_config_sha256', 'pool_slot', 'name', 'service_id', 'cards', 'binding')


def require(value, why):
    if not value:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def terminal(state, stopped):
    statuses = stopped.get('instance_statuses')
    return (state.get('phase') == 'RELEASED' and state.get('platform_status') == 'STOPPED'
            and state.get('release_verified') is True and state.get('canary_passed') is False
            and stopped.get('complete') is True and stopped.get('status') == 'STOPPED'
            and bool(state.get('service_id')) and stopped.get('service_id') == state['service_id']
            and isinstance(statuses, list) and bool(statuses) and all(s in TERMINAL for s in statuses)
            and statuses == state.get('instance_statuses'))


def commit(state, request_bytes, stopped_snapshot_bytes, native_observation, *, disabled, now):
    """Construct proof after a fresh, independently owned native observation.

    This function does not supply that observation. The real adapter must perform
    exact native.assert_owned + GetServing/instances, under the lifecycle lock.
    Persist the supplied request and STOPPED bytes immutably alongside the proof.
    """
    request, stopped = json.loads(request_bytes), json.loads(stopped_snapshot_bytes)
    require(disabled, 'retirement_requires_slot_disabled')
    require(request.get('schema_version') == 'corpus.slot_retirement_request.v1'
            and request.get('retire') is True, 'explicit_retirement_request_required')
    require(all(request.get(k) == state.get(k) for k in IDENTITY), 'retirement_request_identity_mismatch')
    require(type(request.get('requested_at')) in (int, float), 'retirement_request_time_required')
    require(terminal(state, stopped), 'current_stopped_proof_invalid')
    require(native_observation.get('ownership_verified') is True
            and native_observation.get('service_id') == state['service_id']
            and native_observation.get('status') == 'STOPPED'
            and native_observation.get('instance_statuses') == stopped['instance_statuses'],
            'fresh_native_stop_evidence_required')
    observed_at = native_observation.get('observed_at')
    require(type(observed_at) in (int, float) and 0 <= now - observed_at <= 60
            and request['requested_at'] <= observed_at, 'native_stop_evidence_stale')
    return {'schema_version': 'corpus.slot_retirement.v1', 'complete': True,
            **{k: state[k] for k in IDENTITY}, 'retired_at': now,
            'request_sha256': sha(request_bytes), 'stopped_snapshot_sha256': sha(stopped_snapshot_bytes),
            'native_observation': dict(native_observation), 'restart_allowed': False}


def verify(state, proof, request_bytes, stopped_snapshot_bytes, current_stopped, *, disabled):
    """Persistent retirement does not expire; fresh reads are required at commit.

    A later routine cleanup may rewrite current STOPPED.json. Its immutable
    retirement snapshot remains hash bound, while the current proof must still
    semantically demonstrate that this same resource is terminal.
    """
    require(disabled, 'retired_slot_reenabled')
    require(proof.get('schema_version') == 'corpus.slot_retirement.v1'
            and proof.get('complete') is True and proof.get('restart_allowed') is False,
            'retirement_commit_missing')
    require(all(proof.get(k) == state.get(k) for k in IDENTITY), 'retirement_identity_changed')
    require(proof.get('request_sha256') == sha(request_bytes)
            and proof.get('stopped_snapshot_sha256') == sha(stopped_snapshot_bytes), 'retirement_evidence_changed')
    historical_state = {**state, 'instance_statuses': json.loads(stopped_snapshot_bytes)['instance_statuses']}
    rebuilt = commit(historical_state, request_bytes, stopped_snapshot_bytes, proof['native_observation'],
                     disabled=disabled, now=proof['retired_at'])
    require(rebuilt == proof, 'retirement_commit_changed')
    require(terminal(state, current_stopped), 'retired_resource_not_terminal')
    return True


def summary(rows, verified_retired_slots=(), *, external_cards=0, extra_cards=0, ceiling=128):
    """verified_retired_slots must come from verify(), never a raw state flag."""
    retired = set(verified_retired_slots)
    slots = {r['pool_slot'] for r in rows}
    require(retired <= slots and len(slots) == len(rows), 'retirement_or_slot_identity_invalid')
    require(len({r['name'] for r in rows}) == len(rows), 'duplicate_name')
    ids = [r['service_id'] for r in rows if r.get('service_id')]
    require(len(ids) == len(set(ids)), 'duplicate_resource_id')
    require(all(type(r['cards']) is int and r['cards'] > 0 for r in rows), 'invalid_cards')
    require(type(external_cards) is int and external_cards >= 0
            and type(extra_cards) is int and extra_cards >= 0, 'invalid_additional_cards')
    active = [r for r in rows if r['pool_slot'] not in retired]
    active_cards = sum(r['cards'] for r in active)
    total = active_cards + external_cards + extra_cards
    require(total <= ceiling, 'card_budget_exceeded')
    return {'historical_slot_count': len(rows), 'historical_reserved_cards': sum(r['cards'] for r in rows),
            'retired_slot_count': len(retired), 'retired_cards': sum(r['cards'] for r in rows if r['pool_slot'] in retired),
            'active_reserved_slots': len(active), 'active_reserved_cards': active_cards,
            'budget_reserved_cards': total, 'next_slot': max(slots, default=0) + 1}


def eligible_unsent(rows, verified_retired_slots, sent_slots, target):
    """Existing/unknown sent reservations consume target before proven-unsent.

    sent_slots includes every existing service ID and CREATE_TRANSMISSION, even
    an unknown dispatched outcome. TARGET=0 parks all proven-unsent. Existing
    running work is not stopped by a lower target.
    """
    require(type(target) is int and target >= 0, 'invalid_target')
    active = {r['pool_slot'] for r in rows} - set(verified_retired_slots)
    sent = set(sent_slots) & active
    return set(sorted(active - sent)[:max(0, target - len(sent))])


def cleanup_coverage(rows, released_slots):
    """Retired entries remain in final cleanup and the watchdog equality gate."""
    return set(released_slots) == {r['pool_slot'] for r in rows}
