"""Read a bound manager ledger and fail closed before releasing a stage budget.

This command performs no platform operations.  A business-stage COMPLETE file
is deliberately irrelevant: only the manager's validated retirement ledger and
the owning local controller lock establish that the allocation budget is free.
Run this on the manager host; flock on shared storage is not a cross-host proof.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import socket

from corpus_pipeline.resource_manager.engine import Controller, _digest, spec_identity
from corpus_pipeline.resource_manager.models import WorkloadSpec


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _bound_configuration(path):
    path = Path(path).resolve(strict=True)
    config = _read(path)
    _require(config.get('adapter') == 'corpus_pipeline.unified.fleet:StageAdapter',
             'not_a_bound_unified_stage_manager')
    contract = config['adapter_config']['contract']
    spec = WorkloadSpec(**config['workload'])
    directory = Path(config['state_dir']).resolve()
    _require(contract['schema'] == 'corpus.unified.fleet.v1'
             and contract['stage'] in ('frontpass', 'sscd_search'), 'invalid_stage_contract')
    _require(str(directory) == contract['state_directory']
             and spec.scope == contract['scope'] and spec.profile == contract['profile']
             and spec.profile == contract['stage']
             and spec.max_cards == contract['stage_max_cards'], 'manager_contract_changed')
    identity_path = Path(contract['identity_path'])
    _require(_sha(identity_path) == contract['identity_sha256'], 'fleet_identity_changed')
    identity = _read(identity_path)
    _require(identity['schema'] == contract['schema']
             and spec.scope == 'unified-' + contract['identity_sha256'][:16]
             and str(directory) == str(Path(identity['manager_directory']) / spec.profile / 'controller')
             and path == Path(identity['manager_directory']) / (spec.profile + '.json')
             and identity['config_path'] == contract['config_path']
             and identity['config_sha256'] == contract['config_sha256']
             and identity['stage_budgets'][spec.profile] == spec.max_cards
             and identity['total_max_cards'] == contract['total_max_cards']
             and identity['account_ceiling'] == contract['account_ceiling'], 'fleet_binding_changed')
    _require(_sha(contract['config_path']) == contract['config_sha256'], 'pipeline_config_changed')
    _require(_sha(identity['template_path']) == identity['template_sha256'], 'fleet_template_changed')
    return path, spec, directory


def status(manager_config):
    """Return a single JSON-compatible proof; unobservable reservations are null.

    Only ``released is True and reserved == 0`` permits another stage.  In
    particular, a never-started manager is not a completed/released stage.
    """
    result = {'released': False, 'reserved': None, 'phase': 'unknown',
              'manager_config': str(Path(manager_config).resolve()),
              'read_only': True, 'proof': 'manager_verified_retirement.v1'}
    lock = None
    try:
        path, spec, directory = _bound_configuration(manager_config)
        result.update(scope=spec.scope, profile=spec.profile, state_dir=str(directory))
        state_path = directory / 'STATE.json'
        required_path = directory / 'STATE_REQUIRED.json'
        if not state_path.exists():
            _require(not required_path.exists(), 'required_state_missing')
            result.update(phase='not_started', reserved=0, reason='stage_not_started')
            return result
        _require(required_path.is_file(), 'required_state_marker_missing')
        owner = _read(directory / '.owner/HOST.json')
        result['owner_host'] = owner.get('hostname')
        same_host = owner.get('hostname') == socket.gethostname()
        result['same_controller_host'] = same_host
        # Opening read/write without O_CREAT never invents a missing lock proof.
        lock = (directory / '.controller.lock').open('r+')
        controller_running = None
        if same_host:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                controller_running = True
            else:
                controller_running = False
        result['controller_running'] = controller_running
        controller = Controller(None, spec, directory)
        state = controller._load()  # Validates required marker, spec and every intent SHA.
        _require(state.get('scope') == spec.scope and state.get('profile') == spec.profile,
                 'state_scope_or_profile_changed')
        rows = state['resources']
        reserved = 0
        retired = 0
        problems = []
        for key, row in rows.items():
            cards = row['intent']['placement']['quota']['cards']
            _require(type(cards) is int and 1 <= cards <= spec.max_cards, 'invalid_reserved_card_count')
            verified = (row.get('state') == 'STOPPED' and row.get('retired_verified') is True
                        and isinstance(row.get('resource_id'), str) and bool(row['resource_id'])
                        and isinstance(row.get('observed_at'), (int, float))
                        and not row.get('action_required') and not row.get('observation_error_type'))
            if verified:
                retired += 1
            else:
                reserved += cards
                problems.append(key)
        result.update(phase=state.get('phase'), reserved=reserved, resources=len(rows),
                      verified_retired_resources=retired, unresolved_intents=problems,
                      state_identity_sha256=_digest(spec_identity(spec)),
                      state_sha256=_digest(state), updated_at=state.get('updated_at'))
        if not same_host:
            result['reason'] = 'query_requires_controller_host'
        elif controller_running:
            result['reason'] = 'controller_still_running'
        elif state.get('phase') != 'completed':
            result['reason'] = 'manager_not_completed'
        elif reserved or problems:
            result['reason'] = 'retirement_not_fully_verified'
        else:
            result.update(released=True, reason='all_allocations_verified_retired')
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result.update(released=False, reserved=None, reason='unverifiable_manager_ledger',
                      error_type=type(error).__name__, error=str(error))
        return result
    finally:
        if lock is not None:
            lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'released'))
    parser.add_argument('--manager-config', required=True, type=Path)
    args = parser.parse_args(argv)
    result = status(args.manager_config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if args.action == 'status' or result['released'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
