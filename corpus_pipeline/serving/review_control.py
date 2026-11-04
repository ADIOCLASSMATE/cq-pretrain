"""Readiness and release gates for an independently owned native review pool."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from .budget_pool import BudgetPool, audited_unsent_backend, process_identity, read, require


def controller_alive(pool):
    for name in ('GUARD_HEARTBEAT.json', 'RUN_PROCESS.json'):
        path = pool.root / name
        if path.exists():
            identity = read(path).get('identity')
            if identity and process_identity(identity.get('pid', -1)) == identity:
                return True
    return False


def ready(pool, minimum=1):
    require(type(minimum) is int and 1 <= minimum <= pool.config['replicas'], 'Invalid readiness threshold')
    qualified, errors = [], []
    if not pool.disabled() and controller_alive(pool):
        binding = pool.backend.binding()[1]
        ledger = pool.ledger()
        for number in ledger['slots']:
            try:
                state = pool.state(number)
                require(state.get('binding') == binding, 'Runtime binding changed')
                require(state.get('canary_passed') is True and state.get('phase') == 'QUALIFIED', 'Not qualified')
                path = pool.directory(number) / 'registry.json'
                entries = read(path)['endpoints']
                require(len(entries) == 1, 'Expected one native endpoint')
                entry = entries[0]
                require(entry.get('ready') is True and entry.get('ready_expires_at', 0) > time.time(), 'Expired lease')
                require(entry.get('supports_images') is True and entry.get('transport_mode') == 'native_openai'
                        and entry.get('native_stream') is True and entry.get('backend') == 'inspire_serving',
                        'Wrong native image transport')
                keys = ('service_id', 'model', 'revision', 'precision', 'processor_revision',
                        'runtime_revision', 'incarnation', 'platform_instance_id', 'launch_contract_sha256')
                require(all(entry.get(key) == state.get(key) and state.get(key) for key in keys),
                        'Registry identity changed')
                receipt = Path(entry['qualification_receipt']).resolve()
                require(receipt.is_relative_to((pool.directory(number) / 'smoke').resolve())
                        and str(receipt) == state.get('qualification_receipt'), 'Unbound smoke receipt')
                proof = read(receipt)
                require(proof.get('passed') is True and proof.get('request_count') == 1
                        and proof.get('qualification_kind') == 'basic_smoke'
                        and proof.get('standard_sse_done') is True and proof.get('image_input')
                        and proof.get('usage', {}).get('prompt_tokens', 0) > 0
                        and proof.get('finish_reason') == 'stop' and proof.get('response_text', '').strip(),
                        'Real image request evidence missing')
                require(all(proof.get(key) == state.get(key) for key in keys if key != 'service_id'),
                        'Stale image request identity')
                qualified.append(number)
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append({'slot': number, 'error_type': type(error).__name__})
    return {'ready': len(qualified) >= minimum, 'ready_replicas': len(qualified),
            'minimum_ready': minimum, 'target_replicas': pool.target(), 'errors': errors}


def released(pool):
    """Observe exact owned resources; never Stop, Start, Create or drop a reservation."""
    ledger = pool.ledger()
    if not pool.disabled():
        return {'released': False, 'reserved': sum(row['cards'] for row in ledger['slots'].values()),
                'reason': 'pool_not_disabled'}
    stopped = pool.root / 'STOPPED.json'
    if not stopped.exists() or read(stopped).get('complete') is not True:
        return {'released': False, 'reserved': sum(row['cards'] for row in ledger['slots'].values()),
                'reason': 'stop_not_complete'}

    def check_slot(number):
        state = pool.state(number)
        try:
            require(state.get('release_verified') is True, 'No release evidence')
            if state.get('cancelled_never_sent') is True:
                directory = pool.directory(number)
                require(not state.get('service_id') and not (directory / 'CREATE_TRANSMISSION.json').exists(),
                        'Cancelled reservation may have been sent')
                proof = read(directory / 'CANCELLED_NEVER_SENT.json')
                require(proof['pool_config_sha256'] == pool.config_sha and proof['name'] == state['name']
                        and proof['binding'] == state['binding'] and proof['live_exact_name_matches'] == 0
                        and proof['create_intent_sha256'] == hashlib.sha256(
                            (directory / 'CREATE_INTENT.json').read_bytes()).hexdigest(), 'Cancellation binding changed')
                lifecycle = audited_unsent_backend(pool.backend, pool.config['backend_file'])
                require(lifecycle is not None, 'Unreviewed never-sent adapter')
                with lifecycle.InspireClient(account='wjx-ascend') as client:
                    require(not any(row.name == state['name'] for row in client.servings.iter(
                        workspace=lifecycle.native.WORKSPACE)), 'Cancelled name exists on platform')
            else:
                require(state.get('service_id') and pool.backend.released(state), 'Owned resource still allocated')
            return {'slot': number, 'released': True, 'reserved': 0}
        except Exception as error:
            return {'slot': number, 'released': False, 'reserved': state['cards'],
                    'error_type': type(error).__name__}

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(check_slot, ledger['slots']))
    reserved = sum(row['reserved'] for row in rows)
    alive = controller_alive(pool)
    return {'released': reserved == 0 and not alive, 'reserved': reserved,
            'controller_alive': alive, 'slots': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('ready', 'released', 'stop'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--min-ready', type=int, default=1)
    args = parser.parse_args()
    pool = BudgetPool(args.config)
    result = (ready(pool, args.min_ready) if args.action == 'ready' else
              released(pool) if args.action == 'released' else pool.stop_once())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
