#!/usr/bin/env python3
"""Start the existing owned review runner while CPU reconciliation proceeds.

The normal guard adopts the exact RUN_PROCESS receipt after preparation. No
alternate pool, model, priority, resource budget, or review decision is created.
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
from corpus_pipeline.resource_manager.engine import atomic_json
from corpus_pipeline.serving.budget_pool import BudgetPool, locked, process_identity
from corpus_pipeline.unified.baseline_fleet_ops import status

def validate(pool, source, stage):
    cfg = pool.config
    if (stage.get('active_source') != source.name or stage.get('phase') != 'processing_source'
            or cfg['priority'] != 1 or cfg['cards_per_replica'] != 2
            or not 2 <= cfg['max_cards'] <= 512 or cfg['replicas'] * 2 > cfg['max_cards']
            or Path(cfg['root']).resolve() != source / 'review_pool'
            or cfg.get('external_reservations')):
        raise ValueError('Prewarm must match the active owned LOW TP2 pool and existing ceiling')
    for path in (source / 'STOP', source.parents[1] / 'STOP', pool.root / 'DISABLED'):
        if path.exists():
            raise ValueError('An explicit stop forbids review prewarm')
    release = status(source / 'fleet/frontpass.json')
    if release.get('released') is not True or release.get('reserved') != 0:
        raise ValueError('Frontpass card reservations have not been verified released')
    return release

def start(config, source_name):
    cfg = json.loads(Path(config).read_text())
    source = (Path(cfg['run']) / 'sources' / source_name).resolve()
    if source_name not in cfg['sources'] or source.parent != (Path(cfg['run']) / 'sources').resolve():
        raise ValueError('Unknown execution source')
    pool = BudgetPool(source / 'review_pool/config.json')
    runtime = json.loads((source / 'ops.json').read_text())['repository']
    stage = json.loads((Path(cfg['run']) / 'ops/stage_control/STATE.json').read_text())
    release = validate(pool, source, stage)
    command = [cfg['sdk_python'], '-m', 'corpus_pipeline.serving.budget_pool', 'run',
               '--config', str(pool.config_path)]
    with locked(pool.root / 'guard.lock', nonblocking=True):
        receipt_path = pool.root / 'RUN_PROCESS.json'
        saved = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        actual = process_identity(saved.get('pid', -1))
        if actual and actual == saved.get('identity') and actual['command'] == command:
            return {'already_running': True, 'pid': actual['pid']}
        # An existing runner without a matching receipt must not be duplicated.
        with locked(pool.root / 'run.lock', nonblocking=True):
            pass
        with (pool.root / 'run.log').open('a') as log:
            child = subprocess.Popen(command, cwd=runtime, env=os.environ.copy(),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        identity = process_identity(child.pid)
        if identity is None or identity['command'] != command:
            raise RuntimeError('Review runner exited or changed identity before receipt capture')
        atomic_json(receipt_path, {'pid': child.pid, 'identity': identity,
                                  'time': time.time(), 'config_sha256': pool.config_sha})
        result = {'started': True, 'pid': child.pid, 'time': time.time(),
                  'pool_config_sha256': pool.config_sha, 'frontpass_release': release,
                  'normal_guard_will_adopt_exact_runner': True, 'priority': 1,
                  'cards_per_replica': 2, 'max_cards': pool.config['max_cards'], 'target_replicas': pool.target()}
        atomic_json(pool.root / 'PREWARM.json', result)
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True); args = parser.parse_args()
    print(json.dumps(start(args.config, args.source), sort_keys=True))
