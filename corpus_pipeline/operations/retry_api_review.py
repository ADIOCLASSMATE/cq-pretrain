"""Audited operational retry epochs; execute the unchanged frozen API screener.

The coordinator/workflow must be paused by the operator before ``run``. This
tool takes the screener's writer lock before altering its mutable work ledger.
It never changes INPUT, frozen code, prompts, original assets, or valid reviews.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())


def connect(root, *, readonly=False):
    db = sqlite3.connect(f"file:{Path(root) / 'index.sqlite'}?mode={'ro' if readonly else 'rw'}", uri=True)
    db.row_factory = sqlite3.Row
    return db


def protected_digest(db, excluded):
    """Bind every stored field of successful rows and their receipt references."""
    h = hashlib.sha256(); count = 0
    excluded = set(excluded)
    for row in db.execute("SELECT * FROM assets ORDER BY asset_id"):
        if row['asset_id'] in excluded:
            continue
        if row['state'] != 'done' or row['reason'] != 'visual_annotation_only_requires_confirmation':
            raise ValueError('unexpected_non_success_outside_retry_scope')
        requests = [dict(r) for r in db.execute("SELECT * FROM requests WHERE asset_id=? ORDER BY role", (row['asset_id'],))]
        if not requests or any(r['state'] != 'ok' or not r['receipt_path'] or not r['receipt_sha'] for r in requests):
            raise ValueError('protected_review_has_no_successful_receipt')
        h.update((canonical({'asset': dict(row), 'requests': requests}) + '\n').encode())
        count += 1
    return {'assets': count, 'sha256': h.hexdigest()}


def snapshot(db):
    assets = [dict(r) for r in db.execute("SELECT * FROM assets WHERE state!='done' OR reason!='visual_annotation_only_requires_confirmation' ORDER BY asset_id")]
    requests = []
    for row in assets:
        if not ((row['state'] == 'retryable' and row['reason'] == 'serving_temporarily_unavailable')
                or (row['state'] == 'done' and row['reason'] == 'api_review_failure_exhausted')):
            raise ValueError('unsupported_unresolved_asset_state')
        reqs = [dict(r) for r in db.execute("SELECT * FROM requests WHERE asset_id=? ORDER BY role", (row['asset_id'],))]
        if len(reqs) != 1 or reqs[0]['role'] != 'primary' or reqs[0]['state'] not in {'failed', 'deferred'}:
            raise ValueError('unsupported_unresolved_request_state')
        if reqs[0]['receipt_path'] or reqs[0]['receipt_sha']:
            raise ValueError('refusing_to_reset_receipt_bearing_request')
        if reqs[0]['state'] == 'failed' and (row['reason'] != 'api_review_failure_exhausted' or reqs[0]['attempts'] != 3):
            raise ValueError('unexpected_exhausted_retry_budget')
        requests.extend(reqs)
    excluded = [r['asset_id'] for r in assets]
    return {'assets': assets, 'requests': requests, 'protected': protected_digest(db, excluded),
            'metrics': [dict(r) for r in db.execute('SELECT * FROM metrics ORDER BY name')]}


def apply_epoch(db, before, epoch_sha):
    """Atomic, single-use reset of exhausted requests, with an in-DB audit marker."""
    db.execute('BEGIN IMMEDIATE')
    try:
        if snapshot(db) != before:
            raise ValueError('ledger_changed_since_retry_snapshot')
        marker = 'operational_retry_epoch:' + epoch_sha
        if db.execute('SELECT 1 FROM metrics WHERE name=?', (marker,)).fetchone():
            raise ValueError('retry_epoch_already_applied')
        ids = [r['asset_id'] for r in before['requests'] if r['state'] == 'failed']
        for aid in ids:
            db.execute("UPDATE requests SET attempts=0,state='pending',last_error=NULL WHERE asset_id=? AND role='primary'", (aid,))
            db.execute("UPDATE assets SET state='pending',decision=NULL,reason=NULL,decision_json=NULL WHERE asset_id=?", (aid,))
        db.execute('INSERT INTO metrics(name,value) VALUES(?,?)', (marker, len(ids)))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return ids


def verify_after(db, before):
    ids = [r['asset_id'] for r in before['assets']]
    if protected_digest(db, ids) != before['protected']:
        raise ValueError('successful_reviews_changed_during_retry')
    unresolved = []
    for aid in ids:
        asset = dict(db.execute('SELECT * FROM assets WHERE asset_id=?', (aid,)).fetchone())
        old_asset = next(r for r in before['assets'] if r['asset_id'] == aid)
        for key in ('asset_id', 'input_sha', 'input_json'):
            if asset[key] != old_asset[key]:
                raise ValueError('retried_asset_identity_changed')
        request = dict(db.execute("SELECT * FROM requests WHERE asset_id=? AND role='primary'", (aid,)).fetchone())
        old_request = next(r for r in before['requests'] if r['asset_id'] == aid)
        if request['request_sha'] != old_request['request_sha']:
            raise ValueError('retried_request_identity_changed')
        if not (asset['state'] == 'done' and asset['reason'] == 'visual_annotation_only_requires_confirmation'
                and request['state'] == 'ok' and request['receipt_path'] and request['receipt_sha']):
            unresolved.append({'asset_id': aid, 'state': asset['state'], 'reason': asset['reason'],
                               'request_state': request['state'], 'attempts': request['attempts']})
    return {'protected': before['protected'], 'retry_assets': len(ids), 'resolved': len(ids) - len(unresolved),
            'unresolved': unresolved, 'complete': not unresolved}


def load_frozen(run):
    runtime = run / 'runtime'
    sys.path.insert(0, str(runtime))
    from corpus_pipeline.postprocess import api_screen
    if Path(api_screen.__file__).resolve() != runtime / 'corpus_pipeline/postprocess/api_screen.py':
        raise ValueError('api_screener_is_not_frozen_run_module')
    identity = json.loads((run / 'api_review/INPUT.json').read_text())
    if sha(api_screen.__file__) != identity['code_sha256']:
        raise ValueError('frozen_api_screen_hash_changed')
    for rel, expected in identity['dependencies'].items():
        if sha(runtime / 'corpus_pipeline' / rel) != expected:
            raise ValueError('frozen_dependency_hash_changed:' + rel)
    if identity['backend'] != 'serving' or identity['mode'] != 'annotate' or identity['max_attempts'] != 3:
        raise ValueError('unsupported_original_screening_policy')
    return api_screen, identity


def make_argv(run, identity, concurrency):
    cfg = json.loads((run / 'CONFIG.json').read_text())
    options = cfg['screening']
    scope = identity['scope']
    args = ['--output', str(run / 'api_review'), '--backend', 'serving', '--mode', identity['mode'],
            '--primary', identity['models']['primary'], '--primary-pool', options['primary_pool'],
            '--prompt', identity['prompt_path'], '--max-tokens', str(identity['max_tokens']),
            '--max-attempts', str(identity['max_attempts']), '--concurrency', str(concurrency),
            '--image-threads', str(min(cfg['compute']['io_workers'], 16)), '--timeout', str(options['timeout']),
            '--reconciliation-root', str(run / 'reconciliation'), '--selection-manifest', scope['selection_manifest']['path'],
            '--stop-file', str(run / 'STOP'), '--progress-interval', '5']
    for item in scope['inputs']:
        args += ['--input', item['path']]
    return args


@contextmanager
def epoch_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def install_transport_repair(path, epoch, expected_sha):
    """Load an explicit, audited local adapter; the frozen files remain intact."""
    import importlib.util
    if sha(path) != expected_sha:
        raise ValueError('transport_repair_source_changed')
    spec = importlib.util.spec_from_file_location('operational_review_transport_repair', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest = module.install(manifest_dir=epoch / 'transport_repair')
    record = {'path': str(path), 'source_sha256': expected_sha, 'manifest': manifest}
    target = epoch / 'TRANSPORT_REPAIR.json'
    if target.exists():
        previous = json.loads(target.read_text())
        if previous['path'] != record['path'] or previous['source_sha256'] != expected_sha:
            raise ValueError('transport_repair_identity_changed')
    else:
        write_new(target, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', 'verify'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--epoch', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--transport-repair', type=Path, help='Explicit operations-only audited image transport adapter')
    args = parser.parse_args(argv)
    run, epoch = args.run.resolve(), args.epoch.resolve()
    if not 1 <= args.concurrency <= 64:
        raise ValueError('recovery_concurrency_must_be_1_to_64')
    if epoch == run or run / 'operations' not in epoch.parents:
        raise ValueError('epoch_must_be_new_operations_subdirectory')
    api, identity = load_frozen(run)
    transport = ({'path': str(args.transport_repair.resolve()), 'source_sha256': sha(args.transport_repair)}
                 if args.transport_repair else None)
    root = run / 'api_review'
    api_args = make_argv(run, identity, args.concurrency)
    if args.command == 'plan':
        with connect(root, readonly=True) as db:
            before = snapshot(db)
        print(canonical({'assets': len(before['assets']), 'reset_requests': sum(r['state'] == 'failed' for r in before['requests']),
                         'protected': before['protected'], 'argv': api_args, 'transport_repair': transport}))
        return
    with epoch_lock(epoch.parent / (epoch.name + '.lock')):
        if args.command == 'verify':
            before = json.loads((epoch / 'BEFORE.json').read_text())['snapshot']
            with connect(root, readonly=True) as db:
                result = verify_after(db, before)
            print(canonical(result))
            return 0 if result['complete'] else 2
        with api.output_lock(root):
            with connect(root) as db:
                if not epoch.exists():
                    before = snapshot(db)
                    if not before['assets']:
                        raise ValueError('no_unresolved_assets')
                    epoch.mkdir()
                    archive = epoch / 'archive'; archive.mkdir()
                    for name in ('INPUT.json', 'COMPLETE.json', 'SUMMARY.json', 'PROGRESS.json', 'decisions.parquet'):
                        shutil.copy2(root / name, archive / name)
                    # Failed cache entries can be overwritten by a real retry. Preserve them all.
                    for path in (root / 'cache').rglob('*.failure.json'):
                        target = archive / path.relative_to(root)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
                    record = {'schema': 'api_review.operational_retry_epoch.v1', 'created_at': time.time(),
                              'reason': 'Resume deferred reviews and one explicit new attempt epoch for exhausted reviews',
                              'operator_script_sha256': sha(__file__), 'input_sha256': sha(root / 'INPUT.json'),
                              'frozen_screener_sha256': identity['code_sha256'], 'snapshot': before, 'api_argv': api_args,
                              'transport_repair': transport,
                              'archive': {str(p.relative_to(archive)): sha(p) for p in archive.rglob('*') if p.is_file()}}
                    write_new(epoch / 'BEFORE.json', record)
                record = json.loads((epoch / 'BEFORE.json').read_text()); before = record['snapshot']
                if (record['input_sha256'] != sha(root / 'INPUT.json') or record['api_argv'] != api_args
                        or record.get('transport_repair') != transport):
                    raise ValueError('retry_epoch_identity_changed')
                epoch_sha = sha(epoch / 'BEFORE.json')
                marker = 'operational_retry_epoch:' + epoch_sha
                if not db.execute('SELECT 1 FROM metrics WHERE name=?', (marker,)).fetchone():
                    reset = apply_epoch(db, before, epoch_sha)
                else:
                    verify_after(db, before)
                    reset = [r['asset_id'] for r in before['requests'] if r['state'] == 'failed']
                if not (epoch / 'APPLIED.json').exists():
                    write_new(epoch / 'APPLIED.json', {'epoch_sha256': epoch_sha, 'reset_asset_ids': reset,
                                                      'time': time.time(), 'unchanged_successful_reviews': before['protected']})
        # The original function retains its own output lock, identity checks,
        # receipt validation, strict parser and export/COMPLETE generation.
        if transport is not None:
            install_transport_repair(Path(transport['path']), epoch, transport['source_sha256'])
        api.main(api_args)
        with connect(root, readonly=True) as db:
            result = verify_after(db, before)
        result['input_unchanged'] = sha(root / 'INPUT.json') == record['input_sha256']
        if not result['input_unchanged']:
            raise ValueError('screening_input_changed_during_retry')
        write_new(epoch / f'AFTER-{time.time_ns()}.json', result)
        print(canonical(result))
        return 0 if result['complete'] else 2


if __name__ == '__main__':
    sys.exit(main())
