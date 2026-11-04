"""Explicit one-extra-attempt recovery for at most three exhausted 429 items.

Does not invoke Codex or change the original five-attempt runtime/config. The
farm's existing claim/commit increments 5 to 6, then terminates another failure.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from .farm import canonical, digest, exclusive, file_sha, atomic_json

PREFIX = 'transport_recovery:'
MAX_ITEMS = 3
ORIGINAL_ATTEMPTS = 5
AUTHORIZATION = ('用户已要求处理全部不确定项及技术失败；本次是显式审计的恢复批次，'
                 '仅为已耗尽五次且有429证据的条目额外授权一次，不改变原始重试预算。')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read_bound(path, sha, within=None):
    path = Path(path).resolve(strict=True)
    require(within is None or path.is_relative_to(within), 'artifact outside owned directory')
    require(file_sha(path) == sha, 'artifact hash mismatch: ' + path.name)
    return json.loads(path.read_text())


def _snapshot(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': file_sha(path), 'size_bytes': Path(path).stat().st_size}


def audited_requeue(root, item_ids, reason, recovery_id):
    root = Path(root).resolve(strict=True)
    require((root / 'state.sqlite').is_file(), 'missing existing farm database')
    require(isinstance(reason, str) and 8 <= len(reason.strip()) <= 2000, 'explicit recovery reason required')
    require(isinstance(recovery_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', recovery_id),
            'invalid recovery ID')
    ids = sorted(item_ids)
    require(1 <= len(ids) <= MAX_ITEMS and len(set(ids)) == len(ids)
            and all(isinstance(aid, str) and re.fullmatch('[a-f0-9]{64}', aid) for aid in ids),
            'require one to three unique SHA256 item IDs')
    reason = reason.strip()
    with exclusive(root):
        db = sqlite3.connect((root / 'state.sqlite').as_uri() + '?mode=rw', uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            metadata = {r['key']: json.loads(r['value']) for r in db.execute('SELECT key,value FROM metadata')}
            prior = metadata.get(PREFIX + recovery_id)
            if prior is not None:
                require(prior['item_ids'] == ids and prior['reason'] == reason, 'recovery ID reused with different request')
                db.rollback()
                return {**prior, 'already_applied': True, 'new_authorizations': 0}
            authorized = [v for k, v in metadata.items() if k.startswith(PREFIX)]
            require(sum(len(v['item_ids']) for v in authorized) + len(ids) <= MAX_ITEMS,
                    'cumulative recovery limit is three items')
            require(not set(ids).intersection(aid for v in authorized for aid in v['item_ids']),
                    'item already received its extra attempt')
            require(not (root / 'COMPLETE.json').exists(), 'completed farm cannot be requeued')
            require(db.execute("SELECT 1 FROM items WHERE status IN ('pending','running','retry') LIMIT 1").fetchone() is None,
                    'farm still has unfinished work')
            runtime, source = metadata['runtime_binding'], metadata['source']
            config = json.loads((root / 'config.json').read_text()) if (root / 'config.json').exists() else {}
            config.update(model='gpt-5.6-sol', reasoning_effort='low')
            require(runtime['config'] == config and int(config.get('max_attempts', 5)) == ORIGINAL_ATTEMPTS,
                    'original runtime/config budget changed')
            for path, sha in runtime['code'].items():
                require(file_sha(path) == sha, 'frozen runtime code changed')
            require(file_sha(source['path']) == source['sha256'], 'source manifest changed')
            export_path = root / 'EXPORT.json'
            export = json.loads(export_path.read_text())
            require(export == metadata.get('complete') and export.get('complete') is False
                    and export.get('audit_mode') == 'strict' and export.get('skipped_gates') == [],
                    'recovery requires current strict partial export')
            audit = _read_bound(export['audit'], export['audit_sha256'], root / 'exports')
            manifest = _read_bound(export['manifest'], export['manifest_sha256'], root / 'exports')
            require(audit.get('passed') is True and audit.get('mode') == 'strict'
                    and audit.get('skipped_gates') == [] and audit.get('export_checked') is True
                    and audit.get('export_manifest_sha256') == export['manifest_sha256']
                    and audit.get('source') == source and manifest.get('source') == source
                    and audit.get('runtime_binding_sha256') == digest(runtime)
                    and manifest.get('runtime_binding_sha256') == digest(runtime), 'strict export binding mismatch')
            old_rows, old_batches, evidence = [], {}, []
            for aid in ids:
                row = db.execute('SELECT * FROM items WHERE item_id=?', (aid,)).fetchone()
                require(row is not None, 'unknown item')
                row = dict(row)
                require(row['status'] == 'failed' and row['attempts'] == ORIGINAL_ATTEMPTS
                        and row['error_type'] == 'codex_exit' and row['result_json'] is None
                        and row['result_sha256'] is None, 'item is not exhausted codex_exit without valid result')
                require(digest(json.loads(row['input_json'])) == row['input_sha256'], 'item input hash mismatch')
                batch = db.execute('SELECT * FROM batches WHERE batch_id=?', (row['batch_id'],)).fetchone()
                require(batch is not None, 'missing failed batch')
                batch = dict(batch); directory = Path(batch['directory']).resolve(strict=True)
                require(directory.is_relative_to(root / 'batches') and batch['status'] == 'failed'
                        and batch['error_type'] == 'codex_exit', 'invalid failed batch')
                m = json.loads(batch['metadata_json'])
                require(m.get('transport_ok') is False and m.get('usage') is None
                        and m.get('exit_code') == 1 and aid in m.get('attached_item_ids', []),
                        'not a pre-result CLI transport failure')
                outcome = _read_bound(directory / 'outcome.json', batch['outcome_sha256'], directory)
                require(outcome.get('metadata') == m and not outcome.get('results'), 'failed outcome binding mismatch')
                artifacts = m['artifacts']
                paths = {}
                for name in ('events', 'last_message', 'decode_evidence', 'stderr'):
                    record = artifacts[name]; path = Path(record['path']).resolve(strict=True)
                    require(path.is_relative_to(directory) and file_sha(path) == record['sha256'],
                            'raw failure artifact changed')
                    paths[name] = path
                require(paths['last_message'].stat().st_size == 0, 'failure already contains a model response')
                decoded = json.loads(paths['decode_evidence'].read_text())
                require(any(v.get('item_id') == aid and v.get('decoded') is True for v in decoded),
                        'input did not decode successfully')
                with gzip.open(paths['events'], 'rt', encoding='utf-8') as stream:
                    events = [json.loads(line) for line in stream]
                errors = [e.get('message', '') if e.get('type') == 'error' else e.get('error', {}).get('message', '')
                          for e in events if e.get('type') in ('error', 'turn.failed')]
                require(errors and all(re.search(r'\b429\s+Too Many Requests\b', message) for message in errors)
                        and any(e.get('type') == 'turn.failed' for e in events)
                        and not any(e.get('type') == 'turn.completed' or e.get('item', {}).get('type') == 'agent_message'
                                    for e in events), 'no exclusive 429 transport-failure evidence')
                require([e.get('thread_id') for e in events if e.get('type') == 'thread.started'] == [m.get('thread_id')]
                        and m.get('thread_id'), 'failure thread identity mismatch')
                old_rows.append(row); old_batches[row['batch_id']] = batch
                evidence.append({'item_id': aid, 'batch_id': row['batch_id'], 'events': artifacts['events'],
                                 'messages': errors, 'original_attempts': row['attempts']})
            archive = root / 'recovery' / recovery_id
            archive.mkdir(parents=True, exist_ok=False)
            backups = {}
            for name in ('EXPORT.json', 'AUDIT.json', 'final_report.json', 'final_report.md', 'config.json',
                         'SUPERVISOR_STATUS.json'):
                path = root / name
                if path.exists():
                    require(path.stat().st_size <= 16 << 20, 'unexpectedly large small-receipt backup')
                    backups[name] = _snapshot(archive / name, path.read_bytes())
            snapshot = {'source': source, 'runtime_binding': runtime, 'old_completion_metadata': metadata.get('complete'),
                        'old_items': old_rows, 'old_batches': old_batches, 'backups': backups, 'failure_evidence': evidence}
            prepared = _snapshot(archive / 'BEFORE.json', (canonical(snapshot) + '\n').encode())
            authorization = {'schema_version': 'corpus.codex_review.transport_recovery.v1', 'recovery_id': recovery_id,
                             'code_path': str(Path(__file__).resolve()), 'code_sha256': file_sha(__file__),
                             'item_ids': ids, 'reason': reason, 'user_prior_instruction': AUTHORIZATION,
                             'original_attempt_limit': ORIGINAL_ATTEMPTS, 'extra_attempts_per_item': 1,
                             'original_attempts': {row['item_id']: row['attempts'] for row in old_rows},
                             'original_total_attempts': sum(row['attempts'] for row in old_rows),
                             'source': source, 'runtime_binding_sha256': digest(runtime),
                             'prepared_snapshot': prepared, 'authorized_at_epoch': time.time(),
                             'model_submitted': False, 'original_runtime_or_budget_modified': False}
            db.execute('INSERT INTO metadata(key,value) VALUES (?,?)', (PREFIX + recovery_id, canonical(authorization)))
            for aid in ids:
                changed = db.execute("""UPDATE items SET status='retry',retry_not_before=0,updated_at=?
                    WHERE item_id=? AND status='failed' AND attempts=5 AND error_type='codex_exit'""",
                                     (authorization['authorized_at_epoch'], aid)).rowcount
                require(changed == 1, 'requeue compare-and-set failed')
            db.commit()
            # If a crash occurs before this convenience receipt, metadata and
            # BEFORE remain authoritative and a repeated ID grants zero budget.
            atomic_json(archive / 'APPLIED.json', authorization)
            return {**authorization, 'already_applied': False, 'new_authorizations': len(ids)}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--item-id', action='append', required=True)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--recovery-id', required=True)
    args = parser.parse_args()
    print(canonical(audited_requeue(args.root, args.item_id, args.reason, args.recovery_id)))


if __name__ == '__main__':
    main()
