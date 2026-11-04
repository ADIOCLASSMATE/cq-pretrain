"""Bounded recovery tests using fake CLI receipts; never submits model calls."""
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from .farm import Farm, atomic_json, canonical, digest, exclusive, file_sha
from .recover_transport import audited_requeue, PREFIX


def fake_failure(items, directory, message='exceeded retry limit, last status: 429 Too Many Requests'):
    paths = {name: directory / filename for name, filename in
             [('events', 'events.jsonl.gz'), ('last_message', 'last_message.json'),
              ('decode_evidence', 'decode_evidence.json'), ('stderr', 'stderr.log')]}
    paths['last_message'].write_text('')
    paths['stderr'].write_text('Reading prompt from stdin...\n')
    atomic_json(paths['decode_evidence'], [{'item_id': i['item_id'], 'decoded': True} for i in items])
    thread = 'fake-thread-' + directory.name
    with gzip.open(paths['events'], 'wt') as stream:
        for value in [{'type': 'thread.started', 'thread_id': thread}, {'type': 'turn.started'},
                      {'type': 'error', 'message': message}, {'type': 'turn.failed', 'error': {'message': message}}]:
            stream.write(canonical(value) + '\n')
    metadata = {'transport_ok': False, 'usage': None, 'exit_code': 1, 'thread_id': thread,
                'attached_item_ids': [i['item_id'] for i in items],
                'artifacts': {k: {'path': str(p), 'sha256': file_sha(p)} for k, p in paths.items()}}
    return {'results': [], 'errors': {i['item_id']: 'codex_exit:429' for i in items},
            'transport_ok': False, 'error_type': 'codex_exit', 'metadata': metadata}


class FakeWorker:
    def __init__(self):
        self.calls = []
    def run_batch(self, items, directory, config):
        self.calls.append([i['item_id'] for i in items])
        return fake_failure(items, directory)
    def validate_result(self, value, aid):
        raise AssertionError('there is no successful fake result')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'farm'; self.root.mkdir()
        self.ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(4)]
        source = self.root.parent / 'input.jsonl'
        self.items = [{'item_id': aid, 'queue_order': i, 'source_ref': {'path': f'/fake/{i}.tar'}}
                      for i, aid in enumerate(self.ids)]
        source.write_text(''.join(canonical(i) + '\n' for i in self.items))
        with exclusive(self.root):
            farm = Farm(self.root); farm.initialize(source); farm.pin_runtime()
            for i, item in enumerate(self.items[:3]):
                bid = f'old-{i}'; directory = self.root / 'batches' / bid; directory.mkdir(parents=True)
                outcome = fake_failure([item], directory)
                atomic_json(directory / 'outcome.json', outcome)
                farm.db.execute('''INSERT INTO batches(batch_id,item_ids_json,status,directory,started_at,
                                  model,reasoning_effort,metadata_json,outcome_sha256,error_type)
                                  VALUES (?,?,'failed',?,1,'gpt-5.6-sol','low',?,?,'codex_exit')''',
                                (bid, canonical([item['item_id']]), str(directory), canonical(outcome['metadata']),
                                 file_sha(directory / 'outcome.json')))
                farm.db.execute("UPDATE items SET status='failed',attempts=5,batch_id=?,error_type='codex_exit' WHERE item_id=?",
                                (bid, item['item_id']))
            farm.db.execute("UPDATE items SET status='succeeded',result_json='{}',result_sha256='preserve-me' WHERE item_id=?",
                            (self.ids[3],))
            directory = self.root / 'exports' / 'old'; directory.mkdir(parents=True)
            base = {'source': farm.metadata('source'), 'runtime_binding_sha256': digest(farm.metadata('runtime_binding'))}
            atomic_json(directory / 'manifest.json', {**base, 'records': 1})
            atomic_json(directory / 'AUDIT.json', {**base, 'mode': 'strict', 'passed': True, 'skipped_gates': [],
                        'export_checked': True, 'export_manifest_sha256': file_sha(directory / 'manifest.json')})
            export = {'complete': False, 'audit_mode': 'strict', 'skipped_gates': [], 'records': 1,
                      'manifest': str(directory / 'manifest.json'), 'manifest_sha256': file_sha(directory / 'manifest.json'),
                      'audit': str(directory / 'AUDIT.json'), 'audit_sha256': file_sha(directory / 'AUDIT.json')}
            atomic_json(self.root / 'EXPORT.json', export)
            atomic_json(self.root / 'final_report.json', {'complete': False, 'valid_results': 1})
            farm.put('complete', export); farm.db.commit(); farm.close()

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        with sqlite3.connect(self.root / 'state.sqlite') as db:
            db.row_factory = sqlite3.Row
            return {row['item_id']: dict(row) for row in db.execute('SELECT * FROM items')}

    def recover(self, ids=None, recovery_id='explicit-429-once'):
        return audited_requeue(self.root, ids or self.ids[:3], 'Explicit extra attempt for HTTP 429 exhaustion', recovery_id)

    def test_requeue_preserves_attempts_success_history_and_archives(self):
        before = self.rows()
        raw = file_sha(self.root / 'batches/old-0/events.jsonl.gz')
        old_export = file_sha(self.root / 'EXPORT.json')
        receipt = self.recover()
        after = self.rows()
        self.assertEqual(receipt['original_total_attempts'], 15)
        self.assertEqual(receipt['extra_attempts_per_item'], 1)
        self.assertEqual(file_sha(receipt['code_path']), receipt['code_sha256'])
        for aid in self.ids[:3]:
            self.assertEqual(after[aid]['status'], 'retry'); self.assertEqual(after[aid]['attempts'], 5)
            self.assertEqual(after[aid]['batch_id'], before[aid]['batch_id'])
        self.assertEqual(before[self.ids[3]], after[self.ids[3]])
        self.assertEqual(raw, file_sha(self.root / 'batches/old-0/events.jsonl.gz'))
        self.assertEqual(old_export, file_sha(self.root / 'EXPORT.json'))
        snapshot = json.loads(Path(receipt['prepared_snapshot']['path']).read_text())
        self.assertEqual(len(snapshot['old_items']), 3)
        self.assertEqual(snapshot['backups']['EXPORT.json']['sha256'], old_export)

    def test_repeated_id_is_noop_and_different_request_rejected(self):
        self.recover(); before = self.rows()
        second = self.recover()
        self.assertTrue(second['already_applied']); self.assertEqual(second['new_authorizations'], 0)
        self.assertEqual(before, self.rows())
        with self.assertRaisesRegex(ValueError, 'different request'):
            self.recover(self.ids[:1])

    def test_real_farm_consumes_exactly_one_extra_attempt_without_core_changes(self):
        self.recover()
        worker = FakeWorker()
        with exclusive(self.root):
            farm = Farm(self.root, worker=worker)
            farm.run(workers=1, initial_workers=1, batch_size=1)
            farm.close()
        self.assertEqual(len(worker.calls), 3)
        for aid in self.ids[:3]:
            self.assertEqual(self.rows()[aid]['status'], 'failed')
            self.assertEqual(self.rows()[aid]['attempts'], 6)
        self.assertEqual(self.rows()[self.ids[3]]['result_sha256'], 'preserve-me')
        with self.assertRaisesRegex(ValueError, 'cumulative recovery limit'):
            self.recover(self.ids[:1], recovery_id='second-extra-not-allowed')

    def test_raw_tamper_rejected_and_no_db_change(self):
        before = self.rows()
        (self.root / 'batches/old-0/events.jsonl.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.recover()
        self.assertEqual(before, self.rows())
        self.assertFalse((self.root / 'recovery').exists())

    def test_content_refusal_even_with_rebound_raw_is_not_429(self):
        directory = self.root / 'batches/old-0'
        outcome = fake_failure([self.items[0]], directory, message='refused content')
        atomic_json(directory / 'outcome.json', outcome)
        with sqlite3.connect(self.root / 'state.sqlite') as db:
            db.execute("UPDATE batches SET metadata_json=?,outcome_sha256=? WHERE batch_id='old-0'",
                       (canonical(outcome['metadata']), file_sha(directory / 'outcome.json')))
        with self.assertRaisesRegex(ValueError, 'no exclusive 429'):
            self.recover()
        self.assertEqual(self.rows()[self.ids[0]]['status'], 'failed')

    def test_unfinished_and_completed_farms_rejected(self):
        with sqlite3.connect(self.root / 'state.sqlite') as db:
            db.execute("UPDATE items SET status='retry' WHERE item_id=?", (self.ids[0],))
        with self.assertRaisesRegex(ValueError, 'unfinished work'):
            self.recover()
        atomic_json(self.root / 'COMPLETE.json', {'complete': True})
        with self.assertRaisesRegex(ValueError, 'completed farm'):
            self.recover()

    def test_more_than_three_and_repeat_item_ids_rejected(self):
        for ids in (self.ids, [self.ids[0], self.ids[0]]):
            with self.assertRaisesRegex(ValueError, 'one to three unique'):
                self.recover(ids)

    def test_commit_receipt_missing_repeat_cannot_grant_new_budget(self):
        self.recover()
        (self.root / 'recovery/explicit-429-once/APPLIED.json').unlink()
        receipt = self.recover()
        self.assertTrue(receipt['already_applied'])
        with sqlite3.connect(self.root / 'state.sqlite') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM metadata WHERE key LIKE ?', (PREFIX + '%',)).fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
