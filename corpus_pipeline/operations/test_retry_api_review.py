import json
import sqlite3
import unittest

from corpus_pipeline.operations import retry_api_review as repair


class RetryEpochTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE assets(asset_id TEXT PRIMARY KEY, input_sha TEXT, input_json TEXT,
              state TEXT, decision TEXT, reason TEXT, decision_json TEXT, attempts INTEGER);
            CREATE TABLE requests(asset_id TEXT, role TEXT, request_sha TEXT, attempts INTEGER,
              state TEXT, receipt_path TEXT, receipt_sha TEXT, last_error TEXT, PRIMARY KEY(asset_id,role));
            CREATE TABLE metrics(name TEXT PRIMARY KEY, value INTEGER);
        ''')
        for aid, state, reason, request_state, attempts in (
            ('good', 'done', 'visual_annotation_only_requires_confirmation', 'ok', 1),
            ('deferred', 'retryable', 'serving_temporarily_unavailable', 'deferred', 0),
            ('exhausted', 'done', 'api_review_failure_exhausted', 'failed', 3),
        ):
            self.db.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?,?)',
                            (aid, 'input-' + aid, json.dumps({'asset_id': aid}), state, 'pending', reason, '{}', 1))
            self.db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?,?,?)',
                            (aid, 'primary', 'request-' + aid, attempts, request_state,
                             'receipt.json' if aid == 'good' else None, 'hash' if aid == 'good' else None,
                             None if aid == 'good' else 'ModelStageError'))
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_only_exhausted_request_is_reset_and_success_is_unchanged(self):
        before = repair.snapshot(self.db)
        deferred = dict(self.db.execute("SELECT * FROM requests WHERE asset_id='deferred'").fetchone())
        self.assertEqual(repair.apply_epoch(self.db, before, 'epoch-1'), ['exhausted'])
        self.assertEqual(dict(self.db.execute("SELECT * FROM requests WHERE asset_id='deferred'").fetchone()), deferred)
        self.assertEqual(repair.protected_digest(self.db, ['deferred', 'exhausted']), before['protected'])
        row = self.db.execute("SELECT state,attempts,request_sha FROM requests WHERE asset_id='exhausted'").fetchone()
        self.assertEqual(tuple(row), ('pending', 0, 'request-exhausted'))
        self.assertEqual(self.db.execute("SELECT attempts FROM assets WHERE asset_id='exhausted'").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT value FROM metrics WHERE name='operational_retry_epoch:epoch-1'").fetchone()[0], 1)

    def test_concurrent_change_refuses_entire_mutation(self):
        before = repair.snapshot(self.db)
        self.db.execute("UPDATE requests SET attempts=2 WHERE asset_id='exhausted'"); self.db.commit()
        with self.assertRaises(ValueError):
            repair.apply_epoch(self.db, before, 'epoch-1')
        self.assertEqual(self.db.execute("SELECT state FROM requests WHERE asset_id='exhausted'").fetchone()[0], 'failed')
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM metrics').fetchone()[0], 0)

    def test_epoch_cannot_reset_same_rows_twice(self):
        before = repair.snapshot(self.db)
        repair.apply_epoch(self.db, before, 'epoch-1')
        with self.assertRaises(ValueError):
            repair.apply_epoch(self.db, before, 'epoch-1')
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM metrics').fetchone()[0], 1)

    def test_receipt_bearing_failure_is_never_reset(self):
        self.db.execute("UPDATE requests SET receipt_path='old.json' WHERE asset_id='exhausted'")
        with self.assertRaisesRegex(ValueError, 'receipt_bearing'):
            repair.snapshot(self.db)

    def test_unknown_failure_is_not_silently_reopened(self):
        self.db.execute("UPDATE assets SET reason='saved_review_integrity_failure' WHERE asset_id='exhausted'")
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            repair.snapshot(self.db)

    def test_success_mutation_fails_verification(self):
        before = repair.snapshot(self.db)
        self.db.execute("UPDATE requests SET receipt_sha='changed' WHERE asset_id='good'")
        with self.assertRaisesRegex(ValueError, 'successful_reviews_changed'):
            repair.verify_after(self.db, before)

    def test_retry_identity_mutation_fails_verification(self):
        before = repair.snapshot(self.db)
        self.db.execute("UPDATE requests SET request_sha='changed' WHERE asset_id='deferred'")
        with self.assertRaisesRegex(ValueError, 'request_identity_changed'):
            repair.verify_after(self.db, before)

    def test_exhausted_does_not_count_as_completed(self):
        before = repair.snapshot(self.db)
        result = repair.verify_after(self.db, before)
        self.assertFalse(result['complete'])
        self.assertEqual(result['resolved'], 0)
        self.assertEqual(len(result['unresolved']), 2)

    def test_only_valid_real_success_states_complete_epoch(self):
        before = repair.snapshot(self.db)
        for aid in ('deferred', 'exhausted'):
            self.db.execute("UPDATE assets SET state='done',reason='visual_annotation_only_requires_confirmation' WHERE asset_id=?", (aid,))
            self.db.execute("UPDATE requests SET state='ok',receipt_path=?,receipt_sha=? WHERE asset_id=?", (aid + '.json', 'valid-' + aid, aid))
        result = repair.verify_after(self.db, before)
        self.assertTrue(result['complete'])
        self.assertEqual(result['resolved'], 2)


if __name__ == '__main__':
    unittest.main()
