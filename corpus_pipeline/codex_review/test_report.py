"""Local fixture-only report tests: no model, network, or production scans."""
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from .farm import Farm, atomic_json, exclusive
from .report import MODEL, EFFORT, canonical, digest, file_sha, generate_report
from .worker import validate_result


def result_for(aid, kind):
    value = {'item_id': aid, 'decision': 'pending', 'assessment': 'uncertain', 'confidence': 'low',
             'image_kind': 'photograph', 'quality': 'degraded_usable',
             'quality_reason': 'Still enough visible semantic content.', 'rejection_reason': 'none',
             'visible_evidence': [], 'alternative_explanation': 'Ordinary imaging is still plausible.',
             'needs_external_evidence': True, 'reason': 'There is insufficient visual evidence.'}
    if kind == 'admitted':
        value.update(decision='admit', assessment='likely_non_ai', confidence='medium', quality='usable',
                     needs_external_evidence=False, visible_evidence=[
                         {'location': 'foreground', 'observation': 'Specific ordinary detail visible.', 'supports': 'non_ai'}])
    elif kind == 'ai':
        value.update(decision='reject', assessment='likely_ai_generated', confidence='high',
                     rejection_reason='ai_generated', needs_external_evidence=False, visible_evidence=[
                         {'location': 'left object', 'observation': 'First independently described visible anomaly.', 'supports': 'ai'},
                         {'location': 'right object', 'observation': 'Second separately described visible anomaly.', 'supports': 'ai'}])
    elif kind == 'quality':
        value.update(decision='reject', confidence='medium', quality='unusable', rejection_reason='quality_unusable',
                     visible_evidence=[{'location': 'entire image', 'observation': 'Content is entirely destroyed.', 'supports': 'quality'}])
    validate_result(value, aid)
    return value


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'farm'
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, partial=False):
        """Construct small self-consistent audited-export contract fixtures."""
        kinds = ['admitted', 'ai', 'quality', 'pending'] + (['failed', 'unprocessed', 'retry'] if partial else [])
        items = []
        for index, kind in enumerate(kinds):
            aid = hashlib.sha256(str(index).encode()).hexdigest()
            prior = {'decision': 'pending', 'model_recommendation': 'pending',
                     'reviews': {'primary': {'assessment': 'uncertain', 'confidence': 'low'}}}
            items.append({'item_id': aid, 'queue_order': index, 'source_ref': {'path': f'/fake/{index}.tar'},
                          'provenance': {'origin': 'unknown'}, 'category': 'model_pending',
                          'prior_review': prior, 'prior_review_sha256': digest(prior)})
        source_path = self.root.parent / 'candidates.jsonl'
        source_path.write_text(''.join(canonical(item) + '\n' for item in items))
        with exclusive(self.root):
            farm = Farm(self.root)
            farm.initialize(source_path)
            farm.pin_runtime()
            rows = []
            for index, (item, kind) in enumerate(zip(items, kinds)):
                aid = item['item_id']
                if kind in {'failed', 'retry', 'unprocessed'}:
                    status = 'pending' if kind == 'unprocessed' else kind
                    farm.db.execute('UPDATE items SET status=?,attempts=?,error_type=? WHERE item_id=?',
                                    (status, 0 if kind == 'unprocessed' else 3, None if kind == 'unprocessed' else 'decode_error', aid))
                    continue
                result = result_for(aid, kind)
                batch = f'batch-{index}'
                directory = self.root / 'batches' / batch
                directory.mkdir(parents=True)
                batch_metadata = {'thread_id': f'fake-thread-{index}', 'artifacts': {}, 'prompt_version': 'v2_test',
                                  'usage': {'input_tokens': 10, 'output_tokens': 10}}
                farm.db.execute('''INSERT INTO batches(batch_id,item_ids_json,status,directory,started_at,
                                model,reasoning_effort,metadata_json,outcome_sha256) VALUES (?,?,?,?,?,?,?,?,?)''',
                                (batch, canonical([aid]), 'succeeded', str(directory), 1, MODEL, EFFORT,
                                 canonical(batch_metadata), '1' * 64))
                farm.db.execute('''UPDATE items SET status='succeeded',attempts=1,batch_id=?,result_json=?,result_sha256=?
                                WHERE item_id=?''', (batch, canonical(result), digest(result), aid))
                rows.append({'item_id': aid, 'batch_id': batch, 'source': item, 'result': result})
            source = farm.metadata('source')
            runtime = farm.metadata('runtime_binding')
            directory = self.root / 'exports' / 'fixture-immutable'
            directory.mkdir(parents=True)
            shard = directory / 'part-00000.jsonl.gz'
            with gzip.open(shard, 'wt') as stream:
                for row in rows:
                    stream.write(canonical(row) + '\n')
            manifest = {'schema_version': 'corpus.codex_review_farm.v1', 'source': source, 'runtime_binding_sha256': digest(runtime),
                        'model': MODEL, 'reasoning_effort': EFFORT, 'audit_mode': 'strict',
                        'complete': not partial, 'records': len(rows),
                        'shards': [{'path': shard.name, 'sha256': file_sha(shard),
                                    'compressed_bytes': shard.stat().st_size, 'records': len(rows)}]}
            atomic_json(directory / 'manifest.json', manifest)
            content = {'schema_version': 'corpus.codex_review_farm.v1', 'mode': 'strict', 'passed': True, 'skipped_gates': [], 'source': source,
                       'runtime_binding_sha256': digest(runtime), 'expected': len(items), 'succeeded': len(rows),
                       'audited_items': len(rows), 'complete_coverage': not partial}
            atomic_json(directory / 'CONTENT_AUDIT.json', content)
            audit = {**content, 'export_checked': True, 'export_manifest': str(directory / 'manifest.json'),
                     'export_manifest_sha256': file_sha(directory / 'manifest.json'),
                     'content_audit_path': str(directory / 'CONTENT_AUDIT.json'),
                     'content_audit_sha256': file_sha(directory / 'CONTENT_AUDIT.json')}
            atomic_json(directory / 'AUDIT.json', audit)
            pointer = {'complete': not partial, 'audit_mode': 'strict', 'skipped_gates': [], 'records': len(rows),
                       'manifest': str(directory / 'manifest.json'), 'manifest_sha256': file_sha(directory / 'manifest.json'),
                       'audit': str(directory / 'AUDIT.json'), 'audit_sha256': file_sha(directory / 'AUDIT.json')}
            atomic_json(self.root / 'EXPORT.json', pointer)
            if not partial:
                atomic_json(self.root / 'COMPLETE.json', pointer)
            farm.put('complete', pointer)
            farm.db.commit(); farm.close()
        self.items, self.directory = items, directory
        return pointer

    def read_gzip(self, filename):
        with gzip.open(self.root / filename, 'rt') as stream:
            return [json.loads(line) for line in stream]

    def test_complete_categories_quality_not_ai_and_provenance(self):
        self.fixture()
        before = (self.root / 'state.sqlite').read_bytes()
        report = generate_report(self.root)
        self.assertTrue(report['complete'])
        self.assertEqual(report['counts'], {'ai_excluded': 1, 'quality_excluded': 1, 'admitted': 1,
                                           'visual_pending': 1, 'technical_failure': 0, 'unprocessed': 0})
        self.assertEqual(report['percent_of_valid_results']['ai_excluded'], 25)
        self.assertEqual(len(self.read_gzip('manual_review.jsonl.gz')), 1)
        self.assertEqual(self.read_gzip('technical_failures.jsonl.gz'), [])
        overlay = self.read_gzip('screening_overlay.jsonl.gz')
        self.assertEqual(len(overlay), 4)
        self.assertEqual(overlay[2]['rejection_reason'], 'quality_unusable')
        self.assertEqual(overlay[2]['result']['assessment'], 'uncertain')
        self.assertTrue(all(not row['source_certification'] for row in overlay))
        self.assertEqual(overlay[0]['prior_review_sha256'], self.items[0]['prior_review_sha256'])
        self.assertEqual(len(self.read_gzip('raw_provenance.jsonl.gz')), 4)
        self.assertEqual(before, (self.root / 'state.sqlite').read_bytes())
        for artifact in report['artifacts'].values():
            self.assertEqual(file_sha(artifact['path']), artifact['sha256'])

    def test_partial_separates_unprocessed_technical_and_visual(self):
        self.fixture(partial=True)
        report = generate_report(self.root)
        self.assertFalse(report['complete'])
        self.assertEqual(report['missing_valid_results'], 3)
        self.assertEqual(report['counts']['technical_failure'], 2)
        self.assertEqual(report['counts']['unprocessed'], 1)
        self.assertEqual(report['counts']['visual_pending'], 1)
        errors = self.read_gzip('technical_failures.jsonl.gz')
        self.assertEqual(len(errors), 3)
        self.assertTrue(all(row['decision'] is None for row in errors))
        self.assertEqual(len(self.read_gzip('screening_overlay.jsonl.gz')), 7)
        self.assertIn('不完整', (self.root / 'final_report.md').read_text())

    def test_tampered_immutable_audit_rejected_before_outputs(self):
        self.fixture()
        (self.directory / 'AUDIT.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            generate_report(self.root)
        self.assertFalse((self.root / 'final_report.json').exists())

    def test_tampered_source_is_rejected(self):
        self.fixture()
        (self.root.parent / 'candidates.jsonl').write_text('{}\n')
        with self.assertRaises((ValueError, KeyError)):
            generate_report(self.root)
        self.assertFalse((self.root / 'final_report.json').exists())

    def test_tampered_export_is_rejected(self):
        self.fixture()
        (self.directory / 'part-00000.jsonl.gz').write_bytes(b'not gzip')
        with self.assertRaisesRegex(ValueError, 'shard integrity'):
            generate_report(self.root)
        self.assertFalse((self.root / 'screening_overlay.jsonl.gz').exists())

    def test_changed_database_result_is_not_covered_by_old_audit(self):
        self.fixture()
        value = result_for(self.items[0]['item_id'], 'pending')
        with sqlite3.connect(self.root / 'state.sqlite') as db:
            db.execute('UPDATE items SET result_json=?,result_sha256=? WHERE item_id=?',
                       (canonical(value), digest(value), self.items[0]['item_id']))
        with self.assertRaisesRegex(ValueError, 'differs from strictly audited export'):
            generate_report(self.root)

    def test_repeated_report_is_atomic_and_gzip_deterministic(self):
        self.fixture()
        first = generate_report(self.root)
        second = generate_report(self.root)
        self.assertEqual(first['artifacts'], second['artifacts'])
        self.assertEqual(json.loads((self.root / 'final_report.json').read_text()), second)
        self.assertFalse(list(self.root.glob('.*.tmp')))

    def test_writer_lock_prevents_racing_report(self):
        self.fixture()
        with (self.root / 'farm.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                generate_report(self.root)

    def test_wrong_prior_hash_is_rejected_even_with_manifest_and_db_updated(self):
        self.fixture()
        # Source mutation alone fails the immutable source hash gate.
        item = self.items[0].copy(); item['prior_review_sha256'] = '0' * 64
        with sqlite3.connect(self.root / 'state.sqlite') as db:
            db.execute('UPDATE items SET input_json=?,input_sha256=? WHERE item_id=?',
                       (canonical(item), digest(item), item['item_id']))
        with self.assertRaisesRegex(ValueError, 'differs from frozen source'):
            generate_report(self.root)

    def test_absent_strict_completion_cannot_publish(self):
        self.fixture()
        (self.root / 'COMPLETE.json').unlink(); (self.root / 'EXPORT.json').unlink()
        with self.assertRaises(FileNotFoundError):
            generate_report(self.root)
        self.assertFalse((self.root / 'final_report.json').exists())

    def test_rehashed_failed_audit_still_cannot_publish(self):
        pointer = self.fixture()
        audit = json.loads((self.directory / 'AUDIT.json').read_text())
        audit['passed'] = False
        atomic_json(self.directory / 'AUDIT.json', audit)
        pointer['audit_sha256'] = file_sha(self.directory / 'AUDIT.json')
        atomic_json(self.root / 'COMPLETE.json', pointer)
        with self.assertRaisesRegex(ValueError, 'did not pass all gates'):
            generate_report(self.root)
        self.assertFalse((self.root / 'final_report.json').exists())

    def test_partial_cannot_claim_complete(self):
        pointer = self.fixture(partial=True)
        pointer['complete'] = True
        atomic_json(self.root / 'COMPLETE.json', pointer)
        with sqlite3.connect(self.root / 'state.sqlite') as db:
            db.execute("UPDATE metadata SET value=? WHERE key='complete'", (canonical(pointer),))
        with self.assertRaisesRegex(ValueError, 'coverage flags mismatch'):
            generate_report(self.root)


if __name__ == '__main__':
    unittest.main()
