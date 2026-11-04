"""Task-scoped selection must not let image-wide T2I rejection erase I2T."""
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from corpus_pipeline.postprocess.reconcile import artifact, digest, file_sha
from . import training_release as release
from .test_training_release import Fixture, sha


def task_part(fixture, name, rows):
    fixture.part(name, rows)
    part = fixture.parts[-1]
    path = fixture.root/'artifacts'/name/'attempt/records.parquet'
    table = pq.ParquetFile(path).read()
    for column, kind in [('processing_policy', pa.string()), ('task_route', pa.string()),
                         ('image_loss_allowed', pa.bool_())]:
        values = pa.array([row.get(column) for row in rows], type=kind)
        if column in table.column_names:
            table = table.set_column(table.column_names.index(column), column, values)
        else:
            table = table.append_column(column, values)
    pq.write_table(table, path)
    part['artifacts'] = [artifact(path, fixture.root) if Path(item['path']).name == path.name else item
                         for item in part['artifacts']]


def opted(name, images, text, route='i2t', **extra):
    return {'name': name, 'images': images, 'text': text, 'processing_policy': release.TASK_POLICY,
            'task_route': route, 'image_loss_allowed': route != 'i2t',
            'status': 'i2t_dedup_only' if route == 'i2t' else 'frontpass_pass_pending_global_dedup', **extra}


class TaskPolicyReleaseTests(unittest.TestCase):
    def test_i2t_keeps_ai_small_quality_and_distinct_supervision_but_quarantines_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'input'; root.mkdir(); fixture = Fixture(root)
            ai = fixture.asset('ai', action='exclude', status='rejected_source')
            small = fixture.asset('small', action='exclude', status='rejected_too_small')
            broken = fixture.asset('broken', action='pending', status='pending_integrity')
            clean = fixture.asset('clean')
            rows = [opted('i2t-a', [ai], 'caption one'), opted('i2t-duplicate', [ai], 'caption one'),
                    opted('i2t-distinct', [ai], 'caption two'),
                    opted('t2i-ai', [ai], 'caption one', route='t2i', status='rejected'),
                    opted('i2t-small', [small], 'small but readable'),
                    opted('i2t-broken', [broken], 'unreadable'),
                    opted('i2t-binding', [clean], 'ambiguous', status='pending',
                          reason_codes=['unresolved_original_image_binding']),
                    opted('i2t-quality', [clean], 'keep original answer'),
                    {'name': 'legacy-ai', 'images': [ai], 'text': 'old selection'},
                    opted('i2t-order-ab', [ai, clean], 'ordered'),
                    opted('i2t-order-ba', [clean, ai], 'ordered'),
                    opted('i2t-missing', [sha('missing')], 'missing asset')]
            task_part(fixture, 'task0', rows); fixture.finish()
            original = pq.ParquetFile(root/'task0.parquet').read().to_pylist()
            quality_path = root/'quality.parquet'
            pq.write_table(pa.Table.from_pylist([{'source_record_sha256': digest(original[7]),
                                                'reason': 'bad_caption'}]), quality_path)
            quality = {'path': str(quality_path), 'sha256': file_sha(quality_path),
                       'size_bytes': quality_path.stat().st_size}
            output = Path(tmp)/'output'
            result = release.build(root, root/'assets.json', output, threads=1, memory_limit='256MB',
                                   record_quality_files=[quality])
            self.assertEqual(result['records'], len(rows))
            selected = {row['sample_id']: row for decision in ('keep', 'exclude', 'quarantine')
                        for row in release.iter_candidate_references(output, decision)}
            def get(name): return selected[sha(name)]
            self.assertEqual(sum(get(n)['decision'] == 'keep' for n in ('i2t-a', 'i2t-duplicate')), 1)
            for name in ('i2t-distinct', 'i2t-small', 'i2t-quality', 'i2t-order-ab', 'i2t-order-ba'):
                self.assertEqual(get(name)['decision'], 'keep', name)
                self.assertIs(get(name)['image_loss_allowed'], False)
                self.assertFalse(get(name)['unresolved_screening'])
            self.assertEqual(get('i2t-quality')['observed_record_quality_reason'], 'bad_caption')
            for name in ('i2t-broken', 'i2t-binding', 'i2t-missing'):
                self.assertEqual(get(name)['decision'], 'quarantine', name)
            for name in ('t2i-ai', 'legacy-ai'):
                self.assertEqual(get(name)['decision'], 'exclude', name)
            self.assertIsNone(get('legacy-ai')['image_loss_allowed'])
            self.assertEqual(len(fixture.assets), 4)  # Task routing never duplicates global asset identities.

    def test_task_routes_remain_distinct_and_false_permission_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'input'; root.mkdir(); fixture = Fixture(root)
            aid = fixture.asset('a')
            task_part(fixture, 'task0', [opted('understand', [aid], 'same text'),
                                        opted('generate', [aid], 'same text', route='t2i')])
            fixture.finish()
            output = Path(tmp)/'output'
            result = release.build(root, root/'assets.json', output, threads=1, memory_limit='256MB')
            self.assertEqual(result['counts'], {'keep': 2})
        for fields in ({'image_loss_allowed': True}, {'processing_policy': None}):
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'input'; root.mkdir(); fixture = Fixture(root)
                aid = fixture.asset('a')
                task_part(fixture, 'task0', [opted('invalid', [aid], 'x', **fields)])
                fixture.finish()
                with self.assertRaisesRegex(ValueError, 'Inconsistent I2T'):
                    release.build(root, root/'assets.json', Path(tmp)/'out', threads=1, memory_limit='256MB')

    def test_shared_detector_pending_is_task_scoped_but_structure_and_corruption_are_not_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'input'; root.mkdir(); fixture = Fixture(root)
            shared = fixture.asset('shared', action='pending', status='pending_integrity')
            fixture.assets[shared]['i2t_integrity_pending'] = False
            broken = fixture.asset('broken', action='pending', status='pending_integrity')
            fixture.assets[broken]['i2t_integrity_pending'] = True
            rows = [opted('i2t-valid', [shared], 'valid'),
                    opted('t2i-pending', [shared], 'generation', route='t2i'),
                    opted('i2t-broken', [broken], 'unreadable'),
                    opted('i2t-corrupt-rule', [shared], 'corrupt')]
            reasons = ['unreferenced_images', 'invalid_image_index', 'image_reference_unresolved',
                       'missing_text', 'unresolved_original_image_binding', 'unsupported_record_structure']
            rows.extend(opted(reason, [shared], reason, status='pending', reason_codes=[reason]) for reason in reasons)
            task_part(fixture, 'task0', rows); fixture.finish()
            original = pq.ParquetFile(root/'task0.parquet').read().to_pylist()
            quality_path = root/'quality.parquet'
            pq.write_table(pa.Table.from_pylist([{'source_record_sha256': digest(original[3]), 'reason': reason}
                                                 for reason in ('bad_caption', 'corrupt_image')]), quality_path)
            quality = {'path': str(quality_path), 'sha256': file_sha(quality_path),
                       'size_bytes': quality_path.stat().st_size}
            output = Path(tmp)/'out'
            release.build(root, root/'assets.json', output, threads=1, memory_limit='256MB', record_quality_files=[quality])
            selected = {row['sample_id']: row for decision in ('keep', 'exclude', 'quarantine')
                        for row in release.iter_candidate_references(output, decision)}
            self.assertEqual(selected[sha('i2t-valid')]['decision'], 'keep')
            for name in ('t2i-pending', 'i2t-broken', 'i2t-corrupt-rule', *reasons):
                self.assertEqual(selected[sha(name)]['decision'], 'quarantine', name)


if __name__ == '__main__':
    unittest.main()
