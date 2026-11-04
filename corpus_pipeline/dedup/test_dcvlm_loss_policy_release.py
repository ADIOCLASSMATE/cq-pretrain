from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import dcvlm_loss_policy_release as release


FLAGS = {'PRIOR_EXCLUDE': 1, 'PRIOR_QUARANTINE': 2, 'DUPLICATE': 4,
         'FRONTPASS_REJECTION': 8, 'SCREENING_EXCLUSION': 16, 'UNRESOLVED_SCREENING': 32,
         'INTEGRITY': 64, 'ORIGINAL_BINDING': 128, 'SAFETY': 256, 'RECORD_QUALITY': 512,
         'NO_IMAGE': 1024, 'DUPLICATE_NOT_SELECTED': 2048, 'MISMATCH': 4096, 'REPAIRED': 8192}


def candidate(rows):
    defaults = {'decision': 'keep', 'decision_reason': 'retain_candidate', 'native_image_count': 1,
                'image_ids': ['a' * 64], 'reason_codes': [], 'status': 'frontpass_pass_pending_global_dedup',
                'confirmed_screen_excluded': False, 'unresolved_screening': False,
                'integrity_quarantine': False, 'confirmed_record_quality_reason': None,
                'record_signature': '0' * 64}
    records = [dict(defaults, record_id=str(i), sample_id=str(i), **row) for i, row in enumerate(rows)]
    return pa.Table.from_pylist(records)


class ScatterTests(unittest.TestCase):
    def files(self):
        return [{'source_path': '/first.parquet', 'rows': 3}, {'source_path': '/second.parquet', 'rows': 2}]

    def test_unsorted_batches_preserve_original_file_row_order(self):
        scatter = release.FlagScatter(self.files())
        scatter.consume(pa.array(['/second.parquet', '/first.parquet']), pa.array([1, 2]), np.array([8, 4]))
        scatter.consume(pa.array(['/first.parquet', '/second.parquet', '/first.parquet']),
                        pa.array([0, 0, 1]), np.array([1, 0, 2]))
        np.testing.assert_array_equal(scatter.complete(), [1, 2, 4, 0, 8])
        np.testing.assert_array_equal(scatter.file_flags(1), [0, 8])

    def test_duplicate_within_batch_and_across_batches_are_rejected(self):
        scatter = release.FlagScatter(self.files())
        with self.assertRaisesRegex(ValueError, 'Duplicate original'):
            scatter.consume(pa.array(['/first.parquet'] * 2), pa.array([1, 1]), np.array([0, 1]))
        scatter.consume(pa.array(['/first.parquet']), pa.array([1]), np.array([0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate original'):
            scatter.consume(pa.array(['/first.parquet']), pa.array([1]), np.array([1]))

    def test_unknown_negative_out_of_bounds_null_and_missing_are_rejected(self):
        for paths, rows, message in [(['/missing'], [0], 'Unknown/null'),
                                     (['/first.parquet'], [-1], 'outside'),
                                     (['/second.parquet'], [2], 'outside'),
                                     (['/first.parquet'], [None], 'Unknown/null')]:
            scatter = release.FlagScatter(self.files())
            with self.assertRaisesRegex(ValueError, message):
                scatter.consume(pa.array(paths), pa.array(rows, type=pa.int64()), np.array([0]))
        with self.assertRaisesRegex(ValueError, 'Missing original'):
            release.FlagScatter(self.files()).complete()


class PolicyTests(unittest.TestCase):
    def test_integrity_and_binding_survive_prior_exclusion_priority(self):
        table = candidate([
            {'decision': 'exclude', 'decision_reason': 'existing_frontpass_rejection',
             'status': 'rejected', 'integrity_quarantine': True},
            {'decision': 'exclude', 'decision_reason': 'confirmed_image_screening_exclusion',
             'confirmed_screen_excluded': True, 'reason_codes': ['unresolved_original_image_binding']},
        ])
        flags = release.base_flags(table, FLAGS, ['b' * 64])
        self.assertEqual(int(flags[0]), FLAGS['PRIOR_EXCLUDE'] | FLAGS['FRONTPASS_REJECTION'] | FLAGS['INTEGRITY'])
        self.assertEqual(int(flags[1]), FLAGS['PRIOR_EXCLUDE'] | FLAGS['SCREENING_EXCLUSION'] | FLAGS['ORIGINAL_BINDING'])

    def test_safety_detects_every_image_slot_independent_of_older_decision(self):
        table = candidate([
            {'image_ids': ['a' * 64, 'b' * 64]},
            {'image_ids': ['b' * 64, 'b' * 64]},
            {'image_ids': [], 'native_image_count': 0},
            {'unresolved_screening': True},
        ])
        flags = release.base_flags(table, FLAGS, ['b' * 64])
        np.testing.assert_array_equal(flags, [FLAGS['SAFETY'], FLAGS['SAFETY'], FLAGS['NO_IMAGE'], FLAGS['UNRESOLVED_SCREENING']])

    def test_restored_quality_groups_are_deduplicated_again(self):
        table = candidate([
            {'decision': 'exclude', 'decision_reason': 'existing_frontpass_rejection', 'status': 'rejected'},
            {'decision': 'exclude', 'decision_reason': 'existing_frontpass_rejection', 'status': 'rejected'},
            {'record_signature': '1' * 64},
        ])
        flags = release.base_flags(table, FLAGS, [])
        def policy(words):
            return words == 0, (words & (FLAGS['INTEGRITY'] | FLAGS['SAFETY'])) == 0
        ranked = release.rerank_bucket(table, flags, policy, FLAGS['DUPLICATE_NOT_SELECTED'], threads=1)
        self.assertFalse(ranked[0] & FLAGS['DUPLICATE_NOT_SELECTED'])
        self.assertTrue(ranked[1] & FLAGS['DUPLICATE_NOT_SELECTED'])
        self.assertFalse(ranked[2] & FLAGS['DUPLICATE_NOT_SELECTED'])

    def test_ranking_prefers_readable_then_image_eligible_then_stable_identity(self):
        table = candidate([{'integrity_quarantine': True},
                           {'unresolved_screening': True}, {}, {}])
        flags = release.base_flags(table, FLAGS, [])
        def policy(words):
            return words == 0, (words & FLAGS['INTEGRITY']) == 0
        ranked = release.rerank_bucket(table, flags, policy, FLAGS['DUPLICATE_NOT_SELECTED'], threads=1)
        self.assertEqual(np.flatnonzero((ranked & FLAGS['DUPLICATE_NOT_SELECTED']) == 0).tolist(), [2])


class BuildFixture:
    """Small immutable published inputs; the lineage verifier has a real smoke test."""
    def __init__(self, root):
        from . import dcvlm_loss_policy_reader as policy
        self.policy = policy
        self.parent = root / 'parent'
        self.candidate = self.parent / 'training_candidate_index'
        self.view = self.parent / 'training_candidate_source_view'
        self.native = root / 'native'
        self.output = root / 'release'
        self.repairs = root / 'repair'
        for directory in (self.candidate, self.view, self.native, self.repairs, self.parent / 'readers'):
            directory.mkdir(parents=True)
        source = self.native / 'data.parquet'
        native = pa.table({'collection': ['dcvlm'] * 9, 'sample_id': [str(i) for i in range(9)],
                           'native_text': [f'original-{i}' for i in range(9)]})
        pq.write_table(native, source)
        source_before = source.read_bytes()
        self.original_native = source_before
        rows = [
            {'record_signature': '00' + 'a' * 62},
            {'record_signature': '00' + 'a' * 62, 'decision': 'exclude',
             'decision_reason': 'identical_ordered_supervision_and_confirmed_images'},
            {'record_signature': '00' + 'b' * 62, 'decision': 'exclude',
             'decision_reason': 'existing_frontpass_rejection', 'status': 'rejected'},
            {'record_signature': '00' + 'b' * 62, 'decision': 'exclude',
             'decision_reason': 'existing_frontpass_rejection', 'status': 'rejected'},
            {'record_signature': '11' + 'c' * 62, 'decision': 'quarantine',
             'decision_reason': 'unresolved_image_integrity_or_missing_asset', 'integrity_quarantine': True},
            {'record_signature': '11' + 'd' * 62, 'decision': 'exclude', 'status': 'rejected',
             'decision_reason': 'existing_frontpass_rejection', 'integrity_quarantine': True},
            {'record_signature': '11' + 'e' * 62, 'decision': 'quarantine', 'integrity_quarantine': True,
             'decision_reason': 'unresolved_original_image_binding', 'reason_codes': ['unresolved_original_image_binding']},
            {'record_signature': '11' + 'f' * 62, 'decision': 'exclude',
             'decision_reason': 'confirmed_image_screening_exclusion', 'confirmed_screen_excluded': True,
             'image_ids': [sorted(policy.SAFETY_DENY_SHA256)[0]]},
            {'record_signature': '11' + '0' * 62, 'native_image_count': 0, 'image_ids': []},
        ]
        table = candidate(rows)
        for name, values in [('original_parquet_path', [str(source)] * 9), ('row_index', list(range(9))),
                             ('source', ['fixture'] * 9), ('category', ['single_i2t'] * 8 + ['pure_text'])]:
            table = table.append_column(name, pa.array(values))
        self.table = table
        buckets = []
        for bucket in ('00', '11'):
            subset = table.filter(pc.starts_with(table['record_signature'], bucket))
            directory = self.candidate / bucket
            directory.mkdir()
            artifacts = []
            for decision in ('keep', 'exclude', 'quarantine'):
                path = directory / (decision + '.parquet')
                pq.write_table(subset.filter(pc.equal(subset['decision'], decision)), path)
                artifacts.append(release.artifact(path, self.candidate))
            buckets.append({'bucket': bucket, 'records': len(subset), 'counts': release._column_counts(subset, 'decision'),
                            'reasons': release._column_counts(subset, 'decision_reason'), 'artifacts': artifacts})
        categories = table.group_by(['source', 'category', 'decision']).aggregate([('record_id', 'count')]).to_pylist()
        categories = [{**{key: row[key] for key in ('source', 'category', 'decision')},
                       'records': row['record_id_count']} for row in categories]
        manifest = {'records': len(table), 'counts': release._column_counts(table, 'decision'),
                    'reasons': release._column_counts(table, 'decision_reason'), 'buckets': buckets,
                    'source_category_counts': categories}
        release.atomic_json(self.candidate / 'manifest.json', manifest)
        old_codes = np.asarray([release.CODES[row.get('decision', 'keep')] for row in rows], dtype=np.uint8)
        (self.view / 'old.u8').write_bytes(old_codes.tobytes())
        view_item = {'source_path': str(source), 'source_sha256': release.file_sha(source),
                     'source_stamp': release.stamp(source), 'rows': 9,
                     'counts': {name: int(np.count_nonzero(old_codes == code)) for name, code in release.CODES.items()},
                     'mask_bytes': 9, 'mask_path': 'old.u8', 'mask_sha256': release.file_sha(self.view / 'old.u8')}
        roots = {'schema': 'corpus.source_root_mapping.v1', 'native_roots': {str(self.native): str(self.native)},
                 'dataset_roots': {policy.DATASET_ID: str(root / 'raw')}}
        root_path = self.parent / 'readers' / 'ROOTS.example.json'
        release.atomic_json(root_path, roots)
        (self.parent / 'readers' / 'portable_refs.py').write_text('# Frozen resolver is not needed for metadata fixture.\n')
        release.atomic_json(self.parent / 'CURRENT.json', {'release_id': 'fixture'})
        self.context = {'parent': self.parent, 'candidate': self.candidate, 'candidate_manifest': manifest,
                        'view': self.view, 'view_manifest': {'files': [view_item]}, 'native': self.native,
                        'roots': roots, 'current': {'release_id': 'fixture'}, 'release': {'records': 9},
                        'provenance': {'ROOTS.example.json': root_path}}
        evidence = self.repairs / 'EVIDENCE_BUNDLE.json'
        release.atomic_json(evidence, {'complete': True, 'fixture': True})
        evidence_sha = release.file_sha(evidence)
        records = pa.table({'record_id': ['4', '5', '6'],
                            'outcome': ['verified_original', 'repaired', 'mismatch'],
                            'reason': ['bounded_decode', 'bounded_decode', 'placeholder_mismatch'],
                            'evidence_sha256': [evidence_sha] * 3})
        pq.write_table(records, self.repairs / 'records.parquet')
        policies = pa.table({'asset_id': ['a' * 64], 'decode_policy': ['first_frame'], 'frame_count': [2],
                             'width': [1], 'height': [1], 'bytes_verified': [True], 'decode_verified': [True],
                             'frame0_rgb_sha256': ['b' * 64]})
        pq.write_table(policies, self.repairs / 'image_policies.parquet')
        self.repair_manifest = self.repairs / 'REPAIR_MANIFEST.json'
        repair = {'schema': 'corpus.dcvlm_integrity_repair.v1', 'complete': True,
                  'parent_candidate_manifest_sha256': release.file_sha(self.candidate / 'manifest.json'),
                  'records': {**release.artifact(self.repairs / 'records.parquet', self.repairs), 'rows': 3},
                  'image_policies': {**release.artifact(self.repairs / 'image_policies.parquet', self.repairs), 'rows': 1},
                  'evidence_artifacts': [release.artifact(evidence, self.repairs)],
                  'counts': {'record_outcomes': dict(Counter(records['outcome'].to_pylist()))}}
        release.atomic_json(self.repair_manifest, repair)
        self.user = root / 'user.json'
        release.atomic_json(self.user, {'schema': 'corpus.user_adult_review_override.v1',
            'decision': 'adult_user_review', 'asset_ids': sorted(policy.SAFETY_DENY_SHA256),
            'user_statement': '我经过审查了，没问题，是成年', 'independent_age_verification': False,
            'allow_understanding': True, 'allow_image_loss': False})

    def build(self):
        with patch.object(release, 'load_parent', return_value=self.context):
            return release.build(self.parent, self.output, repair_manifest=self.repair_manifest,
                                 user_review=self.user, expected_safety_records=1, threads=1, batch_size=2)


class BuilderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = BuildFixture(Path(self.temporary.name))

    def test_restored_i2t_repair_dedup_and_original_metadata_are_consistent(self):
        f = self.fixture
        result = f.build()
        self.assertEqual(result['counts']['records'], 9)
        self.assertEqual(result['counts']['selected_records'], 6)
        self.assertEqual(result['counts']['image_loss_allowed'], 1)
        self.assertEqual(result['counts']['training_blocked'], 1)
        manifest = release._document(f.output / 'manifest.json')
        self.assertEqual(manifest['filtered_duplicates'], 2)
        selected = [row for batch in f.policy.iter_batches(f.output, f.context['roots'], batch_size=2) for row in batch.to_pylist()]
        self.assertEqual([row['sample_id'] for row in selected], ['0', '2', '4', '5', '7', '8'])
        self.assertEqual([row['sample_id'] for row in selected if row['dcvlm_image_loss_allowed']], ['0'])
        all_rows = [row for batch in f.policy.iter_batches(f.output, f.context['roots'], include_unselected=True)
                    for row in batch.to_pylist()]
        self.assertEqual([row['native_text'] for row in all_rows], [f'original-{i}' for i in range(9)])
        self.assertEqual((f.native / 'data.parquet').read_bytes(), f.original_native)
        self.assertEqual(f.build(), result)

    def test_prepublication_failure_never_creates_current_pointer(self):
        f = self.fixture
        with patch.object(f.policy, '_verify_release_payload', side_effect=ValueError('preflight failed')):
            with self.assertRaisesRegex(ValueError, 'preflight failed'):
                f.build()
        self.assertFalse((f.output / 'CURRENT.json').exists())
        self.assertEqual(f.build()['counts']['selected_records'], 6)

    def test_candidate_hash_and_unbound_repair_evidence_fail_closed(self):
        f = self.fixture
        path = f.candidate / f.context['candidate_manifest']['buckets'][0]['artifacts'][0]['path']
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact bytes or SHA'):
            f.build()
        self.assertFalse((f.output / 'CURRENT.json').exists())
        (f.repairs / 'EVIDENCE_BUNDLE.json').write_text('{"complete":false}\n')
        with self.assertRaisesRegex(ValueError, 'artifact SHA mismatch'):
            release.load_repairs(f.repair_manifest, release.file_sha(f.candidate / 'manifest.json'))


if __name__ == '__main__':
    unittest.main()
