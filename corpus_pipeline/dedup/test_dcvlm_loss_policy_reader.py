from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import dcvlm_loss_policy_reader as reader


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


class ReaderFixture:
    def __init__(self, root):
        self.root = root
        self.view = root / 'release'
        self.native = root / 'relocated' / 'native'
        self.old_native = root / 'old' / 'native'
        for path in (self.view / 'readers', self.view / 'flags', self.view / 'repairs', self.native):
            path.mkdir(parents=True)
        shutil.copyfile(reader.__file__, self.view / 'readers' / 'dcvlm_loss_policy_reader.py')
        shutil.copyfile(Path(reader.__file__).parents[1] / 'production' / 'portable_refs.py',
                        self.view / 'readers' / 'portable_refs.py')
        f = reader.FLAGS
        self.words = np.array([
            0, f['PRIOR_EXCLUDE'] | f['DUPLICATE'] | f['DUPLICATE_NOT_SELECTED'],
            f['PRIOR_EXCLUDE'] | f['FRONTPASS_REJECTION'], f['PRIOR_QUARANTINE'] | f['INTEGRITY'],
            f['MISMATCH'], f['PRIOR_QUARANTINE'] | f['INTEGRITY'] | f['REPAIRED'],
            f['NO_IMAGE'], f['PRIOR_EXCLUDE'] | f['SAFETY'] | f['USER_ADULT_REVIEW'],
            f['UNRESOLVED_SCREENING'],
        ], dtype='<u2')
        self.rows = [{'collection': 'dcvlm', 'sample_id': str(i),
                      'segments': [{'kind': 'text', 'text': 'unchanged supervision'}],
                      'image_count': 0 if i == 6 else 1} for i in range(len(self.words))]
        # An exact repeated native row remains visible in the audit iterator.
        self.rows[1] = dict(self.rows[0])
        files = []
        for index, (start, stop) in enumerate(((0, 5), (5, 9))):
            path = self.native / (str(index) + '.parquet')
            pq.write_table(pa.Table.from_pylist(self.rows[start:stop]), path)
            flags_path = self.view / 'flags' / (str(index) + '.u16')
            flags_path.write_bytes(self.words[start:stop].tobytes())
            files.append({'source_path': str(self.old_native / path.name),
                          'source_sha256': reader.sha(path), 'source_bytes': path.stat().st_size,
                          'rows': stop - start, 'flags_path': 'flags/' + flags_path.name,
                          'flags_sha256': reader.sha(flags_path), 'flags_bytes': flags_path.stat().st_size,
                          'flag_counts': dict(Counter(str(int(word)) for word in self.words[start:stop]))})
        self.roots = {'schema': 'corpus.source_root_mapping.v1',
                      'native_roots': {str(self.old_native): str(self.native)},
                      'dataset_roots': {reader.DATASET_ID: str(root)}}
        self.manifest = {'schema': reader.SCHEMA, 'complete': True, 'scope': 'full',
                         'collection': 'dcvlm', 'dataset_id': reader.DATASET_ID,
                         'original_records_modified': False, 'flags_encoding': 'uint16_le',
                         'flag_definitions': reader.FLAGS, 'files': files,
                         'safety_deny_sha256': sorted(reader.SAFETY_DENY_SHA256),
                         'parent_candidate_manifest_sha256': 'a' * 64}
        records = [{'record_id': 'f' * 64, 'outcome': 'repaired', 'reason': 'verified fixture',
                    'evidence_sha256': 'e' * 64}]
        pq.write_table(pa.Table.from_pylist(records), self.view / 'repairs' / 'records.parquet')
        self.set_image_policies([])
        self.user_review = {'schema': 'corpus.user_adult_review_override.v1',
                            'asset_ids': sorted(reader.SAFETY_DENY_SHA256),
                            'user_statement': '我经过审查了，没问题，是成年',
                            'decision': 'adult_user_review', 'independent_age_verification': False,
                            'allow_understanding': True, 'allow_image_loss': False}
        write_json(self.view / 'user_review.json', self.user_review)
        self.evidence_artifacts = []
        self.seal()

    def set_image_policies(self, rows):
        schema = pa.schema([('asset_id', pa.string()), ('decode_policy', pa.string()),
                            ('frame_count', pa.int64()), ('width', pa.int64()), ('height', pa.int64()),
                            ('bytes_verified', pa.bool_()), ('decode_verified', pa.bool_()),
                            ('frame0_rgb_sha256', pa.string())])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), self.view / 'repairs' / 'image_policies.parquet')

    def seal(self):
        binding = {'schema': reader.READER_SCHEMA,
                   'files': [{'name': name, 'sha256': reader.sha(self.view / 'readers' / name)}
                             for name in ('dcvlm_loss_policy_reader.py', 'portable_refs.py')]}
        write_json(self.view / 'readers' / 'READER_BINDING.json', binding)
        self.manifest['reader_binding_sha256'] = reader.sha(self.view / 'readers' / 'READER_BINDING.json')
        repair = {'schema': 'corpus.dcvlm_integrity_repair.v1', 'complete': True,
                  'parent_candidate_manifest_sha256': 'a' * 64,
                  'evidence_artifacts': self.evidence_artifacts}
        for name in ('records', 'image_policies'):
            path = self.view / 'repairs' / (name + '.parquet')
            repair[name] = {'path': path.name, 'sha256': reader.sha(path),
                            'rows': pq.ParquetFile(path).metadata.num_rows}
        write_json(self.view / 'repairs' / 'manifest.json', repair)
        self.manifest['repair_manifest'] = {'path': 'repairs/manifest.json',
                                            'sha256': reader.sha(self.view / 'repairs' / 'manifest.json')}
        if (self.view / 'user_review.json').exists():
            self.manifest['user_review'] = {'path': 'user_review.json',
                                           'sha256': reader.sha(self.view / 'user_review.json')}
        counts = Counter()
        for item in self.manifest['files']:
            counts.update(item['flag_counts'])
        summary = reader.summarize_flag_counts(counts)
        self.manifest.update(records=summary['records'], selected_records=summary['selected_records'],
                             source_files=len(self.manifest['files']), counts=summary)
        self.seal_manifest()

    def seal_manifest(self):
        write_json(self.view / 'manifest.json', self.manifest)
        done = {'complete': True, 'manifest_sha256': reader.sha(self.view / 'manifest.json'),
                'records': self.manifest['records'], 'selected_records': self.manifest['selected_records'],
                'counts': self.manifest['counts']}
        write_json(self.view / 'COMPLETE.json', done)
        current = {'schema': reader.SCHEMA, 'scope': 'full', 'root': '.',
                   'required_collections': ['dcvlm'], 'release_id': done['manifest_sha256'],
                   'manifest': {'path': 'manifest.json', 'sha256': done['manifest_sha256']},
                   'complete': {'path': 'COMPLETE.json', 'sha256': reader.sha(self.view / 'COMPLETE.json')}}
        write_json(self.view / 'CURRENT.json', current)


class DCVLMLossPolicyReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = ReaderFixture(Path(self.temp.name))
        self.f = self.fixture

    def rows(self, **kwargs):
        return [row for batch in reader.iter_batches(self.f.view, self.f.roots, batch_size=2, **kwargs)
                for row in batch.to_pylist()]

    def test_relocation_preserves_all_native_columns_and_separates_selection_from_loss(self):
        all_rows = self.rows(include_unselected=True)
        self.assertEqual(len(all_rows), len(self.f.rows))
        stripped = [{key: row[key] for key in self.f.rows[0]} for row in all_rows]
        self.assertEqual(stripped, self.f.rows)
        selected = self.rows()
        self.assertEqual([row['sample_id'] for row in selected], ['0', '2', '5', '6', '7', '8'])
        self.assertEqual([row['sample_id'] for row in selected if row['dcvlm_image_loss_allowed']], ['0'])
        self.assertTrue(all_rows[1]['dcvlm_training_ready'])
        self.assertFalse(all_rows[1]['dcvlm_selected'])
        self.assertEqual(all_rows[5]['dcvlm_previous_decision'], 'quarantine')
        self.assertTrue(all_rows[5]['dcvlm_training_ready'])
        self.assertFalse(all_rows[5]['dcvlm_image_loss_allowed'])
        self.assertTrue(all_rows[7]['dcvlm_selected'])
        self.assertFalse(all_rows[7]['dcvlm_image_loss_allowed'])
        for item in self.f.manifest['files']:
            self.assertEqual(reader.sha(self.f.native / Path(item['source_path']).name), item['source_sha256'])

    def test_worker_partition_and_counts_cover_each_source_exactly_once(self):
        rows = [row for worker in range(2) for row in self.rows(worker_id=worker, num_workers=2,
                                                              epoch_seed=4, include_unselected=True)]
        self.assertEqual(Counter(row['sample_id'] for row in rows), Counter(row['sample_id'] for row in self.f.rows))
        manifest = reader.verify_release(self.f.view)
        self.assertEqual(manifest['counts'], {'records': 9, 'selected_records': 6, 'unselected_records': 3,
                                             'image_loss_allowed': 1, 'image_loss_disabled': 8,
                                             'training_ready': 7, 'training_blocked': 2,
                                             'previous_keep': 4, 'previous_exclude': 3, 'previous_quarantine': 2})

    def test_flags_native_runtime_and_evidence_tampering_fail_closed(self):
        reader.verify_release(self.f.view)
        flag = self.f.view / self.f.manifest['files'][0]['flags_path']
        before = flag.read_bytes()
        flag.write_bytes(b'\x01\x00' + before[2:])
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            self.rows()
        flag.write_bytes(before)
        native = self.f.native / '0.parquet'
        before = native.read_bytes()
        native.write_bytes(before + b'changed')
        with self.assertRaisesRegex(ValueError, 'native content differs'):
            self.rows()
        native.write_bytes(before)
        runtime = self.f.view / 'readers' / 'portable_refs.py'
        before = runtime.read_bytes()
        runtime.write_bytes(before + b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            reader.verify_release(self.f.view)
        runtime.write_bytes(before)
        evidence = self.f.view / 'repairs' / 'records.parquet'
        evidence.write_bytes(evidence.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            reader.verify_release(self.f.view)

    def test_other_collection_cannot_use_policy_and_unknown_flag_bits_rejected(self):
        path = self.f.native / '0.parquet'
        rows = pq.read_table(path).to_pylist()
        rows[0]['collection'] = 'another_dataset'
        pq.write_table(pa.Table.from_pylist(rows), path)
        self.f.manifest['files'][0].update(source_sha256=reader.sha(path), source_bytes=path.stat().st_size)
        self.f.seal()
        with self.assertRaisesRegex(ValueError, 'another collection'):
            self.rows()
        with self.assertRaisesRegex(ValueError, 'Unknown policy flag'):
            reader.policy_arrays(np.array([32768], dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            reader.policy_arrays(np.array([3], dtype=np.uint16))

    def test_unpublished_or_invalid_coverage_release_is_unreadable(self):
        (self.f.view / 'CURRENT.json').unlink()
        self.assertEqual(reader._verify_release_payload(self.f.view)['records'], 9)
        with self.assertRaises(FileNotFoundError):
            reader.verify_release(self.f.view)
        self.f.seal()
        self.f.manifest['counts']['selected_records'] += 1
        self.f.seal_manifest()
        with self.assertRaisesRegex(ValueError, 'Policy counts differ'):
            reader.verify_release(self.f.view)

    def test_unbound_override_and_repair_fail_closed(self):
        self.f.manifest.pop('user_review')
        self.f.seal_manifest()
        with self.assertRaisesRegex(ValueError, 'lacks a valid bound review'):
            reader.verify_release(self.f.view)
        self.f.seal()
        self.f.manifest.pop('repair_manifest')
        self.f.seal_manifest()
        with self.assertRaisesRegex(ValueError, 'lacks bound repair'):
            reader.verify_release(self.f.view)

    def test_safety_access_checks_precede_any_image_read_and_pin_dataset(self):
        denied = next(iter(reader.SAFETY_DENY_SHA256))
        fake = Mock()
        fake.logical_ref.side_effect = lambda ref: ref
        fake.read_ref_bytes.return_value = b'fixture bytes only'
        release = reader._release(self.f.view)
        with patch.object(reader, '_resolver', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'pinned DCVLM'):
                reader.read_image({'dataset_id': 'another@dataset', 'sha256': denied}, self.f.roots, view=self.f.view)
            fake.read_ref_bytes.assert_not_called()
            without_override = {**release, 'override': frozenset()}
            with patch.object(reader, '_release', return_value=without_override):
                with self.assertRaisesRegex(ValueError, 'bound adult user review'):
                    reader.read_image({'dataset_id': reader.DATASET_ID, 'sha256': denied}, self.f.roots, view=self.f.view)
            fake.read_ref_bytes.assert_not_called()
            result = reader.read_image({'dataset_id': reader.DATASET_ID, 'sha256': denied}, self.f.roots, view=self.f.view)
            self.assertEqual(result, b'fixture bytes only')
            fake.read_ref_bytes.assert_called_once()

    def test_cache_avoids_rehash_but_stat_change_invalidates_it(self):
        reader.verify_release(self.f.view)
        with patch.object(reader, 'sha', wraps=reader.sha) as hashed:
            reader.verify_release(self.f.view)
            reader.verify_release(self.f.view)
            hashed.assert_not_called()
            path = self.f.view / 'COMPLETE.json'
            before = path.stat()
            previous_stamp = reader.stamp(path)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
            self.assertNotEqual(reader.stamp(path), previous_stamp)
            reader.verify_release(self.f.view)
            self.assertGreater(hashed.call_count, 0)

    def test_repair_audit_evidence_is_bound_and_checked_after_cache_invalidation(self):
        path = self.f.view / 'repairs' / 'probe.json'
        path.write_bytes(b'{"fixture": true}')
        self.f.evidence_artifacts = [{'path': path.name, 'sha256': reader.sha(path),
                                      'size_bytes': path.stat().st_size}]
        self.f.seal()
        reader.verify_release(self.f.view)
        before = path.read_bytes()
        path.write_bytes(before.replace(b'true', b'null'))
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            reader.verify_release(self.f.view)
        path.write_bytes(before + b' ')
        with self.assertRaisesRegex(ValueError, 'size differs'):
            reader.verify_release(self.f.view)

    def test_first_frame_policy_matches_bound_pixels_and_restores_decoder_limit(self):
        from PIL import Image
        buffer = io.BytesIO()
        first, second = Image.new('RGB', (3, 2), 'red'), Image.new('RGB', (3, 2), 'blue')
        first.save(buffer, format='GIF', save_all=True, append_images=[second], duration=100, loop=0)
        blob = buffer.getvalue()
        asset = hashlib.sha256(blob).hexdigest()
        policy = {'asset_id': asset, 'decode_policy': 'first_frame', 'frame_count': 2,
                  'width': 3, 'height': 2, 'bytes_verified': True, 'decode_verified': True,
                  'frame0_rgb_sha256': hashlib.sha256(first.tobytes()).hexdigest()}
        self.f.set_image_policies([policy])
        self.f.seal()
        fake = Mock()
        fake.logical_ref.side_effect = lambda ref: ref
        old_limit = Image.MAX_IMAGE_PIXELS
        ref = {'dataset_id': reader.DATASET_ID, 'sha256': asset}
        with patch.object(reader, '_resolver', return_value=fake), patch.object(reader, 'read_image', return_value=blob):
            actual = reader.decode_image(ref, self.f.roots, view=self.f.view)
            self.assertEqual(actual.tobytes(), first.tobytes())
            self.assertEqual(Image.MAX_IMAGE_PIXELS, old_limit)
            with self.assertRaisesRegex(ValueError, 'pixel bound'):
                reader.decode_image(ref, self.f.roots, view=self.f.view, max_pixels=5)
            self.assertEqual(Image.MAX_IMAGE_PIXELS, old_limit)
            policy['frame0_rgb_sha256'] = '0' * 64
            self.f.set_image_policies([policy])
            self.f.seal()
            with self.assertRaisesRegex(ValueError, 'pixels differ'):
                reader.decode_image(ref, self.f.roots, view=self.f.view)


if __name__ == '__main__':
    unittest.main()
