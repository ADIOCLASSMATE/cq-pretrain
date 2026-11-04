"""Real frozen masks and metadata-only union fixtures, without image payloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from corpus_pipeline.dedup.test_dcvlm_loss_policy_release import BuildFixture
from corpus_pipeline.dedup.training_release import text_fingerprint
from corpus_pipeline.postprocess.reconcile import artifact, atomic_json, digest, file_sha, stamp
from . import baseline_union as union


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def text(value, role='assistant', turn=0):
    return text_fingerprint([dict(kind='image', image_index=0, role='user', turn=0),
                             dict(kind='text', text=value, role=role, turn=turn)])


def publish_candidate(root, table, native):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for decision in ('keep', 'exclude', 'quarantine'):
        path = root / 'buckets/aa' / (decision + '.parquet')
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.filter(pc.equal(table['decision'], decision)), path)
        paths.append(artifact(path, root))
    counts = {name: len(table.filter(pc.equal(table['decision'], name)))
              for name in ('keep', 'exclude', 'quarantine')}
    b = root / 'native_bindings/fixture.json'; b.parent.mkdir()
    atomic_json(b, dict(path=str(native), sha256=file_sha(native), stamp=stamp(native)))
    manifest = dict(schema_version='corpus.training_candidate_index.v1', complete=True,
        scope='full_run', records=len(table), counts=counts,
        buckets=[dict(bucket='aa', records=len(table), counts=counts, artifacts=paths)],
        native_metadata_bindings=[artifact(b, root)])
    atomic_json(root / 'manifest.json', manifest)
    atomic_json(root / 'COMPLETE.json', dict(schema_version=manifest['schema_version'], complete=True,
        scope='full_run', records=len(table), counts=counts, manifest_sha256=file_sha(root / 'manifest.json')))
    return manifest


def baseline_fixture(root):
    fixture = BuildFixture(root)
    table = fixture.table
    native_rows = pq.read_table(fixture.native / 'data.parquet').to_pylist()
    for name, values in [('source_record_sha256', [digest(r) for r in native_rows]),
                         ('text_hash', [text('caption-' + str(i)) for i in range(9)]),
                         ('native_format', ['conversation'] * 8 + ['text'])]:
        table = table.append_column(name, pa.array(values))
    fixture.table = table
    manifest = fixture.context['candidate_manifest']
    for bucket in manifest['buckets']:
        subset = table.filter(pc.starts_with(table['record_signature'], bucket['bucket']))
        for item in bucket['artifacts']:
            path = fixture.candidate / item['path']
            pq.write_table(subset.filter(pc.equal(subset['decision'], path.stem)), path)
            item.update(artifact(path, fixture.candidate))
        # Production counts contain all decisions, including zeros.
        bucket['counts'] = {d: len(subset.filter(pc.equal(subset['decision'], d)))
                            for d in ('keep', 'exclude', 'quarantine')}
    path = fixture.candidate / 'native_bindings/fixture.json'; path.parent.mkdir()
    native = fixture.native / 'data.parquet'
    atomic_json(path, dict(path=str(native), sha256=file_sha(native), stamp=stamp(native)))
    manifest.update(schema_version='corpus.training_candidate_index.v1', complete=True, scope='full_run',
                    native_metadata_bindings=[artifact(path, fixture.candidate)])
    atomic_json(fixture.candidate / 'manifest.json', manifest)
    atomic_json(fixture.candidate / 'COMPLETE.json', dict(schema_version=manifest['schema_version'],
        complete=True, scope='full_run', records=manifest['records'], counts=manifest['counts'],
        manifest_sha256=file_sha(fixture.candidate / 'manifest.json')))
    repair = json.loads(fixture.repair_manifest.read_text())
    repair['parent_candidate_manifest_sha256'] = file_sha(fixture.candidate / 'manifest.json')
    atomic_json(fixture.repair_manifest, repair)
    # Provenance keys become filenames in the frozen builder.
    fixture.context['provenance']['candidate_manifest.json'] = fixture.candidate / 'manifest.json'
    fixture.context['provenance']['candidate_COMPLETE.json'] = fixture.candidate / 'COMPLETE.json'
    fixture.build()
    return fixture


def task_candidate(root, changes):
    native = root / 'native/data.parquet'; native.parent.mkdir(parents=True)
    pq.write_table(pa.table({'sample_id': [str(i) for i in range(len(changes))],
                            'native_text': ['new-' + str(i) for i in range(len(changes))]}), native)
    native_rows = pq.read_table(native).to_pylist()
    rows = []
    for i, change in enumerate(changes):
        rows.append(dict(record_id='new-' + str(i), sample_id=str(i), original_parquet_path=str(native),
            row_index=i, source_record_sha256=digest(native_rows[i]), text_hash=text('caption-0'),
            category='single_i2t', native_format='conversation', source='genund', image_ids=['a' * 64],
            processing_policy=union.TASK_POLICY, task_route='i2t', image_loss_allowed=False,
            decision='keep') | change)
    table = pa.Table.from_pylist(rows)
    candidate = root / 'candidate'
    publish_candidate(candidate, table, native)
    return candidate


def reconciliation(root, rows):
    root.mkdir()
    defaults = dict(pixel_hash_count=1, pixel_sha256_conflict=False, dimension_count=1,
                    has_invalid_observation=False, original_width=32, original_height=48)
    path = root / 'buckets/aa/assets.parquet'; path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([defaults | r for r in rows]), path)
    receipt = path.with_name('COMPLETE.json')
    atomic_json(receipt, dict(bucket='aa', artifacts=[artifact(path, root)]))
    atomic_json(root / 'COMPLETE.json', dict(schema='corpus.asset_reconciliation.v1', complete=True,
        scope='full_run', buckets=[dict(bucket='aa', receipt_path=str(receipt.relative_to(root)),
                                        receipt_sha256=file_sha(receipt))]))
    return root


class UnionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixture = baseline_fixture(self.root / 'baseline')
        self.baseline = self.root / 'baseline-index'

    def export(self):
        return union.export_baseline_index(self.fixture.output, self.fixture.candidate, self.baseline, batch_size=2)

    def test_baseline_masks_restore_review_and_repair_without_resurrecting_duplicates(self):
        source_bytes = (self.fixture.native / 'data.parquet').read_bytes()
        old_manifest = (self.fixture.output / 'manifest.json').read_bytes()
        result = self.export()
        self.assertEqual(result['records'], 6)
        self.assertEqual(result['counts']['input_records'], 9)
        self.assertEqual(result['counts']['image_loss_allowed'], 0)
        reader = union.ReferenceReader(self.baseline)
        rows = list(reader.iter_references())
        self.assertEqual({r['record_id'] for r in rows}, {'0', '2', '4', '5', '7', '8'})
        self.assertTrue(next(r for r in rows if r['record_id'] == '0')['dcvlm_image_loss_allowed'])
        self.assertFalse(any(r['image_loss_allowed'] for r in rows))
        self.assertTrue(next(r for r in rows if r['record_id'] == '7')['dcvlm_policy_flags'] & 16384)
        self.assertEqual(reader.read_record(rows[0])['native_record']['native_text'],
                         'original-' + rows[0]['record_id'])
        self.assertEqual((self.fixture.native / 'data.parquet').read_bytes(), source_bytes)
        self.assertEqual((self.fixture.output / 'manifest.json').read_bytes(), old_manifest)
        self.assertEqual(self.export(), result)

    def test_exact_pixel_union_preserves_caption_role_turn_format_task_and_permission(self):
        self.export()
        changes = [{}, {'image_ids': ['b' * 64]}, {'text_hash': text('different-caption')},
                   {'text_hash': text('caption-0', role='user')}, {'text_hash': text('caption-0', turn=1)},
                   {'native_format': 'caption'}, {'category': 'multi_i2t'},
                   {'task_route': 't2i', 'image_loss_allowed': True},
                   {'task_route': 't2i', 'image_loss_allowed': False},
                   {'image_ids': ['c' * 64]}, {'image_ids': ['a' * 64, 'b' * 64]},
                   {'image_ids': ['b' * 64, 'a' * 64]}]
        candidate = task_candidate(self.root / 'new', changes)
        additional = self.root / 'new-index'
        union.export_candidate_index(candidate, additional, batch_size=3)
        recon = reconciliation(self.root / 'reconciliation', [
            dict(asset_id='a' * 64, pixel_sha256='f' * 64),
            dict(asset_id='b' * 64, pixel_sha256='f' * 64),
            dict(asset_id='c' * 64, pixel_sha256='f' * 64, original_width=33)])
        output = self.root / 'union'
        result = union.merge_selected_indices(self.baseline, [additional], output,
                    reconciliations=[recon], threads=1, memory_limit='512MB')
        self.assertEqual(result['counts']['baseline_selected_records'], 6)
        # New 0/1 match baseline; the two repeated exact-image sequences also match.
        self.assertEqual(result['counts']['exact_duplicates_removed'], 3)
        self.assertEqual(result['counts']['selected_records'], 15)
        self.assertEqual(result['counts']['image_loss_allowed'], 1)
        self.assertEqual(result['original_image_bytes_read'], 0)
        self.assertFalse(result['sscd_candidates_used_for_deletion'])
        rows = list(union.ReferenceReader(output).iter_references())
        self.assertTrue(any(r['record_id'] == 'new-9' for r in rows))  # geometry differs
        self.assertEqual(len({image for r in rows for image in r['exact_image_ids']}),
                         result['counts']['unique_image_identities'])
        self.assertEqual(union.merge_selected_indices(self.baseline, [additional], output,
            reconciliations=[recon], threads=1, memory_limit='512MB'), result)

    def test_bytes_fallback_never_drops_different_encoding_without_pixel_proof(self):
        self.export()
        candidate = task_candidate(self.root / 'new', [{'image_ids': ['b' * 64]}])
        additional = self.root / 'new-index'; union.export_candidate_index(candidate, additional)
        result = union.merge_selected_indices(self.baseline, [additional], self.root / 'union',
                                             threads=1, memory_limit='512MB')
        self.assertEqual(result['counts']['selected_records'], 7)
        self.assertEqual(result['image_identity_mode'], 'encoded_sha_only')

    def test_candidate_lineage_and_native_source_binding_are_required(self):
        path = self.fixture.candidate / 'COMPLETE.json'
        value = json.loads(path.read_text()); value['records'] += 1; atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, 'completion/count'):
            self.export()

    def test_original_source_change_and_retired_source_redirect_are_rejected(self):
        source = self.fixture.native / 'data.parquet'
        source.write_bytes(source.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'native metadata binding'):
            self.export()

    def test_flags_and_output_sha_tampering_are_rejected(self):
        baseline = json.loads((self.fixture.output / 'manifest.json').read_text())
        flags = self.fixture.output / baseline['files'][0]['flags_path']
        original = flags.read_bytes(); flags.write_bytes(b'\0' * len(original))
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.export()
        flags.write_bytes(original)
        result = self.export()
        path = self.baseline / result['files'][0]['path']
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            union.verify_reference_release(self.baseline)

    def test_only_explicit_t2i_allows_image_loss_and_nonboolean_is_rejected(self):
        for name, permission in [('i2t', True), ('badtype', 'false')]:
            candidate = task_candidate(self.root / name, [{'image_loss_allowed': permission}])
            with self.assertRaisesRegex(ValueError, 'Only explicit T2I|must be boolean'):
                union.export_candidate_index(candidate, self.root / (name + '-index'))

    def test_coverage_rejects_duplicate_missing_and_invalid_native_locators(self):
        files = [dict(source_path='/native/data.parquet', rows=2)]
        coverage = union._Coverage(files)
        table = pa.table(dict(original_parquet_path=['/native/data.parquet'] * 2, row_index=[0, 0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            coverage.consume(table)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            coverage.complete()
        with self.assertRaisesRegex(ValueError, 'outside'):
            coverage.consume(pa.table(dict(original_parquet_path=['/native/data.parquet'], row_index=[2])))

    def test_reader_supports_relocation_and_checks_new_native_bytes(self):
        self.export()
        moved = self.root / 'relocated'; self.fixture.native.rename(moved)
        reader = union.ReferenceReader(self.baseline, native_roots={str(self.fixture.native): str(moved)})
        reference = next(reader.iter_references())
        self.assertEqual(reader.read_record(reference)['native_record']['sample_id'], reference['sample_id'])
        tampered = dict(reference, row_index=(reference['row_index'] + 1) % 9)
        with self.assertRaisesRegex(ValueError, 'locator does not match'):
            reader.read_record(tampered)
        source = moved / 'data.parquet'; source.write_bytes(source.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            reader.read_record(reference)

    def test_image_order_and_baseline_selection_survive_union_key_collisions(self):
        self.export()
        # Two baseline records can newly converge after identity normalization;
        # publication must still keep both already selected baseline views.
        manifest = json.loads((self.baseline / 'manifest.json').read_text())
        for item in manifest['files']:
            path = self.baseline / item['path']; table = pq.read_table(path)
            table = table.set_column(table.schema.get_field_index('text_hash'), 'text_hash',
                pc.if_else(pc.equal(table['record_id'], '2'), text('caption-0'), table['text_hash']))
            pq.write_table(table, path); item.update(artifact(path, self.baseline))
        union._publish(self.baseline, manifest)
        candidate = task_candidate(self.root / 'new', [{}, {'image_ids': ['a' * 64, 'c' * 64]},
                                                     {'image_ids': ['c' * 64, 'a' * 64]}])
        additional = self.root / 'new-index'; union.export_candidate_index(candidate, additional)
        result = union.merge_selected_indices(self.baseline, [additional], self.root / 'union',
                                             threads=1, memory_limit='512MB')
        self.assertEqual(result['counts']['baseline_selected_records'], 6)
        self.assertEqual(result['counts']['selected_records'], 8)
        self.assertEqual(result['counts']['exact_duplicates_removed'], 1)

    def test_reconciliation_hash_tampering_prevents_union_publication(self):
        self.export()
        recon = reconciliation(self.root / 'reconciliation', [dict(asset_id='a' * 64, pixel_sha256='f' * 64)])
        path = recon / 'buckets/aa/assets.parquet'; path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            union.merge_selected_indices(self.baseline, [], self.root / 'union', reconciliations=[recon])
        self.assertFalse((self.root / 'union/CURRENT.json').exists())

    def test_candidate_empty_bucket_is_a_valid_zero_row_partition(self):
        candidate = task_candidate(self.root / 'new', [{}])
        manifest = json.loads((candidate / 'manifest.json').read_text())
        schema = pq.read_schema(candidate / manifest['buckets'][0]['artifacts'][0]['path'])
        files = []
        for decision in ('keep', 'exclude', 'quarantine'):
            path = candidate / 'buckets/bb' / (decision + '.parquet'); path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_batches([], schema=schema), path); files.append(artifact(path, candidate))
        manifest['buckets'].append(dict(bucket='bb', records=0,
            counts=dict(keep=0, exclude=0, quarantine=0), artifacts=files))
        atomic_json(candidate / 'manifest.json', manifest)
        done = json.loads((candidate / 'COMPLETE.json').read_text()); done['manifest_sha256'] = file_sha(candidate / 'manifest.json')
        atomic_json(candidate / 'COMPLETE.json', done)
        result = union.export_candidate_index(candidate, self.root / 'index')
        self.assertEqual(result['records'], 1)
        self.assertEqual([r['rows'] for r in result['files']], [1, 0])


if __name__ == '__main__':
    unittest.main()
