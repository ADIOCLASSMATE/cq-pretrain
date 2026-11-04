"""Adapter invariants: coverage, joins, streaming and media-free indexing."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from corpus_pipeline import adapters as a


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        a._exists.cache_clear()

    def unit(self, path, adapter, **config):
        return a._unit('genund', 'fixture', path, adapter,
                       {'source_root': str(self.root), 'collection_root': str(self.root), **config})

    def test_discovery_pairs_sidecars_once_and_preserves_missing_caption(self):
        src = self.root / 'source'
        src.mkdir()
        (src / 'a.jpg').write_bytes(b'not decoded')
        (src / 'a.txt').write_text('A red car.')
        (src / 'a.json').write_text('{"image_provenance":"natural"}')
        (src / 'b.jpg').write_bytes(b'not decoded')
        units = list(a.discover_genund(self.root))
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]['adapter'], 'loose_batch')
        records = list(a.iter_records(units[0]))
        self.assertEqual(records[0]['raw']['caption'], 'A red car.')
        self.assertIn('image_only_unpaired', records[1]['issues'])
        self.assertEqual(len(units[0]['config']['rows'][0]['fingerprint']['sidecars']), 2)

    def test_sample_discovery_is_explicitly_bounded(self):
        src = self.root / 'source'
        src.mkdir()
        for i in range(600): (src / f'{i:04}.jpg').touch()
        units = list(a.discover_genund(self.root, max_files_per_source=1))
        self.assertEqual(len(units), 1)
        scope = units[0]['config']['discovery_scope']
        self.assertEqual(scope['entries_examined'], 512)
        self.assertFalse(scope['complete'])
        self.assertFalse(scope['representative_sample'])

    def test_sample_depth_first_reaches_leaf_in_broad_tree(self):
        src = self.root / 'source'
        src.mkdir()
        for i in range(100):
            leaf = src / f'subset{i:03}' / 'nested' / 'images'
            leaf.mkdir(parents=True)
            (leaf / 'a.jpg').touch()
            (leaf / 'a.txt').write_text('caption')
        units = list(a.discover_genund(self.root, max_files_per_source=2))
        self.assertTrue(any(unit['adapter'] == 'loose_batch' for unit in units))
        record = next(a.iter_records(next(unit for unit in units if unit['adapter'] == 'loose_batch')))
        self.assertEqual(record['raw']['caption'], 'caption')
        self.assertTrue(record['image_refs'][0]['exists'])

    def test_sample_descends_before_huge_parent_exhausts_entry_budget(self):
        from contextlib import contextmanager
        src = self.root / 'source'
        src.mkdir()
        for i in range(5000):
            (src / f'part{i:05}').mkdir()
        image = src / 'part00000' / 'a.jpg'
        image.touch()
        (image.with_suffix('.txt')).write_text('caption')
        original_scandir = a.os.scandir
        @contextmanager
        def ordered_scandir(path):
            # Make the test deterministic; production preserves filesystem order.
            with original_scandir(path) as entries:
                yield iter(sorted(entries, key=lambda entry: entry.name))
        with patch.object(a.os, 'scandir', ordered_scandir):
            units = list(a.discover_genund(self.root, max_files_per_source=1))
        self.assertEqual(units[0]['adapter'], 'loose_batch')
        record = next(a.iter_records(units[0]))
        self.assertEqual(record['image_refs'][0]['path'], str(image))
        self.assertLessEqual(units[0]['config']['discovery_scope']['entries_examined'], 512)

    def test_sample_prefers_paired_images_before_unsupported_archives(self):
        src = self.root / 'source'
        src.mkdir()
        (src / 'aaa.zip').touch()
        (src / 'zzz.jpg').touch()
        (src / 'zzz.txt').write_text('caption')
        units = list(a.discover_genund(self.root, max_files_per_source=1))
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]['adapter'], 'loose_batch')

    def test_partial_sidecar_discovers_image_outside_directory_prefix(self):
        src = self.root / 'source'
        src.mkdir()
        image = src / 'a.jpg'
        image.touch()
        sidecar = src / 'a.txt'
        sidecar.write_text('caption')
        def only_sidecar(root, requested, scopes, source):
            scopes[source] = {'mode': 'bounded_filesystem_prefix', 'complete': False}
            yield sidecar, 'file'
        with patch.object(a, '_bounded_walk', only_sidecar):
            units = list(a.discover_genund(self.root, max_files_per_source=1))
        self.assertEqual(units[0]['adapter'], 'loose_batch')
        record = next(a.iter_records(units[0]))
        self.assertEqual(record['image_refs'][0]['path'], str(image))
        self.assertEqual(record['raw']['caption'], 'caption')

    def test_orphan_caption_is_not_accepted_as_plain_text(self):
        path = self.root / 'metadata.json'
        path.write_text('[{"caption":"orphan caption"}]')
        record = next(a.iter_records(self.unit(path, 'json')))
        self.assertIn('caption_image_reference_missing', record['issues'])
        path = self.root / 'orphan.txt'
        path.write_text('possible sidecar')
        record = next(a.iter_records(self.unit(path, 'text')))
        self.assertIn('orphan_text_sidecar_requires_schema', record['issues'])

    def test_json_array_streams_across_chunks_and_rejects_truncation(self):
        path = self.root / 'large.json'
        path.write_text(json.dumps([{'caption': 'a' * 100000}, {'caption': 'second'}]))
        values = list(a._json_records(path))
        self.assertEqual(len(values[0]['caption']), 100000)
        self.assertEqual(values[1]['caption'], 'second')
        path.write_text('[{"caption":"first"},')
        with self.assertRaises(ValueError): list(a._json_records(path))

    def test_jsonl_locator_and_url_reference(self):
        path = self.root / 'records.jsonl'
        path.write_text('\n{"image":"https://example.test/image.jpg","caption":"caption"}\n')
        record = next(a.iter_records(self.unit(path, 'jsonl')))
        self.assertEqual(record['locator']['line_number'], 2)
        self.assertEqual(record['image_refs'][0]['kind'], 'url')
        self.assertFalse(record['image_refs'][0]['exists'])

    def test_generic_url_caption_cannot_become_pure_text(self):
        path = self.root / 'records.json'
        path.write_text('[{"url":"https://example.test/image.jpg","caption":"image caption"}]')
        record = next(a.iter_records(self.unit(path, 'json')))
        self.assertEqual(record['image_refs'][0]['kind'], 'url')

    def test_relative_image_path_and_prefix_mapping(self):
        path = self.root / 'records.json'
        (self.root / 'a.jpg').touch()
        path.write_text('[{"image":"/old/a.jpg","caption":"x"}]')
        record = next(a.iter_records(self.unit(path, 'json', path_prefix_map={'/old': str(self.root)})))
        self.assertTrue(record['image_refs'][0]['exists'])
        self.assertEqual(record['image_refs'][0]['path'], str(self.root / 'a.jpg'))

    def test_structured_image_keeps_edit_lineage_without_adding_base_image(self):
        path = self.root / 'records.json'
        (self.root / 'edited.jpg').touch()
        for key in ('edit_source_origin', 'source_image_origin', 'original_image_origin',
                    'edit_source_provenance'):
            with self.subTest(key=key):
                value = {'origin': 'natural', 'path': 'base.jpg'} if key == 'edit_source_provenance' else 'natural'
                path.write_text(json.dumps([{'image': {'path': 'edited.jpg',
                    'image_origin': 'ai_edited', key: value}, 'caption': 'A transformed scene.'}]))
                record = next(a.iter_records(self.unit(path, 'json')))
                self.assertEqual(len(record['image_refs']), 1)
                ref = record['image_refs'][0]
                self.assertTrue(ref['exists'])
                self.assertEqual(ref['metadata']['image_origin'], 'ai_edited')
                self.assertEqual(ref['metadata'][key], value)

    def test_message_image_metadata_is_local_and_does_not_infer_edit_source(self):
        path = self.root / 'records.json'
        for name in ('natural.jpg', 'edited.jpg'):
            (self.root / name).touch()
        path.write_text(json.dumps([{'messages': [
            {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
                'url': 'natural.jpg', 'metadata': {'image_origin': 'natural'}}}]},
            {'role': 'assistant', 'content': [{'type': 'image_url', 'image_url': {
                'url': 'edited.jpg', 'image_origin': 'ai_edited'}}]},
        ]}]))
        record = next(a.iter_records(self.unit(path, 'json')))
        first, second = record['image_refs']
        self.assertEqual(first['metadata']['metadata']['image_origin'], 'natural')
        self.assertEqual(second['metadata'], {'image_origin': 'ai_edited'})
        self.assertNotIn('edit_source_origin', second['metadata'])

    def test_nested_image_locator_cannot_erase_negative_parent_evidence(self):
        path = self.root / 'records.json'
        (self.root / 'edited.jpg').touch()
        path.write_text(json.dumps([{'image': {
            'is_ai_generated': True,
            'image_url': {'url': 'edited.jpg', 'image_origin': 'natural',
                          'metadata': {'image_origin': 'ai_edited'}},
        }, 'caption': 'A scene.'}]))
        record = next(a.iter_records(self.unit(path, 'json')))
        self.assertEqual(len(record['image_refs']), 1)
        metadata = record['image_refs'][0]['metadata']
        self.assertEqual(metadata['image_origin'], 'natural')
        self.assertEqual(metadata['metadata']['image_origin'], 'ai_edited')
        self.assertIs(metadata['metadata']['metadata']['is_ai_generated'], True)

    def test_tar_out_of_order_members_and_duplicate_are_not_silently_overwritten(self):
        path = self.root / 'data.tar'
        with tarfile.open(path, 'w') as archive:
            for name, data in [('b.txt', b'B'), ('a.jpg', b'image'), ('b.jpg', b'image'),
                               ('a.txt', b'A'), ('a.txt', b'A2')]:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        records = list(a.iter_records(self.unit(path, 'tar')))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]['raw']['caption'], 'B')
        self.assertEqual(records[0]['image_refs'][0]['member'], 'a.jpg')
        self.assertIn('duplicate_tar_member', records[0]['issues'])
        self.assertEqual(records[0]['image_refs'][0]['offset_space'], 'file')

    def test_tar_compressed_offsets_are_labeled(self):
        path = self.root / 'data.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in [('a.jpg', b'image'), ('a.txt', b'caption')]:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        record = next(a.iter_records(self.unit(path, 'tar')))
        self.assertEqual(record['image_refs'][0]['offset_space'], 'decompressed_tar')

    def test_unsupported_container_fails_unit_not_empty_success(self):
        path = self.root / 'data.zip'
        path.touch()
        with self.assertRaisesRegex(ValueError, 'unsupported_container'):
            list(a.iter_records(self.unit(path, 'unsupported')))

    def test_limit_yields_exact_number(self):
        path = self.root / 'records.json'
        path.write_text('[{"text":"one"},{"text":"two"}]')
        self.assertEqual(len(list(a.iter_records(self.unit(path, 'json'), 1))), 1)
        self.assertEqual(list(a.iter_records(self.unit(path, 'json'), 0)), [])

    def test_parquet_excludes_binary_and_retains_null_uncertainty(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('pyarrow not installed')
        path = self.root / 'data.parquet'
        table = pa.table({'caption': ['first', 'second'], 'image': [b'not decoded', None]})
        pq.write_table(table, path)
        records = list(a.iter_records(self.unit(path, 'parquet')))
        self.assertEqual(records[0]['raw'], {'caption': 'first'})
        self.assertIsNone(records[0]['image_refs'][0]['exists'])
        self.assertIn('embedded_presence_unverified', records[0]['issues'])

    def test_dcvlm_rejects_invalid_published_snapshot(self):
        (self.root / 'manifest.json').write_text(json.dumps({'phase': 'complete', 'schema_version': 'dcvlm.pretrain.v1'}))
        (self.root / 'catalog.json').write_text('[{"path":"../escape.parquet","source":"test"}]')
        (self.root / 'COMPLETE.json').write_text(json.dumps({'validation_passed': True, 'catalog_sha256': 'wrong'}))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            list(a.discover_dcvlm(self.root))
        (self.root / 'COMPLETE.json').write_text(json.dumps({'validation_passed': True, 'catalog_sha256': a._sha_file(self.root / 'catalog.json')}))
        with self.assertRaisesRegex(ValueError, 'Unsafe DCVLM catalog path'):
            list(a.discover_dcvlm(self.root))
        (self.root / 'manifest.json').write_text(json.dumps({'phase': 'building', 'schema_version': 'dcvlm.pretrain.v1'}))
        with self.assertRaisesRegex(ValueError, 'phase is not complete'):
            list(a.discover_dcvlm(self.root))

    def test_dcvlm_published_catalog_and_image_locator(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('pyarrow not installed')
        (self.root / 'pool').mkdir()
        (self.root / 'pool' / 'part.tar').touch()
        (self.root / 'manifest.json').write_text(json.dumps({'source_pool': str(self.root / 'pool'), 'phase': 'complete', 'schema_version': 'dcvlm.pretrain.v1'}))
        (self.root / 'catalog.json').write_text('[{"path":"data.parquet","source":"test","rows":1,"sha256":"catalogdigest"}]')
        row = {'source_shard': 'part.tar', 'segments': [{'kind': 'image', 'image_index': 0}],
               'images': [{'member': 'x.jpg', 'offset': 512, 'length': 4, 'sha256': 'imagedigest'}]}
        pq.write_table(pa.Table.from_pylist([row]), self.root / 'data.parquet')
        (self.root / 'COMPLETE.json').write_text(json.dumps({'validation_passed': True, 'catalog_sha256': a._sha_file(self.root / 'catalog.json')}))
        unit = next(a.discover_dcvlm(self.root))
        record = next(a.iter_records(unit))
        self.assertEqual(unit['fingerprint']['sha256'], 'catalogdigest')
        self.assertEqual(record['image_refs'][0]['sha256'], 'imagedigest')
        self.assertTrue(record['image_refs'][0]['exists'])
        self.assertEqual(record['raw']['segments'][0]['image_index'], 0)


if __name__ == '__main__':
    unittest.main()
