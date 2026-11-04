"""Fixture tests only: these do not claim a mounted gen-und validation run."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image

from .common import digest
from .genund_ingest import build_manifest, snapshot_for_manifest, iter_genund_records
from .records import inspect_asset, load_original_record


def fixture(root, members=None, source='laion-aesthetics-12m'):
    directory = Path(root) / source
    directory.mkdir()
    buffer = io.BytesIO()
    Image.new('RGB', (160, 128), 'green').save(buffer, 'PNG')
    members = members or [('nested/a.png', buffer.getvalue()), ('nested/a.txt', b'A green fixture image.'),
                          ('nested/a.json', json.dumps({'caption': 'Alternative text.', 'is_ai_generated': True}).encode())]
    path = directory / '00000.tar'
    with tarfile.open(path, 'w') as archive:
        for name, blob in members:
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            archive.addfile(member, io.BytesIO(blob))
    manifest = build_manifest(root, source, [path])
    return manifest, path, buffer.getvalue()


class GenundIngestTests(unittest.TestCase):
    def test_empty_caption_task_policy_is_recomputed_after_direction_recovery(self):
        for direction in ('i2t','t2i'):
            with self.subTest(direction=direction),tempfile.TemporaryDirectory() as directory:
                _,path,_=fixture(directory,[('a.jpg',b'fixture'),('a.txt',b''),
                    ('a.json',b'{"is_ai_generated":true}')])
                manifest=build_manifest(directory,'laion-aesthetics-12m',[path],
                                        preserve_all_captions=True,direction=direction)
                row,=list(iter_genund_records(manifest,snapshot_for_manifest(manifest),
                                             {'task_routing':'corpus.task_policy.v1'}))
                self.assertEqual(row['normalized']['category'],'single_'+direction)
                self.assertEqual(row['normalized']['task_route'],direction)
                self.assertEqual(row['normalized']['image_loss_allowed'],direction=='t2i')
                self.assertEqual('image_ai_generated' in row['normalized']['reason_codes'],direction=='t2i')
                self.assertEqual(load_original_record(row['locator']),row['raw'])

    def test_unified_ingest_consumes_all_captions_and_direction_permissions(self):
        from corpus_pipeline.unified.ingest import build
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            _, path, _ = fixture(source)
            before = path.read_bytes()
            output = root / 'native'
            spec = {'collection': 'genund', 'adapter': 'genund_caption_tar', 'root': str(source),
                    'source': 'laion-aesthetics-12m', 'preserve_all_captions': True, 'direction': 'i2t'}
            result = build([spec], output, rows_per_file=1, rules={'task_routing': 'corpus.task_policy.v1'})
            self.assertEqual(result['statistics']['samples'], 2)
            catalog = json.loads((output / 'catalog.json').read_text())
            rows = [row for item in catalog for row in pq.read_table(output / item['path']).to_pylist()]
            self.assertEqual({row['segments'][1]['text'] for row in rows},
                             {'A green fixture image.', 'Alternative text.'})
            for row in rows:
                metadata = json.loads(row['adapter_metadata_json'])
                original = json.loads(row['original_supervision_json'])
                self.assertEqual(metadata['direction'], 'i2t')
                self.assertFalse(metadata['image_loss_allowed'])
                self.assertEqual(metadata['processing_policy'], 'corpus.task_policy.v1')
                self.assertEqual(original['processing_policy'], metadata['processing_policy'])
                self.assertEqual(load_original_record(json.loads(row['original_locator_json'])), original)
            self.assertEqual(path.read_bytes(), before)

    def test_explicit_all_captions_are_consumable_and_rebuildable(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _ = fixture(directory)
            manifest = build_manifest(directory, 'laion-aesthetics-12m', [path],
                                      preserve_all_captions=True, direction='i2t')
            rows = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(len(rows), 2)
            self.assertEqual({r['raw']['segments'][1]['text'] for r in rows},
                             {'A green fixture image.', 'Alternative text.'})
            self.assertEqual(len({r['raw']['sample_id'] for r in rows}), 2)
            self.assertEqual(len({r['source_image_refs'][0]['sha256'] for r in rows}), 1)
            for row in rows:
                self.assertEqual(row['normalized']['category'], 'single_i2t')
                self.assertFalse(row['raw']['image_loss_allowed'])
                self.assertEqual(row['raw']['segments'][0]['kind'], 'image')
                self.assertEqual(row['caption_contract']['target_segment_index'], 1)
                self.assertEqual(load_original_record(row['locator']), row['raw'])
                self.assertEqual(digest(load_original_record(row['locator'])), row['source_record_sha256'])
                self.assertEqual(len(row['raw']['source_annotations']), 2)
            changed = deepcopy(rows[0]['locator'])
            changed['caption_variant']['fields'] = ['missing.txt']
            with self.assertRaisesRegex(ValueError, 'not bound'):
                load_original_record(changed)

    def test_all_caption_contract_collapses_exact_text_but_preserves_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _ = fixture(directory, [('a.jpg', b'fixture'), ('a.txt', b'Caption\r\ntext'),
                ('a.json', b'{"caption":"Caption\\ntext"}')])
            manifest = build_manifest(directory, 'laion-aesthetics-12m', [path], preserve_all_captions=True)
            row, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(row['raw']['caption_variant_fields'], ['a.txt', 'a.json#caption'])
            self.assertEqual(row['raw']['segments'][0]['text'], 'Caption\r\ntext')
            self.assertEqual(load_original_record(row['locator']), row['raw'])
            self.assertEqual(len(row['caption_candidates']), 2)

    def test_all_caption_mode_does_not_turn_url_placeholder_into_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _ = fixture(directory, [('a.jpg', b'fixture'), ('a.txt', b'https://example.test/img'),
                ('a.json', b'{"caption":"A real description."}')])
            manifest = build_manifest(directory, 'laion-aesthetics-12m', [path], preserve_all_captions=True)
            row, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(row['raw']['caption_variant_fields'], ['a.json#caption'])
            self.assertIn('https://example.test/img', row['raw']['source_annotations'].values())

    def test_caption_options_are_bound_and_invalid_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _ = fixture(directory)
            for options in ({'direction': 'guess'}, {'preserve_all_captions': 'true'}):
                with self.assertRaises(ValueError):
                    build_manifest(directory, 'laion-aesthetics-12m', [path], **options)
            manifest = build_manifest(directory, 'laion-aesthetics-12m', [path], preserve_all_captions=True)
            manifest['adapter_options']['direction'] = 'i2t'
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                snapshot_for_manifest(manifest)

    def test_url_or_empty_txt_does_not_shadow_real_json_caption(self):
        for bad_text in (b'https://flickr.com/photo123', b''):
            with self.subTest(bad_text=bad_text), tempfile.TemporaryDirectory() as directory:
                manifest, _, _ = fixture(directory, [('a.jpg', b'fixture'), ('a.txt', bad_text),
                    ('a.json', b'{"caption":"A person standing by a tree."}')])
                record, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
                self.assertEqual(record['raw']['segments'][0]['text'], 'A person standing by a tree.')
                self.assertEqual(record['raw']['selected_caption_field'], 'a.json#caption')
                self.assertEqual(record['raw']['source_annotations']['a.txt'], bad_text.decode())
                self.assertEqual(record['normalized']['category'], 'single_t2i')
                self.assertEqual(load_original_record(record['locator']), record['raw'])

    def test_fixed_t2i_contract_preserves_candidates_and_negative_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _ = fixture(directory)
            record, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(record['normalized']['category'], 'single_t2i')
            self.assertEqual(record['normalized']['decision'], 'rejected')
            self.assertEqual(record['normalized']['provenance']['images'][0]['origin'], 'ai_generated')
            self.assertEqual(len(record['caption_candidates']), 2)
            self.assertEqual(record['pairing_state'], 'verified')
            self.assertEqual(record['prepared_adapter_status'], 'ready')
            self.assertEqual(record['raw']['segments'][0]['kind'], 'text')
            self.assertEqual(record['raw']['segments'][1]['kind'], 'image')
            asset = inspect_asset(record['source_image_refs'][0], record['source'], 'fixture')
            self.assertEqual((asset['width'], asset['height']), (160, 128))
            original = load_original_record(record['locator'])
            self.assertEqual(original, record['raw'])
            self.assertEqual(digest(original), record['source_record_sha256'])

    def test_blip_existing_ai_caption_is_marked_without_claiming_ai_image(self):
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.BytesIO()
            Image.new('RGB', (128, 128), 'white').save(buffer, 'PNG')
            manifest, _, _ = fixture(directory, [('a.png', buffer.getvalue()), ('a.txt', b'A white image.')], source='BLIP3o-Pretrain-Long-Caption')
            record, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(record['normalized']['provenance']['text']['origin'], 'ai')
            self.assertEqual(record['normalized']['provenance']['images'][0]['origin'], 'unknown')
            self.assertEqual(record['normalized']['decision'], 'quarantine')

    def test_missing_wrong_stem_and_ambiguous_images_require_metadata_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _ = fixture(directory, [('a.jpg', b'fixture'), ('b.txt', b'Does not pair with a.')])
            records = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r['prepared_adapter_status'] == 'repair_metadata' for r in records))
            self.assertTrue(all(r['pairing_state'] == 'unresolved' for r in records))
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _ = fixture(directory, [('a.jpg', b'x'), ('a.png', b'y'), ('a.txt', b'A caption.')])
            record, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(record['prepared_adapter_status'], 'repair_metadata')
            self.assertEqual(len(record['source_image_refs']), 2)

    def test_empty_existing_caption_is_explicitly_bound_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _ = fixture(directory, [('a.jpg', b'fixture'), ('a.txt', b'')])
            record, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(record['normalized']['category'], 'single_t2i')
            self.assertEqual(record['prepared_adapter_status'], 'ready')
            self.assertTrue(record['caption_fields_inspected'])
            self.assertEqual(record['raw']['segments'][0]['text'], '')
            self.assertIn('missing_text', record['normalized']['reason_codes'])
            self.assertEqual(load_original_record(record['locator']), record['raw'])

    def test_qa_json_is_not_reinterpreted_as_caption(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _ = fixture(directory, [('a.jpg', b'fixture'), ('a.txt', b'A picture.'),
                ('a.json', b'{"question":"How many?","answer":"42"}')])
            record, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(record['prepared_adapter_status'], 'repair_metadata')
            self.assertIn('unsupported_task_annotation_requires_adapter', record['normalized']['issues'])

    def test_unknown_source_paths_changes_and_duplicate_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, path, _ = fixture(directory)
            with self.assertRaisesRegex(ValueError, 'unsupported gen-und source'):
                build_manifest(directory, 'unknown', [path])
            with self.assertRaisesRegex(ValueError, 'below its declared source'):
                build_manifest(directory, 'BLIP3o-Pretrain-Long-Caption', [path])
            with self.assertRaisesRegex(ValueError, 'unique'):
                build_manifest(directory, 'laion-aesthetics-12m', [path, path])
            changed = deepcopy(manifest)
            changed['shards'][0]['fingerprint']['mtime_ns'] += 1
            with self.assertRaisesRegex(ValueError, 'changed'):
                snapshot_for_manifest(changed)
            record, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            changed_locator = deepcopy(record['locator'])
            changed_locator['components'][0]['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                load_original_record(changed_locator)

    def test_duplicate_tar_members_and_unsafe_paths_do_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _ = fixture(directory, [('a.jpg', b'fixture'), ('a.txt', b'One'), ('a.txt', b'Two')])
            record, = list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))
            self.assertEqual(record['prepared_adapter_status'], 'repair_metadata')
            self.assertIn('duplicate_tar_member', record['normalized']['issues'])
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _ = fixture(directory, [('../a.jpg', b'fixture')])
            with self.assertRaisesRegex(ValueError, 'unsafe TAR member'):
                list(iter_genund_records(manifest, snapshot_for_manifest(manifest), {}))


if __name__ == '__main__':
    unittest.main()
