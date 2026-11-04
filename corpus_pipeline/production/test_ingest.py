"""Input integrity, streaming boundaries and source-specific task protection."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from .common import file_sha
from .ingest import build_shard_manifest, iter_records
from .records import prepared_caption_contract, snapshot


def fixture(root):
    root = Path(root)
    (root / 'data').mkdir()
    pool = root / 'source_pool'
    (pool / 'pixmo_cap').mkdir(parents=True)
    (pool / 'pixmo_cap/000000.tar').write_bytes(b'fixture locator only')
    rows = []
    for index in range(6):
        member = str(index) + '.jpg'
        rows.append({'sample_id': str(index).zfill(64), 'source': 'pixmo_cap',
                     'source_shard': 'pixmo_cap/000000.tar', 'source_key': str(index),
                     'format': 'conversation', 'modality': 'multimodal', 'image_count': 1,
                     'segments': [{'kind': 'image', 'role': 'user', 'text': None, 'image_index': 0,
                                   'source_member': member, 'turn': 0},
                                  {'kind': 'text', 'role': 'user', 'text': 'Please provide a descriptive caption for this image.',
                                   'image_index': None, 'source_member': str(index) + '.conversations.txt', 'turn': 0},
                                  {'kind': 'text', 'role': 'assistant', 'text': 'A real caption ' + str(index),
                                   'image_index': None, 'source_member': str(index) + '.conversations.txt', 'turn': 1}],
                     'images': [{'member': member, 'offset': 0, 'length': 1, 'sha256': str(index).zfill(64), 'width': 200, 'height': 200}]})
    path = root / 'data/000000.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    (root / 'catalog.json').write_text(json.dumps([{'path': 'data/000000.parquet', 'rows': len(rows),
        'bytes': path.stat().st_size, 'source': 'pixmo_cap', 'source_shard': 'pixmo_cap/000000.tar'}]))
    (root / 'manifest.json').write_text(json.dumps({'schema_version': 'dcvlm.pretrain.v1', 'phase': 'complete', 'source_pool': str(pool)}))
    (root / 'COMPLETE.json').write_text(json.dumps({'validation_passed': True, 'catalog_sha256': file_sha(root / 'catalog.json')}))
    return rows


class IngestTests(unittest.TestCase):
    def test_equal_rgba_bytes_at_different_geometry_do_not_exact_deduplicate(self):
        import hashlib
        import io
        from PIL import Image
        from .records import inspect_asset
        with tempfile.TemporaryDirectory() as directory:
            assets = []
            for index, size in enumerate(((128, 256), (256, 128))):
                stream = io.BytesIO()
                Image.new('RGB', size, 'white').save(stream, 'PNG')
                blob = stream.getvalue()
                path = Path(directory) / f'{index}.tar'
                path.write_bytes(blob)
                assets.append(inspect_asset({'path': str(path), 'offset': 0, 'length': len(blob),
                                             'sha256': hashlib.sha256(blob).hexdigest()}, 'fixture', 'photo'))
            self.assertNotEqual(assets[0]['pixel_sha256'], assets[1]['pixel_sha256'])
            self.assertEqual(assets[0]['pixel_identity_version'], 'sha256_rgba_geometry_v2')

    def test_source_and_prompt_contract_does_not_reclassify_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = fixture(directory)[0]
        self.assertEqual(prepared_caption_contract(raw, 'single_i2t')['allowed_training_lengths'], ['short', 'medium', 'long'])
        raw['segments'][1]['text'] = 'Generate a short caption describing what you see in this image.'
        self.assertEqual(prepared_caption_contract(raw, 'single_i2t')['allowed_training_lengths'], ['short'])
        raw['source'] = 'pixmo_ask_model_anything'
        self.assertIsNone(prepared_caption_contract(raw, 'single_i2t'))
        raw['source'] = 'pixmo_cap'
        raw['segments'][1]['text'] = 'How many objects are there?'
        self.assertIsNone(prepared_caption_contract(raw, 'single_i2t'))
        self.assertIsNone(prepared_caption_contract(raw, 'multi_i2t'))

    def test_row_slice_crosses_row_groups_and_batches_without_direction_change(self):
        with tempfile.TemporaryDirectory() as directory:
            originals = fixture(directory)
            manifest = build_shard_manifest(directory)
            manifest['shards'][0].update(row_start=1, row_stop=5)
            rows = list(iter_records(manifest, snapshot(directory), {}, batch_size=1))
            self.assertEqual([r['original_sample_id'] for r in rows], [r['sample_id'] for r in originals[1:5]])
            self.assertEqual([r['locator']['row_group'] for r in rows], [0, 1, 1, 2])
            self.assertTrue(all(r['raw']['segments'] == originals[i + 1]['segments'] for i, r in enumerate(rows)))
            self.assertTrue(all(r['normalized']['category'] == 'single_i2t' for r in rows))
            self.assertTrue(all(r['pairing_state'] == 'verified' and r['caption_fields_inspected'] for r in rows))
            self.assertEqual(len(list(iter_records(manifest, snapshot(directory), {}, batch_size=2, max_records=2))), 2)

    def test_input_fingerprint_and_duplicate_ranges_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory)
            manifest = build_shard_manifest(directory)
            snap = snapshot(directory)
            manifest['shards'].append(deepcopy(manifest['shards'][0]))
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                list(iter_records(manifest, snap, {}))
            manifest['shards'].pop()
            manifest['shards'][0]['parquet_fingerprint']['mtime_ns'] += 1
            with self.assertRaisesRegex(ValueError, 'changed'):
                list(iter_records(manifest, snap, {}))

    def test_native_image_binding_and_catalog_source_are_verified(self):
        from .records import load_raw_record
        with tempfile.TemporaryDirectory() as directory:
            originals = fixture(directory)
            manifest = build_shard_manifest(directory)
            spec = {**manifest['shards'][0], 'parquet_row_index': 0}
            snap = snapshot(directory)
            raw = deepcopy(originals[0])
            raw['segments'][0]['source_member'] = 'another.jpg'
            with self.assertRaisesRegex(ValueError, 'member binding'):
                load_raw_record(raw, spec, snap, {})
            raw = deepcopy(originals[0])
            raw['source'] = 'another_source'
            with self.assertRaisesRegex(ValueError, 'catalog'):
                load_raw_record(raw, spec, snap, {})

    def test_unknown_caption_prompt_routes_to_metadata_repair(self):
        from .records import load_raw_record
        with tempfile.TemporaryDirectory() as directory:
            originals = fixture(directory)
            manifest = build_shard_manifest(directory)
            raw = originals[0]
            raw['segments'][1]['text'] = 'Unregistered source task.'
            record = load_raw_record(raw, {**manifest['shards'][0], 'parquet_row_index': 0}, snapshot(directory), {})
            self.assertEqual(record['prepared_adapter_status'], 'repair_metadata')
            self.assertFalse(record['caption_fields_inspected'])
            self.assertIsNone(record['caption_contract'])

    def test_explicit_empty_native_caption_recovers_only_original_i2t_direction(self):
        from .records import load_raw_record
        with tempfile.TemporaryDirectory() as directory:
            originals = fixture(directory)
            manifest = build_shard_manifest(directory)
            raw = originals[0]
            raw['format'] = 'caption'
            raw['segments'] = [raw['segments'][0], raw['segments'][2]]
            for segment in raw['segments']:
                segment['role'] = None
                segment['turn'] = None
            raw['segments'][1]['text'] = ''
            spec = {**manifest['shards'][0], 'parquet_row_index': 0}
            record = load_raw_record(raw, spec, snapshot(directory), {})
            self.assertEqual(record['normalized']['category'], 'single_i2t')
            self.assertEqual(record['normalized']['adapter_original_classification']['category'], 'unclassified')
            self.assertIn('missing_text', record['normalized']['reason_codes'])
            self.assertTrue(record['caption_fields_inspected'])
            raw['segments'].reverse()
            reversed_record = load_raw_record(raw, spec, snapshot(directory), {})
            self.assertEqual(reversed_record['prepared_adapter_status'], 'repair_metadata')
            self.assertFalse(reversed_record['caption_fields_inspected'])

    def test_unsupported_collection_and_snapshot_mismatch_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory)
            manifest = build_shard_manifest(directory)
            snap = snapshot(directory)
            manifest['collection'] = 'genund'
            with self.assertRaisesRegex(ValueError, 'explicit source adapter'):
                list(iter_records(manifest, snap, {}))
            manifest['collection'] = 'dcvlm'
            manifest['snapshot_fingerprint'] = {}
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                list(iter_records(manifest, snap, {}))


if __name__ == '__main__':
    unittest.main()
