"""Loose-source contracts preserve both views and exact image/QA bindings."""
import hashlib
import io
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from corpus_pipeline.production.common import canonical, file_sha
from .genund_loose import (BLIP, DENSE, PIXMO, prepare_blip_units, prepare_dense_units,
                           build_pixmo_image_index, prepare_pixmo_units, iter_unit_rows,
                           read_original_annotation, prepare_dense_repair_units)
from .ingest import arrow_schema

DATASET = 'local/gen-und-mm-pretrain@' + 'a' * 40


def image(path, color='red'):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (160, 128), color).save(path)
    return file_sha(path)


class LooseSourceTests(unittest.TestCase):
    def test_dense4v_sealed_legacy_zip_proof_keeps_exact_image_sha_required(self):
        from .dense_recovery import read_member
        from .test_dense_recovery import Reader
        from .pixmo_recovery import stamp
        from .genund_loose import _legacy_dense4v_zip_evidence
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); root = base / 'mount'
                local = root / DENSE / '000000/123.png'; sha = image(local)
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, 'w', compression=compression) as writer:
                    writer.writestr('123.png', local.read_bytes())
                with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as reader:
                    original, evidence = read_member(Reader(archive.getvalue()), reader.getinfo('123.png'))
                self.assertEqual(hashlib.sha256(original).hexdigest(), sha)
                value = {'image_id': '123', 'image_path': 'DenseFusion-4V-100K/000000/123.png',
                         'caption': 'Original supplemental caption.'}
                annotation = base / 'original.jsonl'; annotation.write_text(canonical(value) + '\n')
                contract = 'genund.dense4v_exact_png_range_binding.v1'
                proof = {**evidence, 'original_sha256': sha, 'revision': 'b' * 40,
                         'archive_path': 'images/DenseFusion-4V-100K/000000.zip',
                         'archive_lfs_sha256': hashlib.sha256(archive.getvalue()).hexdigest(),
                         'member': '123.png', 'full_archive_sha_validated': False}
                self.assertTrue(_legacy_dense4v_zip_evidence(proof, value['image_path'], contract))
                for changed in ({'member': '456.png'}, {'crc32': '0'}, {'method': True},
                                {'archive_path': 'images/another.zip'}, {'revision': ''},
                                {'requested_range': [0, 1]}):
                    self.assertFalse(_legacy_dense4v_zip_evidence({**proof, **changed}, value['image_path'], contract))
                binding = {'original_path': value['image_path'], 'status': 'bound',
                           'relative_path': str(local.relative_to(root)), 'local_sha256': sha, 'proof': proof}
                database = base / 'bindings.sqlite'; db = sqlite3.connect(database)
                db.execute('CREATE TABLE bindings(original_path TEXT,status TEXT,result_json TEXT)')
                db.execute('INSERT INTO bindings VALUES(?,?,?)', (value['image_path'], 'bound', canonical(binding)))
                db.commit(); db.close()
                before = file_sha(database)
                index = {'contract': contract, 'path': str(database), 'sha256': before, 'fingerprint': stamp(database)}
                unit = prepare_dense_units(root, annotation, base / 'plan', dataset_id=DATASET, binding_index=index)[0]
                rows = list(iter_unit_rows(unit)); self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]['images'], rows[1]['images'])
                self.assertEqual(rows[0]['images'][0]['sha256'], sha)
                pa.Table.from_pylist(rows, schema=arrow_schema())
                image(local, 'blue')
                with self.assertRaisesRegex(ValueError, 'does not match annotation SHA'):
                    list(iter_unit_rows(unit))
                self.assertEqual(file_sha(database), before)

    def test_dense_repaired_original_preserves_locator_and_uses_new_portable_root(self):
        from .pixmo_recovery import stamp
        from corpus_pipeline.production.records import read_asset_bytes
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'; root.mkdir()
            recovered_root = base / 'recovered'; recovered_id = 'inspire/dense-recovered@' + 'b' * 64
            sha = image(recovered_root / 'repaired.png')
            value = {'image_id': '123', 'image_path': DENSE + '/000000/123.png', 'caption': 'Unchanged original.'}
            annotation = base / 'official.jsonl'; annotation.write_text(canonical(value) + '\n')
            line = annotation.read_bytes()
            locator = {'annotation_path': str(annotation), 'offset': 0, 'length': len(line),
                       'row_index': 0, 'annotation_sha256': hashlib.sha256(line).hexdigest()}
            binding = {'original_path': value['image_path'], 'status': 'bound', 'annotation_locator': locator,
                       'relative_path': 'repaired.png', 'local_sha256': sha,
                       'asset_dataset_id': recovered_id, 'asset_root': str(recovered_root),
                       'proof': {'method': 'recovered_official_zip_png', 'original_sha256': sha}}
            database = base / 'bindings.sqlite'; db = sqlite3.connect(database)
            db.execute('CREATE TABLE bindings(original_path TEXT,status TEXT,result_json TEXT)')
            db.execute('INSERT INTO bindings VALUES(?,?,?)', (value['image_path'], 'bound', canonical(binding)))
            db.commit(); db.close()
            index = {'contract': 'genund.dense1m_recovered_png_binding.v1', 'path': str(database),
                     'sha256': file_sha(database), 'fingerprint': stamp(database)}
            units = prepare_dense_repair_units(root, annotation, index, base / 'plan', dataset_id=DATASET)
            self.assertEqual(units[0]['portable_datasets'], [recovered_id])
            self.assertEqual(units[0]['recovered_root'], str(recovered_root))
            rows = list(iter_unit_rows(units[0])); self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['images'], rows[1]['images'])
            self.assertEqual(read_asset_bytes(rows[0]['images'][0], roots={recovered_id: str(recovered_root)}),
                             (recovered_root / 'repaired.png').read_bytes())
            for row in rows:
                self.assertEqual(read_original_annotation(json.loads(row['original_locator_json'])), value)
                self.assertEqual(json.loads(row['original_locator_json'])['annotation_variant'], 'repaired_original_1m')
            self.assertFalse((root / value['image_path']).exists())

    def test_pixmo_recovered_original_root_and_unbound_qa_remain_distinct(self):
        from .pixmo_recovery import _save_official_image, stamp
        from corpus_pipeline.production.records import read_asset_bytes
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'; root.mkdir()
            recovered_root = base / 'recovered'
            source = io.BytesIO(); Image.new('RGB', (11, 13), 'red').save(source, format='PNG')
            blob = source.getvalue(); sha = hashlib.sha256(blob).hexdigest()
            recovered_id = 'inspire/pixmo-official@' + 'b' * 64
            binding = _save_official_image(blob, sha, recovered_root, recovered_id)
            binding['original_sha256'] = sha
            with self.assertRaisesRegex(ValueError, 'differs'):
                _save_official_image(blob, 'e' * 64, recovered_root, recovered_id)
            values = [{'image_url': 'https://example.test/a', 'image_sha256': key,
                       'question': question, 'answer': 'Red.'} for key, question in
                      [(sha, 'What color?'), (sha, 'What is visible?'), ('f' * 64, 'Unbound QA?')]]
            annotation = base / 'qa.parquet'; pq.write_table(pa.Table.from_pylist(values), annotation)
            database = base / 'bindings.sqlite'; db = sqlite3.connect(database)
            db.execute('CREATE TABLE bindings(original_sha256 TEXT,status TEXT,result_json TEXT)')
            db.execute('INSERT INTO bindings VALUES(?,?,?)', (sha, 'bound', canonical(binding)))
            db.execute('INSERT INTO bindings VALUES(?,?,?)', ('f' * 64, 'unbound', canonical({'original_sha256': 'f' * 64, 'status': 'unbound'})))
            db.commit(); db.close()
            index = {'contract': 'genund.pixmo_verified_reencoding.v1', 'path': str(database),
                     'sha256': file_sha(database), 'fingerprint': stamp(database)}
            units = prepare_pixmo_units(root, [annotation], None, base / 'plan', dataset_id=DATASET, binding_index=index)
            self.assertEqual(units[0]['portable_datasets'], [recovered_id])
            self.assertEqual(units[0]['recovered_asset_bindings'], {recovered_id: str(recovered_root)})
            self.assertEqual(json.loads((base / 'plan/PLAN.json').read_text())['identity']['portable_datasets'],
                             [recovered_id])
            rows = [row for unit in units for row in iter_unit_rows(unit)]
            self.assertEqual(len(rows), 3); self.assertEqual(rows[0]['images'], rows[1]['images'])
            pa.Table.from_pylist(rows, schema=arrow_schema())
            self.assertEqual(read_asset_bytes(rows[0]['images'][0], roots={recovered_id: str(recovered_root)}), blob)
            from corpus_pipeline.production.portable_refs import resolve_ref
            # This is the same allow-list gate used by production.load_raw_record,
            # populated exactly as parallel_ingest.finalize_source builds it.
            allowed = sorted({value for unit in units for value in
                              [unit['dataset_id'], *unit.get('portable_datasets', [])]})
            self.assertTrue(resolve_ref(rows[0]['images'][0], {recovered_id: str(recovered_root)},
                                        allowed_datasets=allowed).is_file())
            with self.assertRaisesRegex(ValueError, 'not allowed'):
                resolve_ref(rows[0]['images'][0], {recovered_id: str(recovered_root)},
                            allowed_datasets=[DATASET])
            self.assertEqual(rows[2]['images'], [])
            self.assertEqual(json.loads(rows[2]['adapter_metadata_json'])['prepared_adapter_status'], 'quarantine')
            self.assertEqual(json.loads(rows[2]['original_supervision_json'])['source_annotation'], values[2])
            self.assertEqual(read_original_annotation(json.loads(rows[0]['original_locator_json'])), values[0])

    def test_blip_two_views_one_asset_movable_refs_original_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'
            for stem in ('one', 'two', 'three'):
                image(root / BLIP / 'raw' / (stem + '.jpg'))
                (root / BLIP / 'raw' / (stem + '.txt')).write_text('原始 caption\n')
            before = file_sha(root / BLIP / 'raw/one.jpg')
            units = prepare_blip_units(root, base / 'plan', dataset_id=DATASET, records_per_unit=2)
            self.assertEqual([unit['record_count'] for unit in units], [2, 1])
            with patch('corpus_pipeline.unified.genund_loose.os.scandir', side_effect=AssertionError('rescan')):
                self.assertEqual(prepare_blip_units(root, base / 'plan', dataset_id=DATASET, records_per_unit=2), units)
                rows = [row for unit in units for row in iter_unit_rows(unit)]
            self.assertEqual(len(rows), 6)
            pa.Table.from_pylist(rows, schema=arrow_schema())
            for left, right in zip(rows[::2], rows[1::2]):
                self.assertEqual(left['images'], right['images'])
                self.assertNotEqual(left['sample_id'], right['sample_id'])
                self.assertEqual([x['kind'] for x in left['segments']], ['image', 'text'])
                self.assertEqual([x['kind'] for x in right['segments']], ['text', 'image'])
                self.assertFalse(json.loads(left['original_supervision_json'])['image_loss_allowed'])
                self.assertTrue(json.loads(right['original_supervision_json'])['image_loss_allowed'])
                self.assertEqual(read_original_annotation(json.loads(left['original_locator_json']),
                                                          dataset_roots={DATASET: str(root)}), '原始 caption\n')
                self.assertFalse(Path(left['images'][0]['path']).is_absolute())
            self.assertEqual(file_sha(root / BLIP / 'raw/one.jpg'), before)
            self.assertFalse(any(path.suffix == '.jpg' for path in (base / 'plan').rglob('*')))

    def test_blip_pairing_is_exact_and_missing_text_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'
            image(root / BLIP / 'raw/one.jpg')
            (root / BLIP / 'raw/two.txt').write_text('Not paired.')
            units = prepare_blip_units(root, base / 'plan', dataset_id=DATASET)
            with self.assertRaises(FileNotFoundError):
                list(iter_unit_rows(units[0]))

    def test_dense_streamed_ranges_preserve_every_caption_and_original_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'
            image(root / DENSE / '000000/123.png')
            image(root / DENSE / '000001/456.png', 'blue')
            originals = [{'image_id': key, 'image_path': DENSE + '/' + shard + '/' + key + '.png',
                          'caption': caption, 'url': 'https://example.test/' + key}
                         for key, shard, caption in [('123', '000000', 'First caption.'),
                                                     ('123', '000000', 'Another caption.'),
                                                     ('456', '000001', 'Different image.')]]
            annotation = base / 'original.jsonl'
            annotation.write_text(''.join(canonical(value) + '\n' for value in originals))
            before = file_sha(annotation)
            units = prepare_dense_units(root, annotation, base / 'plan', dataset_id=DATASET, records_per_unit=2)
            self.assertEqual([unit['record_count'] for unit in units], [2, 1])
            rows = [row for unit in units for row in iter_unit_rows(unit)]
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]['images'], rows[2]['images'])
            self.assertNotEqual(rows[0]['text_hash'], rows[2]['text_hash'])
            self.assertEqual({row['segments'][1]['text'] for row in rows[::2]},
                             {value['caption'] for value in originals})
            for row, original in zip(rows[::2], originals):
                self.assertEqual(read_original_annotation(json.loads(row['original_locator_json'])), original)
            self.assertEqual(file_sha(annotation), before)
            annotation.write_text(annotation.read_text().replace('First caption.', 'Other caption.'))
            with self.assertRaisesRegex(ValueError, 'annotation (file changed|byte range SHA mismatch)'):
                list(iter_unit_rows(units[0]))

    def test_dense_rejects_image_id_path_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'
            image(root / DENSE / '000000/123.png')
            annotation = base / 'original.jsonl'
            annotation.write_text(canonical({'image_id': '456', 'image_path': DENSE + '/000000/123.png', 'caption': 'Text.'}) + '\n')
            unit = prepare_dense_units(root, annotation, base / 'plan', dataset_id=DATASET)[0]
            with self.assertRaisesRegex(ValueError, 'binding disagree'):
                list(iter_unit_rows(unit))

    def test_dense_missing_and_empty_images_keep_ordered_original_in_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'
            (root / DENSE / '000000').mkdir(parents=True)
            (root / DENSE / '000000/empty.png').write_bytes(b'')
            originals = [{'image_id': key, 'image_path': DENSE + '/000000/' + key + '.png',
                          'caption': 'Original caption ' + key} for key in ('missing', 'empty')]
            annotation = base / 'original.jsonl'
            annotation.write_text(''.join(canonical(value) + '\n' for value in originals))
            unit = prepare_dense_units(root, annotation, base / 'plan', dataset_id=DATASET)[0]
            rows = list(iter_unit_rows(unit))
            self.assertEqual(len(rows), 4)
            pa.Table.from_pylist(rows, schema=arrow_schema())
            for ordinal, row in enumerate(rows):
                metadata = json.loads(row['adapter_metadata_json'])
                original = json.loads(row['original_supervision_json'])
                self.assertEqual(metadata['prepared_adapter_status'], 'quarantine')
                self.assertEqual(metadata['binding_contract'], 'unresolved_original_image_binding.v1')
                self.assertEqual(metadata['pairing_state'], 'unresolved')
                self.assertEqual(row['images'], [])
                self.assertFalse(original['image_loss_allowed'])
                self.assertEqual(original['source_annotation'], originals[ordinal // 2])
                self.assertEqual([s['kind'] for s in original['segments']],
                                 ['image', 'text'] if ordinal % 2 == 0 else ['text', 'image'])

    def test_pixmo_image_index_resumes_completed_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'
            first = root / PIXMO / '0001/one.jpg'
            second = root / PIXMO / '0002/two.jpg'
            image(first); image(second, 'blue')
            def interrupted(path):
                if Path(path) == second:
                    raise OSError('simulated read interruption')
                return file_sha(path)
            with patch('corpus_pipeline.unified.genund_loose.file_sha', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'read interruption'):
                    build_pixmo_image_index(root, base / 'images.sqlite')
            self.assertFalse((base / 'images.sqlite').exists())
            def resumed(path):
                if Path(path) == first:
                    raise AssertionError('completed image was hashed again')
                return file_sha(path)
            with patch('corpus_pipeline.unified.genund_loose.file_sha', side_effect=resumed):
                result = build_pixmo_image_index(root, base / 'images.sqlite')
            self.assertEqual(result, build_pixmo_image_index(root, base / 'images.sqlite'))

    def test_pixmo_all_qa_preserved_single_asset_no_positional_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'mount'
            checksum = image(root / PIXMO / '0000/999.jpg')
            values = [{'image_url': 'https://example.test/image', 'image_sha256': checksum,
                       'question': question, 'answer': answer} for question, answer in
                      [('Color?', 'Red.'), ('Shape?', 'A rectangle.')]]
            annotation = base / 'qa.parquet'
            pq.write_table(pa.Table.from_pylist(values), annotation, row_group_size=1)
            index = build_pixmo_image_index(root, base / 'images.sqlite')
            units = prepare_pixmo_units(root, [annotation], index, base / 'plan', dataset_id=DATASET)
            rows = [row for unit in units for row in iter_unit_rows(unit)]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['images'], rows[1]['images'])
            self.assertEqual([s['text'] for s in rows[0]['segments']], [None, 'Color?', 'Red.'])
            self.assertEqual([s['role'] for s in rows[0]['segments']], ['user', 'user', 'assistant'])
            for row, original in zip(rows, values):
                self.assertFalse(json.loads(row['original_supervision_json'])['image_loss_allowed'])
                self.assertEqual(read_original_annotation(json.loads(row['original_locator_json'])), original)
            other = base / 'unbound.parquet'
            pq.write_table(pa.Table.from_pylist([{**values[0], 'image_sha256': 'f' * 64}]), other)
            unit = prepare_pixmo_units(root, [other], index, base / 'other-plan', dataset_id=DATASET)[0]
            with self.assertRaisesRegex(ValueError, 'no verified local image binding'):
                list(iter_unit_rows(unit))


if __name__ == '__main__':
    unittest.main()
