"""Real decoder/TAR and read-only SQLite tests; no inference or network."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from . import inputs


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='codex-input-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.number = 0

    def picture(self, size=(40, 20), *, mode='RGB', color='red', format='PNG', **save):
        data = io.BytesIO()
        Image.new(mode, size, color).save(data, format=format, **save)
        return data.getvalue()

    def item(self, raw):
        path = self.root / f'{self.number}.tar'
        self.number += 1
        with tarfile.open(path, 'w') as archive:
            member = tarfile.TarInfo('image.bin')
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        with tarfile.open(path, 'r') as archive:
            offset = archive.getmember('image.bin').offset_data
        aid = hashlib.sha256(raw).hexdigest()
        return {'item_id': aid, 'source_ref': {'kind': 'tar', 'path': str(path),
                'offset': offset, 'length': len(raw), 'sha256': aid}, 'source_bytes': len(raw),
                'source_container_bytes': path.stat().st_size, 'source_mtime_ns': path.stat().st_mtime_ns}

    def test_real_decode_hashes_and_metadata_free_attachment(self):
        item = self.item(self.picture())
        evidence = inputs.prepare_image(item, self.root / 'attachments', 0)
        self.assertTrue(evidence['decoded'])
        self.assertFalse(evidence['crop'])
        self.assertEqual((evidence['decoded_width'], evidence['decoded_height']), (40, 20))
        self.assertEqual(evidence['source_sha256'], item['item_id'])
        self.assertEqual(evidence['sent_sha256'], hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest())
        with Image.open(evidence['path']) as image:
            image.load()
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(image.info, {})

    def test_whole_image_resize_preserves_both_ends_and_aspect(self):
        picture = Image.new('RGB', (400, 100), 'black')
        picture.paste('red', (0, 0, 100, 100)); picture.paste('blue', (300, 0, 400, 100))
        raw = io.BytesIO(); picture.save(raw, format='PNG')
        evidence = inputs.prepare_image(self.item(raw.getvalue()), self.root / 'a', 0,
                                        max_side=200, max_pixels=10000)
        self.assertEqual((evidence['width'], evidence['height']), (200, 50))
        with Image.open(evidence['path']) as image:
            self.assertEqual(image.getpixel((0, 20)), (255, 0, 0))
            self.assertEqual(image.getpixel((199, 20)), (0, 0, 255))

    def test_exif_orientation_is_normalized(self):
        exif = Image.Exif(); exif[274] = 6
        evidence = inputs.prepare_image(self.item(self.picture((20, 40), format='JPEG', exif=exif)),
                                        self.root / 'a', 0)
        self.assertEqual((evidence['decoded_width'], evidence['decoded_height']), (20, 40))
        self.assertEqual((evidence['width'], evidence['height']), (40, 20))
        with Image.open(evidence['path']) as image:
            self.assertFalse(image.getexif())

    def test_large_jpeg_icc_is_stripped_without_changing_pillow_limits(self):
        before = Image.MAX_IMAGE_PIXELS
        evidence = inputs.prepare_image(self.item(self.picture(format='JPEG', icc_profile=b'x' * (2 << 20))),
                                        self.root / 'a', 0)
        self.assertIn('icc_profile', evidence['removed_metadata_keys'])
        with Image.open(evidence['path']) as image:
            image.load(); self.assertNotIn('icc_profile', image.info)
        self.assertEqual(Image.MAX_IMAGE_PIXELS, before)

    def test_alpha_uses_explicit_white_background(self):
        evidence = inputs.prepare_image(self.item(self.picture(mode='RGBA', color=(0, 0, 0, 0))),
                                        self.root / 'a', 0)
        self.assertEqual(evidence['alpha_background'], 'white')
        with Image.open(evidence['path']) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_multiframe_is_rejected_not_silently_frame_zero(self):
        raw = io.BytesIO()
        Image.new('RGB', (20, 20), 'red').save(raw, format='GIF', save_all=True,
                                             append_images=[Image.new('RGB', (20, 20), 'blue')])
        with self.assertRaisesRegex(inputs.ImagePreparationError, 'multiframe'):
            inputs.prepare_image(self.item(raw.getvalue()), self.root / 'a', 0)

    def test_truncated_jpeg_requires_full_decode_even_if_small(self):
        raw = self.picture((200, 200), format='JPEG')[:-50]
        with Image.open(io.BytesIO(raw)) as header:
            self.assertEqual(header.size, (200, 200))
        with self.assertRaisesRegex(inputs.ImagePreparationError, 'complete_image_decode_failed'):
            inputs.prepare_image(self.item(raw), self.root / 'a', 0)

    def test_hash_range_and_mtime_fail_closed(self):
        item = self.item(self.picture())
        for field, value, expected in [('length', inputs.MAX_SOURCE_BYTES + 1, 'oversized'),
                                       ('offset', 999999, 'outside_tar'), ('sha256', '0' * 64, 'mismatch')]:
            bad = {**item, 'source_ref': {**item['source_ref'], field: value}}
            with self.assertRaisesRegex(inputs.InputIntegrityError, expected):
                inputs.prepare_image(bad, self.root / 'a', 0)
        with self.assertRaisesRegex(inputs.InputIntegrityError, 'mtime_changed'):
            inputs.prepare_image({**item, 'source_mtime_ns': 1}, self.root / 'a', 0)
        # Keep matching ID/ref but change actual TAR bytes.
        with Path(item['source_ref']['path']).open('r+b') as stream:
            stream.seek(item['source_ref']['offset']); stream.write(b'BAD')
        item.pop('source_mtime_ns')
        with self.assertRaisesRegex(inputs.InputIntegrityError, 'sha256_or_length'):
            inputs.prepare_image(item, self.root / 'a', 0)

    def test_source_pixel_limit_and_unsafe_decoder_configuration(self):
        item = self.item(self.picture())
        opened = MagicMock()
        opened.__enter__.return_value = SimpleNamespace(size=(100_000_001, 1))
        with patch.object(inputs.Image, 'open', return_value=opened):
            with self.assertRaisesRegex(inputs.ImagePreparationError, 'pixel_limit'):
                inputs.prepare_image(item, self.root / 'a', 0)
        with patch.object(inputs.ImageFile, 'LOAD_TRUNCATED_IMAGES', True):
            with self.assertRaisesRegex(inputs.ImagePreparationError, 'unsafe_decoder'):
                inputs.prepare_image(item, self.root / 'a', 0)

    def test_attachment_never_overwrites_existing_file(self):
        item = self.item(self.picture())
        evidence = inputs.prepare_image(item, self.root / 'a', 0)
        before = Path(evidence['path']).read_bytes()
        with self.assertRaises(FileExistsError):
            inputs.prepare_image(item, self.root / 'a', 0)
        self.assertEqual(Path(evidence['path']).read_bytes(), before)

    def database(self):
        path = self.root / 'source/index.sqlite'; path.parent.mkdir()
        connection = sqlite3.connect(path)
        connection.execute('CREATE TABLE assets(asset_id TEXT PRIMARY KEY,input_sha TEXT,input_json TEXT,state TEXT,reason TEXT,decision_json TEXT)')
        rows = []
        for n, (state, reason, recommendation) in enumerate([
                ('done', inputs.PENDING_REASON, 'pending'), ('done', inputs.FAILURE_REASON, None),
                ('done', inputs.PENDING_REASON, 'admit'), ('done', inputs.PENDING_REASON, 'reject'),
                ('retryable', inputs.FAILURE_REASON, None), ('done', 'other_reason', 'pending')]):
            item = self.item(self.picture(color=(n * 20, 10, 30)))
            provenance = {'origin': 'mixed', 'evidence': []}
            record = {'asset_id': item['item_id'], 'global_status': 'ai_suspect_pending_api',
                      'original_ref_json': inputs._canonical(item['source_ref']),
                      'provenance_json': inputs._canonical(provenance)}
            previous = {'decision': 'pending', 'reason': reason, 'model_recommendation': recommendation,
                'detector_alarm_cleared': False, 'source_certification': False,
                'input_sha256': inputs._digest(record), 'provenance_sha256': inputs._digest(provenance),
                'reviews': {} if recommendation is None else {'primary': {'assessment': 'uncertain'}}}
            row = (item['item_id'], inputs._digest(record), inputs._canonical(record), state, reason,
                   inputs._canonical(previous))
            rows.append(row)
        connection.executemany('INSERT INTO assets VALUES(?,?,?,?,?,?)', reversed(rows))
        connection.commit(); connection.close()
        return path, rows

    def test_export_exact_selection_canonical_order_and_read_only(self):
        source, rows = self.database()
        before = source.read_bytes()
        report = inputs.export_candidates(source, self.root / 'out/candidates.jsonl')
        lines = [json.loads(s) for s in Path(report['output']).read_text().splitlines()]
        expected = sorted(row[0] for row in rows[:2])
        self.assertEqual([r['item_id'] for r in lines], expected)
        self.assertEqual([r['queue_order'] for r in lines], [0, 1])
        self.assertEqual(report['counts'], {'model_pending': 1, 'technical_failure': 1})
        self.assertEqual(len({r['source_path'] for r in lines}), 2)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(report['sha256'], inputs._file_sha(report['output']))
        again = inputs.export_candidates(source, self.root / 'out/second.jsonl')
        self.assertEqual(report['sha256'], again['sha256'])
        with self.assertRaises(FileExistsError):
            inputs.export_candidates(source, self.root / 'out/candidates.jsonl')

    def test_selected_input_digest_tamper_prevents_manifest_publication(self):
        source, rows = self.database()
        connection = sqlite3.connect(source)
        connection.execute('UPDATE assets SET input_sha=? WHERE asset_id=?', ('0' * 64, rows[0][0]))
        connection.commit(); connection.close()
        output = self.root / 'out/candidates.jsonl'
        with self.assertRaisesRegex(inputs.InputIntegrityError, 'input_digest'):
            inputs.export_candidates(source, output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
