import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from corpus_pipeline.context_review.extract import extract_one


class SourceExtractionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'source.tar'

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, entries, target, url=None):
        with tarfile.open(self.path, 'w', format=tarfile.PAX_FORMAT) as archive:
            for name, value in entries:
                raw = value if isinstance(value, bytes) else value.encode()
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
        with tarfile.open(self.path) as archive:
            info = archive.getmember(target)
            raw = archive.extractfile(info).read()
        digest = hashlib.sha256(raw).hexdigest()
        return {'item_id': digest, 'review_number': 1, 'source_name': 'fixture',
                'source_ref': {'member': target, 'offset': info.offset_data, 'length': info.size,
                               'path': str(self.path), 'sha256': digest, 'source_url': url}}

    def test_exact_key_caption_and_no_human_context(self):
        row = self.fixture([
            ('a.jpg', b'previous image'), ('a.txt', 'NOT TARGET'),
            ('b.jpg', b'target image'),
            ('b.json', json.dumps({'caption': 'rough wall texture', 'url': 'https://example.com/b',
                                   'sha256': 'upstream_before_processing_hash'})),
            ('b.txt', 'rough wall texture'),
            ('bb.jpg', b'next image'), ('bb.txt', 'OTHER CAPTION'),
        ], 'b.jpg', 'https://example.com/b')
        human = {'decision': 'exclude_quality', 'review_number': 1}
        candidate, evidence = extract_one(row, human)
        self.assertEqual(candidate['source_context']['caption_texts'], ['rough wall texture'])
        self.assertEqual(candidate['source_context']['context_status'], 'complete')
        self.assertEqual(candidate['task_type'], 'caption_pair')
        self.assertEqual(candidate['human_review'], human)
        self.assertNotIn('exclude_quality', json.dumps(candidate['source_context']))
        self.assertEqual({m['member'] for m in evidence['members']}, {'b.jpg', 'b.json', 'b.txt'})

    def test_interleaved_numeric_order_and_other_images_not_attached(self):
        row = self.fixture([
            ('a.txt', 'previous'),
            ('b.0.jpg', b'not displayed image'), ('b.1.txt', 'before target'),
            ('b.10.txt', 'late text'), ('b.2.jpg', b'displayed image'), ('b.3.txt', 'after target'),
            ('b.filters.json', '{}'), ('c.txt', 'next'),
        ], 'b.2.jpg')
        candidate, _ = extract_one(row, {'decision': 'keep'})
        context = candidate['source_context']
        self.assertEqual(candidate['task_type'], 'interleaved')
        self.assertEqual([t['sequence_index'] for t in context['interleaved_segments']], [1, 3, 10])
        images = context['metadata_summary']['record_images']
        self.assertEqual([(i['member'], i['displayed']) for i in images], [('b.0.jpg', False), ('b.2.jpg', True)])
        self.assertEqual(context['context_status'], 'complete')
        self.assertTrue(any('not displayed' in x for x in context['context_limitations']))

    def test_metadata_url_mismatch_rejected(self):
        row = self.fixture([('b.jpg', b'target'), ('b.json', '{"url":"https://wrong.example/"}'),
                            ('b.txt', 'caption')], 'b.jpg', 'https://expected.example/')
        with self.assertRaisesRegex(ValueError, 'URL mismatch'):
            extract_one(row, {})

    def test_image_bytes_hash_mismatch_rejected(self):
        row = self.fixture([('b.jpg', b'target'), ('b.txt', 'caption')], 'b.jpg')
        row['item_id'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            extract_one(row, {})

    def test_missing_text_is_explicit(self):
        row = self.fixture([('b.jpg', b'target'), ('b.json', '{"width":32}')], 'b.jpg')
        candidate, _ = extract_one(row, {})
        self.assertEqual(candidate['task_type'], 'unknown')
        self.assertEqual(candidate['source_context']['context_status'], 'missing')

    def test_region_qa_preserves_bbox_and_global_caption(self):
        region = {'bbox': [10, 20, 30, 40], 'questions': ['What color?'], 'answers': ['green'],
                  'region_caption': 'green leaf'}
        row = self.fixture([('b.0.json', json.dumps(region)), ('b.jpg', b'target'),
                            ('b.txt', 'a tree')], 'b.jpg')
        candidate, evidence = extract_one(row, {})
        self.assertEqual(candidate['task_type'], 'conversation')
        self.assertEqual(candidate['source_context']['caption_texts'], ['a tree'])
        self.assertIn('bbox', candidate['source_context']['conversation_texts'][0])
        self.assertEqual(evidence['members'][0]['json'], region)

    def test_duplicate_source_member_rejected(self):
        row = self.fixture([('b.jpg', b'target'), ('b.txt', 'first'), ('b.txt', 'second')], 'b.jpg')
        with self.assertRaisesRegex(ValueError, 'Duplicate TAR member'):
            extract_one(row, {})


if __name__ == '__main__':
    unittest.main()
