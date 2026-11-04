"""Asset integrity gates: interrupted downloads cannot publish corrupt weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import download_modelscope as download


class Response:
    def __init__(self, payload=None, data=b'', content_range=None):
        self.payload = payload
        self.data = data
        self.status_code = 206
        self.headers = {'content-range': content_range}

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_bytes(self, **kwargs):
        yield self.data


class FakeClient:
    data = b'actual model bytes'
    wrong_range = False

    def __init__(self, **kwargs):
        assert kwargs['trust_env'] is False

    def get(self, *args, **kwargs):
        return Response({'Data': {'Files': [{
            'Path': 'model.safetensors', 'Size': len(self.data),
            'Sha256': hashlib.sha256(self.data).hexdigest(), 'Revision': 'locked-mirror',
        }]}})

    def stream(self, *args, **kwargs):
        begin = 1 if self.wrong_range else 0
        return Response(data=self.data, content_range=f'bytes {begin}-{len(self.data)-1}/{len(self.data)}')

    def close(self):
        pass


class AssetIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'model'
        self.output.mkdir()
        self.digest = hashlib.sha256(FakeClient.data).hexdigest()
        (self.root / 'download_manifest.json').write_text(json.dumps({
            'revision': 'locked-hf', 'files': [{'name': 'model.safetensors',
                'size': len(FakeClient.data), 'expected_lfs_sha256': self.digest}]}))
        self.patches = [patch.object(download, 'ROOT', self.root),
                        patch.object(download.httpx, 'Client', FakeClient),
                        patch.object(download.time, 'sleep'),
                        patch('sys.argv', ['download', '--output', str(self.output), '--chunk-mib', '1', '--workers', '2'])]
        for p in self.patches:
            p.start()
        FakeClient.wrong_range = False

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_interrupted_sparse_file_resumes_and_publishes_only_verified_final(self):
        part = self.output / 'model.safetensors.part'
        part.write_bytes(b'\0' * len(FakeClient.data))
        (self.root / 'model.safetensors.chunks.json').write_text(json.dumps({
            'identity': {'name': 'model.safetensors', 'size': len(FakeClient.data),
                         'sha256': self.digest, 'chunk_size': 1024 * 1024}, 'completed_chunks': []}))
        download.main()
        self.assertEqual((self.output / 'model.safetensors').read_bytes(), FakeClient.data)
        self.assertFalse(part.exists())
        receipt = json.loads((self.root / 'WEIGHTS_READY.json').read_text())
        self.assertTrue(receipt['all_required_files_verified'])
        self.assertEqual(receipt['files'][0]['sha256'], self.digest)

    def test_tampered_same_size_final_is_rejected(self):
        (self.output / 'model.safetensors').write_bytes(b'X' * len(FakeClient.data))
        with self.assertRaisesRegex(ValueError, 'existing_final_file_invalid'):
            download.main()
        self.assertFalse((self.root / 'WEIGHTS_READY.json').exists())

    def test_mismatched_range_never_marks_complete_or_publishes(self):
        FakeClient.wrong_range = True
        with self.assertRaisesRegex(RuntimeError, 'content_range_mismatch'):
            download.main()
        self.assertFalse((self.root / 'WEIGHTS_READY.json').exists())
        self.assertFalse((self.output / 'model.safetensors').exists())


if __name__ == '__main__':
    unittest.main()
