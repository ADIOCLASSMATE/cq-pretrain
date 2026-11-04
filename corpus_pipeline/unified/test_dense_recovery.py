import hashlib
import io
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from PIL import Image

from .dense_recovery import read_member, verify_local_candidates, _zip_worker


class Reader:
    def __init__(self, blob):self.blob=blob
    def read_at(self, offset, length):return self.blob[offset:offset+length]


class DenseRecoveryTests(unittest.TestCase):
    def test_binding_method_is_not_overwritten_by_zip_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / 'output'; output.mkdir()
            local = root / '123.png'; Image.new('RGB', (13, 17), 'red').save(local)
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as writer:
                writer.writestr('123.png', local.read_bytes())
            class RangeArchive(io.BytesIO):
                transferred = 0
                def read_at(self, offset, length):
                    self.seek(offset); return self.read(length)
            blob = archive.getvalue()
            spec = {'rfilename': 'images/DenseFusion-4V-100K/000000.zip', 'size': len(blob),
                    'lfs': {'sha256': hashlib.sha256(blob).hexdigest()}}
            rows = [{'image_path': 'DenseFusion-4V-100K/000000/123.png', 'image_id': '123'}]
            with patch('corpus_pipeline.unified.dense_recovery.HTTPRangeFile', return_value=RangeArchive(blob)):
                result = _zip_worker(spec, rows, {'123': ['123.png']}, root, output, {})
            self.assertEqual(result['counts'], {'bound': 1})
            with sqlite3.connect(result['path']) as db:
                proof = json.loads(db.execute('SELECT result_json FROM bindings').fetchone()[0])['proof']
            self.assertEqual(proof['method'], 'official_png_identical_encoded_sha256')
            self.assertEqual(proof['zip_compression'], zipfile.ZIP_DEFLATED)

    def test_zip_range_png_sha_binding_rejects_same_id_different_bytes(self):
        png=io.BytesIO();Image.new('RGB',(13,17),'red').save(png,format='PNG')
        for compression in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
            archive=io.BytesIO()
            with zipfile.ZipFile(archive,'w',compression=compression) as writer:
                writer.writestr('000000/123.png',png.getvalue())
            with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as reader:
                original,evidence=read_member(Reader(archive.getvalue()),reader.getinfo('000000/123.png'))
            self.assertEqual(original,png.getvalue())
            self.assertEqual(evidence['size'],len(original))
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);(root/'right').mkdir();(root/'wrong').mkdir()
                (root/'right/123.png').write_bytes(original)
                Image.new('RGB',(13,17),'blue').save(root/'wrong/123.png')
                sha,matches=verify_local_candidates(root,['wrong/123.png','right/123.png'],original)
                self.assertEqual(matches,['right/123.png'])
                self.assertEqual(sha,hashlib.sha256(original).hexdigest())

    def test_zip_member_corruption_is_rejected(self):
        png=io.BytesIO();Image.new('RGB',(5,5),'red').save(png,format='PNG')
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as writer:writer.writestr('x.png',png.getvalue())
        with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as reader:info=reader.getinfo('x.png')
        bad=bytearray(archive.getvalue());bad[40]^=1
        with self.assertRaisesRegex(ValueError,'CRC mismatch'):
            read_member(Reader(bytes(bad)),info)


if __name__=='__main__':unittest.main()
