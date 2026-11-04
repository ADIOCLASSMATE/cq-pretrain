"""Chinese hashtag matching and bounded batch behavior; no live requests."""
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import quote
import zipfile

from PIL import Image
import instagram_chinese_batch as batch
from instagram_browser_collect import discover, normalize_tag, photo_candidates
from test_instagram_browser_flow import FakeContext, FakePage, embed


class ChineseBatchTest(unittest.TestCase):
    def test_chinese_discovery_and_exact_encoded_caption_tag(self):
        tag = normalize_tag('#人像摄影')
        url = 'https://www.instagram.com/explore/tags/' + quote(tag) + '/'
        page = FakePage({url: [['https://www.instagram.com/p/ChinesePost/']]})
        with tempfile.TemporaryDirectory() as tmp:
            found = list(discover(FakeContext(page), tag, Path(tmp), 0, 1))
        self.assertEqual(found[0][1]['requested_tag'], '人像摄影')
        self.assertEqual(page.visits, [url])
        encoded_caption = embed(tag=quote(tag))
        self.assertEqual(photo_candidates(encoded_caption, 'ChinesePost', tag)['status'], 'verified_photo')
        self.assertEqual(photo_candidates(embed(tag='人像摄影作品'), 'ChinesePost', tag)['status'], 'exact_hashtag_missing')

    def exercise(self, statuses):
        invoked = []
        raw = io.BytesIO()
        Image.new('RGB', (600, 800), 'blue').save(raw, 'JPEG')
        jpeg = raw.getvalue()
        digest = hashlib.sha256(jpeg).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root/'profile'
            profile.mkdir()
            (profile/'secret.txt').write_text('FIXTURE_LOGIN_SECRET')
            args = SimpleNamespace(tags=['写真','人像摄影','生活记录'], limit=1, output=root/'output',
                                   profile=profile, headless=True, browser_executable=None, proxy_server=None)

            def collect(child):
                invoked.append(child.tag)
                self.assertEqual(child.profile, profile)
                child.output.mkdir(parents=True)
                status = statuses[len(invoked)-1]
                rows = []
                if status == 'complete':
                    (child.output/'images').mkdir()
                    (child.output/'images/photo.jpg').write_bytes(jpeg)
                    rows.append({'path':'images/photo.jpg','sha256':digest,'width':600,'height':800,'requested_tag':child.tag})
                (child.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                (child.output/'summary.json').write_text(json.dumps({'status':status,'downloaded':len(rows),
                    'attempted_posts':1,'discovered_posts':1}))
                return 0 if status == 'complete' else 2

            with patch.object(batch, 'run_collector', collect), contextlib.redirect_stdout(io.StringIO()):
                code = batch.run_batch(args)
            summary = json.loads((args.output/'summary.json').read_text())
            with zipfile.ZipFile(args.output/'results.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any('profile' in name or 'secret' in name for name in archive.namelist()))
                for name in archive.namelist():
                    self.assertNotIn(b'FIXTURE_LOGIN_SECRET', archive.read(name))
            return code, summary, invoked

    def test_three_tags_share_login_and_report_cross_tag_duplicates(self):
        code, summary, invoked = self.exercise(['complete'] * 3)
        self.assertEqual(code, 0)
        self.assertEqual(invoked, ['写真','人像摄影','生活记录'])
        self.assertEqual(summary['downloaded'], 3)
        self.assertEqual(summary['unique_images'], 1)
        self.assertEqual(summary['repeated_images_across_tags'], 2)

    def test_platform_limit_prevents_starting_the_next_tag(self):
        code, summary, invoked = self.exercise(['complete', 'http_access_limited'])
        self.assertEqual(code, 2)
        self.assertEqual(invoked, ['写真','人像摄影'])
        self.assertEqual(summary['groups'][2]['status'], 'not_run')
        self.assertEqual(summary['stop_reason'], 'http_access_limited')

    def test_profile_cannot_be_archived_inside_batch_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(tags=['写真'], limit=1, output=Path(tmp)/'output', profile=Path(tmp)/'output/profile')
            with self.assertRaises(ValueError):
                batch.run_batch(args)


if __name__ == '__main__':
    unittest.main()
