"""Real Chromium with intercepted synthetic pages, never live account data."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit
import zipfile

from PIL import Image
import instagram_browser_collect as collector
from test_instagram_browser_flow import embed


@unittest.skipUnless(os.environ.get('IG_TEST_BROWSER'), 'Set IG_TEST_BROWSER for real Chromium fixture tests')
class BrowserIntegrationTest(unittest.TestCase):
    def exercise(self, mode, tag='portrait'):
        from playwright.sync_api import BrowserType, Browser, Page
        original_launch = BrowserType.launch_persistent_context
        original_new_context = Browser.new_context
        original_goto = Page.goto
        image_requests = []
        codes = ['Broken', 'WrongTag', 'Video', 'Photo1', 'Photo2', 'Photo3', 'Photo4', 'Photo5']

        def route_fixture(route):
            parsed = urlsplit(route.request.url)
            if parsed.hostname == 'test.cdninstagram.com':
                if not route.request.is_navigation_request():
                    route.abort()
                    return
                image_requests.append(route.request.all_headers())
                color = int(parsed.path.split('Photo')[1].split('.')[0]) * 35
                image = Image.new('RGB', (600, 800), (color, 30, 50))
                raw = io.BytesIO()
                image.save(raw, 'JPEG')
                route.fulfill(status=200, content_type='image/jpeg', body=raw.getvalue())
            elif parsed.hostname == 'www.instagram.com':
                if parsed.path == '/accounts/login/':
                    route.fulfill(status=200, content_type='text/html', body='<html><body>Fixture login</body></html>')
                elif parsed.path.startswith('/explore/'):
                    if parsed.path.startswith('/explore/tags/'):
                        self.assertEqual(unquote(parsed.path.split('/')[3]), tag)
                    links = ''.join(f'<a href="https://www.instagram.com/p/{code}/">post</a>' for code in codes)
                    route.fulfill(status=200, content_type='text/html', body=f'<html><body>{links}</body></html>')
                elif parsed.path.endswith('/embed/captioned/'):
                    code = parsed.path.split('/')[2]
                    if mode == 'unreadable':
                        route.fulfill(status=200, content_type='text/html', body='<html><body><div id="root"></div></body></html>')
                    elif mode == 'limited':
                        route.fulfill(status=429, content_type='text/plain', body='Fixture limit')
                    elif code == 'Broken' or mode == 'all_broken':
                        route.abort('connectionreset')
                    else:
                        markup = embed(kind='GraphVideo' if code == 'Video' else 'GraphImage',
                                       tag=tag+'other' if code == 'WrongTag' else tag)
                        markup = markup.replace('/photo.jpg', '/' + code + '.jpg')
                        route.fulfill(status=200, content_type='text/html', body=markup)
                else:
                    route.abort()
            else:
                route.abort()

        def launch(browser_type, *args, **kwargs):
            context = original_launch(browser_type, *args, **kwargs)
            context.route('**/*', route_fixture)
            # Fake cookie only, generated within the test and never used with Instagram.
            context.add_cookies([{'name': 'sessionid', 'value': 'FIXTURE_ONLY_SECRET', 'url': 'https://www.instagram.com/'}])
            return context

        def new_context(browser, *args, **kwargs):
            context = original_new_context(browser, *args, **kwargs)
            context.route('**/*', route_fixture)
            return context

        def goto(page, *args, **kwargs):
            kwargs['timeout'] = 5000  # Synthetic pages have no external network work.
            return original_goto(page, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix='ig-chromium-fixture-') as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            stack.enter_context(patch.object(BrowserType, 'launch_persistent_context', launch))
            stack.enter_context(patch.object(Browser, 'new_context', new_context))
            stack.enter_context(patch.object(Page, 'wait_for_timeout', lambda *args: None))
            stack.enter_context(patch.object(Page, 'goto', goto))
            stack.enter_context(patch.object(collector.time, 'sleep', lambda *args: None))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            args = SimpleNamespace(tag=tag, output=root/'output', profile=root/'profile',
                headless=True, browser_executable=Path(os.environ['IG_TEST_BROWSER']), proxy_server=None,
                login_timeout=0, max_scrolls=1, max_posts=60, min_side=512, limit=5)
            code = collector.run(args)
            output = args.output
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['collector_version'], '2.1')
            self.assertEqual(len(summary['collector_sha256']), 64)
            outcomes = [json.loads(line) for line in (output/'outcomes.jsonl').read_text().splitlines()]
            errors = [json.loads(line) for line in (output/'errors.jsonl').read_text().splitlines()]
            with zipfile.ZipFile(output/'results.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any('profile' in name for name in archive.namelist()))
                for name in archive.namelist():
                    self.assertNotIn(b'FIXTURE_ONLY_SECRET', archive.read(name))
            for headers in image_requests:
                self.assertNotIn('cookie', headers)
                self.assertNotIn('authorization', headers)
            self.assertEqual(summary['discovered_posts'], 8)
            self.assertEqual(summary['attempted_posts'], len(outcomes))
            return code, summary, outcomes, errors, len(image_requests)

    def test_first_failure_does_not_prevent_five_photos(self):
        code, summary, outcomes, errors, images = self.exercise('mixed')
        self.assertEqual(code, 0, (summary, outcomes, errors))
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['downloaded'], 5)
        self.assertEqual(images, 5)
        self.assertEqual(outcomes[0]['status'], 'post_browser_error')
        self.assertEqual(outcomes[1]['status'], 'exact_hashtag_missing')
        self.assertEqual(outcomes[2]['status'], 'no_verified_photo')
        self.assertEqual(errors[0]['stage'], 'post_embed_navigation')

    def test_chinese_tag_collects_five_matching_images(self):
        code, summary, outcomes, errors, images = self.exercise('mixed', tag='人像摄影')
        self.assertEqual(code, 0, summary)
        self.assertEqual(summary['requested_tag'], '人像摄影')
        self.assertEqual(summary['downloaded'], 5)
        self.assertEqual(images, 5)
        self.assertEqual(outcomes[1]['status'], 'exact_hashtag_missing')

    def test_three_consecutive_network_failures_stop_with_diagnostics(self):
        code, summary, outcomes, errors, images = self.exercise('all_broken')
        self.assertEqual(code, 2)
        self.assertEqual(summary['status'], 'repeated_browser_errors')
        self.assertEqual(len(errors), 3)
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(images, 0)

    def test_platform_limit_stops_immediately(self):
        code, summary, outcomes, errors, images = self.exercise('limited')
        self.assertEqual(code, 2)
        self.assertEqual(summary['status'], 'http_access_limited')
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(images, 0)

    def test_unreadable_pages_stop_before_exhausting_all_candidates(self):
        code, summary, outcomes, errors, images = self.exercise('unreadable')
        self.assertEqual(code, 2)
        self.assertEqual(summary['status'], 'post_parser_unavailable')
        self.assertEqual(summary['attempted_posts'], 3)
        self.assertEqual([o['status'] for o in outcomes], ['post_data_unrecognized'] * 3)
        self.assertEqual(errors[0]['page_structure']['caption_containers'], 0)
        self.assertEqual(images, 0)


if __name__ == '__main__':
    unittest.main()
