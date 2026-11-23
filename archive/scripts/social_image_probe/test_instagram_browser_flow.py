"""Offline fixtures for tag-first discovery and photo-only collection."""
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from instagram_browser_collect import (
    StopCollection, canonical_post_url, discover, photo_candidates, save_photo,
    read_embed_evidence, safe_browser_error, wait_for_login,
)


def embed(kind="GraphImage", tag="portrait", children=None, shortcode="NewPost"):
    caption = f'<div class="Caption"><a href="/explore/tags/{tag}/">#{tag}</a></div>'
    media = f'<article data-media-type="{kind}"><img class="EmbeddedMediaImage" src="https://test.cdninstagram.com/photo.jpg"></article>'
    if children is not None:
        data = {"gql_data": {"shortcode_media": {"__typename": kind,
            "shortcode": shortcode, "edge_sidecar_to_children": {
                "edges": [{"node": child} for child in children]}}}}
        media += '<script>' + json.dumps({"contextJSON": json.dumps(data)}) + '</script>'
    return caption + media


class FakeLocator:
    def __init__(self, page):
        self.page = page

    def inner_text(self, **kwargs):
        return self.page.body

    def evaluate_all(self, expression):
        batches = self.page.batches.get(self.page.url, [[]])
        return batches[min(self.page.scroll, len(batches) - 1)]


class FakePage:
    def __init__(self, batches, redirect=None):
        self.batches = batches
        self.redirect = redirect
        self.url, self.body = "about:blank", ""
        self.visits, self.scroll, self.closed = [], 0, False
        self.mouse = self

    def goto(self, url, **kwargs):
        self.visits.append(url)
        self.url, self.scroll = self.redirect or url, 0

    def locator(self, selector):
        return FakeLocator(self)

    def wait_for_timeout(self, ms):
        pass

    def wheel(self, x, y):
        self.scroll += 1

    def close(self):
        self.closed = True


class FakeContext:
    def __init__(self, page):
        self.page = page

    def new_page(self):
        return self.page

    def cookies(self, url):
        return []


class BrowserFlowTest(unittest.TestCase):
    def test_post_urls_exclude_reels_and_external_links(self):
        self.assertEqual(canonical_post_url('https://www.instagram.com/author/p/ABC-12/?x=1'),
                         'https://www.instagram.com/p/ABC-12/')
        for value in ['https://www.instagram.com/reel/ABC/',
                      'https://example.com/p/ABC/', 'https://www.instagram.com/author/']:
            self.assertIsNone(canonical_post_url(value))

    def test_exact_caption_tag_and_static_type_are_both_required(self):
        self.assertEqual(len(photo_candidates(embed(), 'NewPost', 'portrait')['images']), 1)
        for markup in [embed(tag='portraitphotography'), embed(kind='GraphVideo'),
                       embed() + '<video></video>']:
            self.assertFalse(photo_candidates(markup, 'NewPost', 'portrait')['images'])

    def test_unparsed_page_is_not_a_verified_tag_mismatch(self):
        shell = '<html><body><div id="root"></div><script type="application/json">{}</script></body></html>'
        self.assertEqual(photo_candidates(shell, 'NewPost', 'portrait')['status'], 'post_data_unrecognized')
        no_caption = '<article data-media-type="GraphImage"><img class="EmbeddedMediaImage" src="https://test.cdninstagram.com/photo.jpg"></article>'
        self.assertEqual(photo_candidates(no_caption, 'NewPost', 'portrait')['status'], 'caption_unavailable')
        empty_caption = no_caption + '<div class="Caption">No hashtags here.</div>'
        self.assertEqual(photo_candidates(empty_caption, 'NewPost', 'portrait')['status'], 'exact_hashtag_missing')

    def test_waiting_for_rendered_post_can_recover_an_initial_shell(self):
        page = FakePage({})
        page.url = 'https://www.instagram.com/p/NewPost/embed/captioned/'
        page.content = lambda: embed()
        result = read_embed_evidence(page, b'<html><body></body></html>', 'NewPost', 'portrait')
        self.assertEqual(result['status'], 'verified_photo')
        self.assertEqual(result['read_source'], 'rendered_dom')

    def test_http_200_login_form_is_not_a_tag_mismatch(self):
        page = FakePage({})
        with self.assertRaises(StopCollection) as caught:
            read_embed_evidence(page, b'<html><input type="password" value="SECRET"></html>', 'NewPost', 'portrait')
        self.assertEqual(caught.exception.status, 'needs_user_login_or_verification')

    def test_carousel_requires_explicit_nonvideo_and_matching_post(self):
        children = [
            {'id': 'photo', 'is_video': False, 'display_url': 'https://test.cdninstagram.com/photo.jpg'},
            {'id': 'video', 'is_video': True, 'display_url': 'https://test.cdninstagram.com/cover.jpg'},
            {'id': 'unknown', 'display_url': 'https://test.cdninstagram.com/unknown.jpg'},
            {'id': 'external', 'is_video': False, 'display_url': 'https://example.com/image.jpg'},
        ]
        markup = embed(kind='GraphSidecar', children=children)
        photos = photo_candidates(markup, 'NewPost', 'portrait')['images']
        self.assertEqual([p['child_id'] for p in photos], ['photo'])
        self.assertFalse(photo_candidates(markup, 'WrongPost', 'portrait')['images'])

    def test_discovery_starts_from_tag_scrolls_and_deduplicates(self):
        tag_url = 'https://www.instagram.com/explore/tags/portrait/'
        search_url = 'https://www.instagram.com/explore/search/keyword/?q=%23portrait'
        one, two, three = ['https://www.instagram.com/p/' + code + '/' for code in ['One', 'Two', 'Three']]
        page = FakePage({tag_url: [[one, 'https://www.instagram.com/reel/Video/'], [one, two]],
                         search_url: [[two, three]]})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            found = list(discover(FakeContext(page), 'portrait', output, 4, 60))
            origins = [json.loads(line) for line in (output / 'discovered_posts.jsonl').read_text().splitlines()]
        self.assertEqual([p for p, _ in found], [one, two, three])
        self.assertEqual(page.visits, [tag_url, search_url])
        self.assertEqual(origins[1]['scroll'], 1)
        self.assertEqual(origins[2]['route'], 'hashtag_search')
        self.assertTrue(page.closed)

    def test_redirected_home_feed_is_not_a_discovery_source(self):
        page = FakePage({'https://www.instagram.com/': [['https://www.instagram.com/p/Unrelated/']]},
                        redirect='https://www.instagram.com/')
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list(discover(FakeContext(page), 'portrait', Path(tmp), 4, 60)), [])

    def test_page_candidates_are_saved_before_first_download(self):
        tag_url = 'https://www.instagram.com/explore/tags/portrait/'
        page = FakePage({tag_url: [[f'https://www.instagram.com/p/Post{i}/' for i in range(24)]]})
        with tempfile.TemporaryDirectory() as tmp:
            discovery = discover(FakeContext(page), 'portrait', Path(tmp), 4, 60)
            next(discovery)
            discovery.close()  # Simulate a failure immediately after the first yield.
            self.assertEqual(len((Path(tmp) / 'discovered_posts.jsonl').read_text().splitlines()), 24)

    def test_error_diagnostics_never_copy_raw_exception_details(self):
        error = RuntimeError('page.goto: net::ERR_SOCKS_CONNECTION_FAILED https://user:SECRET@example.com/?token=SECRET\nCookie: sessionid=SECRET')
        detail = safe_browser_error(error, 'post_embed_navigation', 'Post1')
        self.assertEqual(detail['reason'], 'proxy_connection')
        self.assertEqual(detail['network_code'], 'ERR_SOCKS_CONNECTION_FAILED')
        self.assertNotIn('SECRET', json.dumps(detail))
        self.assertNotIn('example.com', json.dumps(detail))

    def test_discovery_budget_and_login_gate_stop(self):
        tag_url = 'https://www.instagram.com/explore/tags/portrait/'
        page = FakePage({tag_url: [['https://www.instagram.com/p/One/', 'https://www.instagram.com/p/Two/']]})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(list(discover(FakeContext(page), 'portrait', Path(tmp), 4, 1))), 1)
        page = FakePage({}, redirect='https://www.instagram.com/accounts/login/')
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(StopCollection):
            list(discover(FakeContext(page), 'portrait', Path(tmp), 4, 60))

    def test_unsigned_headless_run_stops_without_exporting_cookies(self):
        page = FakePage({})
        with self.assertRaises(StopCollection) as caught:
            wait_for_login(FakeContext(page), page, 600, True)
        self.assertEqual(caught.exception.status, 'needs_user_login')

    def test_saved_photo_has_no_metadata_and_duplicate_is_rejected(self):
        image = Image.new('RGB', (600, 800), 'blue')
        exif = Image.Exif()
        exif[0x010E] = 'fixture private description'
        raw = io.BytesIO()
        image.save(raw, 'JPEG', exif=exif)
        with tempfile.TemporaryDirectory() as tmp:
            output, hashes = Path(tmp), set()
            (output / 'images').mkdir()
            result = save_photo(raw.getvalue(), output, 512, hashes)
            with Image.open(output / result['path']) as clean:
                self.assertFalse(clean.getexif())
            self.assertIsNone(save_photo(raw.getvalue(), output, 512, hashes))


if __name__ == '__main__':
    unittest.main()
