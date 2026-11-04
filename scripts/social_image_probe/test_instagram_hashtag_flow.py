"""Offline workflow tests; these are not live Instagram collection results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from collect_instagram_hashtag import collect, discover_posts, normalize_tag


def post(identifier, caption="#portrait", kind="IMAGE"):
    return {"id": identifier, "caption": caption, "media_type": kind,
            "permalink": f"https://www.instagram.com/p/TEST_{identifier}/",
            "media_url": f"https://scontent.test.cdninstagram.com/{identifier}.jpg"}


class FakeGraph:
    def __init__(self):
        self.calls = []

    def get(self, edge, params):
        self.calls.append((edge, dict(params)))
        if edge == "ig_hashtag_search":
            assert params["q"] == "portrait"
            return {"data": [{"id": "888"}]}
        if edge == "888/top_media" and not params.get("after"):
            return {"data": [post("1", "#portraitphotography"), post("2", kind="VIDEO"),
                             post("3"), post("4", "#Portrait")],
                    "paging": {"next": "https://graph.facebook.com/NEXT", "cursors": {"after": "page2"}}}
        if edge == "888/top_media":
            assert params["after"] == "page2"
            return {"data": [post("3"), post("5", "Morning #portrait 🌞")]}
        if edge == "888/recent_media":
            album = post("6", kind="CAROUSEL_ALBUM")
            album["children"] = {"data": [post("video_child", kind="VIDEO"), post("image_child")]}
            return {"data": [post("4"), album, post("7"), post("8")]}
        raise AssertionError(edge)


class FakeImages:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        image = Image.new("RGB", (32, 40), (len(self.calls) * 30, 20, 10))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue(), {"http_status": 200}


class HashtagFlowTest(unittest.TestCase):
    def test_discovers_pages_and_downloads_five_posts_without_input_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "images").mkdir()
            graph, images = FakeGraph(), FakeImages()
            with contextlib.redirect_stdout(io.StringIO()):
                result = collect(graph, "123", normalize_tag("#Portrait"), output, images,
                                 limit=5, max_pages=3, max_posts=100, min_side=16)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["downloaded"], 5)
            self.assertEqual([x[0] for x in graph.calls],
                             ["ig_hashtag_search", "888/top_media", "888/top_media", "888/recent_media"])
            self.assertEqual(graph.calls[0][1]["q"], "portrait")
            self.assertFalse(any("video_child" in url for url in images.calls))
            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["matched_tags"] == ["portrait"] for r in rows))
            self.assertEqual(len({r["sha256"] for r in rows}), 5)
            self.assertEqual(result["outcomes"]["exact_caption_hashtag_missing"], 1)
            self.assertEqual(result["outcomes"]["no_verified_image_media"], 1)
            discovered = [json.loads(line) for line in (output / "discovered_posts.jsonl").read_text().splitlines()]
            self.assertEqual(len({r["post_id"] for r in discovered}), 7)
            self.assertNotIn("8", {r["post_id"] for r in discovered})

    def test_discovery_obeys_post_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = FakeGraph()
            rows = list(discover_posts(graph, "portrait", "123", 3, 2, Path(directory)))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(graph.calls), 2)

    def test_invalid_hashtags_are_rejected(self):
        for value in ("", "#", "portrait landscape", "../portrait"):
            with self.assertRaises(ValueError):
                normalize_tag(value)


if __name__ == '__main__':
    unittest.main()
