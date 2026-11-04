"""Regression checks for exact author-caption hashtag selection."""
import unittest

from collect_instagram_photos import Embed


class CaptionTagsTest(unittest.TestCase):
    def test_similar_tags_do_not_match(self):
        parsed = Embed('<div class="Caption"><a href="/explore/tags/portraitphotography/">'
                       '#portraitphotography</a><a href="/explore/tags/selfportrait/">'
                       '#selfportrait</a></div>')
        self.assertNotIn('portrait', parsed.tags)

    def test_comments_and_other_page_content_do_not_match(self):
        parsed = Embed('<a href="/explore/tags/portrait/">#portrait</a>'
                       '<script>var tag = "#portrait";</script>'
                       '<div class="Caption">An untagged post<br />'
                       '<div class="CaptionComments"><a href="/explore/tags/portrait/">'
                       '#portrait</a></div></div>')
        self.assertEqual(parsed.tags, set())

    def test_caption_tag_survives_line_breaks_and_normalizes_case(self):
        parsed = Embed('<div class="Caption"><br /><span><a '
                       'href="/explore/tags/Portrait/?utm_source=ig_embed">#Portrait</a>'
                       '</span></div><a href="/explore/tags/landscape/">#landscape</a>')
        self.assertEqual(parsed.tags, {'portrait'})
        self.assertIn('portrait', parsed.tag_links)

    def test_external_links_and_plain_keywords_do_not_match(self):
        parsed = Embed('<div class="Caption">Portrait in the morning.'
                       '<a href="https://example.com/explore/tags/portrait/">portrait</a>'
                       '</div>')
        self.assertEqual(parsed.tags, set())


if __name__ == '__main__':
    unittest.main()
