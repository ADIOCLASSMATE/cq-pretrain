"""An upstream caption may only be attached to an image it provably describes.

A published row fixes a visual slot. An upstream record that supplied one of
the caption alternatives may store its own copy of that picture; the copy is
usable only when a pixel identity or content hash ties it back to the slot.
The 32-GPU L40S run died at the same step on every restart because one drawn
alternative named an owner whose copy had neither (data/upstream.py), and the
occurrence check correctly refused the pairing by raising.

These tests pin the repair: such an alternative is passed over deterministically
instead of killing the job, and -- the property that makes the change safe --
any draw the check already accepted is left exactly where it was.
"""

import json
import unittest

from data import captions
from data.upstream import CONTRACT, candidate_bindable, occurrences_verifiable, owner_for, tie

MATCHING = "TextAtlas5M/CoverBook"
MISMATCHED = "openuni"


def image(identity, sha):
    return {"exact_identity": identity, "sha256": sha, "_dataset_roots": {"root": "/dev/null"}}


def origin(source, record):
    return {"source": source, "record_id": record}


def key_of(source, record):
    return json.dumps([source, record], ensure_ascii=False, separators=(",", ":"))


def published_row(owners, contexts, candidates):
    """A one-image, one-caption published row carrying an upstream binding."""
    return {
        "sample_id": "group:fixture",
        "source": MATCHING,
        "format": "caption",
        "segments": [{"kind": "image", "image_index": 0}, {"kind": "text", "text": ""}],
        "images": [image("pixel:aaa:500x490", "sha-published")],
        "caption_segment_index": 1,
        "caption_candidates_json": json.dumps(candidates),
        "_upstream_sampling": CONTRACT,
        "_upstream_binding": {"owners": owners, "contexts": contexts},
    }


def context(images):
    return {"segments": [{"kind": "image", "image_index": 0}, {"kind": "text", "text": ""}], "images": images}


GOOD = key_of(MATCHING, "row:1")
BAD = key_of(MISMATCHED, "row:2")
# The published copy, and an upstream copy that is a different file with no
# pixel identity -- exactly the shape that stopped the production run.
GOOD_CONTEXT = context([image("pixel:aaa:500x490", "sha-published")])
BAD_CONTEXT = context([image(None, "sha-elsewhere")])


class VerifiabilityTests(unittest.TestCase):
    def test_matching_pixel_identity_is_verifiable(self):
        row = published_row({GOOD: 0}, [GOOD_CONTEXT], [])
        self.assertTrue(occurrences_verifiable(row, GOOD_CONTEXT))

    def test_different_file_without_pixel_identity_is_not(self):
        row = published_row({BAD: 0}, [BAD_CONTEXT], [])
        self.assertFalse(occurrences_verifiable(row, BAD_CONTEXT))

    def test_a_differing_occurrence_count_is_not_verifiable(self):
        row = published_row({GOOD: 0}, [GOOD_CONTEXT], [])
        two = context([image("pixel:aaa:500x490", "sha-published")] * 2)
        two["segments"].insert(1, {"kind": "image", "image_index": 1})
        self.assertFalse(occurrences_verifiable(row, two))


class OwnerDrawTests(unittest.TestCase):
    def test_a_verifiable_draw_is_left_untouched(self):
        """The property the repair rests on: accepted draws do not move."""
        candidate = {"text": "a", "provenance": [origin(MATCHING, "row:1"), origin("TextGround4M", "row:3")]}
        row = published_row({GOOD: 0, key_of("TextGround4M", "row:3"): 1}, [GOOD_CONTEXT, GOOD_CONTEXT], [candidate])
        drawn = tie(sorted([GOOD, key_of("TextGround4M", "row:3")]), row["sample_id"], 42, 0, "caption-origin")
        self.assertEqual(owner_for(row, candidate, 42, 0)[0], drawn)

    def test_an_unverifiable_owner_is_replaced_by_a_verifiable_one(self):
        candidate = {"text": "a", "provenance": [origin(MATCHING, "row:1"), origin(MISMATCHED, "row:2")]}
        row = published_row({GOOD: 0, BAD: 1}, [GOOD_CONTEXT, BAD_CONTEXT], [candidate])
        self.assertEqual(owner_for(row, candidate, 42, 0)[0], GOOD)

    def test_an_owner_absent_from_the_binding_is_not_usable(self):
        candidate = {"text": "a", "provenance": [origin(MISMATCHED, "row:2")]}
        row = published_row({GOOD: 0}, [GOOD_CONTEXT], [candidate])
        self.assertIsNone(owner_for(row, candidate, 42, 0)[0])

    def test_no_verifiable_owner_reports_none(self):
        candidate = {"text": "a", "provenance": [origin(MISMATCHED, "row:2")]}
        row = published_row({BAD: 0}, [BAD_CONTEXT], [candidate])
        self.assertIsNone(owner_for(row, candidate, 42, 0)[0])
        self.assertFalse(candidate_bindable(row, candidate, 42, 0))


class SelectionTests(unittest.TestCase):
    def fixture(self):
        """Two alternatives: one bindable, one whose only owner cannot be verified."""
        good = {"text": "published picture", "provenance": [origin(MATCHING, "row:1")]}
        bad = {"text": "borrowed caption", "provenance": [origin(MISMATCHED, "row:2")]}
        return published_row({GOOD: 0, BAD: 1}, [GOOD_CONTEXT, BAD_CONTEXT], [good, bad])

    def test_selection_never_returns_an_unverifiable_pairing(self):
        for seed in range(12):
            for epoch in range(2):
                row = captions.select(dict(self.fixture()), seed, epoch)
                self.assertEqual(row["segments"][1]["text"], "published picture")
                self.assertEqual(row["selected_upstream"]["origin"], origin(MATCHING, "row:1"))

    def test_selection_is_reproducible(self):
        first = captions.select(dict(self.fixture()), 42, 0)
        second = captions.select(dict(self.fixture()), 42, 0)
        self.assertEqual(first["caption_choice"], second["caption_choice"])
        self.assertEqual(first["segments"], second["segments"])

    def test_a_row_whose_alternatives_all_bind_is_unaffected(self):
        both = [
            {"text": "one", "provenance": [origin(MATCHING, "row:1")]},
            {"text": "two", "provenance": [origin("TextGround4M", "row:3")]},
        ]
        row = published_row({GOOD: 0, key_of("TextGround4M", "row:3"): 1}, [GOOD_CONTEXT, GOOD_CONTEXT], both)
        # The ordinary draw decides, and each alternative keeps its own owner.
        prepared = captions.select(dict(row), 42, 0)
        chosen = prepared["caption_choice"]
        self.assertIn(chosen, (0, 1))
        expected = both[chosen]["provenance"][0]
        self.assertEqual(prepared["selected_upstream"]["origin"], expected)
        self.assertEqual(prepared["segments"][1]["text"], both[chosen]["text"])

    def test_nothing_bindable_raises_with_the_sample_id(self):
        only_bad = [{"text": "borrowed", "provenance": [origin(MISMATCHED, "row:2")]}]
        row = published_row({BAD: 0}, [BAD_CONTEXT], only_bad)
        with self.assertRaises(ValueError) as caught:
            captions.select(dict(row), 42, 0)
        self.assertIn("group:fixture", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
