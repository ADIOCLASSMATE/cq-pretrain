"""Skipping the whole-archive hash must not weaken what the bytes are.

read_ref_bytes returns a byte range out of a pinned TAR. Two different claims
can be checked about it:

  * the member claim -- these bytes hash to the pinned image SHA256;
  * the container claim -- they came out of an archive that hashes to the
    pinned archive SHA256.

Only the first constrains what training consumes, and it costs the size of the
image. The second costs the size of the archive, which for this corpus is four
orders of magnitude larger per image. These tests pin that dropping the second
leaves the first, and every structural check around it, fully enforced.
"""

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from corpus_pipeline.production.portable_refs import read_ref_bytes


def build_archive(root, members):
    """A real uncompressed TAR; returns its path and each member's byte range."""
    path = root / "images.tar"
    with tarfile.open(path, "w") as archive:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, __import__("io").BytesIO(body))
    ranges = {}
    with tarfile.open(path) as archive:
        for info in archive.getmembers():
            ranges[info.name] = (info.offset_data, info.size)
    return path, ranges


class ArchiveVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.body = bytes(range(256)) * 40
        name = hashlib.sha256(self.body).hexdigest()
        self.archive, ranges = build_archive(self.root, {name: self.body, "other": b"x" * 100})
        offset, length = ranges[name]
        self.dataset = "local/test-assets@20261124"
        roots = self.root / "roots.json"
        roots.write_text(json.dumps({self.dataset: str(self.root)}))
        self.roots = str(roots)
        self.ref = dict(
            dataset_id=self.dataset,
            kind="tar",
            path="images.tar",
            member=name,
            sha256=name,
            offset=offset,
            length=length,
            archive_sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            archive_size=self.archive.stat().st_size,
        )

    def read(self, **overrides):
        return read_ref_bytes({**self.ref, **overrides}, roots=self.roots, verify_archive=False)

    def test_the_pinned_image_is_returned_without_the_archive_hash(self):
        self.assertEqual(self.read(), self.body)

    def test_both_modes_return_the_same_bytes(self):
        full = read_ref_bytes(self.ref, roots=self.roots, verify_archive=True)
        self.assertEqual(full, self.read())

    def test_a_wrong_member_hash_is_still_rejected(self):
        """The check that actually constrains the data stays on."""
        with self.assertRaises(ValueError):
            self.read(sha256="0" * 64)

    def test_bytes_that_do_not_hash_to_the_pin_are_rejected(self):
        """Corrupt the archive in place: the member hash must catch it."""
        raw = bytearray(self.archive.read_bytes())
        raw[self.ref["offset"]] ^= 0xFF
        self.archive.write_bytes(bytes(raw))
        with self.assertRaises(ValueError):
            self.read()

    def test_a_changed_archive_size_is_still_rejected(self):
        with self.assertRaises(ValueError):
            self.read(archive_size=self.ref["archive_size"] + 1)

    def test_a_range_beyond_the_archive_is_still_rejected(self):
        with self.assertRaises(ValueError):
            self.read(offset=self.ref["archive_size"] + 10)

    def test_the_archive_hash_is_not_read_when_it_is_skipped(self):
        """A wrong archive_sha256 is irrelevant once the member is verified.

        This is the behavioural difference, stated explicitly: the container
        claim is no longer made, and the member claim alone decides.
        """
        self.assertEqual(self.read(archive_sha256="0" * 64), self.body)
        with self.assertRaises(ValueError):
            read_ref_bytes({**self.ref, "archive_sha256": "0" * 64}, roots=self.roots, verify_archive=True)

    def test_full_verification_remains_the_default(self):
        """Release acceptance must keep attesting to containers."""
        with self.assertRaises(ValueError):
            read_ref_bytes({**self.ref, "archive_sha256": "0" * 64}, roots=self.roots)


if __name__ == "__main__":
    unittest.main()
