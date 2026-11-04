"""Regression checks for conservative presence evidence; never decode images."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).with_name("check_genund_presence.py")
SPEC = importlib.util.spec_from_file_location("check_genund_presence", SCRIPT)
presence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(presence)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


INVALID_IMAGE = b"Not a decodable image; physical presence is sufficient."
CAPTION = "A red square on a white background."


class PresenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="genund-presence-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def dataset(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def assert_status(self, result, image, annotation):
        self.assertEqual(result["image"], image, result)
        self.assertEqual(result["annotation"], annotation, result)

    def write_tar(self, path, members, mode="w"):
        with tarfile.open(path, mode) as archive:
            for name, payload in members:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    def test_invalid_image_and_caption_are_presence_evidence(self):
        path = self.dataset("loose")
        (path / "0001.jpg").write_bytes(INVALID_IMAGE)
        (path / "0001.txt").write_text(CAPTION)
        self.assert_status(presence.inspect_dataset(path), "YES", "YES")

    def test_readme_does_not_supply_annotation(self):
        path = self.dataset("documentation")
        (path / "0001.jpg").write_bytes(INVALID_IMAGE)
        (path / "README.txt").write_text("Dataset documentation.")
        self.assert_status(presence.inspect_dataset(path), "YES", "NO")

    def test_metadata_image_references_do_not_establish_image_presence(self):
        for name, reference in (
            ("remote", "https://example.invalid/image.jpg"),
            ("missing", "missing_image.jpg"),
        ):
            with self.subTest(name=name):
                path = self.dataset(name)
                (path / "records.json").write_text(
                    json.dumps({"image": reference, "caption": CAPTION})
                )
                self.assert_status(presence.inspect_dataset(path), "UNKNOWN", "YES")

    def test_tar_url_metadata_does_not_supply_annotation(self):
        path = self.dataset("url_metadata")
        self.write_tar(path / "part.tar", [
            ("0001.jpg", INVALID_IMAGE),
            ("0001.json", b'{"url":"https://example.invalid/image.jpg"}'),
        ])
        result = presence.inspect_dataset(path)
        self.assertEqual(result["image"], "YES")
        self.assertNotEqual(result["annotation"], "YES")

    def test_each_tar_records_its_own_json_caption(self):
        path = self.dataset("multiple_shards")
        for index in range(3):
            self.write_tar(path / f"{index:03d}.tar", [
                ("0001.jpg", INVALID_IMAGE),
                ("0001.json", json.dumps({"caption": CAPTION}).encode()),
                ("unneeded.bin", b"Do not scan past conclusive evidence."),
            ])
        result = presence.inspect_dataset(path, sample_shards=3, seed=0)
        self.assert_status(result, "YES", "YES")
        self.assertEqual(len(result["container_samples"]), 3)
        for sample in result["container_samples"]:
            with self.subTest(shard=sample["path"]):
                self.assertTrue(sample["text_member_observed"], sample)
                self.assertEqual(sample["members_checked"], 2, sample)

    def test_compressed_archives_reveal_image_and_caption(self):
        members = [
            ("0001.jpg", INVALID_IMAGE),
            ("0001.json", json.dumps({"caption": CAPTION}).encode()),
        ]
        for index, (extension, mode) in enumerate((
            (".tar.gz", "w:gz"), (".tar.bz2", "w:bz2"),
            (".tar.xz", "w:xz"), (".tgz", "w:gz"), (".zip", None),
        )):
            with self.subTest(extension=extension):
                path = self.dataset(f"compressed_{index}")
                archive_path = path / ("part" + extension)
                if mode:
                    self.write_tar(archive_path, members, mode)
                else:
                    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                        for name, payload in members:
                            archive.writestr(name, payload)
                self.assert_status(presence.inspect_dataset(path), "YES", "YES")

    def test_directory_limits_prevent_absence_claims(self):
        path = self.dataset("truncated")
        (path / "first.bin").write_bytes(b"noise")
        (path / "second.bin").write_bytes(b"noise")
        nested = self.dataset("nested")
        (nested / "child").mkdir()
        (nested / "child" / "0001.jpg").write_bytes(INVALID_IMAGE)
        for target, options in (
            (path, {"max_entries": 1}),
            (path, {"max_dir_entries": 1}),
            (nested, {"max_depth": 0}),
            (nested, {"max_dirs": 1}),
        ):
            with self.subTest(options=options):
                result = presence.inspect_dataset(target, **options)
                self.assert_status(result, "UNKNOWN", "UNKNOWN")
                self.assertEqual(result["verdict"], "UNCONFIRMED")

    def test_member_limit_preserves_observed_image_without_claiming_missing_text(self):
        path = self.dataset("member_limit")
        self.write_tar(path / "part.tar", [
            ("0001.jpg", INVALID_IMAGE), ("0001.txt", CAPTION.encode()),
        ])
        self.assert_status(presence.inspect_dataset(path, max_members=1), "YES", "UNKNOWN")

    def test_compressed_byte_limit_stops_before_large_member_is_skipped(self):
        path = self.dataset("compressed_byte_limit")
        self.write_tar(path / "part.tar.gz", [
            ("0001.jpg", b"x" * 65536),
            ("0001.txt", CAPTION.encode()),
        ], "w:gz")
        result = presence.inspect_dataset(path, max_archive_bytes=4096)
        self.assert_status(result, "YES", "UNKNOWN")
        self.assertTrue(any("byte limit" in error for error in result["errors"]), result)

    @unittest.skipIf(pa is None, "pyarrow is optional")
    def test_parquet_requires_nonempty_bytes_or_existing_image_path(self):
        schema = pa.schema([
            pa.field("image", pa.struct([
                pa.field("bytes", pa.binary()), pa.field("path", pa.string()),
            ])),
            pa.field("caption", pa.string()),
        ])
        for name, image, expected_image in (
            ("bytes", {"bytes": INVALID_IMAGE, "path": None}, "YES"),
            ("null", {"bytes": None, "path": None}, "UNKNOWN"),
            ("empty", {"bytes": b"", "path": None}, "UNKNOWN"),
            ("missing_path", {"bytes": None, "path": "absent.jpg"}, "UNKNOWN"),
            ("remote_path", {"bytes": None, "path": "https://example.invalid/image.jpg"}, "UNKNOWN"),
        ):
            with self.subTest(name=name):
                path = self.dataset(name)
                table = pa.Table.from_pylist([{"image": image, "caption": CAPTION}], schema=schema)
                pq.write_table(table, path / "part.parquet")
                self.assert_status(presence.inspect_dataset(path), expected_image, "YES")


if __name__ == "__main__":
    unittest.main()
