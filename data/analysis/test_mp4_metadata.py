import io
import struct
import tempfile
import unittest
from pathlib import Path

from mp4_metadata import SplitFile, boxes, presentation_timing, read_moov, rle_table


class TimingTests(unittest.TestCase):
    def test_last_sample_duration_is_not_an_interframe_interval(self):
        result = presentation_timing([(2, 10), (1, 30)])
        self.assertEqual(result["timing_class"], "constant")
        self.assertEqual(result["pts_span_ticks"], 20)
        self.assertEqual(result["sample_duration_ticks"], 50)

    def test_b_frame_composition_offsets(self):
        result = presentation_timing([(4, 10)], [(1, 20), (1, 40), (2, 10)])
        self.assertEqual(result["timing_class"], "constant")
        self.assertEqual(result["pts_span_ticks"], 30)

    def test_variable_timing(self):
        self.assertEqual(presentation_timing([(1, 10), (1, 20), (1, 30)])["timing_class"], "variable")

    def test_quantized_uniform_timing(self):
        self.assertEqual(presentation_timing([(1, 41), (2, 42), (1, 41)])["timing_class"], "quantized_uniform")

    def test_duplicate_pts_rejected(self):
        with self.assertRaises(ValueError):
            presentation_timing([(2, 10)], [(1, 10), (1, 0)])

    def test_version_zero_ctts_signed_offsets_match_ffmpeg(self):
        payload = struct.pack(">II", 0, 3) + struct.pack(">IIIIII", 1, 0, 1, 10, 1, 2**32 - 10)
        offsets = rle_table(payload, signed=True)
        self.assertEqual(offsets, [(1, 0), (1, 10), (1, -10)])
        self.assertEqual(presentation_timing([(3, 10)], offsets)["timing_class"], "constant")

    def test_ctts_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            presentation_timing([(3, 10)], [(2, 0)])

    def test_split_file_boundary_and_seek(self):
        with tempfile.TemporaryDirectory() as directory:
            parts = [Path(directory) / name for name in ("aa", "ab")]
            parts[0].write_bytes(b"123")
            parts[1].write_bytes(b"4567")
            with SplitFile(parts) as stream:
                self.assertEqual(stream.read(5), b"12345")
                stream.seek(-3, 2)
                self.assertEqual(stream.read(), b"567")

    def test_moov_after_media_data(self):
        blob = struct.pack(">I4s", 12, b"mdat") + b"1234" + struct.pack(">I4s", 11, b"moov") + b"abc"
        self.assertEqual(read_moov(io.BytesIO(blob), len(blob)), b"abc")
        with self.assertRaises(ValueError):
            list(boxes(struct.pack(">I4s", 100, b"mdat")))


if __name__ == "__main__":
    unittest.main()
