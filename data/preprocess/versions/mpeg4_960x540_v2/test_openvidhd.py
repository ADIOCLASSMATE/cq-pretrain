import csv
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import av
import numpy as np
import torch

from data.preprocess.archive import SplitFile, extract_video
from data.preprocess.manifest import atomic_jsonl, exclusive_run, journal_latest, read_jsonl
from data.preprocess.openvidhd import ANALYSIS, eligible, part_work
from data.preprocess.video import add_output_stream, transcode, verify_output, audit_source_correspondence


def create_video(path, count, rate, width=1920, height=1080, *, origin=0, lossy_bframes=False):
    with av.open(str(path), "w", options={"video_track_timescale": str(rate.numerator)}) as container:
        stream = container.add_stream("libx264" if lossy_bframes else "libx264rgb", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p" if lossy_bframes else "rgb24"
        stream.time_base = Fraction(1, rate.numerator)
        stream.codec_context.time_base = stream.time_base
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "18" if lossy_bframes else "0", "preset": "medium", "threads": "1"}
        if lossy_bframes:
            stream.options["x264-params"] = "bframes=3:b-adapt=0"
        for index in range(count):
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[:] = [index % 256, (index * 3) % 256, (index * 7) % 256]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.time_base = stream.time_base
            frame.pts = (index + origin) * rate.denominator
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_selection_boundaries(self):
        metric = {"width": "1920", "height": "1080", "frame_count": "161"}
        self.assertTrue(eligible(metric, " caption \n"))
        self.assertFalse(eligible(metric, " \n"))
        self.assertFalse(eligible(dict(metric, frame_count="160"), "x"))
        self.assertFalse(eligible(dict(metric, width="1080", height="1920"), "x"))

    def test_split_zip_and_crc(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            memory = io.BytesIO()
            payload = bytes(range(256)) * 4096
            with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("OpenVidHD/original.mp4", payload)
            raw = memory.getvalue()
            paths = [directory / "aa", directory / "ab", directory / "ac"]
            for path, data in zip(paths, (raw[:100], raw[100:12345], raw[12345:])):
                path.write_bytes(data)
            with SplitFile(paths) as joined, zipfile.ZipFile(joined) as archive:
                info = archive.getinfo("OpenVidHD/original.mp4")
                record = {"source": {"member": info.filename, "bytes": info.file_size, "zip_crc32": info.CRC}}
                digest = extract_video(archive, record, directory / "out.mp4")
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            damaged = bytearray(paths[1].read_bytes())
            damaged[500] ^= 1
            paths[1].write_bytes(damaged)
            with SplitFile(paths) as joined, zipfile.ZipFile(joined) as archive:
                with self.assertRaises(zipfile.BadZipFile):
                    extract_video(archive, record, directory / "bad.mp4")

    def test_journal_torn_tail_and_latest_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part1.jsonl"
            rows = [{"sample_id": "original", "status": "failure"},
                    {"sample_id": "original", "status": "success"}]
            atomic_jsonl(path, rows)
            with path.open("ab") as handle:
                handle.write(b'{"sample_id": "torn')
            self.assertEqual(journal_latest(path)["original"]["status"], "success")
            journal_latest(path, repair_tail=True)
            self.assertEqual(read_jsonl(path), rows)
            path.write_bytes(b'{bad}\n')
            with self.assertRaises(json.JSONDecodeError):
                journal_latest(path, repair_tail=True)

    def test_caption_and_filename_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            row = {"sample_id": "---_iRTHryQ_13_0to241.mp4", "caption": '  中文 "caption"\nline 2 ☃\t'}
            atomic_jsonl(path, [row])
            self.assertEqual(read_jsonl(path), [row])

    def test_batch_wrapper_success_and_failure_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "project/data/preprocess"
            scripts.mkdir(parents=True)
            shutil.copyfile(Path(__file__).parents[1] / "batch.sh", scripts / "batch.sh")
            (scripts / "run.sh").write_text(
                '#!/bin/bash\nprintf "%s\\n" "$1" >> "$CALLS"\n'
                'if [[ "$1" == process ]]; then exit "${FAIL_PROCESS:-0}"; fi\n')
            calls = root / "calls"
            env = dict(os.environ, CALLS=str(calls))
            output = root / "output with spaces"
            command = ["bash", str(scripts / "batch.sh"), "pilot", str(output), "4"]
            completed = subprocess.run(command, env=env, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(calls.read_text().splitlines(), ["select", "process", "verify"])
            self.assertEqual((output / "logs/exit_code").read_text(), "0\n")
            calls.unlink()
            completed = subprocess.run(command, env=dict(env, FAIL_PROCESS="2"), capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(calls.read_text().splitlines(), ["select", "process"])
            self.assertEqual((output / "logs/exit_code").read_text(), "2\n")

    def test_committed_resume_orphan_rebuild_and_config_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.mp4"
            create_video(source, 5, Fraction(25))
            archive_path = root / "part.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(source, "original.mp4")
                info = archive.getinfo("original.mp4")
            stat = archive_path.stat()
            config = {"fingerprint": "test-identity", "archive_identities": {"1": [
                {"path": str(archive_path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}]}}
            record = {"sample_id": "original.mp4", "video": "original.mp4", "caption": '原始\n"caption"',
                      "video_path": "videos/part1/original.mp4", "source": {
                          "part": 1, "member": "original.mp4", "bytes": info.file_size,
                          "zip_crc32": info.CRC, "fps_num": 25, "fps_den": 1, "num_frames": 5}}
            journal = root / "_state/part1.jsonl"
            part_work(root, 1, [record], config, 0)
            self.assertEqual(journal_latest(journal)["original.mp4"]["status"], "success")
            with patch("data.preprocess.video.transcode", side_effect=AssertionError("Must skip intact output")) as mocked:
                part_work(root, 1, [record], config, 0)
                mocked.assert_not_called()
            # Emulate crash after MP4 rename but before durable journal commit.
            journal.unlink()
            with patch("data.preprocess.video.transcode", wraps=transcode) as mocked:
                part_work(root, 1, [record], config, 0)
                self.assertEqual(mocked.call_count, 1)
            # Changed bytes must not be silently accepted as a resumed success.
            path = root / record["video_path"]
            data = bytearray(path.read_bytes())
            data[50] ^= 1
            path.write_bytes(data)
            with patch("data.preprocess.video.transcode", side_effect=ValueError("injected failure")):
                part_work(root, 1, [record], config, 0)
            self.assertEqual(journal_latest(journal)["original.mp4"]["status"], "failure")
            part_work(root, 1, [record], config, 0)
            self.assertEqual(journal_latest(journal)["original.mp4"]["status"], "success")
            with self.assertRaisesRegex(ValueError, "configuration"):
                part_work(root, 1, [record], dict(config, fingerprint="changed"), 0)
            with exclusive_run(root):
                with self.assertRaises(BlockingIOError):
                    with exclusive_run(root):
                        pass

    def test_even_odd_numbered_frames_and_nonzero_origin(self):
        for count, rate in ((161, Fraction(2997, 100)), (162, Fraction(30000, 1001))):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                src, dst = Path(directory) / "src.mp4", Path(directory) / "out.mp4"
                create_video(src, count, rate, origin=30)
                selected = {"source": {"num_frames": count, "fps_num": rate.numerator, "fps_den": rate.denominator}}
                record = transcode(src, dst, selected)
                self.assertEqual(record["num_frames"], 81)
                self.assertNotEqual(record["source"]["first_pts"], 0)
                audit = audit_source_correspondence(src, dst, record)
                self.assertTrue(audit["source_correspondence_passed"])
                self.assertTrue(audit["pre_encoder_rgb_sha256_matched"])
                with av.open(str(dst)) as container:
                    self.assertEqual(container.streams.video[0].codec_context.name, "mpeg4")
                    for j, frame in enumerate(container.decode(video=0)):
                        expected = np.array([2*j % 256, 6*j % 256, 14*j % 256], dtype=np.int16)
                        # Numbered, spatially uniform colours identify selected frames;
                        # YUV420 conversion and q=2 are expected to change RGB values.
                        difference = frame.to_ndarray(format="rgb24").astype(np.int16) - expected
                        self.assertLessEqual(int(np.abs(difference).max()), 6)
                        self.assertEqual(frame.format.name, "yuv420p")
                with self.assertRaisesRegex(ValueError, "Decoded RGB hash"):
                    verify_output(dst, record, "incorrect hash")
                bad = dict(record, validation=dict(record["validation"], pre_encoder_rgb_sha256="incorrect"))
                with self.assertRaisesRegex(ValueError, "pre-encoder RGB hash"):
                    audit_source_correspondence(src, dst, bad)
                with self.assertRaisesRegex(ValueError, "decoded count"):
                    transcode(src, Path(directory) / "bad.mp4",
                              {"source": dict(selected["source"], num_frames=count+1)})

    def test_bframe_presentation_order(self):
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory) / "src.mp4", Path(directory) / "dst.mp4"
            create_video(src, 12, Fraction(25), lossy_bframes=True)
            with av.open(str(src)) as container:
                self.assertTrue(any(f.pict_type == av.video.frame.PictureType.B for f in container.decode(video=0)))
            record = transcode(src, dst, {"source": {"num_frames": 12, "fps_num": 25, "fps_den": 1}})
            self.assertTrue(audit_source_correspondence(src, dst, record)["source_correspondence_passed"])

    def test_all_207_eligible_rational_rates(self):
        with (ANALYSIS / "native_fps_stride2_81frames_counts.csv").open() as handle:
            rates = [Fraction(r["source_fps_exact"]) / 2 for r in csv.DictReader(handle)
                     if int(r["source_ge161_count"]) > 0]
        self.assertEqual(len(rates), 207)
        with tempfile.TemporaryDirectory() as directory:
            for rate in rates:
                path = Path(directory) / "rate.mp4"
                with av.open(str(path), "w", options={"video_track_timescale": str(rate.numerator)}) as container:
                    stream = add_output_stream(container, rate, 32, 16)
                    for j in range(3):
                        rgb = np.empty((16, 32, 3), dtype=np.uint8)
                        rgb[:] = [j, 3*j, 7*j]
                        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        frame.time_base = Fraction(1, rate.numerator)
                        frame.pts = j * rate.denominator
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                record = {"fps_num": rate.numerator, "fps_den": rate.denominator,
                          "num_frames": 3, "width": 32, "height": 16}
                checks = verify_output(path, record)
                self.assertEqual(checks["decoded_frames"], 3)
                self.assertTrue(checks["pts_passed"])


if __name__ == "__main__":
    unittest.main()
