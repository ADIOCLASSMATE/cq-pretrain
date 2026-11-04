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
from data.preprocess.manifest import (atomic_jsonl, exclusive_run, journal_latest,
                                     read_jsonl, require_readable, sha256)
from data.preprocess.openvidhd import ANALYSIS, VARIANTS, eligible, part_work, reconcile
from data.preprocess.video import (add_output_stream, transcode, verify_output,
                                   audit_source_correspondence, spatial_plan, resize_rgb)


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


def output_paths(root, record):
    paths = {v: root / v / record["video_path"] for v in VARIANTS}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


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
            root = Path(directory)
            memory = io.BytesIO()
            payload = bytes(range(256)) * 4096
            with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("OpenVidHD/original.mp4", payload)
            raw = memory.getvalue()
            paths = [root / "aa", root / "ab", root / "ac"]
            for path, data in zip(paths, (raw[:100], raw[100:12345], raw[12345:])):
                path.write_bytes(data)
            with SplitFile(paths) as joined, zipfile.ZipFile(joined) as archive:
                info = archive.getinfo("OpenVidHD/original.mp4")
                record = {"source": {"member": info.filename, "bytes": info.file_size}}
                digest = extract_video(archive, record, root / "out.mp4", info.CRC)
                with self.assertRaisesRegex(ValueError, "identity"):
                    extract_video(archive, record, root / "wrong.mp4", info.CRC + 1)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            damaged = bytearray(paths[1].read_bytes())
            damaged[500] ^= 1
            paths[1].write_bytes(damaged)
            with SplitFile(paths) as joined, zipfile.ZipFile(joined) as archive:
                with self.assertRaises(zipfile.BadZipFile):
                    extract_video(archive, record, root / "bad.mp4", info.CRC)

    def test_journal_torn_tail_and_variant_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part1.jsonl"
            rows = [{"sample_id": "original", "variant": "960x544", "status": "success"},
                    {"sample_id": "original", "variant": "832x480", "status": "failure"},
                    {"sample_id": "original", "variant": "832x480", "status": "success"}]
            atomic_jsonl(path, rows)
            with path.open("ab") as handle:
                handle.write(b'{"sample_id": "torn')
            latest = journal_latest(path, repair_tail=True, per_variant=True)
            self.assertEqual(set(latest), {("original", v) for v in VARIANTS})
            self.assertEqual(latest["original", "832x480"]["status"], "success")
            self.assertEqual(read_jsonl(path), rows)
            path.write_bytes(b'{bad}\n')
            with self.assertRaises(json.JSONDecodeError):
                journal_latest(path, repair_tail=True, per_variant=True)

    def test_readability_caption_and_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            row = {"sample_id": "---_iRTHryQ_13_0to241.mp4", "caption": '  中文 "caption"\nline 2 ☃\t'}
            atomic_jsonl(path, [row])
            require_readable(row)
            self.assertEqual(read_jsonl(path), [row])
            for key in ("source_sha256", "output_hash", "config_fingerprint", "zip_crc32"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Integrity field"):
                    require_readable(dict(row, source={key: "abc"}))

    def test_spatial_plans_and_uniform_image(self):
        expected = {"960x544": (967, 544, 3, 0), "832x480": (853, 480, 10, 0)}
        rgb = np.full((1080, 1920, 3), [50, 100, 200], dtype=np.uint8)
        for variant, size in VARIANTS.items():
            plan = spatial_plan(**size)
            self.assertEqual(tuple(plan[k] for k in ("resize_width", "resize_height", "crop_x", "crop_y")),
                             expected[variant])
            actual = resize_rgb(rgb, **size)
            self.assertEqual(actual.shape, (size["height"], size["width"], 3))
            self.assertTrue(np.all(actual == [50, 100, 200]))
            self.assertEqual(size["width"] % 32, 0)
            self.assertEqual(size["height"] % 32, 0)

    def test_batch_wrapper_success_and_failure_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "project/data/preprocess"
            scripts.mkdir(parents=True)
            shutil.copyfile(Path(__file__).parents[1] / "batch.sh", scripts / "batch.sh")
            (scripts / "run.sh").write_text(
                '#!/bin/bash\nprintf "%s\\n" "$1" >> "$CALLS"\n'
                'if [[ "$1" == process ]]; then exit "$' '{FAIL_PROCESS:-0}"; fi\n')
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

    def test_resume_partial_variant_failure_orphan_and_clean_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.mp4"
            create_video(source, 5, Fraction(25))
            archive_path = root / "part.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(source, "original.mp4")
                info = archive.getinfo("original.mp4")
            stat = archive_path.stat()
            config = {"run_id": "test-v3", "archive_identities": {"1": [
                {"path": str(archive_path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}]}}
            row = {"sample_id": "original.mp4", "video": "original.mp4", "caption": '原始\n"caption"',
                   "video_path": "videos/part1/original.mp4", "source": {
                       "part": 1, "member": "original.mp4", "bytes": info.file_size,
                       "fps_num": 25, "fps_den": 1, "num_frames": 5}}
            source_checks = {"original.mp4": info.CRC}
            atomic_jsonl(root / "metadata/selected.jsonl", [row])
            journal = root / "_state/part1.jsonl"
            args = (root, 1, [row], config, source_checks, 0)
            part_work(*args)
            events = journal_latest(journal, per_variant=True)
            self.assertTrue(all(e["status"] == "success" for e in events.values()))
            keep = root / "960x544" / row["video_path"]
            saved = (sha256(keep), keep.stat().st_mtime_ns)
            with patch("data.preprocess.video.transcode", side_effect=AssertionError("Must skip")) as mocked:
                part_work(*args)
                mocked.assert_not_called()
            bad = root / "832x480" / row["video_path"]
            data = bytearray(bad.read_bytes())
            data[50] ^= 1
            bad.write_bytes(data)
            with patch("data.preprocess.video.transcode", side_effect=ValueError("injected variant failure")):
                part_work(*args)
            summary = reconcile(root, config)
            self.assertEqual(summary["variants"]["960x544"]["success"], 1)
            self.assertEqual(summary["variants"]["832x480"]["failure"], 1)
            self.assertEqual(read_jsonl(root / "960x544/samples.jsonl"), [])
            self.assertEqual(read_jsonl(root / "pairs.jsonl"), [])
            self.assertEqual(saved, (sha256(keep), keep.stat().st_mtime_ns))
            with patch("data.preprocess.video.transcode", wraps=transcode) as mocked:
                part_work(*args)
                self.assertEqual(set(mocked.call_args.args[1]), {"832x480"})
            events = journal_latest(journal, per_variant=True)
            atomic_jsonl(journal, [events["original.mp4", "960x544"]])
            with patch("data.preprocess.video.transcode", wraps=transcode) as mocked:
                part_work(*args)
                self.assertEqual(set(mocked.call_args.args[1]), {"832x480"})
            summary = reconcile(root, config)
            self.assertEqual(summary["success"], 1)
            self.assertEqual(summary["committed_video_files"], 2)
            self.assertEqual(saved, (sha256(keep), keep.stat().st_mtime_ns))
            for v in VARIANTS:
                records = read_jsonl(root / v / "samples.jsonl")
                require_readable(records)
                self.assertEqual(records[0]["sample_id"], row["sample_id"])
                self.assertEqual(records[0]["caption"], row["caption"])
                self.assertTrue((root / v / records[0]["video_path"]).is_file())
            require_readable(read_jsonl(journal))
            require_readable(read_jsonl(root / "pairs.jsonl"))
            self.assertEqual(len(read_jsonl(root / "metadata/integrity.jsonl")), 2)
            with self.assertRaisesRegex(ValueError, "configuration"):
                part_work(root, 1, [row], dict(config, run_id="changed"), source_checks, 0)
            with exclusive_run(root):
                with self.assertRaises(BlockingIOError):
                    with exclusive_run(root):
                        pass

    def test_dual_even_odd_numbered_frames_and_nonzero_origin(self):
        for count, rate in ((161, Fraction(2997, 100)), (162, Fraction(30000, 1001))):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                src = root / "src.mp4"
                create_video(src, count, rate, origin=30)
                selected = {"video_path": "videos/part1/source.mp4", "source":
                    {"num_frames": count, "fps_num": rate.numerator, "fps_den": rate.denominator}}
                paths = output_paths(root, selected)
                with patch("data.preprocess.video.av.open", wraps=av.open) as opened:
                    records, integrity, errors = transcode(src, paths, selected, VARIANTS)
                    self.assertEqual(sum(c.args[0] == str(src) for c in opened.call_args_list), 1)
                self.assertEqual(errors, {})
                audits = audit_source_correspondence(src, root, records, integrity)
                for v, record in records.items():
                    self.assertEqual(record["num_frames"], 81)
                    self.assertNotEqual(record["source"]["first_pts"], 0)
                    require_readable(record)
                    self.assertTrue(audits[v]["source_correspondence_passed"])
                    with av.open(str(paths[v])) as container:
                        for j, frame in enumerate(container.decode(video=0)):
                            expected = np.array([2*j % 256, 6*j % 256, 14*j % 256], dtype=np.int16)
                            difference = frame.to_ndarray(format="rgb24").astype(np.int16) - expected
                            self.assertLessEqual(int(np.abs(difference).max()), 6)
                    with self.assertRaisesRegex(ValueError, "Decoded RGB hash"):
                        verify_output(paths[v], record, "wrong")
                wrong = dict(integrity)
                wrong["832x480"] = dict(wrong["832x480"], pre_encoder_rgb_sha256="incorrect")
                with self.assertRaisesRegex(ValueError, "pre-encoder RGB hash"):
                    audit_source_correspondence(src, root, records, wrong)
                with self.assertRaisesRegex(ValueError, "decoded count"):
                    transcode(src, paths, dict(selected, source=dict(selected["source"], num_frames=count+1)), VARIANTS)

    def test_encoder_failure_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "src.mp4"
            create_video(src, 5, Fraction(25))
            selected = {"video_path": "videos/source.mp4", "source": {"num_frames": 5, "fps_num": 25, "fps_den": 1}}

            def one_fails(container, rate, width, height):
                if width == 832:
                    raise ValueError("injected encoder failure")
                return add_output_stream(container, rate, width, height)

            with patch("data.preprocess.video.add_output_stream", side_effect=one_fails):
                records, integrity, errors = transcode(src, output_paths(root, selected), selected, VARIANTS)
            self.assertEqual(set(records), {"960x544"})
            self.assertEqual(set(integrity), {"960x544"})
            self.assertEqual(set(errors), {"832x480"})

    def test_bframe_presentation_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "src.mp4"
            create_video(src, 12, Fraction(25), lossy_bframes=True)
            with av.open(str(src)) as container:
                self.assertTrue(any(f.pict_type == av.video.frame.PictureType.B for f in container.decode(video=0)))
            selected = {"video_path": "videos/source.mp4", "source": {"num_frames": 12, "fps_num": 25, "fps_den": 1}}
            records, integrity, errors = transcode(src, output_paths(root, selected), selected, VARIANTS)
            self.assertEqual(errors, {})
            self.assertTrue(all(a["source_correspondence_passed"] for a in
                                audit_source_correspondence(src, root, records, integrity).values()))

    def test_all_207_rational_rates_both_geometries(self):
        with (ANALYSIS / "native_fps_stride2_81frames_counts.csv").open() as handle:
            rates = [Fraction(r["source_fps_exact"]) / 2 for r in csv.DictReader(handle)
                     if int(r["source_ge161_count"]) > 0]
        self.assertEqual(len(rates), 207)
        with tempfile.TemporaryDirectory() as directory:
            for rate in rates:
                for variant, size in VARIANTS.items():
                    path = Path(directory) / "rate.mp4"
                    with av.open(str(path), "w", options={"video_track_timescale": str(rate.numerator)}) as container:
                        stream = add_output_stream(container, rate, **size)
                        for j in range(3):
                            rgb = np.full((size["height"], size["width"], 3), [j, 3*j, 7*j], dtype=np.uint8)
                            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                            frame.time_base = Fraction(1, rate.numerator)
                            frame.pts = j * rate.denominator
                            for packet in stream.encode(frame):
                                container.mux(packet)
                        for packet in stream.encode():
                            container.mux(packet)
                    record = dict(size, fps_num=rate.numerator, fps_den=rate.denominator, num_frames=3)
                    checks, _ = verify_output(path, record)
                    self.assertEqual(checks["decoded_frames"], 3)
                    self.assertTrue(checks["pts_passed"])


if __name__ == "__main__":
    unittest.main()
