from fractions import Fraction
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data.preprocess.compare_codecs import (MODES, checked_decode, encode, error_statistics,
                                          psnr, ssim_rgb, summarize)


class CodecComparisonTests(unittest.TestCase):
    def test_error_arithmetic_without_uint8_overflow(self):
        reference = np.zeros((2, 2, 3), dtype=np.uint8)
        distorted = np.full_like(reference, 255)
        stats = error_statistics(reference, distorted)
        self.assertEqual(stats["sse"], 12 * 255 ** 2)
        self.assertEqual(stats["sae"], 12 * 255)
        self.assertEqual(psnr(stats["sse"], stats["values"]), 0)
        self.assertTrue(math.isinf(psnr(0, 12)))

    def test_global_psnr_from_total_squared_error(self):
        rows = []
        for n, mse in [(1, 1), (3, 9)]:
            rows.append(dict(mode="mpeg4_q2", num_frames=n, values=3*n,
                sse=mse*3*n, sae=n*3, output_bytes=n, encode_seconds_median=n,
                decode_seconds_median=n/2, ssim_rgb=0.9+n/100,
                psnr_rgb_db=psnr(mse*3*n, 3*n), worst_frame_psnr_rgb_db=20,
                worst_frame_ssim_rgb=0.8, media_checks_passed=True))
        # Mirror only the inputs for the second candidate, keeping the oracle independent.
        rows.extend([dict(row, mode="h264_crf18") for row in list(rows)])
        result = summarize(rows, [], {"reference_bytes": 8, "repeats": 3})[0]
        self.assertAlmostEqual(result["psnr_rgb_db"], 10*math.log10(255**2/7))
        self.assertAlmostEqual(result["ssim_rgb"], (0.91+3*0.93)/4)
        self.assertEqual(result["encode_fps"], 1)

    def test_real_codec_pts_and_ssim_identity(self):
        rng = np.random.default_rng(7)
        frames = rng.integers(0, 256, (3, 540, 960, 3), dtype=np.uint8)
        rate = Fraction(2997, 200)
        record = dict(num_frames=3, fps_num=2997, fps_den=200)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            reference = directory / "reference.rgb"
            frames.tofile(reference)
            scores = ssim_rgb(reference, reference, rate, directory / "identity.txt")
            self.assertEqual(scores, [1.0]*3)
            for mode in MODES:
                path = directory / (mode+".mp4")
                encode(frames, path, rate, mode)
                self.assertEqual(checked_decode(path, record)["frames"], 3)


if __name__ == "__main__":
    unittest.main()
