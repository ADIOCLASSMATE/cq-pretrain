"""Bitwise input compatibility, including SSCD's EXIF and alpha contracts."""
from __future__ import annotations

import hashlib
import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')

import numpy as np
from PIL import Image
import torch

from corpus_pipeline.detector_calibration.infer import preprocessing
from . import fast_prepare
from .inference import MODEL_NAMES, ResidentInferenceWorker, _sscd_input


class FastPrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.worker = ResidentInferenceWorker.__new__(ResidentInferenceWorker)
        cls.worker.torch = torch
        cls.worker.identity_sha256 = 'bitwise-preprocessing-test'
        cls.worker.bundles = {name: {'transform': preprocessing(name)[0]}
                              for name in ('cf384', 'unifd')}
        y, x = np.indices((384, 512), dtype=np.uint16)
        cls.rgb = Image.fromarray(np.stack(((x + y) % 256, (x * 3) % 256,
                                           (y * 7) % 256), axis=2).astype('uint8'), 'RGB')

    def asset(self, image, format='PNG', **save_options):
        stream = io.BytesIO()
        image.save(stream, format=format, **save_options)
        blob = stream.getvalue()
        return {'sha256': hashlib.sha256(blob).hexdigest(), 'blob_loader': lambda: blob}

    def compare(self, asset, names=MODEL_NAMES):
        expected, reference = ResidentInferenceWorker._prepare(self.worker, asset, names)
        actual, tensors = fast_prepare.prepare_asset(self.worker, asset, names)
        expected.pop('decode_seconds')
        actual.pop('decode_seconds')
        self.assertEqual(expected, actual)
        self.assertEqual(reference is None, tensors is None)
        if tensors is not None:
            self.assertEqual(set(reference), set(tensors))
            for name in names:
                self.assertTrue(torch.equal(reference[name], tensors[name]), name)
        return actual

    def test_opaque_rgb_png_and_jpeg(self):
        for format in ('PNG', 'JPEG'):
            with self.subTest(format=format):
                result = self.compare(self.asset(self.rgb, format))
                self.assertEqual(result['status'], 'prepared')

    def test_exif_orientations_one_through_eight(self):
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                exif = Image.Exif()
                exif[274] = orientation
                result = self.compare(self.asset(self.rgb, 'JPEG', exif=exif))
                expected_size = (384, 512) if orientation in (5, 6, 7, 8) else (512, 384)
                self.assertEqual((result['oriented_width'], result['oriented_height']), expected_size)

    def test_varying_rgba_transparency(self):
        rgba = self.rgb.convert('RGBA')
        rgba.putalpha(Image.fromarray(np.tile(np.arange(512, dtype=np.uint16) % 256,
                                             (384, 1)).astype('uint8'), 'L'))
        self.compare(self.asset(rgba))

    def test_palette_transparency(self):
        palette = self.rgb.quantize(colors=32)
        self.compare(self.asset(palette, transparency=3))

    def test_rgb_transparency_metadata_uses_reference(self):
        asset = self.asset(self.rgb, transparency=(20, 40, 60))
        self.compare(asset)
        with Image.open(io.BytesIO(asset['blob_loader']())) as opened:
            opened.load()
            with patch.object(fast_prepare, '_sscd_input', wraps=_sscd_input) as fallback:
                fast_prepare.sscd_input(opened)
                fallback.assert_called_once()

    def test_grayscale_cmyk_and_la(self):
        for mode, format in [('L', 'PNG'), ('LA', 'PNG'), ('CMYK', 'JPEG')]:
            with self.subTest(mode=mode):
                self.compare(self.asset(self.rgb.convert(mode), format))

    def test_small_boundaries(self):
        for width, height, status in [(127, 200, 'too_small'), (128, 200, 'prepared')]:
            with self.subTest(width=width):
                self.assertEqual(self.compare(self.asset(self.rgb.resize((width, height))))['status'], status)

    def test_corrupt_bytes_hash_and_declared_size(self):
        corrupt = b'not an image'
        self.compare({'sha256': hashlib.sha256(corrupt).hexdigest(), 'blob_loader': lambda: corrupt})
        self.compare({**self.asset(self.rgb), 'sha256': '0' * 64})
        self.compare({**self.asset(self.rgb), 'original_width': 513})

    def test_multiframe_rejection(self):
        asset = self.asset(self.rgb, 'GIF', save_all=True,
                           append_images=[self.rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)])
        self.assertEqual(self.compare(asset)['status'], 'error')

    def test_model_subsets(self):
        asset = self.asset(self.rgb)
        for names in [('sscd',), ('cf384',), ('unifd',), ('cf384', 'unifd')]:
            with self.subTest(names=names):
                self.compare(asset, names)

    def test_rgb_fast_path_avoids_reference_full_copy(self):
        with self.rgb.copy() as opened:
            with patch.object(fast_prepare, '_sscd_input', side_effect=AssertionError('unexpected fallback')):
                value, meta = fast_prepare.sscd_input(opened)
            expected, expected_meta = _sscd_input(opened)
            self.assertTrue(np.array_equal(value, expected))
            self.assertEqual(meta, expected_meta)

    def test_streamed_hash_keeps_original_geometry_and_pixel_bytes_without_concat(self):
        real_sha256 = hashlib.sha256
        initial_lengths, update_lengths = [], []
        class Hasher:
            def __init__(self, data=b''):
                initial_lengths.append(len(data))
                self.inner = real_sha256(data)
            def update(self, data):
                update_lengths.append(len(data));self.inner.update(data)
            def hexdigest(self):
                return self.inner.hexdigest()
        with self.rgb.copy() as opened:
            expected, expected_meta = _sscd_input(opened)
            with patch.object(fast_prepare.hashlib, 'sha256', Hasher):
                actual, actual_meta = fast_prepare.sscd_input(opened)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertEqual(actual_meta, expected_meta)
            self.assertEqual(initial_lengths, [0])
            self.assertEqual(update_lengths, [12, opened.width * opened.height * 4])
        hashes = []
        for size in ((13, 17), (17, 13)):
            with Image.new('RGB', size, (20, 40, 60)) as opened:
                _, actual = fast_prepare.sscd_input(opened)
                _, expected = _sscd_input(opened)
                self.assertEqual(actual, expected)
                hashes.append(actual['pixel_sha256'])
        self.assertNotEqual(*hashes)


if __name__ == '__main__':
    unittest.main()
