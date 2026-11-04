"""Pixel-exact preparation for opaque, unrotated RGB images.

The qualified reference remains unchanged. RGB images with no transparency and
no EXIF rotation need neither a full-size EXIF copy nor white alpha compositing.
Other SSCD inputs use the original preprocessing. Detector crop/resize and all
normalization arithmetic are the original operations. Execution qualification
must bind this file's hash before this optional path is used in production.
"""
from __future__ import annotations

import hashlib
import io
import struct
import time

import numpy as np
from PIL import Image

from .inference import MAX_PIXELS, _read_blob, _sscd_input


def _opaque_rgb(opened):
    return opened.mode == 'RGB' and 'transparency' not in opened.info


def sscd_input(opened):
    """Keep the reference RGBA pixel identity and exact 320px model tensor."""
    if not _opaque_rgb(opened) or opened.getexif().get(274) not in (None, 1):
        return _sscd_input(opened)
    rgba = opened.convert('RGBA')
    rgba.load()
    # Incremental updates hash the identical byte stream without allocating a
    # second full-size RGBA byte string for header + pixels concatenation.
    pixel_digest = hashlib.sha256()
    pixel_digest.update(b'RGBA' + struct.pack('>II', *rgba.size))
    pixel_digest.update(rgba.tobytes())
    pixel_sha = pixel_digest.hexdigest()
    tensor = np.asarray(opened.resize((320, 320), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tensor = ((tensor - np.array([.485, .456, .406], np.float32)) /
              np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1).copy()
    return tensor, {'pixel_sha256': pixel_sha, 'oriented_width': opened.width,
                    'oriented_height': opened.height}


def prepare_asset(worker, asset, names):
    """Same result/error contract as ResidentInferenceWorker._prepare."""
    started = time.perf_counter()
    expected = asset.get('sha256', asset.get('image_sha256'))
    result = {'asset_id': asset.get('asset_id', expected), 'image_sha256': expected,
              'model_identity_sha256': worker.identity_sha256, 'scores': {}, 'status': 'error'}
    try:
        blob, sha = _read_blob(asset)
        with Image.open(io.BytesIO(blob)) as opened:
            result.update(sha256=sha, original_width=opened.width, original_height=opened.height)
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError('image pixel limit exceeded')
            if getattr(opened, 'n_frames', 1) != 1:
                raise ValueError('multiframe input requires a frame-specific adapter')
            if min(opened.size) < 128:
                result.update(status='too_small', size_status='rejected_too_small',
                              reason='original_image_short_edge_lt_128')
                return result, None
            result['size_status'] = 'pass'
            for key, value in zip(('width', 'height'), opened.size):
                declared = asset.get('original_' + key, asset.get(key))
                if declared is not None and declared != value:
                    raise ValueError('declared original image dimensions mismatch')
            opened.load()
            tensors = {}
            if any(name != 'sscd' for name in names):
                # Official detector preprocessing deliberately ignores EXIF.
                rgb = opened if _opaque_rgb(opened) else opened.convert('RGB')
                for name in names:
                    if name != 'sscd':
                        tensors[name] = worker.bundles[name]['transform'](rgb)
            if 'sscd' in names:
                array, metadata = sscd_input(opened)
                tensors['sscd'] = worker.torch.from_numpy(array)
                result.update(metadata)
        result['status'] = 'prepared'
        return result, tensors
    except Exception as exc:
        result.update(error_stage='decode', error_type=type(exc).__name__, error=str(exc)[:400])
        return result, None
    finally:
        result['decode_seconds'] = time.perf_counter() - started
