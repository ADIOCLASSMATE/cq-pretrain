"""A damaged EXIF block must not stop training: the picture is used as stored."""
import io

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

from data.images import upright


def _jpeg(exif):
    picture = Image.fromarray((np.arange(48 * 32 * 3) % 251).astype(np.uint8).reshape(32, 48, 3))
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", **({"exif": exif} if exif is not None else {}))
    return buffer.getvalue()


def _load(blob):
    image = Image.open(io.BytesIO(blob))
    image.load()
    return image


def test_unreadable_exif_keeps_the_stored_pixels(capfd):
    blob = _jpeg(b"Exif\x00\x00" + b" " * 64)
    # The JPEG opener swallows this error for its own DPI lookup; the RGB copy
    # the reader works on parses the block again and raises.
    with pytest.raises(SyntaxError):
        _load(blob).convert("RGB").getexif()
    result = upright(_load(blob).convert("RGB"), "abc")
    assert result.size == (48, 32)
    assert np.array_equal(np.asarray(result), np.asarray(_load(blob).convert("RGB")))
    assert '"exif_unreadable": "abc"' in capfd.readouterr().err


def test_valid_orientation_is_still_applied():
    exif = Image.Exif()
    exif[0x0112] = 6
    result = upright(_load(_jpeg(exif.tobytes())))
    assert result.size == (32, 48)


def test_plain_image_is_untouched():
    assert upright(_load(_jpeg(None))).size == (48, 32)
