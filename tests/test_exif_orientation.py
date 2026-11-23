"""A damaged EXIF block must not stop training: the picture is used as stored."""

import io

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageOps

from corpus_pipeline.dedup.dcvlm_loss_policy_reader import decode_image_bytes


def _jpeg(exif):
    picture = Image.fromarray((np.arange(48 * 32 * 3) % 251).astype(np.uint8).reshape(32, 48, 3))
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", **({"exif": exif} if exif is not None else {}))
    return buffer.getvalue()


def test_unreadable_exif_keeps_the_stored_pixels(capfd):
    blob = _jpeg(b"Exif\x00\x00" + b" " * 64)
    with Image.open(io.BytesIO(blob)) as image:
        stored = image.convert("RGB")
    # The reader's next call on the decoded picture; this is what raised in training.
    with pytest.raises(SyntaxError):
        ImageOps.exif_transpose(stored)
    result = ImageOps.exif_transpose(decode_image_bytes(blob))
    assert result.size == (48, 32)
    assert np.array_equal(np.asarray(result), np.asarray(stored))
    assert "exif_unreadable" in capfd.readouterr().err


def test_valid_orientation_is_still_applied(capfd):
    exif = Image.Exif()
    exif[0x0112] = 6
    assert ImageOps.exif_transpose(decode_image_bytes(_jpeg(exif.tobytes()))).size == (32, 48)
    assert "exif_unreadable" not in capfd.readouterr().err


def test_plain_image_is_untouched():
    assert ImageOps.exif_transpose(decode_image_bytes(_jpeg(None))).size == (48, 32)
