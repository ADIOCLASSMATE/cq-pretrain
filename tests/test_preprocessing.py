"""Pixel/posterior parity with the source Resize(BICUBIC) + CenterCrop contract."""
import numpy as np
from PIL import Image
import pytest
import torch

from scripts.prepare_latents import image_tensor
from models.codec import KL16Codec
from test_evaluation_runtime import SmallVAE


@pytest.mark.parametrize("original,size,resized,crop", [
    ((768, 768), 256, (256, 256), (0, 0, 256, 256)),
    ((901, 601), 256, (383, 256), (64, 0, 320, 256)),
    ((601, 901), 256, (256, 383), (0, 64, 256, 320)),
    ((47, 32), 32, (47, 32), (8, 0, 40, 32)),
    ((49, 33), 32, (47, 32), (8, 0, 40, 32)),
    ((7, 11), 32, (32, 50), (0, 9, 32, 41)),
    ((32, 32), 32, (32, 32), (0, 0, 32, 32)),
])
def test_source_pixels_and_posterior_for_downsample_odd_crop_and_upsample(tmp_path, original, size, resized, crop):
    pixels = np.random.default_rng(9).integers(0, 256, size=(original[1], original[0], 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    path = tmp_path / "source.png"
    source.save(path)
    # Explicit source geometries distinguish truncating Resize from rounding,
    # and CenterCrop's round-to-even from floor; high-frequency pixels expose BOX.
    reference_image = source.resize(resized, Image.Resampling.BICUBIC).crop(crop)
    reference = torch.from_numpy(np.array(reference_image, copy=True)).permute(2, 0, 1).float() / 255
    actual = image_tensor(path, size)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    vae = SmallVAE()
    posterior = vae.encode((reference.unsqueeze(0) - .5) / .5)
    expected = torch.cat((posterior.mean, posterior.std), 1).mul(.2325)
    expected = expected.permute(0, 2, 3, 1).reshape(1, -1, 32)
    torch.testing.assert_close(KL16Codec(vae).encode_posterior(actual.unsqueeze(0)), expected, rtol=0, atol=0)


def test_direct_source_torchvision_pipeline(tmp_path):
    """Optional evaluation dependency provides the original transform oracle."""
    pytest.importorskip("torchvision")
    from torchvision import transforms
    rng = np.random.default_rng(18)
    dimensions = [(768, 768), (901, 601), (601, 901), (47, 32), (49, 33), (7, 11)]
    dimensions += [tuple(map(int, pair)) for pair in rng.integers(7, 1100, size=(18, 2))]
    for index, (width, height) in enumerate(dimensions):
        size = 256 if index % 2 == 0 else 32
        source = Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8))
        path = tmp_path / "source.png"
        source.save(path)
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size), transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])
        torch.testing.assert_close(image_tensor(path, size).mul(2).sub(1), transform(source), rtol=0, atol=0)
