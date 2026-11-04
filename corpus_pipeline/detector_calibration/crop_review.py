"""Show original pixels and the exact spatial crops used by each detector.

This only builds inspection contact sheets, never alters dataset images.
Input is a review JSONL with image_path/image_sha256/review_id fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.review.read_text().splitlines() if line]
    by_id = {row['review_id']: row for row in rows}
    selected = [by_id[key] for key in args.ids]
    args.output.mkdir(exist_ok=True, parents=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    spatial = [None,
        transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), transforms.CenterCrop(224)]),
        transforms.Compose([transforms.Resize(440, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), transforms.CenterCrop(384)]),
        transforms.CenterCrop(224)]
    manifest = []
    for row in selected:
        canvas = Image.new('RGB', (1600, 460), '#dddddd')
        draw = ImageDraw.Draw(canvas)
        with Image.open(row['image_path']) as raw:
            original = raw.convert('RGB')  # Deliberately no EXIF transpose: detector convention.
            for i, (label, transform) in enumerate(zip(('Original', 'CF224 crop', 'CF384 crop', 'UniFD native crop'), spatial)):
                crop = original.copy() if transform is None else transform(original)
                crop.thumbnail((390, 390))
                x = i * 400
                draw.text((x + 8, 8), row['review_id'] + ' ' + label, fill='black', font=font)
                canvas.paste(crop, (x + (400 - crop.width) // 2, 42 + (390 - crop.height) // 2))
        path = args.output / (row['review_id'] + '.jpg')
        canvas.save(path, quality=96)
        manifest.append({'review_id': row['review_id'], 'image_sha256': row['image_sha256'], 'panel': str(path.resolve()), 'exif_transpose': False})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
