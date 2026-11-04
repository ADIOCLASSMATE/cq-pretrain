"""Export exact benchmark-decoder RGB pixels for the visual comparison figures."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np

from .compare_codecs import error_statistics
from .manifest import atomic_json


def frame_at(path, index, rate):
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for number, frame in enumerate(container.decode(video=0)):
            if number == index:
                if frame.pts * frame.time_base != Fraction(index, 1) / rate:
                    raise ValueError("Visual comparison timestamp mismatch")
                return frame.to_ndarray(format="rgb24")
    raise ValueError("Missing comparison frame")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    with (root / "per_sample.csv").open() as handle:
        samples = list(csv.DictReader(handle))
    with (root / "per_frame.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    sample_ids = list(dict.fromkeys(r["sample_id"] for r in samples))
    destination = root / "visual_data"
    destination.mkdir(exist_ok=True)
    manifest = []
    for number, name in enumerate(sample_ids, start=1):
        rows = [r for r in samples if r["sample_id"] == name]
        selected = max((r for r in metrics if r["sample_id"] == name), key=lambda r:int(r["sse"]))
        index = int(selected["frame_index"])
        rate = Fraction(int(rows[0]["fps_num"]), int(rows[0]["fps_den"]))
        pixels = {"reference": frame_at(rows[0]["reference_path"], index, rate)}
        quality = {}
        for row in rows:
            mode = row["mode"]
            pixels[mode] = frame_at(root / row["video_path"], index, rate)
            stored = next(r for r in metrics if r["sample_id"] == name and r["mode"] == mode
                          and int(r["frame_index"]) == index)
            actual = error_statistics(pixels["reference"], pixels[mode])
            if actual["sse"] != int(stored["sse"]) or actual["sae"] != int(stored["sae"]):
                raise ValueError("Figure RGB bytes differ from measured quality pixels")
            quality[mode] = {k:float(stored[k]) for k in ("psnr_rgb_db", "mae_rgb", "ssim_rgb")}
        np.savez_compressed(destination / f"sample{number:02d}.npz", **pixels)
        manifest.append(dict(number=number, sample_id=name, frame_index=index,
            timestamp_seconds=float(Fraction(index, 1)/rate),
            selection="Frame with maximum RGB squared error across the two codecs for this clip",
            pixel_data=f"visual_data/sample{number:02d}.npz", quality=quality))
    atomic_json(root / "visual_data/manifest.json", manifest)
    print(json.dumps(dict(exported_frames=len(manifest), quality_pixels_reverified=True)), flush=True)


if __name__ == "__main__":
    main()
