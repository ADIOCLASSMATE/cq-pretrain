"""Compare MPEG-4 q=2 / H.264 CRF18 on the exact ten validated pilot RGB clips.

The lossless pilot is the pixel reference. This script never changes the frozen
preprocessing implementation, its dataset, or the STOP marker on production.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time

import av
import numpy as np

from .manifest import atomic_json, atomic_jsonl, exclusive_run, read_jsonl, sha256

MODES = {
    "mpeg4_q2": {"encoder": "mpeg4", "pixel_format": "yuv420p", "qscale": 2},
    "h264_crf18": {"encoder": "libx264", "pixel_format": "yuv420p", "crf": "18", "preset": "medium"},
}
WIDTH, HEIGHT = 960, 540
FRAME_BYTES = WIDTH * HEIGHT * 3


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def error_statistics(reference, distorted):
    """Integer arithmetic avoids uint8 subtraction and squared-error overflow."""
    difference = distorted.astype(np.int16) - reference.astype(np.int16)
    absolute = np.abs(difference)
    return dict(sse=int(np.square(difference, dtype=np.int32).sum(dtype=np.int64)),
                sae=int(absolute.sum(dtype=np.int64)),
                max_absolute_error=int(absolute.max()), values=int(reference.size))


def psnr(sse, values):
    return math.inf if sse == 0 else 10 * math.log10(255 ** 2 / (sse / values))


def checked_decode(path, record, raw_destination=None):
    """Video read + decode + RGB24 conversion, no quality calculations in timing."""
    rate = Fraction(record["fps_num"], record["fps_den"])
    count = 0
    handle = Path(raw_destination).open("wb") if raw_destination else None
    digest = hashlib.sha256() if handle else None
    start = time.perf_counter()
    try:
        with av.open(str(path)) as container:
            require(len(container.streams) == 1, "Expected one video-only stream")
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            require(stream.average_rate == rate, "FPS changed")
            require(stream.duration * stream.time_base == Fraction(record["num_frames"], 1) / rate,
                    "Encoded duration changed")
            for frame in container.decode(stream):
                require((frame.width, frame.height) == (WIDTH, HEIGHT), "Geometry changed")
                require(frame.pts * frame.time_base == Fraction(count, 1) / rate, "Timestamp changed")
                require(not frame.is_corrupt, "Corrupt decoded frame")
                rgb = frame.to_ndarray(format="rgb24")
                if handle:
                    block = rgb.tobytes()
                    handle.write(block)
                    digest.update(block)
                count += 1
        require(count == record["num_frames"], "Decoded count changed")
    finally:
        if handle:
            handle.close()
    return dict(seconds=time.perf_counter() - start, frames=count,
                rgb_sha256=digest.hexdigest() if digest else None)


def encode(frames, path, rate, mode):
    """Time RGB -> MP4, including frame construction, colour conversion and mux."""
    settings = MODES[mode]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    start = time.perf_counter()
    with av.open(str(temporary), "w", format="mp4",
                 options={"video_track_timescale": str(rate.numerator)}) as container:
        stream = container.add_stream(settings["encoder"], rate=rate)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = settings["pixel_format"]
        stream.time_base = Fraction(1, rate.numerator)
        stream.codec_context.time_base = stream.time_base
        stream.codec_context.thread_count = 1
        if mode == "mpeg4_q2":
            stream.codec_context.flags |= av.codec.context.Flags.qscale
            stream.codec_context.global_quality = 2 * 118  # FF_QP2LAMBDA
        else:
            stream.options = {"crf": "18", "preset": "medium", "threads": "1"}
        for index, rgb in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index * rate.denominator
            frame.time_base = stream.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    elapsed = time.perf_counter() - start
    os.replace(temporary, path)
    return elapsed


def ssim_rgb(reference_raw, distorted_raw, rate, destination):
    """FFmpeg SSIM on exact PyAV-decoded RGB bytes, not a second decoder's RGB."""
    raw_options = ["-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "960x540",
                   "-framerate", str(rate)]
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-threads", "1",
               "-filter_complex_threads", "1", *raw_options, "-i", str(reference_raw),
               *raw_options, "-i", str(distorted_raw), "-filter_complex",
               f"[0:v]format=gbrp[r];[1:v]format=gbrp[d];[d][r]ssim=stats_file={destination}:shortest=1[out]",
               "-map", "[out]", "-f", "null", "-"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = []
    with Path(destination).open() as handle:
        for line in handle:
            match = re.search(r"\bn:(\d+).*\bAll:([0-9.e+-]+)", line)
            require(match is not None, "Unrecognized SSIM statistics")
            require(int(match[1]) == len(result) + 1, "SSIM frame alignment changed")
            result.append(float(match[2]))
    return result


def compare_quality(frames, raw_distorted, ssim_values, record):
    distorted = np.memmap(raw_distorted, dtype=np.uint8, mode="r", shape=frames.shape)
    require(len(ssim_values) == len(frames), "Missing SSIM frames")
    rows = []
    for index in range(len(frames)):
        stats = error_statistics(frames[index], distorted[index])
        rows.append(dict(sample_id=record["sample_id"], frame_index=index,
                         original_frame_index=index * 2,
                         timestamp_seconds=float(Fraction(index * record["fps_den"], record["fps_num"])),
                         **stats, psnr_rgb_db=psnr(stats["sse"], stats["values"]),
                         mae_rgb=stats["sae"] / stats["values"], ssim_rgb=ssim_values[index]))
    del distorted
    return rows


def summarize(rows, timings, config):
    summaries = []
    for mode in MODES:
        samples = [r for r in rows if r["mode"] == mode]
        frames = sum(r["num_frames"] for r in samples)
        values = sum(r["values"] for r in samples)
        sse = sum(r["sse"] for r in samples)
        sae = sum(r["sae"] for r in samples)
        encoding_seconds = sum(r["encode_seconds_median"] for r in samples)
        decoding_seconds = sum(r["decode_seconds_median"] for r in samples)
        summaries.append(dict(mode=mode, samples=len(samples), frames=frames,
            total_bytes=sum(r["output_bytes"] for r in samples),
            size_ratio_to_lossless=sum(r["output_bytes"] for r in samples)/config["reference_bytes"],
            sse=sse, sae=sae, values=values,
            psnr_rgb_db=psnr(sse, values), mae_rgb=sae/values,
            ssim_rgb=sum(r["ssim_rgb"] * r["num_frames"] for r in samples)/frames,
            worst_clip_psnr_rgb_db=min(r["psnr_rgb_db"] for r in samples),
            worst_frame_psnr_rgb_db=min(r["worst_frame_psnr_rgb_db"] for r in samples),
            worst_frame_ssim_rgb=min(r["worst_frame_ssim_rgb"] for r in samples),
            encode_seconds=encoding_seconds, encode_fps=frames/encoding_seconds,
            decode_seconds=decoding_seconds, decode_fps=frames/decoding_seconds,
            encode_seconds_repeat_totals=[sum(t["seconds"] for t in timings if t["stage"] == "encode"
                and t["mode"] == mode and t["repeat"] == i) for i in range(config["repeats"])],
            decode_seconds_repeat_totals=[sum(t["seconds"] for t in timings if t["stage"] == "decode"
                and t["mode"] == mode and t["repeat"] == i) for i in range(config["repeats"])],
            all_media_checks_passed=all(r["media_checks_passed"] for r in samples)))
    return summaries


def run(args):
    pilot, output = args.pilot.resolve(), args.output.resolve()
    validation = json.loads((pilot / "validation.json").read_text())
    records = read_jsonl(pilot / "samples.jsonl")
    require(validation["all_passed"] and validation["source_correspondence"] and len(records) == 10,
            "Expected the independently validated ten pilot samples")
    ffmpeg_version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout
    config = dict(created_at=now(), code_sha256=sha256(Path(__file__)),
        pilot=str(pilot), reference_manifest_sha256=sha256(pilot / "samples.jsonl"),
        reference_validation_sha256=sha256(pilot / "validation.json"),
        samples=len(records), total_frames=sum(r["num_frames"] for r in records),
        reference_bytes=sum(r["output_bytes"] for r in records), repeats=args.repeats,
        width=WIDTH, height=HEIGHT, modes=MODES, python=platform.python_version(),
        av=av.__version__, numpy=np.__version__,
        ffmpeg_libraries={k: list(v) for k,v in av.library_versions.items()},
        ffmpeg_cli=ffmpeg_version, threads=1,
        scope="Same ten deliberately selected pilot clips; all frames, same resolution and PTS",
        timing="Sequential CPU, one thread per codec; RGB source warmed before encoding; encoding includes RGB-to-YUV and mux; decode includes file read and RGB conversion, excludes metrics. Three repetitions, alternating codec order, summed per-video medians. File cache normally warm; not cold-storage throughput.",
        quality="PSNR and MAE compare exact PyAV RGB pixels to verified pre-encode RGB; SSIM runs FFmpeg on those same RGB bytes, gbrp, equal channel mean. Global PSNR uses total squared error, not mean dB. Global SSIM/MAE weighted by frame/pixel count. Includes YUV420 colour-conversion and chroma-subsampling loss.",
        quality_reference="Already verified lossless 960x540 stride-2 RGB. Does not assess source-video compression or resizing loss; does not measure downstream training quality. Quality and final file size use the last encoding repetition; all repeat sizes and hashes are retained because an encoder need not produce byte-identical bitstreams.")
    configuration_path = output / "comparison_config.json"
    if configuration_path.exists():
        previous = json.loads(configuration_path.read_text())
        for key in ("code_sha256", "reference_manifest_sha256", "reference_validation_sha256", "repeats", "modes", "av", "numpy"):
            require(previous[key] == config[key], f"Cannot resume changed comparison: {key}")
        config = previous
    else:
        atomic_json(configuration_path, config)
    state = output / "_state"
    scratch = output / "_scratch"
    state.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    all_samples, all_frames, timings = [], [], []
    started = time.monotonic()
    for number, record in enumerate(records, start=1):
        completed_path = state / f"sample{number:02d}.json"
        if completed_path.exists():
            saved = json.loads(completed_path.read_text())
            require(saved["sample_id"] == record["sample_id"], "Resume sample changed")
            for row in saved["samples"]:
                require(sha256(output / row["video_path"]) == row["sha256"], "Saved comparison video changed")
            all_samples.extend(saved["samples"])
            all_frames.extend(saved["frames"])
            timings.extend(saved["timings"])
            continue
        reference = pilot / record["video_path"]
        require(sha256(reference) == record["output_sha256"], "Reference MP4 changed")
        rate = Fraction(record["fps_num"], record["fps_den"])
        sample_timings = []
        with tempfile.TemporaryDirectory(prefix=f"sample{number:02d}-", dir=scratch) as directory:
            directory = Path(directory)
            reference_raw = directory / "reference.rgb"
            reference_decode = checked_decode(reference, record, reference_raw)
            require(reference_decode["rgb_sha256"] == record["validation"]["rgb_sha256"],
                    "Reference differs from original resized RGB digest")
            require(reference_raw.stat().st_size == record["num_frames"] * FRAME_BYTES, "Raw reference size changed")
            # Keep only this clip in RAM. Avoid disk page faults biasing codec order.
            frames = np.fromfile(reference_raw, dtype=np.uint8).reshape((-1, HEIGHT, WIDTH, 3))
            paths = {mode: output / "videos" / mode / f"part{record['source']['part']}" / record["video"] for mode in MODES}
            hashes = {}
            for repeat in range(args.repeats):
                modes = list(MODES)
                if (number + repeat) % 2:
                    modes.reverse()
                for mode in modes:
                    seconds = encode(frames, paths[mode], rate, mode)
                    digest = sha256(paths[mode])
                    hashes[mode] = digest
                    sample_timings.append(dict(sample_id=record["sample_id"], mode=mode,
                        stage="encode", repeat=repeat, seconds=seconds, frames=len(frames),
                        output_bytes=paths[mode].stat().st_size, output_sha256=digest))
                print(json.dumps(dict(sample=number, id=record["sample_id"], encoding_repeat=repeat+1)), flush=True)
            require(hashlib.sha256(frames).hexdigest() == record["validation"]["rgb_sha256"],
                    "Shared RGB input was modified during encoding")
            decode_paths = dict(paths, lossless_rgb=reference)
            for repeat in range(args.repeats):
                modes = list(decode_paths)
                shift = (number + repeat) % len(modes)
                for mode in modes[shift:] + modes[:shift]:
                    measurement = checked_decode(decode_paths[mode], record)
                    sample_timings.append(dict(sample_id=record["sample_id"], mode=mode,
                        stage="decode", repeat=repeat, seconds=measurement["seconds"], frames=len(frames),
                        output_bytes=decode_paths[mode].stat().st_size, output_sha256=""))
            sample_rows, frame_rows = [], []
            for mode in MODES:
                distorted_raw = directory / "distorted.rgb"
                checked_decode(paths[mode], record, distorted_raw)
                stats_path = output / "ssim_logs" / mode / f"sample{number:02d}.txt"
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                similarities = ssim_rgb(reference_raw, distorted_raw, rate, stats_path)
                quality = compare_quality(frames, distorted_raw, similarities, record)
                for row in quality:
                    row["mode"] = mode
                frame_rows.extend(quality)
                total_sse = sum(r["sse"] for r in quality)
                total_sae = sum(r["sae"] for r in quality)
                total_values = sum(r["values"] for r in quality)
                enc_times = [t["seconds"] for t in sample_timings if t["mode"] == mode and t["stage"] == "encode"]
                dec_times = [t["seconds"] for t in sample_timings if t["mode"] == mode and t["stage"] == "decode"]
                sample_rows.append(dict(sample_id=record["sample_id"], mode=mode,
                    video_path=str(paths[mode].relative_to(output)), reference_path=str(reference),
                    sha256=hashes[mode], num_frames=len(frames), fps_num=rate.numerator, fps_den=rate.denominator,
                    output_bytes=paths[mode].stat().st_size, reference_bytes=record["output_bytes"],
                    source_bytes=record["source"]["bytes"],
                    sse=total_sse, sae=total_sae, values=total_values,
                    psnr_rgb_db=psnr(total_sse, total_values), mae_rgb=total_sae/total_values,
                    ssim_rgb=float(np.mean(similarities)),
                    worst_frame_psnr_rgb_db=min(r["psnr_rgb_db"] for r in quality),
                    worst_frame_ssim_rgb=min(similarities),
                    worst_frame_index=max(quality, key=lambda r:r["sse"])["frame_index"],
                    max_absolute_error=max(r["max_absolute_error"] for r in quality),
                    encode_seconds_median=statistics.median(enc_times),
                    encode_seconds_min=min(enc_times), encode_seconds_max=max(enc_times),
                    encode_bytes_min=min(t["output_bytes"] for t in sample_timings if t["mode"] == mode and t["stage"] == "encode"),
                    encode_bytes_max=max(t["output_bytes"] for t in sample_timings if t["mode"] == mode and t["stage"] == "encode"),
                    decode_seconds_median=statistics.median(dec_times),
                    decode_seconds_min=min(dec_times), decode_seconds_max=max(dec_times),
                    media_checks_passed=True))
                distorted_raw.unlink()
            del frames
            saved = dict(sample_id=record["sample_id"], samples=sample_rows, frames=frame_rows, timings=sample_timings)
            atomic_json(completed_path, saved)
            all_samples.extend(sample_rows)
            all_frames.extend(frame_rows)
            timings.extend(sample_timings)
        atomic_json(output / "progress.json", dict(updated_at=now(), completed_samples=number, total_samples=10,
                    elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(completed_sample=number, id=record["sample_id"], rows=sample_rows)), flush=True)
    summaries = summarize(all_samples, timings, config)
    require(len(all_samples) == 20 and len(all_frames) == 2 * config["total_frames"],
            "Incomplete comparison population")
    require(all(s["samples"] == 10 and s["frames"] == config["total_frames"] for s in summaries),
            "Candidate populations differ")
    write_csv(output / "per_sample.csv", all_samples)
    write_csv(output / "per_frame.csv", all_frames)
    write_csv(output / "timings.csv", timings)
    manifests = []
    originals = {r["sample_id"]: r for r in records}
    for row in all_samples:
        original = originals[row["sample_id"]]
        manifests.append(dict(sample_id=original["sample_id"], video=original["video"],
            video_path=row["video_path"], caption=original["caption"], mode=row["mode"],
            width=WIDTH, height=HEIGHT, num_frames=row["num_frames"],
            fps_num=row["fps_num"], fps_den=row["fps_den"], fps=row["fps_num"]/row["fps_den"],
            original_source=original["source"]))
    atomic_jsonl(output / "samples.jsonl", manifests)
    lossless_seconds = sum(statistics.median([t["seconds"] for t in timings if t["sample_id"] == r["sample_id"]
        and t["mode"] == "lossless_rgb" and t["stage"] == "decode"]) for r in records)
    report = dict(complete=True, completed_at=now(), samples=10, total_frames=config["total_frames"],
                  reference_bytes=config["reference_bytes"], lossless_decode_seconds=lossless_seconds,
                  lossless_decode_fps=config["total_frames"]/lossless_seconds, results=summaries,
                  all_media_checks_passed=all(r["all_media_checks_passed"] for r in summaries))
    atomic_json(output / "summary.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    require(args.repeats >= 3, "At least three timing repetitions required")
    with exclusive_run(args.output):
        run(args)


if __name__ == "__main__":
    main()
