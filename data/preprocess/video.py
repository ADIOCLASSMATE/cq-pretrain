"""Decode once, retain source frames 0,2,4,..., and encode two MPEG-4 variants."""
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import math
from pathlib import Path

import av
import numpy as np
import torch
from torch.nn import functional as F


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spatial_plan(width, height):
    scale = max(width / 1920, height / 1080)
    rw, rh = max(width, round(1920 * scale)), max(height, round(1080 * scale))
    return dict(method="bicubic_resize_center_crop", resize_width=rw, resize_height=rh,
                crop_x=(rw - width) // 2, crop_y=(rh - height) // 2,
                crop_width=width, crop_height=height, antialias=True, align_corners=False)


def resize_rgb(rgb, width, height):
    plan = spatial_plan(width, height)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    resized = F.interpolate(tensor, size=(plan["resize_height"], plan["resize_width"]),
                            mode="bicubic", align_corners=False, antialias=True)
    x, y = plan["crop_x"], plan["crop_y"]
    cropped = resized[:, :, y:y + height, x:x + width]
    return cropped.clamp(0, 255).round().to(torch.uint8)[0].permute(1, 2, 0).contiguous().numpy()


def add_output_stream(container, rate, width, height):
    stream = container.add_stream("mpeg4", rate=rate)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.time_base = Fraction(1, rate.numerator)
    stream.codec_context.time_base = stream.time_base
    stream.codec_context.thread_count = 1
    stream.codec_context.flags |= av.codec.context.Flags.qscale
    stream.codec_context.global_quality = 2 * 118  # FF_QP2LAMBDA; equivalent to -q:v 2
    return stream


def verify_output(path, record, expected_decoded_rgb_sha256=None):
    """Return readable checks and a separate decoded-pixel digest."""
    rate = Fraction(record["fps_num"], record["fps_den"])
    digest = hashlib.sha256()
    count = 0
    with av.open(str(path)) as container:
        require(len(container.streams) == len(container.streams.video) == 1, "Expected video-only MP4")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        require(stream.codec_context.name == "mpeg4" and stream.codec_context.codec_tag == "mp4v",
                "Expected MPEG-4 Part 2 in MP4")
        require(stream.time_base == Fraction(1, rate.numerator), "Output time base mismatch")
        require(stream.average_rate == rate, "Output average rate mismatch")
        require(stream.duration * stream.time_base == Fraction(record["num_frames"], 1) / rate,
                "Output duration mismatch")
        for count, frame in enumerate(container.decode(stream), start=1):
            require((frame.width, frame.height) == (record["width"], record["height"]),
                    "Output geometry mismatch")
            require(frame.format.name == "yuv420p", "Expected YUV420P pixel format")
            require(not frame.is_corrupt, f"Corrupt output frame {count - 1}")
            require(frame.pts * frame.time_base == Fraction(count - 1, 1) / rate, "Output PTS mismatch")
            digest.update(frame.to_ndarray(format="rgb24").tobytes())
    require(count == record["num_frames"], "Output decoded frame count mismatch")
    if expected_decoded_rgb_sha256 is not None:
        require(digest.hexdigest() == expected_decoded_rgb_sha256, "Decoded RGB hash mismatch")
    checks = dict(decoded_frames=count, geometry_passed=True, pts_passed=True,
                  codec_passed=True, pixel_format="yuv420p")
    return checks, digest.hexdigest()


def transcode(source_path, output_paths, selected, variants):
    """Generate requested variants using one source decode, isolating encoder errors."""
    source = dict(selected["source"])
    rate = Fraction(source["fps_num"], source["fps_den"])
    output_rate = rate / 2
    time_base = Fraction(1, output_rate.numerator)
    states, errors = {}, {}
    decoded = encoded = 0
    first_time = None
    try:
        for name, path in output_paths.items():
            states[name] = dict(container=None)
            try:
                container = av.open(str(path), "w", format="mp4",
                                    options={"video_track_timescale": str(output_rate.numerator)})
                states[name]["container"] = container
                states[name].update(stream=add_output_stream(container, output_rate, **variants[name]),
                                    digest=hashlib.sha256())
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        with av.open(str(source_path)) as src:
            require(len(src.streams.video) == 1, "Expected one source video stream")
            video = src.streams.video[0]
            video.codec_context.thread_count = 1
            for index, frame in enumerate(src.decode(video)):
                require((frame.width, frame.height) == (1920, 1080), "Source geometry mismatch")
                require(frame.pts is not None, "Missing source PTS")
                current = frame.pts * frame.time_base
                if first_time is None:
                    first_time = current
                    source.update(first_pts=frame.pts, time_base_num=frame.time_base.numerator,
                                  time_base_den=frame.time_base.denominator)
                require(current - first_time == Fraction(index, 1) / rate,
                        f"Nonuniform or unexpected source PTS at frame {index}")
                require(not frame.is_corrupt, f"Corrupt source frame {index}")
                decoded += 1
                if index % 2:
                    continue
                original_rgb = frame.to_ndarray(format="rgb24")
                for name, state in states.items():
                    if name in errors:
                        continue
                    try:
                        rgb = resize_rgb(original_rgb, **variants[name])
                        state["digest"].update(rgb.tobytes())
                        target = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        target.time_base, target.pts = time_base, encoded * output_rate.denominator
                        for packet in state["stream"].encode(target):
                            state["container"].mux(packet)
                    except Exception as exc:
                        errors[name] = f"{type(exc).__name__}: {exc}"
                encoded += 1
            require(decoded == source["num_frames"], "Source decoded count differs from census")
            require(encoded == (source["num_frames"] + 1) // 2, "Stride-2 frame count mismatch")
        for name, state in states.items():
            if name not in errors:
                try:
                    for packet in state["stream"].encode():
                        state["container"].mux(packet)
                except Exception as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
    finally:
        for name, state in states.items():
            if state["container"] is not None:
                try:
                    state["container"].close()
                except Exception as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
    records, integrity = {}, {}
    for name, path in output_paths.items():
        if name in errors:
            continue
        try:
            size = variants[name]
            record = dict(selected, variant=name, **size, source=dict(source), num_frames=encoded,
                          fps_num=output_rate.numerator, fps_den=output_rate.denominator,
                          fps=float(output_rate), duration_seconds=float(Fraction(encoded, 1) / output_rate),
                          pts_span_seconds=float(Fraction(encoded - 1, 1) / output_rate),
                          spatial_transform=spatial_plan(**size))
            checks, digest = verify_output(path, record)
            checks.update(source_decoded_frames=decoded, source_pts_passed=True)
            record.update(validation=checks, output_bytes=Path(path).stat().st_size)
            records[name] = record
            integrity[name] = dict(pre_encoder_rgb_sha256=states[name]["digest"].hexdigest(),
                                   decoded_rgb_sha256=digest)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return records, integrity, errors


def audit_source_correspondence(source_path, output_root, records, integrity):
    """Rebuild both encoder inputs from one independent original-source decode."""
    first = next(iter(records.values()))
    rate = Fraction(first["source"]["fps_num"], first["source"]["fps_den"])
    stats = {name: dict(rgb_sse=0, rgb_sae=0, rgb_values=0) for name in records}
    digests = {name: hashlib.sha256() for name in records}
    checked = total = 0
    with ExitStack() as stack:
        src = stack.enter_context(av.open(str(source_path)))
        src.streams.video[0].codec_context.thread_count = 1
        outputs = {}
        for name, record in records.items():
            require(record["spatial_transform"] == spatial_plan(record["width"], record["height"]),
                    "Incorrect spatial transform metadata")
            dst = stack.enter_context(av.open(str(Path(output_root) / name / record["video_path"])))
            dst.streams.video[0].codec_context.thread_count = 1
            outputs[name] = iter(dst.decode(video=0))
        origin = None
        for i, frame in enumerate(src.decode(video=0)):
            total += 1
            if origin is None:
                origin = frame.pts * frame.time_base
            require(frame.pts * frame.time_base - origin == Fraction(i, 1) / rate,
                    "Independent source timestamp mismatch")
            if i % 2:
                continue
            original_rgb = frame.to_ndarray(format="rgb24")
            for name, record in records.items():
                target = next(outputs[name])
                require(target.pts * target.time_base == frame.pts * frame.time_base - origin,
                        "Output time does not match selected source time")
                expected = resize_rgb(original_rgb, record["width"], record["height"])
                digests[name].update(expected.tobytes())
                difference = target.to_ndarray(format="rgb24").astype(np.int16) - expected.astype(np.int16)
                stats[name]["rgb_sse"] += int(np.square(difference, dtype=np.int32).sum(dtype=np.int64))
                stats[name]["rgb_sae"] += int(np.abs(difference).sum(dtype=np.int64))
                stats[name]["rgb_values"] += expected.size
            checked += 1
        require(total == first["source"]["num_frames"], "Independent source count mismatch")
        for name, record in records.items():
            require(next(outputs[name], None) is None and checked == record["num_frames"],
                    "Independent output count mismatch")
            require(digests[name].hexdigest() == integrity[name]["pre_encoder_rgb_sha256"],
                    "Source-selected pre-encoder RGB hash mismatch")
            s = stats[name]
            s.update(source_correspondence_passed=True, compared_output_frames=checked,
                     original_indices="2*j, j=0..num_frames-1", encoder_input_matched=True,
                     psnr_rgb_db=10 * math.log10(255 ** 2 * s["rgb_values"] / s["rgb_sse"]) if s["rgb_sse"] else None,
                     mae_rgb=s["rgb_sae"] / s["rgb_values"])
    return stats
