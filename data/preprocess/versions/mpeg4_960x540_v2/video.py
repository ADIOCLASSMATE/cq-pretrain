"""Streaming stride-2 resize, MPEG-4 Part 2 q=2, and exact rational PTS audit."""
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


def resize_rgb(rgb, width=960, height=540):
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    resized = F.interpolate(tensor, size=(height, width), mode="bicubic",
                            align_corners=False, antialias=True)
    return resized.clamp(0, 255).round().to(torch.uint8)[0].permute(1, 2, 0).contiguous().numpy()


def add_output_stream(container, rate, width=960, height=540):
    """Use the same native MPEG-4 q=2 settings as the ten-sample comparison."""
    stream = container.add_stream("mpeg4", rate=rate)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.time_base = Fraction(1, rate.numerator)
    stream.codec_context.time_base = stream.time_base
    stream.codec_context.thread_count = 1
    stream.codec_context.flags |= av.codec.context.Flags.qscale
    stream.codec_context.global_quality = 2 * 118  # FF_QP2LAMBDA, equivalent to -q:v 2
    return stream


def verify_output(path, record, expected_decoded_rgb_sha256=None):
    rate = Fraction(record["fps_num"], record["fps_den"])
    digest = hashlib.sha256()
    count = 0
    with av.open(str(path)) as container:
        require(len(container.streams) == 1 and len(container.streams.video) == 1,
                "Expected video-only MP4")
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
            require(frame.pts * frame.time_base == Fraction(count - 1, 1) / rate,
                    "Output PTS mismatch")
            digest.update(frame.to_ndarray(format="rgb24").tobytes())
    require(count == record["num_frames"], "Output decoded frame count mismatch")
    if expected_decoded_rgb_sha256 is not None:
        require(digest.hexdigest() == expected_decoded_rgb_sha256, "Decoded RGB hash mismatch")
    return {"decoded_frames": count, "geometry_passed": True, "pts_passed": True,
            "codec_passed": True, "pixel_format": "yuv420p",
            "decoded_rgb_sha256": digest.hexdigest()}


def transcode(source_path, output_path, selected):
    """Validate every source presentation timestamp, encoding only frames 0,2,4,..."""
    source = selected["source"]
    rate = Fraction(source["fps_num"], source["fps_den"])
    output_rate = rate / 2
    time_base = Fraction(1, output_rate.numerator)
    record = dict(selected, width=960, height=540,
                  num_frames=(source["num_frames"] + 1) // 2,
                  fps_num=output_rate.numerator, fps_den=output_rate.denominator,
                  fps=float(output_rate))
    record["source"] = dict(source)
    record["duration_seconds"] = float(Fraction(record["num_frames"], 1) / output_rate)
    record["pts_span_seconds"] = float(Fraction(record["num_frames"] - 1, 1) / output_rate)
    digest = hashlib.sha256()
    decoded = encoded = 0
    first_time = None
    with av.open(str(source_path)) as src, av.open(
            str(output_path), "w", format="mp4",
            options={"video_track_timescale": str(output_rate.numerator)}) as dst:
        require(len(src.streams.video) == 1, "Expected one source video stream")
        video = src.streams.video[0]
        video.codec_context.thread_count = 1
        out = add_output_stream(dst, output_rate)
        for index, frame in enumerate(src.decode(video)):
            require((frame.width, frame.height) == (1920, 1080), "Source geometry mismatch")
            require(frame.pts is not None, "Missing source PTS")
            current = frame.pts * frame.time_base
            if first_time is None:
                first_time = current
                record["source"].update(first_pts=frame.pts,
                    time_base_num=frame.time_base.numerator, time_base_den=frame.time_base.denominator)
            require(current - first_time == Fraction(index, 1) / rate,
                    f"Nonuniform or unexpected source PTS at frame {index}")
            require(not frame.is_corrupt, f"Corrupt source frame {index}")
            decoded += 1
            if index % 2:
                continue
            rgb = resize_rgb(frame.to_ndarray(format="rgb24"))
            digest.update(rgb.tobytes())
            target = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            target.time_base = time_base
            target.pts = encoded * output_rate.denominator
            for packet in out.encode(target):
                dst.mux(packet)
            encoded += 1
        require(decoded == source["num_frames"], "Source decoded count differs from census")
        require(encoded == record["num_frames"], "Stride-2 frame count mismatch")
        for packet in out.encode():
            dst.mux(packet)
    validation = verify_output(output_path, record)
    validation.update(source_decoded_frames=decoded, source_pts_passed=True,
                      pre_encoder_rgb_sha256=digest.hexdigest())
    record["validation"] = validation
    record["output_bytes"] = Path(output_path).stat().st_size
    return record


def audit_source_correspondence(source_path, output_path, record):
    """Rebuild the encoder input from source frames 2*j and measure lossy error."""
    rate = Fraction(record["source"]["fps_num"], record["source"]["fps_den"])
    checked = total = 0
    digest = hashlib.sha256()
    sse = sae = values = 0
    with av.open(str(source_path)) as src, av.open(str(output_path)) as dst:
        src.streams.video[0].codec_context.thread_count = 1
        dst.streams.video[0].codec_context.thread_count = 1
        outputs = iter(dst.decode(video=0))
        origin = None
        for i, frame in enumerate(src.decode(video=0)):
            total += 1
            if origin is None:
                origin = frame.pts * frame.time_base
            if i != 2 * checked:
                continue
            target = next(outputs)
            require(target.pts * target.time_base == frame.pts * frame.time_base - origin,
                    "Output time does not match selected source time")
            expected = resize_rgb(frame.to_ndarray(format="rgb24"))
            digest.update(expected.tobytes())
            difference = target.to_ndarray(format="rgb24").astype(np.int16) - expected.astype(np.int16)
            sse += int(np.square(difference, dtype=np.int32).sum(dtype=np.int64))
            sae += int(np.abs(difference).sum(dtype=np.int64))
            values += expected.size
            require(target.pts * target.time_base == Fraction(i, 1) / rate,
                    "Independent timestamp mismatch")
            checked += 1
        require(next(outputs, None) is None, "Extra output frames")
    require(total == record["source"]["num_frames"] and checked == record["num_frames"],
            "Independent count mismatch")
    require(digest.hexdigest() == record["validation"]["pre_encoder_rgb_sha256"],
            "Source-selected pre-encoder RGB hash mismatch")
    return {"source_correspondence_passed": True, "compared_output_frames": checked,
            "original_indices": "2*j, j=0..num_frames-1",
            "pre_encoder_rgb_sha256_matched": True, "rgb_sse": sse, "rgb_sae": sae,
            "rgb_values": values, "psnr_rgb_db": 10 * math.log10(255 ** 2 * values / sse) if sse else None,
            "mae_rgb": sae / values}
