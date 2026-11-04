"""Read MP4 dimensions and complete sample timing tables without RGB decoding.

Only ordinary (non-fragmented) MP4 video tracks are accepted. Structural errors
raise; inconsistent sample/timing counts are retained as invalid_timing.
"""
from __future__ import annotations

import bisect
import io
import struct
from pathlib import Path

import numpy as np


class SplitFile(io.RawIOBase):
    """Seekable concatenation, including ZIP files split into aa/ab parts."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.ends = [0]
        for path in self.paths:
            self.ends.append(self.ends[-1] + path.stat().st_size)
        self.handles = {}
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, self.ends[-1])[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.ends[-1] - self.position
        result = []
        while size > 0 and self.position < self.ends[-1]:
            index = bisect.bisect_right(self.ends, self.position) - 1
            if index not in self.handles:
                self.handles[index] = self.paths[index].open("rb")
            handle = self.handles[index]
            handle.seek(self.position - self.ends[index])
            block = handle.read(min(size, self.ends[index + 1] - self.position))
            if not block:
                raise EOFError("Truncated archive part")
            result.append(block)
            self.position += len(block)
            size -= len(block)
        return b"".join(result)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        super().close()


def boxes(data):
    data = memoryview(data)
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("Truncated MP4 box header")
        size, kind = struct.unpack_from(">I4s", data, offset)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size = len(data) - offset
        if size < header or offset + size > len(data):
            raise ValueError(f"Invalid box size: {kind!r} {size}")
        yield kind, data[offset + header:offset + size]
        offset += size


def one(data, kind):
    matches = [payload for name, payload in boxes(data) if name == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected one {kind!r}, found {len(matches)}")
    return matches[0]


def optional(data, kind):
    return next((payload for name, payload in boxes(data) if name == kind), None)


def read_moov(handle, file_size):
    while handle.tell() < file_size:
        start = handle.tell()
        header = handle.read(8)
        if len(header) != 8:
            raise EOFError("No complete moov box")
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            size = struct.unpack(">Q", handle.read(8))[0]
            header_size = 16
        elif size == 0:
            size = file_size - start
        if size < header_size or start + size > file_size:
            raise ValueError("Top-level MP4 box exceeds file")
        if kind == b"moov":
            payload = handle.read(size - header_size)
            if len(payload) != size - header_size:
                raise EOFError("Truncated moov")
            return payload
        # ZipExtFile.seek discards decompressed data in bounded chunks.
        handle.seek(size - header_size, io.SEEK_CUR)
    raise ValueError("Missing moov")


def rle_table(payload, signed=False):
    count = struct.unpack_from(">I", payload, 4)[0]
    if len(payload) != 8 + count * 8:
        raise ValueError("Invalid timing table length")
    return [struct.unpack_from(">Ii" if signed else ">II", payload, 8 + i * 8)
            for i in range(count)]


def presentation_timing(stts, ctts=None):
    count = sum(n for n, _ in stts)
    if count < 2 or any(n <= 0 or delta <= 0 for n, delta in stts):
        raise ValueError("Invalid frame timing")
    duration = sum(n * delta for n, delta in stts)
    # The last sample duration is not an inter-frame PTS interval.
    interval_runs = list(stts)
    n, delta = interval_runs[-1]
    interval_runs[-1] = (n - 1, delta)
    delta_set = {d for n, d in interval_runs if n}
    if ctts is None and len(delta_set) == 1:
        delta = next(iter(delta_set))
        return dict(timing_class="constant", pts_span_ticks=(count - 1) * delta,
                    pts_interval_min_ticks=delta, pts_interval_max_ticks=delta,
                    uniform_grid_max_error_ticks=0.0, frame_count=count,
                    sample_duration_ticks=duration)
    deltas = np.repeat([d for _, d in stts], [n for n, _ in stts]).astype(np.int64)
    pts = np.concatenate(([0], np.cumsum(deltas[:-1])))
    if ctts is not None:
        if sum(n for n, _ in ctts) != count:
            raise ValueError("ctts/stts sample count mismatch")
        offsets = np.repeat([d for _, d in ctts], [n for n, _ in ctts])
        pts = np.sort(pts + offsets)
    intervals = np.diff(pts)
    if intervals.min() <= 0:
        raise ValueError("Non-increasing presentation timestamps")
    error = float(np.abs(pts - np.linspace(pts[0], pts[-1], count)).max())
    classification = ("constant" if intervals.min() == intervals.max() else
                      "quantized_uniform" if error <= 1.00000001 else "variable")
    return dict(timing_class=classification, pts_span_ticks=int(pts[-1] - pts[0]),
                pts_interval_min_ticks=int(intervals.min()),
                pts_interval_max_ticks=int(intervals.max()),
                uniform_grid_max_error_ticks=error, frame_count=count,
                sample_duration_ticks=duration)


def parse_moov(moov):
    if optional(moov, b"mvex") is not None:
        raise ValueError("Fragmented MP4 requires separate inspection")
    tracks = []
    for name, track in boxes(moov):
        if name != b"trak":
            continue
        mdia = one(track, b"mdia")
        if bytes(one(mdia, b"hdlr")[8:12]) != b"vide":
            continue
        tracks.append((track, mdia))
    if len(tracks) != 1:
        raise ValueError(f"Expected one video track, found {len(tracks)}")
    track, mdia = tracks[0]
    mdhd = one(mdia, b"mdhd")
    if mdhd[0] == 0:
        timescale, track_duration = struct.unpack_from(">II", mdhd, 12)
    elif mdhd[0] == 1:
        timescale, track_duration = struct.unpack_from(">IQ", mdhd, 20)
    else:
        raise ValueError("Unsupported mdhd version")
    if timescale <= 0 or track_duration <= 0:
        raise ValueError("Invalid media timescale or duration")
    stbl = one(one(mdia, b"minf"), b"stbl")
    stsd = one(stbl, b"stsd")
    descriptions = list(boxes(stsd[8:]))
    if len(descriptions) != 1:
        raise ValueError("Multiple video sample descriptions require inspection")
    codec, description = descriptions[0]
    width, height = struct.unpack_from(">HH", description, 24)
    if not width or not height:
        raise ValueError("Invalid video dimensions")
    stts = rle_table(one(stbl, b"stts"))
    ctts_box = optional(stbl, b"ctts")
    # FFmpeg mov_read_ctts reads offsets into signed int32 for both versions.
    # Some source files store negative offsets even with a version-0 header.
    # https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/mov.c
    ctts = rle_table(ctts_box, signed=True) if ctts_box is not None else None
    try:
        timing = presentation_timing(stts, ctts)
    except ValueError as exc:
        if str(exc) != "ctts/stts sample count mismatch":
            raise
        # Preserve these videos in the complete census. A malformed timing
        # table is a measured category, not evidence of constant frame rate.
        timing = dict(timing_class="invalid_timing", timing_error=str(exc),
                      frame_count=sum(n for n, _ in stts),
                      sample_duration_ticks=sum(n * delta for n, delta in stts),
                      pts_span_ticks=None, pts_interval_min_ticks=None,
                      pts_interval_max_ticks=None, uniform_grid_max_error_ticks=None)
    stsz = optional(stbl, b"stsz")
    encoded_sample_count = struct.unpack_from(">I", stsz, 8)[0] if stsz is not None else timing["frame_count"]
    if encoded_sample_count != timing["frame_count"]:
        previous = timing.get("timing_error", "")
        timing.update(timing_class="invalid_timing",
                      timing_error=(previous + "; stsz/stts sample count mismatch").strip("; "),
                      pts_span_ticks=None, pts_interval_min_ticks=None,
                      pts_interval_max_ticks=None, uniform_grid_max_error_ticks=None)
    edts = optional(track, b"edts")
    edit_count = 0
    leading_empty_edits = 0
    if edts is not None:
        elst = one(edts, b"elst")
        edit_count = struct.unpack_from(">I", elst, 4)[0]
        step = 20 if elst[0] == 1 else 12
        media_times = []
        for i in range(edit_count):
            rate = struct.unpack_from(">hh", elst, 8 + i * step + step - 4)
            if rate != (1, 0):
                raise ValueError("Non-unit edit-list playback rate")
            offset = 8 + i * step + (8 if elst[0] == 1 else 4)
            media_times.append(struct.unpack_from(">q" if elst[0] == 1 else ">i", elst, offset)[0])
        # Leading empty edits only shift all video PTS by a constant. They do
        # not change the media-track sample count or inter-frame intervals.
        if edit_count > 1:
            if not (all(t == -1 for t in media_times[:-1]) and media_times[-1] >= 0):
                raise ValueError("Multiple nonempty timeline edits require separate inspection")
            leading_empty_edits = edit_count - 1
    tkhd = one(track, b"tkhd")
    matrix_offset = 52 if tkhd[0] == 1 else 40
    matrix = struct.unpack_from(">9i", tkhd, matrix_offset)
    return dict(width=width, height=height, codec=codec.decode("ascii"),
                timescale=timescale, track_duration_ticks=track_duration,
                media_duration_seconds=track_duration / timescale,
                duration_seconds=timing["sample_duration_ticks"] / timescale,
                fps=timing["frame_count"] * timescale / timing["sample_duration_ticks"],
                stts_entries=len(stts), ctts_entries=len(ctts or []),
                encoded_sample_count=encoded_sample_count,
                edit_list_entries=edit_count,
                leading_empty_edits=leading_empty_edits,
                rotated=bool(matrix[1] or matrix[3]), **timing)


def probe(handle, size):
    return parse_moov(read_moov(handle, size))
