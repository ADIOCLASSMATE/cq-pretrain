"""Shared fixed/native video preparation for online inputs and offline caches."""

from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .sequence import native_video_metadata


def decode_video(video):
    path = Path(video.get("frames_path") or video.get("path") or "")
    if not path.is_file():
        raise FileNotFoundError(f"Raw video required but unavailable: {path}")
    if path.suffix == ".npy":
        frames = np.load(path, allow_pickle=False)
        fps = video.get("fps")
        if fps is None or float(fps) <= 0:
            raise ValueError("NPY frames require explicit positive source fps")
        times = np.arange(len(frames), dtype=np.float64) / float(fps)
    else:
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "Raw video decoding requires PyAV: install the video extra"
            ) from exc
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            decoded = [
                (frame.to_ndarray(format="rgb24"), float(frame.time))
                for frame in container.decode(stream)
            ]
        if not decoded:
            raise ValueError(f"Empty video: {path}")
        frames = np.stack([x[0] for x in decoded])
        times = np.array([x[1] for x in decoded])
        times -= times[0]
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Decoded frames must be uint8 [T,H,W,3]")
    if (
        len(times) < 2
        or not np.isfinite(times).all()
        or not bool((np.diff(times) > 0).all())
    ):
        raise ValueError("A video needs at least two frames with increasing PTS")
    return frames, times, hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_raw_video(video, settings):
    frames, times, source_hash = decode_video(video)
    n, fps = int(settings["num_frames"]), float(settings["fps"])
    start = float(video.get("start_seconds", 0))
    wanted = start + np.arange(n) / fps
    if wanted[0] < times[0] - 1e-6 or wanted[-1] > times[-1] + 1e-5:
        raise ValueError(
            "Requested clip exceeds source timestamps; choose an explicit shorter clip/frame configuration"
        )
    indices = np.abs(times[:, None] - wanted[None]).argmin(0)
    rgb = torch.from_numpy(frames[indices].copy()).permute(0, 3, 1, 2).float()
    h, w = int(settings["height"]), int(settings["width"])
    scale = max(h / rgb.shape[2], w / rgb.shape[3])
    size = (max(h, round(rgb.shape[2] * scale)), max(w, round(rgb.shape[3] * scale)))
    rgb = F.interpolate(
        rgb, size=size, mode="bicubic", align_corners=False, antialias=True
    ).clamp(0, 255)
    y, x = (size[0] - h) // 2, (size[1] - w) // 2
    rgb = rgb[:, :, y : y + h, x : x + w].round().to(torch.uint8)
    clip = dict(
        source_sha256=source_hash,
        frame_indices=indices.tolist(),
        timestamps=times[indices].tolist(),
        num_frames=n,
        fps=fps,
        height=h,
        width=w,
        spatial_transform="bicubic_resize_center_crop",
    )
    return rgb, clip


def qwen_video_inputs(rgb, clip, processor, settings):
    count = settings.get("qwen_num_frames") or len(rgb)
    indices = np.linspace(0, len(rgb) - 1, int(count)).round().astype(int)
    timestamps = [clip["timestamps"][i] for i in indices]
    kwargs = dict(settings.get("qwen_processor", {}))
    if set(kwargs) & {"do_sample_frames", "fps", "num_frames", "video_metadata"}:
        raise ValueError(
            "Qwen sampling is owned by qwen_num_frames; do not resample inside Processor"
        )
    result = processor.video_processor(
        videos=[rgb[indices]],
        do_sample_frames=False,
        video_metadata=[native_video_metadata(timestamps)],
        return_metadata=True,
        return_tensors="pt",
        **kwargs,
    )
    return result["pixel_values_videos"], result["video_grid_thw"][0].long(), timestamps


def prepare_native_video(video, *, expected_num_frames=None):
    """Keep decoded pixels/PTS; trim only the tail to a shared Wan 4k+1 clip."""
    if video.get("start_seconds", 0) != 0:
        raise ValueError(
            "Native preparation uses the whole video; start_seconds must be 0"
        )
    frames, times, source_hash = decode_video(video)
    if expected_num_frames is not None and len(frames) != expected_num_frames:
        raise ValueError(
            f"Decoded frame count {len(frames)} differs from expected {expected_num_frames}"
        )
    if expected_num_frames is not None and (expected_num_frames - 1) % 4:
        raise ValueError("An exact native clip must already have 4k+1 frames")
    if len(frames) < 5:
        raise ValueError("Native Wan preparation requires at least 5 source frames")
    _, h, w, _ = frames.shape
    if h % 16 or w % 16:
        raise ValueError(
            "Native Wan H/W must be divisible by 16; no resize is performed"
        )
    path = Path(video.get("frames_path") or video["path"])
    if path.suffix == ".npy":
        fps = float(video["fps"])
    else:
        import av

        # decode_video preserves PTS; read the container rate without decoding again.
        with av.open(str(path)) as container:
            rate = container.streams.video[0].average_rate
        if rate is None or rate <= 0:
            raise ValueError(
                f"Native video requires a positive container frame rate: {path}"
            )
        fps = float(rate)
    n = (len(frames) - 1) // 4 * 4 + 1
    rgb = torch.from_numpy(frames[:n]).permute(0, 3, 1, 2)
    clip = dict(
        source_sha256=source_hash,
        source_num_frames=len(frames),
        frame_indices=list(range(n)),
        timestamps=times[:n].tolist(),
        num_frames=n,
        fps=fps,
        height=h,
        width=w,
        spatial_transform="identity",
    )
    return rgb, clip


def native_qwen_video_inputs(rgb, clip, processor, settings):
    """Use Qwen's sampler once, then patchify at the original spatial resolution."""
    from transformers.video_utils import VideoMetadata

    video_processor = processor.video_processor
    kwargs = dict(settings.get("qwen_processor", {}))
    # The sampler reads these attributes rather than call-time kwargs. Restore
    # them afterwards so a shared Processor cannot leak settings across datasets.
    limits = {
        key: kwargs.pop(key) for key in ("min_frames", "max_frames") if key in kwargs
    }
    count = settings.get("qwen_num_frames")
    if count is not None and count > len(rgb):
        raise ValueError(
            "qwen_num_frames exceeds the aligned clip; refusing repeated sampling"
        )
    fps = kwargs.pop("fps", video_processor.fps if count is None else None)
    factor = video_processor.patch_size * video_processor.merge_size
    if any(size % factor for size in rgb.shape[2:]):
        raise ValueError(
            f"Native Qwen H/W must be divisible by {factor}; no resize is performed"
        )
    previous = {key: getattr(video_processor, key) for key in limits}
    try:
        for key, value in limits.items():
            setattr(video_processor, key, value)
        indices = video_processor.sample_frames(
            VideoMetadata(total_num_frames=len(rgb), fps=clip["fps"]),
            num_frames=count,
            fps=fps,
        )
    finally:
        for key, value in previous.items():
            setattr(video_processor, key, value)
    times = [clip["timestamps"][i] for i in indices]
    kwargs["do_resize"] = False
    result = video_processor(
        videos=[rgb[indices]],
        do_sample_frames=False,
        video_metadata=[native_video_metadata(times)],
        return_metadata=True,
        return_tensors="pt",
        **kwargs,
    )
    return result["pixel_values_videos"], result["video_grid_thw"][0].long(), times


def prepare_video(video, settings):
    if settings.get("mode", "fixed") == "native":
        expected = settings.get("expected_num_frames", video.get("num_frames"))
        rgb, clip = prepare_native_video(video, expected_num_frames=expected)
        for key in ("height", "width"):
            if video.get(key) is not None and clip[key] != video[key]:
                raise ValueError(f"Decoded {key} differs from manifest")
        if video.get("fps_num") is not None and video.get("fps_den") is not None:
            expected_fps = video["fps_num"] / video["fps_den"]
            if not np.isclose(clip["fps"], expected_fps, rtol=1e-10, atol=1e-10):
                raise ValueError("Container FPS differs from manifest")
        return rgb, clip
    return prepare_raw_video(video, settings)


def prepare_qwen_inputs(rgb, clip, processor, settings):
    function = (
        native_qwen_video_inputs
        if settings.get("mode", "fixed") == "native"
        else qwen_video_inputs
    )
    return function(rgb, clip, processor, settings)
