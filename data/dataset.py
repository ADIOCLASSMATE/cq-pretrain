"""One-video T2V/V2T records, shared raw preprocessing and cache resolution."""

from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .cache import cache_identities, load_cache
from .sequence import make_video_sequence, native_video_metadata


def read_records(source, *, prompt_only=False):
    path = Path(source["path"]).expanduser().resolve()
    if path.suffix == ".jsonl":
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    elif path.suffix == ".json":
        value = json.loads(path.read_text())
        records = value if isinstance(value, list) else [value]
    else:
        raise ValueError("Dataset paths must identify a JSONL or JSON manifest")
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    result = []
    for index, row in enumerate(records):
        row = deepcopy(row)
        task = row.get("task", source.get("task"))
        if task not in {"t2v", "v2t"}:
            raise ValueError(
                f"{path}:{index + 1}: specify task=t2v/v2t in the row or source"
            )
        if source.get("task") and row.get("task", source["task"]) != source["task"]:
            raise ValueError("Record task conflicts with source task")
        videos = row.get("videos")
        if videos is None:
            videos = [
                {
                    "id": "v0",
                    "path": row.get("video_path"),
                    "frames_path": row.get("frames_path"),
                    "fps": row.get("fps"),
                    "qwen_cache": row.get("qwen_cache"),
                    "wan_cache": row.get("wan_cache"),
                }
            ]
        if len(videos) != 1:
            raise ValueError("Stage one supports exactly one video per sample")
        video = videos[0]
        video.setdefault("id", "v0")
        for key in ("path", "frames_path", "qwen_cache", "wan_cache"):
            if video.get(key):
                video[key] = str((path.parent / video[key]).resolve())
        if "messages" not in row:
            caption = row.get(source.get("caption_field", "caption"))
            has_caption = isinstance(caption, str) and bool(caption.strip())
            if not has_caption and not (prompt_only and task == "v2t"):
                raise ValueError("Simple records require a nonempty caption")
            visual = {"type": "video", "video_id": video["id"]}
            text = lambda value: {"type": "text", "text": value}
            if task == "t2v":
                row["messages"] = [
                    {"role": "user", "content": [text(caption)]},
                    {"role": "assistant", "content": [visual]},
                ]
            else:
                instruction = row.get("instruction", source.get("instruction", "Describe this video."))
                row["messages"] = [
                    {"role": "user", "content": [visual, text(instruction)]},
                ]
                if has_caption:
                    row["messages"].append(
                        {"role": "assistant", "content": [text(caption)]}
                    )
        if not row["messages"]:
            raise ValueError("A record requires at least one message")
        occurrences = []
        for message in row["messages"]:
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            for item in message["content"]:
                if item["type"] == "video":
                    if item.get("video_id", video["id"]) != video["id"]:
                        raise ValueError("Message video_id does not match videos[0]")
                    occurrences.append(message["role"])
                elif item["type"] != "text":
                    raise ValueError(
                        "Stage one messages contain text and one video only"
                    )
        if occurrences != (["assistant"] if task == "t2v" else ["user"]):
            raise ValueError(
                "T2V target must be in assistant; V2T input must be in user"
            )
        if row["messages"][-1]["role"] != "assistant" and not (prompt_only and task == "v2t"):
            raise ValueError("Training records must end with an assistant target")
        row.update(
            task=task,
            videos=videos,
            sample_id=str(row.get("sample_id", f"{path.name}:{index}")),
            source_name=source.get("name", path.stem),
        )
        result.append(row)
    return result


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


class VideoDataset(Dataset):
    def __init__(self, source, processor, config, *, records=None, prompt_only=False):
        self.records = read_records(source, prompt_only=prompt_only) if records is None else records
        self.processor, self.config, self.prompt_only = processor, config, prompt_only
        self.settings = deepcopy(config["data"]["video"])
        self.cache_settings = config["data"]["representations"]
        self.identities = cache_identities(config)
        if (
            not config["model"]["freeze_vision"]
            and self.cache_settings["qwen"] != "online"
        ):
            raise ValueError("Unfrozen ViT requires data.representations.qwen=online")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        video = row["videos"][0]
        caches = {}
        for kind in ("qwen", "wan"):
            if kind == "wan" and (row["task"] != "t2v" or self.prompt_only):
                continue
            mode = self.cache_settings[kind]
            path = video.get(f"{kind}_cache")
            if mode == "cache" and not path:
                raise ValueError(f"{row['sample_id']}: {kind} cache required")
            if mode != "online" and path:
                caches[kind] = load_cache(
                    path,
                    kind,
                    checkpoint=self.identities.get(kind),
                    preprocessing=self.settings,
                )
        raw = clip = None
        need_wan = row["task"] == "t2v" and not self.prompt_only
        if "qwen" not in caches or (need_wan and "wan" not in caches):
            raw, clip = prepare_raw_video(video, self.settings)
        for cached in caches.values():
            if clip is not None and cached["metadata"]["clip"] != clip:
                raise ValueError(
                    f"{row['sample_id']}: Qwen/Wan inputs describe different clips"
                )
            clip = cached["metadata"]["clip"]
        if "qwen" in caches:
            entry = caches["qwen"]
            pixels, grid, times, features = (
                None,
                entry["grid_thw"],
                entry["metadata"]["qwen_timestamps"],
                entry["features"],
            )
        else:
            pixels, grid, times = qwen_video_inputs(
                raw, clip, self.processor, self.settings
            )
            features = None
        sequence = make_video_sequence(
            row["messages"],
            self.processor,
            grid,
            times,
            max_seq_length=self.config["data"]["max_seq_length"],
            prompt_only=self.prompt_only,
        )
        result = dict(
            sequence,
            task=row["task"],
            sample_id=row["sample_id"],
            source_name=row["source_name"],
            video_features=features,
            pixel_values_videos=pixels,
            video_grid_thw=grid,
            clip=clip,
        )
        if need_wan:
            result["wan_latent"] = caches["wan"]["latent"] if "wan" in caches else None
            result["wan_rgb"] = (
                None
                if "wan" in caches
                else raw.permute(1, 0, 2, 3).float().div(127.5).sub(1)
            )
        return result
