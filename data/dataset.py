"""One-video T2V/V2T records, shared raw preprocessing and cache resolution."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .cache import cache_identities, load_cache, encoder_contracts
from .manifest import read_manifest, resolve_video, source_profiles
from .sequence import make_video_sequence
from .video import decode_video, prepare_raw_video, qwen_video_inputs, prepare_video, prepare_qwen_inputs


def read_records(source, *, prompt_only=False):
    path = Path(source["path"]).expanduser().resolve()
    records = read_manifest(path)
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
        video = resolve_video(row, path)
        videos = [video]
        if row.get("latent_profile"):
            row["latent_profile"] = str((path.parent / row["latent_profile"]).resolve())
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


class VideoDataset(Dataset):
    def __init__(self, source, processor, config, *, records=None, prompt_only=False):
        self.records = read_records(source, prompt_only=prompt_only) if records is None else records
        self.processor, self.config, self.prompt_only = processor, config, prompt_only
        self.settings = deepcopy(config["data"]["video"])
        self.cache_settings = config["data"]["representations"]
        self.profiles = source_profiles(source, self.records)
        self.contracts = encoder_contracts(config) if self.profiles else {}
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
                profile = self.profiles.get(row.get("latent_profile"))
                caches[kind] = load_cache(
                    path,
                    kind,
                    checkpoint=None if profile else self.identities.get(kind),
                    contract=self.contracts.get(kind) if profile else None,
                    sample_id=row["sample_id"] if profile else None,
                    info=row.get("latent_info", {}).get(kind) if profile else None,
                    preprocessing=self.settings,
                )
        raw = clip = None
        need_wan = row["task"] == "t2v" and not self.prompt_only
        if "qwen" not in caches or (need_wan and "wan" not in caches):
            raw, clip = prepare_video(video, self.settings)
        for cached in caches.values():
            if clip is not None and cached["metadata"]["clip"] != clip:
                raise ValueError(
                    f"{row['sample_id']}: Qwen/Wan inputs describe different clips"
                )
            clip = cached["metadata"]["clip"]
        if self.settings.get("mode") == "native":
            for key in ("num_frames", "height", "width"):
                if video.get(key) is not None and clip[key] != video[key]:
                    raise ValueError(f"{row['sample_id']}: cached clip {key} differs from manifest")
        if "qwen" in caches:
            entry = caches["qwen"]
            pixels, grid, times, features = (
                None,
                entry["grid_thw"],
                entry["metadata"]["qwen_timestamps"],
                entry["features"],
            )
        else:
            pixels, grid, times = prepare_qwen_inputs(
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
