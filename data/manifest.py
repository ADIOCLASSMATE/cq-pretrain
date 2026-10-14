"""Manifest identity checks and strict matching of captions to cache members."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple


def image_validation_indices(dataset, *, classes_path, per_class=2, seed=424242):
    """Source ImageNet loss subset, including its class and final shuffle order.

    The source groups by str(class_index), sorts official image identities,
    shuffles each class, takes per_class rows, then shuffles the selected rows.
    Resolve selected IDs back to cache positions so cache reordering is safe.
    """
    if not classes_path:
        raise ValueError("Loss validation requires evaluation.image_classes (the source classes.json)")
    if dataset.dataset_split != "val":
        raise ValueError("Image loss validation requires the independent val split")
    rows = json.loads(Path(classes_path).read_text(encoding="utf-8"))["classes"]
    classes = {str(row["synset"]): int(row["class_index"]) for row in rows}
    if len(classes) != len(rows) or len(set(classes.values())) != len(rows):
        raise ValueError("Image class metadata has duplicate identities")
    if set(dataset.synsets.values()) != set(classes):
        raise ValueError("Validation manifest and class metadata have different classes")
    groups = {str(index): [] for index in classes.values()}
    identities = set()
    for img_id, synset in dataset.synsets.items():
        identity = f"val/{Path(dataset.source_paths_full[img_id]).stem}"
        if identity in identities:
            raise ValueError("Validation manifest has duplicate official image identities")
        identities.add(identity)
        groups[str(classes[synset])].append((identity, img_id))
    counts = {len(group) for group in groups.values()}
    if len(counts) != 1 or min(counts) < per_class:
        raise ValueError("Image loss validation requires balanced classes with enough examples")
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        pool = sorted(groups[key])
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    rng.shuffle(selected)
    positions = {int(img_id): index for index, img_id in enumerate(dataset.img_ids)}
    if len(positions) != len(dataset):
        raise ValueError("Validation cache has duplicate image IDs")
    if any(img_id not in positions for _, img_id in selected):
        raise ValueError("Selected ImageNet validation sample is missing from the posterior cache")
    return [positions[img_id] for _, img_id in selected]


def relative_image_path(path: str) -> str:
    parts = Path(path).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


def read_manifest(
    manifest_jsonl: Optional[str]
) -> Tuple[
    Dict[int, str],
    Dict[int, str],
    Dict[int, str],
    Dict[int, str],
]:
    if not manifest_jsonl:
        return {}, {}, {}, {}
    path = Path(manifest_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)
    synsets: Dict[int, str] = {}
    source_paths: Dict[int, str] = {}
    source_paths_full: Dict[int, str] = {}
    manifest_splits: Dict[int, str] = {}
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            img_id = int(row["img_id"])
            if img_id in synsets:
                raise ValueError(f"duplicate img_id={img_id} in manifest {path}")
            synsets[img_id] = str(row.get("synset", ""))
            source_path = row.get("source_path")
            if source_path:
                normalized_source_path = str(source_path)
                source_paths[img_id] = relative_image_path(
                    normalized_source_path
                )
                source_paths_full[img_id] = normalized_source_path
            split = str(row.get("split", "")).strip().lower()
            if not split and source_path:
                path_parts = {
                    part.lower() for part in Path(str(source_path)).parts
                }
                if "train" in path_parts:
                    split = "train"
                elif "val" in path_parts:
                    split = "val"
            if split not in {"train", "val"}:
                raise ValueError(
                    f"{path}:{line_number} has no unambiguous train/val split"
                )
            manifest_splits[img_id] = split
    return synsets, source_paths, source_paths_full, manifest_splits


def read_synset_names(
    synset_mapping_path: Optional[str]
) -> Dict[str, Tuple[str, str]]:
    if not synset_mapping_path:
        return {}
    path = Path(synset_mapping_path)
    if not path.exists():
        raise FileNotFoundError(path)
    names: Dict[str, Tuple[str, str]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            synset, _, raw_names = line.partition(" ")
            class_names = raw_names.strip()
            class_name = (
                class_names.split(",", 1)[0].strip() if class_names else synset
            )
            names[synset] = (class_name, class_names or class_name)
    return names


class CaptionManifest:
    """Resolve all caption identities, rejecting missing or conflicting members."""

    def __init__(self, *, img_ids, source_paths, text_key, list_key, list_text_key,
                 path_key, id_key, include_original):
        self.img_ids = img_ids
        self.source_paths = source_paths
        self.caption_text_key = text_key
        self.caption_list_key = list_key
        self.caption_list_text_key = list_text_key
        self.caption_path_key = path_key
        self.caption_id_key = id_key
        self.caption_include_original = include_original

    def paths(self, caption_jsonl: str | Sequence[str]) -> List[Path]:
        if isinstance(caption_jsonl, (str, Path)):
            return [Path(caption_jsonl)]
        return [Path(path) for path in caption_jsonl]

    def captions_from_row(
        self, row: Dict, path: Path, line_number: int
    ) -> Tuple[str, ...]:
        captions: List[str] = []
        if self.caption_list_key and self.caption_list_key in row:
            raw_captions = row[self.caption_list_key]
            if not isinstance(raw_captions, list):
                raise ValueError(
                    f"{path}:{line_number} field {self.caption_list_key!r} "
                    "must be a list"
                )
            for item in raw_captions:
                if isinstance(item, str):
                    caption = item.strip()
                elif isinstance(item, dict):
                    if not self.caption_include_original:
                        source = str(item.get("source", "")).strip().lower()
                        caption_slot = item.get("caption_slot", None)
                        is_original_slot = False
                        if caption_slot is not None:
                            try:
                                is_original_slot = int(caption_slot) < 0
                            except (TypeError, ValueError) as exc:
                                raise ValueError(
                                    f"{path}:{line_number} has invalid "
                                    f"caption_slot={caption_slot!r}"
                                ) from exc
                        if source == "original" or is_original_slot:
                            continue
                    caption = str(item.get(self.caption_list_text_key, "")).strip()
                else:
                    raise ValueError(
                        f"{path}:{line_number} contains a non-string/non-object "
                        f"caption entry: {type(item).__name__}"
                    )
                if caption:
                    captions.append(caption)
        else:
            caption = str(row.get(self.caption_text_key, "")).strip()
            if caption:
                captions.append(caption)

        if not captions:
            return ()
        if len(set(captions)) != len(captions):
            raise ValueError(f"{path}:{line_number} contains duplicate captions")
        return tuple(captions)

    def _load_caption_rows(
        self,
        caption_jsonl: str | Sequence[str],
    ) -> Tuple[
        Dict[int, Tuple[str, ...]],
        Dict[str, Tuple[str, ...]],
        Dict[str, Tuple[str, ...]],
    ]:
        captions_by_img_id: Dict[int, Tuple[str, ...]] = {}
        captions_by_path: Dict[str, Tuple[str, ...]] = {}
        captions_by_id: Dict[str, Tuple[str, ...]] = {}
        for path in self.paths(caption_jsonl):
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open() as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    captions = self.captions_from_row(row, path, line_number)
                    if not captions:
                        continue
                    row_path = row.get(self.caption_path_key)
                    row_id = row.get(self.caption_id_key)
                    row_img_id = row.get("img_id")
                    if row_img_id is not None:
                        normalized_img_id = int(row_img_id)
                        if normalized_img_id in captions_by_img_id:
                            raise ValueError(
                                f"duplicate caption img_id={normalized_img_id} "
                                f"in {path}"
                            )
                        expected_path = self.source_paths.get(normalized_img_id)
                        if expected_path is None:
                            # Caption manifests may be supersets of a cache;
                            # ignore non-members without retaining their text.
                            continue
                        if (
                            row_path
                            and relative_image_path(str(row_path))
                            != expected_path
                        ):
                            raise ValueError(
                                f"caption img_id/path mismatch for "
                                f"img_id={normalized_img_id} in {path}"
                            )
                        if row_id and str(row_id) != Path(expected_path).stem:
                            raise ValueError(
                                f"caption img_id/id mismatch for "
                                f"img_id={normalized_img_id} in {path}"
                            )
                        captions_by_img_id[normalized_img_id] = captions
                        continue
                    if row_path:
                        relative_path = relative_image_path(str(row_path))
                        if relative_path in captions_by_path:
                            raise ValueError(
                                f"duplicate caption path={relative_path!r} in {path}"
                            )
                        captions_by_path[relative_path] = captions
                    if row_id:
                        normalized_id = str(row_id)
                        if normalized_id in captions_by_id:
                            raise ValueError(
                                f"duplicate caption id={normalized_id!r} in {path}"
                            )
                        captions_by_id[normalized_id] = captions
        return captions_by_img_id, captions_by_path, captions_by_id

    def load(
        self,
        caption_jsonl: str | Sequence[str],
    ) -> Dict[int, Tuple[str, ...]]:
        captions_by_img_id, captions_by_path, captions_by_id = self._load_caption_rows(
            caption_jsonl
        )
        captions: Dict[int, Tuple[str, ...]] = {}
        missing: List[str] = []
        for img_id_tensor in self.img_ids:
            img_id = int(img_id_tensor.item())
            rel_path = self.source_paths.get(img_id)
            caption_by_img_id = captions_by_img_id.get(img_id)
            caption_by_path = captions_by_path.get(rel_path or "")
            caption_by_id = (
                captions_by_id.get(Path(rel_path).stem) if rel_path else None
            )
            candidates = [
                candidate
                for candidate in (
                    caption_by_img_id,
                    caption_by_path,
                    caption_by_id,
                )
                if candidate is not None
            ]
            if any(candidate != candidates[0] for candidate in candidates[1:]):
                raise ValueError(
                    f"caption img_id/path/id conflict for img_id={img_id}, "
                    f"path={rel_path!r}"
                )
            caption = candidates[0] if candidates else None
            if caption is None:
                if len(missing) < 5:
                    missing.append(rel_path or str(img_id))
                continue
            captions[img_id] = caption

        missing_count = int(self.img_ids.numel()) - len(captions)
        if missing_count:
            raise ValueError(
                f"Missing {missing_count} captions for cache img_ids. "
                f"Examples: {missing}. Use a caption manifest matching the cache."
            )
        return captions

