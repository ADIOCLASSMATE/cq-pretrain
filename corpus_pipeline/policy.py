"""Ordered record normalization and conservative, metadata-based provenance policy.

No image decoding, model inference, network access, or source mutation occurs here.
An accepted image has positive provenance evidence; absence of an AI flag is not
evidence that an image is natural. Source and per-image evidence are both retained.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "corpus-index.v1"
CATEGORIES = ("single_t2i", "single_i2t", "multi_t2i", "multi_i2t",
              "interleaved", "pure_text", "mixed_multimodal", "unclassified")
T2I_CATEGORIES = frozenset({"single_t2i", "multi_t2i"})
NON_NATURAL = {"ai_generated", "ai_edited", "programmatic", "rendered"}
ROLE_MAP = {"human": "user", "gpt": "assistant", "bot": "assistant",
            "user": "user", "assistant": "assistant", "system": "system",
            "developer": "developer", "tool": "tool"}
IMAGE_ALIASES = {"real": "natural", "photograph": "natural", "photographic": "natural",
                 "ai": "ai_generated", "ai-generated": "ai_generated",
                 "ai-edited": "ai_edited", "computer_generated": "rendered",
                 "programmatically_generated": "programmatic"}
TEXT_ALIASES = {"ai_generated": "ai", "generated": "ai", "synthetic": "ai",
                "human_written": "human", "natural": "human"}
IMAGE_KEYS = ("image_origin", "image_provenance", "image_generation_status")
TEXT_KEYS = ("caption_origin", "caption_provenance", "text_provenance", "text_origin")
EDIT_SOURCE_KEYS = ("edit_source_origin", "source_image_origin", "original_image_origin", "edit_source_provenance")
PLACEHOLDER = re.compile(r"<image(?:_(\d+))?>")
_SOURCE_CACHE: dict[int, tuple[Any, dict]] = {}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def _status(value: Any, text: bool = False) -> str:
    if isinstance(value, dict):
        value = value.get("status", value.get("origin", "unknown"))
    if not isinstance(value, str):
        return "unknown"
    value = value.lower().strip().replace(" ", "_")
    aliases = TEXT_ALIASES if text else IMAGE_ALIASES
    value = aliases.get(value, value)
    allowed = {"ai", "human", "mixed", "unknown"} if text else NON_NATURAL | {"natural", "mixed", "unknown"}
    return value if value in allowed else "unknown"


def _paths(locator: Any) -> list[str]:
    if isinstance(locator, str):
        return [locator.replace("\\", "/")]
    if isinstance(locator, (list, tuple)):
        return [s for item in locator for s in _paths(item)]
    if isinstance(locator, dict):
        keys = ("path", "member", "source_shard", "relative_path", "paths")
        found = [s for key in keys for s in _paths(locator.get(key))]
        if locator.get("path") and locator.get("member"):
            found.append(str(locator["path"]).rstrip("/") + "/" + str(locator["member"]))
        return found
    return []


def _matches(entry: dict, collection: str, source: str, paths: list[str]) -> bool:
    if entry.get("inventory_only"):
        return False
    match = entry.get("match") or {}
    if not match:
        match = {"collection": entry.get("collection", entry.get("corpus")),
                 "source": entry.get("source", entry.get("source_id"))}
    known_keys = {"collection", "corpus", "source", "source_id", "source_glob", "path_glob", "path_prefix"}
    if set(match) - known_keys:
        raise ValueError("Unsupported source match keys: " + ",".join(sorted(set(match) - known_keys)))
    corpus_match = match.get("collection", match.get("corpus"))
    name_match = match.get("source", match.get("source_id"))
    if corpus_match and str(corpus_match).casefold() != str(collection).casefold():
        return False
    if name_match and str(name_match).casefold() != str(source).casefold():
        return False
    if match.get("source_glob") and not fnmatch.fnmatchcase(source.casefold(), match["source_glob"].casefold()):
        return False
    if match.get("path_glob") and not any(fnmatch.fnmatchcase(p.casefold(), match["path_glob"].casefold()) for p in paths):
        return False
    if match.get("path_prefix") and not any(p.casefold().startswith(match["path_prefix"].casefold()) for p in paths):
        return False
    return True


def _specificity(entry: dict) -> tuple[int, int]:
    match = entry.get("match") or {}
    pattern = str(match.get("path_prefix", match.get("path_glob", "")))
    return (bool(pattern), len(pattern.replace("*", "").replace("?", "")))


def resolve_source_config(collection: str, source: str, locator: Any,
                          rules: dict | None = None, registry: Any = None) -> dict:
    """Resolve source priors; callers must resolve again for each image locator.

    Specific path rules override broad unknown/mixed priors. All matching known
    negative priors remain visible in ``_matched_rules`` and win in evaluation.
    """
    entries = registry if registry is not None else (rules or {}).get("sources", [])
    if isinstance(entries, dict):
        entries = entries.get("sources", [])
    # Registry snapshots are immutable during a run. Cache source-level candidates
    # so hundreds of rules are not scanned for every one of 100M+ records.
    cache_key = id(entries)
    cached = _SOURCE_CACHE.get(cache_key)
    if cached is None or cached[0] is not entries:
        if len(_SOURCE_CACHE) >= 8:
            _SOURCE_CACHE.clear()
        cached = (entries, {})
        _SOURCE_CACHE[cache_key] = cached
    source_key = (collection.casefold(), source.casefold())
    candidates = cached[1].get(source_key)
    if candidates is None:
        candidates = []
        for entry in entries:
            match = entry.get("match") or {}
            source_match = {key: value for key, value in match.items() if key not in ("path_glob", "path_prefix")}
            if _matches({**entry, "match": source_match}, collection, source, []):
                candidates.append(entry)
        cached[1][source_key] = candidates
    paths = _paths(locator)
    matched = [dict(e) for e in candidates if _matches(e, collection, source, paths)]
    matched.sort(key=_specificity)
    result = dict(matched[-1]) if matched else {}
    result["_matched_rules"] = matched
    return result


def _safe_metadata(raw: dict) -> dict:
    result = {}
    selected = set(IMAGE_KEYS + TEXT_KEYS + EDIT_SOURCE_KEYS) | {
        "is_ai_generated", "is_generated", "ai_generated", "is_ai_edited",
        "is_caption_ai_generated", "caption_is_synthetic", "caption_generated_by",
        "caption_model", "direction", "task", "task_type", "usage", "split",
        "exclude_reason", "source_shard", "is_synthetic_image", "image_is_synthetic",
        "processing_policy", "image_loss_allowed", "dcvlm_image_loss_allowed"}
    for key in selected:
        if key in raw and isinstance(raw[key], (str, bool, int, float, dict, list, type(None))):
            result[key] = raw[key]
    if isinstance(raw.get("metadata"), dict):
        result["metadata"] = _safe_metadata(raw["metadata"])
    return result


def _ref(item: Any, envelope: dict, position: int, source_config: dict) -> dict:
    if isinstance(item, str):
        item = {"kind": "url" if item.startswith(("https://", "http://")) else "file", "path": item}
    elif isinstance(item, (bytes, bytearray, memoryview)):
        item = {"kind": "embedded", "field": "images", "index": position, "exists": len(item) > 0}
    elif isinstance(item, dict):
        original = item
        item = {k: v for k, v in item.items() if k not in ("bytes", "data")}
        if isinstance(original.get("bytes"), (bytes, bytearray, memoryview)):
            item.update(kind="embedded", exists=len(original["bytes"]) > 0,
                        field=f"images[{position}].bytes")
        elif item.get("url") and not item.get("path"):
            item.update(kind="url", path=item["url"])
    else:
        return {"kind": "unknown", "index": position, "exists": False}
    item = dict(item)
    if item.get("path") is not None and not isinstance(item["path"], str):
        return {"kind": "unknown", "index": position, "exists": False,
                "invalid_path_type": type(item["path"]).__name__}
    if not item.get("kind"):
        item["kind"] = "tar" if item.get("member") is not None else "file" if item.get("path") else "unknown"
    if item["kind"] == "tar" and not item.get("path"):
        raw = envelope.get("raw", {})
        shard = raw.get("source_shard", envelope.get("source_shard"))
        root = envelope.get("source_pool", source_config.get("source_pool"))
        if shard:
            item["path"] = os.path.join(str(root), shard) if root else shard
    if item.get("path") and item["kind"] in ("file", "tar"):
        metadata = json.loads(item.get("metadata_json") or "{}")
        if "dataset_id" in item or "dataset_id" in metadata:
            from corpus_pipeline.production.portable_refs import logical_ref, resolve_ref
            item = logical_ref(item)
            # Runtime locations must never replace the portable source identity.
            if "exists" not in item:
                item["exists"] = resolve_ref(item).is_file()
        elif not os.path.isabs(item["path"]):
            base = envelope.get("source_root", source_config.get("source_root"))
            base = base or str(Path(envelope.get("locator", {}).get("path", ".")).parent)
            item["path"] = os.path.normpath(os.path.join(base, item["path"]))
        if "exists" not in item:
            item["exists"] = Path(item["path"]).is_file()
    item["index"] = position
    return item


def normalize(envelope: dict, source_config: dict | None = None) -> dict:
    """Normalize common DCVLM/gen-und structures without inventing supervision."""
    source_config = source_config or envelope.get("source_config") or {}
    raw = envelope.get("raw")
    issues = list(envelope.get("issues") or [])
    if not isinstance(raw, dict):
        raw = {}
        issues.append("raw_record_not_object")
    collection, source = str(envelope.get("collection", "")), str(envelope.get("source", ""))
    locator = dict(envelope.get("locator") or {})
    refs = envelope.get("image_refs")
    explicit_image_keys = ("images", "image", "image_path", "image_url", "img")
    if refs is None or (not refs and any(raw.get(key) for key in explicit_image_keys)):
        refs = next((raw[key] for key in explicit_image_keys if raw.get(key) is not None), [])
    if not refs and isinstance(raw.get("url"), str) and raw["url"].startswith(("http://", "https://")):
        if isinstance(raw.get("caption"), str) or ("width" in raw and "height" in raw):
            # Common LAION-like URL-only records are not pure-text documents.
            refs = [{"kind": "url", "path": raw["url"], "exists": False}]
    if refs is None:
        refs = []
    if not isinstance(refs, list):
        refs = [refs]
    refs = [item for item in refs if item is not None]
    if (not refs and any(key in raw for key in explicit_image_keys)
            and (raw.get("format") == "caption" or "caption" in raw or "prompt" in raw)):
        issues.append("declared_image_missing")
    images = [_ref(item, envelope, i, source_config) for i, item in enumerate(refs)]
    raw_images = raw.get("images", raw.get("image", []))
    raw_images = raw_images if isinstance(raw_images, list) else [raw_images]
    raw_images = [item for item in raw_images if item is not None]
    if len(raw_images) == len(images):
        for image, original in zip(images, raw_images):
            if isinstance(original, dict):
                # Adapters resolve a locator but must not erase image-local flags.
                image["metadata"] = {**_safe_metadata(original), **image.get("metadata", {})}
    segments, notes = [], []
    fmt = str(raw.get("format", ""))
    kind = "unknown"
    image_cursor = 0

    def normalize_role(value, required=False):
        if value is None and not required:
            return None
        if not isinstance(value, str) or value not in ROLE_MAP:
            issues.append("unsupported_message_role")
            return value if isinstance(value, str) else None
        return ROLE_MAP[value]

    def add_text(text: Any, role=None, turn=None):
        if not isinstance(text, str):
            if text is not None:
                issues.append("non_string_text")
            return
        segments.append({"kind": "text", "text": text, "role": role, "turn": turn})

    def add_image(index: Any, role=None, turn=None):
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(images):
            issues.append("unresolved_image_index")
        segments.append({"kind": "image", "image_index": index, "role": role, "turn": turn})

    def text_with_images(text: Any, role=None, turn=None):
        nonlocal image_cursor
        if not isinstance(text, str):
            add_text(text, role, turn)
            return
        end = 0
        for match in PLACEHOLDER.finditer(text):
            if match.start() > end:
                add_text(text[end:match.start()], role, turn)
            index = int(match.group(1)) if match.group(1) else image_cursor
            add_image(index, role, turn)
            image_cursor = max(image_cursor, index + 1)
            end = match.end()
        if end < len(text) or not text:
            add_text(text[end:], role, turn)

    native = raw.get("segments")
    messages = raw.get("messages", raw.get("conversations"))
    if isinstance(native, list):
        kind = "native_segments"
        for item in native:
            if not isinstance(item, dict):
                issues.append("invalid_segment")
                continue
            role = normalize_role(item.get("role"))
            if item.get("kind", item.get("type")) == "image":
                add_image(item.get("image_index"), role, item.get("turn"))
            elif item.get("kind", item.get("type")) == "text":
                add_text(item.get("text"), role, item.get("turn"))
            else:
                issues.append("unsupported_segment_kind")
    elif isinstance(messages, list):
        kind = "conversation"
        fmt = fmt or "conversation"
        for turn, msg in enumerate(messages):
            if not isinstance(msg, dict):
                issues.append("invalid_message")
                continue
            role_value = msg.get("role", msg.get("from"))
            role = normalize_role(role_value, required=True)
            content = msg.get("content", msg.get("value"))
            if isinstance(content, str):
                text_with_images(content, role, turn)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        issues.append("unsupported_message_content")
                        continue
                    part_kind = part.get("type", part.get("kind"))
                    if part_kind in ("text", "input_text", "output_text"):
                        text_with_images(part.get("text", ""), role, turn)
                    elif part_kind in ("image", "image_url", "input_image", "output_image"):
                        index = part.get("image_index", image_cursor)
                        if not isinstance(index, int) or isinstance(index, bool):
                            add_image(index, role, turn)
                            continue
                        if index >= len(images):
                            image_value = part.get("image_url", part.get("url", part.get("image", part.get("path"))))
                            if isinstance(image_value, dict):
                                image_value = image_value.get("url", image_value.get("path"))
                            if image_value is not None:
                                images.append(_ref(image_value, envelope, len(images), source_config))
                        add_image(index, role, turn)
                        image_cursor = max(image_cursor, index + 1)
                    else:
                        issues.append("unsupported_message_content")
            else:
                issues.append("unsupported_message_content")
        if images and not any(s["kind"] == "image" for s in segments):
            first_user = next((i for i, s in enumerate(segments) if s.get("role") == "user"), None)
            if len(images) == 1 and first_user is not None:
                segments.insert(first_user, {"kind": "image", "image_index": 0,
                                            "role": "user", "turn": segments[first_user].get("turn")})
                notes.append("single_image_attached_to_first_user_by_convention")
            else:
                issues.append("ambiguous_image_placement")
    elif isinstance(raw.get("texts"), list) and images:
        # Array positions are meaningful only when text and image slots alternate.
        texts, slots = raw["texts"], raw.get("images")
        if isinstance(slots, list) and len(texts) == len(slots):
            kind, fmt = "aligned_interleaved", "interleaved"
            index = 0
            for txt, img in zip(texts, slots):
                if txt and img:
                    issues.append("ambiguous_interleaved_slot_order")
                if txt is not None:
                    add_text(txt)
                if img is not None:
                    add_image(index)
                    index += 1
        else:
            issues.append("unaligned_interleaved_arrays")
    else:
        text_key = next((key for key in ("caption", "recaption", "re_caption", "prompt", "text")
                         if isinstance(raw.get(key), str)), None)
        if text_key:
            kind = "caption_pair" if images else "text"
            text_tasks = {"pure_text", "text", "text_conversation", "text_generation", "language_modeling"}
            declared_tasks = {str(raw.get(key, "")).lower() for key in ("format", "task", "task_type")}
            declared_tasks.update(str(source_config.get(key, "")).lower() for key in ("format", "task", "task_type"))
            if not images and text_key != "text" and not declared_tasks.intersection(text_tasks):
                issues.append("image_annotation_without_image")
            if images:
                fmt = fmt or "caption"
                direction = str(raw.get("direction", raw.get("task", ""))).lower()
                if direction in ("t2i", "text_to_image", "single_t2i") or text_key == "prompt":
                    add_text(raw[text_key])
                    for index in range(len(images)):
                        add_image(index)
                else:
                    for index in range(len(images)):
                        add_image(index)
                    add_text(raw[text_key])
            else:
                add_text(raw[text_key])
        elif images:
            issues.append("image_without_text_annotation")
        else:
            issues.append("unsupported_record_structure")
    referenced = {s.get("image_index") for s in segments if s["kind"] == "image"}
    if images and any(i not in referenced for i in range(len(images))):
        issues.append("unreferenced_images")
    if not any(s["kind"] == "text" and s.get("text", "").strip() for s in segments):
        issues.append("missing_text")
    if fmt == "caption" or kind == "caption_pair":
        caption_parts = [s.get("text", "").strip() for s in segments if s["kind"] == "text"]
        if sum(len(part) for part in caption_parts) + len(caption_parts) <= 8193:
            caption_text = " ".join(caption_parts).strip()
            if re.fullmatch(r"https?://[^\s<>]+", caption_text, re.IGNORECASE):
                issues.append("text_is_only_url")
    record = dict(schema_version=SCHEMA_VERSION, sample_id=_digest({"collection": collection,
        "source": source, "locator": locator}), collection=collection, source=source,
        locator=locator, source_sample_id=raw.get("sample_id", raw.get("id", raw.get("__key__"))),
        format=fmt, structure=kind, segments=segments, images=images, metadata=_safe_metadata(raw),
        issues=sorted(set(str(x) for x in issues)), normalization_notes=notes,
        source_config=source_config)
    result = _classification(record)
    record.update(result)
    return record


def eligible_task_views(collection: str, category: str, candidates: list[str]) -> list[str]:
    """Training views preserve DCVLM's native task; T2I is sourced from gen-und.

    Native categories and ordered segments remain independent evidence, even if
    the collection policy disallows a category as a training view.
    """
    collection = str(collection).casefold()
    if collection != "genund":
        candidates = [view for view in candidates if view == category and view not in T2I_CATEGORIES]
    return list(dict.fromkeys(candidates))


def _classification(record: dict) -> dict:
    result = _native_classification(record)
    result["eligible_views"] = eligible_task_views(
        record.get("collection", ""), result["category"], result["eligible_views"])
    return result


def _native_classification(record: dict) -> dict:
    segments, images = record.get("segments", []), record.get("images", [])
    text = any(s.get("kind") == "text" and s.get("text", "").strip() for s in segments)
    has_image = any(s.get("kind") == "image" for s in segments)
    groups = []
    for s in segments:
        if s.get("role") not in ("user", "assistant"):
            continue
        group = (s.get("role"), s.get("turn"))
        if not groups or groups[-1] != group:
            groups.append(group)
    rounds, waiting = 0, False
    for role, _ in groups:
        if role == "user":
            waiting = True
        elif role == "assistant" and waiting:
            rounds += 1
            waiting = False
    result = {"category": "unclassified", "eligible_views": [], "completed_rounds": rounds}
    if "image_annotation_without_image" in record.get("issues", []):
        return result
    if not images and not has_image and text:
        result.update(category="pure_text", eligible_views=["pure_text"])
        return result
    if images and has_image and not text and record.get('verified_empty_caption_direction') in ('i2t','t2i'):
        category='single_'+record['verified_empty_caption_direction']
        result.update(category=category,eligible_views=[category])
        return result
    if not images or not has_image or not text:
        return result
    if record.get("format") == "interleaved":
        result.update(category="interleaved", eligible_views=["interleaved"])
        return result
    if groups:
        user_image = any(s.get("kind") == "image" and s.get("role") == "user" for s in segments)
        assistant_image = any(s.get("kind") == "image" and s.get("role") == "assistant" for s in segments)
        if not rounds or waiting:
            return result
        if user_image and assistant_image:
            cat = "mixed_multimodal"
        elif assistant_image:
            cat = "multi_t2i" if rounds > 1 else "single_t2i"
        elif user_image:
            cat = "multi_i2t" if rounds > 1 else "single_i2t"
        else:
            cat = "mixed_multimodal"
        result.update(category=cat, eligible_views=[cat])
        return result
    if record.get("format") == "caption" or record.get("structure") == "caption_pair":
        image_positions = [i for i, s in enumerate(segments) if s["kind"] == "image"]
        text_positions = [i for i, s in enumerate(segments) if s["kind"] == "text" and s.get("text", "").strip()]
        if len(images) == 1 and len(image_positions) == 1:
            if max(text_positions) < min(image_positions):
                cat = "single_t2i"
            elif max(image_positions) < min(text_positions):
                cat = "single_i2t"
            else:
                return result
            result.update(category=cat, eligible_views=["single_t2i", "single_i2t"])
        else:
            result.update(category="mixed_multimodal", eligible_views=["mixed_multimodal"])
    return result


def classify(record: dict) -> str:
    """Return the primary category; eligible views remain separate from records."""
    return _classification(record)["category"]


def _explicit_evidence(meta: dict, text: bool = False, where: str = "record") -> list[dict]:
    evidence = []
    if not isinstance(meta, dict):
        return evidence
    keys = TEXT_KEYS if text else IMAGE_KEYS
    for key in keys:
        if key in meta:
            status = _status(meta[key], text=text)
            evidence.append({"status": status, "kind": "structured_metadata", "field": key,
                             "where": where, "value": meta[key]})
            raw_status = meta[key]
            if isinstance(raw_status, dict):
                raw_status = raw_status.get("status", raw_status.get("origin"))
            if not text and isinstance(raw_status, str) and raw_status.lower().strip() in {"synthetic", "generated"}:
                evidence[-1]["ambiguous_synthetic"] = True
    bool_keys = ("is_caption_ai_generated", "caption_is_synthetic") if text else (
        "is_ai_generated", "ai_generated")
    for key in bool_keys:
        if meta.get(key) is True:
            evidence.append({"status": "ai" if text else "ai_generated", "kind": "structured_metadata",
                             "field": key, "where": where, "value": True})
    if not text:
        for key in ("is_generated", "is_synthetic_image", "image_is_synthetic"):
            if meta.get(key) is True:
                evidence.append({"status": "unknown", "kind": "structured_metadata", "field": key,
                                 "where": where, "value": True, "ambiguous_synthetic": True})
    if not text and meta.get("is_ai_edited") is True:
        evidence.append({"status": "ai_edited", "kind": "structured_metadata",
                         "field": "is_ai_edited", "where": where, "value": True})
    if text:
        for key in ("caption_generated_by", "caption_model"):
            if isinstance(meta.get(key), str) and meta[key].strip():
                value = meta[key].lower().strip()
                if value not in {"human", "manual", "none", "unknown", "n/a"}:
                    evidence.append({"status": "ai", "kind": "structured_metadata",
                                     "field": key, "where": where, "value": meta[key]})
    if isinstance(meta.get("metadata"), dict):
        evidence.extend(_explicit_evidence(meta["metadata"], text, where + ".metadata"))
    return evidence


def _source_evidence(config: dict, text: bool = False) -> list[dict]:
    matched = config.get("_matched_rules")
    entries = matched if matched else ([config] if config else [])
    evidence = []
    for entry in entries:
        origin = entry.get("caption_origin", entry.get("text_provenance", "unknown")) if text else entry.get("image_origin", entry.get("image_provenance", "unknown"))
        evidence.append({"status": _status(origin, text), "kind": "source_registry",
            "rule_id": entry.get("id"), "evidence_level": entry.get("evidence_level", "source_declared"),
            "scope": entry.get("scope"), "evidence": entry.get("evidence", []),
            "specificity": list(_specificity(entry)), "selected": entry is entries[-1]})
    return evidence


def _edit_source_evidence(meta: dict, where: str) -> list[dict]:
    """Only explicit editing lineage; ordinary image_origin is not parentage."""
    if not isinstance(meta, dict):
        return []
    evidence = [{"status": _status(meta[key]), "kind": "structured_edit_lineage",
                 "field": key, "where": where, "value": meta[key]}
                for key in EDIT_SOURCE_KEYS if key in meta]
    for key in EDIT_SOURCE_KEYS:
        value = meta.get(key)
        if not isinstance(value, dict):
            continue
        # Do not let an origin=natural label hide a contradictory generated flag,
        # another status, or nested metadata about the same original image.
        evidence.extend(_explicit_evidence(value, where=where + "." + key))
        nested, location = value, where + "." + key
        while isinstance(nested, dict):
            for field in ("origin", "status"):
                if field in nested:
                    evidence.append({"status": _status(nested[field]), "kind": "structured_edit_lineage",
                                     "field": field, "where": location, "value": nested[field]})
            nested = nested.get("metadata")
            location += ".metadata"
    if isinstance(meta.get("metadata"), dict):
        evidence.extend(_edit_source_evidence(meta["metadata"], where + ".metadata"))
    for item in evidence:
        value = item.get("value")
        if isinstance(value, str) and value.lower().strip() in {"synthetic", "generated"}:
            item["ambiguous_synthetic"] = True
    return evidence


def image_policy_decision(origin: str, edit_source_origin: str, rules: dict) -> dict:
    """Apply the image-origin gate without discarding provenance or other gates."""
    mode = rules.get("policy", rules.get("policy_mode", "ai_only"))
    rejected = set(rules.get("rejected_image_origins", NON_NATURAL if mode == "natural_only"
                             else {"ai_generated", "ai_edited"}))
    allow_edits = rules.get("allow_ai_edits_from_natural_sources", True)
    if mode == "ai_only" and allow_edits and origin == "ai_edited":
        if edit_source_origin == "natural":
            return {"decision": "accepted", "reason": None,
                    "exception": "ai_edit_from_natural_source"}
        if edit_source_origin in ("unknown", "mixed"):
            return {"decision": "quarantine", "reason": "ai_edit_source_" + edit_source_origin,
                    "exception": None}
        return {"decision": "rejected", "reason": "ai_edit_source_not_natural", "exception": None}
    if origin in rejected:
        return {"decision": "rejected", "reason": "image_" + origin, "exception": None}
    if origin in ("unknown", "mixed"):
        return {"decision": "quarantine", "reason": "image_provenance_" + origin, "exception": None}
    return {"decision": "accepted", "reason": None, "exception": None}


def _decide_origin(evidence: list[dict], text: bool = False) -> str:
    statuses = {e["status"] for e in evidence}
    if text:
        if "ai" in statuses:
            return "ai"
        if "mixed" in statuses:
            return "mixed"
    else:
        for negative in ("ai_generated", "ai_edited", "programmatic", "rendered"):
            if negative in statuses:
                return negative
        if any(e.get("ambiguous_synthetic") for e in evidence):
            return "unknown"
    explicit = [e for e in evidence if e["kind"] != "source_registry" and e["status"] != "unknown"]
    selected = [e for e in evidence if e.get("selected") and e["status"] != "unknown"]
    usable = explicit or selected
    if not usable:
        return "unknown"
    chosen = {e["status"] for e in usable}
    if "mixed" in chosen or len(chosen) > 1:
        return "mixed"
    return next(iter(chosen))


def _resolved(ref: dict) -> bool:
    kind = ref.get("kind")
    if kind == "url" or kind == "unknown" or ref.get("exists") is not True:
        return False
    if kind == "embedded":
        return bool(ref.get("field") or ref.get("path") or ref.get("index") is not None)
    if kind == "file":
        return bool(ref.get("path"))
    if kind == "tar":
        return bool(ref.get("path") and (ref.get("member") or (
            isinstance(ref.get("offset"), int) and ref["offset"] >= 0 and
            isinstance(ref.get("length"), int) and ref["length"] > 0)))
    return False


def evaluate(record: dict, rules: dict | None = None) -> dict:
    """Apply whole-record image exclusion and retain-and-mark AI text policy."""
    rules = rules or {}
    mode = rules.get("policy", rules.get("policy_mode", "ai_only"))
    if mode not in ("natural_only", "ai_only"):
        raise ValueError(f"Unsupported provenance policy: {mode}")
    defaults = rules.get("defaults") or {}
    for key, expected in {"unknown_image": "quarantine", "unresolved_image": "quarantine",
                          "ai_caption": "retain_and_mark", "non_natural_action": "reject",
                          "disallowed_image_action": "reject"}.items():
        if key in defaults and defaults[key] != expected:
            raise ValueError(f"Unsupported safety default: {key}={defaults[key]}")
    rejected = set(rules.get("rejected_image_origins", NON_NATURAL if mode == "natural_only" else {"ai_generated", "ai_edited"}))
    mandatory = NON_NATURAL if mode == "natural_only" else {"ai_generated", "ai_edited"}
    if not mandatory <= rejected:
        raise ValueError("rejected_image_origins omits required origins for selected policy")
    if not isinstance(rules.get("allow_ai_edits_from_natural_sources", True), bool):
        raise ValueError("allow_ai_edits_from_natural_sources must be a boolean")
    result = dict(record)
    result.update(_classification(record))
    from .production.task_policy import enabled, fields
    task_aware = enabled(rules)
    if task_aware:
        result.update(fields(result))
        # A caption's native ordering has one selected training direction.
        # Do not implicitly expose its inverse view under different filtering.
        result['eligible_views'] = [result['category']] if result['eligible_views'] else []
    i2t = task_aware and result['task_route'] == 'i2t'
    collection, source = record.get("collection", ""), record.get("source", "")
    meta = record.get("metadata", {})
    base = record.get("source_config") or {}
    record_paths = _paths(record.get("locator")) + _paths(meta)
    image_origins, reasons, reject_reasons = [], [], []
    record_config = resolve_source_config(collection, source, record_paths, rules)
    if record_config.get("_matched_rules"):
        base = record_config
    for entry in base.get("_matched_rules", [base]):
        if entry.get("usage") == "evaluation" or entry.get("exclude_reason"):
            reject_reasons.append(str(entry.get("exclude_reason", "evaluation_dataset")))
    for index, ref in enumerate(record.get("images", [])):
        matched = resolve_source_config(collection, source, record_paths + _paths(ref), rules)
        config = matched if matched.get("_matched_rules") else base
        evidence = _source_evidence(config) + _explicit_evidence(meta) + _explicit_evidence(ref, where=f"images[{index}]")
        origin = _decide_origin(evidence)
        resolved = _resolved(ref)
        lineage = _edit_source_evidence(ref, f"images[{index}]")
        # A record-level claim has an unambiguous target only for one image.
        if len(record.get("images", [])) == 1:
            lineage.extend(_edit_source_evidence(meta, "record"))
        for entry in config.get("_matched_rules", [config]):
            lineage.extend(_edit_source_evidence(entry, "source_registry:" + str(entry.get("id", ""))))
        parent_origin = _decide_origin(lineage) if origin == "ai_edited" else None
        gate = ({'decision': 'accepted', 'reason': None, 'exception': 'i2t_dedup_only'}
                if i2t else image_policy_decision(origin, parent_origin or "unknown", rules))
        image_origins.append({"image_index": index, "origin": origin, "resolved": resolved, "evidence": evidence,
                              "ai_edited": origin == "ai_edited", "edit_source_origin": parent_origin,
                              "edit_source_evidence": lineage if origin == "ai_edited" else [],
                              "policy_exception": gate["exception"]})
        if task_aware:
            image_origins[-1].update({key: result[key] for key in
                ('processing_policy', 'task_route', 'image_loss_allowed')})
        if gate["decision"] == "rejected":
            reject_reasons.append(gate["reason"])
        elif gate["decision"] == "quarantine":
            reasons.append(gate["reason"])
        if not resolved:
            reasons.append("image_reference_unresolved")
        entries = config.get("_matched_rules", [config])
        for entry in entries:
            if entry.get("usage") == "evaluation" or entry.get("exclude_reason"):
                reject_reasons.append(str(entry.get("exclude_reason", "evaluation_dataset")))
    # Text evidence may include an image-specific recap rule, without propagating
    # any image naturalness judgement to a different image in the document.
    text_evidence = _source_evidence(base, text=True) + _explicit_evidence(meta, text=True)
    for ref in record.get("images", []):
        config = resolve_source_config(collection, source, record_paths + _paths(ref), rules)
        text_evidence.extend(_source_evidence(config, text=True))
    text_origin = _decide_origin(text_evidence, text=True)
    if result["category"] == "unclassified":
        reasons.append("unclassified_structure")
    if record.get("issues"):
        reasons.extend(record["issues"])
    if meta.get("usage") == "evaluation" or meta.get("exclude_reason"):
        reject_reasons.append(str(meta.get("exclude_reason", "evaluation_dataset")))
    decision = "rejected" if reject_reasons else "quarantine" if reasons else "accepted"
    hashes = [str(ref["sha256"]).lower() for ref in record.get("images", [])
              if isinstance(ref.get("sha256"), str) and re.fullmatch(r"[a-fA-F0-9]{64}", ref["sha256"])]
    result.update(decision=decision, reason_codes=sorted(set(reject_reasons + reasons)),
                  rules_version=rules.get("version", "unversioned"), policy=mode,
                  provenance={"images": image_origins, "text": {"origin": text_origin,
                      "ai_generated": True if text_origin == "ai" else False if text_origin == "human" else None,
                      "action": "retain_and_mark", "evidence": text_evidence}},
                  dedup_keys={"image_sha256": sorted(set(hashes))},
                  record_sha256=_digest({"segments": record.get("segments"), "images": record.get("images")}))
    result["eligible_for_training"] = decision == "accepted"
    return result
