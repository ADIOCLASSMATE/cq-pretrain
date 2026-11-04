"""Deterministic task/geometry checks complement independent visual verification."""
from __future__ import annotations
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, name, maximum=24000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum, "invalid text: " + name)


def semantic_contract(record):
    category = record["normalized"]["category"]
    if category in {"pure_text", "interleaved"}:
        return category
    if category in {"multi_i2t", "multi_t2i", "mixed_multimodal"}:
        return "dialogue"
    if record["raw"]["format"] == "caption":
        return "caption"
    # A visual conversation is QA unless an explicit prepared adapter marks its
    # semantic caption contract. A model cannot reclassify it to justify rewrites.
    return record.get("prepared_semantic_task", "qa")


def annotation_validator(record):
    image_count = len(record["source_image_refs"])
    segments = record["raw"]["segments"]
    expected = record["normalized"]["category"]
    def validate(value):
        require(set(value) == {"native_task", "semantic_task", "image_captions", "assistant_revisions", "concerns", "layout_candidates"}, "annotation schema mismatch")
        require(value["native_task"] == expected, "native task changed")
        require(value["semantic_task"] == semantic_contract(record), "prepared semantic task changed")
        if expected in ("pure_text", "interleaved"):
            require(value["semantic_task"] == expected, "pure-text/interleaved task changed")
        if expected == "multi_i2t":
            require(value["semantic_task"] == "dialogue", "multi-turn task collapsed")
        require(all(isinstance(value[key], list) for key in ("image_captions", "assistant_revisions", "concerns", "layout_candidates")), "expected annotation lists")
        seen = set()
        for caption in value["image_captions"]:
            require(set(caption) == {"image_index", "short", "medium", "long"}, "caption schema mismatch")
            index = caption["image_index"]
            require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < image_count and index not in seen, "invalid/duplicate caption image index")
            seen.add(index)
            for name in ("short", "medium", "long"):
                text(caption[name], name, 8000)
        if semantic_contract(record) == "caption":
            require(seen == set(range(image_count)), "caption record requires all images captioned")
        changed = set()
        for revision in value["assistant_revisions"]:
            require(set(revision) == {"segment_index", "text", "reason"}, "revision schema mismatch")
            index = revision["segment_index"]
            require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(segments) and index not in changed, "bad revision index")
            require(segments[index]["kind"] == "text" and segments[index].get("role") == "assistant", "revision would alter user/image/context")
            text(revision["text"], "revision"); text(revision["reason"], "reason")
            changed.add(index)
        for concern in value["concerns"]:
            text(concern, "concern")
        seen = set()
        for candidate in value["layout_candidates"]:
            require(set(candidate) == {"image_index", "eligible", "reason"}, "layout candidate schema mismatch")
            index = candidate["image_index"]
            require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < image_count and index not in seen, "bad layout index")
            require(isinstance(candidate["eligible"], bool), "layout eligible must be boolean")
            text(candidate["reason"], "layout reason"); seen.add(index)
        if not image_count:
            require(not value["image_captions"] and not value["layout_candidates"], "pure text invented images")
    return validate


def quality_validator(value):
    require(set(value) in ({"verdict", "issues", "reason"}, {"verdict", "issues", "reason", "caption_checks"}), "quality schema mismatch")
    require(value["verdict"] in {"pass", "revise", "uncertain", "fail"}, "invalid quality verdict")
    require(isinstance(value["issues"], list), "quality issues must be list")
    text(value["reason"], "quality reason")
    for issue in value["issues"]:
        require(set(issue) == {"field", "severity", "reason"}, "quality issue schema mismatch")
        require(issue["severity"] in {"minor", "major"}, "invalid severity")
        text(issue["field"], "issue field"); text(issue["reason"], "issue reason")
    require(not (value["verdict"] == "pass" and any(i["severity"] == "major" for i in value["issues"])), "major error cannot pass")
    if "caption_checks" in value:
        require(isinstance(value["caption_checks"], list) and value["caption_checks"], "caption checks missing")
        seen = set()
        for check in value["caption_checks"]:
            require(set(check) == {"image_index", "variant", "verdict", "issues", "reason"}, "caption check schema mismatch")
            require(type(check["image_index"]) is int and check["image_index"] >= 0 and check["variant"] in {"short", "medium", "long"}, "caption check index/variant mismatch")
            key = (check["image_index"], check["variant"])
            require(key not in seen, "duplicate caption check")
            seen.add(key)
            quality_validator({name: check[name] for name in ("verdict", "issues", "reason")})
        require(value["verdict"] != "pass" or all(c["verdict"] == "pass" for c in value["caption_checks"]),
                "unchecked/uncertain/failed caption variant cannot pass triplet")


def caption_quality_validator(image_count):
    def validate(value):
        quality_validator(value)
        require("caption_checks" in value and {(check["image_index"], check["variant"]) for check in value["caption_checks"]}
                == {(i, variant) for i in range(image_count) for variant in ("short", "medium", "long")},
                "every caption variant must be individually reviewed")
    return validate


def layout_validator(image_index):
    def validate(value):
        require(set(value) == {"image_index", "coordinate_system", "summary", "elements"}, "layout schema mismatch")
        require(value["image_index"] == image_index and value["coordinate_system"] == "normalized_0_1000", "layout binding/coordinate mismatch")
        text(value["summary"], "layout summary")
        require(isinstance(value["elements"], list) and 0 < len(value["elements"]) <= 100, "invalid layout element count")
        ids = set()
        for element in value["elements"]:
            require(set(element) == {"id", "kind", "text", "bbox", "description"}, "element schema mismatch")
            text(element["id"], "element id")
            require(element["id"] not in ids, "duplicate element id"); ids.add(element["id"])
            require(element["kind"] in {"text", "object", "chart", "region", "ui"}, "unknown element kind")
            require(element["text"] is None or isinstance(element["text"], str), "invalid OCR text")
            text(element["description"], "element description")
            box = element["bbox"]
            require(isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1000 for v in box), "invalid box bounds")
            require(box[0] < box[2] and box[1] < box[3], "empty/inverted box")
    return validate


def pair_validator(value):
    require(set(value) == {"same_visual_content", "text_or_numbers_differ", "interchangeable_for_supervision", "reason"}, "pair schema mismatch")
    for key in ("same_visual_content", "text_or_numbers_differ", "interchangeable_for_supervision"):
        require(value[key] is None or isinstance(value[key], bool), "pair verdict must be bool/null")
    require(not (value["interchangeable_for_supervision"] is True and (value["same_visual_content"] is not True or value["text_or_numbers_differ"] is not False)), "inconsistent pair verdict")
    text(value["reason"], "pair reason")
