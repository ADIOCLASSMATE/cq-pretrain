"""Strict extraction for model JSON: never promote a nested object to an answer."""
from __future__ import annotations

import json
import math


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON number")
    return number


def parse_json_response(text: str) -> dict:
    """Accept one complete top-level object, allowing prose or a Markdown fence.

    A lexical scan finds whole top-level containers, respecting JSON strings and
    escapes. A malformed outer container, multiple containers, an array, repeated
    keys, or nonfinite numbers causes rejection rather than partial recovery.
    Bracket-like prose outside JSON is conservatively considered ambiguous.
    """
    if not isinstance(text, str):
        raise ValueError("Response text must be a string")
    spans = []
    position = 0
    while position < len(text):
        if text[position] not in "{[":
            if text[position] in "}]":
                raise ValueError("Unmatched closing JSON container")
            position += 1
            continue
        start = position
        stack = []
        quoted = False
        escaped = False
        while position < len(text):
            character = text[position]
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in "{[":
                stack.append(character)
            elif character in "}]":
                if not stack or (stack[-1], character) not in (("{", "}"), ("[", "]")):
                    raise ValueError("Mismatched JSON container")
                stack.pop()
                if not stack:
                    break
            position += 1
        if stack or quoted:
            raise ValueError("Incomplete outer JSON container")
        spans.append(text[start:position + 1])
        position += 1
    if len(spans) != 1:
        raise ValueError("Expected exactly one complete JSON object")
    value = json.loads(spans[0], object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError("Top-level JSON response must be an object")
    return value
