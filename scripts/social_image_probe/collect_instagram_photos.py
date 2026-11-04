#!/usr/bin/env python3
"""Collect verified static image posts from public Instagram embed pages.

Input is a bounded list of known public post URLs, not hashtag search results.
Require the exact requested hashtag in the author's caption. Only GraphImage
is accepted. GraphVideo/GraphSidecar/unknown are skipped.
"""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from PIL import Image
from collect import Fetcher, append, dump


class Embed(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.media_types, self.images, self.tags = [], [], set()
        self.tag_links = {}
        self._contexts = []
        self.video_count = 0
        self.caption_count = 0
        self.image_count = 0
        self.json_script_count = 0
        self.password_input_count = 0
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        self.caption_count += int("Caption" in classes)
        self.image_count += int(tag == "img")
        self.json_script_count += int(tag == "script" and attrs.get("type") == "application/json")
        self.password_input_count += int(tag == "input" and attrs.get("type", "").lower() == "password")
        parent_caption, parent_excluded = self._contexts[-1][1:] if self._contexts else (False, False)
        in_caption = parent_caption or "Caption" in classes
        excluded = parent_excluded or tag in {"script", "style"} or any(c.startswith("CaptionComments") for c in classes)
        if tag not in self.VOID_TAGS:
            self._contexts.append((tag, in_caption, excluded))
        if attrs.get("data-media-type"):
            self.media_types.append(attrs["data-media-type"])
        if tag == "video":
            self.video_count += 1
        if tag == "img" and "EmbeddedMediaImage" in attrs.get("class", "").split():
            self.images.append(attrs.get("src", ""))
        if tag == "a" and in_caption and not excluded:
            href = attrs.get("href", "")
            link = urlsplit(href)
            match = re.fullmatch(r"/explore/tags/([^/]+)/?", link.path)
            if link.hostname not in {None, "instagram.com", "www.instagram.com"}:
                return
            if match:
                hashtag = unquote(match.group(1)).casefold()
                self.tags.add(hashtag)
                self.tag_links[hashtag] = href

    def handle_endtag(self, tag):
        for index in range(len(self._contexts) - 1, -1, -1):
            if self._contexts[index][0] == tag:
                del self._contexts[index:]
                break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--min-side", type=int, default=512)
    args = p.parse_args()
    seeds = json.loads(args.seeds.read_text())
    if not isinstance(seeds, list) or len(seeds) > 40 or not 1 <= args.limit <= 10 or args.min_side < 1:
        p.error("Require <=40 seeds, limit 1..10, min-side >=1")
    for seed in seeds:
        u = urlsplit(seed["post_url"])
        if u.scheme != "https" or u.hostname != "www.instagram.com" or not re.fullmatch(r"/p/[\w-]+/", u.path):
            p.error("Only public https://www.instagram.com/p/SHORTCODE/ URLs are supported")
        if seed["category"] not in {"landscape", "portrait"}:
            p.error("Unknown category")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    for sub in ("images", "evidence"):
        (args.output / sub).mkdir()
    for name in ("manifest.jsonl", "provenance.private.jsonl"):
        (args.output / name).touch()
    dump(args.output / "seeds.json", seeds)
    fetcher = Fetcher(args.output, delay=1.5, timeout=20)
    counts, types, seen, outcomes = Counter(), Counter(), set(), []
    for seed in seeds:
        category, url = seed["category"], seed["post_url"]
        if counts[category] >= args.limit:
            continue
        shortcode = urlsplit(url).path.split("/")[2]
        body, response = fetcher.get(url + "embed/captioned/", evidence=f"{shortcode}.html")
        outcome = {"post_url": url, "category": category, "http_status": response.get("http_status")}
        if "error" in response:
            outcome.update(status="request_failed", reason=response["error"])
        else:
            parsed = Embed(body.decode("utf-8", errors="replace"))
            media_type = parsed.media_types[0] if parsed.media_types else "unknown"
            types[media_type] += 1
            outcome["source_media_type"] = media_type
            # An image file by itself could be a video poster. Require the post's
            # explicit GraphImage type, one primary image and no video element.
            if parsed.media_types != ["GraphImage"] or len(parsed.images) != 1 or parsed.video_count:
                outcome.update(status="not_verified_single_photo", primary_images=len(parsed.images),
                               video_elements=parsed.video_count)
            elif category not in parsed.tags:
                outcome.update(status="exact_hashtag_missing", required_tag=category,
                               caption_hashtags=sorted(parsed.tags))
            elif not (urlsplit(parsed.images[0]).hostname or "").endswith(".cdninstagram.com"):
                outcome["status"] = "unexpected_image_host"
            else:
                image_url = parsed.images[0]
                image_bytes, image_response = fetcher.get(image_url, max_bytes=20 * 1024 * 1024)
                if "error" in image_response:
                    outcome.update(status="image_download_failed", reason=image_response["error"])
                else:
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as image:
                            image.load()
                            width, height = image.size
                            if min(image.size) < args.min_side:
                                raise ValueError("image below minimum side")
                            clean = Image.new("RGB", image.size)
                            clean.paste(image.convert("RGB"))
                            buffer = io.BytesIO()
                            clean.save(buffer, "JPEG", quality=95)
                            cleaned = buffer.getvalue()
                        source_sha = hashlib.sha256(image_bytes).hexdigest()
                        if source_sha in seen:
                            outcome["status"] = "duplicate"
                        else:
                            seen.add(source_sha)
                            relative = Path("images") / (source_sha + ".jpg")
                            (args.output / relative).write_bytes(cleaned)
                            # A source article may discover candidates, but inclusion
                            # requires an exact linked hashtag in the author's caption.
                            record = {"sample_id": source_sha, "source": "instagram", "requested_tag": category,
                                      "category_basis": "exact_caption_hashtag", "tag_filter_used": True,
                                      "matched_tags": [category], "path": str(relative), "width": width, "height": height,
                                      "bytes": len(cleaned), "sha256": hashlib.sha256(cleaned).hexdigest(),
                                      "source_sha256": source_sha, "source_bytes": len(image_bytes),
                                      "source_media_type": "GraphImage", "sample_kind": "static_image_post",
                                      "original_resolution": "not_guaranteed", "ai_status": "not_checked",
                                      "rights_status": "not_reviewed", "processing": "metadata_removed_jpeg_quality95",
                                      "model_api_calls": 0}
                            append(args.output / "manifest.jsonl", record)
                            append(args.output / "provenance.private.jsonl",
                                   {"sample_id": source_sha, "post_url": url, "embed_url": url + "embed/captioned/",
                                    "image_url": image_url, "source_tags": sorted(parsed.tags),
                                    "matched_tag_href": parsed.tag_links[category],
                                    "tag_evidence_scope": "author_caption_hashtag_link",
                                    "discovery_source": seed.get("source_url"), "evidence": f"evidence/{shortcode}.html",
                                    "source_media_type": "GraphImage", "video_elements": 0})
                            counts[category] += 1
                            outcome.update(status="downloaded", size=[width, height])
                    except (OSError, ValueError, Image.DecompressionBombError) as exc:
                        outcome.update(status="invalid_image", reason=str(exc))
        outcomes.append(outcome)
        append(args.output / "outcomes.jsonl", outcome)
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
    dump(args.output / "summary.json", {"downloaded": sum(counts.values()), "category_counts": dict(counts),
         "media_types_seen": dict(types), "outcome_counts": dict(Counter(x["status"] for x in outcomes)),
         "http_transactions": fetcher.count, "model_api_calls": 0, "authentication": "none",
         "tag_search_implemented": False, "exact_caption_tag_filter": True, "carousel_supported": False,
         "limitations": ["Needs known public post URLs; no autonomous hashtag discovery proven.",
                         "Photo files are Instagram-served copies, not camera originals.",
                         "GraphImage distinguishes static posts, not authentic photography from AI images."]})


if __name__ == "__main__":
    main()
