#!/usr/bin/env python3
"""Download bounded Instagram previews from saved anonymous browser DOM results."""
import argparse
from collections import defaultdict
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

from PIL import Image
from collect import Fetcher, append, dump

INCLUDE = {
    "landscape": {"landscape", "landscapephotography", "naturephotography"},
    "portrait": {"portrait", "portraitphotography", "portraits"},
}
EXCLUDE = {"painting", "paintingforsale", "fineartpainting", "pencildrawing", "pencilportrait", "drawing", "sketch", "illustration"}


class Images(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.anchors, self.images = [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            self.anchors.append(attrs.get("href", ""))
        elif tag == "img":
            self.images.append({"src": attrs.get("src", ""), "alt": attrs.get("alt", ""),
                                "href": urljoin("https://www.instagram.com", self.anchors[-1] if self.anchors else "")})

    def handle_endtag(self, tag):
        if tag == "a" and self.anchors:
            self.anchors.pop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True, help="landscape=browser.json or portrait=browser.html; may repeat")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()
    if not 1 <= args.limit <= 20:
        p.error("limit must be 1..20")
    grouped = defaultdict(list)
    for value in args.input:
        group, filename = value.split("=", 1)
        if group not in INCLUDE:
            p.error("unknown category: " + group)
        path = Path(filename)
        text = path.read_text()
        data = json.loads(text)["images"] if path.suffix == ".json" else Images(text).images
        grouped[group].extend((image, str(path)) for image in data)
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    fetcher = Fetcher(args.output, delay=1.0, timeout=20)
    counts, failures, seen, seen_posts = defaultdict(int), [], set(), set()
    for group, images in grouped.items():
        for item, evidence in images:
            if counts[group] >= args.limit:
                break
            href = item.get("href", "")
            post_path = urlsplit(href).path
            if not re.fullmatch(r"/(?:reel|p)/[\w-]+/?", post_path) or post_path in seen_posts:
                continue
            if not (urlsplit(item.get("src", "")).hostname or "").endswith(".cdninstagram.com"):
                continue
            tags = {s.casefold() for s in re.findall(r"#(\w+)", item.get("alt", ""))}
            matched = sorted(tags & INCLUDE[group])
            if not matched or tags & EXCLUDE:
                continue
            seen_posts.add(post_path)
            data, response = fetcher.get(item["src"], max_bytes=20 * 1024 * 1024)
            if "error" in response:
                failures.append({"group": group, "reason": response["error"]})
                continue
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if min(image.size) < 256:
                        raise ValueError("preview shorter side below 256")
                    size = image.size
                    clean = Image.new("RGB", size)
                    clean.paste(image.convert("RGB"))
                    buffer = io.BytesIO()
                    clean.save(buffer, "JPEG", quality=95)
                    clean_bytes = buffer.getvalue()
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                failures.append({"group": group, "reason": str(exc)})
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            relative = Path("images") / (digest + ".jpg")
            (args.output / relative).write_bytes(clean_bytes)
            record = {"sample_id": digest, "source": "instagram", "requested_tag": group,
                      "matched_tags": matched, "path": str(relative), "width": size[0], "height": size[1],
                      "bytes": len(clean_bytes), "sha256": hashlib.sha256(clean_bytes).hexdigest(),
                      "source_sha256": digest, "ai_status": "not_checked", "rights_status": "not_reviewed",
                      "sample_kind": "reel_cover" if post_path.startswith("/reel/") else "post_preview",
                      "original_photo": False, "model_api_calls": 0}
            append(args.output / "manifest.jsonl", record)
            append(args.output / "provenance.private.jsonl", {"sample_id": digest, "post_url": href,
                   "image_url": item["src"], "source_tags": sorted(tags), "evidence": evidence})
            counts[group] += 1
            print(json.dumps({"category": group, "count": counts[group], "matched_tags": matched,
                              "size": size, "kind": record["sample_kind"]}), flush=True)
    dump(args.output / "summary.json", {"downloaded": sum(counts.values()), "tag_counts": dict(counts),
         "model_api_calls": 0, "include_tags": {k: sorted(v) for k, v in INCLUDE.items()},
         "exclude_tags": sorted(EXCLUDE), "min_side": 256, "failures": failures,
         "limitation": "Previews/covers, not original photos. No model or semantic verification."})


if __name__ == "__main__":
    main()
