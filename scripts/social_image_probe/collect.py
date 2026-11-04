#!/usr/bin/env python3
"""Bounded anonymous HTTP probe, plus Flickr public-tag image collection.

No inference, model API, browser automation, account credentials or paid API.
Instagram/X/Xiaohongshu are probes, not implemented full crawler adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import quote, urlencode, urlsplit

import requests
from PIL import Image


ZH_TAGS = {"landscape": "风景", "portrait": "写真"}
SOURCES = ("instagram", "x", "xiaohongshu", "flickr")
USER_AGENT = "PublicImageTagProbe/0.1 (small internal feasibility test)"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.meta = {}
        self.links = []
        self.image_count = 0
        self.title = ""
        self.in_title = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            self.meta[attrs.get("property", attrs.get("name", ""))] = attrs.get("content", "")
        elif tag == "a":
            self.links.append(attrs.get("href", ""))
        elif tag == "img":
            self.image_count += 1
        elif tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


class Fetcher:
    def __init__(self, output, delay, timeout):
        self.output, self.delay, self.timeout = output, delay, timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.last_request = 0.0
        self.count = 0

    def get(self, url, *, evidence=None, max_bytes=3 * 1024 * 1024):
        # One request at a time; no retry loops, proxy rotation or login handling.
        time.sleep(max(0, self.last_request + self.delay - time.monotonic()))
        start = time.monotonic()
        record = {"requested_url": url, "started_at": datetime.now(timezone.utc).isoformat()}
        body = b""
        try:
            with self.session.get(url, timeout=(10, self.timeout), stream=True) as response:
                record.update(http_status=response.status_code, final_url=response.url,
                              content_type=response.headers.get("content-type", ""),
                              redirects=[r.status_code for r in response.history])
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"response exceeds {max_bytes} bytes")
                    chunks.append(chunk)
                body = b"".join(chunks)
                record["bytes"] = len(body)
                response.raise_for_status()
        except (requests.RequestException, ValueError) as exc:
            record["error"] = str(exc)
        self.last_request = time.monotonic()
        self.count += 1
        record["seconds"] = round(self.last_request - start, 3)
        append(self.output / "requests.jsonl", record)
        if evidence:
            path = self.output / "evidence" / evidence
            path.write_bytes(body)
            dump(path.with_suffix(path.suffix + ".response.json"), record)
        return body, record


def source_url(source, tag):
    if source == "instagram":
        return "https://www.instagram.com/explore/tags/" + quote(tag, safe="") + "/"
    if source == "x":
        return "https://x.com/search?" + urlencode({"q": f"#{tag} filter:images", "src": "typed_query", "f": "live"})
    if source == "xiaohongshu":
        return "https://www.xiaohongshu.com/search_result?" + urlencode({"keyword": ZH_TAGS.get(tag, tag), "source": "web_search_result_notes"})
    return "https://www.flickr.com/services/feeds/photos_public.gne?" + urlencode({"tags": tag, "tagmode": "all", "format": "json", "nojsoncallback": 1})


def probe(fetcher, source, tag):
    body, response = fetcher.get(source_url(source, tag), evidence=f"{source}_{tag}.html")
    text = body.decode("utf-8", errors="replace")
    page = Page(text)
    result = {"source": source, "tag": tag, "requested_url": response["requested_url"],
              "final_url": response.get("final_url"), "http_status": response.get("http_status"),
              "title": page.title, "downloaded": 0, "html_img_elements": page.image_count,
              "query_mode": "keyword_discovery_only" if source == "xiaohongshu" else "hashtag_entry"}
    if "error" in response:
        result.update(status="request_failed", detail=response["error"])
    elif source == "x" and any(s in response.get("final_url", "") for s in ["login", "onboarding"]):
        result.update(status="login_redirect", detail="Search redirected to login; no search results collected.")
    elif source == "instagram":
        post_links = [s for s in page.links if re.search(r"/(p|reel)/[^/]+", s)]
        result.update(status="no_post_images_in_http_response" if not post_links else "post_links_require_adapter",
                      post_links=len(post_links), og_image_present=bool(page.meta.get("og:image")),
                      detail="Popular-page OG cover is not a verified tagged post and is not counted as a sample.")
    elif source == "xiaohongshu":
        search_state = re.search(r'"search"\s*:\s*\{.{0,3000}?"feeds"\s*:\s*\[\s*\]', text, re.S)
        result.update(status="empty_search_state" if search_state else "search_requires_adapter",
                      login_hint_present="登录探索更多内容" in text,
                      detail="Keyword page is not proof of a matching hashtag; logos and static UI assets are excluded.")
    else:
        result.update(status="no_supported_search_payload", detail="HTTP response needs a platform-specific adapter.")
    return result


def collect_flickr(fetcher, tag, limit, min_side, seen):
    body, response = fetcher.get(source_url("flickr", tag), evidence=f"flickr_{tag}.json")
    result = {"source": "flickr", "tag": tag, "http_status": response.get("http_status"),
              "requested_url": response["requested_url"], "query_mode": "exact_feed_tag",
              "feed_items": 0, "tag_matches": 0, "downloaded": 0, "duplicates": 0, "failures": []}
    if "error" in response:
        return dict(result, status="request_failed", detail=response["error"])
    try:
        payload = json.loads(body)
        items = payload["items"]
        if not isinstance(items, list):
            raise ValueError("feed items is not a list")
    except (ValueError, KeyError) as exc:
        return dict(result, status="invalid_feed", detail=str(exc))
    result["feed_items"] = len(items)
    matches = [item for item in items if tag.casefold() in item.get("tags", "").casefold().split()]
    result["tag_matches"] = len(matches)
    for index, item in enumerate(matches):
        if result["downloaded"] >= limit:
            break
        # Follow the actual photo page to discover a larger image, not guessed CDN URLs.
        page_url = item.get("link", "")
        if urlsplit(page_url).hostname != "www.flickr.com":
            result["failures"].append({"index": index, "reason": "unexpected_photo_host"})
            continue
        page_body, page_response = fetcher.get(page_url, evidence=f"flickr_{tag}_{index}.html")
        if "error" in page_response:
            result["failures"].append({"index": index, "reason": page_response["error"]})
            continue
        image_url = Page(page_body.decode("utf-8", errors="replace")).meta.get("og:image", "")
        if not (urlsplit(image_url).hostname or "").endswith(".staticflickr.com"):
            result["failures"].append({"index": index, "reason": "missing_photo_og_image"})
            continue
        image_bytes, image_response = fetcher.get(image_url, max_bytes=20 * 1024 * 1024)
        if "error" in image_response:
            result["failures"].append({"index": index, "reason": image_response["error"]})
            continue
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                width, height, fmt = image.width, image.height, image.format
                if min(width, height) < min_side:
                    raise ValueError(f"short side {min(width, height)} below {min_side}")
                if fmt not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError(f"unsupported image format: {fmt}")
                # Re-encode the use copy without EXIF/XMP or other embedded metadata.
                clean = Image.new("RGB", image.size)
                clean.paste(image.convert("RGB"))
                buffer = io.BytesIO()
                clean.save(buffer, format="JPEG", quality=95)
                clean_bytes = buffer.getvalue()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            result["failures"].append({"index": index, "reason": str(exc)})
            continue
        original_sha = hashlib.sha256(image_bytes).hexdigest()
        if original_sha in seen:
            result["duplicates"] += 1
            continue
        seen.add(original_sha)
        relative = Path("images") / f"{original_sha}.jpg"
        (fetcher.output / relative).write_bytes(clean_bytes)
        record = {"sample_id": original_sha, "source": "flickr", "requested_tag": tag,
                  "matched_tag": tag, "path": str(relative), "width": width, "height": height,
                  "sha256": hashlib.sha256(clean_bytes).hexdigest(), "source_sha256": original_sha,
                  "bytes": len(clean_bytes), "source_bytes": len(image_bytes),
                  "processing": "metadata_removed_jpeg_quality95", "ai_status": "not_checked",
                  "rights_status": "not_reviewed", "model_api_calls": 0}
        append(fetcher.output / "manifest.jsonl", record)
        append(fetcher.output / "provenance.private.jsonl",
               {"sample_id": original_sha, "post_url": page_url, "image_url": image_url,
                "source_tags": item.get("tags", "").split(), "published": item.get("published"),
                "image_variant": "photo_page_og_image_not_guaranteed_original"})
        result["downloaded"] += 1
        print(json.dumps({"source": "flickr", "tag": tag, "downloaded": result["downloaded"], "size": [width, height]}, ensure_ascii=False), flush=True)
    result["status"] = "collected" if result["downloaded"] else "no_images_collected"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New private output directory")
    parser.add_argument("--tags", nargs="+", default=["landscape", "portrait"])
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--limit", type=int, default=5, help="Photos per tag, max 20")
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.limit <= 20 or args.delay < 0.5 or args.min_side < 1 or args.timeout <= 0:
        parser.error("Require limit 1..20, delay >=0.5, min-side >=1, timeout >0")
    if any(not re.fullmatch(r"[\w-]{1,64}", tag) for tag in args.tags):
        parser.error("Use one tag per argument, letters/numbers/underscore/hyphen only")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["evidence", "images"]:
        (args.output / name).mkdir()
    for name in ["manifest.jsonl", "provenance.private.jsonl"]:
        (args.output / name).touch()
    started = time.monotonic()
    fetcher = Fetcher(args.output, args.delay, args.timeout)
    results, seen = [], set()
    for source in args.sources:
        for tag in args.tags:
            result = collect_flickr(fetcher, tag, args.limit, args.min_side, seen) if source == "flickr" else probe(fetcher, source, tag)
            results.append(result)
            dump(args.output / "results.partial.json", results)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {"completed_at": datetime.now(timezone.utc).isoformat(),
               "seconds": round(time.monotonic() - started, 2), "http_transactions": fetcher.count,
               "model_api_calls": 0, "model_inference": False, "authentication": "none",
               "tags": args.tags, "min_side": args.min_side, "limit_per_tag": args.limit,
               "downloaded": sum(r["downloaded"] for r in results), "results": results,
               "limitations": ["HTTP only; no JavaScript execution or authenticated search tested.",
                               "Instagram/X/Xiaohongshu are diagnostics, not implemented full collectors.",
                               "Xiaohongshu query is keyword discovery, not verified hashtag matching.",
                               "Flickr feed is a bounded recent feed, without pagination or exhaustive search.",
                               "Photo-page OG images may be resized; original resolution is not guaranteed.",
                               "No AI detection, semantic quality assessment or rights clearance.",
                               "Metadata removal is not full anonymization of recognizable people."]}
    dump(args.output / "summary.json", summary)
    print(json.dumps({"output": str(args.output.resolve()), "downloaded": summary["downloaded"],
                      "seconds": summary["seconds"], "model_api_calls": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
