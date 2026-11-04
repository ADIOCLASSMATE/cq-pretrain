#!/usr/bin/env python3
"""Discover Instagram posts from a hashtag, then collect verified image media.

No input post list and no model API. The graph provider needs authorized
Instagram Hashtag Search access; it is not an anonymous search workaround.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import requests
from PIL import Image

from collect import Fetcher, append, dump


FIELDS = "id,caption,media_type,media_url,permalink,children{id,media_type,media_url}"


def normalize_tag(value):
    value = value.strip().removeprefix("#").casefold()
    if not re.fullmatch(r"\w{1,64}", value):
        raise ValueError("Expected one hashtag, with optional leading #")
    return value


def caption_tags(caption):
    return {tag.casefold() for tag in re.findall(r"(?<!\w)#(\w+)", caption or "")}


def image_children(post):
    if post.get("media_type") == "IMAGE":
        return [post]
    if post.get("media_type") == "CAROUSEL_ALBUM":
        return [child for child in post.get("children", {}).get("data", [])
                if child.get("media_type") == "IMAGE"]
    return []


class GraphError(RuntimeError):
    pass


class GraphClient:
    def __init__(self, token, version, output, delay=1.5):
        if not re.fullmatch(r"v\d+\.\d+", version):
            raise ValueError("IG_API_VERSION must be the version configured for your Meta app, e.g. vNN.0")
        self.token, self.output, self.delay = token, output, delay
        self.base = "https://graph.facebook.com/" + version + "/"
        self.session = requests.Session()
        self.last = 0.0
        self.count = 0

    def get(self, edge, params):
        if not re.fullmatch(r"[\w/]+", edge):
            raise ValueError("Invalid Graph edge")
        time.sleep(max(0, self.last + self.delay - time.monotonic()))
        self.count += 1
        event = {"sequence": self.count, "edge": edge, "params": params}
        try:
            # The credential is never passed in a URL or to the image CDN.
            response = self.session.get(self.base + edge, params=params,
                headers={"Authorization": "Bearer " + self.token}, timeout=(10, 30),
                allow_redirects=False)
            event["http_status"] = response.status_code
            try:
                payload = response.json()
            except ValueError:
                raise GraphError("Graph returned a non-JSON response") from None
            if not isinstance(payload, dict):
                raise GraphError("Graph returned an unexpected response shape")
            # Only retain pagination cursors; next URLs can contain access tokens.
            def sanitize(value):
                if isinstance(value, dict):
                    return {k: sanitize(v) for k, v in value.items()
                            if k not in {"access_token", "appsecret_proof", "next", "previous"}}
                if isinstance(value, list):
                    return [sanitize(v) for v in value]
                if isinstance(value, str):
                    return value.replace(self.token, "[REDACTED]")
                return value
            dump(self.output / "evidence" / f"graph_{self.count:03}.json", sanitize(payload))
            if response.status_code != 200 or "error" in payload:
                error = payload.get("error", {})
                event["error_code"] = error.get("code")
                raise GraphError(f"Graph request failed: HTTP {response.status_code}, code {error.get('code')}; check authorization/permissions")
            return payload
        except requests.RequestException:
            raise GraphError("Graph network request failed") from None
        finally:
            self.last = time.monotonic()
            append(self.output / "graph_requests.jsonl", event)


def discover_posts(client, tag, user_id, max_pages, max_posts, output):
    """Search by tag first, then page two hashtag feeds; never read seed files."""
    result = client.get("ig_hashtag_search", {"user_id": user_id, "q": tag})
    matches = result.get("data", [])
    if not matches:
        append(output / "discovery_events.jsonl", {"stage": "hashtag_search", "tag": tag, "status": "no_hashtag"})
        return
    if len(matches) != 1 or not re.fullmatch(r"\d+", str(matches[0].get("id", ""))):
        raise GraphError("Hashtag search did not return one unambiguous ID")
    hashtag_id = str(matches[0]["id"])
    append(output / "discovery_events.jsonl", {"stage": "hashtag_search", "tag": tag, "hashtag_id": hashtag_id})
    seen = set()
    for feed in ("top_media", "recent_media"):
        after, cursors = None, set()
        for page in range(1, max_pages + 1):
            params = {"user_id": user_id, "fields": FIELDS, "limit": 50}
            if after:
                params["after"] = after
            result = client.get(f"{hashtag_id}/{feed}", params)
            rows = result.get("data", [])
            if not isinstance(rows, list):
                raise GraphError("Hashtag feed did not return a list")
            append(output / "discovery_events.jsonl", {"stage": "feed_page", "feed": feed,
                   "page": page, "returned_posts": len(rows), "hashtag_id": hashtag_id})
            for post in rows:
                post_id = str(post.get("id", ""))
                if not post_id or post_id in seen:
                    continue
                seen.add(post_id)
                source = {"provider": "meta_hashtag_search", "requested_tag": tag,
                          "hashtag_id": hashtag_id, "feed": feed, "page": page,
                          "post_id": post_id, "post_url": post.get("permalink")}
                append(output / "discovered_posts.jsonl", source)
                yield post, source
                if len(seen) >= max_posts:
                    return
            paging = result.get("paging", {})
            after = paging.get("cursors", {}).get("after")
            if not rows or not paging.get("next") or not after or after in cursors:
                break
            cursors.add(after)


def collect(client, user_id, tag, output, fetcher, limit, max_pages, max_posts, min_side):
    outcomes, hashes, count, discovered = Counter(), set(), 0, 0
    for post, discovery in discover_posts(client, tag, user_id, max_pages, max_posts, output):
        discovered += 1
        tags = caption_tags(post.get("caption"))
        outcome = {"post_id": str(post["id"]), "requested_tag": tag}
        if tag not in tags:
            outcome["status"] = "exact_caption_hashtag_missing"
        else:
            images = image_children(post)
            if not images:
                outcome["status"] = "no_verified_image_media"
            else:
                # One image per distinct post makes a five-image preview five posts.
                outcome["status"] = "image_unavailable"
                for media in images:
                    url = media.get("media_url", "")
                    parsed = urlsplit(url)
                    if parsed.scheme != "https" or not (parsed.hostname or "").endswith((".cdninstagram.com", ".fbcdn.net")):
                        continue
                    raw, response = fetcher.get(url, max_bytes=20 * 1024 * 1024)
                    if "error" in response:
                        continue
                    try:
                        with Image.open(io.BytesIO(raw)) as image:
                            image.load()
                            if min(image.size) < min_side:
                                continue
                            size = image.size
                            clean = Image.new("RGB", size)
                            clean.paste(image.convert("RGB"))
                            buffer = io.BytesIO()
                            clean.save(buffer, "JPEG", quality=95)
                            data = buffer.getvalue()
                    except (OSError, ValueError, Image.DecompressionBombError):
                        continue
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in hashes:
                        outcome["status"] = "duplicate_image"
                        continue
                    hashes.add(digest)
                    relative = f"images/{digest}.jpg"
                    (output / relative).write_bytes(data)
                    record = {"sample_id": digest, "source": "instagram", "requested_tag": tag,
                              "category_basis": "exact_caption_hashtag", "tag_filter_used": True,
                              "matched_tags": [tag], "path": relative, "width": size[0], "height": size[1],
                              "sha256": digest, "sample_kind": "static_image_post",
                              "source_media_type": "IMAGE", "parent_media_type": post.get("media_type"),
                              "model_api_calls": 0, "ai_status": "not_checked", "rights_status": "not_reviewed"}
                    append(output / "manifest.jsonl", record)
                    append(output / "provenance.private.jsonl", {**discovery, "sample_id": digest,
                           "source_tags": sorted(tags), "caption": post.get("caption"),
                           "media_id": media.get("id"), "image_url": url, "source_media_type": "IMAGE"})
                    count += 1
                    outcome.update(status="downloaded", path=relative)
                    break
        append(output / "outcomes.jsonl", outcome)
        outcomes[outcome["status"]] += 1
        print(json.dumps(outcome), flush=True)
        if count >= limit:
            break
    return {"downloaded": count, "discovered_posts": discovered,
            "outcomes": dict(outcomes), "status": "complete" if count >= limit else "partial_or_empty"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=3, help="Maximum pages per feed")
    parser.add_argument("--max-posts", type=int, default=100)
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        tag = normalize_tag(args.tag)
    except ValueError as exc:
        parser.error(str(exc))
    if not (1 <= args.limit <= 100 and 1 <= args.max_pages <= 20 and 1 <= args.max_posts <= 1000 and args.min_side > 0):
        parser.error("Bounds: limit 1..100, max-pages 1..20, max-posts 1..1000, min-side >0")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    for sub in ("images", "evidence"):
        (args.output / sub).mkdir()
    for name in ("manifest.jsonl", "provenance.private.jsonl", "discovered_posts.jsonl", "outcomes.jsonl"):
        (args.output / name).touch()
    summary = {"requested_tag": tag, "target": args.limit, "input_post_urls": 0,
               "model_api_calls": 0, "provider": "meta_hashtag_search", "downloaded": 0,
               "live_network_run": False}
    dump(args.output / "config.json", {"tag": tag, "limit": args.limit, "max_pages_per_feed": args.max_pages,
         "max_posts": args.max_posts, "min_side": args.min_side, "input_post_urls": 0})
    missing = [key for key in ("IG_ACCESS_TOKEN", "IG_USER_ID", "IG_API_VERSION") if not os.environ.get(key)]
    if missing:
        summary.update(status="needs_authorized_hashtag_search", missing_config=missing)
    else:
        try:
            user_id = os.environ["IG_USER_ID"]
            if not re.fullmatch(r"\d+", user_id):
                raise ValueError("IG_USER_ID must be a numeric Instagram professional-account ID")
            client = GraphClient(os.environ["IG_ACCESS_TOKEN"], os.environ["IG_API_VERSION"], args.output)
            fetcher = Fetcher(args.output, delay=1.5, timeout=20)
            summary["live_network_run"] = True
            summary.update(collect(client, user_id, tag, args.output, fetcher,
                                   args.limit, args.max_pages, args.max_posts, args.min_side))
            summary.update(graph_requests=client.count, image_requests=fetcher.count)
        except (GraphError, ValueError) as exc:
            saved = len((args.output / "manifest.jsonl").read_text().splitlines())
            summary.update(status="search_or_fetch_failed", error=str(exc), downloaded=saved)
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
