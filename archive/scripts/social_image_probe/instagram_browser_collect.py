#!/usr/bin/env python3
"""Local, user-signed-in Instagram hashtag discovery. No model or Meta API key."""
import argparse
import base64
from collections import Counter
from contextlib import suppress
from datetime import datetime
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote, urlsplit
from urllib.request import getproxies
import zipfile

from PIL import Image, ImageOps
from collect_instagram_photos import Embed

COLLECTOR_VERSION = "2.1"
UNREADABLE_POST_STATUSES = {"post_data_unrecognized", "caption_unavailable"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def normalize_tag(value):
    value = value.strip().removeprefix("#").casefold()
    if not re.fullmatch(r"\w{1,64}", value):
        raise ValueError("请输入单个标签，例如 portrait 或 #portrait")
    return value


def canonical_post_url(value):
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname not in {"www.instagram.com", "instagram.com"}:
        return None
    match = re.fullmatch(r"/(?:[\w.]+/)?p/([\w-]+)/?", parsed.path)
    return "https://www.instagram.com/p/" + match[1] + "/" if match else None


def valid_image_url(value):
    parsed = urlsplit(value)
    return parsed.scheme == "https" and (parsed.hostname or "").endswith((".cdninstagram.com", ".fbcdn.net"))


def photo_candidates(embed_html, shortcode, tag):
    """Verify author hashtag and media type, including explicit non-video children."""
    parsed = Embed(embed_html)
    result = {"shortcode": shortcode, "source_media_types": parsed.media_types,
              "caption_hashtags": sorted(parsed.tags), "matched_tag": tag if tag in parsed.tags else None,
              "matched_tag_href": parsed.tag_links.get(tag), "images": [],
              "page_structure": {"html_characters": len(embed_html), "caption_containers": parsed.caption_count,
                  "image_elements": parsed.image_count, "video_elements": parsed.video_count,
                  "json_scripts": parsed.json_script_count, "password_inputs": parsed.password_input_count}}
    if not parsed.media_types:
        result["status"] = "post_data_unrecognized"
        return result
    if not parsed.caption_count:
        result["status"] = "caption_unavailable"
        return result
    if tag not in parsed.tags:
        result["status"] = "exact_hashtag_missing"
        return result
    if parsed.media_types == ["GraphImage"] and len(parsed.images) == 1 and not parsed.video_count:
        if valid_image_url(parsed.images[0]):
            result["images"] = [{"url": parsed.images[0], "kind": "single_photo", "media_type": "GraphImage"}]
    elif parsed.media_types == ["GraphSidecar"]:
        # The embed contains a JSON string inside contextJSON. Decode twice;
        # never eval JS or interpret the carousel cover alone as a photo.
        decoder = json.JSONDecoder()
        for match in re.finditer(r'"contextJSON"\s*:\s*', embed_html):
            try:
                encoded, _ = decoder.raw_decode(embed_html, match.end())
                data = json.loads(encoded) if isinstance(encoded, str) else encoded
                media = data.get("gql_data", {}).get("shortcode_media", {})
                if media.get("shortcode") != shortcode or media.get("__typename") != "GraphSidecar":
                    continue
                for edge in media.get("edge_sidecar_to_children", {}).get("edges", []):
                    child = edge.get("node", {})
                    if child.get("is_video") is not False:
                        continue
                    url = child.get("display_url", "")
                    if valid_image_url(url):
                        result["images"].append({"url": url, "kind": "carousel_photo",
                            "media_type": "carousel_child_is_video_false", "child_id": child.get("id")})
                break
            except (ValueError, TypeError, AttributeError):
                continue
    result["status"] = "verified_photo" if result["images"] else "no_verified_photo"
    return result


class StopCollection(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def gate_reason(page):
    path = urlsplit(page.url).path
    if any(part in path for part in ("/challenge", "/checkpoint", "/accounts/login")):
        return "needs_user_login_or_verification"
    text = page.locator("body").inner_text(timeout=5000)[:1200].casefold()
    if any(phrase in text for phrase in ("try again later", "please wait a few minutes", "稍后重试", "请稍后再试")):
        return "rate_limited"
    return None


def wait_for_login(context, page, seconds, headless):
    page.goto("https://www.instagram.com/accounts/login/", wait_until="domcontentloaded", timeout=45000)
    print("请在打开的 Instagram 官网窗口登录；密码、验证码由你输入。", flush=True)
    deadline = time.monotonic() + (0 if headless else seconds)
    while True:
        # Inspect presence only. Never print, export or serialize cookie values.
        signed_in = any(cookie["name"] == "sessionid" for cookie in context.cookies("https://www.instagram.com/"))
        challenged = any(s in urlsplit(page.url).path for s in ("/challenge", "/checkpoint"))
        if signed_in and not challenged:
            print("已检测到登录状态，开始搜索标签。", flush=True)
            return
        if time.monotonic() >= deadline:
            raise StopCollection("needs_user_login", "等待登录结束；重新运行即可继续，登录资料仅保留在本机专用浏览器目录。")
        page.wait_for_timeout(1000)


def discover(context, tag, output, max_scrolls, max_posts):
    """Only sources constructed from the tag. Never reads seed lists or accounts."""
    page = context.new_page()
    seen = set()
    routes = [("hashtag_page", "https://www.instagram.com/explore/tags/" + quote(tag) + "/"),
              ("hashtag_search", "https://www.instagram.com/explore/search/keyword/?q=" + quote("#" + tag))]
    try:
        for route, url in routes:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(3500)
            stagnant = 0
            for scroll in range(max_scrolls + 1):
                gate = gate_reason(page)
                if gate:
                    raise StopCollection(gate, "Instagram 要求重新登录、人工验证或稍后再试；已停止自动采集。")
                # Scope discovery to Instagram search/tag pages, not a redirected home feed.
                current_path = urlsplit(page.url).path
                if not current_path.startswith(("/explore/tags/", "/explore/search/", "/popular/")):
                    append_json(output / "discovery_events.jsonl", {"route": route, "scroll": scroll,
                        "status": "unexpected_navigation", "path": current_path})
                    break
                links = page.locator("a[href]").evaluate_all("items => items.map(a => a.href)")
                found = list(dict.fromkeys(p for p in map(canonical_post_url, links) if p))
                fresh = [p for p in found if p not in seen]
                append_json(output / "discovery_events.jsonl", {"route": route, "requested_url": url,
                    "final_path": current_path, "scroll": scroll, "photo_post_links": len(found),
                    "reel_links": sum("/reel/" in link for link in links), "new_post_links": len(fresh)})
                batch = []
                # Persist the full bounded page before the first download can fail.
                for post_url in fresh[:max_posts - len(seen)]:
                    seen.add(post_url)
                    origin = {"requested_tag": tag, "route": route, "search_url": url,
                              "scroll": scroll, "post_url": post_url}
                    append_json(output / "discovered_posts.jsonl", origin)
                    batch.append((post_url, origin))
                print(f"本页发现 {len(batch)} 个新帖，累计 {len(seen)} 个。", flush=True)
                for post_url, origin in batch:
                    yield post_url, origin
                if len(seen) >= max_posts:
                    return
                stagnant = stagnant + 1 if not fresh else 0
                if stagnant >= 3 or scroll >= max_scrolls:
                    break
                page.mouse.wheel(0, 1000)
                page.wait_for_timeout(2500)
    finally:
        page.close()


def safe_browser_error(exc, stage, shortcode=None):
    """Keep useful network categories without exporting URLs, headers or tokens."""
    message = str(exc).casefold()
    code = next((value for value in (
        "ERR_PROXY_CONNECTION_FAILED", "ERR_TUNNEL_CONNECTION_FAILED", "ERR_NO_SUPPORTED_PROXIES",
        "ERR_SOCKS_CONNECTION_FAILED", "ERR_CERT_AUTHORITY_INVALID", "ERR_CERT_DATE_INVALID",
        "ERR_SSL_PROTOCOL_ERROR", "ERR_NAME_NOT_RESOLVED", "ERR_CONNECTION_RESET",
        "ERR_CONNECTION_CLOSED", "ERR_CONNECTION_REFUSED", "ERR_INTERNET_DISCONNECTED",
        "ERR_TIMED_OUT", "ERR_ABORTED", "ECONNRESET", "ECONNREFUSED", "ENOTFOUND", "ETIMEDOUT",
    ) if value.casefold() in message), None)
    if "timeout" in message or "timed out" in message or code in {"ERR_TIMED_OUT", "ETIMEDOUT"}:
        reason = "timeout"
    elif "proxy" in message or "socks" in message or "tunnel" in message:
        reason = "proxy_connection"
    elif any(part in message for part in ("certificate", "cert_", "ssl", "tls")):
        reason = "certificate_or_tls"
    elif "has been closed" in message or "targetclosed" in type(exc).__name__.casefold():
        reason = "browser_closed"
    elif "interrupted by another navigation" in message:
        reason = "navigation_interrupted"
    elif code:
        reason = "network_connection"
    else:
        reason = "unclassified_browser_error"
    return {"stage": stage, "shortcode": shortcode, "reason": reason, "network_code": code,
            "error_type": "TimeoutError" if type(exc).__name__ == "TimeoutError" else "BrowserError"}


def get_browser_body(page, url, byte_limit, *, image=False):
    """Use Chromium's network stack, the same one that opened the search page."""
    if image and not valid_image_url(url):
        raise ValueError("Unexpected image host")
    response = page.goto(url, timeout=30000, wait_until="domcontentloaded")
    if response is None:
        return None, None
    if response.status in (401, 403, 429):
        raise StopCollection("http_access_limited", f"Instagram 返回 HTTP {response.status}，已停止。")
    if response.status != 200:
        return None, response.status
    if image:
        if not valid_image_url(response.url) or not response.headers.get("content-type", "").lower().startswith("image/"):
            return None, response.status
    else:
        gate = gate_reason(page)
        if gate:
            raise StopCollection(gate, "帖子页面要求登录、人工验证或稍后再试；已停止。")
    length = response.headers.get("content-length", "0")
    if length.isdigit() and int(length) > byte_limit:
        return None, response.status
    body = response.body()
    return (body if len(body) <= byte_limit else None), response.status


def read_embed_evidence(page, raw, shortcode, tag):
    """Distinguish unreadable pages from a verified caption missing the tag."""
    evidence = photo_candidates(raw.decode("utf-8", errors="replace"), shortcode, tag)
    evidence["read_source"] = "navigation_response"
    for attempt in range(6):
        if evidence["page_structure"]["password_inputs"]:
            raise StopCollection("needs_user_login_or_verification", "帖子响应显示登录表单，已停止；未将它计作标签不匹配。")
        if evidence["status"] not in UNREADABLE_POST_STATUSES or attempt == 5:
            break
        page.wait_for_timeout(1000)
        gate = gate_reason(page)
        if gate:
            raise StopCollection(gate, "等待帖子加载时遇到登录、验证或限流，已停止。")
        rendered = page.content()
        if len(rendered) > 4 * 1024 * 1024:
            break
        evidence = photo_candidates(rendered, shortcode, tag)
        evidence["read_source"] = "rendered_dom"
    path = urlsplit(page.url).path
    evidence["final_page_kind"] = (
        "expected_embed" if path == f"/p/{shortcode}/embed/captioned/" else
        "expected_post" if path == f"/p/{shortcode}/" else "other")
    return evidence


def save_photo(raw, output, min_side, hashes):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            image = ImageOps.exif_transpose(image)
            if min(image.size) < min_side:
                return None
            clean = Image.new("RGB", image.size)
            clean.paste(image.convert("RGB"))
            buffer = io.BytesIO()
            clean.save(buffer, "JPEG", quality=95)
            data, size = buffer.getvalue(), clean.size
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    digest = hashlib.sha256(data).hexdigest()
    if digest in hashes:
        return None
    hashes.add(digest)
    relative = "images/" + digest + ".jpg"
    (output / relative).write_bytes(data)
    return {"sample_id": digest, "sha256": digest, "path": relative,
            "width": size[0], "height": size[1], "bytes": len(data)}


def write_gallery(output, tag, records):
    cards = []
    for index, record in enumerate(records, 1):
        encoded = base64.b64encode((output / record["path"]).read_bytes()).decode()
        cards.append(f'<figure><img src="data:image/jpeg;base64,{encoded}"><figcaption>'
                     f'{index} · #{html.escape(tag)} · {record["width"]}×{record["height"]}</figcaption></figure>')
    (output / "gallery.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>Instagram 标签照片</title>'
        '<style>body{font:16px system-ui;margin:32px;background:#f5f5f5}.grid{display:flex;flex-wrap:wrap;gap:16px}'
        'figure{margin:0;background:white;padding:12px}img{width:300px;height:400px;object-fit:contain}</style>'
        f'<h1>#{html.escape(tag)} · {len(records)} 张</h1><p>来自标签搜索发现的新帖，已核对原帖标签和照片类型。'
        '未进行 AI 图检测，模型 API 调用 0 次。</p><div class="grid">'+''.join(cards)+'</div></html>', encoding="utf-8")


def collect_posts(context, image_context, args, output, tag, records, outcomes, hashes, progress, browser_error):
    embed_page, image_page = None, None
    discovery = discover(context, tag, output, args.max_scrolls, args.max_posts)
    consecutive_errors = 0
    consecutive_unreadable = 0
    try:
        embed_page, image_page = context.new_page(), image_context.new_page()
        progress.update(stage="discovery", shortcode=None)
        for post_url, origin in discovery:
            shortcode = post_url.split("/")[4]
            progress.update(stage="post_embed_navigation", shortcode=shortcode,
                            attempted_posts=progress["attempted_posts"] + 1)
            outcome = {"post_url": post_url, "status": "interrupted"}
            append_json(output / "attempted_posts.jsonl", {"post_url": post_url, "shortcode": shortcode})
            try:
                body, http_status = get_browser_body(embed_page, post_url + "embed/captioned/", 4 * 1024 * 1024)
                outcome.update(http_status=http_status, status="empty_response")
                if body:
                    progress["stage"] = "post_embed_parse"
                    evidence = read_embed_evidence(embed_page, body, shortcode, tag)
                    write_json(output / "evidence" / (shortcode + ".json"), evidence)
                    outcome["status"] = evidence["status"]
                    if evidence["status"] in UNREADABLE_POST_STATUSES:
                        consecutive_unreadable += 1
                        append_json(output / "errors.jsonl", {"stage": "post_embed_parse", "shortcode": shortcode,
                            "reason": evidence["status"], "page_structure": evidence["page_structure"],
                            "final_page_kind": evidence["final_page_kind"]})
                        print("未读到可核验的帖子数据，已记录页面结构。", flush=True)
                        if consecutive_unreadable >= 3:
                            raise StopCollection("post_parser_unavailable", "连续三帖无法解析，已提前停止；这不代表原帖没有目标标签。")
                    else:
                        consecutive_unreadable = 0
                    for media in evidence["images"]:
                        time.sleep(1.5)
                        progress["stage"] = "photo_download"
                        raw, _ = get_browser_body(image_page, media["url"], 20 * 1024 * 1024, image=True)
                        progress["stage"] = "photo_decode_and_save"
                        saved = save_photo(raw, output, args.min_side, hashes) if raw else None
                        if saved:
                            record = {**saved, "source": "instagram", "requested_tag": tag,
                                "category_basis": "exact_caption_hashtag", "tag_filter_used": True,
                                "matched_tags": [tag], "sample_kind": "static_image_post",
                                "source_media_type": media["media_type"], "model_api_calls": 0,
                                "ai_status": "not_checked"}
                            records.append(record)
                            append_json(output / "manifest.jsonl", record)
                            append_json(output / "provenance.private.jsonl", {**origin,
                                "sample_id": saved["sample_id"], "source_tags": evidence["caption_hashtags"],
                                "matched_tag_href": evidence["matched_tag_href"], "media": media,
                                "evidence": "evidence/" + shortcode + ".json"})
                            outcome["status"] = "downloaded"
                            print(f"已保存 {len(records)}/{args.limit} 张符合 #{tag} 的照片", flush=True)
                            break
                    if evidence["images"] and outcome["status"] != "downloaded":
                        outcome["status"] = "image_download_invalid_small_or_duplicate"
                consecutive_errors = 0
            except browser_error as exc:
                detail = safe_browser_error(exc, progress["stage"], shortcode)
                append_json(output / "errors.jsonl", detail)
                outcome.update(status="post_browser_error", error=detail)
                consecutive_errors += 1
                consecutive_unreadable = 0
                print(f"本帖失败：{detail['stage']} / {detail['reason']}；已记录诊断。", flush=True)
                if detail["reason"] == "browser_closed" or consecutive_errors >= 3:
                    raise StopCollection("repeated_browser_errors", "连续网络错误或浏览器已关闭，详情见 errors.jsonl。") from None
                # A failed navigation may still be loading chrome-error://. New
                # tabs prevent that pending navigation from cancelling the next post.
                with suppress(browser_error):
                    embed_page.close()
                with suppress(browser_error):
                    image_page.close()
                embed_page, image_page = context.new_page(), image_context.new_page()
            except StopCollection as stop:
                if stop.status != "post_parser_unavailable":
                    outcome["status"] = stop.status
                outcome["stage"] = progress["stage"]
                raise
            finally:
                append_json(output / "outcomes.jsonl", outcome)
                outcomes[outcome["status"]] += 1
            if len(records) >= args.limit:
                break
            progress.update(stage="discovery", shortcode=None)
            time.sleep(1.5)
    finally:
        with suppress(browser_error):
            discovery.close()
        for page in (embed_page, image_page):
            if page is not None:
                with suppress(browser_error):
                    page.close()


def run(args):
    from playwright.sync_api import sync_playwright, Error as BrowserError
    tag = normalize_tag(args.tag)
    os.umask(0o077)
    output, profile = args.output.expanduser().resolve(), args.profile.expanduser().resolve()
    if output == profile or output in profile.parents or profile in output.parents:
        raise ValueError("登录资料和输出必须使用互不包含的独立目录。")
    output.mkdir(parents=True, exist_ok=False)
    profile.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    (output / "evidence").mkdir()
    for filename in ("manifest.jsonl", "provenance.private.jsonl", "discovered_posts.jsonl", "discovery_events.jsonl", "outcomes.jsonl", "errors.jsonl", "attempted_posts.jsonl"):
        (output / filename).touch()
    records, outcomes, hashes = [], Counter(), set()
    summary = {"requested_tag": tag, "target": args.limit, "input_post_urls": 0, "model_api_calls": 0,
               "provider": "local_logged_in_browser", "collector_version": COLLECTOR_VERSION,
               "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "download_transport": "chromium_navigation", "downloaded": 0, "status": "starting"}
    progress = {"stage": "browser_launch", "shortcode": None, "attempted_posts": 0}
    write_json(output / "summary.json", summary)
    try:
        with sync_playwright() as p:
            options = {"headless": args.headless, "viewport": {"width": 1360, "height": 950}}
            if args.browser_executable:
                options["executable_path"] = str(args.browser_executable)
            elif sys.platform == "darwin" and Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").exists():
                options["channel"] = "chrome"
            if args.proxy_server:
                options["proxy"] = {"server": args.proxy_server}
            context = p.chromium.launch_persistent_context(str(profile), **options)
            try:
                progress["stage"] = "login"
                login = context.pages[0] if context.pages else context.new_page()
                wait_for_login(context, login, args.login_timeout, args.headless)
                login.goto("about:blank")
                progress["stage"] = "image_context_setup"
                # Use an isolated browser context for CDN images, with no Instagram cookies.
                image_options = {"proxy": options["proxy"]} if "proxy" in options else {}
                image_context = context.browser.new_context(service_workers="block", **image_options)
                try:
                    collect_posts(context, image_context, args, output, tag, records, outcomes, hashes, progress, BrowserError)
                finally:
                    with suppress(BrowserError):
                        image_context.close()
                summary["status"] = "complete" if len(records) >= args.limit else "discovery_exhausted"
            finally:
                with suppress(BrowserError):
                    context.close()
    except StopCollection as stop:
        summary.update(status=stop.status, message=stop.message)
    except BrowserError as exc:
        detail = safe_browser_error(exc, progress["stage"], progress["shortcode"])
        append_json(output / "errors.jsonl", detail)
        summary.update(status="browser_or_network_error", error=detail,
                       message="浏览器或网络失败，已记录阶段及脱敏错误分类，见 errors.jsonl。")
    except KeyboardInterrupt:
        summary.update(status="interrupted")
    finally:
        summary.update(downloaded=len(records), outcomes=dict(outcomes),
                       last_stage=progress["stage"], attempted_posts=progress["attempted_posts"],
                       discovered_posts=len((output / "discovered_posts.jsonl").read_text().splitlines()))
        write_json(output / "summary.json", summary)
        write_gallery(output, tag, records)
        # This archive is made from output only; the login profile is elsewhere.
        with zipfile.ZipFile(output / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for file in output.rglob("*"):
                if file.is_file() and file.name != "results.zip":
                    archive.write(file, file.relative_to(output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果目录：" + str(output))
    return 0 if summary["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=COLLECTOR_VERSION)
    parser.add_argument("--tag", default="portrait")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-scrolls", type=int, default=12)
    parser.add_argument("--max-posts", type=int, default=60)
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--login-timeout", type=int, default=600)
    parser.add_argument("--headless", action="store_true", help="Diagnostic mode: never waits for interactive login")
    parser.add_argument("--browser-executable", type=Path)
    proxies = getproxies()
    parser.add_argument("--proxy-server", default=proxies.get("https") or proxies.get("http"))
    parser.add_argument("--profile", type=Path, default=Path.home() / "Library/Application Support/InstagramTagCollector/browser-profile")
    parser.add_argument("--output", type=Path, default=Path.home() / "Pictures/InstagramTagSamples" / ("portrait_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    if not (1 <= args.limit <= 100 and 0 <= args.max_scrolls <= 100 and 1 <= args.max_posts <= 1000 and args.min_side > 0 and 0 <= args.login_timeout <= 1800):
        parser.error("Invalid bounded collection settings")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
