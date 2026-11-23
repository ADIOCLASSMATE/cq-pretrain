#!/usr/bin/env python3
"""Compare Chinese hashtags using the verified v2.1 browser collector."""
import argparse
import base64
from collections import Counter
from datetime import datetime
import html
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.request import getproxies
import zipfile

from instagram_browser_collect import COLLECTOR_VERSION, normalize_tag, run as run_collector, write_json

BATCH_VERSION = "1.0"
DEFAULT_TAGS = ["写真", "人像摄影", "生活记录"]
CONTINUE_STATUSES = {"complete", "discovery_exhausted"}


def write_outputs(output, summary):
    records, sections = [], []
    for group in summary["groups"]:
        manifest = output / group["directory"] / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
        cards = []
        for index, row in enumerate(rows, 1):
            relative = Path(group["directory"]) / row["path"]
            image_path = (output / relative).resolve()
            if not image_path.is_relative_to(output) or image_path.is_symlink():
                raise ValueError("Image path outside batch output")
            record = {**row, "path": relative.as_posix(), "batch_group": group["tag"]}
            records.append(record)
            encoded = base64.b64encode(image_path.read_bytes()).decode()
            cards.append(f'<figure><img loading="lazy" src="data:image/jpeg;base64,{encoded}">'
                         f'<figcaption>{index} · {row["width"]}×{row["height"]}</figcaption></figure>')
        group["downloaded"] = len(rows)
        sections.append(f'<section><h2>#{html.escape(group["tag"])} · {len(rows)}/{summary["limit_per_tag"]}</h2>'
                        f'<p>运行状态：{html.escape(group["status"])}</p><div class="grid">{"".join(cards)}</div></section>')
    hashes = Counter(row["sha256"] for row in records)
    summary.update(downloaded=len(records), unique_images=len(hashes),
                   repeated_images_across_tags=len(records) - len(hashes))
    (output / "batch_manifest.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding="utf-8")
    write_json(output / "summary.json", summary)
    (output / "gallery.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>中文标签对照</title>'
        '<style>body{font:16px system-ui;margin:32px;background:#f5f5f5}.grid{display:flex;flex-wrap:wrap;gap:16px}'
        'section{margin-bottom:36px}figure{margin:0;padding:12px;background:white}img{width:280px;height:360px;object-fit:contain}</style>'
        '<h1>Instagram 中文标签对照</h1><p>依据本次搜索发现的新帖及原帖精确标签收集静态图。'
        '未判断画面题材、是否绘画或是否 AI 生成。模型 API 调用 0 次。</p>'
        f'<p>合计 {len(records)} 个标签样本，{len(hashes)} 张不同图片；跨标签重复 {len(records)-len(hashes)} 张。</p>'
        + ''.join(sections) + '</html>', encoding="utf-8")
    with zipfile.ZipFile(output / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in output.rglob("*"):
            if file.is_file() and not file.is_symlink() and file.name != "results.zip":
                archive.write(file, file.relative_to(output))


def run_batch(args):
    tags = list(dict.fromkeys(normalize_tag(tag) for tag in args.tags))
    if not 1 <= len(tags) <= 6 or not 1 <= args.limit <= 10:
        raise ValueError("Require 1..6 tags and 1..10 images per tag")
    output, profile = args.output.expanduser().resolve(), args.profile.expanduser().resolve()
    if output == profile or output in profile.parents or profile in output.parents:
        raise ValueError("Batch output and browser login profile must be separate")
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=False)
    groups = [{"tag": tag, "directory": f"{index:02d}_{tag}", "status": "not_run", "downloaded": 0}
              for index, tag in enumerate(tags, 1)]
    summary = {"batch_version": BATCH_VERSION, "collector_version": COLLECTOR_VERSION,
        "requested_tags": tags, "limit_per_tag": args.limit, "target": len(tags) * args.limit,
        "input_post_urls": 0, "model_api_calls": 0, "status": "running", "groups": groups}
    write_json(output / "summary.json", summary)
    try:
        for group in groups:
            print(f'\n开始 #{group["tag"]}，目标 {args.limit} 张。', flush=True)
            child_args = SimpleNamespace(tag=group["tag"], limit=args.limit,
                output=output / group["directory"], profile=profile,
                max_scrolls=12, max_posts=60, min_side=512, login_timeout=600,
                headless=args.headless, browser_executable=args.browser_executable,
                proxy_server=args.proxy_server)
            run_collector(child_args)
            child = json.loads((child_args.output / "summary.json").read_text())
            group.update({key: child.get(key) for key in ("status", "downloaded", "attempted_posts", "discovered_posts")})
            write_json(output / "summary.json", summary)
            if child["status"] not in CONTINUE_STATUSES:
                summary.update(status="stopped", stopped_at_tag=group["tag"], stop_reason=child["status"])
                print("本组需要处理登录、网络或解析问题，后续标签暂停。", flush=True)
                break
        else:
            summary["status"] = "complete" if all(g["status"] == "complete" for g in groups) else "partial"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as exc:
        summary.update(status="batch_error", error_type=type(exc).__name__)
    finally:
        write_outputs(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("汇总目录：" + str(output), flush=True)
    return 0 if summary["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=BATCH_VERSION)
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path.home() / "Pictures/InstagramTagSamples" /
                        ("chinese_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--profile", type=Path, default=Path.home() / "Library/Application Support/InstagramTagCollector/browser-profile")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--browser-executable", type=Path)
    proxies = getproxies()
    parser.add_argument("--proxy-server", default=proxies.get("https") or proxies.get("http"))
    return run_batch(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
