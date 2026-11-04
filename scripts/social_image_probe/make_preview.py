#!/usr/bin/env python3
"""Validate local samples and build offline galleries/contact sheets with Pillow."""
import argparse
import base64
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    hashes = set()
    tag_checked = 0
    for run in args.runs:
        provenance = {r["sample_id"]: r for r in map(json.loads, (run / "provenance.private.jsonl").read_text().splitlines())}
        for line in (run / "manifest.jsonl").read_text().splitlines():
            record = json.loads(line)
            image_path = run / record["path"]
            raw = image_path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == record["sha256"]
            assert record["sha256"] not in hashes, "duplicate across runs"
            hashes.add(record["sha256"])
            matched = record.get("matched_tags", [record.get("matched_tag")])
            source_tags = {t.casefold() for t in provenance[record["sample_id"]]["source_tags"]}
            if record.get("tag_filter_used", True):
                assert matched and set(matched).issubset(source_tags)
                tag_checked += 1
            with Image.open(image_path) as image:
                image.load()
                assert image.size == (record["width"], record["height"])
                assert not image.getexif()
                assert not any(k in image.info for k in ("exif", "xmp", "XML:com.adobe.xmp"))
                is_photo = record["source"] == "flickr" or record.get("sample_kind") == "static_image_post"
                assert min(image.size) >= (512 if is_photo else 256)
            record["absolute_path"] = str(image_path.resolve())
            record["matched_tags"] = matched
            groups[record["source"], record["requested_tag"]].append(record)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    title_font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    sections, summary = [], []
    for (source, tag), rows in groups.items():
        assert len(rows) == 5, f"Expected 5 samples for {source}/{tag}"
        cell_w, cell_h, gap, header = 320, 400, 16, 70
        sheet = Image.new("RGB", (3 * cell_w + 4 * gap, header + 2 * cell_h + 3 * gap), "#f3f4f6")
        draw = ImageDraw.Draw(sheet)
        kinds = {r.get("sample_kind", "post_image") for r in rows}
        labels = {"reel_cover": "Reels covers", "static_image_post": "static photo posts",
                  "post_preview": "post previews", "post_image": "post images"}
        kind = labels.get(next(iter(kinds)), "images") if len(kinds) == 1 else "mixed image types"
        draw.text((gap, 18), f"{source.title()} | {tag} | 5 {kind}", font=title_font, fill="#111827")
        cards = []
        for index, record in enumerate(rows):
            x = gap + (index % 3) * (cell_w + gap)
            y = header + gap + (index // 3) * (cell_h + gap)
            draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=10, fill="white")
            image_path = Path(record["absolute_path"])
            with Image.open(image_path) as image:
                preview = ImageOps.contain(image, (cell_w - 16, 340))
                sheet.paste(preview, (x + (cell_w - preview.width) // 2, y + 8 + (340 - preview.height) // 2))
            label = f"#{index + 1:02}  {record['width']} x {record['height']}"
            draw.text((x + 12, y + 354), label, fill="#111827", font=small_font)
            draw.text((x + 12, y + 376), ', '.join(record['matched_tags'])[:35], fill="#4b5563", font=small_font)
            data_url = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode()
            cards.append(f'<figure><a href="{data_url}" target="_blank"><img loading="lazy" src="{data_url}"></a><figcaption>{html.escape(label)}<br>{html.escape(", ".join(record["matched_tags"]))}</figcaption></figure>')
        filename = f"{source}_{tag}.jpg"
        sheet.save(args.output / filename, quality=92)
        sections.append(f'<section><h2>{source.title()} · {tag} · {kind}</h2><div class="grid">'+''.join(cards)+'</div></section>')
        summary.append({"source": source, "category": tag, "count": len(rows), "sample_kind": kind,
                        "contact_sheet": filename, "samples": rows})
    gallery = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>图片采集样本</title><style>body{font:16px system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;background:#f3f4f6;color:#111827}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}figure{margin:0;background:white;border-radius:10px;overflow:hidden}img{width:100%;height:360px;object-fit:contain}figcaption{padding:12px}section{margin:36px 0}</style><h1>图片采集样本 · '+str(len(hashes))+' 张</h1><p>静态照片帖与视频封面按媒体类型标记。均为平台提供的图片版本，不保证为相机原图。分组与标签依据见索引记录，未执行 AI 检测或视觉分类。模型 API 调用 0 次。此页面完全离线。</p>'+''.join(sections)+'</html>'
    (args.output / "gallery.html").write_text(gallery)
    report = {"total_images": len(hashes), "all_file_and_metadata_checks_passed": True,
              "tag_verified_images": tag_checked,
              "category_note": "Rows without tag_filter_used use their recorded source category; not a hashtag match.",
              "model_api_calls": 0, "groups": summary}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "images": len(hashes),
                      "groups": [{k:v for k,v in r.items() if k != "samples"} for r in summary]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
