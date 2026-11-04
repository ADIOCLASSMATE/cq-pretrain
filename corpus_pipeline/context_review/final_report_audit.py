"""Independent source, offline-page, CSV and real-browser checks for the final report.

Only audit evidence is written. Frozen inference, policy and report inputs stay
unchanged. The full 70,492-row overlay is audited by the separate import audit.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlsplit
import zipfile

from corpus_pipeline.codex_review.farm import canonical, digest, file_sha
from corpus_pipeline.context_review import report as source_report
from corpus_pipeline.context_review.apply_quality import verify_version


def require(value, message):
    if not value:
        raise AssertionError(message)


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def check_binding(value):
    path = Path(value["path"])
    require(file_sha(path) == value["sha256"], "artifact changed: " + str(path))
    if "bytes" in value:
        require(path.stat().st_size == value["bytes"], "artifact size changed")


class Cards(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cards = []
        self.card = None
        self.depth = 0
        self.pre = None
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        require(not any(k.lower().startswith("on") for k in attrs), "unexpected inline event handler")
        if tag == "script":
            self.scripts += 1
            require("src" not in attrs, "external script")
        if tag == "details" and attrs.get("class") == "card":
            require(self.card is None, "nested card")
            self.card = {"attrs": attrs, "pre": [], "text": [], "images": [], "links": []}
        if self.card is None:
            return
        if tag == "details":
            self.depth += 1
        if tag == "pre":
            require(self.pre is None, "nested pre")
            self.pre = []
        if tag == "img":
            self.card["images"].append(attrs)
        if tag == "a":
            self.card["links"].append(attrs.get("href"))

    def handle_data(self, data):
        if self.card is not None:
            self.card["text"].append(data)
        if self.pre is not None:
            self.pre.append(data)

    def handle_endtag(self, tag):
        if self.card is None:
            return
        if tag == "pre":
            self.card["pre"].append("".join(self.pre))
            self.pre = None
        if tag == "details":
            self.depth -= 1
            if self.depth == 0:
                self.card["text"] = "".join(self.card["text"])
                self.cards.append(self.card)
                self.card = None


def data_audit(root):
    directory = root / "strict_blur_20261025" / "report"
    complete = source_report.strict_json((directory / "COMPLETE.json").read_bytes())
    require(complete["complete"] is True and complete["records"] == 688, "report incomplete")
    check_binding(complete["report"])
    report = source_report.strict_json((directory / "REPORT.json").read_bytes())
    require(report["complete"] is True, "report not complete")
    frozen = {"report_complete": binding(directory / "COMPLETE.json"),
              "report": binding(directory / "REPORT.json"), **report["artifacts"],
              "policy_pointer": report["policy_pointer"]}
    for value in frozen.values():
        check_binding(value)
    pointer_path = root.parent / "QUALITY_POLICY_CURRENT.json"
    pointer = source_report.strict_json(pointer_path.read_bytes())
    application, verified = verify_version(pointer["directory"], pointer["revision_sha256"])
    require(pointer == verified, "applied policy pointer differs")
    require(report["applied_quality_decisions"] == pointer["quality_decisions"], "report points to other decisions")
    with gzip.open(pointer["quality_decisions"]["path"], "rt") as stream:
        applied = [source_report.strict_json(line) for line in stream]
    sources, _ = source_report.load_completed(root)
    source_by_id = {r["candidate"]["item_id"]: r for r in sources}
    by_number = {r["review_number"]: r for r in applied}
    require(len(applied) == len(by_number) == len(source_by_id) == 688, "nonunique coverage")
    require(set(by_number) == set(range(1, 689)), "review-number coverage")
    require({r["item_id"] for r in applied} == set(source_by_id), "source identity coverage")
    with (directory / "per_item.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 688 and len({r["item_id"] for r in csv_rows}) == 688, "CSV coverage")
    csv_by_id = {r["item_id"]: r for r in csv_rows}
    parser = Cards()
    # Preserve literal CR/LF in source text; text-mode universal-newline
    # conversion would create a false truncation finding for CRLF captions.
    html = (directory / "review.html").read_bytes().decode("utf-8")
    parser.feed(html)
    require(parser.card is None and len(parser.cards) == 688 and parser.scripts == 1, "HTML card/script coverage")
    cards = {int(c["attrs"]["data-review"]): c for c in parser.cards}
    require(set(cards) == set(by_number), "HTML unique review-number coverage")
    with zipfile.ZipFile(directory / "review.zip") as archive:
        require(archive.namelist() == ["review.html"], "ZIP unexpected members")
        require(archive.read("review.html") == (directory / "review.html").read_bytes(), "ZIP page differs")
    image_checks, context_texts, evidence_texts, independent_count = [], 0, 0, 0
    tasks, human, origins = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for number, row in sorted(by_number.items()):
        original = source_by_id[row["item_id"]]
        candidate, policy = original["candidate"], row["policy_input"]
        require(digest(candidate) == policy["source_input_sha256"], "source policy binding")
        card, csv_row = cards[number], csv_by_id[row["item_id"]]
        values = {k: row[k] for k in ("review_number", "item_id", "decision", "report_category",
                  "rejection_reason", "blur_status", "image_origin_followup", "reason")}
        values.update(source_name=candidate["source_name"], task_type=candidate["task_type"],
                      original_human_decision=candidate["human_review"]["decision"],
                      context_action=policy["context_action"],
                      upstream_topic=policy["context_assessment"]["context_interpretation"],
                      source_tar=candidate["source_ref"]["path"], source_member=candidate["source_ref"]["member"])
        require(csv_row == {k: source_report.csv_safe(v) for k, v in values.items()}, "CSV value mismatch #" + str(number))
        attrs = card["attrs"]
        require(attrs["id"] == "item-" + str(number) and "open" not in attrs, "card initial state")
        require((attrs["data-source"], attrs["data-task"], attrs["data-human"], attrs["data-rec"]) ==
                (candidate["source_name"], candidate["task_type"], candidate["human_review"]["decision"], row["report_category"]),
                "HTML filter category mismatch #" + str(number))
        require(len(card["images"]) == 1 and "src" not in card["images"][0], "original image eager/missing")
        raw = base64.b64decode(card["images"][0]["data-src"].split(";base64,", 1)[1], validate=True)
        picture = source_report.original_image(candidate)
        require(raw == picture["raw"] and hashlib.sha256(raw).hexdigest() == candidate["source_ref"]["sha256"],
                "HTML bytes differ from original TAR image #" + str(number))
        image_checks.append({"review_number": number, "sha256": hashlib.sha256(raw).hexdigest(),
                             "bytes": len(raw), "width": picture["width"], "height": picture["height"]})
        context = candidate["source_context"]
        texts = context["caption_texts"] + context["conversation_texts"] + [x["text"] for x in context["interleaved_segments"]]
        for value in texts:
            require(value in card["pre"], "source context truncated/changed #" + str(number))
            context_texts += 1
        for member in original["source_evidence"]["members"]:
            if "text" in member:
                require(member["text"] in card["pre"], "raw source evidence text changed #" + str(number))
                evidence_texts += 1
        metadata = json.dumps(context["metadata_summary"], ensure_ascii=False, indent=2)
        require(metadata in card["pre"], "full image-slot metadata missing")
        require(json.dumps(candidate["human_review"], ensure_ascii=False, indent=2) in card["pre"], "human history missing")
        json_blocks = []
        for value in card["pre"]:
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                continue
            if isinstance(decoded, dict):
                json_blocks.append(decoded)
        blur_blocks = [v for v in json_blocks if "blur_assessment" in v and "independent_blur_adjudication" in v]
        require(len(blur_blocks) == 1, "blur evidence panel missing/duplicated")
        require(blur_blocks[0]["blur_assessment"] == policy["blur_assessment"]
                and blur_blocks[0]["independent_blur_adjudication"] == policy.get("blur_adjudication"), "raw/adjudicated evidence overwritten")
        independent_count += policy.get("blur_adjudication") is not None
        require(row["reason"] in card["text"], "applied reason absent")
        for url in card["links"]:
            require(urlsplit(url).scheme in {"http", "https"} and urlsplit(url).netloc, "unsafe source URL")
        tasks[candidate["task_type"]][row["report_category"]] += 1
        human[candidate["human_review"]["decision"]][row["report_category"]] += 1
        origins[candidate["source_name"]][row["report_category"]] += 1
    summary = {"records": 688, "applied_counts": dict(Counter(r["report_category"] for r in applied)),
               "blur_counts": dict(Counter(r["blur_status"] for r in applied)),
               "by_task": {k: dict(v) for k, v in tasks.items()}, "by_source": {k: dict(v) for k, v in origins.items()},
               "previous_human_cross": {k: dict(v) for k, v in human.items()}}
    require(summary == report["summary"], "report summary differs from applied decisions")
    require(independent_count == 21, "21 independent reviews not preserved")
    require(by_number[1]["report_category"] == "ai_excluded", "human AI exclusion overridden")
    require(by_number[657]["report_category"] == "quality_excluded", "severe pixelation override lost")
    require(by_number[548]["report_category"] == "visual_pending" and "来源" in by_number[548]["reason"], "origin hold lost")
    for value in frozen.values():
        check_binding(value)
    return {"summary": summary, "source_image_sha256_verified": 688,
            "context_text_blocks_verified": context_texts, "full_source_evidence_blocks_verified": evidence_texts,
            "independent_adjudications_visible": independent_count,
            "pending_review_numbers": sorted(r["review_number"] for r in applied if r["report_category"] == "visual_pending"),
            "uncertain_review_numbers": sorted(r["review_number"] for r in applied if r["blur_status"] == "uncertain"),
            "input_bindings": frozen, "image_checks": image_checks}, by_number, directory


def browser_audit(root, directory, applied):
    from playwright.sync_api import sync_playwright
    errors, requests = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/root/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome",
                                             headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1360, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url) if request.url.startswith(("http:", "https:")) else None)
        page.goto((directory / "review.html").as_uri(), wait_until="load")
        require(page.locator(".card").count() == 688, "browser coverage")
        require(page.locator(".card[open]").count() == 0 and page.locator(".card img[src]").count() == 0, "browser eager-load")
        filter_checks = []
        for selector, value_key in (("source", "source_name"), ("task", "task_type"),
                                    ("human", "human"), ("rec", "report_category")):
            expected = Counter((r["baseline_screening"]["human_review"]["decision"] if value_key == "human" else
                                r["policy_input"].get(value_key, r.get(value_key))) for r in applied.values())
            for value, count in expected.items():
                page.select_option("#" + selector, value)
                page.wait_for_function("n => document.querySelectorAll('.card:not([hidden])').length === n", arg=count)
                require(page.locator("#count").inner_text() == str(count) + " / 688 项", "browser filter counter")
                filter_checks.append({"filter": selector, "value": value, "matched": count})
            page.click("#reset")
            page.wait_for_function("document.querySelectorAll('.card:not([hidden])').length === 688")
        page.fill("#q", "#168")
        page.wait_for_function("document.querySelectorAll('.card:not([hidden])').length === 1")
        page.locator("#item-168 > summary").click()
        page.wait_for_function("document.querySelector('#item-168 img').complete && document.querySelector('#item-168 img').naturalWidth > 0")
        page.screenshot(path=str(root / "strict-final-browser.png"), full_page=False)
        page.click("#reset")
        page.click("#collapse")
        require(page.locator(".card[open]").count() == 0, "collapse failed")
        decoded = page.evaluate("""async () => {
          const images = [...document.querySelectorAll('.card img')];
          const out = [];
          for (let i = 0; i < images.length; i += 16) {
            out.push(...await Promise.all(images.slice(i, i + 16).map(async image => {
              image.src = image.dataset.src;
              await image.decode();
              return {review_number: Number(image.closest('.card').dataset.review),
                      width: image.naturalWidth, height: image.naturalHeight};
            })));
          }
          return out;
        }""")
        require(len(decoded) == 688 and all(v["width"] > 0 and v["height"] > 0 for v in decoded), "browser image decode")
        require(not errors and not requests, "browser error or external network")
        version = browser.version
        browser.close()
    return {"passed": True, "browser": "Chromium " + version, "images_decoded": len(decoded),
            "filter_checks": filter_checks, "default_collapsed": True, "on_demand_image_loading": True,
            "search_review_168_passed": True, "collapse_passed": True,
            "javascript_errors": errors, "external_network_requests": requests,
            "screenshot": binding(root / "strict-final-browser.png")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    result, rows, directory = data_audit(root)
    if not args.data_only:
        result["browser"] = browser_audit(root, directory, rows)
    for value in result["input_bindings"].values():
        check_binding(value)
    result.update(schema_version="corpus.strict_quality.final_report_audit.v1", passed=not args.data_only,
                  data_checks_passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  audit_code=binding(__file__), full_70492_overlay_audited_by_separate_agent=True,
                  training_dataset_published=False, source_certification=False)
    output = args.output or root / "FINAL_REPORT_AUDIT.json"
    require(not output.exists(), "audit artifact already exists")
    output.write_text(canonical(result) + "\n")
    print(canonical({k: v for k, v in result.items() if k not in {"image_checks", "input_bindings", "browser"}}))


if __name__ == "__main__":
    main()
