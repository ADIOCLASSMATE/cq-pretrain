"""Publish a separate, audited context-quality report after strict farm completion.

This layer never updates existing human decisions or AI screening results. The
offline browser embeds the exact source image bytes and fully escaped text.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit
import uuid
import zipfile

from PIL import Image

from corpus_pipeline.codex_review.farm import canonical, digest, exclusive, file_sha
from corpus_pipeline.context_review import worker

SCHEMA = "corpus.context_review.report.v1"
EXPECTED_RECORDS = 688
LABELS = {
    "caption_pair": "图文对 / I2T", "conversation": "对话 / 区域问答", "interleaved": "交错序列中的目标图", "unknown": "类型未定",
    "keep": "建议保留", "flag_low_information": "保留并标注低信息", "exclude_pair": "建议排除图文对",
    "drop_image_keep_text": "建议删除目标图槽，正文候选保留", "needs_context": "上下文不足",
    "exclude_ai": "人工 AI 排除", "exclude_quality": "人工质量排除", "unsure": "人工未定",
    "good": "内容对齐且有用", "usable_low_information": "可用但信息较弱", "unusable": "不适合当前图文任务",
    "image_information_destroyed": "图像信息损伤", "model_image_text_mismatch": "当前图像与原文不符（模型建议）", "none": "不适用",
}
DISCLAIMER = ("本报告只复核所提供目标图与同记录原始文本的训练用途；未查看同记录其他图像，"
              "不能认证整条交错序列。删除图槽并保留正文是候选处理建议，正文仍需独立校验。"
              "这不是 AI 来源认证，也不是最终训练集发布；原人工判定和原筛选记录均未改写。")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(values):
        output = {}
        for key, value in values:
            if key in output:
                raise ValueError("duplicate_json_key:" + key)
            output[key] = value
        return output
    def invalid(value):
        raise ValueError("nonfinite_json:" + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_bound(path, sha=None):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    if sha is not None:
        _require(_sha(raw) == sha, "artifact_hash_mismatch:" + str(path))
    return strict_json(raw), {"path": str(path), "bytes": len(raw), "sha256": _sha(raw)}


def _jsonl(path):
    with Path(path).open("rb") as stream:
        return [strict_json(line) for line in stream if line.strip()]


def load_completed(root):
    """Verify completion, source, raw context, export and SQLite row bindings."""
    root = Path(root).resolve(strict=True)
    completion, complete_binding = _read_bound(root / "COMPLETE.json")
    _require(completion.get("complete") is True and completion.get("audit_mode") == "strict"
             and completion.get("skipped_gates") == [] and completion.get("records") == EXPECTED_RECORDS,
             "strict_complete_688_required")
    manifest, manifest_binding = _read_bound(completion["manifest"], completion["manifest_sha256"])
    audit, audit_binding = _read_bound(completion["audit"], completion["audit_sha256"])
    _require(audit.get("passed") is True and audit.get("mode") == "strict"
             and audit.get("complete_coverage") is True and audit.get("skipped_gates") == []
             and audit.get("export_checked") is True and audit.get("succeeded") == EXPECTED_RECORDS
             and audit.get("context_items_checked") == EXPECTED_RECORDS
             and audit.get("context_prompt_binding_checked") is True,
             "context_strict_audit_not_passed")
    _require(audit.get("export_manifest") == manifest_binding["path"]
             and audit.get("export_manifest_sha256") == manifest_binding["sha256"],
             "audit_manifest_binding_mismatch")
    _require(manifest.get("complete") is True and manifest.get("audit_mode") == "strict"
             and manifest.get("records") == EXPECTED_RECORDS and manifest.get("outputs") == EXPECTED_RECORDS
             and manifest.get("model") == worker.MODEL and manifest.get("reasoning_effort") == worker.EFFORT,
             "manifest_contract_mismatch")
    source = manifest["source"]
    candidates_path = Path(source["path"]).resolve(strict=True)
    _require(file_sha(candidates_path) == source["sha256"] and source["count"] == EXPECTED_RECORDS,
             "candidate_manifest_binding_mismatch")
    candidates = _jsonl(candidates_path)
    by_id = {row["item_id"]: row for row in candidates}
    _require(len(by_id) == len(candidates) == EXPECTED_RECORDS
             and {row["review_number"] for row in candidates} == set(range(1, EXPECTED_RECORDS + 1)),
             "candidate_coverage_mismatch")
    extraction, extraction_binding = _read_bound(root / "EXTRACTION.json")
    _require(extraction.get("complete") is True and extraction.get("records") == EXPECTED_RECORDS,
             "extraction_incomplete")
    for name in ("candidates.jsonl", "source_evidence.jsonl"):
        entry = extraction["artifacts"][name]
        _require(file_sha(entry["path"]) == entry["sha256"], "extraction_artifact_changed:" + name)
    extraction_validation, validation_binding = _read_bound(root / "EXTRACTION_VALIDATION.json")
    _require(extraction_validation.get("tests_passed") is True
             and extraction_validation.get("all_688_evidence_sha_and_context_sha_recomputed") is True
             and extraction_validation.get("all_record_member_keys_match") is True
             and extraction_validation.get("all_records_have_exactly_one_target_image") is True
             and extraction_validation.get("candidates_sha256") == source["sha256"]
             and extraction_validation.get("source_evidence_sha256") == extraction["artifacts"]["source_evidence.jsonl"]["sha256"],
             "extraction_validation_not_passed_or_binding_changed")
    interruption, interruption_binding = _read_bound(root / "INTERRUPTION_TEST.json")
    _require(interruption.get("passed") is True, "interruption_resume_test_not_passed")
    source_audit, source_audit_binding = _read_bound(root / "SOURCE_AUDIT.json")
    _require(source_audit.get("passed") is True
             and all(source_audit.get("coverage", {}).get(key) == EXPECTED_RECORDS
                     for key in ("expected", "observed", "unique_item_ids"))
             and source_audit["input_artifacts"]["candidates"]["sha256"] == source["sha256"]
             and source_audit["input_artifacts"]["source_evidence"]["sha256"] == extraction["artifacts"]["source_evidence.jsonl"]["sha256"],
             "independent_source_audit_not_passed_or_binding_changed")
    _require(Path(extraction["artifacts"]["candidates.jsonl"]["path"]).resolve() == candidates_path,
             "extraction_candidate_path_mismatch")
    evidence = _jsonl(extraction["artifacts"]["source_evidence.jsonl"]["path"])
    evidence_by_id = {row["item_id"]: row for row in evidence}
    _require(len(evidence_by_id) == len(evidence) == EXPECTED_RECORDS and set(evidence_by_id) == set(by_id),
             "source_evidence_coverage_mismatch")
    for item_id, candidate in by_id.items():
        record = evidence_by_id[item_id]
        _require(digest(record) == candidate["source_evidence_sha256"]
                 and digest(candidate["source_context"]) == record["context_sha256"]
                 and record["source_ref"] == candidate["source_ref"]
                 and record["review_number"] == candidate["review_number"],
                 "source_evidence_content_mismatch:" + item_id)
    human_binding = extraction["human_overrides"]
    _require(file_sha(human_binding["path"]) == human_binding["sha256"], "human_source_hash_mismatch")
    with gzip.open(human_binding["path"], "rb") as stream:
        human_rows = [strict_json(line) for line in stream]
    humans = {row["item_id"]: row["human_review"] for row in human_rows}
    _require(len(humans) == len(human_rows) == EXPECTED_RECORDS and set(humans) == set(by_id)
             and all(humans[key] == by_id[key]["human_review"] for key in humans), "frozen_human_source_mismatch")
    db = sqlite3.connect((root / "state.sqlite").as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        metadata = {key: strict_json(value) for key, value in db.execute("SELECT key,value FROM metadata")}
        statuses = dict(db.execute("SELECT status,count(*) FROM items GROUP BY status"))
        _require(statuses == {"succeeded": EXPECTED_RECORDS}, "database_not_completely_succeeded")
        _require(metadata["complete"] == completion and metadata["source"] == source,
                 "database_completion_source_mismatch")
        runtime = metadata["runtime_binding"]
        _require(digest(runtime) == manifest["runtime_binding_sha256"] == audit["runtime_binding_sha256"],
                 "runtime_binding_mismatch")
        _require(runtime["code"].get(str(Path(worker.__file__).resolve())) == file_sha(worker.__file__),
                 "worker_code_changed_since_inference")
        db_rows = {item_id: (input_sha, result_sha, batch_id)
                   for item_id, input_sha, result_sha, batch_id in db.execute(
                       "SELECT item_id,input_sha256,result_sha256,batch_id FROM items")}
    finally:
        db.close()
    output, seen = [], set()
    for shard in manifest["shards"]:
        path = (Path(manifest_binding["path"]).parent / shard["path"]).resolve(strict=True)
        _require(path.is_relative_to(Path(manifest_binding["path"]).parent), "shard_path_outside_export")
        _require(path.stat().st_size == shard["compressed_bytes"] and file_sha(path) == shard["sha256"],
                 "shard_hash_or_size_mismatch")
        count = 0
        with gzip.open(path, "rb") as stream:
            for line in stream:
                exported = strict_json(line)
                _require(set(exported) == {"item_id", "batch_id", "source", "result"}, "export_fields_mismatch")
                item_id = exported["item_id"]
                _require(item_id in by_id and item_id not in seen, "export_unknown_or_duplicate_id")
                source_item, result = exported["source"], exported["result"]
                _require(source_item == by_id[item_id], "export_candidate_mismatch")
                worker.validate_result(result, item_id, source_item)
                _require(db_rows[item_id] == (digest(source_item), digest(result), exported["batch_id"]),
                         "export_database_row_mismatch")
                output.append({"candidate": source_item, "result": result,
                               "source_evidence": evidence_by_id[item_id], "batch_id": exported["batch_id"]})
                seen.add(item_id)
                count += 1
        _require(count == shard["records"] == shard["outputs"], "shard_count_mismatch")
    _require(seen == set(by_id) and len(output) == EXPECTED_RECORDS, "export_exact_coverage_mismatch")
    _require(file_sha(root / "COMPLETE.json") == complete_binding["sha256"], "completion_changed")
    output.sort(key=lambda row: row["candidate"]["review_number"])
    bindings = {"completion": complete_binding, "manifest": manifest_binding, "strict_audit": audit_binding,
                "extraction": extraction_binding, "source_manifest": source,
                "extraction_validation": validation_binding, "interruption_test": interruption_binding,
                "independent_source_audit": source_audit_binding,
                "source_evidence": extraction["artifacts"]["source_evidence.jsonl"],
                "human_overrides": extraction["human_overrides"],
                "runtime_binding_sha256": digest(runtime)}
    return output, bindings


def original_image(candidate):
    """Read exact immutable TAR payload and validate dimensions without rewriting."""
    ref = candidate["source_ref"]
    path = Path(ref["path"])
    before = path.stat()
    if "source_container_bytes" in candidate:
        _require(before.st_size == candidate["source_container_bytes"]
                 and before.st_mtime_ns == candidate["source_mtime_ns"], "source_tar_stat_changed")
    offset, length = ref["offset"], ref["length"]
    _require(type(offset) is int and type(length) is int and offset >= 0 and 0 < length <= 128 << 20
             and offset + length <= before.st_size, "invalid_image_byte_range")
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(length)
    _require(len(raw) == length and _sha(raw) == candidate["item_id"] == ref["sha256"],
             "original_image_hash_mismatch")
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "tar_changed_during_read")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        fmt, width, height = image.format, image.width, image.height
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif",
            "BMP": "image/bmp"}.get(fmt)
    _require(mime is not None, "unsupported_original_browser_image_format:" + str(fmt))
    return {"raw": raw, "mime": mime, "format": fmt, "width": width, "height": height,
            "sha256": _sha(raw), "bytes": len(raw)}


def exclusion_cause(result):
    if result["recommendation"] not in {"exclude_pair", "drop_image_keep_text"}:
        return "none"
    # Lost information takes precedence: a missing portrait subject does not
    # establish that the original caption was attached to the wrong image.
    return "image_information_destroyed" if result["image_quality"] == "unusable" else "model_image_text_mismatch"


def make_overlay(row):
    candidate, result = row["candidate"], row["result"]
    old = candidate["human_review"]["decision"]
    recommendation = result["recommendation"]
    exclusions = {"exclude_pair", "drop_image_keep_text"}
    retention = {"keep", "flag_low_information"}
    disagreement = ((old == "keep" and recommendation in exclusions)
                    or (old in {"exclude_quality", "exclude_ai"} and recommendation in retention))
    return {"schema_version": "corpus.context_quality.overlay.v1", "item_id": candidate["item_id"],
            "review_number": candidate["review_number"], "source_name": candidate["source_name"],
            "task_type": candidate["task_type"], "source_ref": candidate["source_ref"],
            "source_context": candidate["source_context"],
            "source_context_sha256": digest(candidate["source_context"]),
            "source_evidence_sha256": candidate["source_evidence_sha256"],
            "human_review": candidate["human_review"], "context_review": result,
            "exclusion_cause": exclusion_cause(result),
            "context_result_sha256": digest(result), "batch_id": row["batch_id"],
            "disagrees_with_human": disagreement,
            "human_ai_exclusion_preserved": old == "exclude_ai", "human_decision_modified": False,
            "ai_provenance_reassessed": False, "source_certification": False,
            "scope": "target_image_and_supplied_source_text_only", "final_training_publication": False,
            "application_note": ("人工 AI 排除仍然有效；本次内容质量建议不能解除它。" if old == "exclude_ai" else
                                 "独立的上下文质量建议；尚未覆盖原人工判定或发布训练数据。")}


def summarize(rows):
    recommendations = Counter()
    causes = Counter()
    quality = Counter()
    human = Counter()
    tasks, sources, crossing = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    kept_to_exclude, quality_to_usable, ai_to_usable, need_context, examples = [], [], [], [], []
    for row in rows:
        source, result = row["candidate"], row["result"]
        rec = result["recommendation"]
        old = source["human_review"]["decision"]
        recommendations[rec] += 1
        cause = exclusion_cause(result)
        if cause != "none":
            causes[cause] += 1
        quality[result["training_quality"]] += 1
        human[old] += 1
        tasks[source["task_type"]][rec] += 1
        sources[source["source_name"]][rec] += 1
        crossing[old][rec] += 1
        brief = {"review_number": source["review_number"], "item_id": source["item_id"],
                 "source_name": source["source_name"], "task_type": source["task_type"],
                 "human_decision": old, "recommendation": rec,
                 "exclusion_cause": cause,
                 "context_interpretation": result["context_interpretation"], "reason": result["reason"]}
        if old == "keep" and rec in {"exclude_pair", "drop_image_keep_text"}:
            kept_to_exclude.append(brief)
        if old == "exclude_quality" and rec in {"keep", "flag_low_information"}:
            quality_to_usable.append(brief)
        if old == "exclude_ai" and rec in {"keep", "flag_low_information"}:
            ai_to_usable.append(brief)
        if rec == "needs_context":
            need_context.append(brief)
        if source["review_number"] in {191, 247, 309, 598, 597, 486, 168, 1}:
            examples.append(brief)
    return {"records": len(rows), "recommendations": dict(recommendations), "exclusion_causes": dict(causes), "training_quality": dict(quality),
            "original_human_decisions": dict(human),
            "by_task_type": {k: dict(v) for k, v in tasks.items()},
            "by_source": {k: dict(v) for k, v in sources.items()},
            "human_context_cross": {k: dict(v) for k, v in crossing.items()},
            "human_kept_now_proposed_exclusions": kept_to_exclude,
            "human_quality_excluded_now_proposed_usable": quality_to_usable,
            "human_ai_excluded_now_contextually_usable_but_still_ai_excluded": ai_to_usable,
            "needs_context": need_context, "examples": examples}


def safe_url(value):
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        return None
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme.lower() in {"http", "https"} and parsed.netloc else None
    except ValueError:
        return None


def esc(value):
    return html.escape(str(value), quote=True)


def _pre(value):
    return "<pre>" + esc(value) + "</pre>"


def render_card(row, picture):
    candidate, result, evidence = row["candidate"], row["result"], row["source_evidence"]
    number, source, task = candidate["review_number"], candidate["source_name"], candidate["task_type"]
    old, rec = candidate["human_review"]["decision"], result["recommendation"]
    context = candidate["source_context"]
    attrs = {"review": number, "source": source, "task": task, "human": old, "rec": rec}
    out = ['<details class="card" id="item-' + str(number) + '" '
           + " ".join('data-' + key + '="' + esc(value) + '"' for key, value in attrs.items()) + ">",
           "<summary><b>#" + str(number) + "</b> · " + esc(source) + " · " + esc(LABELS.get(task, task))
           + ' <span class="badge">原人工：' + esc("保留" if old == "keep" else LABELS.get(old, old)) + "</span>"
           + ' <span class="badge new">上下文：' + esc(LABELS.get(rec, rec)) + "</span></summary>",
           '<section class="body"><div class="image-panel"><img alt="原始目标图 #' + str(number)
           + '" data-src="data:' + picture["mime"] + ";base64," + base64.b64encode(picture["raw"]).decode()
           + '"><p>' + str(picture["width"]) + " × " + str(picture["height"]) + " · " + str(picture["bytes"])
           + " 字节 · 原始 " + esc(picture["format"]) + "；未重采样、未重编码。</p></div>",
           '<div class="analysis"><h3>上下文质量意见</h3><dl>',
           "<dt>上游在讲什么</dt><dd>" + esc(result["context_interpretation"]) + "</dd>",
           "<dt>图中可见什么</dt><dd>" + esc(result["visible_content"]) + "</dd>",
           "<dt>判断依据</dt><dd>" + esc(result["reason"]) + "</dd>",
           "<dt>排除主因</dt><dd>" + esc(LABELS[exclusion_cause(result)]) + "</dd>",
           "<dt>质量 / 对齐 / 训练用途</dt><dd>" + esc(result["image_quality"] + " / " + result["alignment"]
                                                                  + " / " + result["training_quality"]) + "</dd></dl><ul>"]
    for entry in result["evidence"]:
        out.append("<li>" + esc(entry["kind"]) + "：" + esc(entry["detail"]) + "</li>")
    out.append("</ul><p class=note>" + esc(make_overlay(row)["application_note"]) + "</p></div>")
    out.append("<h3>完整原始文本与位置</h3><p>context_status=" + esc(context["context_status"]) + "</p>")
    for heading, key in (("Caption", "caption_texts"), ("对话 / 区域问答", "conversation_texts")):
        for index, text in enumerate(context[key], 1):
            out.append("<h4>" + heading + " " + str(index) + "</h4>" + _pre(text))
    images = context["metadata_summary"].get("record_images", [])
    if task == "interleaved":
        out.append('<p class="note">下方只显示本条目标图。其他图像仅列位置，未查看其像素，不能据此认证整个序列。</p>')
    slots = [(entry.get("sequence_index"), "text", entry) for entry in context["interleaved_segments"]]
    slots += [(entry.get("sequence_index"), "image", entry) for entry in images]
    slots.sort(key=lambda entry: (entry[0] is None, entry[0] or 0, entry[1], entry[2]["member"]))
    for position, kind, value in slots:
        if kind == "text":
            out.append("<h4>文本位置 " + esc(position) + " · " + esc(value["member"]) + "</h4>" + _pre(value["text"]))
        else:
            title = "【当前目标图，已显示】" if value.get("is_target") else "【其他图像，未显示 / 未复核】"
            out.append('<p class="slot">' + title + " 位置 " + esc(position) + " · " + esc(value["member"]) + "</p>")
    if context["context_limitations"]:
        out.append("<h4>已知限制</h4>" + _pre(json.dumps(context["context_limitations"], ensure_ascii=False, indent=2)))
    out.append("<details><summary>原始 metadata 与可追溯位置</summary>" + _pre(json.dumps(context["metadata_summary"], ensure_ascii=False, indent=2)))
    out.append(_pre(json.dumps(candidate["source_ref"], ensure_ascii=False, indent=2)))
    link = safe_url(candidate["source_ref"].get("source_url") or context["metadata_summary"].get("url"))
    if link:
        out.append('<p><a target="_blank" rel="noopener noreferrer" href="' + esc(link) + '">打开数据记录中的来源链接</a>（远端内容未用于本次判断）</p>')
    out.append("</details><details><summary>同记录完整源文本文件（审计原文，含 JSON metadata）</summary>")
    for member in evidence["members"]:
        if "text" in member:
            out.append("<h4>" + esc(member["member"]) + "</h4>" + _pre(member["text"]))
    out.append("</details><details><summary>原人工判定（独立保留）</summary>"
               + _pre(json.dumps(candidate["human_review"], ensure_ascii=False, indent=2)) + "</details>")
    out.append("<p class=id>图像 SHA256：" + esc(candidate["item_id"]) + "</p></section></details>")
    return "\n".join(out)


STYLE = """body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#17212b}main{max-width:1280px;margin:auto;padding:20px}h1{font-size:26px}.intro,.filters,.card{background:#fff;border:1px solid #d8dfe6;border-radius:8px;padding:14px;margin:12px 0}.filters{position:sticky;top:0;z-index:3;box-shadow:0 2px 8px #0001;display:flex;gap:8px;flex-wrap:wrap}input,select,button{font:inherit;padding:6px;max-width:100%}summary{cursor:pointer}.card>summary{padding:8px}.badge{display:inline-block;font-size:13px;background:#edf0f3;border-radius:4px;padding:2px 6px;margin:2px}.new{background:#e4f1ff}.body{border-top:1px solid #ddd;padding-top:16px}img{display:block;max-width:100%;max-height:80vh;height:auto;width:auto;margin:auto}.image-panel{text-align:center;background:#f7f7f7;padding:10px}.analysis{padding:12px 0}dt{font-weight:700}dd{margin:0 0 8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:12px;border-radius:4px;max-height:60vh;overflow:auto}.note{color:#7c4b03;background:#fff6df;padding:8px}.slot{padding:10px;background:#e8f0f7}.id{font:12px monospace;overflow-wrap:anywhere}.card[hidden]{display:none}details details{border-top:1px solid #ddd;padding:10px 0}.count{font-weight:700}a{color:#1262a4}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}"""
SCRIPT = """const cards=[...document.querySelectorAll('.card')];const controls=['q','source','task','human','rec'].map(id=>document.getElementById(id));const searchable=new Map(cards.map(c=>[c,c.textContent.toLocaleLowerCase()]));function filter(){const [q,source,task,human,rec]=controls.map(c=>c.value.trim().toLocaleLowerCase());let n=0;for(const c of cards){const show=(!q||searchable.get(c).includes(q))&&(!source||c.dataset.source.toLocaleLowerCase()===source)&&(!task||c.dataset.task.toLocaleLowerCase()===task)&&(!human||c.dataset.human.toLocaleLowerCase()===human)&&(!rec||c.dataset.rec.toLocaleLowerCase()===rec);c.hidden=!show;if(show)n++;}document.getElementById('count').textContent=n+' / '+cards.length+' 项';}let timer;for(const c of controls)c.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(filter,120);});for(const c of cards)c.addEventListener('toggle',()=>{if(c.open){const img=c.querySelector('img[data-src]');if(img&&!img.src)img.src=img.dataset.src;}});document.getElementById('collapse').addEventListener('click',()=>{for(const c of cards)c.open=false;});document.getElementById('reset').addEventListener('click',()=>{for(const c of controls)c.value='';filter();});if(location.hash){const c=document.getElementById(location.hash.slice(1));if(c&&c.classList.contains('card')){c.open=true;setTimeout(()=>c.scrollIntoView(),50);}}filter();"""


def write_html(path, rows, summary):
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DCVLM 688 项：原始上下文与训练质量复核</title><style>' + STYLE + "</style><main><h1>DCVLM 688 项：原始上下文与训练质量复核</h1>")
        stream.write('<div class="intro"><p>' + esc(DISCLAIMER) + "</p><p>每项均附源 TAR 的原始图像字节及完整文本。图像按需展开加载；全部内容可离线查看。点击来源链接才会访问网络。</p><p>"
                     + " · ".join(esc(LABELS.get(key, key)) + "：" + str(value) for key, value in summary["recommendations"].items()) + "</p></div>")
        stream.write('<div class="filters"><input id="q" placeholder="搜索编号、来源或完整原文" aria-label="搜索">')
        fields = [("source", "来源", sorted(summary["by_source"])), ("task", "任务类型", sorted(summary["by_task_type"])),
                  ("human", "原人工判定", sorted(summary["original_human_decisions"])),
                  ("rec", "新上下文建议", sorted(summary["recommendations"]))]
        for key, name, values in fields:
            stream.write('<select id="' + key + '" aria-label="' + name + '"><option value="">全部' + name + "</option>")
            for value in values:
                label = "人工保留" if key == "human" and value == "keep" else LABELS.get(value, value)
                stream.write('<option value="' + esc(value) + '">' + esc(label) + "</option>")
            stream.write("</select>")
        stream.write('<button id="reset">重置筛选</button><button id="collapse">全部折叠</button><span id="count" class="count"></span></div>')
        image_audit = []
        for row in rows:
            picture = original_image(row["candidate"])
            stream.write(render_card(row, picture))
            image_audit.append({"item_id": row["candidate"]["item_id"], "review_number": row["candidate"]["review_number"],
                                **{k: v for k, v in picture.items() if k != "raw"}})
        stream.write("</main><script>" + SCRIPT + "</script></html>")
    return image_audit


CSV_FIELDS = ["review_number", "item_id", "source_name", "task_type", "original_human_decision",
              "context_recommendation", "exclusion_cause", "image_quality", "alignment", "training_quality",
              "context_interpretation", "visible_content", "reason", "disagrees_with_human",
              "human_ai_exclusion_preserved", "source_tar", "source_member", "source_url"]


def csv_safe(value):
    text = str(value)
    # Spreadsheet software must not execute untrusted source strings as formulas.
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r")) else text


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            candidate, result = row["candidate"], row["result"]
            overlay = make_overlay(row)
            values = {key: candidate[key] for key in ("review_number", "item_id", "source_name", "task_type")}
            values.update(original_human_decision=candidate["human_review"]["decision"], context_recommendation=result["recommendation"],
                          exclusion_cause=exclusion_cause(result),
                          disagrees_with_human=overlay["disagrees_with_human"], human_ai_exclusion_preserved=overlay["human_ai_exclusion_preserved"],
                          source_tar=candidate["source_ref"]["path"], source_member=candidate["source_ref"]["member"],
                          source_url=candidate["source_ref"].get("source_url", ""))
            values.update({key: result[key] for key in ("image_quality", "alignment", "training_quality", "context_interpretation", "visible_content", "reason")})
            writer.writerow({key: csv_safe(value) for key, value in values.items()})


def _md(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(summary, destination):
    lines = ["# DCVLM 688 项：原始上下文与训练质量复核", "", DISCLAIMER, "",
             "所有 688 项均按原始图像与源文本逐项复核，原人工判定保留为独立列。", "",
             "| 上下文建议 | 数量 |", "|---|---:|"]
    lines += ["| " + LABELS.get(key, key) + " | " + str(value) + " |" for key, value in summary["recommendations"].items()]
    lines.extend(["", "排除主因分层：" + "；".join(LABELS[key] + " " + str(value) + " 项" for key, value in summary["exclusion_causes"].items()) + "。",
                  "图像信息损伤优先于对齐标签计数：即使模型同时输出 mismatched，主体已被抹除的案例也不能表述为已证明原始图文配错。",
                  "其余图文不符项同样只是针对当前图像的模型建议，不能证明上游历史配错图；模型也可能低估视觉损伤。"])
    recs = worker.RECOMMENDATIONS
    for title, key in (("按任务类型", "by_task_type"), ("按来源", "by_source"), ("原人工判定 × 新建议", "human_context_cross")):
        lines.extend(["", "## " + title, "", "| 类别 | " + " | ".join(LABELS[r] for r in recs) + " | 总数 |",
                      "|---|" + "---:|" * (len(recs) + 1)])
        for name, counts in summary[key].items():
            label = "人工保留" if key == "human_context_cross" and name == "keep" else LABELS.get(name, name)
            lines.append("| " + _md(label) + " | " + " | ".join(str(counts.get(rec, 0)) for rec in recs) + " | " + str(sum(counts.values())) + " |")
    conflict_sections = [("原人工保留，本次建议排除图文对或删除目标图槽", "human_kept_now_proposed_exclusions"),
                         ("原人工质量排除，本次认为仍可用（不自动覆盖）", "human_quality_excluded_now_proposed_usable"),
                         ("原人工 AI 排除的内容质量可用项（AI 排除仍然有效）", "human_ai_excluded_now_contextually_usable_but_still_ai_excluded"),
                         ("仍需补充上下文", "needs_context")]
    for title, key in conflict_sections:
        values = summary[key]
        lines.extend(["", "## " + title, "", "共 " + str(len(values)) + " 项。编号："
                      + ("、".join("#" + str(v["review_number"]) for v in values) if values else "无") + "。", ""])
        if values:
            lines += ["| 编号 | 来源 / 任务 | 上游在讲什么 | 建议与理由 |", "|---:|---|---|---|"]
            for value in values:
                lines.append("| #" + str(value["review_number"]) + " | " + _md(value["source_name"] + " / " + value["task_type"])
                             + " | " + _md(value["context_interpretation"]) + " | " + _md(LABELS[value["recommendation"]] + "：" + value["reason"]) + " |")
    lines.extend(["", "## 代表样例", "", "| 编号 | 来源 / 任务 | 上游在讲什么 | 建议与理由 |", "|---:|---|---|---|"])
    for value in summary["examples"]:
        lines.append("| #" + str(value["review_number"]) + " | " + _md(value["source_name"] + " / " + value["task_type"])
                     + " | " + _md(value["context_interpretation"]) + " | " + _md(LABELS[value["recommendation"]] + "：" + value["reason"]) + " |")
    lines += ["", "## 结果文件", "", "- [离线逐图复核页面](" + str(destination / "review.html") + ")：原图、完整原文、图槽位置及旧人工/新质量意见。",
              "- [CSV 索引](" + str(destination / "per_item.csv") + ")：688 行摘要，可筛选。",
              "- [独立上下文质量层](" + str(destination / "context_quality_overlay.jsonl.gz") + ")：包含完整源文本、原人工判定、新建议和证据绑定。",
              "- [离线页面 ZIP](" + str(destination / "review.zip") + ")。", "",
              "严格完成审计、导出与 SQLite 记录逐项比对、完整覆盖校验通过后才生成本报告。",
              "原图按源 TAR 字节区间读取并核验 SHA256，未重采样或重新编码。", ""]
    return "\n".join(lines)


def build_report(root, output=None):
    root = Path(root).resolve(strict=True)
    destination = Path(output).resolve() if output else root / "report"
    _require(not destination.exists(), "report_output_already_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
    with exclusive(root):
        rows, bindings = load_completed(root)
        stage.mkdir()
        summary = summarize(rows)
        _require(summary["original_human_decisions"] == {"exclude_ai": 1, "keep": 225, "exclude_quality": 462},
                 "frozen_human_decision_scope_mismatch")
        write_csv(stage / "per_item.csv", rows)
        with gzip.open(stage / "context_quality_overlay.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as stream:
            for row in rows:
                stream.write(canonical(make_overlay(row)) + "\n")
        image_audit = write_html(stage / "review.html", rows, summary)
        (stage / "REPORT.md").write_text(render_markdown(summary, destination), encoding="utf-8")
        with zipfile.ZipFile(stage / "review.zip", "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(stage / "review.html", "review.html")
        with gzip.open(stage / "context_quality_overlay.jsonl.gz", "rt", encoding="utf-8") as stream:
            overlays = [strict_json(line) for line in stream]
        _require(overlays == [make_overlay(row) for row in rows], "overlay_roundtrip_mismatch")
        with (stage / "per_item.csv").open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        _require(len(csv_rows) == EXPECTED_RECORDS
                 and [row["item_id"] for row in csv_rows] == [row["candidate"]["item_id"] for row in rows],
                 "csv_coverage_mismatch")
        report = {"schema_version": SCHEMA, "complete": True,
                  "created_at_beijing": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  "scope": DISCLAIMER, "source_bindings": bindings, "summary": summary,
                  "human_decisions_modified": False, "source_images_modified": False,
                  "ai_provenance_reassessed": False, "final_training_publication": False,
                  "generator_sha256": file_sha(__file__),
                  "artifacts": {name: {"path": str(destination / name), "bytes": (stage / name).stat().st_size,
                                        "sha256": file_sha(stage / name)} for name in
                                ("REPORT.md", "per_item.csv", "context_quality_overlay.jsonl.gz", "review.html", "review.zip")}}
        (stage / "REPORT.json").write_text(canonical(report) + "\n", encoding="utf-8")
        report_audit = {"schema_version": SCHEMA, "passed": True, "records": len(rows), "strict_source_audit_verified": True,
                        "source_export_database_equality_checked": True, "source_context_full_coverage_checked": True,
                        "csv_roundtrip_checked": True, "overlay_roundtrip_checked": True,
                        "html_original_image_hashes_verified": True, "html_resampled_or_reencoded": False,
                        "image_records": image_audit, "report_sha256": file_sha(stage / "REPORT.json")}
        (stage / "AUDIT.json").write_text(canonical(report_audit) + "\n", encoding="utf-8")
        (stage / "COMPLETE.json").write_text(canonical({"complete": True, "records": len(rows),
            "report_sha256": file_sha(stage / "REPORT.json"), "audit_sha256": file_sha(stage / "AUDIT.json"),
            "source_complete_sha256": bindings["completion"]["sha256"]}) + "\n", encoding="utf-8")
        _require(file_sha(root / "COMPLETE.json") == bindings["completion"]["sha256"], "source_completion_changed_during_report")
        os.replace(stage, destination)
    return {"directory": str(destination), "records": len(rows), "recommendations": summary["recommendations"],
            "report": str(destination / "REPORT.md"), "offline_html": str(destination / "review.html")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(canonical(build_report(args.root, args.output)))


if __name__ == "__main__":
    main()
