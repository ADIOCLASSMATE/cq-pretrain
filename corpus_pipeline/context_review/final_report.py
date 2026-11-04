"""Combine immutable context passes and targeted visual adjudication for review.

Produces recommendations only. Does not rewrite corpus data or human decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import sqlite3
import zipfile

from corpus_pipeline.context_review import report as base
from corpus_pipeline.codex_review.farm import canonical, digest, file_sha

PARTIAL = "retain_caption_prune_invalid_qa"
LABELS = {**base.LABELS, PARTIAL: "保留原图/主caption，局部QA需修剪"}
EXCLUSIONS = {"exclude_pair", "drop_image_keep_text"}
RETENTION = {"keep", "flag_low_information"}
PROVENANCE_FOLLOWUP = {
    548: "GAN实验语境明确，但目标子图及纯生成/真人编辑路径未确认，需图槽级来源核验。",
    478: "数据集蒸馏论文实验图，需确认目标是原样本、初始化还是优化后的图像。",
    489: "数据集蒸馏论文实验图，需确认目标是原样本、初始化还是优化后的图像。",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def load_recheck(parent, initial):
    root = parent / "exclusion_recheck"
    completion = base.strict_json((root / "COMPLETE.json").read_bytes())
    require(completion.get("complete") is True and completion.get("records") == 208
            and completion.get("audit_mode") == "strict" and completion.get("skipped_gates") == [],
            "recheck_not_strictly_complete")
    manifest, _ = base._read_bound(completion["manifest"], completion["manifest_sha256"])
    audit, _ = base._read_bound(completion["audit"], completion["audit_sha256"])
    require(audit.get("passed") is True and audit.get("complete_coverage") is True
            and audit.get("mode") == "strict" and audit.get("skipped_gates") == []
            and audit.get("context_items_checked") == 208 and audit.get("export_checked") is True
            and audit.get("initial_exclusion_scope_checked") is True
            and audit.get("export_manifest_sha256") == completion["manifest_sha256"], "recheck_audit_failed")
    require(manifest.get("records") == manifest.get("outputs") == manifest.get("source", {}).get("count") == 208
            and manifest.get("complete") is True and manifest.get("model") == base.worker.MODEL
            and manifest.get("reasoning_effort") == base.worker.EFFORT, "recheck_manifest_contract")
    interruption = base.strict_json((root / "INTERRUPTION_TEST.json").read_bytes())
    require(interruption.get("passed") is True, "recheck_interruption_test_missing")
    selection = base.strict_json((root / "SELECTION.json").read_bytes())
    require(selection.get("passed") is True and selection["output"]["sha256"] == manifest["source"]["sha256"]
            and file_sha(root / "candidates.jsonl") == manifest["source"]["sha256"]
            and file_sha(root / "SELECTION.json") == audit["selection_sha256"], "recheck_selection_changed")
    expected = {r["candidate"]["item_id"]: r for r in initial if r["result"]["recommendation"] in EXCLUSIONS}
    require(len(expected) == 208, "initial_exclusion_count_changed")
    db = sqlite3.connect((root / "state.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        require(dict(db.execute("SELECT status,count(*) FROM items GROUP BY status")) == {"succeeded": 208},
                "recheck_database_incomplete")
        database = {i: (inp, res, b) for i, inp, res, b in db.execute(
            "SELECT item_id,input_sha256,result_sha256,batch_id FROM items")}
        metadata = {k: base.strict_json(v) for k, v in db.execute("SELECT key,value FROM metadata")}
        require(metadata["complete"] == completion and metadata["source"] == manifest["source"], "recheck_db_binding")
        require(digest(metadata["runtime_binding"]) == manifest["runtime_binding_sha256"], "recheck_runtime_binding")
    finally:
        db.close()
    found = {}
    for shard in manifest["shards"]:
        path = Path(completion["manifest"]).parent / shard["path"]
        require(file_sha(path) == shard["sha256"], "recheck_shard_hash")
        count = 0
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = base.strict_json(line)
                item_id = row["item_id"]
                require(item_id in expected and item_id not in found, "recheck_duplicate_or_extra")
                require(row["source"] == expected[item_id]["candidate"], "recheck_source_changed")
                require(database[item_id] == (digest(row["source"]), digest(row["result"]), row["batch_id"]),
                        "recheck_result_db_mismatch")
                base.worker.validate_result(row["result"], item_id, row["source"])
                found[item_id] = row
                count += 1
        require(count == shard["records"], "recheck_shard_count")
    require(set(found) == set(expected), "recheck_coverage")
    return found, {name: bound(root / name) for name in ("COMPLETE.json", "SELECTION.json", "INTERRUPTION_TEST.json")}


def combine(initial, recheck, adjudications):
    by_id = {r["candidate"]["item_id"]: r for r in initial}
    manual = {}
    for entry in adjudications["entries"]:
        item_id = entry["item_id"]
        require(item_id in by_id and item_id not in manual, "bad_manual_coverage")
        original = by_id[item_id]
        require(entry["review_number"] == original["candidate"]["review_number"]
                and entry["source_input_sha256"] == digest(original["candidate"])
                and entry["source_evidence_sha256"] == original["candidate"]["source_evidence_sha256"]
                and entry["raw_model_result_sha256"] == digest(original["result"]), "manual_binding_mismatch")
        require(entry["original_bytes_sha_verified"] is True and entry["full_original_image_actually_viewed"] is True,
                "manual_visual_evidence_missing")
        require(entry["final_recommendation"] in RETENTION | EXCLUSIONS | {PARTIAL}, "bad_manual_action")
        manual[item_id] = entry
    output = []
    for row in initial:
        c, first = row["candidate"], row["result"]
        item_id = c["item_id"]
        second = recheck.get(item_id)
        effective = dict(second["result"] if second else first)
        layer = "lenient_recheck" if second else "initial_context_review"
        targeted = manual.get(item_id)
        if targeted:
            layer = "independent_targeted_visual_adjudication"
            action = targeted["final_recommendation"]
            independent = targeted["independent_assessment"]
            effective.update(recommendation=action, reason=targeted["reason"], evidence=targeted["evidence"],
                visible_content=independent["visual_evidence"],
                image_quality=("unusable" if action in EXCLUSIONS else independent.get("suggested_image_quality",
                    "usable" if targeted["review_number"] in {233,258,665} else "degraded_usable")),
                alignment=independent["suggested_alignment"],
                training_quality="unusable" if action in EXCLUSIONS else "usable_low_information",
                human_review_not_used=False)
        output.append({"candidate": c, "source_evidence": row["source_evidence"], "result": effective,
            "batch_id": row["batch_id"], "effective_layer": layer,
            "initial_context_review": first, "lenient_recheck": second,
            "targeted_adjudication": targeted})
    return output


def summary(rows):
    counts, tasks, cross, sources = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        c, v = row["candidate"], row["result"]
        rec = v["recommendation"]
        counts[rec] += 1
        tasks[c["task_type"]][rec] += 1
        cross[c["human_review"]["decision"]][rec] += 1
        sources[c["source_name"]][rec] += 1
    return {"records": len(rows), "recommendations": dict(counts),
            "by_task_type": {k: dict(v) for k, v in tasks.items()},
            "human_context_cross": {k: dict(v) for k, v in cross.items()},
            "by_source": {k: dict(v) for k, v in sources.items()},
            "original_human_decisions": dict(Counter(r["candidate"]["human_review"]["decision"] for r in rows))}


def write_viewer(path, rows, statistics):
    # The existing renderer escapes all source text and embeds verified original
    # bytes. Its raw-model records are preserved separately below each card.
    base.LABELS.update(LABELS)
    base.write_html(path, rows, statistics)
    text = path.read_text()
    text = text.replace("<h1>DCVLM 688 项：原始上下文与训练质量复核</h1>",
        "<h1>DCVLM 688 项：结合原文后的训练质量建议</h1><p class=note>这是两轮模型检查与重点原图复核的合并建议。原人工判定独立保留；未修改训练数据。"
        "#679 必须先局部修剪错误QA，不能直接放行整条对话。#548 另有GAN实验来源线索，#478/#489为数据集蒸馏实验图，质量可用不代表图像来源通过。"
        "首轮和二审原结果分别保留在每项末尾，不把合并意见冒充原始模型输出。</p>")
    panels = {}
    for row in rows:
        c = row["candidate"]
        layers = {"effective_layer": row["effective_layer"], "initial_context_review": row["initial_context_review"],
                  "lenient_recheck": row["lenient_recheck"]["result"] if row["lenient_recheck"] else None,
                  "targeted_adjudication": row["targeted_adjudication"],
                  "image_origin_followup": PROVENANCE_FOLLOWUP.get(c["review_number"])}
        panel = "<details><summary>各轮原始意见与独立复核（可追溯）</summary>" + base._pre(json.dumps(layers, ensure_ascii=False, indent=2)) + "</details>"
        panels[c["item_id"]] = panel
    seen = set()
    def insert(match):
        item_id = match.group(1)
        require(item_id in panels and item_id not in seen, "viewer_item_not_unique")
        seen.add(item_id)
        return panels[item_id] + match.group(0)
    text = re.sub(r"<p class=id>图像 SHA256：([0-9a-f]{64})</p>",insert,text)
    require(seen == set(panels),"viewer_item_missing")
    path.write_text(text)


def markdown(stats, destination, root):
    counts = stats["recommendations"]
    actions = ["keep", "flag_low_information", PARTIAL, "exclude_pair", "drop_image_keep_text", "needs_context"]
    lines = ["# DCVLM 人工复核队列：原始上下文与训练用途复查", "",
        "本次逐项检查原人工队列全部688张：583项图文对、9项对话/区域问答、96个交错序列目标图槽。"
        "逐项提供源TAR中原图和同记录完整文本；首轮之后，对全部208项排除建议另用宽松口径独立二审，并对争议项查看完整原图。", "",
        "当前阶段的标准是‘没有明确错误就通过’，这里的‘可用’表示对通用多模态预训练有价值，不等于高质量精细caption、人物身份认证或1M分辨率阶段准入。", "",
        "| 最终质量建议 | 数量 |", "|---|---:|"]
    lines += [f"| {LABELS[k]} | {counts.get(k,0)} |" for k in actions if counts.get(k)]
    lines += ["", "原人工AI排除的#1继续单独受AI门控限制，内容质量保留不解除它。#548存在GAN实验来源线索，#478/#489涉及数据集蒸馏实验图，需要按具体图槽核验；不能因为其适合解释论文实验便放过图像来源规则。", "",
              "## 按任务类型", "", "| 类型 | " + " | ".join(LABELS[k] for k in actions[:-1]) + " |", "|---|" + "---:|"*5]
    for task, values in stats["by_task_type"].items():
        lines.append("| " + LABELS.get(task,task) + " | " + " | ".join(str(values.get(k,0)) for k in actions[:-1]) + " |")
    lines += ["", "## 上游在讲什么，适合学什么", "",
        "- DataComp/ReLAION里的大量纹理来自建材、布料、墙纸、金属表面与素材商品页。#191混凝土、#247水彩底纹、#309拉丝钢的原caption就描述这些东西；适合材料、颜色与纹理识别。SKU/品牌堆砌、缺少具体属性的标题信息较弱，但不能只因画面简单而排除。",
        "- OmniCC本批65个目标里，47个来自NFL官方CDN、49个URL含/t_lazy/。正文是比赛、训练营、球员动态等，后接相册和摄影署名。可辨的运动场景仍有弱监督价值，但不是每张都有精准caption；不能把只有摄影署名的相册强制拆成精确I2T。/t_lazy/仅是原URL字面证据，未假设已经取回高清版本。",
        "- MINT PDF/HTML和textbook包含论文实验、报告、商品技术页和教材。#98是背景分离实验的掩模；#478/#489属于数据集蒸馏论文图；这类图必须结合原文判断，黑白块或小图拼接不等于坏图。装饰图、导航图则标低信息，不能据此删除整篇正文。",
        "- #597/#598的原文字段指向人物绘画/婴儿，但本地图已经近乎只剩色块。它们与真正的纹理商品图不同，当前图文对应的训练信息已严重受损。",
        "- #679的坐佛、蓝天、黄叶与主caption相容，错误集中在部分区域QA：把一尊说成两尊、把扇形叶描述成狭长尖叶等。建议保留原图和主caption，局部隔离错误QA及复述这些错误的区域caption；具体member/索引已记录，尚未改写数据。", "",
        "## 交错序列的边界", "",
        "96个已查看目标对应80条源record，record中共3731个图槽；本次只查看这些目标图，不能认证其他图，也不能把目标图可用等同于整条序列全部高质量。20条record超过64图，7条超过128图，最长332图；训练打包应保留文本/图片位置并分段，不直接硬截成丢失局部上下文的大图集。",
        "本批是检测器与复核流程筛出的困难样本，统计比例不能推算全DCVLM、全OmniCC或其真实AI率。", "",
        "## 对原人工意见的影响", "", "| 原人工结果 | " + " | ".join(LABELS[k] for k in actions[:-1]) + " |", "|---|" + "---:|"*5]
    for old, values in stats["human_context_cross"].items():
        lines.append("| " + ("人工保留" if old=="keep" else LABELS.get(old,old)) + " | " + " | ".join(str(values.get(k,0)) for k in actions[:-1]) + " |")
    lines += ["", "这些是新的独立质量建议，未覆盖此前人工决定、未删除源图、未发布最终训练集。模型判断仍可能有误，审计保证输入/输出与覆盖可追溯，不等于语义100%正确。", "",
        "## 建议后续管道采用的判定单位", "",
        "1. 先解析源结构：图文对保留caption，对话保留问题/回答，交错保留图槽与全文位置。",
        "2. 分离来源AI门控、视觉损伤、图文一致性、信息量。低信息和待核验不能混成错误；源文本说到生成实验时新增来源线索。",
        "3. 人工仅复核有明确争议的‘原图+原文+理由’，不要再只展示孤立图片。",
        "4. 错误落到最小单元：局部QA修剪，坏图只移除图槽；正文是否可作纯文本另行校验，不自动全篇删除或全篇通过。", "",
        "## 文件", "",
        f"- [688项原图、完整上下文与各轮意见]({destination / 'review.html'})",
        f"- [CSV逐项索引]({destination / 'per_item.csv'})",
        f"- [机器可读分层建议]({destination / 'context_quality_overlay.jsonl.gz'})",
        f"- [离线页面ZIP]({destination / 'review.zip'})",
        f"- [9项完整原图重点复核及679局部修剪索引]({root / 'TARGETED_ADJUDICATION.json'})",
        f"- [交错序列源分析]({root / 'INTERLEAVED_ASSESSMENT.md'})", ""]
    return "\n".join(lines)


def build(root):
    root = Path(root).resolve(strict=True)
    initial, bindings = base.load_completed(root)
    second, recheck_bindings = load_recheck(root, initial)
    adjudication = base.strict_json((root / "TARGETED_ADJUDICATION.json").read_bytes())
    require(len(adjudication["entries"]) == 9, "expected_nine_targeted_reviews")
    expected_manual = {r["candidate"]["item_id"] for r in initial
                       if r["result"]["recommendation"] in EXCLUSIONS and r["result"]["image_quality"] != "unusable"}
    require({e["item_id"] for e in adjudication["entries"]} == expected_manual,"targeted_selection_mismatch")
    for entry in adjudication["input_artifacts"].values():
        require(file_sha(entry["path"]) == entry["sha256"],"targeted_input_artifact_changed")
    additional = base.strict_json((root / "ADDITIONAL_ADJUDICATION.json").read_bytes())
    require({e["review_number"] for e in additional["entries"]} == {107,248,315,530,576},"additional_scope_mismatch")
    adjudication["entries"] += additional["entries"]
    rows = combine(initial, second, adjudication)
    require(len(rows) == 688 and len({r["candidate"]["item_id"] for r in rows}) == 688, "final_coverage")
    statistics = summary(rows)
    destination = root / "final_report"
    require(not destination.exists(), "final_report_already_exists")
    destination.mkdir()
    fields = ["review_number", "item_id", "source_name", "task_type", "original_human_decision",
              "initial_recommendation", "recheck_recommendation", "effective_recommendation", "effective_layer",
              "context_interpretation", "reason", "human_ai_exclusion_preserved", "partial_repair_required",
              "image_origin_followup"]
    overlays = []
    with (destination / "per_item.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            c,v = row["candidate"],row["result"]
            review = {k:c[k] for k in ["review_number","item_id","source_name","task_type"]}
            review.update(original_human_decision=c["human_review"]["decision"],
                initial_recommendation=row["initial_context_review"]["recommendation"],
                recheck_recommendation=row["lenient_recheck"]["result"]["recommendation"] if row["lenient_recheck"] else "",
                effective_recommendation=v["recommendation"],effective_layer=row["effective_layer"],
                context_interpretation=v["context_interpretation"],reason=v["reason"],
                human_ai_exclusion_preserved=c["human_review"]["decision"]=="exclude_ai",
                partial_repair_required=v["recommendation"]==PARTIAL,
                image_origin_followup=PROVENANCE_FOLLOWUP.get(c["review_number"], ""))
            writer.writerow({k:base.csv_safe(val) for k,val in review.items()})
            overlays.append({"schema_version":"corpus.context_review.final_advice.v1",**review,
                "source_input":c,"source_evidence_sha256":c["source_evidence_sha256"],
                "effective_assessment":v,"initial_context_review":row["initial_context_review"],
                "lenient_recheck":row["lenient_recheck"],"targeted_adjudication":row["targeted_adjudication"],
                "training_data_modified":False,"ai_origin_certified":False,"advice_only":True})
    with gzip.open(destination / "context_quality_overlay.jsonl.gz","wt",encoding="utf-8") as stream:
        for row in overlays:stream.write(canonical(row)+"\n")
    write_viewer(destination / "review.html",rows,statistics)
    (destination / "REPORT.md").write_text(markdown(statistics,destination,root))
    with zipfile.ZipFile(destination / "review.zip","x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(destination / "review.html","review.html")
    with gzip.open(destination / "context_quality_overlay.jsonl.gz","rt") as stream:
        require([base.strict_json(x) for x in stream]==overlays,"overlay_roundtrip")
    extras = {name:bound(root/name) for name in ("TARGETED_ADJUDICATION.json","ROOT_VISUAL_SPOTCHECK.json",
                                               "ADDITIONAL_ADJUDICATION.json",
                                               "INTERLEAVED_ASSESSMENT.json","INTERLEAVED_ASSESSMENT.md")}
    output = {"schema_version":"corpus.context_review.final_report.v1","complete":True,"records":688,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"source_bindings":bindings,
        "recheck_bindings":recheck_bindings,"independent_evidence":extras,"summary":statistics,
        "generator_sha256":file_sha(__file__),"training_data_modified":False,"human_decisions_modified":False,
        "artifacts":{name:bound(destination/name) for name in ("REPORT.md","per_item.csv","context_quality_overlay.jsonl.gz","review.html","review.zip")}}
    (destination/"REPORT.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    (destination/"COMPLETE.json").write_text(canonical({"complete":True,"records":688,"report":bound(destination/"REPORT.json"),
        "note":"Final advice layers are complete. No corpus publication or replacement of human decisions."})+"\n")
    return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True,type=Path)
    args=parser.parse_args();print(canonical(build(args.root)["summary"]))
