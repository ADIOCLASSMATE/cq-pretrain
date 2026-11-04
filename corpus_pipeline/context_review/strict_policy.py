"""Bind strict blur results to source-context review and the user's latest rule."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sqlite3
import zipfile

from corpus_pipeline.context_review import report as base, final_report as context_report, blur_worker
from corpus_pipeline.codex_review.farm import canonical, digest, file_sha

USER_POLICY = "你直接帮我弄好，带模糊的直接不要"
LABELS = {"admitted":"保留", "quality_excluded":"质量排除", "ai_excluded":"AI排除", "visual_pending":"隔离待核"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def bound(path, records=None):
    path = Path(path).resolve(strict=True)
    result = {"path":str(path),"sha256":file_sha(path)}
    if records is not None:result["records"]=records
    return result


def load_blur(root, initial):
    path = root / "strict_blur_20261025"
    complete, complete_binding = base._read_bound(path/"COMPLETE.json")
    require(complete.get("complete") is True and complete.get("records")==688
        and complete.get("audit_mode")=="strict" and complete.get("skipped_gates")==[], "blur_not_complete")
    manifest, manifest_binding = base._read_bound(complete["manifest"],complete["manifest_sha256"])
    audit, audit_binding = base._read_bound(complete["audit"],complete["audit_sha256"])
    require(audit.get("passed") is True and audit.get("mode")=="strict" and audit.get("skipped_gates")==[]
        and audit.get("complete_coverage") is True and audit.get("context_items_checked")==688
        and audit.get("context_prompt_binding_checked") is True and audit.get("export_checked") is True
        and audit.get("all_original_image_scope_checked") is True
        and audit.get("export_manifest_sha256")==complete["manifest_sha256"],"blur_audit_invalid")
    require(manifest.get("records")==manifest.get("outputs")==manifest["source"]["count"]==688
        and manifest.get("model")==blur_worker.MODEL and manifest.get("reasoning_effort")==blur_worker.EFFORT,
        "blur_manifest_invalid")
    require(manifest["source"]["sha256"]==file_sha(root/"candidates.jsonl")
        ==file_sha(path/"candidates.jsonl"),"blur_input_changed")
    expected = {r["candidate"]["item_id"]:r["candidate"] for r in initial}
    db=sqlite3.connect((path/"state.sqlite").as_uri()+"?mode=ro",uri=True)
    try:
        require(dict(db.execute("SELECT status,count(*) FROM items GROUP BY status"))=={"succeeded":688},"blur_db_incomplete")
        database={i:(a,b,c) for i,a,b,c in db.execute("SELECT item_id,input_sha256,result_sha256,batch_id FROM items")}
        meta={k:base.strict_json(v) for k,v in db.execute("SELECT key,value FROM metadata")}
        require(meta["complete"]==complete and digest(meta["runtime_binding"])==manifest["runtime_binding_sha256"],"blur_runtime_binding")
    finally:db.close()
    output={}
    for shard in manifest["shards"]:
        source=Path(complete["manifest"]).parent/shard["path"]
        require(file_sha(source)==shard["sha256"],"blur_shard_changed")
        count=0
        with gzip.open(source,"rt") as stream:
            for line in stream:
                row=base.strict_json(line);aid=row["item_id"]
                require(aid in expected and aid not in output and row["source"]==expected[aid],"blur_source_coverage")
                blur_worker.validate_result(row["result"],aid,row["source"])
                require(database[aid]==(digest(row["source"]),digest(row["result"]),row["batch_id"]),"blur_export_db_mismatch")
                output[aid]=row;count+=1
        require(count==shard["records"],"blur_shard_count")
    require(set(output)==set(expected),"blur_exact_coverage")
    return output,{"completion":complete_binding,"manifest":manifest_binding,"audit":audit_binding}


def prepare(root):
    root=Path(root).resolve(strict=True)
    initial, first_bindings=base.load_completed(root)
    second, second_bindings=context_report.load_recheck(root,initial)
    manual=base.strict_json((root/"TARGETED_ADJUDICATION.json").read_bytes())
    manual["entries"]+=base.strict_json((root/"ADDITIONAL_ADJUDICATION.json").read_bytes())["entries"]
    contexts=context_report.combine(initial,second,manual)
    blurs,blur_bindings=load_blur(root,initial)
    overrides={}
    override_path=root/"STRICT_BLUR_ADJUDICATION_COMBINED.json"
    adjudication=base.strict_json(override_path.read_bytes())
    require(adjudication.get("passed") is True,"blur_adjudication_not_passed")
    if adjudication:
        for entry in adjudication["entries"]:
            aid=entry["item_id"]
            require(aid in blurs and aid not in overrides and entry["raw_blur_sha256"]==digest(blurs[aid]["result"])
                and entry["source_input_sha256"]==digest(blurs[aid]["source"])
                and entry["full_original_image_actually_viewed"] is True,"blur_adjudication_binding")
            require(entry["blur_status"] in {"clear","blur_present","uncertain"},"blur_adjudication_status")
            require(entry["review_number"]==blurs[aid]["source"]["review_number"]
                and entry.get("reason") and entry.get("evidence"),"blur_adjudication_evidence")
            overrides[aid]=entry
    expected_overrides={aid for aid,row in blurs.items()
        if row["result"]["blur_status"]=="uncertain" or row["source"]["review_number"]==324}
    require(set(overrides)==expected_overrides and adjudication.get("records")==len(overrides),
        "blur_adjudication_incomplete")
    destination=root/"strict_blur_20261025"/"policy_bundle"
    require(not destination.exists(),"bundle_already_exists")
    destination.mkdir()
    rows=[]
    for row in contexts:
        c,v=row["candidate"],row["result"];aid=c["item_id"]
        raw=blurs[aid]["result"];override=overrides.get(aid)
        status=override["blur_status"] if override else raw["blur_status"]
        reason=override["reason"] if override else raw["reason"]
        context_action=v["recommendation"]
        if override and override.get("context_action_override"):
            require(c["review_number"]==657 and override["context_action_override"]=="exclude_pair"
                and override.get("context_override_reason") and override.get("context_override_evidence"),
                "invalid_independent_context_override")
            context_action=override["context_action_override"]
            if status=="clear":reason=override["context_override_reason"]
        origin_note=context_report.PROVENANCE_FOLLOWUP.get(c["review_number"])
        if status=="clear" and origin_note:
            reason="未检出明确模糊，但图像来源仍待核实："+str(origin_note)
        rows.append({"schema_version":"corpus.quality_policy.decision.v1","item_id":aid,
            "review_number":c["review_number"],"source_ref":c["source_ref"],"source_input_sha256":digest(c),
            "blur_status":status,"context_action":context_action,
            "image_origin_followup":c["review_number"] in context_report.PROVENANCE_FOLLOWUP,
            "reason":reason,"authority":"user_policy_applied_by_model",
            "source_name":c["source_name"],"task_type":c["task_type"],
            "blur_assessment":raw,"blur_result_sha256":digest(raw),"blur_adjudication":override,
            "context_assessment":v,"context_layer":row["effective_layer"],
            "image_origin_note":origin_note,
            "source_evidence_sha256":c["source_evidence_sha256"]})
    decisions=destination/"FINAL_QUALITY_DECISIONS.jsonl.gz"
    with gzip.open(decisions,"wt",encoding="utf-8") as stream:
        for row in rows:stream.write(canonical(row)+"\n")
    manifest={"schema_version":"corpus.quality_policy.bundle.v1","records":688,"user_policy":USER_POLICY,
        "input_human_current":bound(root.parent/"HUMAN_REVIEW_CURRENT.json"),
        "candidates":bound(root/"candidates.jsonl",688),"strict_blur_manifest":blur_bindings["manifest"],
        "decisions":bound(decisions,688)}
    if override_path.exists():manifest["blur_adjudication"]=bound(override_path)
    (destination/"MANIFEST.json").write_text(canonical(manifest)+"\n")
    (destination/"EVIDENCE.json").write_text(canonical({"initial_context":first_bindings,"second_context":second_bindings,
        "strict_blur":blur_bindings,"targeted":bound(root/"TARGETED_ADJUDICATION.json"),
        "additional":bound(root/"ADDITIONAL_ADJUDICATION.json"),
        "blur_adjudication":bound(override_path) if override_path.exists() else None,
        "blur_counts":dict(Counter(r["blur_status"] for r in rows)),"code_sha256":file_sha(__file__)})+"\n")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True,type=Path)
    args=parser.parse_args();print(canonical(prepare(args.root)))
