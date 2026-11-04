"""Persist quarantine-only review decisions and finalize a bounded canary offline.

This is deliberately not a general historical-release retraction mechanism.
Only a run-local, parentless canary candidate may be replaced here.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time

from .common import atomic_json, digest, file_sha, jsonl
from .publish import commit_release, release_contents, validate_release
from .records import load_record, snapshot


def load_admission_overrides(output, records):
    path = Path(output) / "admission_overrides.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get("schema_version") != "production.quarantine_only_review.v1":
        raise ValueError("unsupported admission review schema")
    by_id = {record["original_sample_id"]: record for record in records}
    decisions = {}
    for decision in data["decisions"]:
        sample_id = decision["original_sample_id"]
        if sample_id in decisions or sample_id not in by_id:
            raise ValueError("duplicate or unbound reviewed record")
        record = by_id[sample_id]
        if decision.get("action") != "quarantine" or decision.get("origin") not in {"assistant_visual_review", "human_review"}:
            raise ValueError("manual admission can only quarantine; never auto-approve")
        if not decision.get("reason_code") or not decision.get("reason"):
            raise ValueError("quarantine decision requires evidence reason")
        if decision.get("source_record_sha256") != record["source_record_sha256"]:
            raise ValueError("manual review source record changed")
        if decision.get("image_sha256") not in {image["sha256"] for image in record["source_image_refs"]}:
            raise ValueError("manual review image binding mismatch")
        decisions[sample_id] = {**decision, "override_sha256": file_sha(path)}
    return decisions


def apply_admission_override(result, decision):
    if decision.get("action") != "quarantine":
        raise ValueError("manual decision cannot approve an unknown or rejected source")
    result = deepcopy(result)
    result.setdefault("model_stage_publishable_before_manual_review", result.get("publishable", False))
    result["publishable"] = False
    result["manual_admission_decision"] = decision
    result["reason_codes"] = sorted(set(result.get("reason_codes", []) + [decision["reason_code"]]))
    return result


def finalize_canary(output, manifest):
    output = Path(output).resolve()
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = json.loads((output / "summary.json").read_text())
        complete = json.loads((output / "COMPLETE.json").read_text())
        if not complete.get("canary_completed") or complete["summary_sha256"] != file_sha(output / "summary.json"):
            raise ValueError("offline admission finalization requires a completed real canary")
        existing_receipt = output / "manual_adjudication.json"
        if existing_receipt.exists():
            previous = json.loads(existing_receipt.read_text())
            if (previous.get("override_sha256") == file_sha(output / "admission_overrides.json")
                    and summary.get("post_model_adjudication_sha256") == file_sha(existing_receipt)):
                current = json.loads((output / "releases/current.json").read_text())
                if current["release_id"] != previous["publication"]["release_id"]:
                    raise ValueError("current release changed after manual admission")
                validate_release(current["path"])
                return {**previous, "reused_existing_adjudication": True}
        base = Path(__file__).resolve().parents[1]
        plan = json.loads(Path(manifest).read_text())
        source_snapshot = snapshot(plan["snapshot_root"])
        if source_snapshot["fingerprint"] != plan["snapshot_fingerprint"]:
            raise ValueError("source snapshot changed before final admission")
        rules = json.loads((base / "config/rules.json").read_text())
        rules["sources"] = json.loads((base / "config/sources.json").read_text())["sources"]
        records = [load_record(spec, source_snapshot, rules) for spec in plan["records"]]
        decisions = load_admission_overrides(output, records)
        if not decisions:
            raise ValueError("no explicit quarantine decisions")
        records_by_id = {record["original_sample_id"]: record for record in records}
        processed = {sample_id: json.loads((output / "record_results" / (sample_id + ".json")).read_text()) for sample_id in records_by_id}
        if any(result["attempt_id"] != summary["attempt_id"] for result in processed.values()):
            raise ValueError("record results mix execution attempts")
        original_result_hashes = {sample_id: file_sha(output / "record_results" / (sample_id + ".json")) for sample_id in decisions}
        for sample_id, decision in decisions.items():
            processed[sample_id] = apply_admission_override(processed[sample_id], decision)
        archive = output / "pre_manual_adjudication"
        archive.mkdir(exist_ok=True)
        summary_sha, complete_sha = file_sha(output / "summary.json"), file_sha(output / "COMPLETE.json")
        atomic_json(archive / ("summary_" + summary_sha + ".json"), summary)
        atomic_json(archive / ("complete_" + complete_sha + ".json"), complete)
        pointer = output / "releases/current.json"
        current = json.loads(pointer.read_text())
        current_manifest = json.loads((Path(current["path"]) / "manifest.json").read_text())
        run_identity = json.loads((output / "run.json").read_text())["identity"]
        if (current_manifest.get("parent_release") is not None or current_manifest.get("scope") != "canary_not_full_corpus"
                or current_manifest.get("run_identity") != run_identity):
            raise ValueError("historical or foreign publication requires a proper retraction workflow")
        # No accepted historical parent is discarded: this bounded run has only
        # an unaccepted candidate, and its immutable directory remains intact.
        pointer.rename(archive / ("current_" + current["release_id"] + ".json"))
        assets = {asset["asset_id"]: asset for asset in jsonl(output / "asset_index.jsonl")}
        dedup = json.loads((output / "dedup/result.json").read_text())
        publication = commit_release(output / "releases", records, assets, processed, dedup, run_identity=run_identity)
        if not publication.get("committed"):
            raise ValueError("manual review left no accepted records")
        _, _, views, chain_length = release_contents(publication["path"])
        visible = {view["original_sample_id"] for view in views}
        if visible.intersection(decisions) or chain_length != 1:
            raise ValueError("reviewed exclusion leaked through release ancestry")
        receipt = {"schema_version": "production.offline_manual_adjudication.v1", "time_unix": time.time(),
                   "attempt_id": summary["attempt_id"], "code_sha256": file_sha(__file__),
                   "prior_summary_sha256": summary_sha, "prior_complete_sha256": complete_sha,
                   "prior_release_id": current["release_id"], "override_sha256": file_sha(output / "admission_overrides.json"),
                   "original_result_sha256": original_result_hashes, "manual_quarantined_record_ids": sorted(decisions),
                   "no_new_model_requests": True, "original_model_outputs_unchanged": True,
                   "publication": publication, "reader_validation": validate_release(publication["path"]),
                   "reviewed_ids_visible": [], "release_chain_length": chain_length,
                   "scope": "replace_this_run_parentless_canary_candidate; not_general_historical_retraction"}
        receipt_path = output / "manual_adjudication.json"
        atomic_json(receipt_path, receipt)
        for sample_id in decisions:
            atomic_json(output / "record_results" / (sample_id + ".json"), processed[sample_id])
        summary.update(publication=publication, publishable_records=sum(result["publishable"] for result in processed.values()),
                       manual_admission_quarantines=len(decisions), post_model_adjudication_sha256=file_sha(receipt_path))
        atomic_json(output / "summary.json", summary)
        complete.update(summary_sha256=file_sha(output / "summary.json"), published_training_records=publication["records"],
                        post_model_adjudication_sha256=file_sha(receipt_path), no_new_model_requests_in_adjudication=True)
        atomic_json(output / "COMPLETE.json", complete)
        supervisor_path = output / "supervisor.json"
        if supervisor_path.exists():
            supervisor = json.loads(supervisor_path.read_text())
            atomic_json(archive / ("supervisor_" + file_sha(supervisor_path) + ".json"), supervisor)
            supervisor.update(execution_completion_sha256=supervisor.get("completion_sha256"),
                              completion_sha256=file_sha(output / "COMPLETE.json"),
                              post_model_adjudication_sha256=file_sha(receipt_path))
            atomic_json(supervisor_path, supervisor)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_canary(args.output, args.manifest), ensure_ascii=False))


if __name__ == "__main__":
    main()
