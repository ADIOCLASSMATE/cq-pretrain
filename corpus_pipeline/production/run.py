"""Real complete-record canary: source/size/AI/SSCD/annotation/QC/layout/publish."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

from . import prompts
from .ai import detector_scores, frozen_detector_identity, paid_review, route_asset
from .client import DynamicClient
from .common import atomic_json, digest, file_sha, jsonl, write_jsonl, require_production_registry
from .publish import commit_release
from .records import DEFAULT_ROOT, build_canary, inspect_asset, load_record, read_asset_bytes, snapshot
from .review import load_admission_overrides, apply_admission_override
from .validation import annotation_validator, layout_validator, pair_validator, quality_validator, semantic_contract, caption_quality_validator

BASE = Path(__file__).resolve().parents[1]


def validate_dedup_resume(result, eligible_assets):
    from corpus_pipeline.dedup.sscd import MODEL_SHA256, PREPROCESS
    expected = {"model": "sscd_disc_mixup", "weight_sha256": MODEL_SHA256, "preprocess": PREPROCESS,
                "dimensions": 512, "dtype": "float32", "device": "cpu",
                "torch": importlib.metadata.version("torch")}
    receipt = result["receipt"]
    if receipt["model_identity"] != expected or receipt["code_sha256"] != file_sha(BASE / "dedup/sscd.py"):
        raise ValueError("SSCD cached descriptor/code identity changed; use a new output run")
    if receipt["candidate_threshold"] != .75 or receipt["search"]["top_k"] != 20:
        raise ValueError("SSCD cached candidate search configuration changed")
    recorded = {row["asset_id"] for row in result["assets"]} | {row["asset_id"] for row in result["failed"]}
    if recorded != {asset["asset_id"] for asset in eligible_assets}:
        raise ValueError("SSCD cached asset scope changed")


def model_context(record, phase=None):
    context = {"native_task": record["normalized"]["category"], "original_format": record["raw"]["format"],
            "prepared_semantic_task": semantic_contract(record),
            "caption_contract": record.get("caption_contract"),
            "image_count": len(record["source_image_refs"]), "segments": record["raw"]["segments"],
            "all_images_supplied": True}
    if phase is not None:
        if phase not in {"early_mixed", "highres_1mp"}:
            raise ValueError("unsupported annotation phase")
        context.update(training_phase=phase, annotation_quality_policy="staged_visible_grounding.v1")
    if semantic_contract(record) == "dialogue":
        context["segments"] = [{**segment, "segment_index": index}
                               for index, segment in enumerate(record["raw"]["segments"])]
        context["editable_assistant_segment_indices"] = [index for index, segment in enumerate(record["raw"]["segments"])
                                                         if segment["kind"] == "text" and segment.get("role") == "assistant"]
        context["segment_index_definition"] = "Zero-based original segment position, counting image segments too; use only editable_assistant_segment_indices."
    return context


def verification_context(record, context, annotation):
    changes = {item["segment_index"]: item["text"] for item in annotation.get("assistant_revisions", [])}
    targets = [{"segment_index": index, "text": changes.get(index, segment.get("text")),
                "value_origin": "proposed_revision" if index in changes and changes[index] != segment.get("text") else "retained_original_source_target"}
               for index, segment in enumerate(record["raw"]["segments"])
               if segment["kind"] == "text" and segment.get("role") == "assistant"]
    return {"original_record": context, "candidate_annotation": annotation,
            "annotation_is_delta": True, "effective_assistant_targets": targets}


async def annotate_record(record, assets, client, *, allow_layout=True, phase=None):
    context = model_context(record, phase=phase)
    checked_triplet = phase is not None and semantic_contract(record) == "caption"
    verify_prompt = prompts.CAPTION_VERIFY if checked_triplet else prompts.VERIFY
    verify_validator = caption_quality_validator(len(record["source_image_refs"])) if checked_triplet else quality_validator
    verify_tokens = 4500 if checked_triplet else 2200
    bound = [assets[image["sha256"]] for image in record["source_image_refs"]]
    generator = await client.request("generator", "typed_annotation.v1", context, bound,
                                     prompts.ANNOTATE, annotation_validator(record), max_tokens=5000)
    quality = await client.request("verifier", "typed_quality.v1", verification_context(record, context, generator["parsed"]), bound,
                                   verify_prompt, verify_validator, max_tokens=verify_tokens)
    result = {"status": "ok", "annotation": generator, "quality": quality, "layouts": [],
              "annotation_quality_pass": quality["parsed"]["verdict"] == "pass" and not generator["parsed"]["concerns"]}
    # One repair round, preserving original context. Failed/uncertain QC never caches a publication success.
    if quality["parsed"]["verdict"] == "revise":
        repair_context = {**context, "previous_annotation": generator["parsed"], "independent_review": quality["parsed"]}
        repaired = await client.request("generator", "typed_annotation_repair.v1", repair_context, bound,
                                        prompts.ANNOTATE, annotation_validator(record), max_tokens=5000)
        checked = await client.request("verifier", "typed_repair_quality.v1",
            verification_context(record, context, repaired["parsed"]), bound,
            verify_prompt, verify_validator, max_tokens=verify_tokens)
        result.update(original_annotation=generator, original_quality=quality,
                      annotation=repaired, quality=checked,
                      annotation_quality_pass=checked["parsed"]["verdict"] == "pass" and not repaired["parsed"]["concerns"])
    if record.get("quality_review_request"):
        previous = result["annotation"]
        reviewed = await client.request("generator", "typed_annotation_adjudication.v1",
            {**context, "previous_annotation": previous["parsed"], "external_visual_review": record["quality_review_request"],
             "review_handling": "Recheck the supplied images. Remove unsupported precision or identity claims identified by the review; preserve the native task and original constraints. The review is fallible evidence, not ground truth. Record uncertainty rather than inventing a resolution."},
            bound, prompts.ANNOTATE, annotation_validator(record), max_tokens=5000)
        checked = await client.request("verifier", "typed_adjudication_quality.v1",
            {**verification_context(record, context, reviewed["parsed"]), "external_visual_review": record["quality_review_request"]},
            bound, verify_prompt, verify_validator, max_tokens=verify_tokens)
        result.update(pre_adjudication_annotation=previous, pre_adjudication_quality=result["quality"],
                      external_review=record["quality_review_request"], annotation=reviewed, quality=checked,
                      annotation_quality_pass=checked["parsed"]["verdict"] == "pass" and not reviewed["parsed"]["concerns"])
    if allow_layout and result["annotation_quality_pass"]:
        choices = [item for item in result["annotation"]["parsed"]["layout_candidates"] if item["eligible"]]
        for candidate in choices[:1]:
            index = candidate["image_index"]
            try:
                layout = await client.request("generator", "selected_layout.v1",
                    {"requested_original_image_index": index, "selection_evidence": candidate["reason"]}, [bound[index]],
                    prompts.LAYOUT, layout_validator(index), max_tokens=4500)
                review = await client.request("verifier", "layout_quality.v1",
                    {"candidate_layout": layout["parsed"], "original_image_index": index}, [bound[index]],
                    prompts.LAYOUT_VERIFY, quality_validator, max_tokens=2200)
                result["layouts"].append({"status": "ok", "annotation": layout, "quality": review})
            except Exception as error:
                result["layouts"].append({"status": "failed", "image_index": index,
                    "error_type": type(error).__name__, "native_annotation_remains_usable": True})
    return result


async def run(args):
    require_production_registry(args.endpoints)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("production run already has a writer") from None
    started = time.time()
    plan = json.loads(Path(args.manifest).read_text())
    source_snapshot = snapshot(plan["snapshot_root"])
    if source_snapshot["fingerprint"] != plan["snapshot_fingerprint"]:
        raise ValueError("canary snapshot fingerprint changed")
    rules = json.loads((BASE / "config/rules.json").read_text())
    rules["sources"] = json.loads((BASE / "config/sources.json").read_text())["sources"]
    identity = {"manifest_sha256": file_sha(args.manifest), "rules_sha256": digest(rules),
                "minimum_original_short_edge": 128, "scope": "full_selected_records_all_images_canary"}
    run_path = output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["identity"] != identity:
        raise ValueError("run identity changed; choose a new output directory")
    code_identity = {str(path.relative_to(BASE)): file_sha(path) for path in (
        *[p for p in sorted(Path(__file__).parent.glob("*.py")) if not p.name.startswith("test_")],
        BASE / "policy.py", BASE / "adapters.py", BASE / "detector_calibration/evaluate_policy.py",
        BASE / "detector_calibration/infer.py", BASE / "detector_calibration/thresholds.json", BASE / "dedup/sscd.py")}
    attempt_id = digest(code_identity)[:16] + "_" + uuid.uuid4().hex[:12]
    attempt = {"attempt_id": attempt_id, "identity": identity, "started_at_unix": started,
               "endpoints_registry": str(Path(args.endpoints).resolve()), "code_and_config_sha256": code_identity,
               "frozen_detector_identity": frozen_detector_identity(args.cached_detector_scores),
               "cpu_cache_reuse": "verified_content_and_frozen_model_identity_only",
               "model_cache_reuse": "input_prompt_processor_runtime_precision_request_and_processing_code_identity_must_match"}
    review_path = output / "quality_review_requests.json"
    review_requests = json.loads(review_path.read_text()) if review_path.exists() else []
    if not isinstance(review_requests, list):
        raise ValueError("quality review requests must be an explicit list")
    attempt["external_quality_review_sha256"] = file_sha(review_path) if review_path.exists() else None
    atomic_json(output / "attempts" / (attempt_id + ".json"), attempt)
    if not run_path.exists():
        atomic_json(run_path, {"identity": identity, "initial_attempt_id": attempt_id,
                              "endpoints_registry": str(Path(args.endpoints).resolve()),
                              "source_snapshot": source_snapshot["fingerprint"], "source_images_read_only": True})
    (output / "COMPLETE.json").unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        records = list(pool.map(lambda spec: load_record(spec, source_snapshot, rules), plan["records"]))
    if len({r["original_sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate original records in canary")
    admission_overrides = load_admission_overrides(output, records)
    by_id = {record["original_sample_id"]: record for record in records}
    for request in review_requests:
        record = by_id[request["original_sample_id"]]
        if request.get("origin") not in {"assistant_visual_review", "human_review"} or not request.get("issues"):
            raise ValueError("quality review request lacks explicit origin/issues")
        if request.get("image_sha256") not in {image["sha256"] for image in record["source_image_refs"]}:
            raise ValueError("quality review request image binding mismatch")
        if request.get("evidence_file") and file_sha(output / request["evidence_file"]) != request.get("evidence_sha256"):
            raise ValueError("external visual review evidence changed")
        if record.get("quality_review_request"):
            raise ValueError("duplicate quality review request for record")
        record["quality_review_request"] = request
    refs = {}
    for record in records:
        for image in record["source_image_refs"]:
            refs.setdefault(image["sha256"], (image, record["source"], record["domain"]))
    assets, failures = {}, {}
    def inspect(item):
        sha, (ref, source, domain) = item
        try:
            return sha, inspect_asset(ref, source, domain), None
        except Exception as error:
            return sha, None, {"asset_id": sha, "error_type": type(error).__name__, "stage": "original_image_integrity"}
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        for sha, asset, failure in pool.map(inspect, refs.items()):
            if failure:
                failures[sha] = failure
            else:
                assets[sha] = asset
    # Preserve every source association and collect conflicts before any admission.
    explicitly_rejected_assets = set()
    provenance_by_sha = {}
    for record in records:
        for image, provenance in zip(record["source_image_refs"], record["normalized"]["provenance"]["images"]):
            sha = image["sha256"]
            provenance_by_sha.setdefault(sha, []).append(provenance)
            if sha in assets:
                ref = {"source": record["source"], "original_sample_id": record["original_sample_id"], "image_ref": image}
                if ref not in assets[sha]["source_refs"]:
                    assets[sha]["source_refs"].append(ref)
            if provenance["origin"] == "ai_generated" or (provenance["origin"] == "ai_edited" and provenance.get("edit_source_origin") not in ("natural", "unknown", "mixed")):
                explicitly_rejected_assets.add(sha)
    eligible_assets = [asset for sha, asset in assets.items() if asset["size_decision"] == "pass" and sha not in explicitly_rejected_assets]
    write_jsonl(output / "asset_index.jsonl", assets.values())
    write_jsonl(output / "asset_failures.jsonl", failures.values())
    scores = await detector_scores(eligible_assets, output, cached_scores=args.cached_detector_scores, threads=args.io_workers)
    ai_evidence = {}
    for asset in eligible_assets:
        sha = asset["sha256"]
        provenance = provenance_by_sha[sha][0]
        route = route_asset(asset, provenance, scores[sha])
        if route["requires_paid_review"]:
            route["paid_review"] = await paid_review(asset, output, args.credential_example, args.paid_model)
        ai_evidence[sha] = route
    write_jsonl(output / "ai_evidence.jsonl", ai_evidence.values())
    from corpus_pipeline.dedup.sscd import process_assets
    dedup_dir = output / "dedup"
    if (dedup_dir / "COMPLETE.json").exists():
        complete = json.loads((dedup_dir / "COMPLETE.json").read_text())
        if file_sha(dedup_dir / "result.json") != complete["result_sha256"]:
            raise ValueError("dedup result checksum changed")
        dedup = json.loads((dedup_dir / "result.json").read_text())
        validate_dedup_resume(dedup, eligible_assets)
    else:
        if dedup_dir.exists():
            dedup_dir = output / ("dedup_retry_" + str(int(time.time())))
        dedup_inputs = [{**asset, "width": asset["original_width"], "height": asset["original_height"],
                         "blob_loader": lambda ref=asset["original_ref"]: read_asset_bytes(ref)} for asset in eligible_assets]
        dedup = await asyncio.to_thread(process_assets, dedup_inputs, output_dir=dedup_dir,
            history_path=Path(args.dedup_history) if args.dedup_history else output / "dedup_assets.sqlite",
            batch_size=8, threads=args.io_workers)
    dedup_passed = {asset["asset_id"] for asset in dedup["assets"]}
    client = DynamicClient(args.endpoints, output / "model_cache", concurrency=args.model_concurrency,
                           timeout=args.request_timeout, wait_ready=args.wait_endpoints)
    processed = {}
    semaphore = asyncio.Semaphore(args.record_concurrency)
    async def process(record):
        async with semaphore:
            sample_id = record["original_sample_id"]
            ids = [image["sha256"] for image in record["source_image_refs"]]
            reasons = []
            if record["normalized"]["decision"] != "accepted":
                reasons.extend(record["normalized"]["reason_codes"])
            if any(sha in explicitly_rejected_assets for sha in ids):
                reasons.append("cross_source_explicit_negative_evidence")
            if any(sha not in assets for sha in ids):
                reasons.append("required_image_integrity_failure")
            if any(sha in assets and assets[sha]["size_decision"] != "pass" for sha in ids):
                reasons.append("required_image_short_edge_lt_128")
            if any(sha not in dedup_passed for sha in ids):
                reasons.append("required_asset_dedup_stage_incomplete")
            if any(ai_evidence.get(sha, {}).get("review", {}).get("decision") == "inference_error" for sha in ids):
                reasons.append("detector_inference_incomplete")
            if any(ai_evidence.get(sha, {}).get("requires_paid_review") for sha in ids):
                reasons.append("ai_suspicion_requires_provenance_adjudication")
            if sample_id in admission_overrides:
                reasons.append(admission_overrides[sample_id]["reason_code"])
            hard_failure = any(reason in reasons for reason in ("cross_source_explicit_negative_evidence", "required_image_integrity_failure", "required_image_short_edge_lt_128", "required_asset_dedup_stage_incomplete"))
            eligible_for_annotation = not hard_failure and (not reasons or record["audit_only"])
            result = {"original_sample_id": sample_id, "source": record["source"],
                      "attempt_id": attempt_id, "stage_code_identity_sha256": digest(code_identity),
                      "native_task": record["normalized"]["category"], "all_required_images_checked": True,
                      "required_images": len(ids), "stage1_decision": record["normalized"]["decision"],
                      "reason_codes": sorted(set(reasons)), "audit_only": record["audit_only"],
                      "publishable": False, "status": "not_annotated"}
            if eligible_for_annotation:
                try:
                    generated = await annotate_record(record, assets, client,
                        allow_layout=record["domain"] in {"chart", "document", "rendered"})
                    result.update(generated)
                    result["publishable"] = not reasons and not record["audit_only"] and generated["annotation_quality_pass"]
                    if not generated["annotation_quality_pass"]:
                        result["reason_codes"].append("annotation_quality_not_passed")
                except Exception as error:
                    result.update(status="model_stage_failed", error_type=type(error).__name__,
                                  failure_kind=getattr(error, "failure_kind", "implementation_or_data"),
                                  retryable_after_endpoint_change=getattr(error, "retryable_after_endpoint_change", False))
            if sample_id in admission_overrides:
                result = apply_admission_override(result, admission_overrides[sample_id])
            processed[sample_id] = result
            atomic_json(output / "attempt_records" / attempt_id / (sample_id + ".json"), result)
            atomic_json(output / "record_results" / (sample_id + ".json"), result)
            print(json.dumps({"record": sample_id[:12], "source": record["source"], "status": result["status"], "publishable": result["publishable"]}), flush=True)
    pair_reviews = []
    try:
        await asyncio.gather(*(process(record) for record in records))
        for relation in dedup["relations"]:
            first, second = relation["asset_id"], relation["other_asset_id"]
            if first not in assets or second not in assets:
                pair_reviews.append({**relation, "status": "historical_asset_resolver_required", "automatic_merge": False})
                continue
            try:
                context = {"comparison_purpose": "copy_candidate_validation", "source_tasks_may_differ": True}
                review = await client.request("verifier", "sscd_pair_quality.v1", context,
                    [assets[first], assets[second]], prompts.PAIR_VERIFY, pair_validator, max_tokens=2200)
                pair_reviews.append({**relation, "status": "ok", "review": review,
                                     "automatic_merge": False, "supervision_records_preserved": True,
                                     "pair_decision": "shared_exposure_candidate" if review["parsed"]["interchangeable_for_supervision"] is True
                                     else "retain_distinct_assets" if review["parsed"]["interchangeable_for_supervision"] is False else "unresolved_keep_separate",
                                     "sampling_cluster_id": digest(sorted([first, second])) if review["parsed"]["interchangeable_for_supervision"] is True else None})
            except Exception as error:
                pair_reviews.append({**relation, "status": "failed", "error_type": type(error).__name__, "automatic_merge": False,
                                     "failure_kind": getattr(error, "failure_kind", "implementation_or_data"),
                                     "retryable_after_endpoint_change": getattr(error, "retryable_after_endpoint_change", False)})
    finally:
        await client.close()
    write_jsonl(output / "pair_reviews.jsonl", pair_reviews)
    # A missing pair review does not justify merging or dropping records. Publication
    # keeps both originals and records unresolved candidate edges explicitly.
    publication = commit_release(Path(args.publish_dir) if args.publish_dir else output / "releases",
                                  records, assets, processed, dedup, run_identity=identity)
    summary = {"schema_version": "production.canary_result.v1", "records": len(records), "unique_assets": len(assets),
               "all_image_slots": sum(len(r["source_image_refs"]) for r in records), "asset_failures": len(failures),
               "size_rejections": sum(a["size_decision"] != "pass" for a in assets.values()),
               "native_task_counts": dict(Counter(r["normalized"]["category"] for r in records)),
               "record_status_counts": dict(Counter(r["status"] for r in processed.values())),
               "publishable_records": sum(r["publishable"] for r in processed.values()),
               "real_annotations": sum(r.get("annotation", {}).get("real_model_inference") is True for r in processed.values()),
               "real_quality_reviews": sum(r.get("quality", {}).get("real_model_inference") is True for r in processed.values()),
               "layout_generated": sum(item.get("annotation", {}).get("real_model_inference") is True for r in processed.values() for item in r.get("layouts", [])),
               "layout_passed": sum(item.get("quality", {}).get("parsed", {}).get("verdict") == "pass" for r in processed.values() for item in r.get("layouts", [])),
               "optional_layout_failures": sum(item.get("status") == "failed" for r in processed.values() for item in r.get("layouts", [])),
               "paid_review_results": sum("paid_review" in row for row in ai_evidence.values()),
               "paid_review_schema_passed": sum(row.get("paid_review", {}).get("status") == "ok" for row in ai_evidence.values()),
               "sscd": dedup["receipt"], "pair_review_counts": dict(Counter(r["status"] for r in pair_reviews)),
               "publication": publication, "minimum_original_short_edge": 128,
               "full_selected_records_checked": True, "full_corpus_processed": False,
               "duration_seconds": time.time() - started, "attempt_id": attempt_id,
               "stage_code_identity_sha256": digest(code_identity), "sampling_cluster_applied_to_training": False}
    atomic_json(output / "attempt_summaries" / (attempt_id + ".json"), summary)
    atomic_json(output / "summary.json", summary)
    operational_failure = any(r["status"] == "model_stage_failed" for r in processed.values()) or bool(failures) or any(r["status"] == "failed" for r in pair_reviews)
    stages = {"all_image_slots_checked": not failures,
              "real_typed_generation": summary["real_annotations"] > 0,
              "real_independent_quality_review": summary["real_quality_reviews"] > 0,
              "real_paid_alert_review": summary["paid_review_schema_passed"] > 0,
              "real_sscd_pair_review": any(row["status"] == "ok" for row in pair_reviews),
              "real_layout_generated_and_passed": summary["layout_generated"] > 0 and summary["layout_passed"] > 0,
              "published_and_read_back": publication.get("reader_validation", {}).get("success") is True and publication.get("records", 0) > 0}
    atomic_json(output / "COMPLETE.json", {"canary_completed": not operational_failure and all(stages.values()),
                "required_stage_acceptance": stages,
                "all_records_have_terminal_state": True, "published_training_records": publication.get("records", 0),
                "full_corpus_processed": False, "summary_sha256": file_sha(output / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", default=str(DEFAULT_ROOT))
    prepare.add_argument("--associations", default=str(BASE / "runs/detector_calibration_20261022/sample_associations.jsonl"))
    prepare.add_argument("--samples", default=str(BASE / "runs/detector_calibration_20261022/samples.jsonl"))
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--per-source", type=int, default=2)
    execute = sub.add_parser("run")
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--output", required=True)
    execute.add_argument("--endpoints", default=str(BASE / "serving/endpoints.json"))
    execute.add_argument("--cached-detector-scores", default=str(BASE / "runs/detector_calibration_20261022/full_cpu_fp32/scores.jsonl"))
    execute.add_argument("--credential-example", default=str(BASE.parents[1] / "unified-mm/test_api.py"))
    execute.add_argument("--paid-model", default="kimi-k3")
    execute.add_argument("--dedup-history", help="Checked-asset feature catalog; this is never source admission evidence")
    execute.add_argument("--publish-dir", help="Existing release store to append/upsert; defaults to run-local releases")
    execute.add_argument("--io-workers", type=int, default=8)
    execute.add_argument("--model-concurrency", type=int, default=6)
    execute.add_argument("--record-concurrency", type=int, default=4)
    execute.add_argument("--request-timeout", type=int, default=300)
    execute.add_argument("--wait-endpoints", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = build_canary(args.root, args.associations, args.samples, args.output, args.per_source)
        print(json.dumps({"output": args.output, "records": len(value["records"])}))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
