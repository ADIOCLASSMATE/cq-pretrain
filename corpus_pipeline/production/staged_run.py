"""Resumable staged corpus processing: ingest -> gates -> plan -> execute -> release.

Each work unit is bounded; only indexes, annotations and receipts are written.
Original datasets are read-only. Screening decisions never certify image provenance.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import fcntl
import json
from pathlib import Path
import time

from .common import atomic_json, canonical, digest, file_sha, jsonl, write_jsonl, require_production_registry
from .records import snapshot
from .ingest import iter_records
from .planner import plan_record
from .state import State
from .stages import inspect_records, ai_stage, dedup_stage, pair_stage, review_original, selected_layouts, annotate_record
from .staged_publish import build_candidate, commit_staged_release
from .validation import semantic_contract
from .potential import annotate_potential, compile_source_contracts
from .quality_gate import (load_quality_policy, apply_quality_gate, binding_from_identity,
                           HARD_REASONS as QUALITY_HARD_REASONS)

BASE = Path(__file__).resolve().parents[1]


def sampled(record, fraction):
    return int(digest({"sample": record["original_sample_id"], "policy": "native_audit_v1"})[:16], 16) / 2**64 < fraction


def admission(record, reasons, image_screening=()):
    reasons = list(reasons)
    image_screening = list(image_screening)
    if not reasons and (Counter(row["asset_id"] for row in image_screening) != Counter(ref["sha256"] for ref in record["source_image_refs"])
                        or any(row.get("screening", {}).get("decision") != "admit" for row in image_screening)):
        reasons.append("required_image_screening_incomplete")
    evidence = {"original_record": record["source_record_sha256"],
                "source_normalization_decision": record["normalized"]["decision"],
                "source_normalization_reasons": record["normalized"].get("reason_codes", []),
                "image_screening_policy": "image_screening.v3", "image_screening": list(image_screening),
                "image_provenance_certified_by_screening": False}
    if reasons:
        hard = {"required_image_short_edge_lt_128", "cross_source_explicit_negative_evidence", "ai_screening_rejected_generated"} | QUALITY_HARD_REASONS
        decision = "reject" if hard.intersection(reasons) or record["normalized"]["decision"] == "rejected" else "pending"
        return {"decision": decision, "reason_codes": sorted(set(reasons)), "evidence": evidence}
    return {"decision": "admit", "reason_codes": [], "evidence": {**evidence,
        "retention_policy": "early_keep_text_unless_confirmed_wrong.v2",
        "quality_scope": "not_individually_visual_reviewed", "original_record": record["source_record_sha256"]}}


def bound_quality(record, plan, receipt, phase):
    target = (record.get("caption_contract") or {}).get("target_segment_index")
    fields = {f"segments[{target}].text"} if target is not None else {
        f"segments[{i}].text" for i, segment in enumerate(record["raw"]["segments"])
        if segment.get("kind") == "text" and segment.get("role") in {None, "assistant"}}
    return {"scope": "per_annotation_visual_review", "verified": True,
            "verdict": receipt["parsed"]["verdict"], "evidence_ref": receipt["cache_key"],
            "source_record_sha256": record["source_record_sha256"],
            "candidate_ids": [candidate["candidate_id"] for candidate in plan["candidates"]
                              if candidate.get("binding_scope") == "source_local" and candidate.get("field") in fields], "phase": phase}


def early_text_eligible(generated):
    """Uncertainty is retained as evidence, never converted into a QC pass."""
    quality = generated["quality"]["parsed"]
    return (quality.get("verdict") in {"pass", "uncertain"}
            and all(check.get("verdict") in {"pass", "uncertain"}
                    for check in quality.get("caption_checks", [])))


async def process_batch(records, state, client, args, directory, identity):
    quality_state = getattr(args, "quality_policy_state", None)
    if quality_state is not None and binding_from_identity(identity) != quality_state["binding"]:
        raise ValueError("Quality policy state must be bound in batch run identity")
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "COMPLETE.json"
    if complete_path.exists():
        receipt = json.loads(complete_path.read_text())
        if receipt["identity"] != identity:
            raise ValueError("Batch input/configuration identity changed")
        for name, sha in receipt["artifacts_sha256"].items():
            if file_sha(directory / name) != sha:
                raise ValueError("Completed batch artifact checksum changed: " + name)
        # Source bytes are checked again by release reader before actual training.
        return {**json.loads((directory / "summary.json").read_text()), "resumed_completed_batch": True}
    assets, failures, reasons = await inspect_records(records, state, workers=args.io_workers)
    write_jsonl(directory / "assets.jsonl", assets.values())
    write_jsonl(directory / "asset_failures.jsonl", failures.values())
    quality_receipts = apply_quality_gate(records, reasons, quality_state) if quality_state is not None else {}
    if quality_state is not None:
        write_jsonl(directory / "quality_gate_evidence.jsonl", [
            {"record_id": r["original_sample_id"], **quality_receipts[r["original_sample_id"]]} for r in records])
    ai = await ai_stage(records, assets, reasons, client, directory, cached_scores=args.cached_detector_scores, workers=args.io_workers, state=state)
    screening_by_record = {}
    for row in ai:
        screening_by_record.setdefault(row["record_id"], []).append(row)
    dedup = await dedup_stage(records, assets, reasons, directory,
        Path(args.dedup_history).resolve() if args.dedup_history else Path(args.output).resolve() / "dedup_assets.sqlite",
        workers=args.io_workers)
    for row in dedup["assets"]:
        assets[row["asset_id"]]["canonical_asset_id"] = row["canonical_asset_id"]
    pairs = await pair_stage(dedup, assets, client,
        Path(args.dedup_history).resolve() if args.dedup_history else Path(args.output).resolve() / "dedup_assets.sqlite")
    plans, admissions, results, candidates = {}, {}, {}, []
    semaphore = asyncio.Semaphore(args.record_concurrency)

    async def process(record):
        async with semaphore:
            sid = record["original_sample_id"]
            imported = state.candidates(sid)
            plan = plan_record(record, phase=args.phase, gate_reasons=reasons[sid], candidates=imported or None)
            result = {"record_id": sid, "source": record["source"], "native_task": record["normalized"]["category"],
                      "initial_action": plan["action"], "model_executed": False, "model_workflow_attempted": False,
                      "layouts": [], "status": "planned"}
            decision = admission(record, reasons[sid], screening_by_record.get(sid, []))
            generated_candidates = []
            if decision["decision"] == "admit":
                try:
                    # A stratified-source pilot can set fraction=1; large runs
                    # use reproducible random baseline plus planner risk signals.
                    if plan["requires_review"] or sampled(record, args.audit_fraction):
                        if plan["action"] not in {"repair_metadata", "not_in_stage_pool", "generate_missing"} and semantic_contract(record) != "pure_text":
                            result["model_workflow_attempted"] = True
                            try:
                                review = await review_original(record, assets, client)
                            except Exception as exc:
                                # Optional audit failure supplies no evidence that
                                # existing usable native supervision is wrong.
                                if args.phase != "early_mixed" or plan["requires_generation"]:
                                    raise
                                review = None
                                result["original_review_incomplete"] = {"error_type": type(exc).__name__,
                                    "failure_kind": getattr(exc, "failure_kind", "data_or_implementation")}
                                decision["evidence"].update(retained_with_uncertainty=True,
                                    quality_flags=["optional_original_review_incomplete"], qualified_coverage=False)
                            if review is not None:
                                result.update(original_review=review, model_executed=True)
                            if review is not None and semantic_contract(record) == "caption":
                                evidence = bound_quality(record, plan, review, args.phase)
                                plan = plan_record(record, phase=args.phase, candidates=imported or None,
                                    quality_evidence={"candidates": {cid: evidence for cid in evidence["candidate_ids"]}})
                            elif review is not None and review["parsed"]["verdict"] == "uncertain" and args.phase == "early_mixed":
                                decision["evidence"].update(quality_receipt=review, retained_with_uncertainty=True,
                                    quality_flags=["native_supervision_review_uncertain"], qualified_coverage=False)
                            elif review is not None and review["parsed"]["verdict"] != "pass":
                                decision = {"decision": "pending", "reason_codes": ["native_supervision_review_not_passed"],
                                            "evidence": {"quality_receipt": review}}
                            elif review is not None:
                                decision["evidence"].update(quality_scope="per_annotation_visual_review", quality_receipt=review)
                    if decision["decision"] == "admit" and plan["requires_generation"]:
                        result["model_workflow_attempted"] = True
                        generated = await annotate_record(record, assets, client, allow_layout=False, phase=args.phase)
                        result.update(generation=generated, model_executed=True)
                        if generated["annotation_quality_pass"] or (args.phase == "early_mixed" and early_text_eligible(generated)):
                            # Caption contracts currently bind exactly one image;
                            # multi-image native tasks are never collapsed here.
                            caps = generated["annotation"]["parsed"]["image_captions"]
                            if len(caps) != 1 or caps[0]["image_index"] != 0 or len(record["source_image_refs"]) != 1:
                                raise ValueError("Caption adapter requires one exact image view")
                            candidate = build_candidate(record, origin="generated",
                                captions={key: caps[0][key] for key in ("short", "medium", "long")},
                                provenance={"phase": args.phase, "model": generated["annotation"]["identity"]["model"],
                                    "source": "pipeline_generation", "ai_generated_text": True,
                                    "caption_contract": record.get("caption_contract"),
                                    "actual_image_inputs": generated["annotation"].get("image_inputs", [])},
                                quality_receipt={"verdict": generated["quality"]["parsed"]["verdict"], "scope": "per_annotation_visual_review", "evidence": generated["quality"]},
                                generator_receipt=generated["annotation"])
                            generated_candidates.append(candidate)
                            if not generated["annotation_quality_pass"]:
                                decision["evidence"].update(retained_with_uncertainty=True, qualified_coverage=False,
                                    quality_flags=["generated_text_uncertain_or_generator_concerns"],
                                    quality_receipt=generated["quality"],
                                    generator_concerns=generated["annotation"]["parsed"].get("concerns", []))
                        else:
                            decision = {"decision": "pending", "reason_codes": ["generated_annotation_quality_not_passed"],
                                        "evidence": {"quality_receipt": generated["quality"]}}
                    if plan["action"] in {"repair_metadata", "review_caption", "blocked_by_existing_gate"} and not generated_candidates:
                        decision = {"decision": "pending", "reason_codes": plan["reason_codes"], "evidence": {"plan_id": plan["plan_id"]}}
                    # Layout selection is independent of recaption. Only explicitly
                    # selected images get geometry annotation and a second review.
                    if args.layout == "select" and decision["decision"] == "admit" and record["domain"] in {"chart", "document", "rendered"}:
                        try:
                            result["layouts"] = await selected_layouts(record, assets, client)
                        except Exception as exc:
                            result["layouts"] = [{"status": "pending", "error_type": type(exc).__name__}]
                except Exception as exc:
                    result.update(status="model_pending", error_type=type(exc).__name__,
                        failure_kind=getattr(exc, "failure_kind", "data_or_implementation"))
                    decision = {"decision": "pending", "reason_codes": ["model_execution_or_validation_incomplete"],
                                "evidence": {"plan_id": plan["plan_id"], "error_type": type(exc).__name__}}
            if plan.get("retained_with_uncertainty"):
                decision["evidence"].update(retained_with_uncertainty=True, qualified_coverage=False,
                    planner_quality_flags=plan.get("quality_flags", []),
                    uncertain_candidate_refs=plan.get("uncertain_candidate_refs", []))
            if quality_state is not None:
                # Preserve the size/quality gate receipt even when an optional
                # model workflow replaces its own pending evidence envelope.
                decision["evidence"]["image_quality"] = quality_receipts[sid]
            if result["status"] == "planned":
                result["status"] = decision["decision"]
            result.update(final_action=plan["action"], admission=decision,
                          candidate_ids=[c["candidate_id"] for c in generated_candidates])
            plans[sid], admissions[sid], results[sid] = plan, decision, result
            selected_catalog_ids = set(plan.get("selected_catalog_candidate_ids", []))
            candidates.extend(candidate for candidate in imported if candidate["candidate_id"] in selected_catalog_ids)
            candidates.extend(generated_candidates)

    await asyncio.gather(*(process(record) for record in records))
    # Model calls never replace record identity or original supervision. The
    # publisher validates task/candidate/receipt constraints independently.
    publication = commit_staged_release(Path(args.publish_dir) if args.publish_dir else Path(args.output) / "releases",
        records, assets, plans, run_identity=identity, phase=args.phase, admissions=admissions, candidates=candidates,
        revocations=[{"kind": "asset", "id": sha, "reason": "cross_source_explicit_negative_evidence"}
            for sha in sorted({ref["sha256"] for r in records for ref in r["source_image_refs"] if state.rejected_asset(ref["sha256"])})]
            + [{"kind": "asset", "id": sha, "reason": "image_screening_rejected_generated"}
               for sha in state.screening_rejections()]
            + [{"kind": "asset", "id": slot["asset_id"], "reason": slot["reason_code"],
                "quality_policy": quality_state["binding"], "decision_sha256": slot["decision_sha256"]}
               for slot in {slot["asset_id"]: slot for receipt in quality_receipts.values() for slot in receipt["slots"]
                            if slot["reason_code"] in QUALITY_HARD_REASONS}.values()]
            + (getattr(args, "manual_revocations", []) if identity.get("batch", 0) == 0 else []))
    write_jsonl(directory / "plans.jsonl", [plans[r["original_sample_id"]] for r in records])
    write_jsonl(directory / "results.jsonl", [results[r["original_sample_id"]] for r in records])
    write_jsonl(directory / "candidates.jsonl", candidates)
    write_jsonl(directory / "pending.jsonl", [results[r["original_sample_id"]] for r in records if admissions[r["original_sample_id"]]["decision"] != "admit"])
    write_jsonl(directory / "layout_sidecars.jsonl", [{"record_id": sid, **layout} for sid, result in results.items() for layout in result["layouts"]])
    write_jsonl(directory / "dedup_relations.jsonl", dedup["relations"])
    write_jsonl(directory / "pair_reviews.jsonl", pairs)
    source_contracts = getattr(args, "layout_source_contracts", {"schema_version": "prepared.layout_source_contracts.v1", "contracts": []})
    potential = [annotate_potential({**record, "record_admission": admissions[record["original_sample_id"]]}, assets,
        source_contracts=source_contracts, pixel_threshold=getattr(args, "potential_min_pixels", 1048576)) for record in records]
    write_jsonl(directory / "annotation_potential.jsonl", potential)
    summary = {"records": len(records), "assets": len(assets), "image_slots": sum(len(r["source_image_refs"]) for r in records),
        "task_counts": dict(Counter(r["normalized"]["category"] for r in records)),
        "action_counts": dict(Counter(p["action"] for p in plans.values())),
        "admission_counts": dict(Counter(a["decision"] for a in admissions.values())),
        "reason_counts": dict(Counter(reason for a in admissions.values() for reason in a["reason_codes"])),
        "source_counts": dict(Counter(r["source"] for r in records)),
        "model_executed_records": sum(r["model_executed"] for r in results.values()),
        "model_workflow_attempted_records": sum(r["model_workflow_attempted"] for r in results.values()),
        "model_executed_records_definition": "completed_annotation_or_original_review_workflows_including_cache; excludes_layout; not_fresh_HTTP_calls",
        "new_candidates": sum(len(r["candidate_ids"]) for r in results.values()), "sscd": dedup["receipt"],
        "ai_reviews": sum("api_review" in r for r in ai), "publication": publication,
        "image_screening_counts": dict(Counter(row["screening"]["decision"] for row in ai)),
        "image_screening_reason_counts": dict(Counter(row["screening"]["reason"] for row in ai)),
        "quality_gate_enabled": quality_state is not None,
        "quality_gate_slot_counts": dict(Counter(slot["action"] for receipt in quality_receipts.values() for slot in receipt["slots"])),
        "all_required_image_slots_checked": True, "full_corpus_processed": False,
        "layout_counts": dict(Counter(l["status"] for r in results.values() for l in r["layouts"])),
        "layout_source_counts": dict(Counter(row["layout_source_route"] for row in potential)),
        "resolution_eligible_image_slots": sum(row["resolution_eligible_image_slots"] for row in potential),
        "resolution_potential_records": sum(row["any_resolution_eligible"] for row in potential),
        "admitted_resolution_potential_records": sum(row["any_resolution_eligible"] and row["training_admitted"] for row in potential),
        "retained_uncertainty_records": sum(a["decision"] == "admit" and a["evidence"].get("retained_with_uncertainty", False) for a in admissions.values()),
        "future_highres_pool_automatically_selected": False,
        "pair_review_counts": dict(Counter(p["status"] for p in pairs)),
        "near_duplicate_policy": "candidate_edges_only_no_automatic_record_deletion", "resumed_completed_batch": False}
    atomic_json(directory / "summary.json", summary)
    artifacts = ["summary.json", "plans.jsonl", "results.jsonl", "assets.jsonl", "asset_failures.jsonl", "candidates.jsonl",
                 "pending.jsonl", "layout_sidecars.jsonl", "dedup_relations.jsonl", "pair_reviews.jsonl", "ai_evidence.jsonl", "annotation_potential.jsonl"]
    if quality_state is not None:
        artifacts.append("quality_gate_evidence.jsonl")
    atomic_json(complete_path, {"identity": identity, "artifacts_sha256": {name: file_sha(directory / name) for name in artifacts},
                              "all_records_accounted": True, "all_records_admitted": all(a["decision"] == "admit" for a in admissions.values())})
    return summary


async def run(args):
    if hasattr(args, "quality_policy"):
        args.quality_policy_state = load_quality_policy(args.quality_policy)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This run already has an active writer") from None
    started = time.time()
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get("collection", "dcvlm") == "genund":
        from .genund_ingest import snapshot_for_manifest, iter_genund_records
        snap = snapshot_for_manifest(manifest)
        record_iterator = iter_genund_records
    else:
        snap = snapshot(manifest["snapshot_root"])
        record_iterator = iter_records
    if snap["fingerprint"] != manifest["snapshot_fingerprint"]:
        raise ValueError("Input snapshot changed")
    rules = json.loads((BASE / "config/rules.json").read_text())
    rules["sources"] = json.loads((BASE / "config/sources.json").read_text())["sources"]
    layout_config_path = Path(args.layout_source_config)
    args.layout_source_contracts = compile_source_contracts(json.loads(layout_config_path.read_text()))
    code = {str(p.relative_to(BASE)): file_sha(p) for p in sorted(BASE.rglob("*.py"))
            if not p.name.startswith("test_") and not any(part in {"runs", "runtime", "__pycache__"} for part in p.relative_to(BASE).parts)}
    identity = {"schema_version": "production.staged_run.v2", "phase": args.phase,
        "manifest_sha256": file_sha(args.manifest), "rules_sha256": digest(rules), "code_sha256": digest(code),
        "backend": args.backend, "generator": args.generator, "verifier": args.verifier,
        "audit_fraction": args.audit_fraction, "layout": args.layout, "batch_size": args.batch_size,
        "layout_source_contracts_sha256": file_sha(layout_config_path), "potential_min_pixels": args.potential_min_pixels,
        "image_screening_policy": "image_screening.v3",
        "image_screening_prompt_sha256": file_sha(BASE / "production/image_screening_prompt_v3.txt"),
        "detector_thresholds_sha256": file_sha(BASE / "detector_calibration/thresholds.json"),
        "max_records": args.max_records, "min_original_short_edge": 128,
        "candidate_catalog_sha256": file_sha(args.candidate_catalog) if args.candidate_catalog else None,
        "base_release_sha256": file_sha(args.base_release) if args.base_release else None,
        "revocations_sha256": file_sha(args.revocations) if args.revocations else None,
        "endpoints_path": str(Path(args.endpoints).resolve()) if args.backend == "serving" else None,
        "dedup_history_path": str(Path(args.dedup_history).resolve()) if args.dedup_history else str(output / "dedup_assets.sqlite"),
        "detector_scores_sha256": file_sha(args.cached_detector_scores)}
    if getattr(args, "quality_policy_state", None) is not None:
        identity["quality_policy"] = args.quality_policy_state["binding"]
    state = State(output / "state.sqlite", identity)
    args.manual_revocations = list(jsonl(args.revocations)) if args.revocations else []
    client = None
    try:
        atomic_json(output / "run.json", {"identity": identity, "code_identity": code, "source_snapshot": snap["fingerprint"],
            "source_images_read_only": True, "scope": "explicit_manifest_selection", "started_unix": started})
        count = state.ingest(record_iterator(manifest, snap, rules, batch_size=args.io_batch_size, max_records=args.max_records))
        if args.candidate_catalog:
            state.import_candidates(jsonl(args.candidate_catalog))
        if args.backend == "api":
            from .api_client import QwenAPIClient
            client = QwenAPIClient(args.credential_example, Path(args.model_cache) if args.model_cache else output / "model_cache", generator=args.generator, verifier=args.verifier,
                                   concurrency=args.model_concurrency, timeout=args.request_timeout)
        elif args.backend == "serving":
            from .client import DynamicClient
            require_production_registry(args.endpoints)
            client = DynamicClient(args.endpoints, Path(args.model_cache) if args.model_cache else output / "model_cache", concurrency=args.model_concurrency,
                                    timeout=args.request_timeout, wait_ready=args.wait_endpoints)
        else:
            from .api_client import DeferredClient
            client = DeferredClient()
        sums, histograms, parts = Counter(), {}, []
        for number, records in enumerate(state.batches(args.batch_size)):
            batch_identity = {"run": identity, "records": [r["source_record_sha256"] for r in records], "batch": number}
            batch_id = f"{number:08d}_" + digest(batch_identity)[:16]
            result = await process_batch(records, state, client, args, output / "batches" / batch_id, batch_identity)
            state.save_batch(batch_id, result)
            for key in ("records", "assets", "image_slots", "model_executed_records", "model_workflow_attempted_records", "new_candidates", "ai_reviews", "resumed_completed_batch", "resolution_eligible_image_slots", "resolution_potential_records", "admitted_resolution_potential_records", "retained_uncertainty_records"):
                sums[key] += result[key]
            for key in ("task_counts", "action_counts", "admission_counts", "reason_counts", "source_counts", "layout_counts", "layout_source_counts", "pair_review_counts", "image_screening_counts", "image_screening_reason_counts"):
                histograms.setdefault(key, Counter()).update(result[key])
            if result["publication"].get("path"):
                parts.append(result["publication"]["path"])
            progress = {"processed_records": sums["records"], "selected_records": count, "batch": number,
                        "action_counts": dict(histograms["action_counts"]), "admission_counts": dict(histograms["admission_counts"])}
            atomic_json(output / "progress.json", progress)
            print(canonical(progress), flush=True)
        from .staged_publish import build_dataset_release
        dataset = build_dataset_release(output / "dataset_release.json", parts, phase=args.phase, run_identity=identity,
                                        base_release=args.base_release)
        summary = {"schema_version": "production.staged_summary.v2", "phase": args.phase, **dict(sums),
            **{key: dict(value) for key, value in histograms.items()}, "duration_seconds": time.time() - started,
            "selected_records": count, "http_and_cache_this_invocation": dict(getattr(client, "stats", {})),
            "dataset_release": dataset, "full_corpus_processed": False,
            "scope": "explicit_manifest_selection", "source_images_modified": False,
            "asset_counts_are_per_batch_observations": True}
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "COMPLETE.json", {"all_selected_records_accounted": sums["records"] == count,
            "summary_sha256": file_sha(output / "summary.json"), "dataset_release_sha256": file_sha(output / "dataset_release.json"),
            "pending_is_not_training_admission": True, "full_corpus_processed": False})
        print(canonical(summary), flush=True)
        return summary
    finally:
        if client:
            await client.close()
        state.close()
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("early_mixed", "highres_1mp"), default="early_mixed")
    parser.add_argument("--backend", choices=("serving", "api", "defer"), default="serving")
    parser.add_argument("--endpoints", default=str(BASE / "serving/endpoints.json"))
    parser.add_argument("--credential-example", default=str(BASE.parents[1] / "unified-mm/test_api.py"))
    parser.add_argument("--generator", default="qwen3.8-max")
    parser.add_argument("--verifier", default="qwen3.8-flash")
    parser.add_argument("--candidate-catalog")
    parser.add_argument("--model-cache", help="Shared exact-request response cache; still validated by full request identity")
    parser.add_argument("--base-release", help="Existing frozen same-phase dataset_release.json to append")
    parser.add_argument("--revocations", help="Evidence-bearing record/candidate/asset tombstones JSONL")
    parser.add_argument("--quality-policy", default=str(BASE / "config/quality_policy.json"),
                        help="Bound blur policy; default reviewed_subset preserves existing gates outside its reviewed SHA set")
    parser.add_argument("--publish-dir")
    parser.add_argument("--dedup-history")
    parser.add_argument("--cached-detector-scores", default=str(BASE / "runs/detector_calibration_20261022/full_cpu_fp32/scores.jsonl"))
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--record-concurrency", type=int, default=16)
    parser.add_argument("--model-concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--wait-endpoints", type=int, default=30)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--audit-fraction", type=float, default=0.0)
    parser.add_argument("--layout", choices=("off", "source", "select"), default="source",
                        help="source/off: metadata tags only; select: opt-in per-image model selection and annotation")
    parser.add_argument("--layout-source-config", default=str(BASE / "prepared/config/layout_source_contracts.json"))
    parser.add_argument("--potential-min-pixels", type=int, default=1048576,
                        help="Original pixel-area tag only; does not select a future corpus")
    args = parser.parse_args(argv)
    if not 0 <= args.audit_fraction <= 1 or any(getattr(args, key) < 1 for key in
        ("io_workers", "io_batch_size", "batch_size", "record_concurrency", "model_concurrency", "request_timeout")):
        parser.error("positive concurrency/batch/timeout and audit fraction in [0,1] required")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be positive")
    if args.potential_min_pixels < 1:
        parser.error("--potential-min-pixels must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
