"""Source rules remain authoritative; detector/API outputs never certify provenance."""
from __future__ import annotations
import asyncio
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

from corpus_pipeline.api_review.client import ReviewClient, credentials_from_example
from corpus_pipeline.api_review.aggregate import validate_parsed
from corpus_pipeline.detector_calibration import evaluate_policy
from .common import atomic_json, digest, file_sha, jsonl, write_jsonl
from .records import read_asset_bytes
from .client import image_payload

BASE = Path(__file__).resolve().parents[1]
FROZEN_DETECTOR_FILES = {
    "scores.jsonl": "407cecb811e068337c26ec9b47457c936eeb75f0e6bd3b1e2690428d2fb1d0f2",
    "modelinfo.cf384.json": "b335a86c2468ecb6a877dd991db625507480905acd62e6fa4e0c6aa85717008b",
    "modelinfo.unifd.json": "b993b73c9486f5d00708fa8875c813c65b994a2b310fbb8ab9f5d7f33ba80409",
    "inference_run.json": "5c147ca74a8032547731f94eca1ff5589cbee7055dc5e268590e8fc6ae05596a",
}


def frozen_detector_identity(cached_scores):
    """Accept only the explicitly audited frozen cache, never arbitrary SHA-keyed scores."""
    if not cached_scores:
        raise ValueError("production detector requires a pinned model/runtime reference")
    cached_scores = Path(cached_scores)
    paths = {name: cached_scores if name == "scores.jsonl" else cached_scores.parent / name
             for name in FROZEN_DETECTOR_FILES}
    for name, path in paths.items():
        if file_sha(path) != FROZEN_DETECTOR_FILES[name]:
            raise ValueError("unrecognized frozen detector artifact: " + name)
    run = json.loads(paths["inference_run.json"].read_text())
    if file_sha(BASE / "detector_calibration/infer.py") != run["code_sha256"]:
        raise ValueError("detector implementation differs from frozen cache")
    runtime = {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "timm")}}
    if any(runtime[name] != run[name] for name in runtime):
        raise ValueError("detector runtime differs from frozen cache")
    models = {name: json.loads(paths["modelinfo." + name + ".json"].read_text()) for name in ("cf384", "unifd")}
    return {"artifact_sha256": dict(FROZEN_DETECTOR_FILES), "inference_code_sha256": run["code_sha256"],
            "runtime": runtime, "models": {name: {key: value for key, value in model.items()
                if key not in {"load_s", "checkpoint"}} for name, model in models.items()},
            "cache_scope": "only_this_frozen_20261022_cpu_fp32_run"}


def completed_detector_stage(stage, expected, assets):
    """Read a prior actual inference only after verifying its complete receipt."""
    stage = Path(stage)
    complete = json.loads((stage / "COMPLETE.json").read_text())
    if complete.get("success") is not True or complete.get("scores_sha256") != file_sha(stage / "scores.jsonl"):
        raise ValueError("local detector completion/checksum mismatch")
    run = json.loads((stage / "inference_run.json").read_text())
    if run.get("code_sha256") != expected["inference_code_sha256"] or any(run.get(k) != v for k, v in expected["runtime"].items()):
        raise ValueError("local detector runtime/code identity mismatch")
    if run.get("device") != "cpu" or run.get("dtype") != "float32" or set(run.get("models", [])) != {"cf384", "unifd"}:
        raise ValueError("local detector precision/models mismatch")
    for name, identity in expected["models"].items():
        actual = json.loads((stage / ("modelinfo." + name + ".json")).read_text())
        if any(actual.get(key) != value for key, value in identity.items()):
            raise ValueError("local detector weight/preprocessing identity mismatch: " + name)
    rows = list(jsonl(stage / "scores.jsonl"))
    desired = {asset["sha256"] for asset in assets}
    if len(rows) != len(desired) or {row["image_sha256"] for row in rows} != desired:
        raise ValueError("local detector completed asset scope mismatch")
    if any(row.get("scores", {}).get(name, {}).get("status") != "ok" for row in rows for name in ("cf384", "unifd")):
        raise ValueError("failed detector output is not a successful cache")
    return {row["image_sha256"]: row for row in rows}


async def detector_scores(assets, output, *, cached_scores=None, threads=8):
    output = Path(output)
    model_identity = frozen_detector_identity(cached_scores)
    cached = {row["image_sha256"]: row for row in jsonl(cached_scores)} if cached_scores else {}
    usable = {sha: row for sha, row in cached.items()
              if all(row.get("scores", {}).get(name, {}).get("status") == "ok" for name in ("cf384", "unifd"))}
    missing = [asset for asset in assets if asset["sha256"] not in usable]
    evidence = {"cached_scores_sha256": file_sha(cached_scores) if cached_scores else None,
                "cached_count": len(assets) - len(missing), "new_count": len(missing),
                "frozen_model_identity": model_identity}
    if missing:
        stage = output / ("detector_" + digest([a["sha256"] for a in missing])[:16])
        if (stage / "COMPLETE.json").exists():
            usable.update(completed_detector_stage(stage, model_identity, missing))
            evidence.update(local_completed_stage_reused=len(missing), new_count=0,
                            local_completed_scores_sha256=file_sha(stage / "scores.jsonl"),
                            local_completion_sha256=file_sha(stage / "COMPLETE.json"))
            missing = []
    if missing:
        stage = output / ("detector_" + digest([a["sha256"] for a in missing])[:16])
        scratch = output / ".detector_images"
        scratch.mkdir(parents=True, exist_ok=True)
        manifest = []
        for asset in missing:
            path = scratch / (asset["sha256"] + ".image")
            path.write_bytes(read_asset_bytes(asset["original_ref"]))
            manifest.append({"image_sha256": asset["sha256"], "image_path": str(path.resolve())})
        samples = output / "new_detector_inputs.jsonl"
        write_jsonl(samples, manifest)
        command = [sys.executable, str(BASE / "detector_calibration/infer.py"), "--samples", str(samples),
                   "--models-dir", str(BASE / "models/ai_detectors"),
                   "--output", str(stage), "--models", "cf384", "unifd", "--device", "cpu",
                   "--batch-size", "4", "--threads", str(threads), "--retry-errors"]
        process = await asyncio.create_subprocess_exec(*command)
        result = await process.wait()
        if result != 0:
            raise RuntimeError("detector_process_failed:" + str(result))
        for name, expected in model_identity["models"].items():
            actual = json.loads((stage / ("modelinfo." + name + ".json")).read_text())
            if any(actual.get(key) != value for key, value in expected.items()):
                raise ValueError("new detector inference model/preprocessing identity mismatch: " + name)
        for row in jsonl(stage / "scores.jsonl"):
            if all(row.get("scores", {}).get(name, {}).get("status") == "ok" for name in ("cf384", "unifd")):
                usable[row["image_sha256"]] = row
        shutil.rmtree(scratch)
        evidence["new_scores_path"] = str(stage / "scores.jsonl")
    selected = {asset["sha256"]: usable.get(asset["sha256"]) for asset in assets}
    atomic_json(output / "detector_reuse_receipt.json", evidence)
    return selected


def equivalent_paid_result(previous, identity, metadata):
    # Transport implementation changes do not change an already completed
    # request when prompt/model/budgets and exact transmitted image are equal.
    fields = ("image_sha256", "model", "prompt_sha256", "max_tokens", "thinking_budget", "preprocessing")
    return (previous.get("status") == "ok" and not validate_parsed(previous.get("parsed"))
            and all(previous.get("identity", {}).get(key) == identity.get(key) for key in fields)
            and previous.get("source_certification") is False and previous.get("automatic_policy_override") is False
            and previous.get("sent_sha256") == metadata["sent_sha256"]
            and digest(previous.get("image_preprocessing")) == digest(metadata))


async def paid_review(asset, output, credential_example, model="kimi-k3"):
    prompt_path = BASE / "api_review/blind_prompt_v1.txt"
    prompt = prompt_path.read_text()
    identity = {"image_sha256": asset["sha256"], "model": model, "prompt_sha256": file_sha(prompt_path),
                "max_tokens": 5000, "thinking_budget": 2400,
                "preprocessing": "original_bytes_or_full_image_lossless_white_alpha_max2048.v2",
                "processing_code_identity": {name: file_sha(Path(__file__).with_name(name)) for name in ("ai.py", "client.py")}}
    target = Path(output) / "paid_reviews" / (digest(identity) + ".json")
    if target.exists():
        previous = json.loads(target.read_text())
        if previous.get("identity") == identity and previous.get("status") == "ok" and not validate_parsed(previous.get("parsed")):
            return {**previous, "cache_hit": True}
    prepared, metadata = await asyncio.to_thread(image_payload, asset)
    for previous_path in sorted(target.parent.glob("*.json")):
        previous = json.loads(previous_path.read_text())
        if equivalent_paid_result(previous, identity, metadata):
            reuse = {"source_result_path": str(previous_path.resolve()), "source_result_sha256": file_sha(previous_path),
                     "current_request_identity": identity, "same_sent_image_sha256": metadata["sent_sha256"],
                     "equivalence": "exact_prompt_model_budgets_and_transmitted_image", "new_api_call": False}
            atomic_json(Path(output) / "paid_review_reuse" / (digest(identity) + ".json"), reuse)
            return {**previous, "cache_hit": True, "reuse_receipt": reuse}
    header, encoded = prepared["image_url"]["url"].split(",", 1)
    payload = {"type": "image", "source": {"type": "base64", "media_type": header[5:].split(";", 1)[0],
               "data": encoded}}
    client = ReviewClient(credentials_from_example(credential_example), timeout=180)
    try:
        row = await client.review(model=model, image_payload=payload, prompt=prompt, max_tokens=5000, thinking_budget=2400)
    finally:
        await client.close()
    if row.get("status") == "ok":
        errors = validate_parsed(row.get("parsed"))
        if errors:
            row.update(status="schema_error", validation_errors=errors)
    row.update(identity=identity, image_sha256=asset["sha256"], source_certification=False,
               automatic_policy_override=False, sent_sha256=metadata["sent_sha256"],
               image_preprocessing=metadata, cache_hit=False)
    atomic_json(target, row)
    return row


def route_asset(asset, provenance, prediction):
    config = evaluate_policy.thresholds(json.loads((BASE / "detector_calibration/thresholds.json").read_text()))
    origin = provenance["origin"]
    # evaluate_one is a calibration/holdout reporting API. Reuse its validated
    # score and threshold primitives without inventing an evaluation split for
    # new production assets.
    raw = prediction.get("scores", {}) if isinstance(prediction, dict) else {}
    scores = {name: evaluate_policy.checked_score(raw.get(name)) for name in ("cf384", "unifd")}
    errors = {name: row["error"] for name, row in scores.items() if row["status"] != "ok"}
    protected = origin in {"programmatic", "rendered"} or provenance.get("policy_exception") == "ai_edit_from_natural_source"
    flags = {name: None if errors else scores[name]["prob"] >= config["review_thresholds"][name]
             for name in ("cf384", "unifd")}
    decision = ("inference_error" if errors else ("protected_source_audit" if protected else "ai_suspect_review")
                if any(flags.values()) else "no_model_alert")
    result = {"image_sha256": asset["sha256"], "decision": decision, "review_flags": flags, "scores": scores,
              "inference_errors": errors, "protected_by_source_or_verified_edit": protected,
              "split": None, "scope": "production_canary_not_calibration_or_holdout",
              "automatic_deletion": False, "training_eligibility_decision": None}
    return {"asset_id": asset["asset_id"], "origin": origin, "review": result,
            "source_policy_decision_is_separate": True,
            "model_scores_are_unvalidated_evidence": True,
            "requires_paid_review": result["decision"] == "ai_suspect_review"}
