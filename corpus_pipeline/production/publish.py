"""Atomic reference releases and a reader that revalidates original dependencies."""
from __future__ import annotations
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import time
import uuid

from .common import atomic_json, digest, file_sha, jsonl, write_jsonl
from .records import parquet_record, read_asset_bytes


def materialize(view, annotation, assets, *, caption_length=None, epoch_seed=0):
    raw, _ = parquet_record(view["locator"]["path"], view["locator"]["row_index"])
    if raw["sample_id"] != view["original_sample_id"] or digest(raw) != view["source_record_sha256"]:
        raise ValueError("published original record changed")
    segments = deepcopy(raw["segments"])
    loaded = []
    for asset_id in view["asset_ids"]:
        asset = assets[asset_id]
        if asset["short_edge"] < 128:
            raise ValueError("published asset violates minimum image size")
        loaded.append(read_asset_bytes(asset["original_ref"]))
    if len(loaded) != raw["image_count"]:
        raise ValueError("published record lost a required image")
    for change in annotation["assistant_revisions"]:
        index = change["segment_index"]
        if segments[index]["kind"] != "text" or segments[index].get("role") != "assistant":
            raise ValueError("invalid published assistant revision")
        segments[index]["text"] = change["text"]
    contract = view.get("caption_contract")
    if (raw["format"] == "caption" or contract) and len(annotation["image_captions"]) == 1:
        text_positions = ([contract["target_segment_index"]] if contract else
                          [i for i, segment in enumerate(segments) if segment["kind"] == "text"])
        if len(text_positions) != 1:
            raise ValueError("caption rewrite needs one unambiguous target text segment")
        if contract and segments[text_positions[0]].get("role") != "assistant":
            raise ValueError("caption contract must preserve user instruction")
        allowed = contract["allowed_training_lengths"] if contract else ["short", "medium", "long"]
        if caption_length is None:
            draw = int(digest([view["asset_ids"], epoch_seed])[:16], 16) / float(16 ** 16)
            weights = view["caption_length_weights"]
            total = sum(weights[name] for name in allowed)
            cumulative = 0
            caption_length = allowed[-1]
            for name in allowed:
                cumulative += weights[name] / total
                if draw < cumulative:
                    caption_length = name
                    break
        if caption_length not in allowed:
            raise ValueError("caption length violates original instruction contract")
        segments[text_positions[0]]["text"] = annotation["image_captions"][0][caption_length]
    return {"sample_id": raw["sample_id"], "native_task": view["native_task"],
            "segments": segments, "images": loaded, "annotation": annotation, "caption_length": caption_length}


def release_contents(path):
    """Resolve immutable parent deltas and upsert records without losing history."""
    path = Path(path).resolve()
    chain, seen = [], set()
    while True:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["release_id"] in seen:
            raise ValueError("cycle in release parent chain")
        seen.add(manifest["release_id"])
        for filename, expected in manifest["artifacts_sha256"].items():
            if file_sha(path / filename) != expected:
                raise ValueError("release artifact mismatch: " + filename)
        chain.append((path, manifest))
        parent = manifest.get("parent_release")
        if parent is None:
            break
        parent_path = path.parent / parent
        expected_parent = manifest.get("parent_manifest_sha256")
        if expected_parent and file_sha(parent_path / "manifest.json") != expected_parent:
            raise ValueError("parent release manifest changed")
        path = parent_path
    assets, annotations, views = {}, {}, {}
    for directory, manifest in reversed(chain):
        assets.update({a["asset_id"]: a for a in jsonl(directory / "assets.jsonl")})
        annotations.update({a["annotation_id"]: a["annotation"] for a in jsonl(directory / "annotations.jsonl")})
        for view in jsonl(directory / "views.jsonl"):
            views[(view["collection"], view["original_sample_id"])] = view
    return assets, annotations, list(views.values()), len(chain)


def validate_release(path):
    path = Path(path)
    assets, annotations, views, chain_length = release_contents(path)
    count, image_slots = 0, 0
    for view in views:
        if view["native_task"] in ("single_t2i", "multi_t2i") or view["collection"] != "dcvlm":
            raise ValueError("DCVLM release changed task direction")
        result = materialize(view, annotations[view["annotation_id"]], assets)
        count += 1; image_slots += len(result["images"])
    return {"success": True, "records_materialized": count, "all_image_slots_read": image_slots,
            "source_sha256_verified": True, "direction_preserved": True, "release_chain_length": chain_length,
            "caption_length_selection": "deterministic_asset_epoch_weighted_one_variant"}


def commit_release(directory, records, assets, processed, dedup, *, run_identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = json.loads((directory / "current.json").read_text()) if (directory / "current.json").exists() else None
        eligible = [record for record in records if processed[record["original_sample_id"]].get("publishable")]
        if not eligible:
            return {"committed": False, "reason": "no_records_passed_all_gates", "records": 0}
        canonical = {a["asset_id"]: a["canonical_asset_id"] for a in dedup["assets"]}
        batch_identity = {"run_identity": run_identity,
                    "records": [(r["original_sample_id"], processed[r["original_sample_id"]]["annotation"]["cache_key"],
                                 processed[r["original_sample_id"]]["quality"]["cache_key"],
                                 [item["annotation"]["cache_key"] for item in processed[r["original_sample_id"]].get("layouts", [])
                                  if item.get("quality", {}).get("parsed", {}).get("verdict") == "pass"]) for r in eligible]}
        batch_hash = digest(batch_identity)
        if old:
            old_path = Path(old["path"])
            previous = json.loads((old_path / "manifest.json").read_text())
            if previous.get("batch_identity_sha256") == batch_hash:
                reader_receipt = validate_release(old_path)
                return {"committed": True, "reused_existing_release": True, "release_id": old["release_id"],
                        "path": str(old_path), "records": previous["records"], "unique_assets": previous["unique_assets"],
                        "layout_annotations": previous["layout_annotations"], "reader_validation": reader_receipt}
        identity = {**batch_identity, "parent_release": old and old["release_id"]}
        release_id = digest(identity)
        staging = directory / (".staging_" + release_id + "_" + uuid.uuid4().hex)
        staging.mkdir()
        selected_assets = {image["sha256"] for record in eligible for image in record["source_image_refs"]}
        write_jsonl(staging / "assets.jsonl", [assets[key] for key in sorted(selected_assets)])
        annotations, views, layouts = [], [], []
        for record in eligible:
            result = processed[record["original_sample_id"]]
            annotation = result["annotation"]["parsed"]
            annotation_id = digest({"record": record["original_sample_id"], "annotation": annotation})
            annotations.append({"annotation_id": annotation_id, "annotation": annotation,
                                "ai_generated_text": True, "generator_identity": result["annotation"]["identity"],
                                "verifier_identity": result["quality"]["identity"], "quality": result["quality"]["parsed"]})
            ids = [image["sha256"] for image in record["source_image_refs"]]
            view = {"view_id": digest({"sample_id": record["original_sample_id"], "annotation_id": annotation_id}),
                    "collection": "dcvlm", "source": record["source"], "original_sample_id": record["original_sample_id"],
                    "source_record_sha256": record["source_record_sha256"], "locator": record["locator"],
                    "native_task": record["normalized"]["category"], "asset_ids": ids,
                    "canonical_asset_ids": [canonical[i] for i in ids], "annotation_id": annotation_id,
                    "caption_contract": record.get("caption_contract"),
                    "caption_length_weights": {"short": .25, "medium": .5, "long": .25},
                    "asset_exposure_multiplier": 1, "preserve_original_supervision_reference": True}
            views.append(view)
            for layout in result.get("layouts", []):
                if layout.get("quality", {}).get("parsed", {}).get("verdict") == "pass":
                    layouts.append({"original_sample_id": record["original_sample_id"], "layout": layout["annotation"]["parsed"],
                                    "generator_identity": layout["annotation"]["identity"], "quality": layout["quality"]["parsed"],
                                    "does_not_replace_native_training_view": True})
        write_jsonl(staging / "annotations.jsonl", annotations)
        write_jsonl(staging / "views.jsonl", views)
        write_jsonl(staging / "layouts.jsonl", layouts)
        manifest = {"schema_version": "production.reference_release.v1", "release_id": release_id,
                    "parent_release": old and old["release_id"], "incremental_release": True,
                    "parent_manifest_sha256": file_sha(Path(old["path"]) / "manifest.json") if old else None,
                    "batch_identity_sha256": batch_hash, "logical_record_operation": "upsert_by_collection_original_sample_id",
                    "run_identity": run_identity, "records": len(views), "unique_assets": len(selected_assets),
                    "layout_annotations": len(layouts), "minimum_original_short_edge": 128,
                    "no_source_image_copies": True, "scope": "canary_not_full_corpus",
                    "artifacts_sha256": {name: file_sha(staging / name) for name in ("assets.jsonl", "annotations.jsonl", "views.jsonl", "layouts.jsonl")}}
        atomic_json(staging / "manifest.json", manifest)
        reader_receipt = validate_release(staging)
        atomic_json(staging / "reader_validation.json", reader_receipt)
        atomic_json(staging / "COMPLETE.json", {"published_training": True, "release_id": release_id,
                    "reader_validation_sha256": file_sha(staging / "reader_validation.json"), "manifest_sha256": file_sha(staging / "manifest.json")})
        target = directory / release_id
        if target.exists():
            # Resume a crash after directory commit but before current-pointer
            # update. Only remove the private staging directory created here.
            if target.is_symlink() or file_sha(target / "manifest.json") != file_sha(staging / "manifest.json"):
                raise ValueError("existing release identity/content conflict")
            validate_release(target)
            shutil.rmtree(staging)
        else:
            staging.replace(target)
        atomic_json(directory / "current.json", {"release_id": release_id, "path": str(target.resolve()),
                                                  "parent_release": manifest["parent_release"]})
        return {"committed": True, "release_id": release_id, "path": str(target), "records": len(views),
                "unique_assets": len(selected_assets), "layout_annotations": len(layouts), "reader_validation": reader_receipt}
