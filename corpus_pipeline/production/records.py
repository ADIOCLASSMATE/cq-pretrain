"""Load complete native records and every referenced image, never sample one image."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import tarfile

from PIL import Image, ImageOps

from corpus_pipeline.policy import normalize, evaluate, resolve_source_config
from .common import atomic_json, file_sha, digest, jsonl

DEFAULT_ROOT = Path("/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z")

# Fixed, observed source/structure contracts. Unknown prompts are not guessed by
# a model; adding them requires a prepared adapter change and a new attempt.
CAPTION_INSTRUCTIONS = {
    "Write a terse but informative summary of the picture.": ["short"],
    "Summarize the visual content of the image.": ["short", "medium", "long"],
    "Present a compact description of the photo’s key features.": ["short"],
    "Create a compact narrative representing the image presented.": ["short"],
    "Render a clear and concise summary of the photo.": ["short"],
    "Describe the image concisely.": ["short"],
    "Offer a succinct explanation of the picture presented.": ["short"],
    "Share a concise interpretation of the image provided.": ["short"],
    "Relay a brief, clear account of the picture shown.": ["short"],
    "Provide a brief description of the given image.": ["short"],
    "Give a short and clear explanation of the subsequent image.": ["short"],
}
CONVERSATION_CAPTION_INSTRUCTIONS = {
    "coco_caption": CAPTION_INSTRUCTIONS,
    "openimages_caption": CAPTION_INSTRUCTIONS,
    "pixmo_cap": {
        "Write a concise and meaningful caption for the following image.": ["short"],
        "Please write a brief caption describing this image.": ["short"],
        "Please provide a descriptive caption for this image.": ["short", "medium", "long"],
        "Could you give me a succinct caption for this image?": ["short"],
        "Generate a short caption describing what you see in this image.": ["short"],
    },
}


def prepared_caption_contract(raw, category):
    if raw["image_count"] != 1:
        return None
    if raw.get("format") == "caption":
        targets = [i for i, s in enumerate(raw["segments"]) if s["kind"] == "text"]
        image_positions = [i for i, s in enumerate(raw["segments"]) if s["kind"] == "image"]
        if (len(targets) != 1 or len(image_positions) != 1 or image_positions[0] >= targets[0]
                or any(s.get("role") for s in raw["segments"])):
            return None
        target_text = raw["segments"][targets[0]].get("text")
        missing = target_text is None or (isinstance(target_text, str) and not target_text.strip())
        if category != "single_i2t" and not (category == "unclassified" and missing):
            return None
        return {"adapter": "dcvlm.native_caption.v2", "target_segment_index": targets[0],
                "image_index": 0, "allowed_training_lengths": ["short", "medium", "long"],
                "language_policy": "preserve_original_target_language_and_explicit_user_language_constraints",
                "preserve_original_user_instruction": True}
    if category != "single_i2t":
        return None
    instruction_contract = CONVERSATION_CAPTION_INSTRUCTIONS.get(raw["source"])
    if instruction_contract is None:
        return None
    user_text = [s for s in raw["segments"] if s["kind"] == "text" and s.get("role") == "user"]
    targets = [index for index, s in enumerate(raw["segments"])
               if s["kind"] == "text" and s.get("role") == "assistant"]
    if len(user_text) != 1 or len(targets) != 1:
        return None
    user_index = next(i for i, s in enumerate(raw["segments"]) if s is user_text[0])
    if targets[0] <= user_index:
        return None
    if any(s.get("role") not in (None, "user", "assistant") for s in raw["segments"]):
        return None
    allowed = instruction_contract.get(user_text[0]["text"].strip())
    if allowed is None:
        return None
    return {"adapter": f"dcvlm.{raw['source']}.observed_instructions.v2", "target_segment_index": targets[0],
            "image_index": 0, "allowed_training_lengths": allowed,
            "language_policy": "preserve_original_target_language_and_explicit_user_language_constraints",
            "preserve_original_user_instruction": True}


def _source_membership(manifest, entry):
    if manifest.get('catalog_contract') != 'mixed_source_tar.v1':
        return None
    values = entry.get('sources')
    if (manifest.get('schema_version') != 'corpus.native.v2'
            or entry.get('contract') != 'mixed_source_tar.v1'
            or entry.get('source') != '__mixed__'
            or not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value or value == '__mixed__' for value in values)
            or len(set(values)) != len(values)):
        raise ValueError('invalid mixed TAR source membership')
    return values


def snapshot(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    complete = json.loads((root / "COMPLETE.json").read_text())
    if manifest.get("phase") != "complete" or manifest.get("schema_version") not in ("dcvlm.pretrain.v1", "corpus.native.v2"):
        raise ValueError("snapshot is not complete native v1/v2")
    if complete.get("validation_passed") is not True or complete.get("catalog_sha256") != file_sha(root / "catalog.json"):
        raise ValueError("DCVLM catalog validation mismatch")
    if manifest['schema_version'] == 'corpus.native.v2' and (complete.get('complete') is not True or
            complete.get('manifest_sha256') != file_sha(root / 'manifest.json')):
        raise ValueError('native v2 manifest validation mismatch')
    catalog = json.loads((root / "catalog.json").read_text())
    if not isinstance(catalog, list):
        raise ValueError("native DCVLM catalog must be a list")
    index = {}
    for item in catalog:
        relative = Path(item["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or root not in path.parents:
            raise ValueError("DCVLM catalog path escapes bound snapshot")
        if str(path) in index:
            raise ValueError("duplicate DCVLM catalog entry")
        if manifest['schema_version'] == 'corpus.native.v2' and (item.get('collection') not in ('dcvlm', 'genund') or
                not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64):
            raise ValueError('native v2 catalog requires collection and content hash')
        _source_membership(manifest, item)
        index[str(path)] = item
    return {"root": str(root), "manifest": manifest, "catalog_index": index,
            "fingerprint": {name: file_sha(root / name) for name in ("manifest.json", "catalog.json", "COMPLETE.json")}}


def parquet_record(path, row_index):
    import pyarrow.parquet as pq
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError("invalid parquet row index")
    source = pq.ParquetFile(path)
    begin = 0
    for group in range(source.num_row_groups):
        count = source.metadata.row_group(group).num_rows
        if row_index < begin + count:
            # Read one row group, not the full shard; all columns/images are kept.
            return source.read_row_group(group).slice(row_index - begin, 1).to_pylist()[0], group
        begin += count
    raise IndexError("parquet row index outside shard")


def load_original_record(locator):
    """Resolve a published original view from its bound, read-only source locator."""
    if locator.get("adapter") == "genund.exact_stem_caption_tar.v1":
        from .genund_ingest import original_record
        return original_record(locator)
    if locator.get("adapter") not in (None, "dcvlm.native.v1", "corpus.native.v2"):
        raise ValueError("unsupported original record locator adapter")
    return parquet_record(locator["path"], locator["row_index"])[0]


def load_record(spec, source_snapshot, rules, *, dataset_roots=None):
    root = Path(source_snapshot["root"])
    path = Path(spec["parquet_path"]).resolve()
    if root not in path.parents:
        raise ValueError("record locator outside bound snapshot")
    raw, group = parquet_record(path, spec["parquet_row_index"])
    return load_raw_record(raw, spec, source_snapshot, rules, row_group=group, dataset_roots=dataset_roots)


def load_raw_record(raw, spec, source_snapshot, rules, *, row_group=None, dataset_roots=None):
    """Adapt a native row already read by a bounded Parquet batch iterator.

    Pairing describes the native record's binding, not semantic image/text QC or
    successful image decoding. Those independent gates remain mandatory.
    """
    root = Path(source_snapshot["root"]).resolve()
    path = Path(spec["parquet_path"]).resolve()
    if root not in path.parents:
        raise ValueError("record locator outside bound snapshot")
    if "catalog_index" in source_snapshot:
        entry = source_snapshot["catalog_index"].get(str(path))
        membership = _source_membership(source_snapshot['manifest'], entry) if entry is not None else None
        source_matches = (raw['source'] in membership if membership is not None else
                          entry is not None and entry['source'] == raw['source'])
        if entry is None or not source_matches or entry['source_shard'] != raw['source_shard']:
            raise ValueError("native row does not match published catalog entry")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("native Parquet published size mismatch")
        if source_snapshot['manifest']['schema_version'] == 'corpus.native.v2' and entry['collection'] != raw.get('collection'):
            raise ValueError('native row collection does not match published catalog')
    expected_stat = spec.get("parquet_fingerprint")
    if expected_stat:
        current = path.stat()
        if expected_stat != {"size": current.st_size, "mtime_ns": current.st_mtime_ns}:
            raise ValueError("native Parquet file changed after manifest creation")
    if spec.get("sample_id") and spec["sample_id"] != raw["sample_id"]:
        raise ValueError("original sample_id no longer matches parquet locator")
    v2 = source_snapshot['manifest']['schema_version'] == 'corpus.native.v2'
    collection = raw.get('collection', 'dcvlm') if v2 else 'dcvlm'
    metadata = json.loads(raw['adapter_metadata_json']) if v2 else {}
    policy_raw = raw
    if v2:
        # Reimported native v2 snapshots retain their previous complete row as
        # evidence; unwrap only for provenance normalization, never source SHA.
        layers, original = [], raw
        while 'original_supervision_json' in original:
            layers.append(original)
            original = json.loads(original['original_supervision_json'])
            if len(layers) > 32:
                raise ValueError('native wrapper nesting exceeds metadata budget')
        policy_raw = dict(original)
        for layer in reversed(layers):
            policy_raw = {**policy_raw, **layer}
    if v2:
        from .portable_refs import logical_ref, resolve_ref
        image_refs = []
        allowed_roots = [Path(value).resolve() for value in source_snapshot['manifest']['allowed_image_roots']]
        for image in raw.get('images', []):
            logical = logical_ref(image)
            if 'dataset_id' in logical:
                resolve_ref(logical, dataset_roots,
                            allowed_datasets=source_snapshot['manifest'].get('portable_datasets', []))
                image_refs.append({**logical, 'exists': True})
                continue
            image_path = Path(image['path'])
            if (image['kind'] != 'tar' or not image_path.is_absolute() or image_path.suffix.lower() != '.tar'
                    or not any(parent in image_path.resolve().parents for parent in allowed_roots)
                    or image['offset_data'] != image['offset'] or image['size'] != image['length']):
                raise ValueError('native v2 image reference violates bound storage contract')
            image_refs.append({**json.loads(image['metadata_json']), **image, 'exists': image_path.is_file()})
        policy_raw['images'] = image_refs
    else:
        source_pool = Path(source_snapshot["manifest"]["source_pool"]).resolve()
        shard = Path(raw["source_shard"])
        tar_path = (source_pool / shard).resolve()
        if shard.is_absolute() or ".." in shard.parts or source_pool not in tar_path.parents:
            raise ValueError("native source TAR escapes bound source pool")
        image_refs = [{**image, "kind": "tar", "path": str(tar_path), "exists": tar_path.is_file()}
                      for image in raw.get("images", [])]
    locator = {"path": str(path), "row_index": spec["parquet_row_index"], "row_group": row_group,
               "source_shard": raw["source_shard"], "source_key": raw["source_key"]}
    if v2:
        locator['adapter'] = 'corpus.native.v2'
        locator['original_locator'] = json.loads(raw['original_locator_json'])
    policy_locator = locator['original_locator'] if v2 else locator
    envelope = {"collection": collection, "source": raw["source"], "locator": policy_locator,
                "raw": policy_raw, "image_refs": image_refs, "issues": metadata.get('issues', [])}
    config = resolve_source_config(collection, raw["source"], policy_locator, rules)
    normalized = evaluate(normalize(envelope, config), rules)
    if len(raw.get("images", [])) != raw.get("image_count"):
        raise ValueError("native record image_count mismatch")
    indexes = [s["image_index"] for s in raw["segments"] if s["kind"] == "image"]
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(image_refs) for i in indexes):
        raise ValueError("native record contains invalid image index")
    quarantine = metadata.get('binding_contract') == 'unresolved_original_image_binding.v1'
    if quarantine and (metadata.get('pairing_state') != 'unresolved'
                       or metadata.get('prepared_adapter_status') != 'quarantine'
                       or 'unresolved_original_image_binding' not in metadata.get('issues', [])
                       or indexes or any(part.get('image_index') is not None for part in raw['segments'])):
        raise ValueError('invalid unresolved original binding quarantine')
    if not quarantine and set(indexes) != set(range(len(image_refs))):
        raise ValueError("native record has unbound images")
    for segment in raw["segments"]:
        if segment["kind"] == "image" and segment.get("source_member") != image_refs[segment["image_index"]].get("member"):
            raise ValueError("native image segment member binding mismatch")
    caption = prepared_caption_contract(raw, normalized["category"]) if collection == 'dcvlm' else metadata.get('caption_contract')
    if caption and normalized["category"] == "unclassified" and raw.get("format") == "caption":
        # A registered native caption row can have an explicitly empty target.
        # Its image-before-text order still proves I2T; no direction is flipped.
        normalized["adapter_original_classification"] = {"category": normalized["category"],
                                                          "reason_codes": list(normalized["reason_codes"])}
        # The verified native segment order is authoritative for either
        # collection; a new I2T gen-und input must never become T2I here.
        image_positions=[i for i,s in enumerate(raw['segments']) if s['kind']=='image']
        text_positions=[i for i,s in enumerate(raw['segments']) if s['kind']=='text']
        direction=( 'i2t' if image_positions and text_positions and max(image_positions)<min(text_positions)
                    else 't2i' if image_positions and text_positions and max(text_positions)<min(image_positions)
                    else None )
        declared=policy_raw.get('direction',metadata.get('direction'))
        if declared in ('i2t','t2i') and declared!=direction:
            raise ValueError('Native caption direction contradicts ordered supervision')
        if direction:
            original_classification=normalized['adapter_original_classification']
            record=normalize(envelope,config)
            record['verified_empty_caption_direction']=direction
            normalized=evaluate(record,rules)
            normalized['adapter_original_classification']=original_classification
            normalized['normalization_notes'].append('empty_native_caption_task_recovered_from_explicit_segment_order')
    unresolved_caption_schema = (raw["source"] in CONVERSATION_CAPTION_INSTRUCTIONS or raw.get("format") == "caption") and caption is None
    return {"collection": collection, "original_sample_id": raw["sample_id"], "source": raw["source"], "locator": locator,
            "source_record_sha256": digest(raw), "raw": raw, "normalized": normalized,
            "source_image_refs": image_refs, "domain": spec.get("domain", "unknown"),
            "pairing_state": metadata.get('pairing_state', "verified"), "pairing_evidence_scope": metadata.get('pairing_evidence_scope', "published_native_record_all_image_members_bound_not_semantic_QC"),
            "caption_fields_inspected": metadata.get('caption_fields_inspected', bool(caption)),
            "prepared_adapter_status": metadata.get('prepared_adapter_status', "repair_metadata" if unresolved_caption_schema else "ready"),
            "selection_reason": spec.get("selection_reason"), "audit_only": bool(spec.get("audit_only")),
            "caption_contract": caption, **({"prepared_semantic_task": "caption"} if caption else {}),
            **({'caption_candidates': metadata['caption_candidates']} if 'caption_candidates' in metadata else {})}


def read_asset_bytes(ref, *, roots=None):
    from .portable_refs import read_ref_bytes
    return read_ref_bytes(ref, roots)


def inspect_asset(ref, source, domain):
    blob = read_asset_bytes(ref)
    with Image.open(io.BytesIO(blob)) as image:
        original_width, original_height = image.size
        if original_width * original_height > 100_000_000:
            raise ValueError("max pixel budget exceeded")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("multi-frame image requires explicit frame adapter")
        image.load()
        oriented = ImageOps.exif_transpose(image)
        width, height = oriented.size
        if width * height > 100_000_000:
            raise ValueError("max pixel budget exceeded")
        # Geometry is part of exact decoded-image identity. Equal byte streams
        # laid out at different widths/heights are not the same image view.
        pixel = hashlib.sha256(f"RGBA:{width}:{height}\0".encode() + oriented.convert("RGBA").tobytes()).hexdigest()
    return {"asset_id": ref["sha256"], "sha256": ref["sha256"], "width": width, "height": height,
            "original_width": original_width, "original_height": original_height,
            "pixel_sha256": pixel, "pixel_identity_version": "sha256_rgba_geometry_v2",
            "domain": domain, "source_refs": [{"source": source, "image_ref": ref}],
            "decode_status": "ok", "short_edge": min(width, height),
            "size_decision": "pass" if min(width, height) >= 128 else "reject_short_edge_lt_128",
            "original_ref": ref}


def build_canary(root, associations, samples, output, per_source=2):
    snap = snapshot(root)
    sample_by_sha = {s["image_sha256"]: s for s in jsonl(samples)}
    groups = defaultdict(list)
    for row in jsonl(associations):
        groups[row["source"]].append(row)
    selected, seen = [], set()
    targets = [("coco_caption", "photo", False), ("visdial", "photo", False),
               ("plotqa", "chart", False), ("synthdog", "document", False),
               ("clevr_math", "rendered", False), ("websight", "document", False),
               ("omnicc_matteo", "interleaved", True), ("mmdu", "multi_image", True),
               ("casia", "handwriting_size_rejection_control", True)]
    for source, domain, audit_only in targets:
        candidates = sorted(groups[source], key=lambda r: (r["source_image_count"], r["sample_id"]))
        added = 0
        for row in candidates:
            if row["sample_id"] in seen or row["source_image_count"] > 4:
                continue
            chars = sum(len(s.get("text") or "") for s in row["segments"])
            if chars > 18000:
                continue
            meta = sample_by_sha[row["image_sha256"]]
            if source != "casia" and min(meta["decoded_width"], meta["decoded_height"]) < 128:
                continue
            if row["source_image_count"] > 1:
                full, _ = parquet_record(row["parquet_path"], row["parquet_row_index"])
                if any(min(image.get("width", 0), image.get("height", 0)) < 128 for image in full["images"]):
                    continue
            selected.append({"sample_id": row["sample_id"], "parquet_path": row["parquet_path"],
                             "parquet_row_index": row["parquet_row_index"], "domain": domain,
                             "selection_reason": "fixed_source_stratum_full_record_canary", "audit_only": audit_only})
            seen.add(row["sample_id"])
            added += 1
            if added >= (1 if audit_only else per_source):
                break
    # Verified high-SSCD but semantically different puzzle pair: a real pair-QC
    # challenge, not synthetic images or a lowered-threshold demonstration.
    pair_ids = {"bed9fa607c6cd06f020894cb617a45bb757c2a652e656ec8f3cf6ff8c441fa2d",
                "03a131ccccd97273e70ac58fbfa0e437795674790e50bc92a5e07ece9390b8aa"}
    for candidates in groups.values():
        for row in candidates:
            if row["image_sha256"] in pair_ids and row["sample_id"] not in seen:
                selected.append({"sample_id": row["sample_id"], "parquet_path": row["parquet_path"],
                                 "parquet_row_index": row["parquet_row_index"], "domain": "puzzle",
                                 "selection_reason": "real_sscd_high_similarity_nonduplicate_control", "audit_only": True})
                seen.add(row["sample_id"]); pair_ids.remove(row["image_sha256"])
    catalog = json.loads((Path(root) / "catalog.json").read_text())
    if isinstance(catalog, dict):
        catalog = catalog.get("shards", catalog.get("files", []))
    for entry in catalog:
        if entry["source"] == "dolly":
            selected.append({"parquet_path": str(Path(root) / entry["path"]), "parquet_row_index": 0,
                             "domain": "pure_text", "selection_reason": "pure_text_skip_visual_control", "audit_only": False})
            break
    value = {"schema_version": "production.canary.v1", "snapshot_root": str(Path(root).resolve()),
             "snapshot_fingerprint": snap["fingerprint"], "scope": "fixed_full_record_canary_not_population_estimate",
             "all_images_required": True, "min_short_edge": 128, "records": selected}
    atomic_json(output, value)
    return value
