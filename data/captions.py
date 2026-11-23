"""One caption per image/conditioning group, reproducible across workers."""

import hashlib
import json
from collections import OrderedDict

CONTRACT = "all_captions_uniform_seed_epoch_v1"


def apply_alternatives(catalog, shard, rows):
    """Apply additions pinned to one published shard, sample and original image sequence."""
    binding = catalog.shards[shard].get("text_alternatives")
    if not binding:
        return
    from corpus_pipeline.production.common import digest, file_sha

    cache = catalog._cache().setdefault("text_alternatives", OrderedDict())
    if shard not in cache:
        path = (catalog.path / binding["path"]).resolve(strict=True)
        if not path.is_relative_to(catalog.path) or file_sha(path) != binding["sha256"]:
            raise ValueError("Text alternatives checksum or path changed")
        cache[shard] = json.loads(path.read_text())
    cache.move_to_end(shard)
    while len(cache) > 4:
        cache.popitem(last=False)
    for row in rows:
        for patch in cache[shard].get(row["sample_id"], ()):
            slot = patch["event"]
            if row["format"] == "task_group":
                index = patch["sequence"]
                sequence = row["sequences"][index]
                event = sequence[slot]
                target = sequence[patch["target_event"]]
                if target.get("instruction_event") != slot or not target["image_loss_allowed"]:
                    raise ValueError("Text alternative has no permitted target")
                indices = [*target["input_indices"], target["image_index"]]
                options = event["text_candidates"]
            else:
                event = row["segments"][slot]
                indices = list(range(len(row["images"])))
                options = (
                    json.loads(row["caption_candidates_json"])
                    if row.get("caption_candidates_json")
                    else [dict(text=event["text"], provenance=event.get("source_member") or "original")]
                )
            if (
                event["kind"] != "text"
                or patch["anchor"] not in {x["text"] for x in options}
                or [row["images"][i]["sha256"] for i in indices] != patch["image_sha256s"]
            ):
                raise ValueError("Text alternative differs from its bound supervision")
            known = {" ".join(v["text"].split()) for v in options}
            added = []
            for candidate in patch["candidates"]:
                key = " ".join(candidate["text"].split())
                if not key or not candidate["provenance"]:
                    raise ValueError("Empty text alternative or provenance")
                if key not in known:
                    options.append(candidate)
                    known.add(key)
                    added.append(key)
            if row["format"] == "task_group":
                identities = [row["images"][i]["exact_identity"] for i in indices]
                for new_text in added:
                    key = digest([[new_text], identities[:-1], identities[-1]])
                    if key not in target["transition_keys"]:
                        target["transition_keys"].append(key)
                    if key not in row["transition_keys"]:
                        row["transition_keys"].append(key)
                row["events"] = row["sequences"][0]
            else:
                row.update(caption_segment_index=slot, caption_candidates_json=json.dumps(options, ensure_ascii=False))


def _rebind(row, candidates, eligible, key, seed, epoch):
    """Take the first caption alternative, in a fixed order, that binds the image.

    Reached only when the drawn alternative names no upstream owner able to
    prove it holds the published picture; the published occurrence check
    rejects that pairing, so such a draw never produced a document. Every
    alternative that does bind is left where the ordinary draw put it, which is
    why this cannot change a row the ordinary draw already accepted.
    """
    from data.upstream import candidate_bindable

    options = []
    for position, owners in eligible or []:
        option = candidates[position]
        if owners is not None:
            option = dict(option, provenance=[option["provenance"][i] for i in owners])
        options.append((position, option))
    if not eligible:
        options = list(enumerate(candidates))
    order = sorted(range(len(options)), key=lambda i: hashlib.sha256(key + f":rebind:{i}".encode()).digest())
    for position in order:
        chosen, option = options[position]
        if candidate_bindable(row, option, seed, epoch):
            return chosen, option
    raise ValueError(f"No caption alternative binds the published visual occurrence: {row['sample_id']}")


def select(row, seed, epoch):
    if "caption_members" in row:
        return select_image_caption(row, seed, epoch)
    encoded = row.get("caption_candidates_json")
    if encoded is None:
        return row
    candidates = json.loads(encoded)
    index = row["caption_segment_index"]
    if not candidates or type(index) is not int or not 0 <= index < len(row["segments"]):
        raise ValueError("Invalid caption group")
    if row["segments"][index]["kind"] != "text":
        raise ValueError("Caption slot must be text")
    key = f"{CONTRACT}:{int(seed)}:{int(epoch)}:{row['sample_id']}".encode()
    chosen = int.from_bytes(hashlib.sha256(key).digest(), "big") % len(candidates)
    candidate = candidates[chosen]
    if eligible := row.get("stage1_captions"):
        from data.length_choices import branch, pick
        from data.upstream import origin_key

        scopes = []
        for ix, owners in eligible:
            origins = candidates[ix]["provenance"]
            if owners is None:
                scopes.append(1)
            else:
                allowed = {origin_key(origins[i]) for i in owners}
                scopes.append(branch([int(k in allowed) for k in dict.fromkeys(map(origin_key, origins))]))
        tree = branch(scopes)
        k = pick(tree["weights"] if isinstance(tree, dict) else [1] * len(scopes), key.decode() + ":8k")
        chosen, owners = eligible[k]
        candidate = candidates[chosen]
        if owners is not None:
            candidate = dict(candidate, provenance=[candidate["provenance"][i] for i in owners])
    if row.get("_upstream_sampling"):
        from data.upstream import candidate_bindable

        if not candidate_bindable(row, candidate, seed, epoch):
            chosen, candidate = _rebind(row, candidates, eligible, key, seed, epoch)
    caption = candidate["text"]
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Caption must be nonempty text")
    if row.get("_upstream_sampling"):
        from data.upstream import caption_record

        row = caption_record(row, candidate, seed, epoch)
        index = row["caption_segment_index"]
    segments = [dict(segment) for segment in row["segments"]]
    segments[index]["text"] = caption
    # Published token counts of one alternative do not describe another one.
    return dict(row, segments=segments, text_tokens=None, caption_choice=chosen)


def select_image_caption(group, seed, epoch):
    import pyarrow.compute as pc
    from corpus_pipeline.production.portable_refs import logical_ref
    from corpus_pipeline.dedup.dcvlm_loss_policy_reader import SAFETY_DENY_SHA256
    from data.loss_permissions import image_loss_allowed

    members = group["caption_members"]
    # A placeholder can have over 60,000 owners. Decode only the sampled owner's metadata.
    active = pc.indices_nonzero(members.field("eligible")).to_numpy()
    if not len(active):
        raise ValueError("An image group has no eligible caption")
    key = f"openuni-image-v1:{seed}:{epoch}:{group['sample_id']}".encode()
    index = int(active[int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(active)])
    choice = members[index].as_py()
    roots = group["_dataset_roots"]
    if choice["extra"]:
        row = json.loads(choice["extra"])
        row["caption_candidates_json"] = json.dumps(row.pop("_caption_options_for_training"), ensure_ascii=False)
        row["images"] = [dict(ref, _dataset_roots=roots) for ref in row["images"]]
        row = select(row, seed, epoch)
        row["upstream_caption_choice"] = row["caption_choice"]
    else:
        ref = logical_ref(json.loads(choice["image_json"]))
        ref["_dataset_roots"] = roots
        if not image_loss_allowed(ref) or ref["sha256"] in SAFETY_DENY_SHA256:
            raise ValueError("OpenUni generation candidate has a denied target")
        origin = dict(source="openuni", record_id=choice["sample_id"], subset=choice["subset"], field=choice["field"])
        row = dict(
            source="openuni",
            subset=choice["subset"],
            format="caption",
            direction="T2I",
            source_shard=choice["source_shard"],
            original_locator_json=choice["original_locator_json"],
            adapter_metadata_json=choice["adapter_metadata_json"],
            text_tokens=None,
            segments=[
                dict(kind="text", text=choice["caption"], source_member=choice["field"]),
                dict(kind="image", image_index=0, source_member=ref["member"]),
            ],
            images=[ref],
            image_loss_allowed=True,
            caption_provenance=origin,
            selected_upstream=dict(source="openuni", origin=origin),
        )
    row.update(sample_id=group["sample_id"], caption_choice=index)
    return row
