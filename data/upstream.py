"""Bind a sampled text to the publication and original example which supplied it."""

from collections import OrderedDict
import hashlib
import json

import pyarrow.parquet as pq
from corpus_pipeline.production.common import file_sha
from data.loss_permissions import bound_flags, occurrence_images


CONTRACT = "draw_existing_candidate_then_bound_upstream_v1"


def origin_key(origin):
    if not isinstance(origin, dict):
        return "native"
    source = origin.get("source") or origin.get("dataset")
    for field in ("record_id", "manifest_row", "original_record_id"):
        if origin.get(field) is not None:
            return json.dumps([source, origin[field]], ensure_ascii=False, separators=(",", ":"))
    raise ValueError("Caption origin has no original record identity")


def attach(catalog, shard, rows):
    if not getattr(catalog, "upstream_sampling", None):
        return
    side = catalog.shards[shard].get("upstream_bindings")
    entries = {}
    if side:
        cache = catalog._cache().setdefault("upstream_bindings", OrderedDict())
        key = side["path"]
        if key not in cache:
            path = (catalog.path / key).resolve(strict=True)
            if not path.is_relative_to(catalog.path) or file_sha(path) != side["sha256"]:
                raise ValueError("Upstream binding file changed")
            table = pq.ParquetFile(path).read(columns=["sample_id", "binding_json"])
            cache[key] = ({sid: i for i, sid in enumerate(table.column(0).to_pylist())}, table.column(1))
        cache.move_to_end(key)
        while len(cache) > 2:
            cache.popitem(last=False)
        entries, payloads = cache[key]
    for row in rows:
        row["_upstream_sampling"] = CONTRACT
        if row["sample_id"] in entries:
            row["_upstream_binding"] = json.loads(payloads[entries[row["sample_id"]]].as_py())


def tie(owners, sample_id, seed, epoch, slot):
    if len(owners) == 1:
        return owners[0]
    key = f"{CONTRACT}:{sample_id}:{seed}:{epoch}:{slot}".encode()
    from data.length_choices import index

    i = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(owners)
    return owners[index(len(owners), i, ("upstream", sample_id, slot, len(owners)))]


def with_roots(images, row):
    roots = next((r["_dataset_roots"] for r in row["images"] if "_dataset_roots" in r), None)
    policies = {r["sha256"]: r["_decode_policy"] for r in row["images"] if "_decode_policy" in r}
    return [
        dict(
            ref,
            **({"_dataset_roots": roots} if roots is not None else {}),
            **({"_decode_policy": policies[ref["sha256"]]} if ref.get("sha256") in policies else {}),
        )
        for ref in images
    ]


def bound_ref(ref, original, row):
    if not ref.get("exact_identity") or ref["exact_identity"] != original.get("exact_identity"):
        raise ValueError("Bound view image differs from its published logical slot")
    result = with_roots([ref], row)[0]
    if "conditioning_mask" in original:
        result.setdefault("conditioning_mask", original["conditioning_mask"])
    result.update(bound_flags(original, ref))
    return result


def bound_context(published, binding, key):
    """The published row rebuilt on one upstream owner's own context."""
    row = dict(published)
    for field in (
        "source_policy",
        "metadata",
        "caption_provenance",
        "original_locator_json",
        "adapter_metadata_json",
    ):
        row.pop(field, None)
    context = binding["contexts"][binding["owners"][key]]
    row.update(context, images=with_roots(context["images"], published))
    return row


def occurrences_verifiable(published, bound):
    """True when every bound image occurrence provably binds its published slot.

    The same comparison constrain_bound_occurrences enforces, asked instead of
    asserted. An upstream record can name its own copy of a picture: that copy
    is only usable when a pixel identity or content hash ties it to the slot
    the publication fixed. A copy with neither cannot be shown to be the same
    picture, so the caption it carries must not be attached to this image.
    """
    original = [s for s in published["segments"] if s["kind"] == "image"]
    selected = [s for s in bound["segments"] if s["kind"] == "image"]
    if len(original) != len(selected):
        return False
    for left, right in zip(original, selected):
        logical, _ = occurrence_images(published, left)
        image, _ = occurrence_images(bound, right)
        if not any(logical.get(key) and logical[key] == image.get(key) for key in ("exact_identity", "sha256")):
            return False
    return True


def verifiable_owners(published, keys):
    """Those owner keys whose bound context holds the published visual slots."""
    binding = published.get("_upstream_binding") or {}
    owners = binding.get("owners") or {}
    return [
        key
        for key in keys
        if key in owners and occurrences_verifiable(published, bound_context(published, binding, key))
    ]


def owner_for(published, candidate, seed, epoch):
    """(owner key, origin) for this candidate, or (None, None) if none binds.

    The drawn owner is kept whenever its context is verifiable, so a caption
    that bound before keeps binding exactly as it did. Only a draw the published
    occurrence check would reject is repaired, by redrawing among the owners
    that do bind. That repair can therefore never change a row which already
    produced a document.
    """
    raw = candidate["provenance"]
    origins = raw if isinstance(raw, list) else [raw]
    distinct = {origin_key(origin): origin for origin in origins}
    keys = sorted(distinct)
    drawn = tie(keys, published["sample_id"], seed, epoch, "caption-origin")
    if not published.get("_upstream_binding"):
        return drawn, distinct[drawn]
    usable = verifiable_owners(published, keys)
    if drawn in usable:
        return drawn, distinct[drawn]
    if not usable:
        return None, None
    repaired = tie(usable, published["sample_id"], seed, epoch, "caption-origin-repair")
    return repaired, distinct[repaired]


def candidate_bindable(published, candidate, seed, epoch):
    """True when this caption alternative has an owner that binds the image."""
    return owner_for(published, candidate, seed, epoch)[0] is not None


def constrain_bound_occurrences(published, bound):
    """Upstream alternatives bind the same visual slots, even if their asset indices differ."""
    bound["segments"] = [dict(s) for s in bound["segments"]]
    original = (s for s in published["segments"] if s["kind"] == "image")
    selected = (s for s in bound["segments"] if s["kind"] == "image")
    for left, right in zip(original, selected, strict=True):
        logical, prior_media = occurrence_images(published, left)
        selected_image, selected_media = occurrence_images(bound, right)
        if not any(
            logical.get(key) and logical[key] == selected_image.get(key) for key in ("exact_identity", "sha256")
        ):
            raise ValueError("Upstream alternative does not match its published visual occurrence")
        owners = (published, logical, prior_media, left, bound, selected_image, selected_media, right)
        flags = bound_flags(*owners)
        if flags.get("generation_target") and not (left.get("generation_target") or right.get("generation_target")):
            del flags["generation_target"]
        right.update(flags)


def caption_record(row, candidate, seed, epoch):
    # Identical text remains one draw. Select a documented owner only after that draw.
    key, origin = owner_for(row, candidate, seed, epoch)
    if key is None:
        raise ValueError(f"No upstream owner binds the published visual occurrence: {row['sample_id']}")
    binding = row.get("_upstream_binding")
    if binding:
        published = row
        row = bound_context(published, binding, key)
        # Still the authority on the pairing, and now also applies its flags.
        constrain_bound_occurrences(published, row)
    elif isinstance(origin, dict) and origin.get("source") and origin["source"] != row.get("source"):
        raise ValueError("Cross-source caption has no bound image context")
    else:
        row = dict(row)
    if isinstance(origin, dict):
        row["caption_provenance"] = origin
    row["selected_upstream"] = dict(source=row.get("source"), origin=origin)
    if not isinstance(origin, dict) and row.get("original_locator_json"):
        row["selected_upstream"]["original_locator"] = json.loads(row["original_locator_json"])
    row.pop("_upstream_binding", None)
    row.pop("caption_candidates_json", None)
    return row


def view_options(row, index):
    """Candidate positions in the rendered segment sequence, including an inserted region mask."""
    view = row["views"][index]
    if view["task"] == "description":
        return {2: view["text_candidates"]}
    if view["task"] == "layout_generation":
        return {1: [c for c in row["descriptions"] if c["image_index"] == view["image_index"]]}
    if "stop" not in view:
        return {}
    seq = row["sequences"][view["sequence"]][: view["stop"]]
    insert = len(seq)
    if view["task"] == "region_edit":
        insert = next(
            i for i, e in enumerate(seq) if e["kind"] == "image" and e["image_index"] == view["mask_source_index"]
        )
    return {i + int(i >= insert): e["text_candidates"] for i, e in enumerate(seq) if e.get("text_candidates")}


def _handoff(draw, owner):
    """A human instruction and its accepted rewrite follow the same original frames."""
    if not draw["metadata"].get("handoff_item_id"):
        return
    tv = draw["training_view"]
    tracks = {}
    for index, event in enumerate(draw["segments"]):
        if event.get("event_type") != "instruction":
            continue
        opts = {}
        for choice, option in enumerate(event.get("text_candidates", ())):
            provenance = option["provenance"]
            for field in provenance if isinstance(provenance, list) else [provenance]:
                if field in ("original", "accepted_synthesis"):
                    opts[field] = (choice, option)
        if len(opts) == 2:
            tracks[index] = opts
    # Preserve the old first distinct instruction draw. All later instructions follow it.
    first = next(
        (i for i, opts in tracks.items() if opts["original"][1]["text"] != opts["accepted_synthesis"][1]["text"]), None
    )
    track = "accepted_synthesis"
    if first is not None:
        track = next(k for k, (_, o) in tracks[first].items() if o["text"] == draw["segments"][first]["text"])
    owner.update(handoff_item_id=draw["metadata"]["handoff_item_id"], instruction_variant=track)
    for change in tv["text_choices"]:
        slot = change["event"]
        if slot in tracks:
            choice, option = tracks[slot][track]
            draw["segments"][slot]["text"] = option["text"]
            change.update(choice=choice, provenance=track)
    derived = tv["task"] in ("evaluation", "post_result_control", "description")
    owner["publication"] = "accepted_handoff" if derived or track == "accepted_synthesis" else "original"
    if owner["publication"] == "original":
        events = draw["segments"]
        keep = [i for i, e in enumerate(events) if e.get("event_type") != "evaluation"]
        position = {old: new for new, old in enumerate(keep)}
        draw["segments"] = [events[i] for i in keep]
        for event in draw["segments"]:
            if event.get("instruction_event") is not None:
                event["instruction_event"] = position[event["instruction_event"]]
            if "matching_prior_image_events" in event:
                event["matching_prior_image_events"] = [position[i] for i in event["matching_prior_image_events"]]
        tv["text_choices"] = [dict(c, event=position[c["event"]]) for c in tv["text_choices"] if c["event"] in position]
        draw["metadata"] = dict(handoff_item_id=owner["handoff_item_id"], original=draw["metadata"]["original"])


def _resolved(draw, owner):
    if draw["training_view"]["task"] == "region_edit":
        insert = next(i for i, e in enumerate(draw["segments"]) if e.get("conditioning_mask"))
        for event in draw["segments"]:
            if event.get("instruction_event") is not None and event["instruction_event"] >= insert:
                event["instruction_event"] += 1
            if "matching_prior_image_events" in event:
                event["matching_prior_image_events"] = [
                    i + int(i >= insert) for i in event["matching_prior_image_events"]
                ]
    _handoff(draw, owner)
    for key in ("conditions", "layout", "mask_bindings"):
        if key in draw["typed_conditions"]:
            draw[key] = draw["typed_conditions"][key]
        else:
            draw.pop(key, None)
    for event in draw["segments"]:
        event.pop("text_candidates", None)
    # Only the selected occurrence sequence and its bound metadata leave the sampler.
    for field in (
        "_upstream_binding",
        "events",
        "sequences",
        "views",
        "condition_variants",
        "descriptions",
        "provenance",
        "sources",
        "control_annotations",
        "preferences",
        "preference",
        "transition_keys",
    ):
        draw.pop(field, None)
    draw["selected_upstream"] = owner
    return draw


def task_record(row, draw, seed, epoch):
    from data.task_groups import _render

    index = draw["training_view"]["index"]
    binding = row.get("_upstream_binding")
    if binding and "bound_views" in binding:
        bound = tie(binding["bound_views"][index], row["family_id"], seed, epoch, f"view:{index}")
        members = [row["provenance"][i] for i in bound["members"]]
        owner = dict(
            source=row["source"],
            view=index,
            sequence=row["views"][index].get("sequence"),
            members=[m["family_id"] for m in members],
        )
        draw["metadata"] = (
            members[0]["metadata"]
            if len(members) == 1
            else dict(members=members, request_id=row["metadata"].get("request_id"))
        )
        for slot, ref in bound.get("images", {}).items():
            i = int(slot)
            draw["images"][i] = bound_ref(ref, draw["images"][i], row)
        for event, ref in bound.get("occurrences", {}).items():
            segment = draw["segments"][int(event)]
            segment["upstream_image"] = bound_ref(ref, draw["images"][segment["image_index"]], row)
        if "mask_bindings" in bound:
            draw["typed_conditions"]["mask_bindings"] = bound["mask_bindings"]
        return _resolved(draw, owner)
    if not binding:
        owner = dict(source=row["source"], view=index, sequence=row["views"][index].get("sequence"))
        if row["source"] == "gptedit":
            choices = draw["training_view"]["text_choices"]
            alt = next(
                (
                    x
                    for x in choices
                    if isinstance(x["provenance"], dict) and x["provenance"].get("source") == "GPT-Image-Edit-1.5M"
                ),
                None,
            )
            if alt:
                src = alt["provenance"]
                publication = tie(
                    sorted(set(src["original_metadata"])), row["family_id"], seed, epoch, "instruction-origin"
                )
                owner.update(
                    publication=publication,
                    variant_id=src["variant_id"],
                    input=src["input_path"],
                    output=src["output_path"],
                )
                draw["metadata"] = dict(upstream=owner, instruction=draw["segments"][alt["event"]]["text"])
            else:
                annotation = row["metadata"].get("annotation", {})
                owner["publication"] = (
                    "audit/qc"
                    if annotation.get("decision") or row["metadata"].get("subtype") == "qc"
                    else "ultraedit/conversations"
                )
        return _resolved(draw, owner)

    route = binding["views"][index]
    choices = [(b, v, view_options(binding["branches"][b], v)) for b, v in route["refs"]]
    driver = route["driver"]
    original_pools = view_options(row, index)
    slots = list(original_pools)
    if driver is not None:
        slots.remove(driver)
        slots.insert(0, driver)
    for slot in slots:
        selected = draw["segments"][slot]["text"]
        eligible = [ref for ref in choices if any(c["text"] == selected for c in ref[2].get(slot, ()))]
        if eligible:
            choices = eligible
        elif slot == driver:
            raise ValueError("Selected upstream candidate has no matching source view")
    b, v, pools = tie(choices, row["family_id"], seed, epoch, f"view:{index}")
    branch = binding["branches"][b]
    scoped = _render(branch, seed, epoch, view_index=v)
    chosen = []
    for slot, options in pools.items():
        previous = draw["segments"][slot]["text"]
        available = {c["text"] for c in options}
        text = (
            previous
            if previous in available
            else tie(
                [o["text"] for o in original_pools[slot] if o["text"] in available],
                row["family_id"],
                seed,
                epoch,
                f"bound-text:{index}:{b}:{slot}",
            )
        )
        local = next(i for i, c in enumerate(options) if c["text"] == text)
        scoped["segments"][slot]["text"] = text
        chosen.append(dict(event=slot, choice=local, provenance=options[local]["provenance"]))
    scoped["sample_id"] = row["sample_id"]
    scoped["images"] = with_roots(scoped["images"], row)
    constrain_bound_occurrences(draw, scoped)
    scoped["training_view"].update(index=index, source_view_index=v, text_choices=chosen)
    return _resolved(scoped, dict(source=branch["source"], member=branch["source_member"], view=v))
