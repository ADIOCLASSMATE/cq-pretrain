"""A release occurrence can supply context and a diffusion target in one CQ forward."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from data.packing import Packer
from data.codec import DevicePreparer
from models.cq_backbone import backbone
from models.modeling_umm import JointModel
from tests.test_wan_objectives import Documents, read_documents, setup


@pytest.mark.parametrize("flags", [(True, True, True), (True, False, True)], ids=["all", "first-third"])
@pytest.mark.parametrize("visibility", ["visual_block", "causal"])
def test_tititi_reads_targets_and_content(setup, flags, visibility):
    cfg, tokens = setup
    cfg = replace(
        cfg, content_visibility=visibility, visual_uncond_probability=0.0, visual_instruction_dropout_probability=0.0
    )
    # The first two visual occurrences deliberately share an asset, with independent permissions.
    segments = []
    for i, enabled in enumerate(flags):
        segments.extend(
            [
                dict(kind="text", text=f"T{i + 1}"),
                dict(kind="image", image_index=(0, 0, 1)[i], generation_target=enabled),
            ]
        )
    row = dict(sample_id="tititi", format="interleaved", images=[dict(path="a"), dict(path="b")], segments=segments)
    (doc,) = read_documents([row], lambda s: [20 + int(s[1])])
    for e in doc["elements"]:
        if e["type"] == "image":
            e["pixels"] = np.zeros((64, 96 if e["locator"]["path"] == "a" else 64, 3), dtype=np.uint8)
    active = [i for i, enabled in enumerate(flags) if enabled]
    torch.manual_seed(803)
    model = JointModel(cfg).eval()

    def pack(p_hist):
        return Packer(Documents([doc]), replace(cfg, visual_instruction_dropout_probability=p_hist), 64, tokens).next()

    full, receipt = pack(0.0)
    hist, hist_receipt = pack(1.0)
    empty, empty_receipt = Packer(Documents([doc]), replace(cfg, visual_uncond_probability=1.0), 64, tokens).next()
    assert full.generation_image_indices.tolist() == hist.generation_image_indices.tolist() == active
    assert full.generation.query_counts == tuple((6, 6, 4)[i] for i in active)
    assert hist_receipt["instruction_dropped_images"] == sum(flags[1:])
    assert receipt["instruction_dropout_eligible_images"] == sum(flags[1:])
    assert empty_receipt["unconditional_images"] == len(active)
    assert empty.mainline.query_read_sigma[empty.mainline.query_image[0]].eq(0).all()
    assert torch.equal(full.text_indices, hist.text_indices)
    assert torch.equal(full.mainline.query_rope, hist.mainline.query_rope)
    assert torch.equal(full.vision.content_indices, hist.vision.content_indices)
    text_positions = [(full.ids[0] == token).nonzero().item() for token in (21, 22, 23)]
    spans = [r[r.ge(0)] for r in full.image_indices]
    assert len(spans) == 3 and len(full.vision.content_indices) == sum(map(len, spans))
    from evaluations.sampling import make_null_batch, make_history_batch

    for branch, want in ((make_null_batch(full, cfg), [0] * 3), (make_history_batch(full, cfg), text_positions)):
        assert torch.equal(branch.mainline.query_rope, full.mainline.query_rope[:, full.mainline.query_image[0]])
        for i in active:
            rows = branch.mainline.query_positions.eq(spans[i][0])
            assert branch.mainline.query_read_sigma[rows].eq(want[i]).all()

    for batch in (full, hist):
        plan = batch.mainline
        assert torch.equal(plan.cq_mask, plan.cq_predicate.dense())
        for i, span in enumerate(spans):
            queries = (plan.query_image[0] & plan.query_positions.eq(span[0])).nonzero().flatten()
            if not flags[i]:
                assert not len(queries) and plan.condition_indices[i].eq(-1).all()
                continue
            # Hist is a distinct training draw only when an earlier prefix exists.
            cut = text_positions[i] if batch is hist and i > 0 else int(span[0])
            assert len(queries) == len(span) and plan.query_read_sigma[queries].eq(cut).all()
            visible = torch.arange(cut)
            for q in queries:
                read = (~plan.cq_mask[0, 0, batch.ids.numel() + q]).nonzero().flatten()
                assert torch.equal(read, visible)
                delta = plan.delta_plan
                event = delta.hidden_event[batch.ids.numel() + q]
                writes = delta.event_write[:event]
                assert torch.equal(delta.content_order[writes[writes < len(delta.content_order)]], visible)
            mapped = plan.condition_indices[i, : len(span)]
            assert torch.equal(mapped, plan.output_positions[queries])
    with torch.no_grad():
        features = torch.randn(len(full.vision.content_indices), cfg.width)
        query_rows = empty.mainline.output_positions[empty.mainline.query_image[0]]
        changed_ids = empty.ids.clone()
        changed_ids[0, text_positions] = 32
        assert torch.equal(
            backbone(model, empty, vision_features=features)[query_rows],
            backbone(model, replace(empty, ids=changed_ids), vision_features=features + 2)[query_rows],
        )
        from evaluations.sampling import make_null_batch, make_history_batch
    for branch, want in ((make_null_batch(full, cfg), [0] * 3), (make_history_batch(full, cfg), text_positions)):
        assert torch.equal(branch.mainline.query_rope, full.mainline.query_rope[:, full.mainline.query_image[0]])
        for i in active:
            rows = branch.mainline.query_positions.eq(spans[i][0])
            assert branch.mainline.query_read_sigma[rows].eq(want[i]).all()

    for batch in (full, hist):
        hidden = backbone(model, batch, vision_features=features)
        gathered = batch.mainline.gather_image_conditions(hidden, batch.generation_image_indices)
        consumed = gathered.reshape(-1, cfg.width)[batch.generation.condition_indices]
        torch.testing.assert_close(
            consumed, hidden[batch.mainline.output_positions[batch.mainline.query_image[0]]], atol=0, rtol=0
        )
        for j, span in enumerate(spans):
            changed = features.clone()
            changed[torch.isin(batch.vision.content_indices, span)] += 2
            altered_image = backbone(model, batch, vision_features=changed)
            words = batch.ids.clone()
            words[0, text_positions[j]] = 32
            altered_text = backbone(model, replace(batch, ids=words), vision_features=features)
            for i in active:
                target = batch.mainline.condition_indices[i, : len(spans[i])]
                reads_image = not torch.equal(altered_image[target], hidden[target])
                reads_text = not torch.equal(altered_text[target], hidden[target])
                assert reads_image == (j < i)
                assert reads_text == (j < i or (j == i and (batch is full or i == 0)))
    ready, _ = DevicePreparer(cfg, device="cpu", encoder="fake").prepare(full, receipt)
    output = model(ready, compute_image_loss=False)
    loss = model.renderer(output["image_conditions"], ready)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert len(ready.generation.lengths) == len(active)
    assert model.renderer.text_embedding[0].weight.grad.abs().sum() > 0


def test_release_permissions_independent_of_task_and_role():
    def t(role, n):
        return dict(kind="text", role=role, turn=n, text="text")

    def im(role, n, **fields):
        return dict(kind="image", image_index=0, role=role, turn=n, **fields)

    row = dict(
        sample_id="same-media",
        format="conversation",
        direction="I2T",
        images=[dict(path="a")],
        segments=[
            t("assistant", 0),
            im("user", 1, generation_target=True),
            t("user", 2),
            im("assistant", 3, image_loss_allowed=False),
            t("tool", 4),
            im("tool", 5),
        ],
    )
    for record, expected in (
        (row, [True, False, False]),
        (dict(row, image_loss_allowed=True), [True, False, False]),
        (dict(row, image_loss_allowed=False), [False] * 3),
        (dict(row, images=[dict(path="a", dcvlm_image_loss_allowed=False)]), [False] * 3),
    ):
        (doc,) = read_documents([record])
        assert [e["generation_target"] for e in doc["elements"] if e["type"] == "image"] == expected


def test_empty_history_is_local_to_each_packed_document(setup):
    from data.batch_layout import build_plan
    from tests.test_wan_objectives import picture, text

    cfg, tokens = setup
    cfg = replace(cfg, visual_uncond_probability=0.0, visual_instruction_dropout_probability=1.0)
    docs = [dict(id=str(i), templated=False, elements=[text(21 + i), picture()]) for i in range(2)]
    batch, receipt = Packer(Documents(docs), cfg, 64, tokens).next()
    cutoffs = batch.generation_instruction_starts
    assert cutoffs.tolist() == [(batch.segments[0] == seg).nonzero()[0].item() for seg in (0, 1)]
    assert receipt["instruction_dropout_eligible_images"] == receipt["instruction_dropped_images"] == 0
    plan = build_plan(replace(batch, generation_instruction_dropped=cutoffs.ge(0)), cfg)
    for target, boundary in zip(batch.image_indices, cutoffs, strict=True):
        queries = plan.query_image[0] & plan.query_positions.eq(target[0])
        assert plan.query_read_sigma[queries].eq(batch.sigma[0, boundary]).all()
        assert not plan.cq_valid[0, batch.ids.numel() :][queries].any()


def test_release_entry_controls_packing_and_native_jsonl_permissions(tmp_path):
    from data.sources import open_catalog
    from training.train import parse_args

    ungranted = dict(
        images=[dict(path="image.png")],
        segments=[dict(kind="text", text="an apple"), dict(kind="image", image_index=0)],
        generation_target=True,
    )
    native = dict(
        segments=[dict(kind="image", image_index=0, generation_target=flag) for flag in (True, False, True)],
        images=[dict(path="image.png")],
    )
    (tmp_path / "records.jsonl").write_text("\n".join(map(json.dumps, [ungranted, native])))
    entry = tmp_path / "sources.yaml"
    entry.write_text(
        "training_geometry: {sequence_length: 4096, shuffle_block_rows: 2}\nsources: [{path: records.jsonl}]"
    )
    recipe = tmp_path / "experiment.yaml"
    recipe.write_text("training: {sequence_length: 8192, shuffle_block_rows: 4}")
    for root in (tmp_path, entry):
        args = ["--config", str(recipe), "--data-root", str(root), "--output", str(tmp_path)]
        parsed = parse_args(args)
        assert (parsed.sequence_length, parsed.shuffle_block_rows) == (4096, 2)
        assert parse_args([*args, "--sequence-length", "6144"]).sequence_length == 6144
        catalog = open_catalog(root, block_rows=parsed.shuffle_block_rows)
        try:
            docs = read_documents(catalog.read_rows(*catalog.block(0)))
            assert [[e["generation_target"] for e in d["elements"] if e["type"] == "image"] for d in docs] == [
                [False],
                [True, False, True],
            ]
        finally:
            catalog.close()


def test_native_publication_marks_gqa_direction_and_retires_row_only_visual_grants(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data.catalog import Catalog
    from corpus_pipeline.production.occurrence_targets import publish

    source = tmp_path / "original"
    (source / "reports").mkdir(parents=True)
    (source / "tokenizer").mkdir()
    row = dict(
        sample_id="text",
        format="interleaved",
        source_shard="",
        images=[],
        text_tokens=1,
        segments=[dict(kind="text", text="hello")],
        image_loss_allowed=False,
    )
    picture = dict(kind="image", image_index=0)
    rows = [row, dict(row, sample_id="gqa", images=[dict(path="a.jpg")], segments=[picture, *row["segments"]])]
    shard = source / "data.parquet"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    catalog = json.dumps([dict(path=shard.name, rows=2, bytes=shard.stat().st_size)]).encode()
    (source / "catalog.json").write_bytes(catalog)
    (source / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version="dcvlm.pretrain.v1",
                phase="complete",
                source_pool=str(tmp_path),
                tokenizer_lock=dict(files=[]),
            )
        )
    )
    (source / "sources.json").write_text('{"raw_shards":[]}')
    (source / "reports/validation.json").write_text('{"passed":true}')
    (source / "COMPLETE.json").write_text(
        json.dumps(dict(validation_passed=True, catalog_sha256=hashlib.sha256(catalog).hexdigest()))
    )
    old = Catalog(source)
    try:
        assert old.read_rows(0, 0, 1)[0]["sample_id"] == "text"
        with pytest.raises(ValueError, match="generation_target"):
            old.read_rows(0, 1, 1)
    finally:
        old.close()

    publish(source, tmp_path / "published", direction="I2T")
    current = Catalog(tmp_path / "published")
    try:
        (gqa,) = current.read_rows(0, 1, 1)
        assert gqa["direction"] == "I2T" and gqa["segments"][0]["generation_target"] is False
    finally:
        current.close()


def test_rebinding_upstream_assets_keeps_vetoes_on_the_corresponding_occurrence():
    from data.upstream import bound_ref, constrain_bound_occurrences

    def occurrence(i, **kw):
        return dict(kind="image", image_index=i, **kw)

    published = dict(
        format="interleaved",
        sample_id="bound",
        image_loss_allowed=True,
        images=[dict(path="A", exact_identity="A"), dict(path="B", exact_identity="B", dcvlm_image_loss_allowed=False)],
        segments=[occurrence(0), occurrence(1), occurrence(0, generation_target=False), occurrence(0)],
    )
    bound = dict(
        published,
        images=[dict(path="B", exact_identity="B"), dict(path="A", exact_identity="A")],
        segments=[
            occurrence(1),
            occurrence(0),
            occurrence(1),
            occurrence(1, upstream_image=dict(path="A", generation_target=False)),
        ],
    )
    constrain_bound_occurrences(published, bound)
    bound["segments"][0]["generation_target"] = True
    (doc,) = read_documents([bound])
    assert [e["generation_target"] for e in doc["elements"]] == [True, False, False, False]
    assert (
        bound_ref(dict(exact_identity="A"), dict(exact_identity="A", generation_target=False), published)[
            "generation_target"
        ]
        is False
    )
    with pytest.raises(ValueError, match="visual occurrence"):
        constrain_bound_occurrences(published, dict(bound, images=list(reversed(bound["images"]))))


def test_pinned_training_entry_projects_its_selected_view_without_adding_a_shuffle_level(tmp_path):
    import yaml
    from data.sources import open_catalog

    (tmp_path / "rows.jsonl").write_text(
        json.dumps(dict(segments=[dict(kind="image", image_index=0)], images=[dict(path="a")], image_loss_allowed=True))
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema_version="corpus.combined_training.v1",
                children=[],
                sources=[dict(path="rows.jsonl")],
                training_geometry=dict(sequence_length=8192),
            )
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "COMPLETE.json").write_text(json.dumps(dict(complete=True, manifest_sha256=digest)))
    entry = tmp_path / "sources.yaml"
    config = dict(catalog=manifest.name, catalog_sha256=digest)
    entry.write_text(yaml.safe_dump(config))
    catalog = open_catalog(tmp_path, block_rows=1)
    try:
        rows = catalog.read_rows(*catalog.block(0))
        assert rows[0]["sample_id"] == "0:rows:0"
        assert not read_documents(rows)[0]["elements"][0]["generation_target"]
    finally:
        catalog.close()
    entry.write_text(yaml.safe_dump(dict(config, training_geometry=dict(sequence_length=4096))))
    with pytest.raises(ValueError, match="geometry conflicts"):
        open_catalog(tmp_path)
    entry.write_text(yaml.safe_dump(dict(config, catalog_sha256="bad")))
    with pytest.raises(ValueError, match="pinned catalog"):
        open_catalog(tmp_path)
