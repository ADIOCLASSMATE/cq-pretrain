"""Real cache/JSONL tests for targets, isolation and acknowledged data recovery."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from data.cache import load_sharded_posterior
from data.dataset import ImageNetFlowCacheDataset, POSTERIOR_CACHE_FORMAT, POSTERIOR_STATS_LAYOUT
from data.dataloader import build_dataloaders
from data.packing import collate_imagenet_flow_cache, collate_segment_packed
from data.sequence import build_selfless_sigma
from data.text_dataset import ClimbMixOnlineBatchDataset


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [10 + sum(word.encode()) % 200 for word in text.split()]


def image_files(root, split="train", count=4):
    root.mkdir(parents=True, exist_ok=True)
    means = torch.arange(count * 4 * 2).reshape(count, 4, 2).float()
    torch.save(dict(posterior_stats=torch.cat([means, torch.ones_like(means)], -1),
                    img_ids=torch.arange(1, count + 1),
                    metadata=dict(format=POSTERIOR_CACHE_FORMAT, stats_layout=POSTERIOR_STATS_LAYOUT)), root / "cache.pt")
    manifests, captions = [], []
    for i in range(1, count + 1):
        path = f"n00000001/{split}_{i}.JPEG"
        manifests.append(dict(img_id=i, synset="n00000001", source_path=f"/images/{split}/{path}", split=split))
        captions.append(dict(img_id=i, path=path, id=f"{split}_{i}", captions=["first caption", "second detailed caption"]))
    (root / "manifest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in manifests))
    (root / "captions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in captions))
    (root / "synsets.txt").write_text("n00000001 example, sample\n")
    return dict(cache_path=str(root / "cache.pt"), manifest_jsonl=str(root / "manifest.jsonl"),
                caption_jsonl=str(root / "captions.jsonl"), conditioning_mode="caption",
                synset_mapping_path=str(root / "synsets.txt"), image_tokens_per_img=4, image_latent_dim=2)


def dataset(root, **kwargs):
    return ImageNetFlowCacheDataset(**image_files(root), tokenizer=Tokenizer(), boi_token_id=3,
                                   eoi_token_id=4, mask_token_id=5, eos_token_id=2, **kwargs)


def text_files(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "text.jsonl"
    path.write_text("".join(json.dumps(dict(text=" ".join(f"word{i + j}" for j in range(5)))) + "\n" for i in range(30)))
    return [str(path)]


def config(root, *, workers=0):
    train, val = image_files(root / "train"), image_files(root / "val", "val", 3)
    return dict(model=dict(boi_token_id=3, eoi_token_id=4, mask_token_id=5, image_tokens_per_img=4, image_latent_dim=2),
                training=dict(seed=43, max_train_steps=10),
                # Generic data tests intentionally inspect every fixture row.
                evaluation=dict(loss_image_per_class=None),
                data=dict(schedule=["text", "t2i", "text", "i2t"], image={**train, "validation": val},
                          text=dict(shard_paths=text_files(root / "text"), sequence_length=8, tokenizer_batch_documents=2),
                          sources={name: dict(micro_batch_size=2, dataloader_workers=workers)
                                   for name in ("text", "t2i", "i2t")}))


def assert_batch_equal(a, b):
    assert a["task"] == b["task"]
    for name in ("input_ids", "labels", "sigma", "segment_ids", "position_ids", "image_latents", "image_loss_mask"):
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


def close_loader_workers(loader):
    """Shut down test workers while their queues and tensors are still alive."""
    iterators = list(getattr(loader, "_iterators", {}).values())
    owned = list(getattr(loader, "image_loaders", {}).values())
    owned += [loader, getattr(loader, "_text_loader", None)]
    iterators += [getattr(item, "_iterator", None) for item in owned if item is not None]
    seen = set()
    for iterator in iterators:
        if iterator is None or id(iterator) in seen:
            continue
        seen.add(id(iterator))
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is None:
            continue
        workers = list(iterator._workers)
        shutdown()
        for worker in workers:
            worker.join(timeout=10)
            assert not worker.is_alive(), "test left a DataLoader worker running"
            assert worker.exitcode == 0, f"DataLoader worker exited with {worker.exitcode}"


def test_posterior_is_fresh_per_epoch_and_validation_is_fixed(tmp_path):
    ds = dataset(tmp_path)
    ds.set_training_indices([0, 1])
    first = ds[0]
    assert torch.equal(first["image_latents"], ds[0]["image_latents"])
    fixed = ds[2]["image_latents"]
    ds.set_epoch(1)
    assert not torch.equal(first["image_latents"], ds[0]["image_latents"])
    assert torch.equal(fixed, ds[2]["image_latents"])
    assert ds[0]["caption_index"] != first["caption_index"]


def test_epoch_reaches_persistent_workers(tmp_path):
    ds = dataset(tmp_path)
    loader = DataLoader(ds, batch_size=1, num_workers=1, persistent_workers=True, multiprocessing_context="spawn", timeout=30, collate_fn=collate_imagenet_flow_cache)
    try:
        # Drain each finite epoch before advancing the shared epoch value. This
        # leaves no outstanding tensor transfers when workers are stopped.
        a = list(loader)[0]["image_latents"]
        ds.set_epoch(1)
        b = list(loader)[0]["image_latents"]
        assert not torch.equal(a, b)
    finally:
        close_loader_workers(loader)


def test_joint_sigma_and_same_position_i2t_targets(tmp_path):
    ds = dataset(tmp_path, caption_sequence_modes=["i2t"])
    sample = ds[0]
    sigma = build_selfless_sigma(sample, image_tokens=4)
    image = sample["token_types"] == 1
    assert sigma[image].unique().numel() == 1
    assert not sample["image_loss_mask"].any()
    targets = sample["labels"] != -100
    assert torch.equal(sample["labels"][targets], sample["input_ids"][targets])
    assert sample["labels"][-1] == Tokenizer.eos_token_id
    start = int(sample["image_start"])
    assert sigma[start + 4] < sigma[start]
    assert (sigma[targets] > sigma[start]).all()


def test_t2i_and_i2t_targets_are_disjoint(tmp_path):
    ds = dataset(tmp_path, caption_sequence_modes=["t2i", "i2t"])
    tasks = []
    for epoch in (0, 1):
        ds.set_epoch(epoch)
        sample = ds[0]
        tasks.append(sample["task_mode"])
        assert bool(sample["image_loss_mask"].any()) != bool(sample["labels"].ne(-100).any())
    assert set(tasks) == {"t2i", "i2t"}


def test_caption_identity_mismatch_fails_before_sampling(tmp_path):
    params = image_files(tmp_path)
    path = Path(params["caption_jsonl"])
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["path"] = "n00000001/other.JPEG"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="mismatch"):
        ImageNetFlowCacheDataset(**params, tokenizer=Tokenizer(), boi_token_id=3, eoi_token_id=4, mask_token_id=5, eos_token_id=2)


def test_caption_missing_member_is_rejected(tmp_path):
    params = image_files(tmp_path)
    path = Path(params["caption_jsonl"])
    path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="Missing 1 captions"):
        ImageNetFlowCacheDataset(**params, tokenizer=Tokenizer(), boi_token_id=3, eoi_token_id=4, mask_token_id=5, eos_token_id=2)


def test_packing_preserves_local_positions_labels_and_segment_isolation(tmp_path):
    ds = dataset(tmp_path)
    samples = [ds[0], ds[1]]
    packed = collate_segment_packed(samples, nominal_capacity=64, image_uncond_prob=1)
    flat = collate_imagenet_flow_cache(samples)
    assert packed["input_ids"].shape[0] == 1
    assert set(packed["segment_ids"].unique().tolist()) == {-1, 0, 1}
    for segment in (0, 1):
        active = packed["segment_ids"][0].eq(segment)
        n = int(active.sum())
        torch.testing.assert_close(packed["input_ids"][0, active], flat["input_ids"][segment, :n])
        torch.testing.assert_close(packed["position_ids"][:, 0, active], flat["position_ids"][:, segment, :n])
    assert torch.equal(packed["image_uncond_mask"], packed["image_latent_mask"])
    spans = packed["image_span_table"]
    for row, segment, start, end, _ in spans:
        assert torch.all(packed["segment_ids"][row, start:end] == segment)


@pytest.mark.parametrize("workers", [0, 1])
def test_mixed_resume_uses_only_consumed_cursors_even_after_prefetch(tmp_path, workers):
    cfg = config(tmp_path, workers=workers)
    loader, validation = build_dataloaders(cfg, Tokenizer())
    resumed = None
    try:
        it = iter(loader)
        tasks = []
        for _ in range(5):
            batch = next(it)
            tasks.append(batch["task"])
            loader.mark_consumed(batch)
        assert tasks == ["text", "t2i", "text", "i2t", "text"]
        saved = loader.state_dict()
        expected = [next(it) for _ in range(9)]
        assert loader.state_dict() == saved
        resumed, _ = build_dataloaders(cfg, Tokenizer())
        resumed.load_state_dict(saved)
        resumed_it = iter(resumed)
        for wanted in expected:
            got = next(resumed_it)
            assert_batch_equal(got, wanted)
            resumed.mark_consumed(got)
        assert resumed.state_dict()["micro_step"] == 14
        assert validation.task_lengths == {"t2i": 2, "i2t": 2}
    finally:
        close_loader_workers(loader)
        if resumed is not None:
            close_loader_workers(resumed)


def test_out_of_order_commit_fails_and_validation_does_not_advance_training(tmp_path):
    loader, validation = build_dataloaders(config(tmp_path), Tokenizer())
    before = loader.state_dict()
    list(validation)
    assert loader.state_dict() == before
    iterator = iter(loader)
    first, second = next(iterator), next(iterator)
    with pytest.raises(ValueError, match="in order"):
        loader.mark_consumed(second)
    loader.mark_consumed(first)
    loader.mark_consumed(second)


def test_changed_schedule_cannot_restore_a_cursor(tmp_path):
    cfg = config(tmp_path)
    loader, _ = build_dataloaders(cfg, Tokenizer())
    cfg["data"]["schedule"] = ["t2i", "text", "i2t", "text"]
    other, _ = build_dataloaders(cfg, Tokenizer())
    with pytest.raises(ValueError, match="schedule"):
        other.load_state_dict(loader.state_dict())


def test_text_documents_keep_labels_at_same_position_and_exclude_first_token(tmp_path):
    ds = ClimbMixOnlineBatchDataset(shard_paths=text_files(tmp_path), tokenizer=Tokenizer(), eos_token_id=2,
                                   sequence_length=8, micro_batch_size=2, tokenizer_batch_documents=2)
    batch = next(iter(ds))
    active = batch["labels"].ne(-100)
    assert torch.equal(batch["input_ids"][active], batch["labels"][active])
    for row in range(2):
        for segment in batch["segment_ids"][row].unique():
            if segment < 0:
                continue
            first = batch["segment_ids"][row].eq(segment).nonzero()[0]
            assert batch["labels"][row, first].item() == -100
    assert batch["labels"][batch["token_types"].eq(2)].eq(2).any()


def test_fixed_text_validation_is_finite_and_deterministic(tmp_path):
    cfg = config(tmp_path)
    probe = tmp_path / "probe.jsonl"
    probe.write_text(json.dumps(dict(text="a validation document with several words")) + "\n")
    cfg["data"]["text"]["validation"] = dict(jsonl=str(probe), sequence_length=8)
    train, validation = build_dataloaders(cfg, Tokenizer())
    assert validation.task_lengths == dict(t2i=2, i2t=2, text=1)
    first, second = list(validation), list(validation)
    for a, b in zip(first, second):
        assert_batch_equal(a, b)
    assert train.state_dict()["micro_step"] == 0


def test_sharded_posterior_cache_preserves_storage_identity(tmp_path):
    metadata = dict(format=POSTERIOR_CACHE_FORMAT, stats_layout=POSTERIOR_STATS_LAYOUT)
    torch.save(dict(posterior_stats=torch.ones(1, 4, 4, dtype=torch.float16), img_ids=torch.tensor([7])), tmp_path / "shard.pt")
    torch.save(dict(img_ids=torch.tensor([1]), storage_img_ids=torch.tensor([7]), shard_rows=torch.tensor([[0, 0]])), tmp_path / "rows.pt")
    index = tmp_path / "index.json"
    index.write_text(json.dumps(dict(schema="kl16_posterior_shard_index_v1", row_index="rows.pt", shards=["shard.pt"], token_shape=[4, 4], metadata=metadata)))
    cache = load_sharded_posterior(index)
    assert torch.equal(cache["posterior_stats"][0], torch.ones(4, 4, dtype=torch.float16))
    assert cache["img_ids"].tolist() == [1]


def test_changed_batch_size_cannot_silently_restore_different_samples(tmp_path):
    cfg = config(tmp_path)
    train, _ = build_dataloaders(cfg, Tokenizer())
    state = train.state_dict()
    cfg["data"]["sources"]["t2i"]["micro_batch_size"] = 1
    other, _ = build_dataloaders(cfg, Tokenizer())
    with pytest.raises(ValueError, match="source_contract"):
        other.load_state_dict(state)


def test_standalone_validation_never_opens_training_cache(tmp_path):
    from data.dataloader import build_validation_loader
    cfg = config(tmp_path)
    cfg["data"]["image"]["cache_path"] = "/missing/training/cache.pt"
    cfg["data"]["text"]["shard_paths"] = ["/missing/training/text.jsonl"]
    validation = build_validation_loader(cfg, Tokenizer())
    batches = list(validation)
    assert len(batches) == 4
    assert batches[0]["sample_indices"].tolist() == [0, 1]
    assert batches[0]["img_ids"].tolist() == [1, 2]
    assert batches[0]["synsets"] == ["n00000001", "n00000001"]


def test_training_iteration_and_resume_do_not_advance_model_rng(tmp_path):
    cfg = config(tmp_path)
    train, _ = build_dataloaders(cfg, Tokenizer())
    before = torch.get_rng_state().clone()
    iterator = iter(train)
    for _ in range(8):
        train.mark_consumed(next(iterator))
    assert torch.equal(torch.get_rng_state(), before)
    resumed, _ = build_dataloaders(cfg, Tokenizer())
    resumed.load_state_dict(train.state_dict())
    next(iter(resumed))
    assert torch.equal(torch.get_rng_state(), before)


def test_indexed_caption_and_t2i_prompt_files_are_distinct(tmp_path):
    import struct
    from data.cache import INDEX_SCHEMA
    params = image_files(tmp_path, count=1)

    def write_indexed(path, rows):
        offsets = [0]
        with path.open("wb") as output:
            for row in rows:
                encoded = (json.dumps(row) + "\n").encode()
                output.write(encoded)
                offsets.append(offsets[-1] + len(encoded))
        index = path.with_suffix(".u64")
        index.write_bytes(b"".join(struct.pack("<Q", offset) for offset in offsets))
        return dict(path=path.name, offsets_path=index.name, records=len(rows))

    captions = ["caption source one", "caption source two"]
    prompts = ["prompt source one", "prompt source two"]
    caption = write_indexed(tmp_path / "indexed_caption.jsonl", [dict(manifest_index=0, img_id=1, captions=captions)])
    t2i = write_indexed(tmp_path / "indexed_t2i.jsonl", [dict(image_id="train/train_1", model_result=dict(prompts=[dict(prompt=text) for text in prompts]))])
    mapping = tmp_path / "mapping.bin"
    mapping.write_bytes(struct.pack("<BI", 0, 0))
    manifest = tmp_path / "seek.json"
    manifest.write_text(json.dumps(dict(schema=INDEX_SCHEMA, split="train", records=1,
                                       caption=caption, t2i=dict(shards=[dict(shard_index=0, **t2i)]),
                                       mapping=dict(path=mapping.name))))
    for task in ("t2i", "i2t"):
        ds = ImageNetFlowCacheDataset(**params, tokenizer=Tokenizer(), boi_token_id=3, eoi_token_id=4,
                                     mask_token_id=5, eos_token_id=2, synthetic_text_index_manifest=str(manifest),
                                     caption_sequence_modes=[task])
        sample = ds[0]
        chosen = int(sample["caption_index"])
        if task == "t2i":
            expected = Tokenizer().encode(ds.caption_t2i_prefix + " " + prompts[chosen])
            assert sample["input_ids"][:int(sample["prompt_len"])].tolist() == expected
        else:
            assert sample["labels"][sample["labels"] != -100].tolist() == Tokenizer().encode(captions[chosen]) + [2]


def test_loader_uses_separate_indexed_text_sources_for_train_and_validation(tmp_path):
    import struct
    from data.cache import INDEX_SCHEMA
    cfg = config(tmp_path)
    cfg["data"]["schedule"] = ["t2i", "i2t"]
    for params, split, count in ((cfg["data"]["image"], "train", 4),
                                 (cfg["data"]["image"]["validation"], "val", 3)):
        root = Path(params["manifest_jsonl"]).parent
        def write_rows(name, rows):
            path = root / name
            offsets = [0]
            with path.open("wb") as stream:
                for row in rows:
                    encoded = (json.dumps(row) + "\n").encode()
                    stream.write(encoded)
                    offsets.append(offsets[-1] + len(encoded))
            index = path.with_suffix(".u64")
            index.write_bytes(b"".join(struct.pack("<Q", n) for n in offsets))
            return dict(path=path.name, offsets_path=index.name, records=len(rows))
        caption = write_rows("indexed_caption.jsonl", [dict(manifest_index=i, img_id=i+1,
            captions=[f"indexed caption {split} item{i}"]) for i in range(count)])
        t2i = write_rows("indexed_prompt.jsonl", [dict(image_id=f"{split}/{split}_{i+1}",
            model_result=dict(prompts=[dict(prompt=f"independent prompt {split} item{i}")])) for i in range(count)])
        (root / "mapping.bin").write_bytes(b"".join(struct.pack("<BI", 0, i) for i in range(count)))
        manifest = root / "text-index.json"
        manifest.write_text(json.dumps(dict(schema=INDEX_SCHEMA, split=split, records=count, caption=caption,
            t2i=dict(shards=[dict(shard_index=0, **t2i)]), mapping=dict(path="mapping.bin"))))
        params["synthetic_text_index_manifest"] = str(manifest)
    train, validation = build_dataloaders(cfg, Tokenizer())
    for loaders, split in ((train.image_loaders, "train"), (validation.loaders, "val")):
        for task, loader in loaders.items():
            sample = loader.dataset[0]
            if task == "t2i":
                expected = Tokenizer().encode(loader.dataset.caption_t2i_prefix + f" independent prompt {split} item0")
                assert sample["input_ids"][:int(sample["prompt_len"])].tolist() == expected
            else:
                expected = Tokenizer().encode(f"indexed caption {split} item0") + [2]
                assert sample["labels"][sample["labels"].ne(-100)].tolist() == expected


def test_presets_keep_distinct_train_and_validation_prompt_indexes():
    import yaml
    root = Path(__file__).resolve().parents[1]
    for name in ("unified.yaml", "image.yaml"):
        image = yaml.safe_load((root / "configs" / name).read_text())["data"]["image"]
        train_index = image["synthetic_text_index_manifest"]
        val_index = image["validation"]["synthetic_text_index_manifest"]
        assert train_index != val_index
        assert "/train/" in train_index and "/val/" in val_index


@pytest.mark.parametrize("task", ["text", "t2i", "i2t"])
def test_single_task_schedule_has_independent_optimizer_accumulation(tmp_path, task):
    cfg = config(tmp_path)
    cfg["data"]["schedule"] = [task]
    cfg["training"]["gradient_accumulation_steps"] = 4
    train, _ = build_dataloaders(cfg, Tokenizer())
    assert len(train) == cfg["training"]["max_train_steps"] * 4
    assert train.state_dict(global_step=0)["micro_step"] == 0
    iterator = iter(train)
    train.mark_consumed(next(iterator))
    assert train.schedule_position == 0
    with pytest.raises(ValueError, match="optimizer boundary"):
        train.state_dict(global_step=1)
    for _ in range(3):
        batch = next(iterator)
        assert batch["task"] == task
        train.mark_consumed(batch)
    state = train.state_dict(global_step=1)
    assert state["micro_step"] == 4
    expected = next(iterator)
    resumed, _ = build_dataloaders(cfg, Tokenizer())
    resumed.load_state_dict(state)
    assert_batch_equal(next(iter(resumed)), expected)
    cfg["training"]["gradient_accumulation_steps"] = 2
    other, _ = build_dataloaders(cfg, Tokenizer())
    with pytest.raises(ValueError, match="accumulation_steps"):
        other.load_state_dict(state)
