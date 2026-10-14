"""Loader assembly and exact recovery of consumed multimodal microbatches.

Workers may prefetch freely. Only ``mark_consumed(batch)`` advances the state
saved in a checkpoint; each emitted batch carries its own post-batch cursor.
"""
from __future__ import annotations

import copy
import glob
import inspect
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .dataset import ImageNetFlowCacheDataset
from .packing import collate_imagenet_flow_cache, collate_segment_packed
from .text_dataset import ClimbMixOnlineBatchDataset
from .text_validation import build_text_validation
from .manifest import image_validation_indices

DEFAULT_SCHEDULE = ("text", "t2i", "text", "i2t")
STATE_SCHEMA = "umm_v3_consumed_data_v1"


def _loader_nodes(loader):
    pending, seen = [loader], set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for attr in ("dataloader", "base_dataloader", "dataset", "sampler", "batch_sampler"):
            pending.append(getattr(node, attr, None))


def _set_epoch(loader, epoch, seed):
    for node in _loader_nodes(loader):
        if callable(getattr(node, "set_epoch", None)):
            node.set_epoch(epoch)
        generator = getattr(node, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(seed + epoch)


def _worker_options(workers, prefetch=4):
    result = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    if workers:
        result["prefetch_factor"] = prefetch
    return result


def _dataset(config, params, tokenizer):
    fields = inspect.signature(ImageNetFlowCacheDataset).parameters
    options = {k: v for k, v in params.items() if k in fields}
    model = config.get("model", {})
    for key in ("boi_token_id", "eoi_token_id", "mask_token_id"):
        options.setdefault(key, model[key])
    options.setdefault("image_tokens_per_img", model.get("image_tokens_per_img", 256))
    options.setdefault("image_latent_dim", model.get("image_latent_dim", 16))
    options.setdefault("eos_token_id", tokenizer.eos_token_id)
    options.setdefault("seed", config.get("training", {}).get("seed", 42))
    options.setdefault("image_sigma_order", "joint")
    if "joint" in config.get("data", {}).get("schedule", ()):
        options.setdefault("caption_sequence_modes", ("t2i", "i2t"))
    return ImageNetFlowCacheDataset(tokenizer=tokenizer, **options)


def _task_view(dataset, task, *, training):
    result = copy.copy(dataset)
    result.caption_sequence_modes = dataset.caption_sequence_modes if task == "joint" else (task,)
    result._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()
    result.text_cache, result.sequence_cache = {}, {}
    result.set_training_indices(range(len(result)) if training else [])
    if result.synthetic_text_index is not None:
        result.synthetic_text_index = result.synthetic_text_index.clone()
    if task in {"i2t", "joint"} and result.conditioning_mode != "caption":
        raise ValueError("I2T/joint supervision requires caption conditioning")
    return result


def _assert_independent_splits(train, validation):
    if train.dataset_split != "train" or validation.dataset_split != "val":
        raise ValueError("image training/validation require independent train/val manifests")
    identities = lambda data: {(data.synsets[i], Path(p).name) for i, p in data.source_paths_full.items()}
    overlap = identities(train) & identities(validation)
    if overlap:
        raise ValueError(f"image train/val identity overlap: {sorted(overlap)[:8]}")


def _image_collator(config, params, source, *, training):
    packing = params.get("packing", {})
    if training and packing.get("enabled", False):
        return partial(
            collate_segment_packed,
            nominal_capacity=int(packing.get("nominal_capacity", 2048)),
            image_uncond_prob=float(config.get("model", {}).get("image_uncond_prob", 0)),
            emit_audit_manifest=bool(packing.get("audit_manifests", False)),
        )
    return partial(
        collate_imagenet_flow_cache,
        pad_to_length=source.get("pad_to_length", params.get("pad_to_length")),
        pad_to_multiple_of=params.get("pad_to_multiple_of"),
    )


def _with_task(batch, task, latent_dim):
    batch["task"] = task
    batch["source_name"] = "climbmix" if task == "text" else task
    image_mask = batch["token_types"].eq(1)
    batch.setdefault("image_latent_mask", image_mask)
    if "image_latents" not in batch:
        batch["image_latents"] = torch.zeros(*image_mask.shape, latent_dim)
    return batch


def _image_validation_loader(config, dataset, params, source, task, *, loss_protocol):
    view = _task_view(dataset, task, training=False)
    evaluation = config.get("evaluation", {})
    per_class = evaluation.get("loss_image_per_class", 2)
    if loss_protocol and per_class is not None:
        indices = image_validation_indices(view, classes_path=evaluation.get("image_classes"),
            per_class=int(per_class), seed=int(evaluation.get("seed", 424242)))
        view = Subset(view, indices)
    training = config.get("training", {})
    workers = int(source.get("dataloader_workers", training.get("dataloader_workers", 0)))
    return DataLoader(view,
        batch_size=int(source.get("micro_batch_size", training.get("batch_size", 16))),
        shuffle=False, drop_last=False,
        collate_fn=_image_collator(config, params, source, training=loss_protocol),
        **_worker_options(workers, int(training.get("dataloader_prefetch_factor", 4))))


class ValidationLoader:
    """Visit each task's fixed validation split with an explicit task field."""

    def __init__(self, loaders, latent_dim):
        self.loaders = loaders
        self.latent_dim = latent_dim

    @property
    def task_lengths(self):
        return {task: len(loader) for task, loader in self.loaders.items()}

    def __len__(self):
        return sum(self.task_lengths.values())

    def __iter__(self):
        for task, loader in self.loaders.items():
            for batch in loader:
                yield _with_task(batch, task, self.latent_dim)

    def prepare_with_accelerator(self, accelerator):
        self.loaders = {task: accelerator.prepare_data_loader(loader) for task, loader in self.loaders.items()}
        return self


class ScheduledDataLoader:
    """Fixed task exposure with per-source epochs and acknowledged cursors.

    ``mark_consumed`` must be called in emission order after each successful
    training microbatch. A failed or prefetched batch is replayed after resume.
    ``prepare_with_accelerator`` shards image loaders exactly once; the text
    stream already owns rank-disjoint byte ranges and must not be sharded again.
    """

    def __init__(self, *, schedule, image_loaders, text_options=None, text_workers=0,
                 seed=42, latent_dim=16, max_micro_steps=0, prefetch_factor=4, source_contract=None,
                 accumulation_steps=None):
        self.schedule = tuple(schedule)
        if not self.schedule or set(self.schedule) - {"text", "t2i", "i2t", "joint"}:
            raise ValueError("schedule must contain text, t2i, i2t and/or joint")
        if set(image_loaders) != set(self.schedule) - {"text"}:
            raise ValueError("image loaders must match active image tasks")
        if ("text" in self.schedule) != (text_options is not None):
            raise ValueError("text options must match the schedule")
        if text_workers not in (0, 1):
            raise ValueError("exact text-stream recovery supports zero or one worker per rank")
        if any(len(loader) == 0 for loader in image_loaders.values()):
            raise ValueError("image loader is empty; reduce the microbatch size or disable drop_last")
        self.accumulation_steps = int(len(self.schedule) if accumulation_steps is None else accumulation_steps)
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self.image_loaders = image_loaders
        self.text_options = text_options
        self.text_workers = text_workers
        self.prefetch_factor = prefetch_factor
        self.seed, self.latent_dim = int(seed), int(latent_dim)
        self.max_micro_steps = int(max_micro_steps)
        self.source_contract = copy.deepcopy(source_contract)
        self.rank, self.world_size = 0, 1
        self._prepared = False
        self._iterators = {}
        self._text_loader = None
        self._fetched = self._initial_state()
        self._committed = copy.deepcopy(self._fetched)

    def _initial_state(self):
        return dict(schema=STATE_SCHEMA, schedule=list(self.schedule), rank=self.rank,
                    world_size=self.world_size, seed=self.seed, source_contract=self.source_contract,
                    accumulation_steps=self.accumulation_steps, micro_step=0,
                    image_sources={name: dict(epoch=0, batches_consumed=0) for name in self.image_loaders},
                    text=None)

    def __len__(self):
        if self.max_micro_steps <= 0:
            raise TypeError("infinite loader requires training.max_train_steps for a length")
        return self.max_micro_steps

    @property
    def schedule_position(self):
        return self._committed["micro_step"] % len(self.schedule)

    def prepare_with_accelerator(self, accelerator):
        if self._prepared or self._iterators:
            raise RuntimeError("prepare data loaders exactly once, before iteration")
        self.rank = int(accelerator.process_index)
        self.world_size = int(accelerator.num_processes)
        self.image_loaders = {name: accelerator.prepare_data_loader(loader) for name, loader in self.image_loaders.items()}
        self._fetched = self._initial_state()
        self._committed = copy.deepcopy(self._fetched)
        self._prepared = True
        return self

    def _image_iterator(self, task):
        state = self._fetched["image_sources"][task]
        loader = self.image_loaders[task]
        _set_epoch(loader, state["epoch"], self.seed)
        iterator = iter(loader)
        for index in range(state["batches_consumed"]):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError(f"invalid {task} data cursor: ended before batch {index}") from exc
        return iterator

    def _next_image(self, task):
        if task not in self._iterators:
            self._iterators[task] = self._image_iterator(task)
        state = self._fetched["image_sources"][task]
        try:
            batch = next(self._iterators[task])
        except StopIteration:
            state.update(epoch=state["epoch"] + 1, batches_consumed=0)
            self._iterators[task] = self._image_iterator(task)
            try:
                batch = next(self._iterators[task])
            except StopIteration as exc:
                raise RuntimeError(f"empty distributed {task} data loader") from exc
        state["batches_consumed"] += 1
        return batch

    def _next_text(self):
        if "text" not in self._iterators:
            dataset = ClimbMixOnlineBatchDataset(**self.text_options, rank=self.rank,
                                                world_size=self.world_size, resume_state=self._fetched["text"])
            self._text_loader = DataLoader(
                dataset, batch_size=None,
                generator=torch.Generator().manual_seed(self.seed + self.rank),
                **_worker_options(self.text_workers, self.prefetch_factor),
            )
            self._iterators["text"] = iter(self._text_loader)
        batch = next(self._iterators["text"])
        self._fetched["text"] = batch.pop("stream_state")
        return batch

    def __iter__(self):
        while True:
            position = self._fetched["micro_step"] % len(self.schedule)
            task = self.schedule[position]
            batch = self._next_text() if task == "text" else self._next_image(task)
            _with_task(batch, task, self.latent_dim)
            batch["source_schedule_position"] = position
            self._fetched["micro_step"] += 1
            batch["_data_state"] = copy.deepcopy(self._fetched)
            yield batch

    def mark_consumed(self, batch):
        state = batch.get("_data_state")
        if not isinstance(state, dict):
            raise ValueError("batch has no data cursor")
        expected_step = self._committed["micro_step"] + 1
        if state["micro_step"] != expected_step:
            raise ValueError(f"batches must be committed in order: expected {expected_step}, got {state['micro_step']}")
        self._committed = copy.deepcopy(state)

    def state_dict(self, *, global_step=None):
        if global_step is not None and self._committed["micro_step"] != int(global_step) * self.accumulation_steps:
            raise ValueError("checkpoint step does not match the committed optimizer boundary")
        return copy.deepcopy(self._committed)

    def load_state_dict(self, state):
        if self._iterators:
            raise RuntimeError("load data state before iteration")
        expected = self._initial_state()
        for key in ("schema", "schedule", "rank", "world_size", "seed", "source_contract", "accumulation_steps"):
            if state.get(key) != expected[key]:
                raise ValueError(f"data resume contract changed: {key}")
        if set(state.get("image_sources", {})) != set(self.image_loaders):
            raise ValueError("data resume image sources changed")
        if int(state.get("micro_step", -1)) < 0:
            raise ValueError("invalid consumed micro_step")
        for cursor in state["image_sources"].values():
            if int(cursor.get("epoch", -1)) < 0 or int(cursor.get("batches_consumed", -1)) < 0:
                raise ValueError("invalid image cursor")
        if "text" not in self.schedule and state.get("text") is not None:
            raise ValueError("inactive text source has a saved cursor")
        self._committed = copy.deepcopy(state)
        self._fetched = copy.deepcopy(state)


def build_dataloaders(config: dict, tokenizer):
    """Build independent validation and a committed-cursor training loader.

    Shared image storage is configured under ``data.image`` and text shards
    under ``data.text``. Per-task batch sizes and workers live in
    ``data.sources.{text,t2i,i2t,joint}``; ``data.schedule`` controls exposure.
    Joint batches retain the source dataset's caption_sequence_modes and masks.
    """
    data, training = config.get("data", {}), config.get("training", {})
    model = config.get("model", {})
    schedule = tuple(data.get("schedule", DEFAULT_SCHEDULE))
    active = tuple(dict.fromkeys(schedule))
    sources = data.get("sources", {})
    seed = int(training.get("dataloader_shuffle_seed", training.get("seed", 42)))
    image_loaders, val_loaders = {}, {}
    image_tasks = [task for task in active if task in {"t2i", "i2t", "joint"}]
    if image_tasks:
        params = data["image"]
        if not params.get("validation"):
            raise ValueError("data.image.validation is required for independent image validation")
        base = _dataset(config, params, tokenizer)
        val_params = {**params, **params["validation"]}
        validation = _dataset(config, val_params, tokenizer)
        _assert_independent_splits(base, validation)
        for task in image_tasks:
            source = sources.get(task, {})
            batch_size = int(source.get("micro_batch_size", training.get("batch_size", 16)))
            workers = int(source.get("dataloader_workers", training.get("dataloader_workers", 0)))
            options = _worker_options(workers, int(training.get("dataloader_prefetch_factor", 4)))
            image_loaders[task] = DataLoader(
                _task_view(base, task, training=True), batch_size=batch_size,
                shuffle=True, generator=torch.Generator().manual_seed(seed),
                drop_last=bool(source.get("drop_last", True)),
                collate_fn=_image_collator(config, params, source, training=True), **options,
            )
            val_loaders[task] = _image_validation_loader(
                config, validation, val_params, source, task, loss_protocol=True)
    text_options, text_workers = None, 0
    if "text" in active:
        params, source = data["text"], sources.get("text", {})
        paths = params.get("shard_paths") or sorted(glob.glob(params.get("shard_glob", "")))
        if not paths:
            raise FileNotFoundError("data.text shard paths/glob matched no files")
        fields = inspect.signature(ClimbMixOnlineBatchDataset).parameters
        text_options = {k: v for k, v in params.items() if k in fields and k not in {"rank", "world_size", "resume_state"}}
        text_options.update(shard_paths=paths, eos_token_id=tokenizer.eos_token_id,
                            seed=int(training.get("seed", 42)),
                            micro_batch_size=int(source.get("micro_batch_size", 4)))
        if not text_options.get("tokenizer_path"):
            text_options["tokenizer"] = tokenizer
        text_workers = int(source.get("dataloader_workers", 0))
        text_validation = build_text_validation(params, source, tokenizer, paths)
        if text_validation is not None:
            val_loaders["text"] = text_validation
    latent_dim = int(model.get("image_latent_dim", 16))
    accumulation_steps = int(training.get("gradient_accumulation_steps", len(schedule)))
    train = ScheduledDataLoader(schedule=schedule, image_loaders=image_loaders,
                               text_options=text_options, text_workers=text_workers, seed=seed,
                               latent_dim=latent_dim,
                               max_micro_steps=int(training.get("max_train_steps", 0)) * accumulation_steps,
                               accumulation_steps=accumulation_steps,
                               prefetch_factor=int(training.get("dataloader_prefetch_factor", 4)),
                               source_contract={"image": {k: v for k, v in data.get("image", {}).items() if k != "validation"},
                                                "text": {k: v for k, v in data.get("text", {}).items() if k != "validation"},
                                                "batch_sizes": {name: sources.get(name, {}).get("micro_batch_size", 4 if name == "text" else training.get("batch_size", 16)) for name in active},
                                                "padding": {name: sources.get(name, {}).get("pad_to_length") for name in image_tasks},
                                                "drop_last": {name: sources.get(name, {}).get("drop_last", True) for name in image_tasks}})
    validation = ValidationLoader(val_loaders, latent_dim) if val_loaders else None
    return train, validation


def build_validation_loader(config: dict, tokenizer, *, loss_protocol=True):
    """Assemble evaluation without opening any training cache or text stream."""
    data, training = config.get("data", {}), config.get("training", {})
    sources = data.get("sources", {})
    active = tuple(dict.fromkeys(data.get("schedule", DEFAULT_SCHEDULE)))
    loaders = {}
    image_tasks = [task for task in active if task in {"t2i", "i2t", "joint"}]
    if image_tasks:
        params = data["image"]
        if not params.get("validation"):
            raise ValueError("data.image.validation is required")
        params = {**params, **params["validation"]}
        dataset = _dataset(config, params, tokenizer)
        if dataset.dataset_split != "val":
            raise ValueError("standalone image evaluation requires a val manifest")
        for task in image_tasks:
            source = sources.get(task, {})
            loaders[task] = _image_validation_loader(
                config, dataset, params, source, task, loss_protocol=loss_protocol)
    if "text" in active:
        params = data.get("text", {})
        paths = params.get("shard_paths") or sorted(glob.glob(params.get("shard_glob", "")))
        text_loader = build_text_validation(params, sources.get("text", {}), tokenizer, paths)
        if text_loader is not None:
            loaders["text"] = text_loader
    return ValidationLoader(loaders, int(config.get("model", {}).get("image_latent_dim", 16))) if loaders else None
