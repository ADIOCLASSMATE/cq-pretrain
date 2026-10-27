"""Unified video loaders with deterministic source mixing and consumed cursors."""

from __future__ import annotations
import copy
from functools import partial
import hashlib
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from .dataset import VideoDataset, read_records
from .packing import collate_video_samples
from .cache import cache_identities, digest


def _set_epoch(loader, epoch, seed):
    pending, seen = [loader], set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if callable(getattr(node, "set_epoch", None)):
            node.set_epoch(epoch)
        generator = getattr(node, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(seed + epoch)
        for attr in (
            "base_dataloader",
            "dataloader",
            "dataset",
            "sampler",
            "batch_sampler",
        ):
            pending.append(getattr(node, attr, None))


class ValidationLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __len__(self):
        return sum(len(loader) for loader in self.loaders.values())

    def __iter__(self):
        for task, loader in self.loaders.items():
            for batch in loader:
                batch["task"] = task
                yield batch


class ScheduledDataLoader:
    """Only acknowledged microbatches advance checkpoints, including worker prefetch."""

    def __init__(
        self,
        loaders,
        schedule,
        *,
        seed,
        accumulation_steps,
        max_micro_steps,
        source_contract,
    ):
        if set(loaders) != set(schedule) or any(not len(x) for x in loaders.values()):
            raise ValueError("Every scheduled task needs a nonempty dataset")
        self.loaders, self.schedule = loaders, tuple(schedule)
        self.seed, self.accumulation_steps = int(seed), int(accumulation_steps)
        self.max_micro_steps, self.source_contract = max_micro_steps, source_contract
        self.rank, self.world_size, self._prepared = 0, 1, False
        self._iterators = {}
        self._fetched = self._initial_state()
        self._committed = copy.deepcopy(self._fetched)

    def _initial_state(self):
        return dict(
            schema="umm_video_consumed_v1",
            schedule=list(self.schedule),
            rank=self.rank,
            world_size=self.world_size,
            seed=self.seed,
            source_contract=self.source_contract,
            accumulation_steps=self.accumulation_steps,
            micro_step=0,
            sources={name: dict(epoch=0, batches_consumed=0) for name in self.loaders},
        )

    def __len__(self):
        return self.max_micro_steps

    def prepare_with_accelerator(self, accelerator):
        if self._prepared or self._iterators:
            raise RuntimeError("Prepare loaders once, before iteration")
        self.rank, self.world_size = (
            accelerator.process_index,
            accelerator.num_processes,
        )
        # CPU preprocessing stays in workers. Trainer moves nested tensors later.
        self.loaders = {
            k: accelerator.prepare_data_loader(v, device_placement=False)
            for k, v in self.loaders.items()
        }
        self._fetched = self._initial_state()
        self._committed = copy.deepcopy(self._fetched)
        self._prepared = True
        return self

    def _iterator(self, task):
        state = self._fetched["sources"][task]
        loader = self.loaders[task]
        _set_epoch(
            loader, state["epoch"], self.seed + 100003 * self.schedule.index(task)
        )
        iterator = iter(loader)
        for _ in range(state["batches_consumed"]):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError("Saved data cursor exceeds the epoch") from exc
        return iterator

    def __iter__(self):
        while True:
            task = self.schedule[self._fetched["micro_step"] % len(self.schedule)]
            if task not in self._iterators:
                self._iterators[task] = self._iterator(task)
            cursor = self._fetched["sources"][task]
            try:
                batch = next(self._iterators[task])
            except StopIteration:
                cursor.update(epoch=cursor["epoch"] + 1, batches_consumed=0)
                self._iterators[task] = self._iterator(task)
                try:
                    batch = next(self._iterators[task])
                except StopIteration as exc:
                    raise RuntimeError(f"Empty distributed {task} loader") from exc
            cursor["batches_consumed"] += 1
            self._fetched["micro_step"] += 1
            batch["task"] = task
            batch["_data_state"] = copy.deepcopy(self._fetched)
            yield batch

    def mark_consumed(self, batch):
        state = batch.get("_data_state")
        if (
            not isinstance(state, dict)
            or state["micro_step"] != self._committed["micro_step"] + 1
        ):
            raise ValueError("Consume batches exactly once, in emission order")
        self._committed = copy.deepcopy(state)

    def state_dict(self, *, global_step=None):
        if (
            global_step is not None
            and self._committed["micro_step"] != global_step * self.accumulation_steps
        ):
            raise ValueError("Data cursor and optimizer boundary differ")
        return copy.deepcopy(self._committed)

    def load_state_dict(self, state):
        if self._iterators:
            raise RuntimeError("Restore before iteration")
        expected = self._initial_state()
        for key in (
            "schema",
            "schedule",
            "rank",
            "world_size",
            "seed",
            "source_contract",
            "accumulation_steps",
        ):
            if state.get(key) != expected[key]:
                raise ValueError(f"Data resume contract changed: {key}")
        if (
            set(state.get("sources", {})) != set(self.loaders)
            or int(state.get("micro_step", -1)) < 0
        ):
            raise ValueError("Invalid data state")
        for cursor in state["sources"].values():
            if min(cursor.get("epoch", -1), cursor.get("batches_consumed", -1)) < 0:
                raise ValueError("Invalid data cursor")
        self._committed = copy.deepcopy(state)
        self._fetched = copy.deepcopy(state)


def _sources(config, processor, sources, *, training):
    data, settings = config["data"], config["training"]
    active = tuple(dict.fromkeys(data["schedule"]))
    groups = {task: [] for task in active}
    contract = []
    for source in sources:
        records = read_records(source)
        path = Path(source["path"]).resolve()
        assets = []
        for row in records:
            for key in ("path", "frames_path", "qwen_cache", "wan_cache"):
                name = row["videos"][0].get(key)
                if name:
                    p = Path(name)
                    assets.append(
                        (name, p.stat().st_size, p.stat().st_mtime_ns)
                        if p.is_file()
                        else (name, None)
                    )
        contract.append(
            dict(
                source=source,
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                assets=digest(assets),
            )
        )
        for task in active:
            selected = (
                records
                if task == "mixed"
                else [r for r in records if r["task"] == task]
            )
            if selected:
                if training and any(row["task"] == "v2t" for row in selected):
                    if config["model"]["lambda_text"] <= 0:
                        raise ValueError("V2T training requires positive lambda_text")
                    if (
                        not config["model"]["train_language"]
                        and config["model"]["freeze_vision"]
                    ):
                        raise ValueError(
                            "V2T training needs train_language or an unfrozen online ViT"
                        )
                groups[task].append(
                    (
                        VideoDataset(source, processor, config, records=selected),
                        source.get("weight"),
                    )
                )
    loaders = {}
    for task, members in groups.items():
        if not members:
            raise ValueError(
                f"No records for scheduled task {task}; set task_weights=0 to disable it"
            )
        dataset = ConcatDataset([x[0] for x in members])
        generator = torch.Generator().manual_seed(int(settings["seed"]))
        sampler = None
        if training and any(weight is not None for _, weight in members):
            if task == "mixed":
                raise ValueError(
                    "Preserve mixing does not resample source weights; use balanced mixing"
                )
            weights = []
            for subset, weight in members:
                weight = len(subset) if weight is None else float(weight)
                if not 0 < weight < float("inf"):
                    raise ValueError("Source weight must be finite and positive")
                weights.extend([weight / len(subset)] * len(subset))
            sampler = WeightedRandomSampler(
                weights, len(weights), replacement=True, generator=generator
            )
        workers = int(settings["dataloader_workers"])
        options = dict(
            num_workers=workers, pin_memory=False, persistent_workers=workers > 0
        )
        if workers:
            options["prefetch_factor"] = int(settings["dataloader_prefetch_factor"])
        loaders[task] = DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=training and sampler is None,
            sampler=sampler,
            generator=generator,
            drop_last=False,
            collate_fn=partial(
                collate_video_samples,
                pad_token_id=processor.tokenizer.pad_token_id or 0,
                pad_to_multiple_of=int(data["pad_to_multiple_of"]),
            ),
            **options,
        )
    return loaders, contract


def build_validation_loader(config, processor, sources=None):
    """Build the same ordered validation batches without requiring training data.

    The training schedule, batch size, worker seed and collation deliberately
    remain identical to training validation. Explicit sources replace only the
    validation manifests, never the supervised objective or stream weighting.
    """
    sources = config["data"]["validation_sources"] if sources is None else sources
    if not sources:
        raise ValueError("Provide data.validation_sources or task data.sources")
    config["cache_identity"] = cache_identities(config)
    loaders, _ = _sources(config, processor, sources, training=False)
    return ValidationLoader(loaders)


def build_dataloaders(config, processor):
    data, train = config["data"], config["training"]
    if not data["sources"]:
        raise ValueError(
            "Provide data.sources JSONL paths with row/source task=t2v or v2t"
        )
    config["cache_identity"] = cache_identities(config)
    loaders, contract = _sources(config, processor, data["sources"], training=True)
    train_loader = ScheduledDataLoader(
        loaders,
        data["schedule"],
        seed=train["seed"],
        accumulation_steps=train["gradient_accumulation_steps"],
        max_micro_steps=train["max_train_steps"] * train["gradient_accumulation_steps"],
        source_contract=digest(
            {
                "sources": contract,
                "data": data,
                "cache_identity": config["cache_identity"],
            }
        ),
    )
    validation = None
    if data["validation_sources"]:
        validation = build_validation_loader(config, processor)
    if train["val_every"] and validation is None:
        raise ValueError(
            "val_every requires explicit validation_sources; no automatic train/val split"
        )
    return train_loader, validation
