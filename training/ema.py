"""Rank-sharded FP32 exponential moving averages.

MCore keeps a replicated model after parameter gathering, so every rank can read
the parameters it owns in the EMA layout without further communication. Large tensors
are split into deterministic flat chunks before assignment; this avoids placing
the whole tied token embedding on a single rank.
"""

from __future__ import annotations

import heapq
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from utils.file_sync import run_io_phase


EMA_SCHEMA = "selfless_rank_sharded_fp32_ema_v1"
DEFAULT_EMA_CHUNK_NUMEL = 4 * 1024 * 1024


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported EMA dtype in manifest: {name!r}")
    return dtype


def _ema_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype.is_floating_point else dtype


def _tensor_view_identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    if tensor.numel() == 0:
        storage_identity: Any = ("empty", id(tensor))
    else:
        storage = tensor.untyped_storage()
        storage_identity = (
            tensor.device.type,
            tensor.device.index,
            storage.data_ptr(),
        )
    return (
        storage_identity,
        int(tensor.storage_offset()),
        tuple(int(item) for item in tensor.shape),
        tuple(int(item) for item in tensor.stride()),
        _dtype_name(tensor.dtype),
    )


def _layout_contract(layout: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in layout.items()
        if key not in {"layout_fingerprint", "runtime"}
    }


def build_sharded_ema_layout(
    model: torch.nn.Module,
    *,
    world_size: int,
    chunk_numel: int = DEFAULT_EMA_CHUNK_NUMEL,
) -> dict[str, Any]:
    """Build a deterministic tied-aware, size-balanced EMA layout."""

    world_size = int(world_size)
    chunk_numel = int(chunk_numel)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if chunk_numel <= 0:
        raise ValueError(f"chunk_numel must be positive, got {chunk_numel}")

    state = model.state_dict(keep_vars=True)
    if not state:
        raise ValueError("Cannot build EMA layout for an empty state_dict")

    alias_groups: dict[tuple[Any, ...], list[str]] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{name!r}] is not a tensor")
        alias_groups.setdefault(_tensor_view_identity(tensor), []).append(name)

    tensors: dict[str, dict[str, Any]] = {}
    canonical_for_name: dict[str, str] = {}
    chunk_candidates: list[dict[str, Any]] = []
    next_chunk_index = 0
    for aliases in sorted((sorted(names) for names in alias_groups.values()), key=lambda x: x[0]):
        canonical_name = aliases[0]
        tensor = state[canonical_name]
        if not tensor.is_contiguous() and tensor.numel() > chunk_numel:
            raise ValueError(
                "Large non-contiguous state tensors are unsupported by sharded EMA: "
                f"{canonical_name} shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}"
            )
        source_dtype = _dtype_name(tensor.dtype)
        ema_dtype = _ema_dtype(tensor.dtype)
        ema_dtype_name = _dtype_name(ema_dtype)
        numel = int(tensor.numel())
        tensors[canonical_name] = {
            "aliases": aliases,
            "shape": [int(item) for item in tensor.shape],
            "stride": [int(item) for item in tensor.stride()],
            "numel": numel,
            "source_dtype": source_dtype,
            "ema_dtype": ema_dtype_name,
        }
        for name in aliases:
            canonical_for_name[name] = canonical_name

        for offset in range(0, numel, chunk_numel):
            length = min(chunk_numel, numel - offset)
            chunk_candidates.append(
                {
                    "id": f"chunk_{next_chunk_index:08d}",
                    "tensor": canonical_name,
                    "offset": offset,
                    "numel": length,
                    "bytes": length * torch.empty((), dtype=ema_dtype).element_size(),
                }
            )
            next_chunk_index += 1

    rank_heap = [(0, rank) for rank in range(world_size)]
    heapq.heapify(rank_heap)
    chunks_by_id: dict[str, dict[str, Any]] = {}
    for chunk in sorted(
        chunk_candidates,
        key=lambda item: (-item["bytes"], item["tensor"], item["offset"]),
    ):
        rank_bytes, owner = heapq.heappop(rank_heap)
        chunk["owner"] = owner
        chunks_by_id[chunk["id"]] = chunk
        heapq.heappush(rank_heap, (rank_bytes + chunk["bytes"], owner))

    rank_bytes = [0] * world_size
    rank_chunk_count = [0] * world_size
    for chunk in chunks_by_id.values():
        rank_bytes[chunk["owner"]] += int(chunk["bytes"])
        rank_chunk_count[chunk["owner"]] += 1

    layout: dict[str, Any] = {
        "schema": EMA_SCHEMA,
        "world_size": world_size,
        "chunk_numel": chunk_numel,
        "state_keys": list(state.keys()),
        "canonical_for_name": canonical_for_name,
        "tensors": tensors,
        "chunks": chunks_by_id,
        "rank_bytes": rank_bytes,
        "rank_chunk_count": rank_chunk_count,
    }
    layout["layout_validation"] = "readable_field_equality"
    return layout


def _manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.name == "ema_manifest.json" else path / "ema_manifest.json"


def _validate_readable_manifest_structure(manifest: dict[str, Any]) -> None:
    world_size = int(manifest.get("world_size", 0))
    chunks = manifest.get("chunks")
    tensors = manifest.get("tensors")
    if world_size <= 0 or not isinstance(chunks, dict) or not isinstance(tensors, dict):
        raise ValueError("EMA manifest has an invalid readable layout structure")
    rank_bytes = [0] * world_size
    rank_chunk_count = [0] * world_size
    for chunk_id, chunk in chunks.items():
        if chunk.get("id") != chunk_id:
            raise ValueError(f"EMA chunk id mismatch: {chunk_id!r}")
        owner = int(chunk.get("owner", -1))
        tensor_name = chunk.get("tensor")
        if owner < 0 or owner >= world_size or tensor_name not in tensors:
            raise ValueError(f"EMA chunk has invalid owner/tensor: {chunk_id!r}")
        numel = int(chunk.get("numel", -1))
        offset = int(chunk.get("offset", -1))
        tensor_numel = int(tensors[tensor_name].get("numel", -1))
        expected_bytes = numel * torch.empty(
            (), dtype=_dtype_from_name(tensors[tensor_name]["ema_dtype"])
        ).element_size()
        if numel <= 0 or offset < 0 or offset + numel > tensor_numel:
            raise ValueError(f"EMA chunk range is invalid: {chunk_id!r}")
        if int(chunk.get("bytes", -1)) != expected_bytes:
            raise ValueError(f"EMA chunk byte count is invalid: {chunk_id!r}")
        rank_bytes[owner] += expected_bytes
        rank_chunk_count[owner] += 1
    if manifest.get("rank_bytes") != rank_bytes:
        raise ValueError("EMA manifest rank_bytes does not match its chunks")
    if manifest.get("rank_chunk_count") != rank_chunk_count:
        raise ValueError("EMA manifest rank_chunk_count does not match its chunks")
    for name, metadata in tensors.items():
        ranges = sorted((int(chunk["offset"]), int(chunk["numel"]))
                        for chunk in chunks.values() if chunk["tensor"] == name)
        expected_offset = 0
        for offset, length in ranges:
            if offset != expected_offset:
                raise ValueError(f"EMA chunks overlap or leave a gap in {name}")
            expected_offset += length
        if expected_offset != int(metadata["numel"]):
            raise ValueError(f"EMA chunks do not cover complete tensor {name}")


def load_ema_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing sharded EMA manifest: {manifest_path}. "
            "Legacy ema_state.pt checkpoints are not supported."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EMA_SCHEMA:
        raise ValueError(
            f"Unsupported EMA manifest schema {manifest.get('schema')!r}; "
            f"expected {EMA_SCHEMA!r}."
        )
    if manifest.get("layout_validation") != "readable_field_equality":
        raise ValueError(
            f"EMA manifest must use readable field validation: {manifest_path}"
        )
    _validate_readable_manifest_structure(manifest)
    return manifest


def _expected_chunk_ids(layout: dict[str, Any], rank: int) -> list[str]:
    return sorted(
        chunk_id
        for chunk_id, chunk in layout["chunks"].items()
        if int(chunk["owner"]) == int(rank)
    )


def _shard_filename(rank: int) -> str:
    return f"ema_shard_rank_{int(rank):05d}.safetensors"


def validate_sharded_ema_layout_for_model(
    layout: dict[str, Any], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    state = model.state_dict(keep_vars=True)
    expected_keys = list(layout["state_keys"])
    if list(state.keys()) != expected_keys:
        missing = sorted(set(expected_keys) - set(state))
        unexpected = sorted(set(state) - set(expected_keys))
        raise RuntimeError(
            "EMA/model state mismatch: "
            f"missing={missing}, unexpected={unexpected}, order_matches=False"
        )

    for canonical_name, metadata in layout["tensors"].items():
        aliases = metadata["aliases"]
        identities = {_tensor_view_identity(state[name]) for name in aliases}
        if len(identities) != 1:
            raise RuntimeError(
                f"EMA tied-weight group is no longer tied: {canonical_name}: {aliases}"
            )
        tensor = state[canonical_name]
        if list(tensor.shape) != metadata["shape"]:
            raise RuntimeError(
                f"EMA tensor shape mismatch for {canonical_name}: "
                f"model={list(tensor.shape)}, manifest={metadata['shape']}"
            )
        if _dtype_name(tensor.dtype) != metadata["source_dtype"]:
            raise RuntimeError(
                f"EMA tensor dtype mismatch for {canonical_name}: "
                f"model={tensor.dtype}, manifest={metadata['source_dtype']}"
            )
    return state


class RankShardedEMA:
    """The local rank's portion of a globally sharded FP32 EMA."""

    def __init__(
        self,
        layout: dict[str, Any],
        *,
        rank: int,
        decay: float,
        update_after_step: int,
    ) -> None:
        self.layout = layout
        self.rank = int(rank)
        self.world_size = int(layout["world_size"])
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is outside world_size={self.world_size}")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"EMA decay must be in [0,1), got {self.decay}")
        if self.update_after_step < 0:
            raise ValueError(
                f"EMA update_after_step must be non-negative, got {self.update_after_step}"
            )
        self.local_chunk_ids = _expected_chunk_ids(layout, self.rank)
        self.shards: dict[str, torch.Tensor] = {}
        self.started = False
        self.global_step = 0
        self._source_state: dict[str, torch.Tensor] = {}

    @property
    def local_bytes(self) -> int:
        return int(self.layout["rank_bytes"][self.rank])

    def bind(self, model: torch.nn.Module) -> None:
        state = validate_sharded_ema_layout_for_model(self.layout, model)
        local_names = {
            self.layout["chunks"][chunk_id]["tensor"]
            for chunk_id in self.local_chunk_ids
        }
        self._source_state = {name: state[name] for name in local_names}

    def _source_chunk(self, chunk_id: str) -> torch.Tensor:
        chunk = self.layout["chunks"][chunk_id]
        source = self._source_state[chunk["tensor"]].detach()
        flat = source.view(-1) if source.is_contiguous() else source.reshape(-1)
        return flat.narrow(0, int(chunk["offset"]), int(chunk["numel"]))

    @torch.no_grad()
    def sync_from_model(self) -> None:
        if not self._source_state and self.local_chunk_ids:
            raise RuntimeError("RankShardedEMA must be bound to a model before syncing")
        new_shards: dict[str, torch.Tensor] = {}
        for chunk_id in self.local_chunk_ids:
            chunk = self.layout["chunks"][chunk_id]
            tensor_meta = self.layout["tensors"][chunk["tensor"]]
            source = self._source_chunk(chunk_id)
            ema_dtype = _dtype_from_name(tensor_meta["ema_dtype"])
            if source.dtype == ema_dtype:
                # torch-npu 2.6 may alias a same-device, same-dtype
                # ``to(copy=True)`` result. EMA storage must never alias the
                # live model or its in-place update becomes self-referential.
                shard = source.clone(memory_format=torch.contiguous_format)
            else:
                shard = source.to(
                    dtype=ema_dtype,
                    copy=True,
                    non_blocking=True,
                ).contiguous()
            new_shards[chunk_id] = shard
        self.shards = new_shards

    @torch.no_grad()
    def update_from_model(self) -> None:
        if set(self.shards) != set(self.local_chunk_ids):
            raise RuntimeError("EMA local shards have not been initialized")
        for chunk_id in self.local_chunk_ids:
            ema_value = self.shards[chunk_id]
            source_value = self._source_chunk(chunk_id)
            if source_value.device != ema_value.device:
                source_value = source_value.to(
                    device=ema_value.device,
                    non_blocking=True,
                )
            if ema_value.dtype.is_floating_point:
                # add_ accepts BF16/FP16 sources for an FP32 destination.  Do
                # the exact same conversion inside the fused pointwise op
                # instead of allocating a full-size FP32 temporary per chunk.
                ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source_value)

    @torch.no_grad()
    def initialize_from_model(self, *, global_step: int = 0) -> None:
        self.sync_from_model()
        self.global_step = int(global_step)
        self.started = self.global_step >= self.update_after_step

    @torch.no_grad()
    def maybe_update(self, next_step: int) -> bool:
        next_step = int(next_step)
        if next_step < self.update_after_step:
            self.global_step = next_step
            return self.started
        if not self.started and self.update_after_step > 0:
            self.sync_from_model()
            self.started = True
        else:
            self.update_from_model()
            self.started = True
        self.global_step = next_step
        return self.started

    @contextmanager
    @torch.no_grad()
    def applied_to(self, model: torch.nn.Module):
        """Temporarily install the distributed EMA in a replicated model.

        CPU backups preserve live parameters, buffers and tied storage exactly.
        Broadcast buffers never alias the FP32 EMA. Optimizer/master parameters
        and EMA updates are untouched; callers must invoke this on every rank.
        """
        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        if self.world_size != (dist.get_world_size() if distributed else 1):
            raise RuntimeError("EMA evaluation requires the complete training process group")
        if self.rank != (dist.get_rank() if distributed else 0):
            raise RuntimeError("EMA shard owner does not match the training rank")
        if set(self.shards) != set(self.local_chunk_ids):
            raise RuntimeError("EMA evaluation requires initialized local shards")
        state = validate_sharded_ema_layout_for_model(self.layout, model)
        backups = {
            name: state[name].detach().to("cpu", copy=True)
            for name in self.layout["tensors"]
        }
        try:
            with torch.no_grad():
                for chunk_id, chunk in self.layout["chunks"].items():
                    target = state[chunk["tensor"]]
                    value = torch.empty(
                        int(chunk["numel"]), device=target.device, dtype=target.dtype
                    )
                    if self.rank == int(chunk["owner"]):
                        value.copy_(self.shards[chunk_id])
                    if distributed:
                        dist.broadcast(value, src=int(chunk["owner"]))
                    # reshape may copy noncontiguous buffers: copy back explicitly.
                    flat = target.reshape(-1)
                    flat.narrow(0, int(chunk["offset"]), int(chunk["numel"])).copy_(value)
                    if not target.is_contiguous():
                        target.copy_(flat.reshape(target.shape))
                yield model
        finally:
            with torch.no_grad():
                for name, backup in backups.items():
                    state[name].copy_(backup)

    def save_checkpoint(self, directory: str | Path, accelerator, *, global_step: int) -> Path:
        directory = Path(directory)

        def write_shard():
            directory.mkdir(parents=True, exist_ok=True)
            shard_path = directory / _shard_filename(self.rank)
            temp_path = directory / f".{shard_path.name}.tmp-{os.getpid()}"
            cpu_state = {
                chunk_id: tensor.detach().to(device="cpu", non_blocking=False).contiguous()
                for chunk_id, tensor in self.shards.items()
            }
            save_file(
                cpu_state,
                str(temp_path),
                metadata={
                    "schema": EMA_SCHEMA,
                    "rank": str(self.rank),
                    "world_size": str(self.world_size),
                    "layout_validation": "readable_field_equality",
                },
            )
            os.replace(temp_path, shard_path)
            del cpu_state

        run_io_phase(accelerator, write_shard, description="EMA shard save")

        manifest_path = directory / "ema_manifest.json"
        def write_manifest():
            manifest = dict(self.layout)
            manifest["runtime"] = {
                "global_step": int(global_step),
                "decay": self.decay,
                "update_after_step": self.update_after_step,
                "started": bool(self.started),
            }
            temp_manifest = directory / f".{manifest_path.name}.tmp-{os.getpid()}"
            temp_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_manifest, manifest_path)
        run_io_phase(accelerator, write_manifest, description="EMA manifest save", main_process_only=True)
        return manifest_path

    def load_checkpoint(
        self,
        directory: str | Path,
        accelerator,
        *,
        expected_global_step: int | None = None,
    ) -> None:
        return run_io_phase(
            accelerator,
            lambda: self._load_local_checkpoint(directory, expected_global_step=expected_global_step),
            description="EMA shard restore",
        )

    def _load_local_checkpoint(
        self,
        directory: str | Path,
        *,
        expected_global_step: int | None = None,
    ) -> None:
        directory = Path(directory)
        manifest = load_ema_manifest(directory)
        if int(manifest["world_size"]) != self.world_size:
            raise RuntimeError(
                "Sharded EMA requires the same world size when resuming: "
                f"checkpoint={manifest['world_size']}, current={self.world_size}"
            )
        if _layout_contract(manifest) != _layout_contract(self.layout):
            raise RuntimeError(
                "Readable sharded EMA layout does not match the current "
                "model/config."
            )
        runtime = manifest.get("runtime") or {}
        if float(runtime.get("decay", -1.0)) != self.decay:
            raise RuntimeError(
                f"EMA decay mismatch: checkpoint={runtime.get('decay')}, current={self.decay}"
            )
        if int(runtime.get("update_after_step", -1)) != self.update_after_step:
            raise RuntimeError(
                "EMA update_after_step mismatch: "
                f"checkpoint={runtime.get('update_after_step')}, current={self.update_after_step}"
            )
        checkpoint_step = int(runtime.get("global_step", -1))
        if expected_global_step is not None and checkpoint_step != int(expected_global_step):
            raise RuntimeError(
                f"EMA/global checkpoint step mismatch: ema={checkpoint_step}, "
                f"training={expected_global_step}"
            )

        for rank in range(self.world_size):
            shard_path = directory / _shard_filename(rank)
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing EMA shard for rank {rank}: {shard_path}")

        shard_path = directory / _shard_filename(self.rank)
        expected_ids = self.local_chunk_ids
        loaded: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            actual_ids = sorted(handle.keys())
            if actual_ids != expected_ids:
                raise RuntimeError(
                    f"EMA shard key mismatch for rank {self.rank}: "
                    f"expected={expected_ids}, actual={actual_ids}"
                )
            for chunk_id in expected_ids:
                chunk = self.layout["chunks"][chunk_id]
                tensor_meta = self.layout["tensors"][chunk["tensor"]]
                value = handle.get_tensor(chunk_id)
                expected_dtype = _dtype_from_name(tensor_meta["ema_dtype"])
                if value.shape != (int(chunk["numel"]),) or value.dtype != expected_dtype:
                    raise RuntimeError(
                        f"EMA shard tensor mismatch for {chunk_id}: "
                        f"shape={tuple(value.shape)}, dtype={value.dtype}"
                    )
                source_device = self._source_state[chunk["tensor"]].device
                loaded[chunk_id] = value.to(device=source_device, non_blocking=False)
        self.shards = loaded
        self.started = bool(runtime.get("started", False))
        self.global_step = checkpoint_step


def _selected_canonical_names(
    manifest: dict[str, Any], names: Iterable[str] | None
) -> set[str]:
    if names is None:
        return set(manifest["tensors"])
    selected = set()
    for name in names:
        canonical = manifest["canonical_for_name"].get(name)
        if canonical is None:
            raise KeyError(f"EMA state key is absent from manifest: {name}")
        selected.add(canonical)
    return selected


def merge_sharded_ema_state_dict(
    directory: str | Path,
    *,
    names: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Merge selected EMA tensors on CPU.  No CUDA tensor is created."""

    directory = _manifest_path(directory).parent
    manifest = load_ema_manifest(directory)
    selected_canonical = _selected_canonical_names(manifest, names)
    canonical_state: dict[str, torch.Tensor] = {}
    for canonical_name in selected_canonical:
        metadata = manifest["tensors"][canonical_name]
        canonical_state[canonical_name] = torch.empty(
            metadata["shape"], dtype=_dtype_from_name(metadata["ema_dtype"]), device="cpu"
        )

    seen_chunks: set[str] = set()
    for rank in range(int(manifest["world_size"])):
        shard_path = directory / _shard_filename(rank)
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing EMA shard for rank {rank}: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            actual_ids = set(handle.keys())
            expected_ids = set(_expected_chunk_ids(manifest, rank))
            if actual_ids != expected_ids:
                raise RuntimeError(
                    f"EMA shard key mismatch for rank {rank}: "
                    f"missing={sorted(expected_ids - actual_ids)}, "
                    f"unexpected={sorted(actual_ids - expected_ids)}"
                )
            for chunk_id in sorted(actual_ids):
                chunk = manifest["chunks"][chunk_id]
                canonical_name = chunk["tensor"]
                if canonical_name not in selected_canonical:
                    continue
                value = handle.get_tensor(chunk_id)
                expected_dtype = _dtype_from_name(manifest["tensors"][canonical_name]["ema_dtype"])
                if value.shape != (int(chunk["numel"]),) or value.dtype != expected_dtype:
                    raise RuntimeError(f"EMA merge tensor shape/dtype mismatch: {chunk_id}")
                target = canonical_state[canonical_name].view(-1)
                target.narrow(0, int(chunk["offset"]), int(chunk["numel"])).copy_(value)
                seen_chunks.add(chunk_id)

    expected_selected_chunks = {
        chunk_id
        for chunk_id, chunk in manifest["chunks"].items()
        if chunk["tensor"] in selected_canonical
    }
    if seen_chunks != expected_selected_chunks:
        raise RuntimeError(
            "EMA merge did not cover every selected chunk: "
            f"missing={sorted(expected_selected_chunks - seen_chunks)}"
        )

    selected_names = set(manifest["state_keys"]) if names is None else set(names)
    merged: dict[str, torch.Tensor] = {}
    for name in manifest["state_keys"]:
        if name not in selected_names:
            continue
        canonical_name = manifest["canonical_for_name"][name]
        merged[name] = canonical_state[canonical_name]
    return merged


def load_sharded_ema_checkpoint(
    model: torch.nn.Module,
    directory: str | Path,
) -> dict[str, Any]:
    """Strictly load a complete rank-sharded EMA into ``model`` on CPU."""

    if not str(directory):
        return {"ema_checkpoint": None}
    path = Path(directory)
    manifest = load_ema_manifest(path)
    state_dict = merge_sharded_ema_state_dict(path)
    model.load_state_dict(state_dict, strict=True)
    runtime = manifest.get("runtime") or {}
    return {
        "ema_checkpoint": str(path),
        "global_step": runtime.get("global_step"),
        "decay": runtime.get("decay"),
        "keys": len(state_dict),
        "missing": [],
        "unexpected": [],
    }

