"""Lazy posterior shards and indexed caption JSONL access."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
import struct
from pathlib import Path
from typing import Any

import torch


POSTERIOR_INDEX_SCHEMA = "kl16_posterior_shard_index_v1"


class ShardedPosterior:
    def __init__(self, paths, shard_rows, token_shape, img_ids, limit=None, storage_img_ids=None):
        self.paths = paths
        self.shard_rows = shard_rows
        self.img_ids = img_ids
        self.storage_img_ids = img_ids if storage_img_ids is None else storage_img_ids
        self.limit = len(shard_rows) if limit is None else limit
        self.shape = (self.limit, *token_shape)
        self.ndim = 3
        self.dtype = torch.float16
        self._opened = OrderedDict()

    def is_floating_point(self):
        return True

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.limit)
            if start != 0 or step != 1:
                raise ValueError("sharded posterior only supports prefix slices")
            return type(self)(self.paths, self.shard_rows, self.shape[1:], self.img_ids, stop,
                              self.storage_img_ids)
        index = int(index)
        if index < 0 or index >= self.limit:
            raise IndexError(index)
        shard, row = map(int, self.shard_rows[index])
        if shard not in self._opened:
            payload = torch.load(self.paths[shard], mmap=True, map_location="cpu", weights_only=True)
            stats = payload["posterior_stats"]
            if tuple(stats.shape[1:]) != self.shape[1:] or stats.dtype != self.dtype:
                raise ValueError(f"shard shape/dtype changed: {self.paths[shard]}")
            self._opened[shard] = (stats, payload["img_ids"])
            if len(self._opened) > 32:
                self._opened.popitem(last=False)
        self._opened.move_to_end(shard)
        stats, ids = self._opened[shard]
        if int(ids[row]) != int(self.storage_img_ids[index]):
            raise ValueError(f"shard image identity changed: {self.paths[shard]} row {row}")
        return stats[row]

    def __getstate__(self):
        return {**self.__dict__, "_opened": OrderedDict()}


def load_sharded_posterior(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != POSTERIOR_INDEX_SCHEMA:
        raise ValueError(f"invalid posterior shard index: {path}")
    def resolve(value):
        value = Path(value)
        return str(value if value.is_absolute() else path.parent / value)
    index = torch.load(resolve(manifest["row_index"]), mmap=True, map_location="cpu", weights_only=True)
    ids, rows = index["img_ids"], index["shard_rows"]
    storage_ids = index.get("storage_img_ids", ids)
    if rows.shape != (len(ids), 2) or rows.dtype != torch.int64 or ids.ndim != 1 or ids.dtype != torch.int64:
        raise ValueError("invalid posterior row index")
    if storage_ids.shape != ids.shape or storage_ids.dtype != torch.int64:
        raise ValueError("invalid posterior storage image IDs")
    paths = [resolve(value) for value in manifest["shards"]]
    if len(ids) and (bool((rows < 0).any()) or int(rows[:, 0].max()) >= len(paths)):
        raise ValueError("posterior row index references an invalid shard/row")
    return {
        "posterior_stats": ShardedPosterior(paths, rows, manifest["token_shape"], ids,
                                            storage_img_ids=storage_ids),
        "img_ids": ids,
        "metadata": manifest["metadata"],
    }


INDEX_SCHEMA = "imagenet1k_synthetic_text_seek_index_v1"
MAPPING_STRUCT = struct.Struct("<BI")
OFFSET_STRUCT = struct.Struct("<Q")


def _read_exact_at(descriptor: int, size: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    position = int(offset)
    while remaining:
        chunk = os.pread(descriptor, remaining, position)
        if not chunk:
            break
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    value = b"".join(chunks)
    if len(value) != size:
        raise ValueError(
            f"short indexed read: offset={offset}, expected={size}, got={len(value)}"
        )
    return value


class OffsetJsonlReader:
    """Read one JSONL record with an N+1 little-endian uint64 offset table."""

    def __init__(self, path: Path, offsets_path: Path, row_count: int) -> None:
        self.path = Path(path)
        self.offsets_path = Path(offsets_path)
        self.row_count = int(row_count)
        self._data_fd: int | None = None
        self._offsets_fd: int | None = None
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not self.offsets_path.is_file():
            raise FileNotFoundError(self.offsets_path)
        expected_offsets_bytes = (self.row_count + 1) * OFFSET_STRUCT.size
        actual_offsets_bytes = self.offsets_path.stat().st_size
        if actual_offsets_bytes != expected_offsets_bytes:
            raise ValueError(
                f"corrupt offset table {self.offsets_path}: "
                f"expected {expected_offsets_bytes} bytes, got {actual_offsets_bytes}"
            )
        with self.offsets_path.open("rb") as handle:
            handle.seek(self.row_count * OFFSET_STRUCT.size)
            final_offset = OFFSET_STRUCT.unpack(handle.read(OFFSET_STRUCT.size))[0]
        if final_offset != self.path.stat().st_size:
            raise ValueError(
                f"final offset mismatch for {self.path}: "
                f"offset={final_offset}, bytes={self.path.stat().st_size}"
            )

    def _open(self) -> tuple[int, int]:
        if self._data_fd is None:
            self._data_fd = os.open(self.path, os.O_RDONLY)
        if self._offsets_fd is None:
            self._offsets_fd = os.open(self.offsets_path, os.O_RDONLY)
        return self._data_fd, self._offsets_fd

    def read(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0 or index >= self.row_count:
            raise IndexError(index)
        data_fd, offsets_fd = self._open()
        offset_pair = _read_exact_at(
            offsets_fd,
            2 * OFFSET_STRUCT.size,
            index * OFFSET_STRUCT.size,
        )
        start, end = struct.unpack("<QQ", offset_pair)
        if end <= start:
            raise ValueError(
                f"invalid JSONL offsets for {self.path} row {index}: {start}, {end}"
            )
        encoded = _read_exact_at(data_fd, end - start, start).rstrip(b"\r\n")
        value = json.loads(encoded)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSON in {self.path} row {index}")
        return value

    def close(self) -> None:
        for descriptor in (self._data_fd, self._offsets_fd):
            if descriptor is not None:
                os.close(descriptor)
        self._data_fd = None
        self._offsets_fd = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_data_fd"] = None
        state["_offsets_fd"] = None
        return state

    def __del__(self) -> None:
        self.close()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


class ImageNetSyntheticTextIndex:
    """Resolve aligned caption and T2I records by posterior-cache row."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.caption_reader: OffsetJsonlReader | None = None
        self.t2i_readers: list[OffsetJsonlReader] = []
        self._mapping_fd: int | None = None
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != INDEX_SCHEMA:
            raise ValueError(
                f"unsupported synthetic text index schema in {self.manifest_path}: "
                f"{self.manifest.get('schema')!r}"
            )
        self.split = str(self.manifest.get("split", "")).strip().lower()
        if self.split not in {"train", "val"}:
            raise ValueError(
                "synthetic text index split must be 'train' or 'val', "
                f"got {self.split!r}"
            )
        self.row_count = int(self.manifest["records"])
        base = self.manifest_path.parent

        caption = self.manifest["caption"]
        self.caption_reader = OffsetJsonlReader(
            _resolve(base, caption["path"]),
            _resolve(base, caption["offsets_path"]),
            self.row_count,
        )

        for expected_index, shard in enumerate(self.manifest["t2i"]["shards"]):
            if int(shard["shard_index"]) != expected_index:
                raise ValueError(
                    f"non-contiguous T2I shard index in {self.manifest_path}: "
                    f"{shard['shard_index']} != {expected_index}"
                )
            self.t2i_readers.append(
                OffsetJsonlReader(
                    _resolve(base, shard["path"]),
                    _resolve(base, shard["offsets_path"]),
                    int(shard["records"]),
                )
            )
        if not self.t2i_readers:
            raise ValueError(f"synthetic text index has no T2I shards: {manifest_path}")

        mapping = self.manifest["mapping"]
        self.mapping_path = _resolve(base, mapping["path"])
        expected_mapping_bytes = self.row_count * MAPPING_STRUCT.size
        if self.mapping_path.stat().st_size != expected_mapping_bytes:
            raise ValueError(
                f"corrupt T2I mapping {self.mapping_path}: expected "
                f"{expected_mapping_bytes} bytes, got {self.mapping_path.stat().st_size}"
            )

    def clone(self) -> "ImageNetSyntheticTextIndex":
        return type(self)(self.manifest_path)

    def _open_mapping(self) -> int:
        if self._mapping_fd is None:
            self._mapping_fd = os.open(self.mapping_path, os.O_RDONLY)
        return self._mapping_fd

    def read_caption(self, row_index: int) -> dict[str, Any]:
        if self.caption_reader is None:
            raise RuntimeError("caption index reader is closed")
        return self.caption_reader.read(int(row_index))

    def read_t2i(self, row_index: int) -> dict[str, Any]:
        row_index = int(row_index)
        if row_index < 0 or row_index >= self.row_count:
            raise IndexError(row_index)
        encoded = _read_exact_at(
            self._open_mapping(),
            MAPPING_STRUCT.size,
            row_index * MAPPING_STRUCT.size,
        )
        shard_index, shard_row = MAPPING_STRUCT.unpack(encoded)
        if shard_index >= len(self.t2i_readers):
            raise ValueError(
                f"T2I mapping row {row_index} references shard {shard_index}, "
                f"but only {len(self.t2i_readers)} shards exist"
            )
        return self.t2i_readers[shard_index].read(shard_row)

    def close(self) -> None:
        if self.caption_reader is not None:
            self.caption_reader.close()
        for reader in self.t2i_readers:
            reader.close()
        if self._mapping_fd is not None:
            os.close(self._mapping_fd)
            self._mapping_fd = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_mapping_fd"] = None
        return state

    def __del__(self) -> None:
        self.close()
