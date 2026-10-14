"""Streaming ClimbMix batches with online tokenization and exact cursors.

The dataset deliberately yields complete microbatches (``batch_size=None`` at
the PyTorch DataLoader layer).  A yielded batch carries the state *after* that
batch.  The trainer commits only the state attached to a consumed batch, so a
prefetching worker can run ahead without making checkpoint recovery inexact.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info


CLIMBMIX_STREAM_SCHEMA = "climbmix_online_stream_v1"


def _split_text(text: str, max_chars: int) -> Iterable[str]:
    for start in range(0, len(text), max_chars):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            yield chunk


class _ClimbMixPacker:
    def __init__(
        self,
        *,
        shard_paths: Sequence[Path],
        tokenizer,
        eos_token_id: int,
        sequence_length: int,
        micro_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        tokenizer_batch_documents: int,
        max_document_chars: int,
        resume_state: dict[str, Any] | None,
        excluded_rows: dict[str, set[int]] | None = None,
        exclusion_contract: dict | None = None,
    ) -> None:
        self.shard_paths = tuple(Path(path) for path in shard_paths)
        self.tokenizer = tokenizer
        self.eos_token_id = int(eos_token_id)
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.tokenizer_batch_documents = int(tokenizer_batch_documents)
        self.max_document_chars = int(max_document_chars)
        self.excluded_rows = {str(path): (excluded_rows or {}).get(str(path.resolve()), set())
                              for path in self.shard_paths}
        self.exclusion_contract = exclusion_contract
        self.state = self._normalize_state(resume_state)
        self._handle = None
        self._handle_shard_position: int | None = None

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema": CLIMBMIX_STREAM_SCHEMA,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "pass_index": 0,
            "shard_position": 0,
            "byte_offset": None,
            "pending_segments": [],
            "batches_emitted": 0,
            "physical_tokens_emitted": 0,
            "supervised_tokens_emitted": 0,
            "documents_read": 0,
            "validation_exclusion": self.exclusion_contract,
        }

    def _normalize_state(
        self, state: dict[str, Any] | None
    ) -> dict[str, Any]:
        if state is None:
            return self._initial_state()
        state = copy.deepcopy(state)
        expected = {
            "schema": CLIMBMIX_STREAM_SCHEMA,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    "ClimbMix resume state differs from the stream contract: "
                    f"{key}={state.get(key)!r}, expected={value!r}"
                )
        if state.get("validation_exclusion") != self.exclusion_contract:
            raise ValueError(
                "ClimbMix validation exclusions differ from the saved stream; "
                "keep the original exclusion configuration when resuming. "
                "A new holdout applies only to a new training run."
            )
        shard_position = int(state.get("shard_position", -1))
        if not 0 <= shard_position < len(self.shard_paths):
            raise ValueError(
                f"invalid ClimbMix shard_position={shard_position}"
            )
        state["shard_position"] = shard_position
        state["pass_index"] = int(state.get("pass_index", 0))
        state["byte_offset"] = (
            None
            if state.get("byte_offset") is None
            else int(state["byte_offset"])
        )
        pending = state.get("pending_segments", [])
        if not isinstance(pending, list):
            raise ValueError("ClimbMix pending_segments must be a list")
        for segment in pending:
            if not isinstance(segment, dict) or not isinstance(
                segment.get("tokens"), list
            ):
                raise ValueError("invalid pending ClimbMix segment")
            segment["tokens"] = [int(token) for token in segment["tokens"]]
            segment["offset"] = int(segment.get("offset", 0))
            if not 0 <= segment["offset"] < len(segment["tokens"]):
                raise ValueError("invalid pending ClimbMix token offset")
        state["pending_segments"] = pending
        for key in (
            "batches_emitted",
            "physical_tokens_emitted",
            "supervised_tokens_emitted",
            "documents_read",
        ):
            state[key] = int(state.get(key, 0))
        return state

    def _shard_order(self) -> list[int]:
        order = list(range(len(self.shard_paths)))
        # A rank-specific order spreads a partial-corpus run over all shards,
        # while the byte-range ownership below keeps records disjoint.
        rng = random.Random(
            self.seed
            + 1_000_003 * int(self.state["pass_index"])
            + 97_409 * self.rank
        )
        rng.shuffle(order)
        return order

    def _current_shard(self) -> tuple[Path, int, int]:
        shard_index = self._shard_order()[int(self.state["shard_position"])]
        path = self.shard_paths[shard_index]
        size = path.stat().st_size
        start = size * self.rank // self.world_size
        end = size * (self.rank + 1) // self.world_size
        return path, start, end

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._handle_shard_position = None

    def _open_handle(self):
        shard_position = int(self.state["shard_position"])
        if (
            self._handle is not None
            and self._handle_shard_position == shard_position
        ):
            return self._handle
        self._close_handle()
        path, start, _ = self._current_shard()
        handle = path.open("rb")
        byte_offset = self.state.get("byte_offset")
        if byte_offset is None:
            handle.seek(start)
            if start > 0:
                # The preceding rank owns the line crossing the boundary.
                handle.readline()
            self.state["byte_offset"] = int(handle.tell())
        else:
            handle.seek(int(byte_offset))
        self._handle = handle
        self._handle_shard_position = shard_position
        return handle

    def _advance_shard(self) -> None:
        self._close_handle()
        next_position = int(self.state["shard_position"]) + 1
        if next_position >= len(self.shard_paths):
            self.state["pass_index"] = int(self.state["pass_index"]) + 1
            next_position = 0
        self.state["shard_position"] = next_position
        self.state["byte_offset"] = None

    def _read_text_chunks(self) -> list[str]:
        chunks: list[str] = []
        while len(chunks) < self.tokenizer_batch_documents:
            path, _, end = self._current_shard()
            handle = self._open_handle()
            line_start = int(handle.tell())
            if line_start >= end:
                self._advance_shard()
                continue
            raw_line = handle.readline()
            if not raw_line:
                self._advance_shard()
                continue
            self.state["byte_offset"] = int(handle.tell())
            if line_start in self.excluded_rows.get(str(path), ()):
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid ClimbMix JSON at {path}:{line_start}"
                ) from exc
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"ClimbMix row has no string text at {path}:{line_start}"
                )
            self.state["documents_read"] = int(
                self.state["documents_read"]
            ) + 1
            chunks.extend(_split_text(text, self.max_document_chars))
        return chunks

    def _batch_encode(self, texts: Sequence[str]) -> list[list[int]]:
        if callable(self.tokenizer):
            encoded = self.tokenizer(
                list(texts),
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            if isinstance(encoded, dict):
                input_ids = encoded["input_ids"]
            else:
                input_ids = encoded.input_ids
            return [[int(token) for token in row] for row in input_ids]
        return [
            [
                int(token)
                for token in self.tokenizer.encode(
                    text, add_special_tokens=False
                )
            ]
            for text in texts
        ]

    def _fill_pending_segments(self) -> None:
        texts = self._read_text_chunks()
        for tokens in self._batch_encode(texts):
            if not tokens:
                continue
            self.state["pending_segments"].append(
                {
                    "tokens": tokens + [self.eos_token_id],
                    "offset": 0,
                }
            )

    def _next_block(self) -> dict[str, torch.Tensor | int]:
        length = self.sequence_length
        input_ids = torch.zeros(length, dtype=torch.long)
        labels = torch.full((length,), -100, dtype=torch.long)
        token_types = torch.full((length,), 3, dtype=torch.uint8)
        sigma = torch.full((length,), length, dtype=torch.long)
        segment_ids = torch.full((length,), -1, dtype=torch.long)
        position_ids = torch.zeros(2, length, dtype=torch.long)
        cursor = 0
        segment_id = 0
        supervised = 0

        while cursor < length:
            if not self.state["pending_segments"]:
                self._fill_pending_segments()
            segment = self.state["pending_segments"][0]
            tokens = segment["tokens"]
            offset = int(segment["offset"])
            remaining = length - cursor
            continuation = offset > 0
            if remaining < 2 and (continuation or len(tokens) - offset > 1):
                break

            if continuation:
                take = min(remaining - 1, len(tokens) - offset)
                piece = [tokens[offset - 1], *tokens[offset : offset + take]]
                consumed = take
            else:
                take = min(remaining, len(tokens))
                piece = tokens[:take]
                consumed = take
            piece_length = len(piece)
            if piece_length <= 0:
                raise RuntimeError("ClimbMix packer made no progress")
            end = cursor + piece_length
            piece_tensor = torch.tensor(piece, dtype=torch.long)
            input_ids[cursor:end] = piece_tensor
            token_types[cursor:end] = torch.where(
                piece_tensor.eq(self.eos_token_id),
                torch.full_like(piece_tensor, 2, dtype=torch.uint8),
                torch.zeros_like(piece_tensor, dtype=torch.uint8),
            )
            labels[cursor:end] = piece_tensor
            labels[cursor] = -100
            sigma[cursor:end] = torch.arange(piece_length, dtype=torch.long)
            segment_ids[cursor:end] = segment_id
            local_positions = torch.arange(piece_length, dtype=torch.long)
            position_ids[:, cursor:end] = local_positions.unsqueeze(0)
            supervised += max(0, piece_length - 1)

            new_offset = offset + consumed
            if new_offset >= len(tokens):
                self.state["pending_segments"].pop(0)
            else:
                segment["offset"] = new_offset
            cursor = end
            segment_id += 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "token_types": token_types,
            "sigma": sigma,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
            "valid_tokens": cursor,
            "supervised_tokens": supervised,
        }

    def next_batch(self) -> dict[str, Any]:
        blocks = [self._next_block() for _ in range(self.micro_batch_size)]
        valid_tokens = sum(int(block["valid_tokens"]) for block in blocks)
        supervised_tokens = sum(
            int(block["supervised_tokens"]) for block in blocks
        )
        physical_tokens = self.micro_batch_size * self.sequence_length
        self.state["batches_emitted"] = int(self.state["batches_emitted"]) + 1
        self.state["physical_tokens_emitted"] = int(
            self.state["physical_tokens_emitted"]
        ) + physical_tokens
        self.state["supervised_tokens_emitted"] = int(
            self.state["supervised_tokens_emitted"]
        ) + supervised_tokens
        state = copy.deepcopy(self.state)
        return {
            "source_name": "climbmix",
            "input_ids": torch.stack([block["input_ids"] for block in blocks]),
            "labels": torch.stack([block["labels"] for block in blocks]),
            "token_types": torch.stack(
                [block["token_types"] for block in blocks]
            ),
            "sigma": torch.stack([block["sigma"] for block in blocks]),
            "segment_ids": torch.stack(
                [block["segment_ids"] for block in blocks]
            ),
            "position_ids": torch.stack(
                [block["position_ids"] for block in blocks], dim=1
            ),
            "image_loss_mask": torch.zeros(
                self.micro_batch_size,
                self.sequence_length,
                dtype=torch.bool,
            ),
            "image_span_table": torch.empty(0, 5, dtype=torch.long),
            "pack_stats": (
                valid_tokens,
                0,
                physical_tokens - valid_tokens,
                self.sequence_length,
            ),
            "supervised_text_tokens": supervised_tokens,
            "stream_state": state,
        }

    def close(self) -> None:
        self._close_handle()


class ClimbMixOnlineBatchDataset(IterableDataset):
    """Infinite, rank-sharded stream of fixed-shape pure-text microbatches."""

    def __init__(
        self,
        *,
        shard_paths: Sequence[str | Path],
        tokenizer_path: str | Path | None = None,
        tokenizer=None,
        eos_token_id: int,
        sequence_length: int = 2048,
        micro_batch_size: int = 4,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        tokenizer_batch_documents: int = 32,
        max_document_chars: int = 262_144,
        rayon_num_threads: int = 2,
        resume_state: dict[str, Any] | None = None,
        validation_exclusion_manifest: str | Path | None = None,
    ) -> None:
        super().__init__()
        paths = tuple(Path(path) for path in shard_paths)
        if not paths:
            raise ValueError("ClimbMix requires at least one JSONL shard")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing ClimbMix shards: {missing[:8]}")
        if (tokenizer_path is None) == (tokenizer is None):
            raise ValueError(
                "provide exactly one of tokenizer_path or tokenizer"
            )
        if int(sequence_length) < 2 or int(micro_batch_size) <= 0:
            raise ValueError("invalid ClimbMix sequence or microbatch size")
        if int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
            raise ValueError(
                f"invalid distributed rank/world_size={rank}/{world_size}"
            )
        if int(tokenizer_batch_documents) <= 0:
            raise ValueError("tokenizer_batch_documents must be positive")
        if int(max_document_chars) <= 0:
            raise ValueError("max_document_chars must be positive")
        if int(rayon_num_threads) <= 0:
            raise ValueError("rayon_num_threads must be positive")
        self.shard_paths = paths
        self.tokenizer_path = (
            str(tokenizer_path) if tokenizer_path is not None else None
        )
        self.tokenizer = tokenizer
        self.eos_token_id = int(eos_token_id)
        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.tokenizer_batch_documents = int(tokenizer_batch_documents)
        self.max_document_chars = int(max_document_chars)
        self.rayon_num_threads = int(rayon_num_threads)
        self.resume_state = copy.deepcopy(resume_state)
        self.excluded_rows, self.exclusion_contract = {}, None
        if validation_exclusion_manifest:
            from .text_validation import load_validation_manifest
            _, _, self.excluded_rows, self.exclusion_contract = load_validation_manifest(validation_exclusion_manifest)
            if not set(self.excluded_rows).issubset({str(p.resolve()) for p in paths}):
                raise ValueError("validation exclusions refer to shards outside the training stream")

    def set_resume_state(self, state: dict[str, Any] | None) -> None:
        self.resume_state = copy.deepcopy(state)

    def _load_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            fix_mistral_regex=True,
            local_files_only=True,
        )
        if int(tokenizer.eos_token_id) != self.eos_token_id:
            raise ValueError(
                "ClimbMix tokenizer EOS differs from the frozen model tokenizer: "
                f"{tokenizer.eos_token_id} != {self.eos_token_id}"
            )
        return tokenizer

    def __iter__(self):
        worker = get_worker_info()
        if worker is not None and worker.num_workers != 1:
            raise RuntimeError(
                "exact ClimbMix recovery currently requires exactly one "
                "DataLoader worker per distributed rank"
            )
        os.environ["RAYON_NUM_THREADS"] = str(self.rayon_num_threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        packer = _ClimbMixPacker(
            shard_paths=self.shard_paths,
            tokenizer=self._load_tokenizer(),
            eos_token_id=self.eos_token_id,
            sequence_length=self.sequence_length,
            micro_batch_size=self.micro_batch_size,
            rank=self.rank,
            world_size=self.world_size,
            seed=self.seed,
            tokenizer_batch_documents=self.tokenizer_batch_documents,
            max_document_chars=self.max_document_chars,
            resume_state=self.resume_state,
            excluded_rows=self.excluded_rows,
            exclusion_contract=self.exclusion_contract,
        )
        try:
            while True:
                yield packer.next_batch()
        finally:
            packer.close()


__all__ = [
    "CLIMBMIX_STREAM_SCHEMA",
    "ClimbMixOnlineBatchDataset",
]
