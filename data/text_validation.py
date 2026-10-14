"""A fixed text probe and explicit, resumable training-row exclusions.

Source identities are shard paths and JSONL byte offsets. No corpus hashing or
network access is performed. A newly reserved subset is not retroactively held
out from checkpoints trained before exclusions were enabled.
"""
from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MANIFEST = "public/datasets/climbmix_validation_v1/manifest.json"
SCHEMA = "climbmix_validation_subset_v1"


def load_validation_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or not data.get("records"):
        raise ValueError(f"invalid ClimbMix validation manifest: {path}")
    records = data["records"]
    identities = [(r["shard"], r["byte_offset"]) for r in records]
    if len(set(identities)) != len(records):
        raise ValueError("duplicate ClimbMix validation records")
    shards = {}
    for source in data["shards"]:
        shard = (path.parent / source["path"]).resolve()
        stat = shard.stat()
        if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
            raise ValueError(f"ClimbMix validation source changed: {shard}")
        shards[source["id"]] = shard
    excluded = {str(p): set() for p in shards.values()}
    for row in records:
        offset = row["byte_offset"]
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("invalid validation byte offset")
        shard = shards[row["shard"]]
        if offset >= shard.stat().st_size:
            raise ValueError("validation byte offset exceeds source")
        excluded[str(shard)].add(offset)
    stat = path.stat()
    contract = {"manifest": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return data, shards, excluded, contract


def validation_documents(manifest_path):
    data, shards, _, contract = load_validation_manifest(manifest_path)
    documents = []
    handles = {}
    try:
        for row in data["records"]:
            name = row["shard"]
            handle = handles.setdefault(name, None)
            if handle is None:
                handle = handles[name] = shards[name].open("rb")
            handle.seek(row["byte_offset"])
            record = json.loads(handle.readline())
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("validation document has no nonempty text")
            documents.append({"id": f"{name}:{row['byte_offset']}", "text": text})
    finally:
        for handle in handles.values():
            if handle is not None:
                handle.close()
    return documents, contract


class FixedTextDataset:
    """Finite document probe with the source implementation's seeded windows."""

    def __init__(self, documents, tokenizer, *, sequence_length=2048,
                 max_document_chars=32768, seed=424242):
        import random
        self.rows = []
        self.sequence_length = int(sequence_length)
        self.eos_token_id = int(tokenizer.eos_token_id)
        if self.sequence_length < 2 or max_document_chars < 1:
            raise ValueError("invalid text validation dimensions")
        rng = random.Random(seed)
        for document in documents:
            text = document["text"]
            start = rng.randrange(max(1, len(text) - max_document_chars + 1))
            ids = list(tokenizer.encode(text[start:start + max_document_chars], add_special_tokens=False))
            ids.append(self.eos_token_id)
            if len(ids) > self.sequence_length:
                offset = rng.randrange(len(ids) - self.sequence_length + 1)
                ids = ids[offset:offset + self.sequence_length]
            if len(ids) < 2:
                raise ValueError(f"validation document has no scored targets: {document['id']}")
            self.rows.append((document["id"], ids))
        if not self.rows:
            raise ValueError("text validation dataset is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def collate(self, rows):
        import torch
        size, length = len(rows), self.sequence_length
        ids = torch.zeros(size, length, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        types = torch.full((size, length), 3, dtype=torch.uint8)
        sigma = torch.full_like(ids, length)
        segments = torch.full_like(ids, -1)
        for i, (_, tokens) in enumerate(rows):
            n = len(tokens)
            ids[i, :n] = torch.tensor(tokens)
            labels[i, 1:n] = ids[i, 1:n]
            types[i, :n] = torch.where(ids[i, :n] == self.eos_token_id, 2, 0).to(torch.uint8)
            sigma[i, :n] = torch.arange(n)
            segments[i, :n] = 0
        positions = torch.arange(length).view(1, 1, length).expand(2, size, length)
        return dict(input_ids=ids, labels=labels, token_types=types, sigma=sigma,
                    segment_ids=segments, position_ids=positions,
                    image_loss_mask=torch.zeros_like(ids, dtype=torch.bool),
                    image_span_table=torch.empty(0, 5, dtype=torch.long),
                    sample_ids=[row[0] for row in rows])


def build_text_validation(params, source, tokenizer, training_paths):
    """Return a finite loader only when an explicit fixed probe is configured."""
    from torch.utils.data import DataLoader
    options = dict(params.get("validation", {}))
    manifest = options.get("manifest", params.get("validation_exclusion_manifest"))
    jsonl = options.get("jsonl")
    if not manifest and not jsonl:
        return None
    if jsonl:
        path = Path(jsonl).resolve()
        if path in {Path(p).resolve() for p in training_paths}:
            raise ValueError("text validation JSONL is a training shard")
        documents = []
        with path.open() as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text validation requires nonempty text")
                documents.append(dict(id=str(index), text=text))
    else:
        documents, _ = validation_documents(manifest)
    dataset = FixedTextDataset(documents, tokenizer,
                              sequence_length=int(options.get("sequence_length", params.get("sequence_length", 2048))),
                              max_document_chars=int(options.get("max_document_chars", 32768)),
                              seed=int(options.get("seed", 424242)))
    return DataLoader(dataset, batch_size=int(options.get("batch_size", source.get("micro_batch_size", 4))),
                      shuffle=False, collate_fn=dataset.collate)
