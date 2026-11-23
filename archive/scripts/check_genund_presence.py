#!/usr/bin/env python3
"""Bounded image/text presence audit; no image decoding, extraction or downloads.

Each immediate child of ROOT is a dataset (or select --dataset relative/path).
Loose images and TXT sidecars are checked by file existence. JSON/CSV metadata,
archive headers, and a few Parquet rows are sampled to distinguish image bytes
from URLs. YES means observed, NO requires an exhausted loose-file search, and
UNKNOWN means sampling, unresolved references, errors or unsupported containers
prevent an absence claim. BOTH_PRESENT does not verify image/text pairing.
"""
from __future__ import annotations

import argparse
import bz2
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
import csv
import gzip
from datetime import datetime, timezone
import io
import json
import lzma
import os
from pathlib import Path
import random
import re
import sys
import tarfile
import zipfile

DEFAULT_ROOT = "/inspire/dataset/gen-und-mm-pretrain/01"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".heic"}
TEXT_SUFFIXES = {".txt", ".caption", ".text"}
METADATA_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv"}
IGNORED_NAMES = {"dataset_info.json", "dataset_infos.json", "state.json", "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "cmakelists.txt", "_source.json"}
IGNORED_PREFIXES = ("readme", "license", "licence", "copying", "notice", "requirements", "citation")
IGNORED_DIRS = {".git", ".cache", "__pycache__", ".ipynb_checkpoints"}
TEXT_KEYS = {"text", "texts", "caption", "captions", "recaption", "prompt", "description", "question", "answer", "content", "conversations", "conversation", "messages", "dialog", "dialogue", "turns"}
IMAGE_KEYS = {"image", "images", "img", "jpg", "jpeg", "png", "webp", "imagebytes", "imgbytes", "imagebase64", "imagepath", "imagefile", "imagefilename", "imageurl", "imageurls", "url"}
DEFAULTS = dict(max_entries=20000, max_dir_entries=2000, max_dirs=128,
                max_depth=8, sample_shards=3, max_members=2000,
                sample_rows=8, max_probe_bytes=1024 * 1024,
                max_archive_bytes=64 * 1024 * 1024, seed=0)


class BudgetReader:
    """Bound decompression when a compressed tar must skip large members."""
    def __init__(self, stream, limit):
        self.stream, self.remaining = stream, limit

    def read(self, size=-1):
        if self.remaining <= 0:
            raise OSError("decompressed archive byte limit reached")
        data = self.stream.read(self.remaining if size < 0 else min(size, self.remaining))
        self.remaining -= len(data)
        return data


def canonical(key):
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def text_key(key):
    key = canonical(key)
    return key in TEXT_KEYS or ("caption" in key and not any(x in key for x in ("path", "file", "url", "length", "count")))


def ignored(name):
    name = Path(name).name.lower()
    return name in IGNORED_NAMES or name.startswith(IGNORED_PREFIXES)


def file_kind(path):
    name = str(path).lower()
    for suffix in (".tar.zst", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar", ".zip", ".parquet"):
        if name.endswith(suffix):
            return suffix
    if Path(name).suffix in {".arrow", ".feather", ".7z", ".rar", ".h5", ".hdf5", ".lmdb", ".mdb", ".tfrecord", ".tfrecords", ".gz", ".bz2", ".xz", ".zst"}:
        return "unsupported"
    return None


def json_prefix_records(text, limit):
    """Decode complete objects in a bounded prefix, including large JSON arrays.

    Never regex-match field values as evidence: only successfully parsed JSON
    objects are inspected. An incomplete outer wrapper may contain a complete
    first record. Scan at most 512 candidate starts to bound malformed input.
    """
    decoder = json.JSONDecoder()
    position = yielded = attempts = 0
    while yielded < limit and attempts < 512:
        start = text.find("{", position)
        if start < 0:
            break
        attempts += 1
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            position = start + 1
            continue
        yield value
        yielded += 1
        position = start + length


class Probe:
    def __init__(self, root, corpus_root, options):
        self.root, self.corpus_root, self.options = root, corpus_root, options
        self.examples = {"image": None, "annotation": None, "container": None, "image_reference": None}
        self.notes = set()
        self.errors = []
        self.samples = []
        self.hits = Counter()

    def found(self, kind, source):
        self.hits[kind] += 1
        if self.examples[kind] is None:
            self.examples[kind] = str(source)

    def error(self, source, exc):
        self.notes.add("inspection_error")
        if len(self.errors) < 10:
            self.errors.append(f"{source}: {type(exc).__name__}: {exc}")

    def reference(self, value, source, base):
        self.found("image_reference", source)
        if value.startswith(("http://", "https://", "s3://", "gs://", "oss://")):
            self.notes.add("external_image_reference_observed")
            return
        if value.startswith("data:image/") and "," in value and value.split(",", 1)[1]:
            self.found("image", source + " (inline image data; not decoded)")
            return
        if len(value) > 4096 or "\n" in value:
            self.notes.add("unresolved_image_reference")
            return
        target = Path(value.removeprefix("file://"))
        bases = [base, self.root, self.corpus_root]
        if target.suffix.lower() in IMAGE_SUFFIXES:
            for candidate in ([target] if target.is_absolute() else [b / target for b in bases]):
                try:
                    if candidate.is_file():
                        self.found("image", f"{source} -> {candidate} (path exists)")
                        return
                except OSError as exc:
                    self.error(candidate, exc)
        self.notes.add("unresolved_image_reference")

    def value(self, value, source, base, image_context=False, text_context=False, depth=0):
        if depth > 12:
            self.notes.add("record_depth_limit")
            return
        if isinstance(value, dict):
            for key, child in value.items():
                name = canonical(key)
                child_image = name in IMAGE_KEYS or (image_context and name in {"bytes", "path", "data", "base64"})
                child_text = text_key(key) or (text_context and name in {"value", "utterance"})
                self.value(child, source + "." + str(key), base, child_image, child_text, depth + 1)
        elif isinstance(value, (list, tuple)):
            if len(value) > self.options["sample_rows"]:
                self.notes.add("record_list_sampled")
            for index, child in enumerate(value[:self.options["sample_rows"]]):
                self.value(child, f"{source}[{index}]", base, image_context, text_context, depth + 1)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            if image_context and len(value):
                self.found("image", source + " (nonempty image bytes; not decoded)")
        elif isinstance(value, str) and value.strip():
            if image_context:
                self.reference(value.strip(), source, base)
            if text_context and not value.strip().startswith(("http://", "https://", "s3://", "gs://")):
                self.found("annotation", source + " (text field)")

    def metadata(self, data, suffix, source, base):
        text = data.decode("utf-8-sig", errors="replace")
        if suffix in {".csv", ".tsv"}:
            reader = csv.DictReader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ",")
            records = []
            for index, row in enumerate(reader):
                if index >= self.options["sample_rows"]:
                    break
                records.append(row)
        else:
            records = list(json_prefix_records(text, self.options["sample_rows"]))
        if not records and text.strip():
            self.notes.add("metadata_prefix_unparsed")
        for index, row in enumerate(records):
            self.value(row, f"{source}:record{index}", base)

    def metadata_file(self, path):
        try:
            with path.open("rb") as stream:
                data = stream.read(self.options["max_probe_bytes"] + 1)
            if len(data) > self.options["max_probe_bytes"]:
                self.notes.add("metadata_byte_limit")
            self.notes.add("metadata_records_sampled")
            self.metadata(data[:self.options["max_probe_bytes"]], path.suffix.lower(), str(path), path.parent)
        except (OSError, ValueError, csv.Error) as exc:
            self.error(path, exc)

    def archive(self, path):
        before = self.hits.copy()
        info = {"path": str(path), "kind": file_kind(path), "members_checked": 0}
        self.notes.add("containers_sampled")
        try:
            with ExitStack() as stack:
                if info["kind"] == ".zip":
                    archive = stack.enter_context(zipfile.ZipFile(path))
                    members = iter(archive.infolist())
                    fields = lambda member: (member.filename, not member.is_dir())
                    read = archive.open
                else:
                    if info["kind"] == ".tar.zst":
                        import zstandard
                        raw = stack.enter_context(path.open("rb"))
                        stream = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(raw))
                        archive = stack.enter_context(tarfile.open(fileobj=BudgetReader(stream, self.options["max_archive_bytes"]), mode="r|"))
                    elif info["kind"] != ".tar":
                        opener = (gzip.open if info["kind"] in {".tar.gz", ".tgz"}
                                  else bz2.open if info["kind"] in {".tar.bz2", ".tbz2"}
                                  else lzma.open)
                        stream = stack.enter_context(opener(path, "rb"))
                        archive = stack.enter_context(tarfile.open(fileobj=BudgetReader(stream, self.options["max_archive_bytes"]), mode="r|"))
                    else:
                        archive = stack.enter_context(tarfile.open(path, mode="r:"))
                    members = iter(archive)
                    fields = lambda member: (member.name, member.isfile())
                    read = archive.extractfile
                image_seen = text_seen = False
                metadata_count = 0
                for member in members:
                    if info["members_checked"] >= self.options["max_members"]:
                        self.notes.add("archive_member_limit")
                        break
                    info["members_checked"] += 1
                    name, regular = fields(member)
                    if not regular or ignored(name):
                        continue
                    source = f"{path}::{name}"
                    suffix = Path(name).suffix.lower()
                    if suffix in IMAGE_SUFFIXES:
                        image_seen = True
                        self.found("image", source + " (archive member exists)")
                    elif suffix in TEXT_SUFFIXES:
                        text_seen = True
                        self.found("annotation", source + " (text member exists)")
                    elif suffix in METADATA_SUFFIXES and metadata_count < self.options["sample_rows"]:
                        metadata_count += 1
                        with read(member) as stream:
                            data = stream.read(self.options["max_probe_bytes"])
                        self.metadata(data, suffix, source, path.parent)
                        image_seen = image_seen or self.hits["image"] > before["image"]
                        text_seen = text_seen or self.hits["annotation"] > before["annotation"]
                    elif file_kind(name):
                        self.notes.add("nested_container_not_inspected")
                    if image_seen and text_seen:
                        break
                info["image_member_observed"] = image_seen
                info["text_member_observed"] = text_seen
        except ImportError as exc:
            self.notes.add("missing_optional_dependency")
            self.error(path, exc)
        except (OSError, ValueError, EOFError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
            self.error(path, exc)
        self.samples.append(info)

    def parquet(self, path):
        info = {"path": str(path), "kind": ".parquet", "rows_checked": 0}
        self.notes.add("containers_sampled")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            def relevant(field):
                if canonical(field.name) in IMAGE_KEYS or text_key(field.name):
                    return True
                dtype = field.type
                if pa.types.is_struct(dtype):
                    return any(relevant(child) for child in dtype)
                if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
                    return relevant(dtype.value_field)
                return False

            with pq.ParquetFile(path) as parquet:
                columns = [field.name for field in parquet.schema_arrow if relevant(field)]
                info["columns"] = parquet.schema_arrow.names
                info["sampled_columns"] = columns
                if not columns:
                    self.notes.add("unrecognized_parquet_schema")
                elif parquet.num_row_groups:
                    batches = parquet.iter_batches(batch_size=self.options["sample_rows"], row_groups=[0], columns=columns, use_threads=False)
                    batch = next(batches, None)
                    if batch is not None:
                        for index, row in enumerate(batch.to_pylist()):
                            self.value(row, f"{path}:row{index}", path.parent)
                        info["rows_checked"] = batch.num_rows
        except ImportError as exc:
            self.notes.add("missing_optional_dependency")
            self.error(path, exc)
        except Exception as exc:
            self.error(path, exc)
        self.samples.append(info)


def inspect_dataset(root, max_entries=20000, *, corpus_root=None, **kwargs):
    options = {**DEFAULTS, **kwargs, "max_entries": max_entries}
    root = Path(root).absolute()
    probe = Probe(root, Path(corpus_root or root.parent).absolute(), options)
    rng = random.Random(options["seed"])
    pending = deque([(root, 0)])
    candidates = defaultdict(list)
    visited = directories = 0
    stop = False
    while pending and not stop:
        if directories >= options["max_dirs"]:
            probe.notes.add("directory_limit")
            break
        directory, depth = pending.popleft()
        directories += 1
        subdirs = []
        sidecar_probes = 0
        try:
            with os.scandir(directory) as entries:
                for offset, entry in enumerate(entries):
                    if options["max_dir_entries"] and offset >= options["max_dir_entries"]:
                        probe.notes.add("directory_entry_limit")
                        break
                    if options["max_entries"] and visited >= options["max_entries"]:
                        probe.notes.add("entry_limit")
                        stop = True
                        break
                    visited += 1
                    path = Path(entry.path)
                    try:
                        if entry.is_symlink():
                            probe.notes.add("symlink_not_followed")
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in IGNORED_DIRS:
                                if depth >= options["max_depth"]:
                                    probe.notes.add("depth_limit")
                                else:
                                    subdirs.append(path)
                            continue
                        if not entry.is_file(follow_symlinks=False) or ignored(entry.name):
                            continue
                        suffix = path.suffix.lower()
                        kind = file_kind(path)
                        if suffix in IMAGE_SUFFIXES:
                            probe.found("image", path)
                            if sidecar_probes < 8 and not probe.examples["annotation"]:
                                sidecar_probes += 1
                                for extension in (".txt", ".caption", ".json", ".jsonl"):
                                    sidecar = path.with_suffix(extension)
                                    if sidecar.is_file() and not ignored(sidecar.name):
                                        if extension in TEXT_SUFFIXES:
                                            probe.found("annotation", sidecar)
                                        else:
                                            candidates[extension].append(sidecar)
                        elif suffix in TEXT_SUFFIXES:
                            probe.found("annotation", path)
                        elif suffix in METADATA_SUFFIXES:
                            candidates[suffix].append(path)
                        elif kind:
                            probe.found("container", path)
                            if kind == "unsupported":
                                probe.notes.add("unsupported_container")
                            else:
                                candidates[kind].append(path)
                    except OSError as exc:
                        probe.error(path, exc)
                    if probe.examples["image"] and probe.examples["annotation"]:
                        stop = True
                        break
        except OSError as exc:
            probe.error(directory, exc)
        rng.shuffle(subdirs)
        subdirs.sort(key=lambda path: 0 if any(token in path.name.lower() for token in ("annotation", "caption", "metadata")) else 1)
        pending.extend((path, depth + 1) for path in subdirs)

    for kind, paths in sorted(candidates.items()):
        paths = sorted(set(paths))
        selected = rng.sample(paths, min(options["sample_shards"], len(paths)))
        if len(selected) < len(paths):
            probe.notes.add("candidate_files_sampled")
        for path in selected:
            if kind in METADATA_SUFFIXES:
                probe.metadata_file(path)
            elif kind == ".parquet":
                probe.parquet(path)
            else:
                probe.archive(path)

    def status(kind):
        return "YES" if probe.examples[kind] else ("UNKNOWN" if probe.notes else "NO")

    image, annotation = status("image"), status("annotation")
    verdict = "BOTH_PRESENT" if image == annotation == "YES" else ("UNCONFIRMED" if "UNKNOWN" in (image, annotation) else "MISSING_TYPE")
    return {"dataset": root.name, "path": str(root), "image": image,
            "annotation": annotation, "verdict": verdict,
            "examples": probe.examples, "notes": sorted(probe.notes),
            "errors": probe.errors, "container_samples": probe.samples,
            "entries_checked": visited, "directories_checked": directories}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    parser.add_argument("--dataset", action="append", help="Only inspect this relative directory; repeat for multiple datasets ('.' inspects ROOT itself)")
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default,
                            help=f"{name.replace('_', ' ')} (default: {default})")
    parser.add_argument("--output", type=Path, help="Write JSON and same-stem CSV outside the dataset root (must not already exist)")
    args = parser.parse_args()
    options = {name: getattr(args, name) for name in DEFAULTS}
    for name, value in options.items():
        if value < 0 or (name not in {"seed", "max_entries", "max_dir_entries", "max_depth"} and value == 0):
            parser.error(f"--{name.replace('_', '-')} has an invalid value")
    root = Path(args.root).resolve()
    try:
        if not root.is_dir():
            parser.error(f"Dataset root is not mounted or does not exist: {root}")
        if args.dataset:
            datasets = []
            for relative in args.dataset:
                path = (root / relative).resolve()
                if path != root and root not in path.parents:
                    parser.error("--dataset must stay inside ROOT")
                if not path.is_dir():
                    parser.error(f"Dataset directory does not exist: {path}")
                datasets.append(path)
            datasets = sorted(set(datasets))
        else:
            with os.scandir(root) as entries:
                datasets = sorted(Path(e.path) for e in entries if not e.is_symlink() and e.is_dir() and e.name not in IGNORED_DIRS)
        if not datasets:
            parser.error(f"No dataset directories found in {root}; use --dataset . to inspect ROOT itself")
        if args.output:
            output = args.output.resolve()
            csv_output = output.with_suffix(".csv")
            if output == csv_output:
                parser.error("--output must be a JSON path, not a .csv path")
            for path in (output, csv_output):
                if path == root or root in path.parents:
                    parser.error("Reports must be outside the dataset root")
                if path.exists():
                    parser.error(f"Report already exists: {path}")
    except OSError as exc:
        parser.error(str(exc))

    results = []
    print("dataset\timage\tannotation\tverdict", flush=True)
    for index, path in enumerate(datasets, 1):
        name = str(path.relative_to(root)) if path != root else root.name
        print(f"[{index}/{len(datasets)}] {name}", file=sys.stderr, flush=True)
        result = inspect_dataset(path, corpus_root=root, **options)
        result["dataset"] = name
        results.append(result)
        print(f"{name}\t{result['image']}\t{result['annotation']}\t{result['verdict']}", flush=True)
        for error in result["errors"]:
            print("  " + error, file=sys.stderr, flush=True)
    counts = Counter(r["verdict"] for r in results)
    summary = dict(datasets=len(results), both_present=counts["BOTH_PRESENT"],
                   missing_type=counts["MISSING_TYPE"], unconfirmed=counts["UNCONFIRMED"])
    print("Summary: " + json.dumps(summary), flush=True)
    if args.output:
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {"root": str(root), "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "Sampled presence only; no image decoding, URL fetching, full completeness or pairing verification. UNKNOWN is not absence. Directory and container sampling is bounded, not uniform over the full corpus.",
                  "options": options, "summary": summary, "results": results}
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        with csv_output.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset", "image", "annotation", "verdict", "image_example", "annotation_example", "notes"])
            for result in results:
                writer.writerow([result["dataset"], result["image"], result["annotation"], result["verdict"], result["examples"]["image"], result["examples"]["annotation"], ";".join(result["notes"])])
        print(f"JSON: {output}\nCSV: {csv_output}")


if __name__ == "__main__":
    main()
