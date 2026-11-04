"""Direct, immutable-revision ModelScope download; publishes READY only after SHA checks."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
PROCESSOR_FILES = ("config.json", "generation_config.json", "preprocessor_config.json",
                   "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                   "merges.txt", "vocab.json", "chat_template.jinja")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def verify_tensors(output):
    config = json.loads((output / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]:
        raise ValueError("unexpected_model_architecture")
    description = json.loads((output / "quant_model_description.json").read_text())
    types = Counter(value for value in description.values() if isinstance(value, str))
    if not types["W8A8_DYNAMIC"] or any("FP8" in value for value in types):
        raise ValueError("not_ascend_w8a8_dynamic")
    index = json.loads((output / "quant_model_weights.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    seen = {}
    dtype_bytes = Counter()
    dtype_counts = Counter()
    weight_int8 = 0
    for name in sorted(set(mapping.values())):
        if Path(name).name != name:
            raise ValueError("unsafe_shard_name")
        path = output / name
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("missing_safetensor_header")
            length = struct.unpack("<Q", prefix)[0]
            if not 1 <= length <= 64 << 20:
                raise ValueError("invalid_safetensor_header_size")
            tensors = json.loads(stream.read(length))
        offsets = []
        for key, tensor in tensors.items():
            if key == "__metadata__":
                continue
            if key in seen or mapping.get(key) != name:
                raise ValueError("tensor_index_mismatch")
            seen[key] = name
            start, stop = tensor["data_offsets"]
            if not 0 <= start <= stop <= path.stat().st_size - 8 - length:
                raise ValueError("invalid_tensor_offsets")
            offsets.append((start, stop))
            dtype_bytes[tensor["dtype"]] += stop - start
            dtype_counts[tensor["dtype"]] += 1
            if key.endswith(".weight") and tensor["dtype"] == "I8":
                weight_int8 += 1
                if description.get(key) != "W8A8_DYNAMIC":
                    raise ValueError("int8_weight_without_w8a8_declaration")
        position = 0
        for start, stop in sorted(offsets):
            if start != position:
                raise ValueError("noncontiguous_tensor_offsets")
            position = stop
        if position + 8 + length != path.stat().st_size:
            raise ValueError("unexpected_shard_trailing_bytes")
    if seen != mapping or not weight_int8:
        raise ValueError("missing_tensors_or_no_int8_weights")
    if any("F8" in dtype for dtype in dtype_counts):
        raise ValueError("fp8_is_not_ascend_w8a8")
    if sum(dtype_bytes.values()) != index["metadata"]["total_size"]:
        raise ValueError("index_total_tensor_size_mismatch")
    return {"architecture": config["architectures"][0], "quant_description_types": dict(types),
            "tensor_count": len(seen), "int8_weight_tensor_count": weight_int8,
            "tensor_bytes_by_dtype": dict(dtype_bytes), "tensor_count_by_dtype": dict(dtype_counts),
            "all_index_tensors_present": True, "all_shard_offsets_valid": True,
            "tensor_bytes": sum(dtype_bytes.values())}


def main(args):
    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    revision = manifest["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("immutable_commit_required")
    output = Path(manifest["destination"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    states = ROOT / "download_state"
    states.mkdir(exist_ok=True)
    all_files = manifest["files"]
    pending = []
    handles = {}
    locks = {}
    state = {}
    chunk_size = args.chunk_mib << 20
    for item in all_files:
        name, size, checksum = item["name"], item["size_bytes"], item["sha256"]
        if Path(name).name != name or item["revision"] != revision:
            raise ValueError("unsafe_file_or_revision_mismatch")
        final = output / name
        if final.exists():
            if final.stat().st_size != size or sha(final) != checksum:
                raise ValueError("existing_final_file_invalid:" + name)
            continue
        partial = output / (name + ".part")
        state_path = states / (name + ".json")
        identity = {"file": item, "chunk_size": chunk_size, "revision": revision}
        current = {"identity": identity, "completed_chunks": []}
        if state_path.exists():
            current = json.loads(state_path.read_text())
            if current["identity"] != identity or not partial.exists() or partial.stat().st_size != size:
                raise ValueError("resume_identity_mismatch")
        fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(fd, size)
        handles[name] = fd
        locks[name] = threading.Lock()
        state[name] = current
        done = set(current["completed_chunks"])
        for number, begin in enumerate(range(0, size, chunk_size)):
            if number not in done:
                pending.append((item, number, begin, min(size, begin + chunk_size) - 1))
    # Interleave source files instead of scheduling all ranges of one shard first.
    pending.sort(key=lambda task: (task[1], task[0]["name"]))
    progress_lock = threading.Lock()
    downloaded = [0]
    local = threading.local()

    def fetch(task):
        item, number, begin, end = task
        name, size = item["name"], item["size_bytes"]
        url = ("https://www.modelscope.cn/models/" + manifest["repo_id"] + "/resolve/"
               + revision + "/" + quote(name))
        if not hasattr(local, "client"):
            local.client = requests.Session()
            local.client.trust_env = False
        for attempt in range(5):
            try:
                with local.client.get(url, headers={"Range": f"bytes={begin}-{end}"},
                                      stream=True, timeout=(20, 90)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                        raise ValueError("exact_range_response_required")
                    position = begin
                    for data in response.iter_content(1 << 20):
                        if position + len(data) > end + 1:
                            raise ValueError("range_overflow")
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(handles[name], view, position)
                            if written <= 0:
                                raise OSError("short_pwrite")
                            position += written
                            view = view[written:]
                    if position != end + 1:
                        raise ValueError("range_underflow")
                with locks[name]:
                    os.fsync(handles[name])
                    state[name]["completed_chunks"].append(number)
                    atomic(states / (name + ".json"), state[name])
                with progress_lock:
                    downloaded[0] += end + 1 - begin
                    if len(state[name]["completed_chunks"]) % 8 == 0:
                        print(json.dumps({"event": "download_progress", "new_bytes": downloaded[0],
                                          "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(8, 2 ** attempt))
    print(json.dumps({"event": "download_start", "pending_chunks": len(pending), "workers": args.workers,
                      "revision": revision, "destination": str(output)}), flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(fetch, task) for task in pending]):
                future.result()
    finally:
        for fd in handles.values():
            os.close(fd)
    verified = []
    for item in all_files:
        path = output / item["name"]
        if not path.exists():
            path = output / (item["name"] + ".part")
        actual = sha(path)
        if path.stat().st_size != item["size_bytes"] or actual != item["sha256"]:
            raise ValueError("full_file_hash_mismatch:" + item["name"])
        final = output / item["name"]
        if path != final:
            os.replace(path, final)
        verified.append({**item, "verified": True})
        print(json.dumps({"event": "verified", "file": item["name"], "size_bytes": item["size_bytes"]}), flush=True)
    tensor_report = verify_tensors(output)
    processor = {name: sha(output / name) for name in PROCESSOR_FILES}
    processor_revision = "sha256:" + hashlib.sha256(json.dumps(processor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {"schema": "corpus.qwen36_w8a8.weights_ready.v1", "complete": True,
               "model_path": str(output), "model_name": "Qwen3.6-35B-A3B-W8A8",
               "repo_id": manifest["repo_id"], "revision": revision, "provider": "modelscope",
               "precision": "W8A8_DYNAMIC", "quantization": "ascend", "files": verified,
               "processor_files": processor, "processor_revision": processor_revision,
               "tensor_validation": tensor_report, "total_bytes": sum(item["size_bytes"] for item in all_files),
               "source_manifest_sha256": sha(manifest_path), "download_code_sha256": sha(__file__),
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds_this_invocation": time.monotonic() - started,
               "weights_not_inference_qualified": True, "direct_https": True}
    atomic(ROOT / "WEIGHTS_READY.json", receipt)
    print(json.dumps({"event": "weights_ready", "receipt": str(ROOT / "WEIGHTS_READY.json"),
                      "tensor_validation": tensor_report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24 or not 4 <= args.chunk_mib <= 256:
        parser.error("invalid bounded concurrency/chunk size")
    with (ROOT / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main(args)
        except Exception as error:
            print(json.dumps({"event": "download_failed", "error_type": type(error).__name__}), flush=True)
            raise SystemExit(1) from None
