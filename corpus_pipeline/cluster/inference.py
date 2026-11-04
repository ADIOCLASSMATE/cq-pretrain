"""Resident CF384, UniFD and SSCD workers for bounded CPU / Ascend batches.

No historical detector scores are loaded. Models are loaded exactly once per
worker. This module produces inference evidence, never an admission decision.
NPU and reduced precision executions have separate identities and require their
own numerical qualification before sharing a production feature catalog.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import io
from itertools import islice
import json
import os
from pathlib import Path
import platform
import struct
import time

import numpy as np
from PIL import Image, ImageOps

from corpus_pipeline.production.common import atomic_json, canonical, digest, file_sha


WORKER_VERSION = "resident_image_inference.v1"
MODEL_NAMES = ("cf384", "unifd", "sscd")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "models/ai_detectors"
DEFAULT_SSCD = ROOT / "models/sscd/sscd_disc_mixup.torchscript.pt"
MAX_BYTES = 128 << 20
MAX_PIXELS = 100_000_000


def _configure_npu_precision(torch):
    # torch_npu/npu/npu_config.py defaults convolution HF32 to enabled. The
    # detector operating point came from FP32 CPU inference, so do not inherit
    # backend precision-reduction defaults when requesting an NPU baseline.
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    precision = {"acl_precision_mode_requested": "must_keep_origin_dtype",
                 "conv_allow_hf32": torch.npu.conv.allow_hf32,
                 "matmul_allow_hf32": torch.npu.matmul.allow_hf32}
    if precision["conv_allow_hf32"] is not False or precision["matmul_allow_hf32"] is not False:
        raise RuntimeError("NPU refused strict HF32-disabled precision settings")
    return precision


def _configure_cf_attention(model, device):
    """Use timm's explicit attention math for the strict NPU FP32 baseline."""
    attention = [module for module in model.modules() if hasattr(module, "fused_attn")]
    disabled = str(device).startswith("npu")
    if disabled:
        for module in attention:
            module.fused_attn = False
    return {"fused_attn_disabled_for_npu": disabled, "attention_module_count": len(attention),
            "fused_attn_enabled_modules": sum(bool(module.fused_attn) for module in attention),
            "reason": "explicit_fp32_attention_for_npu_qualification" if disabled else "cpu_loader_default_unchanged"}


def _configure_cf_gelu(model, device, torch):
    """Avoid a backend's GELU approximation in the NPU qualification path."""
    class ExactErfGELU(torch.nn.Module):
        def forward(self, value):
            x = value.float()
            return (0.5 * x * (1.0 + torch.erf(x / (2.0 ** 0.5)))).to(value.dtype)

    count = 0
    if str(device).startswith("npu"):
        for parent in model.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.GELU) and child.approximate == "none":
                    setattr(parent, name, ExactErfGELU())
                    count += 1
    return {"gelu_erf_replacements": count,
            "gelu_execution": "explicit_fp32_erf" if str(device).startswith("npu") else "cpu_loader_default_unchanged"}


def _runtime(device, dtype, threads):
    # CPU hosts can have a torch_npu wheel without a CANN runtime. NPU requests
    # explicitly import the extension; they never silently run on another device.
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16 or bfloat16")
    if device == "cpu":
        selected = torch.device("cpu")
        precision = {"torch_float32_matmul_precision": torch.get_float32_matmul_precision()}
    elif device == "npu" or device.startswith("npu:"):
        import torch_npu  # noqa: F401 -- registers torch.npu explicitly
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable; check CANN and device assignment")
        selected = torch.device(device)
        torch.npu.set_device(selected)
        precision = _configure_npu_precision(torch)
    else:
        raise ValueError("explicit cpu or npu[:index] device required")
    torch.set_num_threads(threads)
    return torch, selected, getattr(torch, dtype), precision


def _load_bundles(models_dir, sscd_path, names):
    """Use the existing pinned loaders without changing their frozen code."""
    from corpus_pipeline.detector_calibration.infer import load_model
    from corpus_pipeline.dedup.sscd import MODEL_SHA256, PREPROCESS
    import torch
    bundles = {}
    for name in names:
        if name != "sscd":
            model, transform, identity = load_model(name, models_dir)
        else:
            if file_sha(sscd_path) != MODEL_SHA256:
                raise ValueError("SSCD checkpoint SHA256 mismatch")
            model = torch.jit.load(str(sscd_path), map_location="cpu").eval()
            transform = None
            identity = {"model": "sscd_disc_mixup", "checkpoint_sha256": MODEL_SHA256,
                        "preprocessing": PREPROCESS, "dimensions": 512,
                        "implementation": "original_pinned_torchscript"}
        bundles[name] = {"model": model, "transform": transform, "identity": identity}
    return bundles


def _sscd_input(opened):
    """Exactly the existing dedup.load_image_asset preprocessing and pixel ID."""
    rgba = ImageOps.exif_transpose(opened).convert("RGBA")
    rgba.load()
    pixel_sha = hashlib.sha256(b"RGBA" + struct.pack(">II", *rgba.size) + rgba.tobytes()).hexdigest()
    image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    tensor = np.asarray(image.resize((320, 320), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tensor = ((tensor - np.array([.485, .456, .406], np.float32)) /
              np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1).copy()
    return tensor, {"pixel_sha256": pixel_sha, "oriented_width": image.width, "oriented_height": image.height}


def _read_blob(asset):
    if callable(asset.get("blob_loader")):
        blob = asset["blob_loader"]()
    elif asset.get("original_ref"):
        from corpus_pipeline.production.records import read_asset_bytes
        blob = read_asset_bytes(asset["original_ref"])
    elif asset.get("image_path"):
        with Path(asset["image_path"]).open("rb") as stream:
            blob = stream.read(MAX_BYTES + 1)
    else:
        raise ValueError("image_path, original_ref or blob_loader required")
    if not isinstance(blob, bytes) or not 0 < len(blob) <= MAX_BYTES:
        raise ValueError("invalid or excessive image bytes")
    expected = asset.get("sha256", asset.get("image_sha256"))
    if not expected or hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("image bytes do not match expected SHA256")
    return blob, expected


class ResidentInferenceWorker:
    """One process, one device, resident models and a bounded decode thread pool.

    The instance is single-consumer; call ``iter_results`` repeatedly to reuse it.
    A scheduler owns cross-process concurrency and assigns distinct NPU devices.
    Each input has a SHA256 and a path, TAR original_ref or bytes loader. Outputs
    retain input order. SSCD embeddings are float32 numpy arrays in Python; the
    benchmark CLI stores them as binary vectors with JSONL byte-offset references.
    """

    def __init__(self, models_dir=DEFAULT_MODELS, sscd_model=DEFAULT_SSCD, *, device,
                 dtype="float32", batch_size=32, threads=8, models=MODEL_NAMES):
        if (type(batch_size) is not int or batch_size < 1 or type(threads) is not int or threads < 1):
            raise ValueError("batch_size and threads must be positive integers")
        names = tuple(models)
        if not names or len(set(names)) != len(names) or set(names) - set(MODEL_NAMES):
            raise ValueError("models must be a nonempty unique subset of cf384, unifd, sscd")
        started = time.perf_counter()
        self.torch, self.device, self.dtype, precision = _runtime(device, dtype, threads)
        self.batch_size, self.names = batch_size, names
        self.bundles = _load_bundles(models_dir, sscd_model, names)
        execution_options = {}
        for name, bundle in self.bundles.items():
            model = bundle["model"].eval()
            if name == "cf384":
                execution_options[name] = {**_configure_cf_attention(model, self.device),
                                           **_configure_cf_gelu(model, self.device, self.torch)}
            # ScriptModule intentionally does not expose requires_grad_().
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.to(device=self.device, dtype=self.dtype)
        self.synchronize()
        versions = {"python": platform.python_version(), "torch": self.torch.__version__}
        for package in ("torchvision", "timm", "safetensors", "torch-npu"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.identity = {"worker_version": WORKER_VERSION, "device": str(self.device), "dtype": dtype,
                         "precision_settings": precision,
                         "execution_options": execution_options,
                         "runtime": versions, "models": {name: bundle["identity"] for name, bundle in self.bundles.items()},
                         "code_sha256": file_sha(__file__),
                         "reference_implementation_sha256": {
                             "detectors": file_sha(ROOT / "detector_calibration/infer.py"),
                             "sscd": file_sha(ROOT / "dedup/sscd.py")},
                         "score_cache_used": False, "automatic_cpu_fallback": False,
                         "numerical_qualification_required": str(self.device) != "cpu" or dtype != "float32"}
        self.identity_sha256 = digest(self.identity)
        self.load_seconds = time.perf_counter() - started
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.closed = False
        self.stats = {"input_images": 0, "complete_images": 0, "batches": 0, "decode_seconds": 0.0,
                      "model_seconds": {name: 0.0 for name in names}, "model_batches": {name: 0 for name in names}}

    def synchronize(self):
        if str(self.device).startswith("npu"):
            self.torch.npu.synchronize(self.device)

    def _prepare(self, asset, names):
        started = time.perf_counter()
        expected = asset.get("sha256", asset.get("image_sha256"))
        result = {"asset_id": asset.get("asset_id", expected), "image_sha256": expected,
                  "model_identity_sha256": self.identity_sha256, "scores": {}, "status": "error"}
        try:
            allow_small = asset.get('allow_small_for_dedup', False)
            if type(allow_small) is not bool or (allow_small and tuple(names) != ('sscd',)):
                raise ValueError('small-image bypass is only valid for explicit SSCD-only dedup')
            blob, sha = _read_blob(asset)
            with Image.open(io.BytesIO(blob)) as opened:
                result.update(sha256=sha, original_width=opened.width, original_height=opened.height)
                if opened.width * opened.height > MAX_PIXELS:
                    raise ValueError("image pixel limit exceeded")
                if getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("multiframe input requires a frame-specific adapter")
                if min(opened.size) < 128 and not allow_small:
                    result.update(status="too_small", size_status="rejected_too_small",
                                  reason="original_image_short_edge_lt_128")
                    return result, None
                result["size_status"] = "i2t_dedup_small_image" if min(opened.size) < 128 else "pass"
                for key, value in zip(("width", "height"), opened.size):
                    declared = asset.get("original_" + key, asset.get(key))
                    if declared is not None and declared != value:
                        raise ValueError("declared original image dimensions mismatch")
                opened.load()
                tensors = {}
                if any(name != "sscd" for name in names):
                    rgb = opened.convert("RGB")  # Official detector path does not transpose EXIF.
                    for name in names:
                        if name != "sscd":
                            tensors[name] = self.bundles[name]["transform"](rgb)
                if "sscd" in names:
                    array, metadata = _sscd_input(opened)
                    tensors["sscd"] = self.torch.from_numpy(array)
                    result.update(metadata)
            result["status"] = "prepared"
            return result, tensors
        except Exception as exc:
            result.update(error_stage="decode", error_type=type(exc).__name__, error=str(exc)[:400])
            return result, None
        finally:
            result["decode_seconds"] = time.perf_counter() - started

    def infer_batch(self, assets, *, models=None):
        if self.closed:
            raise RuntimeError("worker is closed")
        names = self.names if models is None else tuple(models)
        if not names or len(set(names)) != len(names) or set(names) - set(self.names):
            raise ValueError("requested models must be a unique subset of resident models")
        # Take at most batch_size + 1, even if a caller accidentally passes an
        # unbounded generator. Oversized requests never trigger model execution.
        assets = list(islice(iter(assets), self.batch_size + 1))
        if len(assets) > self.batch_size:
            raise ValueError("infer_batch exceeds configured batch_size; use iter_results")
        if not assets:
            return []
        begin = time.perf_counter()
        prepared = list(self.pool.map(lambda asset: self._prepare(asset, names), assets))
        self.stats["decode_seconds"] += time.perf_counter() - begin
        valid = [(result, tensors) for result, tensors in prepared if tensors is not None]
        for name in names:
            if not valid:
                break
            start = time.perf_counter()
            batch, values = None, None
            try:
                batch = self.torch.stack([tensors[name] for _, tensors in valid]).to(device=self.device, dtype=self.dtype)
                self.synchronize()
                with self.torch.inference_mode():
                    values = self.bundles[name]["model"](batch).float()
                self.synchronize()
                if name == "sscd":
                    vectors = values.cpu().numpy()
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    if (vectors.shape != (len(valid), 512) or not np.isfinite(vectors).all()
                            or (norms < 1e-8).any()):
                        raise ValueError("invalid SSCD descriptor")
                    vectors = (vectors / norms).astype(np.float32)
                    for (result, _), vector in zip(valid, vectors):
                        result["sscd"] = {"status": "ok", "embedding": vector, "dimensions": 512}
                else:
                    logits = values.flatten()
                    if logits.shape != (len(valid),) or not self.torch.isfinite(logits).all().item():
                        raise ValueError("invalid detector logits")
                    probabilities = logits.sigmoid().cpu().tolist()
                    for (result, _), logit, prob in zip(valid, logits.cpu().tolist(), probabilities):
                        result["scores"][name] = {"status": "ok", "logit": logit, "prob": prob}
            except Exception as exc:
                failure = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:400]}
                for result, _ in valid:
                    if name == "sscd":
                        result["sscd"] = dict(failure)
                    else:
                        result["scores"][name] = dict(failure)
            finally:
                del batch, values
                self.stats["model_seconds"][name] += time.perf_counter() - start
                self.stats["model_batches"][name] += 1
        for result, _ in valid:
            result["status"] = "ok" if all((result.get("sscd", {}) if name == "sscd" else
                result["scores"].get(name, {})).get("status") == "ok" for name in names) else "incomplete"
        self.stats["batches"] += 1
        self.stats["input_images"] += len(prepared)
        self.stats["complete_images"] += sum(result["status"] == "ok" for result, _ in prepared)
        return [result for result, _ in prepared]

    def iter_results(self, assets, *, models=None):
        iterator = iter(assets)
        while batch := list(islice(iterator, self.batch_size)):
            yield from self.infer_batch(batch, models=models)

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True)
            self.bundles.clear()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _sample_rows(path, limit):
    count = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                if limit is not None and count >= limit:
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("sample JSONL rows must be objects")
                count += 1
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--sscd-model", type=Path, default=DEFAULT_SSCD)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.warmup_batches < 0 or args.repeat < 1 or args.limit is not None and args.limit < 1:
        parser.error("invalid warmup/repeat/limit")
    args.output.mkdir(parents=True, exist_ok=False)
    with ResidentInferenceWorker(args.models_dir, args.sscd_model, device=args.device, dtype=args.dtype,
                                  batch_size=args.batch_size, threads=args.threads, models=args.models) as worker:
        atomic_json(args.output / "model_identity.json", worker.identity)
        warmup = list(islice(_sample_rows(args.samples, args.limit), args.batch_size))
        if not warmup:
            raise ValueError("empty sample input")
        for _ in range(args.warmup_batches):
            rows = worker.infer_batch(warmup)
            if any(row["status"] not in {"ok", "too_small"} for row in rows):
                # Persist useful per-image errors even when a target device or
                # TorchScript operator cannot execute; never fall back to CPU.
                for row in rows:
                    if row.get("sscd", {}).get("status") == "ok":
                        row["sscd"].pop("embedding", None)
                atomic_json(args.output / "warmup_failure.json", rows)
                raise RuntimeError("warmup failed; see warmup_failure.json")
        baseline = json.loads(json.dumps(worker.stats))
        started = time.perf_counter()
        completed, failed, too_small, count = 0, 0, 0, 0
        with (args.output / "results.jsonl").open("w", encoding="utf-8") as stream, \
                (args.output / "sscd.f32").open("wb") as vectors:
            for repetition in range(args.repeat):
                for row in worker.iter_results(_sample_rows(args.samples, args.limit)):
                    row["benchmark_repeat"] = repetition
                    if row.get("sscd", {}).get("status") == "ok":
                        vector = row["sscd"].pop("embedding")
                        offset = vectors.tell()
                        vectors.write(np.asarray(vector, dtype="<f4").tobytes())
                        row["sscd"].update(embedding_ref={"path": "sscd.f32", "byte_offset": offset,
                                                        "length": 512 * 4, "dtype": "little_endian_float32"})
                    stream.write(canonical(row) + "\n")
                    count += 1
                    completed += row["status"] == "ok"
                    too_small += row["status"] == "too_small"
                    failed += row["status"] not in {"ok", "too_small"}
            stream.flush()
            vectors.flush()
            os.fsync(stream.fileno())
            os.fsync(vectors.fileno())
        elapsed = time.perf_counter() - started
        summary = {"worker_version": WORKER_VERSION, "identity_sha256": worker.identity_sha256,
                   "samples_sha256": file_sha(args.samples), "device": str(worker.device), "dtype": args.dtype,
                   "batch_size": args.batch_size, "threads": args.threads, "load_seconds": worker.load_seconds,
                   "warmup_batches": args.warmup_batches, "repeat": args.repeat, "input_images": count,
                   "completed_images": completed, "too_small_images": too_small,
                   "failed_images": failed, "elapsed_seconds": elapsed,
                   "completed_images_per_second": completed / elapsed,
                   "input_images_per_second": count / elapsed,
                   "decode_seconds": worker.stats["decode_seconds"] - baseline["decode_seconds"],
                   "model_seconds": {name: worker.stats["model_seconds"][name] - baseline["model_seconds"][name]
                                     for name in worker.names},
                   "models_loaded_once": True, "score_cache_used": False, "published_training": False,
                   "success": failed == 0, "results_sha256": file_sha(args.output / "results.jsonl"),
                   "vectors_sha256": file_sha(args.output / "sscd.f32")}
        atomic_json(args.output / "COMPLETE.json", summary)
        print(canonical(summary), flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
