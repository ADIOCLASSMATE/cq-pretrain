from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import numpy as np
from PIL import Image
import torch

from .inference import ResidentInferenceWorker, _configure_cf_attention, _configure_cf_gelu, _configure_npu_precision, _load_bundles, _runtime, main
from corpus_pipeline.detector_calibration.infer import preprocessing
from corpus_pipeline.dedup.sscd import load_image_asset


class TinyDetector(torch.nn.Module):
    def __init__(self, bad=False):
        super().__init__()
        self.calls = 0
        self.bad = bad

    def forward(self, batch):
        self.calls += 1
        return batch.mean((1, 2, 3)).unsqueeze(1) * (float("nan") if self.bad else 1)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return (batch.mean((1, 2, 3)).unsqueeze(1) +
                torch.arange(1, 513, device=batch.device, dtype=batch.dtype).unsqueeze(0))


def fixture_bundles(*args):
    names = args[-1]
    return {name: {"model": TinyEncoder() if name == "sscd" else TinyDetector(),
                   "transform": None if name == "sscd" else preprocessing(name)[0],
                   "identity": {"fixture_only": True, "model": name}}
            for name in names}


class ResidentInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.loader = patch("corpus_pipeline.cluster.inference._load_bundles", side_effect=fixture_bundles)
        self.mock_loader = self.loader.start()

    def tearDown(self):
        self.loader.stop()
        self.temp.cleanup()

    def image(self, size=(192, 256), rgba=False, exif=False):
        mode = "RGBA" if rgba else "RGB"
        picture = Image.new(mode, size, (23, 74, 168, 90) if rgba else (23, 74, 168))
        buffer = io.BytesIO()
        metadata = Image.Exif()
        if exif:
            metadata[274] = 6
        picture.save(buffer, format="PNG", exif=metadata)
        blob = buffer.getvalue()
        sha = hashlib.sha256(blob).hexdigest()
        path = self.root / (sha + ".png")
        path.write_bytes(blob)
        return {"image_path": str(path), "image_sha256": sha, "asset_id": sha}, blob

    def worker(self, **kwargs):
        return ResidentInferenceWorker(device="cpu", threads=1, batch_size=kwargs.pop("batch_size", 2), **kwargs)

    def test_npu_precision_explicitly_disables_default_reduced_precision(self):
        fake = SimpleNamespace(npu=SimpleNamespace(set_option=Mock(), conv=SimpleNamespace(allow_hf32=True),
                                                   matmul=SimpleNamespace(allow_hf32=True)))
        settings = _configure_npu_precision(fake)
        fake.npu.set_option.assert_called_once_with({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
        self.assertIs(settings["conv_allow_hf32"], False)
        self.assertIs(settings["matmul_allow_hf32"], False)

    def test_cf_explicit_attention_only_changes_npu_execution(self):
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        for module in model:
            module.fused_attn = True
        cpu = _configure_cf_attention(model, "cpu")
        self.assertEqual(cpu["fused_attn_enabled_modules"], 2)
        self.assertFalse(cpu["fused_attn_disabled_for_npu"])
        npu = _configure_cf_attention(model, "npu:0")
        self.assertEqual(npu["attention_module_count"], 2)
        self.assertEqual(npu["fused_attn_enabled_modules"], 0)
        self.assertTrue(npu["fused_attn_disabled_for_npu"])

    def test_cf_exact_gelu_replaces_only_npu_none_approximation(self):
        model = torch.nn.Sequential(torch.nn.GELU(), torch.nn.GELU(approximate="tanh"))
        original = model[0]
        self.assertEqual(_configure_cf_gelu(model, "cpu", torch)["gelu_erf_replacements"], 0)
        self.assertIs(model[0], original)
        x = torch.linspace(-10, 10, 1000)
        expected = original(x)
        self.assertEqual(_configure_cf_gelu(model, "npu:0", torch)["gelu_erf_replacements"], 1)
        torch.testing.assert_close(model[0](x), expected, atol=1e-6, rtol=1e-5)
        self.assertIsInstance(model[1], torch.nn.GELU)

    def test_models_loaded_once_and_all_three_outputs_checked(self):
        asset, _ = self.image()
        with self.worker() as worker:
            models = {name: bundle["model"] for name, bundle in worker.bundles.items()}
            first = worker.infer_batch([asset])
            second = worker.infer_batch([asset])
            self.assertEqual(self.mock_loader.call_count, 1)
            self.assertEqual([model.calls for model in models.values()], [2, 2, 2])
            self.assertEqual(first[0]["status"], "ok")
            self.assertEqual(first[0]["scores"], second[0]["scores"])
            self.assertEqual(first[0]["sscd"]["embedding"].shape, (512,))
            self.assertEqual(first[0]["sscd"]["embedding"].dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(first[0]["sscd"]["embedding"])), 1.0, places=6)
            self.assertFalse(worker.identity["score_cache_used"])
            self.assertFalse(worker.identity["automatic_cpu_fallback"])

    def test_per_stage_selection_reuses_loaded_models(self):
        asset, _ = self.image()
        with self.worker() as worker:
            row = worker.infer_batch([asset], models=("cf384", "unifd"))[0]
            self.assertEqual(row["status"], "ok")
            self.assertNotIn("sscd", row)
            self.assertEqual(worker.bundles["sscd"]["model"].calls, 0)
            row = worker.infer_batch([asset], models=("sscd",))[0]
            self.assertEqual(row["scores"], {})
            self.assertEqual(worker.bundles["cf384"]["model"].calls, 1)
            self.assertEqual(worker.bundles["sscd"]["model"].calls, 1)

    def test_torchscript_model_loaded_without_unsupported_module_methods(self):
        asset, _ = self.image()
        bundle = fixture_bundles(None, None, ("sscd",))
        bundle["sscd"]["model"] = torch.jit.trace(TinyEncoder(), torch.zeros(1, 3, 320, 320))
        self.mock_loader.side_effect = None
        self.mock_loader.return_value = bundle
        with self.worker(models=("sscd",)) as worker:
            self.assertEqual(worker.infer_batch([asset])[0]["status"], "ok")

    def test_sscd_preprocessing_exactly_matches_reference_for_alpha_and_exif(self):
        for rgba, exif in ((False, False), (True, False), (False, True), (True, True)):
            asset, _ = self.image(rgba=rgba, exif=exif)
            expected_meta, expected = load_image_asset(asset)
            with self.worker(models=("sscd",)) as worker:
                metadata, tensors = worker._prepare(asset, ("sscd",))
            with self.subTest(rgba=rgba, exif=exif):
                np.testing.assert_array_equal(tensors["sscd"].numpy(), expected)
                self.assertEqual(metadata["pixel_sha256"], expected_meta["pixel_sha256"])
                self.assertEqual(metadata["original_width"], expected_meta["width"])
                self.assertEqual(metadata["oriented_width"], expected_meta["oriented_width"])

    def test_official_detector_crops_and_exif_behavior_are_preserved(self):
        asset, _ = self.image(exif=True)
        with self.worker() as worker:
            _, tensors = worker._prepare(asset, ("cf384", "unifd"))
            with Image.open(asset["image_path"]) as image:
                for name in ("cf384", "unifd"):
                    expected = preprocessing(name)[0](image.convert("RGB"))
                    torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)
            self.assertEqual(tensors["cf384"].shape, (3, 384, 384))
            self.assertEqual(tensors["unifd"].shape, (3, 224, 224))

    def test_original_bytes_loaded_once_for_joint_models(self):
        asset, blob = self.image()
        calls = []
        asset.pop("image_path")
        asset["blob_loader"] = lambda: (calls.append(True), blob)[1]
        with self.worker() as worker:
            self.assertEqual(worker.infer_batch([asset])[0]["status"], "ok")
        self.assertEqual(len(calls), 1)

    def test_tar_byte_range_supported_without_extraction(self):
        asset, blob = self.image()
        path = self.root / "fixture.tar"
        path.write_bytes(b"prefix" + blob + b"suffix")
        asset.pop("image_path")
        asset["original_ref"] = {"path": str(path), "offset": 6, "length": len(blob), "sha256": asset["image_sha256"]}
        with self.worker() as worker:
            self.assertEqual(worker.infer_batch([asset])[0]["status"], "ok")

    def test_streaming_consumption_bounded_and_order_preserved(self):
        asset, _ = self.image()
        consumed = []
        def stream():
            for index in range(7):
                consumed.append(index)
                yield {**asset, "asset_id": str(index)}
        with self.worker(batch_size=2) as worker:
            iterator = worker.iter_results(stream())
            self.assertEqual(next(iterator)["asset_id"], "0")
            self.assertEqual(consumed, [0, 1])
            rows = list(iterator)
            self.assertEqual([row["asset_id"] for row in rows], [str(i) for i in range(1, 7)])
            self.assertEqual(worker.stats["batches"], 4)

    def test_small_bad_hash_and_mismatched_size_are_per_image_failures(self):
        good, _ = self.image()
        small, _ = self.image((127, 500))
        for bad in ({**good, "image_sha256": "0" * 64}, {**good, "original_width": 1}):
            with self.worker() as worker:
                rows = worker.infer_batch([bad, good])
                self.assertEqual([row["status"] for row in rows], ["error", "ok"])
                self.assertEqual(rows[0]["error_stage"], "decode")
                self.assertEqual(rows[0]["scores"], {})
        with self.worker() as worker:
            row = worker.infer_batch([small])[0]
            self.assertEqual(row["status"], "too_small")
            self.assertEqual(row["size_status"], "rejected_too_small")
            self.assertEqual(row["original_width"], 127)
            self.assertTrue(all(bundle["model"].calls == 0 for bundle in worker.bundles.values()))

    def test_nonfinite_output_never_appears_successful(self):
        asset, _ = self.image()
        with self.worker() as worker:
            worker.bundles["cf384"]["model"] = TinyDetector(bad=True)
            row = worker.infer_batch([asset])[0]
            self.assertEqual(row["status"], "incomplete")
            self.assertEqual(row["scores"]["cf384"]["status"], "error")
            self.assertEqual(row["scores"]["unifd"]["status"], "ok")
            self.assertEqual(row["sscd"]["status"], "ok")

    def test_forward_exception_preserved_without_unbound_cleanup_error(self):
        asset, _ = self.image()
        with self.worker() as worker:
            with patch.object(worker.bundles["cf384"]["model"], "forward", side_effect=RuntimeError("fixture-forward-error")):
                row = worker.infer_batch([asset])[0]
            self.assertEqual(row["status"], "incomplete")
            self.assertEqual(row["scores"]["cf384"]["error_type"], "RuntimeError")
            self.assertEqual(row["scores"]["cf384"]["error"], "fixture-forward-error")
            self.assertEqual(row["sscd"]["status"], "ok")

    def test_invalid_configuration_and_oversized_batch_fail(self):
        for arguments in ({"batch_size": 0}, {"threads": 0}, {"models": ()}, {"models": ("cf384", "cf384")}):
            with self.assertRaises(ValueError):
                ResidentInferenceWorker(device="cpu", **arguments)
        with self.assertRaises(ValueError):
            _runtime("cuda:0", "float32", 1)
        asset, _ = self.image()
        with self.worker() as worker:
            with self.assertRaises(ValueError):
                worker.infer_batch([asset] * 3)
            with self.assertRaises(ValueError):
                worker.infer_batch([asset], models=("invalid",))
        with self.assertRaises(RuntimeError):
            worker.infer_batch([asset])

    def test_mismatch_in_pinned_sscd_weights_is_rejected(self):
        wrong = self.root / "wrong.pt"
        wrong.write_bytes(b"not model weights")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            _load_bundles(self.root, wrong, ("sscd",))

    def test_benchmark_cli_streams_binary_vectors_and_excludes_warmup(self):
        asset, _ = self.image()
        samples = self.root / "samples.jsonl"
        samples.write_text(json.dumps(asset) + "\n")
        output = self.root / "benchmark"
        with patch("sys.argv", ["inference", "--samples", str(samples), "--output", str(output),
                                "--device", "cpu", "--batch-size", "2", "--threads", "1", "--repeat", "2"]):
            self.assertEqual(main(), 0)
        summary = json.loads((output / "COMPLETE.json").read_text())
        self.assertEqual(summary["input_images"], 2)
        self.assertEqual(summary["completed_images"], 2)
        self.assertEqual(summary["warmup_batches"], 1)
        self.assertTrue(summary["models_loaded_once"])
        self.assertEqual((output / "sscd.f32").stat().st_size, 2 * 512 * 4)
        rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual([r["sscd"]["embedding_ref"]["byte_offset"] for r in rows], [0, 2048])
        self.assertTrue(all("embedding" not in row["sscd"] for row in rows))


if __name__ == "__main__":
    unittest.main()
