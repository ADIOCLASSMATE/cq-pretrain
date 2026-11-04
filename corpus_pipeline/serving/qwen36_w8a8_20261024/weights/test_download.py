"""Small offline fixtures verify the readiness gate never mistakes float weights for W8A8."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("qwen36_weights_download", Path(__file__).with_name("download.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WeightValidationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
        self.shard = "quant_model_weights-00001-of-00001.safetensors"
        self.fixture()

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def fixture(self, *, dtype="I8", declared="W8A8_DYNAMIC", data_bytes=4, index_bytes=4, offset_end=4):
        self.write_json("config.json", {"architectures": ["Qwen3_5MoeForConditionalGeneration"]})
        self.write_json("quant_model_description.json", {self.key: declared, "version": "1.0.0", "metadata": {}})
        self.write_json("quant_model_weights.safetensors.index.json", {
            "weight_map": {self.key: self.shard}, "metadata": {"total_size": index_bytes}})
        header = json.dumps({self.key: {"dtype": dtype, "shape": [4], "data_offsets": [0, offset_end]}}).encode()
        (self.root / self.shard).write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * data_bytes)

    def test_int8_payload_and_ascend_declaration_must_agree(self):
        result = MODULE.verify_tensors(self.root)
        self.assertEqual(result["tensor_bytes_by_dtype"], {"I8": 4})
        self.assertEqual(result["int8_weight_tensor_count"], 1)
        self.assertTrue(result["all_index_tensors_present"])
        for dtype, declared in (("BF16", "FLOAT"), ("BF16", "W8A8_DYNAMIC"), ("F8_E4M3", "W8A8_DYNAMIC"), ("I8", "FLOAT")):
            with self.subTest(dtype=dtype, declared=declared):
                self.fixture(dtype=dtype, declared=declared)
                with self.assertRaises(ValueError):
                    MODULE.verify_tensors(self.root)

    def test_shard_index_and_byte_ranges_are_not_trusted_blindly(self):
        for error in ("wrong_shard", "missing_tensor", "truncated", "total_size"):
            with self.subTest(error=error):
                self.fixture()
                if error in ("wrong_shard", "missing_tensor"):
                    index = json.loads((self.root / "quant_model_weights.safetensors.index.json").read_text())
                    if error == "wrong_shard":
                        index["weight_map"][self.key] = "../outside.safetensors"
                    else:
                        index["weight_map"]["missing.weight"] = self.shard
                    self.write_json("quant_model_weights.safetensors.index.json", index)
                elif error == "truncated":
                    self.fixture(data_bytes=3)
                else:
                    self.fixture(index_bytes=5)
                with self.assertRaises(ValueError):
                    MODULE.verify_tensors(self.root)


if __name__ == "__main__":
    unittest.main()
