"""Boundary tests for static preparation without weakening existing policy."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from corpus_pipeline.prepared.profiles import (
    REPO, compile_profiles, digest, match_profiles, validate_record, validate_snapshot, write_profiles,
)


class PreparedProfilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = compile_profiles()

    def envelope(self, collection, source, raw, path="/data/record.json"):
        return {"collection": collection, "source": source, "locator": {"path": path},
                "raw": raw, "image_refs": [{"kind": "file", "path": "/data/image.jpg", "exists": True}]}

    def test_all_ordered_rules_preserved_including_narrow_overrides(self):
        original = json.loads((REPO / "corpus_pipeline/config/sources.json").read_text())
        self.assertEqual(len(original["sources"]), 634)
        self.assertEqual(self.registry["source_rules"], original["sources"])
        self.assertEqual(self.registry["coverage"]["parent_sources"], {"dcvlm": 165, "genund": 21})
        self.assertEqual(self.registry["coverage"]["missing_inventory_sources"], {"dcvlm": [], "genund": []})
        for index, profile in enumerate(self.registry["profiles"]):
            self.assertEqual(profile["source_rule_ref"]["sha256"], digest(original["sources"][index]))
            self.assertEqual(profile["match"], original["sources"][index]["match"])

    def test_dcvlm_cannot_reverse_caption_to_t2i(self):
        result = validate_record(self.registry, self.envelope("dcvlm", "coco_caption", {"caption": "A dog."}))
        self.assertEqual(result["native_category"], "single_i2t")
        self.assertNotIn("single_t2i", result["candidate_views"])
        self.assertIsNone(result["release_eligibility"])
        self.assertFalse(self.registry["contracts"]["dcvlm"]["classification"]["dcvlm_direction_conversion_allowed"])

    def test_user_minimum_128_applies_to_both_collections_without_exceptions(self):
        for collection in ('dcvlm', 'genund'):
            size = self.registry['contracts'][collection]['size']
            self.assertEqual(size['global_short_side_threshold'], 128)
            self.assertFalse(size['domain_exceptions'])
            self.assertFalse(size['upsampling_can_satisfy_gate'])

    def test_genund_caption_conversion_does_not_convert_qa(self):
        caption = validate_record(self.registry, self.envelope("genund", "laion-aesthetics-12m", {"caption": "A dog."}))
        self.assertIn("single_t2i", caption["candidate_views"])
        qa = validate_record(self.registry, self.envelope("genund", "laion-aesthetics-12m", {
            "messages": [{"role": "user", "content": "<image> How many dogs?"},
                         {"role": "assistant", "content": "One."}]}))
        self.assertEqual(qa["native_category"], "single_i2t")
        self.assertNotIn("single_t2i", qa["candidate_views"])

    def test_unknown_origin_and_inventory_only_never_autoaccept(self):
        result = validate_record(self.registry, self.envelope("dcvlm", "ai2d", {"caption": "A diagram."}))
        self.assertEqual(result["stage1_decision"], "quarantine")
        self.assertIsNone(result["release_eligibility"])
        reg = deepcopy(self.registry)
        index = next(i for i, rule in enumerate(reg["source_rules"]) if rule["id"] == "dcvlm:ai2d")
        reg["source_rules"][index]["inventory_only"] = True
        self.assertEqual(match_profiles(reg, "dcvlm", "ai2d"), [])
        self.assertTrue(all(not profile["preparation"]["automatic_accept"] for profile in reg["profiles"]))

    def test_narrow_generated_path_remains_rejected(self):
        path = "/inspire/dataset/gen-und-mm-pretrain/01/openuni/data/text-to-image-2M/dall-e/0001.json"
        matching = match_profiles(self.registry, "genund", "openuni", [path])
        self.assertGreaterEqual(len(matching), 2)
        self.assertTrue(any(profile["provenance"]["image_origin_prior"] == "ai_generated" for profile in matching))
        result = validate_record(self.registry, self.envelope("genund", "openuni", {"caption": "A dog."}, path))
        self.assertEqual(result["stage1_decision"], "rejected")

    def test_future_snapshot_requires_bound_schema_and_inventory_receipts(self):
        snapshot = "dcvlm-balance-200B-future"
        result = validate_snapshot(self.registry, "dcvlm", snapshot)
        self.assertEqual(result["status"], "requires_validation")
        self.assertFalse(result["historical_counts_inherited"])
        observed = {"schema_validated": True, "inventory_validated": True, "source_names": ["ai2d"],
                    "schema_evidence_sha256": "a" * 64, "inventory_evidence_sha256": "b" * 64,
                    "snapshot_sha256": "c" * 64, "snapshot_id": "other_snapshot"}
        self.assertIn("validation_receipt_snapshot_mismatch", validate_snapshot(
            self.registry, "dcvlm", snapshot, observations=observed)["reason_codes"])
        observed["snapshot_id"] = snapshot
        result = validate_snapshot(self.registry, "dcvlm", snapshot, observations=observed)
        self.assertEqual(result["status"], "validation_receipts_present")
        self.assertFalse(result["production_admission"])
        self.assertFalse(result["historical_counts_inherited"])
        observed["source_names"].append("brand_new_source")
        self.assertEqual(validate_snapshot(self.registry, "dcvlm", snapshot, observations=observed)["status"], "requires_validation")

    def test_pending_adapters_and_historical_t2i50_are_not_promoted(self):
        profiles = {p["profile_id"]: p for p in self.registry["profiles"]}
        history = profiles["dcvlm:ai2d"]["historical_statistics"]
        self.assertTrue(history["retired_t2i50_snapshot"])
        self.assertFalse(history["caption_direction_usable"])
        self.assertTrue(history["counts_not_inherited_by_new_snapshot"])
        recap = profiles["genund:Stable_Diffusion_3_Recaption"]
        self.assertIn("cc12m_imagenet_and_sd3.unlocated_annotations", recap["pending_adapter_refs"])
        self.assertFalse(recap["preparation"]["production_ready"])
        source = profiles["genund:Infinity-MM"]
        self.assertIn("genund.infinity_mm.legacy_source_images_root", source["adapter_rule_refs"])
        self.assertEqual(source["source_contract_ref"]["source_id"], "Infinity-MM")

    def test_reproducible_compilation_and_writes(self):
        again = compile_profiles()
        self.assertEqual(self.registry, again)
        with tempfile.TemporaryDirectory() as directory:
            first = write_profiles(self.registry, directory)
            before = Path(first["json"]).read_bytes()
            write_profiles(again, directory)
            self.assertEqual(before, Path(first["json"]).read_bytes())
            self.assertEqual(len(Path(first["csv"]).read_text().splitlines()), 635)


if __name__ == "__main__":
    unittest.main()
