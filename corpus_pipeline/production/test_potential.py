from copy import deepcopy
import unittest

from .potential import annotate_potential


class PotentialTests(unittest.TestCase):
    def record(self, source="chartqa", collection="dcvlm", decision="accepted", images=("a",)):
        return {"collection": collection, "source": source, "original_sample_id": "record",
                "source_record_sha256": "record-sha", "domain": "unknown",
                "normalized": {"category": "single_i2t", "decision": decision, "reason_codes": []},
                "source_image_refs": [{"sha256": key, "width": 99999, "height": 99999} for key in images]}

    def contracts(self):
        return {"schema_version": "prepared.layout_source_contracts.v1", "contracts": [
            {"collection": "dcvlm", "source": "chartqa", "match": {"collection": "dcvlm", "source": "chartqa"},
             "tag": "layout_candidate", "domain": "chart", "upstream_annotation_type": ["chart_table", "qa"],
             "annotation_availability": "upstream_declared_local_unverified", "local_binding_verified": False,
             "evidence": [{"url": "https://example.invalid/dataset", "claim": "Upstream has chart tables.", "basis": "primary_dataset_page"}],
             "reason": "Source-level chart structure; individual clarity unreviewed."}]}

    def asset(self, key="a", width=1024, height=1024):
        return {"asset_id": key, "original_width": width, "original_height": height, "width": 2048, "height": 2048}

    def test_native_pixel_boundary_does_not_select_future_pool_or_claim_quality(self):
        record = self.record()
        record["record_admission"] = {"decision": "admit", "reason_codes": [], "evidence": {"gates": "passed"}}
        value = annotate_potential(record, {"a": self.asset()}, source_contracts=self.contracts())
        self.assertTrue(value["assets"][0]["resolution_eligible"])
        self.assertEqual(value["assets"][0]["native_pixels"], 1048576)
        self.assertTrue(value["training_admitted"])
        self.assertFalse(value["automatic_api_required"])
        self.assertFalse(value["future_highres_pool_selected"])
        self.assertIsNone(value["visual_quality_claim"])
        self.assertFalse(value["moss_fewshot_qualified"])

    def test_resizing_or_declared_metadata_never_creates_original_resolution(self):
        value = annotate_potential(self.record(), {"a": self.asset(width=512, height=512)}, source_contracts=self.contracts())
        self.assertFalse(value["assets"][0]["resolution_eligible"])
        self.assertEqual(value["assets"][0]["native_pixels"], 262144)
        self.assertEqual(value["assets"][0]["recaption_potential_tag"], "below_pixel_threshold")

    def test_huge_narrow_image_cannot_bypass_global_minimum(self):
        value = annotate_potential(self.record(), {"a": self.asset(width=127, height=20000)}, source_contracts=self.contracts())
        self.assertGreater(value["assets"][0]["native_pixels"], value["pixel_threshold"])
        self.assertFalse(value["assets"][0]["resolution_eligible"])
        self.assertEqual(value["assets"][0]["recaption_potential_tag"], "below_global_minimum_short_edge")

    def test_quarantined_source_still_gets_resolution_tag_without_admission(self):
        record = self.record(decision="quarantine")
        value = annotate_potential(record, {"a": self.asset()}, source_contracts=self.contracts())
        self.assertTrue(value["assets"][0]["resolution_eligible"])
        self.assertFalse(value["training_admitted"])
        self.assertEqual(value["record_admission"]["decision"], "pending")
        self.assertTrue(value["potential_does_not_override_admission"])

    def test_source_accepted_does_not_invent_final_gate_admission(self):
        value = annotate_potential(self.record(), {"a": self.asset()}, source_contracts=self.contracts())
        self.assertFalse(value["training_admitted"])
        self.assertEqual(value["record_admission"]["decision"], "not_evaluated")

    def test_upstream_annotation_claim_is_independent_of_local_recovery(self):
        contracts = self.contracts()
        contracts["contracts"][0]["local_binding_verified"] = True
        value = annotate_potential(self.record(), {"a": self.asset()}, source_contracts=contracts)
        self.assertEqual(value["layout_source_route"], "layout_candidate")
        self.assertEqual(value["upstream_annotation_claim"]["types"], ["chart_table", "qa"])
        self.assertEqual(value["local_recovered_status"]["status"], "not_verified")
        self.assertFalse(value["assets"][0]["visual_quality_verified"])

    def test_unknown_source_is_exact_matched_not_name_or_domain_guessed(self):
        for record in (self.record(source="chartqa_extra"), self.record(collection="genund")):
            value = annotate_potential(record, {"a": self.asset()}, source_contracts=self.contracts())
            self.assertEqual(value["layout_source_route"], "sample_to_decide")
            self.assertEqual(value["upstream_annotation_claim"]["availability"], "unknown")

    def test_missing_dimensions_remain_unknown_and_all_image_slots_count(self):
        record = self.record(images=("a", "b", "c"))
        value = annotate_potential(record, {"a": self.asset(), "b": self.asset("b", width=True)}, source_contracts=self.contracts())
        self.assertEqual(value["image_slots"], 3)
        self.assertEqual(value["resolution_unknown_image_slots"], 2)
        self.assertEqual(value["resolution_eligible_image_slots"], 1)
        self.assertIsNone(value["assets"][2]["resolution_eligible"])
        self.assertFalse(value["all_required_images_resolution_eligible"])

    def test_pure_text_does_not_acquire_visual_potential(self):
        value = annotate_potential(self.record(images=()), {}, source_contracts=self.contracts())
        self.assertFalse(value["any_resolution_eligible"])
        self.assertFalse(value["all_required_images_resolution_eligible"])
        self.assertEqual(value["assets"], [])

    def test_local_annotation_requires_exact_binding_and_preserves_coordinate_description(self):
        record = self.record()
        record["local_layout_annotation"] = {"status": "recovered", "verified": True, "source_record_sha256": "record-sha",
                                            "asset_ids": ["a"], "coordinate_system": "absolute_pixel_xyxy",
                                            "annotation_refs": [{"locator": {"path": "readonly.json", "key": "a"}, "sha256": "annotation-sha"}]}
        value = annotate_potential(record, {"a": self.asset()}, source_contracts=self.contracts())
        self.assertEqual(value["local_recovered_status"]["status"], "recovered")
        self.assertEqual(value["local_recovered_status"]["coordinate_system"], "absolute_pixel_xyxy")
        record["local_layout_annotation"]["asset_ids"] = ["other"]
        value = annotate_potential(record, {"a": self.asset()}, source_contracts=self.contracts())
        self.assertEqual(value["local_recovered_status"]["status"], "not_verified")

    def test_input_is_unchanged_and_invalid_contracts_fail_closed(self):
        record, contracts, assets = self.record(), self.contracts(), {"a": self.asset()}
        original = deepcopy((record, contracts, assets))
        annotate_potential(record, assets, source_contracts=contracts)
        self.assertEqual((record, contracts, assets), original)
        contracts["contracts"].append(deepcopy(contracts["contracts"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            annotate_potential(record, assets, source_contracts=contracts)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            annotate_potential(record, assets, source_contracts=self.contracts(), pixel_threshold=True)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            annotate_potential(record, {"a": self.asset("wrong")}, source_contracts=self.contracts())


if __name__ == "__main__":
    unittest.main()
