"""Adversarial policy tests: fail closed without confusing AI text and images."""
import copy
import unittest

try:
    from .policy import classify, evaluate, normalize, resolve_source_config
except ImportError:
    from policy import classify, evaluate, normalize, resolve_source_config


class PolicyTests(unittest.TestCase):
    def envelope(self, raw=None, images=1, **kwargs):
        return dict(collection="test", source="photos", locator={"path": "/data/part.parquet", "row_index": 3},
                    raw=raw or {"caption": "A tree."},
                    image_refs=[{"kind": "file", "path": f"/data/photo-{i}.jpg", "exists": True}
                                for i in range(images)], **kwargs)

    def evaluated(self, envelope=None, origin="natural", caption="unknown", rules=None):
        record = normalize(envelope or self.envelope(), {"image_origin": origin, "caption_origin": caption})
        return evaluate(record, rules or {"version": "test-v1"})

    def test_genund_natural_caption_pair_has_two_views_one_record(self):
        env = self.envelope()
        env["collection"] = "genund"
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["category"], "single_i2t")
        self.assertEqual(result["eligible_views"], ["single_t2i", "single_i2t"])
        self.assertEqual(len(result["images"]), 1)

    def test_unknown_is_quarantine_not_accepted(self):
        result = self.evaluated(origin="unknown")
        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("image_provenance_unknown", result["reason_codes"])

    def test_ai_caption_retained_and_marked(self):
        result = self.evaluated(caption="ai")
        self.assertEqual(result["decision"], "accepted")
        self.assertTrue(result["provenance"]["text"]["ai_generated"])

    def test_ai_text_does_not_contaminate_image_provenance(self):
        result = self.evaluated(self.envelope({"caption": "An AI-generated painting with a Flux watermark."}))
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["provenance"]["images"][0]["origin"], "natural")
        self.assertIsNone(result["provenance"]["text"]["ai_generated"])

    def test_explicit_natural_only_rejects_all_non_natural_statuses(self):
        for origin in ("ai_generated", "ai_edited", "programmatic", "rendered"):
            with self.subTest(origin=origin):
                self.assertEqual(self.evaluated(origin=origin, rules={"policy": "natural_only"})["decision"], "rejected")

    def test_default_allows_non_ai_programmatic_and_rendered(self):
        for origin in ("programmatic", "rendered"):
            result = self.evaluated(origin=origin)
            self.assertEqual(result["decision"], "accepted")
            self.assertEqual(result["provenance"]["images"][0]["origin"], origin)

    def test_generic_synthetic_flags_do_not_mean_ai_generated(self):
        for flag in ("is_generated", "is_synthetic_image", "image_is_synthetic"):
            env = self.envelope({"caption": "A rendered chart", flag: True})
            self.assertEqual(self.evaluated(env, origin="programmatic")["decision"], "accepted")
            result = self.evaluated(env, origin="natural")
            self.assertEqual(result["decision"], "quarantine")
            self.assertNotIn("image_ai_generated", result["reason_codes"])

    def test_generic_synthetic_origin_quarantined_without_ai_claim(self):
        for value in ("synthetic", "generated"):
            result = self.evaluated(self.envelope({"caption": "A picture", "image_origin": value}))
            self.assertEqual(result["decision"], "quarantine")
            self.assertEqual(result["provenance"]["images"][0]["origin"], "unknown")

    def test_ai_edit_from_natural_original_retained_and_marked(self):
        for field in ("edit_source_origin", "source_image_origin", "original_image_origin", "edit_source_provenance"):
            with self.subTest(field=field):
                value = {"origin": "natural", "source_id": "photo-123"} if field.endswith("provenance") else "natural"
                env = self.envelope({"caption": "A red tree", "image_origin": "ai_edited", field: value})
                result = self.evaluated(env, caption="ai")
                self.assertEqual(result["decision"], "accepted")
                image = result["provenance"]["images"][0]
                self.assertEqual(image["origin"], "ai_edited")
                self.assertTrue(image["ai_edited"])
                self.assertEqual(image["edit_source_origin"], "natural")
                self.assertEqual(image["policy_exception"], "ai_edit_from_natural_source")
                self.assertTrue(image["edit_source_evidence"])
                self.assertTrue(result["provenance"]["text"]["ai_generated"])

    def test_ai_edit_source_unknown_or_mixed_quarantined(self):
        for origin in (None, "unknown", "mixed"):
            raw = {"caption": "A tree", "image_origin": "ai_edited"}
            if origin:
                raw["edit_source_origin"] = origin
            result = self.evaluated(self.envelope(raw))
            self.assertEqual(result["decision"], "quarantine")
            self.assertIn("ai_edit_source_" + (origin or "unknown"), result["reason_codes"])

    def test_edit_of_non_natural_source_rejected(self):
        for origin in ("ai_generated", "ai_edited", "programmatic", "rendered"):
            raw = {"caption": "A tree", "image_origin": "ai_edited", "edit_source_origin": origin}
            result = self.evaluated(self.envelope(raw))
            self.assertEqual(result["decision"], "rejected")
            self.assertIn("ai_edit_source_not_natural", result["reason_codes"])

    def test_generated_target_not_exempted_by_real_reference(self):
        env = self.envelope({"caption": "A tree", "image_origin": "ai_generated",
                             "is_ai_edited": True, "edit_source_origin": "natural"})
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "rejected")
        self.assertIn("image_ai_generated", result["reason_codes"])

    def test_edit_lineage_cannot_spread_from_neighbor_or_record(self):
        env = self.envelope({"format": "interleaved", "edit_source_origin": "natural", "segments": [
            {"kind": "text", "text": "First"}, {"kind": "image", "image_index": 0},
            {"kind": "text", "text": "Edited"}, {"kind": "image", "image_index": 1}]}, images=2)
        env["image_refs"][1]["image_origin"] = "ai_edited"
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("ai_edit_source_unknown", result["reason_codes"])
        env["image_refs"][1]["metadata"] = {"edit_source_provenance": {"origin": "natural", "source_id": "photo-0"}}
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["provenance"]["images"][1]["edit_source_origin"], "natural")

    def test_conflicting_edit_lineage_negative_evidence_wins(self):
        env = self.envelope({"caption": "A tree", "image_origin": "ai_edited", "edit_source_origin": "natural"})
        env["image_refs"][0]["metadata"] = {"edit_source_origin": "ai_generated"}
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(result["provenance"]["images"][0]["edit_source_origin"], "ai_generated")

    def test_conflicting_fields_inside_lineage_object_are_not_ignored(self):
        for lineage in ({"origin": "natural", "is_ai_generated": True},
                        {"status": "natural", "origin": "ai_generated"},
                        {"origin": "natural", "metadata": {"image_origin": "ai_generated"}},
                        {"origin": "natural", "metadata": {"origin": "rendered"}},
                        {"origin": "natural", "is_ai_edited": True}):
            with self.subTest(lineage=lineage):
                env = self.envelope({"caption": "A tree", "image_origin": "ai_edited",
                                     "edit_source_provenance": lineage})
                result = self.evaluated(env)
                self.assertEqual(result["decision"], "rejected")
                self.assertIn("ai_edit_source_not_natural", result["reason_codes"])

    def test_ambiguous_synthetic_original_does_not_qualify_for_edit_exception(self):
        env = self.envelope({"caption": "A tree", "image_origin": "ai_edited",
                             "edit_source_origin": "natural", "original_image_origin": "synthetic"})
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("ai_edit_source_unknown", result["reason_codes"])

    def test_natural_only_does_not_apply_edit_exception(self):
        env = self.envelope({"caption": "A tree", "image_origin": "ai_edited", "edit_source_origin": "natural"})
        self.assertEqual(self.evaluated(env, rules={"policy": "natural_only"})["decision"], "rejected")

    def test_invalid_edit_exception_setting_refused(self):
        with self.assertRaises(ValueError):
            self.evaluated(rules={"allow_ai_edits_from_natural_sources": "true"})

    def test_single_ai_image_rejects_whole_interleaved_document(self):
        env = self.envelope({"format": "interleaved", "segments": [
            {"kind": "text", "text": "First"}, {"kind": "image", "image_index": 0},
            {"kind": "text", "text": "Second"}, {"kind": "image", "image_index": 1}]}, images=2)
        env["image_refs"][1]["image_origin"] = "ai_generated"
        result = self.evaluated(env)
        self.assertEqual(result["category"], "interleaved")
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(len(result["segments"]), 4)

    def test_negative_source_cannot_be_overridden_by_perimage_natural(self):
        env = self.envelope()
        env["image_refs"][0]["image_origin"] = "natural"
        self.assertEqual(self.evaluated(env, origin="ai_generated")["decision"], "rejected")

    def test_false_ai_flag_does_not_prove_natural(self):
        env = self.envelope({"caption": "An actual tree", "is_ai_generated": False})
        self.assertEqual(self.evaluated(env, origin="unknown")["decision"], "quarantine")

    def test_record_ai_flag_rejects_all_images(self):
        env = self.envelope({"caption": "Tree", "is_ai_generated": True})
        self.assertEqual(self.evaluated(env)["decision"], "rejected")

    def test_native_caption_direction_preserved(self):
        for direction, parts in (("single_t2i", ["text", "image"]), ("single_i2t", ["image", "text"])):
            segments = [{"kind": p, "text": "Tree"} if p == "text" else {"kind": p, "image_index": 0} for p in parts]
            result = self.evaluated(self.envelope({"format": "caption", "segments": segments}))
            self.assertEqual(result["category"], direction)
            self.assertEqual([s["kind"] for s in result["segments"]], parts)

    def test_dcvlm_caption_preserves_native_order_without_t2i_training_view(self):
        for direction, parts in (("single_t2i", ["text", "image"]), ("single_i2t", ["image", "text"])):
            with self.subTest(direction=direction):
                segments = [{"kind": part, "text": "Tree"} if part == "text" else
                            {"kind": part, "image_index": 0} for part in parts]
                env = self.envelope({"format": "caption", "segments": segments})
                env["collection"] = "dcvlm"
                result = self.evaluated(env)
                self.assertEqual(result["category"], direction)
                self.assertEqual(result["segments"], normalize(env)["segments"])
                self.assertEqual(result["eligible_views"], ["single_i2t"] if direction == "single_i2t" else [])

    def test_dcvlm_other_native_types_are_unchanged(self):
        cases = [
            ({"conversations": [{"from": "human", "value": "<image>Describe."},
                                {"from": "gpt", "value": "Tree."},
                                {"from": "human", "value": "Color?"},
                                {"from": "gpt", "value": "Green."}]}, 1, "multi_i2t"),
            ({"format": "interleaved", "segments": [{"kind": "text", "text": "A tree"},
                {"kind": "image", "image_index": 0}, {"kind": "text", "text": "in context"}]}, 1, "interleaved"),
            ({"text": "Plain text."}, 0, "pure_text"),
        ]
        for raw, count, category in cases:
            with self.subTest(category=category):
                env = self.envelope(raw, images=count)
                env["collection"] = "dcvlm"
                result = self.evaluated(env)
                self.assertEqual(result["category"], category)
                self.assertEqual(result["eligible_views"], [category])

    def test_qa_never_reversed_into_t2i(self):
        env = self.envelope({"conversations": [{"from": "human", "value": "<image>What color?"},
                                                {"from": "gpt", "value": "Green."}]})
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["eligible_views"], ["single_i2t"])
        self.assertEqual(result["completed_rounds"], 1)

    def test_multi_image_does_not_mean_multi_turn(self):
        env = self.envelope({"messages": [{"role": "user", "content": "<image><image>Compare."},
                                          {"role": "assistant", "content": "Both show trees."}]}, images=2)
        result = self.evaluated(env)
        self.assertEqual(result["category"], "single_i2t")
        self.assertEqual(result["completed_rounds"], 1)

    def test_multi_round_i2t_count_actual_pairs(self):
        messages = [{"from": "human", "value": "<image>Describe."}, {"from": "gpt", "value": "Tree."},
                    {"from": "human", "value": "Color?"}, {"from": "gpt", "value": "Green."}]
        result = self.evaluated(self.envelope({"conversations": messages}))
        self.assertEqual(result["category"], "multi_i2t")
        self.assertEqual(result["completed_rounds"], 2)

    def test_multi_round_t2i_assistant_images(self):
        messages = [{"role": "user", "content": "A tree"}, {"role": "assistant", "content": "<image>"},
                    {"role": "user", "content": "An ocean"}, {"role": "assistant", "content": "<image>"}]
        env = self.envelope({"messages": messages}, images=2)
        env["collection"] = "genund"
        result = self.evaluated(env)
        self.assertEqual(result["category"], "multi_t2i")
        self.assertEqual(result["completed_rounds"], 2)
        self.assertEqual(result["eligible_views"], ["multi_t2i"])

    def test_image_edit_conversation_is_mixed(self):
        messages = [{"role": "user", "content": "<image>Make it red"},
                    {"role": "assistant", "content": "<image>"}]
        result = self.evaluated(self.envelope({"messages": messages}, images=2))
        self.assertEqual(result["category"], "mixed_multimodal")

    def test_pure_ai_text_allowed_even_source_ai_image_prior(self):
        result = self.evaluated(self.envelope({"text": "An AI-authored paragraph"}, images=0), origin="ai_generated", caption="ai")
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["category"], "pure_text")
        self.assertTrue(result["provenance"]["text"]["ai_generated"])

    def test_assistant_only_pure_text_remains_valid(self):
        result = self.evaluated(self.envelope({"messages": [{"role": "assistant", "content": "Hello"}]}, images=0))
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["category"], "pure_text")

    def test_dcvlm_empty_image_list_is_valid_pure_text(self):
        env = self.envelope({"format": "text_conversation", "images": [], "segments": [
            {"kind": "text", "text": "Hello", "role": "assistant", "turn": 0}]}, images=0)
        self.assertEqual(self.evaluated(env)["decision"], "accepted")

    def test_url_and_missing_local_image_quarantine(self):
        for ref in ({"kind": "url", "path": "https://example.com/photo.jpg", "exists": True},
                    {"kind": "file", "path": "/missing/photo.jpg", "exists": False}):
            env = self.envelope()
            env["image_refs"] = [ref]
            self.assertIn("image_reference_unresolved", self.evaluated(env)["reason_codes"])

    def test_url_only_caption_never_becomes_pure_text(self):
        env = self.envelope({"caption": "Tree", "url": "https://example.org/photo.jpg"}, images=0)
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "quarantine")
        self.assertEqual(result["category"], "single_i2t")
        self.assertIn("image_reference_unresolved", result["reason_codes"])

    def test_unpaired_caption_prompt_and_recaption_quarantine(self):
        for key in ("caption", "prompt", "recaption", "re_caption"):
            with self.subTest(key=key):
                result = self.evaluated(self.envelope({key: "A tree"}, images=0))
                self.assertEqual(result["decision"], "quarantine")
                self.assertEqual(result["category"], "unclassified")
                self.assertIn("image_annotation_without_image", result["reason_codes"])

    def test_explicit_pure_text_task_allows_prompt_without_image(self):
        result = self.evaluated(self.envelope({"prompt": "Write a poem", "task": "pure_text"}, images=0))
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["category"], "pure_text")

    def test_url_text_is_not_an_image_caption(self):
        for key in ("caption", "text", "prompt"):
            result = self.evaluated(self.envelope({key: " https://www.flickr.com/photos/user/12345 "}))
            self.assertEqual(result["decision"], "quarantine")
            self.assertIn("text_is_only_url", result["reason_codes"])
        native = {"format": "caption", "segments": [
            {"kind": "image", "image_index": 0}, {"kind": "text", "text": "https://example.org/123"}]}
        self.assertIn("text_is_only_url", self.evaluated(self.envelope(native))["reason_codes"])

    def test_caption_with_a_link_is_not_mistaken_for_url_only(self):
        result = self.evaluated(self.envelope({"caption": "A tree photographed at https://example.org/123"}))
        self.assertEqual(result["decision"], "accepted")

    def test_ambiguous_multiimage_attachment_quarantines(self):
        env = self.envelope({"messages": [{"role": "user", "content": "Compare"},
                                          {"role": "assistant", "content": "Similar"}]}, images=2)
        result = self.evaluated(env)
        self.assertEqual(result["decision"], "quarantine")
        self.assertIn("ambiguous_image_placement", result["reason_codes"])

    def test_bad_index_and_trailing_user_fail_closed(self):
        for raw in ({"format": "caption", "segments": [{"kind": "image", "image_index": 8}, {"kind": "text", "text": "Tree"}]},
                    {"messages": [{"role": "user", "content": "<image>Tree?"}]}):
            self.assertEqual(self.evaluated(self.envelope(raw))["decision"], "quarantine")

    def test_stable_identity_uses_source_and_locator_not_caption_only(self):
        a = normalize(self.envelope())
        b = normalize(self.envelope())
        cenv = self.envelope()
        cenv["source"] = "other"
        self.assertEqual(a["sample_id"], b["sample_id"])
        self.assertNotEqual(a["sample_id"], normalize(cenv)["sample_id"])

    def test_existing_hash_only_used_as_image_dedup_key(self):
        env = self.envelope()
        env["image_refs"][0]["sha256"] = "a" * 64
        self.assertEqual(self.evaluated(env)["dedup_keys"]["image_sha256"], ["a" * 64])

    def test_source_specificity_and_casefold(self):
        rules = {"sources": [
            {"id": "base", "match": {"collection": "test", "source": "photos"}, "image_origin": "mixed"},
            {"id": "natural", "match": {"collection": "test", "source": "photos", "path_glob": "*/REAL/*"}, "image_origin": "natural"}]}
        cfg = resolve_source_config("test", "photos", {"path": "/data/real/1.jpg"}, rules)
        self.assertEqual(cfg["id"], "natural")
        env = self.envelope()
        env["image_refs"][0]["path"] = "/data/real/1.jpg"
        self.assertEqual(self.evaluated(env, origin="mixed", rules=rules)["decision"], "accepted")

    def test_one_natural_subpath_does_not_bless_other_image(self):
        rules = {"sources": [
            {"match": {"collection": "test", "source": "photos"}, "image_origin": "unknown"},
            {"match": {"collection": "test", "source": "photos", "path_glob": "*/real/*"}, "image_origin": "natural"}]}
        env = self.envelope({"format": "interleaved", "segments": [
            {"kind": "text", "text": "A"}, {"kind": "image", "image_index": 0},
            {"kind": "text", "text": "B"}, {"kind": "image", "image_index": 1}]}, images=2)
        env["image_refs"][0]["path"] = "/data/real/1.jpg"
        env["image_refs"][1]["path"] = "/data/unknown/2.jpg"
        result = self.evaluated(env, origin="unknown", rules=rules)
        self.assertEqual([x["origin"] for x in result["provenance"]["images"]], ["natural", "unknown"])
        self.assertEqual(result["decision"], "quarantine")

    def test_specific_natural_does_not_override_broad_known_generated(self):
        rules = {"sources": [
            {"match": {"collection": "test", "source": "photos"}, "image_origin": "ai_generated"},
            {"match": {"collection": "test", "source": "photos", "path_glob": "*/data/*"}, "image_origin": "natural"}]}
        self.assertEqual(self.evaluated(rules=rules)["decision"], "rejected")

    def test_evaluation_set_reports_both_reasons(self):
        rules = {"sources": [{"match": {"collection": "test", "source": "photos"},
                               "image_origin": "ai_generated", "usage": "evaluation"}]}
        result = self.evaluated(rules=rules)
        self.assertEqual(result["decision"], "rejected")
        self.assertIn("evaluation_dataset", result["reason_codes"])
        self.assertIn("image_ai_generated", result["reason_codes"])

    def test_evaluation_exclusion_applies_to_pure_text(self):
        rules = {"sources": [{"match": {"collection": "test", "source": "photos"},
                               "image_origin": "ai_generated", "usage": "evaluation"}]}
        result = self.evaluated(self.envelope({"text": "A test prompt"}, images=0), rules=rules)
        self.assertEqual(result["decision"], "rejected")
        self.assertIn("evaluation_dataset", result["reason_codes"])
        self.assertNotIn("image_ai_generated", result["reason_codes"])

    def test_malformed_role_and_path_quarantine_without_crash(self):
        for raw in ({"segments": [{"kind": "text", "text": "x", "role": {}}]},
                    {"messages": [{"role": {}, "content": "x"}]}):
            result = self.evaluated(self.envelope(raw, images=0))
            self.assertEqual(result["decision"], "quarantine")
        env = self.envelope()
        env["image_refs"] = [{"path": 3}]
        self.assertEqual(self.evaluated(env)["decision"], "quarantine")

    def test_adapter_resolution_preserves_raw_perimage_ai_flag(self):
        env = self.envelope({"image": {"path": "/data/photo.jpg", "is_ai_generated": True}, "caption": "Tree"})
        self.assertEqual(self.evaluated(env)["decision"], "rejected")

    def test_unsafe_rules_refused(self):
        for rules in ({"policy": "natural_only", "rejected_image_origins": ["ai_generated"]},
                      {"defaults": {"unknown_image": "accept"}}):
            with self.assertRaises(ValueError):
                self.evaluated(rules=rules)

    def test_no_input_mutation(self):
        env = self.envelope()
        before = copy.deepcopy(env)
        self.evaluated(env)
        self.assertEqual(env, before)


if __name__ == "__main__":
    unittest.main()
