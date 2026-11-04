import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import pyarrow.parquet as pq
from PIL import Image

from corpus_pipeline.operations import assistant_screening_adoption as adoption
from corpus_pipeline.unified import screening, release, workflow
from corpus_pipeline.unified.test_screening import recon, qrow, write_json
from corpus_pipeline.dedup import decisions as legacy


def parsed(assessment='likely_non_ai', confidence='medium', supports='non_ai'):
    return {'image_form': 'illustration', 'assessment': assessment, 'confidence': confidence,
            'image_description': 'A visible diagram.', 'needs_provenance': False,
            'edit_lineage': 'not_established_from_pixels', 'reason': 'A diagram is a plausible explanation; origin remains unknown.',
            'counterevidence': ['Pixels cannot certify production history.'],
            'visible_evidence': [{'location': 'upper area', 'observation': 'Repeated line symbols.',
                                  'supports': supports, 'alternative_explanation': 'Diagram layout.'}]}


def review(aid='0' * 63 + '2'):
    return {'asset_id': aid, 'completed': True, 'viewed_at_utc': '2026-10-29T02:10:00Z',
            'observed_image': {'path': '/verified/preview.png', 'sha256': 'b' * 64}, 'parsed': parsed()}


class AssistantEvidenceTests(unittest.TestCase):
    def package(self):
        row = review(); aid = row['asset_id']
        prompt = {'path': '/frozen/prompt.txt', 'sha256': 'c' * 64}
        image = {'asset_id': aid, 'original_sha256': aid, 'original_ref': {'sha256': aid},
                 'source_archive_and_member_sha256_verified': True, 'crop': False,
                 'preview_path': '/verified/preview.png', 'preview_sha256': 'b' * 64,
                 'full_resolution_path': '/verified/full.png', 'full_resolution_sha256': 'd' * 64}
        manifest = {'schema': 'corpus.assistant_visual_review_inputs.v1', 'frozen_input_sha256': 'e' * 64,
                    'original_prompt_file_sha256': prompt['sha256'], 'images': [image], 'count': 1}
        package = {'schema': 'corpus.assistant_visual_review.v1',
                   'reviewer': {'authority': adoption.AUTHORITY, 'qwen_inference_claimed': False,
                                'human_review_claimed': False, 'reviewer_type': 'assistant_direct_visual_review'},
                   'user_instruction': adoption.USER_INSTRUCTION, 'prompt': prompt,
                   'visual_input_manifest': {'path': '/manifest.json', 'sha256': 'f' * 64}, 'reviews': [row]}
        return package, manifest, {aid}, prompt, 'e' * 64

    def test_exact_evidence_keeps_assistant_authority(self):
        result = adoption.validate_review_package(*self.package())
        row, evidence = result[0]
        self.assertEqual(row['decision_authority'], adoption.AUTHORITY)
        self.assertEqual(row['screen_action'], 'retain_annotated')
        self.assertFalse(row['quality_reviewed'])
        self.assertEqual(row['evidence_sha256'], adoption.digest(evidence))

    def test_missing_duplicate_extra_or_unviewed_assets_refuse_completion(self):
        for mutation in ('missing', 'duplicate', 'extra', 'unviewed'):
            args = list(self.package()); package = args[0]
            if mutation == 'missing': package['reviews'] = []
            if mutation == 'duplicate': package['reviews'].append(copy.deepcopy(package['reviews'][0]))
            if mutation == 'extra': package['reviews'][0]['asset_id'] = '9' * 64
            if mutation == 'unviewed': package['reviews'][0]['completed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                adoption.validate_review_package(*args)

    def test_authority_prompt_original_and_observed_tampering_rejected(self):
        for mutation in ('authority', 'human', 'prompt', 'original', 'observed'):
            args = list(self.package()); package, manifest = args[:2]
            if mutation == 'authority': package['reviewer']['authority'] = 'qwen_visual_screening'
            if mutation == 'human': package['reviewer']['human_review_claimed'] = True
            if mutation == 'prompt': package['prompt'] = {'path': '/other', 'sha256': 'a' * 64}
            if mutation == 'original': manifest['images'][0]['original_sha256'] = 'a' * 64
            if mutation == 'observed': package['reviews'][0]['observed_image']['sha256'] = 'a' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                adoption.validate_review_package(*args)

    def test_uncertain_and_unproven_edit_remain_pending(self):
        for assessment in ('uncertain', 'possible_ai_edit'):
            row = review(); row['parsed'] = parsed(assessment, 'low', 'ambiguous')
            row['parsed']['needs_provenance'] = True
            result = adoption.normalize_assistant(row, {'actual_review': row})
            self.assertEqual(result['screen_action'], 'pending')

    def test_ai_rejection_needs_two_independent_visible_cues(self):
        row = review(); row['parsed'] = parsed('likely_ai_generated', 'high', 'ai')
        self.assertEqual(adoption.normalize_assistant(row, row)['screen_action'], 'pending')
        row['parsed']['visible_evidence'].append(copy.deepcopy(row['parsed']['visible_evidence'][0]))
        self.assertEqual(adoption.normalize_assistant(row, row)['screen_action'], 'pending')
        row['parsed']['visible_evidence'][1].update(location='lower area', observation='A separate structural anomaly.')
        self.assertEqual(adoption.normalize_assistant(row, row)['screen_action'], 'exclude')

    def test_schema_errors_and_unsupported_certainty_fail(self):
        row = review(); row['parsed']['confidence'] = 'high'
        with self.assertRaises(ValueError): adoption.normalize_assistant(row, row)
        row = review(); row['parsed']['source_certified'] = True
        with self.assertRaises(ValueError): adoption.normalize_assistant(row, row)

    def test_qwen_success_cannot_be_inferred_from_done_alone(self):
        row = qrow(1); request = {'request_state': 'deferred', 'receipt_path': None,
            'receipt_sha': None, 'asset_state': 'done', 'input_sha': '1'}
        with self.assertRaises(ValueError): adoption.normalized_qwen(row, request)
        request.update(request_state='ok', receipt_path='original.json', receipt_sha='a' * 64)
        result = adoption.normalized_qwen(row, request)
        self.assertEqual(result, legacy.normalize_qwen(row))
        self.assertEqual(result['decision_authority'], 'qwen_visual_screening')
        self.assertEqual(result['evidence_sha256'], hashlib.sha256(row['decision_json'].encode()).hexdigest())

    def test_independent_qwen_audit_cannot_be_partial_or_rebound(self):
        sources = {name: {'path': '/' + name, 'sha256': 'a' * 64}
                   for name in ('api_input', 'api_complete', 'decisions')}
        report = {'complete': True, 'error_count': 0, 'checked_success_receipts': 3,
                  'source_bindings': copy.deepcopy(sources)}
        adoption.validate_qwen_verification(report, 3, sources)
        with self.assertRaises(ValueError): adoption.validate_qwen_verification(report, 4, sources)
        report['source_bindings']['decisions']['sha256'] = 'b' * 64
        with self.assertRaises(ValueError): adoption.validate_qwen_verification(report, 3, sources)

    def test_observed_pixels_reproduce_original_not_just_manifest_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'view.png'
            original = Image.new('RGB', (8, 8), 'red'); stream = io.BytesIO(); original.save(stream, 'PNG')
            raw = stream.getvalue(); original.save(path)
            rgba = original.convert('RGBA')
            decoded = hashlib.sha256(b'RGBA' + struct.pack('>II', *rgba.size) + rgba.tobytes()).hexdigest()
            image = {'asset_id': hashlib.sha256(raw).hexdigest(), 'preview_path': str(path),
                     'bound_decoded_rgba_geometry_sha256': decoded}
            observed = adoption.bound(path)
            adoption.verify_observed_pixels(raw, image, observed, decoded)
            Image.new('RGB', (8, 8), 'blue').save(path)
            observed = adoption.bound(path)  # Even a matching new file hash cannot hide changed pixels.
            with self.assertRaisesRegex(ValueError, 'Viewed pixels'):
                adoption.verify_observed_pixels(raw, image, observed, decoded)

    def test_frozen_history_assembly_and_stage_validation_preserve_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); recon(root / 'reconciliation', [1, 2])
            qwen = legacy.normalize_qwen(qrow(1))
            assistant = adoption.normalize_assistant(review(), {'direct_observation': review()})
            qhistory = adoption.write_history(root / 'qhistory', [qwen], {'source': 'original_qwen'})
            ahistory = adoption.write_history(root / 'ahistory', [assistant], {'source': 'assistant'})
            api = root / 'api_review'; api.mkdir()
            write_json(api / 'COMPLETE.json', {'screening_pass_complete': False})
            original_api = (api / 'COMPLETE.json').read_bytes()
            result = screening.build_decisions(root / 'reconciliation', root / 'screening',
                review_root=None, history_decision_manifests=[qhistory, ahistory])
            self.assertTrue(result['complete']); self.assertEqual(result['records'], 2)
            rows = pq.read_table(root / 'screening/screening.parquet').to_pylist()
            self.assertEqual(rows, [qwen, assistant])
            self.assertEqual((api / 'COMPLETE.json').read_bytes(), original_api)
            write_json(root / 'EXECUTION.json', {'frozen': True})
            write_json(root / 'ADOPTION.json', {'effective_authority_changed': True})
            write_json(root / 'receipts/reconcile.json', {'upstream': 'immutable'})
            receipt = release.seal_receipt(root / 'receipts/screening.json', stage='screening',
                collections={'dcvlm': 'a' * 64},
                coverage={'unit': 'assets', 'expected': 2, 'processed': 2, 'missing': 0, 'pending_infra': 0},
                inputs=[root / 'EXECUTION.json', root / 'ADOPTION.json', qhistory, ahistory],
                artifacts=workflow.finalized_files(root / 'screening'),
                dependencies={'reconcile': adoption.sha(root / 'receipts/reconcile.json')})
            self.assertEqual(workflow.validated_receipt(root, 'screening'), receipt)
            (root / 'ADOPTION.json').write_text('{}')
            with self.assertRaises(ValueError): workflow.validated_receipt(root, 'screening')


if __name__ == '__main__':
    unittest.main()
