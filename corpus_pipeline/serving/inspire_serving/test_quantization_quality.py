import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image

from . import quantization_quality as q


def image_fixture(root):
    stream = io.BytesIO()
    Image.new('RGB', (128, 128), 'blue').save(stream, format='PNG')
    blob = stream.getvalue()
    path = root / 'fixture.tar'
    path.write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    ref = {'path': str(path), 'offset': 0, 'length': len(blob), 'sha256': sha}
    return {'asset_id': sha, 'sha256': sha, 'original_ref': ref}


def row(model, parsed, *, precision='BF16', context=None, assets=(), metadata=()):
    return {'status': 'ok', 'real_model_inference': True, 'parsed': parsed,
            'response_model': model, 'response_boot_id': 'boot', 'qualified_boots_at_request': ['boot'],
            'image_inputs': list(metadata), 'identity': {
                'model': model, 'revision': 'weights', 'processor_revision': 'processor',
                'runtime_revision': 'runtime', 'precision': precision,
                'context_sha256': q.digest(context), 'assets': [a['sha256'] for a in assets],
                'request_overrides': {}, 'max_tokens': 5000, 'stage': 'typed_annotation.v1'}}


def case_fixture(root, source='coco_caption', domain='photo', category='single_i2t', fmt='caption'):
    asset = image_fixture(root)
    segments = [{'kind': 'image', 'image_index': 0, 'role': 'user'},
                {'kind': 'text', 'role': 'user', 'text': 'Describe the image.'},
                {'kind': 'text', 'role': 'assistant', 'text': 'A blue square.'}]
    if category == 'multi_i2t':
        segments.extend([{'kind': 'text', 'role': 'user', 'text': 'What color?'},
                         {'kind': 'text', 'role': 'assistant', 'text': 'Blue.'}])
    raw = {'segments': segments, 'format': fmt}
    record = {'original_sample_id': source, 'source': source, 'domain': domain,
              'normalized': {'category': category}, 'raw': raw,
              'source_record_sha256': q.digest(raw), 'source_image_refs': [asset['original_ref']]}
    annotation = {'native_task': category, 'semantic_task': q.semantic_contract(record),
                  'image_captions': [], 'assistant_revisions': [], 'concerns': [], 'layout_candidates': []}
    if annotation['semantic_task'] == 'caption':
        annotation['image_captions'] = [{'image_index': 0, 'short': 'A blue square.',
                                       'medium': 'A blue square.', 'long': 'A blue square.'}]
    context = q.model_context(record)
    metadata = [q.image_payload(asset)[1]]
    generated = row('Qwen-BF16', annotation, context=context, assets=[asset], metadata=metadata)
    generated['identity']['prompt_sha256'] = hashlib.sha256(q.prompts.ANNOTATE.encode()).hexdigest()
    verified = row('Gemma4-31B-IT', {'verdict': 'pass', 'issues': [], 'reason': 'Visible square.'},
                   context=q.verification_context(record, context, annotation))
    verified['identity']['prompt_sha256'] = hashlib.sha256(q.prompts.VERIFY.encode()).hexdigest()
    return {'sample_id': source, 'strata': q.strata_for(record), 'record': record,
            'context': context, 'assets': [asset], 'generation_max_tokens': 5000,
            'baseline_annotation': generated, 'baseline_quality': verified}


def baseline_fixture(root):
    cases = [case_fixture(root), case_fixture(root, 'plotqa', 'chart', fmt='conversation'),
             case_fixture(root, 'synthdog', 'document', fmt='conversation'),
             case_fixture(root, 'clevr_math', 'rendered', fmt='conversation'),
             case_fixture(root, 'visdial', 'photo', 'multi_i2t', 'conversation')]
    value = {'schema_version': q.BASELINE_SCHEMA, 'required_strata': list(q.STRATA),
             'coverage': q.coverage(cases), 'cases': cases, 'production_code_sha256': q.pipeline_code(),
             'annotation_prompt': q.prompts.ANNOTATE, 'verification_prompt': q.prompts.VERIFY}
    value['content_sha256'] = q.digest(value)
    return value


def serving_fixture(root):
    now = time.time()
    proof = {'model': 'Qwen-W8A8', 'revision': 'weights', 'base_url': 'https://candidate.invalid/v1',
             'expected_boots': ['boot'], 'passed': True, 'authentication_passed': True,
             'requests': [{'passed': True, 'boot_id': 'boot', 'expected': {'digits': '123'},
                           'actual': {'digits': '123'}} for _ in range(13)], 'finished_at': now - 1,
             'tiers': [{'concurrency': n} for n in (1, 4, 8)]}
    proof_path = root / 'probe.json'
    proof_path.write_text(json.dumps(proof))
    state = {'role': 'generator', 'tier': 'low', 'precision': 'W8A8_DYNAMIC',
             'platform_status': 'RUNNING', 'observation_error': None,
             'last_observed_at': now, 'heartbeat_at': now, 'probe_status': 'PASSED',
             'model': 'Qwen-W8A8', 'revision': 'weights', 'processor_revision': 'processor',
             'runtime_revision': 'runtime', 'incarnation': 'incarnation', 'expected_boots': ['boot'],
             'endpoint': 'https://candidate.invalid', 'service_id': 'owned', 'request_overrides': {},
             'max_images': 8, 'qualification_receipt': str(proof_path)}
    path = root / 'candidate.json'
    path.write_text(json.dumps(state))
    return path, state, proof


class BaselineTests(unittest.TestCase):
    def test_missing_domain_cannot_be_called_complete(self):
        with self.assertRaisesRegex(ValueError, 'missing required strata'):
            q.coverage([{'strata': ['photo_caption', 'chart_qa']}])

    def test_ui_document_does_not_substitute_for_ocr(self):
        with tempfile.TemporaryDirectory() as temp:
            case = case_fixture(Path(temp), 'websight', 'document', fmt='conversation')
            self.assertNotIn('document_ocr', case['strata'])

    def test_freeze_refuses_unfinished_real_production(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'COMPLETE.json').write_text('{"canary_completed": false}')
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                q.freeze_baseline(root, root / 'manifest.json', root / 'READY.json')
            self.assertFalse((root / 'READY.json').exists())

    def test_valid_frozen_baseline_and_all_original_images_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = baseline_fixture(root)
            q.validate_baseline(baseline)
            (root / 'fixture.tar').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'SHA256/length'):
                q.validate_baseline(baseline)

    def test_baseline_content_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = baseline_fixture(Path(temp))
            baseline['cases'][0]['context']['image_count'] = 0
            with self.assertRaisesRegex(ValueError, 'content changed'):
                q.validate_baseline(baseline)

    def test_uncertain_or_major_error_never_pass(self):
        self.assertFalse(q.strict_quality({'verdict': 'uncertain', 'issues': [], 'reason': 'Unreadable.'}))
        with self.assertRaisesRegex(ValueError, 'major error cannot pass'):
            q.strict_quality({'verdict': 'pass', 'issues': [{'field': 'text', 'severity': 'major', 'reason': 'Wrong OCR.'}], 'reason': 'Bad.'})

    def test_mock_or_unqualified_evidence_is_not_a_real_baseline(self):
        value = row('Qwen-BF16', {})
        value['real_model_inference'] = False
        with self.assertRaisesRegex(ValueError, 'real successful'):
            q.real_row(value, precision='BF16')
        value['real_model_inference'] = True
        value['response_boot_id'] = 'other-boot'
        with self.assertRaisesRegex(ValueError, 'Unqualified response boot'):
            q.real_row(value)

    def test_probe_requires_current_boot_all_requests_and_authentication(self):
        with tempfile.TemporaryDirectory() as temp:
            _, state, proof = serving_fixture(Path(temp))
            q.validate_probe(state, proof)
            for mutation in ({'authentication_passed': False}, {'requests': proof['requests'][:5]}, {'expected_boots': ['old']}):
                with self.assertRaises(ValueError):
                    q.validate_probe(state, {**proof, **mutation})
            with self.assertRaisesRegex(ValueError, 'Stale candidate'):
                q.validate_probe({**state, 'last_observed_at': time.time() - 60}, proof)

    def test_freeze_excludes_final_human_quarantine_and_keeps_five_strata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = baseline_fixture(root)['cases']
            excluded = copy.deepcopy(cases[0])
            excluded['sample_id'] = excluded['record']['original_sample_id'] = 'excluded-photo'
            cases.insert(0, excluded)
            by_id = {c['sample_id']: c for c in cases}
            manifest = root / 'manifest.json'
            fingerprint = {'fixture': 'snapshot-sha'}
            manifest.write_text(json.dumps({'snapshot_root': str(root), 'snapshot_fingerprint': fingerprint,
                                           'records': [{'fixture_id': c['sample_id']} for c in cases]}))
            (root / 'run.json').write_text(json.dumps({'identity': {'manifest_sha256': q.file_sha(manifest)}}))
            (root / 'summary.json').write_text('{"attempt_id": "final-attempt"}')
            (root / 'COMPLETE.json').write_text(json.dumps({'canary_completed': True, 'summary_sha256': q.file_sha(root / 'summary.json')}))
            (root / 'asset_index.jsonl').write_text(json.dumps(cases[0]['assets'][0]) + '\n')
            (root / 'record_results').mkdir()
            for case in cases:
                (root / 'record_results' / (case['sample_id'] + '.json')).write_text(json.dumps({
                    'attempt_id': 'final-attempt', 'original_sample_id': case['sample_id'],
                    'status': 'ok', 'annotation_quality_pass': True,
                    'annotation': case['baseline_annotation'], 'quality': case['baseline_quality']}))
            with patch.object(q, 'snapshot', return_value={'root': str(root), 'fingerprint': fingerprint}), \
                 patch.object(q, 'load_record', side_effect=lambda spec, *_: by_id[spec['fixture_id']]['record']):
                baseline = q.freeze_baseline(root, manifest, root / 'READY.json', excluded_sample_ids=['excluded-photo'])
            self.assertEqual(set(baseline['coverage']), set(q.STRATA))
            self.assertEqual(len(baseline['cases']), 5)
            self.assertNotIn('excluded-photo', {c['sample_id'] for c in baseline['cases']})
            self.assertEqual(baseline['excluded_sample_ids'], ['excluded-photo'])
            q.validate_baseline(baseline)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_registry_does_not_publish_and_fences_changed_boot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, state, _ = serving_fixture(root)
            verifier = {'role': 'verifier', 'model': 'Gemma4-31B-IT', 'ready': True,
                        'backend': 'inspire_serving', 'expected_boots': ['gemma-boot'],
                        'ready_expires_at': time.time() + 30, 'protocol': 'openai'}
            registry = root / 'production.json'
            registry.write_text(json.dumps({'endpoints': [verifier]}))
            before = registry.read_bytes()
            client = q.QualificationClient(path, registry, root / 'private')
            try:
                endpoint, _ = client.endpoint('generator')
                self.assertTrue(endpoint['qualification_only'])
                self.assertTrue(q.load(client.private_path)['qualification_only'])
                self.assertEqual(registry.read_bytes(), before)
                state['expected_boots'] = ['changed']
                path.write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, 'changed during qualification'):
                    client.endpoint('generator')
            finally:
                await client.close()

    async def test_same_prompt_candidate_with_uncertain_review_fails_without_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            case = case_fixture(Path(temp))
            baseline = {'annotation_prompt': q.prompts.ANNOTATE, 'verification_prompt': q.prompts.VERIFY}
            calls = []
            class FakeClient:
                def endpoint(self, role):
                    return {'request_overrides': {}}, 1
                async def request(self, role, stage, context, assets, prompt, validator, **kwargs):
                    calls.append((role, prompt, context))
                    if role == 'generator':
                        value = copy.deepcopy(case['baseline_annotation'])
                        value['identity']['precision'] = 'W8A8_DYNAMIC'
                    else:
                        value = row('Gemma4-31B-IT', {'verdict': 'uncertain', 'issues': [], 'reason': 'Not readable.'})
                    validator(value['parsed'])
                    return value
            result = await q.evaluate_case(case, baseline, FakeClient())
            self.assertFalse(result['passed'])
            self.assertEqual(result['status'], 'quality_failed')
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1:], (q.prompts.ANNOTATE, case['context']))
            self.assertNotIn('baseline_annotation', calls[1][2])

    async def test_cannot_write_over_production_registry(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = Path(temp) / 'registry.json'
            registry.write_text('{"sentinel": true}')
            with self.assertRaisesRegex(ValueError, 'production registry'):
                await q.qualify(SimpleNamespace(output=str(registry), registry=str(registry)))
            self.assertEqual(registry.read_text(), '{"sentinel": true}')

    async def test_one_uncertain_case_blocks_whole_gate_and_binds_baseline_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = baseline_fixture(root)
            baseline_path = root / 'baseline.json'
            baseline_path.write_text(json.dumps(baseline))
            candidate_path, state, _ = serving_fixture(root)
            key = root / 'key'
            key.write_text('test-fixture-not-an-api-credential')
            state['api_key_file'] = str(key)
            candidate_path.write_text(json.dumps(state))
            registry = root / 'production.json'
            registry.write_text('{"endpoints": []}')
            cases_by_context = {q.digest(c['context']): c for c in baseline['cases']}
            class FakeClient:
                def __init__(self, candidate, registry, private, **kwargs):
                    Path(private).mkdir()
                    self.fixed_verifier = {'model': 'Gemma4-31B-IT', 'expected_boots': ['boot']}
                def endpoint(self, role):
                    return {'request_overrides': {}}, 1
                async def request(self, role, stage, context, assets, prompt, validator, **kwargs):
                    if role == 'generator':
                        result = copy.deepcopy(cases_by_context[q.digest(context)]['baseline_annotation'])
                        result['identity']['precision'] = 'W8A8_DYNAMIC'
                    else:
                        is_caption = context['candidate_annotation']['semantic_task'] == 'caption'
                        result = row('Gemma4-31B-IT', {'verdict': 'uncertain' if is_caption else 'pass',
                                                      'issues': [], 'reason': 'Fixture review.'})
                    validator(result['parsed'])
                    return result
                async def close(self):
                    pass
            args = SimpleNamespace(output=str(root / 'quality.json'), registry=str(registry),
                                   candidate_state=str(candidate_path), baseline_manifest=str(baseline_path),
                                   concurrency=2, request_timeout=5, max_duration=5)
            with patch.object(q, 'QualificationClient', FakeClient), patch.dict(q.os.environ):
                receipt = await q.qualify(args)
            self.assertFalse(receipt['passed'])
            self.assertEqual(receipt['status'], 'quality_failed')
            self.assertFalse(receipt['retryable'])
            self.assertTrue(receipt['all_baseline_cases_completed'])
            self.assertEqual(receipt['baseline_sha256'], hashlib.sha256(baseline_path.read_bytes()).hexdigest())
            self.assertEqual(receipt['expected_boots'], ['boot'])
            self.assertFalse(receipt['published_training'])
            self.assertEqual(registry.read_text(), '{"endpoints": []}')


if __name__ == '__main__':
    unittest.main()
