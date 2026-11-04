"""Strict T2I review admission is opt-in and cannot change the I2T route."""
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from corpus_pipeline.production.task_policy import receipt_policy, run_rules
from .config import checked
from .policy import TASK_POLICY, config, evaluate_image, evaluate_record, policy_hash
from .screening import build_decisions
from .test_policy import observation
from .test_screening import recon, review, qrow, history, write_json
from .test_task_routing_screening import reconciliation

STRICT = {'task_routing': TASK_POLICY, 't2i_uncertain': 'exclude'}
LEGACY_TASK = {'task_routing': TASK_POLICY}


class StrictPolicyTests(unittest.TestCase):
    def test_opt_in_hashes_preserve_both_legacy_defaults(self):
        self.assertEqual(policy_hash(), 'e8ac12b2b48f00e2dd05de9245ebae9b51d25b63211dac0aa4bce87d01b9633b')
        self.assertNotIn('t2i_uncertain', config())
        self.assertNotIn('t2i_uncertain', config(LEGACY_TASK))
        self.assertEqual(policy_hash(LEGACY_TASK), policy_hash({**LEGACY_TASK, 't2i_uncertain': 'retain_annotated'}))
        self.assertNotEqual(policy_hash(STRICT), policy_hash(LEGACY_TASK))
        for invalid in ({'t2i_uncertain': 'exclude'}, {**LEGACY_TASK, 't2i_uncertain': 'guess'},
                        {**LEGACY_TASK, 't2i_uncertain': False}, {**LEGACY_TASK, 't2i_uncertain': []}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config(invalid)

    def test_checked_execution_config_routes_same_switch_to_rules_and_receipts(self):
        base = {'schema': 'corpus.unified.pipeline.v1', 'scope': 'validation', 'output': '/tmp/run',
                'inputs': [{'collection': 'genund', 'adapter': 'genund_caption_tar',
                            'root': '/tmp/data', 'direction': 'i2t'}],
                'policy': {'profile': 'task_aware_v1'}}
        old = checked(base)
        self.assertNotIn('t2i_uncertain', old['policy'])
        self.assertEqual(receipt_policy(old), LEGACY_TASK)
        for mode in ('retain_annotated', 'exclude'):
            cfg = checked({**base, 'policy': {**base['policy'], 't2i_uncertain': mode}})
            actual = receipt_policy(cfg)
            self.assertEqual(actual, STRICT if mode == 'exclude' else LEGACY_TASK)
            rules = run_rules(cfg, {'version': 'legacy'})
            self.assertEqual(rules.get('t2i_uncertain', 'retain_annotated'), mode)
            self.assertEqual(policy_hash(actual), policy_hash(STRICT if mode == 'exclude' else LEGACY_TASK))
        with self.assertRaisesRegex(ValueError, 'Unknown early policy'):
            checked({**base, 'policy': {'t2i_uncertain': 'exclude'}})
        with self.assertRaisesRegex(ValueError, 'task_aware_v1'):
            receipt_policy({'policy': {'t2i_uncertain': 'exclude'}})

    def test_completed_uncertain_is_excluded_only_for_t2i_and_never_claimed_ai(self):
        item = observation(processing_policy=TASK_POLICY, task_route='i2t', image_loss_allowed=False,
            status='i2t_dedup_only', pixel_sha256='b' * 64, detector={'status': 'suspect'},
            review={'status': 'uncertain', 'evidence_ids': ['completed-review']})
        old = evaluate_image([item], LEGACY_TASK)
        self.assertEqual(old['task_decisions']['t2i']['action'], 'retain')
        result = evaluate_image([item], STRICT)
        self.assertEqual(result['action'], 'retain')
        t2i = result['task_decisions']['t2i']
        self.assertEqual(t2i['action'], 'exclude')
        self.assertTrue(t2i['processing_complete'])
        self.assertFalse(t2i['review_pending'])
        self.assertIn('t2i_uncertain_review_excluded', t2i['reasons'])
        self.assertNotIn('confirmed_ai_generated', t2i['reasons'])
        for direction, expected in [('i2t', 'retain'), ('t2i', 'exclude')]:
            record = {'record_id': direction, 'image_ids': ['a' * 64], 'processing_policy': TASK_POLICY,
                      'task_route': direction, 'image_loss_allowed': direction == 't2i'}
            if direction == 'i2t':
                record['reason_codes'] = ['t2i_uncertain_review_excluded']
            self.assertEqual(evaluate_record(record, {'a' * 64: result}, policy_config=STRICT)['action'], expected)

    def test_clear_and_non_ai_review_pass_but_incomplete_review_cannot_use_detector_clear(self):
        clear = {'status': 'clear', 'evidence_ids': ['detector-output']}
        self.assertEqual(evaluate_image([observation(detector=clear)], STRICT)['action'], 'retain')
        natural = {'status': 'confirmed_natural', 'evidence_ids': ['completed-review']}
        self.assertEqual(evaluate_image([observation(detector={'status': 'suspect'}, review=natural)], STRICT)['action'], 'retain')
        for incomplete in ({'status': 'error'}, {'status': 'uncertain'},
                           {'status': 'confirmed_natural'},
                           {'status': 'uncertain', 'evidence_ids': ['x'], 'completed': False}):
            result = evaluate_image([observation(detector=clear, review=incomplete)], STRICT)
            self.assertEqual(result['action'], 'pending')
            self.assertFalse(result['processing_complete'])


class StrictScreeningTests(unittest.TestCase):
    def test_completed_uncertain_pending_and_admit_are_excluded_but_likely_non_ai_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); recon(root / 'recon', [1, 2, 3, 4])
            review(root / 'review', root / 'recon', [qrow(1, 'pending', 'uncertain'),
                qrow(2, 'admit', 'uncertain'), qrow(3, 'admit', 'likely_non_ai'),
                qrow(4, 'admit', 'possible_ai_edit')])
            legacy = build_decisions(root / 'recon', root / 'old', root / 'review', policy_config=LEGACY_TASK)
            strict = build_decisions(root / 'recon', root / 'strict', root / 'review', policy_config=STRICT)
            self.assertEqual(legacy['counts'], {'pending': 1, 'retain_annotated': 3})
            self.assertEqual(strict['counts'], {'exclude': 3, 'retain_annotated': 1})
            self.assertEqual(strict['t2i_uncertain_excluded'], 3)
            rows = {r['asset_id']: r for r in pq.read_table(root / 'strict/screening.parquet').to_pylist()}
            self.assertEqual(rows[f'{1:064x}']['ai_decision'], 'uncertain')
            self.assertEqual(rows[f'{3:064x}']['screen_action'], 'retain_annotated')
            self.assertEqual(strict['policy_sha256'], policy_hash(STRICT))
            self.assertEqual(strict['identity']['policy_sha256'], strict['policy_sha256'])
            self.assertEqual(build_decisions(root / 'recon', root / 'strict', root / 'review', policy_config=STRICT), strict)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                build_decisions(root / 'recon', root / 'strict', root / 'review', policy_config=LEGACY_TASK)

    def test_i2t_only_uncertain_bypasses_but_shared_or_t2i_uncertain_excludes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); reconciliation(root / 'recon')
            review(root / 'review', root / 'recon', [qrow(1, 'pending', 'uncertain'), qrow(2, 'pending', 'uncertain')])
            result = build_decisions(root / 'recon', root / 'out', root / 'review', policy_config=STRICT)
            self.assertTrue(result['complete'])
            self.assertEqual(result['i2t_only_screening_bypassed'], 1)
            self.assertEqual(result['counts'], {'exclude': 1})
            self.assertEqual(pq.read_table(root / 'out/screening.parquet')['asset_id'].to_pylist(), [f'{2:064x}'])

    def test_failure_missing_and_invalid_evidence_remain_pending(self):
        for mode in ('failed', 'missing', 'invalid'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); recon(root / 'recon', [1])
                if mode != 'missing':
                    row = qrow(1, state='failed' if mode == 'failed' else 'done')
                    if mode == 'invalid':
                        value = json.loads(row['decision_json']); value['reviews'] = {}
                        row['decision_json'] = json.dumps(value)
                    review(root / 'review', root / 'recon', [row])
                result = build_decisions(root / 'recon', root / 'out',
                    root / 'review' if mode != 'missing' else None, policy_config=STRICT)
                self.assertFalse(result['complete'])
                self.assertEqual(result['pending_infra'], 1)
                self.assertFalse((root / 'out/COMPLETE.json').exists())

    def test_history_hash_must_match_strict_receipt_and_unknown_history_is_not_clear(self):
        from corpus_pipeline.dedup import decisions as old
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); recon(root / 'recon', [1])
            old_manifest = history(root / 'history', [old.normalize_qwen(qrow(1))])
            prior = json.loads(old_manifest.read_text()); prior['policy_sha256'] = policy_hash(LEGACY_TASK)
            write_json(old_manifest, prior)
            with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                build_decisions(root / 'recon', root / 'out', history_decision_manifests=[old_manifest], policy_config=STRICT)
            prior['policy_sha256'] = policy_hash(STRICT); write_json(old_manifest, prior)
            result = build_decisions(root / 'recon', root / 'out', history_decision_manifests=[old_manifest], policy_config=STRICT)
            self.assertEqual(result['policy_sha256'], policy_hash(STRICT))
            row = old.normalize_qwen(qrow(1)); row['ai_decision'] = None
            unknown = history(root / 'unknown', [row])
            pending = build_decisions(root / 'recon', root / 'incomplete', history_decision_manifests=[unknown], policy_config=STRICT)
            self.assertFalse(pending['complete'])
            self.assertEqual(pending['pending_infra'], 1)


class StrictDagTests(unittest.TestCase):
    def test_actual_dag_screening_receipt_and_publication_bind_strict_hash(self):
        from .test_task_workflow import TaskWorkflowTests
        from .test_cache import FakeWorker
        from . import workflow, release
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, cfg, _ = TaskWorkflowTests().fixture(root)
            cfg['policy']['t2i_uncertain'] = 'exclude'
            path.write_text(json.dumps(cfg))
            result = workflow.run(path, local=True, delegate=FakeWorker())
            self.assertTrue(result['complete'])
            manifest = release.read_manifest(root / 'run')
            self.assertEqual(manifest['identity']['policy']['t2i_uncertain'], 'exclude')
            self.assertEqual(manifest['identity']['policy_sha256'], policy_hash(STRICT))
            receipt = json.loads((root / 'run/receipts/screening.json').read_text())
            self.assertEqual(receipt['policy_sha256'], policy_hash(STRICT))
            rows = [r for b in release.iter_batches(root / 'run') for r in b.to_pylist()]
            self.assertTrue(all(r['task_route'] == 'i2t' and r['image_loss_allowed'] is False for r in rows))


if __name__ == '__main__':
    unittest.main()
