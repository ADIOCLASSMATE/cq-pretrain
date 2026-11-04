"""Fixture-only tests for the additive strict-blur application layer."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from corpus_pipeline.context_review import apply_quality as app
from corpus_pipeline.codex_review.report import canonical, digest


def candidate(number=1, decision='keep'):
    aid = f'{number:064x}'
    return {'item_id': aid, 'review_number': number, 'source_ref': {'sha256': aid, 'path': '/fixture/data.tar',
            'member': str(number) + '.jpg', 'kind': 'tar'},
            'human_review': {'item_id': aid, 'review_number': number, 'decision': decision,
                             'note': '', 'updated_at': '2026-10-25T00:00:00Z'}}


def proposed(c, **changes):
    row = {'schema_version': app.DECISION_SCHEMA, 'item_id': c['item_id'], 'review_number': c['review_number'],
           'source_ref': deepcopy(c['source_ref']), 'source_input_sha256': digest(c), 'blur_status': 'clear',
           'context_action': 'keep', 'image_origin_followup': False, 'reason': '原图边缘清晰，与源caption一致。',
           'authority': app.AUTHORITY}
    row.update(changes)
    return row


def baseline(c, category='admitted'):
    return {'schema_version': 'corpus.human_review.resolved_overlay.v1', 'item_id': c['item_id'],
            'decision': 'admit' if category == 'admitted' else 'reject', 'report_category': category,
            'rejection_reason': 'none' if category == 'admitted' else 'ai_generated' if category == 'ai_excluded'
                                else 'quality_unusable',
            'decision_authority': 'human_user', 'source_certification': False,
            'human_review': deepcopy(c['human_review']), 'human_submission': {'sha256': 'a' * 64},
            'model_screening': {'item_id': c['item_id'], 'source_ref': deepcopy(c['source_ref']),
                                'decision': 'pending', 'result': {'unchanged_nested': ['raw', {'evidence': 'pixels'}]}}}


REFERENCE = {'path': '/fixture/MANIFEST.json', 'sha256': 'b' * 64}


class ValidationTests(unittest.TestCase):
    def test_exact_ids_and_fixed_number(self):
        c = candidate()
        p = proposed(c)
        for rows in ([], [p, p], [dict(p, item_id='f' * 64)], [dict(p, review_number=2)],
                     [dict(p, review_number=True)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                app.validate_decisions(rows, [c])

    def test_source_identity_hash_and_authority(self):
        c = candidate()
        for change in ({'source_ref': dict(c['source_ref'], member='other.jpg')},
                       {'source_input_sha256': '0' * 64}, {'authority': 'human_user'},
                       {'blur_status': 'soft'}, {'context_action': 'admit'}, {'image_origin_followup': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                app.validate_decisions([proposed(c, **change)], [c])

    def test_no_input_mutation(self):
        c = candidate()
        p = proposed(c)
        old_c, old_p = deepcopy(c), deepcopy(p)
        result = app.validate_decisions([p], [c])
        result[c['item_id']]['source_ref']['member'] = 'changed'
        self.assertEqual(c, old_c)
        self.assertEqual(p, old_p)

    def test_tampered_bound_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evidence.json'
            path.write_text('{}')
            binding = app.artifact(path)
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                app.check_binding(binding)

    def test_independent_blur_override_requires_exact_original_evidence(self):
        c = candidate()
        raw = {'item_id': c['item_id'], 'blur_status': 'uncertain'}
        entry = {'item_id': c['item_id'], 'source_input_sha256': digest(c), 'raw_blur_sha256': digest(raw),
                 'blur_status': 'clear', 'reason': '完整原图可见规则布料纹理而不是失焦。',
                 'full_original_image_actually_viewed': True,
                 'evidence': [{'location': '全图', 'observation': '纤维边缘连续清晰。'}]}
        app.validate_blur_adjudication(entry, c, raw)
        for changes in ({'raw_blur_sha256': '0' * 64}, {'source_input_sha256': '0' * 64},
                        {'full_original_image_actually_viewed': False}, {'evidence': []},
                        {'evidence': [{'observation': 'no location'}]}, {'reason': ''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                app.validate_blur_adjudication(dict(entry, **changes), c, raw)


class ResolutionTests(unittest.TestCase):
    def test_manual_ai_exclusion_cannot_be_cleared(self):
        c = candidate(decision='exclude_ai')
        old = baseline(c, 'ai_excluded')
        for blur in ('clear', 'blur_present', 'uncertain'):
            row = app.resolve_policy_row(old, proposed(c, blur_status=blur), REFERENCE)
            self.assertEqual((row['decision'], row['report_category']), ('reject', 'ai_excluded'))
            self.assertEqual(row['decision_authority'], 'human_user')
            self.assertEqual(row['baseline_screening'], old)
            self.assertEqual(row['human_review'], c['human_review'])
            self.assertEqual(row['authority'], app.AUTHORITY)

    def test_blur_is_quality_rejection_and_uncertainty_not_admitted(self):
        c = candidate()
        for blur, expected in [('blur_present', ('reject', 'quality_excluded')),
                               ('uncertain', ('pending', 'visual_pending'))]:
            row = app.resolve_policy_row(baseline(c), proposed(c, blur_status=blur), REFERENCE)
            self.assertEqual((row['decision'], row['report_category']), expected)
            self.assertEqual(row['decision_authority'], app.AUTHORITY)

    def test_prior_quality_reconsidered_only_when_clear_and_usable(self):
        c = candidate(decision='exclude_quality')
        old = baseline(c, 'quality_excluded')
        for action, decision in [('keep', 'admit'), ('flag_low_information', 'admit'),
                                 ('exclude_pair', 'reject'), ('drop_image_keep_text', 'reject'),
                                 ('needs_context', 'pending'), ('retain_caption_prune_invalid_qa', 'pending')]:
            row = app.resolve_policy_row(old, proposed(c, context_action=action), REFERENCE)
            self.assertEqual(row['decision'], decision)
            self.assertEqual(row['human_review']['decision'], 'exclude_quality')
            self.assertEqual(row['baseline_screening'], old)

    def test_source_followup_stays_pending(self):
        c = candidate()
        row = app.resolve_policy_row(baseline(c), proposed(c, image_origin_followup=True), REFERENCE)
        self.assertEqual(row['decision'], 'pending')
        self.assertEqual(row['rejection_reason'], 'image_origin_followup_required')

    def test_unselected_row_exact_and_no_nested_mutation(self):
        c = candidate()
        old = baseline(c)
        original = deepcopy(old)
        out = app.resolve_policy_row(old, None, REFERENCE)
        self.assertEqual(out, old)
        out['model_screening']['result']['unchanged_nested'].append('changed')
        self.assertEqual(old, original)
        selected = app.resolve_policy_row(old, proposed(c), REFERENCE)
        selected['baseline_screening']['human_review']['note'] = 'change'
        self.assertEqual(old, original)

    def test_wrong_source_cannot_be_applied(self):
        c = candidate()
        with self.assertRaisesRegex(ValueError, 'source reference'):
            app.resolve_policy_row(baseline(c), proposed(c, source_ref={'sha256': c['item_id']}), REFERENCE)


class PublicationTests(unittest.TestCase):
    def test_atomic_publication_idempotency_and_unchanged_other_row(self):
        # Real gzip/fsync/current-pointer publication; only the expensive external
        # inference/human receipt loader is replaced by a small fixture bundle.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = candidate()
            other = candidate(2)
            policy = proposed(c, blur_status='blur_present')
            rows = [baseline(c), baseline(other)]
            source = root / 'baseline.jsonl.gz'
            with gzip.open(source, 'wt') as stream:
                stream.write(''.join(canonical(row) + '\n' for row in rows))
            decisions = root / 'decisions.jsonl.gz'
            with gzip.open(decisions, 'wt') as stream:
                stream.write(canonical(policy) + '\n')
            manifest = root / 'MANIFEST.json'
            manifest.write_text('{"user_policy":"带模糊的直接不要"}')
            human_pointer = root / 'HUMAN_REVIEW_CURRENT.json'
            human_pointer.write_text('{"immutable":"human"}')
            inputs = {'input_manifest': app.artifact(manifest), 'decisions': app.artifact(decisions),
                      'baseline_overlay': app.artifact(source), 'input_human_current': app.artifact(human_pointer)}
            counts = {key: 2 if key == 'admitted' else 0 for key in app.CATEGORIES}
            loaded = ({'user_policy': '带模糊的直接不要'}, inputs,
                      {'resolved_counts': counts, 'model_scope_records': 2}, {c['item_id']: c}, {c['item_id']: policy})
            original_pointer = human_pointer.read_bytes()
            with patch.object(app, 'load_inputs', return_value=loaded), patch.object(app, 'SCOPE', 1):
                report = app.apply_quality(root, manifest)
                current = app.strict_json((root / 'QUALITY_POLICY_CURRENT.json').read_bytes())
                actual = list(app.json_rows(current['resolved_screening_overlay']['path']))
                self.assertEqual(actual[1], rows[1])
                self.assertEqual(actual[0]['baseline_screening'], rows[0])
                self.assertEqual(actual[0]['decision'], 'reject')
                self.assertEqual(report['unchanged_records'], 1)
                self.assertEqual(report['policy_decision_counts']['quality_excluded'], 1)
                self.assertEqual(human_pointer.read_bytes(), original_pointer)
                self.assertEqual(app.apply_quality(root, manifest), report)
                self.assertEqual(len(list((root / 'quality_policy_versions').iterdir())), 1)
                # A published payload changed after hashing must never be reused.
                Path(current['quality_decisions']['path']).write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                    app.apply_quality(root, manifest)


if __name__ == '__main__':
    unittest.main()
