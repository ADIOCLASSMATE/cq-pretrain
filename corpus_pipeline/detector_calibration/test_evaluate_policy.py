import copy
import unittest

from corpus_pipeline.detector_calibration.evaluate_policy import evaluate

CONFIG = {'review_thresholds': {'cf384': .9, 'unifd': .8},
          'priority_both_thresholds': {'cf384': .95, 'unifd': .9}}


def sample(key, origin='unknown', split='calibration', **extra):
    return {'image_sha256': key, 'source': key, 'source_image_origin': origin,
            'domain': 'fixture', 'calibration_split': split, **extra}


def prediction(key, cf=.99, uni=.99, cf224=.99):
    return {'image_sha256': key, 'scores': {
        'cf384': {'status': 'ok', 'prob': cf}, 'unifd': {'status': 'ok', 'prob': uni},
        'cf224': {'status': 'ok', 'prob': cf224}}}


class ReviewPolicyTest(unittest.TestCase):
    def test_protection_and_unknown_not_promoted(self):
        samples = [sample('program', 'programmatic'), sample('render', 'rendered'),
                   sample('edit', ai_edited_from_real_ground_truth=True),
                   sample('photo', 'natural'), sample('unknown')]
        predictions = [prediction(s['image_sha256']) for s in samples[:-1]] + [prediction('unknown', .01, .01)]
        rows, summary = evaluate(samples, predictions, CONFIG)
        self.assertEqual([r['decision'] for r in rows], ['protected_source_audit'] * 3 + ['ai_suspect_review', 'no_model_alert'])
        self.assertTrue(all(r['training_eligibility_decision'] is None and r['provenance_unchanged'] for r in rows))
        self.assertEqual(rows[-1]['source_image_origin'], 'unknown')
        self.assertEqual(summary['overall']['protected_source_audit_both_high'], 3)
        self.assertIsNone(summary['precision_recall_auc'])

    def test_missing_error_and_nonfinite_never_pass(self):
        samples = [sample('missing'), sample('error'), sample('nan', 'programmatic'), sample('infinite_logit')]
        bad = prediction('error'); bad['scores']['unifd'] = {'status': 'error', 'prob': .99}
        nan = prediction('nan', float('nan'))
        inf = prediction('infinite_logit'); inf['scores']['cf384']['logit'] = float('inf')
        rows, summary = evaluate(samples, [bad, nan, inf], CONFIG)
        self.assertTrue(all(r['decision'] == 'inference_error' for r in rows))
        self.assertFalse(summary['all_selected_primary_inference_valid'])
        self.assertEqual(summary['overall']['decisions']['inference_error'], 4)

    def test_two_model_gate_diagnostic_cf224_and_split_isolation(self):
        samples = [sample('one'), sample('both'), sample('only224'), sample('held', split='holdout')]
        predictions = [prediction('one', .91, .1), prediction('both', .96, .91), prediction('only224', .01, .01, 1), prediction('held')]
        rows, summary = evaluate(samples, predictions, CONFIG, 'calibration')
        self.assertEqual([r['decision'] for r in rows], ['ai_suspect_review', 'ai_suspect_review', 'no_model_alert'])
        self.assertEqual([r['both_high'] for r in rows], [False, True, False])
        self.assertEqual(summary['overall']['raw_both_priority_flagged'], 1)
        self.assertEqual(set(summary['by_split']), {'calibration'})
        held = copy.deepcopy(predictions); held[-1] = prediction('held', .01, .01)
        changed_rows, changed_summary = evaluate(samples, held, CONFIG, 'calibration')
        self.assertEqual(rows, changed_rows)
        self.assertEqual(summary, changed_summary)

    def test_disabled_threshold_and_duplicate_guard(self):
        config = {'review_thresholds': {'cf384': 1.000001, 'unifd': 1.000001},
                  'priority_both_thresholds': {'cf384': 1.000001, 'unifd': 1.000001}}
        rows, summary = evaluate([sample('a')], [prediction('a', 1, 1)], config)
        self.assertEqual(rows[0]['decision'], 'no_model_alert')
        self.assertTrue(summary['thresholds_disabled_above_one']['review_thresholds']['cf384'])
        with self.assertRaises(ValueError):
            evaluate([sample('a')], [prediction('a'), prediction('a')], CONFIG)
        with self.assertRaises(ValueError):
            evaluate([sample('a')], [prediction('outside')], CONFIG)
        with self.assertRaises(ValueError):
            evaluate([sample('a')], [prediction('a')], {'review_thresholds': {'cf224': .9}, 'priority_both_thresholds': {}})


if __name__ == '__main__':
    unittest.main()
