"""Evaluate externally frozen review thresholds without changing corpus policy.

Only CF384 and UniFD gate review. CF224 is diagnostic. No model outcome grants
training eligibility, changes provenance, or deletes an image. The input split
must have been assigned before scoring; thresholds are never fitted here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

PRIMARY_MODELS = ('cf384', 'unifd')
DIAGNOSTIC_MODELS = ('cf224',)
PROTECTED_ORIGINS = {'programmatic', 'rendered'}
DECISIONS = ('ai_suspect_review', 'protected_source_audit', 'no_model_alert', 'inference_error')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def thresholds(config):
    result = {}
    for key in ('review_thresholds', 'priority_both_thresholds'):
        values = config.get(key)
        if not isinstance(values, dict) or set(values) != set(PRIMARY_MODELS):
            raise ValueError(f'{key} must contain exactly cf384 and unifd; CF224 is diagnostic only')
        parsed = {}
        for model, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key}.{model} must be a finite nonnegative number')
            # Values above one deliberately disable that threshold. Never clip to 1.
            parsed[model] = float(value)
        result[key] = parsed
    for model in PRIMARY_MODELS:
        if result['priority_both_thresholds'][model] < result['review_thresholds'][model]:
            raise ValueError('Priority threshold must be at least the review threshold for each model')
    return result


def checked_score(score):
    if not isinstance(score, dict):
        return {'status': 'missing', 'prob': None, 'error': 'missing_model_score'}
    status = score.get('status', 'missing')
    if status != 'ok':
        return {'status': status, 'prob': None, 'error': str(score.get('error', f'model_status_{status}'))}
    prob = score.get('prob')
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(prob):
        return {'status': 'invalid', 'prob': None, 'error': 'missing_or_nonfinite_probability'}
    if not 0 <= prob <= 1:
        return {'status': 'invalid', 'prob': None, 'error': 'probability_out_of_range'}
    logit = score.get('logit')
    if logit is not None and (isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(logit)):
        return {'status': 'invalid', 'prob': None, 'error': 'nonfinite_or_invalid_logit'}
    result = {'status': 'ok', 'prob': float(prob)}
    if logit is not None:
        result['logit'] = float(logit)
    return result


def assigned_split(sample):
    value = sample.get('split', sample.get('calibration_split'))
    if value not in ('calibration', 'holdout'):
        raise ValueError('Every sample needs its preassigned calibration/holdout split')
    return value


def origin(sample):
    return sample.get('source_image_origin', sample.get('image_origin', 'unknown'))


def evaluate_one(sample, prediction, config):
    raw = prediction.get('scores', {}) if isinstance(prediction, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    scores = {model: checked_score(raw.get(model)) for model in (*PRIMARY_MODELS, *DIAGNOSTIC_MODELS)}
    failures = {m: scores[m]['error'] for m in PRIMARY_MODELS if scores[m]['status'] != 'ok'}
    reasons = []
    if origin(sample) in PROTECTED_ORIGINS:
        reasons.append('source_declares_' + origin(sample))
    if sample.get('ai_edited_from_real_ground_truth') is True:
        reasons.append('explicit_verified_ai_edit_from_real_source')
    protected = bool(reasons)
    review_flags = {m: None for m in PRIMARY_MODELS}
    priority_flags = {m: None for m in PRIMARY_MODELS}
    priority = None
    both_high = None
    if failures:
        decision = 'inference_error'
    else:
        review_flags = {m: scores[m]['prob'] >= config['review_thresholds'][m] for m in PRIMARY_MODELS}
        priority_flags = {m: scores[m]['prob'] >= config['priority_both_thresholds'][m] for m in PRIMARY_MODELS}
        both_high = all(priority_flags.values())
        if any(review_flags.values()):
            decision = 'protected_source_audit' if protected else 'ai_suspect_review'
            priority = 'both_high' if both_high else 'single_or_lower_priority_high'
        else:
            decision = 'no_model_alert'
    return {
        'image_sha256': sample['image_sha256'], 'image_path': sample.get('image_path'),
        'sample_id': sample.get('sample_id'), 'source': sample.get('source', 'unknown'),
        'domain': sample.get('source_domain', sample.get('domain', 'unknown')),
        'domain_label_scope': sample.get('domain_label_scope', 'source_stratum_not_per_image_truth'),
        'source_image_origin': origin(sample), 'split': assigned_split(sample),
        'decision': decision, 'review_priority': priority,
        'protected_by_source_or_verified_edit': protected, 'protection_reasons': reasons,
        'ai_edited_from_real_ground_truth': sample.get('ai_edited_from_real_ground_truth'),
        'review_flags': review_flags, 'priority_flags': priority_flags, 'both_high': both_high,
        'scores': scores, 'inference_errors': failures,
        'automatic_deletion': False, 'training_eligibility_decision': None,
        'provenance_unchanged': True, 'model_inferred_authenticity_ground_truth': None,
        'note': 'Review/annotation only. no_model_alert is not acceptance or proof of real origin.',
    }


def group_stats(rows):
    decisions = Counter(r['decision'] for r in rows)
    valid = [r for r in rows if r['decision'] != 'inference_error']
    return {
        'records': len(rows), 'primary_scores_valid': len(valid),
        'decisions': {k: decisions[k] for k in DECISIONS},
        'raw_review_flags': {m: sum(r['review_flags'][m] is True for r in valid) for m in PRIMARY_MODELS},
        'raw_any_model_review_flagged': sum(any(r['review_flags'].values()) for r in valid),
        'raw_both_priority_flagged': sum(r['both_high'] is True for r in valid),
        'protected_records': sum(r['protected_by_source_or_verified_edit'] for r in rows),
        'ai_suspect_review_both_high': sum(r['decision'] == 'ai_suspect_review' and r['both_high'] is True for r in rows),
        'protected_source_audit_both_high': sum(r['decision'] == 'protected_source_audit' and r['both_high'] is True for r in rows),
        'model_score_statuses': {m: dict(Counter(r['scores'][m]['status'] for r in rows)) for m in (*PRIMARY_MODELS, *DIAGNOSTIC_MODELS)},
    }


def evaluate(samples, predictions, frozen_config, selected_split='all'):
    if selected_split not in ('calibration', 'holdout', 'all'):
        raise ValueError('split must be calibration, holdout, or all')
    config = thresholds(frozen_config)
    by_sample = {s['image_sha256']: s for s in samples}
    if len(by_sample) != len(samples):
        raise ValueError('Samples contain duplicate image_sha256')
    by_prediction = {p['image_sha256']: p for p in predictions}
    if len(by_prediction) != len(predictions):
        raise ValueError('Predictions contain duplicate image_sha256; supply the merged index')
    if set(by_prediction) - set(by_sample):
        raise ValueError('Predictions contain images outside the locked sample manifest')
    rows = [evaluate_one(s, by_prediction.get(s['image_sha256']), config) for s in samples
            if selected_split == 'all' or assigned_split(s) == selected_split]
    # Validate splits even when a filter would otherwise skip the affected row.
    for s in samples:
        assigned_split(s)
    grouped = {}
    for dimension in ('source', 'domain', 'source_image_origin'):
        buckets = defaultdict(list)
        for row in rows:
            buckets[(row[dimension], row['split'])].append(row)
        grouped[dimension + '_by_split'] = [
            {dimension: key, 'split': split, **group_stats(values)}
            for (key, split), values in sorted(buckets.items())
        ]
    report = {
        'scope': 'source_stratified_review_policy_diagnostic_not_corpus_prevalence',
        'thresholds_fitted_here': False, 'thresholds': config,
        'thresholds_disabled_above_one': {k: {m: t > 1 for m, t in values.items()} for k, values in config.items()},
        'required_models': list(PRIMARY_MODELS), 'diagnostic_only_models': list(DIAGNOSTIC_MODELS),
        'requested_split': selected_split, 'locked_sample_count': len(samples),
        'prediction_record_count': len(predictions), 'selected_sample_count': len(rows),
        'all_selected_primary_inference_valid': all(r['decision'] != 'inference_error' for r in rows),
        'overall': group_stats(rows),
        'by_split': {name: group_stats([r for r in rows if r['split'] == name]) for name in sorted({r['split'] for r in rows})},
        **grouped,
        'automatic_deletion': False, 'training_eligibility_decision': None,
        'precision_recall_auc': None, 'protected_audit_is_nonblocking_for_this_detector_stage': True,
        'limitations': [
            'Source/domain strata are not per-image ground truth, and sample proportions cannot estimate corpus prevalence.',
            'Natural-source records can be reviewed; low scores never relabel unknown/mixed records as natural or accepted.',
            'Programmatic/rendered source evidence or explicit verified real-source AI editing protects a flagged record from ai_suspect_review.',
            'Missing/error/nonfinite CF384 or UniFD scores result in inference_error even for protected records; no acceptance is issued.',
            'CF224 is diagnostic only; its status and scores do not participate in the two-model gate.',
            'Threshold file is treated as externally frozen. This tool records its hash but cannot prove the user froze it before viewing holdout.',
        ],
    }
    return rows, report


def atomic_write(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--thresholds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('calibration', 'holdout', 'all'), default='all')
    args = parser.parse_args()
    rows, report = evaluate(read_jsonl(args.samples), read_jsonl(args.scores),
                            json.loads(args.thresholds.read_text()), args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    report['inputs_sha256'] = {str(p.resolve()): sha256(p) for p in (args.samples, args.scores, args.thresholds)}
    report['evaluation_code_sha256'] = sha256(__file__)
    index = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
    atomic_write(args.output / 'review_index.jsonl', index)
    report['review_index_sha256'] = sha256(args.output / 'review_index.jsonl')
    atomic_write(args.output / 'policy_evaluation.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'selected_records': len(rows), **report['overall']['decisions']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
