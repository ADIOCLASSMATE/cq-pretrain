"""Fail-closed activation bound to successful real frontpass evidence."""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path

from corpus_pipeline.production.common import file_sha

CODE_FILES = ('inference.py', 'optimized_inference.py', 'frontpass.py', 'tuned_inference.py',
              'fast_prepare.py', 'tuned_frontpass.py', 'tuned_configuration.py', 'node_worker.py')
EVIDENCE_CODE_FILES = CODE_FILES + ('optimization_probe.py', 'tuned_frontpass_probe.py')
EXPECTED_CASES = {('s00030-r00000000', 0): ('as100m', 2048),
                  ('s00030-r00000000', 1): ('as100m', 2048),
                  ('s00016-r00000000', 0): ('anyword3m', 1024),
                  ('s00007-r00000000', 0): ('allava', 8192),
                  **{('primary-' + source, 0): (source, 1024)
                     for source in ('datacomp_1b', 'relaion-2b-en-vol-1-unpacked-TEST',
                                    'relaion-2b-en-vol-2-unpacked-TEST')},
                  **{('diverse-' + source, 0): (source, 16)
                     for source in ('chartqa', 'clevr_math', 'docvqa', 'iam', 'coco_caption')}}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_profile(cfg, agg):
    """Use the same bounded settings for the default and every source override."""
    _require(isinstance(cfg, dict) and set(cfg) == {'batch_size', 'decode_workers', 'cpu_threads',
                                                  'prefetch_batches', 'fast_prepare'}, 'invalid tuned settings')
    for key, maximum in [('batch_size', 1024), ('decode_workers', 32),
                         ('cpu_threads', 16), ('prefetch_batches', 4)]:
        _require(type(cfg[key]) is int and 1 <= cfg[key] <= maximum, 'invalid tuned setting: ' + key)
    _require(type(cfg['fast_prepare']) is bool, 'invalid fast preparation setting')
    _require(isinstance(agg, dict) and set(agg) == {'target_images', 'max_records', 'max_record_bytes'}
             and all(type(v) is int and v > 0 for v in agg.values()), 'invalid tuned aggregation')
    _require(cfg['batch_size'] * (cfg['prefetch_batches'] + 1) <= agg['target_images'] <= 1024
             and agg['max_records'] <= 4096 and agg['max_record_bytes'] <= 64 * 1024 ** 2,
             'unqualified aggregation bound')


def _validate_source_profiles(profiles):
    _require(isinstance(profiles, dict), 'invalid source profiles')
    tested_sources = {source for source, _ in EXPECTED_CASES.values()}
    for source, profile in profiles.items():
        _require(isinstance(source, str) and source in tested_sources,
                 'source profile has no required real-data validation case')
        _require(isinstance(profile, dict) and set(profile) == {'configuration', 'aggregation'},
                 'invalid source profile settings')
        _validate_profile(profile['configuration'], profile['aggregation'])


def select_source_profile(qualified, source):
    """Return an isolated profile for an exact source; absent overrides use default.

    The caller must first validate the qualification with configuration(root).
    This selector never infers profiles from image content or source substrings.
    """
    _require(isinstance(source, str) and bool(source), 'task source must be a nonempty string')
    profile = qualified.get('source_profiles', {}).get(source)
    if profile is None:
        profile = {'configuration': qualified['configuration'], 'aggregation': qualified['aggregation']}
    return deepcopy(profile)


def _validate_comparisons(evidence, cfg, agg, profiles):
    results = evidence.get('results')
    _require(isinstance(results, list) and len(results) == len(EXPECTED_CASES),
             'real-data validation case coverage incomplete')
    seen = set()
    for result in results:
        _require(isinstance(result, dict) and type(result.get('repetition')) is int,
                 'invalid real-data validation case')
        key = (result.get('case'), result['repetition'])
        _require(key in EXPECTED_CASES and key not in seen, 'unexpected or repeated validation case')
        seen.add(key)
        source, records = EXPECTED_CASES[key]
        _require(result.get('source') == source, 'validation source mismatch')
        actual_cfg, actual_agg = result.get('candidate_configuration'), result.get('candidate_aggregation')
        _validate_profile(actual_cfg, actual_agg)
        expected = select_source_profile({'configuration': cfg, 'aggregation': agg,
                                          'source_profiles': profiles}, source)
        _require(actual_cfg == expected['configuration'] and actual_agg == expected['aggregation'],
                 'validation case configuration/aggregation differs from source profile')
        comparison = result.get('comparison', {})
        _require(isinstance(comparison, dict)
                 and comparison.get('record_outcomes_equal') is True
                 and comparison.get('image_metadata_and_outcomes_equal') is True
                 and type(comparison.get('threshold_decision_disagreements')) is int
                 and comparison['threshold_decision_disagreements'] == 0,
                 'validation decisions or metadata changed')
        _require(type(comparison.get('records')) is int and comparison['records'] == records,
                 'validation row coverage mismatch')
        images = comparison.get('image_occurrences')
        _require(type(images) is int and images >= 0, 'invalid validation image coverage')
        if source in profiles:
            _require(images > 0, 'source profile validation has no image coverage')
        errors = comparison.get('max_probability_error')
        _require(isinstance(errors, dict) and set(errors) == {'cf384_prob', 'unifd_prob'}
                 and all(_number(v) and 0 <= v <= .001 for v in errors.values()),
                 'validation detector tolerance exceeded')
        cosine = comparison.get('minimum_sscd_cosine')
        _require(_number(cosine) and .999 <= cosine <= 1.000001, 'validation SSCD tolerance exceeded')
        timings = result.get('timings', {})
        _require(isinstance(timings, dict) and set(timings) == {'baseline', 'candidate'},
                 'validation timing/coverage evidence missing')
        for mode in ('baseline', 'candidate'):
            timing = timings[mode]
            _require(isinstance(timing, dict), 'invalid validation timing')
            counts = timing.get('counts', {})
            _require(isinstance(counts, dict) and type(counts.get('records')) is int
                     and counts['records'] == records
                     and type(counts.get('image_occurrences', 0)) is int
                     and counts.get('image_occurrences', 0) == images,
                     'validation measured record/image coverage mismatch')
            _require(_number(timing.get('elapsed_seconds')) and timing['elapsed_seconds'] > 0,
                     'invalid validation elapsed time')


def configuration(root):
    root = Path(root).resolve()
    path = root / 'TUNED_QUALIFIED.json'
    if not os.path.lexists(path):
        return None
    try:
        receipt = json.loads(path.read_text())
        _require(isinstance(receipt, dict) and receipt.get('success') is True
                 and receipt.get('dtype') == 'float32', 'tuned execution is not qualified')
        hashes = receipt.get('code_sha256', {})
        _require(isinstance(hashes, dict), 'invalid tuned code identity')
        actual = {name: file_sha(Path(__file__).with_name(name)) for name in EVIDENCE_CODE_FILES}
        for name in CODE_FILES:
            _require(hashes.get(name) == actual[name], 'tuned qualified code changed: ' + name)
        cfg = receipt.get('configuration')
        agg = receipt.get('aggregation')
        _validate_profile(cfg, agg)
        profiles = receipt.get('source_profiles', {})
        _validate_source_profiles(profiles)
        _require(type(receipt.get('processes_per_device')) is int and receipt['processes_per_device'] == 1,
                 'only one qualified process per device currently supported')
        verification_value = receipt.get('validation_receipt')
        _require(isinstance(verification_value, str) and Path(verification_value).is_absolute(),
                 'validation receipt must be an absolute path')
        verification = Path(verification_value)
        _require(file_sha(verification) == receipt.get('validation_receipt_sha256'),
                 'tuned validation receipt changed')
        evidence = json.loads(verification.read_text())
        _require(isinstance(evidence, dict) and evidence.get('success') is True
                 and evidence.get('dtype') == 'float32'
                 and str(evidence.get('device', '')).startswith('npu')
                 and evidence.get('scope') == 'bounded_real_frontpass_numerical_and_throughput_comparison_not_AI_accuracy',
                 'tuned real-data verification did not succeed on the required NPU path')
        _validate_profile(evidence.get('candidate_configuration'), evidence.get('candidate_aggregation'))
        evidence_profiles = evidence.get('source_profiles', {})
        _validate_source_profiles(evidence_profiles)
        _require(evidence.get('candidate_configuration') == cfg
                 and evidence.get('candidate_aggregation') == agg,
                 'qualified configuration/aggregation differs from validation evidence')
        _require(evidence_profiles == profiles, 'source profiles differ from validation evidence')
        _require(evidence.get('baseline_configuration') == {'batch_size': 32, 'threads': 4}
                 and evidence.get('baseline_aggregation') == {'target_images': 128, 'max_records': 1024,
                                                            'max_record_bytes': 64 << 20},
                 'validation baseline is not the incumbent optimized worker')
        evidence_hashes = evidence.get('code_sha256', {})
        _require(isinstance(evidence_hashes, dict), 'invalid validation code identity')
        for name in EVIDENCE_CODE_FILES:
            _require(evidence_hashes.get(name) == actual[name], 'validation code changed: ' + name)
        _require(evidence.get('plan_sha256') == file_sha(root / 'plan.json'), 'validation plan identity mismatch')
        execution = evidence.get('candidate_execution_identity', {})
        _require(isinstance(execution, dict) and all(execution.get(key) == value for key, value in cfg.items())
                 and execution.get('model_identity_unchanged') is True
                 and execution.get('base_inference_code_sha256') == actual['inference.py']
                 and execution.get('optimized_inference_code_sha256') == actual['optimized_inference.py']
                 and execution.get('tuned_inference_code_sha256') == actual['tuned_inference.py']
                 and execution.get('fast_prepare_code_sha256') == (actual['fast_prepare.py'] if cfg['fast_prepare'] else None),
                 'candidate execution identity differs from qualified settings')
        model = evidence.get('model_identity', {})
        _require(isinstance(model, dict) and model.get('dtype') == 'float32'
                 and model.get('code_sha256') == actual['inference.py'], 'validation model identity mismatch')
        _validate_comparisons(evidence, cfg, agg, profiles)
        return receipt
    except RuntimeError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError('unreadable or malformed tuned qualification: ' + type(error).__name__) from error
