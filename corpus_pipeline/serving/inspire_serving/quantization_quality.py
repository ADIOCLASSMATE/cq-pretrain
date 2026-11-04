"""Freeze a real BF16 production baseline and independently qualify a W8 Serving.

Freeze only after the production canary has completed:
  python -m corpus_pipeline.serving.inspire_serving.quantization_quality \
    --freeze-baseline --production-run <run> --production-manifest <manifest> \
    --baseline-manifest <BF16_BASELINE_READY.json>

Then run the candidate gate after its image/auth/boot probe has passed:
  python -m corpus_pipeline.serving.inspire_serving.quantization_quality \
    --candidate-state <qwen-low.json> --baseline-manifest <baseline> \
    --registry <production endpoints.json> --output <quant-quality-INCARNATION.json>

Only a private qualification registry is written. No service creation, paid API,
training publication, or mutation of the production registry is implemented.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from corpus_pipeline.production import prompts
from corpus_pipeline.production.client import DynamicClient, endpoint_lease_valid, image_payload, request_overrides
from corpus_pipeline.production.common import atomic_json, digest, file_sha, jsonl
from corpus_pipeline.production.records import load_record, read_asset_bytes, snapshot
from corpus_pipeline.production.run import model_context, verification_context
from corpus_pipeline.production.validation import annotation_validator, quality_validator, semantic_contract

BASE = Path(__file__).resolve().parents[2]
SCHEMA = 'corpus_pipeline.quantization_quality.v1'
BASELINE_SCHEMA = 'corpus_pipeline.quantization_baseline.v1'
STRATA = ('photo_caption', 'chart_qa', 'document_ocr', 'rendered_qa', 'multi_turn_or_multi_image')
BINDINGS = ('model', 'revision', 'precision', 'processor_revision', 'runtime_revision', 'incarnation', 'expected_boots')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def load(path):
    return json.loads(Path(path).read_text())


def now_string():
    return datetime.now(timezone(timedelta(hours=8), 'Asia/Shanghai')).isoformat()


def pipeline_code():
    return {name: file_sha(BASE / 'production' / name)
            for name in ('prompts.py', 'client.py', 'validation.py', 'records.py', 'run.py')}


def strata_for(record):
    task, domain = semantic_contract(record), record['domain']
    result = []
    if domain == 'photo' and task == 'caption':
        result.append('photo_caption')
    if domain == 'chart' and task == 'qa':
        result.append('chart_qa')
    # Only the existing explicit OCR adapter/source counts. A UI/document image
    # or general document QA must not silently stand in for a transcription task.
    if domain == 'document' and record['source'] == 'synthdog' and task == 'qa':
        result.append('document_ocr')
    if domain == 'rendered' and task == 'qa':
        result.append('rendered_qa')
    assistants = sum(s['kind'] == 'text' and s.get('role') == 'assistant' for s in record['raw']['segments'])
    if len(record['source_image_refs']) >= 2 or task == 'dialogue' and assistants >= 2:
        result.append('multi_turn_or_multi_image')
    return result


def coverage(cases):
    counts = Counter(tag for case in cases for tag in case['strata'])
    missing = [tag for tag in STRATA if not counts[tag]]
    require(not missing, 'Baseline missing required strata: ' + ', '.join(missing))
    return {tag: counts[tag] for tag in STRATA}


def strict_quality(value):
    quality_validator(value)
    return value['verdict'] == 'pass' and not any(i['severity'] == 'major' for i in value['issues'])


def real_row(row, *, precision=None, model=None):
    require(row.get('status') == 'ok' and row.get('real_model_inference') is True, 'Missing real successful model evidence')
    identity = row['identity']
    require(row.get('response_model') == identity['model'], 'Response model differs from evidence identity')
    require(row.get('response_boot_id') in (row.get('qualified_boots_at_request') or []), 'Unqualified response boot')
    require(identity.get('revision') and identity.get('processor_revision') and identity.get('runtime_revision'), 'Incomplete model provenance')
    if precision:
        require(identity['precision'] == precision, 'Baseline is not BF16')
    if model:
        require(identity['model'] == model, 'Unexpected verifier model')


def freeze_baseline(production_run, production_manifest, destination, per_stratum=1, excluded_sample_ids=()):
    """Never invent a missing domain or use unreviewed BF16 rows as a baseline."""
    production_run, production_manifest, destination = map(Path, (production_run, production_manifest, destination))
    require(not destination.exists(), 'Baseline is immutable; choose a new path')
    complete = load(production_run / 'COMPLETE.json')
    require(complete.get('canary_completed') is True, 'BF16 production canary has not completed')
    require(complete.get('summary_sha256') == file_sha(production_run / 'summary.json'), 'Production completion summary changed')
    summary = load(production_run / 'summary.json')
    manifest = load(production_manifest)
    snap = snapshot(manifest['snapshot_root'])
    require(snap['fingerprint'] == manifest['snapshot_fingerprint'], 'Source snapshot changed')
    require(load(production_run / 'run.json')['identity']['manifest_sha256'] == file_sha(production_manifest), 'Production manifest differs from completed run')
    rules = load(BASE / 'config/rules.json')
    rules['sources'] = load(BASE / 'config/sources.json')['sources']
    assets = {a['sha256']: a for a in jsonl(production_run / 'asset_index.jsonl')}
    require(1 <= per_stratum <= 10, 'Invalid bounded baseline size')
    excluded, excluded_seen = set(excluded_sample_ids), set()
    selected, counts, rejected = [], Counter(), []
    for spec in manifest['records']:
        record = load_record(spec, snap, rules)
        if record['original_sample_id'] in excluded:
            excluded_seen.add(record['original_sample_id'])
            rejected.append({'id': record['original_sample_id'], 'reason': 'explicit_final_human_adjudication_exclusion'})
            continue
        tags = strata_for(record)
        if not tags or not any(counts[tag] < per_stratum for tag in tags):
            continue
        path = production_run / 'record_results' / (record['original_sample_id'] + '.json')
        if not path.exists():
            rejected.append({'id': record['original_sample_id'], 'reason': 'missing_real_result'})
            continue
        result = load(path)
        require(result.get('attempt_id') == summary.get('attempt_id'), 'Record evidence is not from the final completed attempt')
        require(result.get('original_sample_id') == record['original_sample_id'], 'Record result identity mismatch')
        if result.get('status') != 'ok' or result.get('annotation_quality_pass') is not True:
            rejected.append({'id': record['original_sample_id'], 'reason': 'BF16_annotation_not_passed'})
            continue
        annotation, quality = result['annotation'], result['quality']
        real_row(annotation, precision='BF16')
        real_row(quality, model='Gemma4-31B-IT')
        require(strict_quality(quality['parsed']) and not annotation['parsed']['concerns'], 'Uncertain or failed baseline annotation')
        annotation_validator(record)(annotation['parsed'])
        context = model_context(record)
        if annotation['identity']['stage'] == 'typed_annotation_adjudication.v1':
            context = {**context, 'previous_annotation': result['pre_adjudication_annotation']['parsed'],
                       'external_visual_review': result['external_review'],
                       'review_handling': 'Recheck the supplied images. Remove unsupported precision or identity claims identified by the review; preserve the native task and original constraints. The review is fallible evidence, not ground truth. Record uncertainty rather than inventing a resolution.'}
        elif annotation['identity']['stage'] == 'typed_annotation_repair.v1':
            context = {**context, 'previous_annotation': result['original_annotation']['parsed'],
                       'independent_review': result['original_quality']['parsed']}
        else:
            require(annotation['identity']['stage'] == 'typed_annotation.v1', 'Unrecognized baseline generation stage')
        require(annotation['identity']['context_sha256'] == digest(context), 'Cannot reconstruct identical baseline request context')
        require(annotation['identity']['prompt_sha256'] == hashlib.sha256(prompts.ANNOTATE.encode()).hexdigest(), 'Baseline annotation prompt changed')
        require(quality['identity']['prompt_sha256'] == hashlib.sha256(prompts.VERIFY.encode()).hexdigest(), 'Baseline verifier prompt changed')
        verification = verification_context(record, model_context(record), annotation['parsed'])
        if result.get('external_review'):
            verification['external_visual_review'] = result['external_review']
        require(quality['identity']['context_sha256'] == digest(verification), 'Cannot reconstruct final baseline verification context')
        bound = [assets[ref['sha256']] for ref in record['source_image_refs']]
        metadata = [image_payload(asset)[1] for asset in bound]
        require(digest(metadata) == digest(annotation.get('image_inputs')), 'Baseline transmitted image processing changed')
        require([a['sha256'] for a in bound] == annotation['identity']['assets'], 'Baseline omitted/reordered required images')
        selected.append({'sample_id': record['original_sample_id'], 'strata': tags,
                         'record': record, 'assets': bound, 'context': context,
                         'generation_stage': annotation['identity']['stage'],
                         'generation_max_tokens': annotation['identity']['max_tokens'],
                         'external_visual_review': result.get('external_review'),
                         'baseline_annotation': annotation, 'baseline_quality': quality,
                         'baseline_record_admission': {key: result.get(key) for key in
                             ('publishable', 'audit_only', 'manual_admission_decision', 'reason_codes')},
                         'source_result_path': str(path.resolve()), 'source_result_sha256': file_sha(path)})
        counts.update(tags)
    require(excluded_seen == excluded, 'Excluded record ID not found in this baseline manifest')
    covered = coverage(selected)
    baseline = {'schema_version': BASELINE_SCHEMA, 'frozen_at': now_string(),
                'production_run': str(production_run.resolve()), 'production_manifest': str(production_manifest.resolve()),
                'production_manifest_sha256': file_sha(production_manifest),
                'production_complete_sha256': file_sha(production_run / 'COMPLETE.json'),
                'manual_admission_evidence': {name: file_sha(production_run / name) for name in
                    ('manual_adjudication.json', 'admission_overrides.json') if (production_run / name).exists()},
                'production_code_sha256': pipeline_code(), 'annotation_prompt': prompts.ANNOTATE,
                'verification_prompt': prompts.VERIFY, 'required_strata': list(STRATA),
                'coverage': covered, 'cases': selected, 'not_selected': rejected,
                'excluded_sample_ids': sorted(excluded), 'maximum_per_stratum': per_stratum,
                'qualification_only': True, 'published_training': False,
                'scope': 'fixed BF16 canary regression gate; not population-wide equivalence'}
    baseline['content_sha256'] = digest(baseline)
    atomic_json(destination, baseline)
    return baseline


def validate_baseline(baseline):
    require(baseline.get('schema_version') == BASELINE_SCHEMA, 'Unsupported baseline schema')
    require(baseline.get('content_sha256') == digest({k: v for k, v in baseline.items() if k != 'content_sha256'}), 'Frozen baseline content changed')
    require(baseline.get('required_strata') == list(STRATA), 'Required coverage changed')
    require(baseline.get('production_code_sha256') == pipeline_code(), 'Production processing code changed; requalify baseline')
    require(baseline['annotation_prompt'] == prompts.ANNOTATE and baseline['verification_prompt'] == prompts.VERIFY, 'Frozen prompts changed')
    require(coverage(baseline['cases']) == baseline['coverage'], 'Coverage receipt mismatch')
    for case in baseline['cases']:
        record = case['record']
        require(strata_for(record) == case['strata'], 'Domain coverage relabeled')
        require(record['source_record_sha256'] == digest(record['raw']), 'Frozen source record changed')
        real_row(case['baseline_annotation'], precision='BF16')
        real_row(case['baseline_quality'], model='Gemma4-31B-IT')
        require(strict_quality(case['baseline_quality']['parsed']), 'Baseline quality is not passed')
        require(not case['baseline_annotation']['parsed']['concerns'], 'Baseline contains unresolved concerns')
        annotation_validator(record)(case['baseline_annotation']['parsed'])
        require(digest(case['context']) == case['baseline_annotation']['identity']['context_sha256'], 'Frozen context changed')
        require([a['sha256'] for a in case['assets']] == [r['sha256'] for r in record['source_image_refs']], 'Required image slots omitted/reordered')
        for asset in case['assets']:
            require(asset['sha256'] == asset['original_ref']['sha256'], 'Image asset binding changed')
            read_asset_bytes(asset['original_ref'])  # Recheck original TAR bytes; no copies.


def candidate_identity(state):
    identity = {k: state[k] for k in (*BINDINGS, 'service_id', 'endpoint', 'request_overrides')}
    identity['expected_boots'] = sorted(set(identity['expected_boots']))
    return identity


def validate_probe(state, proof, now=None):
    now = time.time() if now is None else now
    require(all(isinstance(state.get(k), str) and state[k] for k in BINDINGS if k != 'expected_boots'), 'Candidate identity is incomplete')
    require(state.get('role') == 'generator' and state.get('tier') == 'low', 'Only LOW generator qualification is supported')
    require(state.get('precision') == 'W8A8_DYNAMIC', 'Unsupported candidate precision')
    require(state.get('platform_status') == 'RUNNING' and not state.get('observation_error'), 'Candidate platform observation failed')
    for name in ('last_observed_at', 'heartbeat_at'):
        age = now - state.get(name, 0)
        require(math.isfinite(age) and 0 <= age < 20, 'Stale candidate observation')
    require(state.get('probe_status') in ('PASSED', 'AWAIT_INDEPENDENT_QUANTIZED_DATA_QUALITY_GATE',
                                        'QUANTIZED_QUALITY_RUNNING'), 'Candidate has not passed visual probe')
    require(proof.get('passed') is True and proof.get('authentication_passed') is True, 'Visual/auth probe did not pass')
    require(proof.get('model') == state['model'] and proof.get('revision') == state['revision'], 'Probe model/revision mismatch')
    require(proof.get('base_url') == state['endpoint'].rstrip('/') + '/v1', 'Probe endpoint mismatch')
    boots = state['expected_boots']
    require(isinstance(boots, list) and boots and proof.get('expected_boots') == boots, 'Probe boot mismatch')
    requests = proof.get('requests', [])
    require(len(requests) >= 13 and all(r.get('passed') is True and r.get('boot_id') in boots
            and isinstance(r.get('expected'), dict) and r['expected'] and r.get('actual') == r['expected']
            for r in requests), 'Incomplete or inconsistent visual request evidence')
    finished = proof.get('finished_at')
    require(isinstance(finished, (int, float)) and not isinstance(finished, bool) and math.isfinite(finished)
            and 0 < finished <= now, 'Visual probe has no valid completion time')
    require({x.get('concurrency') for x in proof.get('tiers', [])} >= {1, 4, 8}, 'Concurrent vision qualification incomplete')


class QualificationClient(DynamicClient):
    """The candidate is visible solely inside this private validation client."""

    def __init__(self, candidate_path, production_registry, private_dir, *, concurrency=2, timeout=600):
        self.candidate_path = Path(candidate_path)
        self.production_registry = Path(production_registry)
        self.initial = candidate_identity(load(self.candidate_path))
        self.fixed_verifier = None
        private_dir = Path(private_dir)
        private_dir.mkdir(parents=True, exist_ok=False)
        self.private_path = private_dir / 'qualification_registry.json'
        require(self.private_path.resolve() != self.production_registry.resolve(), 'Cannot overwrite production registry')
        super().__init__(self.private_path, private_dir / 'cache', concurrency=concurrency, timeout=timeout, wait_ready=1)

    def endpoint(self, role):
        state = load(self.candidate_path)
        require(candidate_identity(state) == self.initial, 'Candidate boot/runtime/precision changed during qualification')
        proof = load(state['qualification_receipt'])
        validate_probe(state, proof)
        registry = load(self.production_registry)
        require(not registry.get('qualification_only'), 'Verifier must come from production-safe registry')
        verified = [x for x in registry['endpoints'] if x['role'] == 'verifier' and x.get('ready') and endpoint_lease_valid(x)]
        require(len(verified) == 1 and verified[0]['model'] == 'Gemma4-31B-IT', 'Exactly one qualified Gemma verifier required')
        verifier = verified[0]
        verifier_binding = {k: verifier.get(k) for k in (*BINDINGS, 'service_id', 'base_url', 'request_overrides')}
        if self.fixed_verifier is None:
            self.fixed_verifier = verifier_binding
        require(verifier_binding == self.fixed_verifier, 'Verifier changed during qualification')
        generator = {**self.initial, 'base_url': state['endpoint'].rstrip('/') + '/v1',
                     'role': 'generator', 'backend': 'inspire_serving', 'protocol': 'openai',
                     'api_key_env': state.get('api_key_env', 'INF_API_KEY'), 'supports_images': True,
                     'max_images': state['max_images'], 'max_output_tokens': 12000,
                     'ready': True, 'ready_expires_at': min(state['last_observed_at'] + 20, time.time() + 15),
                     'qualification_only': True}
        atomic_json(self.private_path, {'schema_version': 'corpus_pipeline.endpoints.v1',
                    'registry_generation': 1, 'qualification_only': True, 'published_training': False,
                    'endpoints': [generator, verifier]})
        self.private_path.chmod(0o600)
        return super().endpoint(role)


async def evaluate_case(case, baseline, client):
    generator_endpoint, _ = client.endpoint('generator')
    require(request_overrides(generator_endpoint) == case['baseline_annotation']['identity']['request_overrides'],
            'Candidate decoding/processor options differ from BF16 baseline')
    metadata = [image_payload(asset)[1] for asset in case['assets']]
    require(digest(metadata) == digest(case['baseline_annotation']['image_inputs']), 'Transmitted pixels differ from BF16 baseline')
    generated = await client.request('generator', 'quantization_quality_annotation.v1', case['context'], case['assets'],
                                    baseline['annotation_prompt'], annotation_validator(case['record']),
                                    max_tokens=case['generation_max_tokens'])
    require(generated['identity']['max_tokens'] == case['generation_max_tokens'], 'Candidate required a different output budget than BF16 baseline')
    review_context = verification_context(case['record'], model_context(case['record']), generated['parsed'])
    if case.get('external_visual_review'):
        review_context['external_visual_review'] = case['external_visual_review']
    reviewed = await client.request('verifier', 'quantization_quality_review.v1', review_context,
                                   case['assets'], baseline['verification_prompt'], quality_validator, max_tokens=2200)
    real_row(generated)
    real_row(reviewed, model='Gemma4-31B-IT')
    passed = strict_quality(reviewed['parsed']) and not generated['parsed']['concerns']
    # No repair round hides a regression: one same-prompt candidate and one
    # independent visual review must pass for every baseline-pass record.
    return {'sample_id': case['sample_id'], 'strata': case['strata'], 'passed': passed,
            'status': 'passed' if passed else 'quality_failed', 'retryable': False,
            'annotation': generated, 'quality': reviewed, 'all_required_images_supplied': True}


async def qualify(args):
    output = Path(args.output).resolve()
    registry_path = Path(args.registry).resolve()
    require(output != registry_path, 'Cannot write quality receipt over production registry')
    require(not output.exists(), 'Quality receipt is immutable; choose a fresh output or inspect the existing result')
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + '.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not output.exists(), 'Quality receipt already exists')
        return await _qualify_locked(args, output, registry_path)
    finally:
        lock.close()


async def _qualify_locked(args, output, registry_path):
    started = time.time()
    state = load(args.candidate_state)
    binding = {key: state[key] for key in BINDINGS}
    binding['expected_boots'] = sorted(set(binding['expected_boots']))
    baseline_path = Path(args.baseline_manifest)
    baseline_hash = file_sha(baseline_path)
    baseline = load(baseline_path)
    validate_baseline(baseline)
    validate_probe(state, load(state['qualification_receipt']))
    require(state.get('api_key_env', 'INF_API_KEY') == 'INF_API_KEY', 'Unexpected native Serving credential variable')
    os.environ['INF_API_KEY'] = Path(state['api_key_file']).read_text().strip()
    client = QualificationClient(args.candidate_state, registry_path,
                                 output.with_suffix('.details'), concurrency=args.concurrency, timeout=args.request_timeout)
    results = []
    semaphore = asyncio.Semaphore(args.concurrency)
    async def one(case):
        async with semaphore:
            try:
                result = await evaluate_case(case, baseline, client)
            except Exception as error:
                result = {'sample_id': case['sample_id'], 'strata': case['strata'], 'passed': False,
                          'status': 'operational_error', 'error_type': type(error).__name__,
                          'retryable': getattr(error, 'retryable_after_endpoint_change', False)}
            results.append(result)
            atomic_json(output.with_suffix('.details') / (case['sample_id'] + '.json'), result)
    final_error = None
    try:
        await asyncio.wait_for(asyncio.gather(*(one(case) for case in baseline['cases'])), timeout=args.max_duration)
        client.endpoint('generator')
        client.endpoint('verifier')
        require(file_sha(baseline_path) == baseline_hash, 'Frozen baseline changed during qualification')
    except Exception as error:
        final_error = {'type': type(error).__name__, 'retryable': isinstance(error, (TimeoutError, asyncio.TimeoutError))}
    finally:
        await client.close()
    complete = len(results) == len(baseline['cases'])
    passed = complete and not final_error and all(r['passed'] for r in results)
    quality_failure = any(r['status'] == 'quality_failed' for r in results)
    retryable = not quality_failure and (bool(final_error and final_error['retryable']) or
                bool(results) and all(r['passed'] or r.get('retryable') for r in results))
    receipt = {'schema_version': SCHEMA, **binding, 'candidate_service_id': state['service_id'],
               'baseline_sha256': baseline_hash, 'baseline_manifest': str(baseline_path.resolve()),
               'baseline_content_sha256': baseline['content_sha256'],
               'worker_sha256': file_sha(__file__), 'production_code_sha256': pipeline_code(),
               'verifier_identity': client.fixed_verifier, 'qualification_only': True, 'published_training': False,
               'started_at': started, 'finished_at': time.time(), 'finished_at_asia_shanghai': now_string(),
               'passed': bool(passed), 'status': 'passed' if passed else 'quality_failed' if quality_failure else 'operational_error',
               'retryable': bool(not passed and retryable), 'cases': sorted(results, key=lambda x: x['sample_id']),
               'required_strata': list(STRATA), 'coverage': baseline['coverage'],
               'all_baseline_cases_completed': complete, 'error': final_error,
               'scope': 'fixed real BF16 canary regression gate; not a guarantee of all corpus labels'}
    atomic_json(output, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-baseline', action='store_true')
    parser.add_argument('--production-run')
    parser.add_argument('--production-manifest')
    parser.add_argument('--exclude-sample-id', action='append', default=[])
    parser.add_argument('--per-stratum', type=int, default=1)
    parser.add_argument('--baseline-manifest', required=True)
    parser.add_argument('--candidate-state')
    parser.add_argument('--registry')
    parser.add_argument('--output')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--max-duration', type=float, default=1800)
    args = parser.parse_args(argv)
    if args.freeze_baseline:
        require(args.production_run and args.production_manifest, 'Freeze requires completed production run and manifest')
        result = freeze_baseline(args.production_run, args.production_manifest, args.baseline_manifest,
                                 args.per_stratum, args.exclude_sample_id)
        print(json.dumps({'baseline_manifest': args.baseline_manifest, 'coverage': result['coverage'], 'published_training': False}))
    else:
        require(args.candidate_state and args.registry and args.output, 'Gate requires candidate state, registry, output')
        require(1 <= args.concurrency <= 8 and 0 < args.request_timeout <= 900 and 0 < args.max_duration <= 3600, 'Invalid bounded execution settings')
        result = asyncio.run(qualify(args))
        print(json.dumps({key: result[key] for key in ('status', 'passed', 'retryable', 'coverage')}))
        return 0 if result['passed'] else 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
