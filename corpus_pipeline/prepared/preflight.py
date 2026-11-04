"""Read-only, sample-scoped DCVLM preflight; no admission or model inference.

Re-read original TAR byte ranges, validate the locked SHA256, decode dimensions,
compare candidate short-edge thresholds, and retain original task associations.
The output is a shadow audit, never a published training dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import time

from PIL import Image, __version__ as pillow_version

from corpus_pipeline import policy
from corpus_pipeline.detector_calibration import evaluate_policy


DEFAULT_THRESHOLDS = (32, 64, 128, 192, 224, 256, 384, 480, 512)
RUN_ROOT = Path('corpus_pipeline/runs/detector_calibration_20261022')
SCHEMA_VERSION = 'prepared-preflight.v1'


def sha256(path):
    return evaluate_policy.sha256(path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def indexed(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f'duplicate {key}: {value}')
        result[value] = row
    return result


def small_image_flags(width, height, candidates=DEFAULT_THRESHOLDS):
    """A dimension equal to a threshold is retained by that dimension test."""
    return {str(t): min(width, height) < t for t in candidates}


def probe_asset(sample, prediction, frozen, candidates=DEFAULT_THRESHOLDS,
                max_bytes=128 * 1024 * 1024, max_pixels=100_000_000):
    """Decode only original TAR bytes. Never consult image_path copies."""
    row = {
        'schema_version': SCHEMA_VERSION,
        'image_sha256': sample['image_sha256'], 'source': sample['source'],
        'domain': sample.get('source_domain', sample.get('domain', 'unknown')),
        'domain_label_scope': sample.get('domain_label_scope', 'source_stratum'),
        'source_image_origin': sample.get('source_image_origin', 'unknown'),
        'source_evidence': sample.get('source_evidence', []),
        'source_evidence_level': sample.get('source_evidence_level'),
        'caption_origin': sample.get('caption_origin', 'unknown'),
        'original_locator': {k: sample[k] for k in ('tar_path', 'tar_member', 'tar_offset', 'tar_length')},
        'hash_verified': False, 'decode_status': 'not_attempted',
        'short_edge_rejected_by_candidate': {str(t): None for t in candidates},
        'sample_only': True, 'automatic_deletion': False,
        'training_eligibility': None, 'published_training': False,
        'sscd_dedup_status': 'not_run', 'recaption_status': 'not_run',
        'layout_status': 'not_run',
    }
    # Scores are evidence for routing only; raw integrity still gates their reuse.
    routing = evaluate_policy.evaluate_one(sample, prediction, frozen)
    row['detector_evidence'] = {
        k: routing[k] for k in ('decision', 'review_flags', 'scores', 'inference_errors',
                               'protected_by_source_or_verified_edit', 'protection_reasons')
    }
    row['detector_scores_reused'] = True
    row['detector_route'] = 'input_integrity_error'
    try:
        offset, length = sample['tar_offset'], sample['tar_length']
        if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
                or isinstance(length, bool) or not isinstance(length, int)
                or length <= 0 or length > max_bytes):
            raise ValueError('invalid_tar_byte_range_or_max_bytes_exceeded')
        with Path(sample['tar_path']).open('rb') as stream:
            stream.seek(offset)
            blob = stream.read(length)
        if len(blob) != length:
            raise ValueError('truncated_tar_byte_range')
        row['bytes_read'] = len(blob)
        digest = hashlib.sha256(blob).hexdigest()
        row['observed_sha256'] = digest
        if digest != sample['image_sha256']:
            row['integrity_status'] = 'hash_mismatch'
            return row
        row['hash_verified'] = True
    except (OSError, ValueError, OverflowError) as error:
        row['integrity_status'] = 'source_read_error'
        row['error'] = f'{type(error).__name__}: {error}'[:500]
        return row
    try:
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError('invalid_dimensions_or_max_pixels_exceeded')
            row.update(width=width, height=height, short_edge=min(width, height),
                       pixels=width * height, image_format=image.format,
                       image_mode=image.mode, frame_count=getattr(image, 'n_frames', 1),
                       decode_scope='first_frame')
            image.load()
            row['decode_status'] = 'ok'
        row['manifest_dimensions_match'] = (
            width == sample.get('decoded_width') and height == sample.get('decoded_height'))
        row['short_edge_rejected_by_candidate'] = small_image_flags(width, height, candidates)
        row['integrity_status'] = 'ok'
        row['detector_route'] = routing['decision']
        row['record_admission_complete'] = False
    except Exception as error:
        row['integrity_status'] = 'decode_error'
        row['decode_status'] = 'error'
        row['error'] = f'{type(error).__name__}: {error}'[:500]
    return row


def record_association(association, asset):
    """Preserve every supervision record even when its image was seen before."""
    count = association.get('source_image_count')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('association source_image_count must be a positive integer')
    segments = association.get('segments')
    if not isinstance(segments, list):
        raise ValueError('association segments must be a list')
    normalized = {
        'collection': 'dcvlm', 'format': association.get('format'),
        'segments': segments, 'images': [{'index': i} for i in range(count)],
        'issues': [],
    }
    category = policy.classify(normalized)
    sampled_valid = asset['integrity_status'] == 'ok'
    return {
        'schema_version': SCHEMA_VERSION,
        'sample_id': association['sample_id'], 'image_sha256': association['image_sha256'],
        'source': association['source'], 'collection': 'dcvlm',
        'native_task_type': category,
        'task_views_preserving_native_direction': policy.eligible_task_views('dcvlm', category, [category]),
        'format': association.get('format'), 'segments': segments,
        'caption_origin': association.get('caption_origin', 'unknown'),
        'record_image_count': count, 'sampled_image_count': 1,
        'sampled_image_index': association['image_index'],
        'sampled_asset_checks_complete': sampled_valid,
        'all_record_images_observed': count == 1,
        'all_record_images_integrity_checked': count == 1 and sampled_valid,
        'record_admission_complete': False,
        'remaining_image_count': count - 1,
        'source_locator': {k: association.get(k) for k in (
            'parquet_path', 'parquet_row_index', 'source_shard', 'source_key',
            'row_ordinal', 'annotation_file', 'image_ref')},
        'association_locator_independently_read': False,
        'canonical_asset_original_locator': asset['original_locator'],
        'integrity_status': asset['integrity_status'],
        'detector_route': asset['detector_route'],
        'short_edge_rejected_by_candidate': asset['short_edge_rejected_by_candidate'],
        'sample_only': True, 'automatic_deletion': False,
        'training_eligibility': None, 'published_training': False,
        'note': 'Only the sampled canonical image asset was probed; other images and alternate TAR locators were not read. Native supervision is preserved; no T2I inversion.',
    }


def threshold_stats(rows, candidates):
    rows = list(rows)
    values = {}
    for threshold in candidates:
        counts = Counter(r['short_edge_rejected_by_candidate'][str(threshold)] for r in rows)
        values[str(threshold)] = {
            'sampled_count': len(rows), 'would_reject_short_edge_lt': counts[True],
            'dimension_test_pass_only': counts[False], 'unresolved_integrity_or_decode': counts[None],
        }
    return values


def grouped_thresholds(rows, dimension, candidates):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(dimension, 'unknown')].append(row)
    return {name: threshold_stats(group, candidates) for name, group in sorted(groups.items())}


def json_write(path, value):
    # All files are new in an exclusively created run directory.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def jsonl_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def run(args):
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.monotonic()
    if args.workers < 1 or args.workers > 64:
        raise ValueError('workers must be in [1,64]')
    candidates = tuple(sorted(set(args.size_thresholds)))
    if not candidates or any(t < 1 for t in candidates):
        raise ValueError('size thresholds must be positive integers')
    if args.output.exists():
        raise FileExistsError(f'output already exists; use a new run directory: {args.output}')
    input_paths = [args.samples, args.associations, args.scores, args.thresholds]
    if args.api_decisions:
        input_paths.append(args.api_decisions)
    input_hashes = {str(p.resolve()): sha256(p) for p in input_paths}
    code_paths = [Path(__file__), Path(policy.__file__), Path(evaluate_policy.__file__)]
    code_hashes = {str(p.resolve()): sha256(p) for p in code_paths}
    samples = indexed(read_jsonl(args.samples), 'image_sha256')
    predictions = indexed(read_jsonl(args.scores), 'image_sha256')
    if set(predictions) - set(samples):
        raise ValueError('scores contain images outside the locked sample manifest')
    associations = list(read_jsonl(args.associations))
    indexed(associations, 'sample_id')  # This locked sampler chose one image per record.
    if {a['image_sha256'] for a in associations} != set(samples):
        raise ValueError('associations must cover exactly the sampled image assets')
    frozen = evaluate_policy.thresholds(json.loads(args.thresholds.read_text()))
    api = indexed(read_jsonl(args.api_decisions), 'image_sha256') if args.api_decisions else {}
    args.output.mkdir(parents=True, exist_ok=False)
    assets = {}
    asset_path = args.output / 'asset_index.jsonl'
    # executor.map submits only the bounded sample manifest, while each worker
    # holds at most one image's bytes. No image copies or decoded arrays persist.
    with asset_path.open('x', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        def process(sample):
            return probe_asset(sample, predictions.get(sample['image_sha256']), frozen,
                               candidates, args.max_bytes, args.max_pixels)
        for row in pool.map(process, samples.values()):
            prior = api.get(row['image_sha256'])
            if prior is not None:
                row['prior_api_evidence_reference'] = {
                    'path': str(args.api_decisions.resolve()),
                    'file_sha256': input_hashes[str(args.api_decisions.resolve())],
                    **{k: prior.get(k) for k in ('review_id', 'final_disposition',
                         'manual_review_required', 'operational_retry_required', 'rule_version')},
                    'used_as_training_approval': False,
                }
            assets[row['image_sha256']] = row
            jsonl_line(stream, row)
            if len(assets) % 400 == 0:
                print(json.dumps({'assets_probed': len(assets), 'total': len(samples)}), flush=True)
    records = []
    with (args.output / 'record_index.jsonl').open('x', encoding='utf-8') as stream:
        for association in associations:
            row = record_association(association, assets[association['image_sha256']])
            records.append(row)
            jsonl_line(stream, row)
    rows = list(assets.values())
    multiplicities = Counter(a['image_sha256'] for a in associations)
    categories_by_source = defaultdict(Counter)
    for row in records:
        categories_by_source[row['source']][row['native_task_type']] += 1
    summary = {
        'schema_version': SCHEMA_VERSION, 'sample_only': True,
        'scope': 'locked_source_stratified_sample_not_full_corpus_or_prevalence_estimate',
        'unique_sampled_assets': len(rows), 'record_associations': len(records),
        'sources_in_associations': len(categories_by_source),
        'bytes_read_from_original_tars': sum(r.get('bytes_read', 0) for r in rows),
        'integrity_statuses': dict(Counter(r['integrity_status'] for r in rows)),
        'manifest_dimension_mismatches': sum(r.get('manifest_dimensions_match') is False for r in rows),
        'non_single_frame_assets': sum(r.get('frame_count', 1) != 1 for r in rows),
        'detector_routes': dict(Counter(r['detector_route'] for r in rows)),
        'prior_api_references': sum('prior_api_evidence_reference' in r for r in rows),
        'assets_with_multiple_supervision_records': sum(v > 1 for v in multiplicities.values()),
        'additional_supervision_associations_preserved': len(records) - len(rows),
        'native_task_types': dict(Counter(r['native_task_type'] for r in records)),
        'native_task_types_by_source': {k: dict(v) for k, v in sorted(categories_by_source.items())},
        'records_with_unobserved_images': sum(r['remaining_image_count'] > 0 for r in records),
        'unobserved_image_slots': sum(r['remaining_image_count'] for r in records),
        'short_edge_thresholds': threshold_stats(rows, candidates),
        'short_edge_thresholds_by_domain': grouped_thresholds(rows, 'domain', candidates),
        'short_edge_thresholds_by_primary_source': grouped_thresholds(rows, 'source', candidates),
        'short_edge_thresholds_by_record_source': grouped_thresholds(records, 'source', candidates),
        'thresholds_selected_here': False, 'size_comparison': 'short_edge < candidate_threshold',
        'frozen_detector_thresholds': frozen,
        'automatic_deletion': False, 'training_eligibility': None,
        'published_training': False, 'record_admission_complete': False,
        'new_model_inference_calls': 0, 'new_api_calls': 0,
        'limitations': [
            'Images were already sampled by the previous sampler; this audit cannot measure the source corpus unreadable-image rate.',
            'Pure-text records were not sampled here; zero observed pure-text records is not proof of their absence.',
            'One image per sampled original record; multi-image records have unchecked images and cannot be admitted as complete records.',
            'Source domains are strata, not per-image truth; counts cannot estimate population fractions.',
            'Only canonical original TAR locations were probed; alternative TAR locations in other associations were preserved but not independently read.',
            'Reused detector scores only route review; no_model_alert does not establish non-AI provenance.',
            'Prior API dispositions are linked audit evidence only, not accepted training decisions.',
            'SSCD, recaption, layout generation, quality validation and training publication were not run.',
        ],
    }
    json_write(args.output / 'summary.json', summary)
    # Detect manifest/code mutations before declaring a completed audit.
    if any(sha256(Path(p)) != digest for p, digest in {**input_hashes, **code_hashes}.items()):
        raise RuntimeError('input_or_code_changed_during_audit; no COMPLETE marker written')
    receipt = {
        'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': round(time.monotonic() - start_clock, 3),
        'inputs_sha256': input_hashes, 'code_sha256': code_hashes,
        'runtime': {'python': platform.python_version(), 'pillow': pillow_version, 'workers': args.workers},
        'config': {'candidate_short_edges': candidates, 'max_bytes_per_asset': args.max_bytes,
                   'max_pixels_per_asset': args.max_pixels, 'read_source': 'original_tar_byte_ranges'},
        'artifacts_sha256': {name: sha256(args.output / name) for name in (
            'asset_index.jsonl', 'record_index.jsonl', 'summary.json')},
    }
    json_write(args.output / 'receipt.json', receipt)
    json_write(args.output / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'audit_completed': True,
        'all_sampled_assets_integrity_verified': all(r['integrity_status'] == 'ok' for r in rows),
        'sample_only': True, 'published_training': False,
        'training_eligibility': None, 'automatic_deletion': False,
        'record_admission_complete': False, 'receipt_sha256': sha256(args.output / 'receipt.json'),
    })
    print(json.dumps({k: summary[k] for k in (
        'unique_sampled_assets', 'record_associations', 'integrity_statuses',
        'detector_routes', 'native_task_types', 'records_with_unobserved_images')}, ensure_ascii=False), flush=True)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--samples', type=Path, default=RUN_ROOT / 'samples.jsonl')
    result.add_argument('--associations', type=Path, default=RUN_ROOT / 'sample_associations.jsonl')
    result.add_argument('--scores', type=Path, default=RUN_ROOT / 'full_cpu_fp32/scores.jsonl')
    result.add_argument('--thresholds', type=Path, default=Path('corpus_pipeline/detector_calibration/thresholds.json'))
    result.add_argument('--api-decisions', type=Path)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--workers', type=int, default=8)
    result.add_argument('--size-thresholds', type=int, nargs='+', default=list(DEFAULT_THRESHOLDS))
    result.add_argument('--max-bytes', type=int, default=128 * 1024 * 1024)
    result.add_argument('--max-pixels', type=int, default=100_000_000)
    return result


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
