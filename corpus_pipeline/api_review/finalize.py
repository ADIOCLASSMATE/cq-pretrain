"""Combine two audit rounds without converting model agreement into ground truth."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from PIL import Image


def apply_chartx_mapping(blind, mapping):
    """Apply reviewed author provenance only after rechecking exact local pixels."""
    if blind.get('source') != 'chartx' or mapping.get('upstream_repository') != 'InternScience/ChartX':
        raise ValueError('Source mapping is outside the reviewed ChartX provenance rule')
    if mapping.get('image_sha256') != blind['image_sha256'] or mapping.get('metadata_match_count') != 1:
        raise ValueError('Ambiguous or mismatched provenance mapping')
    matches = mapping.get('matches', [])
    if len(matches) != 1 or not matches[0].get('rgba_pixel_equal'):
        raise ValueError('Exact pixel provenance evidence required')
    match = matches[0]
    original = Path(blind['image_path']).read_bytes()
    upstream_path = Path(match['upstream_image_path'])
    if hashlib.sha256(original).hexdigest() != blind['image_sha256']:
        raise ValueError('Local image changed since review')
    if hashlib.sha256(upstream_path.read_bytes()).hexdigest() != match['upstream_image_sha256']:
        raise ValueError('Upstream evidence image changed')
    with Image.open(blind['image_path']) as local, Image.open(upstream_path) as upstream:
        if local.size != upstream.size or local.convert('RGBA').tobytes() != upstream.convert('RGBA').tobytes():
            raise ValueError('Upstream and local pixels differ')
    return {**blind, 'prior_source_rule_decision': blind['source_rule_decision'],
            'source_rule_decision': 'protected_source', 'source_image_origin': 'programmatic',
            'verified_provenance': {
                'type': 'exact_rgba_pixel_match_to_official_ChartX_archive',
                'repository': mapping['upstream_repository'], 'revision': mapping['upstream_revision'],
                'image_name': match['imgname'], 'archive_member': match['archive_member'],
                'source_url': match['source_url'], 'metadata_url': match['metadata_url'],
                'upstream_sha256': match['upstream_image_sha256'],
                'production_method_evidence': 'https://arxiv.org/html/2402.12185#S3.SS3',
                'data_and_code_origin': 'ai_generated_or_assisted',
                'text_origin': 'ai_generated_or_assisted',
            }}


def decide(blind, second=None):
    rule = blind['source_rule_decision']
    first = blind['review_status']
    final = second['review_status'] if second else first
    reason = ''
    if rule in {'protected_source', 'source_reject'}:
        status, reason = rule, 'Registered provenance takes precedence over visual opinions.'
    elif final == 'api_incomplete' or first == 'api_incomplete':
        status, reason = 'api_incomplete', 'Missing or invalid review is an operational failure, not uncertainty.'
    elif final == 'unanimous_non_ai_candidate':
        prior_strong_ai = any(
            r.get('parsed', {}).get('assessment') == 'likely_ai_generated'
            and r.get('parsed', {}).get('confidence') == 'high'
            for r in blind['reviews'].values())
        if second and prior_strong_ai:
            status, reason = 'cannot_determine', 'Second-round reversal of a strong AI claim needs a human to check the claimed cue.'
        else:
            status, reason = 'likely_non_ai_candidate', 'Three reviewers lean non-AI; this is not proof of production history.'
    elif first == final == 'unanimous_high_ai_candidate' and second:
        status, reason = 'likely_ai_generated_candidate', 'Strong agreement persisted after evidence review; still an unvalidated exclusion candidate.'
    else:
        status, reason = 'cannot_determine', 'Evidence or reviewer agreement is insufficient; retain an explicit unknown label.'
    return {
        **{key: blind.get(key) for key in (
            'image_sha256', 'image_path', 'review_id', 'source', 'domain', 'pilot_group',
            'source_image_origin', 'source_rule_decision')},
        'prior_source_rule_decision': blind.get('prior_source_rule_decision'),
        'verified_provenance': blind.get('verified_provenance'),
        'final_disposition': status, 'reason': reason,
        'blind_review_status': first,
        'adjudication_review_status': second['review_status'] if second else 'not_requested',
        'manual_review_required': status in {'cannot_determine', 'likely_ai_generated_candidate'},
        'operational_retry_required': first == 'api_incomplete' or final == 'api_incomplete',
        'automatic_deletion': False, 'training_eligibility': None,
        'verified_real_source_edit': 'Preserve and annotate only with verified parent-image/edit provenance; pixels cannot establish it.',
        'adjudication': second.get('reviews') if second else None,
        'rule_version': 'api_review_pilot_final_v1_unvalidated',
    }


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finalize(blind, adjudicated):
    second = {row['image_sha256']: row for row in adjudicated}
    if len(second) != len(adjudicated):
        raise ValueError('Duplicate second-round decision')
    first_shas = {row['image_sha256'] for row in blind}
    if len(first_shas) != len(blind) or not set(second) <= first_shas:
        raise ValueError('Duplicate or unmatched image decision')
    rows = [decide(row, second.get(row['image_sha256'])) for row in blind]
    return rows, {
        'total_images': len(rows), 'second_round_images': len(second),
        'counts': dict(Counter(row['final_disposition'] for row in rows)),
        'by_pilot_group': {
            group: dict(Counter(r['final_disposition'] for r in rows if r['pilot_group'] == group))
            for group in sorted({r['pilot_group'] for r in rows})},
        'human_queue_count': sum(r['manual_review_required'] for r in rows),
        'operational_retry_count': sum(r['operational_retry_required'] for r in rows),
        'automatic_deletion': False, 'training_eligibility': None,
        'accuracy_precision_recall': None,
        'scope': 'Small exploratory pilot. Dependent second-round judgments are not additional independent votes.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blind-decisions', required=True)
    p.add_argument('--adjudication-decisions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--chartx-mapping', help='Reviewed exact upstream ChartX pixel mappings; revalidated locally')
    args = p.parse_args()
    blind = read_rows(args.blind_decisions)
    if args.chartx_mapping:
        mappings = read_rows(args.chartx_mapping)
        index = {r['image_sha256']: r for r in mappings}
        if len(index) != len(mappings) or not set(index) <= {r['image_sha256'] for r in blind}:
            raise ValueError('Duplicate or unmatched ChartX mapping')
        blind = [apply_chartx_mapping(r, index[r['image_sha256']]) if r['image_sha256'] in index else r for r in blind]
    rows, stats = finalize(blind, read_rows(args.adjudication_decisions))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, records in [('decisions.jsonl', rows), ('human_queue.jsonl', [r for r in rows if r['manual_review_required']]),
                          ('retry_queue.jsonl', [r for r in rows if r['operational_retry_required']])]:
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    stats['input_sha256'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in (
        ('blind_decisions', args.blind_decisions), ('adjudication_decisions', args.adjudication_decisions), ('finalizer', __file__))}
    if args.chartx_mapping:
        stats['input_sha256']['chartx_mapping'] = hashlib.sha256(Path(args.chartx_mapping).read_bytes()).hexdigest()
    stats['new_exact_source_matches'] = sum(r['verified_provenance'] is not None for r in rows)
    (out / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
