"""Conservative early-training policy with optional explicit task routing.

Image observations are joined by the SHA256 of the original encoded image only.
`origin_confirmed` requires `origin_evidence`; detector/review objects require
`evidence_ids`. A completed uncertain review is usable with an explicit pending
annotation. Missing inference is unfinished work and cannot be published as clear.
"""
from __future__ import annotations

import hashlib
import json
import re

ASSET_ID = re.compile(r'[0-9a-f]{64}\Z')
DEFAULT_CONFIG = {
    'version': 'corpus.unified.early_policy.v1',
    'min_short_side': 128,
    'recaption_min_pixels': 1048576,
    'full_recaption': False,
    'layout_api': False,
    'sscd_candidates_automatically_deleted': False,
}
QUALITY_REASONS = {'blur', 'placeholder', 'corrupt_image', 'bad_caption'}
TASK_POLICY = 'corpus.task_policy.v1'


def config(value=None):
    result = {**DEFAULT_CONFIG, **(value or {})}
    if set(result) - set(DEFAULT_CONFIG) - {'task_routing', 't2i_uncertain'}:
        raise ValueError('Unknown early-policy configuration key')
    if 'task_routing' in result and result['task_routing'] != TASK_POLICY:
        raise ValueError('Unsupported task routing policy')
    if 't2i_uncertain' in result:
        if result.get('task_routing') != TASK_POLICY:
            raise ValueError('t2i_uncertain requires the task-aware policy')
        if result['t2i_uncertain'] not in ('retain_annotated', 'exclude'):
            raise ValueError('Unsupported t2i_uncertain policy')
        # The explicit legacy default and an absent switch are semantically
        # identical. Do not invalidate historical policy hashes for either.
        if result['t2i_uncertain'] == 'retain_annotated':
            result.pop('t2i_uncertain')
    for key in ['min_short_side', 'recaption_min_pixels']:
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError('Policy dimensions must be positive integers')
    if any(result[key] is not False for key in ['full_recaption', 'layout_api', 'sscd_candidates_automatically_deleted']):
        raise ValueError('Early policy does not rewrite all captions, call layout APIs, or delete SSCD candidates')
    return result


def policy_hash(value=None):
    return hashlib.sha256(json.dumps(config(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _evidence(value):
    return isinstance(value, list) and bool(value) and all(isinstance(x, str) and x for x in value)


def _task_route(value, cfg):
    """Only explicit, bound new-policy metadata enables an I2T exemption."""
    if cfg.get('task_routing') != TASK_POLICY or value.get('processing_policy') != TASK_POLICY:
        return None
    route = value.get('task_route')
    if (route not in {'i2t', 't2i', 'mixed', 'text', 'unknown'}
            or type(value.get('image_loss_allowed')) is not bool
            or (route == 'i2t' and value['image_loss_allowed'] is not False)):
        raise ValueError('Invalid task-specific image loss policy')
    return route


def _i2t_image_decision(result, rows, dimensions, quality):
    """Skip visual-origin/quality gates, retaining the actual decode contract."""
    reasons = []
    corrupt = [item for item in quality if item['reason'] == 'corrupt_image']
    if corrupt:
        action, complete = 'exclude', True
        reasons.append('confirmed_corrupt_image')
    else:
        if len(dimensions) != 1:
            reasons.append('conflicting_dimensions_for_exact_asset' if dimensions else 'missing_image_dimensions')
        if any(row.get('error_stage') == 'decode' or row.get('has_invalid_observation') is True
               or row.get('pixel_sha256_conflict') is True
               or row.get('decode_status') in {'error', 'failed', 'corrupt'}
               or row.get('integrity_status') in {'error', 'failed', 'corrupt'} for row in rows):
            reasons.append('image_integrity_unresolved')
        pixels = {row['pixel_sha256'] for row in rows
                  if ASSET_ID.fullmatch(str(row.get('pixel_sha256', '')))}
        if len(pixels) > 1:reasons.append('conflicting_decoded_pixels_for_exact_asset')
        decoded = any((row.get('status') == 'i2t_dedup_only'
                       or (row.get('sscd') or {}).get('status') == 'ok')
                      and ASSET_ID.fullmatch(str(row.get('pixel_sha256', '')))
                      or row.get('decode_status') == 'ok' and _evidence(row.get('decode_evidence'))
                      for row in rows)
        if not decoded:
            reasons.append('image_decode_evidence_missing')
        action, complete = ('pending', False) if reasons else ('retain', True)
        if not reasons:reasons.append('i2t_dedup_only')
    return {**result, 'action': action, 'processing_complete': complete,
            'eligible_for_training': action == 'retain' and complete,
            'review_pending': False, 'needs_detector': False, 'reasons': reasons,
            'conflicts': [reason for reason in reasons if reason.startswith('conflicting_')],
            'processing_policy': TASK_POLICY, 'task_route': 'i2t', 'image_loss_allowed': False}


def quality_decision(value):
    """Validate a scoped, confirmed exclusion; record text errors never spread to images."""
    if value.get('scope') not in {'asset', 'source_record'}:
        raise ValueError('Quality scope must be asset or source_record')
    target = value.get('asset_id') if value['scope'] == 'asset' else value.get('source_record_sha256')
    if not isinstance(target, str) or not target:
        raise ValueError('Quality evidence is missing its exact target')
    if not ASSET_ID.fullmatch(target):
        raise ValueError('Quality evidence needs an exact asset or source-record SHA256')
    if value.get('reason') not in QUALITY_REASONS or value.get('confirmed') is not True or not _evidence(value.get('evidence_ids')):
        raise ValueError('Quality exclusion requires a confirmed reason and evidence')
    if value['reason'] == 'bad_caption' and value['scope'] != 'source_record':
        raise ValueError('A bad caption excludes its record, not other captions using the image')
    return dict(value)


def evaluate_image(observations, policy_config=None):
    cfg = config(policy_config)
    rows = list(observations)
    if not rows:
        raise ValueError('At least one image observation is required')
    assets = {row.get('asset_id') for row in rows}
    if len(assets) != 1 or not ASSET_ID.fullmatch(str(next(iter(assets)))):
        raise ValueError('Only observations of the same exact encoded SHA256 may be combined')
    aid = next(iter(assets))
    collections = sorted({row['collection_id'] for row in rows})
    dimensions = {(r['width'], r['height']) for r in rows
                  if type(r.get('width')) is int and type(r.get('height')) is int and r['width'] > 0 and r['height'] > 0}
    reasons = []; annotations = set(); conflicts = []; origins = set(); confirmed_ai = False
    confirmed_non_ai = False; real_edit = False; clear = False; suspect = False
    uncertain_review = False; final_non_ai = False; protected = False
    strict_t2i = cfg.get('t2i_uncertain') == 'exclude'; incomplete_review = False
    evidence = set(); quality = []
    tags = sorted({tag for row in rows for tag in row.get('layout_source_tags', [])})
    for row in rows:
        origin = row.get('origin', 'unknown')
        origins.add(origin)
        origin_ok = row.get('origin_confirmed') is True and _evidence(row.get('origin_evidence'))
        if origin_ok:
            evidence.update(row['origin_evidence'])
            confirmed_ai |= origin == 'ai_generated'
            protected |= origin in {'programmatic', 'rendered'}
            confirmed_non_ai |= origin in {'programmatic', 'rendered'}
            confirmed_non_ai |= origin == 'natural' and row.get('origin_scope') == 'asset'
        parent = row.get('real_parent_asset_id')
        parent_proven = row.get('real_parent_verified') is True and ASSET_ID.fullmatch(str(parent)) and _evidence(row.get('parent_evidence'))
        if origin == 'ai_edited' and parent_proven:
            real_edit = True; confirmed_non_ai = True; evidence.update(row['parent_evidence'])
        for item in row.get('quality_decisions', []):
            item = quality_decision(item)
            if item['scope'] == 'asset' and item['asset_id'] == aid:
                quality.append(item); evidence.update(item['evidence_ids'])
        detector = row.get('detector') or {}
        if detector.get('status') == 'clear' and _evidence(detector.get('evidence_ids')):
            clear = True; evidence.update(detector['evidence_ids'])
        if detector.get('status') in {'suspect', 'error'}:
            suspect = True
        review = row.get('review') or {}
        status = review.get('status', 'not_requested')
        if status not in {'not_requested', 'pending', 'error', 'uncertain', 'confirmed_ai_generated',
                          'confirmed_natural', 'confirmed_programmatic', 'confirmed_real_edit'}:
            raise ValueError('Unknown review status: ' + str(status))
        if status in {'pending', 'error'}:
            suspect = True
        if strict_t2i and status != 'not_requested' and (
                status in {'pending', 'error'} or not _evidence(review.get('evidence_ids'))
                or review.get('completed', True) is not True):
            incomplete_review = True
        if _evidence(review.get('evidence_ids')) and review.get('completed', True) is True:
            evidence.update(review['evidence_ids'])
            confirmed_ai |= status == 'confirmed_ai_generated'
            if status in {'confirmed_natural', 'confirmed_programmatic'}:
                final_non_ai = True; confirmed_non_ai = True
            if status == 'confirmed_real_edit' and parent_proven:
                real_edit = True; final_non_ai = True; confirmed_non_ai = True
            uncertain_review |= status == 'uncertain'
        if origin == 'ai_edited' and not parent_proven:
            annotations.add('ai_edit_parent_unverified'); suspect = True
    if len(dimensions) > 1:
        conflicts.append('conflicting_dimensions_for_exact_asset')
    if confirmed_ai and confirmed_non_ai:
        conflicts.append('conflicting_confirmed_origin_evidence')
    potential = bool(dimensions and max(w*h for w, h in dimensions) >= cfg['recaption_min_pixels'])
    action = 'pending'; complete = False; review_pending = True; needs_detector = False
    if len(dimensions) == 1 and min(next(iter(dimensions))) < cfg['min_short_side']:
        action = 'exclude'; complete = True; review_pending = False; reasons.append('image_too_small')
    elif quality:
        action = 'exclude'; complete = True; review_pending = False
        reasons.extend('confirmed_' + q['reason'] for q in quality)
    elif conflicts:
        reasons.extend(conflicts)
    elif confirmed_ai:
        action = 'exclude'; complete = True; review_pending = False; reasons.append('confirmed_ai_generated')
    elif not dimensions:
        reasons.append('missing_image_dimensions')
    elif incomplete_review:
        reasons.append('review_evidence_incomplete')
    elif real_edit:
        action = 'retain'; complete = True; review_pending = False
        annotations.add('ai_edited_with_verified_real_parent'); reasons.append('verified_real_image_edit')
    elif protected or final_non_ai:
        action = 'retain'; complete = True; review_pending = False
        reasons.append('confirmed_programmatic_or_rendered' if protected else 'confirmed_non_ai_review')
    elif uncertain_review:
        action = 'exclude' if strict_t2i else 'retain'
        complete = True; review_pending = not strict_t2i
        reasons.append('t2i_uncertain_review_excluded' if strict_t2i else 'completed_review_uncertain')
        annotations.add('origin_uncertain_pending_review')
    elif suspect:
        reasons.append('visual_review_required')
    elif clear:
        action = 'retain'; complete = True; review_pending = False; reasons.append('detector_no_alert_not_certified')
    else:
        needs_detector = True; reasons.append('detector_result_missing')
    if protected:
        annotations.add('programmatic_or_rendered_protected')
    result = {'asset_id': aid, 'collections': collections, 'action': action,
            'processing_complete': complete, 'review_pending': review_pending,
            'eligible_for_training': action == 'retain' and complete,
            'needs_detector': needs_detector, 'reasons': sorted(set(reasons)),
            'annotations': sorted(annotations), 'conflicts': conflicts,
            'evidence_ids': sorted(evidence), 'observed_origins': sorted(origins),
            'recaption_resolution_potential': potential, 'layout_source_tags': tags,
            'recaption_action': 'none', 'layout_api_action': 'none',
            'source_certification': False, 'policy_sha256': policy_hash(cfg)}
    if cfg.get('task_routing') == TASK_POLICY:
        routes = [_task_route(row, cfg) for row in rows]
        by_task = {'t2i': dict(result)}
        if 'i2t' in routes:
            by_task['i2t'] = _i2t_image_decision(result, rows, dimensions, quality)
            if all(route == 'i2t' for route in routes):
                result = dict(by_task['i2t'])
        result['task_decisions'] = by_task
    return result


def evaluate_record(record, image_decisions, quality_decisions=(), policy_config=None):
    """Return selection metadata without editing source text, task type, or image slots."""
    cfg = config(policy_config); ids = list(record.get('image_ids', [])); reasons = []
    quality = [quality_decision(q) for q in quality_decisions]
    bad = [q for q in quality if q['scope'] == 'source_record' and
           q['source_record_sha256'] == record.get('source_record_sha256')]
    decisions = [image_decisions.get(aid) for aid in ids]
    route = _task_route(record, cfg)
    structural = []
    if route == 'i2t':
        bad = [item for item in bad if item['reason'] == 'corrupt_image']
        decisions = [decision.get('task_decisions', {}).get('i2t') if decision else None
                     for decision in decisions]
        structural = list(record.get('issues', [])) + list(record.get('reason_codes', []))
        ignored = {'image_provenance_unknown', 'image_provenance_mixed', 'image_generated_by_ai',
                   'confirmed_ai_generated', 'image_too_small', 'rejected_too_small',
                   't2i_uncertain_review_excluded',
                   'blur', 'placeholder', 'bad_caption', 'confirmed_blur',
                   'confirmed_placeholder', 'confirmed_bad_caption'}
        structural = sorted(set(structural) - ignored)
        if record.get('binding_valid') is False or record.get('integrity_valid') is False:
            structural.append('record_binding_or_integrity_invalid')
    else:
        # A later generation/legacy occurrence must not inherit the convenient
        # top-level I2T result of an image previously observed only in I2T.
        decisions = [decision.get('task_decisions', {}).get('t2i', decision) if decision else None
                     for decision in decisions]
    if structural:
        action = 'pending'; complete = False; reasons.extend(structural)
    elif bad or any(d and d['action'] == 'exclude' for d in decisions):
        action = 'exclude'; complete = True
        reasons.extend('confirmed_' + q['reason'] for q in bad)
        if any(d and d['action'] == 'exclude' for d in decisions): reasons.append('excluded_exact_image_asset')
    elif any(d is None or not d['processing_complete'] for d in decisions):
        action = 'pending'; complete = False; reasons.append('image_screening_incomplete')
    else:
        action = 'retain'; complete = True; reasons.append('retain_original_supervision')
    result = {'record_id': record['record_id'], 'action': action, 'processing_complete': complete,
            'review_pending': any(d and d['review_pending'] for d in decisions),
            'reasons': reasons, 'text_origin': record.get('text_origin', 'unknown'),
            'ai_text_annotated': record.get('text_origin') in {'ai_generated', 'synthetic', 'ai'},
            'task_type': record.get('task_type', record.get('category')),
            'native_format': record.get('native_format', record.get('format')),
            'ordered_image_ids': ids, 'source_record_modified': False,
            'policy_sha256': policy_hash(cfg)}
    if route is not None:
        result.update(processing_policy=TASK_POLICY, task_route=route,
                      image_loss_allowed=record['image_loss_allowed'])
    return result
