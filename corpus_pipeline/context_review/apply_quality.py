"""Publish a bounded machine quality-policy revision without altering human history.

This applies the new visible-blur policy only to the frozen 688-image review
queue. The full overlay retains the other 69,804 rows exactly as logical JSON.
It is an image-screening index, never a final training release or source proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import gzip
from pathlib import Path
import shutil
import tempfile

from corpus_pipeline.codex_review.farm import atomic_json, exclusive, file_sha
from corpus_pipeline.codex_review.human_import import (
    CATEGORIES, fsync_dir, fsync_file, json_rows, read_bound, strict_json,
    verify_version as verify_human_version, write_line,
)
from corpus_pipeline.codex_review.report import digest, require
from corpus_pipeline.context_review import blur_worker

SCHEMA = 'corpus.quality_policy.application.v1'
BUNDLE_SCHEMA = 'corpus.quality_policy.bundle.v1'
DECISION_SCHEMA = 'corpus.quality_policy.decision.v1'
AUTHORITY = 'user_policy_applied_by_model'
SCOPE = 688
CONTEXT_ACTIONS = {'keep', 'flag_low_information', 'exclude_pair', 'drop_image_keep_text',
                   'needs_context', 'retain_caption_prune_invalid_qa'}


def artifact(path, records=None):
    path = Path(path).resolve(strict=True)
    result = {'path': str(path), 'sha256': file_sha(path), 'bytes': path.stat().st_size}
    if records is not None:
        result['records'] = records
    return result


def check_binding(binding):
    require(isinstance(binding, dict) and isinstance(binding.get('path'), str)
            and isinstance(binding.get('sha256'), str), 'missing artifact binding')
    path = Path(binding['path']).resolve(strict=True)
    require(file_sha(path) == binding['sha256'], 'artifact SHA mismatch: ' + str(path))
    if 'bytes' in binding:
        require(path.stat().st_size == binding['bytes'], 'artifact size mismatch')
    return path


def validate_decisions(rows, candidates):
    """Bind every proposed rule application to a unique original source image."""
    expected = {row['item_id']: row for row in candidates}
    require(len(expected) == len(candidates), 'duplicate candidate ID')
    require(len({row['review_number'] for row in candidates}) == len(candidates),
            'duplicate candidate review number')
    decisions = {}
    mandatory = {'schema_version', 'item_id', 'review_number', 'source_ref', 'source_input_sha256',
                 'blur_status', 'context_action', 'image_origin_followup', 'reason', 'authority'}
    for row in rows:
        require(isinstance(row, dict) and mandatory <= set(row), 'missing policy decision fields')
        aid = row['item_id']
        require(isinstance(aid, str) and aid in expected and aid not in decisions,
                'unknown or duplicate policy image ID')
        source = expected[aid]
        require(row['schema_version'] == DECISION_SCHEMA and row['authority'] == AUTHORITY,
                'policy decision authority/schema differs')
        require(type(row['review_number']) is int and row['review_number'] == source['review_number'],
                'policy review number does not match image ID')
        require(row['source_ref'] == source['source_ref'] and row['source_ref'].get('sha256') == aid
                and row['source_input_sha256'] == digest(source), 'policy source binding differs')
        require(row['blur_status'] in {'clear', 'blur_present', 'uncertain'}
                and row['context_action'] in CONTEXT_ACTIONS, 'invalid policy action')
        require(type(row['image_origin_followup']) is bool, 'invalid source followup flag')
        require(isinstance(row['reason'], str) and row['reason'].strip(), 'missing policy reason')
        decisions[aid] = deepcopy(row)
    require(set(decisions) == set(expected), 'policy decisions must cover the exact candidate set')
    return decisions


def resolve_policy_row(baseline, policy, reference):
    """Keep source/AI precedence separate from blur and contextual quality."""
    if policy is None:
        return deepcopy(baseline)
    require(policy['item_id'] == baseline['item_id'], 'policy applies to another image')
    require(policy['source_ref'] == baseline['model_screening']['source_ref'],
            'baseline/policy source reference differs')
    human = baseline.get('human_review')
    require(human is not None and human['review_number'] == policy['review_number'],
            'policy image was not in the original human queue')
    authority = AUTHORITY
    basis = 'new_quality_policy'
    if baseline['report_category'] == 'ai_excluded' or human['decision'] == 'exclude_ai':
        decision, category, reason = 'reject', 'ai_excluded', baseline['rejection_reason']
        authority = baseline['decision_authority']
        basis = 'existing_ai_exclusion_preserved'
    elif policy['blur_status'] == 'blur_present':
        decision, category, reason = 'reject', 'quality_excluded', 'visible_blur_excluded'
    elif policy['blur_status'] == 'uncertain':
        decision, category, reason = 'pending', 'visual_pending', 'blur_review_uncertain'
    elif policy['image_origin_followup']:
        decision, category, reason = 'pending', 'visual_pending', 'image_origin_followup_required'
    elif policy['context_action'] in {'exclude_pair', 'drop_image_keep_text'}:
        decision, category, reason = 'reject', 'quality_excluded', 'context_quality_excluded'
    elif policy['context_action'] in {'needs_context', 'retain_caption_prune_invalid_qa'}:
        decision, category, reason = 'pending', 'visual_pending', 'context_repair_or_review_required'
    else:
        require(policy['blur_status'] == 'clear' and policy['context_action'] in {'keep', 'flag_low_information'},
                'unhandled quality action')
        decision, category, reason = 'admit', 'admitted', 'none'
    # Keep the familiar complete model/human fields, plus an exact prior envelope.
    result = deepcopy(baseline)
    result.update(schema_version='corpus.quality_policy.resolved_overlay.v1',
                  decision=decision, report_category=category, rejection_reason=reason,
                  decision_authority=authority, authority=AUTHORITY, decision_basis=basis,
                  review_number=policy['review_number'], source_ref=deepcopy(policy['source_ref']),
                  blur_status=policy['blur_status'], context_recommendation=policy['context_action'],
                  image_origin_followup=policy['image_origin_followup'], reason=policy['reason'],
                  baseline_screening=deepcopy(baseline), policy_input=deepcopy(policy),
                  quality_policy_manifest=deepcopy(reference), source_certification=False)
    return result


def validate_blur_adjudication(entry, candidate, raw_result):
    require(isinstance(entry, dict) and entry.get('item_id') == candidate['item_id']
            and entry.get('source_input_sha256') == digest(candidate)
            and entry.get('raw_blur_sha256') == digest(raw_result), 'independent blur adjudication binding differs')
    require(entry.get('full_original_image_actually_viewed') is True
            and entry.get('blur_status') in {'clear', 'blur_present', 'uncertain'}
            and isinstance(entry.get('reason'), str) and entry['reason'].strip(),
            'independent blur adjudication lacks actual image review')
    evidence = entry.get('evidence')
    require(isinstance(evidence, list) and len(evidence) > 0, 'independent blur adjudication lacks evidence')
    for cue in evidence:
        require(isinstance(cue, dict) and ((isinstance(cue.get('location'), str) and cue['location'].strip()
                and isinstance(cue.get('observation'), str) and cue['observation'].strip())
                or isinstance(cue.get('artifact'), dict)), 'invalid independent blur evidence')
        if 'artifact' in cue:
            check_binding(cue['artifact'])


def load_blur_export(binding, candidates, decisions, adjudications=None):
    """Reuse the frozen strict export, and verify each decision's blur label."""
    manifest_path = check_binding(binding)
    manifest, _ = read_bound(manifest_path, binding['sha256'])
    run_root = manifest_path.parent.parent.parent
    complete, complete_binding = read_bound(run_root / 'COMPLETE.json')
    require(complete.get('complete') is True and complete.get('records') == SCOPE
            and complete.get('audit_mode') == 'strict' and complete.get('skipped_gates') == []
            and Path(complete['manifest']).resolve() == manifest_path
            and complete['manifest_sha256'] == binding['sha256'], 'blur export not strictly complete')
    audit, audit_binding = read_bound(complete['audit'], complete['audit_sha256'])
    require(audit.get('passed') is True and audit.get('mode') == 'strict'
            and audit.get('skipped_gates') == [] and audit.get('export_checked') is True
            and audit.get('complete_coverage') is True
            and audit.get('all_original_image_scope_checked') is True
            and audit.get('export_manifest_sha256') == binding['sha256'], 'strict blur audit failed')
    require(manifest.get('complete') is True and manifest.get('audit_mode') == 'strict'
            and manifest.get('records') == manifest.get('outputs') == manifest['source']['count'] == SCOPE,
            'blur export coverage differs')
    inputs = {'blur_completion': complete_binding, 'blur_audit': audit_binding,
              'blur_source': artifact(check_binding(manifest['source']))}
    source = {row['item_id']: row for row in candidates}
    adjudications = adjudications or {}
    used_adjudications = set()
    seen = set()
    for index, shard in enumerate(manifest['shards']):
        path = manifest_path.parent / shard['path']
        require(path.resolve().parent == manifest_path.parent, 'invalid blur shard path')
        check_binding({'path': str(path), 'sha256': shard['sha256']})
        inputs['blur_shard_' + str(index)] = artifact(path)
        count = 0
        for row in json_rows(path):
            aid = row['item_id']
            require(aid in source and aid not in seen and row['source'] == source[aid],
                    'blur export source identity differs')
            blur_worker.validate_result(row['result'], aid, source[aid])
            override = decisions[aid].get('blur_adjudication')
            if override is None:
                require(row['result']['blur_status'] == decisions[aid]['blur_status'],
                        'policy blur label differs from actual strict inference')
            else:
                require(adjudications.get(aid) == override, 'blur override not in bound adjudication artifact')
                validate_blur_adjudication(override, source[aid], row['result'])
                require(override['blur_status'] == decisions[aid]['blur_status'],
                        'policy blur label differs from independent adjudication')
                used_adjudications.add(aid)
                for evidence_index, cue in enumerate(override['evidence']):
                    if 'artifact' in cue:
                        inputs['adjudication_evidence_' + aid + '_' + str(evidence_index)] = artifact(
                            check_binding(cue['artifact']))
            seen.add(aid)
            count += 1
        require(count == shard['records'] == shard['outputs'], 'blur shard count differs')
    require(seen == set(source), 'blur export exact image coverage differs')
    require(used_adjudications == set(adjudications), 'unused or missing independent blur adjudication')
    return inputs


def load_inputs(root, manifest_path):
    manifest, manifest_binding = read_bound(manifest_path)
    require(manifest.get('schema_version') == BUNDLE_SCHEMA and manifest.get('records') == SCOPE
            and isinstance(manifest.get('user_policy'), str) and '模糊' in manifest['user_policy'],
            'invalid quality bundle policy or scope')
    inputs = {'input_manifest': manifest_binding}
    for key in ('input_human_current', 'candidates', 'strict_blur_manifest', 'decisions'):
        inputs[key] = artifact(check_binding(manifest[key]))
    require(Path(inputs['input_human_current']['path']) == root / 'HUMAN_REVIEW_CURRENT.json',
            'bundle binds another human pointer')
    current, _ = read_bound(inputs['input_human_current']['path'], inputs['input_human_current']['sha256'])
    human_dir = root / 'human_reviews' / current['submission_sha256']
    human_report, human_pointer = verify_human_version(human_dir, current['submission_sha256'])
    require(current == human_pointer, 'current human version binding differs')
    baseline_path = human_dir / 'resolved_screening_overlay.jsonl.gz'
    inputs['baseline_overlay'] = artifact(baseline_path, human_report['model_scope_records'])
    for key in ('report', 'completion'):
        inputs['human_' + key] = current[key]
    candidates = list(json_rows(inputs['candidates']['path']))
    require(len(candidates) == manifest['candidates'].get('records') == SCOPE
            and manifest['decisions'].get('records') == SCOPE, 'quality candidate scope differs')
    decisions = validate_decisions(json_rows(inputs['decisions']['path']), candidates)
    adjudications = {}
    if manifest.get('blur_adjudication') is not None:
        inputs['blur_adjudication'] = artifact(check_binding(manifest['blur_adjudication']))
        adjudication_artifact, _ = read_bound(inputs['blur_adjudication']['path'], inputs['blur_adjudication']['sha256'])
        require(isinstance(adjudication_artifact.get('entries'), list), 'invalid blur adjudication artifact')
        for entry in adjudication_artifact['entries']:
            require(isinstance(entry, dict) and entry.get('item_id') in decisions
                    and entry['item_id'] not in adjudications, 'unknown or duplicate independent blur adjudication')
            adjudications[entry['item_id']] = entry
    inputs.update(load_blur_export(manifest['strict_blur_manifest'], candidates, decisions, adjudications))
    candidate_map = {row['item_id']: row for row in candidates}
    # These three source-record contexts explicitly introduced unresolved AI
    # experiment lineage. A quality-only pass cannot silently certify it.
    for row in candidates:
        if row['review_number'] in {478, 489, 548}:
            require(decisions[row['item_id']]['image_origin_followup'] is True,
                    'known AI-experiment source followup was silently cleared')
    return manifest, inputs, human_report, candidate_map, decisions


def verify_version(directory, revision_sha):
    directory = Path(directory).resolve(strict=True)
    complete, complete_binding = read_bound(directory / 'COMPLETE.json')
    require(complete.get('schema_version') == SCHEMA and complete.get('complete') is True
            and complete.get('revision_sha256') == revision_sha, 'quality completion differs')
    report, report_binding = read_bound(directory / 'REPORT.json', complete['report_sha256'])
    require(report['revision_sha256'] == revision_sha and report['input_manifest']['sha256'] == revision_sha,
            'quality revision identity differs')
    for name, binding in report['artifacts'].items():
        require(Path(name).name == name, 'invalid quality artifact filename')
        check_binding({'path': str(directory / name), **binding})
    audit, _ = read_bound(directory / 'AUDIT.json', report['artifacts']['AUDIT.json']['sha256'])
    require(audit.get('passed') is True and audit.get('skipped_gates') == [], 'quality application audit failed')
    pointer = {'schema_version': SCHEMA, 'directory': str(directory), 'revision_sha256': revision_sha,
               'input_manifest': report['input_manifest'], 'report': report_binding, 'completion': complete_binding,
               'input_human_current': report['input_bindings']['input_human_current'],
               'quality_decisions': {'path': str(directory / 'quality_decisions.jsonl.gz'),
                                     **report['artifacts']['quality_decisions.jsonl.gz']},
               'resolved_screening_overlay': {'path': str(directory / 'resolved_screening_overlay.jsonl.gz'),
                                               **report['artifacts']['resolved_screening_overlay.jsonl.gz']},
               'training_dataset_published': False, 'source_certification': False}
    return report, pointer


def apply_quality(root, manifest_path, supersedes=None):
    root = Path(root).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    with exclusive(root):
        manifest, inputs, human_report, candidates, decisions = load_inputs(root, manifest_path)
        revision_sha = inputs['input_manifest']['sha256']
        parent = root / 'quality_policy_versions'
        parent.mkdir(exist_ok=True)
        directory = parent / revision_sha
        current_path = root / 'QUALITY_POLICY_CURRENT.json'
        previous = None
        if current_path.exists():
            previous, _ = read_bound(current_path)
            old_report, old_pointer = verify_version(parent / previous['revision_sha256'], previous['revision_sha256'])
            require(previous == old_pointer, 'current quality pointer binding differs')
            if previous['revision_sha256'] != revision_sha:
                require(supersedes == previous['revision_sha256'], 'another quality revision requires explicit supersedes')
        else:
            require(supersedes is None, 'quality supersedes has no current parent')
        if directory.exists():
            report, pointer = verify_version(directory, revision_sha)
            require(previous is None or previous['revision_sha256'] == revision_sha,
                    'cannot reactivate a historical revision through an idempotent retry')
            for binding in inputs.values():
                check_binding(binding)
            atomic_json(current_path, pointer)
            fsync_dir(root)
            return report
        temporary = Path(tempfile.mkdtemp(prefix='.apply-', dir=parent))
        try:
            (temporary / 'input_manifest.json').write_bytes(manifest_path.read_bytes())
            shutil.copyfile(inputs['decisions']['path'], temporary / 'input_decisions.jsonl.gz')
            reference = inputs['input_manifest']
            filenames = ('resolved_screening_overlay.jsonl.gz', 'quality_decisions.jsonl.gz')
            counts, before = Counter({k: 0 for k in CATEGORIES}), Counter({k: 0 for k in CATEGORIES})
            selected_counts = Counter({k: 0 for k in CATEGORIES})
            expected, selected, unchanged = {}, set(), set()
            with ExitStack() as stack:
                outputs = {name: stack.enter_context(gzip.GzipFile(filename='', fileobj=stack.enter_context(
                    (temporary / name).open('wb')), mode='wb', compresslevel=1, mtime=0)) for name in filenames}
                for row in json_rows(inputs['baseline_overlay']['path']):
                    aid = row['item_id']
                    require(aid not in expected and row['report_category'] in CATEGORIES, 'invalid baseline identity/category')
                    before[row['report_category']] += 1
                    policy = decisions.get(aid)
                    if policy is not None:
                        require(row['human_review'] == candidates[aid]['human_review'], 'baseline human decision changed')
                        selected.add(aid)
                    else:
                        unchanged.add(aid)
                    resolved = resolve_policy_row(row, policy, reference)
                    counts[resolved['report_category']] += 1
                    expected[aid] = digest(resolved)
                    write_line(outputs[filenames[0]], resolved)
                    if policy is not None:
                        selected_counts[resolved['report_category']] += 1
                        write_line(outputs[filenames[1]], resolved)
            require(before == human_report['resolved_counts'] and len(expected) == human_report['model_scope_records'],
                    'baseline coverage or category count differs')
            require(selected == set(decisions) and len(selected) == SCOPE, 'policy IDs missing from baseline')
            # Reopen every complete output row. Equality hashes cover all nested
            # history, not merely the decision labels or aggregate counts.
            for name in filenames:
                seen = set()
                for row in json_rows(temporary / name):
                    aid = row['item_id']
                    require(aid not in seen and expected.get(aid) == digest(row), 'quality output content differs')
                    seen.add(aid)
                require(seen == (set(expected) if name == filenames[0] else selected), 'quality output coverage differs')
            for binding in inputs.values():
                check_binding(binding)
            audit = {'schema_version': SCHEMA, 'passed': True, 'skipped_gates': [],
                     'records': len(expected), 'policy_records': len(selected), 'unchanged_records': len(unchanged),
                     'independent_blur_adjudication_ids': sorted(aid for aid, decision in decisions.items()
                                                               if decision.get('blur_adjudication') is not None),
                     'checks': ['exact_original_candidate_set', 'strict_blur_export_bound_by_hash',
                                'each_blur_label_matches_actual_inference_or_bound_independent_adjudication',
                                'human_current_and_baseline_hash_bound',
                                'source_ref_and_candidate_hash_per_item', 'existing_ai_exclusions_preserved',
                                'all_other_rows_logically_unchanged', 'complete_baseline_history_retained',
                                'all_output_rows_reopened_and_digest_verified', 'all_inputs_unchanged'],
                     'input_bindings': inputs}
            atomic_json(temporary / 'AUDIT.json', audit)
            combined = deepcopy(human_report.get('combined_qwen_population'))
            if combined is not None:
                check_binding(combined['source_snapshot'])
                combined['counts'] = {key: combined['counts'][key] - before[key] + counts[key] for key in CATEGORIES}
                require(sum(combined['counts'].values()) == combined['population'], 'combined rollup denominator differs')
                combined['semantics'] = 'Arithmetic screened-population rollup; only the 688-image review subset received this new blur gate. Not a training release.'
            report = {'schema_version': SCHEMA, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                      'revision_sha256': revision_sha, 'input_manifest': reference, 'input_bindings': inputs,
                      'previous_quality_revision': previous, 'user_policy': manifest['user_policy'],
                      'model_scope_records': len(expected), 'policy_records': len(selected),
                      'unchanged_records': len(unchanged), 'counts_before': dict(before), 'resolved_counts': dict(counts),
                      'policy_decision_counts': dict(selected_counts),
                      'combined_qwen_population': combined, 'source_images_modified': False,
                      'human_history_modified': False, 'model_results_modified': False,
                      'training_dataset_published': False, 'source_certification': False,
                      'policy_coverage': '688 reviewed image assets only; 842426 population is arithmetic, not newly blur-screened.',
                      'import_code': artifact(__file__)}
            report['artifacts'] = {name: {'sha256': file_sha(temporary / name),
                                         'bytes': (temporary / name).stat().st_size,
                                         **({'records': len(expected) if name == filenames[0] else SCOPE}
                                            if name in filenames else {})}
                                   for name in (*filenames, 'input_manifest.json', 'input_decisions.jsonl.gz', 'AUDIT.json')}
            atomic_json(temporary / 'REPORT.json', report)
            atomic_json(temporary / 'COMPLETE.json', {'schema_version': SCHEMA, 'complete': True,
                'revision_sha256': revision_sha, 'report_sha256': file_sha(temporary / 'REPORT.json'),
                'meaning': 'quality policy overlay applied to reviewed subset, not a final training release'})
            for path in temporary.iterdir():
                fsync_file(path)
            fsync_dir(temporary)
            temporary.rename(directory)
            fsync_dir(parent)
            report, pointer = verify_version(directory, revision_sha)
            for binding in inputs.values():
                check_binding(binding)
            atomic_json(current_path, pointer)
            fsync_dir(root)
            return report
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--supersedes', help='Exact current machine-policy revision SHA for a new revision')
    args = parser.parse_args()
    result = apply_quality(args.root, args.manifest, args.supersedes)
    print(__import__('json').dumps({key: result[key] for key in
        ('revision_sha256', 'model_scope_records', 'policy_records', 'resolved_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
