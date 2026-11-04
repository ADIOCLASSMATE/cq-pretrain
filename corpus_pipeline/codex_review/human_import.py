"""Import explicit human decisions as an audited, additive screening overlay.

The source model export, SQLite state, review package and images remain intact.
Only an atomically published human version and its CURRENT pointer are written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .farm import atomic_json, exclusive, file_sha
from .report import canonical, digest, require

SCHEMA = 'corpus.human_review.import.v1'
CHOICES = {
    'keep': ('admit', 'admitted', 'none'),
    'exclude_ai': ('reject', 'ai_excluded', 'ai_generated'),
    'exclude_quality': ('reject', 'quality_excluded', 'quality_unusable'),
    'unsure': ('pending', 'visual_pending', 'none'),
}
CATEGORIES = ('admitted', 'ai_excluded', 'quality_excluded', 'visual_pending',
              'technical_failure', 'unprocessed')


def strict_json(raw):
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('nonfinite JSON number: ' + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)


def timestamp(value):
    require(isinstance(value, str), 'timestamp must be a string')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            'timestamp must include a timezone')
    return parsed


def validate_submission(payload, queue_id, queue_rows):
    require(isinstance(payload, dict) and set(payload) == {
        'schema_version', 'queue_id', 'queue_count', 'exported_at', 'decisions'},
        'unexpected submission fields')
    require(payload['schema_version'] == 'corpus.human_review.decisions.v1', 'wrong submission schema')
    require(payload['queue_id'] == queue_id, 'wrong review queue')
    require(type(payload['queue_count']) is int and payload['queue_count'] == len(queue_rows),
            'wrong review queue count')
    exported_at = timestamp(payload['exported_at'])
    require(isinstance(payload['decisions'], list), 'decisions must be an array')
    queue = {row['item_id']: row['review_number'] for row in queue_rows}
    require(len(queue) == len(queue_rows) and len(set(queue.values())) == len(queue),
            'duplicate review queue identity')
    decisions = {}
    for row in payload['decisions']:
        require(isinstance(row, dict) and set(row) == {
            'item_id', 'review_number', 'decision', 'note', 'updated_at'}, 'unexpected decision fields')
        aid = row['item_id']
        require(isinstance(aid, str) and aid in queue and aid not in decisions, 'unknown or duplicate image ID')
        require(type(row['review_number']) is int and row['review_number'] == queue[aid],
                'review number does not match image ID')
        require(isinstance(row['decision'], str) and row['decision'] in CHOICES, 'unknown human decision')
        require(isinstance(row['note'], str) and len(row['note']) <= 100000, 'invalid human note')
        require(timestamp(row['updated_at']) <= exported_at, 'decision timestamp follows export')
        decisions[aid] = deepcopy(row)
    return decisions


def resolve_row(model_row, human_row, submission_reference):
    decision = tuple(model_row[key] for key in ('decision', 'report_category', 'rejection_reason'))
    if human_row is not None:
        require(human_row['item_id'] == model_row['item_id'], 'human decision belongs to another image')
        require(model_row['report_category'] == 'visual_pending' and model_row['decision'] == 'pending',
                'human queue cannot override nonpending model result')
        require(human_row['decision'] in CHOICES, 'unknown human decision')
        decision = CHOICES[human_row['decision']]
    return {
        'schema_version': 'corpus.human_review.resolved_overlay.v1',
        'item_id': model_row['item_id'],
        'decision': decision[0], 'report_category': decision[1], 'rejection_reason': decision[2],
        'decision_authority': 'human_user' if human_row is not None else 'codex_model',
        'model_screening': deepcopy(model_row), 'human_review': deepcopy(human_row),
        'human_submission': deepcopy(submission_reference) if human_row is not None else None,
        'source_certification': False,
    }


def validate_correction(previous_decisions, new_decisions, evidence, parent_sha):
    require(isinstance(evidence, dict)
            and evidence.get('schema_version') == 'corpus.human_review.chat_confirmation.v1'
            and evidence.get('parent_submission_sha256') == parent_sha, 'correction parent evidence differs')
    require(set(previous_decisions) == set(new_decisions), 'correction cannot silently add or drop decisions')
    changes = evidence.get('changes')
    require(isinstance(changes, list) and len(changes) > 0, 'correction requires explicit changes')
    confirmed = {}
    for change in changes:
        require(isinstance(change, dict) and set(change) == {
            'item_id', 'review_number', 'from_decision', 'to_decision', 'user_message'},
            'unexpected correction fields')
        aid = change['item_id']
        require(isinstance(aid, str) and aid in previous_decisions and aid not in confirmed,
                'unknown or repeated corrected image')
        old, new = previous_decisions[aid], new_decisions[aid]
        require(type(change['review_number']) is int
                and change['review_number'] == old['review_number'] == new['review_number']
                and old['item_id'] == new['item_id'] == aid, 'correction image identity differs')
        require(change['from_decision'] == old['decision'] and change['to_decision'] == new['decision']
                and old['decision'] != new['decision'] and new['decision'] in CHOICES,
                'correction decision transition differs')
        require(isinstance(change['user_message'], str) and change['user_message'].strip(),
                'missing explicit user confirmation')
        allowed = {'decision', 'updated_at', 'note'}
        require({key: value for key, value in old.items() if key not in allowed}
                == {key: value for key, value in new.items() if key not in allowed},
                'correction modified identity fields')
        confirmed[aid] = change
    require({aid for aid in previous_decisions if previous_decisions[aid] != new_decisions[aid]} == set(confirmed),
            'unconfirmed decision changes')
    return deepcopy(changes)


def read_bound(path, expected=None):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    require(expected is None or expected == sha, 'artifact SHA mismatch: ' + str(path))
    return strict_json(raw), {'path': str(path), 'sha256': sha}


def json_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield strict_json(line)


def write_line(stream, row):
    stream.write((canonical(row) + '\n').encode('utf-8'))


def fsync_file(path):
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_context(root):
    package, package_binding = read_bound(root / 'manual_review_ui/MANIFEST.json')
    report, report_binding = read_bound(root / 'final_report.json', package['source_report_sha256'])
    receipt, receipt_binding = read_bound(root / 'COMPLETE.json', report['completion_receipt']['sha256'])
    require(receipt == package['source_complete'] and receipt['complete'] is True
            and receipt['audit_mode'] == 'strict' and receipt['skipped_gates'] == []
            and report['complete'] is True and report['strict_audit_passed'] is True,
            'source model review is not strictly complete')
    bindings = {'review_package': package_binding, 'model_report': report_binding,
                'model_completion': receipt_binding}
    for key in ('audit', 'content_audit', 'export_manifest'):
        entry = report[key]
        value, bindings[key] = read_bound(entry['path'], entry['sha256'])
        if key != 'export_manifest':
            require(value['passed'] is True and value['mode'] == 'strict' and value['skipped_gates'] == [],
                    'source audit failed')
        if key == 'audit':
            require(entry['sha256'] == receipt['audit_sha256']
                    and value['export_checked'] is True
                    and value['export_manifest_sha256'] == receipt['manifest_sha256'], 'audit binding differs')
        elif key == 'export_manifest':
            require(entry['sha256'] == receipt['manifest_sha256'] and value['records'] == report['expected'],
                    'model export coverage differs')
    for filename in ('manual_review.jsonl.gz', 'screening_overlay.jsonl.gz'):
        entry = report['artifacts'][filename]
        path = root / filename
        require(Path(entry['path']).resolve() == path and file_sha(path) == entry['sha256'],
                'model artifact binding differs: ' + filename)
        bindings[filename] = {'path': str(path), 'sha256': entry['sha256']}
    queue_id = bindings['manual_review.jsonl.gz']['sha256']
    require(package['queue_id'] == queue_id, 'package queue identity differs')
    queue_path = root / 'manual_review_ui/queue.jsonl'
    queue_sha = file_sha(queue_path)
    require(queue_sha == package['files']['queue.jsonl']['sha256'], 'UI queue SHA differs')
    bindings['ui_queue'] = {'path': str(queue_path), 'sha256': queue_sha}
    queue = list(json_rows(queue_path))
    pending = list(json_rows(root / 'manual_review.jsonl.gz'))
    pending.sort(key=lambda row: (row['queue_order'], row['item_id']))
    require(len(queue) == len(pending) == package['records'] == report['counts']['visual_pending'],
            'pending queue coverage differs')
    for index, (ui, model) in enumerate(zip(queue, pending), 1):
        require(ui['review_number'] == index and ui['item_id'] == model['item_id']
                and ui['result_sha256'] == model['result_sha256']
                and ui['source_ref'] == model['source_ref'], 'UI queue is not bound to model record')
    require(len({row['item_id'] for row in pending}) == len(pending), 'duplicate pending model image')
    return package, report, bindings, queue, {row['item_id']: row for row in pending}


def combined_counts(root, report, counts):
    """Optional arithmetic rollup of the previously snapshotted Qwen population."""
    path = root.parent / 'status_reports/final_model_counts.json'
    if not path.exists():
        return None
    prior, binding = read_bound(path)
    upstream = report['prior_qwen']
    require(prior['summary_sha256'] == upstream['source_artifacts']['SUMMARY.json']['sha256']
            and prior['complete_sha256'] == upstream['source_artifacts']['COMPLETE.json']['sha256']
            and prior['summary_counts_match'] is True and not prior['schema_anomalies'],
            'Qwen count snapshot binding differs')
    for name in ('SUMMARY.json', 'COMPLETE.json'):
        entry = upstream['source_artifacts'][name]
        read_bound(entry['path'], entry['sha256'])
    rec = prior['model_recommendation']
    require(rec['pending']['count'] + prior['technical_failure_exhausted']['count'] == report['expected'],
            'Codex scope does not replace all previous unresolved items')
    final = dict(counts)
    final['admitted'] += rec['admit']['count']
    final['ai_excluded'] += rec['reject']['count']
    require(sum(final.values()) == prior['eligible_assets'], 'combined denominator differs')
    return {'population': prior['eligible_assets'], 'counts': final, 'source_snapshot': binding,
            'semantics': 'Qwen/Codex筛选建议加人工决定；不是已认证的AI来源数量，也不是训练集发布。'}


def markdown_report(report):
    human = report['human_decision_counts']
    text = [
        '# DCVLM 人工复核导入结果', '',
        '人工结果已作为独立覆盖层合并。未修改模型原始判断、数据库、原始图像或复核页面。', '',
        f"生成时间（UTC）：{report['created_at_utc']}",
        f"提交 SHA256：`{report['submission']['sha256']}`", '',
        f"原队列 {report['queue_count']} 张，明确提交 {report['submitted_count']} 条；"
        f"保留 {human.get('keep', 0)}、AI 排除 {human.get('exclude_ai', 0)}、"
        f"质量排除 {human.get('exclude_quality', 0)}、仍不确定 {human.get('unsure', 0)}。", '',
        '| 范围 | 保留 | AI 排除建议 | 质量排除 | 待复核 | 技术失败 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    rows = [(str(report['model_scope_records']) + ' 条 Codex 复核', report['resolved_counts'])]
    if report['combined_qwen_population']:
        combined = report['combined_qwen_population']
        rows.append((str(combined['population']) + ' 条检测器告警总队列', combined['counts']))
    for name, counts in rows:
        text.append('| ' + name + ' | ' + ' | '.join(str(counts[key]) for key in
                    ('admitted', 'ai_excluded', 'quality_excluded', 'visual_pending', 'technical_failure')) + ' |')
    text += ['', '人工未确定项继续 pending，不默认通过。质量排除只针对数据集内实际图像质量，'
             '不计为 AI 生成证据；来源 URL 更清晰也不会自动替换 TAR 图像。', '',
             '筛选结论不等于来源认证。此导入没有发布最终训练集，也没有完成全库 SSCD 去重。', '',
             '## 待复核', '']
    for row in report['remaining_pending']:
        text.append(f"- #{row['review_number']}，`{row['item_id']}`，人工决定：{row['human_decision']}。")
    if not report['remaining_pending']:
        text.append('无。')
    if report.get('correction'):
        text += ['', '## 人工补判', '',
                 '本版本保留原始提交版本，并按用户在会话中的明确补判修订：']
        for change in report['correction']['changes']:
            text.append(f"- #{change['review_number']}：`{change['from_decision']}` → "
                        f"`{change['to_decision']}`；用户原话：{change['user_message']}。")
    text += ['', '## 产物', '',
             '- `submission.json`：用户提交的原始字节。',
             '- `human_overrides.jsonl.gz`：仅显式提交的人工作用层。',
             '- `resolved_screening_overlay.jsonl.gz`：模型全量结果与人工决定合并；原模型完整保留在 `model_screening`。',
             '- `remaining_manual_review.jsonl.gz`：仍未确定项。',
             '- `AUDIT.json`：全量逐项合并核验；`COMPLETE.json` 只表示本次导入完成。',
             '- 运行根目录 `HUMAN_REVIEW_CURRENT.json` 指向当前生效的人工作用层。', '']
    return '\n'.join(text)


def verify_version(directory, submission_sha):
    complete, complete_binding = read_bound(directory / 'COMPLETE.json')
    require(complete['schema_version'] == SCHEMA and complete['complete'] is True
            and complete['submission_sha256'] == submission_sha, 'human completion differs')
    report, report_binding = read_bound(directory / 'REPORT.json', complete['report_sha256'])
    for filename, artifact in report['artifacts'].items():
        require(Path(filename).name == filename, 'invalid artifact filename')
        path = directory / filename
        require(file_sha(path) == artifact['sha256'] and path.stat().st_size == artifact['bytes'],
                'human artifact integrity failure: ' + filename)
    return report, {'schema_version': SCHEMA, 'directory': str(directory),
                    'submission_sha256': submission_sha, 'report': report_binding,
                    'completion': complete_binding, 'all_reviews_resolved': report['all_reviews_resolved']}


def import_submission(root, submission_path, supersedes=None, correction_evidence=None):
    root = Path(root).resolve(strict=True)
    submission_path = Path(submission_path).resolve(strict=True)
    raw = submission_path.read_bytes()
    payload = strict_json(raw)
    submission_sha = hashlib.sha256(raw).hexdigest()
    with exclusive(root):
        package, model_report, bindings, queue, pending = load_context(root)
        decisions = validate_submission(payload, package['queue_id'], queue)
        require(decisions, 'empty submission has no explicit decisions to import')
        parent = root / 'human_reviews'
        parent.mkdir(exist_ok=True)
        directory = parent / submission_sha
        current_path = root / 'HUMAN_REVIEW_CURRENT.json'
        correction = None
        evidence_raw = None
        if current_path.exists():
            current, _ = read_bound(current_path)
            if current['submission_sha256'] != submission_sha:
                require(supersedes == current['submission_sha256'] and correction_evidence is not None,
                        'another human version is current; explicit parent and correction evidence required')
                parent_dir = parent / supersedes
                previous_report, previous_pointer = verify_version(parent_dir, supersedes)
                require(current == previous_pointer, 'current human version binding differs')
                previous_payload, _ = read_bound(parent_dir / 'submission.json', supersedes)
                previous = validate_submission(previous_payload, package['queue_id'], queue)
                evidence_raw = Path(correction_evidence).read_bytes()
                evidence = strict_json(evidence_raw)
                changes = validate_correction(previous, decisions, evidence, supersedes)
                correction = {'parent': previous_pointer, 'changes': changes,
                              'evidence': {'path': str(directory / 'correction_evidence.json'),
                                           'sha256': hashlib.sha256(evidence_raw).hexdigest()}}
        else:
            require(supersedes is None and correction_evidence is None, 'correction has no current parent')
        if directory.exists():
            report, pointer = verify_version(directory, submission_sha)
            require(correction is None or report.get('correction') == correction,
                    'existing version belongs to a different correction history')
            atomic_json(current_path, pointer)
            return report
        temporary = Path(tempfile.mkdtemp(prefix='.import-', dir=parent))
        try:
            (temporary / 'submission.json').write_bytes(raw)
            if evidence_raw is not None:
                (temporary / 'correction_evidence.json').write_bytes(evidence_raw)
            reference = {'path': str(directory / 'submission.json'), 'sha256': submission_sha,
                         'queue_id': package['queue_id']}
            filenames = ('resolved_screening_overlay.jsonl.gz', 'human_overrides.jsonl.gz',
                         'remaining_manual_review.jsonl.gz')
            counts = Counter({key: 0 for key in CATEGORIES})
            output_counts = Counter()
            original_counts = Counter({key: 0 for key in CATEGORIES})
            expected_rows, seen_human, seen_pending = {}, set(), set()
            with ExitStack() as stack:
                streams = {name: stack.enter_context(gzip.GzipFile(filename='', fileobj=
                           stack.enter_context((temporary / name).open('wb')), mode='wb', compresslevel=1, mtime=0))
                           for name in filenames}
                for model in json_rows(root / 'screening_overlay.jsonl.gz'):
                    aid = model['item_id']
                    require(aid not in expected_rows, 'duplicate model image ID')
                    require(model['report_category'] in CATEGORIES and model['processing_status'] == 'succeeded'
                            and digest(model['result']) == model['result_sha256'], 'invalid model result')
                    original_counts[model['report_category']] += 1
                    if model['report_category'] == 'visual_pending':
                        require(aid in pending and model == pending[aid], 'pending row differs from reviewed queue')
                        seen_pending.add(aid)
                    human = decisions.get(aid)
                    resolved = resolve_row(model, human, reference)
                    counts[resolved['report_category']] += 1
                    expected_rows[aid] = digest(resolved)
                    write_line(streams[filenames[0]], resolved)
                    output_counts[filenames[0]] += 1
                    if human is not None:
                        seen_human.add(aid)
                        write_line(streams[filenames[1]], resolved)
                        output_counts[filenames[1]] += 1
                    if resolved['report_category'] == 'visual_pending':
                        write_line(streams[filenames[2]], resolved)
                        output_counts[filenames[2]] += 1
            require(original_counts == model_report['counts'] and len(expected_rows) == model_report['expected'],
                    'original model count or coverage differs')
            require(seen_human == set(decisions) and seen_pending == set(pending), 'human or pending coverage differs')
            # Reopen every produced gzip and compare every complete record, not just counters.
            for name in filenames:
                seen = set()
                for row in json_rows(temporary / name):
                    aid = row['item_id']
                    require(aid not in seen and expected_rows.get(aid) == digest(row), 'resolved content audit failed')
                    seen.add(aid)
                expected = (set(expected_rows) if name == filenames[0] else set(decisions) if name == filenames[1]
                            else {aid for aid in pending if aid not in decisions or decisions[aid]['decision'] == 'unsure'})
                require(seen == expected and len(seen) == output_counts[name], 'resolved coverage audit failed')
            # Inputs must remain the same throughout the merge, including reports and model gzip.
            for binding in bindings.values():
                require(file_sha(binding['path']) == binding['sha256'], 'source changed during human import')
            audit = {'schema_version': SCHEMA, 'passed': True, 'skipped_gates': [],
                     'scope': 'human merge; previous model strict audit reused by content hash',
                     'records': len(expected_rows), 'submitted_records': len(decisions),
                     'checks': ['submission_schema_and_unique_identity', 'exact_queue_binding',
                                'source_strict_audit_binding', 'model_result_digests',
                                'explicit_human_decisions_only', 'full_output_row_equality_and_coverage',
                                'all_gzip_artifacts_reopened', 'source_files_unchanged'],
                     'input_bindings': bindings}
            atomic_json(temporary / 'AUDIT.json', audit)
            queue_by_id = {row['item_id']: row for row in queue}
            remaining = [{'review_number': queue_by_id[aid]['review_number'], 'item_id': aid,
                          'human_decision': decisions[aid]['decision'] if aid in decisions else None,
                          'source_ref': pending[aid]['source_ref']}
                         for aid in pending if aid not in decisions or decisions[aid]['decision'] == 'unsure']
            report = {'schema_version': SCHEMA, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                      'submission': reference, 'original_submission_path': str(submission_path),
                      'queue_id': package['queue_id'], 'queue_count': len(queue), 'submitted_count': len(decisions),
                      'human_decision_counts': dict(Counter(row['decision'] for row in decisions.values())),
                      'model_scope_records': len(expected_rows), 'model_counts_before': dict(original_counts),
                      'resolved_counts': dict(counts), 'remaining_pending': remaining,
                      'all_queue_items_submitted': len(decisions) == len(queue),
                      'all_reviews_resolved': counts['visual_pending'] == 0,
                      'source_certification': False, 'source_images_modified': False,
                      'model_database_modified': False, 'training_dataset_published': False,
                      'correction': correction,
                      'import_code': {'path': str(Path(__file__).resolve()), 'sha256': file_sha(__file__)},
                      'combined_qwen_population': combined_counts(root, model_report, counts),
                      'input_bindings': bindings}
            (temporary / 'REPORT.md').write_text(markdown_report(report), encoding='utf-8')
            extra_artifacts = ('correction_evidence.json',) if correction else ()
            report['artifacts'] = {name: {'sha256': file_sha(temporary / name),
                                         'bytes': (temporary / name).stat().st_size,
                                         **({'records': output_counts[name]} if name in filenames else {})}
                                   for name in (*filenames, 'submission.json', 'AUDIT.json', 'REPORT.md', *extra_artifacts)}
            atomic_json(temporary / 'REPORT.json', report)
            atomic_json(temporary / 'COMPLETE.json', {
                'schema_version': SCHEMA, 'complete': True, 'submission_sha256': submission_sha,
                'report_sha256': file_sha(temporary / 'REPORT.json'),
                'meaning': 'human import completed, not all reviews resolved or training release'})
            for path in temporary.iterdir():
                fsync_file(path)
            fsync_dir(temporary)
            temporary.rename(directory)
            fsync_dir(parent)
            report, pointer = verify_version(directory, submission_sha)
            atomic_json(current_path, pointer)
            fsync_dir(root)
            return report
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--supersedes', help='Exact current submission SHA, required for an explicit correction')
    parser.add_argument('--correction-evidence', type=Path)
    args = parser.parse_args()
    report = import_submission(args.root, args.submission, args.supersedes, args.correction_evidence)
    print(json.dumps({key: report[key] for key in ('submission', 'submitted_count', 'human_decision_counts',
                                                  'resolved_counts', 'all_reviews_resolved')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
