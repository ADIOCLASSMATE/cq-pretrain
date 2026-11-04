"""Derived reports for a strictly audited Codex export; never changes source data.

The farm's existing writer lock is held while a read-only snapshot is scanned
once. A partial strict export is reportable but cannot claim full completion.
The final JSON is the commit record for the atomically replaced report files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

SCHEMA = 'corpus.codex_review.report.v1'
FARM_SCHEMA = 'corpus.codex_review_farm.v1'
MODEL, EFFORT = 'gpt-5.6-sol', 'low'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_json(path, expected=None, within=None):
    path = Path(path).resolve(strict=True)
    if within is not None:
        require(path.is_relative_to(within), 'audit path escapes farm export root')
    actual = file_sha(path)
    require(expected is None or actual == expected, 'bound artifact hash mismatch: ' + path.name)
    return json.loads(path.read_text()), {'path': str(path), 'sha256': actual}


def source_index(source):
    """Hash/parse the frozen manifest once, retaining only small identity entries."""
    entries, h = {}, hashlib.sha256()
    with Path(source['path']).open('rb') as stream:
        for index, line in enumerate(stream):
            h.update(line)
            item = json.loads(line)
            aid = item['item_id']
            require(aid not in entries, 'duplicate source item ID')
            entries[aid] = (digest(item), item.get('queue_order', index))
    require(h.hexdigest() == source['sha256'] and len(entries) == source['count'],
            'source manifest hash or coverage mismatch')
    return entries


def export_rows(manifest, directory):
    total = 0
    for shard in manifest['shards']:
        path = (directory / shard['path']).resolve(strict=True)
        require(path.is_relative_to(directory) and file_sha(path) == shard['sha256']
                and path.stat().st_size == shard['compressed_bytes'], 'export shard integrity mismatch')
        count = 0
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                count += 1
                yield json.loads(line)
        require(count == shard['records'], 'export shard record count mismatch')
        total += count
    require(total == manifest['records'], 'export total count mismatch')


def prior_evidence(source):
    """Optional upstream Qwen bindings are rechecked, never used as new labels."""
    path = Path(source['path']).with_suffix('.manifest.json')
    if not path.exists():
        return {'available': False, 'reason': 'source manifest has no upstream evidence sidecar'}
    evidence, binding = bound_json(path)
    require(evidence.get('sha256') == source['sha256']
            and evidence.get('item_count') == source['count'], 'upstream sidecar binding mismatch')
    prior = {'available': True, 'manifest_evidence': binding,
             'source_database': evidence.get('source_database'),
             'selected_counts': evidence.get('counts'), 'source_artifacts': evidence.get('source_artifacts', {})}
    for name, entry in prior['source_artifacts'].items():
        value, _ = bound_json(entry['path'], entry['sha256'])
        if name == 'INPUT.json':
            prior['input_fields'] = {key: value.get(key) for key in
                                     ('mode', 'backend', 'models', 'model_contracts', 'prompt_path', 'prompt_sha256')}
        elif name == 'SUMMARY.json':
            prior['summary_counts'] = value.get('counts')
        elif name == 'COMPLETE.json':
            prior['screening_pass_complete'] = value.get('screening_pass_complete')
            prior['all_reviews_resolved'] = value.get('all_reviews_resolved')
    return prior


def _write_json_line(stream, value):
    stream.write((canonical(value) + '\n').encode())


def generate_report(root) -> dict:
    root = Path(root).resolve(strict=True)
    # The farm creates this lock before initializing/exporting. Opening read-only
    # avoids creating or mutating its state during reporting.
    with (root / 'farm.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _locked_report(root)


def _locked_report(root):
    pointer_path = root / ('COMPLETE.json' if (root / 'COMPLETE.json').exists() else 'EXPORT.json')
    pointer, pointer_binding = bound_json(pointer_path)
    require(type(pointer.get('complete')) is bool and pointer.get('audit_mode') == 'strict'
            and pointer.get('skipped_gates') == [], 'completion does not claim strict unskipped audit')
    require(pointer_path.name != 'COMPLETE.json' or pointer['complete'], 'false COMPLETE marker')
    audit, audit_binding = bound_json(pointer['audit'], pointer['audit_sha256'], root / 'exports')
    manifest, manifest_binding = bound_json(pointer['manifest'], pointer['manifest_sha256'], root / 'exports')
    content, content_binding = bound_json(audit['content_audit_path'], audit['content_audit_sha256'],
                                          Path(audit_binding['path']).parent)
    require(audit.get('export_manifest') == manifest_binding['path']
            and audit.get('export_manifest_sha256') == manifest_binding['sha256']
            and audit.get('export_checked') is True, 'audit does not bind checked export')
    for value in (audit, content):
        require(value.get('schema_version') == FARM_SCHEMA and value.get('passed') is True and value.get('mode') == 'strict'
                and value.get('skipped_gates') == [], 'strict audit did not pass all gates')
    require(manifest.get('schema_version') == FARM_SCHEMA and manifest.get('audit_mode') == 'strict' and manifest.get('model') == MODEL
            and manifest.get('reasoning_effort') == EFFORT, 'export model or audit contract mismatch')

    db = sqlite3.connect((root / 'state.sqlite').as_uri() + '?mode=ro', uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    temporary_files = []
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        metadata = {row['key']: json.loads(row['value']) for row in db.execute('SELECT key,value FROM metadata')}
        require(metadata.get('schema') == FARM_SCHEMA, 'database farm schema mismatch')
        source, runtime = metadata['source'], metadata['runtime_binding']
        runtime_sha = digest(runtime)
        require(metadata.get('complete') == pointer, 'completion pointer differs from database export checkpoint')
        require(runtime.get('model') == MODEL and runtime.get('reasoning_effort') == EFFORT,
                'runtime model binding mismatch')
        for value in (manifest, audit, content):
            require(value.get('source') == source and value.get('runtime_binding_sha256') == runtime_sha,
                    'source or runtime audit binding mismatch')
        successes = manifest['records']
        total = source['count']
        complete = successes == total
        require(pointer['records'] == successes and manifest['complete'] == complete
                and pointer['complete'] == complete, 'completion coverage flags mismatch')
        for value in (audit, content):
            require(value.get('expected') == total and value.get('succeeded') == successes
                    and value.get('audited_items') == successes
                    and value.get('complete_coverage') == complete, 'audit coverage mismatch')
        from .worker import validate_result
        worker_path = Path(__file__).with_name('worker.py').resolve()
        require(runtime.get('code', {}).get(str(worker_path)) == file_sha(worker_path),
                'current validator differs from audited worker')
        expected = source_index(source)
        prior_source = prior_evidence(source)
        exported = export_rows(manifest, Path(manifest_binding['path']).parent)
        status_counts, counts, quality_counts, failure_types = Counter(), Counter(), Counter(), Counter()
        old_recommendations, old_assessments, old_confidences, source_categories = (Counter() for _ in range(4))
        batch_seen, output_counts = set(), Counter()
        filenames = ('manual_review.jsonl.gz', 'technical_failures.jsonl.gz',
                     'screening_overlay.jsonl.gz', 'raw_provenance.jsonl.gz')
        output_paths = {}
        with ExitStack() as stack:
            streams = {}
            for filename in filenames:
                fd, temporary = tempfile.mkstemp(prefix='.' + filename + '.', suffix='.tmp', dir=root)
                temporary_files.append(Path(temporary)); output_paths[filename] = Path(temporary)
                raw = stack.enter_context(os.fdopen(fd, 'wb'))
                streams[filename] = stack.enter_context(gzip.GzipFile(filename='', fileobj=raw, mode='wb',
                                                                      compresslevel=1, mtime=0))
            scanned = 0
            query = '''SELECT i.*,b.directory AS batch_directory,b.metadata_json AS batch_metadata_json,
                       b.model AS batch_model,b.reasoning_effort AS batch_effort,
                       b.outcome_sha256 AS batch_outcome_sha256
                       FROM items i LEFT JOIN batches b ON i.batch_id=b.batch_id ORDER BY i.queue_order,i.item_id'''
            for row in db.execute(query):
                scanned += 1
                aid, status = row['item_id'], row['status']
                item = json.loads(row['input_json'])
                require(expected.pop(aid, None) == (row['input_sha256'], row['queue_order'])
                        and digest(item) == row['input_sha256'], 'database item differs from frozen source')
                require(status in {'pending', 'running', 'retry', 'failed', 'succeeded'}, 'unknown farm item status')
                status_counts[status] += 1
                prior = item.get('prior_review')
                require(isinstance(prior, dict) and item.get('prior_review_sha256') == digest(prior),
                        'prior review hash mismatch')
                primary = prior.get('reviews', {}).get('primary') or {}
                old_recommendations[str(prior.get('model_recommendation'))] += 1
                old_assessments[str(primary.get('assessment'))] += 1
                old_confidences[str(primary.get('confidence'))] += 1
                source_categories[str(item.get('category'))] += 1
                batch_id = row['batch_id']
                raw_reference = None
                if batch_id is not None:
                    require(row['batch_directory'] is not None, 'item references missing batch')
                    raw_reference = {'batch_id': batch_id, 'directory': row['batch_directory'],
                                     'outcome_sha256': row['batch_outcome_sha256']}
                    if batch_id not in batch_seen:
                        batch_seen.add(batch_id)
                        info = json.loads(row['batch_metadata_json'] or '{}')
                        _write_json_line(streams['raw_provenance.jsonl.gz'], {
                            **raw_reference, 'model': row['batch_model'], 'reasoning_effort': row['batch_effort'],
                            'thread_id': info.get('thread_id'), 'usage': info.get('usage'),
                            'prompt_version': info.get('prompt_version'), 'artifacts': info.get('artifacts', {}),
                            'strict_audit_reference': audit_binding})
                        output_counts['raw_provenance.jsonl.gz'] += 1
                overlay = {'item_id': aid, 'queue_order': row['queue_order'], 'processing_status': status,
                           'decision': None, 'rejection_reason': None, 'quality': None, 'quality_reason': None,
                           'source_certification': False, 'scope': 'derived_visual_screening_recommendation',
                           'source_ref': item['source_ref'], 'source_manifest': source,
                           'candidate_input_sha256': row['input_sha256'],
                           'upstream_input_sha256': item.get('source_input_sha256'),
                           'prior_review_sha256': item['prior_review_sha256'],
                           'prior_summary': {'category': item.get('category'), 'origin': prior.get('origin'),
                                             'model_recommendation': prior.get('model_recommendation'),
                                             'primary_assessment': primary.get('assessment'),
                                             'primary_confidence': primary.get('confidence')},
                           'prior_review_reference': {'manifest': source['path'], 'item_id': aid,
                                                      'queue_order': row['queue_order'], 'field': 'prior_review'},
                           'provenance_sha256': digest(item.get('provenance')),
                           'raw_provenance_reference': raw_reference, 'result_sha256': row['result_sha256']}
                if status == 'succeeded':
                    result = json.loads(row['result_json'])
                    validate_result(result, aid)
                    require(digest(result) == row['result_sha256'], 'database result hash mismatch')
                    prior_export = next(exported, None)
                    require(prior_export is not None and prior_export.get('item_id') == aid
                            and prior_export.get('source') == item and prior_export.get('result') == result
                            and prior_export.get('batch_id') == batch_id, 'database differs from strictly audited export')
                    require(row['batch_model'] == MODEL and row['batch_effort'] == EFFORT,
                            'successful batch model mismatch')
                    category = ('ai_excluded' if result['rejection_reason'] == 'ai_generated' else
                                'quality_excluded' if result['rejection_reason'] == 'quality_unusable' else
                                'admitted' if result['decision'] == 'admit' else 'visual_pending')
                    counts[category] += 1
                    quality_counts[result['quality']] += 1
                    overlay.update({key: result[key] for key in ('decision', 'rejection_reason', 'quality', 'quality_reason')})
                    overlay.update(result=result, report_category=category)
                    if result['decision'] == 'pending':
                        _write_json_line(streams['manual_review.jsonl.gz'], overlay)
                        output_counts['manual_review.jsonl.gz'] += 1
                else:
                    category = 'unprocessed' if row['attempts'] == 0 and status == 'pending' else 'technical_failure'
                    counts[category] += 1
                    error_type = row['error_type'] or ('not_attempted' if category == 'unprocessed' else 'missing_valid_result')
                    failure_types[error_type] += 1
                    overlay.update(report_category=category, attempts=row['attempts'], error_type=error_type,
                                   last_error_sha256=hashlib.sha256((row['last_error'] or '').encode()).hexdigest(),
                                   error_reference={'database': str(root / 'state.sqlite'), 'item_id': aid,
                                                    'field': 'last_error'})
                    _write_json_line(streams['technical_failures.jsonl.gz'], overlay)
                    output_counts['technical_failures.jsonl.gz'] += 1
                _write_json_line(streams['screening_overlay.jsonl.gz'], overlay)
                output_counts['screening_overlay.jsonl.gz'] += 1
            require(not expected and scanned == total and status_counts['succeeded'] == successes
                    and next(exported, None) is None, 'final coverage or exported success mismatch')
        db.rollback()
        # Verify immutable pointers did not change even if an external writer
        # ignored the advisory farm lock. The audited exports pin all successes.
        require(file_sha(pointer_path) == pointer_binding['sha256']
                and file_sha(audit_binding['path']) == audit_binding['sha256']
                and file_sha(manifest_binding['path']) == manifest_binding['sha256'], 'export changed during report')
        artifacts = {}
        for filename, temporary in output_paths.items():
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            artifacts[filename] = {'path': str(root / filename), 'sha256': file_sha(temporary),
                                   'compressed_bytes': temporary.stat().st_size, 'records': output_counts[filename]}
        categories = ('ai_excluded', 'quality_excluded', 'admitted', 'visual_pending', 'technical_failure', 'unprocessed')
        report = {'schema_version': SCHEMA, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                  'complete': complete, 'strict_audit_passed': True,
                  'coverage_warning': None if complete else '未全覆盖：仍有条目没有有效模型结果，不能把它们计为视觉不确定。',
                  'expected': total, 'valid_results': successes, 'missing_valid_results': total - successes,
                  'counts': {key: counts[key] for key in categories}, 'status_counts': dict(status_counts),
                  'quality_counts_valid_results': dict(quality_counts), 'technical_error_types_non_success': dict(failure_types),
                  'percent_of_valid_results': {key: 100 * counts[key] / successes if successes else None
                                              for key in categories[:4]},
                  'model': MODEL, 'reasoning_effort': EFFORT, 'runtime_binding_sha256': runtime_sha,
                  'audit': audit_binding, 'content_audit': content_binding, 'export_manifest': manifest_binding,
                  'completion_receipt': pointer_binding, 'source': source, 'artifacts': artifacts,
                  'prior_qwen': {**prior_source, 'selected_recommendations': dict(old_recommendations),
                                 'selected_primary_assessments': dict(old_assessments),
                                 'selected_primary_confidences': dict(old_confidences), 'selected_categories': dict(source_categories)},
                  'interpretation': {'source_certification': False, 'original_database_modified': False,
                                     'original_provenance_modified': False,
                                     'results_are': '筛选建议，不是来源认证；质量排除与 AI 生成判断分开统计。',
                                     'denominator_scope': '仅本次候选清单，不代表原始 DCVLM 全库分布。',
                                     'technical_failures_file_includes': '所有非成功项；report_category 区分未处理与技术失败。'}}
        labels = {'ai_excluded': 'AI 生成排除建议', 'quality_excluded': '质量过差排除建议',
                  'admitted': '通过建议', 'visual_pending': '视觉待人工复核',
                  'technical_failure': '技术失败／暂无有效结果', 'unprocessed': '尚未处理'}
        lines = [f"{'完整' if complete else '不完整'}筛选报告：有效结果 {successes:,}/{total:,}。",
                 '', '严格内容与导出审计已通过。' + ('' if complete else report['coverage_warning']), '',
                 f'模型：{MODEL}；推理强度：{EFFORT}。', '', '| 类别 | 数量 |', '|---|---:|']
        lines += [f'| {labels[key]} | {counts[key]:,} |' for key in categories]
        lines += ['', '这些是图像筛选建议，不认证图像来源。质量排除不计入 AI 排除；未处理及技术失败不计入视觉不确定。',
                  '统计分母仅限本次候选清单，不代表原始 DCVLM 全库分布。',
                  '原 Qwen 数据库及来源证据保持不变。成功但仍 pending 的条目在 manual_review.jsonl.gz；',
                  '所有非成功项在 technical_failures.jsonl.gz，其中 report_category 明确区分未处理项。',
                  'screening_overlay.jsonl.gz 覆盖全部输入，raw_provenance.jsonl.gz 索引真实批次原始回执。',
                  '', f"输入 SHA256：`{source['sha256']}`。", f"严格审计 SHA256：`{audit_binding['sha256']}`。", '']
        md = '\n'.join(lines).encode()
        fd, name = tempfile.mkstemp(prefix='.final_report.md.', suffix='.tmp', dir=root)
        temporary_files.append(Path(name))
        with os.fdopen(fd, 'wb') as stream:
            stream.write(md); stream.flush(); os.fsync(stream.fileno())
        report['markdown'] = {'path': str(root / 'final_report.md'), 'sha256': hashlib.sha256(md).hexdigest()}
        for filename, temporary in output_paths.items():
            os.replace(temporary, root / filename)
        os.replace(name, root / 'final_report.md')
        fd, name = tempfile.mkstemp(prefix='.final_report.json.', suffix='.tmp', dir=root)
        temporary_files.append(Path(name))
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(canonical(report) + '\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, root / 'final_report.json')
        return report
    finally:
        db.close()
        for temporary in temporary_files:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(canonical(generate_report(args.root)))


if __name__ == '__main__':
    main()
