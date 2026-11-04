"""Bind existing screening evidence into a small asset-level release overlay.

This performs no inference. Qwen's outer annotation-only ``pending`` field is
not its recommendation. The published quality/human overlay takes precedence,
while an earlier explicit human AI exclusion cannot be cleared by a model.
Only exact asset IDs are covered; similarity does not propagate exclusions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = 'corpus.dedup.screening_decisions.v1'
ASSET = re.compile(r'[0-9a-f]{64}\Z')
ARROW_SCHEMA = pa.schema([
    ('asset_id', pa.string()), ('screen_action', pa.string()),
    ('screen_reason', pa.string()), ('quality_reviewed', pa.bool_()),
    ('ai_decision', pa.string()), ('strict_blur_reviewed', pa.bool_()),
    ('decision_authority', pa.string()), ('evidence_sha256', pa.string()),
])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def bound(path, expected=None):
    path = Path(path).resolve(strict=True)
    sha = file_sha(path)
    require(expected is None or sha == expected, 'SHA mismatch: ' + str(path))
    return {'path': str(path), 'sha256': sha, 'size_bytes': path.stat().st_size}


def _human_ai_excluded(row):
    """The quality revision embeds the earlier complete human envelope."""
    for _ in range(8):
        human = row.get('human_review') or {}
        if human.get('decision') == 'exclude_ai':
            return True
        row = row.get('baseline_screening') or {}
        if not row:
            return False
    raise ValueError('unexpectedly deep human screening history')


def normalize_quality(row):
    aid = row.get('item_id')
    require(isinstance(aid, str) and ASSET.fullmatch(aid), 'invalid overlay asset ID')
    category = row.get('report_category')
    expected = {'admitted': 'admit', 'ai_excluded': 'reject',
                'quality_excluded': 'reject', 'visual_pending': 'pending'}
    require(category in expected and row.get('decision') == expected[category],
            'inconsistent quality overlay decision/category')
    authority = row.get('decision_authority')
    require(isinstance(authority, str) and authority, 'missing overlay authority')
    human_ai = _human_ai_excluded(row)
    assessment = ((row.get('model_screening') or {}).get('result') or {}).get('assessment')
    ai = assessment if assessment in {'likely_non_ai', 'likely_ai_generated', 'possible_ai_edit'} else 'uncertain'
    reason = row.get('rejection_reason') or 'none'
    action = {'admitted': 'retain_annotated', 'ai_excluded': 'exclude',
              'quality_excluded': 'exclude', 'visual_pending': 'pending'}[category]
    if human_ai:
        action, reason, ai, authority = 'exclude', 'ai_generated', 'human_confirmed_ai_generated', 'human_user'
    elif category == 'ai_excluded':
        ai = 'likely_ai_generated'
    elif category == 'admitted' and ai == 'uncertain':
        ai = 'screening_passed_not_certified'
    return {'asset_id': aid, 'screen_action': action, 'screen_reason': reason,
            'quality_reviewed': True, 'ai_decision': ai,
            'strict_blur_reviewed': row.get('blur_status') in {'clear', 'blur_present', 'uncertain'},
            'decision_authority': authority, 'evidence_sha256': digest(row)}


def normalize_qwen(row):
    aid = row.get('asset_id')
    require(isinstance(aid, str) and ASSET.fullmatch(aid), 'invalid Qwen asset ID')
    evidence = json.loads(row['decision_json'])
    require(isinstance(evidence, dict), 'non-object Qwen evidence')
    require(evidence.get('input_sha256', row['input_sha256']) == row['input_sha256'],
            'Qwen input binding mismatch')
    primary = (evidence.get('reviews') or {}).get('primary') or {}
    recommendation = evidence.get('model_recommendation')
    assessment = primary.get('assessment', 'uncertain')
    reason = primary.get('reason') or evidence.get('reason') or row.get('reason') or 'unresolved_screening'
    action = 'pending'
    # Never infer AI from image form (charts/renders) or from a detector score.
    if recommendation == 'reject' and primary.get('decision') == 'reject' and (
        assessment == 'likely_ai_generated' and primary.get('confidence') == 'high'
        and reason == 'ai_visual_review_high_confidence_generated'
    ):
        action = 'exclude'
    elif recommendation == 'admit' and primary.get('decision') == 'admit':
        action = 'retain_annotated'
    return {'asset_id': aid, 'screen_action': action, 'screen_reason': reason,
            'quality_reviewed': False, 'ai_decision': assessment,
            'strict_blur_reviewed': False, 'decision_authority': 'qwen_visual_screening',
            'evidence_sha256': hashlib.sha256(row['decision_json'].encode()).hexdigest()}


@contextmanager
def lock(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def prepare(qwen_root, quality_pointer, output):
    qwen_root, quality_pointer, output = map(Path, (qwen_root, quality_pointer, output))
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with lock(output.parent / ('.' + output.name + '.lock')):
        qcomplete_path = qwen_root / 'COMPLETE.json'
        inputs = {'qwen_complete': bound(qcomplete_path), 'quality_pointer': bound(quality_pointer)}
        qcomplete = json.loads(qcomplete_path.read_text())
        require(qcomplete.get('screening_pass_complete') is True, 'Qwen screening incomplete')
        items = [r for r in qcomplete.get('artifacts', []) if Path(r['path']).name == 'decisions.parquet']
        require(len(items) == 1, 'missing/ambiguous Qwen decisions binding')
        qpath = (qwen_root / items[0]['path']).resolve(strict=True)
        require(qpath.parent == qwen_root.resolve(), 'Qwen decision path escapes root')
        inputs['qwen_decisions'] = bound(qpath, items[0]['sha256'])
        pointer = json.loads(quality_pointer.read_text())
        for name in ('completion', 'resolved_screening_overlay'):
            entry = pointer[name]
            inputs['quality_' + name] = bound(entry['path'], entry['sha256'])
            if 'bytes' in entry:
                require(inputs['quality_' + name]['size_bytes'] == entry['bytes'], 'quality size mismatch')
        complete = json.loads(Path(inputs['quality_completion']['path']).read_text())
        require(complete.get('complete') is True and complete.get('revision_sha256') == pointer.get('revision_sha256'),
                'quality completion/revision mismatch')
        identity = {'schema': SCHEMA, 'inputs': inputs, 'runtime_sha256': file_sha(__file__),
                    'quality_revision_sha256': pointer['revision_sha256'],
                    'reconciliation_complete_sha256': qcomplete.get('scope', {}).get('reconciliation_complete_sha256')}
        identity_sha = digest(identity)
        if output.exists():
            require((output / 'COMPLETE.json').is_file(), 'existing output is incomplete; choose a new output')
            prior = json.loads((output / 'COMPLETE.json').read_text())
            require(prior.get('complete') is True and prior.get('identity_sha256') == identity_sha,
                    'existing decisions identity differs')
            for item in prior['files']:
                checked = bound(output / item['path'], item['sha256'])
                require(checked['size_bytes'] == item['size_bytes'], 'output artifact size mismatch')
            return prior
        overlay = {}
        with gzip.open(inputs['quality_resolved_screening_overlay']['path'], 'rt', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = normalize_quality(json.loads(line))
                require(item['asset_id'] not in overlay, 'duplicate quality overlay asset')
                overlay[item['asset_id']] = item
        overlay_count = len(overlay)
        require(overlay_count == pointer['resolved_screening_overlay']['records'], 'quality overlay row count mismatch')
        temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
        try:
            seen = set()
            counts, reasons, ai_counts = Counter(), Counter(), Counter()
            quality_count = strict_count = 0
            target = temporary / 'screening.parquet'
            with pq.ParquetWriter(target, ARROW_SCHEMA, compression='zstd') as writer:
                for batch in pq.ParquetFile(qpath).iter_batches(batch_size=4096):
                    rows = []
                    for raw in batch.to_pylist():
                        aid = raw['asset_id']
                        require(aid not in seen, 'duplicate Qwen asset')
                        seen.add(aid)
                        base = normalize_qwen(raw)
                        item = overlay.pop(aid, base)
                        rows.append(item)
                        counts[item['screen_action']] += 1
                        reasons[item['screen_reason']] += 1
                        ai_counts[item['ai_decision']] += 1
                        quality_count += int(item['quality_reviewed'])
                        strict_count += int(item['strict_blur_reviewed'])
                    writer.write_table(pa.Table.from_pylist(rows, schema=ARROW_SCHEMA))
            require(not overlay, 'quality overlay contains assets outside Qwen screening scope')
            # Bind the precise snapshot consumed, including a pointer change during processing.
            for binding in inputs.values():
                require(file_sha(binding['path']) == binding['sha256'], 'input changed during screening export')
            artifact = bound(target)
            artifact['path'] = target.name
            result = {'schema': SCHEMA, 'complete': True, 'identity_sha256': identity_sha,
                      'identity': identity, 'files': [artifact], 'records': len(seen),
                      'overlay_records': overlay_count, 'counts': dict(counts), 'reason_counts': dict(reasons),
                      'ai_decision_counts': dict(ai_counts), 'quality_reviewed_records': quality_count,
                      'strict_blur_reviewed_records': strict_count,
                      'source_certification': False, 'training_dataset_published': False,
                      'pending_policy': 'retain annotation; never relabel uncertain as confirmed AI',
                      'scope': 'exact asset IDs only; no screening-label propagation across near-duplicate edges',
                      'quality_scope': 'quality_reviewed is existing quality assessment; strict_blur_reviewed marks the bounded stricter pass'}
            for name, value in [('IDENTITY.json', identity), ('COMPLETE.json', result)]:
                with (temporary / name).open('w') as handle:
                    json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
                    handle.write('\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            with target.open('rb') as handle:
                os.fsync(handle.fileno())
            os.rename(temporary, output)
            return result
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen-root', required=True)
    parser.add_argument('--quality-pointer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = prepare(args.qwen_root, args.quality_pointer, args.output)
    print(json.dumps({k: result[k] for k in ('complete', 'records', 'counts', 'quality_reviewed_records',
                                          'strict_blur_reviewed_records')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
