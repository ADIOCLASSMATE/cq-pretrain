"""Sparse, hash-bound screening assembly; no API calls and no guessed completions."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from corpus_pipeline.dedup import decisions as legacy
from corpus_pipeline.postprocess.reconcile import atomic_json, locked
from .policy import config as checked_policy, policy_hash, quality_decision

SCHEMA = 'corpus.unified.screening.v1'


def _bound(path, inputs, expected=None):
    value = legacy.bound(path, expected)
    inputs[str(Path(path).resolve())] = value
    return Path(value['path'])


def _normal(row):
    legacy.require(legacy.ASSET.fullmatch(str(row.get('asset_id'))), 'Invalid historical exact asset ID')
    legacy.require(row.get('screen_action') in {'exclude', 'pending', 'retain_annotated'}, 'Invalid historical screening action')
    legacy.require(legacy.ASSET.fullmatch(str(row.get('evidence_sha256'))) and bool(row.get('decision_authority')), 'Historical decision lacks evidence')
    return {field.name: row.get(field.name) for field in legacy.ARROW_SCHEMA}


def _is_ai_exclusion(row):
    return row['screen_action'] == 'exclude' and ('ai_generated' in str(row.get('ai_decision')) or
           row['screen_reason'] in {'ai_generated', 'ai_visual_review_high_confidence_generated'})


def _is_non_ai(row):
    return row['screen_action'] == 'retain_annotated' and row.get('ai_decision') in {
        'likely_non_ai', 'confirmed_natural', 'screening_passed_not_certified', 'human_confirmed_non_ai'}


def _is_integrity_exclusion(row):
    return row['screen_action'] == 'exclude' and row['screen_reason'] in {'corrupt_image', 'confirmed_corrupt_image'}


def _reduce(aid, rows):
    if any(_is_ai_exclusion(row) for row in rows) and any(_is_non_ai(row) for row in rows):
        return None, 'conflicting_confirmed_ai_evidence'
    priority = {'exclude': 0, 'retain_annotated': 1, 'pending': 2}
    chosen = dict(min(rows, key=lambda r: (priority[r['screen_action']], not r['quality_reviewed'], r['evidence_sha256'])))
    if len(rows) > 1:
        chosen['evidence_sha256'] = legacy.digest(sorted(r['evidence_sha256'] for r in rows))
        chosen['decision_authority'] = 'joined_exact_asset_evidence'
        chosen['quality_reviewed'] = any(r['quality_reviewed'] for r in rows)
        chosen['strict_blur_reviewed'] = any(r['strict_blur_reviewed'] for r in rows)
    return chosen, None


def _strict_t2i_review(row):
    """A completed uncertain assessment is a selection exclusion, not an AI fact.

    Detector no-alert images never enter this sparse review overlay. I2T-only
    images are bypassed before reduction, and mixed-use I2T records bypass this
    asset-level exclusion downstream. Infrastructure/missing evidence is handled
    separately and must never be promoted to a completed negative decision.
    """
    if row['screen_action'] == 'exclude':
        return row
    assessment = row.get('ai_decision')
    known_non_ai = {'likely_non_ai', 'confirmed_natural', 'human_confirmed_non_ai',
                    'confirmed_programmatic', 'confirmed_real_edit'}
    if row['screen_action'] == 'retain_annotated' and assessment in known_non_ai:
        return row
    completed_labels = known_non_ai | {'uncertain', 'possible_ai_edit', 'likely_ai_generated',
                                        'screening_passed_not_certified'}
    if assessment not in completed_labels:
        return None
    return {**row, 'screen_action': 'exclude', 'screen_reason': 't2i_uncertain_review_excluded'}


def _current_assets(recon, buckets, wanted, inputs, bypassed=None, integrity_targets=()):
    """C++ column scan/semi-join; never materialize the complete image catalog in Python."""
    import duckdb
    con = duckdb.connect(); con.execute('SET threads=2'); con.execute("SET memory_limit='1GB'")
    found = set()
    try:
        for prefix in sorted({aid[:2] for aid in wanted}):
            if prefix not in buckets: continue
            item = buckets[prefix]; path = _bound(recon/item['path'], inputs, item['sha256'])
            con.register('wanted_assets', pa.table({'asset_id': sorted(aid for aid in wanted if aid.startswith(prefix))}))
            fields = pq.ParquetFile(path).schema_arrow.names
            flag = 'coalesce(i2t_only,false)' if 'i2t_only' in fields else 'false'
            rows = con.execute('SELECT asset_id,' + flag +
                ' FROM read_parquet(?) SEMI JOIN wanted_assets USING(asset_id)', [str(path)]).fetchall()
            found.update(aid for aid, skip in rows if not skip or aid in integrity_targets)
            if bypassed is not None:
                bypassed.update(aid for aid, skip in rows if skip)
    finally:
        con.close()
    return found


def _selection(output, pending, candidates, api_paths, recon_complete, scope):
    """Versioned API queue, preserving every original reconciliation field verbatim."""
    selected = {row['asset_id'] for row in pending if row['asset_id'] in candidates
                and row['reason'] != 'conflicting_confirmed_ai_evidence'}
    if not selected: return None
    revision = legacy.digest({'assets': sorted(selected), 'reconciliation': legacy.file_sha(recon_complete), 'scope': scope})
    directory = output/'api_selections'/revision; directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'API_SELECTION.json'
    if marker.exists():
        manifest = json.loads(marker.read_text())
        for item in [manifest['pending_snapshot'], *manifest['selected_files']]: legacy.bound(item['path'], item['sha256'])
        return {'manifest': legacy.bound(marker), 'files': manifest['selected_files'], 'assets': len(selected)}
    target = directory/'remaining_api_review.parquet'; value_set = pa.array(sorted(selected), type=pa.string())
    schema = pq.ParquetFile(api_paths[0]).schema_arrow; written = 0
    with pq.ParquetWriter(target, schema, compression='zstd') as writer:
        for path in api_paths:
            reader = pq.ParquetFile(path)
            legacy.require(reader.schema_arrow.equals(schema, check_metadata=False), 'Reconciliation API schemas differ')
            for batch in reader.iter_batches(batch_size=8192):
                chosen = batch.filter(pc.is_in(batch.column('asset_id'), value_set=value_set))
                if chosen.num_rows: writer.write_batch(chosen); written += chosen.num_rows
    legacy.require(written == len(selected), 'Incremental API selection coverage differs')
    snapshot = directory/'pending_snapshot.parquet'
    pq.write_table(pa.Table.from_pylist([{**row, 'needs_api': row['asset_id'] in selected} for row in pending]), snapshot)
    manifest = {'schema': 'corpus.api_review.selection.v1', 'scope': scope,
                'reconciliation_complete': legacy.bound(recon_complete),
                'source_files': [legacy.bound(path) for path in api_paths],
                'pending_snapshot': legacy.bound(snapshot), 'selected_files': [legacy.bound(target)],
                'selected_assets': len(selected), 'match': 'exact_original_reconciliation_rows',
                'global_screening_scope_authorized': False}
    atomic_json(marker, manifest)
    return {'manifest': legacy.bound(marker), 'files': manifest['selected_files'], 'assets': len(selected)}


def build_decisions(reconciliation_root, output, review_root=None, quality_decisions=(),
                    history_decision_manifests=(), policy_config=None, scope=None):
    policy_config = checked_policy(policy_config)
    strict_t2i = policy_config.get('t2i_uncertain') == 'exclude'
    recon = Path(reconciliation_root).resolve(strict=True); output = Path(output).resolve()
    inputs = {}; complete_path = _bound(recon/'COMPLETE.json', inputs)
    recon_complete = json.loads(complete_path.read_text())
    legacy.require(recon_complete.get('complete') is True, 'Reconciliation incomplete')
    scope = scope or ('full' if recon_complete.get('scope') == 'full_run' else 'validation')
    legacy.require(scope in {'validation', 'full'}, 'Invalid screening scope')
    candidates = set(); source_candidates = set(); bypassed = set(); bucket_assets = {}; api_paths = []
    for bucket in recon_complete['buckets']:
        path = _bound(recon/bucket['receipt_path'], inputs, bucket['receipt_sha256'])
        receipt = json.loads(path.read_text())
        for item in receipt['artifacts']:
            name = Path(item['path']).name
            if name == 'assets.parquet': bucket_assets[receipt['bucket']] = item
            if name != 'api_review.parquet': continue
            path = _bound(recon/item['path'], inputs, item['sha256'])
            api_paths.append(path)
            reader = pq.ParquetFile(path)
            columns = ['asset_id'] + (['i2t_only'] if 'i2t_only' in reader.schema_arrow.names else [])
            for batch in reader.iter_batches(columns=columns):
                ids = batch.column(0).to_pylist()
                legacy.require(not source_candidates.intersection(ids) and len(ids) == len(set(ids)), 'Duplicate API-review asset')
                legacy.require(all(legacy.ASSET.fullmatch(str(aid)) for aid in ids), 'Invalid API-review asset')
                source_candidates.update(ids)
                skip = batch.column('i2t_only').to_pylist() if 'i2t_only' in columns else [False] * len(ids)
                candidates.update(aid for aid, only_i2t in zip(ids, skip) if only_i2t is not True)
                bypassed.update(aid for aid, only_i2t in zip(ids, skip) if only_i2t is True)
    legacy.require(len(source_candidates) == recon_complete['counts']['api_review_assets'], 'API-review coverage mismatch')
    rules = [quality_decision(rule) for rule in quality_decisions]
    asset_rules = [rule for rule in rules if rule['scope'] == 'asset']
    targets = candidates | {rule['asset_id'] for rule in asset_rules}
    evidence = defaultdict(list); history_scopes = []; history_imports = []; errors = {}
    for manifest_path in history_decision_manifests:
        path = _bound(manifest_path, inputs); prior = json.loads(path.read_text())
        legacy.require(prior.get('complete') is True, 'Historical screening is incomplete')
        legacy.require(prior.get('schema') in {SCHEMA, legacy.SCHEMA}, 'Unsupported historical screening policy')
        if prior.get('policy_sha256'):
            legacy.require(prior['policy_sha256'] == policy_hash(policy_config), 'Historical screening policy mismatch')
        imported_ids = set(); excluded_ids = set()
        for item in prior['files']:
            if Path(item['path']).name != 'screening.parquet': continue
            data = _bound(path.parent/item['path'], inputs, item['sha256'])
            seen = set()
            for batch in pq.ParquetFile(data).iter_batches(batch_size=8192):
                for row in batch.to_pylist():
                    if row['asset_id'] not in targets and row['screen_action'] != 'exclude': continue
                    legacy.require(row['asset_id'] not in seen, 'Duplicate historical screening asset')
                    seen.add(row['asset_id']); evidence[row['asset_id']].append(_normal(row)); imported_ids.add(row['asset_id'])
                    if row['screen_action'] == 'exclude': excluded_ids.add(row['asset_id'])
        history_imports.append((str(path), prior, imported_ids, excluded_ids))
    extra = (targets | set(evidence)) - candidates
    integrity_targets = {rule['asset_id'] for rule in asset_rules if rule['reason'] == 'corrupt_image'}
    integrity_targets.update(aid for aid, rows in evidence.items() if any(_is_integrity_exclusion(row) for row in rows))
    current = _current_assets(recon, bucket_assets, extra, inputs, bypassed, integrity_targets) if extra else set()
    out_of_scope = extra - current - bypassed
    targets = candidates | current
    evidence = defaultdict(list, {aid: [row for row in rows if aid not in bypassed or _is_integrity_exclusion(row)]
                                 for aid, rows in evidence.items() if aid in targets})
    asset_rules = [rule for rule in asset_rules if rule['asset_id'] in targets and
                   (rule['asset_id'] not in bypassed or rule['reason'] == 'corrupt_image')]
    for path, prior, imported, excluded in history_imports:
        history_scopes.append({'manifest': path, 'imported_assets': len(imported & targets),
            'imported_exclusions_without_current_alert': len(excluded & current),
            'out_of_scope_exclusions': len(excluded - targets - bypassed),
            'i2t_only_exclusions_bypassed': len(excluded & bypassed), 'policy': prior['schema'],
            'match': 'exact_original_asset_id', 'detector_identity_reused': False,
            'note': 'Confirmed image exclusions apply across detector versions; historical retain only resolves current alerts/explicit quality targets.',
            'historical_scope': prior.get('identity', {}).get('reconciliation_complete_sha256')})
    if review_root is not None:
        root = Path(review_root).resolve(); marker = root/'COMPLETE.json'
        if marker.exists():
            _bound(marker, inputs); review = json.loads(marker.read_text())
            legacy.require(review.get('screening_pass_complete') is True, 'API screening pass incomplete')
            legacy.require(review.get('scope', {}).get('reconciliation_complete_sha256') == legacy.file_sha(complete_path),
                           'API screening belongs to another reconciliation snapshot')
            choices = [x for x in review.get('artifacts', []) if Path(x['path']).name == 'decisions.parquet']
            legacy.require(len(choices) == 1, 'Missing unique API decisions binding')
            item = choices[0]; path = _bound(root/item['path'], inputs, item['sha256']); seen = set()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                for row in batch.to_pylist():
                    aid = row['asset_id']
                    legacy.require(aid in source_candidates and aid not in seen, 'Unexpected/duplicate API-review asset')
                    seen.add(aid)
                    if aid in bypassed:
                        continue
                    if row.get('state') != 'done':
                        errors[aid] = 'review_infrastructure_' + str(row.get('state')); continue
                    try:
                        raw_evidence = json.loads(row['decision_json'])
                        primary = (raw_evidence.get('reviews') or {}).get('primary') or {}
                        legacy.require(primary.get('assessment') in {'likely_ai_generated', 'likely_non_ai', 'possible_ai_edit', 'uncertain'}
                                       and primary.get('decision') in {'admit', 'reject', 'pending'}, 'Missing completed visual assessment')
                        normalized = legacy.normalize_qwen(row)
                    except (ValueError, KeyError, TypeError):
                        errors[aid] = 'invalid_review_evidence'; continue
                    evidence[aid].append(normalized)
        else:
            # No fabricated result; historical evidence may cover some or all of this snapshot.
            pass
    for rule in asset_rules:
        aid = rule['asset_id']
        evidence[aid].append({'asset_id': aid, 'screen_action': 'exclude',
            'screen_reason': 'confirmed_' + rule['reason'], 'quality_reviewed': True,
            'ai_decision': 'not_inferred_from_quality', 'strict_blur_reviewed': rule['reason'] == 'blur',
            'decision_authority': 'confirmed_scoped_quality_rule', 'evidence_sha256': legacy.digest(rule)})
    rows = []; pending = []; conflicts = []
    for aid in sorted(targets):
        if not evidence[aid]:
            pending.append({'asset_id': aid, 'reason': errors.get(aid, 'review_result_missing')}); continue
        item, conflict = _reduce(aid, evidence[aid])
        if conflict:
            pending.append({'asset_id': aid, 'reason': conflict})
            conflicts.append({'asset_id': aid, 'reason': conflict, 'evidence': evidence[aid]}); continue
        if strict_t2i:
            item = _strict_t2i_review(item)
            if item is None:
                pending.append({'asset_id': aid, 'reason': 'review_evidence_incomplete'}); continue
        rows.append(item)
    identity = {'schema': SCHEMA, 'policy_sha256': policy_hash(policy_config),
                'reconciliation_complete_sha256': legacy.file_sha(complete_path),
                'inputs': inputs, 'history_scopes': history_scopes, 'quality_rules': rules,
                'source_review_candidates': len(source_candidates), 'i2t_only_screening_bypassed': len(bypassed),
                'code_sha256': legacy.file_sha(__file__), 'scope': scope}
    with locked(output):
        if (output/'COMPLETE.json').exists():
            previous = json.loads((output/'COMPLETE.json').read_text())
            legacy.require(previous['identity_sha256'] == legacy.digest(identity), 'Completed screening inputs changed')
            for item in [*previous['files'], *previous.get('record_quality_files', [])]:
                legacy.bound(output/item['path'], item['sha256'])
            return previous
        if pending:
            pq.write_table(pa.Table.from_pylist(pending), output/'pending.parquet', compression='zstd')
            atomic_json(output/'conflicts.json', conflicts)
            result = {'schema': SCHEMA, 'complete': False, 'pending_infra': len(pending),
                      'review_candidates': len(candidates), 'resolved_assets': len(rows),
                      'i2t_only_screening_bypassed': len(bypassed),
                      'conflicts': len(conflicts), 'identity': identity, 'out_of_scope_assets': len(out_of_scope),
                      'api_selection': _selection(output, pending, candidates, api_paths, complete_path, scope)}
            atomic_json(output/'PENDING.json', result); return result
        target = output/'screening.parquet'
        pq.write_table(pa.Table.from_pylist(rows, schema=legacy.ARROW_SCHEMA), target, compression='zstd')
        record_rules = [rule for rule in rules if rule['scope'] == 'source_record']; record_files = []
        if record_rules:
            path = output/'record_quality.parquet'
            pq.write_table(pa.Table.from_pylist([{'record_id': q.get('record_id'),
                'source_record_sha256': q['source_record_sha256'], 'reason': q['reason'],
                'decision': 'exclude', 'evidence_sha256': legacy.digest(q), 'evidence_json': json.dumps(q, sort_keys=True)}
                for q in record_rules]), path, compression='zstd')
            record_files.append({**legacy.bound(path), 'path': path.name})
        for binding in inputs.values(): legacy.bound(binding['path'], binding['sha256'])
        result = {'schema': SCHEMA, 'complete': True, 'policy_sha256': policy_hash(policy_config),
            'identity_sha256': legacy.digest(identity), 'identity': identity, 'pending_infra': 0,
            'reconciliation_complete_sha256': identity['reconciliation_complete_sha256'],
            'records': len(rows), 'review_candidates': len(candidates),
            'i2t_only_screening_bypassed': len(bypassed),
            'out_of_scope_assets': len(out_of_scope),
            'historical_exclusions_without_current_alert': len(
                {aid for _, _, _, excluded in history_imports for aid in excluded} &
                {row['asset_id'] for row in rows if row['screen_action'] == 'exclude'} - candidates),
            'review_uncertain': sum(row['screen_action'] == 'pending' for row in rows),
            'counts': dict(Counter(row['screen_action'] for row in rows)),
            'files': [{**legacy.bound(target), 'path': target.name}], 'record_quality_files': record_files,
            'source_certification': False, 'published_training': False,
            'scope': scope, 'matching_scope': 'exact original asset IDs; record-quality rules remain record scoped'}
        if strict_t2i:
            result['t2i_uncertain_excluded'] = sum(row['screen_reason'] == 't2i_uncertain_review_excluded' for row in rows)
        atomic_json(output/'IDENTITY.json', identity); atomic_json(output/'COMPLETE.json', result)
        (output/'PENDING.json').unlink(missing_ok=True)
        return result
