"""Build a DCVLM-only loss-policy view over an immutable published release.

The native rows and raw TARs stay untouched. Every original row receives an
auditable policy word; exact ordered-supervision duplicates are ranked again
after changing quality exclusions into image-loss restrictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from corpus_pipeline.postprocess.reconcile import (
    artifact, atomic_json, bind_identity, digest, file_sha, locked, require, stamp,
)
from .source_view import CODES, verify_mask

CANDIDATE_COLUMNS = (
    'original_parquet_path', 'row_index', 'record_id', 'sample_id', 'record_signature',
    'decision', 'decision_reason', 'native_image_count', 'image_ids', 'reason_codes',
    'status', 'confirmed_screen_excluded', 'unresolved_screening',
    'integrity_quarantine', 'confirmed_record_quality_reason', 'source', 'category',
)


def _document(path):
    return json.loads(Path(path).read_text())


def _bound_file(root, item):
    """Validate exact bytes and reject paths outside the declared local root."""
    root = Path(root).resolve(strict=True)
    path = (root / item['path']).resolve(strict=True)
    require(path.is_relative_to(root), 'Bound artifact escapes its root')
    before = stamp(path)
    require(before['size'] == item['size_bytes'] and file_sha(path) == item['sha256'],
            'Bound artifact bytes or SHA changed: ' + str(path))
    require(stamp(path) == before, 'Bound artifact changed while hashing')
    return path


def _receipt_artifact(receipt, path):
    path = str(Path(path).resolve(strict=True))
    found = [item for item in receipt['artifacts'] if item['path'] == path]
    require(len(found) == 1, 'Published receipt must bind exactly one requested artifact')
    return found[0]


def _verified_document(path, item):
    path = Path(path).resolve(strict=True)
    _bound_file(path.parent, {**item, 'path': path.name})
    return _document(path)


def load_parent(parent, dataset_id):
    """Verify the published metadata graph without re-reading native payloads.

    Candidate Parquet payloads are verified separately before each scan. Native
    SHA/size identities must agree in the published catalog and source view;
    their original stamps must still match. The portable reader hashes native
    bytes when consuming them, including after a filesystem relocation.
    """
    parent = Path(parent).resolve(strict=True)
    current_path = parent / 'CURRENT.json'
    current = _document(current_path)
    require(current.get('schema') == 'corpus.unified.release.v1'
            and current.get('scope') == 'full'
            and current.get('required_collections') == ['dcvlm'],
            'Parent must be a full DCVLM-only published release')
    release_path = _bound_file(parent, current['manifest'])
    complete_path = _bound_file(parent, current['complete'])
    release = _document(release_path)
    complete = _document(complete_path)
    require(release.get('complete') is True and complete.get('complete') is True
            and release.get('scope') == complete.get('scope') == 'full'
            and release.get('required_collections') == complete.get('required_collections') == ['dcvlm']
            and complete['release_id'] == current['release_id'] == digest(release['identity'])
            and complete['manifest_sha256'] == current['manifest']['sha256']
            and complete['records'] == release['records']
            and complete['counts'] == release['counts'], 'Parent publication identity differs')
    identity = release['identity']
    require(set(identity['collections']) == {'dcvlm'} and identity['required_collections'] == ['dcvlm'],
            'Parent includes another collection')
    provenance = {'parent_current.json': current_path,
                  'parent_release_manifest.json': release_path,
                  'parent_release_COMPLETE.json': complete_path}
    receipts = {}
    for stage in ('normalize', 'records', 'source_view'):
        binding = identity['stages'][stage]
        path = _bound_file(parent, binding)
        value = _document(path)
        require(value.get('complete') is True and value.get('stage') == stage
                and value.get('collections') == identity['collections']
                and value.get('semantics', {}).get('scope') == 'full', 'Invalid parent stage receipt')
        receipts[stage] = value
        provenance[stage + '_receipt.json'] = path
    candidate = parent / 'training_candidate_index'
    view = parent / 'training_candidate_source_view'
    documents = {}
    for name, root, stage in (('candidate', candidate, 'records'), ('source_view', view, 'source_view')):
        for filename in ('COMPLETE.json', 'manifest.json'):
            path = root / filename
            value = _verified_document(path, _receipt_artifact(receipts[stage], path))
            documents[(name, filename)] = value
            provenance[name + '_' + filename] = path
        marker = documents[(name, 'COMPLETE.json')]
        manifest = documents[(name, 'manifest.json')]
        require(marker.get('complete') is True and manifest.get('complete') is True
                and marker['manifest_sha256'] == file_sha(root / 'manifest.json')
                and marker['records'] == manifest['records'] == release['records']
                and marker['counts'] == manifest['counts']
                and marker['scope'] == manifest['scope'] == 'full_run',
                'Parent stage manifest/completion coverage differs')
    candidate_manifest = documents[('candidate', 'manifest.json')]
    view_manifest = documents[('source_view', 'manifest.json')]
    require(candidate_manifest['counts'] == release['counts']
            and view_manifest['counts'] == {**release['counts'], 'not_indexed': 0}
            and view_manifest['codes'] == CODES
            and view_manifest['original_records_modified'] is False
            and view_manifest['input_candidate_manifest_sha256'] == file_sha(candidate / 'manifest.json'),
            'Published candidate/source-view relation differs')
    for name in ('manifest', 'complete'):
        binding = release['reader'][name]
        require(_bound_file(parent, binding) == view / ('manifest.json' if name == 'manifest' else 'COMPLETE.json'),
                'Published reader points to another source view')
    roots_path = parent / 'readers' / 'ROOTS.example.json'
    roots = _document(roots_path)
    require(roots.get('schema') == 'corpus.source_root_mapping.v1'
            and set(roots['dataset_roots']) == {dataset_id} and len(roots['native_roots']) == 1,
            'Root mapping is not the pinned DCVLM baseline dataset')
    native = Path(next(iter(roots['native_roots']))).resolve(strict=True)
    require(roots['native_roots'][str(native)] == str(native), 'Unexpected native-root remapping in parent')
    native_documents = {}
    for filename in ('COMPLETE.json', 'manifest.json', 'catalog.json'):
        path = native / filename
        native_documents[filename] = _verified_document(path, _receipt_artifact(receipts['normalize'], path))
        provenance['native_' + filename] = path
    native_complete = native_documents['COMPLETE.json']
    native_manifest = native_documents['manifest.json']
    catalog = native_documents['catalog.json']
    require(native_complete.get('complete') is True and native_complete.get('validation_passed') is True
            and native_complete.get('schema_version') == native_manifest.get('schema_version') == 'corpus.native.v2'
            and native_complete['scope'] == native_manifest['scope'] == 'full_declared_inputs'
            and native_complete['manifest_sha256'] == file_sha(native / 'manifest.json')
            and native_complete['catalog_sha256'] == file_sha(native / 'catalog.json')
            and native_manifest['collections'] == ['dcvlm']
            and native_manifest['portable_datasets'] == [dataset_id]
            and native_manifest['source_files_modified'] is False
            and native_manifest['original_row_fields_modified'] is False
            and native_complete['statistics']['samples'] == native_manifest['statistics']['samples'] == release['records'],
            'Native baseline snapshot identity or coverage differs')
    catalog_by_path = {}
    for item in catalog:
        path = (native / item['path']).resolve(strict=True)
        require(path.is_relative_to(native) and str(path) not in catalog_by_path
                and item['collection'] == 'dcvlm' and type(item['rows']) is int and item['rows'] > 0,
                'Invalid/duplicate native catalog entry')
        catalog_by_path[str(path)] = item
    files = view_manifest['files']
    require(len(files) == len(catalog_by_path) == view_manifest['source_files']
            and sum(x['rows'] for x in files) == view_manifest['native_rows'] == release['records'],
            'Native/source-view row coverage differs')
    seen = set()
    for item in files:
        path = item['source_path']
        require(path not in seen and path in catalog_by_path, 'Source view has a duplicate/unknown native file')
        seen.add(path)
        entry = catalog_by_path[path]
        require(item['source_sha256'] == entry['sha256'] and item['rows'] == entry['rows']
                and item['source_stamp']['size'] == entry['bytes']
                and stamp(path) == item['source_stamp'], 'Native file/catalog/source-view binding changed')
    provenance['ROOTS.example.json'] = roots_path
    return {'parent': parent, 'release': release, 'current': current, 'candidate': candidate,
            'candidate_manifest': candidate_manifest, 'view': view, 'view_manifest': view_manifest,
            'native': native, 'roots': roots, 'provenance': provenance}


def _numpy(values, dtype=None):
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=dtype)


def _list_has_any(values, needles):
    """Vectorized per-record list membership, including empty/null lists."""
    values = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    result = np.zeros(len(values), dtype=bool)
    flat = pc.list_flatten(values)
    if len(flat):
        selected = _numpy(pc.fill_null(pc.is_in(flat, value_set=pa.array(list(needles), type=flat.type)), False), bool)
        parents = _numpy(pc.list_parent_indices(values), np.int64)
        result[parents[selected]] = True
    return result


class FlagScatter:
    """Exactly-once vectorized native locator coverage with an invalid sentinel."""
    def __init__(self, files):
        self.files = files
        self.paths = pa.array([item['source_path'] for item in files])
        require(len(set(self.paths.to_pylist())) == len(files), 'Duplicate native source path')
        self.lengths = np.asarray([item['rows'] for item in files], dtype=np.int64)
        require(np.all(self.lengths > 0), 'Native rows must be positive')
        self.offsets = np.r_[np.int64(0), np.cumsum(self.lengths)[:-1]]
        self.flags = np.full(int(self.lengths.sum()), 0xffff, dtype='<u2')
        self.seen = 0

    def consume(self, paths, row_indices, flags):
        require(pa.types.is_integer(row_indices.type), 'Original row locator must use integer indices')
        ids = pc.index_in(paths, value_set=self.paths)
        require(ids.null_count == 0 and row_indices.null_count == 0, 'Unknown/null original row locator')
        ids = _numpy(ids, np.int64)
        rows = _numpy(row_indices, np.int64)
        require(len(rows) == len(flags) and np.all(rows >= 0) and np.all(rows < self.lengths[ids]),
                'Original row locator outside native file')
        positions = self.offsets[ids] + rows
        require(len(np.unique(positions)) == len(positions) and np.all(self.flags[positions] == 0xffff),
                'Duplicate original row locator')
        require(np.all(np.asarray(flags, dtype=np.uint64) < 0xffff), 'Invalid policy word')
        self.flags[positions] = flags
        self.seen += len(rows)

    def complete(self):
        require(self.seen == len(self.flags) and np.all(self.flags != 0xffff), 'Missing original row locator coverage')
        return self.flags

    def file_flags(self, number):
        start = int(self.offsets[number])
        return self.flags[start:start + int(self.lengths[number])]


def base_flags(table, definitions, safety_deny):
    """Derive independent policy facts, never hiding integrity behind exclusion."""
    required = ('decision', 'decision_reason', 'native_image_count', 'status',
                'confirmed_screen_excluded', 'unresolved_screening', 'integrity_quarantine')
    require(all(table[name].null_count == 0 for name in required), 'Null policy input')
    decisions = table['decision']
    require(bool(pc.all(pc.is_in(decisions, value_set=pa.array(['keep', 'exclude', 'quarantine']))).as_py()),
            'Unknown prior decision')
    require(bool(pc.all(pc.greater_equal(table['native_image_count'], 0)).as_py()), 'Negative native image count')
    flags = np.zeros(len(table), dtype='<u2')
    def add(name, values):
        flags[np.asarray(values, dtype=bool)] |= definitions[name]
    def equal(name, value):
        return _numpy(pc.equal(table[name], value), bool)
    add('PRIOR_EXCLUDE', equal('decision', 'exclude'))
    add('PRIOR_QUARANTINE', equal('decision', 'quarantine'))
    add('DUPLICATE', equal('decision_reason', 'identical_ordered_supervision_and_confirmed_images'))
    add('FRONTPASS_REJECTION', equal('status', 'rejected'))
    add('SCREENING_EXCLUSION', _numpy(table['confirmed_screen_excluded'], bool))
    add('UNRESOLVED_SCREENING', _numpy(table['unresolved_screening'], bool))
    add('INTEGRITY', _numpy(table['integrity_quarantine'], bool))
    add('ORIGINAL_BINDING', _list_has_any(table['reason_codes'], ['unresolved_original_image_binding']))
    add('SAFETY', _list_has_any(table['image_ids'], safety_deny))
    add('RECORD_QUALITY', _numpy(pc.is_valid(table['confirmed_record_quality_reason']), bool))
    add('NO_IMAGE', equal('native_image_count', 0))
    return flags


def rerank_bucket(table, flags, policy_arrays, duplicate_flag, *, threads=2):
    """Re-elect one readable representative, including formerly rejected groups.

    Old exclusions were evaluated before duplicate decisions. A group containing
    only quality-rejected rows therefore has no old duplicate labels at all.
    Ranking every signature is necessary to avoid restoring all its repeats.
    """
    import duckdb
    require(len(table) == len(flags), 'Policy/candidate row count mismatch')
    for name in ('record_signature', 'sample_id', 'record_id'):
        require(table[name].null_count == 0, 'Null duplicate identity')
    require(pc.count_distinct(table['record_id']).as_py() == len(table), 'Duplicate candidate record ID')
    policy_result = policy_arrays(flags)
    allowed, ready = policy_result[:2]
    keys = table.select(['record_signature', 'sample_id', 'record_id'])
    keys = keys.append_column('ordinal', pa.array(np.arange(len(table), dtype=np.int64)))
    keys = keys.append_column('readable', pa.array(ready)).append_column('image_allowed', pa.array(allowed))
    con = duckdb.connect()
    try:
        con.execute('SET threads = ' + str(int(threads)))
        con.register('policy_keys', keys)
        losers = con.execute('''SELECT ordinal FROM (
            SELECT ordinal,row_number() OVER (PARTITION BY record_signature
              ORDER BY readable DESC,image_allowed DESC,sample_id,record_id) AS n
            FROM policy_keys) WHERE n>1''').fetchnumpy()['ordinal']
    finally:
        con.close()
    result = flags.copy()
    result[losers] |= duplicate_flag
    require(np.count_nonzero((result & duplicate_flag) == 0) == pc.count_distinct(table['record_signature']).as_py(),
            'Reranking did not produce exactly one representative per signature')
    return result


def _column_counts(table, column):
    return {item['values']: item['counts'] for item in pc.value_counts(table[column]).to_pylist()}


def _read_bucket(candidate, bucket, executor, *, batch_size=131072):
    """Hash all three original files before reading projected Arrow batches."""
    entries = bucket['artifacts']
    require(len(entries) == 3 and {Path(item['path']).name for item in entries}
            == {'keep.parquet', 'exclude.parquet', 'quarantine.parquet'},
            'Candidate bucket must contain its three decision partitions')
    def checked(item):
        before = stamp(candidate / item['path'])
        path = _bound_file(candidate, item)
        require(stamp(path) == before, 'Candidate changed while hashing')
        return path, before
    paths = list(executor.map(checked, entries))
    tables = []
    observed = []
    for path, before in paths:
        require(stamp(path) == before, 'Candidate changed between hash and read')
        with pq.ParquetFile(path) as reader:
            require(set(CANDIDATE_COLUMNS).issubset(reader.schema_arrow.names), 'Candidate required fields missing')
            batches = list(reader.iter_batches(batch_size=batch_size, columns=list(CANDIDATE_COLUMNS)))
            # Empty decision partitions must preserve the exact projected schema.
            table = pa.Table.from_batches(batches, schema=pa.schema([reader.schema_arrow.field(x) for x in CANDIDATE_COLUMNS]))
        require(stamp(path) == before, 'Candidate changed during read')
        tables.append(table)
        observed.append((path, before))
    table = pa.concat_tables(tables)
    require(len(table) == bucket['records'], 'Candidate bucket row count differs')
    require(_column_counts(table, 'decision') == {k: v for k, v in bucket['counts'].items() if v},
            'Candidate bucket decision counts differ')
    require(_column_counts(table, 'decision_reason') == {k: v for k, v in bucket['reasons'].items() if v},
            'Candidate bucket reason counts differ')
    require(table['record_signature'].null_count == 0 and bool(pc.all(pc.match_substring_regex(
        table['record_signature'], '^' + bucket['bucket'] + '[0-9a-f]{62}$')).as_py()),
        'Candidate signature belongs to another bucket')
    return table, observed


def _local_artifact(root, item):
    relative = Path(item['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Portable artifact path must be relative')
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), 'Portable artifact escapes root')
    before = stamp(path)
    require(file_sha(path) == item['sha256'], 'Portable artifact SHA mismatch')
    if 'size_bytes' in item:
        require(before['size'] == item['size_bytes'], 'Portable artifact size mismatch')
    require(stamp(path) == before, 'Portable artifact changed while hashing')
    return path


def load_repairs(manifest_path, candidate_sha256):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _document(manifest_path)
    require(manifest.get('schema') == 'corpus.dcvlm_integrity_repair.v1' and manifest.get('complete') is True
            and manifest.get('parent_candidate_manifest_sha256') == candidate_sha256,
            'Repair manifest does not bind this complete candidate release')
    records_path = _local_artifact(manifest_path.parent, manifest['records'])
    policies_path = _local_artifact(manifest_path.parent, manifest['image_policies'])
    records = pq.read_table(records_path)
    require(set(('record_id', 'outcome', 'reason', 'evidence_sha256')).issubset(records.column_names)
            and len(records) == manifest['records']['rows']
            and all(records[name].null_count == 0 for name in ('record_id', 'outcome', 'reason', 'evidence_sha256'))
            and pc.count_distinct(records['record_id']).as_py() == len(records),
            'Repair record coverage/identity is invalid')
    require(bool(pc.all(pc.is_in(records['outcome'], value_set=pa.array(
        ['verified_original', 'repaired', 'mismatch', 'unresolved']))).as_py())
        and bool(pc.all(pc.match_substring_regex(records['evidence_sha256'], '^[0-9a-f]{64}$')).as_py()),
        'Repair record outcome/evidence binding is invalid')
    if 'record_outcomes' in manifest.get('counts', {}):
        require(_column_counts(records, 'outcome') == manifest['counts']['record_outcomes'],
                'Repair outcome counts differ from manifest')
    policies = pq.read_table(policies_path)
    required = ('asset_id', 'decode_policy', 'frame_count', 'width', 'height', 'bytes_verified',
                'decode_verified', 'frame0_rgb_sha256')
    require(set(required).issubset(policies.column_names) and len(policies) == manifest['image_policies']['rows']
            and all(policies[name].null_count == 0 for name in required)
            and pc.count_distinct(policies['asset_id']).as_py() == len(policies),
            'Repair image policy coverage/identity is invalid')
    require(bool(pc.all(pc.match_substring_regex(policies['asset_id'], '^[0-9a-f]{64}$')).as_py())
            and bool(pc.all(pc.is_in(policies['decode_policy'], value_set=pa.array(['original', 'first_frame']))).as_py())
            and all(bool(pc.all(pc.greater(policies[name], 0)).as_py()) for name in ('frame_count', 'width', 'height'))
            and bool(pc.all(pc.equal(policies['bytes_verified'], True)).as_py())
            and bool(pc.all(pc.equal(policies['decode_verified'], True)).as_py())
            and bool(pc.all(pc.match_substring_regex(policies['frame0_rgb_sha256'], '^[0-9a-f]{64}$')).as_py()),
            'Unverified/invalid image repair policy')
    widths = _numpy(policies['width'], np.int64)
    heights = _numpy(policies['height'], np.int64)
    require(np.all(widths <= 200_000_000) and np.all(heights <= 200_000_000)
            and np.all(widths * heights <= 200_000_000), 'Repair image exceeds the verified reader pixel bound')
    evidence = manifest.get('evidence_artifacts', [])
    require(isinstance(evidence, list) and bool(evidence)
            and len({item['path'] for item in evidence}) == len(evidence),
            'Repair audit evidence artifacts are missing or repeated')
    evidence_paths = [_local_artifact(manifest_path.parent, item) for item in evidence]
    require(bool(pc.all(pc.is_in(records['evidence_sha256'], value_set=pa.array(
        [item['sha256'] for item in evidence]))).as_py()), 'Repair record references an unbound evidence artifact')
    return {'path': manifest_path, 'sha256': file_sha(manifest_path), 'manifest': manifest,
            'records_path': records_path, 'policies_path': policies_path, 'records': records,
            'outcomes': records['outcome'].combine_chunks(), 'seen': np.zeros(len(records), dtype=bool),
            'evidence_paths': evidence_paths}


def apply_repairs(table, flags, repairs, definitions):
    indexes = pc.index_in(table['record_id'], value_set=repairs['records']['record_id'])
    present = _numpy(pc.is_valid(indexes), bool)
    anomalies = (flags & (definitions['PRIOR_QUARANTINE'] | definitions['INTEGRITY'] | definitions['ORIGINAL_BINDING'])) != 0
    require(np.all(~anomalies | present), 'Integrity/binding anomaly is missing its repair outcome')
    selected = _numpy(pc.drop_null(indexes), np.int64)
    require(len(np.unique(selected)) == len(selected) and not np.any(repairs['seen'][selected]),
            'Repair record matched more than once')
    repairs['seen'][selected] = True
    outcomes = pc.take(repairs['outcomes'], selected)
    positions = np.flatnonzero(present)
    successful = _numpy(pc.is_in(outcomes, value_set=pa.array(['verified_original', 'repaired'])), bool)
    mismatch = _numpy(pc.equal(outcomes, 'mismatch'), bool)
    flags[positions[successful]] |= definitions['REPAIRED']
    flags[positions[mismatch]] |= definitions['MISMATCH']
    return flags


def load_user_review(path, safety_deny):
    path = Path(path).resolve(strict=True)
    value = _document(path)
    require(value.get('schema') == 'corpus.user_adult_review_override.v1'
            and value.get('decision') == 'adult_user_review'
            and value.get('independent_age_verification') is False
            and value.get('allow_understanding') is True and value.get('allow_image_loss') is False
            and isinstance(value.get('user_statement'), str) and bool(value['user_statement'].strip())
            and len(value.get('asset_ids', [])) == len(set(value.get('asset_ids', [])))
            and set(value.get('asset_ids', [])) == set(safety_deny), 'Invalid user adult-review override')
    return {'path': path, 'sha256': file_sha(path), 'value': value}


def _copy_verified(source, destination, expected_sha256):
    source = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(file_sha(destination) == expected_sha256, 'Existing copied artifact differs')
        return
    before = stamp(source)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    with source.open('rb') as src, temporary.open('wb') as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    require(stamp(source) == before and file_sha(temporary) == expected_sha256, 'Artifact changed during copy')
    os.replace(temporary, destination)


def _write_flags(path, flags):
    data = np.asarray(flags, dtype='<u2').tobytes()
    expected = hashlib.sha256(data).hexdigest()
    if path.exists():
        require(path.stat().st_size == len(data) and file_sha(path) == expected, 'Existing policy mask differs')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial')
        with temporary.open('wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return expected


def _histogram(flags):
    histogram = np.bincount(flags, minlength=65536)
    return {str(int(value)): int(histogram[value]) for value in np.flatnonzero(histogram)}


def build(parent, output, *, repair_manifest, user_review, hash_workers=4, threads=2,
          batch_size=131072, expected_safety_records=8):
    from . import dcvlm_loss_policy_reader as policy
    require(type(hash_workers) is int and hash_workers > 0 and type(threads) is int and threads > 0
            and type(batch_size) is int and batch_size > 0, 'Invalid worker/batch configuration')
    context = load_parent(parent, policy.DATASET_ID)
    output = Path(output).resolve()
    require(not output.is_relative_to(context['parent']) and not context['parent'].is_relative_to(output)
            and not output.is_relative_to(context['native']), 'Output must be a separate sibling release')
    candidate_sha = file_sha(context['candidate'] / 'manifest.json')
    repairs = load_repairs(repair_manifest, candidate_sha)
    user = load_user_review(user_review, policy.SAFETY_DENY_SHA256)
    reader_source = Path(policy.__file__).resolve(strict=True)
    portable_source = context['parent'] / 'readers' / 'portable_refs.py'
    documentation_source = Path(__file__).with_name('DCVLM_LOSS_POLICY.md')
    provenance_bindings = {name: {'sha256': file_sha(path), 'size_bytes': path.stat().st_size}
                           for name, path in context['provenance'].items()}
    identity = {'schema': policy.SCHEMA, 'dataset_id': policy.DATASET_ID,
                'parent_current_sha256': file_sha(context['parent'] / 'CURRENT.json'),
                'parent_candidate_manifest_sha256': candidate_sha,
                'builder_sha256': file_sha(__file__), 'reader_sha256': file_sha(reader_source),
                'portable_refs_sha256': file_sha(portable_source),
                'documentation_sha256': file_sha(documentation_source),
                'repair_manifest_sha256': repairs['sha256'], 'user_review_sha256': user['sha256'],
                'expected_safety_records': expected_safety_records}
    with locked(output):
        bind_identity(output, identity)
        if (output / 'CURRENT.json').exists():
            policy.verify_release(output)
            return _document(output / 'COMPLETE.json')
        def progress(phase, **values):
            atomic_json(output / 'PROGRESS.json', {'phase': phase, 'updated_at': time.time(), **values})
        progress('verifying_and_reranking_candidates', completed_buckets=0,
                 total_buckets=len(context['candidate_manifest']['buckets']), records=0)
        files = context['view_manifest']['files']
        scatter = FlagScatter(files)
        totals = Counter()
        reasons = Counter()
        categories = Counter()
        input_stamps = []
        bucket_statistics = []
        buckets = context['candidate_manifest']['buckets']
        require(len({bucket['bucket'] for bucket in buckets}) == len(buckets), 'Repeated candidate signature bucket')
        with ThreadPoolExecutor(max_workers=hash_workers) as executor:
            for number, bucket in enumerate(buckets):
                table, observed = _read_bucket(context['candidate'], bucket, executor, batch_size=batch_size)
                input_stamps.extend(observed)
                words = base_flags(table, policy.FLAGS, policy.SAFETY_DENY_SHA256)
                words[(words & policy.FLAGS['SAFETY']) != 0] |= policy.FLAGS['USER_ADULT_REVIEW']
                apply_repairs(table, words, repairs, policy.FLAGS)
                words = rerank_bucket(table, words, policy.policy_arrays, policy.FLAGS['DUPLICATE_NOT_SELECTED'], threads=threads)
                scatter.consume(table['original_parquet_path'], table['row_index'], words)
                totals.update(_column_counts(table, 'decision'))
                reasons.update(_column_counts(table, 'decision_reason'))
                # Arrow's count aggregation does not need to materialize Python rows.
                grouped = table.group_by(['source', 'category', 'decision']).aggregate([('record_id', 'count')])
                for row in grouped.to_pylist():
                    categories[(row['source'], row['category'], row['decision'])] += row['record_id_count']
                bucket_statistics.append({'bucket': bucket['bucket'], 'records': len(table),
                                          'flag_counts': _histogram(words)})
                progress('verifying_and_reranking_candidates', completed_buckets=number + 1,
                         total_buckets=len(buckets), records=scatter.seen)
        scatter.complete()
        parent_manifest = context['candidate_manifest']
        require(scatter.seen == parent_manifest['records'] and dict(totals) == parent_manifest['counts']
                and dict(reasons) == parent_manifest['reasons'], 'Candidate totals/reasons differ from parent manifest')
        expected_categories = {(row['source'], row['category'], row['decision']): row['records']
                               for row in parent_manifest['source_category_counts']}
        require(len(expected_categories) == len(parent_manifest['source_category_counts'])
                and dict(categories) == expected_categories, 'Candidate source/category coverage differs')
        require(np.all(repairs['seen']), 'Repair evidence contains unmatched record identities')
        require(np.count_nonzero(scatter.flags & policy.FLAGS['SAFETY']) == expected_safety_records,
                'Safety-asset occurrence coverage differs from the audited parent')
        progress('verifying_parent_masks_and_writing_policy', source_files=len(files))
        outputs = []
        all_flag_counts = Counter()
        for number, item in enumerate(files):
            words = scatter.file_flags(number)
            original_mask = verify_mask(context['view'], item)
            derived = np.ones(len(words), dtype=np.uint8)
            derived[(words & policy.FLAGS['PRIOR_EXCLUDE']) != 0] = CODES['exclude']
            derived[(words & policy.FLAGS['PRIOR_QUARANTINE']) != 0] = CODES['quarantine']
            require(np.array_equal(original_mask, derived), 'Policy locators do not match published source-view decisions')
            require(stamp(item['source_path']) == item['source_stamp'], 'Native metadata changed during policy build')
            name = hashlib.sha256(item['source_path'].encode()).hexdigest() + '.u16'
            path = output / 'flags' / name
            flags_sha = _write_flags(path, words)
            flag_counts = _histogram(words)
            all_flag_counts.update(flag_counts)
            outputs.append({'source_path': item['source_path'], 'source_sha256': item['source_sha256'],
                            'source_bytes': item['source_stamp']['size'], 'rows': item['rows'],
                            'flags_path': str(path.relative_to(output)), 'flags_sha256': flags_sha,
                            'flags_bytes': len(words) * 2, 'flag_counts': flag_counts})
            policy.verify_flags(output, outputs[-1])
            if (number + 1) % 256 == 0:
                progress('verifying_parent_masks_and_writing_policy', completed_files=number + 1, source_files=len(files))
        for path, before in input_stamps:
            require(stamp(path) == before, 'Candidate payload changed during policy build')
        require(file_sha(context['parent'] / 'CURRENT.json') == identity['parent_current_sha256'], 'Parent pointer changed during build')
        counts = policy.summarize_flag_counts(dict(all_flag_counts))
        require(counts['records'] == parent_manifest['records'], 'Final policy coverage differs')
        progress('freezing_reader_and_provenance', counts=counts)
        for name, source in context['provenance'].items():
            _copy_verified(source, output / 'provenance' / name, provenance_bindings[name]['sha256'])
        _copy_verified(reader_source, output / 'readers' / reader_source.name, identity['reader_sha256'])
        _copy_verified(portable_source, output / 'readers' / 'portable_refs.py', identity['portable_refs_sha256'])
        _copy_verified(Path(__file__), output / 'runtime' / Path(__file__).name, identity['builder_sha256'])
        _copy_verified(documentation_source, output / 'README.md', identity['documentation_sha256'])
        _copy_verified(context['provenance']['ROOTS.example.json'], output / 'readers' / 'ROOTS.example.json',
                       provenance_bindings['ROOTS.example.json']['sha256'])
        reader_binding = {'schema': policy.READER_SCHEMA, 'files': [
            {'name': reader_source.name, 'sha256': identity['reader_sha256']},
            {'name': 'portable_refs.py', 'sha256': identity['portable_refs_sha256']}]}
        atomic_json(output / 'readers' / 'READER_BINDING.json', reader_binding)
        repair_destination = output / 'repairs' / repairs['path'].name
        _copy_verified(repairs['path'], repair_destination, repairs['sha256'])
        for key, source_key in (('records', 'records_path'), ('image_policies', 'policies_path')):
            item = repairs['manifest'][key]
            _copy_verified(repairs[source_key], repair_destination.parent / item['path'], item['sha256'])
        for item, source in zip(repairs['manifest']['evidence_artifacts'], repairs['evidence_paths']):
            _copy_verified(source, repair_destination.parent / item['path'], item['sha256'])
        _copy_verified(user['path'], output / 'provenance' / 'USER_ADULT_REVIEW.json', user['sha256'])
        atomic_json(output / 'audit' / 'BUCKETS.json', {'complete': True, 'buckets': bucket_statistics,
                    'parent_counts': dict(totals), 'parent_reasons': dict(reasons),
                    'repair_records_matched': int(np.count_nonzero(repairs['seen']))})
        _, training_ready, _ = policy.policy_arrays(scatter.flags)
        duplicate_flags = (scatter.flags & policy.FLAGS['DUPLICATE_NOT_SELECTED']) != 0
        manifest = {'schema': policy.SCHEMA, 'complete': True, 'scope': 'full', 'collection': 'dcvlm',
                    'dataset_id': policy.DATASET_ID, 'records': counts['records'],
                    'selected_records': counts['selected_records'], 'source_files': len(outputs),
                    'policy_files_verified': len(outputs),
                    'original_records_modified': False, 'records_filtered': True,
                    'deduplication_preserved': True, 'filtered_duplicates': int(np.count_nonzero(training_ready & duplicate_flags)),
                    'duplicate_flag_records': int(np.count_nonzero(duplicate_flags)),
                    'flags_encoding': 'uint16_le', 'flag_definitions': policy.FLAGS,
                    'counts': counts, 'flag_counts': dict(all_flag_counts), 'files': outputs,
                    'reader_binding_sha256': file_sha(output / 'readers' / 'READER_BINDING.json'),
                    'safety_deny_sha256': sorted(policy.SAFETY_DENY_SHA256),
                    'parent_candidate_manifest_sha256': candidate_sha,
                    'parent_release_id': context['current']['release_id'],
                    'repair_manifest': artifact(repair_destination, output),
                    'user_review': artifact(output / 'provenance' / 'USER_ADULT_REVIEW.json', output),
                    'runtime': artifact(output / 'runtime' / Path(__file__).name, output),
                    'documentation': artifact(output / 'README.md', output),
                    'audit': artifact(output / 'audit' / 'BUCKETS.json', output),
                    'provenance': [artifact(output / 'provenance' / name, output) for name in sorted(provenance_bindings)],
                    'identity_sha256': file_sha(output / 'IDENTITY.json'),
                    'image_payloads_modified': False, 'source_certification': False,
                    'other_collections_policy_changed': False}
        atomic_json(output / 'manifest.json', manifest)
        result = {'schema': policy.SCHEMA, 'complete': True, 'scope': 'full', 'records': counts['records'],
                  'selected_records': counts['selected_records'], 'counts': counts,
                  'manifest_sha256': file_sha(output / 'manifest.json'), 'completed_at': time.time()}
        atomic_json(output / 'COMPLETE.json', result)
        policy._verify_release_payload(output)
        atomic_json(output / 'CURRENT.json', {'schema': policy.SCHEMA, 'scope': 'full', 'root': '.',
                    'release_id': result['manifest_sha256'], 'required_collections': ['dcvlm'],
                    'manifest': artifact(output / 'manifest.json', output),
                    'complete': artifact(output / 'COMPLETE.json', output)})
        policy.verify_release(output)
        progress('published', complete=True, counts=counts)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repair-manifest', required=True)
    parser.add_argument('--user-review', required=True)
    parser.add_argument('--hash-workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=131072)
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.output, repair_manifest=args.repair_manifest,
                           user_review=args.user_review, hash_workers=args.hash_workers,
                           threads=args.threads, batch_size=args.batch_size)))


if __name__ == '__main__':
    main()
