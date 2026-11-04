"""Metadata-only union of an immutable DCVLM loss view and task-aware candidates.

The frozen DCVLM masks, including repairs and review overrides, remain the
authority for selection. Exporting reads their original candidate index, never
the retired native small pool or image payloads. Every baseline selected row is
preserved. A new understanding-only permission is applied in this derived view.

Exact deduplication uses ordered supervision AND the native rendering contract,
task route, loss permission and ordered, geometry-bound decoded identities.
SSCD similarity is deliberately not an input. Pixel evidence is reused from
completed reconciliations; without it the conservative fallback is encoded SHA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time
import types

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from corpus_pipeline.postprocess.reconcile import (
    artifact, atomic_json, bind_identity, digest, file_sha, locked, quoted, require,
    stamp, validate_artifacts,
)
from corpus_pipeline.dedup.source_view import source_bindings
from corpus_pipeline.dedup.training_release import connection, parquet_sql

INDEX_SCHEMA = 'corpus.selected_reference_index.v1'
UNION_SCHEMA = 'corpus.baseline_union.v1'
TASK_POLICY = 'corpus.task_policy.v1'
KEY_POLICY = 'ordered_supervision_native_contract_task_permission_exact_pixels_v1'
SHA = re.compile(r'[0-9a-f]{64}\Z')
COLUMNS = ['record_id', 'sample_id', 'original_parquet_path', 'row_index',
           'source_record_sha256', 'text_hash', 'category', 'native_format',
           'source', 'image_ids']


def _json(path):
    return json.loads(Path(path).read_text())


def _binding(path):
    path = Path(path).resolve(strict=True)
    before = stamp(path)
    result = dict(path=str(path), sha256=file_sha(path), size_bytes=path.stat().st_size)
    require(stamp(path) == before, 'Input changed while binding')
    return result


def _verify(item):
    path = Path(item['path']).resolve(strict=True)
    require(path.stat().st_size == item['size_bytes'] and file_sha(path) == item['sha256'],
            'Bound source SHA/size changed: ' + str(path))
    return path


def _contained(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Artifact path must be relative')
    target = (root / path).resolve(strict=True)
    require(target.is_relative_to(root), 'Artifact escaped source root')
    return target


def _frozen_reader(view):
    """Execute the release-bound reader without creating bytecode in that release."""
    path = view / 'readers/dcvlm_loss_policy_reader.py'
    binding = _json(view / 'readers/READER_BINDING.json')
    item = next(x for x in binding['files'] if x['name'] == path.name)
    require(file_sha(path) == item['sha256'], 'Frozen baseline reader changed')
    module = types.ModuleType('_union_frozen_dcvlm_reader')
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def _candidate(candidate):
    candidate = Path(candidate).resolve(strict=True)
    manifest, done = _json(candidate / 'manifest.json'), _json(candidate / 'COMPLETE.json')
    require(manifest.get('schema_version') == 'corpus.training_candidate_index.v1'
            and manifest.get('complete') is True and done.get('complete') is True
            and manifest.get('scope') == done.get('scope') == 'full_run',
            'Expected a completed full candidate index')
    require(done.get('manifest_sha256') == file_sha(candidate / 'manifest.json')
            and done.get('records') == manifest['records']
            and done.get('counts') == manifest['counts'], 'Candidate completion/count binding differs')
    buckets = manifest['buckets']
    require(len({b['bucket'] for b in buckets}) == len(buckets)
            and sum(b['records'] for b in buckets) == manifest['records'],
            'Candidate bucket coverage differs')
    totals = Counter()
    for bucket in buckets:
        require(re.fullmatch(r'[0-9a-f]{2}', bucket['bucket']) is not None,
                'Invalid candidate bucket')
        require(len(bucket['artifacts']) == 3
                and {Path(x['path']).name for x in bucket['artifacts']}
                == {'keep.parquet', 'exclude.parquet', 'quarantine.parquet'},
                'Candidate bucket requires all three decisions')
        require(sum(bucket['counts'].values()) == bucket['records'], 'Candidate bucket count differs')
        totals.update(bucket['counts'])
    require(dict(totals) == manifest['counts'], 'Candidate decision counts differ')
    return candidate, manifest


def audit_baseline(loss_view, candidate_index):
    """Verify published lineage and current native locators without reading pixels."""
    view = Path(loss_view).resolve(strict=True)
    reader = _frozen_reader(view)
    baseline = reader.verify_release(view)
    candidate, manifest = _candidate(candidate_index)
    require(file_sha(candidate / 'manifest.json') == baseline['parent_candidate_manifest_sha256'],
            'Candidate does not belong to this baseline loss-policy release')
    provenance = {Path(x['path']).name: x for x in baseline['provenance']}
    for name in ('manifest.json', 'COMPLETE.json'):
        bound = provenance.get('candidate_' + name)
        require(bound is not None and file_sha(candidate / name) == bound['sha256'],
                'Baseline candidate provenance binding differs')
    bindings = source_bindings(candidate, manifest)
    require(set(bindings) == {x['source_path'] for x in baseline['files']},
            'Candidate native source coverage differs from selected baseline')
    for item in baseline['files']:
        bound = bindings[item['source_path']]
        require(bound['sha256'] == item['source_sha256']
                and bound['stamp']['size'] == item['source_bytes'],
                'Baseline native source SHA/size differs')
    return dict(view=view, reader=reader, baseline=baseline, candidate=candidate,
                manifest=manifest, bindings=bindings)


class _Coverage:
    def __init__(self, files):
        self.paths = pa.array([x['source_path'] for x in files])
        self.lengths = np.asarray([x['rows'] for x in files], dtype=np.int64)
        require(len(set(self.paths.to_pylist())) == len(files) and np.all(self.lengths > 0),
                'Invalid native file coverage')
        self.offsets = np.r_[np.int64(0), np.cumsum(self.lengths)[:-1]]
        self.seen = np.zeros(int(self.lengths.sum()), dtype=bool)

    def consume(self, table):
        indices = table['row_index']
        ids = pc.index_in(table['original_parquet_path'], value_set=self.paths)
        require(pa.types.is_integer(indices.type) and indices.null_count == ids.null_count == 0,
                'Unknown/null original row locator')
        ids, indices = ids.to_numpy(), indices.to_numpy()
        require(np.all(indices >= 0) and np.all(indices < self.lengths[ids]),
                'Original row locator outside native source')
        positions = self.offsets[ids] + indices
        require(len(np.unique(positions)) == len(positions) and not np.any(self.seen[positions]),
                'Duplicate original row locator')
        self.seen[positions] = True
        return positions

    def complete(self):
        require(bool(np.all(self.seen)), 'Missing original row locator coverage')


def _native_files(bindings, baseline=None):
    counts = {} if baseline is None else {x['source_path']: x['rows'] for x in baseline['files']}
    return [dict(source_path=path, source_sha256=b['sha256'], source_stamp=b['stamp'],
                 rows=counts.get(path) or pq.ParquetFile(path).metadata.num_rows)
            for path, b in sorted(bindings.items())]


def _export_table(table, source_key, source_kind, flags=None, reader=None):
    for name in COLUMNS:
        require(name in table.column_names and table[name].null_count == 0,
                'Missing/null candidate column: ' + name)
    for name in ('source_record_sha256', 'text_hash'):
        require(pc.all(pc.match_substring_regex(table[name], '^[0-9a-f]{64}$')).as_py() is not False,
                'Invalid candidate SHA column: ' + name)
    images = table['image_ids'].combine_chunks().cast(pa.list_(pa.string()))
    require(images.values.null_count == 0 and
            pc.all(pc.match_substring_regex(images.values, '^[0-9a-f]{64}$')).as_py() is not False,
            'Invalid encoded image identity')
    n = len(table)
    if flags is not None:
        old_allowed, ready, selected = reader.policy_arrays(flags)
        require(bool(np.all(ready & selected)), 'Nonselected baseline row reached export')
        categories = table['category']
        routes = pc.if_else(pc.is_in(categories, value_set=pa.array(['single_i2t', 'multi_i2t'])),
                            'i2t', pc.if_else(pc.equal(categories, 'pure_text'), 'text',
                                              pc.if_else(pc.equal(categories, 'interleaved'), 'mixed', 'unknown')))
        allowed = pa.array(np.zeros(n, dtype=bool))
        old = pa.array(old_allowed)
        words = pa.array(flags, type=pa.uint16())
    else:
        for name in ('processing_policy', 'task_route', 'image_loss_allowed'):
            require(name in table.column_names and table[name].null_count == 0,
                    'Candidate task policy is incomplete')
        require(pa.types.is_boolean(table['image_loss_allowed'].type), 'Image permission must be boolean')
        routes, allowed = table['task_route'], table['image_loss_allowed']
        require(pc.all(pc.equal(table['processing_policy'], TASK_POLICY)).as_py() is not False
                and pc.all(pc.is_in(routes, value_set=pa.array(['i2t', 't2i', 'mixed', 'text', 'unknown']))).as_py() is not False,
                'Unknown task policy or route')
        require(pc.any(pc.and_(allowed, pc.not_equal(routes, 't2i'))).as_py() is not True,
                'Only explicit T2I views may enable image loss')
        old, words = pa.nulls(n, pa.bool_()), pa.nulls(n, pa.uint16())
    result = table.select(COLUMNS).set_column(COLUMNS.index('image_ids'), 'image_ids', images)
    result = result.set_column(COLUMNS.index('row_index'), 'row_index', result['row_index'].cast(pa.int64()))
    for name, value in [('source_key', pa.repeat(source_key, n)), ('source_kind', pa.repeat(source_kind, n)),
                        ('processing_policy', pa.repeat(TASK_POLICY, n)), ('task_route', routes),
                        ('image_loss_allowed', allowed), ('dcvlm_image_loss_allowed', old),
                        ('dcvlm_policy_flags', words)]:
        result = result.append_column(name, value)
    return result


def _publish(output, manifest):
    atomic_json(output / 'manifest.json', manifest)
    sha = file_sha(output / 'manifest.json')
    done = dict(schema=manifest['schema'], complete=True, manifest_sha256=sha,
                records=manifest['records'], counts=manifest['counts'])
    atomic_json(output / 'COMPLETE.json', done)
    atomic_json(output / 'CURRENT.json', dict(schema=manifest['schema'], complete=True,
                release_id=sha, manifest=artifact(output / 'manifest.json', output),
                completion=artifact(output / 'COMPLETE.json', output)))
    return manifest


def verify_reference_release(output, *, verify_payloads=True):
    output = Path(output).resolve(strict=True)
    current = _json(output / 'CURRENT.json')
    validate_artifacts(output, dict(artifacts=[current['manifest'], current['completion']]))
    manifest, done = _json(output / 'manifest.json'), _json(output / 'COMPLETE.json')
    require(manifest.get('schema') in (INDEX_SCHEMA, UNION_SCHEMA) and manifest.get('complete') is True
            and done.get('complete') is True and current.get('complete') is True
            and current['release_id'] == done['manifest_sha256'] == current['manifest']['sha256']
            and manifest['records'] == done['records'] and manifest['counts'] == done['counts'],
            'Reference publication SHA/count binding differs')
    require(sum(x['rows'] for x in manifest['files']) == manifest['records'], 'Reference file coverage differs')
    if verify_payloads:
        validate_artifacts(output, dict(artifacts=manifest['files'] + manifest.get('auxiliary_files', [])))
        for item in manifest['files']:
            require(pq.ParquetFile(_contained(output, item['path'])).metadata.num_rows == item['rows'],
                    'Reference Parquet row coverage differs')
    return manifest


def _export(candidate_index, output, *, loss_view=None, batch_size=65536):
    require(type(batch_size) is int and batch_size > 0, 'Invalid batch size')
    baseline = audit_baseline(loss_view, candidate_index) if loss_view is not None else None
    candidate, manifest = (baseline['candidate'], baseline['manifest']) if baseline else _candidate(candidate_index)
    bindings = baseline['bindings'] if baseline else source_bindings(candidate, manifest)
    files = _native_files(bindings, baseline['baseline'] if baseline else None)
    source_key = file_sha(Path(loss_view) / 'manifest.json') if baseline else file_sha(candidate / 'manifest.json')
    source_kind = 'dcvlm_loss_view' if baseline else 'task_candidate'
    source = dict(source_key=source_key, source_kind=source_kind,
                  candidate_manifest=_binding(candidate / 'manifest.json'),
                  candidate_complete=_binding(candidate / 'COMPLETE.json'), native_files=files)
    if baseline:
        source['loss_view'] = str(baseline['view'])
        source['loss_manifest'] = _binding(baseline['view'] / 'manifest.json')
        source['loss_current'] = _binding(baseline['view'] / 'CURRENT.json')
    identity = dict(schema=INDEX_SCHEMA, source=source, code_sha256=file_sha(__file__),
                    derived_baseline_permission='all_selected_understanding_views_false',
                    task_policy=TASK_POLICY)
    output = Path(output).resolve()
    require(output != candidate and not output.is_relative_to(candidate), 'Output overlaps candidate input')
    if baseline:
        require(output != baseline['view'] and not output.is_relative_to(baseline['view']),
                'Output overlaps immutable baseline release')
    require(not any(output.is_relative_to(Path(path).parent) for path in bindings),
            'Output overlaps native source directory')
    with locked(output):
        bind_identity(output, identity)
        if (output / 'CURRENT.json').exists():
            return verify_reference_release(output)
        started = time.time()
        progress = dict(schema='corpus.selected_reference_export_progress.v1', status='running',
                        pid=os.getpid(), started_at=started, updated_at=started,
                        source_key=source_key, input_records=manifest['records'],
                        processed_records=0, selected_records=0, buckets_complete=0,
                        buckets_total=len(manifest['buckets']))
        atomic_json(output / 'PROGRESS.json', progress)
        coverage = _Coverage(files)
        flags = None
        if baseline:
            by_path = {x['source_path']: x for x in baseline['baseline']['files']}
            flags = np.concatenate([baseline['reader'].verify_flags(baseline['view'], by_path[x['source_path']]) for x in files])
        receipts, total, selected = [], 0, 0
        permission_counts = Counter()
        for bucket in manifest['buckets']:
            validate_artifacts(candidate, bucket)
            path = output / 'buckets' / bucket['bucket'] / 'references.parquet'
            receipt_path = path.with_name('COMPLETE.json')
            resumed = _json(receipt_path) if receipt_path.exists() else None
            if resumed:
                validate_artifacts(output, resumed)
                require(len(resumed['artifacts']) == 1 and
                        pq.ParquetFile(output / resumed['artifacts'][0]['path']).metadata.num_rows == resumed['records'],
                        'Resumed reference Parquet row coverage differs')
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.partial.parquet')
            writer = None
            bucket_input, bucket_selected = 0, 0
            bucket_permissions = Counter()
            try:
                for item in bucket['artifacts']:
                    input_path = _contained(candidate, item['path'])
                    before = stamp(input_path)
                    decision = input_path.stem
                    actual = 0
                    input_columns = COLUMNS + ([] if baseline else ['processing_policy', 'task_route', 'image_loss_allowed'])
                    for batch in pq.ParquetFile(input_path).iter_batches(batch_size=batch_size, columns=input_columns):
                        table = pa.Table.from_batches([batch])
                        positions = coverage.consume(table)
                        actual += len(table)
                        words = flags[positions] if baseline else None
                        mask = baseline['reader'].policy_arrays(words)[2] if baseline else np.full(len(table), decision == 'keep')
                        chosen = table.filter(pa.array(mask))
                        exported = _export_table(chosen, source_key, source_kind,
                                                 words[mask] if baseline else None,
                                                 baseline['reader'] if baseline else None)
                        if not resumed:
                            if writer is None:
                                writer = pq.ParquetWriter(temp, exported.schema, compression='zstd')
                            writer.write_table(exported)
                        bucket_selected += len(exported)
                        enabled = pc.sum(exported['image_loss_allowed'].cast(pa.int64())).as_py() or 0
                        bucket_permissions['image_loss_allowed'] += enabled
                        bucket_permissions['image_loss_disabled'] += len(exported) - enabled
                    require(actual == bucket['counts'].get(decision, 0), 'Candidate artifact row count differs')
                    require(stamp(input_path) == before, 'Candidate changed during export')
                    bucket_input += actual
                require(bucket_input == bucket['records'], 'Candidate input coverage differs')
                if not resumed and writer is None:
                    empty = pa.Table.from_batches([], schema=pq.read_schema(input_path)).select(input_columns)
                    exported = _export_table(empty, source_key, source_kind,
                        np.array([], dtype=np.uint16) if baseline else None,
                        baseline['reader'] if baseline else None)
                    pq.write_table(exported, temp, compression='zstd')
            finally:
                if writer is not None:
                    writer.close()
            if not resumed:
                os.replace(temp, path)
                receipt = dict(complete=True, input_records=bucket_input, records=bucket_selected,
                               counts=dict(bucket_permissions), artifacts=[dict(artifact(path, output), rows=bucket_selected)])
                atomic_json(receipt_path, receipt)
            else:
                receipt = resumed
                require(receipt['input_records'] == bucket_input and receipt['records'] == bucket_selected
                        and receipt['counts'] == dict(bucket_permissions), 'Resumed export coverage differs')
            receipts.append(receipt)
            total += bucket_input
            selected += bucket_selected
            permission_counts.update(bucket_permissions)
            progress.update(updated_at=time.time(), processed_records=total, selected_records=selected,
                            buckets_complete=len(receipts), last_bucket=bucket['bucket'])
            atomic_json(output / 'PROGRESS.json', progress)
        coverage.complete()
        expected = baseline['baseline']['selected_records'] if baseline else manifest['counts'].get('keep', 0)
        require(total == manifest['records'] and selected == expected, 'Selected reference count differs')
        for value in bindings.values():
            require(stamp(value['path']) == value['stamp'], 'Native source binding changed during export')
        counts = dict(input_records=total, selected_records=selected, unselected_records=total - selected,
                      **dict(permission_counts))
        result = _publish(output, dict(schema=INDEX_SCHEMA, complete=True, records=selected, counts=counts,
            identity_sha256=file_sha(output / 'IDENTITY.json'), sources=[source],
            files=[item for receipt in receipts for item in receipt['artifacts']],
            original_records_modified=False, original_image_bytes_read=0, image_payloads_copied=0,
            baseline_selected_preserved=bool(baseline), task_policy=TASK_POLICY))
        progress.update(status='complete', updated_at=time.time())
        atomic_json(output / 'PROGRESS.json', progress)
        return result


def export_baseline_index(loss_view, candidate_index, output, *, batch_size=65536):
    return _export(candidate_index, output, loss_view=loss_view, batch_size=batch_size)


def export_candidate_index(candidate_index, output, *, batch_size=65536):
    return _export(candidate_index, output, batch_size=batch_size)


def _reconciliation_inputs(root):
    root = Path(root).resolve(strict=True)
    complete = _json(root / 'COMPLETE.json')
    require(complete.get('complete') is True and complete.get('scope') == 'full_run'
            and complete.get('schema') == 'corpus.asset_reconciliation.v1',
            'Expected completed full image reconciliation')
    files = []
    seen = set()
    for bucket in complete['buckets']:
        receipt = _contained(root, bucket['receipt_path'])
        require(file_sha(receipt) == bucket['receipt_sha256'] and bucket['bucket'] not in seen,
                'Reconciliation receipt binding differs')
        seen.add(bucket['bucket'])
        value = _json(receipt)
        items = [x for x in value['artifacts'] if Path(x['path']).name == 'assets.parquet']
        require(len(items) == 1, 'Reconciliation requires one asset table per bucket')
        validate_artifacts(root, dict(artifacts=items))
        files.append(dict(items[0], path=str(_contained(root, items[0]['path']))))
    require(files, 'Reconciliation contains no asset tables')
    return dict(completion=_binding(root / 'COMPLETE.json'), files=files)


def _copy(con, query, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.parquet')
    con.execute(f'COPY ({query}) TO {quoted(temp)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)')
    os.replace(temp, path)
    return path


def _stage(con, output, name, build):
    """Bound, restartable metadata stages; a receipt commits only complete files."""
    marker = output / 'stages' / (name + '.json')
    if marker.exists():
        value = _json(marker)
        validate_artifacts(output, value)
        return [output / x['path'] for x in value['artifacts']]
    paths = build()
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, dict(complete=True, artifacts=[artifact(p, output) for p in paths]))
    return paths


def merge_selected_indices(baseline_index, additional_indices, output, *, reconciliations=(),
                           threads=8, memory_limit='8GB'):
    """Publish a lightweight union; baseline selection is never reranked/deleted.

    Additional indices are ordered by desired representative priority. They must
    be produced by export_candidate_index. Completed reconciliations authorize
    exact pixel reuse only; missing or ambiguous pixel evidence uses encoded SHA.
    """
    roots = [Path(baseline_index).resolve(strict=True)] + [Path(p).resolve(strict=True) for p in additional_indices]
    require(len(roots) == len(set(roots)), 'Repeated selected index')
    manifests = [verify_reference_release(p) for p in roots]
    require(manifests[0].get('baseline_selected_preserved') is True
            and all(m['schema'] == INDEX_SCHEMA for m in manifests), 'Union requires exported baseline and candidate indices')
    source_keys = [s['source_key'] for m in manifests for s in m['sources']]
    require(len(source_keys) == len(set(source_keys)), 'Source index supplied more than once')
    evidence = [_reconciliation_inputs(p) for p in reconciliations]
    identity = dict(schema=UNION_SCHEMA, key_policy=KEY_POLICY, code_sha256=file_sha(__file__),
                    inputs=[_binding(p / 'manifest.json') for p in roots], reconciliation=evidence,
                    baseline_rows_always_preserved=True, native_bytes_read=0, image_bytes_read=0)
    output = Path(output).resolve()
    require(all(output != p and not output.is_relative_to(p) for p in roots), 'Union output overlaps selected index')
    with locked(output):
        bind_identity(output, identity)
        if (output / 'CURRENT.json').exists():
            return verify_reference_release(output)
        con = connection(output, threads, memory_limit)
        try:
            sources = []
            for priority, (root, manifest) in enumerate(zip(roots, manifests)):
                paths = [_contained(root, x['path']) for x in manifest['files']]
                sources.append(f'SELECT *, {priority}::INTEGER AS source_priority FROM {parquet_sql(paths)}')
            con.execute('CREATE VIEW refs AS ' + ' UNION ALL '.join(sources))
            n = con.execute('SELECT count(*) FROM refs').fetchone()[0]
            require(n == sum(m['records'] for m in manifests), 'Union input row count differs')
            invalid = con.execute("""SELECT count(*) FROM refs WHERE
                (image_loss_allowed AND task_route <> 't2i') OR image_loss_allowed IS NULL OR
                text_hash IS NULL OR native_format IS NULL OR category IS NULL OR
                source_key IS NULL OR record_id IS NULL""").fetchone()[0]
            require(invalid == 0, 'Invalid union task permission or exact supervision key')
            repeated = con.execute('SELECT count(*) FROM (SELECT source_key, record_id FROM refs GROUP BY ALL HAVING count(*)>1)').fetchone()[0]
            require(repeated == 0, 'Duplicate selected source record identity')

            def build_identity():
                from corpus_pipeline.dedup.asset_release import identity_key_sql
                evidence_paths = [x['path'] for item in evidence for x in item['files']]
                if evidence_paths:
                    con.execute('CREATE VIEW observations AS SELECT *, ' + identity_key_sql() +
                                ' AS proven_identity FROM ' + parquet_sql(evidence_paths, union_by_name=True))
                    # The decoded digest includes oriented RGBA geometry; explicitly
                    # retain geometry too so inconsistent producer metadata cannot merge.
                    proof = """SELECT asset_id,
                        CASE WHEN count(DISTINCT CASE WHEN starts_with(proven_identity,'pixel:')
                            THEN proven_identity || ':' || original_width::VARCHAR || 'x' || original_height::VARCHAR END)=1
                        THEN min(CASE WHEN starts_with(proven_identity,'pixel:')
                            THEN proven_identity || ':' || original_width::VARCHAR || 'x' || original_height::VARCHAR END)
                        ELSE 'bytes:' || asset_id END AS exact_identity
                        FROM observations GROUP BY asset_id"""
                    con.execute('CREATE VIEW proof AS ' + proof)
                    query = """SELECT used.asset_id, coalesce(proof.exact_identity, 'bytes:' || used.asset_id) AS exact_identity
                        FROM (SELECT DISTINCT unnest(image_ids) AS asset_id FROM refs) used
                        LEFT JOIN proof USING(asset_id)"""
                else:
                    query = "SELECT DISTINCT unnest(image_ids) AS asset_id, 'bytes:' || asset_id AS exact_identity FROM refs"
                return [_copy(con, query, output / 'work' / 'image_identities.parquet')]
            identities = _stage(con, output, 'identities', build_identity)[0]
            con.execute('CREATE VIEW image_identity AS SELECT * FROM ' + parquet_sql([identities]))

            def build_keyed():
                # UNNEST by slot preserves repeated images and their original order.
                query = """WITH slots AS (
                    SELECT source_key, record_id, unnest(image_ids) AS asset_id,
                           generate_subscripts(image_ids,1) AS slot FROM refs), ordered AS (
                    SELECT source_key, record_id, list(exact_identity ORDER BY slot) AS exact_image_ids
                    FROM slots JOIN image_identity USING(asset_id) GROUP BY source_key,record_id), keyed AS (
                    SELECT r.*, r.source_key || ':' || r.record_id AS reference_id,
                        coalesce(o.exact_image_ids, []::VARCHAR[]) AS exact_image_ids,
                        sha256(to_json(struct_pack(text_hash := r.text_hash, native_format := r.native_format,
                          category := r.category, images := coalesce(o.exact_image_ids, []::VARCHAR[]),
                          task_route := r.task_route, image_loss_allowed := r.image_loss_allowed))) AS union_signature
                    FROM refs r LEFT JOIN ordered o USING(source_key,record_id))
                    SELECT *, substr(union_signature,1,2) AS union_bucket FROM keyed"""
                # A bounded number of output partitions keeps each dedup window small.
                directory = output / 'work' / 'keyed'
                directory.mkdir(parents=True, exist_ok=True)
                # No receipt means an earlier attempt did not commit; overwrite its
                # own temporary stage only, never any source or published artifact.
                import shutil
                for item in directory.iterdir():
                    if item.is_dir(): shutil.rmtree(item)
                    else: item.unlink()
                con.execute(f'COPY ({query}) TO {quoted(directory)} (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (union_bucket))')
                return sorted(directory.glob('union_bucket=*/*.parquet'))
            keyed_paths = _stage(con, output, 'keyed', build_keyed)
            grouped = {}
            for path in keyed_paths:
                grouped.setdefault(path.parent.name.split('=', 1)[1], []).append(path)
            selected_files, provenance_files = [], []
            counts = Counter(input_records=n)
            for bucket, paths in sorted(grouped.items()):
                require(SHA.fullmatch(bucket + '0' * 62) is not None, 'Invalid union signature bucket')
                con.execute('CREATE OR REPLACE VIEW bucket AS SELECT * FROM ' + parquet_sql(paths, hive_partitioning=False))
                con.execute('''CREATE OR REPLACE VIEW ranked AS SELECT *,
                    first_value(reference_id) OVER (PARTITION BY union_signature
                        ORDER BY source_priority, source_key, record_id) AS representative_id
                    FROM bucket''')
                def build_bucket():
                    selected = _copy(con, '''SELECT * EXCLUDE(representative_id) FROM ranked
                        WHERE source_priority=0 OR reference_id=representative_id''',
                        output / 'references' / (bucket + '.parquet'))
                    provenance = _copy(con, '''SELECT source_key, record_id, reference_id, union_signature,
                        CASE WHEN source_priority=0 THEN reference_id ELSE representative_id END AS selected_reference_id,
                        source_priority=0 OR reference_id=representative_id AS selected
                        FROM ranked''', output / 'provenance' / (bucket + '.parquet'))
                    return [selected, provenance]
                selected, provenance = _stage(con, output, 'bucket-' + bucket, build_bucket)
                selected_files.append(dict(artifact(selected, output), rows=pq.ParquetFile(selected).metadata.num_rows))
                provenance_files.append(dict(artifact(provenance, output), rows=pq.ParquetFile(provenance).metadata.num_rows))
            selected_paths = [output / x['path'] for x in selected_files]
            con.execute('CREATE VIEW selected AS SELECT * FROM ' + parquet_sql(selected_paths))
            counts['selected_records'], counts['baseline_selected_records'], counts['image_loss_allowed'] = con.execute(
                'SELECT count(*), count(*) FILTER (WHERE source_priority=0), count(*) FILTER (WHERE image_loss_allowed) FROM selected').fetchone()
            counts['image_loss_disabled'] = counts['selected_records'] - counts['image_loss_allowed']
            counts['exact_duplicates_removed'] = n - counts['selected_records']
            counts['additional_selected_records'] = counts['selected_records'] - counts['baseline_selected_records']
            require(counts['baseline_selected_records'] == manifests[0]['records'], 'Union changed baseline selection')
            require(sum(x['rows'] for x in provenance_files) == n, 'Union provenance coverage differs')

            def build_assets():
                query = """WITH slots AS (SELECT reference_id, source_key, record_id,
                        unnest(image_ids) AS asset_id, unnest(exact_image_ids) AS exact_identity,
                        generate_subscripts(image_ids,1)-1 AS image_index, source_priority FROM selected),
                    ranked AS (SELECT *, row_number() OVER (PARTITION BY exact_identity
                        ORDER BY source_priority, reference_id, image_index) AS rank FROM slots)
                    SELECT * EXCLUDE(rank,source_priority) FROM ranked WHERE rank=1"""
                return [_copy(con, query, output / 'assets.parquet')]
            assets = _stage(con, output, 'assets', build_assets)[0]
            counts['unique_image_identities'] = pq.ParquetFile(assets).metadata.num_rows
            all_sources = [s for m in manifests for s in m['sources']]
            return _publish(output, dict(schema=UNION_SCHEMA, complete=True, records=counts['selected_records'],
                counts=dict(counts), files=selected_files,
                auxiliary_files=provenance_files + [artifact(assets, output)], sources=all_sources,
                input_manifests=identity['inputs'], reconciliation=evidence, key_policy=KEY_POLICY,
                identity_sha256=file_sha(output / 'IDENTITY.json'), baseline_selected_preserved=True,
                original_records_modified=False, image_payloads_copied=0, original_image_bytes_read=0,
                native_metadata_bytes_read=0, sscd_candidates_used_for_deletion=False,
                image_identity_mode='reused_pixel_geometry_else_encoded_sha' if evidence else 'encoded_sha_only',
                reader='corpus_pipeline.unified.baseline_union.ReferenceReader',
                limitation='Reference view requires its immutable native metadata and original dataset roots; no training Catalog bridge.'))
        finally:
            con.close()


class ReferenceReader:
    """Verify references and lazily read original native rows with explicit relocation.

    This reader adds permissions without rewriting the source row. Its image
    references remain original; DCVLM read_image/decode_image should be delegated
    to the frozen release reader, which preserves the approved repair behavior.
    """
    def __init__(self, release, *, native_roots=None):
        self.root = Path(release).resolve(strict=True)
        self.manifest = verify_reference_release(self.root, verify_payloads=False)
        self.native_roots = native_roots or {}
        self.bindings = {(s['source_key'], f['source_path']): f
                         for s in self.manifest['sources'] for f in s['native_files']}
        self._checked = {}

    def iter_references(self, batch_size=8192):
        for item in self.manifest['files']:
            validate_artifacts(self.root, dict(artifacts=[item]))
            path = _contained(self.root, item['path'])
            require(pq.ParquetFile(path).metadata.num_rows == item['rows'], 'Reference row coverage differs')
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()

    def read_record(self, reference):
        key = (reference['source_key'], reference['original_parquet_path'])
        require(key in self.bindings, 'Unknown native source binding')
        binding = self.bindings[key]
        original = Path(key[1])
        candidates = [(Path(old), Path(new)) for old, new in self.native_roots.items()
                      if original.is_relative_to(Path(old))]
        if candidates:
            old, new = max(candidates, key=lambda pair: len(pair[0].parts))
            new = new.resolve(strict=True)
            path = (new / original.relative_to(old)).resolve(strict=True)
            require(path.is_relative_to(new), 'Relocated native source escaped its root')
        else:
            path = original.resolve(strict=True)
        before = stamp(path)
        if self._checked.get(str(path)) != before:
            require(file_sha(path) == binding['source_sha256'] and stamp(path) == before,
                    'Native source SHA changed')
            self._checked[str(path)] = before
        reader = pq.ParquetFile(path)
        require(reader.metadata.num_rows == binding['rows'], 'Native source row coverage changed')
        index = reference['row_index']
        require(type(index) is int and 0 <= index < binding['rows'], 'Invalid native row locator')
        for group in range(reader.num_row_groups):
            size = reader.metadata.row_group(group).num_rows
            if index < size:
                row = reader.read_row_group(group).slice(index, 1).to_pylist()[0]
                require(stamp(path) == before, 'Native source changed during reading')
                require(digest(row) == reference['source_record_sha256'],
                        'Reference locator does not match the bound native record')
                break
            index -= size
        # Never mutate original JSON columns: a derived envelope holds permissions.
        return dict(native_record=row, source_reference=reference,
                    processing_policy=TASK_POLICY, task_route=reference['task_route'],
                    image_loss_allowed=reference['image_loss_allowed'],
                    dcvlm_image_loss_allowed=reference.get('dcvlm_image_loss_allowed'),
                    dcvlm_policy_flags=reference.get('dcvlm_policy_flags'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export-baseline')
    p.add_argument('--loss-view', required=True); p.add_argument('--candidate-index', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('export-candidate')
    p.add_argument('--candidate-index', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('merge')
    p.add_argument('--baseline-index', required=True); p.add_argument('--additional-index', action='append', default=[])
    p.add_argument('--reconciliation', action='append', default=[]); p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=8); p.add_argument('--memory-limit', default='8GB')
    p = sub.add_parser('verify'); p.add_argument('--release', required=True)
    args = parser.parse_args(argv)
    if args.command == 'export-baseline':
        value = export_baseline_index(args.loss_view, args.candidate_index, args.output)
    elif args.command == 'export-candidate':
        value = export_candidate_index(args.candidate_index, args.output)
    elif args.command == 'merge':
        value = merge_selected_indices(args.baseline_index, args.additional_index, args.output,
            reconciliations=args.reconciliation, threads=args.threads, memory_limit=args.memory_limit)
    else:
        value = verify_reference_release(args.release)
    print(json.dumps(dict(schema=value['schema'], records=value['records'], counts=value['counts']), indent=2))


if __name__ == '__main__':
    main()
