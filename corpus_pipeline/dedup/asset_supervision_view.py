"""One exact image asset, multiple ordered supervision references.

This derived view copies metadata only. It does not rewrite original images,
merge SSCD neighbours, flatten conversations, or change training sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from corpus_pipeline.postprocess.reconcile import (
    artifact, atomic_json, bind_identity, digest, file_sha, locked, quoted,
    require, stamp, validate_artifacts,
)
from .source_view import source_bindings
from .training_release import connection, parquet_sql

SCHEMA = 'corpus.asset_supervision_view.v1'
FILES = ('assets.parquet', 'supervisions.parquet', 'asset_supervisions.parquet',
         'supervision_provenance.parquet')


def _identifier(value):
    return '"' + value.replace('"', '""') + '"'


def _read_manifest(root):
    root = Path(root).resolve(strict=True)
    complete = json.loads((root / 'COMPLETE.json').read_text())
    require(complete.get('complete') is True and complete.get('schema') == SCHEMA
            and complete['manifest_sha256'] == file_sha(root / 'manifest.json'),
            'Asset supervision view is incomplete or changed')
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest.get('schema') == SCHEMA, 'Unsupported asset supervision view')
    require(manifest.get('identity_sha256') == file_sha(root / 'IDENTITY.json'),
            'Asset supervision input identity changed')
    identity=json.loads((root / 'IDENTITY.json').read_text())
    require(identity.get('schema') == SCHEMA
            and manifest.get('input_candidate_manifest_sha256') == identity.get('candidate_manifest_sha256')
            and manifest.get('input_candidate_complete_sha256') == identity.get('candidate_complete_sha256'),
            'Asset supervision candidate binding changed')
    require(complete.get('counts') == manifest.get('counts'), 'Asset supervision completion counts differ')
    validate_artifacts(root, manifest)
    require({item['path'] for item in manifest['artifacts']} == set(FILES),
            'Asset supervision view file coverage changed')
    return root, manifest


def build(candidate, output, *, threads=2, memory_limit='4GB'):
    """Join a sealed exact-identity candidate index to SHA-bound native metadata."""
    candidate = Path(candidate).resolve(strict=True)
    output = Path(output).resolve()
    require(not output.is_relative_to(candidate) and not candidate.is_relative_to(output),
            'Derived output must not overlap the candidate input')
    complete = json.loads((candidate / 'COMPLETE.json').read_text())
    require(complete.get('complete') is True and
            complete['manifest_sha256'] == file_sha(candidate / 'manifest.json'),
            'Candidate index is incomplete or changed')
    manifest = json.loads((candidate / 'manifest.json').read_text())
    require(manifest.get('same_image_different_supervision_preserved') is True
            and manifest.get('sscd_candidates_used_for_deletion') is False,
            'Input must preserve distinct supervision and use exact image equivalence')
    bindings = source_bindings(candidate, manifest)
    paths = []
    for bucket in manifest['buckets']:
        validate_artifacts(candidate, bucket)
        paths.extend(candidate / item['path'] for item in bucket['artifacts'])
    require(paths and bindings, 'Candidate source bindings are empty')
    for path, binding in bindings.items():
        require(file_sha(path) == binding['sha256'] and stamp(path) == binding['stamp'],
                'Original native metadata changed')
        require(not Path(path).is_relative_to(output), 'Output overlaps original native metadata')
    identity = {'schema': SCHEMA, 'code_sha256': file_sha(__file__),
                'candidate': str(candidate), 'candidate_complete_sha256': file_sha(candidate / 'COMPLETE.json'),
                'candidate_manifest_sha256': complete['manifest_sha256']}
    with locked(output):
        bind_identity(output, identity)
        if (output / 'COMPLETE.json').exists():
            _read_manifest(output)
            return json.loads((output / 'COMPLETE.json').read_text())
        con = connection(output, threads, memory_limit)
        try:
            con.execute('CREATE TEMP VIEW decisions AS SELECT * FROM ' + parquet_sql(paths))
            decision_columns = {row[0] for row in con.execute('DESCRIBE decisions').fetchall()}
            con.execute('CREATE TEMP VIEW native AS SELECT * FROM ' +
                        parquet_sql(sorted(bindings), filename=True, file_row_number=True, union_by_name=True))
            native_columns = [row[0] for row in con.execute('DESCRIBE native').fetchall()
                              if row[0] not in {'filename', 'file_row_number'}]
            require({'segments', 'images', 'sample_id', 'text_hash', 'image_count'} <= set(native_columns),
                    'Native metadata lacks ordered supervision or image bindings')
            require(all(set(pq.ParquetFile(path).schema_arrow.names) == set(native_columns) for path in bindings),
                    'Native metadata schemas differ; normalize before constructing the asset view')
            native_json = 'to_json(struct_pack(' + ','.join(
                _identifier(key) + ':=n.' + _identifier(key) for key in native_columns) + '))'
            permission = ('coalesce(d.image_loss_allowed,NOT coalesce(d.unresolved_screening,true))' if 'image_loss_allowed' in decision_columns
                          else 'NOT coalesce(d.unresolved_screening,true)')
            direction = """CASE WHEN d.category IN ('single_i2t','multi_i2t') THEN 'i2t'
                     WHEN d.category IN ('single_t2i','multi_t2i') THEN 't2i'
                     WHEN d.category='pure_text' THEN 'text' ELSE 'mixed' END"""
            if 'task_route' in decision_columns:
                direction = 'coalesce(d.task_route,(' + direction + '))'
            # The original candidate retains the screening gate. I2T permission
            # cannot be promoted by another T2I occurrence of the same image.
            con.execute(f'''CREATE TEMP VIEW joined AS SELECT d.*,
                n.segments,n.images AS original_image_refs,{native_json} AS native_record_json,
                {direction} AS task_direction,
                (({direction})='t2i' AND {permission}) AS task_image_loss_allowed
                FROM decisions d JOIN native n
                  ON d.original_parquet_path=n.filename AND d.row_index=n.file_row_number
                WHERE d.decision='keep' OR d.decision_reason='identical_ordered_supervision_and_confirmed_images' ''')
            expected = con.execute("SELECT count(*) FROM decisions WHERE decision='keep' OR decision_reason='identical_ordered_supervision_and_confirmed_images'").fetchone()[0]
            count, unique, invalid = con.execute('''SELECT count(*),count(DISTINCT record_id),
                count(*) FILTER(WHERE len(original_image_refs)<>len(image_ids)
                  OR len(canonical_image_ids)<>len(image_ids) OR segments IS NULL)
                FROM joined''').fetchone()
            require(count == unique == expected and invalid == 0, 'Native supervision coverage or image order mismatch')
            con.execute('''CREATE TEMP VIEW selected AS SELECT * FROM joined WHERE decision='keep' ''')
            kept = con.execute('SELECT count(*) FROM selected').fetchone()[0]
            require(kept == manifest['counts'].get('keep', 0), 'Selected supervision count differs from candidate')
            require(con.execute('SELECT count(DISTINCT record_signature) FROM selected').fetchone()[0] == kept,
                    'Exact duplicate supervision remains in candidate keep rows')
            con.execute('''CREATE TEMP VIEW occurrences AS SELECT s.record_signature AS supervision_id,
                s.canonical_image_ids[u.position] AS canonical_asset_id,
                s.image_ids[u.position] AS original_asset_id,u.position-1 AS image_index,
                to_json(u.image_ref) AS image_ref_json
                FROM selected s, UNNEST(s.original_image_refs) WITH ORDINALITY u(image_ref,position)''')
            require(con.execute("""SELECT count(*) FROM occurrences WHERE
                canonical_asset_id IS NULL OR original_asset_id IS NULL OR
                NOT regexp_full_match(canonical_asset_id,'[0-9a-f]{64}') OR
                NOT regexp_full_match(original_asset_id,'[0-9a-f]{64}') OR
                json_extract_string(image_ref_json,'$.sha256') IS DISTINCT FROM original_asset_id""").fetchone()[0] == 0,
                'Image references do not match the original encoded identities')
            queries = {
                'supervisions.parquet': '''SELECT record_signature AS supervision_id,record_id,category,
                    task_direction AS direction,task_image_loss_allowed AS image_loss_allowed,
                    canonical_image_ids AS ordered_image_ids,image_ids AS original_image_ids,
                    segments,original_image_refs,source_record_sha256,native_record_json,
                    original_parquet_path,row_index FROM selected ORDER BY supervision_id''',
                'assets.parquet': '''SELECT canonical_asset_id,
                    arg_min(original_asset_id,struct_pack(prefer_canonical:=original_asset_id<>canonical_asset_id,
                        asset:=original_asset_id,reference:=image_ref_json)) AS representative_original_asset_id,
                    arg_min(image_ref_json,struct_pack(prefer_canonical:=original_asset_id<>canonical_asset_id,
                        asset:=original_asset_id,reference:=image_ref_json)) AS image_ref_json,
                    count(DISTINCT supervision_id)::BIGINT AS supervision_count,
                    count(DISTINCT original_asset_id)::BIGINT AS original_asset_count
                    FROM occurrences GROUP BY canonical_asset_id ORDER BY canonical_asset_id''',
                'asset_supervisions.parquet': '''SELECT canonical_asset_id,supervision_id,
                    list(image_index ORDER BY image_index) AS image_indices
                    FROM occurrences GROUP BY canonical_asset_id,supervision_id
                    ORDER BY canonical_asset_id,supervision_id''',
                'supervision_provenance.parquet': '''SELECT j.record_signature AS supervision_id,j.record_id,
                    j.source,j.source_record_sha256,j.original_parquet_path,j.row_index,j.native_record_json,
                    j.decision='keep' AS selected_representative
                    FROM joined j JOIN selected s ON s.record_id=
                      CASE WHEN j.decision='keep' THEN j.record_id ELSE j.duplicate_of_record_id END
                      AND j.record_signature=s.record_signature
                    ORDER BY supervision_id,j.record_id''',
            }
            for name, query in queries.items():
                temporary = output / (name + '.partial')
                con.execute(f'COPY ({query}) TO {quoted(temporary)} (FORMAT PARQUET,COMPRESSION ZSTD)')
                temporary.replace(output / name)
            counts = {name.removesuffix('.parquet'): pq.ParquetFile(output / name).metadata.num_rows for name in FILES}
            require(counts['supervision_provenance'] == expected,
                    'Duplicate supervision provenance lost its retained representative')
            for path, binding in bindings.items():
                require(stamp(path) == binding['stamp'], 'Native metadata changed during view build')
            result = {'schema': SCHEMA, 'complete': True, 'counts': counts,
                      'scope': manifest['scope'], 'identity_sha256': file_sha(output / 'IDENTITY.json'),
                      'input_candidate_manifest_sha256': identity['candidate_manifest_sha256'],
                      'input_candidate_complete_sha256': identity['candidate_complete_sha256'],
                      'image_payloads_copied': False, 'original_records_modified': False,
                      'same_image_different_supervision_preserved': True,
                      'one_asset_row_per_exact_canonical_image': True,
                      'training_sampler_changed': False,
                      'multi_image_order_preserved': True,
                      'artifacts': [artifact(output / name, output) for name in FILES]}
            atomic_json(output / 'manifest.json', result)
            done = {'schema': SCHEMA, 'complete': True, 'counts': counts,
                    'manifest_sha256': file_sha(output / 'manifest.json')}
            atomic_json(output / 'COMPLETE.json', done)
            return done
        finally:
            con.close()


def _decoded_supervision(row):
    raw = json.loads(row.pop('native_record_json'))
    require(digest(raw) == row['source_record_sha256'], 'Supervision no longer matches its bound native record')
    row['native_record'] = raw
    return row


def iter_assets(view, *, batch_size=8192):
    """Stream one metadata row per exact image. Image bytes are not read."""
    require(type(batch_size) is int and batch_size > 0, 'batch_size must be positive')
    root, _ = _read_manifest(view)
    for batch in pq.ParquetFile(root / 'assets.parquet').iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            row['image_ref'] = json.loads(row.pop('image_ref_json'))
            yield row


def iter_supervisions(view, *, batch_size=8192):
    """Stream all distinct ordered supervision with shared canonical image IDs."""
    require(type(batch_size) is int and batch_size > 0, 'batch_size must be positive')
    root, _ = _read_manifest(view)
    for batch in pq.ParquetFile(root / 'supervisions.parquet').iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield _decoded_supervision(row)


def iter_asset_supervisions(view, canonical_asset_id):
    """Read all complete tasks using an image, preserving repeated image slots."""
    import duckdb
    root, _ = _read_manifest(view)
    con = duckdb.connect()
    try:
        cursor = con.execute('''SELECT s.*,l.image_indices FROM read_parquet(?) s
            JOIN read_parquet(?) l USING(supervision_id) WHERE l.canonical_asset_id=?
            ORDER BY s.supervision_id''', [str(root / 'supervisions.parquet'),
                                         str(root / 'asset_supervisions.parquet'), canonical_asset_id])
        names = [item[0] for item in cursor.description]
        while rows := cursor.fetchmany(128):
            for row in rows:
                yield _decoded_supervision(dict(zip(names, row)))
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory-limit', default='4GB')
    args = parser.parse_args()
    print(json.dumps(build(args.candidate, args.output, threads=args.threads, memory_limit=args.memory_limit)))


if __name__ == '__main__':
    main()
