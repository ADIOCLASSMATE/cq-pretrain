"""Reconcile immutable frontpass image observations by exact encoded-image SHA.

One partition pass, then 256 bounded DuckDB aggregations. No original images,
models, queue mutations, API calls, vector reads or training-record deletion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

SCHEMA = 'corpus.asset_reconciliation.v1'
BUCKET_SCHEMA = 'corpus.asset_reconciliation.bucket.v1'
PARTITION_SCHEMA = 'corpus.asset_reconciliation.partition.v1'
HEX64 = re.compile(r'^[0-9a-f]{64}$')
BUCKETS = tuple(f'{i:02x}' for i in range(256))
VECTOR_BYTES = 512 * 4


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}


def quoted(value):
    return "'" + str(value).replace("'", "''") + "'"


def artifact(path, root):
    return {'path': str(Path(path).relative_to(root)), 'sha256': file_sha(path),
            'size_bytes': Path(path).stat().st_size}


def validate_artifacts(root, receipt):
    for item in receipt['artifacts']:
        path = (root / item['path']).resolve()
        require(path.is_relative_to(root), 'Output artifact escaped output root')
        require(path.is_file() and path.stat().st_size == item['size_bytes'], 'Missing/changed output artifact')
        require(file_sha(path) == item['sha256'], 'Output artifact SHA mismatch')


def load_inputs(run_root, selected_task_ids=None):
    """The final index binds exactly one already-committed attempt per task."""
    root = Path(run_root).resolve()
    raw = (root / 'frontpass_index.json').read_bytes()
    index_hash = hashlib.sha256(raw).hexdigest()
    complete_raw = (root / 'FRONTPASS_COMPLETE.json').read_bytes()
    complete, index = json.loads(complete_raw), json.loads(raw)
    require(complete.get('complete') is True and index.get('complete') is True, 'Frontpass is not complete')
    require(complete.get('index_sha256') == index_hash, 'Final index SHA mismatch')
    require({k: v for k, v in complete.items() if k != 'index_sha256'} ==
            {k: v for k, v in index.items() if k != 'parts'}, 'Final index and completion disagree')
    require(index.get('scope') == 'offline_frontpass_only', 'Unexpected input release scope')
    selected = None if selected_task_ids is None else tuple(sorted(set(selected_task_ids)))
    require(selected is None or bool(selected), 'Empty bounded selection')
    parts, seen, images_seen, total_records = [], set(), set(), 0
    mixed_snapshot = None
    for original in index['parts']:
        task = original['task']; task_id = task['task_id']
        require(isinstance(task_id, str) and task_id and '/' not in task_id and '..' not in task_id,
                'Invalid task identity')
        require(task_id not in seen, 'Duplicate final-index task')
        seen.add(task_id)
        require(type(task.get('row_start')) is int and type(task.get('row_end')) is int
                and 0 <= task['row_start'] < task['row_end'], 'Invalid task row range')
        total_records += task['row_end'] - task['row_start']
        require(isinstance(task.get('source'), str) and bool(task['source']), 'Missing source identity')
        if selected is not None and task_id not in selected:
            continue
        mixed_sources = None
        if task['source'] == '__mixed__':
            if mixed_snapshot is None:
                from corpus_pipeline.production.records import snapshot
                require(isinstance(index.get('snapshot_root'), str)
                        and isinstance(index.get('snapshot_fingerprint'), dict),
                        'Mixed-source task lacks a bound native snapshot')
                mixed_snapshot = snapshot(index['snapshot_root'])
                require(mixed_snapshot['fingerprint'] == index['snapshot_fingerprint']
                        and mixed_snapshot['manifest'].get('catalog_contract') == 'mixed_source_tar.v1',
                        'Mixed-source snapshot contract or fingerprint differs')
            entry = mixed_snapshot['catalog_index'].get(str(Path(task['parquet_path']).resolve()))
            require(entry is not None and entry.get('source') == '__mixed__'
                    and entry.get('contract') == 'mixed_source_tar.v1'
                    and entry.get('source_shard') == task.get('source_shard')
                    and entry.get('collection') == task.get('collection', 'dcvlm')
                    and task['row_end'] <= entry['rows'], 'Mixed-source task differs from bound catalog')
            mixed_sources = entry.get('sources')
            require(isinstance(mixed_sources, list) and mixed_sources
                    and all(isinstance(value, str) and 0 < len(value) <= 512 and value.strip() == value
                            and value != '__mixed__' and all(ord(char) >= 32 and ord(char) != 127 for char in value)
                            for value in mixed_sources)
                    and len(mixed_sources) == len(set(mixed_sources)), 'Invalid mixed-source catalog identities')
        bound = {}
        token = None
        for item in original['artifacts']:
            path = (root / item['path']).resolve()
            require(path.is_relative_to(root / 'artifacts' / task_id), 'Artifact outside committed task')
            relative = path.relative_to(root / 'artifacts' / task_id)
            require(len(relative.parts) == 2, 'Unexpected attempt artifact layout')
            if token is None:
                token = relative.parts[0]
            require(token == relative.parts[0], 'Mixed attempt artifacts')
            require(path.name not in bound and HEX64.fullmatch(item['sha256']), 'Invalid artifact binding')
            require(type(item.get('size_bytes')) is int and item['size_bytes'] >= 0, 'Invalid artifact size')
            bound[path.name] = {**item, 'absolute_path': str(path)}
        require({'images.parquet', 'summary.json', 'sscd.f32'} <= bound.keys(), 'Incomplete image/vector binding')
        require(bound['images.parquet']['absolute_path'] not in images_seen, 'Repeated image artifact')
        images_seen.add(bound['images.parquet']['absolute_path'])
        parts.append({'task': task, 'token': token, 'artifacts': bound,
                      'model_identity_sha256': original['model_identity_sha256'],
                      **({'mixed_sources': list(mixed_sources)} if mixed_sources is not None else {})})
    require(total_records == index['counts']['records'], 'Final index record coverage mismatch')
    require(selected is None or set(selected) <= seen, 'Unknown selected task')
    return {'root': root, 'index': index, 'index_sha256': index_hash,
            'complete_sha256': hashlib.sha256(complete_raw).hexdigest(), 'parts': parts,
            'selected_task_ids': selected, 'scope': 'full_run' if selected is None else 'bounded_validation'}


def verify_part(part):
    """Hash each image shard once before its sole partition read; vectors later."""
    bound = part['artifacts']
    image, summary, vector = (bound[name] for name in ('images.parquet', 'summary.json', 'sscd.f32'))
    before = stamp(image['absolute_path'])
    require(before['size'] == image['size_bytes'] and file_sha(image['absolute_path']) == image['sha256'],
            'Committed images.parquet changed')
    require(stamp(image['absolute_path']) == before, 'Images changed while hashing')
    summary_raw = Path(summary['absolute_path']).read_bytes()
    require(len(summary_raw) == summary['size_bytes'] and hashlib.sha256(summary_raw).hexdigest() == summary['sha256'],
            'Committed summary changed')
    value = json.loads(summary_raw)
    require(value['task'] == part['task'] and value['token'] == part['token'], 'Summary task/token mismatch')
    require(value['counts']['records'] == part['task']['row_end'] - part['task']['row_start'], 'Summary row mismatch')
    require(value['model_identity_sha256'] == part['model_identity_sha256'], 'Model identity mismatch')
    vector_stamp = stamp(vector['absolute_path'])
    require(vector_stamp['size'] == vector['size_bytes'] and vector_stamp['size'] % VECTOR_BYTES == 0,
            'SSCD file missing or size invalid; contents must be hashed by vector consumer')
    return {'images_file': image['absolute_path'], 'task_id': part['task']['task_id'],
            'source': part['task']['source'], 'attempt_token': part['token'],
            'allowed_sources': part.get('mixed_sources', [part['task']['source']]),
            'sscd_file': vector['absolute_path'], 'sscd_file_sha256': vector['sha256'],
            'sscd_file_size_bytes': vector['size_bytes'], 'model_identity_sha256': part['model_identity_sha256'],
            'expected_images': value['counts'].get('image_occurrences', 0),
            'image_stamp': before, 'vector_stamp': vector_stamp}


@contextmanager
def locked(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.LOCK').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def bind_identity(output, identity):
    path = output / 'IDENTITY.json'
    if path.exists():
        require(json.loads(path.read_text()) == identity, 'Output belongs to different input/code/config')
    else:
        require(not any(p.name != '.LOCK' for p in output.iterdir()), 'Nonempty unbound output directory')
        atomic_json(path, identity)


def connect(output, threads, memory_limit):
    import duckdb
    connection = duckdb.connect()
    connection.execute(f'SET threads = {threads}')
    connection.execute('SET memory_limit = ' + quoted(memory_limit))
    connection.execute('SET temp_directory = ' + quoted(output / 'duckdb_spill'))
    connection.execute('SET preserve_insertion_order = false')
    connection.execute('SET partitioned_write_max_open_files = 32')
    return connection


def stage_chunk(connection, output, identity_hash, number, parts, workers):
    import pyarrow as pa
    name = f'batch-{number:06d}'
    final = output / 'partitions' / name
    receipt_path = final / 'COMPLETE.json'
    part_digest = digest([{'task': p['task'], 'token': p['token'], 'artifacts': p['artifacts']} for p in parts])
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get('identity_sha256') == identity_hash and receipt.get('parts_sha256') == part_digest,
                'Partition checkpoint identity mismatch')
        validate_artifacts(output, receipt)
        return receipt
    require(not final.exists(), 'Incomplete partition destination; refusing to guess its ownership')
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + name + '-', dir=final.parent))
    try:
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
            metadata = list(pool.map(verify_part, parts))
        join_fields = ('images_file', 'task_id', 'source', 'allowed_sources', 'attempt_token', 'sscd_file',
                       'sscd_file_sha256', 'sscd_file_size_bytes', 'model_identity_sha256')
        connection.register('input_parts', pa.Table.from_pylist([{key: row[key] for key in join_fields} for row in metadata]))
        files = [row['images_file'] for row in metadata]
        expected = sum(row['expected_images'] for row in metadata)
        # Only the explicitly bound mixed-TAR contract may override a task's
        # source. Ordinary releases retain their original task-source semantics.
        invalid_sources = connection.execute("""SELECT count(*)
            FROM read_parquet(?, filename=true, hive_partitioning=false) i
            JOIN input_parts m ON i.filename=m.images_file
            WHERE m.source='__mixed__' AND (
                json_type(try_cast(i.provenance_json AS JSON),'$.source') IS DISTINCT FROM 'VARCHAR'
                OR NOT coalesce(list_contains(m.allowed_sources,
                    json_extract_string(try_cast(i.provenance_json AS JSON),'$.source')),false))""", [files]).fetchone()[0]
        require(invalid_sources == 0, 'Mixed-source observation lacks a valid bound source identity')
        query = """SELECT i.* EXCLUDE(filename), m.* EXCLUDE(images_file,source,allowed_sources),
            CASE WHEN m.source='__mixed__' THEN json_extract_string(i.provenance_json,'$.source')
                 ELSE m.source END AS source,
            CASE WHEN regexp_full_match(i.asset_id, '[0-9a-f]{64}') THEN substr(i.asset_id, 1, 2)
                 ELSE '__invalid__' END AS bucket
            FROM read_parquet(?, filename=true, hive_partitioning=false) i
            JOIN input_parts m ON i.filename=m.images_file"""
        count = connection.execute('COPY (' + query + ') TO ' + quoted(temporary / 'data') +
            " (FORMAT PARQUET, PARTITION_BY(bucket), COMPRESSION ZSTD, ROW_GROUP_SIZE 32768)", [files]).fetchone()[0]
        require(count == expected, 'Partition observation count does not match summaries')
        require(not (temporary / 'data' / 'bucket=__invalid__').exists(), 'Malformed asset SHA; refusing unsafe grouping')
        for row in metadata:
            require(stamp(row['images_file']) == row['image_stamp'], 'Images changed during partition scan')
            require(stamp(row['sscd_file']) == row['vector_stamp'], 'SSCD file changed during partition scan')
        files_out = sorted(temporary.glob('data/bucket=*/*.parquet'))
        artifacts = []
        for path in files_out:
            item = artifact(path, temporary)
            item['path'] = str((final / item['path']).relative_to(output))
            artifacts.append(item)
        receipt = {'schema': PARTITION_SCHEMA, 'identity_sha256': identity_hash, 'parts_sha256': part_digest,
                   'number': number, 'image_occurrences': count, 'input_parts': metadata,
                   'artifacts': artifacts, 'vector_integrity': 'size_bound_only_consumer_must_verify_sha256'}
        atomic_json(temporary / 'COMPLETE.json', receipt)
        os.replace(temporary, final)
        return receipt
    finally:
        connection.unregister('input_parts')
        if temporary.exists():
            shutil.rmtree(temporary)


# Decode/original-ref pixel hashes use different historical conventions. Only
# frontpass pixel_sha256 values are compared to each other, never to ref JSON.
OBSERVATIONS_SQL = r"""
WITH parsed AS (
 SELECT *, try_cast(original_ref_json AS JSON) AS ref,
   try_cast(provenance_json AS JSON) AS prov, try_cast(error_json AS JSON) AS errors
 FROM bucket_input
), fields AS (
 SELECT *, json_extract_string(prov,'$.origin') AS origin,
   coalesce(json_extract_string(prov,'$.processing_policy')='corpus.task_policy.v1'
      AND json_extract_string(prov,'$.task_route')='i2t'
      AND json_type(prov,'$.image_loss_allowed')='BOOLEAN'
      AND try_cast(json_extract(prov,'$.image_loss_allowed') AS BOOLEAN)=false,false) AS i2t_dedup_only,
   json_extract_string(prov,'$.policy_exception') AS policy_exception,
   json_extract_string(prov,'$.edit_source_origin') AS edit_parent,
   coalesce(json_type(ref)='OBJECT' AND json_extract_string(ref,'$.sha256')=asset_id
     AND length(json_extract_string(ref,'$.path'))>0
     AND (try_cast(json_extract(ref,'$.exists') AS BOOLEAN) IS NOT FALSE)
     AND CASE WHEN json_extract_string(ref,'$.kind')='tar' THEN
       length(json_extract_string(ref,'$.member'))>0
       AND try_cast(json_extract(ref,'$.offset') AS BIGINT)>=0
       AND try_cast(json_extract(ref,'$.length') AS BIGINT)>0
       ELSE json_extract_string(ref,'$.kind') IN ('file','loose','path') END, false) AS reference_valid,
   coalesce(regexp_full_match(pixel_sha256,'[0-9a-f]{64}') AND original_width>0 AND original_height>0
     AND coalesce(json_extract_string(errors,'$.error_stage'),'')<>'decode', false) AS decode_valid,
   coalesce(isfinite(cf384_prob) AND cf384_prob BETWEEN 0 AND 1
     AND isfinite(unifd_prob) AND unifd_prob BETWEEN 0 AND 1, false) AS scores_valid
 FROM parsed
), flags AS (
 SELECT *, coalesce(decode_valid AND sscd_byte_offset>=0 AND sscd_byte_offset%2048=0
     AND sscd_byte_offset+2048<=sscd_file_size_bytes
     AND json_extract(errors,'$.models.sscd') IS NULL, false) AS vector_valid,
   coalesce(NOT i2t_dedup_only AND (status IN ('rejected_known_ai_source','rejected_ai_edit_nonreal_parent')
      OR origin='ai_generated' OR (origin='ai_edited' AND edit_parent IS NOT NULL
                                  AND edit_parent NOT IN ('natural','unknown','mixed'))), false) AS explicit_negative,
   coalesce(origin IN ('programmatic','rendered') OR policy_exception='ai_edit_from_natural_source',false) AS protected_evidence,
   (status='pending_inference_or_integrity_error' OR NOT reference_valid
      OR coalesce(json_type(prov)<>'OBJECT',true) OR coalesce(json_type(errors)<>'OBJECT',true)
      OR (pixel_sha256 IS NOT NULL AND NOT regexp_full_match(pixel_sha256,'[0-9a-f]{64}'))
      OR (cf384_prob IS NOT NULL AND (NOT isfinite(cf384_prob) OR cf384_prob NOT BETWEEN 0 AND 1))
      OR (unifd_prob IS NOT NULL AND (NOT isfinite(unifd_prob) OR unifd_prob NOT BETWEEN 0 AND 1))
      OR status IS NULL OR status NOT IN ('rejected_known_ai_source','rejected_ai_edit_nonreal_parent',
          'rejected_too_small','protected_source','pending_inference_or_integrity_error',
          'pending_edit_lineage','ai_suspect_pending_api','no_detector_alert','i2t_dedup_only')
      OR (status='i2t_dedup_only' AND NOT i2t_dedup_only)) AS basic_invalid
 FROM fields
), checked AS (
 SELECT *, (basic_invalid
     OR (sscd_byte_offset IS NOT NULL AND NOT vector_valid)
     OR (status IN ('protected_source','ai_suspect_pending_api','no_detector_alert','pending_edit_lineage','i2t_dedup_only') AND NOT vector_valid)
     OR (status IN ('ai_suspect_pending_api','no_detector_alert') AND NOT scores_valid)
     OR (status='protected_source' AND NOT protected_evidence)) AS invalid_observation
 FROM flags
)
SELECT *, (status='protected_source' AND protected_evidence AND vector_valid AND NOT invalid_observation) AS valid_protected,
 (status='no_detector_alert' AND scores_valid AND vector_valid AND NOT invalid_observation) AS valid_noalert,
 (status='ai_suspect_pending_api') AS any_suspect
FROM checked
"""


ASSETS_SQL = r"""
WITH grouped AS (
 SELECT asset_id, count(*)::BIGINT AS occurrence_count, count(DISTINCT source)::BIGINT AS source_count,
   to_json(histogram(status))::VARCHAR AS counts_by_status,
   count(DISTINCT pixel_sha256) FILTER(WHERE regexp_full_match(pixel_sha256,'[0-9a-f]{64}')) AS pixel_hash_count,
   min(pixel_sha256) FILTER(WHERE regexp_full_match(pixel_sha256,'[0-9a-f]{64}')) AS representative_pixel_hash,
   max(cf384_prob) FILTER(WHERE isfinite(cf384_prob) AND cf384_prob BETWEEN 0 AND 1) AS cf384_prob_max,
   max(unifd_prob) FILTER(WHERE isfinite(unifd_prob) AND unifd_prob BETWEEN 0 AND 1) AS unifd_prob_max,
   bool_or(explicit_negative) AS explicit_negative, bool_or(protected_evidence) AS protected_evidence,
   bool_or(valid_protected) AS valid_protected, bool_or(any_suspect) AS any_suspect,
   bool_or(valid_noalert) AS valid_noalert, bool_or(invalid_observation) AS has_invalid_observation,
   bool_or(i2t_dedup_only AND invalid_observation) AS has_invalid_i2t_observation,
   bool_or(status='rejected_too_small') AS any_too_small, bool_and(status='rejected_too_small') AS all_too_small,
   bool_or(vector_valid) AS any_vector,
   bool_or(vector_valid AND NOT i2t_dedup_only) AS any_filtered_usage_vector,
   bool_or(i2t_dedup_only) AS has_i2t_usage,
   bool_and(i2t_dedup_only) AS i2t_only,
   count(DISTINCT original_width::VARCHAR || 'x' || original_height::VARCHAR)
      FILTER(WHERE original_width>0 AND original_height>0) AS dimension_count
 FROM observations GROUP BY asset_id
), ranked AS (
 SELECT *, row_number() OVER(PARTITION BY asset_id ORDER BY
    (vector_valid AND reference_valid) DESC, invalid_observation ASC,
    task_id, sample_id, image_index, original_ref_json, provenance_json) AS rank
 FROM observations
), combined AS (
 SELECT g.* EXCLUDE(representative_pixel_hash),
   CASE WHEN pixel_hash_count=1 THEN representative_pixel_hash ELSE NULL END AS pixel_sha256,
   pixel_hash_count>1 AS pixel_sha256_conflict,
   (has_invalid_observation OR pixel_hash_count>1 OR dimension_count>1 OR (any_too_small AND any_filtered_usage_vector)) AS integrity_pending,
   CASE WHEN has_i2t_usage THEN (has_invalid_i2t_observation OR pixel_hash_count>1 OR dimension_count>1)
        ELSE NULL END AS i2t_integrity_pending,
   r.task_id,r.source,r.sample_id,r.image_index,r.attempt_token,r.sscd_file,r.sscd_byte_offset,
   r.sscd_file_sha256,r.sscd_file_size_bytes,r.model_identity_sha256,
   r.original_ref_json,r.provenance_json,r.original_width,r.original_height,
   r.vector_valid AND r.reference_valid AS representative_vector_valid
 FROM grouped g JOIN ranked r USING(asset_id) WHERE rank=1
)
SELECT * EXCLUDE(integrity_pending,representative_vector_valid),
 CASE WHEN has_i2t_usage AND integrity_pending THEN 'pending_integrity'
      WHEN explicit_negative THEN 'rejected_source'
      WHEN integrity_pending THEN 'pending_integrity'
      WHEN i2t_only THEN 'i2t_dedup_only'
      WHEN all_too_small THEN 'rejected_too_small'
      WHEN valid_protected THEN 'protected_source'
      WHEN any_suspect THEN 'ai_suspect_pending_api'
      WHEN valid_noalert THEN 'no_detector_alert'
      ELSE 'pending' END AS global_status,
 list_filter([CASE WHEN has_invalid_observation THEN 'invalid_observation' END,
              CASE WHEN pixel_hash_count>1 THEN 'pixel_sha256_conflict' END,
              CASE WHEN dimension_count>1 THEN 'dimension_conflict' END,
              CASE WHEN any_too_small AND any_filtered_usage_vector THEN 'size_status_conflict' END], x -> x IS NOT NULL) AS exception_reasons,
 (representative_vector_valid AND (NOT integrity_pending OR
     (has_i2t_usage AND NOT i2t_integrity_pending)) AND (NOT explicit_negative OR has_i2t_usage)) AS sscd_usable,
 512::INTEGER AS sscd_dimension, 'float32'::VARCHAR AS sscd_dtype,
 true AS sscd_normalized
FROM combined ORDER BY asset_id
"""


def aggregate_bucket(connection, output, identity_hash, bucket, paths, empty_template):
    final = output / 'buckets' / bucket
    receipt_path = final / 'COMPLETE.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get('schema') == BUCKET_SCHEMA and receipt.get('identity_sha256') == identity_hash
                and receipt.get('bucket') == bucket, 'Bucket checkpoint identity mismatch')
        validate_artifacts(output, receipt)
        return receipt
    require(not final.exists(), 'Incomplete bucket destination; refusing unsafe overwrite')
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + bucket + '-', dir=final.parent))
    try:
        if paths:
            connection.execute('CREATE OR REPLACE TEMP VIEW bucket_input AS SELECT * FROM read_parquet(' +
                '[' + ','.join(quoted(path) for path in paths) + '], hive_partitioning=false)')
        else:
            connection.execute('CREATE OR REPLACE TEMP VIEW bucket_input AS SELECT * FROM read_parquet(' +
                               quoted(empty_template) + ', hive_partitioning=false) LIMIT 0')
        # Materialize only this bucket; DuckDB spills under the configured memory cap.
        connection.execute('CREATE OR REPLACE TEMP TABLE observations AS ' + OBSERVATIONS_SQL)
        connection.execute('CREATE OR REPLACE TEMP TABLE assets AS ' + ASSETS_SQL)
        connection.execute('SET preserve_insertion_order = true')
        for filename, predicate in (('assets.parquet', 'true'),
                                   ('api_review.parquet', "global_status='ai_suspect_pending_api'"),
                                   ('exceptions.parquet', 'len(exception_reasons)>0')):
            connection.execute('COPY (SELECT * FROM assets WHERE ' + predicate + ' ORDER BY asset_id) TO ' +
                quoted(temporary / filename) + ' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 32768)')
        occurrences, assets, api, errors, usable = connection.execute("""SELECT coalesce(sum(occurrence_count),0),count(*),
            count(*) FILTER(WHERE global_status='ai_suspect_pending_api'),
            count(*) FILTER(WHERE len(exception_reasons)>0), count(*) FILTER(WHERE sscd_usable) FROM assets""").fetchone()
        statuses = dict(connection.execute('SELECT global_status,count(*) FROM assets GROUP BY global_status').fetchall())
        artifacts = []
        for path in sorted(temporary.glob('*.parquet')):
            item = artifact(path, temporary); item['path'] = str((final / item['path']).relative_to(output)); artifacts.append(item)
        receipt = {'schema': BUCKET_SCHEMA, 'identity_sha256': identity_hash, 'bucket': bucket,
                   'counts': {'image_occurrences': occurrences, 'assets': assets, 'api_review_assets': api,
                              'exception_assets': errors, 'sscd_usable_assets': usable, 'global_status': statuses},
                   'observation_partitions': [str(Path(path).relative_to(output)) for path in paths], 'artifacts': artifacts}
        atomic_json(temporary / 'COMPLETE.json', receipt)
        os.replace(temporary, final)
        return receipt
    finally:
        connection.execute('SET preserve_insertion_order = false')
        connection.execute('DROP TABLE IF EXISTS assets')
        connection.execute('DROP TABLE IF EXISTS observations')
        if temporary.exists():
            shutil.rmtree(temporary)


def run(run_root, output_root, *, threads=16, memory_limit='64GB', batch_parts=128,
        selected_task_ids=None, max_new_chunks=None, max_new_buckets=None):
    """Optional max_new_* stop at durable boundaries, primarily for resumability tests."""
    import duckdb
    import pyarrow
    require(type(threads) is int and 1 <= threads <= 16, 'threads must be in 1..16')
    require(type(batch_parts) is int and 1 <= batch_parts <= 512, 'batch_parts must be in 1..512')
    require(isinstance(memory_limit, str) and re.fullmatch(r'[1-9][0-9]*(MB|GB)', memory_limit), 'Explicit MB/GB memory limit required')
    for limit in (max_new_chunks, max_new_buckets):
        require(limit is None or type(limit) is int and limit >= 0, 'Work bounds must be nonnegative integers')
    inputs = load_inputs(run_root, selected_task_ids)
    output = Path(output_root).resolve()
    require(not output.is_relative_to(inputs['root'] / 'artifacts') and output != inputs['root']
            and not output.is_relative_to(inputs['root'] / 'states'), 'Unsafe output location')
    identity = {'schema': SCHEMA, 'run_root': str(inputs['root']), 'output_root': str(output),
                'input_index_sha256': inputs['index_sha256'], 'input_complete_sha256': inputs['complete_sha256'],
                'code_sha256': file_sha(__file__), 'duckdb_version': duckdb.__version__,
                'pyarrow_version': pyarrow.__version__,
                'configuration': {'threads': threads, 'memory_limit': memory_limit, 'batch_parts': batch_parts,
                                  'buckets': 256, 'selected_task_ids': inputs['selected_task_ids']}, 'scope': inputs['scope']}
    # JSON normalization gives resumed tuples/lists identical identity representations.
    identity = json.loads(canonical(identity)); identity_hash = digest(identity)
    with locked(output):
        bind_identity(output, identity)
        if (output / 'COMPLETE.json').exists():
            result = json.loads((output / 'COMPLETE.json').read_text())
            require(result['identity_sha256'] == identity_hash, 'Final checkpoint identity mismatch')
            for item in result['partition_receipts']:
                path = output / item['path']
                require(file_sha(path) == item['sha256'], 'Final partition receipt changed')
                validate_artifacts(output, json.loads(path.read_text()))
            for item in result['buckets']:
                path = output / item['receipt_path']
                require(file_sha(path) == item['receipt_sha256'], 'Final bucket receipt changed')
                validate_artifacts(output, json.loads(path.read_text()))
            return result
        connection = connect(output, threads, memory_limit)
        try:
            partitions, new_chunks = [], 0
            for number, start in enumerate(range(0, len(inputs['parts']), batch_parts)):
                is_new = not (output / 'partitions' / f'batch-{number:06d}' / 'COMPLETE.json').exists()
                if is_new and max_new_chunks is not None and new_chunks >= max_new_chunks:
                    return {'complete': False, 'phase': 'partitioning', 'identity_sha256': identity_hash}
                receipt = stage_chunk(connection, output, identity_hash, number, inputs['parts'][start:start+batch_parts], threads)
                partitions.append(receipt); new_chunks += is_new
                atomic_json(output / 'PROGRESS.json', {'identity_sha256': identity_hash, 'phase': 'partitioning',
                            'completed_chunks': len(partitions), 'total_parts': len(inputs['parts'])})
            total_refs = sum(p['image_occurrences'] for p in partitions)
            if inputs['scope'] == 'full_run':
                require(total_refs == inputs['index']['counts']['image_occurrences'], 'Full image-reference coverage mismatch')
            by_bucket = {bucket: [] for bucket in BUCKETS}
            for partition in partitions:
                for item in partition['artifacts']:
                    path = output / item['path']; bucket = path.parent.name.removeprefix('bucket=')
                    require(bucket in by_bucket, 'Unexpected image bucket')
                    by_bucket[bucket].append(path)
            all_paths = [path for paths in by_bucket.values() for path in paths]
            # A text-only selected scope is valid; construct a zero-row template
            # from the first real input schema plus its bound metadata.
            if not all_paths:
                import pyarrow as pa
                import pyarrow.parquet as pq
                require(bool(inputs['parts']), 'No parts selected')
                first = inputs['parts'][0]
                schema = pq.read_schema(first['artifacts']['images.parquet']['absolute_path'])
                for name in ('task_id','source','attempt_token','sscd_file','sscd_file_sha256','model_identity_sha256'):
                    schema = schema.append(pa.field(name, pa.string()))
                schema = schema.append(pa.field('sscd_file_size_bytes', pa.int64()))
                template = output / 'EMPTY_TEMPLATE.parquet'
                if not template.exists():
                    pq.write_table(pa.Table.from_batches([], schema=schema), template)
            else:
                template = all_paths[0]
            buckets, new_buckets = [], 0
            for bucket, paths in by_bucket.items():
                is_new = not (output / 'buckets' / bucket / 'COMPLETE.json').exists()
                if is_new and max_new_buckets is not None and new_buckets >= max_new_buckets:
                    return {'complete': False, 'phase': 'aggregating', 'identity_sha256': identity_hash}
                receipt = aggregate_bucket(connection, output, identity_hash, bucket, paths, template)
                buckets.append(receipt); new_buckets += is_new
                atomic_json(output / 'PROGRESS.json', {'identity_sha256': identity_hash, 'phase': 'aggregating',
                                                     'completed_buckets': len(buckets), 'total_buckets': 256})
            totals = {key: sum(b['counts'][key] for b in buckets) for key in
                      ('image_occurrences','assets','api_review_assets','exception_assets','sscd_usable_assets')}
            require(totals['image_occurrences'] == total_refs, 'Aggregation lost image observations')
            statuses = {}
            for bucket in buckets:
                for key, value in bucket['counts']['global_status'].items():
                    statuses[key] = statuses.get(key, 0) + value
            totals['global_status'] = statuses
            result = {'schema': SCHEMA, 'complete': True, 'scope': inputs['scope'], 'identity_sha256': identity_hash,
                      'input_index_sha256': inputs['index_sha256'], 'counts': totals,
                      'buckets': [{'bucket': b['bucket'], 'receipt_path': f'buckets/{b["bucket"]}/COMPLETE.json',
                                   'receipt_sha256': file_sha(output / 'buckets' / b['bucket'] / 'COMPLETE.json')} for b in buckets],
                      'partition_receipts': [{'path': f'partitions/batch-{p["number"]:06d}/COMPLETE.json',
                          'sha256': file_sha(output / 'partitions' / f'batch-{p["number"]:06d}' / 'COMPLETE.json')} for p in partitions],
                      'finished_at': datetime.now(timezone.utc).isoformat(), 'original_records_deleted': 0,
                      'global_sscd_search_complete': False, 'api_calls': 0, 'training_release': False,
                      'vector_integrity': 'reference_and_size_validated_vector_consumer_must_verify_full_sha256'}
            atomic_json(output / 'COMPLETE.json', result)
            atomic_json(output / 'PROGRESS.json', {'identity_sha256': identity_hash, 'phase': 'complete', 'counts': totals})
            return result
        finally:
            connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--memory-limit', default='64GB')
    parser.add_argument('--batch-parts', type=int, default=128)
    parser.add_argument('--task-id', action='append', help='Explicit bounded validation; cannot publish full_run')
    args = parser.parse_args(argv)
    print(canonical(run(args.run_root, args.output_root, threads=args.threads,
                        memory_limit=args.memory_limit, batch_parts=args.batch_parts,
                        selected_task_ids=args.task_id)))


if __name__ == '__main__':
    main()
