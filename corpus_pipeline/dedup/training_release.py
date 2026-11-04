"""Resumable DCVLM training-candidate references, never a quality certification.

Image identity sharing does not delete supervision. Only equal ordered confirmed
image identities AND equal role/turn/order-aware text hashes and native task
contracts may remove a redundant record. Original images and rows are untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import unicodedata

from corpus_pipeline.postprocess.reconcile import (
    atomic_json, artifact, bind_identity, canonical, digest, file_sha, load_inputs,
    locked, quoted, require, stamp, validate_artifacts,
)

SCHEMA = 'corpus.training_candidate_index.v1'
POLICY = 'confirmed_identity_and_identical_ordered_supervision_v1'
TASK_POLICY = 'corpus.task_policy.v1'


def bound_files(manifest, *, identity=False):
    path = Path(manifest).resolve(strict=True)
    value = json.loads(path.read_text())
    require(value.get('complete') is True, 'Input manifest is incomplete')
    if identity:
        require(value.get('canonical_equivalence_only') is True,
                'Asset map must certify canonical equivalence, not SSCD similarity')
    files = value.get('files')
    require(isinstance(files, list) and files, 'Bound manifest requires nonempty files')
    checked = []
    for row in files:
        target = Path(row['path'])
        if not target.is_absolute(): target = path.parent / target
        target = target.resolve(strict=True)
        size = row.get('size_bytes', row.get('bytes'))
        require(target.suffix == '.parquet' and type(size) is int, 'Expected bound Parquet input')
        require(target.stat().st_size == size and file_sha(target) == row['sha256'],
                'Bound Parquet input changed: ' + str(target))
        checked.append({'path': str(target), 'sha256': row['sha256'], 'size_bytes': size,
                        'stamp': stamp(target)})
    require(len({x['path'] for x in checked}) == len(checked), 'Repeated manifest input')
    return {'manifest': str(path), 'manifest_sha256': file_sha(path), 'files': checked}


def connection(output, threads, memory_limit):
    import duckdb
    require(type(threads) is int and 1 <= threads <= 128, 'threads outside 1..128')
    con = duckdb.connect()
    con.execute(f'SET threads={threads}')
    con.execute('SET memory_limit=' + quoted(memory_limit))
    con.execute('SET temp_directory=' + quoted(output / 'spill'))
    con.execute('SET preserve_insertion_order=false')
    con.execute('SET partitioned_write_max_open_files=32')
    return con


def parquet_sql(paths, **options):
    flags = ''.join(',' + key + '=' + str(value).lower() for key, value in options.items())
    return 'read_parquet([' + ','.join(quoted(x) for x in paths) + ']' + flags + ')'


def verify_stamps(bindings):
    for item in bindings:
        require(stamp(item['path']) == item['stamp'], 'Input changed while processing: ' + item['path'])


def text_fingerprint(segments):
    """Exact native dcvlm_curate.common.content_fingerprints text contract."""
    values=[]
    for segment in segments:
        value=segment['text'] if segment['kind']=='text' else segment['image_index']
        if segment['kind']=='text':
            value=unicodedata.normalize('NFC',value.replace('\r\n','\n').replace('\r','\n'))
        values.append([segment['kind'],segment.get('role'),segment.get('turn'),value])
    return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def bind_native_metadata(con, output, path, before):
    """Freeze full metadata bytes and verify a deterministic row against frontpass.

    The original frontpass binding is size/mtime. This adds a SHA for subsequent
    restart checks plus one full-row and text-hash audit per metadata file; it
    deliberately does not claim every original row was rehashed.
    """
    import pyarrow.parquet as pq
    marker=output/'native_bindings'/(hashlib.sha256(str(path).encode()).hexdigest()+'.json')
    if marker.exists():
        receipt=json.loads(marker.read_text())
        require(receipt['stamp']==before,'Original metadata changed since prior batch')
        return
    sample=con.execute('''SELECT row_index,source_record_sha256,text_hash FROM prepared_batch
        WHERE original_parquet_path=? ORDER BY row_index LIMIT 1''',[str(path)]).fetchone()
    require(sample is not None,'Missing native verification sample')
    index,expected_source,expected_text=sample
    source=pq.ParquetFile(path);offset=0;raw=None
    for group in range(source.num_row_groups):
        count=source.metadata.row_group(group).num_rows
        if index<offset+count:
            for batch in source.iter_batches(batch_size=128,row_groups=[group]):
                if index<offset+batch.num_rows:
                    raw=batch.slice(index-offset,1).to_pylist()[0];break
                offset+=batch.num_rows
            break
        offset+=count
    require(raw is not None and digest(raw)==expected_source,'Native full-record sample differs from committed frontpass')
    require(text_fingerprint(raw['segments'])==raw['text_hash']==expected_text,'Native supervision fingerprint mismatch')
    sha=file_sha(path)
    require(stamp(path)==before,'Native metadata changed during verification')
    atomic_json(marker,{'path':str(path),'stamp':before,'sha256':sha,'sample_row_index':index,
        'sample_source_record_sha256':expected_source,'sample_text_hash':expected_text,
        'all_source_records_rehashed':False})


def prepare_batch(con, output, number, parts):
    """Commit one bounded batch with complete source-row identity checks."""
    import pyarrow as pa
    directory = output / 'prepared' / f'{number:06d}'
    receipt = directory / 'COMPLETE.json'
    if receipt.exists():
        value = json.loads(receipt.read_text()); validate_artifacts(output, value)
        return value
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'records.parquet'; temporary = directory / 'records.partial.parquet'
    meta, observed = [], []
    for part in parts:
        task = part['task']; binding = part['artifacts'].get('records.parquet')
        require(binding is not None, 'Committed task lacks records.parquet')
        path = Path(binding['absolute_path']); before = stamp(path)
        require(before['size'] == binding['size_bytes'] and file_sha(path) == binding['sha256'],
                'Committed record index changed')
        require(stamp(path) == before, 'Record index changed during hashing')
        native = Path(task['parquet_path']).resolve(strict=True); native_stat = stamp(native)
        require(task.get('parquet_fingerprint') == {'size': native_stat['size'], 'mtime_ns': native_stat['mtime_ns']},
                'Original metadata fingerprint changed')
        observed += [{'path': str(path), 'stamp': before}, {'path': str(native), 'stamp': native_stat}]
        meta.append({'index_path': str(path), 'task_id': task['task_id'], 'source': task['source'],
                     'source_shard': task['source_shard'], 'parquet_path': str(native),
                     'allowed_sources': part.get('mixed_sources', [task['source']]),
                     'row_start': task['row_start'], 'row_end': task['row_end']})
    con.register('batch_tasks', pa.Table.from_pylist(meta))
    source = parquet_sql([x['index_path'] for x in meta], filename=True, union_by_name=True)
    native = parquet_sql(sorted({x['parquet_path'] for x in meta}), filename=True, file_row_number=True)
    con.execute(f'''CREATE OR REPLACE TEMP TABLE prepared_batch AS
      SELECT t.task_id || ':' || CAST(r.row_index AS VARCHAR) AS record_id,
        t.task_id,t.source,t.source_shard,t.allowed_sources,t.parquet_path AS original_parquet_path,r.* EXCLUDE(filename),
        n.sample_id AS native_sample_id,n.source AS native_source,n.source_shard AS native_source_shard,
        n.text_hash,n.format AS native_format,n.image_count AS native_image_count
      FROM {source} r JOIN batch_tasks t ON r.filename=t.index_path
      LEFT JOIN (SELECT filename,file_row_number,sample_id,source,source_shard,text_hash,format,image_count FROM {native}) n
        ON n.filename=t.parquet_path AND n.file_row_number=r.row_index''')
    # Older frontpass snapshots have none of these fields. Nulls preserve their
    # original policy; only the explicit, internally consistent v1 contract opts in.
    columns = {row[0] for row in con.execute('DESCRIBE prepared_batch').fetchall()}
    for name, kind in [('processing_policy', 'VARCHAR'), ('task_route', 'VARCHAR'),
                       ('image_loss_allowed', 'BOOLEAN')]:
        if name not in columns:
            con.execute(f'ALTER TABLE prepared_batch ADD COLUMN {name} {kind}')
    invalid_task_policy = con.execute('''SELECT count(*) FROM prepared_batch WHERE
        (processing_policy=? AND task_route='i2t' AND image_loss_allowed IS DISTINCT FROM false)
        OR (status='i2t_dedup_only' AND NOT coalesce(
            processing_policy=? AND task_route='i2t' AND image_loss_allowed=false,false))''',
        [TASK_POLICY, TASK_POLICY]).fetchone()[0]
    require(invalid_task_policy == 0, 'Inconsistent I2T task policy or image loss flag')
    expected = sum(x['row_end']-x['row_start'] for x in meta)
    count, unique, invalid = con.execute('''SELECT count(*),count(DISTINCT record_id),count(*) FILTER (WHERE
        native_sample_id IS NULL OR native_sample_id<>sample_id OR text_hash IS NULL OR
        NOT regexp_full_match(text_hash,'[0-9a-f]{64}') OR native_format IS NULL OR category IS NULL OR
        native_image_count IS NULL OR image_ids IS NULL OR image_statuses IS NULL OR
        native_image_count<>len(image_ids) OR len(image_ids)<>len(image_statuses) OR
        status IS NULL OR status NOT IN ('rejected','pending','frontpass_pass_pending_global_dedup','i2t_dedup_only') OR
        source_record_sha256 IS NULL OR NOT regexp_full_match(source_record_sha256,'[0-9a-f]{64}'))
        FROM prepared_batch''').fetchone()
    require(count == unique == expected and invalid == 0, 'Record/native metadata coverage or structure mismatch')
    invalid_sources=con.execute('''SELECT count(*) FROM prepared_batch WHERE
        native_source IS NULL OR native_source_shard IS DISTINCT FROM source_shard
        OR NOT coalesce(list_contains(allowed_sources,native_source),false)
        OR (source<>'__mixed__' AND native_source IS DISTINCT FROM source)''').fetchone()[0]
    require(invalid_sources==0,'Native source identity differs from bound task/catalog')
    invalid_images=con.execute("""SELECT count(*) FROM prepared_batch b,UNNEST(image_ids) u(asset_id)
        WHERE asset_id IS NULL OR NOT regexp_full_match(asset_id,'[0-9a-f]{64}')""").fetchone()[0]
    require(invalid_images==0,'Invalid original image identity')
    invalid_ranges = con.execute('''SELECT count(*) FROM prepared_batch r JOIN batch_tasks t USING(task_id)
        WHERE r.row_index<t.row_start OR r.row_index>=t.row_end''').fetchone()[0]
    require(invalid_ranges == 0, 'Native row locator outside committed range')
    for native in sorted({x['parquet_path'] for x in meta}):
        bind_native_metadata(con,output,Path(native),stamp(native))
    con.execute(f'''COPY (SELECT * EXCLUDE(native_sample_id,native_source,native_source_shard,source,source_shard,allowed_sources),
        native_source AS source,native_source_shard AS source_shard FROM prepared_batch) TO {quoted(temporary)}
        (FORMAT PARQUET,COMPRESSION ZSTD)''')
    verify_stamps(observed); os.replace(temporary, target)
    value = {'schema_version': SCHEMA, 'complete': True, 'records': count,
             'task_ids': [x['task_id'] for x in meta], 'input_stamps': observed,
             'artifacts': [artifact(target, output)]}
    atomic_json(receipt, value); con.unregister('batch_tasks')
    con.execute('DROP TABLE prepared_batch')
    return value


def normalize_and_partition(con, output, batches, asset_map, record_quality=()):
    """One global join; DuckDB spills when the full asset map exceeds RAM."""
    marker = output / 'partitioned' / 'COMPLETE.json'
    if marker.exists():
        value = json.loads(marker.read_text()); validate_artifacts(output, value); return value
    temporary = output / 'partitioned.partial'
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    bases = [str(output / item['artifacts'][0]['path']) for item in batches]
    con.execute('CREATE OR REPLACE TEMP VIEW base AS SELECT *, coalesce('
                'processing_policy=' + quoted(TASK_POLICY) + " AND task_route='i2t' AND image_loss_allowed=false,false)"
                ' AS i2t_dedup_only FROM ' + parquet_sql(bases, union_by_name=True))
    if record_quality:
        con.execute("CREATE OR REPLACE TEMP VIEW record_quality AS SELECT source_record_sha256,min(reason) AS reason, bool_or(reason='corrupt_image') AS has_corrupt_image FROM "+
                    parquet_sql([x['path'] for x in record_quality])+' GROUP BY source_record_sha256')
        require(con.execute("SELECT count(*) FROM record_quality WHERE source_record_sha256 IS NULL OR NOT regexp_full_match(source_record_sha256,'[0-9a-f]{64}') OR reason IS NULL").fetchone()[0]==0,
                'Record quality rule must bind exact original record SHA')
    else:
        con.execute('CREATE OR REPLACE TEMP VIEW record_quality AS SELECT NULL::VARCHAR AS source_record_sha256,NULL::VARCHAR AS reason,false AS has_corrupt_image WHERE false')
    con.execute('CREATE OR REPLACE TEMP VIEW identity_map AS SELECT * FROM ' + parquet_sql([x['path'] for x in asset_map['files']]))
    total, unique, invalid = con.execute('''SELECT count(*),count(DISTINCT asset_id), count(*) FILTER(WHERE
       asset_id IS NULL OR canonical_asset_id IS NULL OR NOT regexp_full_match(asset_id,'[0-9a-f]{64}') OR
       NOT regexp_full_match(canonical_asset_id,'[0-9a-f]{64}')) FROM identity_map''').fetchone()
    require(total == unique and invalid == 0, 'Ambiguous or invalid canonical asset map')
    invalid_screen = con.execute("""SELECT count(*) FROM identity_map WHERE screening_action IS NULL OR
       screening_action NOT IN ('exclude','pending','retain_annotated') OR screening_reason IS NULL OR global_status IS NULL""").fetchone()[0]
    require(invalid_screen == 0, 'Invalid screening decisions in asset identity map')
    columns = {row[0] for row in con.execute('DESCRIBE identity_map').fetchall()}
    component = 'm.sscd_component_id' if 'sscd_component_id' in columns else 'NULL::UBIGINT'
    integrity = "m.global_status='pending_integrity'"
    if 'i2t_integrity_pending' in columns:
        integrity = ("CASE WHEN b.i2t_dedup_only THEN coalesce(m.i2t_integrity_pending,"
                     "m.global_status='pending_integrity') ELSE m.global_status='pending_integrity' END")
    # Crucially screen by the original SHA, never by another image's canonical SHA.
    con.execute(f'''CREATE OR REPLACE TEMP TABLE image_context AS
      SELECT b.record_id,
        list(COALESCE(m.canonical_asset_id,u.asset_id) ORDER BY u.position) AS canonical_image_ids,
        bool_or(m.screening_action='exclude') AS screen_excluded,
        bool_or(m.screening_action='pending') AS screen_pending,
        bool_or(m.asset_id IS NULL OR ({integrity}) OR (b.i2t_dedup_only AND
          m.screening_action='exclude' AND m.screening_reason IN ('corrupt_image','confirmed_corrupt_image'))) AS integrity_quarantine,
        list(struct_pack(image_index:=u.position-1,asset_id:=u.asset_id,
             action:=m.screening_action,reason:=m.screening_reason,evidence_sha256:={quoted(asset_map['manifest_sha256'])})
             ORDER BY u.position) FILTER(WHERE m.asset_id IS NOT NULL AND (m.screening_action<>'retain_annotated' OR m.quality_reviewed)) AS screening_evidence,
        list(DISTINCT {component}) FILTER(WHERE {component} IS NOT NULL) AS sscd_component_ids
      FROM base b CROSS JOIN UNNEST(b.image_ids) WITH ORDINALITY u(asset_id,position)
      LEFT JOIN identity_map m ON m.asset_id=u.asset_id
      GROUP BY b.record_id''')
    con.execute(f'''CREATE OR REPLACE TEMP VIEW enriched AS SELECT b.*,
        COALESCE(i.canonical_image_ids,[]::VARCHAR[]) AS canonical_image_ids,
        (NOT b.i2t_dedup_only AND COALESCE(i.screen_excluded,false)) AS confirmed_screen_excluded,
        NOT b.i2t_dedup_only AND (COALESCE(i.screen_pending,false) OR (b.status='pending' AND
          (len(b.reason_codes)=0 OR len(list_filter(b.reason_codes,r->r NOT IN
             ('ai_suspect_pending_api','pending_edit_lineage','pending_inference_or_integrity_error')))>0)))
          AS unresolved_screening,
        i.screening_evidence,i.sscd_component_ids,
        CASE WHEN NOT b.i2t_dedup_only THEN q.reason ELSE NULL END AS confirmed_record_quality_reason,
        q.reason AS observed_record_quality_reason,
        COALESCE(i.integrity_quarantine,false) OR list_contains(b.image_statuses,'pending_inference_or_integrity_error')
          OR list_contains(b.reason_codes,'unresolved_original_image_binding')
          OR (b.i2t_dedup_only AND (b.status<>'i2t_dedup_only' OR coalesce(q.has_corrupt_image,false))) AS integrity_quarantine,
        (NOT b.i2t_dedup_only AND (b.status='rejected' OR COALESCE(i.screen_excluded,false) OR q.reason IS NOT NULL)) AS preexcluded,
        CASE WHEN b.processing_policy={quoted(TASK_POLICY)} THEN
          sha256(to_json(struct_pack(category:=b.category,native_format:=b.native_format,
              text_hash:=b.text_hash,ordered_images:=COALESCE(i.canonical_image_ids,[]::VARCHAR[]),
              task_route:=b.task_route,image_loss_allowed:=b.image_loss_allowed)))
        ELSE sha256(to_json(struct_pack(category:=b.category,native_format:=b.native_format,
              text_hash:=b.text_hash,ordered_images:=COALESCE(i.canonical_image_ids,[]::VARCHAR[])))) END AS record_signature
        FROM base b LEFT JOIN image_context i USING(record_id)
        LEFT JOIN record_quality q ON q.source_record_sha256=b.source_record_sha256''')
    con.execute(f'''COPY (SELECT *,substr(record_signature,1,2) AS bucket FROM enriched)
        TO {quoted(temporary / 'data')} (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(bucket))''')
    con.execute('DROP TABLE image_context')
    verify_stamps(asset_map['files'])
    for batch in batches: verify_stamps(batch['input_stamps'])
    destination = output / 'partitioned'
    if destination.exists(): shutil.rmtree(destination)
    temporary.rename(destination)
    files = sorted(destination.glob('data/bucket=*/*.parquet'))
    observed = con.execute('SELECT count(*) FROM ' + parquet_sql(files, hive_partitioning=True)).fetchone()[0]
    expected = sum(x['records'] for x in batches)
    require(observed == expected, 'Canonicalization lost or multiplied records')
    value = {'schema_version': SCHEMA, 'complete': True, 'records': observed, 'asset_map_records': total,
             'artifacts': [artifact(path, output) for path in files]}
    atomic_json(marker, value); return value


def finalize_bucket(con, output, bucket, paths):
    directory = output / 'buckets' / bucket; marker = directory / 'COMPLETE.json'
    if marker.exists():
        value = json.loads(marker.read_text()); validate_artifacts(output, value); return value
    directory.mkdir(parents=True, exist_ok=True)
    source = parquet_sql(paths, hive_partitioning=True)
    # Prefer an eligible record over an excluded one, then a non-pending record.
    con.execute(f'''CREATE OR REPLACE TEMP TABLE ranked AS SELECT *,
      first_value(record_id) OVER (PARTITION BY record_signature
        ORDER BY preexcluded,integrity_quarantine,unresolved_screening,sample_id,record_id) AS representative_record_id
      FROM {source}''')
    con.execute('''CREATE OR REPLACE TEMP VIEW final_bucket AS SELECT *,
      CASE WHEN list_contains(reason_codes,'unresolved_original_image_binding') THEN 'quarantine'
           WHEN preexcluded THEN 'exclude'
           WHEN integrity_quarantine THEN 'quarantine'
           WHEN record_id<>representative_record_id THEN 'exclude' ELSE 'keep' END AS decision,
      CASE WHEN list_contains(reason_codes,'unresolved_original_image_binding') THEN 'unresolved_original_image_binding'
           WHEN status='rejected' AND NOT i2t_dedup_only THEN 'existing_frontpass_rejection'
           WHEN confirmed_screen_excluded THEN 'confirmed_image_screening_exclusion'
           WHEN confirmed_record_quality_reason IS NOT NULL THEN 'confirmed_record_quality_exclusion'
           WHEN integrity_quarantine THEN 'unresolved_image_integrity_or_missing_asset'
           WHEN record_id<>representative_record_id THEN 'identical_ordered_supervision_and_confirmed_images'
           WHEN unresolved_screening THEN 'retain_candidate_with_unresolved_screening'
           ELSE 'retain_candidate' END AS decision_reason,
      CASE WHEN NOT preexcluded AND NOT integrity_quarantine AND record_id<>representative_record_id
           THEN representative_record_id ELSE NULL END AS duplicate_of_record_id,
      false AS source_certification,false AS published_training
      FROM ranked''')
    invalid_links=con.execute('''SELECT count(*) FROM final_bucket f LEFT JOIN final_bucket r
        ON f.duplicate_of_record_id=r.record_id WHERE f.duplicate_of_record_id IS NOT NULL
        AND (r.record_id IS NULL OR r.decision<>'keep' OR r.record_signature<>f.record_signature)''').fetchone()[0]
    require(invalid_links==0,'Duplicate record points to a missing/ineligible/different representative')
    total = con.execute('SELECT count(*) FROM final_bucket').fetchone()[0]
    counts = dict(con.execute('SELECT decision,count(*) FROM final_bucket GROUP BY decision').fetchall())
    reasons = dict(con.execute('SELECT decision_reason,count(*) FROM final_bucket GROUP BY decision_reason').fetchall())
    categories = [dict(zip(('source','category','decision','records'),row)) for row in con.execute(
        'SELECT source,category,decision,count(*) FROM final_bucket GROUP BY source,category,decision').fetchall()]
    for decision in ('keep','exclude','quarantine'):
        path = directory / (decision + '.parquet'); temp = directory / (decision + '.partial.parquet')
        con.execute(f'''COPY (SELECT * EXCLUDE(bucket) FROM final_bucket WHERE decision={quoted(decision)})
            TO {quoted(temp)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        os.replace(temp,path)
    value = {'schema_version': SCHEMA, 'complete': True, 'bucket': bucket, 'records': total,
             'counts': counts, 'reasons': reasons, 'source_category_counts': categories,
             'artifacts': [artifact(directory/(x+'.parquet'),output) for x in ('keep','exclude','quarantine')]}
    atomic_json(marker,value); con.execute('DROP VIEW final_bucket'); con.execute('DROP TABLE ranked')
    return value


def build(run_root, asset_manifest, output, *, threads=16, memory_limit='16GB',
          batch_parts=128, task_ids=None, stop_after_prepare_batches=None, record_quality_files=()):
    require(type(batch_parts) is int and batch_parts>0, 'batch_parts must be positive')
    output = Path(output).resolve()
    inputs = load_inputs(run_root, task_ids)
    ranges={}
    for part in inputs['parts']:
        task=part['task'];ranges.setdefault(task['parquet_path'],[]).append((task['row_start'],task['row_end']))
    for spans in ranges.values():
        spans.sort()
        require(all(a[1]<=b[0] for a,b in zip(spans,spans[1:])),'Overlapping committed native row ranges')
    assets = bound_files(asset_manifest,identity=True)
    quality=[]
    for entry in record_quality_files:
        path=Path(entry['path']).resolve(strict=True)
        require(file_sha(path)==entry['sha256'] and path.stat().st_size==entry['size_bytes'],
                'Record quality evidence changed')
        quality.append({**entry,'path':str(path),'stamp':stamp(path)})
    identity = {'schema_version': SCHEMA,'policy': POLICY,'code_sha256': file_sha(__file__),
                'frontpass_index_sha256': inputs['index_sha256'], 'frontpass_complete_sha256': inputs['complete_sha256'],
                'scope': inputs['scope'],'selected_task_ids':None if inputs['selected_task_ids'] is None else list(inputs['selected_task_ids']),
                'asset_map':assets,'batch_parts':batch_parts,'record_quality_files':quality}
    with locked(output):
        bind_identity(output, identity)
        if (output/'COMPLETE.json').exists():
            complete = json.loads((output/'COMPLETE.json').read_text())
            require(complete['manifest_sha256']==file_sha(output/'manifest.json'), 'Published index manifest changed')
            manifest = json.loads((output/'manifest.json').read_text())
            for part in manifest['buckets']: validate_artifacts(output,part)
            validate_artifacts(output,{'artifacts':manifest['native_metadata_bindings']})
            for binding in manifest['native_metadata_bindings']:
                verify_stamps([json.loads((output/binding['path']).read_text())])
            return complete
        con=connection(output,threads,memory_limit)
        try:
            batches=[]
            for start in range(0,len(inputs['parts']),batch_parts):
                batches.append(prepare_batch(con,output,start//batch_parts,inputs['parts'][start:start+batch_parts]))
                atomic_json(output/'PROGRESS.json',{'phase':'prepare','completed_parts':min(start+batch_parts,len(inputs['parts'])),
                    'total_parts':len(inputs['parts']),'records':sum(x['records'] for x in batches),'updated_at':time.time()})
                if stop_after_prepare_batches and len(batches)>=stop_after_prepare_batches:
                    return {'complete':False,'phase':'prepare','prepared_batches':len(batches)}
            partitioned=normalize_and_partition(con,output,batches,assets,quality)
            buckets=[]
            for directory in sorted((output/'partitioned/data').glob('bucket=*')):
                buckets.append(finalize_bucket(con,output,directory.name.split('=')[1],sorted(directory.glob('*.parquet'))))
                atomic_json(output/'PROGRESS.json',{'phase':'record_dedup','completed_buckets':len(buckets),
                    'records':sum(x['records'] for x in buckets),'updated_at':time.time()})
            total=sum(x['records'] for x in buckets)
            counts=Counter();reasons=Counter();categories=Counter()
            for item in buckets:
                counts.update(item['counts']);reasons.update(item['reasons'])
                for row in item['source_category_counts']:categories[(row['source'],row['category'],row['decision'])]+=row['records']
            require(total==partitioned['records']==sum(x['records'] for x in batches), 'Final record coverage mismatch')
            require(sum(counts.values())==total,'Keep/exclude coverage mismatch')
            manifest={'schema_version':SCHEMA,'complete':True,'scope':inputs['scope'],'policy':POLICY,
                'records':total,'counts':dict(counts),'reasons':dict(reasons),'buckets':buckets,
                'native_metadata_bindings':[artifact(path,output) for path in sorted((output/'native_bindings').glob('*.json'))],
                'source_category_counts':[dict(source=k[0],category=k[1],decision=k[2],records=v) for k,v in sorted(categories.items())],
                'input_identity_sha256':file_sha(output/'IDENTITY.json'),'published_training':False,
                'source_certification':False,'source_records_deleted':0,'native_task_types_preserved':True,
                'same_image_different_supervision_preserved':True,'sscd_candidates_used_for_deletion':False,
                'original_image_bytes_read':False,'meaning':'Auditable training candidate keep/exclude references, not complete quality certification',
                'limitations':['Unresolved screening remains annotated in retained candidates.',
                    'Image identity is shared; original image slots are never rewritten to the representative.',
                    'Duplicate text identity uses existing NFC/newline-normalized, role/turn/order-aware native text_hash.',
                    'Native metadata files are fully SHA-bound; one deterministic full row per file is rechecked against frontpass. The reader verifies every consumed full row.',
                    'Blur review coverage is limited to the bound upstream reviewed population.']}
            verify_stamps(assets['files'])
            verify_stamps(quality)
            for batch in batches:verify_stamps(batch['input_stamps'])
            atomic_json(output/'manifest.json',manifest)
            complete={'schema_version':SCHEMA,'complete':True,'scope':inputs['scope'],'records':total,
                'counts':dict(counts),'manifest_sha256':file_sha(output/'manifest.json'),
                'published_training':False,'source_records_deleted':0,'completed_at':time.time()}
            atomic_json(output/'COMPLETE.json',complete);return complete
        finally:con.close()


def iter_candidate_references(output, decision='keep', batch_size=8192):
    """Yield verified row locators; reading source data remains the loader's job."""
    import pyarrow.parquet as pq
    require(decision in {'keep','exclude','quarantine'},'Unsupported candidate view')
    output=Path(output).resolve(strict=True)
    complete=json.loads((output/'COMPLETE.json').read_text())
    require(complete.get('complete') is True and complete['manifest_sha256']==file_sha(output/'manifest.json'),
            'Candidate manifest is incomplete or changed')
    manifest=json.loads((output/'manifest.json').read_text())
    for bucket in manifest['buckets']:
        selected=[item for item in bucket['artifacts'] if Path(item['path']).name==decision+'.parquet']
        require(len(selected)==1,'Missing/ambiguous candidate view')
        validate_artifacts(output,{'artifacts':selected})
        for batch in pq.ParquetFile(output/selected[0]['path']).iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True);parser.add_argument('--asset-manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=16);parser.add_argument('--memory-limit',default='16GB')
    parser.add_argument('--batch-parts',type=int,default=128);parser.add_argument('--task-id',action='append')
    args=parser.parse_args()
    print(canonical(build(args.run_root,args.asset_manifest,args.output,
        threads=args.threads,memory_limit=args.memory_limit,batch_parts=args.batch_parts,task_ids=args.task_id)))

if __name__=='__main__':main()
