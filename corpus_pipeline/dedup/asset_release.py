"""Build exact visual identities without using SSCD scores as proof of equality.

Source images stay read-only. The frontpass pixel hash includes oriented image
dimensions and every RGBA byte. Only unambiguous, integral observations share an
identity; screening decisions remain attached to the original encoded SHA.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import duckdb

from corpus_pipeline.postprocess.reconcile import atomic_json, file_sha, quoted


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": file_sha(path), "size_bytes": path.stat().st_size}


def verify(item):
    path = Path(item['path'])
    before = path.stat()
    require(before.st_size == item['size_bytes'], 'Input size changed: ' + str(path))
    require(file_sha(path) == item['sha256'], 'Input hash changed: ' + str(path))
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Input changed during verification')
    return item


def identity_key_sql():
    # Prefixes distinguish an encoded SHA fallback from a decoded pixel SHA.
    return """CASE WHEN pixel_hash_count=1 AND NOT pixel_sha256_conflict
        AND dimension_count=1 AND NOT has_invalid_observation
        AND regexp_full_match(pixel_sha256, '[0-9a-f]{64}')
        AND original_width>0 AND original_height>0
        THEN 'pixel:' || pixel_sha256 ELSE 'bytes:' || asset_id END"""


def build(reconciliation, screening_manifest, output, threads=8, memory='64GB'):
    reconciliation, output = Path(reconciliation).absolute(), Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.LOCK').open('a+') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_complete = reconciliation / 'COMPLETE.json'
        source = json.loads(source_complete.read_text())
        require(source.get('complete') is True, 'Reconciliation incomplete')
        sources = []
        for bucket in source['buckets']:
            receipt = reconciliation / bucket['receipt_path']
            require(file_sha(receipt) == bucket['receipt_sha256'], 'Reconciliation receipt mismatch')
            value = json.loads(receipt.read_text())
            row = next(a for a in value['artifacts'] if Path(a['path']).name == 'assets.parquet')
            sources.append({**row, 'path': str(reconciliation / row['path'])})
        screen_path = Path(screening_manifest).absolute()
        screen = json.loads(screen_path.read_text())
        require(screen.get('complete') is True, 'Screening decision binding incomplete')
        screen_files = [{**i, 'path': str((screen_path.parent / i['path']).absolute())}
                        for i in screen['files']]
        inputs = sources + screen_files
        identity = {'schema': 'corpus.exact_asset_release.v1',
                    'reconciliation': artifact(source_complete),
                    'screening_manifest': artifact(screen_path),
                    'code_sha256': file_sha(__file__),
                    'identity_policy': 'unambiguous_oriented_RGBA_pixel_sha256_else_encoded_SHA256',
                    'screening_propagates_across_identity': False,
                    'inputs': inputs}
        identity_path = output / 'IDENTITY.json'
        if identity_path.exists():
            require(json.loads(identity_path.read_text()) == identity, 'Output identity changed')
        else:
            atomic_json(identity_path, identity)
        identity_sha = file_sha(identity_path)
        complete_path = output / 'COMPLETE.json'
        if complete_path.exists():
            done = json.loads(complete_path.read_text())
            require(done['identity_sha256'] == identity_sha, 'Completion identity changed')
            for item in done['files']:
                verify(item)
            return done
        def progress(phase, **values):
            atomic_json(output / 'PROGRESS.json', {'phase': phase, 'updated_at': time.time(), **values})
        progress('verifying_inputs', files=len(inputs))
        with ThreadPoolExecutor(max_workers=min(threads, 8)) as pool:
            list(pool.map(verify, inputs))
        db = duckdb.connect(str(output / 'work.duckdb'))
        db.execute(f'SET threads={threads}')
        db.execute('SET memory_limit=' + quoted(memory))
        db.execute('SET temp_directory=' + quoted(output / 'spill'))
        db.execute('SET preserve_insertion_order=false')
        db.execute('SET partitioned_write_max_open_files=32')
        files = '[' + ','.join(quoted(i['path']) for i in sources) + ']'
        screens = '[' + ','.join(quoted(i['path']) for i in screen_files) + ']'
        source_columns = {row[0] for row in db.execute('DESCRIBE SELECT * FROM read_parquet(' + files + ')').fetchall()}
        i2t_integrity = 'a.i2t_integrity_pending' if 'i2t_integrity_pending' in source_columns else 'NULL::BOOLEAN'
        db.execute('CREATE OR REPLACE TEMP VIEW screen AS SELECT * FROM read_parquet(' + screens + ')')
        require(db.execute('SELECT count(*)=count(DISTINCT asset_id) FROM screen').fetchone()[0],
                'Duplicate screening asset identity')
        progress('materializing_assets')
        db.execute('DROP TABLE IF EXISTS base')
        db.execute("""CREATE TABLE base AS SELECT a.asset_id,
            """ + identity_key_sql() + """ AS visual_identity,
            a.global_status, CASE WHEN s.screen_action='exclude' AND
                s.screen_reason IN ('corrupt_image','confirmed_corrupt_image') THEN true
                ELSE """ + i2t_integrity + """ END AS i2t_integrity_pending, a.original_width, a.original_height,
            a.occurrence_count, a.source, a.sscd_usable,
            CASE WHEN a.global_status IN ('rejected_source','rejected_too_small') THEN 'exclude'
                 WHEN s.screen_action='exclude' THEN 'exclude'
                 WHEN a.global_status='pending_integrity' THEN 'pending'
                 WHEN s.screen_action IS NOT NULL THEN s.screen_action
                 WHEN a.global_status IN ('no_detector_alert','protected_source') THEN 'retain_annotated'
                 ELSE 'pending' END AS screening_action,
            CASE WHEN a.global_status IN ('rejected_source','rejected_too_small','pending_integrity')
                 THEN a.global_status ELSE coalesce(s.screen_reason,a.global_status) END AS screening_reason,
            coalesce(s.quality_reviewed,false) AS quality_reviewed,
            coalesce(s.strict_blur_reviewed,false) AS strict_blur_reviewed
            FROM read_parquet(""" + files + """, hive_partitioning=false) a
            LEFT JOIN screen s USING(asset_id)""")
        counts = db.execute('SELECT count(*),count(DISTINCT asset_id) FROM base').fetchone()
        expected = source.get('counts', {}).get('assets')
        require(counts[0] == counts[1] and (expected is None or counts[0] == expected), 'Asset coverage mismatch')
        # Prefer an existing usable asset, while never forwarding its decision to
        # another asset. This mapping does not replace source image bytes.
        progress('selecting_exact_representatives', assets=counts[0])
        db.execute('DROP TABLE IF EXISTS canonical')
        db.execute("""CREATE TABLE canonical AS SELECT visual_identity,
            arg_min(asset_id, struct_pack(screen_rank := CASE screening_action
                WHEN 'retain_annotated' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
                usable_rank := CASE WHEN sscd_usable THEN 0 ELSE 1 END, sha := asset_id))
                AS canonical_asset_id,
            count(*)::BIGINT AS exact_identity_size
            FROM base GROUP BY visual_identity""")
        dest = output / 'assets'
        if dest.exists():
            # Only deterministic uncommitted output of this bound run is removed.
            import shutil
            shutil.rmtree(dest)
        progress('writing_asset_map', assets=counts[0])
        db.execute("""COPY (SELECT b.asset_id,c.canonical_asset_id,
            CASE WHEN b.visual_identity LIKE 'pixel:%' THEN 'exact_oriented_rgba'
                 ELSE 'exact_encoded_bytes' END AS canonical_relation,
            c.exact_identity_size,b.global_status,b.i2t_integrity_pending,b.screening_action,b.screening_reason,
            b.quality_reviewed,b.strict_blur_reviewed,b.original_width,b.original_height,b.source,b.occurrence_count,
            b.sscd_usable,substr(b.asset_id,1,2) AS bucket
            FROM base b JOIN canonical c USING(visual_identity)) TO """ + quoted(dest) +
            ' (FORMAT PARQUET, PARTITION_BY(bucket), COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)')
        files_out = [artifact(p) for p in sorted(dest.glob('bucket=*/*.parquet'))]
        actions = dict(db.execute('SELECT screening_action,count(*) FROM base GROUP BY 1').fetchall())
        groups, redundant = db.execute('SELECT count(*),sum(exact_identity_size-1) FROM canonical').fetchone()
        stats = {'assets': counts[0], 'exact_visual_identities': groups,
                 'encoded_assets_sharing_exact_pixels': int(redundant or 0), 'screening_actions': actions,
                 'quality_reviewed_assets': db.execute('SELECT count(*) FROM base WHERE quality_reviewed').fetchone()[0]}
        result = {'complete': True, 'schema': identity['schema'], 'identity_sha256': identity_sha,
                  'identity_policy': identity['identity_policy'], 'files': files_out, 'counts': stats,
                  'near_similarity_is_equivalence': False, 'canonical_equivalence_only': True,
                  'source_records_deleted': False,
                  'screening_propagates_across_identity': False}
        atomic_json(complete_path, result)
        progress('complete', **stats)
        db.close()
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reconciliation',required=True);p.add_argument('--screening-manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--memory',default='64GB')
    args=p.parse_args()
    result=build(args.reconciliation,args.screening_manifest,args.output,args.threads,args.memory)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))


if __name__=='__main__':
    main()
