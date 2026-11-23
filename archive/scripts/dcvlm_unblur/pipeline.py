"""Finish the recovery queue, apply verified replacements, and publish current indices."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from collections import Counter

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from prepare import NATIVE, PUBLIC, ROOT, RUN, save
from rewrite import DATASET_ID, file_sha, rewrite

REPO = Path(__file__).resolve().parents[2]
OLD_ID = 'mlfoundations/dcvlm-baseline-200b@251523f5a04d7a852223454182cea63bc9f95063'


def invalidate_old_bindings():
    note = dict(reason='DataComp URL restoration changes image bytes and TAR offsets',
                repair_run=str(RUN), original_dataset_id=OLD_ID, new_dataset_id=DATASET_ID,
                old_indices_and_image_features_require_rebuilding=True, time=time.time())
    save(ROOT / 'REPAIR_IN_PROGRESS.json', note)
    save(RUN / 'INVALIDATED_INPUTS.json', note)
    bases = [PUBLIC / 'preprocessing' / name for name in (
        'dcvlm_baseline_full_20261028_v3', 'dcvlm_baseline_200b_20261028_v2',
        'genund_six_with_dcvlm_20261103_v1')]
    for base in bases:
        roots = base / 'source_roots.json'
        if roots.exists():
            current = json.loads(roots.read_text())
            if OLD_ID in current:
                backup = roots.with_name('source_roots.pre_unblur_20261104.json')
                if not backup.exists():
                    shutil.copy2(roots, backup)
                del current[OLD_ID]
                save(roots, current)
        save(base / 'BASELINE_REPAIR_INVALIDATION.json', note)
        paths = [base / 'native/COMPLETE.json'] if base.name.startswith('dcvlm') else [
            base / 'baseline_reuse/COMPLETE.json', base / 'baseline_reuse/CURRENT.json']
        for path in paths:
            if path.exists():
                backup = path.with_name(path.stem + '.pre_unblur_20261104.json')
                if not backup.exists():
                    os.replace(path, backup)
    download = PUBLIC / 'datasets/_downloads/dcvlm_baseline_200b_20261027'
    save(download / 'SUPERSEDED_BY_REPAIR.json', note)


def prepare_apply():
    db = duckdb.connect(config={'threads': 4, 'memory_limit': '8GB'})
    recovery = RUN / 'recovery.parquet'
    schema = pa.schema([('upstream_shard', pa.string()), ('sha256', pa.string()),
                        ('status', pa.string()), ('recovery_json', pa.large_string())])
    with pq.ParquetWriter(str(recovery) + '.tmp', schema, compression='zstd') as writer:
        for metadata_path in sorted((RUN / 'upstream').glob('*/metadata.json')):
            path = metadata_path.with_name('results.jsonl')
            latest = {}
            for row in json.loads(metadata_path.read_text()):
                latest[row['baseline_sha256']] = dict(baseline_sha256=row['baseline_sha256'],
                    original_key=row.get('original_key'), status=row['status'] if row['status'] != 'identified' else 'not_processed_before_deadline')
            for line in path.read_text().splitlines() if path.exists() else []:
                try:
                    item = json.loads(line)
                    latest[item['baseline_sha256']] = item
                except json.JSONDecodeError:
                    pass
            records = list(latest.values())
            writer.write_table(pa.Table.from_pylist([
                dict(upstream_shard=path.parent.name, sha256=r['baseline_sha256'],
                     status=r['status'], recovery_json=json.dumps(dict(r, upstream_evidence=str(path.parent / 'metadata.json'))))
                for r in records], schema=schema))
    os.replace(str(recovery) + '.tmp', recovery)
    db.execute(f'''COPY (SELECT t.*, coalesce(r.status, 'not_processed_before_deadline') AS status,
        r.recovery_json FROM read_parquet(?) t
        LEFT JOIN read_parquet(?) r ON t.upstream_shard=r.upstream_shard AND t.sha256=r.sha256)
        TO '{RUN / 'sample_audit.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)''',
        [str(RUN / 'targets.parquet'), str(recovery)])
    counts = dict(db.execute('SELECT status,count(*) FROM read_parquet(?) GROUP BY status',
                             [str(RUN / 'sample_audit.parquet')]).fetchall())
    assert None not in counts and sum(counts.values()) == 4968838
    if not (RUN / 'DEADLINE.json').exists():
        assert not any(counts.get(k) for k in ('not_processed_before_deadline', 'metadata_range_failed', 'not_matched_in_old_index'))
    # An unanswered failure-policy question never authorizes dropping training records.
    coverage = dict(record_counts=counts, unresolved_records=sum(v for k, v in counts.items()
                    if k not in ('restored', 'no_blur_annotation', 'already_original_bytes')),
                    failure_policy='retain_unrestorable_pending_user_decision', samples_deleted=0)
    save(RUN / 'COVERAGE.json', coverage)
    grouped = db.execute('''SELECT source_shard, list(struct_pack(source_key := source_key,
        recovery_json := recovery_json)) FROM read_parquet(?) WHERE status='restored'
        GROUP BY source_shard''', [str(RUN / 'sample_audit.parquet')]).fetchall()
    by_shard = {shard: {r['source_key']: json.loads(r['recovery_json']) for r in rows} for shard, rows in grouped}
    directory = RUN / 'apply_jobs'
    directory.mkdir(exist_ok=True)
    jobs = []
    for entry in json.loads((RUN / 'original_catalog.json').read_text()):
        name = Path(entry['source_shard']).stem + '.json'
        save(directory / name, dict(entry=entry, replacements=by_shard.get(entry['source_shard'], {})))
        jobs.append(name)
    invalidate_old_bindings()
    save(RUN / 'APPLY_READY.json', dict(jobs=jobs, coverage=coverage, created=time.time()))


def apply_worker(worker):
    jobs = json.loads((RUN / 'APPLY_READY.json').read_text())['jobs']
    workers = len(json.loads((RUN / 'workers.json').read_text()))
    offset = len(jobs) * worker // workers
    jobs = jobs[offset:] + jobs[:offset]
    while True:
        pending = [name for name in jobs if not (RUN / 'rewrite' / Path(name).stem / 'APPLIED.json').exists()]
        if not pending:
            break
        for name in pending:
            if shutil.disk_usage(ROOT).free < 512 * 1024 ** 3:
                raise RuntimeError('Less than 512 GiB free; retaining staged work')
            job = json.loads((RUN / 'apply_jobs' / name).read_text())
            result = rewrite(job['entry'], job['replacements'])
            if result:
                print('APPLIED', result['source_shard'], result['changed_images'], flush=True)
        time.sleep(2)
    save(RUN / f'apply_worker_{worker:02d}.json', dict(complete=True, ended=time.time()))


def finalize():
    catalog = json.loads((RUN / 'original_catalog.json').read_text())
    output = RUN / 'native'
    output.mkdir(exist_ok=True)
    manifest, receipts, restored = [], [], 0
    dataset_ids = set()
    source_counts, format_counts, adapter_statuses, image_slots = Counter(), Counter(), Counter(), 0
    for entry in catalog:
        receipt = json.loads((RUN / 'rewrite' / Path(entry['source_shard']).stem / 'APPLIED.json').read_text())
        index = Path(receipt['index'])
        assert file_sha(index) == receipt['index_sha256']
        assert (ROOT / entry['source_shard']).stat().st_size == receipt['new_size']
        target = output / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(index, target)
        assert pq.read_metadata(target).num_rows == entry['rows'] == receipt['records']
        entry.update(bytes=target.stat().st_size, sha256=receipt['index_sha256'])
        manifest.append(dict(path=entry['source_shard'], size=receipt['new_size'], sha256=receipt['new_sha256']))
        marker = target.with_name('COMPLETE.json')
        if not marker.exists():
            os.link(RUN / 'rewrite' / Path(entry['source_shard']).stem / 'APPLIED.json', marker)
        receipts.append(dict(path=str(marker.relative_to(output)), sha256=file_sha(marker)))
        source_counts.update(receipt['source_counts'])
        format_counts.update(receipt['format_counts'])
        adapter_statuses.update(receipt['adapter_statuses'])
        image_slots += receipt['image_slots']
        restored += receipt['changed_images']
        dataset_ids.add(receipt['dataset_id'])
    coverage = json.loads((RUN / 'COVERAGE.json').read_text())
    assert restored == coverage['record_counts'].get('restored', 0)
    save(output / 'catalog.json', catalog)
    save(output / 'source_roots.json', {identity: str(ROOT) for identity in dataset_ids})
    save(RUN / 'repaired_manifest.json', dict(dataset_id=DATASET_ID, files=manifest))
    stats = dict(samples=sum(x['rows'] for x in catalog), sources=dict(source_counts),
        formats=dict(format_counts), image_slots=image_slots, input_units_planned=len(catalog),
        input_units_processed=len(catalog), adapter_statuses=dict(adapter_statuses))
    plan_sha = file_sha(RUN / 'APPLY_READY.json')
    save(output / 'manifest.json', dict(schema_version='corpus.native.v2', phase='complete',
        scope='full_declared_inputs', catalog_contract='mixed_source_tar.v1', collections=['dcvlm'],
        enumeration_scope='user_authorized_datacomp_url_restoration', statistics=stats,
        portable_datasets=sorted(dataset_ids), allowed_image_roots=[], plan_sha256=plan_sha,
        unit_receipts=receipts, source_files_modified=True, original_row_fields_modified=True,
        image_payloads_copied=False, quality_features_recomputed=False,
        source_inventory_sha256=file_sha(RUN / 'repaired_manifest.json')))
    save(output / 'COMPLETE.json', dict(schema_version='corpus.native.v2', complete=True,
        validation_passed=True, validation_scope='changed_archives_all_member_bytes_and_offsets; unchanged_archives_original_verified_index',
        scope='full_declared_inputs', catalog_contract='mixed_source_tar.v1', statistics=stats,
        catalog_sha256=file_sha(output / 'catalog.json'), manifest_sha256=file_sha(output / 'manifest.json'),
        plan_sha256=plan_sha))
    # Use the project's actual native-index reader as the final manifest compatibility check.
    import sys
    sys.path.insert(0, str(REPO))
    from corpus_pipeline.production.records import snapshot
    snapshot(output)
    result = dict(stage='applied', dataset_id=DATASET_ID, dataset_root=str(ROOT),
                  native_index=str(output), restored_records=restored, **coverage,
                  all_datacomp_blur_removed=coverage['unresolved_records'] == 0, ended=time.time(),
                  deadline=json.loads((RUN / 'DEADLINE.json').read_text()) if (RUN / 'DEADLINE.json').exists() else None)
    save(ROOT / 'DATACOMP_REPAIR.json', result)
    (ROOT / 'REPAIR_IN_PROGRESS.json').unlink(missing_ok=True)
    save(RUN / 'APPLIED.json', result)
    save(RUN / 'STATUS.json', dict(updated=time.time(), phase='complete' if result['all_datacomp_blur_removed'] else 'partial_applied',
        upstream_complete=len(list((RUN / 'upstream').glob('*/DONE.json'))),
        upstream_total=json.loads((RUN / 'PREPARED.json').read_text())['upstream_shards'] if (RUN / 'PREPARED.json').exists() else 0,
        rewritten_shards=sum(not json.loads((RUN / 'rewrite' / Path(e['source_shard']).stem / 'APPLIED.json').read_text()).get('unchanged', False) for e in catalog),
        validated_shards=len(catalog), total_shards=len(catalog), restored_records=restored,
        unresolved_records=coverage['unresolved_records'], all_datacomp_blur_removed=result['all_datacomp_blur_removed']))
    return result


def main():
    with (RUN / 'controller.lock').open('w') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool = None
        last_launch = 0
        expected = json.loads((RUN / 'PREPARED.json').read_text())['upstream_shards']
        while not (RUN / 'APPLIED.json').exists():
            done = sum((p.parent / 'METADATA_RECHECKED.json').exists() for p in (RUN / 'upstream').glob('*/DONE.json'))
            applied = len(list((RUN / 'rewrite').glob('*/APPLIED.json')))
            heartbeats = [json.loads(p.read_text()) for p in RUN.glob('worker_*.heartbeat.json')]
            live = [h for h in heartbeats if time.time()-h['updated'] < 90]
            save(RUN / 'STATUS.json', dict(updated=time.time(), upstream_complete=done,
                upstream_total=expected, rewritten_shards=applied, total_shards=10752,
                live_workers=len(live), upstream_bytes_per_second=sum(h.get('upstream_bytes_per_second', 0) for h in live),
                phase='applying' if (RUN / 'APPLY_READY.json').exists() else 'recovering'))
            deadline = json.loads((RUN / 'DEADLINE.json').read_text()) if (RUN / 'DEADLINE.json').exists() else None
            cutoff = deadline and time.time() >= deadline['deadline'] - 1800
            if cutoff and not (RUN / 'STOP_RECOVERY.json').exists():
                save(RUN / 'STOP_RECOVERY.json', dict(time=time.time(), reason='Reserve final 30 minutes for verified in-place application'))
            if (done == expected or cutoff) and not (RUN / 'APPLY_READY.json').exists():
                if done == expected and not (RUN / 'LOCAL_DONORS_APPLIED.json').exists():
                    save(RUN / 'STATUS.json', dict(updated=time.time(), phase='checking_local_originals',
                        upstream_complete=done, upstream_total=expected, rewritten_shards=0, total_shards=10752))
                    from local_donors import main as find_local_originals, apply as apply_local_originals, retry_deferred_originals
                    if not (RUN / 'FINAL_URL_RETRY.json').exists():
                        retry_deferred_originals()
                    apply_local_originals(find_local_originals())
                prepare_apply()
            if applied == 10752:
                finalize()
                break
            if time.time() - last_launch > 120 and (pool is None or pool.poll() is not None):
                with (RUN / 'pool.log').open('ab') as log:
                    pool = subprocess.Popen(['/root/.local/share/uv/tools/inspire-skill/bin/python',
                        str(REPO / 'scripts/dcvlm_unblur/start_pool.py')], stdout=log, stderr=subprocess.STDOUT)
                last_launch = time.time()
            time.sleep(30)
        # Release only the named CPU instances created for this repair run.
        env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
        released = []
        for worker in json.loads((RUN / 'workers.json').read_text()):
            success = False
            for attempt in range(3):
                result = subprocess.run(['inspire', '--account', 'wjx-ascend', '--json', 'notebook', 'stop',
                    worker['name'], '--workspace', '昇腾卡公共空间'], env=env, capture_output=True, text=True, timeout=120)
                try:
                    success = json.loads(result.stdout).get('success') is True
                except json.JSONDecodeError:
                    pass
                if success:
                    break
                time.sleep(5)
            released.append(dict(name=worker['name'], success=success, returncode=result.returncode))
            save(RUN / 'CPU_RELEASE.json', released)
        save(RUN / ('CPU_RELEASED.json' if all(r['success'] for r in released) else 'CPU_RELEASE_INCOMPLETE.json'),
             dict(updated=time.time(), stopped=sum(r['success'] for r in released), total=len(released)))


if __name__ == '__main__':
    main()
