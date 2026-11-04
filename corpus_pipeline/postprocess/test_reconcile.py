from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from . import reconcile as r


IMAGE_SCHEMA = pa.schema([
    ('asset_id', pa.string()), ('sample_id', pa.string()), ('image_index', pa.int32()),
    ('original_ref_json', pa.string()), ('provenance_json', pa.string()), ('status', pa.string()),
    ('original_width', pa.int32()), ('original_height', pa.int32()), ('pixel_sha256', pa.string()),
    ('cf384_prob', pa.float32()), ('unifd_prob', pa.float32()), ('sscd_byte_offset', pa.int64()),
    ('resolution_potential', pa.bool_()), ('error_json', pa.string())])


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def observation(key, *, status='no_detector_alert', origin='unknown', pixel=None, **changes):
    asset = sha(key)
    protected = status == 'protected_source'
    small = status == 'rejected_too_small'
    negative = status == 'rejected_known_ai_source'
    result = {'asset_id': asset, 'sample_id': 'sample-' + key, 'image_index': 0,
              'original_ref_json': json.dumps({'kind': 'tar', 'path': '/fixtures/original.tar',
                  'member': key+'.jpg', 'offset': 512, 'length': 100, 'sha256': asset, 'exists': True,
                  'pixel_sha256': sha('old-convention-'+key)}),
              'provenance_json': json.dumps({'origin': origin, 'evidence': [{'selected': True}],
                                           'policy_exception': None, 'edit_source_origin': None}),
              'status': status, 'original_width': 64 if small else 640,
              'original_height': 64 if small else 480,
              'pixel_sha256': None if small or negative else (pixel or sha('pixels-'+key)),
              'cf384_prob': None if protected or small or negative else .2,
              'unifd_prob': None if protected or small or negative else .3,
              'sscd_byte_offset': None if small or negative else 0,
              'resolution_potential': False, 'error_json': '{}'}
    result.update(changes)
    return result


def mixed_frontpass_fixture(base):
    """Real TAR/native/queue/frontpass transport with synthetic model evidence."""
    from corpus_pipeline.unified.test_ingest import balanced
    from corpus_pipeline.unified.ingest import build
    from corpus_pipeline.unified.test_cache import FakeWorker
    from corpus_pipeline.production.records import snapshot
    from corpus_pipeline.cluster import frontpass, queue, monitor
    spec, tar = balanced(base/'raw', [
        ('first','instruction','source-a','human:<image>Question A?<EOCL>\ngpt:Answer A.',1),
        ('second','instruction','source-b','human:<image>Question B?<EOCL>\ngpt:Answer B.',1)])
    staging=base/'staging';build([spec],staging)
    catalog=json.loads((staging/'catalog.json').read_text())
    table=pa.concat_tables([pq.ParquetFile(staging/entry['path']).read() for entry in catalog])
    native=base/'native';native.mkdir();path=native/'mixed.parquet';pq.write_table(table,path)
    manifest=json.loads((staging/'manifest.json').read_text());manifest['catalog_contract']='mixed_source_tar.v1'
    entry={'path':path.name,'bytes':path.stat().st_size,'rows':table.num_rows,'sha256':r.file_sha(path),
        'collection':'dcvlm','source':'__mixed__','source_shard':'shards/000/000000.tar',
        'contract':'mixed_source_tar.v1','sources':['source-a','source-b']}
    r.atomic_json(native/'manifest.json',manifest);r.atomic_json(native/'catalog.json',[entry])
    r.atomic_json(native/'COMPLETE.json',{'complete':True,'validation_passed':True,
        'manifest_sha256':r.file_sha(native/'manifest.json'),'catalog_sha256':r.file_sha(native/'catalog.json')})
    run=base/'frontpass';frontpass.prepare(run,dataset=native,rows_per_task=16384)
    claim=queue.claim(run,'fixture-worker');worker=FakeWorker();snap=snapshot(native)
    summary,artifacts=frontpass.process_task(run,claim,worker,snap,frontpass.load_rules(),
        SimpleNamespace(entries={}),{'cf384':1.,'unifd':1.})
    queue.complete(run,claim['task_id'],claim['token'],artifacts=artifacts,
        processed_records=summary['counts']['records'],processed_images=summary['counts']['image_occurrences'])
    monitor.finalize(run,queue.status(run))
    return run,native,tar,table.to_pylist()


class Fixture:
    def __init__(self, root):
        self.root = root; self.parts = []; self.counts = Counter(); self.categories = Counter()

    def part(self, source, rows):
        index = len(self.parts); task_id = f'task-{index:04d}'; token = f'token-{index:04d}'
        directory = self.root / 'artifacts' / task_id / token; directory.mkdir(parents=True)
        task = {'task_id': task_id, 'source': source, 'source_shard': source+'/000.tar',
                'parquet_path': '/fixtures/'+source+'.parquet', 'row_start': 0, 'row_end': max(1, len(rows))}
        rows = [dict(row) for row in rows]
        vectors = []
        for row in rows:
            if row['sscd_byte_offset'] is not None and row['sscd_byte_offset'] == 0:
                row['sscd_byte_offset'] = len(vectors) * r.VECTOR_BYTES
                vectors.append(struct.pack('<512f', 1., *([0.] * 511)))
        (directory/'sscd.f32').write_bytes(b''.join(vectors))
        pq.write_table(pa.Table.from_pylist(rows, schema=IMAGE_SCHEMA), directory/'images.parquet')
        counts = Counter({'records': task['row_end'], 'image_occurrences': len(rows)})
        counts.update({'image:'+status: count for status, count in Counter(row['status'] for row in rows).items()})
        self.counts.update(counts); self.categories['single_i2t'] += task['row_end']
        summary = {'task': task, 'token': token, 'counts': dict(counts), 'model_identity_sha256': sha('model')}
        r.atomic_json(directory/'summary.json', summary)
        artifacts = [r.artifact(path, self.root) for path in sorted(directory.iterdir())]
        self.parts.append({'task': task, 'artifacts': artifacts, 'model_identity_sha256': sha('model')})
        return task_id

    def finish(self):
        value = {'schema': 'dcvlm.offline_frontpass_release.v1', 'scope': 'offline_frontpass_only',
                 'complete': True, 'counts': dict(self.counts), 'categories': dict(self.categories),
                 'parts': self.parts, 'api_calls': 0, 'finished_at': '2026-10-23T12:41:50+00:00'}
        r.atomic_json(self.root/'frontpass_index.json', value)
        r.atomic_json(self.root/'FRONTPASS_COMPLETE.json', {k:v for k,v in value.items() if k != 'parts'} |
                      {'index_sha256': r.file_sha(self.root/'frontpass_index.json')})


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.root = self.base/'run'; self.root.mkdir()
        self.output = self.base/'out'
        self.fixture = Fixture(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def run_reconcile(self, **kwargs):
        return r.run(self.root, self.output, threads=1, memory_limit='256MB', batch_parts=1, **kwargs)

    def assets(self):
        rows = []
        for path in sorted(self.output.glob('buckets/*/assets.parquet')):
            part = pq.ParquetFile(path).read().to_pylist()
            ids = [row['asset_id'] for row in part]
            self.assertEqual(ids, sorted(set(ids)))
            self.assertTrue(all(value.startswith(path.parent.name) for value in ids))
            rows.extend(part)
        return {row['asset_id']: row for row in rows}

    def test_global_policy_preserves_evidence_and_never_repairs_bad_occurrences(self):
        first = [observation('shared-protected', status='ai_suspect_pending_api'),
                 observation('negative', status='protected_source', origin='rendered'),
                 observation('pixel-conflict'), observation('broken-reference'),
                 observation('model-error'), observation('suspect', status='ai_suspect_pending_api', cf384_prob=.999),
                 observation('ordinary'), observation('small', status='rejected_too_small'),
                 observation('other-sha', status='ai_suspect_pending_api'),
                 observation('same-pixel-different-sha', status='protected_source', origin='rendered', pixel=sha('pixels-other-sha')),
                 observation('invalid-vector', sscd_byte_offset=2**48),
                 observation('size-conflict', status='rejected_too_small'),
                 observation('missing-lineage', status='pending_edit_lineage', origin='ai_edited')]
        second = [observation('shared-protected', status='protected_source', origin='programmatic'),
                  observation('negative', status='rejected_known_ai_source', origin='ai_generated', original_ref_json='broken'),
                  observation('pixel-conflict', pixel=sha('different-pixels')),
                  observation('broken-reference', original_ref_json=json.dumps({'kind':'tar','path':'/wrong','sha256':sha('different')})),
                  observation('model-error', status='pending_inference_or_integrity_error', error_json='{"error_stage":"decode"}'),
                  observation('suspect'), observation('size-conflict')]
        self.fixture.part('source-a', first); self.fixture.part('source-b', second); self.fixture.finish()
        result = self.run_reconcile(); assets = self.assets()
        def get(key): return assets[sha(key)]
        self.assertTrue(result['complete']); self.assertEqual(result['scope'], 'full_run')
        self.assertEqual(result['counts']['image_occurrences'], len(first)+len(second))
        self.assertEqual(result['counts']['assets'], len(first))
        self.assertEqual(get('shared-protected')['global_status'], 'protected_source')
        self.assertEqual(get('shared-protected')['source_count'], 2)
        self.assertEqual(get('shared-protected')['occurrence_count'], 2)
        self.assertEqual(json.loads(get('shared-protected')['counts_by_status']), {'ai_suspect_pending_api':1,'protected_source':1})
        self.assertEqual(get('negative')['global_status'], 'rejected_source')
        self.assertTrue(get('negative')['explicit_negative']); self.assertTrue(get('negative')['has_invalid_observation'])
        self.assertFalse(get('negative')['sscd_usable'])
        for key in ('pixel-conflict','broken-reference','model-error','invalid-vector','size-conflict'):
            self.assertEqual(get(key)['global_status'], 'pending_integrity', key)
            self.assertFalse(get(key)['sscd_usable'], key)
        self.assertTrue(get('pixel-conflict')['pixel_sha256_conflict'])
        self.assertIsNone(get('pixel-conflict')['pixel_sha256'])
        self.assertEqual(get('suspect')['global_status'], 'ai_suspect_pending_api')
        self.assertAlmostEqual(get('suspect')['cf384_prob_max'], .999, places=6)
        self.assertEqual(get('ordinary')['global_status'], 'no_detector_alert')
        self.assertTrue(get('ordinary')['sscd_usable'])
        self.assertEqual(get('ordinary')['sscd_dimension'], 512)
        self.assertEqual(get('small')['global_status'], 'rejected_too_small')
        self.assertEqual(get('missing-lineage')['global_status'], 'pending')
        # A protected *different encoded SHA* with matching pixels cannot rescue this asset.
        self.assertEqual(get('other-sha')['global_status'], 'ai_suspect_pending_api')
        self.assertEqual(result['counts']['api_review_assets'], 2)
        self.assertEqual(len(result['buckets']), 256)
        self.assertTrue((self.root/'artifacts/task-0000/token-0000/images.parquet').exists())
        all_observations = sum(pq.ParquetFile(path).metadata.num_rows for path in self.output.glob('partitions/*/data/bucket=*/*.parquet'))
        self.assertEqual(all_observations, len(first)+len(second))

    def test_resume_partitions_then_buckets_is_exactly_once_and_identity_bound(self):
        self.fixture.part('a', [observation('repeat')])
        self.fixture.part('b', [observation('repeat'), observation('another')]); self.fixture.finish()
        result = self.run_reconcile(max_new_chunks=1)
        self.assertFalse(result['complete']); self.assertEqual(result['phase'], 'partitioning')
        with patch.object(r, 'verify_part', wraps=r.verify_part) as verify:
            result = self.run_reconcile(max_new_buckets=1)
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(result['phase'], 'aggregating')
        with patch.object(r, 'verify_part', side_effect=AssertionError('source should not be rescanned')):
            result = self.run_reconcile()
        self.assertEqual(result['counts']['assets'], 2)
        self.assertEqual(result['counts']['image_occurrences'], 3)
        self.assertEqual(self.run_reconcile(), result)
        with self.assertRaisesRegex(ValueError, 'different input/code/config'):
            r.run(self.root, self.output, threads=2, memory_limit='256MB', batch_parts=1)

    def test_output_corruption_and_final_index_change_are_rejected(self):
        self.fixture.part('a',[observation('a')]); self.fixture.finish()
        self.run_reconcile()
        path=self.output/'buckets'/sha('a')[:2]/'assets.parquet'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'Missing/changed output artifact'):
            self.run_reconcile()
        with (self.root/'frontpass_index.json').open('ab') as stream:stream.write(b' ')
        with self.assertRaisesRegex(ValueError,'Final index SHA mismatch'):
            self.run_reconcile()

    def test_input_hash_corruption_fails_before_any_aggregate_publish(self):
        self.fixture.part('a',[observation('a')]);self.fixture.finish()
        path=self.root/'artifacts/task-0000/token-0000/images.parquet'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'Committed images.parquet changed'):
            self.run_reconcile()
        self.assertFalse((self.output/'COMPLETE.json').exists())

    def test_bounded_selection_and_empty_image_shard(self):
        task=self.fixture.part('text',[])
        self.fixture.part('not-selected',[observation('a')]);self.fixture.finish()
        result=self.run_reconcile(selected_task_ids=[task])
        self.assertEqual(result['scope'],'bounded_validation')
        self.assertEqual(result['counts']['assets'],0)
        self.assertEqual(result['counts']['image_occurrences'],0)
        self.assertEqual(len(result['buckets']),256)
        with self.assertRaisesRegex(ValueError,'different input/code/config'):
            self.run_reconcile()

    def test_invalid_asset_identity_blocks_partition_commit(self):
        self.fixture.part('a',[observation('a',asset_id='not-a-sha')]);self.fixture.finish()
        with self.assertRaisesRegex(ValueError,'Malformed asset SHA'):
            self.run_reconcile()
        self.assertFalse((self.output/'partitions/batch-000000/COMPLETE.json').exists())

    def test_summary_wrong_token_and_namespace_are_rejected(self):
        self.fixture.part('a',[observation('a')]);self.fixture.finish()
        part=r.load_inputs(self.root)['parts'][0]
        path=Path(part['artifacts']['summary.json']['absolute_path'])
        value=json.loads(path.read_text());value['token']='other';r.atomic_json(path,value)
        with self.assertRaisesRegex(ValueError,'Committed summary changed'):
            self.run_reconcile()

    def test_mixed_tar_sources_are_restored_from_bound_per_image_provenance(self):
        run,native,tar,originals=mixed_frontpass_fixture(self.base/'mixed')
        self.root=run
        parts=r.load_inputs(run)['parts'];self.assertEqual(parts[0]['mixed_sources'],['source-a','source-b'])
        self.assertEqual(parts[0]['task']['source'],'__mixed__')
        self.run_reconcile()
        rows=list(self.assets().values());self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['source_count'],2)
        self.assertEqual(rows[0]['occurrence_count'],2)
        observations=[row for path in self.output.glob('partitions/*/data/bucket=*/*.parquet')
            for row in pq.ParquetFile(path).read().to_pylist()]
        self.assertEqual({row['source'] for row in observations},{'source-a','source-b'})

    def test_mixed_provenance_rejects_missing_invalid_or_unlisted_source(self):
        for bad in ({}, {'source':'not-in-catalog'}, {'source':7}, None):
            with self.subTest(provenance=bad):
                base=self.base/('case-'+str(len(list(self.base.glob('case-*')))))
                run,_,_,_=mixed_frontpass_fixture(base)
                index=json.loads((run/'frontpass_index.json').read_text())
                part=index['parts'][0]
                item=next(a for a in part['artifacts'] if Path(a['path']).name=='images.parquet')
                path=run/item['path'];table=pq.ParquetFile(path).read();rows=table.to_pylist()
                rows[0]['provenance_json']='not-json' if bad is None else json.dumps(bad)
                pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),path)
                item.update(r.artifact(path,run));r.atomic_json(run/'frontpass_index.json',index)
                complete=json.loads((run/'FRONTPASS_COMPLETE.json').read_text())
                complete['index_sha256']=r.file_sha(run/'frontpass_index.json');r.atomic_json(run/'FRONTPASS_COMPLETE.json',complete)
                with self.assertRaisesRegex(ValueError,'valid bound source identity'):
                    r.run(run,base/'reconciliation',threads=1,memory_limit='256MB')
                self.assertFalse((base/'reconciliation/COMPLETE.json').exists())

    def test_mixed_task_requires_the_frozen_catalog_contract(self):
        run,native,_,_=mixed_frontpass_fixture(self.base/'missing-contract')
        manifest=json.loads((native/'manifest.json').read_text());manifest.pop('catalog_contract')
        r.atomic_json(native/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'manifest validation mismatch'):
            r.load_inputs(run)


if __name__ == '__main__':
    unittest.main()
