from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .test_training_release import Fixture,sha
from . import training_release as training
from . import source_view as view
from corpus_pipeline.postprocess.reconcile import atomic_json, artifact, file_sha


class SourceViewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'input';self.source.mkdir()
        self.fixture=Fixture(self.source);a=self.fixture.asset('a');bad=self.fixture.asset('bad',action='exclude')
        broken=self.fixture.asset('broken',action='pending',status='pending_integrity')
        self.fixture.part('one',[
            {'name':'keep-one','images':[a],'text':'x'},
            {'name':'exclude','images':[bad],'text':'bad'},
            {'name':'quarantine','images':[broken],'text':'broken'}])
        self.fixture.part('two',[{'name':'keep-two','images':[a],'text':'y'}])
        self.fixture.finish();self.candidate=self.root/'candidate';self.output=self.root/'view'
        training.build(self.source,self.source/'assets.json',self.candidate,threads=1,memory_limit='256MB')
    def build(self):return view.build(self.candidate,self.output,threads=1,memory_limit='256MB')
    def test_masks_preserve_full_native_fields_and_hash_each_file_once(self):
        result=self.build();self.assertEqual(result['records'],4);self.assertEqual(result['mask_bytes'],4)
        self.assertEqual(result['counts'],{'keep':2,'exclude':1,'quarantine':1,'not_indexed':0})
        expected=[]
        for path in (self.source/'one.parquet',self.source/'two.parquet'):
            expected.append(pq.read_table(path).slice(0,1).to_pylist()[0])
        original=view.file_sha;calls=[]
        def counted(path):calls.append(str(path));return original(path)
        with patch.object(view,'file_sha',side_effect=counted):
            actual=[row for batch in view.iter_native_batches(self.output,batch_size=1) for row in batch.to_pylist()]
        self.assertEqual(actual,expected)
        for path in (self.source/'one.parquet',self.source/'two.parquet'):
            self.assertEqual(calls.count(str(path)),1)
        self.assertEqual(self.build(),result)
        shards=[]
        for worker in range(2):
            shards.append([row['sample_id'] for batch in view.iter_native_batches(self.output,epoch_seed=7,worker_id=worker,num_workers=2) for row in batch.to_pylist()])
        self.assertFalse(set(shards[0])&set(shards[1]));self.assertEqual(set(shards[0]+shards[1]),{x['sample_id'] for x in expected})
    def test_mask_tamper_and_source_tamper_fail_before_yielding(self):
        self.build();manifest=json.loads((self.output/'manifest.json').read_text());item=manifest['files'][0]
        mask=self.output/item['mask_path'];before=mask.read_bytes();mask.write_bytes(bytes([0])+before[1:])
        with self.assertRaisesRegex(ValueError,'mask byte count or SHA'):
            list(view.iter_native_batches(self.output))
        mask.write_bytes(before)
        source=Path(item['source_path']);source.touch()
        with self.assertRaisesRegex(ValueError,'source metadata changed'):
            list(view.iter_native_batches(self.output))
    def test_duplicate_locator_rejected_across_stream_batches_and_bounded_unknown_rows(self):
        path=self.source/'one.parquet';binding={'path':str(path),'sha256':view.file_sha(path),'stamp':view.stamp(path)}
        builder=view.MaskBuilder(self.output,{str(path):binding},False)
        builder.consume(str(path),np.array([0]),np.array([1],dtype=np.uint8))
        with self.assertRaisesRegex(ValueError,'Duplicate or unsorted'):
            builder.consume(str(path),np.array([0]),np.array([0],dtype=np.uint8))
        builder.finish();self.assertEqual(builder.files[0]['counts']['not_indexed'],2)
        full=view.MaskBuilder(self.root/'full',{str(path):binding},True)
        full.consume(str(path),np.array([0]),np.array([1],dtype=np.uint8))
        with self.assertRaisesRegex(ValueError,'unindexed native rows'):full.finish()

    def test_new_candidate_permissions_reach_primary_release_reader_without_rewriting_native(self):
        from corpus_pipeline.unified import release
        source=self.root/'task-input';source.mkdir();fixture=Fixture(source)
        aid=fixture.asset('task-image')
        fixture.part('mixed',[
            {'name':'new-i2t','images':[aid],'text':'New image understanding','category':'interleaved'},
            {'name':'legacy','images':[aid],'text':'Existing legacy policy'}])
        fixture.part('generation',[{'name':'new-t2i','images':[aid],'text':'Generation prompt','category':'single_t2i'}])
        originals={str(path):file_sha(path) for path in source.glob('*.parquet')}
        for part in fixture.parts:
            path=source/'artifacts'/part['task']['task_id']/'attempt/records.parquet'
            table=pq.read_table(path);rows=table.to_pylist();schema=table.schema
            for key,kind in [('processing_policy',pa.string()),('task_route',pa.string()),('image_loss_allowed',pa.bool_())]:
                if key not in schema.names:schema=schema.append(pa.field(key,kind))
            for row in rows:
                if row['sample_id']==sha('legacy'):continue
                route='i2t' if row['sample_id']==sha('new-i2t') else 't2i'
                row.update(processing_policy=view.TASK_POLICY,task_route=route,image_loss_allowed=route=='t2i')
                if route=='i2t':row['status']='i2t_dedup_only'
            # Schema inference follows every row, including a nullable legacy row.
            for row in rows:
                for key in ('processing_policy','task_route','image_loss_allowed'):row.setdefault(key,None)
            pq.write_table(pa.Table.from_pylist(rows,schema=schema),path)
            part['artifacts']=[artifact(p,source) for p in path.parent.iterdir()]
        fixture.finish();candidate=self.root/'task-candidate';output=self.root/'task-view'
        training.build(source,source/'assets.json',candidate,threads=1,memory_limit='256MB')
        result=view.build(candidate,output,threads=1,memory_limit='256MB')
        self.assertEqual(result['counts']['keep'],3)
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['task_policy_overlay']['codes'],view.POLICY_CODES)
        self.assertTrue(all('task_policy_path' in item for item in manifest['files']))
        publication=self.root/'reader-fixture';publication.mkdir()
        atomic_json(publication/'manifest.json',{'reader':{'kind':'native_source_view','path':str(output),
                    'complete':release.artifact(output/'COMPLETE.json'),
                    'manifest':release.artifact(output/'manifest.json')}})
        atomic_json(publication/'COMPLETE.json',{'complete':True,'manifest_sha256':file_sha(publication/'manifest.json')})
        rows={row['sample_id']:row for batch in release.iter_batches(publication,batch_size=1) for row in batch.to_pylist()}
        self.assertFalse(rows[sha('new-i2t')]['image_loss_allowed'])
        self.assertEqual(rows[sha('new-i2t')]['task_route'],'i2t')
        self.assertEqual(rows[sha('new-i2t')]['processing_policy'],view.TASK_POLICY)
        self.assertTrue(rows[sha('new-t2i')]['image_loss_allowed'])
        self.assertIsNone(rows[sha('legacy')]['image_loss_allowed'])
        for path,expected_hash in originals.items():
            self.assertEqual(file_sha(path),expected_hash)
            for original in pq.read_table(path).to_pylist():
                returned=rows[original['sample_id']]
                self.assertEqual({key:returned[key] for key in original},original)
        self.assertEqual(view.build(candidate,output,threads=1,memory_limit='256MB'),result)
        policy_file=output/manifest['files'][0]['task_policy_path']
        before=policy_file.read_bytes();policy_file.write_bytes(bytes([0])+before[1:])
        with self.assertRaisesRegex(ValueError,'Task policy sidecar byte count or SHA'):
            list(release.iter_batches(publication))

if __name__=='__main__':unittest.main()
