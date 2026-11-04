from pathlib import Path
import json
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from corpus_pipeline.cluster.frontpass import record_decision, schemas
from corpus_pipeline.postprocess.reconcile import atomic_json, artifact, digest, file_sha
from . import training_release as t


def sha(value):return digest(value)


class Fixture:
    def __init__(self,root):
        self.root=root;self.parts=[];self.assets={};self.records=0
    def asset(self,name,canonical=None,action='retain_annotated',status='no_detector_alert'):
        aid=sha(name);self.assets[aid]={'asset_id':aid,'canonical_asset_id':sha(canonical or name),
            'canonical_relation':'exact_oriented_rgba','global_status':status,'screening_action':action,
            'screening_reason':'test_reason','quality_reviewed':False,'sscd_component_id':42}
        return aid
    def part(self,name,rows):
        directory=self.root/'artifacts'/name/'attempt';directory.mkdir(parents=True)
        native=self.root/(name+'.parquet');raw=[]
        for item in rows:
            segments=[{'kind':'image','role':'user','turn':0,'text':None,'image_index':i} for i in range(len(item['images']))]
            segments.append({'kind':'text','role':item.get('role','assistant'),'turn':item.get('turn',0),'text':item['text'],'image_index':None})
            raw.append({'sample_id':sha(item['name']),'text_hash':t.text_fingerprint(segments),
                'source':item.get('source','test'),'source_shard':item.get('source_shard',name+'.tar'),
                'format':item.get('format','caption'),'image_count':len(item['images']),'segments':segments})
        table=pa.Table.from_pylist(raw);pq.write_table(table,native);raw=table.to_pylist()
        records=[]
        for i,(item,original) in enumerate(zip(rows,raw)):
            records.append({'sample_id':original['sample_id'],'row_index':i,'source_record_sha256':digest(original),
                'category':item.get('category','single_i2t'),'text_origin':item.get('text_origin','unknown'),
                'text_retained':True,'image_ids':item['images'],
                'image_statuses':item.get('image_statuses',['no_detector_alert']*len(item['images'])),
                'status':item.get('status','frontpass_pass_pending_global_dedup'),'reason_codes':item.get('reason_codes',[]),
                'layout_source_tag':'sample_to_decide','recaption_resolution_potential':False})
        pq.write_table(pa.Table.from_pylist(records,schema=schemas()[0]),directory/'records.parquet')
        pq.write_table(pa.table({'asset_id':pa.array([],type=pa.string())}),directory/'images.parquet')
        (directory/'sscd.f32').write_bytes(b'');atomic_json(directory/'summary.json',{})
        stat=native.stat();task={'task_id':name,'source':'test','source_shard':name+'.tar',
            'parquet_path':str(native),'parquet_fingerprint':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},
            'row_start':0,'row_end':len(rows)}
        self.parts.append({'task':task,'model_identity_sha256':sha('model'),
            'artifacts':[artifact(p,self.root) for p in directory.iterdir()]});self.records+=len(rows)
    def finish(self):
        index={'schema':'dcvlm.offline_frontpass_release.v1','scope':'offline_frontpass_only','complete':True,
            'counts':{'records':self.records},'parts':self.parts}
        atomic_json(self.root/'frontpass_index.json',index)
        atomic_json(self.root/'FRONTPASS_COMPLETE.json',{k:v for k,v in index.items() if k!='parts'}|
            {'index_sha256':file_sha(self.root/'frontpass_index.json')})
        path=self.root/'assets.parquet';pq.write_table(pa.Table.from_pylist(list(self.assets.values())),path)
        atomic_json(self.root/'assets.json',{'complete':True,'canonical_equivalence_only':True,
            'files':[{'path':str(path),'sha256':file_sha(path),'size_bytes':path.stat().st_size}]})


class TrainingReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'input';self.root.mkdir();self.f=Fixture(self.root);self.out=self.root.parent/'output'
    def build(self,**kwargs):
        return t.build(self.root,self.root/'assets.json',self.out,threads=1,memory_limit='256MB',batch_parts=1,**kwargs)
    def rows(self):
        return {r['sample_id']:r for path in self.out.glob('buckets/*/*.parquet') for r in pq.read_table(path).to_pylist()}
    def test_bad_caption_excludes_only_bound_record_before_duplicate_selection(self):
        aid=self.f.asset('shared')
        self.f.part('task0',[{'name':'bad','images':[aid],'text':'wrong'},
                            {'name':'alternative','images':[aid],'text':'correct'},
                            {'name':'unreviewed-same-text','images':[aid],'text':'wrong'}])
        self.f.finish()
        raw=pq.read_table(self.root/'task0.parquet').to_pylist()[0]
        path=self.root/'quality.parquet'
        pq.write_table(pa.Table.from_pylist([{'source_record_sha256':digest(raw),'reason':'bad_caption'}]),path)
        binding={'path':str(path),'sha256':file_sha(path),'size_bytes':path.stat().st_size}
        result=self.build(record_quality_files=[binding]);self.assertTrue(result['complete'])
        rows=self.rows();self.assertEqual(rows[sha('bad')]['decision_reason'],'confirmed_record_quality_exclusion')
        self.assertEqual(rows[sha('alternative')]['decision'],'keep')
        self.assertEqual(rows[sha('unreviewed-same-text')]['decision'],'keep')
    def test_supervision_order_categories_and_original_asset_screening_are_protected(self):
        a=self.f.asset('a');b=self.f.asset('b','a');c=self.f.asset('c');bad=self.f.asset('bad','a','exclude')
        pending=self.f.asset('pending',action='pending');broken=self.f.asset('broken',action='pending',status='pending_integrity')
        rows=[{'name':'a-caption','images':[a],'text':'cat'},
              {'name':'b-caption','images':[b],'text':'cat'},
              {'name':'other-caption','images':[b],'text':'dog'},
              {'name':'bad-caption','images':[bad],'text':'cat'},
              {'name':'qa','images':[a],'text':'cat','format':'conversation'},
              {'name':'roles','images':[a],'text':'cat','role':'user'},
              {'name':'turns','images':[a],'text':'cat','turn':1},
              {'name':'order-ab','images':[a,c],'text':'same','category':'interleaved'},
              {'name':'order-ba','images':[c,a],'text':'same','category':'interleaved'},
              {'name':'sscd-candidate','images':[c],'text':'cat'},
              {'name':'pending','images':[pending],'text':'u','status':'pending'},
              {'name':'broken','images':[broken],'text':'i'},
              {'name':'missing','images':[sha('absent')],'text':'m'},
              {'name':'small','images':[c],'text':'small','status':'rejected'},
              {'name':'text1','images':[],'text':'pure text','category':'pure_text'},
              {'name':'text2','images':[],'text':'pure text','category':'pure_text'}]
        self.f.part('task0',rows[:8]);self.f.part('task1',rows[8:]);self.f.finish()
        result=self.build();self.assertTrue(result['complete']);self.assertEqual(result['records'],16)
        found=self.rows();get=lambda key:found[sha(key)]
        self.assertEqual(sum(get(k)['decision']=='keep' for k in ('a-caption','b-caption')),1)
        for name in ('other-caption','qa','roles','turns','order-ab','order-ba','sscd-candidate','pending'):
            self.assertEqual(get(name)['decision'],'keep',name)
        self.assertEqual(get('bad-caption')['decision_reason'],'confirmed_image_screening_exclusion')
        for name in ('broken','missing'):self.assertEqual(get(name)['decision'],'quarantine')
        self.assertEqual(get('small')['decision_reason'],'existing_frontpass_rejection')
        self.assertEqual(sum(get(k)['decision']=='keep' for k in ('text1','text2')),1)
        for row in found.values():
            self.assertEqual(row['original_parquet_path'],str(self.root/(row['task_id']+'.parquet')))
            self.assertFalse(row['published_training'])
            if row['duplicate_of_record_id']:
                representative=next(r for r in found.values() if r['record_id']==row['duplicate_of_record_id'])
                self.assertEqual(representative['decision'],'keep')
        self.assertEqual(self.build(),result)
    def test_resume_and_native_tamper_fail_closed(self):
        a=self.f.asset('a');self.f.part('task0',[{'name':'first','images':[a],'text':'x'}])
        self.f.part('task1',[{'name':'second','images':[a],'text':'x'}]);self.f.finish()
        value=self.build(stop_after_prepare_batches=1);self.assertFalse(value['complete'])
        before=file_sha(self.out/'prepared/000000/records.parquet')
        result=self.build();self.assertEqual(result['counts'],{'keep':1,'exclude':1})
        self.assertEqual(before,file_sha(self.out/'prepared/000000/records.parquet'))
        # A source mutation is rejected on a fresh bound output, before publication.
        p=self.root/'task1.parquet';table=pq.read_table(p);pq.write_table(table,p)
        self.out=self.root.parent/'changed'
        with self.assertRaisesRegex(ValueError,'fingerprint changed'):self.build()
    def test_text_hash_tamper_and_duplicate_asset_mapping_are_rejected(self):
        a=self.f.asset('a');self.f.part('task0',[{'name':'first','images':[a],'text':'x'}]);self.f.finish()
        p=self.root/'assets.parquet';rows=pq.read_table(p).to_pylist();pq.write_table(pa.Table.from_pylist(rows+rows),p)
        atomic_json(self.root/'assets.json',{'complete':True,'canonical_equivalence_only':True,
            'files':[{'path':str(p),'sha256':file_sha(p),'size_bytes':p.stat().st_size}]})
        with self.assertRaisesRegex(ValueError,'Ambiguous or invalid canonical'):self.build()

    def test_final_admit_clears_old_detector_alert_but_not_unrelated_record_concerns(self):
        a=self.f.asset('a')
        self.f.part('task0',[
            {'name':'cleared','images':[a],'text':'x','status':'pending','reason_codes':['ai_suspect_pending_api']},
            {'name':'structure','images':[a],'text':'y','status':'pending','reason_codes':['unclassified_structure']}])
        self.f.finish();self.build();rows=self.rows()
        self.assertFalse(rows[sha('cleared')]['unresolved_screening'])
        self.assertTrue(rows[sha('structure')]['unresolved_screening'])

    def test_unresolved_original_binding_stays_quarantined_after_frontpass(self):
        aid=self.f.asset('shared')
        bad=self.f.asset('already-screen-excluded',action='exclude')
        reason='unresolved_original_image_binding'
        # Image inference can succeed even though the original placeholder
        # binding is ambiguous. A pending record must not become a keep row.
        def pending(images):
            statuses=['no_detector_alert']*len(images)
            status,reasons=record_decision({'source_image_refs':images,
                'normalized':{'decision':'quarantine','reason_codes':[reason]}},statuses)
            self.assertEqual(status,'pending')
            return {'status':status,'reason_codes':reasons,'image_statuses':statuses}
        self.f.part('task0',[
            {'name':'ambiguous','images':[aid],'text':'literal <image> and unchanged supervision',**pending([aid])},
            {'name':'clear','images':[aid],'text':'literal <image> and unchanged supervision'},
            {'name':'text-binding','images':[],'text':'original <image>:tag','category':'pure_text',**pending([])},
            {'name':'excluded-binding','images':[bad],'text':'preserve the full source',**pending([bad])},
            {'name':'screening-only','images':[aid],'text':'uncertain visual review','status':'pending',
             'reason_codes':['ai_suspect_pending_api']}])
        self.f.finish()
        native=self.root/'task0.parquet';before=file_sha(native)
        originals={r['sample_id']:r for r in pq.read_table(native).to_pylist()}
        complete=self.build();self.assertTrue(complete['complete'])
        rows=self.rows();self.assertEqual(len(rows),5)
        for name in ('ambiguous','text-binding','excluded-binding'):
            row=rows[sha(name)]
            self.assertEqual(row['decision'],'quarantine',name)
            self.assertEqual(row['decision_reason'],reason)
            self.assertTrue(row['integrity_quarantine'])
            self.assertIsNone(row['duplicate_of_record_id'])
            self.assertEqual(row['source_record_sha256'],digest(originals[sha(name)]))
        for name in ('clear','screening-only'):
            self.assertEqual(rows[sha(name)]['decision'],'keep',name)
        self.assertEqual(file_sha(native),before)
        self.assertEqual(complete['records'],5)
        self.assertEqual(complete['counts'],{'keep':2,'quarantine':3})

    def test_mixed_tar_end_to_end_preserves_original_source_rows_in_source_view(self):
        from corpus_pipeline.postprocess.test_reconcile import mixed_frontpass_fixture
        from corpus_pipeline.postprocess import reconcile
        from corpus_pipeline.unified.screening import build_decisions
        from . import asset_release,source_view
        base=self.root.parent/'mixed';run,native,tar,originals=mixed_frontpass_fixture(base)
        raw_sha=file_sha(tar);metadata_sha=file_sha(native/'mixed.parquet')
        recon=base/'reconciliation';reconcile.run(run,recon,threads=1,memory_limit='256MB')
        screen=base/'screening';build_decisions(recon,screen)
        assets=base/'assets';asset_release.build(recon,screen/'COMPLETE.json',assets,threads=1,memory='256MB')
        output=base/'candidate';complete=t.build(run,assets/'COMPLETE.json',output,threads=1,memory_limit='256MB')
        self.assertEqual(complete['records'],2)
        rows=[row for path in output.glob('buckets/*/keep.parquet') for row in pq.ParquetFile(path).read().to_pylist()]
        expected={row['sample_id']:row for row in originals}
        self.assertEqual(len(rows),2)
        for row in rows:
            original=expected[row['sample_id']]
            self.assertEqual(row['source'],original['source'])
            self.assertEqual(row['source_shard'],original['source_shard'])
            self.assertEqual(row['source_record_sha256'],digest(original))
        view=base/'source_view';source_view.build(output,view,threads=1,memory_limit='256MB')
        restored=[row for batch in source_view.iter_native_batches(view,batch_size=1) for row in batch.to_pylist()]
        self.assertEqual(restored,originals)
        self.assertEqual(file_sha(tar),raw_sha)
        self.assertEqual(file_sha(native/'mixed.parquet'),metadata_sha)

    def test_single_source_task_rejects_native_source_or_shard_mismatch(self):
        for field,value in [('source','different-source'),('source_shard','different.tar')]:
            with self.subTest(field=field):
                self.root=self.root.parent/field;self.root.mkdir();self.f=Fixture(self.root)
                aid=self.f.asset('a');self.f.part('task0',[{'name':'sample','images':[aid],'text':'original',field:value}])
                self.f.finish();self.out=self.root.parent/(field+'-output')
                with self.assertRaisesRegex(ValueError,'Native source identity differs'):
                    self.build()

if __name__=='__main__':unittest.main()
