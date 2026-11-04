from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

from corpus_pipeline.postprocess.reconcile import digest,atomic_json,file_sha,artifact
from .cache import CachedInferenceWorker,EvidenceCache,CacheCorruptionError,asset_sha


IDENTITY={'worker':'unit-test','device':'cpu','dtype':'float32','models':{'cf384':'pinned-cf','unifd':'pinned-uf','sscd':'pinned-sscd'}}


class FakeWorker:
    def __init__(self,identity=None,fail=False):
        self.identity=deepcopy(identity or IDENTITY);self.identity_sha256=digest(self.identity)
        self.names=('cf384','unifd','sscd');self.batch_size=3;self.calls=[];self.stats={};self.fail=fail;self.closed=False
    def iter_results(self,assets,*,models=None):
        assets=list(assets);names=tuple(models or self.names);self.calls.append(([asset_sha(a) for a in assets],names))
        from corpus_pipeline.cluster.inference import _read_blob
        for asset in assets:
            _read_blob(asset);sha=asset_sha(asset)
            value={'image_sha256':sha,'model_identity_sha256':self.identity_sha256,'status':'error' if self.fail else 'ok',
                   'original_width':256,'original_height':256,'scores':{},'source':'DO_NOT_CACHE','provenance':{'origin':'protected'}}
            if not self.fail:
                for name in names:
                    if name=='sscd':
                        vector=np.zeros(512,dtype=np.float32);vector[0]=1
                        value.update(pixel_sha256=digest(['pixels',sha]),sscd={'status':'ok','dimensions':512,'embedding':vector})
                    else:value['scores'][name]={'status':'ok','prob':.2,'logit':-1.3862943611198906}
            yield value
    def close(self):self.closed=True


def put_from_process(arguments):
    root,sha=arguments;cache=EvidenceCache(root,IDENTITY)
    value={'image_sha256':sha,'model_identity_sha256':digest(IDENTITY),'status':'ok','original_width':256,'original_height':256,
           'scores':{'cf384':{'status':'ok','prob':.3}}}
    cache.put(value);return cache.get_many([sha])[sha][0]['scores']['cf384']['prob']


class EvidenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.assets=[]
        for i in range(5):
            blob=('encoded-image-'+str(i)).encode();path=self.root/f'image-{i}';path.write_bytes(blob)
            self.assets.append({'sha256':hashlib.sha256(blob).hexdigest(),'image_path':str(path)})
    def worker(self,**kwargs):
        delegate=kwargs.pop('delegate',None) or FakeWorker();return CachedInferenceWorker(delegate,self.root/'cache',**kwargs)
    def test_cross_run_hits_exact_sha_not_path_preserve_order_and_source_free(self):
        first=self.worker();expected=list(first.iter_results(self.assets));self.assertEqual(len(first.delegate.calls),1)
        second=self.worker();moved=[]
        for i,asset in enumerate(reversed(self.assets)):
            path=self.root/f'relocated-{i}';path.write_bytes(Path(asset['image_path']).read_bytes());moved.append({**asset,'image_path':str(path),'provenance':{'origin':'mixed'}})
        actual=list(second.iter_results(moved));self.assertEqual(second.delegate.calls,[])
        self.assertEqual([x['image_sha256'] for x in actual],[x['sha256'] for x in moved])
        for result in actual:
            self.assertNotIn('source',result);self.assertNotIn('provenance',result)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['sscd']['embedding'].dtype,np.float32)
        self.assertEqual(second.identity,first.identity);self.assertEqual(second.identity_sha256,first.identity_sha256)
    def test_partial_models_retry_failures_and_identity_drift(self):
        worker=self.worker();list(worker.iter_results(self.assets[:1],models=('sscd',)))
        result=list(worker.iter_results(self.assets[:1]))[0]
        self.assertEqual(worker.delegate.calls[-1][1],('cf384','unifd'))
        self.assertIn('sscd',result);self.assertEqual(set(result['scores']),{'cf384','unifd'})
        failed=self.worker(delegate=FakeWorker(fail=True));list(failed.iter_results(self.assets[1:2]));list(failed.iter_results(self.assets[1:2]))
        self.assertEqual(len(failed.delegate.calls),2)
        other=deepcopy(IDENTITY);other['device']='npu:0';changed=self.worker(delegate=FakeWorker(other))
        list(changed.iter_results(self.assets[:1]));self.assertEqual(len(changed.delegate.calls),1)
    def test_corrupt_payload_and_identity_are_rejected(self):
        worker=self.worker();list(worker.iter_results(self.assets[:1]));sha=self.assets[0]['sha256']
        with sqlite3.connect(worker.cache.path(sha)) as db:db.execute('UPDATE evidence SET checksum=?',('0'*64,))
        with self.assertRaises(CacheCorruptionError):list(worker.iter_results(self.assets[:1]))
        marker=worker.cache.root/'IDENTITY.json';value=json.loads(marker.read_text());value['identity']['dtype']='float16';marker.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'Cache model identity changed'):self.worker()
    def test_wrong_current_bytes_or_dimensions_never_reuse_pass(self):
        worker=self.worker();list(worker.iter_results(self.assets[:1]));asset=self.assets[0]
        changed={**asset,'original_width':300};self.assertEqual(list(worker.iter_results([changed]))[0]['status'],'error')
        wrong=self.root/'wrong';wrong.write_bytes(b'other pixels')
        self.assertEqual(list(worker.iter_results([{**asset,'image_path':str(wrong)}]))[0]['status'],'error')
        # A second occurrence in the same first-miss batch must be checked too.
        cold=self.worker();new=self.assets[4]
        result=list(cold.iter_results([new,{**new,'image_path':str(wrong)}]))
        self.assertEqual([x['status'] for x in result],['ok','error'])
    def test_same_host_process_races_and_sealed_cross_writer_reuse(self):
        sha=self.assets[0]['sha256'];root=str(self.root/'race')
        with ProcessPoolExecutor(max_workers=3) as pool:
            self.assertEqual(list(pool.map(put_from_process,[(root,sha)]*6)),[.3]*6)
        cache=EvidenceCache(root,IDENTITY);other=EvidenceCache(root,IDENTITY,writer_id='isolated-new-worker')
        self.assertFalse(other.get_many([sha])) # Never reads another live WAL namespace.
        sealed=cache.seal();self.assertEqual(sealed['records'],1)
        self.assertEqual(other.get_many([sha])[sha][0]['scores']['cf384']['prob'],.3)
        file=Path(sealed['path']).parent/sealed['files'][0]['path'];file.write_bytes(file.read_bytes()+b'bad')
        with self.assertRaisesRegex(ValueError,'Sealed cache artifact changed'):other.get_many([sha])
    def test_unknown_and_invalid_vector_not_cached(self):
        worker=self.worker();sha=self.assets[0]['sha256']
        self.assertFalse(worker.cache.put({'image_sha256':sha,'status':'unknown'}))
        value={'image_sha256':sha,'model_identity_sha256':worker.identity_sha256,'status':'ok','scores':{},
               'original_width':256,'original_height':256,'pixel_sha256':digest('pixels'),
               'sscd':{'status':'ok','embedding':np.zeros(512,dtype=np.float32)}}
        with self.assertRaisesRegex(ValueError,'Invalid normalized'):worker.cache.put(value)
        self.assertFalse(worker.cache.get_many([sha]))
    def test_bounded_lookup_preserves_multibatch_prefetch_and_public_batches(self):
        worker=self.worker();seen=[]
        def inputs():
            for i in range(30):
                blob=('many-'+str(i)).encode();path=self.root/f'many-{i}';path.write_bytes(blob)
                seen.append(i);yield {'sha256':hashlib.sha256(blob).hexdigest(),'image_path':str(path)}
        stream=worker.iter_batches(inputs());first,values=next(stream)
        self.assertEqual(len(seen),12) # Four delegate batches, not unbounded collection.
        self.assertEqual(len(worker.delegate.calls[0][0]),12)
        self.assertEqual(len(first),3);self.assertEqual(len(values),3)
        rest=list(stream)
        self.assertEqual(sum(len(batch) for batch,_ in rest),27)
        self.assertEqual([len(ids) for ids,_ in worker.delegate.calls],[12,12,6])
        self.assertTrue(all(len(batch)<=worker.batch_size for batch,_ in rest))
        self.assertEqual(worker.identity_sha256,digest(IDENTITY))
        self.assertEqual(worker.execution_identity['cache_lookup_window'],12)
        with self.assertRaisesRegex(ValueError,'lookup_window'):self.worker(lookup_window=4097)

    def test_sscd_small_image_opt_in_never_reads_or_writes_cache_or_history(self):
        worker=self.worker();asset=self.assets[0];sha=asset_sha(asset)
        worker.cache.put({'image_sha256':sha,'model_identity_sha256':worker.identity_sha256,
            'status':'too_small','original_width':8,'original_height':12})
        worker.histories=[mock.Mock()]
        worker.histories[0].lookup_many.side_effect=AssertionError('history must be bypassed')
        original=worker.delegate.iter_results
        def small(assets,*,models):
            for value in original(assets,models=models):
                value.update(original_width=8,original_height=12)
                yield value
        worker.delegate.iter_results=small
        with (mock.patch.object(worker.cache,'get_many',side_effect=AssertionError('cache read')),
              mock.patch.object(worker.cache,'put_many',side_effect=AssertionError('cache write'))):
            for _ in range(2):
                result=list(worker.iter_results([{**asset,'allow_small_for_dedup':True}],models=('sscd',)))[0]
                self.assertEqual(result['status'],'ok');self.assertEqual(result['original_width'],8)
                self.assertEqual(result['sscd']['status'],'ok')
        self.assertEqual(len(worker.delegate.calls),2)
        self.assertEqual(worker.cache_stats['dedup_only_bypass_images'],2)
        self.assertEqual(worker.cache_stats['dedup_only_bypass_calls'],2)
        self.assertEqual(worker.cache.get_many([sha])[sha][0]['status'],'too_small')

    def test_mixed_same_sha_requests_keep_separate_decode_contracts_and_order(self):
        worker=self.worker();asset=self.assets[0];sha=asset_sha(asset)
        worker.cache.put({'image_sha256':sha,'model_identity_sha256':worker.identity_sha256,
            'status':'too_small','original_width':8,'original_height':12})
        rows=[asset,{**asset,'allow_small_for_dedup':True},asset]
        result=list(worker.iter_results(rows,models=('sscd',)))
        self.assertEqual([row['status'] for row in result],['too_small','ok','too_small'])
        self.assertEqual(worker.delegate.calls,[([sha],('sscd',))])

    def test_small_image_bypass_rejects_detectors_and_nonboolean_flags_before_cache(self):
        worker=self.worker();asset=self.assets[0]
        with mock.patch.object(worker.cache,'get_many',side_effect=AssertionError('cache read')):
            with self.assertRaisesRegex(ValueError,'SSCD-only'):
                list(worker.iter_results([{**asset,'allow_small_for_dedup':True}],models=('cf384','sscd')))
            for value in (None,1,'true'):
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'boolean'):
                    list(worker.iter_results([{**asset,'allow_small_for_dedup':value}],models=('sscd',)))
        self.assertEqual(worker.delegate.calls,[])

    def test_small_image_bypass_checks_delegate_order_and_current_bytes(self):
        worker=self.worker();asset={**self.assets[0],'allow_small_for_dedup':True}
        with mock.patch.object(worker.delegate,'iter_results',return_value=iter([])):
            with self.assertRaisesRegex(ValueError,'incomplete or reordered'):
                list(worker.iter_results([asset],models=('sscd',)))
        wrong=self.root/'wrong-bypass';wrong.write_bytes(b'wrong encoded image')
        with self.assertRaisesRegex(ValueError,'SHA256'):
            list(worker.iter_results([{**asset,'image_path':str(wrong)}],models=('sscd',)))


class HistoricalEvidenceTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from .history import register_history
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.front=self.root/'front';self.recon=self.root/'recon';part=self.front/'parts/part0';part.mkdir(parents=True)
        self.shas=['0a'+'1'*62,'0a'+'2'*62];self.identity=digest(IDENTITY)
        self.vector=np.zeros((2,512),dtype='<f4');self.vector[:,0]=1;self.vector.tofile(part/'sscd.f32')
        task={'task_id':'test-part','parquet_path':'/not-used-current-source.parquet','row_start':0,'row_end':2}
        images=[];assets=[]
        for i,sha in enumerate(self.shas):
            pixel=digest(['pixels',sha]);images.append({'asset_id':sha,'sample_id':str(i),'image_index':0,
                'status':'no_detector_alert' if i==0 else 'protected_source',
                'original_width':256,'original_height':256,'pixel_sha256':pixel,
                'cf384_prob':.2 if i==0 else None,'unifd_prob':.3 if i==0 else None,
                'sscd_byte_offset':i*2048,'error_json':'{}','original_ref_json':json.dumps({'sha256':sha,'path':'/old/unmounted/source'})})
            assets.append({'asset_id':sha,'task_id':'test-part','sample_id':str(i),'image_index':0,
                'model_identity_sha256':self.identity,'sscd_file':str(part/'sscd.f32'),
                'sscd_byte_offset':i*2048,'sscd_file_sha256':file_sha(part/'sscd.f32'),
                'sscd_file_size_bytes':4096,'original_width':256,'original_height':256,
                'pixel_sha256':pixel,'has_invalid_observation':False,'pixel_sha256_conflict':False,
                'dimension_count':1,'cf384_prob_max':.99,'unifd_prob_max':.99})
        pq.write_table(pa.Table.from_pylist(images),part/'images.parquet')
        atomic_json(part/'summary.json',{'task':task,'model_identity_sha256':self.identity})
        index={'complete':True,'counts':{'records':2},'parts':[{'task':task,'model_identity_sha256':self.identity,
            'artifacts':[artifact(part/name,self.front) for name in ['summary.json','images.parquet','sscd.f32']]}]}
        atomic_json(self.front/'frontpass_index.json',index)
        atomic_json(self.front/'FRONTPASS_COMPLETE.json',{'complete':True,'counts':index['counts'],'index_sha256':file_sha(self.front/'frontpass_index.json')})
        bucket=self.recon/'buckets/0a';bucket.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(assets),bucket/'assets.parquet')
        atomic_json(bucket/'COMPLETE.json',{'bucket':'0a','artifacts':[artifact(bucket/'assets.parquet',self.recon)]})
        atomic_json(self.recon/'COMPLETE.json',{'complete':True,'counts':{'assets':2},'input_index_sha256':file_sha(self.front/'frontpass_index.json'),
            'buckets':[{'bucket':'0a','receipt_path':'buckets/0a/COMPLETE.json','receipt_sha256':file_sha(bucket/'COMPLETE.json')}]})
        self.registry=self.root/'history';register_history(self.front,self.recon,self.registry)
    def test_history_uses_original_observation_and_partial_stage_not_maximum(self):
        from .history import HistoryIndex
        with HistoryIndex(self.registry) as history:
            result=history.lookup_many(self.shas,self.identity)
            self.assertEqual(result[self.shas[0]]['scores']['cf384']['prob'],.2)
            self.assertEqual(result[self.shas[1]]['scores'],{})
            self.assertEqual(history.last_report['complete_hits'],1)
            self.assertEqual(history.last_report['partial_hits'],1)
            self.assertTrue(np.array_equal(result[self.shas[1]]['sscd']['embedding'],self.vector[1]))
            # A new device/dtype is incompatible even with identical original bytes.
            other=deepcopy(IDENTITY);other['device']='npu:0'
            self.assertEqual(history.lookup_many(self.shas,digest(other)),{})
            self.assertEqual(history.last_report['identity_incompatible'],2)
        manifest=json.loads((self.registry/'manifest.json').read_text())
        self.assertEqual(manifest['per_asset_files_created'],0)
        self.assertEqual(len(list(self.registry.glob('*.json'))),2)
    def test_history_rechecks_changed_artifact_and_registration_identity(self):
        from .history import HistoryIndex,register_history
        register_history(self.front,self.recon,self.registry) # Exact replay is safe.
        with HistoryIndex(self.registry) as history:
            history.lookup_many(self.shas,self.identity)
            path=self.front/'parts/part0/sscd.f32';prior=path.stat();data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data)
            os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns)) # ctime still invalidates the memo.
            with self.assertRaisesRegex(ValueError,'Historical artifact changed'):history.lookup_many(self.shas,self.identity)
        marker=self.registry/'manifest.json';value=json.loads(marker.read_text());value['indexed_assets']=999;marker.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'manifest changed'):HistoryIndex(self.registry)
    def test_sha_bucket_verification_survives_small_task_file_lru(self):
        from .history import HistoryIndex
        with HistoryIndex(self.registry,max_verified_files=1) as history:
            with mock.patch('corpus_pipeline.unified.history.file_sha',wraps=file_sha) as checked:
                history.lookup_many(self.shas,self.identity);history.lookup_many(self.shas,self.identity)
                bucket=self.recon/'buckets/0a/assets.parquet'
                self.assertEqual(sum(Path(call.args[0])==bucket for call in checked.call_args_list),1)
            self.assertLessEqual(len(history.verified),1)
            self.assertLessEqual(len(history.fixed_verified),515)


if __name__=='__main__':unittest.main()
