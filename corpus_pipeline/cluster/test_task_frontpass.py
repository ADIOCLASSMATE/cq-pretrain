"""Task routing is opt-in; I2T only extracts dedup evidence from valid bytes."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
from PIL import Image
import pyarrow.parquet as pq
import torch

from . import frontpass, inference
from .optimized_inference import OptimizedResidentInferenceWorker
from .test_frontpass import native_fixture, THRESHOLDS, record
from corpus_pipeline.production.records import snapshot
from corpus_pipeline.production.ingest import iter_records


class TinyWorker(inference.ResidentInferenceWorker):
    def __init__(self):
        self.torch,self.device,self.dtype=torch,torch.device('cpu'),torch.float32
        self.identity_sha256='fixture-task-routing';self.batch_size=32
        self.names=('cf384','unifd','sscd');self.closed=False;self.calls=[]
        self.pool=ThreadPoolExecutor(max_workers=2)
        self.bundles={name:{'model':lambda x:torch.full((len(x),1),8.),
                           'transform':lambda image:torch.zeros(3,4,4)} for name in ('cf384','unifd')}
        self.bundles['sscd']={'model':lambda x:torch.ones(len(x),512)}
        self.stats={'input_images':0,'complete_images':0,'batches':0,'decode_seconds':0.,
                    'model_seconds':dict.fromkeys(self.names,0.),'model_batches':dict.fromkeys(self.names,0)}

    def infer_batch(self,assets,*,models=None):
        assets=list(assets)
        if assets:self.calls.append((tuple(models),[a['sha256'] for a in assets]))
        return super().infer_batch(assets,models=models)


def mark(item, route):
    item=deepcopy(item);normalized=item['normalized']
    normalized.update(processing_policy=frontpass.TASK_POLICY,task_route=route,
                      image_loss_allowed=route=='t2i',decision='accepted',reason_codes=[])
    if route=='t2i' and any(p['origin']=='ai_generated' for p in normalized['provenance']['images']):
        normalized.update(decision='rejected',reason_codes=['image_generated_by_ai'])
    return item


class TaskFrontpassTests(unittest.TestCase):
    def load_fixture(self,directory):
        dataset=Path(directory)/'native';_,images,task=native_fixture(dataset);snap=snapshot(dataset)
        manifest={'schema_version':'production.input.v2','collection':'dcvlm',
                  'snapshot_root':snap['root'],'snapshot_fingerprint':snap['fingerprint'],
                  'shards':[{**task,'row_stop':task['row_end']}]}
        rows=list(iter_records(manifest,snap,{}))
        return snap,images,task,rows

    def process(self,directory,snap,task,rows,worker,*,record_batch=128):
        task={**task,'row_end':len(rows)}
        claim={'task':task,'task_id':'test','token':'attempt'}
        with patch.object(frontpass,'iter_records',return_value=iter(rows)):
            summary,_=frontpass.process_task(Path(directory)/'run',claim,worker,snap,
                {'task_routing':frontpass.TASK_POLICY},SimpleNamespace(entries={}),THRESHOLDS,
                record_batch=record_batch)
        out=Path(directory)/'run/artifacts/test/attempt'
        return summary,pq.read_table(out/'records.parquet').to_pylist(),pq.read_table(out/'images.parquet').to_pylist()

    def test_i2t_ai_source_and_small_images_only_run_sscd(self):
        with tempfile.TemporaryDirectory() as directory,TinyWorker() as worker:
            snap,images,task,rows=self.load_fixture(directory)
            rows=[mark(row,'text' if not row['source_image_refs'] else 'i2t') for row in rows]
            summary,records,assets=self.process(directory,snap,task,rows,worker)
            self.assertTrue(worker.calls)
            self.assertTrue(all(models==('sscd',) for models,_ in worker.calls))
            self.assertTrue(all(r['status']=='i2t_dedup_only' for r in records if r['image_ids']))
            self.assertEqual({r['status'] for r in assets},{'i2t_dedup_only'})
            self.assertTrue(all(r['cf384_prob'] is None and r['unifd_prob'] is None for r in assets))
            self.assertTrue(any(r['asset_id']==images['small']['sha256'] and r['sscd_byte_offset'] is not None for r in assets))
            self.assertTrue(all(json.loads(r['provenance_json'])['image_loss_allowed'] is False for r in assets))
            self.assertEqual(summary['api_calls'],0)

    def test_known_ai_t2i_does_not_invoke_any_model(self):
        with tempfile.TemporaryDirectory() as directory,TinyWorker() as worker:
            snap,_,task,rows=self.load_fixture(directory)
            _,records,assets=self.process(directory,snap,task,[mark(rows[4],'t2i')],worker)
            self.assertEqual(worker.calls,[])
            self.assertEqual(records[0]['status'],'rejected')
            self.assertEqual(assets[0]['status'],'rejected_known_ai_source')

    def test_shared_image_keeps_i2t_when_t2i_is_suspect_in_both_orders(self):
        for reverse in (False,True):
            with self.subTest(reverse=reverse),tempfile.TemporaryDirectory() as directory,TinyWorker() as worker:
                snap,_,task,rows=self.load_fixture(directory)
                understanding=mark(rows[1],'i2t');generation=mark(rows[1],'t2i')
                selected=[generation,understanding] if reverse else [understanding,generation]
                _,records,assets=self.process(directory,snap,task,selected,worker,record_batch=1)
                self.assertEqual(sorted(r['status'] for r in records),['i2t_dedup_only','pending'])
                self.assertTrue(all(r['status']=='i2t_dedup_only' for r in assets if r['task_route']=='i2t'))
                self.assertTrue(all(r['status']=='ai_suspect_pending_api' for r in assets if r['task_route']=='t2i'))

    def test_shared_small_image_keeps_i2t_but_rejects_t2i_in_both_orders(self):
        for reverse in (False,True):
            with self.subTest(reverse=reverse),tempfile.TemporaryDirectory() as directory,TinyWorker() as worker:
                snap,_,task,rows=self.load_fixture(directory)
                selected=[mark(rows[5],'i2t'),mark(rows[5],'t2i')]
                if reverse:selected.reverse()
                _,records,assets=self.process(directory,snap,task,selected,worker,record_batch=1)
                self.assertEqual(sorted(r['status'] for r in records),['i2t_dedup_only','rejected'])
                self.assertEqual(sorted(r['status'] for r in assets),['i2t_dedup_only','rejected_too_small'])

    def test_i2t_integrity_failure_is_pending_and_policy_fields_are_required(self):
        item=record();item['normalized'].update(processing_policy=frontpass.TASK_POLICY,
            task_route='i2t',image_loss_allowed=False,reason_codes=['image_reference_unresolved'])
        self.assertEqual(frontpass.record_decision(item,['pending_inference_or_integrity_error'])[0],'pending')
        self.assertEqual(frontpass.route({'origin':'ai_generated'},
            {'error_stage':'decode','sscd':{'status':'ok'}},THRESHOLDS,task_route='i2t'),
            'pending_inference_or_integrity_error')
        item['normalized']['image_loss_allowed']=True
        with self.assertRaisesRegex(ValueError,'task policy'):frontpass.task_metadata(item)
        with self.assertRaisesRegex(ValueError,'bound I2T'):frontpass.record_decision(record(),['i2t_dedup_only'])

    def test_rules_are_frozen_and_digest_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            snap,_,_,_=self.load_fixture(directory)
            rules={'task_routing':frontpass.TASK_POLICY,'sources':[]}
            plan=frontpass.prepare(Path(directory)/'planned',dataset=snap['root'],rules=rules)
            rules['sources'].append({'changed':True})
            self.assertEqual(frontpass.rules_for_plan(plan)['sources'],[])
            plan['rules']['sources'].append({'tampered':True})
            with self.assertRaisesRegex(ValueError,'identity changed'):frontpass.rules_for_plan(plan)

    def test_decoder_small_image_opt_in_is_sscd_only_and_checks_bytes(self):
        for cls in (inference.ResidentInferenceWorker,OptimizedResidentInferenceWorker):
            with self.subTest(worker=cls.__name__),TinyWorker() as worker:
                body=io.BytesIO();Image.new('RGB',(8,12),'red').save(body,format='PNG');blob=body.getvalue()
                asset={'sha256':hashlib.sha256(blob).hexdigest(),'blob_loader':lambda:blob}
                ordinary,tensors=cls._prepare(worker,asset,('sscd',))
                self.assertEqual(ordinary['status'],'too_small');self.assertIsNone(tensors)
                allowed={**asset,'allow_small_for_dedup':True}
                prepared,tensors=cls._prepare(worker,allowed,('sscd',))
                self.assertEqual(prepared['status'],'prepared');self.assertEqual(set(tensors),{'sscd'})
                failed,tensors=cls._prepare(worker,allowed,('cf384','sscd'))
                self.assertEqual(failed['error_stage'],'decode');self.assertIsNone(tensors)
                failed,tensors=cls._prepare(worker,{**allowed,'sha256':'0'*64},('sscd',))
                self.assertEqual(failed['error_stage'],'decode');self.assertIsNone(tensors)


if __name__=='__main__':unittest.main()
