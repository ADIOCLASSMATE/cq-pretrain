"""Joint end-to-end fixtures; synthetic inference here is never real evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pyarrow.parquet as pq
from .config import checked
from .test_ingest import balanced,genund,image_bytes
from .test_cache import FakeWorker
from . import workflow,release


class WorkflowTests(unittest.TestCase):
    def config(self,root):
        first,_=balanced(root/'balanced')
        second,_=genund(root/'genund',[('a.png',image_bytes()),('a.txt','A green square.'),
            ('b.png',image_bytes('blue')),('b.txt','A blue square.')])
        value={'schema':'corpus.unified.pipeline.v1','scope':'validation','inputs':[first,second],
               'output':str(root/'run'),'compute':{'threads':1,'io_workers':1,'batch_size':4,'rows_per_task':2,
                                                'memory_limit':'512MB'},'screening':{'backend':'history_only'},
               'sscd':{'block_rows':2,'query_tile':2,'database_tile':2,'candidate_batch':2,'min_free_gib':0}}
        path=root/'config.json';path.write_text(json.dumps(value));return path

    def test_joint_dag_resume_and_preserve_task_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.config(root);worker=FakeWorker()
            first=workflow.run(config,local=True,stop_after='frontpass',delegate=worker)
            self.assertEqual(first['stage'],'frontpass')
            count=len(worker.calls)
            result=workflow.run(config,local=True,delegate=worker)
            self.assertTrue(result['complete']);self.assertEqual(len(worker.calls),count)
            manifest=release.read_manifest(root/'run')
            self.assertEqual(manifest['records'],6)
            self.assertEqual(set(manifest['per_collection']),{'dcvlm','genund'})
            self.assertFalse(manifest['sscd_candidates_automatically_deleted'])
            self.assertEqual(manifest['counts'],{'keep':6,'exclude':0,'quarantine':0})
            observed=[row for batch in release.iter_batches(root/'run',batch_size=2) for row in batch.to_pylist()]
            self.assertEqual(len(observed),6)
            self.assertEqual(workflow.run(config,local=True)['release_id'],result['release_id'])
            receipt=root/'run/receipts/frontpass.json';value=json.loads(receipt.read_text());value['dependencies']={}
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'Missing required dependency'):
                workflow.run(config,local=True)

    def test_configuration_rejects_fake_full_subset_and_policy_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.config(root);value=json.loads(config.read_text());value['scope']='full'
            with self.assertRaisesRegex(ValueError,'some shards or sources'):checked(value)
            value['scope']='validation';value['policy']={'sscd_automatic_delete':True}
            with self.assertRaisesRegex(ValueError,'Unsupported policy'):checked(value)

    def test_full_single_collection_requires_explicit_complete_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.config(root);value=json.loads(path.read_text())
            value['scope']='full';value['inputs']=value['inputs'][:1]
            with self.assertRaisesRegex(ValueError,'required_collections'):checked(value)
            value['required_collections']=['dcvlm']
            self.assertEqual(checked(value)['required_collections'],['dcvlm'])
            for invalid in ([],['dcvlm','dcvlm'],['other'],['genund'],'dcvlm'):
                with self.subTest(required_collections=invalid):
                    with self.assertRaisesRegex(ValueError,'required_collections'):
                        checked({**value,'required_collections':invalid})
            with self.assertRaisesRegex(ValueError,'record sampling'):
                checked({**value,'max_records':1})
            with self.assertRaisesRegex(ValueError,'some shards or sources'):
                checked({**value,'inputs':[{**value['inputs'][0],'paths':['one.tar']}]})
            self.assertEqual(workflow.preflight(path)['required_collections'],['dcvlm','genund'])

    def test_baseline_manifest_pending_cannot_be_published_as_full_normalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'raw';source.mkdir();control=root/'download';control.mkdir()
            (control/'manifest.json').write_text(json.dumps({'repo':'mlfoundations/dcvlm-baseline-200b',
                                                            'destination':str(source)}))
            value={'schema':'corpus.unified.pipeline.v1','scope':'full','required_collections':['dcvlm'],
                   'inputs':[{'collection':'dcvlm','adapter':'dcvlm_baseline_wds','root':str(source),
                              'download_control':str(control)}],'output':str(root/'run')}
            config=root/'config.json';config.write_text(json.dumps(value));cfg=checked(value)
            self.assertTrue(workflow.preflight(config)['ready'])
            with patch('corpus_pipeline.unified.baseline_index.prepare'), \
                    patch('corpus_pipeline.unified.baseline_index.status',return_value={
                        'phase':'waiting_for_download','units_complete':1,'units_total':2}), \
                    patch('corpus_pipeline.unified.baseline_index.finalize') as finalize:
                with self.assertRaisesRegex(workflow.Pending,'1/2 TAR units'):
                    workflow.execute_stage(cfg,root/'run','normalize')
                finalize.assert_not_called()
            self.assertFalse((root/'run/receipts/normalize.json').exists())
            with self.assertRaisesRegex(ValueError,'Cache must not overlap'):
                checked({**value,'cache':{'directory':str(source/'cache')}})
            with self.assertRaisesRegex(ValueError,'Output must not overlap'):
                checked({**value,'output':str(source/'preprocessing')})
            (control/'manifest.json').write_text(json.dumps({'repo':'mlfoundations/dcvlm-baseline-200b',
                                                            'destination':str(root/'wrong')}))
            self.assertFalse(workflow.preflight(config)['ready'])

    def test_baseline_separate_index_pending_and_status_use_same_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';index=run/'native_index'
            index.mkdir(parents=True);(index/'PLAN.json').write_text('{}')
            value={'schema':'corpus.unified.pipeline.v1','scope':'full','required_collections':['dcvlm'],
                   'inputs':[{'collection':'dcvlm','adapter':'dcvlm_baseline_wds','root':str(root/'raw'),
                              'download_control':str(root/'download'),'index_output':str(index)}],
                   'output':str(run),'compute':{'rows_per_task':16384,'index_rows_per_file':8192}}
            config=root/'pipeline.json';config.write_text(json.dumps(value));cfg=checked(value)
            (run/'STATUS.json').write_text(json.dumps({'stage':'normalize','complete':False}))
            progress={'phase':'indexing','units_complete':1,'units_total':3}
            with patch('corpus_pipeline.unified.baseline_index.prepare') as prepare, \
                    patch('corpus_pipeline.unified.baseline_index.status',return_value=progress) as status:
                with self.assertRaisesRegex(workflow.Pending,'1/3 TAR units'):
                    workflow.execute_stage(cfg,run,'normalize')
                prepare.assert_called_once_with(str(root/'download'),index,rows_per_file=8192)
                self.assertEqual(workflow.status(config)['progress'],progress)
                self.assertEqual(status.call_args.args,(str(root/'download'),index))
            for invalid in (root/'outside',run,run/'native'):
                with self.subTest(index_output=invalid),self.assertRaisesRegex(ValueError,'Separate baseline index'):
                    checked({**value,'inputs':[{**value['inputs'][0],'index_output':str(invalid)}]})


if __name__=='__main__':unittest.main()
