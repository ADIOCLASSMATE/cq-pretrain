"""Task-aware policy through the actual DAG, using fake model evidence only."""
import json
from pathlib import Path
import tempfile
import unittest

from . import workflow,release
from .config import checked
from .test_ingest import write_tar,image_bytes
from .test_cache import FakeWorker
from corpus_pipeline.production.common import file_sha
from corpus_pipeline.dedup.asset_supervision_view import iter_assets,iter_supervisions


class TaskWorkflowTests(unittest.TestCase):
    def fixture(self,root):
        tar=write_tar(root/'genund/laion-aesthetics-12m/00000.tar',[
            ('a.png',image_bytes()),('a.txt','A green square.'),
            ('a.json',json.dumps({'caption':'A green shape on a plain canvas.',
                                  'image_origin':'ai_generated'})),
            ('b.png',image_bytes('blue')),('b.txt','A blue square.')])
        inp={'collection':'genund','adapter':'genund_caption_tar','root':str(root/'genund'),
             'source':'laion-aesthetics-12m','direction':'i2t'}
        cfg={'schema':'corpus.unified.pipeline.v1','scope':'validation','inputs':[inp],
             'policy':{'profile':'task_aware_v1'},'output':str(root/'run'),
             'compute':{'threads':1,'io_workers':1,'batch_size':4,'rows_per_task':2,'memory_limit':'512MB'},
             'screening':{'backend':'history_only'},
             'sscd':{'block_rows':2,'query_tile':2,'database_tile':2,'candidate_batch':2,'min_free_gib':0}}
        path=root/'config.json';path.write_text(json.dumps(cfg))
        return path,cfg,tar

    def test_configuration_requires_direction_and_all_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,cfg,_=self.fixture(Path(tmp))
            value=checked(cfg)
            self.assertIs(value['inputs'][0]['preserve_all_captions'],True)
            self.assertIs(value['policy']['i2t_image_loss'],False)
            cfg['inputs'][0].pop('direction')
            with self.assertRaisesRegex(ValueError,'explicit i2t/t2i'):
                checked(cfg)

    def test_i2t_ai_multi_caption_single_asset_and_permission_through_dag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,cfg,tar=self.fixture(root);original=file_sha(tar)
            worker=FakeWorker()
            result=workflow.run(path,local=True,delegate=worker)
            self.assertTrue(result['complete'])
            self.assertEqual(file_sha(tar),original)
            self.assertTrue(worker.calls)
            self.assertTrue(all(names==('sscd',) for _,names in worker.calls))
            manifest=release.read_manifest(root/'run')
            self.assertEqual(manifest['identity']['policy']['task_routing'],'corpus.task_policy.v1')
            rows=[row for batch in release.iter_batches(root/'run') for row in batch.to_pylist()]
            self.assertEqual(len(rows),3)
            self.assertTrue(all(row['image_loss_allowed'] is False for row in rows))
            self.assertTrue(all(row['task_route']=='i2t' for row in rows))
            view=manifest['reader']['asset_supervision_view']['path']
            self.assertEqual(len(list(iter_assets(view))),2)
            self.assertEqual(len(list(iter_supervisions(view))),3)
            self.assertFalse((root/'run/api_review').exists())
            self.assertEqual(workflow.run(path,local=True)['release_id'],result['release_id'])

    def test_ai_excluded_t2i_cannot_remove_shared_i2t(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,cfg,_=self.fixture(root)
            write_tar(root/'t2i/laion-aesthetics-12m/00000.tar',[
                ('a.png',image_bytes()),('a.txt','A green square.'),
                ('a.json',json.dumps({'caption':'A green square.','image_origin':'ai_generated'}))])
            cfg['inputs'].append({**cfg['inputs'][0],'root':str(root/'t2i'),'direction':'t2i'})
            path.write_text(json.dumps(cfg));worker=FakeWorker()
            result=workflow.run(path,local=True,delegate=worker)
            self.assertTrue(result['complete'])
            manifest=release.read_manifest(root/'run')
            self.assertEqual(manifest['counts'],{'keep':3,'exclude':1,'quarantine':0})
            rows=[r for b in release.iter_batches(root/'run') for r in b.to_pylist()]
            self.assertTrue(all(r['task_route']=='i2t' and r['image_loss_allowed'] is False for r in rows))
            self.assertTrue(all(names==('sscd',) for _,names in worker.calls))


if __name__=='__main__':unittest.main()
