import importlib.util,json,tempfile,unittest
from pathlib import Path
from corpus_pipeline.unified import workflow,release
from corpus_pipeline.unified import test_task_workflow
from corpus_pipeline.unified.test_cache import FakeWorker
from corpus_pipeline.dedup.training_release import iter_candidate_references
from . import exact_release as m

class ExactReleaseTests(unittest.TestCase):
    def selected_fixture(self,root,overrides):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from corpus_pipeline.dedup.training_release import artifact
        from corpus_pipeline.production.common import file_sha
        root.mkdir(parents=True);part=root/'keep.parquet'
        base={'decision':'keep','processing_policy':'corpus.task_policy.v1','task_route':'t2i',
            'image_loss_allowed':True,'unresolved_screening':False,'integrity_quarantine':False,
            'preexcluded':False,'status':'frontpass_pass_pending_global_dedup',
            'reason_codes':[],'screening_evidence':[],'text_hash':'different caption'}
        schema=pa.schema([(key,pa.bool_() if isinstance(value,bool) else pa.string()) for key,value in base.items()
            if key not in ('reason_codes','screening_evidence')]+[
                ('reason_codes',pa.list_(pa.string())),
                ('screening_evidence',pa.list_(pa.struct([('action',pa.string())])))])
        rows=[{**base,**change} for change in overrides]
        pq.write_table(pa.Table.from_pylist(rows,schema=schema),part)
        manifest={'counts':{'keep':len(rows)},'buckets':[{'artifacts':[artifact(part,root)]}]}
        (root/'manifest.json').write_text(json.dumps(manifest))
        (root/'COMPLETE.json').write_text(json.dumps({'complete':True,'manifest_sha256':file_sha(root/'manifest.json')}))
        return root
    def test_final_gate_rejects_unresolved_or_unpermitted_t2i_and_invalid_i2t(self):
        screen={'complete':True,'pending_infra':0,'review_uncertain':0,'counts':{}}
        cases=[{'unresolved_screening':True},{'unresolved_screening':None},
            {'screening_evidence':[{'action':'pending'}]},
            {'screening_evidence':[{'action':'exclude'}]},
            {'image_loss_allowed':False},{'status':'pending','reason_codes':['caption_recovery_pending']},
            {'status':'pending','reason_codes':[]},{'integrity_quarantine':True},
            {'task_route':'i2t','image_loss_allowed':True},{'processing_policy':None}]
        with tempfile.TemporaryDirectory() as d:
            for index,change in enumerate(cases):
                with self.subTest(change=change):
                    root=self.selected_fixture(Path(d)/str(index),[change])
                    with self.assertRaisesRegex(ValueError,'Strict selection rejects'):
                        m.strict_selection_check(root,screen)
    def test_final_gate_retains_i2t_and_all_captions_but_requires_complete_review(self):
        screen={'complete':True,'pending_infra':0,'review_uncertain':0,'counts':{}}
        with tempfile.TemporaryDirectory() as d:
            root=self.selected_fixture(Path(d)/'candidates',[
                {'task_route':'i2t','image_loss_allowed':False,'unresolved_screening':True},
                {'text_hash':'caption one'},
                {'text_hash':'caption two','status':'pending','reason_codes':['ai_suspect_pending_api'],
                 'screening_evidence':[{'action':'retain_annotated'}]}])
            result=m.strict_selection_check(root,screen)
            self.assertEqual((result['selected_records'],result['i2t_records'],result['t2i_records']),(3,1,2))
            for change in ({'complete':False},{'pending_infra':1},{'review_uncertain':1},{'counts':{'pending':1}}):
                with self.subTest(change=change):
                    with self.assertRaisesRegex(ValueError,'completed non-pending T2I screening'):
                        m.strict_selection_check(root,{**screen,**change})
    def fixture(self,root):
        path,cfg,_=test_task_workflow.TaskWorkflowTests().fixture(root)
        cfg['policy']['t2i_uncertain']='exclude';cfg['sscd']['dedup_mode']=m.MODE
        path.write_text(json.dumps(cfg));return path,cfg
    def decisions(self,root):
        result=[]
        for decision in ('keep','exclude','quarantine'):
            for row in iter_candidate_references(root/'training_candidate_index',decision=decision):
                result.append((decision,row['source'],row['category'],row['native_format'],row['text_hash'],
                        tuple(row['canonical_image_ids']),row['task_route'],row['image_loss_allowed']))
        return sorted(result)
    def test_exact_only_matches_advisory_path_and_never_claims_graph_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);full,fcfg=self.fixture(parent/'advisory');exact,ecfg=self.fixture(parent/'exact')
            workflow.run(full,local=True,delegate=FakeWorker())
            workflow.run(exact,local=True,delegate=FakeWorker(),stop_after='screening')
            result=m.publish(exact,allow_validation=True)
            root=Path(ecfg['output']);self.assertEqual(self.decisions(root),self.decisions(Path(fcfg['output'])))
            self.assertEqual(result['counts'],{'keep':3,'exclude':0,'quarantine':0})
            gate=json.loads(Path(result['selection_gate']['path']).read_text())
            self.assertEqual(gate['invalid_task_permissions'],0)
            self.assertEqual(gate['unresolved_t2i_records'],0)
            self.assertFalse(result['advisory_search_performed']);self.assertTrue(result['frontpass_embeddings_retained'])
            self.assertFalse((root/'sscd_search').exists());self.assertFalse((root/'sscd_graph').exists())
            self.assertFalse((root/'receipts/sscd_search.json').exists())
            self.assertFalse((root/'CURRENT.json').exists());self.assertTrue((root/'EXACT_CURRENT.json').exists())
            self.assertEqual(m.publish(exact,allow_validation=True),result)
            receipt=root/'exact_release/records.receipt.json';v=json.loads(receipt.read_text());v['complete']=False;receipt.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'Changed stage artifact'):m.verify(root)
    def test_mixed_t2i_source_rejection_and_exact_duplicate_match_advisory(self):
        from corpus_pipeline.unified.test_ingest import write_tar,image_bytes
        with tempfile.TemporaryDirectory() as d:
            configs=[]
            for label in ('advisory','exact'):
                parent=Path(d)/label;path,cfg=self.fixture(parent)
                write_tar(parent/'genund/laion-aesthetics-12m/00001.tar',[
                    ('duplicate.png',image_bytes()),('duplicate.txt','A green square.')])
                write_tar(parent/'t2i/laion-aesthetics-12m/00000.tar',[
                    ('a.png',image_bytes()),('a.txt','A green square.'),
                    ('a.json',json.dumps({'caption':'A green square.','image_origin':'ai_generated'})),
                    ('b.png',image_bytes('blue')),('b.txt','A blue square.')])
                cfg['inputs'].append({**cfg['inputs'][0],'root':str(parent/'t2i'),'direction':'t2i'})
                path.write_text(json.dumps(cfg));configs.append((path,cfg))
            workflow.run(configs[0][0],local=True,delegate=FakeWorker())
            workflow.run(configs[1][0],local=True,delegate=FakeWorker(),stop_after='screening')
            result=m.publish(configs[1][0],allow_validation=True)
            self.assertEqual(result['counts'],{'keep':4,'exclude':2,'quarantine':0})
            self.assertEqual(self.decisions(Path(configs[0][1]['output'])),self.decisions(Path(configs[1][1]['output'])))
    def test_missing_review_and_unopted_mode_cannot_publish(self):
        with tempfile.TemporaryDirectory() as d:
            p,cfg=self.fixture(Path(d));workflow.run(p,local=True,delegate=FakeWorker(),stop_after='reconcile')
            with self.assertRaisesRegex(ValueError,'Unfinished prerequisite screening'):m.publish(p,allow_validation=True)
            self.assertFalse((Path(cfg['output'])/'EXACT_CURRENT.json').exists())
            with self.assertRaisesRegex(ValueError,'full source scope'):m.publish(p)

if __name__=='__main__':unittest.main()
