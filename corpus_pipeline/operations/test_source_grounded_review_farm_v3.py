"""Temporary-only transport and exact v2/v3 epoch compatibility regression tests."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import unittest

from corpus_pipeline.operations import source_grounded_review_farm_v3 as f
from corpus_pipeline.operations import test_source_grounded_review_farm as fixtures

OLD = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/dcvlm_baseline_full_20261028_v3/operations/source_grounded_review_20261029/code-v2/source_grounded_review_farm.py')


def module(path):
    spec=importlib.util.spec_from_file_location('test_farm_epoch',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


class TransportMigrationTests(fixtures.FarmTests):
    # Inherited tests run unchanged against v3, including strict transform audit,
    # partial success preservation, signal drain, and owned-child recovery.
    def setUp(self):
        self.original=fixtures.f;fixtures.f=f
        super().setUp()

    def tearDown(self):
        super().tearDown();fixtures.f=self.original

    def fixture_batch(self, *, extra=None, wrong_scope=False, completed=True):
        self.init();d=self.root/'batches'/'fixture';db=f.connect(self.root)
        binding=f.meta(db,'source_binding');db.close()
        f.execute_batch(d,'fixture',self.items,str(self.binary),binding)
        raw=json.loads((d/'last-message.json').read_text())
        if wrong_scope: raw['reviews'][0]['item_id']='0'*64
        text=f.canonical(raw);(d/'last-message.json').write_text(text)
        events=[{'type':'thread.started','thread_id':'fixture-thread'},
                {'type':'error','message':'Reconnecting... 2/5 (unexpected status 403 Forbidden)'},
                {'type':'item.completed','item':{'id':'item_0','type':'error','message':'Falling back from WebSockets to HTTPS transport. failed to connect; unexpected status 403 Forbidden'}}]
        if extra: events.append(extra)
        events.append({'type':'item.completed','item':{'type':'agent_message','text':text}})
        if completed: events.append({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':100}})
        with gzip.open(d/'events.jsonl.gz','wt') as out: out.write('\n'.join(f.canonical(x) for x in events))
        (d/'stderr.log').write_text('ERROR failed to connect to websocket: HTTP error 403 Forbidden\n')
        return d,events

    def test_real_fallback_repro_keeps_completed_result(self):
        d,events=self.fixture_batch();old=module(OLD)
        self.assertEqual(old.parse_events('\n'.join(map(f.canonical,events)))['tool_events'],1)
        outcome=f.collect(d,self.items,0)
        self.assertEqual(len(outcome['accepted']),4);self.assertTrue(outcome['transport_healthy'])
        self.assertIsNone(outcome['error_type'])

    def test_real_tool_or_unknown_error_still_rejected(self):
        d,_=self.fixture_batch(extra={'type':'item.completed','item':{'type':'command_execution','command':'echo fixture'}})
        outcome=f.collect(d,self.items,0)
        self.assertEqual(outcome['accepted'],{});self.assertEqual(outcome['error_type'],'contract_violation')
        events=f.parse_events(f.canonical({'type':'item.completed','item':{'type':'error','message':'unrecognized error'}}))
        self.assertEqual(events['tool_events'],1)

    def test_completed_bad_ids_not_connection(self):
        d,_=self.fixture_batch(wrong_scope=True);outcome=f.collect(d,self.items,0)
        self.assertEqual(outcome['error_type'],'validation_error')
        self.assertEqual(outcome['transport_signature'],'validation_error')
        db=f.connect(self.root);before=f.meta(db,'scheduler');db.close()
        self.assertEqual(f.update_scheduler(before,outcome,64),before)

    def test_incomplete_fallback_is_transport_failure(self):
        d,_=self.fixture_batch(completed=False);outcome=f.collect(d,self.items,0)
        self.assertEqual(outcome['accepted'],{});self.assertEqual(outcome['error_type'],'transport_error')

    def migrated_fixture(self):
        old=module(OLD);old.initialize(self.root,self.source,4)
        old.run(self.root,batch_size=1,ceiling=1,initial_concurrency=1,max_batches=1,binary=str(self.binary))
        directory=self.base/'code-v3';directory.mkdir();code=directory/'source_grounded_review_farm.py'
        shutil.copyfile(f.__file__,code);new=module(code)
        binding=json.loads((OLD.parent/'BINDING.json').read_text());binding['code']=new.bound(code)
        new.atomic(directory/'BINDING.json',binding)
        before=new.connect(self.root,readonly=True)
        protected=[dict(x) for x in before.execute("SELECT item_id,result_sha256,attempts,batch_id FROM items WHERE status='succeeded'")]
        scheduler=new.meta(before,'scheduler');before.close()
        identity=new.migrate(self.root,OLD)
        return old,new,identity,protected,scheduler

    def test_mixed_epochs_preserve_success_and_strict_audit(self):
        old,new,identity,protected,scheduler=self.migrated_fixture()
        self.assertEqual(identity['original_code_sha256'],new.sha(OLD))
        db=new.connect(self.root,readonly=True)
        self.assertEqual(new.meta(db,'code_sha256'),new.sha(OLD));self.assertEqual(new.meta(db,'scheduler'),scheduler);db.close()
        new.run(self.root,ceiling=1,initial_concurrency=1,binary=str(self.binary))
        report=new.export(self.root);self.assertTrue(report['complete'])
        db=new.connect(self.root,readonly=True)
        for row in protected:
            now=dict(db.execute('SELECT item_id,result_sha256,attempts,batch_id FROM items WHERE item_id=?',(row['item_id'],)).fetchone())
            self.assertEqual(now,row)
        db.close()
        with self.assertRaisesRegex(ValueError,'Batch identity'): old.strict_audit(self.root)
        self.assertTrue(new.strict_audit(self.root)['passed'])

    def test_migration_rejects_active_and_history_tamper(self):
        old,new,identity,protected,_=self.migrated_fixture()
        db=new.connect(self.root)
        with db:db.execute("UPDATE items SET result_sha256=? WHERE item_id=?",('0'*64,protected[0]['item_id']))
        db.close()
        with self.assertRaisesRegex(ValueError,'successful item changed'):new.controller_identity(self.root)
        record={'code_sha256':new.sha(OLD)}
        with self.assertRaisesRegex(ValueError,'outside registered epoch'):new.parser_for_batch(identity,old,'unregistered-v2-batch',record)

    def test_scientific_contract_exactly_unchanged(self):
        self.assertEqual(f.science_identity(OLD),f.science_identity(f.__file__))
        old=module(OLD)
        self.assertEqual(old.INSTRUCTION,f.INSTRUCTION);self.assertEqual(old.SCHEMA,f.SCHEMA)
        self.assertEqual(old.prompt_for(self.items),f.prompt_for(self.items))

    def test_migration_requires_drain_and_no_live_child(self):
        _,new,_,_,_=self.migrated_fixture()
        (self.root/'FARM_CODE_MIGRATION.json').unlink()  # temporary fixture only
        db=new.connect(self.root)
        with db:db.execute("UPDATE items SET status='running' WHERE status='pending'")
        with self.assertRaisesRegex(ValueError,'Drain all'):new.migrate(self.root,OLD)
        with db:db.execute("UPDATE items SET status='pending' WHERE status='running'")
        directory=Path(db.execute('SELECT directory FROM batches').fetchone()[0]);db.close()
        new.atomic(directory/'CHILD.json',{'identity':new.process_identity(os.getpid()),'deadline':0})
        with self.assertRaisesRegex(ValueError,'child still alive'):new.migrate(self.root,OLD)


if __name__=='__main__':unittest.main()
