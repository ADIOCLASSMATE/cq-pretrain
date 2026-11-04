import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq

from .review_health import ProgressHealth, load_credential, preflight
from .api_screen import Ledger, Screener, execute
from .test_api_screen import sample, bound_receipt, PRIMARY, SECONDARY, PROMPT


class HealthTests(unittest.TestCase):
    def test_old_success_does_not_mask_a_later_failed_window(self):
        guard=ProgressHealth({'actual_responses':1000},{'http_attempts':2000},now=0)
        self.assertIsNone(guard.check({'actual_responses':1001},{'http_attempts':2001},now=10))
        self.assertIsNone(guard.check({'actual_responses':1001},{'http_attempts':2500},now=129))
        result=guard.check({'actual_responses':1001},{'http_attempts':2500},now=131)
        self.assertEqual(result['reason'],'no_valid_review_response')

    def test_preemption_wait_without_http_attempts_is_not_failure(self):
        guard=ProgressHealth({}, {}, now=0)
        self.assertIsNone(guard.check({'serving_transient_deferrals':10000},{},now=10000))

    def test_credentials_load_in_process_and_errors_never_contain_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'private';path.write_text('private-fixture-value\n');env={}
            load_credential(path,environ=env)
            self.assertEqual(env,{'INF_API_KEY':'private-fixture-value'})
            path.write_text('private fixture value')
            with self.assertRaisesRegex(ValueError,'^invalid_serving_credential$'):load_credential(path,environ={})
            with self.assertRaisesRegex(ValueError,'unavailable'):load_credential(environ={})


class PreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_uncached_bound_response_required_and_queue_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(Path(tmp),{'fixture':True})
            self.addCleanup(ledger.close);ledger.insert(sample());ledger.db.commit()
            class Clients:
                backend='serving';endpoint_identity={'primary':'c'*64};cached=False
                async def request(self,role,asset,prompt,context,max_tokens):
                    value=bound_receipt(asset['asset_id'],PRIMARY,prompt=prompt,context=context,backend='serving')
                    value.update(cache_hit=self.cached,real_model_inference=True)
                    return value
            clients=Clients();worker=Screener(ledger,clients,models={'primary':PRIMARY},mode='annotate',prompt=PROMPT)
            result=await preflight(worker,process_id='current-process')
            self.assertTrue(result['complete'])
            self.assertEqual(ledger.summary()['counts']['state:pending'],1)
            self.assertEqual(ledger.db.execute('SELECT count(*) FROM requests').fetchone()[0],0)
            clients.cached=True
            with self.assertRaisesRegex(ValueError,'real_image_preflight_failed'):
                await preflight(worker,process_id='next-process')

    async def test_production_execute_never_starts_bulk_when_probe_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input.parquet';pq.write_table(pa.Table.from_pylist([sample()]),source)
            prompt=root/'prompt';prompt.write_text(PROMPT)
            class Client:
                backend='serving';models={'primary':PRIMARY};endpoint_identity={'primary':'c'*64};contracts={}
                async def request(self,*args):raise RuntimeError('fixture authentication failure')
                async def close(self):pass
                def stats(self):return {}
            args=SimpleNamespace(input=[str(source)],reconciliation_root=None,canary=1,primary=PRIMARY,secondary=SECONDARY,
                mode='annotate',backend='serving',prompt=str(prompt),output=str(root/'out'),stop_file=None,
                image_threads=1,primary_registry=['unused'],primary_pool=None,secondary_registry=None,secondary_pool=None,
                concurrency=1,timeout=1,wait_ready=1,max_attempts=1,max_tokens=1600,progress_interval=1)
            with patch.dict('os.environ',{'INF_API_KEY':'fixture'}), \
                    patch('corpus_pipeline.postprocess.api_screen.ShardedServingClients',return_value=Client()), \
                    patch.object(Screener,'run') as bulk:
                with self.assertRaisesRegex(RuntimeError,'authentication'):await execute(args)
                bulk.assert_not_called()
            self.assertFalse((root/'out/COMPLETE.json').exists())
            import sqlite3
            db=sqlite3.connect(root/'out/index.sqlite')
            self.assertEqual(db.execute('SELECT state FROM assets').fetchall(),[('pending',)])
            self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],0);db.close()


if __name__=='__main__':unittest.main()
