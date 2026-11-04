"""Production pool admission regression tests; fake registries and no HTTP/resources."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from corpus_pipeline.postprocess import api_screen as screen
from corpus_pipeline.postprocess.test_api_screen import FakeDynamic, sample, PROMPT
from corpus_pipeline.production.common import atomic_json


class PoolAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        dynamic_patch = patch.object(screen, 'DynamicClient', FakeDynamic)
        dynamic_patch.start(); self.addCleanup(dynamic_patch.stop)
        self.cls = screen.ShardedServingClients; FakeDynamic.calls = []
        self.pool = self.root / 'pool.json'

    def registry(self, number, ready=True):
        path = self.root / f'{number}.json'
        atomic_json(path, {'endpoints':[{'role':'verifier', 'ready':ready, 'ready_expires_at':9999999999,
            'backend':'inspire_serving', 'supports_images':True, 'expected_boots':['fixture-boot'],
            'model':screen.PRIMARY, 'revision':'revision', 'processor_revision':'processor',
            'runtime_revision':'runtime', 'precision':'BF16', 'request_overrides':{}}]})
        return path

    def create(self, count=1, cap=512, concurrency=2048, *, configured=True, weights=None):
        rows=[]
        for n in range(count):
            row={'path':str(self.registry(n)), 'weight':weights[n] if weights else 1}
            if configured: row['max_inflight']=cap
            rows.append(row)
        atomic_json(self.pool, {'schema_version':'corpus.screening_pool.v1', 'registries':rows})
        return self.cls({}, self.root / 'cache', concurrency=concurrency, wait_ready=.01,
                        pools={'primary':self.pool})

    async def parked(self, client):
        task=asyncio.create_task(client.acquire('primary'))
        await asyncio.sleep(.035)
        self.assertFalse(task.done()); self.assertEqual(FakeDynamic.calls, [])
        client.stop.set()
        with self.assertRaises(screen.ServingUnavailable): await task
        self.assertEqual(client.counters['request_invocations'], 0)
        self.assertGreaterEqual(client.counters['readiness_wait_windows'], 1)

    async def test_single_ready_replica_stops_at_512_even_with_global_2048(self):
        client=self.create()
        key=client.entries('primary')[0]['key']; self.assertEqual(client.entries('primary')[0]['max_inflight'],512)
        client.inflight[key]=511
        self.assertEqual(await client.acquire('primary'),key)
        await self.parked(client)
        self.assertEqual(client.inflight[key],512)
        await client.close()

    async def test_four_replicas_each_have_cap_and_global_limit_still_applies(self):
        client=self.create(4)
        entries=client.entries('primary')
        for entry in entries:client.inflight[entry['key']]=511
        acquired=[await client.acquire('primary') for _ in range(4)]
        self.assertEqual(set(acquired),{row['key'] for row in entries})
        self.assertEqual(sum(client.inflight.values()),2048)
        await self.parked(client); await client.close()
        # Cap larger than weighted/global share never overrides either limit.
        client=self.create(4,concurrency=1000)
        entries=client.entries('primary')
        for entry in entries:client.inflight[entry['key']]=249
        for _ in range(4):await client.acquire('primary')
        self.assertEqual(set(client.inflight.values()),{250})
        await self.parked(client); await client.close()

    async def test_preemption_does_not_increase_surviving_replica_pressure(self):
        client=self.create(4)
        entries=client.entries('primary')
        for entry in entries:client.inflight[entry['key']]=512
        for n in (1,2,3):
            self.registry(n,ready=False)
            client.inflight[entries[n]['key']]=0
        self.assertEqual(len(client.ready_entries('primary')),1)
        await self.parked(client)
        self.assertEqual(client.inflight[entries[0]['key']],512)
        await client.close()

    async def test_missing_cap_retains_original_weighted_behavior(self):
        client=self.create(2,concurrency=4,configured=False,weights=[1,3])
        keys=[await client.acquire('primary') for _ in range(4)]
        self.assertEqual([key[1] for key in keys].count(str(self.root/'0.json')),1)
        self.assertEqual([key[1] for key in keys].count(str(self.root/'1.json')),3)
        self.assertNotIn('max_inflight',client.entries('primary')[0])
        await self.parked(client);await client.close()

    async def test_invalid_explicit_caps_fail_closed(self):
        for value in (None,True,False,0,-1,1.0,'512',[],{}):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError,'invalid_serving_pool_max_inflight'):
                    self.create(cap=value)

    async def test_cap_refresh_does_not_change_contract_prompt_or_request_decision(self):
        client=self.create(cap=512)
        models=deepcopy(client.models);contracts=deepcopy(client.contracts)
        asset,_=screen.prepare_asset(sample())
        first=await client.request('primary',asset,PROMPT,screen.review_context(asset['asset_id']),1600)
        pool=json.loads(self.pool.read_text());pool['registries'][0]['max_inflight']=128;atomic_json(self.pool,pool)
        second=await client.request('primary',asset,PROMPT,screen.review_context(asset['asset_id']),1600)
        self.assertEqual(client.entries('primary')[0]['max_inflight'],128)
        self.assertEqual(models,client.models);self.assertEqual(contracts,client.contracts)
        self.assertEqual(first,second)
        await client.close()

    async def test_wait_does_not_consume_retries_and_interruption_refunds_reservation(self):
        client=self.create(cap=1)
        key=await client.acquire('primary')  # Occupy the only slot without HTTP.
        ledger=screen.Ledger(self.root/'ledger',{'fixture':True})
        self.addCleanup(ledger.close)
        worker=screen.Screener(ledger,client,mode='annotate',concurrency=2,max_attempts=3,prompt=PROMPT)
        asset,provenance=screen.prepare_asset(sample())
        task=asyncio.create_task(worker.review('primary',asset,provenance))
        await asyncio.sleep(.04)
        before=ledger.db.execute('SELECT attempts,state FROM requests').fetchone()
        self.assertEqual(before,(1,'inflight'))  # Existing durable pre-send reservation.
        await asyncio.sleep(.04)
        self.assertEqual(ledger.db.execute('SELECT attempts,state FROM requests').fetchone(),before)
        self.assertEqual(FakeDynamic.calls,[]);self.assertEqual(client.counters['request_invocations'],0)
        worker.stop.set();self.assertIsNone(await task)
        self.assertEqual(ledger.db.execute('SELECT attempts,state FROM requests').fetchone(),(0,'deferred'))
        self.assertEqual(client.inflight[key],1)
        await client.close()


if __name__ == '__main__': unittest.main()
