import tempfile,json,hashlib,unittest
from pathlib import Path
from types import SimpleNamespace
from . import worker_health as m
class SharedTests(unittest.TestCase):
    def fixture(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);r=Path(t.name);(r/'EXECUTION.json').write_text('{}')
        c={'stage':'frontpass','config_path':str(r/'pipeline.json'),'config_sha256':'config'}
        inner=SimpleNamespace(config={'worker_state_dir':str(r/'workers')},_workload_command=lambda _:['argv'])
        adapter=SimpleNamespace(contract=c,inner=inner)
        row={'resource_id':'resource','state':'RUNNING','runtime_id':'host:boot','launch':{'operation_id':'op'},'intent':{'operation_id':'allocation','placement':{'quota':{'cards':1}}}}
        p=r/'fleet/pools/frontpass/allocation.json';p.parent.mkdir(parents=True)
        binding={'schema':'corpus.unified.worker_pool.v1','allocation':'allocation','cards':1,'config_path':c['config_path'],'config_sha256':'config','execution_sha256':hashlib.sha256(b'{}').hexdigest(),'stage':'frontpass'}
        pool={'updated_at':100,'status':'running','fatal':False,'binding':binding,'runtime_id':'host:boot','host':'host','allocation':'allocation','cards':1,'stage':'frontpass','active_children':{'0':123},'pid':122};p.write_text(json.dumps(pool))
        d=r/'workers/resource';(d/'operations').mkdir(parents=True);(d/'CURRENT_LAUNCH.json').write_text(json.dumps({'operation_id':'op'}));(d/'operations/op.json').write_text(json.dumps({'phase':'running','operation_id':'op','kind':'launch','command':['argv'],'runtime_id':'host:boot','child':{'runtime_id':'host:boot','pid':122}}))
        return adapter,row,p,pool
    def test_bound_fresh_running_receipt(self):
        a,row,p,pool=self.fixture();h=m.health(a,row,now=105);self.assertTrue(h['ready']);self.assertEqual(h['launch_operation_id'],'op');self.assertEqual(row['health_evidence']['kind'],'fresh_bound_pool_heartbeat')
    def test_stale_restarted_or_inactive_not_assumed_healthy(self):
        a,row,p,pool=self.fixture();self.assertIsNone(m.health(a,row,now=116));row['runtime_id']='host:newboot';self.assertIsNone(m.health(a,row,now=105));row['runtime_id']='host:boot';pool['active_children']={};p.write_text(json.dumps(pool));self.assertIsNone(m.health(a,row,now=105))
    def test_changed_binding_fails_closed(self):
        a,row,p,pool=self.fixture();pool['binding']['config_sha256']='other';p.write_text(json.dumps(pool))
        with self.assertRaisesRegex(ValueError,'binding'):m.health(a,row,now=105)
if __name__=='__main__':unittest.main()
