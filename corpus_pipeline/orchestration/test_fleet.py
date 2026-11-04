import importlib.util,tempfile,unittest
from pathlib import Path
from corpus_pipeline.resource_manager.models import WorkloadSpec,Placement,Quota
from . import fleet as m
class SingleTests(unittest.TestCase):
    def controller(self):
        return m.SingleCardController(None,WorkloadSpec(scope='test',profile='test',max_cards=128,command=['true'],allowed_cards=[1,2,4,8,16]),'/tmp/unused-single-test')
    def test_old_pending_cards_reserved_new_only_one(self):
        c=self.controller();c.state={'resources':{'old':{'intent':{'placement':{'node':'native-scheduler','quota':{'cards':16,'cpu':64,'memory':1024}}},'resource_id':'old','state':'PENDING','ready':False,'retired_verified':False}}}
        snapshot={'nodes':[{'name':'native-scheduler','gpu':128,'cpu':1024,'memory':8192}],'quotas':[{'cards':1,'cpu':8,'memory':64},{'cards':8,'cpu':64,'memory':512}], 'resources':{'old':{'state':'PENDING'}},'global_remaining_cards':112}
        nodes,quotas,reservations,remaining=c._inputs(snapshot)
        self.assertEqual([(q.cards,q.cpu,q.memory)for q in quotas],[(1,8,64)]);self.assertEqual(reservations[0].cards,16);self.assertFalse(reservations[0].is_retired);self.assertEqual(remaining,112)
        c.creates_this_poll=8;self.assertEqual(c._inputs(snapshot)[1],[])
        c.creates_this_poll=0;snapshot['quotas']=snapshot['quotas'][1:]
        with self.assertRaisesRegex(ValueError,'catalog'):c._inputs(snapshot)
    def test_poll_burst_launches_before_creating_next_eight(self):
        from corpus_pipeline.resource_manager.test_engine import FakeAdapter
        with tempfile.TemporaryDirectory() as d:
            a=FakeAdapter();a.nodes={'node-a':(128,1024,8192)}
            c=m.SingleCardController(a,WorkloadSpec(scope='test',profile='test',max_cards=128,command=['true'],allowed_cards=[1,16]),d)
            self.assertEqual(c.poll()['reserved_cards'],8)
            for v in a.resources.values():v['state']='RUNNING'
            self.assertEqual(c.poll()['reserved_cards'],16)
            launch=[i for i,x in enumerate(a.calls)if x[0]=='launch'];creates=[i for i,x in enumerate(a.calls)if x[0]=='create']
            self.assertEqual(len(launch),8);self.assertLess(max(launch),creates[8])
    def test_known_running_transport_failure_does_not_release_or_block_allocations(self):
        from corpus_pipeline.resource_manager.test_engine import FakeAdapter
        class TransportAdapter(FakeAdapter):
            def health(self,*args):raise RuntimeError('Remote worker operation was not confirmed')
        with tempfile.TemporaryDirectory() as d:
            a=TransportAdapter();a.nodes={'node-a':(128,1024,8192)}
            c=m.SingleCardController(a,WorkloadSpec(scope='test',profile='test',max_cards=128,command=['true'],allowed_cards=[1,16],completion_command=['true']),d)
            c.poll()
            for v in a.resources.values():v['state']='RUNNING'
            result=c.poll();self.assertEqual(result['reserved_cards'],16)
            old=[x for x in c.state['resources'].values()if x.get('worker_transport_pending')]
            self.assertEqual(len(old),8)
            self.assertTrue(all(x['state']=='RUNNING' and x['ready'] is False and not x['retired_verified'] for x in old))
            self.assertFalse(any(x.get('observation_error_type')for x in old))
            a.observe=lambda _:(_ for _ in ()).throw(ValueError('Owned identity mismatch'))
            c.poll();self.assertEqual(len(a.resources),16)
            self.assertTrue(any(x.get('observation_error_type')for x in c.state['resources'].values()))
    def test_transport_type_requires_inspire_class_and_fresh_owned_observe(self):
        import json
        from corpus_pipeline.resource_manager.test_engine import FakeAdapter
        TransportError=type('TransportError',(Exception,),{'__module__':'inspire.platform.errors'})
        self.assertTrue(m.remote_transport_unconfirmed(TransportError()))
        self.assertFalse(m.remote_transport_unconfirmed(type('TransportError',(Exception,),{})()))
        self.assertFalse(m.remote_transport_unconfirmed(ValueError('owned identity')))
        class FlakyAdapter(FakeAdapter):
            def health(self,*args):raise TransportError('temporary')
        with tempfile.TemporaryDirectory() as d:
            a=FlakyAdapter();a.nodes={'node-a':(128,1024,8192)}
            c=m.SingleCardController(a,WorkloadSpec(scope='test',profile='test',max_cards=128,command=['true'],allowed_cards=[1]),d)
            c.poll()
            for v in a.resources.values():v['state']='RUNNING'
            self.assertEqual(c.poll()['reserved_cards'],16)
            self.assertTrue(any(x.get('worker_transport_pending')for x in c.state['resources'].values()))
    def test_create_wrong_quota_fails_before_platform_mutation(self):
        c=self.controller()
        with self.assertRaisesRegex(ValueError,'Only a real'):c._create(Placement('native-scheduler',Quota(8,64,512)))
        self.assertFalse(c._migration({}))
if __name__=='__main__':unittest.main()
