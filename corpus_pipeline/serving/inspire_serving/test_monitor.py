import importlib.util
from pathlib import Path
import tempfile
import time
import unittest
import sys
from unittest.mock import patch
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('native_monitor',Path(__file__).with_name('monitor.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(Path(__file__).parent))
from downsize import capacity_choice


class NativeServingTests(unittest.TestCase):
    def test_ownership_preserves_exact_priority_and_native_identity(self):
        payload = {key: key + '-owned' for key in ('workspace_id', 'project_id', 'model_id',
            'model_version', 'mirror_id', 'command', 'port', 'logic_compute_group_id')}
        payload['resource_spec_price'] = {'gpu_count': 2, 'cpu_count': 16,
            'memory_size_gib': 128, 'gpu_type': 'Ascend910B', 'gpu_info': {'gpu_type': 'Ascend910B'}}
        for priority, tier in ((1, 'low'), (3, 'low'), (4, 'high'), (6, 'high')):
            state = {'service_id': 'sv-owned', 'name': 'owned', 'priority': priority,
                'tier': tier, 'expected_nodes': 1, 'payload': payload}
            view = {'project': m.PROJECT, 'priority': priority, 'replicas': 1, 'nodes_per_replica': 1}
            detail = SimpleNamespace(ref=SimpleNamespace(key='sv-owned'), name='owned',
                raw=payload, to_dict=lambda: view)
            with self.subTest(priority=priority):
                m.assert_owned(state, detail)
                view['priority'] = 3 if priority == 1 else 1
                with self.assertRaises(AssertionError):
                    m.assert_owned(state, detail)
                view['priority'] = priority
                state['tier'] = 'high' if tier == 'low' else 'low'
                with self.assertRaises(AssertionError):
                    m.assert_owned(state, detail)

    def state(self):
        now=time.time()
        return {'tier':'low','auto_recovery_enabled':True,'service_id':'sv-owned',
                'was_running':True,'last_running_at':now-2,'stopped_transition_at':now}

    def event(self,message='This pod was preempted by a higher priority workload'):
        return {'reason':'Preempted','message':message,'created_at':str(int(time.time()*1000)),
                'object_id':'project/sv-owned-0'}

    def test_high_never_low_restart(self):
        s=self.state();s['tier']='high'
        self.assertIsNone(m.recovery_decision(s,'STOPPED',[self.event()]))

    def test_http_failure_does_not_trigger_mutation(self):
        for status in ('RUNNING','PENDING','FAILED'):
            self.assertIsNone(m.recovery_decision(self.state(),status,[self.event()]))

    def test_actively_preempting_others_is_not_victim(self):
        self.assertIsNone(m.recovery_decision(self.state(),'STOPPED',[self.event('This pod will preempt pod another-project/other')]))

    def test_stale_event_does_not_undo_manual_stop(self):
        event=self.event();event['created_at']=str(int((time.time()-600)*1000))
        self.assertIsNone(m.recovery_decision(self.state(),'STOPPED',[event]))

    def test_future_and_nonfinite_event_rejected(self):
        for timestamp in (str(int((time.time()+30)*1000)),'NaN','Infinity'):
            event=self.event();event['created_at']=timestamp
            self.assertIsNone(m.recovery_decision(self.state(),'STOPPED',[event]))

    def test_current_victim_restarts_once(self):
        s=self.state();events=[self.event()]
        selected=m.recovery_decision(s,'STOPPED',events)
        self.assertIsNotNone(selected)
        s['restart_event']=selected
        self.assertIsNone(m.recovery_decision(s,'STOPPED',events))

    def test_unrelated_event_rejected(self):
        event=self.event();event['object_id']='project/sv-other-0'
        self.assertIsNone(m.recovery_decision(self.state(),'STOPPED',[event]))

    def test_high_preferred_until_acceptance(self):
        high={'tier':'high','ready':True};low={'tier':'low','ready':True}
        self.assertIs(m.choose([high,low]),high)
        self.assertIs(m.choose([high,low],True),low)

    def test_failover_only_to_qualified_low(self):
        high={'tier':'high','ready':False};low={'tier':'low','ready':True}
        self.assertIs(m.choose([high,low]),low)
        low['ready']=False
        self.assertFalse(m.choose([high,low])['ready'])

    def test_expired_high_lease_yields_to_fresh_low(self):
        high={'tier':'high','ready':True,'ready_expires_at':time.time()-1}
        low={'tier':'low','ready':True,'ready_expires_at':time.time()+30}
        self.assertIs(m.choose([high,low]),low)

    def test_planned_retirement_publishes_pending_low(self):
        high={'tier':'high','ready':False};low={'tier':'low','ready':False}
        self.assertIs(m.choose([high,low],True),low)

    def test_instance_restart_changes_identity(self):
        class I:
            raw={'name':'sv-owned-0','node':'n','created_at':'1','started_at':'2','restarts':0,'term':0}
        before=m.digest(m.instance_signature([I()]))
        I.raw['restarts']=1
        self.assertNotEqual(before,m.digest(m.instance_signature([I()])))

    def test_worker_client_created_inside_worker(self):
        # Actual SDK enforces creator-thread affinity; live observations are
        # additionally audited. This assertion protects the worker architecture.
        source=Path(m.__file__).read_text()
        worker=source.split('def worker(path):',1)[1].split('def main():',1)[0]
        self.assertIn("with InspireClient(account='wjx-ascend') as client:",worker)
        self.assertNotIn('ThreadPoolExecutor',source)

    def test_atomic_save_private(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'state.json';m.save(path,{'x':1})
            self.assertEqual(m.load(path),{'x':1})
            self.assertEqual(path.stat().st_mode&0o777,0o600)

    def test_downsize_requires_two_distinct_capacity_observations(self):
        self.assertEqual(capacity_choice([{'time':0,'complete_free_nodes':1}],32),'wait')
        self.assertEqual(capacity_choice([{'time':0,'complete_free_nodes':1},{'time':10,'complete_free_nodes':1}],32),'wait')
        self.assertEqual(capacity_choice([{'time':0,'complete_free_nodes':1},{'time':16,'complete_free_nodes':1}],32),'w8a8_16')

    def test_ample_capacity_keeps_bf16(self):
        self.assertEqual(capacity_choice([{'time':0,'complete_free_nodes':2}],32),'restart_same')

    def test_no_capacity_waits_without_new_service(self):
        self.assertEqual(capacity_choice([{'time':0,'complete_free_nodes':0}],32),'wait')

    def test_16_card_attestation_and_stale_hccl(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'deployment-group';run.mkdir()
            state={'role':'generator','cards':16,'expected_nodes':1,'runtime_layout':'prefixed_groups',
                   'runtime_root':str(root),'deployment_id':'deployment','model':'quant-model','precision':'W8A8_DYNAMIC',
                   'revision':'quant-r1','processor_revision':'processor-r1','request_overrides':{}}
            version={**{k:'1.0' for k in m.PACKAGES},**{k:state[k] for k in ('model','precision','revision','processor_revision')},
                     'deployment_id':run.name,'boot_id':'current-boot','serving_code_revision':'loaded-code-hash'}
            m.save(run/'versions-node0.json',version)
            m.save(run/'node0.json',{'time':time.time(),'deployment_id':run.name,'boot_id':'current-boot','hostname':'node-one'})
            for rank in range(16):
                m.save(run/f'hccl-rank-{rank:02}.json',{'correctness_passed':True,'world_size':16,'boot_id':'current-boot'})
            class Instance:
                status='INSTANCE_RUNNING';rank=0
                raw={'name':'project/service-0','node':'node-one','created_at':'1','started_at':'2','restarts':0,'term':0}
            report=m.attest(state,[Instance()])
            self.assertEqual(report['expected_boots'],['current-boot'])
            m.save(run/'hccl-rank-15.json',{'correctness_passed':True,'world_size':16,'boot_id':'previous-boot'})
            with self.assertRaises(AssertionError):
                m.attest(state,[Instance()])

    def test_quantized_quality_receipt_requires_complete_bound_baseline(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'RUNTIME',Path(directory)):
            state={k:k+'-value' for k in ('model','revision','processor_revision','precision','incarnation','runtime_revision')}
            state['expected_boots']=['boot-current']
            state['service_id']='owned-candidate'
            strata=['photo_caption','chart_qa','document_ocr','rendered_qa','multi_turn_or_multi_image']
            baseline={'required_strata':strata,'coverage':{x:1 for x in strata},'cases':[{'sample_id':x} for x in 'abcde']}
            path=Path(directory)/'BF16_BASELINE_READY.json';m.save(path,baseline)
            proof={**state,'schema_version':'corpus_pipeline.quantization_quality.v1','qualification_only':True,
                   'published_training':False,'baseline_sha256':m.hashlib.sha256(path.read_bytes()).hexdigest(),
                   'required_strata':baseline['required_strata'],'coverage':baseline['coverage'],
                   'candidate_service_id':state['service_id'],'worker_sha256':m.hashlib.sha256((m.ROOT/'quantization_quality.py').read_bytes()).hexdigest(),
                   'all_baseline_cases_completed':True,'cases':[{'sample_id':x,'passed':True} for x in 'abcde'],
                   'passed':True,'status':'passed','finished_at':time.time()}
            output=Path(directory)/f'quant-quality-{state["incarnation"]}.json';m.save(output,proof)
            self.assertTrue(m.quality_if_needed(state,Path(directory)/'state.json',state))
            proof['cases'].pop();m.save(output,proof)
            self.assertFalse(m.quality_if_needed(state,Path(directory)/'state.json',state))
            proof['cases'].append({'sample_id':'e','passed':True});proof['expected_boots']=['previous-boot'];m.save(output,proof)
            self.assertFalse(m.quality_if_needed(state,Path(directory)/'state.json',state))


if __name__=='__main__':
    unittest.main()
