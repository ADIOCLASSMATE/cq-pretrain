"""Safety-critical recovery transitions; all platform operations are fakes."""
import copy
import json
from pathlib import Path
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest import mock

import supervisor as module
import watchdog


def fixture():
    payload = {'name': 'corpus-test-r01', 'command': 'bash test.sh', 'workspace_id': 'ws',
        'project_id': 'pr', 'logic_compute_group_id': 'group', 'framework': 'pytorch',
        'task_priority': 3, 'exclude_nodes': [], 'envs': [{'name': 'CORPUS_DEPLOYMENT_ID', 'value': 'test-r01'}],
        'framework_config': [{'image': 'image:fixed', 'image_type': 'SOURCE_PRIVATE', 'instance_count': 2,
            'gpu_count': 16, 'cpu': 128, 'mem_gi': 1024, 'shm_gi': 128}]}
    row = copy.deepcopy(payload)
    row.update(job_id='owned-job', status='job_queuing', priority_name='3', priority_level='LOW')
    state = {'job_id': 'owned-job', 'owned_jobs': ['owned-job'], 'payload': payload,
             'seen_running': True, 'confirmed_preemption': True}
    return state, row


class RecoverySafety(unittest.TestCase):
    def test_http_timeout_is_never_preemption(self):
        self.assertEqual(module.decision('RUNNING', True, False, False), 'observe')
        self.assertEqual(module.decision('UNKNOWN', True, False, False), 'observe')

    def test_queue_before_first_run_never_restarts(self):
        self.assertEqual(module.decision('QUEUING', False, False), 'observe')
        self.assertEqual(module.decision('QUEUING', False, True), 'observe')

    def test_confirmed_preemption_after_run_recovery(self):
        self.assertEqual(module.decision('QUEUING', True, True), 'recover')
        self.assertEqual(module.decision('FAILED', True, True), 'recover')

    def test_running_overrides_old_preemption_event(self):
        self.assertEqual(module.decision('RUNNING', True, True, False), 'observe')

    def test_unexplained_exit_needs_diagnosis(self):
        self.assertEqual(module.decision('FAILED', True, False), 'diagnose')
        self.assertEqual(module.decision('CANCELLED', True, False), 'diagnose')

    def test_no_ownership_no_control(self):
        state, row = fixture()
        state['owned_jobs'] = ['other-job']
        with self.assertRaises(ValueError):
            module.assert_owned(state, {'owned-job': row})

    def test_no_overlapping_generation(self):
        state, row = fixture()
        rows = {'owned-job': row, 'old-job': {'status': 'job_running'}}
        with self.assertRaises(ValueError):
            module.assert_owned(state, rows)

    def test_high_priority_refused(self):
        state, row = fixture()
        row['priority_level'] = 'HIGH'
        with self.assertRaises(ValueError):
            module.assert_owned(state, {'owned-job': row})

    def test_live_image_normalization_preserves_image_check(self):
        state, row = fixture()
        row['framework_config'][0]['image_type'] = 'SOURCE_PUBLIC'
        row['envs'] = []
        module.validate_payload(state['payload'], row)
        row['framework_config'][0]['image'] = 'other-image'
        with self.assertRaises(ValueError):
            module.validate_payload(state['payload'], row)

    def test_command_or_resource_change_refused(self):
        for field, value in [('command', 'other-command'), ('project_id', 'other-project')]:
            state, row = fixture()
            row[field] = value
            with self.assertRaises(ValueError):
                module.validate_payload(state['payload'], row)
        state, row = fixture()
        row['framework_config'][0]['gpu_count'] = 8
        with self.assertRaises(ValueError):
            module.validate_payload(state['payload'], row)

    def test_stale_heartbeat_not_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'test-r01'
            path.mkdir()
            (path/'node0.json').write_text(json.dumps({'deployment_id':'test-r01','time':0,'ip':'10.1.2.3'}))
            self.assertIsNone(module.fresh_gateway({'heartbeat_root':temporary,'deployment_id':'test-r01'}))

    def test_no_free_nodes_never_stops_queue(self):
        state, row = fixture()
        state['original_payload_path'] = 'unused'
        api = mock.Mock()
        api.nodes.return_value = []
        with mock.patch.object(module, 'choose_profile', return_value=None):
            module.recover(api, state, Path('unused'), {'owned-job':row})
        api.stop.assert_not_called()
        api.create.assert_not_called()
        self.assertEqual(state['phase'], 'WAITING_IDLE_NODES_AFTER_PREEMPTION')

    def test_precision_ladder_selects_16_when_only_one_node_fits(self):
        state, row = fixture()
        state.update(role='generator',budget_cards=32)
        full=state['payload'];small=copy.deepcopy(full);small['framework_config'][0]['instance_count']=1
        profiles=[{'profile_id':'bf16','cards':32,'payload_path':'full'}, {'profile_id':'w8a8','cards':16,'payload_path':'small'}]
        def node(name, occupied=False):
            return {'name':name,'status':'Ready','cordon_type':'','resource_pool':'online',
              'tasks_associated':{'count':int(occupied),'tasks':[{'name':'high-priority'}] if occupied else []},
              'gpu':{'available':16,'total':16,'used':0},'cpu':{'available':128,'total':128,'used':0},
              'memory':{'available':1024,'total':1024,'used':0}}
        with mock.patch.object(module,'profiles_for',return_value=profiles), mock.patch.object(module,'load',side_effect=lambda p:full if p=='full' else small), mock.patch.object(module,'apply_fixed_exclusions',side_effect=lambda p,a:p):
            result=module.choose_profile(state,[node('idle'),node('occupied',True)])
            self.assertEqual(result[0]['profile_id'],'w8a8')
            self.assertEqual(result[1],['idle'])
            self.assertIn('occupied',result[2]['exclude_nodes'])
            result=module.choose_profile(state,[node('idle1'),node('idle2')])
            self.assertEqual(result[0]['profile_id'],'bf16')

    def test_observation_timeout_does_not_create_or_stop(self):
        state,row=fixture();state.update(disabled=False,phase='OBSERVING')
        api=mock.Mock();api.records.side_effect=TimeoutError('live API timeout')
        with mock.patch.object(module,'load',return_value=state), mock.patch.object(module,'profiles_for',return_value=[]):
            with self.assertRaises(TimeoutError):module.tick(api,Path('unused'))
        api.create.assert_not_called();api.stop.assert_not_called()

    def test_aggressor_preemption_event_is_not_victim(self):
        self.assertIsNone(module.preemption_kind({'reason':'Preempted','message':'This job will preempt another job'}))
        self.assertEqual(module.preemption_kind({'reason':'Evicted','message':'Pod is preempted by high priority job'}),'preemption_observed')
        self.assertEqual(module.preemption_kind({'reason':'Killing','message':'Stopping container'}),'termination_requested')

    def test_watchdog_refuses_reused_pid(self):
        receipt={'pid':123,'starttime_ticks':1}
        with mock.patch.object(watchdog,'process_identity',return_value={'pid':123,'starttime_ticks':2,'command':[x.encode() for x in watchdog.COMMAND]}):
            self.assertFalse(watchdog.is_same_process(receipt))

    def test_new_revision_not_ready_before_new_image_probe(self):
        state,row=fixture();state.update(role='generator',model='Qwen-W8',revision='quant-revision',precision='W8A8',max_images=16,deployment_id='w8-r02')
        result=module.endpoint(state,'http://10.1.2.3:8000/v1',False)
        self.assertFalse(result['ready']);self.assertEqual(result['revision'],'quant-revision')
        with tempfile.TemporaryDirectory() as temporary:
            registry=Path(temporary)/'endpoints.json'
            with mock.patch.object(module,'REGISTRY',registry):
                module.publish('generator',result)
                self.assertFalse(json.loads(registry.read_text())['endpoints'][0]['ready'])
                module.publish('generator',{**result,'ready':True})
                final=json.loads(registry.read_text());self.assertEqual(final['registry_generation'],2)
                self.assertTrue(final['endpoints'][0]['ready'])

    def test_probe_abnormal_exit_finishes_receipt_and_bounded_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'runtime/probes/test-r01-attempt01';output.mkdir(parents=True)
            (output/'result.json').write_text(json.dumps({'passed':False,'requests':[]}))
            (output/'process.json').write_text(json.dumps({'pid':123,'starttime_ticks':7,'command':['probe'],'created_at':time.time()}))
            state={'deployment_id':'test-r01','revision':'rev'}
            child=mock.Mock();child.poll.return_value=-9
            with mock.patch.object(module,'ROOT',root), mock.patch.object(module,'PROBES',{'test-r01-attempt01':child}), mock.patch.object(module,'process_identity',return_value=None), mock.patch.object(module.subprocess,'Popen') as spawn:
                self.assertFalse(module.probe_if_needed(state,'http://10.1.2.3:8000/v1'))
                report=json.loads((output/'result.json').read_text())
                self.assertEqual(report['error_type'],'ProbeProcessExited');self.assertIn('finished_at',report)
                report['finished_at']=time.time()-60;(output/'result.json').write_text(json.dumps(report))
                self.assertFalse(module.probe_if_needed(state,'http://10.1.2.3:8000/v1'))
                self.assertEqual(state['probe_attempt'],2);spawn.assert_not_called()

    def test_probe_wallclock_bound_checks_exact_identity(self):
        receipt={'pid':123,'starttime_ticks':7,'command':['probe'],'created_at':time.time()-2000}
        same={'pid':123,'starttime_ticks':7,'command':[b'probe']}
        with mock.patch.object(module,'process_identity',return_value=same):
            self.assertEqual(module.inspect_probe_process(receipt),('ProbeWallClockTimeout',True))
        with mock.patch.object(module,'process_identity',return_value={**same,'starttime_ticks':8}):
            self.assertEqual(module.inspect_probe_process(receipt),('ProbeProcessExited',False))

    def test_runtime_attestation_requires_this_deployment_and_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            state={'deployment_id':'run1','model':'Qwen','precision':'BF16','revision':'rev','processor_revision':'processor-hash',
                   'heartbeat_root':temporary,'payload':{'framework_config':[{'instance_count':1}]}}
            self.assertFalse(module.attest_runtime(state))
            directory=Path(temporary)/'run1';directory.mkdir()
            report={k:'1.0' for k in ('vllm','vllm-ascend','torch','torch-npu','transformers')}
            report.update({k:state[k] for k in ('deployment_id','model','precision','revision','processor_revision')})
            path=directory/'versions-node0.json';path.write_text(json.dumps(report))
            self.assertTrue(module.attest_runtime(state))
            report['model']='wrong-model';path.write_text(json.dumps(report))
            self.assertFalse(module.attest_runtime(state))

    def test_clients_stay_in_their_creation_threads(self):
        barrier=threading.Barrier(2)
        def factory(*args,**kwargs):
            return {'owner_thread':threading.get_ident()}
        def worker():
            first=module.api_for_current_thread()
            barrier.wait(timeout=5)
            second=module.api_for_current_thread()
            self.assertIs(first,second)
            self.assertEqual(second['owner_thread'],threading.get_ident())
            return id(first)
        with mock.patch.object(module,'APIS',{}), mock.patch.object(module,'API',side_effect=factory) as create:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures=[executor.submit(worker) for _ in range(2)]
                self.assertEqual(len({f.result() for f in futures}),2)
            self.assertEqual(create.call_count,2)

    def test_parallel_registry_publish_preserves_both_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry=Path(temporary)/'endpoints.json'
            with mock.patch.object(module,'REGISTRY',registry):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures=[executor.submit(module.publish,role,{'role':role,'ready':False,'revision':str(index)})
                             for index in range(10) for role in ('generator','verifier')]
                    for future in futures:future.result()
            value=json.loads(registry.read_text())
            self.assertEqual({v['role'] for v in value['endpoints']},{'generator','verifier'})
            self.assertEqual(value['registry_generation'],20)

    def test_initial_downsize_requires_two_live_capacity_observations(self):
        state={'allow_initial_capacity_qualification':True,'seen_running':False,
               'platform_status':'QUEUING','profile_id':'bf16'}
        api=mock.Mock();api.nodes.return_value=[]
        with mock.patch.object(module,'choose_profile',return_value=({'profile_id':'w8a8'},['idle'],{})), mock.patch.object(module.time,'time',side_effect=lambda:clock[0]):
            clock=[100.0]
            self.assertFalse(module.consider_initial_qualification(api,state,[{'node':''}]))
            clock[0]=114.0
            self.assertFalse(module.consider_initial_qualification(api,state,[{'node':''}]))
            clock[0]=116.0
            self.assertTrue(module.consider_initial_qualification(api,state,[{'node':''}]))
        self.assertEqual(api.nodes.call_count,2)
        api.create.assert_not_called();api.stop.assert_not_called()

    def test_initial_downsize_does_not_interrupt_partially_allocated_job(self):
        state={'allow_initial_capacity_qualification':True,'seen_running':False,
               'platform_status':'QUEUING','profile_id':'bf16'}
        api=mock.Mock()
        self.assertFalse(module.consider_initial_qualification(api,state,[{'node':'allocated-node'}]))
        api.nodes.assert_not_called();api.stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
