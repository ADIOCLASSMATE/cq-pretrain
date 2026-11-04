"""Offline lifecycle/identity/budget tests; all platform APIs are fake."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import high_lab as lab


def payload(name, priority=6, command='fixture-command'):
    return {'name':name, 'workspace_id':'workspace', 'project_id':'project', 'model_id':'model',
        'model_version':1, 'mirror_id':'mirror', 'command':command, 'port':8000,
        'logic_compute_group_id':'group', 'task_priority':priority, 'replicas':1, 'node_num_per_replica':2,
        'resource_spec_price':{'gpu_count':16,'cpu_count':128,'memory_size_gib':1024,'gpu_type':'ASCEND_910B_64G'}}


class FakeServings:
    def __init__(self):
        self.resources={};self.creates=[];self.stops=[];self.create_error=None;self.accept_before_error=False
        self.instance_states={};self.plan_hook=None

    def add(self, body, identifier, status='PENDING'):
        raw=deepcopy(body);raw['resource_spec_price']['gpu_info']={'gpu_type':body['resource_spec_price']['gpu_type']}
        detail=SimpleNamespace(name=body['name'],ref=SimpleNamespace(key=identifier),status=status,raw=raw)
        detail.to_dict=lambda:{'project':lab.native.PROJECT,'priority':body['task_priority'],'replicas':1,
            'nodes_per_replica':2,'endpoint':'https://example.invalid'}
        self.resources[body['name']]=detail
        return detail

    def get(self, ref, **kwargs):
        if hasattr(ref,'key'):return next(r for r in self.resources.values() if r.ref.key==ref.key)
        return self.resources[ref]

    def iter(self, **kwargs):return list(self.resources.values())

    def instances(self, ref, **kwargs):
        detail=self.get(ref)
        statuses=self.instance_states.get(ref.key,['INSTANCE_STOPPED' if detail.status=='STOPPED' else 'INSTANCE_RUNNING']*2)
        return [SimpleNamespace(status=status) for status in statuses]

    def plan(self, spec):
        body=payload(spec.name,spec.priority,spec.command)
        if self.plan_hook:self.plan_hook(body)
        return SimpleNamespace(priority=spec.priority,create_kwargs=body)

    def create(self, spec, operation_id):
        self.creates.append((spec,operation_id))
        if self.create_error and not self.accept_before_error:raise self.create_error
        result=self.add(payload(spec.name,spec.priority,spec.command),'owned-high-'+str(len(self.creates)))
        if self.create_error:raise self.create_error
        return result

    def stop(self, ref, **kwargs):
        self.stops.append(ref.key);self.get(ref).status='STOPPED'
        self.instance_states[ref.key]=['INSTANCE_STOPPED']*2


class HighLabTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'lab';self.low=Path(self.tmp.name)/'low'
        self.root.mkdir();self.low.mkdir();(self.root/'profiles').mkdir();(self.root/'runtime').mkdir()
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(lab,'ROOT',self.root));self.stack.enter_context(patch.object(lab,'LOW',self.low))
        self.stack.enter_context(patch.object(lab.native,'RUNTIME',self.root))
        self.servings=FakeServings()
        class Client:
            servings=self.servings
            def __enter__(self):return self
            def __exit__(self,*_):pass
        self.client=self.stack.enter_context(patch.object(lab,'InspireClient',side_effect=lambda **kwargs:Client()))
        self.stack.enter_context(patch.object(lab,'ServingCreateSpec',side_effect=lambda **kwargs:SimpleNamespace(**kwargs)))
        self.budget=self.stack.enter_context(patch.object(lab,'observable_project_budget',return_value={
            'observable_total_with_request':32,'ceiling':256,'records':{}}))
        self.base={'schema_version':'corpus_pipeline.native_serving.v1','role':'generator','model':'Qwen3.5-397B-A17B',
            'revision':'revision-fixed','processor_revision':'processor-fixed','precision':'BF16','max_images':16,
            'api_key_file':'/fixture/no-credential','api_key_env':'INF_API_KEY','request_overrides':{},
            'name':'low-owned','service_name':'low-owned','service_id':'owned-low','payload':payload('low-owned',3),
            'tier':'low','priority':3,'cards':32,'expected_nodes':2,'expected_replicas':1}
        lab.native.save(self.low/'qwen-01.json',self.base)
        (self.low/'DISABLED').touch()
        lab.native.save(self.low/'STOPPED.json',{'complete':True,'services':[{'service_id':'owned-low'}]})
        self.servings.add(self.base['payload'],'owned-low','STOPPED')
        for name in lab.RUNTIME_FILES:(self.root/'runtime'/name).write_text('fixture '+name)
        for profile in lab.PROFILES:
            n,tokens=lab.PROFILE_LIMITS[profile]
            config={'model_name':self.base['model'],'revision':self.base['revision'],
                'processor_revision':self.base['processor_revision'],'precision':'BF16','nodes':2,'data_parallel_size':4,
                'epoch':'infra-20261024','max_num_seqs':n,'max_model_len':65536,'max_images':16,
                'max_num_batched_tokens':tokens,'cudagraph_capture_sizes':[2**i for i in range(n.bit_length())]}
            if 'k' in profile:config['enable_json_keepalive']=True
            if profile.endswith('h'):config['enable_mc2_hierarchy_comm']=True
            if profile.endswith('m'):
                config['num_speculative_tokens']=1
                config['cudagraph_capture_sizes']=[2**(i+1) for i in range(n.bit_length())]
            lab.native.save(self.root/'profiles'/(profile+'.json'),config)
        self.refresh_manifest()

    def refresh_manifest(self):
        files={str(f.relative_to(self.root)):lab.file_sha(f) for directory in ('runtime','profiles')
               for f in (self.root/directory).iterdir() if f.is_file()}
        lab.native.save(self.root/'RUNTIME_MANIFEST.json',{'files':files})

    def state(self, profile='s64'):return lab.native.load(lab.paths(profile)[0])

    def evidence(self, state, *, boot='boot-one'):
        expected={k:v for k,v in state['config_snapshot'].items() if 'key' not in k and k!='processor_files'}
        report={'runtime_config':expected,'serving_code_revision':state['binding']['serving_code_revision']}
        return {'incarnation':boot,'runtime_revision':'runtime-fixed','expected_boots':[boot],
                'reports':[deepcopy(report),deepcopy(report)]}

    def test_high32_create_is_idempotent_and_binds_manifest(self):
        result=lab.create('s64');again=lab.create('s64')
        self.assertEqual(result['service_id'],again['service_id']);self.assertEqual(len(self.servings.creates),1)
        spec,operation=self.servings.creates[0]
        self.assertEqual((spec.priority,spec.nodes_per_replica,spec.replicas,spec.quota),(6,2,1,'16,128,1024'))
        self.assertFalse(spec.auto_scaling);self.assertEqual(spec.name,operation)
        self.assertIn('CORPUS_CONFIG=',spec.command);self.assertIn('profiles/s64.json',spec.command)
        state=self.state();self.assertEqual(state['cards'],32);self.assertFalse(state['auto_recovery_enabled'])
        self.assertEqual(state['binding']['runtime_manifest_sha256'],lab.file_sha(self.root/'RUNTIME_MANIFEST.json'))
        self.assertNotIn('endpoint',state)

    def test_all_explicit_seq_token_profiles_bind_without_topology_changes(self):
        for profile,(sequences,tokens) in lab.PROFILE_LIMITS.items():
            config,_,bound=lab.binding(profile)
            self.assertEqual((config['max_num_seqs'],config['max_num_batched_tokens']),(sequences,tokens))
            self.assertEqual((config['nodes'],config['data_parallel_size']),(2,4))
            self.assertEqual(bound['profile_sha256'],lab.file_sha(self.root/'profiles'/(profile+'.json')))

    def test_production_handoff_blocks_create_under_lab_lock(self):
        (self.root/'PRODUCTION_HANDOFF.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'handed off'):lab.create('s32k')
        self.assertEqual(self.servings.creates,[])

    def test_transport_and_hierarchy_flags_are_explicit_and_boolean(self):
        original=lab.native.load(self.root/'profiles/s64kh.json')
        for key,value in [('enable_json_keepalive',False),('enable_mc2_hierarchy_comm','true')]:
            p=self.root/'profiles/s64kh.json';config=deepcopy(original);config[key]=value
            lab.native.save(p,config);self.refresh_manifest()
            with self.assertRaisesRegex(ValueError,'Unexpected'):lab.binding('s64kh')

    def test_mtp_is_a_separate_single_token_profile(self):
        p=self.root/'profiles/s64km.json';config=lab.native.load(p);config['num_speculative_tokens']=True
        lab.native.save(p,config);self.refresh_manifest()
        with self.assertRaisesRegex(ValueError,'MTP'):lab.binding('s64km')

    def test_release_receipt_is_required_to_free_reservation(self):
        lab.create('s64');lab.stop('s64',timeout=0)
        receipt=lab.paths('s64')[2]/'STOPPED.json'
        data=lab.native.load(receipt);data['complete']=False;lab.native.save(receipt,data)
        with self.assertRaisesRegex(ValueError,'release receipt'):lab.create('s32')
        self.assertEqual(len(self.servings.creates),1)

    def test_low_requires_disabled_receipt_and_fresh_terminal_instances(self):
        (self.low/'DISABLED').unlink()
        with self.assertRaisesRegex(ValueError,'disabled'):lab.create('s64')
        (self.low/'DISABLED').touch();self.servings.resources['low-owned'].status='RUNNING'
        with self.assertRaisesRegex(ValueError,'fully stopped'):lab.create('s64')
        self.servings.resources['low-owned'].status='STOPPED';self.servings.instance_states['owned-low']=['INSTANCE_UNKNOWN']
        with self.assertRaisesRegex(ValueError,'fully stopped'):lab.create('s64')
        self.assertEqual(self.servings.creates,[])

    def test_plan_time_low_restart_blocks_send(self):
        self.servings.plan_hook=lambda _body:setattr(self.servings.resources['low-owned'],'status','RUNNING')
        with self.assertRaisesRegex(ValueError,'fully stopped'):lab.create('s64')
        self.assertEqual(self.servings.creates,[])

    def test_unknown_create_no_replay_and_blocks_other_profile(self):
        self.servings.create_error=TimeoutError('fixture')
        with self.assertRaises(TimeoutError):lab.create('s64')
        with self.assertRaisesRegex(ValueError,'unresolved'):lab.create('s64')
        with self.assertRaisesRegex(ValueError,'reserves cards'):lab.create('s128')
        self.assertEqual(len(self.servings.creates),1)

    def test_lost_reply_adopts_exact_id_without_second_create(self):
        self.servings.create_error=TimeoutError('fixture');self.servings.accept_before_error=True
        with self.assertRaises(TimeoutError):lab.create('s64')
        result=lab.create('s64')
        self.assertEqual(result['service_id'],'owned-high-1');self.assertEqual(len(self.servings.creates),1)

    def test_missing_state_keeps_immutable_intent_reservation(self):
        lab.create('s64');lab.paths('s64')[0].unlink()
        with self.assertRaisesRegex(ValueError,'reserves cards'):lab.create('s32')
        self.assertEqual(lab.create('s64')['service_id'],'owned-high-1')
        self.assertEqual(len(self.servings.creates),1)

    def test_only_verified_stop_releases_and_fresh_external_restart_reserves_again(self):
        lab.create('s64');state=self.state();self.servings.resources[state['name']].status='STOPPED'
        with self.assertRaisesRegex(ValueError,'reserves cards'):lab.create('s32')
        lab.stop('s64',timeout=0)
        self.servings.resources[state['name']].status='RUNNING'
        with self.assertRaisesRegex(ValueError,'reserves cards'):lab.create('s32')
        self.servings.resources[state['name']].status='STOPPED'
        self.assertEqual(lab.create('s32')['cards'],32)

    def test_config_change_blocks_observation_and_create_but_still_allows_owned_cleanup(self):
        lab.create('s64');config=lab.native.load(self.root/'profiles/s64.json');config['max_num_batched_tokens']=8192
        lab.native.save(self.root/'profiles/s64.json',config)
        with self.assertRaisesRegex(ValueError,'Manifest file changed'):lab.create('s64')
        with patch.object(lab.native,'probe') as probe:
            self.assertFalse(lab.observe('s64',once=True)['canary_passed']);probe.assert_not_called()
        self.assertTrue(lab.stop('s64',timeout=0)['release_verified'])

    def test_current_identity_drift_prevents_stop(self):
        lab.create('s64');state=self.state();self.servings.resources[state['name']].ref.key='unowned'
        with self.assertRaisesRegex(AssertionError,'identity'):lab.stop('s64',timeout=0)
        self.assertEqual(self.servings.stops,[])

    def test_unknown_instance_status_cannot_release_stopped_service(self):
        lab.create('s64');state=self.state();self.servings.resources[state['name']].status='STOPPED'
        self.servings.instance_states[state['service_id']]=['INSTANCE_UNKNOWN']
        with self.assertRaisesRegex(RuntimeError,'not verified'):lab.stop('s64',timeout=0)
        receipt=lab.native.load(lab.paths('s64')[2]/'STOPPED.json')
        self.assertFalse(receipt['complete']);self.assertFalse(self.state()['release_verified'])

    def test_stop_only_exact_owned_id(self):
        lab.create('s64');self.servings.add(payload('unrelated'),'other-job','RUNNING')
        result=lab.stop('s64',timeout=0)
        self.assertTrue(result['release_verified']);self.assertEqual(self.servings.stops,['owned-high-1'])
        self.assertEqual(self.servings.resources['unrelated'].status,'RUNNING')

    def test_ready_requires_loaded_config_and_probe_then_expires_on_new_boot(self):
        lab.create('s64');state=self.state();self.servings.resources[state['name']].status='RUNNING'
        with patch.object(lab.native,'attest',return_value=self.evidence(state)) as attest, \
                patch.object(lab.native,'health',return_value=True),patch.object(lab.native,'probe',return_value=True) as probe:
            self.assertTrue(lab.observe('s64',once=True)['canary_passed'])
            registry=lab.native.load(lab.paths('s64')[2]/'registry.json')['endpoints'][0]
            self.assertEqual(registry['role'],'verifier');self.assertTrue(registry['ready'])
            self.assertGreater(registry['ready_expires_at'],lab.time.time())
            self.assertNotIn('endpoint',self.state())
            attest.return_value=self.evidence(state,boot='boot-two');probe.return_value=False
            self.assertFalse(lab.observe('s64',once=True)['canary_passed'])
            self.assertIsNone(self.state()['qualification_receipt'])
            bad=self.evidence(state);bad['reports'][0]['runtime_config']['max_num_seqs']=999
            attest.return_value=bad;probe.reset_mock()
            self.assertFalse(lab.observe('s64',once=True)['canary_passed']);probe.assert_not_called()

    def test_high_stopped_is_observed_without_auto_start_or_create(self):
        lab.create('s64');state=self.state();self.servings.resources[state['name']].status='STOPPED'
        with patch.object(lab.native,'probe') as probe:
            self.assertFalse(lab.observe('s64',once=True)['canary_passed']);probe.assert_not_called()
        self.assertEqual(len(self.servings.creates),1);self.assertEqual(self.servings.stops,[])


if __name__=='__main__':unittest.main()
