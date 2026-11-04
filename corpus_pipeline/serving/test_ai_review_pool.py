"""Offline pool safety tests. No SDK connection or platform mutation is made."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import ai_review_pool as pool


class FakeServing:
    def __init__(self):
        self.resources = {}; self.creates = []; self.starts = []; self.stops = []; self.plan_hook = None
        self.instance_rows = {}
        self.create_error = None; self.accept_before_error = False; self.event_rows = []

    def add(self, name, identifier, status='RUNNING'):
        detail = SimpleNamespace(name=name, ref=SimpleNamespace(key=identifier), status=status)
        detail.to_dict = lambda: {'endpoint':'https://example.invalid'}
        self.resources[name] = detail
        return detail

    def iter(self, **kwargs):
        return list(self.resources.values())

    def get(self, name, **kwargs):
        if hasattr(name, 'key'):
            return next(r for r in self.resources.values() if r.ref.key == name.key)
        return self.resources[name]

    def instances(self, ref, **kwargs):
        return self.instance_rows.get(ref.key, [])

    def events(self, ref, **kwargs):
        return SimpleNamespace(items=self.event_rows)

    def plan(self, spec):
        if self.plan_hook:
            self.plan_hook()
        value = {'name':spec.name,'quota':spec.quota,'nodes_per_replica':spec.nodes_per_replica,
                 'replicas':spec.replicas,'command':spec.command}
        return SimpleNamespace(priority=3,create_kwargs=value,to_dict=lambda:value)

    def create(self, spec, operation_id):
        self.creates.append((spec, operation_id))
        if self.create_error and not self.accept_before_error:
            raise self.create_error
        result=self.add(spec.name,'created-'+str(len(self.creates)),status='PENDING')
        if self.create_error:
            raise self.create_error
        return result

    def start(self, ref, **kwargs):
        self.starts.append(ref.key)

    def stop(self, ref, **kwargs):
        self.stops.append(ref.key)
        self.get(ref).status = 'STOPPED'
        self.instance_rows[ref.key] = [SimpleNamespace(status='INSTANCE_STOPPED')]


class FakeClient:
    def __init__(self, servings):
        self.servings=servings;self.entries=0;self.exits=0

    def __enter__(self):
        self.entries+=1;return self

    def __exit__(self,*args):
        self.exits+=1

    def close(self):
        self.exits+=1


class AiReviewPoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.stack=ExitStack();self.addCleanup(self.stack.close);self.addCleanup(self.temporary.cleanup)
        self.servings=FakeServing();self.client=FakeClient(self.servings)
        self.stack.enter_context(patch.object(pool,'ROOT',self.root))
        self.stack.enter_context(patch.object(pool,'states',lambda root=None:sorted((root or self.root).glob('qwen-[0-9][0-9].json'))))
        self.client_factory=self.stack.enter_context(patch.object(pool,'InspireClient',return_value=self.client))
        self.stack.enter_context(patch.object(pool,'ServingCreateSpec',side_effect=lambda **kwargs:SimpleNamespace(**kwargs)))
        self.stack.enter_context(patch.object(pool,'observable_project_budget',return_value={'ceiling':256,'observable_total_with_request':64}))
        self.stack.enter_context(patch.object(pool.time,'time',return_value=1000.))
        self.owned=self.stack.enter_context(patch.object(pool.native,'assert_owned',side_effect=self.assert_owned))
        self.attest=self.stack.enter_context(patch.object(pool.native,'attest',return_value={'incarnation':'boot-current'}))
        self.health=self.stack.enter_context(patch.object(pool.native,'health',return_value=True))
        self.base=self.row(1)
        pool.native.save(self.root/'qwen-01.json',self.base)
        self.servings.add(self.base['name'],self.base['service_id'])

    @staticmethod
    def assert_owned(state, detail):
        if state['service_id'] != detail.ref.key or state['name'] != detail.name:
            raise AssertionError('Wrong service identity')

    def row(self, number, **changes):
        row={'schema_version':'test-pool.v1','name':f'fixture-qwen-{number:02}',
             'service_name':f'fixture-qwen-{number:02}','service_id':f'owned-{number:02}',
             'cards':32,'expected_nodes':2,'expected_replicas':1,'priority':3,'precision':'BF16',
             'role':'generator','tier':'low','model':'fixture-model','revision':'fixture-revision',
             'processor_revision':'fixture-processor','runtime_root':'/fixtures/runtime',
             'runtime_layout':'prefixed_groups','max_images':4,'api_key_file':'/fixtures/not-read.key',
             'api_key_env':'FIXTURE_API_KEY','request_overrides':{},'auto_recovery_enabled':True,
             'payload':{},'canary_passed':True,'heartbeat_at':1000.,'incarnation':'boot-current'}
        row.update(changes);return row

    def save_base(self, **changes):
        self.base.update(changes);pool.native.save(self.root/'qwen-01.json',self.base)

    def test_proxy_bypass_preserves_existing_routes_and_proxy_settings(self):
        environ = {'NO_PROXY': 'localhost,127.0.0.1,.internal.example,10.0.0.0/8',
                   'HTTPS_PROXY': 'https://proxy.example.invalid:8080',
                   'http_proxy': 'http://proxy.example.invalid:8080',
                   'ALL_PROXY': 'socks5://proxy.example.invalid:1080',
                   'UNRELATED_SETTING': 'keep'}
        old = dict(environ)
        pool.configure_management_proxy_bypass(environ)
        self.assertEqual(environ['NO_PROXY'], environ['no_proxy'])
        self.assertEqual(environ['NO_PROXY'].split(','), old['NO_PROXY'].split(',') +
                         ['qz.sii.edu.cn', 'keycloak-inspire-prod.sii.edu.cn'])
        self.assertEqual({k: v for k, v in environ.items() if k not in ('NO_PROXY', 'no_proxy')},
                         {k: v for k, v in old.items() if k != 'NO_PROXY'})

    def test_proxy_bypass_merges_both_spellings_without_duplicate_hosts(self):
        environ = {'NO_PROXY': ' localhost, QZ.SII.EDU.CN, , .internal.example ',
                   'no_proxy': 'localhost,lower-only.example,qz.sii.edu.cn'}
        pool.configure_management_proxy_bypass(environ)
        self.assertEqual(environ['NO_PROXY'], environ['no_proxy'])
        self.assertEqual(environ['NO_PROXY'].split(','),
                         ['localhost', 'QZ.SII.EDU.CN', '.internal.example',
                          'lower-only.example', 'keycloak-inspire-prod.sii.edu.cn'])
        once = dict(environ)
        pool.configure_management_proxy_bypass(environ)
        self.assertEqual(environ, once)

    def test_proxy_bypass_empty_environment_adds_only_verified_domains(self):
        environ = {}
        pool.configure_management_proxy_bypass(environ)
        expected = 'qz.sii.edu.cn,keycloak-inspire-prod.sii.edu.cn'
        self.assertEqual(environ, {'NO_PROXY': expected, 'no_proxy': expected})

    def test_every_cli_action_configures_bypass_before_dispatch(self):
        for action in ('provision', 'monitor', 'guard', 'stop'):
            with self.subTest(action=action), patch.dict(pool.os.environ, {'no_proxy': 'keep.example'}, clear=True), \
                    patch.object(sys, 'argv', ['ai_review_pool', action]), \
                    patch.object(pool, action) as handler:
                def dispatched(*args):
                    self.assertEqual(pool.os.environ['NO_PROXY'], pool.os.environ['no_proxy'])
                    self.assertEqual(set(pool.os.environ['no_proxy'].split(',')),
                                     {'keep.example', 'qz.sii.edu.cn', 'keycloak-inspire-prod.sii.edu.cn'})
                handler.side_effect = dispatched
                pool.main()
                handler.assert_called_once_with(*((4,) if action == 'provision' else ()))

    def test_pool_cap_counts_pending_unknown_and_stopped_reservations(self):
        rows=[self.row(i,phase=phase,platform_status=status,service_id=None if i==2 else f'owned-{i}')
              for i,phase,status in [(1,'OBSERVING','RUNNING'),(2,'CREATE_INTENT','UNKNOWN'),
                                      (3,'OBSERVING','PENDING'),(4,'OBSERVING','STOPPED')]]
        self.assertEqual(pool.validate_budget(rows),128)
        with self.assertRaisesRegex(ValueError,'128-card'):
            pool.validate_budget(rows,32)
        self.assertEqual(pool.validate_budget(rows[:2]),64)

    def test_duplicate_names_and_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate service name'):
            pool.validate_budget([self.row(1),self.row(2,name=self.row(1)['name'])])
        with self.assertRaisesRegex(ValueError,'Duplicate service identity'):
            pool.validate_budget([self.row(1),self.row(2,service_id=self.row(1)['service_id'])])

    def test_precision_priority_or_topology_change_is_rejected(self):
        for changes in ({'precision':'W8A8'},{'priority':6},{'cards':16},{'expected_nodes':1},{'expected_replicas':2}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError,'Unqualified'):
                pool.validate_budget([self.row(1,**changes)])

    def test_disabled_pool_does_not_connect_or_create(self):
        (self.root/'DISABLED').touch()
        with self.assertRaisesRegex(RuntimeError,'disabled'):
            pool.provision(4)
        self.client_factory.assert_not_called();self.assertEqual(self.servings.creates,[])

    def test_disable_during_plan_preserves_reservation_without_sending(self):
        self.servings.plan_hook=lambda:(self.root/'DISABLED').touch()
        with self.assertRaisesRegex(RuntimeError,'disabled'):
            pool.provision(2)
        self.assertEqual(self.servings.creates,[])
        state=pool.native.load(self.root/'qwen-02.json')
        self.assertEqual(state['phase'],'CREATE_INTENT')
        self.assertNotIn('service_id',state)
        self.assertEqual(pool.validate_budget([pool.native.load(p) for p in pool.states()]),64)

    def test_no_current_qualification_cannot_expand(self):
        for changes in ({'canary_passed':False},{'heartbeat_at':969.},{'heartbeat_at':1001.}):
            with self.subTest(changes=changes):
                base=self.row(1,**changes);pool.native.save(self.root/'qwen-01.json',base)
                with self.assertRaisesRegex(RuntimeError,'qualification'):
                    pool.provision(2)
        self.assertEqual(self.servings.creates,[])

    def test_changed_boot_unhealthy_or_nonrunning_base_cannot_expand(self):
        self.attest.return_value={'incarnation':'new-boot'}
        with self.assertRaisesRegex(RuntimeError,'expired'):
            pool.provision(2)
        self.attest.return_value={'incarnation':'boot-current'};self.health.return_value=False
        with self.assertRaisesRegex(RuntimeError,'expired'):
            pool.provision(2)
        self.health.return_value=True;self.servings.resources[self.base['name']].status='PENDING'
        with self.assertRaisesRegex(RuntimeError,'expired'):
            pool.provision(2)
        self.assertEqual(self.servings.creates,[])

    def test_expansion_stays_bf16_and_repeated_provision_never_creates_twice(self):
        pool.provision(4);pool.provision(4)
        self.assertEqual(len(self.servings.creates),3)
        self.assertEqual(pool.validate_budget([pool.native.load(p) for p in pool.states()]),128)
        for spec, operation in self.servings.creates:
            self.assertEqual(spec.quota,'16,128,1024');self.assertEqual(spec.nodes_per_replica,2)
            self.assertEqual(spec.replicas,1);self.assertEqual(spec.priority,3)
            self.assertIn('bf16',spec.model);self.assertNotIn('w8',spec.command.lower())
            self.assertEqual(operation,spec.name)
        for path in pool.states():
            self.assertEqual(pool.native.load(path)['precision'],'BF16')

    def test_unknown_create_is_not_retransmitted_when_no_matching_object(self):
        self.servings.create_error=TimeoutError('bounded simulated transport failure')
        with self.assertRaises(TimeoutError):pool.provision(2)
        with self.assertRaisesRegex(RuntimeError,'unresolved'):pool.provision(2)
        self.assertEqual(len(self.servings.creates),1)
        self.assertEqual(pool.validate_budget([pool.native.load(p) for p in pool.states()]),64)

    def test_accepted_but_lost_create_reply_reconciles_exact_identity_without_resend(self):
        self.servings.create_error=TimeoutError('lost reply');self.servings.accept_before_error=True
        with self.assertRaises(TimeoutError):pool.provision(2)
        pool.provision(2)
        self.assertEqual(len(self.servings.creates),1)
        self.assertEqual(pool.native.load(self.root/'qwen-02.json')['service_id'],'created-1')

    def test_unresolved_observer_keeps_polling_and_reservation(self):
        state=self.row(2,service_id=None,phase='CREATE_INTENT',canary_passed=False)
        path=self.root/'qwen-02.json';pool.native.save(path,state)
        sleeps=[]
        def sleep(_):
            sleeps.append(1)
            if len(sleeps)==2:(self.root/'DISABLED').touch()
        with patch.object(pool.time,'sleep',side_effect=sleep),patch.object(pool,'publish') as publish:
            pool.worker(path)
        self.assertEqual(len(sleeps),2)
        saved=pool.native.load(path)
        self.assertFalse(saved['canary_passed']);self.assertIsNone(saved.get('service_id'))
        self.assertEqual(saved['observation_error']['type'],'RuntimeError')
        self.assertEqual(self.servings.creates,[])
        self.assertEqual(pool.validate_budget([pool.native.load(p) for p in pool.states()]),64)

    def test_same_preemption_event_starts_only_owned_id_once(self):
        self.save_base(platform_status='RUNNING',was_running=True,last_running_at=998.)
        self.servings.resources[self.base['name']].status='STOPPED'
        self.servings.event_rows=[{'reason':'Preempted','message':'This pod was preempted by higher priority',
                                  'created_at':'1000000','object_id':'workspace/'+self.base['service_id']+'-0'}]
        self.attest.side_effect=ValueError('No running instance')
        sleeps=[]
        def sleep(_):
            sleeps.append(1)
            if len(sleeps)==2:(self.root/'DISABLED').touch()
        with patch.object(pool.time,'sleep',side_effect=sleep),patch.object(pool,'publish'):
            pool.worker(self.root/'qwen-01.json')
        self.assertEqual(self.servings.starts,[self.base['service_id']])
        self.assertEqual(self.servings.creates,[])
        self.assertEqual(pool.native.load(self.root/'qwen-01.json')['precision'],'BF16')

    def test_authentication_failure_reopens_client_without_restart(self):
        class AuthenticationError(Exception):pass
        replacement=FakeClient(self.servings)
        self.client_factory.side_effect=[self.client,replacement]
        original_get=self.servings.get;calls=[]
        def get(*args,**kwargs):
            calls.append(1)
            if len(calls)==1:raise AuthenticationError('simulated expired auth')
            return original_get(*args,**kwargs)
        self.attest.return_value={'incarnation':'boot-current','runtime_revision':'fixture-runtime','expected_boots':['fixture-boot']}
        sleeps=[]
        def sleep(_):
            sleeps.append(1)
            if len(sleeps)==2:(self.root/'DISABLED').touch()
        with patch.object(self.servings,'get',side_effect=get),patch.object(pool.time,'sleep',side_effect=sleep),\
                patch.object(pool.native,'probe',return_value=True),patch.object(pool,'publish'):
            pool.worker(self.root/'qwen-01.json')
        self.assertEqual(self.client_factory.call_count,2)
        self.assertGreaterEqual(self.client.exits,1);self.assertGreaterEqual(replacement.exits,1)
        saved=pool.native.load(self.root/'qwen-01.json')
        self.assertIsNone(saved['observation_error']);self.assertTrue(saved['canary_passed'])
        self.assertEqual(self.servings.starts,[]);self.assertEqual(self.servings.creates,[])

    def test_stop_only_exact_pool_owned_ids_and_verify_release(self):
        second = self.row(2)
        pool.native.save(self.root / 'qwen-02.json', second)
        self.servings.add(second['name'], second['service_id'])
        foreign = self.servings.add('fixture-qwen-03', 'another-experiment')
        with patch.object(pool, 'publish') as publish:
            pool.stop()
        self.assertEqual(self.servings.stops, ['owned-01', 'owned-02'])
        self.assertEqual(foreign.status, 'RUNNING')
        self.assertTrue((self.root / 'DISABLED').exists())
        self.assertEqual(publish.call_count, 2)
        self.assertTrue(all(call.args[2] is False for call in publish.call_args_list))
        receipt = pool.native.load(self.root / 'STOPPED.json')
        self.assertTrue(receipt['complete'])
        self.assertEqual({r['service_id'] for r in receipt['services']}, {'owned-01', 'owned-02'})
        self.assertEqual(self.servings.creates, [])
        self.assertEqual(self.servings.starts, [])

    def test_stop_rejects_current_identity_drift_before_mutation(self):
        self.servings.resources[self.base['name']].ref.key = 'replacement-not-owned'
        with patch.object(pool, 'publish') as publish, self.assertRaisesRegex(AssertionError, 'identity'):
            pool.stop()
        self.assertTrue((self.root / 'DISABLED').exists())
        self.assertEqual(self.servings.stops, [])
        publish.assert_not_called()
        self.assertFalse((self.root / 'STOPPED.json').exists())
        self.assertEqual(self.servings.resources[self.base['name']].status, 'RUNNING')

    def test_stopped_service_with_unknown_instance_cannot_claim_release(self):
        self.servings.resources[self.base['name']].status = 'STOPPED'
        self.servings.instance_rows[self.base['service_id']] = [SimpleNamespace(status='INSTANCE_UNKNOWN')]
        with patch.object(pool, 'publish'), patch.object(pool.time, 'monotonic', side_effect=[0., 301.]), \
                patch.object(pool.time, 'sleep') as sleep, \
                self.assertRaisesRegex(RuntimeError, 'release not yet verified'):
            pool.stop()
        receipt = pool.native.load(self.root / 'STOPPED.json')
        self.assertFalse(receipt['complete'])
        self.assertEqual(receipt['services'][0]['instance_statuses'], ['INSTANCE_UNKNOWN'])
        self.assertEqual(self.servings.stops, [])
        sleep.assert_not_called()
        self.assertEqual(pool.validate_budget([pool.native.load(p) for p in pool.states()]), 32)

    def test_recovery_rechecks_disable_after_observation(self):
        self.servings.resources[self.base['name']].status = 'STOPPED'
        self.attest.side_effect = ValueError('No running instance')
        def decision(*args):
            (self.root / 'DISABLED').touch()
            return 'preemption-after-observation'
        with patch.object(pool.native, 'recovery_decision', side_effect=decision), \
                patch.object(pool.time, 'sleep'), patch.object(pool, 'publish'):
            pool.worker(self.root / 'qwen-01.json')
        self.assertEqual(self.servings.starts, [])
        state = pool.native.load(self.root / 'qwen-01.json')
        self.assertFalse(state['canary_passed'])
        self.assertNotIn('restart_event', state)

    def test_recovery_rechecks_disable_after_budget_before_start(self):
        self.servings.resources[self.base['name']].status = 'STOPPED'
        self.attest.side_effect = ValueError('No running instance')
        def budget(*args):
            (self.root / 'DISABLED').touch()
            return {'ceiling': 256}
        with patch.object(pool.native, 'recovery_decision', return_value='preemption'), \
                patch.object(pool, 'observable_project_budget', side_effect=budget), \
                patch.object(pool.time, 'sleep'), patch.object(pool, 'publish'):
            pool.worker(self.root / 'qwen-01.json')
        self.assertEqual(self.servings.starts, [])
        self.assertNotIn('restart_event', pool.native.load(self.root / 'qwen-01.json'))

    def test_stop_waits_for_inflight_restart_with_cross_process_lock(self):
        self.servings.resources[self.base['name']].status = 'STOPPED'
        self.attest.side_effect = ValueError('No running instance')
        entered = threading.Event(); release = threading.Event(); waiting = threading.Event()
        order = []; errors = []; lock_probe = []
        original_flock = pool.fcntl.flock
        original_start = self.servings.start; original_stop = self.servings.stop
        def start(ref, **kwargs):
            # A separate process must see the lifecycle lock held during Start.
            probe = subprocess.run([sys.executable, '-c',
                'import fcntl,sys\nf=open(sys.argv[1],"a")\n'
                'try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
                'except BlockingIOError: sys.exit(73)\n',
                str(self.root / 'lifecycle.lock')], capture_output=True, timeout=5)
            lock_probe.append(probe.returncode)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release simulated Start')
            original_start(ref, **kwargs)
            self.servings.get(ref).status = 'RUNNING'
            order.append('start_finished')
        def stop(ref, **kwargs):
            order.append('stop_sent')
            original_stop(ref, **kwargs)
        def flock(handle, operation):
            if threading.current_thread().name == 'test-pool-stop' and Path(handle.name).name == 'lifecycle.lock':
                waiting.set()
            return original_flock(handle, operation)
        def invoke(function, *args):
            try:
                function(*args)
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=invoke, args=(pool.worker, self.root / 'qwen-01.json'), daemon=True)
        stopper = threading.Thread(target=invoke, args=(pool.stop,), name='test-pool-stop', daemon=True)
        with patch.object(pool.native, 'recovery_decision', return_value='preemption'), \
                patch.object(self.servings, 'start', side_effect=start), \
                patch.object(self.servings, 'stop', side_effect=stop), \
                patch.object(pool.fcntl, 'flock', side_effect=flock), \
                patch.object(pool.time, 'sleep'), patch.object(pool, 'publish'):
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(lock_probe, [73])
                stopper.start()
                self.assertTrue(waiting.wait(5))
                self.assertTrue((self.root / 'DISABLED').exists())
                self.assertEqual(self.servings.stops, [])
                self.assertFalse((self.root / 'STOPPED.json').exists())
            finally:
                release.set()
                worker.join(5)
                if stopper.ident is not None:
                    stopper.join(5)
        self.assertFalse(worker.is_alive()); self.assertFalse(stopper.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(order, ['start_finished', 'stop_sent'])
        self.assertEqual(self.servings.starts, ['owned-01'])
        self.assertEqual(self.servings.stops, ['owned-01'])
        self.assertTrue(pool.native.load(self.root / 'STOPPED.json')['complete'])
        self.assertEqual(self.servings.resources[self.base['name']].status, 'STOPPED')


if __name__=='__main__':
    unittest.main()
