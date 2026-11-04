from pathlib import Path
from contextlib import nullcontext
import tempfile
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
from .sscd_npu_fleet import target_from_progress,PlacementRecoveringController,ready_to_finalize,Fleet
from . import sscd_npu_fleet as f
from .sscd_npu_pool import complete,qualified,sha,await_allocation_lock,exit_code
from . import sscd_npu_pool as pool
from corpus_pipeline.resource_manager.engine import atomic_json,Controller
from corpus_pipeline.resource_manager.models import WorkloadSpec


class PolicyTests(unittest.TestCase):
    def test_inherited_lock_busy_waits_then_recovers_instead_of_false_success(self):
        sleeps=[]
        with patch.object(pool.fcntl,'flock',side_effect=[BlockingIOError(),BlockingIOError(),None]):
            self.assertTrue(await_allocation_lock(object(),lambda:False,sleeps.append))
        self.assertEqual(sleeps,[2,2])
        with patch.object(pool.fcntl,'flock',side_effect=AssertionError('stopped waiter must not lock')):
            self.assertFalse(await_allocation_lock(object(),lambda:True,sleeps.append))
        self.assertEqual(exit_code(finished=False),75)
        self.assertEqual(exit_code(finished=True),0)
        self.assertEqual(exit_code(finished=False,fatal=True),2)
    def test_scale_only_from_recent_five_minute_business_progress_with_bounded_step(self):
        previous={'observed_at':1000,'total_pairs':1000000,'completed_pairs':0}
        current={'updated_at':1300,'total_pairs':1000000,'completed_pairs':300}
        kw=dict(now=1300,deadline=2000,target=32,maximum=128)
        value=target_from_progress([previous],current,**kw)
        self.assertEqual(value['target_cards'],48);self.assertEqual(value['pairs_per_second'],1)
        self.assertEqual(target_from_progress([],current,**kw)['target_cards'],32)
        self.assertEqual(target_from_progress([previous],current,last_change=1290,**kw)['target_cards'],32)
        current['updated_at']=1000
        self.assertEqual(target_from_progress([previous],current,**kw)['target_cards'],32)

    def test_no_progress_does_not_provoke_capacity_loop_and_maximum_respected(self):
        previous={'observed_at':1000,'total_pairs':1000000,'completed_pairs':1}
        current={'updated_at':1300,'total_pairs':1000000,'completed_pairs':1}
        kw=dict(now=1300,deadline=2000,target=125,maximum=128)
        self.assertEqual(target_from_progress([previous],current,**kw)['target_cards'],125)
        current['completed_pairs']=301
        self.assertEqual(target_from_progress([previous],current,**kw)['target_cards'],128)
        current['completed_pairs']=1000000
        self.assertEqual(target_from_progress([previous],current,**kw)['reason'],'all_pairs_processed')

    def test_gate_requires_full_calibration_and_completion_requires_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            atomic_json(root/'calibration'/'COMPLETE.json',{'complete':True,'passed':True,'qualified_for_full_run':False})
            self.assertFalse(qualified(root))
            atomic_json(root/'PLAN.json',{'plan':'test'})
            atomic_json(root/'calibration'/'COMPLETE.json',{'complete':True,'passed':True,'qualified_for_full_run':True,'scope':'full_run','plan_sha256':sha(root/'PLAN.json')})
            self.assertTrue(qualified(root))
            atomic_json(root/'COMPLETE.json',{'complete':True,'all_pairs_covered':True,'completed_pairs':1,'total_pairs':2})
            self.assertFalse(complete(root))
            atomic_json(root/'COMPLETE.json',{'complete':True,'all_pairs_covered':True,'completed_pairs':2,'total_pairs':2,
                                            'plan_sha256':sha(root/'PLAN.json'),'calibration_sha256':sha(root/'calibration'/'COMPLETE.json')})
            self.assertTrue(complete(root))
            atomic_json(root/'PLAN.json',{'plan':'changed'})
            self.assertFalse(qualified(root));self.assertFalse(complete(root))

    def test_finalization_requires_all_tasks_and_pairs_not_just_idle_workers(self):
        self.assertTrue(ready_to_finalize({'total_tasks':10,'completed_tasks':10,'total_pairs':100,'completed_pairs':100}))
        for v in ({'total_tasks':10,'completed_tasks':10,'total_pairs':100,'completed_pairs':99},
                  {'total_tasks':10,'completed_tasks':9,'total_pairs':100,'completed_pairs':100},{}):
            self.assertFalse(ready_to_finalize(v))

    def test_pending_recovery_only_stops_confirmed_own_resource_once(self):
        obj=object.__new__(PlacementRecoveringController);obj.pending_timeout=300;obj.clock=lambda:1000
        row={'state':'PENDING','resource_id':'owned','accepted_at':0}
        with patch.object(Controller,'_observe'),patch.object(obj,'_register_replacement') as replacement,patch.object(obj,'_stop_row') as stop:
            obj._observe(row);self.assertEqual(row['retirement_reason'],'placement_wait_timeout');stop.assert_called_once_with(row);replacement.assert_called_once_with(row)
            row['stop']={'phase':'sent'};obj._observe(row);self.assertEqual(stop.call_count,1)
            other={'state':'PENDING','accepted_at':0};obj._observe(other);self.assertEqual(stop.call_count,1)


class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name);runtime=root/'runtime';out=root/'out'
        (runtime/'corpus_pipeline/postprocess').mkdir(parents=True)
        (runtime/'corpus_pipeline/postprocess/sscd_exact_npu.py').write_text('# frozen\n')
        atomic_json(out/'PLAN.json',{'plan':1});atomic_json(out/'COMPLETE.json',{'test':1})
        self.fleet=object.__new__(Fleet);self.fleet.root=root/'fleet';self.fleet.output=out;self.fleet.policy={}
        self.fleet.config_sha256='a'*64;self.fleet.state={};self.fleet.state_path=self.fleet.root/'FLEET.json'
        self.fleet.c={'adapter_config':{'repository':str(runtime),'remote_python':'/python'}}
    def tearDown(self):self.tmp.cleanup()

    def test_launch_is_durable_then_running_restart_does_not_duplicate_and_success_is_checked(self):
        with patch.object(f.command_runner,'status',return_value={'launch_operation_id':None}),\
             patch.object(f.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as launch:
            self.assertFalse(self.fleet.finalize_tick());launch.assert_called_once()
        operation=self.fleet.state['finalizer']['operation_id']
        self.fleet.state=f.read(self.fleet.state_path)
        with patch.object(f.command_runner,'status',return_value={'launch_operation_id':operation,'running':True}),\
             patch.object(f.subprocess,'run',side_effect=AssertionError('must not duplicate')):
            self.assertFalse(self.fleet.finalize_tick())
        with patch.object(f.command_runner,'status',return_value={'launch_operation_id':operation,'completed':True,'exit_code':0}),\
             patch.object(f,'complete',return_value=True):
            self.assertTrue(self.fleet.finalize_tick())
        self.assertTrue((self.fleet.output/'fleet/FINALIZER_SUCCESS.json').exists())

    def test_failed_or_lost_finalizer_does_not_blindly_restart(self):
        self.fleet.state['finalizer']={'operation_id':'once'}
        for health in ({'launch_operation_id':'once','completed':True,'exit_code':1},
                       {'launch_operation_id':'once','process_gone':True}):
            with patch.object(f.command_runner,'status',return_value=health),\
                 patch.object(f.subprocess,'run',side_effect=AssertionError('no retry')):
                with self.assertRaises(RuntimeError):self.fleet.finalize_tick()

    def test_verified_finalization_requests_stop_before_poll_even_without_running_resources(self):
        self.fleet.state={'samples':[],'target_cards':32,'last_change':0}
        self.fleet.spec=WorkloadSpec(scope='test-scope',profile='test-profile',max_cards=32,command=['true'],allowed_cards=[1])
        self.fleet.maximum=128;self.fleet.policy={'deadline_epoch':9999999999};self.fleet.stop=threading.Event()
        def finishing_poll(**kwargs):
            stop=f.read(self.fleet.root/'STOP_REQUEST.json')
            self.assertEqual((stop['scope'],stop['profile']),('test-scope','test-profile'))
            return {'phase':'completed','resources':0}
        self.fleet.controller=SimpleNamespace(locked=lambda:nullcontext(),poll=finishing_poll)
        with patch.object(f,'qualified',return_value=True),patch.object(f.signal,'signal'),\
             patch.object(self.fleet,'business_status',return_value={'total_tasks':1,'completed_tasks':1,'total_pairs':10,'completed_pairs':10,'updated_at':f.time.time()}),\
             patch.object(self.fleet,'finalize_tick',return_value=True):
            result=self.fleet.run()
        self.assertEqual(result['resource_summary']['phase'],'completed')

    def test_business_stop_drains_without_status_scaling_or_finalizing_and_preserves_stop(self):
        self.fleet.state={'samples':[],'target_cards':32,'last_change':0}
        self.fleet.spec=WorkloadSpec(scope='test-scope',profile='test-profile',max_cards=32,command=['true'],allowed_cards=[1])
        self.fleet.stop=threading.Event()
        (self.fleet.output/'STOP').write_text('disk reserve')
        phases=iter(['stopping','completed'])
        def stopped_poll(**kwargs):
            stop=f.read(self.fleet.root/'STOP_REQUEST.json')
            self.assertEqual(stop['scope'],'test-scope')
            return {'phase':next(phases)}
        self.fleet.controller=SimpleNamespace(locked=lambda:nullcontext(),poll=stopped_poll)
        with patch.object(f,'qualified',return_value=True),patch.object(f.signal,'signal'),\
             patch.object(self.fleet.stop,'wait'),\
             patch.object(self.fleet,'business_status',side_effect=AssertionError('no stale-rate scaling')),\
             patch.object(self.fleet,'finalize_tick',side_effect=AssertionError('not completed')):
            result=self.fleet.run()
        self.assertEqual(result['phase'],'await_repair')
        self.assertFalse(f.read(self.fleet.root/'ALARM.json')['automatic_restart_allowed'])
        self.assertEqual((self.fleet.output/'STOP').read_text(),'disk reserve')
        with patch.object(pool,'qualified',return_value=True),patch.object(pool.signal,'signal'),\
             patch.dict(pool.os.environ,{'RESOURCE_MANAGER_ALLOCATION_ID':'owned','RESOURCE_MANAGER_CARDS':'1'}),\
             patch.object(pool.subprocess,'Popen',side_effect=AssertionError('STOP forbids worker')):
            self.assertEqual(pool.run(self.fleet.output,'/frozen/env.sh'),75)


if __name__=='__main__':unittest.main()
