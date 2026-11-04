"""Offline receipt, resource-ordering, fencing and real local process tests."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from . import baseline_pipeline_supervisor as sup

FAKE = '''import json,os,pathlib,signal,sys,time
name,root=sys.argv[1:3]; root=pathlib.Path(root)
p=root/('calls-'+name); p.write_text(str(int(p.read_text())+1 if p.exists() else 1))
mode=os.environ.get('FAKE_MODE','sleep')
if mode=='sleep':
 stopped=[False]; signal.signal(signal.SIGTERM,lambda *a:stopped.__setitem__(0,True))
 while not stopped[0]: time.sleep(.02)
elif mode=='released': print(json.dumps({'released':True,'reserved':0}))
elif mode=='false_released': print(json.dumps({'released':True,'reserved':1}))
elif mode=='ready': print(json.dumps({'ready':True,'ready_replicas':1}))
elif mode=='false_ready': print(json.dumps({'ready':True,'ready_replicas':True}))
elif mode=='fail': print('actual fixture error',file=sys.stderr); sys.exit(7)
elif mode=='stage_complete': print(json.dumps({'phase':'stage_complete'})); sys.exit(2)
'''


class PipelineSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'run'; self.root.mkdir()
        self.script = self.root/'fake.py'; self.script.write_text(FAKE)
        self.repo = str(Path(sup.__file__).resolve().parents[2])
        self.ops = self.root/'ops.json'
        self.config = dict(schema=sup.SCHEMA, run_root=str(self.root), control=str(self.root/'control'),
            repository=self.repo, deadline_utc='2026-10-29T04:18:20Z', queue_port=28691,
            poll_seconds=1, retry_seconds=1, max_retry_seconds=10, max_failures=3, stop_grace_seconds=1,
            commands={name:dict(argv=[sys.executable,str(self.script),name,str(self.root),'28691'],
                                cwd=str(self.root),env={'FAKE_MODE':'sleep' if name in sup.WORKFLOWS or name=='queue_rpc' else 'once'})
                      for name in sup.COMMANDS},
            receipts={name:str(self.root/(name+'.json')) for name in sup.RECEIPTS}, gates={})
        for name in sup.GATES:
            gate=dict(checks=[dict(path=str(self.root/(name+'.json')),equals={'complete':True})])
            if name.endswith('_released'):
                gate['checks'][0]['equals']['reserved']=0;gate['zero_key']='reserved'
            self.config['gates'][name]=gate
        self.config=self.write_config(self.config)
        self.controllers=[];self.extra=[]
        self.addCleanup(self.cleanup)

    def write_config(self, config):
        config=deepcopy(config);config.pop('ops_path',None);config.pop('deadline_epoch',None)
        sup.atomic_json(self.ops,config)
        return sup.checked_ops(self.ops)

    def controller(self):
        obj=sup.Supervisor(self.config);self.controllers.append(obj);return obj

    def cleanup(self):
        for obj in self.controllers:
            obj.stop(tuple(obj.config['commands']),time.time())
        for child in self.extra:
            if child.poll() is None:child.terminate()
        deadline=time.time()+3
        while time.time()<deadline:
            running=[]
            for obj in self.controllers:
                running.extend(child for child in obj.children.values() if child.poll() is None)
            running.extend(child for child in self.extra if child.poll() is None)
            if not running:break
            time.sleep(.03)
        for obj in self.controllers:
            for command in obj.config['commands'].values():
                for identity in sup.process.matching_processes(command['argv'],command['cwd']):
                    sup.process.signal_owned(identity,command['argv'],command['cwd'],signal.SIGKILL)
            for child in obj.children.values():
                if child.poll() is None:child.kill()
                child.wait(timeout=3)
        for child in self.extra:
            if child.poll() is None:child.kill()
            child.wait(timeout=3)

    def receipt(self,name,**extra):
        value=dict(complete=True,stage=name,coverage=dict(expected=7,processed=7,missing=0,pending_infra=0),semantics={'scope':'full'})
        value.update(extra);sup.atomic_json(self.config['receipts'][name],value)

    def gate(self,name,**extra):
        value=dict(complete=True,reserved=0);value.update(extra)
        sup.atomic_json(self.root/(name+'.json'),value)

    def wait_phase(self,controller,name,phase):
        deadline=time.time()+5
        while time.time()<deadline:
            row=controller.reconcile(name,time.time())
            if row['phase']==phase:return row
            time.sleep(.03)
        self.fail(str(row))

    def test_missing_interfaces_and_nonzero_release_gate_fail_closed(self):
        broken=deepcopy(self.config);broken['commands'].pop('review_stop')
        with self.assertRaisesRegex(ValueError,'interfaces'):self.write_config(broken)
        self.config=self.write_config(self.config)
        self.gate('frontpass_released',reserved=1)
        self.assertFalse(sup.gate_passed(self.config,'frontpass_released'))
        self.gate('frontpass_released',reserved=False)
        self.assertFalse(sup.gate_passed(self.config,'frontpass_released'))
        broken=deepcopy(self.config);broken['queue_port']=28692
        with self.assertRaisesRegex(ValueError,'queue_rpc'):self.write_config(broken)

    def test_restart_adopts_exact_runner_without_second_command(self):
        first=self.controller();first.bind();first.step()
        identity=first.state['commands']['normalize']['identity']
        self.assertIsNotNone(identity)
        second=self.controller();second.bind();second.step()
        self.assertEqual(second.state['commands']['normalize']['identity'],identity)
        self.assertEqual(second.state['commands']['normalize']['attempts'],1)
        self.assertFalse(second.children)

    def test_popen_ledger_window_is_adopted(self):
        first=self.controller();first.step()
        identity=first.state['commands']['normalize']['identity']
        first.state['commands']['normalize']['identity']=None;first.save()
        second=self.controller();second.step()
        self.assertEqual(second.state['commands']['normalize']['identity'],identity)
        self.assertEqual(second.state['commands']['normalize']['attempts'],1)

    def test_durable_exit_two_is_stage_success_but_never_business_completion(self):
        self.config['commands']['normalize']['env']['FAKE_MODE']='stage_complete'
        self.config=self.write_config(self.config)
        first=self.controller();first.step()
        row=self.wait_phase(first,'normalize','succeeded');first.save()
        self.assertEqual(row['result']['exit_code'],2);self.assertEqual(row['failures'],0)
        second=self.controller();second.step()
        self.assertEqual(second.state['phase'],'normalize')
        self.assertFalse((second.control/'COMPLETE.json').exists())

    def test_frontpass_release_precedes_review_and_sscd_waits_for_plan(self):
        obj=self.controller();obj.state['phase']='frontpass_release'
        calls=[]
        def ensure(name,*args,**kwargs):
            calls.append(name);return dict(phase='succeeded',alive=False)
        with patch.object(obj,'ensure',side_effect=ensure),patch.object(obj,'stop',return_value=True):
            obj.step(100);self.assertEqual(obj.state['phase'],'frontpass_release')
            self.assertNotIn('review_start',calls)
            self.gate('frontpass_released',reserved=1);obj.step(101)
            self.assertEqual(obj.state['phase'],'frontpass_release')
            self.gate('frontpass_released');obj.step(102)
            self.assertEqual(obj.state['phase'],'review_start')
            obj.step(103);self.assertEqual(obj.state['phase'],'screening')
            obj.state['phase']='sscd';calls.clear();obj.step(104)
            self.assertIn('sscd_workflow',calls);self.assertNotIn('sscd_manager',calls)
            self.gate('sscd_plan');obj.step(105);self.assertIn('sscd_manager',calls)

    def test_stop_signals_only_owned_process_and_never_completes_business(self):
        obj=self.controller();obj.step()
        unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        self.extra.append(unrelated);self.gate('index_released')
        (obj.control/'STOP').touch()
        real=obj.ensure
        with patch.object(obj,'ensure',side_effect=lambda name,*a,**k:dict(phase='succeeded') if name=='index_stop' else real(name,*a,**k)):
            deadline=time.time()+5
            while time.time()<deadline and not obj.step():time.sleep(.05)
        self.assertEqual(obj.state['phase'],'stopped')
        self.assertIsNone(unrelated.poll())
        self.assertFalse((obj.control/'COMPLETE.json').exists())
        self.assertFalse(sup.read_json(obj.control/'STOPPED.json')['complete'])

    def test_deadline_and_false_publication_do_not_create_complete(self):
        obj=self.controller()
        with patch.object(obj,'ensure',return_value={'phase':'running'}):
            obj.step(self.config['deadline_epoch']+10)
        self.assertTrue(obj.state['deadline_overdue']);self.assertFalse(obj.state['complete'])
        obj.state['phase']='publish';sup.atomic_json(self.config['receipts']['publication'],{'complete':True})
        obj.step(self.config['deadline_epoch']+11)
        self.assertIn('CURRENT pointer',obj.state['last_error']['error'])
        self.assertFalse((obj.control/'COMPLETE.json').exists())

    def test_release_probe_requires_zero_reserved_and_resource_starts_are_bounded(self):
        cfg=deepcopy(self.config)
        cfg['commands']['frontpass_released']=dict(argv=[sys.executable,str(self.script),'frontpass_released',str(self.root)],cwd=str(self.root),env={'FAKE_MODE':'false_released'})
        cfg['gates']['frontpass_released']={'command':'frontpass_released'}
        self.config=self.write_config(cfg);obj=self.controller()
        self.assertFalse(obj.gate('frontpass_released',time.time()))
        row=self.wait_phase(obj,'frontpass_released','succeeded')
        self.assertFalse(row['released']);self.assertFalse(obj.gate('frontpass_released',time.time()))
        obj.state['commands']['frontpass_manager']=dict(attempts=3,failures=0,next_start=0,phase='succeeded',handled_attempt=3)
        with patch.object(obj,'reconcile',return_value=obj.state['commands']['frontpass_manager']),patch.object(subprocess,'Popen') as popen:
            obj.ensure('frontpass_manager',time.time(),repeat=True)
            popen.assert_not_called()
        self.assertEqual(obj.state['commands']['frontpass_manager']['phase'],'retry_exhausted')

    def test_full_receipt_requires_full_coverage(self):
        self.receipt('normalize',coverage=dict(expected=7,processed=6,missing=1,pending_infra=0))
        with self.assertRaisesRegex(ValueError,'coverage'):sup.receipt_passed(self.config,'normalize')
        self.receipt('normalize');self.assertTrue(sup.receipt_passed(self.config,'normalize'))

    def test_guard_readiness_probe_requires_actual_numeric_replica(self):
        cfg=deepcopy(self.config)
        cfg['commands']['review_ready']=dict(argv=[sys.executable,str(self.script),'review_ready',str(self.root)],
                                            cwd=str(self.root),env={'FAKE_MODE':'false_ready'})
        cfg['gates']['review_ready']={'command':'review_ready'}
        self.config=self.write_config(cfg);obj=self.controller()
        self.assertFalse(obj.gate('review_ready',time.time()))
        row=self.wait_phase(obj,'review_ready','succeeded');self.assertFalse(row['ready'])
        self.assertFalse(obj.gate('review_ready',time.time()))

    def test_review_stop_exit_zero_repeats_until_live_release(self):
        obj=self.controller();calls=[]
        rows={'review_stop':{'phase':'succeeded'},'review_released':{'phase':'succeeded','released':False}}
        with patch.object(obj,'reconcile',side_effect=lambda name,*a:rows[name]),patch.object(obj,'ensure',side_effect=lambda name,*a,**k:calls.append((name,k))):
            self.assertFalse(obj.released_review(100))
        self.assertEqual(calls,[('review_released',{'repeat':True}),('review_stop',{'repeat':True})])

    def test_operator_stop_keeps_resource_manager_alive_until_release(self):
        obj=self.controller();obj.stop_requested=True
        obj.state['commands']['frontpass_manager']={'attempts':1,'failures':0,'phase':'running','next_start':0}
        stopped=[];started=[]
        def stop(names,*args):stopped.append(tuple(names));return True
        def ensure(name,*args,**kwargs):started.append((name,kwargs));return {'phase':'succeeded'}
        with patch.object(obj,'stop',side_effect=stop),patch.object(obj,'ensure',side_effect=ensure),patch.object(obj,'gate',return_value=False):
            self.assertFalse(obj.step(100))
        self.assertTrue(all('frontpass_manager' not in names for names in stopped))
        self.assertIn(('frontpass_stop',{}),started)
        self.assertIn(('frontpass_manager',{'repeat':True,'cleanup':True}),started)
        self.assertEqual(obj.state['phase'],'stopping')

    def test_complete_requires_bound_full_publication_and_released_resources(self):
        obj=self.controller();obj.state['phase']='publish'
        for name in sup.RECEIPTS[:-1]:self.receipt(name)
        release=self.root/'releases'/'version';release.mkdir(parents=True)
        manifest=release/'manifest.json';sup.atomic_json(manifest,dict(complete=True,scope='full'))
        marker=release/'COMPLETE.json';sup.atomic_json(marker,dict(complete=True,scope='full',manifest_sha256=sup.sha(manifest)))
        pointer=dict(scope='full',complete=dict(path=str(marker),sha256=sup.sha(marker)),
                     manifest=dict(path=str(manifest),sha256=sup.sha(manifest)))
        sup.atomic_json(self.config['receipts']['publication'],pointer)
        with patch.object(obj,'gate',return_value=True),patch.object(obj,'released_review',return_value=True),patch.object(obj,'stop',return_value=True):
            self.assertTrue(obj.step(self.config['deadline_epoch']+1))
        self.assertTrue(obj.state['complete'])
        self.assertFalse(sup.read_json(obj.control/'COMPLETE.json')['deadline_met'])

    def test_lockf_excludes_other_controller(self):
        obj=self.controller()
        script="import fcntl,sys;f=open(sys.argv[1],'a+');\ntry:fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:sys.exit(17)\n"
        with sup.process.exclusive(obj.control/'controller.lock'):
            result=subprocess.run([sys.executable,'-c',script,str(obj.control/'controller.lock')])
        self.assertEqual(result.returncode,17)


if __name__=='__main__':unittest.main()
