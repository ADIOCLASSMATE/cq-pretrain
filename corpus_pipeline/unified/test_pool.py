"""No accelerator/platform tests of allocation ownership and durable recovery."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from corpus_pipeline.production.common import atomic_json,file_sha
from . import pool


class Process:
    def __init__(self,pid,rc=None):self.pid=pid;self.returncode=rc
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        if self.returncode is None:raise subprocess.TimeoutExpired('fixture',timeout)
        return self.returncode


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.config=self.root/'config.json';self.output=self.root/'run';self.output.mkdir()
        self.cfg={'output':str(self.output),'resources':{'max_cards':8,'frontpass_max_cards':4,'sscd_max_cards':4}}
        atomic_json(self.config,self.cfg);atomic_json(self.output/'EXECUTION.json',{'fixture':True})
        self.clock=100.;self.created=[];self.launch_options=[];self.after_sleep=lambda:None
        self.returncode=0;self.done=lambda:False;self.killed=[]
        mocks=[patch.dict(os.environ,{'RESOURCE_MANAGER_ALLOCATION_ID':'test-allocation','RESOURCE_MANAGER_CARDS':'1'}),
               patch('corpus_pipeline.unified.workflow.verify_execution',return_value=self.cfg),
               patch('corpus_pipeline.unified.fleet.preflight',return_value={'passed':True}),
               patch.object(pool,'business_complete',side_effect=lambda *args:self.done()),
               patch.object(pool.time,'time',side_effect=lambda:self.clock),
               patch.object(pool.time,'monotonic',side_effect=lambda:self.clock),
               patch.object(pool.time,'sleep',side_effect=self.sleep),
               patch.object(pool.subprocess,'Popen',side_effect=self.spawn),
               patch.object(pool.os,'killpg',side_effect=self.kill)]
        for mock in mocks:mock.start();self.addCleanup(mock.stop)
    def sleep(self,seconds):
        self.clock+=seconds;self.after_sleep()
        if self.clock>1000:raise AssertionError('Pool loop did not terminate')
    def spawn(self,argv,**options):
        result=Process(10000+len(self.created),self.returncode)
        self.created.append(result);self.launch_options.append((argv,options));return result
    def kill(self,pid,number):
        self.killed.append((pid,number))
        for process in self.created:
            if process.pid==pid:process.returncode=-number
    def state(self,stage='sscd_search'):
        return pool.read(self.output/f'fleet/pools/{stage}/test-allocation.json')
    def test_clean_local_exit_restarts_until_global_completion(self):
        self.done=lambda:len(self.created)>=2
        self.assertEqual(pool.run(self.config,'sscd_search'),0)
        self.assertEqual(len(self.created),2)
        self.assertEqual(self.state()['status'],'complete');self.assertEqual(self.state()['failures'],{'0':0})
        argv,options=self.launch_options[0]
        self.assertIn('sscd_search',argv);self.assertIn('npu:0',argv)
        self.assertTrue(options['start_new_session']);self.assertEqual(len(options['pass_fds']),1)
        self.assertEqual(self.state()['active_children'],{})
    def test_failure_limit_is_durable_and_last_exit_wins(self):
        self.returncode=7
        self.assertEqual(pool.run(self.config,'sscd_search'),2)
        self.assertEqual(len(self.created),3);self.assertEqual(self.state()['failures'],{'0':3})
        stop=self.output/'sscd_search/STOP'
        self.assertEqual(pool.read(stop)['reason'],'worker_failure_limit')
        self.assertEqual(pool.run(self.config,'sscd_search'),75)
        stop.unlink() # Simulate an explicit repair attempt; persisted failures must still win.
        path=self.output/'fleet/pools/sscd_search/test-allocation.json'
        value=pool.read(path);value['failures']['0']=2;atomic_json(path,value)
        self.assertEqual(pool.run(self.config,'sscd_search'),2)
        self.assertEqual(len(self.created),3) # An interrupted state save cannot reset failures.
        self.assertEqual(self.state()['failures'],{'0':3})
        self.assertTrue((path.parent/'test-allocation.ALARM.json').exists())
    def test_frontpass_completion_waits_for_cache_seal_child_exit(self):
        self.returncode=None
        self.done=lambda:bool(self.created)
        observed=[]
        def after_sleep():
            observed.append(self.state('frontpass')['status'])
            if len(observed)>=3:self.created[0].returncode=0
        self.after_sleep=after_sleep
        self.assertEqual(pool.run(self.config,'frontpass'),0)
        self.assertIn('draining_completed_children',observed)
        self.assertEqual(len(self.created),1);self.assertEqual(self.killed,[])
    def test_signal_kills_group_and_returns_recoverable_not_success(self):
        self.returncode=None;prior=signal.getsignal(signal.SIGTERM)
        self.after_sleep=lambda:signal.raise_signal(signal.SIGTERM)
        self.assertEqual(pool.run(self.config,'sscd_search'),75)
        self.assertEqual(self.state()['status'],'stopped');self.assertEqual(self.state()['stop_reason'],'signal')
        self.assertEqual(self.killed,[(10000,signal.SIGTERM)])
        self.assertEqual(signal.getsignal(signal.SIGTERM),prior)
    def test_stage_disk_stop_kills_child_and_never_reports_complete(self):
        self.returncode=None
        def stop():
            area=self.output/'sscd_search';area.mkdir(exist_ok=True);(area/'STOP').touch()
        self.after_sleep=stop
        self.assertEqual(pool.run(self.config,'sscd_search'),75)
        self.assertEqual(self.state()['stop_reason'],'business_stop')
        self.assertEqual(self.state()['active_children'],{});self.assertEqual(len(self.killed),1)
        self.assertEqual(pool.run(self.config,'sscd_search'),75)
        self.assertEqual(len(self.created),1)
    def test_waiting_parent_reads_failure_checkpoint_after_lock(self):
        self.returncode=7
        self.assertEqual(pool.run(self.config,'sscd_search'),2)
        (self.output/'sscd_search/STOP').unlink()
        directory=self.output/'fleet/pools/sscd_search';path=directory/'test-allocation.json'
        value=pool.read(path);value['failures']['0']=0;atomic_json(path,value)
        last=directory/'test-allocation.device0.last_exit.json';last.unlink()
        def acquire(lock,should_stop,wait):
            value['failures']['0']=3;atomic_json(path,value)
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
        with patch.object(pool,'await_allocation_lock',side_effect=acquire):
            self.assertEqual(pool.run(self.config,'sscd_search'),2)
        self.assertEqual(len(self.created),3);self.assertEqual(self.state()['failures'],{'0':3})
    def test_bad_checkpoint_identity_cannot_be_adopted(self):
        self.done=lambda:True
        self.assertEqual(pool.run(self.config,'frontpass'),0)
        path=self.output/'fleet/pools/frontpass/test-allocation.json';value=pool.read(path)
        value['binding']['config_sha256']='0'*64;atomic_json(path,value)
        with self.assertRaisesRegex(ValueError,'another allocation'):pool.run(self.config,'frontpass')
        self.assertFalse(self.created)


class InheritedLockTests(unittest.TestCase):
    def test_child_holds_allocation_lock_after_parent_descriptor_closes(self):
        # This is a real CPU subprocess, not a mocked flock. No worker/model is loaded.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'allocation.lock';parent=path.open('a');fcntl.flock(parent,fcntl.LOCK_EX|fcntl.LOCK_NB)
            child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.buffer.read(1)'],
                                   stdin=subprocess.PIPE,pass_fds=(parent.fileno(),))
            try:
                parent.close()
                with path.open('a') as replacement:
                    with self.assertRaises(BlockingIOError):fcntl.flock(replacement,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    child.communicate(b'x',timeout=5)
                    self.assertTrue(pool.await_allocation_lock(replacement,lambda:False,wait=lambda _:None))
            finally:
                if child.poll() is None:child.kill();child.wait()


if __name__=='__main__':unittest.main()
