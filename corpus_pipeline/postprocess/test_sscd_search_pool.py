import fcntl
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
from . import sscd_search_pool as p
from . import sscd_cluster_runner as r

class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.lock=open(self.root/'worker.lock','a+'); fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def tearDown(self):
        if not self.lock.closed:self.lock.close()
        self.temp.cleanup()
    def test_children_inherit_descriptor_and_process_group(self):
        fd=self.lock.fileno();group=os.getpgrp()
        def command(output,name,lane):
            code=f"import os,json,pathlib; pathlib.Path({str(self.root/('lane'+str(lane)))!r}).write_text(json.dumps({{'pid':os.getpid(),'pgid':os.getpgrp(),'inode':os.fstat({fd}).st_ino}}))"
            return [sys.executable,'-c',code]
        with patch.object(p,'validate'),patch.object(p,'worker_command',side_effect=command):
            self.assertEqual(p.run_pool(self.root,'owned',4,3,fd),0)
        for lane in range(4):
            row=json.loads((self.root/('lane'+str(lane))).read_text())
            self.assertEqual(row['pgid'],group)
            self.assertEqual(row['inode'],os.fstat(fd).st_ino)
    def test_one_failed_lane_stops_and_reaps_other_lanes(self):
        children=[];real=subprocess.Popen
        def spawn(*a,**kw):
            child=real(*a,**kw);children.append(child);return child
        def command(output,name,lane):
            return [sys.executable,'-c','import time,sys;time.sleep(.2);sys.exit(7)' if lane==0 else 'import time;time.sleep(30)']
        with patch.object(p,'validate'),patch.object(p,'worker_command',side_effect=command),patch.object(p.subprocess,'Popen',side_effect=spawn):
            self.assertEqual(p.run_pool(self.root,'owned',4,3,self.lock.fileno(),grace_seconds=.2),1)
        self.assertEqual(len(children),4)
        self.assertTrue(all(child.poll() is not None for child in children))
    def test_partial_spawn_failure_cleans_started_child(self):
        real=subprocess.Popen;children=[]
        def spawn(*a,**kw):
            if children:raise OSError('spawn failed')
            child=real(*a,**kw);children.append(child);return child
        with patch.object(p,'validate'),patch.object(p,'worker_command',return_value=[sys.executable,'-c','import time;time.sleep(30)']),patch.object(p.subprocess,'Popen',side_effect=spawn):
            with self.assertRaises(OSError):p.run_pool(self.root,'owned',4,3,self.lock.fileno(),grace_seconds=.2)
        self.assertIsNotNone(children[0].poll())
    def test_plan_thread_mismatch_fails_before_spawn(self):
        with patch('corpus_pipeline.postprocess.sscd_parallel_search.load_plan',return_value={'identity':{'threads':14}}):
            with self.assertRaisesRegex(ValueError,'thread counts differ'):p.validate(self.root,4,3,self.lock.fileno())
    def test_orphan_child_keeps_daemon_lock_after_wrapper_kill(self):
        fd=self.lock.fileno();marker=self.root/'orphan.json'
        child_code=f"import os,json,pathlib,time;pathlib.Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'pgid':os.getpgrp(),'fd':os.fstat({fd}).st_ino}}));time.sleep(30)"
        helper=("from unittest.mock import patch\nfrom corpus_pipeline.postprocess import sscd_search_pool as p\n"
                f"with patch.object(p,'validate'),patch.object(p,'worker_command',return_value={[sys.executable,'-c',child_code]!r}):\n"
                f" p.run_pool({str(self.root)!r},'owned',1,1,{fd},grace_seconds=.2)\n")
        process=subprocess.Popen([sys.executable,'-c',helper],pass_fds=(fd,),start_new_session=True)
        self.lock.close()
        try:
            until=time.monotonic()+5
            while not marker.exists() and time.monotonic()<until:time.sleep(.02)
            self.assertTrue(marker.exists())
            self.assertEqual(json.loads(marker.read_text())['pgid'],process.pid)
            process.kill();process.wait(timeout=5)
            with open(self.root/'worker.lock','a+') as other:
                with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            process.wait(timeout=5)
        with open(self.root/'worker.lock','a+') as other:
            until=time.monotonic()+5
            while True:
                try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>=until:raise
                    time.sleep(.02)

class RunnerWiringTests(unittest.TestCase):
    def test_search_command_forwards_live_lock_and_pool_counts(self):
        c={'python':sys.executable,'output':'/tmp/isolated','search_processes':4,'search_threads':3,'threads':14}
        command=r.worker_command(c,'search','node08',17)
        self.assertIn(r.SEARCH_POOL,command)
        self.assertEqual(command[command.index('--processes')+1],'4')
        self.assertEqual(command[command.index('--threads')+1],'3')
        self.assertEqual(command[command.index('--inherited-lock-fd')+1],'17')
        self.assertNotIn(r.SEARCH_POOL,r.worker_command(c,'build','node08',17))
    def test_config_defaults_and_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'resources').mkdir();r.save(root/'resources/PLAN.json',{'instances':[{'name':'cpu','owned':True}]})
            c={k:str(root) for k in ('root','output','python','runtime','sdk_python','input','adopt')}
            c.update(threads=14,notebooks=['cpu'],max_instances=8,runtime_sha256={});cp=root/'config.json';r.save(cp,c)
            self.assertEqual((r.config(cp)['search_processes'],r.config(cp)['search_threads']),(1,14))
            c.update(search_processes=4,search_threads=3);r.save(cp,c);self.assertEqual(r.config(cp)['search_threads'],3)
            c['search_threads']=4;r.save(cp,c)
            with self.assertRaisesRegex(ValueError,'CPU thread budget'):r.config(cp)

if __name__=='__main__':unittest.main()
