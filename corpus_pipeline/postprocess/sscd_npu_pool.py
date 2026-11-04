"""One resumable SSCD process per allocated NPU; no platform operations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from corpus_pipeline.resource_manager.engine import atomic_json


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete(output):
    path=Path(output)/'COMPLETE.json'
    if not path.exists(): return False
    v=read(path)
    plan=Path(output)/'PLAN.json';calibration=Path(output)/'calibration'/'COMPLETE.json'
    return (plan.exists() and calibration.exists() and qualified(output)
            and v.get('plan_sha256')==sha(plan) and v.get('calibration_sha256')==sha(calibration)
            and v.get('complete') is True and v.get('all_pairs_covered') is True
            and type(v.get('total_pairs')) is int and v['total_pairs']>=0
            and v.get('completed_pairs')==v['total_pairs'])


def qualified(output):
    path=Path(output)/'calibration'/'COMPLETE.json'
    if not path.exists(): return False
    v=read(path)
    plan=Path(output)/'PLAN.json'
    return (plan.exists() and v.get('complete') is True and v.get('passed') is True
            and v.get('qualified_for_full_run') is True and v.get('scope')=='full_run'
            and v.get('plan_sha256')==sha(plan))


def await_allocation_lock(lock,should_stop,wait=time.sleep):
    # Orphan stage children deliberately inherit this flock. A replacement pool
    # must stay alive until they release it; returning zero here would make the
    # generic manager permanently classify the allocation as worker_completed.
    while not should_stop():
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            wait(2)
    return False


def exit_code(*,finished,fatal=False):
    return 2 if fatal else 0 if finished else 75


def run(output, worker_env):
    if not qualified(output): raise ValueError('Qualified full-run calibration is required')
    allocation=os.environ.get('RESOURCE_MANAGER_ALLOCATION_ID','')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',allocation): raise ValueError('Explicit allocation identity required')
    cards=int(os.environ.get('RESOURCE_MANAGER_CARDS','0'))
    if not 1<=cards<=16: raise ValueError('Expected 1..16 allocated NPUs')
    root=Path(output);directory=root/'fleet'/'pools';directory.mkdir(parents=True,exist_ok=True)
    statepath=directory/(allocation+'.json')
    stop=False;processes={};next_at={i:0 for i in range(cards)};fatal=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    with (directory/(allocation+'.lock')).open('a') as lock:
        if not await_allocation_lock(lock,lambda:stop or (root/'STOP').exists()):return 75
        # The prior owner may update failures while this replacement waits for
        # inherited child locks. Read only after acquiring ownership and retain
        # a last-exit receipt written before an interrupted heartbeat update.
        old=read(statepath) if statepath.exists() else {}
        failures={str(i):old.get('failures',{}).get(str(i),0) for i in range(cards)}
        for i in range(cards):
            last=directory/(allocation+f'.device{i}.last_exit.json')
            if last.exists():failures[str(i)]=max(failures[str(i)],read(last).get('failures',0))
        try:
            while not stop and not (root/'STOP').exists() and not complete(root):
                for i in range(cards):
                    if (root/'STOP').exists():break
                    proc=processes.get(i)
                    if proc is not None and proc.poll() is not None:
                        rc=proc.returncode;processes.pop(i)
                        if rc:failures[str(i)]+=1
                        next_at[i]=time.time()+min(300,10*2**failures[str(i)])
                        atomic_json(directory/(allocation+f'.device{i}.last_exit.json'),{'at':time.time(),'returncode':rc,'failures':failures[str(i)]})
                    if failures[str(i)]>=3:
                        fatal=True;continue
                    if i not in processes and time.time()>=next_at[i]:
                        argv=['bash',worker_env,'-m','corpus_pipeline.postprocess.sscd_exact_npu',
                              'worker','--output',str(root),'--worker-id',allocation+'-'+str(i),'--device','npu:'+str(i)]
                        with (directory/(allocation+f'.device{i}.log')).open('ab',buffering=0) as log:
                            processes[i]=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                                          start_new_session=True,pass_fds=(lock.fileno(),))
                atomic_json(statepath,{'allocation':allocation,'host':socket.gethostname(),'pid':os.getpid(),
                    'updated_at':time.time(),'cards':cards,'failures':failures,'fatal':fatal,
                    'active_children':{str(i):p.pid for i,p in processes.items() if p.poll() is None},
                    'status':'await_worker_repair' if fatal else 'running'})
                if fatal:break
                time.sleep(5)
        finally:
            for proc in processes.values():
                if proc.poll() is None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            deadline=time.monotonic()+45
            for proc in processes.values():
                if proc.poll() is None:
                    try:proc.wait(timeout=max(.1,deadline-time.monotonic()))
                    except subprocess.TimeoutExpired:
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        proc.wait()
            atomic_json(statepath,{'allocation':allocation,'host':socket.gethostname(),'pid':os.getpid(),
                'updated_at':time.time(),'cards':cards,'failures':failures,'fatal':fatal,'active_children':{},
                'status':'await_worker_repair' if fatal or (root/'STOP').exists() else 'complete' if complete(root) else 'stopped'})
    return exit_code(finished=not (root/'STOP').exists() and complete(root),fatal=fatal)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('run','probe-complete','probe-ready'))
    p.add_argument('--output',required=True);p.add_argument('--worker-env');a=p.parse_args()
    if a.action=='probe-complete':return 0 if complete(a.output) else 1
    if a.action=='probe-ready':
        now=time.time()
        for path in (Path(a.output)/'fleet'/'pools').glob('*.json'):
            v=read(path)
            if v.get('host')==socket.gethostname() and now-v.get('updated_at',0)<30 and v.get('active_children') and not v.get('fatal'):return 0
        return 1
    if not a.worker_env or not Path(a.worker_env).is_absolute():p.error('--worker-env absolute path required')
    return run(a.output,a.worker_env)


if __name__=='__main__':raise SystemExit(main())
