"""Per-allocation process supervision for both unified accelerator stages.

The generic resource manager owns platform lifecycles. This module only owns
its child processes. Children inherit the allocation flock: after a parent is
killed, another pool waits until every orphan releases the lock before loading
another model on those cards. Completion never means merely a local exit zero.
"""
from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from corpus_pipeline.postprocess.sscd_npu_pool import await_allocation_lock,exit_code
from corpus_pipeline.production.common import atomic_json,file_sha
from .config import require

SCHEMA='corpus.unified.worker_pool.v1'
STAGES={'frontpass':'frontpass_max_cards','sscd_search':'sscd_max_cards'}
MAX_FAILURES=3


def read(path):return json.loads(Path(path).read_text())


def stopped(root,stage):
    return (root/'STOP').exists() or (root/stage/'STOP').exists()


def business_complete(config,stage,queue_backend):
    """Independent of pool completion, so a final cache seal can drain safely."""
    from .config import load
    cfg=load(config);root=Path(cfg['output'])
    if stopped(root,stage):return False
    if stage=='frontpass':
        if queue_backend=='rpc':
            from corpus_pipeline.cluster.queue_rpc import status
        else:
            from corpus_pipeline.cluster.queue import status
        value=status(root/'frontpass')
        return value.get('all_tasks_completed') is True and value.get('counts',{}).get('failed',0)==0
    # The fleet's external completion probe also waits for this pool to drain;
    # calling it here would make the two completion gates wait on one another.
    area=root/'sscd_search';marker=area/'COMPLETE.json'
    if not marker.exists():return False
    value=read(marker);plan=read(area/'PLAN.json');calibration=read(area/'calibration/COMPLETE.json')
    return (value.get('complete') is True and value.get('all_pairs_covered') is True and
            calibration.get('passed') is True and value.get('plan_sha256')==file_sha(area/'PLAN.json') and
            calibration.get('plan_sha256')==value['plan_sha256'] and
            value.get('calibration_sha256')==file_sha(area/'calibration/COMPLETE.json') and
            value.get('completed_tasks')==value.get('total_tasks')==plan['total_tasks'] and
            value.get('completed_pairs')==value.get('total_pairs')==plan['total_pairs'])


def run(config,stage,queue_backend='rpc'):
    from .workflow import verify_execution
    from .fleet import preflight
    require(stage in STAGES,'Unsupported accelerator worker stage')
    require(queue_backend in {'local','rpc'},'Unknown queue backend')
    config=Path(config).resolve(strict=True);cfg=verify_execution(config);root=Path(cfg['output'])
    # A durable stop is a recoverable non-success, never an empty completed run.
    if stopped(root,stage):return 75
    preflight(config,stage)
    allocation=os.environ.get('RESOURCE_MANAGER_ALLOCATION_ID','')
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,100}',allocation),'Explicit resource-manager allocation identity required')
    cards=int(os.environ.get('RESOURCE_MANAGER_CARDS','0'))
    require(1<=cards<=16 and cards<=cfg['resources'][STAGES[stage]]<=cfg['resources']['max_cards'],
            'Allocated cards exceed the stage budget or physical allocation limit')
    directory=root/'fleet/pools'/stage;directory.mkdir(parents=True,exist_ok=True)
    statepath=directory/(allocation+'.json')
    binding={'schema':SCHEMA,'config_path':str(config),'config_sha256':file_sha(config),
             'execution_sha256':file_sha(root/'EXECUTION.json'),'allocation':allocation,'stage':stage,'cards':cards}
    runtime_id=socket.gethostname()+':'+Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    signaled=False;fatal=False;finished=False;processes={};next_at={i:0. for i in range(cards)}
    failures={str(i):0 for i in range(cards)};stop_reason=None

    def signal_stop(*_):
        nonlocal signaled
        signaled=True

    handlers={number:signal.getsignal(number) for number in (signal.SIGTERM,signal.SIGINT)}
    for number in handlers:signal.signal(number,signal_stop)

    def save(status):
        atomic_json(statepath,{'binding':binding,'allocation':allocation,'stage':stage,'cards':cards,
            'host':socket.gethostname(),'runtime_id':runtime_id,'pid':os.getpid(),'updated_at':time.time(),
            'failures':failures,'fatal':fatal,'status':status,'stop_reason':stop_reason,
            'active_children':{str(i):p.pid for i,p in processes.items() if p.poll() is None}})

    def check_old(value):
        require(value.get('binding')==binding,'Pool checkpoint belongs to another allocation/configuration')

    try:
      with (directory/(allocation+'.lock')).open('a') as lock:
        if not await_allocation_lock(lock,lambda:signaled or stopped(root,stage),wait=time.sleep):return 75
        # Load after acquiring the lock. A previous parent may have written an
        # exit receipt while this replacement was waiting on orphan children.
        if statepath.exists():
            old=read(statepath);check_old(old)
            for i in range(cards):failures[str(i)]=old.get('failures',{}).get(str(i),0)
        for i in range(cards):
            path=directory/(allocation+f'.device{i}.last_exit.json')
            if path.exists():
                old=read(path);check_old(old)
                failures[str(i)]=max(failures[str(i)],old['failures'])
            require(type(failures[str(i)]) is int and 0<=failures[str(i)]<=MAX_FAILURES,'Invalid persisted failure count')
        try:
            while not signaled and not stopped(root,stage):
                committed=business_complete(config,stage,queue_backend)
                for i in range(cards):
                    proc=processes.get(i)
                    if proc is not None and proc.poll() is not None:
                        rc=proc.returncode;processes.pop(i)
                        if rc:failures[str(i)]=min(MAX_FAILURES,failures[str(i)]+1)
                        next_at[i]=time.time()+(min(300,10*2**failures[str(i)]) if rc else 10)
                        atomic_json(directory/(allocation+f'.device{i}.last_exit.json'),
                            {'binding':binding,'updated_at':time.time(),'returncode':rc,'failures':failures[str(i)]})
                fatal=any(count>=MAX_FAILURES for count in failures.values())
                if fatal:
                    alarm=directory/(allocation+'.ALARM.json')
                    atomic_json(alarm,{'binding':binding,'updated_at':time.time(),
                        'reason':'worker_failure_limit','failures':failures,'requires_operator_repair':True})
                    # Freeze the entire failed stage, not just this allocation:
                    # otherwise replacing the container resets its retry budget.
                    stage_root=root/stage;stage_root.mkdir(parents=True,exist_ok=True)
                    with (stage_root/'.stop.lock').open('a') as stop_lock:
                        fcntl.flock(stop_lock,fcntl.LOCK_EX)
                        marker=stage_root/'STOP'
                        if not marker.exists():
                            atomic_json(marker,{'schema':SCHEMA,'binding':binding,'updated_at':time.time(),
                                'reason':'worker_failure_limit','failures':failures,'requires_operator_repair':True,
                                'alarm':{'path':str(alarm),'sha256':file_sha(alarm)}})
                    break
                if committed and not processes:
                    finished=True;break
                if not committed:
                    for i in range(cards):
                        if signaled or stopped(root,stage):break
                        if i not in processes and time.time()>=next_at[i]:
                            argv=[sys.executable,'-m','corpus_pipeline.unified','worker','--config',str(config),
                                  '--stage',stage,'--device',f'npu:{i}','--queue-backend',queue_backend]
                            try:
                                with (directory/(allocation+f'.device{i}.log')).open('ab',buffering=0) as log:
                                    processes[i]=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                        start_new_session=True,pass_fds=(lock.fileno(),))
                            except OSError as error:
                                failures[str(i)]=min(MAX_FAILURES,failures[str(i)]+1)
                                next_at[i]=time.time()+min(300,10*2**failures[str(i)])
                                atomic_json(directory/(allocation+f'.device{i}.last_exit.json'),
                                    {'binding':binding,'updated_at':time.time(),'returncode':127,
                                     'failures':failures[str(i)],'error_type':type(error).__name__})
                save('draining_completed_children' if committed else 'running')
                time.sleep(5)
        finally:
            stop_reason='signal' if signaled else 'business_stop' if stopped(root,stage) else None
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
            processes.clear()
            # A late STOP wins over a just-observed global completion.
            finished=finished and not signaled and not stopped(root,stage)
            save('await_worker_repair' if fatal else 'complete' if finished else 'stopped')
    finally:
        for number,handler in handlers.items():signal.signal(number,handler)
    return exit_code(finished=finished,fatal=fatal)
