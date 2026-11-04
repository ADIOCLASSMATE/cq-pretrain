"""One bounded query-worker cohort; preserves the daemon's inherited flock.

The daemon starts this wrapper in its own process group. Children intentionally
stay in that group, so stopping the daemon-owned group stops the whole cohort.
Every child retains the lock descriptor even if the daemon or wrapper is killed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SEARCH='corpus_pipeline.postprocess.sscd_parallel_search'


def validate(output, processes, threads, lock_fd):
    if type(processes) is not int or not 1<=processes<=8:
        raise ValueError('Search pool must have between one and eight processes')
    if type(threads) is not int or not 1<=threads<=64:
        raise ValueError('Invalid per-process thread count')
    if type(lock_fd) is not int or lock_fd<3:
        raise ValueError('Inherited daemon lock descriptor required')
    os.fstat(lock_fd)  # Never reopen the lock: an independent open changes flock ownership.
    from .sscd_parallel_search import load_plan
    plan=load_plan(Path(output),'search')
    if plan['identity']['threads']!=threads:
        raise ValueError('Search plan and pool thread counts differ')


def worker_command(output, name, lane):
    # sys.executable keeps the selected virtual environment; do not resolve it.
    return [sys.executable,'-m',SEARCH,'search-worker','--output',str(output),
            '--worker-id',name+'-lane'+str(lane)]


def stop_children(children, grace_seconds):
    for child in children:
        if child.poll() is None:
            try: child.terminate()
            except ProcessLookupError: pass
    deadline=time.monotonic()+grace_seconds
    while any(child.poll() is None for child in children) and time.monotonic()<deadline:
        time.sleep(.05)
    for child in children:
        if child.poll() is None:
            try: child.kill()
            except ProcessLookupError: pass
    for child in children:
        child.wait()


def run_pool(output, name, processes, threads, lock_fd, *, grace_seconds=30):
    validate(output,processes,threads,lock_fd)
    if not isinstance(name,str) or not name or '\0' in name:
        raise ValueError('Nonempty worker identity required')
    if not 0<=grace_seconds<=60:
        raise ValueError('Invalid child shutdown grace period')
    environment=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        environment[key]=str(threads)
    children=[]; stopped=False; previous={}
    def stopping(*_):
        nonlocal stopped
        stopped=True
    for number in (signal.SIGTERM,signal.SIGINT):
        previous[number]=signal.signal(number,stopping)
    try:
        for lane in range(processes):
            if stopped: break
            children.append(subprocess.Popen(worker_command(output,name,lane),
                stdin=subprocess.DEVNULL,env=environment,pass_fds=(lock_fd,),start_new_session=False))
        while not stopped:
            codes=[child.poll() for child in children]
            if any(code is not None and code!=0 for code in codes):
                stop_children(children,grace_seconds)
                return 1
            if all(code==0 for code in codes):
                return 0
            time.sleep(.1)
        stop_children(children,grace_seconds)
        return 130
    finally:
        # Includes a partially spawned cohort when Popen itself raises.
        if any(child.poll() is None for child in children):
            stop_children(children,grace_seconds)
        for number,handler in previous.items():
            signal.signal(number,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker-id',required=True)
    parser.add_argument('--processes',type=int,default=1)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--inherited-lock-fd',type=int,required=True)
    args=parser.parse_args()
    raise SystemExit(run_pool(args.output,args.worker_id,args.processes,args.threads,args.inherited_lock_fd))


if __name__=='__main__':main()
