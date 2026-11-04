"""External owner for one unmodified official vLLM HIGH allocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time
from corpus_pipeline.serving.budget_pool import locked
from corpus_pipeline.serving.qwen36_w8a8_20261024.official_v023.backend import (
    Backend,ROOT,prepare_contract,native,pool,lifecycle,check)
HIGH=ROOT/'high'
NAME='corpus-qwen36-official-v023-high-20261024'


def safe(state):
    return {k:state.get(k) for k in ('service_id','phase','platform_status','canary_passed',
        'readiness_reason','observation_error_type','heartbeat_at','release_verified','incarnation')}


def create():
    with locked(HIGH/'lifecycle.lock'):
        check(not (HIGH/'DISABLED').exists(),'HIGH explicitly disabled')
        backend=Backend()
        if (HIGH/'CREATE_INTENT.json').exists():return safe(native.load(HIGH/'state.json'))
        previous=native.load(ROOT.parent/'high/state.json')
        check(previous.get('release_verified') is True and lifecycle.Backend().released(previous),
              'Previous experimental HIGH is not freshly released')
        prepare_contract()
        state=backend.prepare(HIGH,0,NAME,6)
        lifecycle.persist(HIGH/'state.json',state)
        native.save(HIGH/'CREATE_INTENT.json',{'name':NAME,'cards':2,'time':time.time(),
                    'launch_contract_sha256':state['launch_contract_sha256']})
        try:state=backend.create(HIGH,state)
        except Exception as error:
            state.update(phase='CREATE_UNKNOWN',observation_error_type=type(error).__name__)
            lifecycle.persist(HIGH/'state.json',state)
            raise
        lifecycle.persist(HIGH/'state.json',state)
        return safe(state)


def observe(once=False,event=None):
    event=event or threading.Event();backend=Backend()
    with locked(HIGH/'observer.lock',nonblocking=True):
        native.save(HIGH/'OBSERVER.json',{**native.identity(os.getpid()),'time':time.time()})
        while not event.is_set() and not (HIGH/'DISABLED').exists():
            with locked(HIGH/'lifecycle.lock'):
                state=native.load(HIGH/'state.json')
                try:state=backend.observe(HIGH,state)
                except Exception as error:
                    state.update(canary_passed=False,readiness_reason='OBSERVATION_FAILED',
                        observation_error_type=type(error).__name__,heartbeat_at=time.time())
                    backend.publish(HIGH,state,False)
                lifecycle.persist(HIGH/'state.json',state);native.save(HIGH/'STATUS.json',safe(state))
            if once:return safe(state)
            event.wait(8)
    return {'observer_stopped':True}


def stop():
    HIGH.mkdir(parents=True,exist_ok=True);(HIGH/'DISABLED').write_text('Explicit owned HIGH stop\n')
    with locked(HIGH/'lifecycle.lock'):
        backend=Backend();state=backend.stop(HIGH,native.load(HIGH/'state.json'))
        lifecycle.persist(HIGH/'state.json',state)
        return safe(state)


def main():
    pool.configure_management_proxy_bypass()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['create','observe','stop'])
    parser.add_argument('--once',action='store_true');args=parser.parse_args();event=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:event.set())
    try:
        result=create() if args.action=='create' else stop() if args.action=='stop' else observe(args.once,event)
        print(json.dumps(result,ensure_ascii=False));return 0
    except Exception as error:
        print(json.dumps({'success':False,'error_type':type(error).__name__}));return 1


if __name__=='__main__':raise SystemExit(main())
