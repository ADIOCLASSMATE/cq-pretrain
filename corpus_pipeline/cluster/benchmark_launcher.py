"""Bounded detached benchmark with exit receipt, independent of terminal lifetime."""
import argparse,ctypes,fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from corpus_pipeline.production.common import atomic_json
MODULES={'tuning_probe','concurrency_probe','tuned_frontpass_probe','aggregation_probe'}
LOCK_PATH=Path('/tmp/corpus-dedicated-npu-benchmark.lock')

def specification(directory):
    spec=json.loads((directory/'INTENT.json').read_text())
    command=spec.get('command')
    if (type(spec.get('timeout')) is not int or not 1<=spec['timeout']<=1800
            or spec.get('hostname')!=socket.gethostname()
            or not isinstance(command,list) or len(command)<3
            or command[:2]!=[sys.executable,'-m']
            or command[2] not in {'corpus_pipeline.cluster.'+name for name in MODULES}
            or any(not isinstance(value,str) for value in command)):
        raise ValueError('Benchmark intent does not match bounded local execution contract')
    return spec

def child_exec(directory,parent_pid,lock_fd):
    """Retain the host lock and exit if the detached owner is killed."""
    if type(parent_pid) is not int or parent_pid<=1 or type(lock_fd) is not int or lock_fd<0:
        raise ValueError('Benchmark child requires explicit owner and lock FD')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Cannot install benchmark parent-death guard')
    if os.getppid()!=parent_pid:
        raise RuntimeError('Benchmark owner exited before child startup')
    # pass_fds plus inheritable survives both the wrapper and real Python exec.
    # A SIGKILLed runner therefore cannot release the lock while its child lives.
    os.set_inheritable(lock_fd,True)
    spec=specification(directory)
    os.execv(spec['command'][0],spec['command'])

def run(directory):
    directory=directory.resolve()
    if any((directory/name).exists() for name in ('RUNNING.json','EXIT.json')):
        raise RuntimeError('Benchmark runner already started; use a new directory')
    # Exclusive ownership prevents duplicate --run from overwriting each other's
    # status even before the host-wide execution lock is acquired.
    (directory/'OWNER_STARTED').mkdir()
    lock=None;child=None;code=1;error=None;cleanup_error=None;previous={}
    try:
        spec=specification(directory)
        lock=LOCK_PATH.open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(directory/'RUNNING.json',dict(pid=os.getpid(),hostname=socket.gethostname(),at=time.time()))
        def stop(*_):raise InterruptedError('runner interrupted')
        for signum in (signal.SIGTERM,signal.SIGINT):
            previous[signum]=signal.signal(signum,stop)
        child=subprocess.Popen([sys.executable,'-m','corpus_pipeline.cluster.benchmark_launcher',
            '--child-spec',str(directory),'--parent-pid',str(os.getpid()),'--lock-fd',str(lock.fileno())],
            stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lock.fileno(),))
        atomic_json(directory/'CHILD.json',dict(pid=child.pid,at=time.time()))
        code=child.wait(timeout=spec['timeout'])
    except BaseException as e:
        code=124 if isinstance(e,subprocess.TimeoutExpired) else 130 if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 75 if isinstance(e,BlockingIOError) else 1
        error=dict(type=type(e).__name__,error=str(e)[:500],at=time.time())
        atomic_json(directory/'ERROR.json',error)
    finally:
        # Repeated TERM must not interrupt bounded cleanup or suppress EXIT.
        for signum in previous:signal.signal(signum,signal.SIG_IGN)
        try:
            if child is not None and child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait(timeout=10)
        except BaseException as e:
            cleanup_error=dict(type=type(e).__name__,error=str(e)[:500])
            code=1
        finally:
            atomic_json(directory/'EXIT.json',dict(returncode=code,success=code==0,
                child_returncode=None if child is None else child.poll(),
                cleanup_error=cleanup_error,error_type=None if error is None else error['type'],at=time.time()))
            if lock is not None:lock.close()
            for signum,handler in previous.items():signal.signal(signum,handler)
    return code

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--child-spec',type=Path);p.add_argument('--parent-pid',type=int);p.add_argument('--lock-fd',type=int);p.add_argument('--directory',type=Path);p.add_argument('--module',choices=sorted(MODULES));p.add_argument('--timeout',type=int,default=1800);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.child_spec:return child_exec(a.child_spec,a.parent_pid,a.lock_fd)
    if a.run:return run(a.run)
    if a.directory is None or a.module is None or not 1<=a.timeout<=1800:p.error('directory,module and bounded timeout required')
    d=a.directory.resolve();d.mkdir(parents=True,exist_ok=False)
    args=a.args[1:] if a.args[:1]==['--'] else a.args
    atomic_json(d/'INTENT.json',dict(command=[sys.executable,'-m','corpus_pipeline.cluster.'+a.module,*args],timeout=a.timeout,hostname=socket.gethostname(),at=time.time()))
    with (d/'PROCESS.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-m','corpus_pipeline.cluster.benchmark_launcher','--run',str(d)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(d/'LAUNCHED.json',dict(pid=child.pid,at=time.time(),hostname=socket.gethostname()))
    print(json.dumps(dict(pid=child.pid,directory=str(d))))
    return 0
if __name__=='__main__':raise SystemExit(main())
