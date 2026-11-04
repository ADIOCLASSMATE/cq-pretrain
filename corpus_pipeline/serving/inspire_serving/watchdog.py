"""Independent guardian for native Serving monitor; never launches retired Jobs."""
import fcntl
import os
from pathlib import Path
import subprocess
import time
from monitor import ROOT, RUNTIME, identity, save, load

COMMAND=['/root/.local/share/uv/tools/inspire-skill/bin/python',str(ROOT/'monitor.py')]


def same(receipt):
    current=identity(receipt.get('pid',-1))
    return bool(current and current['command']==COMMAND and current['starttime_ticks']==receipt.get('starttime_ticks'))


def main():
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME/'watchdog.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (RUNTIME/'WATCHDOG_DISABLED').exists():
            receipt=load(RUNTIME/'monitor-launch.json') if (RUNTIME/'monitor-launch.json').exists() else {}
            if not same(receipt):
                with (RUNTIME/'monitor.log').open('a') as output:
                    proc=subprocess.Popen(COMMAND,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                receipt={**identity(proc.pid),'created_at':time.time(),'guardian_pid':os.getpid()}
                save(RUNTIME/'monitor-launch.json',receipt)
            save(RUNTIME/'watchdog-heartbeat.json',{**identity(os.getpid()),'time':time.time(),'monitor_pid':receipt['pid']})
            time.sleep(5)


if __name__=='__main__':
    main()
