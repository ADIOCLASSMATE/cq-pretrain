"""Restart-safe metadata seal controller for the CPU01 allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from corpus_pipeline.production.common import atomic_json
from finalize_finevision_recovery import finalize


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); args=parser.parse_args()
    run=Path(args.run); directory=run/'ops'; directory.mkdir(exist_ok=True)
    heartbeat=directory/'cpu01_metadata.heartbeat.json'; current={'pid':os.getpid(),'state':'running'}
    with (directory/'cpu01_metadata.process.lock').open('a') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return
        def pulse():
            while True:
                atomic_json(heartbeat,{**current,'updated_at':time.time()}); time.sleep(30)
        threading.Thread(target=pulse,daemon=True).start()
        while True:
            try:
                result=finalize(run); current['result']=result
                atomic_json(directory/'finevision_seal_status.json',{**result,'time':time.time()})
                if (run/'ingest_registry/finevision.json').exists():
                    atomic_json(heartbeat,{**current,'state':'complete','updated_at':time.time()}); return
            except Exception as exc:
                atomic_json(directory/'finevision_seal_ERROR.json',{'error_type':type(exc).__name__,'error':str(exc),'time':time.time()})
                current['error']=str(exc)
            time.sleep(60)


if __name__=='__main__': main()
