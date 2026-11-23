"""Start a detached, idempotent worker inside an already-owned CPU instance."""
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

RUN = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/_repairs/dcvlm_datacomp_unblur_20261104')
worker = int(sys.argv[1])
code_sha = hashlib.sha256(Path(__file__).with_name('worker.py').read_bytes()).hexdigest()
launchfile = RUN / f'worker_{worker:02d}.launch.json'
pidfile = RUN / f'worker_{worker:02d}.pid'
if pidfile.exists():
    pid = int(pidfile.read_text())
    command = Path(f'/proc/{pid}/cmdline')
    if command.exists() and b'dcvlm_unblur/worker.py' in command.read_bytes():
        previous = json.loads(launchfile.read_text()) if launchfile.exists() else {}
        if previous.get('code_sha256') == code_sha:
            print(json.dumps(dict(worker=worker, pid=pid, already_running=True)))
            raise SystemExit(0)
        os.kill(pid, 15)
        for _ in range(50):
            if not command.exists() or not command.read_bytes():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('Previous worker has not exited')
assert (RUN / 'PREPARED.json').is_file()
root = Path(json.loads((RUN / 'PREPARED.json').read_text())['root'])
with (root / 'shards/000/000000.tar').open('rb') as stream:
    assert len(stream.read(512)) == 512
with (RUN / f'worker_{worker:02d}.log').open('ab') as log:
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('worker.py')),
                                '--worker', str(worker)], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
pidfile.write_text(str(process.pid))
record = dict(worker=worker, pid=process.pid, host=socket.gethostname(), code_sha256=code_sha)
launchfile.write_text(json.dumps(record))
print(json.dumps(record))
