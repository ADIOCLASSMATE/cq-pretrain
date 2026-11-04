"""External process guardian; adopts only the exact PID + Linux starttime pair."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime'
COMMAND = ['/root/.local/share/uv/tools/inspire-skill/bin/python', str(ROOT/'supervisor.py'), '--interval', '2']


def process_identity(pid):
    try:
        raw = Path(f'/proc/{int(pid)}/stat').read_text()
        # comm may contain spaces and parentheses; parse after its final ')'.
        fields = raw[raw.rfind(')')+2:].split()
        return {'pid': int(pid), 'starttime_ticks': int(fields[19]),
                'command': Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')[:-1]}
    except (OSError, ValueError, IndexError):
        return None


def is_same_process(receipt):
    live = process_identity(receipt.get('pid', -1))
    return bool(live and live['starttime_ticks'] == receipt.get('starttime_ticks') and
                live['command'] == [x.encode() for x in COMMAND])


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def main():
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME/'watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launches = 0
        while not (RUNTIME/'WATCHDOG_DISABLED').exists():
            try:
                receipt = json.loads((RUNTIME/'supervisor-launch.json').read_text())
            except (OSError, ValueError):
                receipt = {}
            if not is_same_process(receipt):
                with (RUNTIME/'supervisor.log').open('a') as log:
                    process = subprocess.Popen(COMMAND, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT.parents[1])
                identity = process_identity(process.pid)
                if identity is not None:
                    save(RUNTIME/'supervisor-launch.json', {'pid': process.pid,
                         'starttime_ticks': identity['starttime_ticks'], 'command': COMMAND,
                         'created_at': time.time(), 'maximum_owned_cards':48, 'guardian_pid':os.getpid()})
                launches += 1
                # Crash loops never turn into aggressive workload mutation loops.
                time.sleep(5)
            own = process_identity(os.getpid())
            save(RUNTIME/'watchdog-heartbeat.json', {'pid':os.getpid(), 'starttime_ticks':own['starttime_ticks'],
                 'time':time.time(), 'supervisor_pid':json.loads((RUNTIME/'supervisor-launch.json').read_text())['pid'],
                 'restarts_this_process':launches, 'interval':15})
            time.sleep(15)


if __name__ == '__main__':
    main()
