"""Atomic JSON writes and file synchronization helpers."""
import json
import os
import time

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def wait_files(paths, timeout=1200):
    deadline = time.monotonic() + timeout
    while not all(p.is_file() for p in paths):
        if time.monotonic() > deadline:
            raise TimeoutError(f'Missing files: {[str(p) for p in paths if not p.exists()]}')
        time.sleep(1)

