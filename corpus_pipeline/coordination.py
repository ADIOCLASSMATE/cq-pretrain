"""Shared-filesystem locks: POSIX record locks plus same-process exclusion.

GPFS flock may be host-local. lockf coordinates hosts, but POSIX locks are
process scoped, so every local contender must also hold this thread lock.
Never unlink a live lock file; that would create a second lock inode.
"""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import threading

_guard = threading.Lock()
_locks = {}


@contextmanager
def locked(path, nonblocking=False):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _guard:
        lock = _locks.setdefault(str(path), threading.Lock())
    if not lock.acquire(blocking=not nonblocking):
        raise BlockingIOError('shared_lock_busy')
    try:
        with path.open('a') as stream:
            # Also cooperate with existing same-host readers of this lock.
            fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
            fcntl.lockf(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
            try:
                yield
            finally:
                fcntl.lockf(stream, fcntl.LOCK_UN)
                fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        lock.release()


@contextmanager
def available_lock(path):
    manager = locked(path, nonblocking=True)
    try:
        manager.__enter__()
    except BlockingIOError:
        yield False
    else:
        try:
            yield True
        finally:
            manager.__exit__(None, None, None)
