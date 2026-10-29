"""Small OS-owned cross-process locks; released by the OS after a worker crash."""
from contextlib import contextmanager
from pathlib import Path
import os
import time


def try_lock(file):
    file.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock(file):
    file.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


@contextmanager
def process_lock(path, timeout=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as file:
        if path.stat().st_size == 0:
            file.write(b'0')
            file.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                try_lock(file)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Another process holds ' + path.name)
                time.sleep(min(.2, max(.01, deadline-time.monotonic())))
        try:
            yield
        finally:
            unlock(file)
