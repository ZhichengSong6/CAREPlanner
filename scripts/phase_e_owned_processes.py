"""Linux subprocess ownership tracking; never kill by global executable name."""
from pathlib import Path
import os
import signal
import subprocess
import time


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]  # kernel starttime; guards PID reuse
    except (FileNotFoundError, ProcessLookupError): return None


def remember_descendants(roots, owned):
    rows = [tuple(map(int, line.split())) for line in
            subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True).splitlines()]
    parents = set(roots)
    while parents:
        children = {pid for pid, ppid in rows if ppid in parents and pid not in roots}
        for pid in children:
            token = identity(pid)
            if token is not None: owned.setdefault(pid, token)
        parents = children


def alive(owned):
    return [pid for pid, token in owned.items() if identity(pid) == token]


def reap_owned(owned):
    signalled = []
    for sig, grace in [(signal.SIGINT, 3.), (signal.SIGTERM, 3.), (signal.SIGKILL, 1.)]:
        for pid in alive(owned):
            try:
                os.kill(pid, sig); signalled.append(dict(pid=pid, signal=int(sig)))
            except ProcessLookupError: pass
        deadline = time.monotonic()+grace
        while alive(owned) and time.monotonic() < deadline: time.sleep(.05)
        if not alive(owned): break
    return dict(signalled=signalled, remaining=alive(owned))
