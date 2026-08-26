#!/usr/bin/env python3
"""Zombie-reaping subreaper front for the annotation console stack.

Why this exists
---------------
On this dev box ``PID 1`` is ``tail -f /dev/null`` (a container placeholder),
not a real init.  A non-init PID 1 never reaps orphaned descendants, so every
console/job subprocess that outlives its parent — an annotation batch's
``ffmpeg``, a ``watch_console.sh`` ``sleep``, a killed worker's children — is
reparented to PID 1 and stays a permanent zombie.  They accumulate over time and
bloat the process table until the machine feels sluggish even when idle.

Running the whole console under this process fixes that: it marks itself a
*child subreaper* (``prctl(PR_SET_CHILD_SUBREAPER)``), so orphaned descendants
reparent **here** instead of to PID 1, and a reap loop harvests them.  Because it
only reaps orphans that bubble up to it (never the backend's own tracked
``subprocess`` children), it does not race the backend's ``subprocess.run``
bookkeeping.

Lifecycle
---------
1. Mark self as child subreaper.
2. Launch ``ensure_console.sh --watch`` as a child (which nohup-launches the
   backend/frontend/health-monitor/watchdog).  Those daemons reparent to this
   process as soon as the launcher shell exits.
3. Loop forever, reaping any descendant that exits (SIGCHLD-driven, with a
   periodic backstop).
4. On SIGTERM/SIGINT, stop the stack and exit.

Existing zombies already parented to PID 1 cannot be adopted retroactively; they
clear only on the next container restart.  This process prevents *new* ones.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ENSURE = _SCRIPT_DIR / "ensure_console.sh"

# From <linux/prctl.h>: PR_SET_CHILD_SUBREAPER = 36.
_PR_SET_CHILD_SUBREAPER = 36

# Marker so the child ensure_console.sh does not try to bootstrap another reaper.
_UNDER_REAPER_ENV = "MEMSTRATA_CONSOLE_UNDER_REAPER"


def _log(message: str) -> None:
    print(f"[console-init {time.strftime('%F %T')}] {message}", flush=True)


def _set_child_subreaper() -> bool:
    """Mark this process as a child subreaper (Linux >= 3.4). Best-effort."""
    libc_name = ctypes.util.find_library("c") or "libc.so.6"
    try:
        libc = ctypes.CDLL(libc_name, use_errno=True)
    except OSError as exc:  # pragma: no cover - non-glibc / exotic libc
        _log(f"cannot load libc ({exc}); subreaper disabled")
        return False
    res = libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    if res != 0:
        errno = ctypes.get_errno()
        _log(f"prctl(PR_SET_CHILD_SUBREAPER) failed errno={errno}; subreaper disabled")
        return False
    return True


def _reap() -> int:
    """Reap every currently-exited descendant. Returns how many were harvested."""
    reaped = 0
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break  # no children remain
        except OSError:
            break
        if pid <= 0:
            break  # children exist but none has exited yet
        reaped += 1
    return reaped


def _stop_stack() -> None:
    """Ask ensure_console.sh to stop the daemons (without re-killing us)."""
    try:
        subprocess.run(
            ["bash", str(_ENSURE), "--stop"],
            cwd=str(_SCRIPT_DIR),
            env={**os.environ, _UNDER_REAPER_ENV: "1"},
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log(f"stop_stack error: {exc}")


def main() -> int:
    ok = _set_child_subreaper()
    _log(f"pid={os.getpid()} subreaper={'ok' if ok else 'FAILED'} ensure={_ENSURE}")

    stopping = {"flag": False}

    def _on_term(_signum, _frame):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    # Bring up the stack as our child so its whole subtree reparents to us.
    try:
        subprocess.Popen(  # noqa: S603 - fixed, repo-local launcher
            ["bash", str(_ENSURE), "--watch"],
            cwd=str(_SCRIPT_DIR),
            env={**os.environ, _UNDER_REAPER_ENV: "1"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log(f"failed to launch console stack: {exc}")
        return 1

    # Install SIGCHLD only after the launcher Popen so it cannot interfere with
    # Popen's own child setup. Python runs signal handlers in the main thread
    # between bytecodes, so os.waitpid here is safe (not a true async context).
    # We only ever reap orphans that reparented to us; the backend keeps managing
    # its own tracked subprocess children, so this never steals its return codes.
    def _on_sigchld(_signum, _frame):
        _reap()

    signal.signal(signal.SIGCHLD, _on_sigchld)
    _reap()  # harvest anything that exited during setup

    # Handler reaps promptly on each child exit; the periodic wake is a backstop.
    while not stopping["flag"]:
        try:
            time.sleep(30)
        except InterruptedError:  # pragma: no cover - EINTR from SIGCHLD
            pass
        _reap()

    _log("received stop signal; stopping console stack")
    _stop_stack()
    deadline = time.time() + 15
    while time.time() < deadline:
        if _reap() == 0:
            time.sleep(0.2)
    _log("exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
