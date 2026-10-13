#!/usr/bin/env python3
"""Kill a job's systemd user unit when truly free memory stays low, so the job dies and the box does not.

The case (2026-10-13): one shallow-water training cell started on two identical GB10 boxes.
On gpu-node-1 free memory fell to ~1 GiB for 1-3 s at a time during its range test, the
NVIDIA driver logged NV_ERR_NO_MEMORY, and the box carried on. On the Turing box the same
cell logged the same error 7 s in and the whole box froze ~80 s later (2026-10-10), with
free memory at 1.0 GiB and 55 GiB of file cache. Both boxes now drop clean cache when free
memory runs low (``machine/memory/keep_free.py``), so a run of low free memory that LASTS
means the job itself is too big for the box; stopping it is cheaper than a frozen box.

Once per ``--interval`` s, while a process matching ``--while-process`` exists (the job's
training process, so a unit still waiting for its turn is never stopped), read MemFree.
If it stays under ``--free-below-gib`` for ``--for-s`` s in a row, SIGKILL every process
in ``--unit`` (memory is freed at once; SIGTERM would let a squeezed job take its time),
stop the unit, log it and exit STOPPED. Exits UNIT_ENDED when the unit stops by itself.
Every entry into and exit from the low state is logged as one JSON line. Stdlib only.

With ``--stop-on-driver-error PATTERN`` (e.g. ``NV_ERR``) the unit is also stopped, exit
DRIVER_ERROR, at the first kernel-log line matching PATTERN since the job's process appeared,
but only during its first ``--driver-error-window-s`` s (default 40 min): on the Turing box the
2026-10-10 freeze followed the driver's first NV_ERR_NO_MEMORY 7 s into a job's start, while a
job that has run cleanly past its start-up is left alone (gpu-node-1 logged hundreds of these
over days and kept running). This replaces the session-bound start-up watcher of 2026-10-13,
which stopped covering a job the moment the session that armed it ended.

    jobs/stop_on_low_free.py --unit <user unit> --while-process '<cmdline substring>' \
        [--free-below-gib 4] [--for-s 10] [--interval 1] \
        [--stop-on-driver-error NV_ERR [--driver-error-window-s 2400]]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "machine" / "memory"))
from keep_free import KIB_PER_GIB, MEMINFO, meminfo

#: How often the unit's state and the process list are re-read (s); MemFree is read every interval.
CHECK_EVERY_S = 5.0


class Exit(IntEnum):
    """How the watch ended."""

    UNIT_ENDED = 0
    STOPPED = 3
    DRIVER_ERROR = 4


def watch(read_free_kib: Callable[[], int], unit_active: Callable[[], bool],
          process_running: Callable[[], bool], stop: Callable[[], None], *,
          floor_kib: int, hold_s: float, interval: float,
          clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep,
          log: Callable[[dict], None] = lambda r: print(json.dumps(r), flush=True),
          driver_error: Callable[[], bool] | None = None,
          driver_window_s: float = 2400.0) -> Exit:
    """Stop the unit once free memory has stayed under ``floor_kib`` for ``hold_s`` s.

    With ``driver_error``, also stop it the first time that returns True while the job's
    process runs and within ``driver_window_s`` s of the process first being seen.
    """
    low_since: float | None = None
    checked = -float("inf")
    running = False
    running_since: float | None = None
    while True:
        now = clock()
        if now - checked >= CHECK_EVERY_S:
            if not unit_active():
                log({"event": "unit_ended"})
                return Exit.UNIT_ENDED
            running = process_running()
            checked = now
            if running and running_since is None:
                running_since = now
            if (running and driver_error is not None
                    and now - running_since <= driver_window_s and driver_error()):
                log({"event": "driver_error_stopping", "since_start_s": round(now - running_since, 1)})
                stop()
                return Exit.DRIVER_ERROR
        free = read_free_kib()
        if running and free < floor_kib:
            if low_since is None:
                low_since = now
                log({"event": "low", "free_gib": round(free / KIB_PER_GIB, 2)})
            if now - low_since >= hold_s:
                log({"event": "stopping", "free_gib": round(free / KIB_PER_GIB, 2),
                     "low_for_s": round(now - low_since, 1)})
                stop()
                return Exit.STOPPED
        elif low_since is not None:
            log({"event": "recovered", "free_gib": round(free / KIB_PER_GIB, 2),
                 "low_for_s": round(now - low_since, 1)})
            low_since = None
        sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--unit", required=True)
    parser.add_argument("--while-process", required=True)
    parser.add_argument("--free-below-gib", type=float, default=4.0)
    parser.add_argument("--for-s", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--stop-on-driver-error", metavar="PATTERN", default=None)
    parser.add_argument("--driver-error-window-s", type=float, default=2400.0)
    args = parser.parse_args(argv)
    systemctl = ["systemctl", "--user"]
    fd = os.open(MEMINFO, os.O_RDONLY)

    def unit_active() -> bool:
        return subprocess.run([*systemctl, "is-active", "--quiet", args.unit], check=False).returncode == 0

    def process_running() -> bool:
        mine = {str(os.getpid()), str(os.getppid())}  # our own argv carries the pattern too
        out = subprocess.run(["pgrep", "-f", args.while_process], capture_output=True, text=True,
                             check=False).stdout.split()
        return any(pid not in mine for pid in out)

    def stop() -> None:
        subprocess.run([*systemctl, "kill", "--signal=SIGKILL", args.unit], check=False)
        # quiet: a --collect unit is already gone once its processes die, and says so on stderr
        subprocess.run([*systemctl, "stop", args.unit], check=False, capture_output=True)

    since: list[int] = []

    def driver_error() -> bool:
        # first call = the check that first saw the process; back off one check plus slack so the
        # job's first seconds (the 2026-10-10 error came 7 s in) are inside the window
        if not since:
            since.append(int(time.time() - CHECK_EVERY_S - 5))
        out = subprocess.run(["journalctl", "-k", "-q", "--no-pager", "--since", f"@{since[0]}",
                              "-g", args.stop_on_driver_error], capture_output=True, text=True,
                             check=False).stdout
        return bool(out.strip())

    print(json.dumps({"event": "started", "unit": args.unit, "while_process": args.while_process,
                      "free_below_gib": args.free_below_gib, "for_s": args.for_s,
                      "stop_on_driver_error": args.stop_on_driver_error,
                      "driver_error_window_s": args.driver_error_window_s}), flush=True)
    return watch(lambda: meminfo(fd)[0], unit_active, process_running, stop,
                 floor_kib=int(args.free_below_gib * KIB_PER_GIB), hold_s=args.for_s,
                 interval=args.interval,
                 driver_error=driver_error if args.stop_on_driver_error else None,
                 driver_window_s=args.driver_error_window_s)


if __name__ == "__main__":
    sys.exit(main())
