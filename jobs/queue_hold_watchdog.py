#!/usr/bin/env python3
"""Resume a held (SIGSTOPped) job queue when the unit holding it ends -- however it ends.

The pattern this guards: a priority job jumps a running queue by SIGSTOPping the queue's
top-level process, and resumes it from an EXIT trap when done. Two ways that strands the
queue for good, both silent:

1. the holder is killed hard (SIGKILL, the OOM killer): no trap runs;
2. the holder never ends -- e.g. it waits for an input that a failed upstream job will
   never produce.

So this watches the holder's systemd unit from OUTSIDE it. When the unit is no longer
active, for any reason, the queue gets SIGCONT. With ``--stuck-if``, a shell command
whose exit 0 means "the holder is waiting on something that will never come", the
holder is stopped after ``--strikes`` consecutive positive polls and the queue is
resumed. Resuming an already running process is a no-op, so racing the holder's own
trap is harmless.

Written 2026-10-03 for the latent cns64 reproduction chain on gpu-node-1, which holds
the shallow-water queue; a one-off copy in that project's gitignored ``outputs/`` was
replaced by this.

    jobs/queue_hold_watchdog.py --holder-unit <unit> --queue-pid <pid> \
        [--stuck-if '<shell command>'] [--strikes 2] [--interval 60]

Run it as its own unit (``systemd-run --user``) so it outlives the session.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from collections.abc import Callable


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def unit_active(unit: str) -> bool:
    """True while the user systemd unit is active (activating counts as not yet ended)."""
    res = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip() in {"active", "activating", "reloading", "deactivating"}


def stop_unit(unit: str) -> None:
    subprocess.run(["systemctl", "--user", "stop", unit], check=False)


def stuck(command: str) -> bool:
    return subprocess.run(["bash", "-c", command], check=False).returncode == 0


def resume(pid: int) -> bool:
    """SIGCONT ``pid``; False if it no longer exists."""
    try:
        os.kill(pid, signal.SIGCONT)
    except ProcessLookupError:
        return False
    return True


def watch(
    holder_unit: str,
    queue_pid: int,
    stuck_if: str | None = None,
    strikes: int = 2,
    interval: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    log(f"watching {holder_unit} (queue {queue_pid})")
    hits = 0
    while unit_active(holder_unit):
        hits = hits + 1 if stuck_if and stuck(stuck_if) else 0
        if hits >= strikes:
            log(f"stuck condition held {hits} polls; stopping {holder_unit}")
            stop_unit(holder_unit)
            break
        sleep(interval)
    log(f"{holder_unit} no longer active")
    if resume(queue_pid):
        log(f"queue {queue_pid} resumed (SIGCONT)")
    else:
        log(f"queue {queue_pid} not present; nothing to resume")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--holder-unit", required=True, help="systemd --user unit holding the queue"
    )
    ap.add_argument(
        "--queue-pid", type=int, required=True, help="the SIGSTOPped queue process"
    )
    ap.add_argument(
        "--stuck-if", help="shell command; exit 0 = holder can never finish"
    )
    ap.add_argument(
        "--strikes", type=int, default=2, help="consecutive stuck polls before stopping"
    )
    ap.add_argument(
        "--interval", type=float, default=60.0, help="seconds between polls"
    )
    a = ap.parse_args()
    if a.strikes < 1:
        ap.error("--strikes must be >= 1")
    return watch(a.holder_unit, a.queue_pid, a.stuck_if, a.strikes, a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
