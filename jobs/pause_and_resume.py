#!/usr/bin/env python3
"""Pause a training unit at a checkpoint, run another job, then resume the first -- however that job ends.

The pattern: job A (a long training run in a systemd --user unit) must give the GPU to
job B for a while, and continue afterwards without losing work. The steps:

1. wait until A's checkpoint file is REWRITTEN after arming (and has stopped changing for
   ``--settle`` seconds), so stopping A loses at most the time since that write;
2. stop A's unit (``systemctl --user stop``: SIGTERM, not SIGSTOP);
3. start B as its own unit and wait while it is active;
4. when B's unit is no longer active, for ANY reason (finished, failed, killed, never
   started), start A's resume command as its own unit.

Why not freeze A with SIGSTOP: Lightning's ``Timer`` measures with ``time.monotonic``, so
a frozen run's clock keeps running and it stops early on thaw; a frozen process also keeps
its GPU memory, which on the GB10 is system RAM. A stopped run releases both, and a
resume from its checkpoint restores the elapsed time.

If A is not active when armed, or ends on its own while the watcher waits for the
checkpoint, A is NOT stopped and NOT resumed: B simply runs. If the checkpoint is not
rewritten within ``--max-wait`` seconds while A is still active, nothing is touched and
the tool exits 2 (A may be hung; a person should look).

Written 2026-10-04 for the shallow-water cell on gpu-node-1, paused for the latent cns64
toy runs (project-a D204); the previous pause was a one-off shell script.

    jobs/pause_and_resume.py --pause-unit <A unit> --checkpoint <A last.ckpt> \
        --run-unit <B unit> --run-command '<B command>' \
        --resume-unit <A resume unit> --resume-command '<A resume command>' \
        [--workdir <dir>] [--settle 60] [--interval 30] [--max-wait 21600]

Run it as its own unit (``systemd-run --user``) so it outlives the session. The commands
run under ``bash -c`` in ``--workdir`` with systemd's user environment, not this shell's:
make them self-contained.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from queue_hold_watchdog import log, stop_unit, unit_active

#: Exit status when the checkpoint was never rewritten while A stayed active.
EXIT_CHECKPOINT_TIMEOUT = 2


def mtime_ns(path: Path) -> int | None:
    """The file's modification time, or None if it does not exist (follows links)."""
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def start_unit(unit: str, command: str, workdir: Path) -> bool:
    """Start ``command`` as a transient user unit; False if systemd refused it."""
    res = subprocess.run(
        [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            f"--working-directory={workdir}",
            "bash",
            "-c",
            command,
        ],
        check=False,
    )
    return res.returncode == 0


def wait_for_checkpoint(
    pause_unit: str,
    checkpoint: Path,
    settle: float,
    interval: float,
    max_wait: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> bool | None:
    """Wait for a fresh, settled checkpoint while A runs.

    Returns:
        True when a rewrite has settled (A may be stopped); False when A is no longer
        active (it ended on its own); None when ``max_wait`` passed first.
    """
    baseline = mtime_ns(checkpoint)
    start = clock()
    log(f"waiting for {checkpoint} to be rewritten (now {baseline})")
    while unit_active(pause_unit):
        if clock() - start > max_wait:
            return None
        current = mtime_ns(checkpoint)
        if current is not None and current != baseline:
            sleep(settle)
            if mtime_ns(checkpoint) == current:
                log(f"checkpoint rewritten and settled for {settle:.0f} s")
                return True
            baseline = current  # still being written; wait for it to settle
            continue
        sleep(interval)
    return False


def pause_and_resume(
    pause_unit: str,
    checkpoint: Path,
    run_unit: str,
    run_command: str,
    resume_unit: str,
    resume_command: str,
    workdir: Path,
    settle: float = 60.0,
    interval: float = 30.0,
    max_wait: float = 21600.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    paused = False
    if unit_active(pause_unit):
        ready = wait_for_checkpoint(
            pause_unit, checkpoint, settle, interval, max_wait, sleep, clock
        )
        if ready is None:
            log(
                f"{checkpoint} not rewritten within {max_wait:.0f} s while "
                f"{pause_unit} stayed active; touching nothing"
            )
            return EXIT_CHECKPOINT_TIMEOUT
        if ready:
            log(f"stopping {pause_unit} (SIGTERM via systemd)")
            stop_unit(pause_unit)
            paused = True
        else:
            log(f"{pause_unit} ended on its own; it will not be resumed")
    else:
        log(f"{pause_unit} not active; it will not be stopped or resumed")

    if start_unit(run_unit, run_command, workdir):
        log(f"started {run_unit}")
        while unit_active(run_unit):
            sleep(interval)
        log(f"{run_unit} no longer active")
    else:
        log(f"systemd refused to start {run_unit}; going straight to the resume")

    if paused:
        if start_unit(resume_unit, resume_command, workdir):
            log(f"resumed {pause_unit} as {resume_unit}")
        else:
            log(f"FAILED to start {resume_unit}; resume {pause_unit} by hand")
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pause-unit", required=True, help="systemd --user unit of job A")
    ap.add_argument(
        "--checkpoint", type=Path, required=True, help="A's checkpoint (last.ckpt)"
    )
    ap.add_argument("--run-unit", required=True, help="unit name to give job B")
    ap.add_argument("--run-command", required=True, help="job B, as a shell command")
    ap.add_argument("--resume-unit", required=True, help="unit name for A's resume")
    ap.add_argument(
        "--resume-command", required=True, help="A's resume, as a shell command"
    )
    ap.add_argument(
        "--workdir", type=Path, default=Path.cwd(), help="where both commands run"
    )
    ap.add_argument(
        "--settle", type=float, default=60.0, help="seconds the rewrite must be stable"
    )
    ap.add_argument(
        "--interval", type=float, default=30.0, help="seconds between polls"
    )
    ap.add_argument(
        "--max-wait",
        type=float,
        default=21600.0,
        help="seconds to wait for a rewrite before giving up (A untouched)",
    )
    a = ap.parse_args()
    if not os.path.isdir(a.workdir):
        ap.error(f"--workdir {a.workdir} is not a directory")
    return pause_and_resume(
        a.pause_unit,
        a.checkpoint,
        a.run_unit,
        a.run_command,
        a.resume_unit,
        a.resume_command,
        a.workdir,
        a.settle,
        a.interval,
        a.max_wait,
    )


if __name__ == "__main__":
    raise SystemExit(main())
