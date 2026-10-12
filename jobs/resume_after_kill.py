#!/usr/bin/env python3
"""Resume a long job from its checkpoint when it dies, once the box can take it again.

The pattern: a training run on a shared box (the Turing DGX) can be killed by the
memory guard when someone else's job needs the memory -- by design, ours goes first.
The run saves its state periodically; what was missing is something that notices the
death and resumes it, without a person, and without starting it again into the same
squeeze. The steps:

1. run the job (``--command``), or wait for an already-running one to end
   (``--watch-pid``);
2. if ``--done-if`` succeeds, it finished: stop;
3. if ``--give-up-if`` succeeds, it failed for a reason a resume will not fix (a
   traceback, a refused gate): stop and say so;
4. otherwise wait until ``--ready-if`` succeeds (card free, memory available),
   polling every ``--interval`` s, at most ``--max-wait`` s;
5. run ``--resume-command`` and go back to 2, at most ``--max-resumes`` times.

Commands run under ``bash -c`` in ``--workdir``. Start it with ``setsid nohup`` so it
outlives the session, or, to survive a REBOOT, as a user unit enabled for
``default.target`` on a box with ``Linger=yes`` (the Turing DGX since 2026-10-12) and
pass ``--state FILE``: the file records that the job was started and how many resumes
were used, so a restart after a reboot resumes instead of starting over, and
``--max-resumes`` bounds the total across reboots. Without it, a box that froze
because of the job would restart the job, freeze again, and loop.

Written 2026-10-10 for the raw-field memorisation toy on the Turing DGX
(project-a), after the owner asked that jobs there be resumable when killed.

    jobs/resume_after_kill.py (--command '<cmd>' | --watch-pid <pid>) \
        --resume-command '<cmd>' --done-if '<cmd>' --ready-if '<cmd>' \
        [--give-up-if '<cmd>'] [--max-resumes 3] [--interval 300] [--max-wait 86400] \
        [--workdir <dir>] [--state <file.json>]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path


class Exit(IntEnum):
    """How the supervisor ended."""

    DONE = 0
    GAVE_UP = 1
    NEVER_READY = 2
    OUT_OF_RESUMES = 3


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def shell(command: str, workdir: Path) -> int:
    """Run ``command`` under bash in ``workdir``; its exit status."""
    return subprocess.run(["bash", "-c", command], cwd=workdir, check=False).returncode


def pid_alive(pid: int) -> bool:
    """True while ``pid`` exists and is not a zombie."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


def load_state(path: Path | None) -> dict:
    """What a previous run of this supervisor recorded; empty without a state file."""
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(path: Path | None, state: dict) -> None:
    if path is not None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)


def wait_ready(
    ready_if: str,
    workdir: Path,
    interval: float,
    max_wait: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> bool:
    """Poll ``ready_if`` until it succeeds; False if ``max_wait`` passes first."""
    start = clock()
    while shell(ready_if, workdir) != 0:
        if clock() - start > max_wait:
            return False
        sleep(interval)
    return True


def supervise(
    resume_command: str,
    done_if: str,
    ready_if: str,
    workdir: Path,
    command: str | None = None,
    watch_pid: int | None = None,
    give_up_if: str | None = None,
    max_resumes: int = 3,
    interval: float = 300.0,
    max_wait: float = 86400.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    state_file: Path | None = None,
) -> Exit:
    state = load_state(state_file)
    if state.get("started"):
        log(f"already started by an earlier run ({state_file}): checking, not restarting")
    elif command is not None:
        save_state(state_file, {**state, "started": True})
        log(f"running: {command}")
        log(f"job exited {shell(command, workdir)}")
    elif watch_pid is not None:
        log(f"watching pid {watch_pid}")
        while pid_alive(watch_pid):
            sleep(interval)
        log(f"pid {watch_pid} has ended")
    resumes = int(state.get("resumes", 0))
    while True:
        if shell(done_if, workdir) == 0:
            log("done")
            return Exit.DONE
        if give_up_if is not None and shell(give_up_if, workdir) == 0:
            log("not done, and the give-up check matched: not resuming")
            return Exit.GAVE_UP
        if resumes >= max_resumes:
            log(f"not done after {resumes} resume(s): giving up")
            return Exit.OUT_OF_RESUMES
        log("not done: waiting until the box can take the job again")
        if not wait_ready(ready_if, workdir, interval, max_wait, sleep, clock):
            log(f"box not ready within {max_wait:.0f} s: giving up")
            return Exit.NEVER_READY
        resumes += 1
        save_state(state_file, {"started": True, "resumes": resumes})
        log(f"resume {resumes}/{max_resumes}: {resume_command}")
        log(f"resume exited {shell(resume_command, workdir)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    start = ap.add_mutually_exclusive_group(required=True)
    start.add_argument("--command", help="the job, run now as a shell command")
    start.add_argument("--watch-pid", type=int, help="an already-running job's pid")
    ap.add_argument(
        "--resume-command", required=True, help="resume, as a shell command"
    )
    ap.add_argument("--done-if", required=True, help="exit 0 = the job finished")
    ap.add_argument("--ready-if", required=True, help="exit 0 = the box can take it")
    ap.add_argument("--give-up-if", help="exit 0 = a failure a resume will not fix")
    ap.add_argument("--max-resumes", type=int, default=3)
    ap.add_argument("--interval", type=float, default=300.0, help="seconds per poll")
    ap.add_argument("--max-wait", type=float, default=86400.0, help="seconds to wait")
    ap.add_argument("--workdir", type=Path, default=Path.cwd())
    ap.add_argument("--state", type=Path, help="JSON file: started + resumes used")
    a = ap.parse_args()
    if not os.path.isdir(a.workdir):
        ap.error(f"--workdir {a.workdir} is not a directory")
    return int(
        supervise(
            a.resume_command,
            a.done_if,
            a.ready_if,
            a.workdir,
            command=a.command,
            watch_pid=a.watch_pid,
            give_up_if=a.give_up_if,
            max_resumes=a.max_resumes,
            interval=a.interval,
            max_wait=a.max_wait,
            state_file=a.state,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
