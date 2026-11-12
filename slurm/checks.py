"""The check a submission states before it is queued (D47): `--check CMD` runs here first and a failure refuses the job;
`--no-check REASON` says why none is possible; the site's `require_check` (off / recommend / enforce) says what happens
when neither is given.

A fault found only when a job starts costs its whole queue wait, and in a chain of jobs every upstream run as well
(2026-11-10: an evaluate step refused after a 7-hour chain). The check runs on the submitting machine, before the wait,
so the fault surfaces in seconds. It is the project's own command (its pipeline's dry run or stub run on the real
inputs); this module only runs it, safely:

- with the GPU hidden (`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` empty): GPU work goes through slurm. A convention,
  not a wall (`docker run --gpus` ignores it); the audit's report of GPU work outside slurm is what finds a breach;
- with `GPUJOBS_CHECK=1` set, so a project's command can tell, and so `submit.py` / `urgent.py` refuse to queue from
  inside a check;
- as the first victim under memory pressure (`oom_score_adj` 1000, which needs no privilege to raise), never a job;
- in its own process group, killed on any exit of the submitting tool (a signal, a time limit, an error), and its
  direct child also by `PR_SET_PDEATHSIG` if the tool is killed outright;
- for at most `CHECK_TIMEOUT_S`.

A sweep submitting many jobs with one check runs it once: the same command, from the same folder, by the same user
that passed within `CHECK_REUSE_S` counts as passed (the user's own cache, so no other user can plant a pass), and
identical checks submitted at once wait on one lock and reuse the first one's pass.

Stdlib only.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

import gpu_jobs
from site_config import Policy

#: The owner's limit on any check (2026-11-07: "tests that take ... 10 minutes, it's just ridiculous"): seconds, never
#: over about four minutes.
CHECK_TIMEOUT_S = 240.0
#: After SIGTERM at the time limit, how long the check's group has to stop before SIGKILL.
CHECK_KILL_GRACE_S = 5.0
#: A sweep submitted in one go shares its inputs: one pass of the same check from the same folder covers it this long.
CHECK_REUSE_S = 600.0
#: Work holding more than about 8 GB belongs in slurm (the owner's rule for this machine): a check this heavy is a job.
CHECK_HEAVY_BYTES = 8 * gpu_jobs.GB
#: Set in the check's environment; a submission from inside a check is refused.
IN_CHECK = "GPUJOBS_CHECK"
#: Where the flags and what makes a good check are explained.
HOW_TO = "README, 'Checks before queuing'"
#: The highest `oom_score_adj`: the kernel's OOM killer, and earlyoom, take this process first.
OOM_SCORE_ADJ_FIRST = 1000
#: prctl(2) option: the signal the child gets when its parent dies.
PR_SET_PDEATHSIG = 1
#: How often the wait looks at the check.
POLL_S = 0.05
#: Linux reports ru_maxrss in KiB.
KIB = 1024


class CheckRefused(Exception):
    """A submission refused by its check or by the site's policy; the message names the remedy."""


@dataclass(frozen=True)
class Outcome:
    """What is kept in the job's record (`check`) and the lines to print."""

    record: dict[str, object]
    lines: tuple[str, ...] = ()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """`--check CMD` / `--no-check REASON`, one or the other."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", default=None, metavar="CMD",
                       help="a fast command (seconds, at most 4 minutes, no GPU) that tries the job's steps on its real "
                            f"inputs, run here before the job is queued; a failure refuses the job ({HOW_TO})")
    group.add_argument("--no-check", default=None, metavar="REASON",
                       help="why no check is possible for this job; kept in its record and shown on its page")


def refuse_nested() -> None:
    """A check tries a job's steps; it must not queue jobs itself."""
    if os.environ.get(IN_CHECK):
        msg = (f"{IN_CHECK} is set: this is running inside a pre-queue check, and a check must not queue jobs; make the "
               "check try the steps without submitting them")
        raise CheckRefused(msg)


def cache_dir() -> Path:
    """The user's own cache folder for passes (`$XDG_CACHE_HOME/gpujobs`, else `~/.cache/gpujobs`)."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "gpujobs"


def settle(check: str | None, no_check: str | None, policy: Policy, workdir: Path, *, dry_run: bool,
           cache: Path | None = None, now: Callable[[], float] = time.time,
           timeout_s: float = CHECK_TIMEOUT_S) -> Outcome:
    """Settle a submission's check under the site's `policy`: run `check` (or reuse a recent pass), take `no_check`'s
    reason, or apply the policy to a submission that states neither. Raises `CheckRefused` (nothing is queued)."""
    if no_check is not None:
        reason = no_check.strip()
        if not reason:
            msg = "--no-check needs a reason: why no check is possible for this job"
            raise CheckRefused(msg)
        return Outcome({"policy": str(policy), "none": reason}, (f"none: {reason}",))
    if check is None:
        if policy is Policy.ENFORCE:
            msg = (f"this site requires a check: pass --check '<a fast command that tries the job's steps on its real "
                   f"inputs>', or --no-check '<why none is possible>' ({HOW_TO})")
            raise CheckRefused(msg)
        if policy is Policy.RECOMMEND:
            said = (f"warning: no --check or --no-check; a fault the job meets at its start will surface only after its "
                    f"queue wait ({HOW_TO})")
            return Outcome({"policy": str(policy)}, (said,))
        return Outcome({"policy": str(policy)})
    argv = shlex.split(check)
    if not argv:
        msg = f"--check is empty: give the command to run ({HOW_TO})"
        raise CheckRefused(msg)
    if dry_run:
        return Outcome({"policy": str(policy), "command": check}, (f"would run `{check}` before queuing",))
    folder = cache or cache_dir()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([argv, str(workdir.resolve())]).encode()).hexdigest()
    with _locked(folder / f"{key}.lock"):  # identical checks at once: the first runs, the rest reuse its pass
        passed = _passed_at(folder, key, now())
        if passed is not None:
            return Outcome({"policy": str(policy), "command": check, "reused_from": passed},
                           (f"`{check}` passed {int(now() - passed)} s ago from this folder; not run again",))
        exit_code, seconds, peak = run(argv, workdir, timeout_s)
        record = {"policy": str(policy), "command": check, "exit": exit_code, "seconds": round(seconds, 1),
                  "peak_bytes": peak}
        if exit_code is None:
            msg = (f"the check `{check}` did not finish within {timeout_s:g} s; nothing was queued. A check runs in "
                   f"seconds: make it try the steps on a small slice of the inputs ({HOW_TO})")
            raise CheckRefused(msg)
        if exit_code != 0:
            msg = (f"the check `{check}` failed (exit {exit_code}) after {seconds:.0f} s; nothing was queued. Fix what it "
                   f"reported and submit again ({HOW_TO})")
            raise CheckRefused(msg)
        _remember(folder, key, now())
        lines = [f"`{check}` passed in {seconds:.1f} s (peak {peak / gpu_jobs.GB:.1f} GB)"]
        if peak > CHECK_HEAVY_BYTES:
            lines.append(f"warning: the check peaked at {peak / gpu_jobs.GB:.1f} GB: work this size is a job, not a "
                         f"check; make the check smaller ({HOW_TO})")
        return Outcome(record, tuple(lines))


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _passes(folder: Path) -> dict[str, float]:
    found = gpu_jobs.read_json(folder / "checks.json", gpu_jobs.current_user()) or {}
    return {key: float(at) for key, at in found.items() if isinstance(at, int | float)}


def _passed_at(folder: Path, key: str, now: float) -> float | None:
    at = _passes(folder).get(key)
    return at if at is not None and 0 <= now - at <= CHECK_REUSE_S else None


def _remember(folder: Path, key: str, now: float) -> None:
    kept = {k: at for k, at in _passes(folder).items() if 0 <= now - at <= CHECK_REUSE_S}
    gpu_jobs.write_json(folder / "checks.json", {**kept, key: now})


def _child_setup() -> None:
    """In the check's process before it runs: the first victim under memory pressure, and killed with its parent."""
    with contextlib.suppress(OSError):
        Path("/proc/self/oom_score_adj").write_text(str(OOM_SCORE_ADJ_FIRST), encoding="ascii")
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


def _kill_group(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def run(argv: list[str], workdir: Path, timeout_s: float = CHECK_TIMEOUT_S,
        grace_s: float = CHECK_KILL_GRACE_S) -> tuple[int | None, float, int]:
    """Run the check: (exit code, None when the time limit stopped it; seconds; its peak resident bytes, the largest
    single process of it as the kernel reports for a reaped child). Its process group is killed whatever happens."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", IN_CHECK: "1"}
    started = time.monotonic()
    child = subprocess.Popen(argv, cwd=workdir, stdin=subprocess.DEVNULL, env=env, start_new_session=True,
                             preexec_fn=_child_setup)  # noqa: PLW1509 - the submitting tool is single-threaded
    stopping = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.getsignal(sig) for sig in stopping}

    def stop(signum: int, _frame: FrameType | None) -> None:
        _kill_group(child.pid, signal.SIGKILL)
        raise SystemExit(128 + signum)

    code: int | None = None
    peak = 0
    try:
        for sig in stopping:
            signal.signal(sig, stop)
        deadline, killed_at = started + timeout_s, None
        while True:
            pid, status, usage = os.wait4(child.pid, os.WNOHANG)
            if pid:
                child.returncode = os.waitstatus_to_exitcode(status)
                code = None if killed_at is not None else child.returncode
                peak = usage.ru_maxrss * KIB
                break
            if killed_at is None and time.monotonic() >= deadline:
                _kill_group(child.pid, signal.SIGTERM)
                killed_at = time.monotonic()
            elif killed_at is not None and time.monotonic() >= killed_at + grace_s:
                _kill_group(child.pid, signal.SIGKILL)
            time.sleep(POLL_S)
    finally:
        _kill_group(child.pid, signal.SIGKILL)  # whatever the check left running in its group
        if child.returncode is None:
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child.pid, 0)
                child.returncode = -signal.SIGKILL
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code, time.monotonic() - started, peak
