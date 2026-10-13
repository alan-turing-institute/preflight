#!/usr/bin/env python
"""Refuse to start a GPU job on this box while another one is already running.

⛔ **WHY THE RULE.** The GB10's CUDA pool IS system RAM, so two concurrent jobs
do not merely contend -- on 2026-09-01 they took the whole machine down, killing
every terminal session on it. Hence the owner's standing rule: one GPU job at a
time per DGX.

⛔ **WHY THIS FILE IS TRACKED.** The guard existed before, at
`notes/design/project_admin/.../assert_gpu_free.py`, and imported its first leg
from `outputs/eval_queue/preflight.py`. BOTH trees are git-excluded -- `notes/`
carries 0 tracked files and `outputs` is in `.gitignore` -- so every DGX safety
device this project had built lived outside version control, could not be
reviewed, and was re-invented per study. That is the durability half of the
2026-09-16 failure. This is the canonical copy; the old path is a shim onto it.

TWO LEGS, AND THE FIRST CANNOT REPLACE THE SECOND
-------------------------------------------------
* Read `/proc` and require a watched module to appear as an actual ``-m``
  argument of a python process, skipping this process's own ancestors.
  ⛔ D135: `pgrep -f` is unusable here -- any wrapping shell carries the pattern
  in its own command line and the guard finds itself. The `[a]utocast` bracket
  trick fixes self-match but NOT an ancestor; verified, it still matched.
* ⛔⛔ **MEASURED 2026-09-05 14:44: leg 1 is BLIND to a GPU job launched as a
  plain script.** With a task holding 4297 MiB of the GB10 it returned nothing.
  The rule is about the BOX, so leg 2 asks the DEVICE who is on it.

⛔⛔ **PRE-LAUNCH ONLY -- DO NOT PUT THIS INSIDE A PER-CELL LOOP.** It is a
gate to run ONCE, before a wave starts. `task14_w4.py` and `task14_w8_3b.py`
both record the hazard in their own words: the guard "refuses the wave because
THIS process holds the card". Leg 1 skips this process's ancestors, which is
not the same as skipping its children -- a wave that re-checks between cells
would refuse itself from cell 2 onward.

⚠ **The older guard at `notes/design/project_admin/.../assert_gpu_free.py` is
deliberately NOT replaced by this one.** It watches the eval module only; this
one watches training too, which is what "one GPU job at a time" actually means
but is also a STRICTER rule, and three live launchers call the old one. Making
a running launcher stricter without tracing its call sites is how the next
failure gets introduced while fixing the last. Converging them is a follow-up
that needs those call sites read first.

⭐ **A DECLARED RESIDENT (`--resident 'name=reason'`).** Excuses ONE long-lived
process of OUR user whose name contains `name` from the budget -- written for the
owner's `llama-server`, which holds ~21 GB of the GB10 permanently, so a queued
grid's wait for a free card could never succeed on gpu-node-1. Printed on every
run, like `--allow-other-users`, so it cannot become an invisible permanent hole.
Another user's process is never excused by it, and unknown ownership is never
ours. Moved here 2026-09-26 from the older guard's `AUTOCAST_GPU_RESIDENT`
(owner: keep it as standing practice, in the tracked preflight); `dgx_submit.sh`
turns that variable into this flag, since an environment variable does not reach
a guard piped over ssh.

A box whose `nvidia-smi` does not answer refuses launch: GPU occupancy is unknown.

Exit 0 = free. Exit 1 = refused, with the occupant named.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

#: `pid, used_memory, process_name` -- fewer fields means a line that is not a
#: compute-app row (a blank line, or a driver message).
_MIN_CSV_FIELDS = 2

#: Modules that mean "a job of ours is already running".
DEFAULT_MODULES: tuple[str, ...] = (
    "autocast.scripts.eval.encoder_processor_decoder",
    "autocast.scripts.train.encoder_processor_decoder",
)


def ancestors(pid: int) -> set[int]:
    """Every PID from ``pid`` up to init, so the caller cannot detect itself."""
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            break
        # The comm field is parenthesised and may itself contain spaces, so
        # split after the LAST ')': ppid is then the second field.
        try:
            pid = int(stat[stat.rindex(")") + 1 :].split()[1])
        except (ValueError, IndexError):
            break
    return seen


def running_jobs(modules: tuple[str, ...] = DEFAULT_MODULES) -> list[tuple[int, str]]:
    """OUR PIDs running one of ``modules`` as ``-m``, excluding our own ancestry.

    ⛔⛔ **OWNERSHIP IS PART OF THE QUESTION THIS LEG ASKS, and leaving it out
    made the gate's own ``--allow-other-users`` declaration impossible to
    honour.** This leg counts *our* jobs against *our* declared budget; the
    separate "another user is on the card" leg is what speaks for everyone
    else. Matching on the module name alone conflated them, because a
    colleague on a SHARED box runs the same module we do: measured 2026-09-21
    on the Turing DGX, another user's
    ``python3 -m autocast.scripts.train.encoder_processor_decoder`` was counted
    as one of ours, so a correctly-declared queueing request was refused with
    "1 job(s) of ours already running" while we had none. The misdirection is
    the cost -- it blamed our own budget for someone else's job.

    ⚠ Unknown ownership still counts as OURS here, which is the conservative
    direction for a budget leg: it can only over-count and refuse. (The
    other-user leg treats unknown ownership as someone else's, which is the
    conservative direction there. The two differ deliberately.)
    """
    found: list[tuple[int, str]] = []
    mine = ancestors(os.getpid())
    me = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in mine:
            continue
        try:
            raw = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in raw if a]
        if not args or "python" not in Path(args[0]).name:
            continue
        owner = owner_of(pid)
        if owner is not None and owner != me:
            continue
        if any(module in args for module in modules):
            found.append((pid, " ".join(args)[:120]))
    return found


def owner_of(pid: int) -> int | None:
    """The uid owning ``pid``, or None when it cannot be read.

    ⛔ Unknown ownership is treated as SOMEONE ELSE'S by the caller. Guessing
    "probably mine" is how a guard talks itself into sharing a card with a
    stranger, and the measured cost of that is a daemon wedged for 9 h 44 m.
    """
    try:
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def compute_apps() -> list[tuple[int, str]] | None:
    """`(pid, description)` for every compute process, or None if unmeasurable.

    ⚠ On the GB10 the driver reports "Not Supported" for the memory columns, so
    this keys on the PRESENCE of a compute-app row, never on a memory total.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    rows: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < _MIN_CSV_FIELDS or not fields[0]:
            return None
        try:
            rows.append((int(fields[0]), ", ".join(fields[1:])))
        except ValueError:
            return None
    return rows


def classify(rows: list[tuple[int, str]]) -> tuple[list[str], list[str]]:
    """Split compute processes into ours and other people's."""
    me = os.getuid()
    mine: list[str] = []
    theirs: list[str] = []
    for pid, description in rows:
        uid = owner_of(pid)
        line = f"pid {pid} ({description})"
        (mine if uid == me else theirs).append(line)
    return mine, theirs


def parse_resident(raw: str | None) -> tuple[str, str] | None:
    """``(name, reason)`` from a ``--resident`` declaration, or ``None``.

    Raises:
        ValueError: On a declaration missing its name or its reason.
    """
    if raw is None:
        return None
    name, _, reason = raw.partition("=")
    if not name.strip() or not reason.strip():
        msg = f"--resident must be '<process name>=<reason>'; got {raw!r}."
        raise ValueError(msg)
    return name.strip(), reason.strip()


def resident_pids(
    apps: list[tuple[int, str]], resident: tuple[str, str] | None
) -> set[int]:
    """The compute processes ``resident`` excuses: OUR user's, name matching."""
    if resident is None:
        return set()
    me = os.getuid()
    return {
        pid
        for pid, description in apps
        if resident[0] in description.rsplit(",", 1)[-1] and owner_of(pid) == me
    }


def main() -> int:
    """Both legs, cheaper and more specific first, against a declared budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="extra module name that counts as a running job (repeatable)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help=(
            "how many of OUR GPU jobs may run at once, this one included."
            " Default 1 -- the standing rule. Raise it only against a MEASURED"
            " sharing result for this workload."
        ),
    )
    parser.add_argument(
        "--allow-other-users",
        default=None,
        metavar="REASON",
        help=(
            "permit starting while another user is on the card, with a reason."
            " Off by default and it should stay off"
        ),
    )
    parser.add_argument(
        "--resident",
        default=None,
        metavar="NAME=REASON",
        help=(
            "one long-lived process of OUR user, whose name contains NAME, that"
            " is not counted against the budget. Printed on every run"
        ),
    )
    arguments = parser.parse_args()
    modules = DEFAULT_MODULES + tuple(arguments.module)

    if arguments.max_jobs < 1:
        print("[gpu] --max-jobs must be at least 1", file=sys.stderr)
        return 2
    try:
        resident = parse_resident(arguments.resident)
    except ValueError as error:
        print(f"[gpu] {error}", file=sys.stderr)
        return 2

    apps = compute_apps()
    if apps is None:
        print("[gpu] REFUSED: nvidia-smi did not answer; GPU occupancy is unknown.",
              file=sys.stderr)
        return 1
    ours_by_pid = {pid for pid, _ in running_jobs(modules)}
    ours_detail = [f"pid {pid}: {cmd}" for pid, cmd in running_jobs(modules)]
    excused = resident_pids(apps, resident)
    for pid, description in apps:
        if pid in excused:
            print(
                f"[gpu] DECLARED RESIDENT, not counted: pid {pid}"
                f" ({description}) -- {resident[1] if resident else ''}"
            )
    apps = [(pid, d) for pid, d in apps if pid not in excused]
    mine, theirs = classify(apps)
    ours_by_pid |= {pid for pid, _ in apps if owner_of(pid) == os.getuid()}

    # ---- another user on the card: refuse unless explicitly declared --------
    if theirs and arguments.allow_other_users is None:
        print("[gpu] REFUSED: another user is on this card:", file=sys.stderr)
        for line in theirs:
            print(f"[gpu]   {line}", file=sys.stderr)
        print(
            "[gpu] The CUDA pool IS system RAM, and the sharing daemon has a"
            " measured deadlock when a second user's client appears (it tore"
            " our server down and waited 9 h 44 m). Declare it with"
            " --allow-other-users 'reason' only if you mean it.",
            file=sys.stderr,
        )
        return 1
    if theirs:
        declared = arguments.allow_other_users
        print(f"[gpu] DECLARED: sharing with another user -- {declared}")
        for line in theirs:
            print(f"[gpu]   {line}")

    # ---- our own occupancy against the declared budget ----------------------
    running = len(ours_by_pid)
    if running + 1 > arguments.max_jobs:
        print(
            f"[gpu] REFUSED: {running} job(s) of ours already running and"
            f" --max-jobs is {arguments.max_jobs}.",
            file=sys.stderr,
        )
        for line in ours_detail + mine:
            print(f"[gpu]   {line}", file=sys.stderr)
        return 1

    print(
        f"[gpu] OK -- {running} job(s) of ours running, budget {arguments.max_jobs};"
        f" {len(theirs)} other user(s) on the card"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
