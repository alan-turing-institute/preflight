#!/usr/bin/env python3
"""Run one slurm job's command: wait for the card, re-check the code, run, measure, record.

Design 3.4-3.5 (`design notes`). `submit.py` writes an
sbatch script whose only line runs this wrapper. In order:

1. WAIT (GPU and urgent jobs) until the keeper holds slurm's GPU lease (`gpu_lease_keeper.py`; only when
   the site has a lease) and the job's own GPUs show no compute process outside
   its cgroup (`nvidia-smi` inside the job sees only the GPUs slurm gave it; run before this job's own
   command starts, so any other process belongs to something else). Polls
   every `--poll` s up to `--wait-limit` s, then exits non-zero saying what it waited for: the job ends
   FAILED and, under `kill_invalid_depend`, its dependents are cancelled. A SIGTERM here ends it at once.
2. RE-CHECK the submission's manifest (`dgx/assert_code_parity.py --check-manifest`) when there is one,
   so a file changed during the queue wait is caught before launch.
3. RUN the command as a CHILD in its own process group, never `exec`: the wrapper must stay alive to
   sample it. On SIGTERM it passes the signal to the whole group (slurm signals only the job script's own
   process: stage C test 3), records it, and keeps waiting while the child saves its checkpoint, until
   `KillWait` ends.
4. MEASURE every `--sample` s: GPU memory, over all the job's GPUs, of the processes in this job's cgroup
   and of docker containers labelled `slurm-job=<id>` (`nvidia-smi`), and the cgroup's `memory.current`/`memory.peak` (the CPU side; GPU memory is not charged to the cgroup on the
   GB10, trial B2), appended to `footprints/<job id>.jsonl`. The per-job node-exporter textfile
   (`metrics/job_<id>.prom`) is not written here: the wrapper runs as the job's user, and only the agent may write
   that directory, so the agent measures the same numbers itself.
5. RECORD the leg's outcome in the job's sidecar (exit status, signal, restart count, peaks).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path
from typing import IO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hostguard"))

import gpu_jobs  # noqa: E402 - path set just above
from gpu_jobs import (  # audit.py imports gpu_mib_by_pid from here
    CGROUP_ROOT,
    cgroup_pids,
    gpu_mib_by_pid,
    job_pids,
    read_int,
)

PREFLIGHT = Path(__file__).resolve().parents[1]
PARITY = PREFLIGHT / "dgx" / "assert_code_parity.py"


class Exit(IntEnum):
    """The wrapper's own exit codes; any other code is the command's."""

    NO_COMMAND = 2
    CARD_NOT_READY = 3  # the wait limit passed
    CODE_CHANGED = 4  # the manifest re-check failed


def foreign_gpu_processes(own_pids: set[int], output: str | None = None) -> list[tuple[int, str]] | None:
    """(pid, gpu uuid) of every compute process on this job's GPUs that is not in `own_pids`; None when nvidia-smi
    does not answer (missing, hung or failing), so the caller cannot mistake an unknown card for a free one."""
    if output is None:
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if done.returncode != 0:
            return None
        output = done.stdout
    found: list[tuple[int, str]] = []
    for line in output.splitlines():
        pid, _, uuid = line.partition(",")
        try:
            if int(pid) not in own_pids:
                found.append((int(pid), uuid.strip()))
        except ValueError:
            continue
    return found


def card_is_free(own_pids: set[int], max_jobs: int = 1) -> tuple[bool, str]:
    """Free when no compute process outside this job's cgroup is on its GPUs; declared sharing skips the check."""
    if max_jobs > 1:
        return True, ""
    foreign = foreign_gpu_processes(own_pids)
    if foreign is None:  # as dgx/assert_gpu_free.py did: occupancy unknown is not free
        return False, "nvidia-smi did not answer, so the card's occupancy is unknown"
    if not foreign:
        return True, ""
    return False, "compute process(es) outside this job: " + ", ".join(f"pid {p} on {u}" for p, u in foreign)


def wait_for_card(
    *,
    needs_lease: bool,
    limit_s: float,
    poll_s: float,
    lease_admitted=gpu_jobs.slurm_lease_admitted,
    card_free=card_is_free,
    clock=time.monotonic,
    sleep=time.sleep,
) -> tuple[bool, str]:
    """True once the lease is held for slurm (if needed) and the card is free; False with the reason at the limit."""
    start = clock()
    while True:
        if needs_lease and not lease_admitted():
            reason = "the keeper does not hold slurm's GPU lease yet (a service or another project has the card)"
        else:
            free, detail = card_free()
            if free:
                return True, ""
            reason = f"the card is not free: {detail}"
        if clock() - start >= limit_s:
            return False, reason
        sleep(poll_s)


def job_cgroup(job_id: str, proc_cgroup: Path = Path("/proc/self/cgroup")) -> Path | None:
    """This job's cgroup directory (the `job_<id>` component of our own cgroup path)."""
    try:
        line = proc_cgroup.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    parts = line.split("::", 1)[-1].strip("/").split("/")
    marker = f"job_{job_id}"
    if marker not in parts:
        return None
    return CGROUP_ROOT.joinpath(*parts[: parts.index(marker) + 1])


def sample(cgroup: Path | None, fallback_pids: set[int], job_id: str = "") -> dict[str, object]:
    usage = gpu_mib_by_pid()
    pids = job_pids(cgroup, fallback_pids, job_id, set(usage))
    gpu = {pid: mib for pid, mib in usage.items() if pid in pids}
    return {
        "t": int(time.time()),
        "gpu_mib": sum(gpu.values()),
        "gpu_by_pid": gpu,
        "mem_current": read_int(cgroup / "memory.current") if cgroup else None,
        "mem_peak": read_int(cgroup / "memory.peak") if cgroup else None,
    }


_WARNED: set[str] = set()


def best_effort(what: str, action: Callable[..., object], *args: object) -> object | None:
    """Bookkeeping never fails a job. The shared state directories are open to every user, so a record file another
    user planted under this job's id (or a full disk) must cost only the record: one warning line per `what`, and
    the command's own exit status stays the job's."""
    try:
        return action(*args)
    except OSError as error:
        if what not in _WARNED:
            _WARNED.add(what)
            print(f"[job] warning: {what} not recorded: {error}", file=sys.stderr, flush=True)
        return None


def open_footprint(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def append_row(out: IO[str], row: dict[str, object]) -> None:
    out.write(json.dumps(row) + "\n")
    out.flush()


def run_and_measure(
    command: list[str], footprint: Path, cgroup: Path | None, sample_s: float, job_id: str = ""
) -> dict[str, object]:
    received: list[int] = []

    # The command runs in its own process group, and SIGTERM is passed on to all of it. Measured 2026-10-16
    # (stage C test 3): on pre-emption slurm signals only the job script's own process when the job has no
    # `srun` steps (`man slurm.conf`, GraceTime: the signals go to "the tasks and the containing batch
    # script"), so a child that is not told never saves; it is SIGKILLed when KillWait ends.
    # The handler is set before the child starts, so a SIGTERM in between is not lost: it waits in `missed` and is
    # passed on once, as soon as the child exists.
    child: subprocess.Popen[bytes] | None = None
    missed: list[int] = []

    def forward(signum: int, _frame: object) -> None:
        received.append(signum)
        if child is None:
            missed.append(signum)
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signum)

    signal.signal(signal.SIGTERM, forward)
    child = subprocess.Popen(command, start_new_session=True)
    if missed:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, missed[0])
    peak_gpu, peak_mem = 0.0, 0
    out = best_effort("footprint", open_footprint, footprint)
    with out or contextlib.nullcontext():
        while True:
            try:
                status = child.wait(timeout=sample_s)
                break
            except subprocess.TimeoutExpired:
                pass
            row = sample(cgroup, {child.pid}, job_id)
            if out:
                best_effort("footprint", append_row, out, row)
            peak_gpu = max(peak_gpu, float(row["gpu_mib"]))
            peak_mem = max(peak_mem, int(row["mem_peak"] or row["mem_current"] or 0))
    final = read_int(cgroup / "memory.peak") if cgroup else None
    return {
        "exit": status,
        "signals": [signal.Signals(s).name for s in received],
        "peak_gpu_mib": peak_gpu,
        "peak_mem_bytes": max(peak_mem, final or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--needs-lease", action="store_true", help="a GPU or urgent job: wait for slurm's lease")
    parser.add_argument("--needs-card", action="store_true", help="a GPU job: wait for a free card")
    parser.add_argument("--wait-limit", type=float, default=3600.0, help="seconds to wait for the card")
    parser.add_argument(
        "--max-jobs", type=int, default=1, help="GPU jobs allowed on the card, this one included (declared sharing)"
    )
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--sample", type=float, default=5.0)
    parser.add_argument("--manifest", default=None, help="the submission's code manifest, re-checked at launch")
    parser.add_argument("--repo", default=None, help="the project checkout the manifest describes")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        print("[job] no command", file=sys.stderr)
        return Exit.NO_COMMAND
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
    print(f"[job] {job_id} leg {restart}: {' '.join(command)}", flush=True)
    if arguments.max_jobs > 1:
        print(f"[job] DECLARED SHARING: up to {arguments.max_jobs} GPU jobs on the card", flush=True)

    site = gpu_jobs.load_site()
    cgroup = job_cgroup(job_id)
    if arguments.needs_lease or arguments.needs_card:
        ok, reason = wait_for_card(
            needs_lease=arguments.needs_lease and site.lease is not None,
            limit_s=arguments.wait_limit,
            poll_s=arguments.poll,
            card_free=(
                (lambda: card_is_free(cgroup_pids(cgroup) if cgroup else {os.getpid()}, arguments.max_jobs))
                if arguments.needs_card
                else lambda: (True, "")
            ),
        )
        if not ok:
            print(f"[job] REFUSED after {arguments.wait_limit:.0f} s: {reason}", file=sys.stderr, flush=True)
            best_effort("sidecar", lambda: gpu_jobs.update_sidecar(
                job_id, last_leg={"leg": restart, "refused": reason, "ended_at": int(time.time())}))
            return Exit.CARD_NOT_READY
        print("[job] card ready", flush=True)

    if arguments.manifest:
        done = subprocess.run(
            [sys.executable, str(PARITY), "--check-manifest", arguments.manifest, "--local-root", arguments.repo or "."],
            check=False,
        )
        if done.returncode != 0:
            print("[job] REFUSED: the code changed since submission (manifest re-check)", file=sys.stderr, flush=True)
            best_effort("sidecar", lambda: gpu_jobs.update_sidecar(
                job_id, last_leg={"leg": restart, "refused": "manifest", "ended_at": int(time.time())}))
            return Exit.CODE_CHANGED

    footprint = gpu_jobs.STATE_DIR / "footprints" / f"{job_id}.jsonl"
    started = int(time.time())
    if site.hostguard:
        # The job is what a memory shortage should end, not a session: 2026-10-18 13:57 earlyoom ended three
        # Claude sessions while a training job holding 68.5 GB of the unified pool sat at oom_score_adj 0 (GPU
        # memory is not in RSS, so its score saw 2.8 GB). Raised here, inherited by the command, so every slurm
        # job is marked whatever environment it loads; hostguard marks only processes in a guarded venv.
        # Imported only here: an install carries hostguard only for a site that turns it on (tools component).
        # Importing it only arms a hook on torch's import; the wrapper never imports torch.
        from hostguard import mark_preferred_victim  # noqa: PLC0415

        mark_preferred_victim()
    outcome = run_and_measure(command, footprint, cgroup, arguments.sample, job_id)
    outcome.update(leg=restart, started_at=started, ended_at=int(time.time()))
    record = gpu_jobs.read_sidecar(job_id, gpu_jobs.current_user()) or {}
    earlier = record.get("legs")
    legs = [*(earlier if isinstance(earlier, list) else []), outcome]
    best_effort("sidecar", lambda: gpu_jobs.update_sidecar(job_id, legs=legs, last_leg=outcome))
    print(f"[job] exit {outcome['exit']}; signals {outcome['signals']}; peak GPU {outcome['peak_gpu_mib']:.0f} MiB",
          flush=True)
    status = int(outcome["exit"])
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
