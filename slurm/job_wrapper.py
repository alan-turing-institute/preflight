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
5. RECORD the leg's outcome in the job's sidecar (exit status, signal, restart count, peaks), and REPORT what the run
   used (D37): two lines after the exit line, `[job] used: memory peak ... of ... asked; cores ... on average of ...
   held` and `[job] for the same work next time: --mem ... --cpus ...` (`usage_lines`), kept in the leg's `usage` so
   `wait.py` can repeat them. Never per user; the project decides its next request.
6. A KILL BY THE HOST'S OOM GUARD is not a finished run. The guard signals a process of the job directly, not through
   slurm, and a training loop that catches SIGTERM exits 0 (2026-10-22: job 428 was recorded COMPLETED at 14.6 of
   23.5 h and its afterok chain started). The agent records each such kill (`agent/oom_kills/`); when the command ended
   without slurm having signalled the wrapper and such a record names this leg, the job ends KILLED_FOR_MEMORY, and a
   `resume` job is first requeued (`scontrol requeue`, its owner's right), at most KILLED_LEGS_LIMIT legs in a row.
7. LAUNCH RE-CHECK on a part-time machine (`--require`, given only to a job `submit.py` found portable; run only when
   slurm put the job in `part_time`, so never on the main node): the paths and architecture `portability.decide` saw at
   submission must still hold here. If not, the wrapper says why, records a `launch_refused` event in this machine's
   state folder and requeues the job. A job cannot bar a node from itself while it runs (slurm: "Job is no longer
   pending execution", measured 2026-10-31), so `node_check.py` (root) reads the event and, once the job waits again,
   sets its `ExcNodeList` if the event's owner is the job's owner. LAUNCH_REFUSALS_LIMIT refusals in a row fail the job.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import IO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hostguard"))

import cores  # noqa: E402 - path set just above
import gpu_jobs  # noqa: E402
import memory_guard  # noqa: E402
import portability  # noqa: E402
from gpu_jobs import (  # audit.py imports gpu_mib_by_pid from here
    CGROUP_ROOT,
    cgroup_pids,
    gpu_mib_by_pid,
    job_pids,
    read_int,
)

#: Slurm sets this in the environment of a job that asked for shards of a card.
SHARDS_ENV = "SLURM_SHARDS_ON_NODE"
PREFLIGHT = Path(__file__).resolve().parents[1]
PARITY = PREFLIGHT / "dgx" / "assert_code_parity.py"


class Exit(IntEnum):
    """The wrapper's own exit codes; any other code is the command's."""

    NO_COMMAND = 2
    CARD_NOT_READY = 3  # the wait limit passed
    CODE_CHANGED = 4  # the manifest re-check failed
    KILLED_FOR_MEMORY = 5  # the host's OOM guard killed the run, whatever the command's own exit status
    LAUNCH_CHECK_FAILED = 6  # a part-time machine was unfit for the job: requeued, or failed after LAUNCH_REFUSALS_LIMIT in a row


#: After the command ends, how long to wait for the agent's record of an OOM-guard kill. The agent writes it as the
#: guard signals, before the process has exited, so it is normally there already.
KILL_RECORD_WAIT_S = 3.0
#: A kill of a process other than the command itself (a data-loader worker, say) counts only if the command ended this
#: soon after it: the guard's own wait for the process (10 s), slurm's KillWait to save a checkpoint (120 s), and room.
KILL_WINDOW_S = 180.0
#: A `resume` job killed this many legs in a row fails instead of being requeued again: slurm's MaxBatchRequeue covers
#: only failed launches, so a memory shortage that persists would otherwise requeue it for ever.
KILLED_LEGS_LIMIT = 3
#: A job whose launch check fails this many times in a row fails instead of being requeued again: the node's exclusion
#: is applied by `node_check.py` within about a minute, and if it never lands the job would otherwise loop for ever.
LAUNCH_REFUSALS_LIMIT = 3


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


def card_is_free(own_pids: set[int], max_jobs: int = 1, shard_job: bool = False) -> tuple[bool, str]:
    """Free when no compute process outside this job's cgroup is on its GPUs; declared sharing skips the check.

    A shard job (slurm sets SLURM_SHARDS_ON_NODE) shares the card with other slurm jobs by design, so it refuses only
    on a compute process that belongs to no slurm job and no lease (`audit.unregistered`)."""
    if shard_job:
        import audit  # lazy: audit imports this module

        usage = gpu_jobs.gpu_usage()
        if usage is None:
            return False, "nvidia-smi did not answer, so the card's occupancy is unknown"
        stray = audit.unregistered(usage)
        if not stray:
            return True, ""
        return False, "compute process(es) outside slurm and the lease protocol: " + ", ".join(f"pid {p}" for p, _, _ in stray)
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


def job_cgroup(proc_cgroup: Path = Path("/proc/self/cgroup")) -> Path | None:
    """This job's cgroup directory: the component right after slurm's `*slurmstepd.scope`.

    That is `job_<id>` on 23.11 (`<node>_slurmstepd.scope`) and the job's SLUID on 26.05 (`slurmstepd.scope`).
    """
    try:
        line = proc_cgroup.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    parts = line.split("::", 1)[-1].strip("/").split("/")
    scopes = [i for i, part in enumerate(parts) if part.endswith("slurmstepd.scope")]
    if not scopes or scopes[0] + 1 >= len(parts):
        return None
    return CGROUP_ROOT.joinpath(*parts[: scopes[0] + 2])


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


#: How often a running command's wrapper looks for a memory stop of its run (D44): the agent escalates after ACK_S.
STOP_POLL_S = 1.0


def carry_out_stop(job_id: str, pid: int, leg_started: float) -> dict[str, object] | None:
    """A memory stop the agent asked for this leg (D44): acknowledge it in the job's record, then SIGKILL the command's
    process group. The kill description, or None when there is no stop of this leg."""
    request = memory_guard.StopRequest.read(job_id, log=lambda line: print(f"[job] {line}", file=sys.stderr, flush=True))
    if request is None or not request.for_leg(leg_started):
        return None
    now = time.time()
    best_effort("sidecar", lambda: gpu_jobs.update_sidecar(job_id, **request.ack_fields(now)))
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
    print(f"[job] stopped by the memory guard: free memory was {gpu_jobs.size_text(request.free)}, and this run held "
          f"{gpu_jobs.size_text(request.held + request.card)} (memory + card) of its {gpu_jobs.size_text(request.ask)} ask",
          file=sys.stderr, flush=True)
    return request.kill_description(pid, now)


def run_and_measure(
    command: list[str], footprint: Path, cgroup: Path | None, sample_s: float, job_id: str = "", leg_started: float = 0.0
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
    stopped: dict[str, object] | None = None
    # waits in short slices so a memory stop is seen within STOP_POLL_S; samples every sample_s
    next_sample = time.monotonic() + sample_s
    out = best_effort("footprint", open_footprint, footprint)
    with out or contextlib.nullcontext():
        while True:
            try:
                status = child.wait(timeout=min(STOP_POLL_S, sample_s))
                break
            except subprocess.TimeoutExpired:
                pass
            if job_id and stopped is None:  # a run outside slurm never reads requests
                stopped = carry_out_stop(job_id, child.pid, leg_started)
            if time.monotonic() < next_sample:
                continue
            next_sample += sample_s
            row = sample(cgroup, {child.pid}, job_id)
            if out:
                best_effort("footprint", append_row, out, row)
            peak_gpu = max(peak_gpu, float(row["gpu_mib"]))
            peak_mem = max(peak_mem, int(row["mem_peak"] or row["mem_current"] or 0))
    final = read_int(cgroup / "memory.peak") if cgroup else None
    return {
        "exit": status,
        "pid": child.pid,
        "signals": [signal.Signals(s).name for s in received],
        "peak_gpu_mib": peak_gpu,
        "peak_mem_bytes": max(peak_mem, final or 0),
        **({"stopped": stopped} if stopped else {}),
    }


#: Bytes in one of slurm's MB.
SLURM_MB = gpu_jobs.SLURM_MB


def cgroup_counters(cgroup: Path | None) -> tuple[int | None, int | None]:
    """(core time in microseconds, processes the memory cap has killed) of the job's cgroup now, None when unreadable.
    `memory.events` is hierarchical, so a kill in the batch step or a task below the job's cgroup counts."""
    if cgroup is None:
        return None, None
    return gpu_jobs.flat_key(cgroup / "cpu.stat", "usage_usec"), gpu_jobs.flat_key(cgroup / "memory.events", "oom_kill")


@dataclass(frozen=True)
class Usage:
    """What one run asked for and used (D37); None for a figure that could not be read."""

    mem_asked_bytes: int | None
    mem_peak_bytes: int | None  # host side, plus the card's on a site whose card shares system memory
    cores_held: int | None
    cores_avg: float | None
    cap_killed: bool = False  # the job's memory cap killed one of its processes during the run
    guard_killed: bool = False  # the host's OOM guard killed it


def measure_usage(cgroup: Path | None, outcome: dict[str, object], before: tuple[int | None, int | None],
                  after: tuple[int | None, int | None], wall_s: float, *, guard_killed: bool, unified: bool) -> Usage:
    """The run's `Usage` from its cgroup and the peaks `run_and_measure` kept. Asked = the cgroup's memory cap (slurm's
    --mem); "max" (no cap) or no cgroup is unknown, and so is the host peak without a cgroup."""
    asked = gpu_jobs.read_int(cgroup / "memory.max") if cgroup else None
    host = int(outcome.get("peak_mem_bytes") or 0) if cgroup else None
    card = int(float(outcome.get("peak_gpu_mib") or 0) * SLURM_MB)
    try:
        held = len(os.sched_getaffinity(0))
    except OSError:
        held = None
    used_usec = None if None in (before[0], after[0]) else after[0] - before[0]  # type: ignore[operator]
    return Usage(
        mem_asked_bytes=asked,
        mem_peak_bytes=None if host is None else host + (card if unified else 0),
        cores_held=held,
        cores_avg=used_usec / 1e6 / wall_s if used_usec is not None and wall_s > 0 else None,
        cap_killed=None not in (before[1], after[1]) and after[1] > before[1],  # type: ignore[operator]
        guard_killed=guard_killed,
    )


def usage_lines(u: Usage, *, gpu_job: bool, cores_cap: int | None, most_mb: int | None) -> list[str]:
    """The run's report (D37): what it used against what it asked for and, when anything is known, the request for the
    same work next time: memory at its peak plus `AUTO_MARGIN`, or after a memory kill at least `bumped_mem_mb` of what
    it was given (the whole box, `0`, when no node gives that much, as for a run that had the whole box); cores, for a job without GPUs
    only (a GPU job's are held for the card, D17/D33), its average times `CORES_MARGIN` within `cores_cap`."""
    if u.mem_peak_bytes is None:
        memory = "memory peak unknown"
    elif u.mem_asked_bytes:
        memory = (f"memory peak {gpu_jobs.size_text(u.mem_peak_bytes)} of {gpu_jobs.size_text(u.mem_asked_bytes)} asked "
                  f"({round(100 * u.mem_peak_bytes / u.mem_asked_bytes)}%)")
    else:
        memory = f"memory peak {gpu_jobs.size_text(u.mem_peak_bytes)} (asked unknown)"
    if u.cores_avg is None:
        cores = "cores in use unknown" + (f" of {u.cores_held} held" if u.cores_held else "")
    elif u.cores_held:
        cores = f"cores {u.cores_avg:.1f} on average of {u.cores_held} held ({round(100 * u.cores_avg / u.cores_held)}%)"
    else:
        cores = f"cores {u.cores_avg:.1f} on average"
    note = ("; the memory cap killed a process" if u.cap_killed else "") + ("; killed by the host's OOM guard" if u.guard_killed else "")
    lines = [f"[job] used: {memory}; {cores}{note}"]
    flags = []
    asked_mb = u.mem_asked_bytes / SLURM_MB if u.mem_asked_bytes else None
    if u.cap_killed or u.guard_killed:
        if asked_mb is not None:  # a run that had the whole box already gets `0` again: 1.2 times it fits no node
            option = gpu_jobs.mem_option(gpu_jobs.bumped_mem_mb(asked_mb), most_mb)
            flags.append(f"--mem {option}" + (" or more" if option != "0" else ""))
    elif u.mem_peak_bytes is not None:
        flags.append(f"--mem {gpu_jobs.mem_option(u.mem_peak_bytes * (1 + gpu_jobs.AUTO_MARGIN) / SLURM_MB, most_mb)}")
    if not gpu_job and u.cores_avg is not None:
        cpus = max(1, math.ceil(u.cores_avg * gpu_jobs.CORES_MARGIN))
        flags.append(f"--cpus {min(cpus, cores_cap) if cores_cap else cpus}")
    if flags:
        lines.append(f"[job] for the same work next time: {' '.join(flags)}")
    return lines


def usage_record(u: Usage, lines: list[str]) -> dict[str, object]:
    """The leg's `usage`: the known figures, whether the cap killed a process, and the printed lines (`wait.py` repeats them)."""
    figures = {"mem_asked_bytes": u.mem_asked_bytes, "mem_peak_bytes": u.mem_peak_bytes, "cores_held": u.cores_held,
               "cores_avg": None if u.cores_avg is None else round(u.cores_avg, 3)}
    return {**{k: v for k, v in figures.items() if v is not None}, "memory_cap_killed": u.cap_killed, "lines": lines}


def guard_kill(
    job_id: str,
    started: float,
    ended: float,
    child_pid: int,
    *,
    read: Callable[[Path], dict[str, object] | None] = gpu_jobs.read_json,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object] | None:
    """The agent's record of the OOM guard killing this leg's command, waiting up to KILL_RECORD_WAIT_S for it; None
    when there is none. A record from an earlier leg, or of a worker killed long before the command ended, is not one."""

    def counts(record: dict[str, object] | None) -> bool:
        if not isinstance(record, dict) or record.get("job") != job_id or not isinstance(record.get("at"), int | float):
            return False
        at = float(record["at"])  # type: ignore[arg-type]
        return started - 1 <= at <= ended + 1 and (record.get("pid") == child_pid or ended - at <= KILL_WINDOW_S)

    deadline = clock() + KILL_RECORD_WAIT_S
    while True:
        record = read(gpu_jobs.oom_kill_path(job_id))
        if counts(record):
            return record
        if clock() >= deadline:
            return None
        sleep(0.25)


def after_kill(job_id: str, kill: dict[str, object], legs: list[object], job_name: str, run=gpu_jobs.run_slurm) -> int:
    """Say the run was killed; requeue a `resume` job unless its last KILLED_LEGS_LIMIT legs were all killed."""
    in_a_row = 0
    for leg in reversed(legs):
        if not (isinstance(leg, dict) and leg.get("killed_by")):
            break
        in_a_row += 1
    print(f"[job] KILLED: {kill.get('guard')} sent {kill.get('signal')} to pid {kill.get('pid')} to free memory; the run "
          "did not finish, whatever the command's exit status", file=sys.stderr, flush=True)
    if gpu_jobs.class_of(job_name) is not gpu_jobs.JobClass.RESUME:
        print("[job] not a resume job: it fails, and the jobs that wait on it are cancelled", file=sys.stderr, flush=True)
    elif in_a_row >= KILLED_LEGS_LIMIT:
        print(f"[job] killed {in_a_row} runs in a row: it fails instead of being requeued again", file=sys.stderr, flush=True)
    elif _requeued(run, job_id):
        print("[job] requeued: it starts again from its checkpoint, and the jobs that wait on it keep waiting", flush=True)
    else:
        print("[job] slurm refused the requeue: the job fails", file=sys.stderr, flush=True)
    return Exit.KILLED_FOR_MEMORY


def _requeued(run: Callable[[list[str]], int], job_id: str) -> bool:
    try:
        return run(["scontrol", "requeue", job_id]) == 0
    except OSError:  # no scontrol here
        return False


def refuse_launch(job_id: str, problems: list[str], run: Callable[[list[str]], int] = gpu_jobs.run_slurm) -> int:
    """Say why this machine cannot run the job, leave the marker `node_check.py` bars the node by, and requeue; fail the
    job on the LAUNCH_REFUSALS_LIMIT-th refusal in a row, or when the refusal cannot be counted or the requeue is refused."""
    print(f"[job] REFUSED at launch on {socket.gethostname().split('.')[0]}: " + "; ".join(problems), file=sys.stderr, flush=True)
    try:
        record = gpu_jobs.read_json(gpu_jobs.sidecar_path(job_id), gpu_jobs.current_user()) or {}
        so_far = record.get("launch_refusals")
        refusals = (so_far if isinstance(so_far, int) else 0) + 1
        gpu_jobs.update_sidecar(job_id, launch_refusals=refusals)
        if refusals >= LAUNCH_REFUSALS_LIMIT:
            print(f"[job] refused {refusals} times in a row: the job fails instead of being requeued again", file=sys.stderr, flush=True)
            return Exit.LAUNCH_CHECK_FAILED
        gpu_jobs.write_event(job_id, gpu_jobs.EventAction.LAUNCH_REFUSED, node=socket.gethostname().split(".")[0], problems=problems)
    except OSError as error:
        print(f"[job] the refusal could not be recorded ({error}): the job fails", file=sys.stderr, flush=True)
        return Exit.LAUNCH_CHECK_FAILED
    if not _requeued(run, job_id):
        print("[job] slurm refused the requeue: the job fails", file=sys.stderr, flush=True)
    else:
        print("[job] requeued: the node is barred from the job once it waits again", flush=True)
    return Exit.LAUNCH_CHECK_FAILED


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
    parser.add_argument("--require", default=None, metavar="JSON",
                        help="[[role, path, where], ...] that must still hold on a part-time machine (portability.decide)")
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
    cgroup = job_cgroup()
    if arguments.needs_lease or arguments.needs_card:
        ok, reason = wait_for_card(
            needs_lease=arguments.needs_lease and site.lease is not None,
            limit_s=arguments.wait_limit,
            poll_s=arguments.poll,
            card_free=(
                (lambda: card_is_free(cgroup_pids(cgroup) if cgroup else {os.getpid()}, arguments.max_jobs,
                                      shard_job=SHARDS_ENV in os.environ))
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

    if arguments.require and os.environ.get("SLURM_JOB_PARTITION") == gpu_jobs.Partition.PART_TIME:
        if problems := portability.launch_problems(json.loads(arguments.require)):
            return refuse_launch(job_id, problems)
        if (gpu_jobs.read_json(gpu_jobs.sidecar_path(job_id), gpu_jobs.current_user()) or {}).get("launch_refusals"):
            best_effort("sidecar", lambda: gpu_jobs.update_sidecar(job_id, launch_refusals=0))  # in a row: this one passed
        print("[job] part-time launch check passed", flush=True)

    footprint = gpu_jobs.STATE_DIR / "footprints" / f"{job_id}.jsonl"
    leg_started = time.time()  # a stop request older than this is an earlier run's
    started = int(leg_started)
    if site.hostguard:
        # The job is what a memory shortage should end, not a session: 2026-10-18 13:57 earlyoom ended three
        # Claude sessions while a training job holding 68.5 GB of the unified pool sat at oom_score_adj 0 (GPU
        # memory is not in RSS, so its score saw 2.8 GB). Raised here, inherited by the command, so every slurm
        # job is marked whatever environment it loads; hostguard marks only processes in a guarded venv.
        # Imported only here: an install carries hostguard only for a site that turns it on (tools component).
        # Importing it only arms a hook on torch's import; the wrapper never imports torch.
        from hostguard import mark_preferred_victim  # noqa: PLC0415

        mark_preferred_victim()
    # The cores this leg was allowed (the cgroup's cpuset reaches this process), by top speed: a run's speed depends on them on a
    # host whose cores differ, and slurm does not choose among them. Read before the command starts; None on a host of alike cores.
    received = best_effort("cores", lambda: cores.CoreSpeeds.read().received(os.sched_getaffinity(0)))
    # core time and memory-cap kills are read right around the command, on a monotonic clock (D37's report)
    before, clock_start = cgroup_counters(cgroup), time.monotonic()
    outcome = run_and_measure(command, footprint, cgroup, arguments.sample, job_id, leg_started=leg_started)
    after, wall_s = cgroup_counters(cgroup), time.monotonic() - clock_start
    ended = int(time.time())
    # slurm's own SIGTERM (pre-emption, cancel, time limit) ends the job its own way; only an unforwarded one is a kill.
    # A memory stop is a kill even when such a SIGTERM came in the same second (D44).
    stopped = outcome.pop("stopped", None)
    killed = stopped or (None if outcome["signals"] else guard_kill(job_id, started, ended, int(outcome["pid"])))  # type: ignore[call-overload]
    usage = measure_usage(cgroup, outcome, before, after, wall_s, guard_killed=killed is not None, unified=site.unified_memory)
    in_service = os.environ.get("SLURM_JOB_PARTITION") == gpu_jobs.Partition.SERVICE and site.service is not None
    report = usage_lines(
        usage, gpu_job=arguments.needs_card or SHARDS_ENV in os.environ,
        cores_cap=site.service.cores if in_service else site.cpu_job_cap or max((n.cpus for n in site.nodes), default=None),
        most_mb=max((n.jobs_mb for n in site.nodes), default=None),
    )
    outcome.update(leg=restart, started_at=started, ended_at=ended, **({"killed_by": killed} if killed else {}),
                   **({"cores": received} if received else {}), usage=usage_record(usage, report))
    # the local record only: legs copied back from a part-time machine must never be written into a user's own sidecar
    record = gpu_jobs.read_json(gpu_jobs.sidecar_path(job_id), gpu_jobs.current_user()) or {}
    earlier = record.get("legs")
    legs = [*(earlier if isinstance(earlier, list) else []), outcome]
    best_effort("sidecar", lambda: gpu_jobs.update_sidecar(job_id, legs=legs, last_leg=outcome))
    print(f"[job] exit {outcome['exit']}; signals {outcome['signals']}; peak GPU {outcome['peak_gpu_mib']:.0f} MiB",
          flush=True)
    print("\n".join(report), flush=True)  # what the run used, and what to ask for next time (D37)
    if killed:
        return after_kill(job_id, killed, legs, os.environ.get("SLURM_JOB_NAME", ""))
    status = int(outcome["exit"])
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
