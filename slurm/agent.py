#!/usr/bin/env python3
"""The GPU-job service: the jobs of slurm that nobody should have to remember, done every 5 seconds.

Plan 3.5 (`plan notes`). One system service, run as SlurmUser
(`gpujobs-agent.service`). It takes no instruction from user-written content except admin-owned files in
`STATE_DIR/requests/`. Each step:

1. BOOT RE-REGISTRATION, once per boot (`agent/boot_id` against the kernel's boot id): once slurmctld has run
   30 s, `scontrol update nodename=N state=undrain` for every node not drained on purpose. That makes slurmd
   register again, which is what makes slurm purge the jobs the node lost (otherwise it waits ~32 minutes).
2. LEVEL, THEN SUBMISSION ORDER (D18): every pending job that is not held gets the priority its level and job id
   give it (`levels.priority_for`, from a base id kept in the state), so a job keeps its place after a requeue, a
   pause or a slurmctld restart: slurm keeps a priority set this way across a requeue and a restart, and the agent
   sets it again after a release. Every placed priority is above any slurm gives by itself, so a job not placed
   yet (just submitted, or released) waits behind them all. The level is the job's QOS, slurm's own field; every
   level is open to everyone, so reading it grants nothing. An admin's hand-set priority is overwritten at the
   next look: move a job by changing its level (`scontrol update jobid=J qos=high`).
3. RESTORE ORDER after urgent work: `requests/restore_order-*.json` written by an admin names the urgent job
   (`urgent`), the jobs it may pre-empt (`victims`) and the jobs held meanwhile (`held`). The held jobs are released
   once the urgent job has left the queue and every victim is running, gone or held (a pre-empted victim can run
   again only after the urgent job; one slurm did not pre-empt never stopped). A file owned by a non-admin, or
   naming no urgent job, is ignored.
4. FAILURE NOTICES for jobs that ended FAILED, OUT_OF_MEMORY, TIMEOUT or NODE_FAIL: a `FAILED` event, and the
   site's `notify_command` if any (argv, no shell, 30 s). Slurm's accounting is asked at most every 30 s
   (`NOTICES_EVERY_S`).
5. JOB METRICS, when the site's metrics mode is not "off": for each RUNNING job with a cgroup on this node, the
   `metrics/job_<id>.prom` node-exporter textfile (GPU and CPU memory, `user` and `project` from the queue), and
   no such file for a job that is not running. The agent measures and writes them itself because `metrics/` is
   writable by SlurmUser alone: a file in a world-writable directory could name any job or user, and a killed
   job wrapper would leave its file for ever. When `nvidia-smi` does not answer the files stay as they are
   (unknown is not zero) and the error is recorded.
6. OOM GUARD KILLS: a thread follows the host OOM guard's journal (`OomWatch`) and, for each process it signals,
   writes `agent/oom_kills/job_<id>.json` naming the job the process belonged to. The guard signals a job's process
   directly, not through slurm, and a training loop that catches SIGTERM can exit 0: without this record the job
   wrapper would report a killed run as finished (2026-10-22, job 428). Records of jobs no longer queued are removed.
7. HEARTBEAT `agent/agent.json`. When `squeue` does not answer, the queue-dependent parts are skipped and the
   state is kept: the agent never acts on a queue it cannot see.

The requests directory is sticky and world-writable, and the installer makes SlurmUser its owner, which lets the agent
delete a request but not rewrite one (SlurmUser owns none of the files): what it has learnt about one (victims seen
paused, request already released) lives in `agent/state.json`, and deleting the file is tried but not relied on.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # the path is set just above
import levels
import site_config

BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
#: slurmctld must have been up this long before the nodes are re-registered (it needs time to read its state).
CTLD_SETTLE_S = 30.0
NOTIFY_TIMEOUT_S = 30
MIB = 1024 * 1024
#: How many ended jobs are remembered.
ENDED_KEPT = 5000

Runner = Callable[..., subprocess.CompletedProcess[str]]

#: The host's OOM guard (machine/earlyoom); its journal is read by unit, the field journald sets itself (`_SYSTEMD_UNIT`
#: cannot be sent by a client), and the agent's unit is in the `systemd-journal` group for it.
OOM_GUARD_UNIT = "earlyoom.service"
#: earlyoom's line before each kill (`kill.c`, v1.7): the process is still alive when it is written. The name that
#: follows is the process's own (/proc/PID/comm, at most 15 bytes), too short to hold a forged second line.
KILL_LINE = re.compile(r'sending (SIGTERM|SIGKILL) to process (\d+) uid \d+ "')
#: Seconds before a stopped journal follower is started again.
WATCH_RETRY_S = 30.0


class EndedBadly(StrEnum):
    """The `JobState` values of a job's latest accounting record (`gpu_jobs.completion_records`) that earn a failure notice."""

    FAILED = "FAILED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"


#: States of a job that has not ended (a requeued job's latest record is PENDING).
_NOT_ENDED = frozenset(
    {
        *gpu_jobs.State,
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESIZING",
        "SIGNALING",
    }
)


def call(runner: Runner, argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one command through `runner`: text, captured, 30 s; a missing or hung command is a failed result."""
    try:
        return runner(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(argv, 1, "", str(exc))


def read_boot_id() -> str:
    return BOOT_ID_FILE.read_text(encoding="utf-8").strip()


def slurmctld_active_s(
    run: Runner = subprocess.run, monotonic: Callable[[], float] = time.monotonic
) -> float | None:
    """Seconds slurmctld has been active, or None when it is not (or systemd does not say).

    `ActiveEnterTimestampMonotonic` is microseconds on CLOCK_MONOTONIC, which is what `time.monotonic` reads.
    """
    done = call(
        run,
        [
            "systemctl",
            "show",
            "slurmctld",
            "-p",
            "ActiveState",
            "-p",
            "ActiveEnterTimestampMonotonic",
        ],
    )
    fields = dict(
        line.split("=", 1) for line in done.stdout.splitlines() if "=" in line
    )
    if done.returncode != 0 or fields.get("ActiveState") != "active":
        return None
    try:
        return monotonic() - int(fields["ActiveEnterTimestampMonotonic"]) / 1e6
    except (KeyError, ValueError):
        return None


def notify_run(argv: list[str], env: dict[str, str]) -> None:
    subprocess.run(
        argv,
        env={**os.environ, **env},
        timeout=NOTIFY_TIMEOUT_S,
        check=False,
        shell=False,
    )


@dataclass
class State:
    """What the agent has seen, kept in `agent/state.json` so a restart does not forget it."""

    base: int | None = None  # the job id placed priorities count from (levels.rebase)
    ended_seen: list[str] | None = (
        None  # None until the first look: the log's history is not news
    )
    released_requests: list[str] = field(
        default_factory=list
    )  # "<file>:<mtime_ns>" released but not deletable


def _by_id(job_id: str) -> tuple[int, str]:
    """Sort key putting job ids in submission order (shorter id first, then text)."""
    return len(job_id), job_id


#: Ended jobs are asked of slurm's accounting (sacct, through slurmdbd) at most this often, not every loop.
NOTICES_EVERY_S = 30.0
#: Failure notices sent per look; the rest wait for the next, so a burst and a slow notify command (30 s each) can
#: never stall the loop (placing jobs, restore order) for long.
NOTICES_PER_LOOK = 3


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def metrics_text(job: gpu_jobs.QueuedJob, gpu_mib: float, mem_bytes: int) -> str:
    """The job's two gauges in Prometheus text format; project is the first `:` field of the job name."""
    project = job.name.split(":", 1)[0] if ":" in job.name else "(other)"
    labels = f'jobid="{_label(job.job_id)}",user="{_label(job.user)}",project="{_label(project)}"'
    return (
        "# TYPE gpujobs_job_gpu_memory_bytes gauge\n"
        f"gpujobs_job_gpu_memory_bytes{{{labels}}} {int(gpu_mib * MIB)}\n"
        "# TYPE gpujobs_job_cpu_memory_bytes gauge\n"
        f"gpujobs_job_cpu_memory_bytes{{{labels}}} {mem_bytes}\n"
    )


def kill_from_entry(line: str) -> tuple[int, str, float] | None:
    """(pid, signal, wall-clock seconds) from one `journalctl -o json` line of the OOM guard; None for any other line."""
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    message = entry.get("MESSAGE") if isinstance(entry, dict) else None
    if isinstance(message, list):  # journald writes text that is not UTF-8 as a list of bytes
        message = bytes(message).decode("utf-8", errors="replace")
    match = KILL_LINE.match(message) if isinstance(message, str) else None
    if match is None:
        return None
    try:
        at = int(entry["__REALTIME_TIMESTAMP"]) / 1e6
    except (KeyError, ValueError):
        at = time.time()
    return int(match[2]), match[1], at


def job_of_pid(pid: int) -> str | None:
    """The slurm job a live process belongs to: its cgroup, else the docker container labelled with the job."""
    import audit  # noqa: PLC0415 - lazy, as in gpu_jobs.job_pids: audit imports job_wrapper

    cgroup = audit.cgroup_of(pid)
    return gpu_jobs.job_of_cgroup(cgroup) or gpu_jobs.container_label(cgroup, gpu_jobs.JOB_LABEL) or None


class OomWatch:
    """Follows the OOM guard's journal in a thread of its own, so each kill is attributed while its process still
    exists (a 5-s step would often look after the process is gone). `error` holds what stopped the follower last."""

    def __init__(self, on_kill: Callable[[int, str, float], None], unit: str = OOM_GUARD_UNIT) -> None:
        self.on_kill, self.error = on_kill, ""
        self.argv = ["journalctl", "--follow", "--lines=0", "--output=json", f"_SYSTEMD_UNIT={unit}"]

    def start(self) -> None:
        threading.Thread(target=self._run, name="oom-watch", daemon=True).start()

    def _run(self) -> None:
        while True:
            self.follow()
            time.sleep(WATCH_RETRY_S)

    def follow(self, popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen) -> None:
        self.error = ""
        try:
            with popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as follower:
                for line in follower.stdout or ():
                    kill = kill_from_entry(line)
                    if kill is not None:
                        self.on_kill(*kill)
                    elif not line.lstrip().startswith("{"):  # journalctl's own complaint, e.g. no permission
                        self.error = f"journal of {OOM_GUARD_UNIT}: {line.strip()}"
            self.error = self.error or f"journalctl stopped following {OOM_GUARD_UNIT}"
        except OSError as exc:
            self.error = f"journal of {OOM_GUARD_UNIT}: {exc}"


def _strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


class Agent:
    """One step at a time, so every decision can be tested with a fake queue, fake scontrol and a fake clock."""

    def __init__(
        self,
        site: site_config.Site,
        state_dir: Path,
        *,
        read_queue: Callable[[], list[gpu_jobs.QueuedJob] | None] = gpu_jobs.read_queue,
        scontrol: Runner = subprocess.run,
        ctld_active_s: Callable[[], float | None] = slurmctld_active_s,
        clock: Callable[[], float] = time.time,
        boot_id: Callable[[], str] = read_boot_id,
        completion: Callable[
            [], dict[str, dict[str, str]] | None
        ] = gpu_jobs.completion_records,
        notify: Callable[[list[str], dict[str, str]], None] = notify_run,
        gpu_usage: Callable[[], dict[int, float] | None] = gpu_jobs.gpu_usage,
        cgroup_root: Path = gpu_jobs.CGROUP_ROOT,
        pid_job: Callable[[int], str | None] = job_of_pid,
    ) -> None:
        self.site = site
        self.state_dir, self.pid_job = state_dir, pid_job
        self.watch: OomWatch | None = None  # main() starts one; a test drives record_kill itself
        self._pid_jobs: dict[int, str] = {}  # each running job's processes at the last step, for a kill seen late
        self.gpu_usage, self.cgroup_root = gpu_usage, cgroup_root
        self.dir = state_dir / "agent"
        self.metrics = state_dir / "metrics"
        self.requests = state_dir / "requests"
        self.read_queue, self.scontrol, self.ctld_active_s = (
            read_queue,
            scontrol,
            ctld_active_s,
        )
        self.clock, self.boot_id, self.completion, self.notify = (
            clock,
            boot_id,
            completion,
            notify,
        )
        saved = gpu_jobs.read_json(self.dir / "state.json") or {}
        ended = saved.get("ended_seen")
        base = saved.get("base")
        self.state = State(
            base=base if isinstance(base, int) else None,
            ended_seen=_strings(ended) if isinstance(ended, list) else None,
            released_requests=_strings(saved.get("released_requests")),
        )
        beat = gpu_jobs.read_json(self.dir / "agent.json") or {}
        self.boot_reconciled_at = beat.get("boot_reconciled_at")
        self.last_error, self.last_error_at = (
            str(beat.get("last_error") or ""),
            beat.get("last_error_at"),
        )
        self.note = ""
        self._undrained: set[str] = (
            set()
        )  # nodes handled this boot, so a retry after a partial failure skips them
        self._ignored: set[str] = set()  # requests already logged as not from an admin
        self._notices_at = -NOTICES_EVERY_S  # the first step looks at once

    # -- one step -------------------------------------------------------------------------------------------

    def step(self) -> dict[str, object]:
        now = self.clock()
        errors: list[str] = []
        self.note = ""
        self._reconcile_boot(now, errors)
        read_at_ns = time.time_ns()  # wall clock, like a request file's mtime
        queue = self.read_queue()
        if queue is None:
            errors.append("squeue did not answer; keeping the last state")
        else:
            self._place_jobs(queue, errors)
            self._restore_order(queue, read_at_ns, errors)
            self._pid_jobs = self._running_pids(queue)
            self._prune_oom_kills(queue)
        if self.watch is not None and self.watch.error:
            errors.append(self.watch.error)
        if now - self._notices_at >= NOTICES_EVERY_S:
            self._notices_at = now
            self._failure_notices(errors)
        if queue is not None and self.site.metrics.mode != site_config.MetricsMode.OFF:
            self._job_metrics(queue, errors)
        if errors:
            self.last_error, self.last_error_at = "; ".join(errors), int(now)
        return self._persist(now)

    def _persist(self, now: float) -> dict[str, object]:
        s = self.state
        gpu_jobs.write_json(
            self.dir / "state.json",
            {
                "base": s.base,
                "ended_seen": s.ended_seen,
                "released_requests": s.released_requests,
            },
        )
        beat: dict[str, object] = {
            "updated": int(now),
            "boot_reconciled_at": self.boot_reconciled_at,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "note": self.note,
        }
        gpu_jobs.write_json(self.dir / "agent.json", beat)
        return beat

    # -- 1. boot re-registration ----------------------------------------------------------------------------

    def _node_drained(self, node: str) -> bool | None:
        """Whether `node` is drained or draining; None when slurm does not say."""
        done = call(self.scontrol, ["scontrol", "show", "node", node, "-o"])
        match = (
            re.search(r"\bState=(\S+)", done.stdout) if done.returncode == 0 else None
        )
        return None if match is None else "DRAIN" in match[1]

    def _reconcile_boot(self, now: float, errors: list[str]) -> None:
        boot = self.boot_id()
        marker = self.dir / "boot_id"
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == boot:
            return
        up = self.ctld_active_s()
        if up is None or up < CTLD_SETTLE_S:
            self.note = (
                "waiting for slurmctld to settle before re-registering the nodes"
            )
            return
        nodes = gpu_jobs.nodes(self.scontrol)
        if not nodes:
            errors.append("boot re-registration: slurm lists no nodes; will retry")
            return
        for node in nodes:
            if node in self._undrained:
                continue
            drained = self._node_drained(node)
            if drained is None:
                errors.append(
                    f"boot re-registration: no state for node {node}; will retry"
                )
                continue
            if (
                not drained
                and call(
                    self.scontrol,
                    ["scontrol", "update", f"nodename={node}", "state=undrain"],
                ).returncode
            ):
                errors.append(
                    f"boot re-registration: undrain of {node} failed; will retry"
                )
                continue
            self._undrained.add(node)  # a drained node was left alone on purpose
        if self._undrained.issuperset(nodes):
            self.dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(boot + "\n", encoding="utf-8")
            self.boot_reconciled_at = int(now)
            self._undrained.clear()

    # -- 2. level, then submission order ------------------------------------------------------------------

    def _place_jobs(self, queue: list[gpu_jobs.QueuedJob], errors: list[str]) -> None:
        self.state.base = levels.rebase(self.state.base, [job.job_id for job in queue])
        for job in queue:
            if job.state == gpu_jobs.State.PENDING and job.priority:  # 0 is slurm's hold, which refuses a priority
                self._place(job, errors)

    def _place(self, job: gpu_jobs.QueuedJob, errors: list[str]) -> None:
        target = levels.priority_for(job.level, job.job_id, self.state.base or 0)
        if job.priority == target:
            return
        if call(self.scontrol, ["scontrol", "update", f"jobid={job.job_id}", f"priority={target}"]).returncode:
            errors.append(f"placing job {job.job_id} failed; will retry")
            return
        print(f"[agent] job {job.job_id} placed at {job.level} (priority {job.priority} -> {target})", flush=True)

    # -- 3. restore order after urgent work -----------------------------------------------------------------

    def _restore_order(
        self, queue: list[gpu_jobs.QueuedJob], read_at_ns: int, errors: list[str]
    ) -> None:
        s = self.state
        jobs = {job.job_id: job for job in queue}
        present: set[str] = set()
        for path in sorted(self.requests.glob("restore_order-*.json")):
            try:
                written_ns = path.stat().st_mtime_ns
                key, owner = f"{path.name}:{written_ns}", gpu_jobs._owner(path)
            except OSError:
                continue
            present.add(key)
            if not site_config.is_admin(self.site, owner):
                if path.name not in self._ignored:
                    self._ignored.add(path.name)
                    print(
                        gpu_jobs.printable(f"[agent] ignoring {path}: owned by {owner}, not an admin"),
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            if written_ns >= read_at_ns:
                # written after this step read the queue (urgent.py submits, then writes): that queue may lack the
                # urgent job, so the request would look finished. The next step judges it.
                continue
            order = gpu_jobs.read_json(path)
            if not order or key in s.released_requests:
                continue
            urgent = str(order.get("urgent", ""))
            if not gpu_jobs.JOB_ID.fullmatch(urgent):
                if path.name not in self._ignored:
                    self._ignored.add(path.name)
                    print(f"[agent] ignoring {path}: it names no urgent job", file=sys.stderr, flush=True)
                continue
            victims = [v for v in _strings(order.get("victims")) if gpu_jobs.JOB_ID.fullmatch(v)]
            # A pre-empted victim is requeued and can run again only once the urgent job's resources are free; a
            # victim slurm did not pre-empt never stopped. So: the urgent job gone, then each victim running, gone,
            # or held (a held one does not start by itself, and must not keep the waiting jobs held forever).
            if urgent in jobs or any(
                v in jobs and not (jobs[v].active or jobs[v].held) for v in victims
            ):
                continue
            held = [
                job_id
                for job_id in _strings(order.get("held"))
                if gpu_jobs.JOB_ID.fullmatch(job_id)
            ]
            if not self._release_in_order(held, jobs, errors):
                errors.append(f"releasing {held} for {path.name} failed; will retry")
                continue
            print(f"[agent] released {held}: the pre-empted jobs run again", flush=True)
            s.released_requests.append(key)
            gpu_jobs.remove_quietly(path)
        s.released_requests = [key for key in s.released_requests if key in present]

    def _release_in_order(self, held: list[str], jobs: dict[str, gpu_jobs.QueuedJob], errors: list[str]) -> bool:
        """Release one job at a time, best place first, each placed at once: a release starts a scheduling pass, and
        with the card free the first job released takes it."""
        def place_of(job_id: str) -> int:
            job = jobs.get(job_id)
            return levels.priority_for(job.level, job_id, self.state.base or 0) if job else 0

        ok = True
        for job_id in sorted(held, key=place_of, reverse=True):
            if call(self.scontrol, ["scontrol", "release", job_id]).returncode:
                ok = False  # slurm refuses a job it no longer knows, or one not held: the rest still go
                continue
            if job_id in jobs:
                self._place(jobs[job_id], errors)  # held in this queue read: priority 0, never its target
        return ok

    # -- 4. failure notices ---------------------------------------------------------------------------------

    def _failure_notices(self, errors: list[str]) -> None:
        s = self.state
        records = self.completion()
        if records is None:
            errors.append("sacct did not answer; failure notices wait")
            return
        ended = {
            job_id: fields
            for job_id, fields in records.items()
            if fields.get("JobState") not in _NOT_ENDED
        }
        if (
            s.ended_seen is None
        ):  # first look at a log: what is already in it is history, not news
            s.ended_seen = sorted(ended, key=_by_id)[-ENDED_KEPT:]  # even when none: the next failure is news
            return
        # A job requeued after it ended (its latest record is a pending leg) may end again, and be news again.
        known = set(s.ended_seen) & set(ended) | (set(s.ended_seen) - set(records))
        for job_id in sorted(set(ended) - known, key=_by_id)[:NOTICES_PER_LOOK]:
            fields = ended[job_id]
            state = fields.get("JobState", "")
            if state in set(EndedBadly):
                name, user = (
                    fields.get("Name", ""),
                    fields.get("UserId", "").split("(")[0],
                )
                gpu_jobs.write_event(
                    job_id, gpu_jobs.EventAction.FAILED, state=state, name=name
                )
                self._notify(job_id, state, user, name, errors)
            known.add(job_id)
        s.ended_seen = sorted(known, key=_by_id)[-ENDED_KEPT:]

    def _notify(
        self, job_id: str, state: str, user: str, name: str, errors: list[str]
    ) -> None:
        if not self.site.notify_command:
            return
        env = {
            "GPUJOBS_JOB": job_id,
            "GPUJOBS_STATE": state,
            "GPUJOBS_USER": user,
            "GPUJOBS_NAME": gpu_jobs.printable(name),  # what the submitter typed
        }
        try:
            self.notify(list(self.site.notify_command), env)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"notify command for job {job_id} failed: {exc}")

    # -- 6. OOM guard kills ----------------------------------------------------------------------------------

    def record_kill(self, pid: int, signal_name: str, at: float) -> None:
        """Called from the watch thread for each process the OOM guard signals."""
        job_id = self.pid_job(pid) or self._pid_jobs.get(pid)
        if job_id is None:
            print(f"[agent] {OOM_GUARD_UNIT} sent {signal_name} to pid {pid}, outside every slurm job", flush=True)
            return
        record = {"job": job_id, "pid": pid, "signal": signal_name, "at": at, "guard": OOM_GUARD_UNIT}
        try:
            gpu_jobs.write_json(gpu_jobs.oom_kill_path(job_id, self.state_dir), record)
        except OSError as exc:
            print(f"[agent] recording the kill of pid {pid} (job {job_id}) failed: {exc}", file=sys.stderr, flush=True)
            return
        print(f"[agent] {OOM_GUARD_UNIT} sent {signal_name} to pid {pid} of job {job_id}", flush=True)

    def _running_pids(self, queue: list[gpu_jobs.QueuedJob]) -> dict[int, str]:
        found: dict[int, str] = {}
        for job in queue:
            cgroup = self._job_cgroup(job.job_id) if job.state == gpu_jobs.State.RUNNING else None
            found.update(dict.fromkeys(gpu_jobs.cgroup_pids(cgroup), job.job_id) if cgroup else {})
        return found

    def _prune_oom_kills(self, queue: list[gpu_jobs.QueuedJob]) -> None:
        """A job's record is read by its wrapper while the job is still queued; once it has left, the record goes."""
        queued = {job.job_id for job in queue}
        for path in (self.dir / "oom_kills").glob("job_*.json"):
            if path.stem.removeprefix("job_") not in queued:
                gpu_jobs.remove_quietly(path)

    # -- 5. job metrics -------------------------------------------------------------------------------------

    def _job_cgroup(self, job_id: str) -> Path | None:
        """The job's cgroup under slurm's scope (`system.slice/<node>_slurmstepd.scope`); None on another node."""
        return next(iter(self.cgroup_root.glob(f"system.slice/*_slurmstepd.scope/job_{job_id}")), None)

    def _job_metrics(self, queue: list[gpu_jobs.QueuedJob], errors: list[str]) -> None:
        # a site without GPUs has no nvidia-smi to ask: its jobs' GPU memory is 0, not unknown
        usage = self.gpu_usage() if any(node.gpus for node in self.site.nodes) else {}
        if usage is None:
            errors.append("nvidia-smi did not answer; the job metrics files stay as they are")
            return
        texts: dict[str, str] = {}
        for job in queue:
            cgroup = self._job_cgroup(job.job_id) if job.state == gpu_jobs.State.RUNNING else None
            if cgroup is None:
                continue
            pids = gpu_jobs.job_pids(cgroup, set(), job.job_id, set(usage))
            mem = gpu_jobs.read_int(cgroup / "memory.current") or 0
            gpu_mib = sum(mib for pid, mib in usage.items() if pid in pids)
            texts[f"job_{job.job_id}.prom"] = metrics_text(job, gpu_mib, mem)
        try:
            self.metrics.mkdir(exist_ok=True)
            for name, text in texts.items():
                temp = self.metrics / f"{name}.tmp"
                temp.write_text(text, encoding="utf-8")  # write, then rename: the exporter never reads half a file
                temp.replace(self.metrics / name)
            for path in self.metrics.glob("job_*.prom*"):
                if path.name not in texts:
                    gpu_jobs.remove_quietly(path)
        except OSError as exc:
            errors.append(f"job metrics: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--interval", type=float, default=5.0, help="seconds between steps"
    )
    parser.add_argument(
        "--once", action="store_true", help="run one step, print the heartbeat and exit"
    )
    arguments = parser.parse_args()

    agent = Agent(site_config.load(), gpu_jobs.STATE_DIR)
    agent.watch = OomWatch(agent.record_kill)
    agent.watch.start()
    while True:
        beat = agent.step()
        if arguments.once:
            print(beat)
            return 0
        time.sleep(arguments.interval)


if __name__ == "__main__":
    sys.exit(main())
