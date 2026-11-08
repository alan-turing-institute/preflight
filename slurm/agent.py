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
3a. PART-TIME MACHINES (D22, sites with part-time nodes): `requests/part_time-<node>-<ns>.json` written by an admin from
   the page: `accept` (resume the node, only when the page's own drain is what holds it, never a health-check or
   foreign one), `close` (drain, reason `gpujobs-admin: closed from the page`), `give_back` (close, then requeue every
   `resume` job running there), `thresholds` (the health check's limits, kept in the node's `Extra` and set again
   whenever slurm's copy differs) and `resources` (jobs' memory and kept cores: rewrites `part_time_nodes.conf`, which
   slurm.conf includes, then `scontrol reconfigure`). A request from a non-admin, or naming no part-time node, is ignored.
3b. ROOM FOR SERVICE JOBS (D20, sites with `[service]`; `gpu_jobs.plan_room` decides): a `service` job (a background
   service's CPU stand-in, placed in the high band) waiting for cores that fits in memory gets them from the `cpu`
   jobs its owner agreed to, which the agent suspends (`scontrol suspend`: slurm takes back their cores, keeps their
   memory, and starts the service job at once), newest first, after holding every waiting `cpu` job (one at a time)
   so none takes the freed cores. GPU and urgent jobs are never held or suspended. A suspended job is resumed only
   when its own cores are free (slurm resumes a job on the cores it had, unchecked: 23.11 `_job_resume_test`), so if
   a GPU job took some meanwhile it waits for that job (the heartbeat says so); then the holds are released. A
   service job not started 60 s after room was made gives it back, and is helped again 5 minutes later. Every change
   is saved first (write-ahead), so a crash cannot leave a suspended job the agent does not know. Slurm's own suspending pre-emption (PreemptMode=SUSPEND) is not used:
   it needs gang scheduling, whose second pre-emption pass killed a `normal` job slurm had just requeued (harness,
   2026-10-23; slurm 23.11.4 `gang.c` `_preempt_job_dequeue` kills a job it cannot requeue, unchanged upstream).
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
import dataclasses
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
import jobs_view
import levels
import memory_guard
import node_check
import site_config

sys.path.insert(0, str(Path(__file__).resolve().parent / "install"))  # the installer ships render.py there

import render  # noqa: E402 - path set just above

BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
#: slurmctld must have been up this long before the nodes are re-registered (it needs time to read its state).
KILL_TASK_FAILED = "Kill task failed"  # slurm's own drain reason (slurmctld log, 2026-11-07: "Kill task failed (JobId=2187 StepId=batch)")
CTLD_SETTLE_S = 30.0
#: How long a service job may take to start once room is made for it; then the room is given back (step 3b).
ROOM_WAIT_S = 60.0
#: How long a service job that did not start is passed over before room is made for it again.
ROOM_RETRY_S = 300.0
NOTIFY_TIMEOUT_S = 30
MIB = 1024 * 1024
#: How often a unified-memory node records the memory used outside its jobs (`status.py --reserve` reads it, D44).
OUTSIDE_EVERY_S = 60.0
#: Days of that record kept.
OUTSIDE_KEEP_DAYS = 14
DAY_S = 86400
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


#: The page's own drain of a part-time node; `accept` resumes a node only while its reason starts so.
ADMIN_REASON_PREFIX = "gpujobs-admin: "
CLOSED_FROM_PAGE = f"{ADMIN_REASON_PREFIX}closed from the page"
PART_TIME_REQUEST = re.compile(r"part_time-(.+)-(\d+)\.json")
#: The node `Extra` keys the health check reads (node_check.EXTRA_KEYS), by the request field that sets each.
THRESHOLD_FIELDS = {"close_below_gb": "close_below_mb", "push_back_below_gb": "push_back_below_mb"}
MAX_THRESHOLD_GB = 1000
#: Job state is copied back from each part-time machine this often (plan B5b); rsync gives up after its own timeout.
COPY_EVERY_S = 60.0
COPY_IO_TIMEOUT_S = 20
COPY_CALL_TIMEOUT_S = 300
COPY_KEY = Path(render.COPY_KEY)  # under the state dir; `install.py part-time add` makes it
KNOWN_HOSTS = COPY_KEY.parent / "known_hosts"  # and this: the node's host key from the admin's own known_hosts (B7.1)
#: What the copy takes from a node, as plain files only: no permissions or links from the node, nothing big, and only
#: the state directories (the node's own manifest in `owners/` says who owned each file there).
COPY_OPTIONS = [
    "--no-perms", "--chmod=D755,F644", "--no-links", "--max-size=50M",
    *(f"--include=/{sub}/***" for sub in ("jobs", "events", "logs", "footprints", "owners")), "--exclude=*",
]
#: rsync's "some files could not be transferred" and "some files vanished": the rest was copied.
RSYNC_PARTIAL = frozenset({23, 24})
RESUME_PROTECTED = frozenset({gpu_jobs.Partition.URGENT, gpu_jobs.Partition.SERVICE})

#: The idle check (Part C2) looks this often, and tells one project about one holder at most this often.
IDLE_EVERY_S = 600.0
IDLE_NOTICE_EVERY_S = 3600.0
#: Holders named per finding (and told per look), largest first.
IDLE_HOLDERS = 3
#: Waiting for room, as opposed to a dependency, a hold or a begin time.
IDLE_WAITING_REASONS = frozenset({gpu_jobs.Reason.RESOURCES, gpu_jobs.Reason.PRIORITY})
#: Partitions whose `resume` jobs slurm requeues for a waiting job; a `cpu` (restart) job is never moved.
MOVABLE_PARTITIONS = frozenset({gpu_jobs.Partition.NORMAL, gpu_jobs.Partition.CPU_FILL, gpu_jobs.Partition.PART_TIME})


class NotifyEvent(StrEnum):
    """What a call of the site's notify command is about (`GPUJOBS_EVENT`)."""

    FAILED = "failed"  # a job ended FAILED, OUT_OF_MEMORY, TIMEOUT or NODE_FAIL
    IDLE = "idle"  # a waste finding (D40): a job waits beside hardware a moved job would put to work
    OVER_ASK = "over_ask"  # a running job holds more memory (host + card) than it asked for (D44)
    MEMORY_STOP = "memory_stop"  # the memory guard stopped a job, and how it ended (D44)


class NotifyRole(StrEnum):
    """Which side of an idle finding a notice is addressed to (`GPUJOBS_ROLE`)."""

    WAITING = "waiting"
    HOLDING = "holding"


class IdleRule(StrEnum):
    """What a waiting job lacks while the machine has something idle."""

    MEMORY = "memory"
    CORES = "cores"
    CPU_CORE_CAP = "cpu_core_cap"


class PartTimeAction(StrEnum):
    """The `action` of a `part_time-<node>-*.json` request."""

    ACCEPT = "accept"
    CLOSE = "close"
    GIVE_BACK = "give_back"
    THRESHOLDS = "thresholds"
    RESOURCES = "resources"


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


def call(runner: Runner, argv: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """Run one command through `runner`: text, captured, 30 s unless told; a missing or hung command is a failed result."""
    try:
        return runner(argv, capture_output=True, text=True, timeout=timeout, check=False)
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
class Holder:
    job: str
    project: str
    amount: str  # with its unit: "24.0 GB", "8 cores", "1 GPU"
    job_class: str
    movable: bool

    @property
    def text(self) -> str:
        return (f"#{self.job} {self.project} holds {self.amount} "
                f"({self.job_class}, {'can be moved' if self.movable else 'cannot be moved'})")


@dataclass
class IdleFinding:
    waiting: str
    project: str
    lacks: str
    rule: IdleRule
    holders: list[Holder]
    idle: str  # what is unused meanwhile
    since: int = 0

    @property
    def text(self) -> str:
        held = "; ".join(h.text for h in self.holders) or "no running job holds it"
        return gpu_jobs.printable(f"Waiting: #{self.waiting} needs {self.lacks}; {held}")


def _gb(mb: float) -> float:
    """Slurm's MB (binary) as decimal GB."""
    return mb * 1048576 / 1e9


def _holder(job: gpu_jobs.QueuedJob, amount: str) -> Holder:
    cls = job.job_class
    movable = cls is gpu_jobs.JobClass.RESUME and job.partition in MOVABLE_PARTITIONS
    return Holder(job.job_id, jobs_view.project_of(job.name), amount, cls.value if cls else "unknown class", movable)


def idle_findings(queue: list[gpu_jobs.QueuedJob], total: dict[str, int], cpu_cap: int | None,
                  total_for: Callable[[str], dict[str, int]] | None = None) -> list[IdleFinding]:
    """One finding per waiting job beside hardware it could use that sits idle for a reason moving a job would cure (D40).

    Two cases count, judged on the nodes the job can start on (`total_for(partitions)`; `total` when not given): the GPU
    unreserved while a job that wants it (a GPU or shards) waits for memory or cores, and cores free while a cpu-only job
    fits in them and waits only for the cpu partition's core cap. Spare memory beside full cores, or spare cores beside
    full memory, is packing, not a finding: moving a job would only change who uses the full resource. Holders are the
    running jobs holding what the job lacks, largest first."""
    active = [j for j in queue if j.active]
    in_cpu = [j for j in active if j.partition == gpu_jobs.Partition.CPU]
    found = []
    for job in queue:
        if job.state != gpu_jobs.State.PENDING or job.reason not in IDLE_WAITING_REASONS:
            continue
        mine = total_for(job.partition) if total_for else total
        free_mem, free_cpus = mine["mem_mb"] - mine["mem_used_mb"], mine["cpus"] - mine["cpus_used"]
        gpu_idle = bool(mine["gpus"]) and not mine["gpus_used"] and not mine.get("shards_used", 0)
        need_mem = mine["mem_mb"] if job.mem_mb == 0 else job.mem_mb or 0
        cpu_only = not job.gpus and not job.shards and gpu_jobs.Partition.CPU in job.partition.split(",")
        held_mem = lambda j: mine["mem_mb"] if j.mem_mb == 0 else j.mem_mb or 0  # noqa: E731
        if (job.gpus or job.shards) and gpu_idle:
            idle = "the GPU unreserved"
            options = (
                (IdleRule.MEMORY, need_mem > free_mem, "all memory" if job.mem_mb == 0 else f"{_gb(need_mem):.1f} GB memory",
                 [(held_mem(j), f"{_gb(held_mem(j)):.1f} GB", j) for j in active]),
                (IdleRule.CORES, job.cpus > free_cpus, f"{job.cpus} cores", [(j.cpus, f"{j.cpus} cores", j) for j in active]),
            )
        elif cpu_only and cpu_cap is not None and need_mem <= free_mem and job.cpus <= free_cpus:
            idle = f"{free_cpus} cores free"
            options = ((IdleRule.CPU_CORE_CAP, job.cpus + sum(j.cpus for j in in_cpu) > cpu_cap,
                        f"{job.cpus} cores under the cpu cap of {cpu_cap}", [(j.cpus, f"{j.cpus} cores", j) for j in in_cpu]),)
        else:
            continue
        for rule, lacking, lacks, held in options:
            if lacking:
                holders = [_holder(j, label) for amount, label, j in sorted(held, key=lambda h: -h[0])
                           if amount > 0 and j.job_id != job.job_id][:IDLE_HOLDERS]
                found.append(IdleFinding(job.job_id, jobs_view.project_of(job.name), lacks, rule, holders, idle))
                break
    return found


#: Waiting jobs named on an idle summary line; the rest are counted.
IDLE_JOBS_NAMED = 6
_IDLE_WAITS_FOR = {IdleRule.MEMORY: "memory", IdleRule.CORES: "cores", IdleRule.CPU_CORE_CAP: "the cpu partition's core cap"}


def idle_summary(found: list[dict[str, object]]) -> list[str]:
    """The findings as one line per (what the jobs wait for, what is unused, who holds it), for the page and status.py."""
    groups: dict[tuple[str, str, str], list[str]] = {}
    for f in found:
        holders = f.get("holders") if isinstance(f.get("holders"), list) else []
        held = "; ".join(
            f"#{h.get('job')} {h.get('project')} holds {h.get('amount')} ({h.get('job_class')}, "
            f"{'can be moved' if h.get('movable') else 'cannot be moved'})" for h in holders if isinstance(h, dict)
        ) or "nobody"
        groups.setdefault((str(f.get("rule", "")), str(f.get("idle", "")), held), []).append(str(f.get("waiting", "?")))
    lines = []
    for (rule, idle, held), waiting in groups.items():
        named = ", ".join(f"#{w}" for w in waiting[:IDLE_JOBS_NAMED])
        more = f" and {len(waiting) - IDLE_JOBS_NAMED} more" if len(waiting) > IDLE_JOBS_NAMED else ""
        jobs = f"Job {named}" if len(waiting) == 1 else f"{len(waiting)} jobs ({named}{more})"
        verb = "waits" if len(waiting) == 1 else "wait"
        lines.append(gpu_jobs.printable(
            f"{jobs} {verb} for {_IDLE_WAITS_FOR.get(rule, rule)} with {idle}; held by {held}"))
    return lines


def read_idle(state_dir: Path | None = None) -> list[dict[str, object]]:
    """The idle findings the agent last wrote (`agent/agent.json`), as plain dicts; [] when there are none."""
    beat = gpu_jobs.read_json((state_dir or gpu_jobs.STATE_DIR) / "agent" / "agent.json") or {}
    found = beat.get("idle")
    return [f for f in found if isinstance(f, dict)] if isinstance(found, list) else []


def read_copied(state_dir: Path | None = None) -> dict[str, int]:
    """When the agent last copied each part-time machine's job state (`agent/agent.json`), by node; {} for none."""
    beat = gpu_jobs.read_json((state_dir or gpu_jobs.STATE_DIR) / "agent" / "agent.json") or {}
    found = beat.get("copied")
    return {k: v for k, v in found.items() if isinstance(v, int)} if isinstance(found, dict) else {}


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
    suspended: list[str] = field(default_factory=list)  # cpu jobs the agent suspended for a service job (3b)
    held: dict[str, int] = field(default_factory=dict)  # job -> when (ns) the agent held it while one is suspended
    helping: dict[str, float] = field(default_factory=dict)  # service job -> when the agent first suspended for it
    given_up: dict[str, float] = field(default_factory=dict)  # service job -> when it did not start once room was made
    part_time_done: list[str] = field(default_factory=list)  # "<file>:<mtime_ns>" of part-time requests carried out
    part_time_extra: dict[str, dict[str, object]] = field(default_factory=dict)  # node -> health-check limits the page set
    part_time_resources: dict[str, dict[str, object]] = field(default_factory=dict)  # node -> jobs_memory_gb, keep_cpus
    idle_since: dict[str, int] = field(default_factory=dict)  # "<waiting job>:<rule>" -> when the finding first showed
    idle_told: dict[str, float] = field(default_factory=dict)  # "<waiting job>:<holder job>" -> when it was last told
    idle_recorded: list[str] = field(default_factory=list)  # "<waiting job>:<holder job>" pairs already written as events


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


def _cpu_seconds(cgroup: Path) -> float | None:
    """CPU time the job's cgroup has used so far (`cpu.stat` usage_usec); None when it cannot be read."""
    usec = gpu_jobs.flat_key(cgroup / "cpu.stat", "usage_usec")
    return None if usec is None else usec / 1e6


def metrics_text(
    job: gpu_jobs.QueuedJob, gpu_mib: float, mem_bytes: int, cpu_seconds: float | None = None, cores_now: float | None = None,
    mem_peak_bytes: int | None = None, cores_peak: float | None = None, mem_avg_bytes: int | None = None,
) -> str:
    """The job's gauges (and its CPU seconds so far, when the cgroup says) in Prometheus text format; project is the first `:` field of the job name."""
    project = job.name.split(":", 1)[0] if ":" in job.name else "(other)"
    labels = f'jobid="{_label(job.job_id)}",user="{_label(job.user)}",project="{_label(project)}"'
    return (
        "# TYPE gpujobs_job_gpu_memory_bytes gauge\n"
        f"gpujobs_job_gpu_memory_bytes{{{labels}}} {int(gpu_mib * MIB)}\n"
        "# TYPE gpujobs_job_cpu_memory_bytes gauge\n"
        f"gpujobs_job_cpu_memory_bytes{{{labels}}} {mem_bytes}\n"
        + (
            "# TYPE gpujobs_job_cpu_seconds_total counter\n"
            f"gpujobs_job_cpu_seconds_total{{{labels}}} {cpu_seconds:.3f}\n"
            if cpu_seconds is not None
            else ""
        )
        + (
            "# TYPE gpujobs_job_cpu_memory_avg_bytes gauge\n"
            f"gpujobs_job_cpu_memory_avg_bytes{{{labels}}} {mem_avg_bytes}\n"
            if mem_avg_bytes
            else ""
        )
        + (
            "# TYPE gpujobs_job_cpu_memory_peak_bytes gauge\n"
            f"gpujobs_job_cpu_memory_peak_bytes{{{labels}}} {mem_peak_bytes}\n"
            if mem_peak_bytes
            else ""
        )
        + (
            "# TYPE gpujobs_job_cpu_cores_peak gauge\n"
            f"gpujobs_job_cpu_cores_peak{{{labels}}} {cores_peak:.3f}\n"
            if cores_peak is not None
            else ""
        )
        + (
            "# TYPE gpujobs_job_cpu_cores gauge\n"
            f"gpujobs_job_cpu_cores{{{labels}}} {cores_now:.3f}\n"
            if cores_now is not None
            else ""
        )
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
    """The slurm job a live process belongs to, by its cgroup."""
    # a docker container's process is never attributed (D44: a label can be forged, and an OOM kill attributed to a job
    # ends its run as KILLED_FOR_MEMORY): only the cgroup slurm made counts
    return gpu_jobs.job_of_cgroup(gpu_jobs.cgroup_of(pid)) or None


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


def may_suspend(job: gpu_jobs.QueuedJob) -> frozenset[str] | None:
    """The cpu jobs a service job's owner agreed it may suspend (`submit.py --suspend` / `--no-suspend`, D9), from the
    job's record, trusted only when its owner is the job's user (slurm's); None, any, when the record names none."""
    consent = (gpu_jobs.read_sidecar(job.job_id, job.user) or {}).get("may_suspend")
    if isinstance(consent, list) and all(isinstance(j, str) and j.isdigit() for j in consent):
        return frozenset(consent)
    return None


def _dicts(value: object) -> dict[str, dict[str, object]]:
    return {k: v for k, v in value.items() if isinstance(v, dict)} if isinstance(value, dict) else {}


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
        part_time_conf: Path = Path(render.PART_TIME_CONF_PATH),
        copier: Runner = subprocess.run,
        guard: memory_guard.MemoryGuard | None = None,
        hostname: str | None = None,
        node_check_path: Path = node_check.SETTINGS_PATH,
        meminfo_path: Path = gpu_jobs.MEMINFO,
        containers: Callable[[], list[gpu_jobs.JobContainer] | None] = gpu_jobs.known_containers,
    ) -> None:
        self.site = site
        # the unified-memory part (D44) watches the agent's own node: the installer refuses a GPU node elsewhere
        here = (hostname or gpu_jobs.this_host()).split(".")[0]
        self.own_node = next((node for node in site.nodes if node.name == here), None)
        self.guard, self.node_check_path, self.meminfo_path = guard, node_check_path, meminfo_path
        self._guard_level: int | None = None  # the close level last handed to the guard, logged when it changes
        self._last_usage: dict[int, float] = {}  # the last card sample, for the guard when nvidia-smi is silent
        self._over_told: set[tuple[str, int]] = set()  # runs already given their over_ask event
        self._outside_at = -OUTSIDE_EVERY_S
        self.copier = copier
        self._copy_at: dict[str, float] = {}  # node -> when it was last tried
        self.copied: dict[str, int] = {}  # node -> when its state was last copied (kept in agent.json)
        self._copy_failing: set[str] = set()  # nodes whose outage is already logged
        self._copy_partial: set[str] = set()  # nodes whose partial copy is already logged
        self.part_time_conf = part_time_conf
        self.state_dir, self.pid_job = state_dir, pid_job
        self.watch: OomWatch | None = None  # main() starts one; a test drives record_kill itself
        self._pid_jobs: dict[int, str] = {}  # each running job's processes at the last step, for a kill seen late
        self.gpu_usage, self.cgroup_root = gpu_usage, cgroup_root
        self.containers = containers  # the root health check's map of labelled containers (the agent has no docker access)
        self.dir = state_dir / "agent"
        self.metrics = state_dir / "metrics"
        self._mem_samples: dict[str, tuple[int, int]] = {}  # job -> (sum of memory samples, how many) since the agent saw it
        self._cores_peak: dict[str, float] = {}  # job -> highest "cores now" this agent has seen for it
        self._cpu_prev: dict[str, tuple[float, float]] = {}  # job -> (wall time, cpu seconds) at the last sample
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
            suspended=_strings(saved.get("suspended")),
            held={k: v for k, v in (saved.get("held") or {}).items() if isinstance(v, int)}
            if isinstance(saved.get("held"), dict) else {},
            helping={k: v for k, v in (saved.get("helping") or {}).items() if isinstance(v, (int, float))}
            if isinstance(saved.get("helping"), dict) else {},
            given_up={k: v for k, v in (saved.get("given_up") or {}).items() if isinstance(v, (int, float))}
            if isinstance(saved.get("given_up"), dict) else {},
            part_time_done=_strings(saved.get("part_time_done")),
            part_time_extra=_dicts(saved.get("part_time_extra")),
            part_time_resources=_dicts(saved.get("part_time_resources")),
            idle_since={k: v for k, v in (saved.get("idle_since") or {}).items() if isinstance(v, int)}
            if isinstance(saved.get("idle_since"), dict) else {},
            idle_told={k: v for k, v in (saved.get("idle_told") or {}).items() if isinstance(v, (int, float))}
            if isinstance(saved.get("idle_told"), dict) else {},
        )
        beat = gpu_jobs.read_json(self.dir / "agent.json") or {}
        saved_copies = beat.get("copied")
        self.copied = {k: v for k, v in saved_copies.items() if isinstance(v, int)} if isinstance(saved_copies, dict) else {}
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
        self._idle_at = -IDLE_EVERY_S
        self.idle: list[IdleFinding] = []  # the last check's findings, kept in agent.json until the next one

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
            if self.site.service is not None or self.state.suspended or self.state.held:  # undo even if removed
                self._make_room(queue, now, errors)
            self._pid_jobs = self._running_pids(queue)
            self._prune_oom_kills(queue)
            if now - self._idle_at >= IDLE_EVERY_S:
                self._idle_at = now
                self._idle_check(queue, now, errors)
        if self.site.part_time_nodes:  # needs no queue: only slurm's node records
            self._part_time(errors)
            self._copy_back(now)
        if self.watch is not None and self.watch.error:
            errors.append(self.watch.error)
        if now - self._notices_at >= NOTICES_EVERY_S:
            self._notices_at = now
            self._failure_notices(errors)
        if queue is not None and (self.site.unified_memory or self.site.metrics.mode != site_config.MetricsMode.OFF):
            self._memory_step(queue, now, errors)
        if errors:
            self.last_error, self.last_error_at = "; ".join(errors), int(now)
        return self._persist(now)

    def _save_state(self) -> None:
        s = self.state
        gpu_jobs.write_json(
            self.dir / "state.json",
            {
                "base": s.base,
                "ended_seen": s.ended_seen,
                "released_requests": s.released_requests,
                "suspended": s.suspended,
                "held": s.held,
                "helping": s.helping,
                "given_up": s.given_up,
                "part_time_done": s.part_time_done,
                "part_time_extra": s.part_time_extra,
                "part_time_resources": s.part_time_resources,
                "idle_since": s.idle_since,
                "idle_told": s.idle_told,
            },
        )

    def _persist(self, now: float) -> dict[str, object]:
        self._save_state()
        beat: dict[str, object] = {
            "updated": int(now),
            "boot_reconciled_at": self.boot_reconciled_at,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "note": self.note,
            "copied": self.copied,
            "idle": [dataclasses.asdict(f) | {"text": f.text} for f in self.idle],
            **({"memory_guard": self.guard.heartbeat()} if self.guard is not None else {}),
        }
        gpu_jobs.write_json(self.dir / "agent.json", beat)
        return beat

    # -- 1. boot re-registration ----------------------------------------------------------------------------

    def _node_drained(self, node: str) -> bool | None:
        """Whether `node` is drained or draining for a reason a reboot does not clear; None when slurm does not say.

        slurm drains a node with KILL_TASK_FAILED when a job's process will not die (2026-11-07: one stuck in the GPU driver);
        a reboot ended that process, so at boot that drain is not kept, and the release is logged.
        """
        done = call(self.scontrol, ["scontrol", "show", "node", node, "-o"])
        match = (
            re.search(r"\bState=(\S+)", done.stdout) if done.returncode == 0 else None
        )
        if match is None:
            return None
        reason = re.search(r"\bReason=(.*?)(?: \S+=|$)", done.stdout)
        if "DRAIN" in match[1] and reason and reason[1].startswith(KILL_TASK_FAILED):
            print(f"[agent] boot: releasing {node}, drained for '{reason[1]}' before the reboot ended that process", flush=True)
            return False
        return "DRAIN" in match[1]

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
        part_time = {n.name for n in self.site.part_time_nodes}  # their health check opens them; one may be switched off
        nodes = [n for n in nodes if n not in part_time]
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
        level = levels.Level.HIGH if job.partition == gpu_jobs.Partition.SERVICE else job.level  # a person waits on it
        target = levels.priority_for(level, job.job_id, self.state.base or 0)
        if job.priority == target:
            return
        if call(self.scontrol, ["scontrol", "update", f"jobid={job.job_id}", f"priority={target}"]).returncode:
            errors.append(f"placing job {job.job_id} failed; will retry")
            return
        print(f"[agent] job {job.job_id} placed at {level} (priority {job.priority} -> {target})", flush=True)

    # -- 3b. room for service jobs --------------------------------------------------------------------------

    def _make_room(self, queue: list[gpu_jobs.QueuedJob], now: float, errors: list[str]) -> None:
        """Step 3b. Every change is saved first: a suspended job the state forgot would never be resumed."""
        s = self.state
        jobs = {job.job_id: job for job in queue}
        s.suspended = [j for j in s.suspended if j in jobs and jobs[j].state == gpu_jobs.State.SUSPENDED]
        s.given_up = {j: at for j, at in s.given_up.items() if j in jobs and now - at < ROOM_RETRY_S}
        for job_id, since in list(s.helping.items()):
            job = jobs.get(job_id)
            if job is None or job.state in gpu_jobs.ON_CORES:
                del s.helping[job_id]  # started, or gone
            elif now - since > ROOM_WAIT_S:
                del s.helping[job_id]
                s.given_up[job_id] = now
                print(f"[agent] service job {job_id} did not start {ROOM_WAIT_S:.0f} s after room was made; "
                      "giving its room back", flush=True)
        service = self.site.service
        node = self.site.nodes[0]  # site_config allows [service] on one node only
        room = gpu_jobs.plan_room(
            queue, set(s.suspended), node.cpus, node.real_memory_mb, service.cores if service else 0,
            {j.job_id: may_suspend(j) for j in queue if j.partition == gpu_jobs.Partition.SERVICE},
            frozenset(s.given_up),
        )
        just_held: set[str] = set()
        if room.suspend and self._hold_waiting(queue, errors, just_held):  # else a cpu job not held could take them
            for job_id in room.suspend:
                s.suspended.append(job_id)  # write-ahead: a failed suspend is pruned at the next step
                s.helping.setdefault(room.helping, now)
                self._save_state()
                if call(self.scontrol, ["scontrol", "suspend", job_id]).returncode:
                    s.suspended.remove(job_id)
                    errors.append(f"suspending job {job_id} for service job {room.helping} failed; will retry")
                    continue
                print(f"[agent] job {job_id} suspended to make room for service job {room.helping}", flush=True)
        for job_id in room.resume:
            if call(self.scontrol, ["scontrol", "resume", job_id]).returncode:
                errors.append(f"resuming job {job_id} failed; will retry")
                continue
            s.suspended.remove(job_id)
            self._save_state()
            print(f"[agent] job {job_id} resumed", flush=True)
        waiting = [j for j in s.suspended if j not in room.resume and not jobs[j].cores]
        if waiting:
            errors.append(f"suspended jobs {waiting} have no cores in squeue; resume them by hand once they are free")
        if s.suspended:
            self._hold_waiting(queue, errors, just_held)
            if not any(j.partition == gpu_jobs.Partition.SERVICE and (j.active or j.waiting) for j in queue):
                self.note = f"suspended jobs {sorted(s.suspended)} wait for their own cores to be free"
        elif s.held:
            self._release_held(jobs, errors)

    def _hold_waiting(self, queue: list[gpu_jobs.QueuedJob], errors: list[str], done: set[str]) -> bool:
        """Hold every waiting `cpu` job, one at a time (a list stops at the first it cannot hold, and the ones before
        are held unrecorded), so none takes a suspended job's cores; `done` (held this step) is skipped and added to.
        A job released by someone else waits again and is held again. GPU and urgent jobs are never held. True when
        all are held."""
        ok = True
        for job in queue:
            if not job.waiting or job.partition != gpu_jobs.Partition.CPU or job.job_id in done:
                continue
            if call(self.scontrol, ["scontrol", "hold", job.job_id]).returncode:
                errors.append(f"holding job {job.job_id} while a cpu job is suspended failed; will retry")
                ok = False
                continue
            self.state.held[job.job_id] = time.time_ns()
            done.add(job.job_id)
            self._save_state()
        return ok

    def _release_held(self, jobs: dict[str, gpu_jobs.QueuedJob], errors: list[str]) -> None:
        """Release what the agent held. A job its owner paused meanwhile (`pause.py`'s `uhold` changes nothing on the
        agent's hold, yet records a PAUSE by its owner) gets the owner's hold instead: an operator's `uhold` turns the
        hold into a user hold in one call (23.11 `job_mgr.c`), which `resume.py` can release."""
        s = self.state
        mine = {j: at for j, at in s.held.items() if j in jobs and jobs[j].held}
        paused = [j for j, at in mine.items() if any(
            e.action == gpu_jobs.EventAction.PAUSE and e.at > at and e.actor == jobs[j].user for e in gpu_jobs.events(j))]
        for job_id in paused:
            if call(self.scontrol, ["scontrol", "uhold", job_id]).returncode:
                errors.append(f"handing job {job_id} back to its owner's pause failed; will retry")
                return
        if self._release_in_order([j for j in mine if j not in paused], jobs, errors) is False:
            errors.append("releasing the jobs held for a service job failed in part; will retry")
            return
        s.held = {}
        self._save_state()

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
            held = [  # one that has left the queue or was released meanwhile needs nothing (slurm refuses it)
                job_id
                for job_id in _strings(order.get("held"))
                if gpu_jobs.JOB_ID.fullmatch(job_id) and job_id in jobs and jobs[job_id].held
            ]
            if not self._release_in_order(held, jobs, errors):
                errors.append(f"releasing {held} for {path.name} failed; will retry")
                continue
            print(f"[agent] released {held}: the pre-empted jobs run again", flush=True)
            s.released_requests.append(key)
            gpu_jobs.remove_quietly(path)
        s.released_requests = [key for key in s.released_requests if key in present]

    # -- 3a. part-time machines ---------------------------------------------------------------------------

    def _part_time(self, errors: list[str]) -> None:
        s = self.state
        nodes = {n.name for n in self.site.part_time_nodes}
        present: set[str] = set()
        self._reapply_extra(errors)  # before the requests: one a request has just set is not set twice
        for path in sorted(self.requests.glob("part_time-*.json")):
            match = PART_TIME_REQUEST.fullmatch(path.name)
            try:
                key, owner = f"{path.name}:{path.stat().st_mtime_ns}", gpu_jobs._owner(path)
            except OSError:
                continue
            present.add(key)
            if key in s.part_time_done:
                continue
            if not site_config.is_admin(self.site, owner):
                self._ignore(path, f"owned by {owner}, not an admin")
                continue
            request = gpu_jobs.read_json(path)
            if match is None or match[1] not in nodes or not request:
                self._ignore(path, "it names no part-time node of this site")
                continue
            try:
                action = PartTimeAction(request.get("action"))
            except ValueError:
                self._ignore(path, f"unknown action {request.get('action')!r}")
                continue
            if self._apply_part_time(action, match[1], request, errors):
                s.part_time_done.append(key)
                gpu_jobs.remove_quietly(path)
        s.part_time_done = [key for key in s.part_time_done if key in present]

    def _copy_back(self, now: float) -> None:
        """Copy each part-time machine's state directory to `remote/<node>/` (read-only, over ssh; plan B5b). A failure
        is logged once per outage and never reaches `errors`: the machine is often off, and that is not an error."""
        if self.site.shared_state:
            return
        key = self.state_dir / COPY_KEY
        for node in self.site.part_time_nodes:
            if now - self._copy_at.get(node.name, -COPY_EVERY_S) < COPY_EVERY_S:
                continue
            self._copy_at[node.name] = now
            dest = self.state_dir / gpu_jobs.REMOTE_DIR / node.name
            dest.mkdir(parents=True, exist_ok=True)
            ssh = (f"ssh -i {key} -o BatchMode=yes -o ConnectTimeout=10 -o UserKnownHostsFile={self.state_dir / KNOWN_HOSTS} "
                   "-o StrictHostKeyChecking=yes")
            done = call(self.copier, ["rsync", "-a", "--delete", f"--timeout={COPY_IO_TIMEOUT_S}", *COPY_OPTIONS, "-e", ssh,
                                      f"{render.SLURM_USER}@{node.address}:", f"{dest}/"], COPY_CALL_TIMEOUT_S)
            lines = (done.stderr or "").strip().splitlines()
            why = gpu_jobs.printable(lines[-1] if lines else f"exit {done.returncode}")
            if done.returncode in {0, *RSYNC_PARTIAL}:
                self.copied[node.name] = int(now)
                if node.name in self._copy_failing:
                    self._copy_failing.discard(node.name)
                    print(f"[agent] job state is copied from {node.name} again", file=sys.stderr, flush=True)
                if done.returncode == 0:
                    self._copy_partial.discard(node.name)
                elif node.name not in self._copy_partial:
                    self._copy_partial.add(node.name)
                    print(f"[agent] job state from {node.name} was copied in part (exit {done.returncode}): {why}",
                          file=sys.stderr, flush=True)
            elif node.name not in self._copy_failing:
                self._copy_failing.add(node.name)
                print(f"[agent] copying job state from {node.name} failed: {why}", file=sys.stderr, flush=True)

    def _ignore(self, path: Path, why: str) -> None:
        if path.name not in self._ignored:
            self._ignored.add(path.name)
            print(gpu_jobs.printable(f"[agent] ignoring {path}: {why}"), file=sys.stderr, flush=True)

    def _node_json(self, node: str) -> dict[str, object] | None:
        done = call(self.scontrol, ["scontrol", "show", "node", node, "--json"])
        try:
            found = json.loads(done.stdout)["nodes"][0] if done.returncode == 0 else None
        except (ValueError, KeyError, IndexError):
            found = None
        return found if isinstance(found, dict) else None

    def _scontrol_ok(self, argv: list[str], errors: list[str]) -> bool:
        done = call(self.scontrol, ["scontrol", *argv])
        if done.returncode != 0:
            errors.append(f"scontrol {' '.join(argv[:2])} failed: {done.stderr.strip() or 'no answer'}")
        return done.returncode == 0

    def _apply_part_time(self, action: PartTimeAction, node: str, request: dict[str, object], errors: list[str]) -> bool:
        """Carry out one request; False when it must be tried again (slurm did not answer or refused)."""
        match action:
            case PartTimeAction.CLOSE:
                return self._close(node, errors)
            case PartTimeAction.GIVE_BACK:
                return self._close(node, errors) and self._requeue_resume_jobs(node, errors)
            case PartTimeAction.ACCEPT:
                info = self._node_json(node)
                if info is None:
                    errors.append(f"could not read node {node}; will retry")
                    return False
                reason = str(info.get("reason") or "")
                if "DRAIN" in (info.get("state") or []) and not reason.startswith(ADMIN_REASON_PREFIX):
                    print(gpu_jobs.printable(f"[agent] not resuming {node}: drained for {reason!r}, not from the page"),
                          flush=True)
                    return True
                return self._scontrol_ok(["update", f"NodeName={node}", "State=RESUME"], errors)
            case PartTimeAction.THRESHOLDS:
                return self._set_thresholds(node, request, errors)
            case PartTimeAction.RESOURCES:
                return self._set_resources(node, request, errors)

    def _close(self, node: str, errors: list[str]) -> bool:
        return self._scontrol_ok(["update", f"NodeName={node}", "State=DRAIN", f"Reason={CLOSED_FROM_PAGE}"], errors)

    def _requeue_resume_jobs(self, node: str, errors: list[str]) -> bool:
        """Clear `node` of jobs: requeue every `resume` job running there (never an urgent or service one) and cancel any
        other, which a part-time machine may not run (B7.1).

        From the JSON queue, never `squeue -o` text: a job name may hold a newline and forge rows (review R)."""
        queue = self.read_queue()
        if queue is None:
            errors.append(f"squeue for {node} failed; will retry")
            return False
        ok = True
        for job in queue:
            if job.state != gpu_jobs.State.RUNNING or job.node != node:
                continue
            if job.job_class is gpu_jobs.JobClass.RESUME and job.partition not in RESUME_PROTECTED:
                ok = self._scontrol_ok(["requeue", job.job_id], errors) and ok
            else:
                done = call(self.scontrol, ["scancel", job.job_id])
                if done.returncode != 0:
                    errors.append(f"scancel {job.job_id} failed: {done.stderr.strip() or 'no answer'}")
                ok = done.returncode == 0 and ok
        return ok

    def _set_thresholds(self, node: str, request: dict[str, object], errors: list[str]) -> bool:
        """The limits go into the node's `Extra`, the health check's override; kept here too, for slurm forgetting them."""
        new: dict[str, object] = {}
        for gb_key, mb_key in THRESHOLD_FIELDS.items():
            if gb_key in request:
                gb = request[gb_key]
                if isinstance(gb, bool) or not isinstance(gb, (int, float)) or not 0 <= gb <= MAX_THRESHOLD_GB:
                    self._reject(node, f"{gb_key} must be a number of GB from 0 to {MAX_THRESHOLD_GB}")
                    return True
                new[mb_key] = round(gb * site_config.MB_PER_GB)
        if "mains_only" in request:
            if not isinstance(request["mains_only"], bool):
                self._reject(node, "mains_only must be true or false")
                return True
            new["mains_only"] = request["mains_only"]
        site_node = next(n for n in self.site.part_time_nodes if n.name == node)
        merged = {**self.state.part_time_extra.get(node, {}), **new}
        close = merged.get("close_below_mb", site_node.close_below_mb)
        push = merged.get("push_back_below_mb", site_node.push_back_below_mb)
        if not new or push > close:  # type: ignore[operator]
            self._reject(node, "no threshold given, or push-back is above close")
            return True
        self.state.part_time_extra[node] = merged
        return self._scontrol_ok(["update", f"NodeName={node}", f"Extra={json.dumps(merged, separators=(',', ':'))}"], errors)

    def _reapply_extra(self, errors: list[str]) -> None:
        """Slurm lost a node's `Extra` (a re-registered node, a slurmctld started without state): set it again."""
        for node, saved in self.state.part_time_extra.items():
            info = self._node_json(node)
            if info is None:
                continue
            try:
                current = json.loads(str(info.get("extra") or ""))
            except ValueError:
                current = None
            if current != saved:
                self._scontrol_ok(["update", f"NodeName={node}", f"Extra={json.dumps(saved, separators=(',', ':'))}"], errors)

    def _set_resources(self, node: str, request: dict[str, object], errors: list[str]) -> bool:
        """Jobs' memory and kept cores: the part-time nodes' file is rewritten (with every node's saved choice), then
        slurm rereads it. The choice is saved first so a failed reconfigure is retried with it."""
        site_node = next(n for n in self.site.part_time_nodes if n.name == node)
        chosen = {**self.state.part_time_resources.get(node, {})}
        chosen.update({k: request[k] for k in ("jobs_memory_gb", "keep_cpus") if k in request})
        try:
            site_config._check_part_time(dataclasses.replace(site_node, **chosen), "request")  # noqa: SLF001
        except (site_config.SiteError, TypeError, ValueError, OverflowError) as exc:  # NaN and infinity raise the last two
            self._reject(node, str(exc))
            return True
        self.state.part_time_resources[node] = chosen
        site = dataclasses.replace(self.site, nodes=tuple(
            dataclasses.replace(n, **self.state.part_time_resources.get(n.name, {})) if n.part_time else n
            for n in self.site.nodes))
        try:
            self.part_time_conf.write_text(render.part_time_nodes_conf(site), encoding="utf-8")  # in place: SlurmUser owns it
        except OSError as exc:
            errors.append(f"could not write {self.part_time_conf}: {exc}")
            return False
        return self._scontrol_ok(["reconfigure"], errors)

    def _reject(self, node: str, why: str) -> None:
        print(gpu_jobs.printable(f"[agent] part-time request for {node} refused: {why}"), file=sys.stderr, flush=True)

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
                self._notify(NotifyEvent.FAILED, job_id, user, name,
                             f"{jobs_view.project_of(name)} job #{job_id} ({name}) ended {state}", errors, state=state)
            known.add(job_id)
        s.ended_seen = sorted(known, key=_by_id)[-ENDED_KEPT:]

    def _notify(self, event: NotifyEvent, job: str, user: str, name: str, message: str, errors: list[str],
                **extra: str) -> None:
        """Run the site's notify command once, with the notice in its environment (the contract is in the README's
        `notify_command` row): the job it is about, that job's real name, project and user, one ready sentence, and
        `extra` fields (GPUJOBS_<KEY>) such as the failed state or an idle finding's other side."""
        if not self.site.notify_command:
            return
        env = {
            "GPUJOBS_EVENT": event.value,
            "GPUJOBS_JOB": job,
            "GPUJOBS_USER": user,
            "GPUJOBS_NAME": gpu_jobs.printable(name),  # what the submitter typed
            "GPUJOBS_PROJECT": gpu_jobs.printable(jobs_view.project_of(name)),
            "GPUJOBS_MESSAGE": gpu_jobs.printable(message),
            **{f"GPUJOBS_{key.upper()}": gpu_jobs.printable(value) for key, value in extra.items()},
        }
        try:
            self.notify(list(self.site.notify_command), env)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"notify command for job {job} failed: {exc}")

    # -- 5b. idle resources while jobs wait (Part C2) ----------------------------------------------------------

    def _idle_check(self, queue: list[gpu_jobs.QueuedJob], now: float, errors: list[str]) -> None:
        """Record who waits while the machine has something unused, log each new finding, and tell the holders' and the
        waiting job's projects (once an hour per waiting job and holder, through the site's notify command)."""
        s = self.state
        rows = gpu_jobs.node_capacity(self.scontrol)
        if not rows:
            errors.append("scontrol did not answer; the idle check waits")
            return
        found = idle_findings(queue, gpu_jobs.usable_capacity(rows), self.site.cpu_job_cap,
                              lambda partitions: gpu_jobs.usable_capacity(rows, partitions))
        seen = {f"{f.waiting}:{f.rule}" for f in found}
        s.idle_since = {k: v for k, v in s.idle_since.items() if k in seen}
        for finding in found:
            key = f"{finding.waiting}:{finding.rule}"
            if key not in s.idle_since:
                print(f"[agent] idle ({finding.rule}): {finding.text}; unused: {finding.idle}", flush=True)
            finding.since = s.idle_since.setdefault(key, int(now))
        s.idle_told = {k: v for k, v in s.idle_told.items() if now - v < IDLE_NOTICE_EVERY_S}
        jobs = {j.job_id: j for j in queue}
        self._record_idle(found, errors)
        told = 0
        for finding in found:
            waiting = jobs.get(finding.waiting)
            for holder in finding.holders:
                key = f"{finding.waiting}:{holder.job}"
                if key in s.idle_told or told >= NOTICES_PER_LOOK:
                    continue
                s.idle_told[key] = now
                told += 1
                held = jobs.get(holder.job)
                moved = "it can be moved" if holder.movable else "it cannot be moved"
                self._notify(NotifyEvent.IDLE, finding.waiting, waiting.user if waiting else "", waiting.name if waiting else "",
                             f"{finding.project} job #{finding.waiting} waits for {finding.lacks} with {finding.idle}; "
                             f"{holder.project} job #{holder.job} holds {holder.amount} ({holder.job_class}, {moved})",
                             errors, role=NotifyRole.WAITING, other_job=holder.job, other_project=holder.project)
                self._notify(NotifyEvent.IDLE, holder.job, held.user if held else "", held.name if held else "",
                             f"{holder.project} job #{holder.job} holds {holder.amount} that {finding.project} job "
                             f"#{finding.waiting} needs while it waits with {finding.idle} ({holder.job_class}, {moved})",
                             errors, role=NotifyRole.HOLDING, other_job=finding.waiting, other_project=finding.project)
        self.idle = found

    def _record_idle(self, found: list[IdleFinding], errors: list[str]) -> None:
        """Write each new (waiting job, holder) pair of a finding as an `idle` event on both jobs, once while the pair
        lasts (D41): the page's timeline and `gpujobs-wait` read them, so nobody has to be messaged."""
        s = self.state
        pairs = {f"{f.waiting}:{h.job}": (f, h) for f in found for h in f.holders}
        s.idle_recorded = [key for key in s.idle_recorded if key in pairs]
        for key, (finding, holder) in pairs.items():
            if key in s.idle_recorded:
                continue
            moved = "it can be moved" if holder.movable else "it cannot be moved"
            sides = (
                (finding.waiting, NotifyRole.WAITING, holder.job, holder.project,
                 (f"waits for {finding.lacks} with {finding.idle}; {holder.project} job #{holder.job} holds "
                  f"{holder.amount} ({holder.job_class}, {moved})")),
                (holder.job, NotifyRole.HOLDING, finding.waiting, finding.project,
                 (f"holds {holder.amount} that {finding.project} job #{finding.waiting} needs while it waits with "
                  f"{finding.idle} ({holder.job_class}, {moved})")),
            )
            try:
                for job, role, other, other_project, message in sides:
                    gpu_jobs.write_event(job, gpu_jobs.EventAction.IDLE, role=role.value, other_job=other,
                                         other_project=other_project, message=message)
            except OSError as exc:
                errors.append(f"recording idle finding {key} failed: {exc}")
                continue
            s.idle_recorded.append(key)

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
            cgroup = self._job_cgroup(job) if job.state == gpu_jobs.State.RUNNING else None
            found.update(dict.fromkeys(gpu_jobs.cgroup_pids(cgroup), job.job_id) if cgroup else {})
        return found

    def _prune_oom_kills(self, queue: list[gpu_jobs.QueuedJob]) -> None:
        """A job's record is read by its wrapper while the job is still queued; once it has left, the record goes."""
        queued = {job.job_id for job in queue}
        for path in (self.dir / "oom_kills").glob("job_*.json"):
            if path.stem.removeprefix("job_") not in queued:
                gpu_jobs.remove_quietly(path)
        if self.guard is not None:  # a stop request goes once its run is gone and recorded (and the hold-off is over)
            by_id = {job.job_id: job for job in queue}
            for request in self.guard.actuator.requests():
                if self.guard.prunable(request, by_id.get(request.job)):
                    gpu_jobs.remove_quietly(memory_guard.stop_path(request.job, self.state_dir))

    # -- 5. job metrics -------------------------------------------------------------------------------------

    def _job_cgroup(self, job: gpu_jobs.QueuedJob) -> Path | None:
        return gpu_jobs.job_cgroup(job, self.cgroup_root)

    def _memory_step(self, queue: list[gpu_jobs.QueuedJob], now: float, errors: list[str]) -> None:
        """One card sample and one list of running jobs with their cgroups, for the metrics and the unified part."""
        # a site without GPUs has no nvidia-smi to ask: its jobs' GPU memory is 0, not unknown
        usage = self.gpu_usage() if any(node.gpus for node in self.site.nodes) else {}
        running = [(job, cgroup) for job in queue
                   if job.state == gpu_jobs.State.RUNNING and (cgroup := self._job_cgroup(job)) is not None]
        metrics = self.site.metrics.mode != site_config.MetricsMode.OFF
        listed = self.containers() if running and (metrics or self.site.unified_memory) else []
        own = {job.job_id: gpu_jobs.run_containers(listed or [], job) for job, _ in running}
        if usage is None:
            errors.append("nvidia-smi did not answer; the job metrics files and the over-ask check wait")
        elif metrics:
            self._job_metrics(running, usage, own, errors)
        if self.site.unified_memory:
            self._unified_memory(queue, running, usage, own, now, errors)
        if usage is not None:
            self._last_usage = usage

    def _job_metrics(self, running: list[tuple[gpu_jobs.QueuedJob, Path]], usage: dict[int, float],
                     own: dict[str, list[gpu_jobs.JobContainer]], errors: list[str]) -> None:
        texts: dict[str, str] = {}
        for job, cgroup in running:
            containers = own[job.job_id]  # the run's docker containers count with it (README "A job's docker containers")
            pids = gpu_jobs.job_pids(cgroup, set(), set(usage), containers)
            mem = (gpu_jobs.read_int(cgroup / "memory.current") or 0) + gpu_jobs.container_bytes(
                containers, gpu_jobs.MemoryFile.CURRENT, self.cgroup_root)
            peak = gpu_jobs.read_int(cgroup / "memory.peak")
            peak = None if peak is None else peak + gpu_jobs.container_bytes(containers, gpu_jobs.MemoryFile.PEAK, self.cgroup_root)
            gpu_mib = sum(mib for pid, mib in usage.items() if pid in pids)
            cpu_s, now = _cpu_seconds(cgroup), time.time()
            before = self._cpu_prev.get(job.job_id)
            cores_now = (cpu_s - before[1]) / (now - before[0]) if cpu_s is not None and before and now > before[0] else None
            if cpu_s is not None:
                self._cpu_prev[job.job_id] = (now, cpu_s)
            if cores_now is not None:
                self._cores_peak[job.job_id] = max(cores_now, self._cores_peak.get(job.job_id, 0.0))
            total, count = self._mem_samples.get(job.job_id, (0, 0))
            self._mem_samples[job.job_id] = (total + mem, count + 1)
            texts[f"job_{job.job_id}.prom"] = metrics_text(
                job, gpu_mib, mem, cpu_s, cores_now, peak, self._cores_peak.get(job.job_id),
                (total + mem) // (count + 1),
            )
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


    # -- 7. memory on a unified-memory node (D44) -------------------------------------------------------------

    def _unified_memory(self, queue: list[gpu_jobs.QueuedJob], running: list[tuple[gpu_jobs.QueuedJob, Path]],
                        usage: dict[int, float] | None, own: dict[str, list[gpu_jobs.JobContainer]], now: float,
                        errors: list[str]) -> None:
        node = self.own_node
        if node is None:
            errors.append("this host is not a node of the site file: the memory guard has nothing to watch")
            return
        jobs_memory = node.jobs_mb * gpu_jobs.SLURM_MB
        level = self._close_level(node, errors)
        if self.guard is not None:
            if level != self._guard_level:
                print(f"[agent] memory guard watching {node.name}: close level {gpu_jobs.size_text(level)}, "
                      f"jobs memory {gpu_jobs.size_text(jobs_memory)}", flush=True)
                self._guard_level = level
            self.guard.update(running, level, jobs_memory, self._last_usage if usage is None else usage)
            self._stops(queue, now, errors)
        if usage is None:
            return  # card memory unknown: no over-ask finding and no outside-use sample this step
        memories = [m for job, cgroup in running if (m := gpu_jobs.JobMemory.measure(job, cgroup, usage, jobs_memory))]
        self._over_ask(memories, {(job.job_id, job.start) for job, _ in running}, errors)
        self._outside(node, memories, [c for cs in own.values() for c in cs], usage, now, errors)

    def _close_level(self, node: site_config.Node, errors: list[str]) -> int:
        """The node's close level in bytes, exactly as its health check computes it: the settings file, then the node's
        `Extra`; the site's level when there is no health check (or its settings cannot be read)."""
        try:
            settings = node_check.load_settings(self.node_check_path, (node.name,))
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            if self.site.health_check is not None:
                errors.append(f"health-check settings unreadable ({exc!r}); the memory guard uses the site's close level")
            return node.close_below_mb * gpu_jobs.MB
        extra = str((self._node_json(node.name) or {}).get("extra") or "")
        settings, note = node_check.apply_extra(settings, extra)
        if note:
            errors.append(f"memory guard: {note}")
        return settings.close_below_mb * gpu_jobs.MB

    def _over_ask(self, memories: list[gpu_jobs.JobMemory], runs: set[tuple[str, int]], errors: list[str]) -> None:
        """One `over_ask` event (and notice) per run that holds at least MIN_EXCESS more than it asked for."""
        self._over_told &= runs
        kinds = frozenset({gpu_jobs.EventAction.OVER_ASK})
        for m in memories:
            if m.excess < memory_guard.MIN_EXCESS or (m.job_id, m.start) in self._over_told:
                continue
            self._over_told.add((m.job_id, m.start))
            if any(e.fields.get("start") == m.start for e in gpu_jobs.agent_notices(m.job_id, kinds)):
                continue  # told before this agent started
            size = gpu_jobs.size_text
            message = (f"job {m.job_id} ({m.name}) uses {size(m.total)} (memory {size(m.held)} + card {size(m.card)}), "
                       f"over its ask of {size(m.ask)}")
            try:
                gpu_jobs.write_event(m.job_id, gpu_jobs.EventAction.OVER_ASK, message=message, start=m.start,
                                     held=m.held, card=m.card, ask=m.ask, total=m.total)
            except OSError as exc:
                errors.append(f"over_ask event for job {m.job_id}: {exc}")
            print(f"[agent] {message}", flush=True)
            self._notify(NotifyEvent.OVER_ASK, m.job_id, m.user, m.name, message, errors)

    def _stops(self, queue: list[gpu_jobs.QueuedJob], now: float, errors: list[str]) -> None:
        """Carry each pending stop forward: scancel when the wrapper did not act, and one event per outcome."""
        guard, by_id = self.guard, {job.job_id: job for job in queue}
        if guard is None:
            return
        for request, step in guard.pending(queue):
            if step is memory_guard.StopStep.ENDED:
                self._stop_event(request, gpu_jobs.MemoryStopEnd.BY_WRAPPER, "its wrapper ended it", errors)
            elif step is memory_guard.StopStep.STUCK:
                print(f"[agent] memory guard: job {request.job} is still running {memory_guard.STUCK_S:.0f} s after "
                      "scancel: a process may be stuck in the driver", flush=True)
                self._stop_event(request, gpu_jobs.MemoryStopEnd.NOT_ENDED, "still running after scancel", errors)
            else:  # ESCALATE
                job = by_id[request.job]
                acked = request.acked(gpu_jobs.read_sidecar(job.job_id, job.user))
                if acked:  # an acknowledgement is the job user's to write: it never makes a run unstoppable
                    request = self._stop_event(request, gpu_jobs.MemoryStopEnd.NOT_ENDED,
                                               f"acknowledged, but still running after {memory_guard.STUCK_S:.0f} s", errors)
                sent = guard.actuator.escalate(request)
                print(f"[agent] memory guard: scancel --signal=KILL {request.job} "
                      f"({'acknowledged, not ended' if acked else 'no acknowledgement'}): {'sent' if sent else 'failed'}",
                      flush=True)
                request = request.updated(self.state_dir, escalated_at=now)
                if not acked:
                    self._stop_event(request, gpu_jobs.MemoryStopEnd.BY_SCANCEL if sent else gpu_jobs.MemoryStopEnd.NOT_ENDED,
                                     "ended by scancel" if sent else "scancel failed", errors)

    def _stop_event(self, request: memory_guard.StopRequest, ended: gpu_jobs.MemoryStopEnd, how: str,
                    errors: list[str]) -> memory_guard.StopRequest:
        size = gpu_jobs.size_text
        message = (f"job {request.job} ({request.name}) was stopped for memory: free memory was {size(request.free)}, "
                   f"the run held {size(request.held + request.card)} (memory {size(request.held)} + card "
                   f"{size(request.card)}) of its {size(request.ask)} ask; {how}")
        try:
            gpu_jobs.write_event(request.job, gpu_jobs.EventAction.MEMORY_STOP, **request.event_fields(ended, message))
        except OSError as exc:
            errors.append(f"memory_stop event for job {request.job}: {exc}")
        print(f"[agent] {message}", flush=True)
        self._notify(NotifyEvent.MEMORY_STOP, request.job, request.user, request.name, message, errors, ended=str(ended))
        return request.updated(self.state_dir, ended=str(ended))

    def _outside(self, node: site_config.Node, memories: list[gpu_jobs.JobMemory], containers: list[gpu_jobs.JobContainer],
                 usage: dict[int, float], now: float, errors: list[str]) -> None:
        """Once a minute, what the node uses outside its jobs, for measuring its reserve (`status.py --reserve`). The
        running jobs' docker containers are theirs, not outside use: their asks already cover them, and counting them
        here too would hold the same memory back twice."""
        if now - self._outside_at < OUTSIDE_EVERY_S:
            return
        self._outside_at = now
        try:
            memory = gpu_jobs.meminfo(self.meminfo_path)
        except (OSError, ValueError) as exc:
            errors.append(f"outside-use record: {exc}")
            return
        used = memory["MemTotal"] - memory["MemAvailable"]
        held, card = sum(m.held for m in memories), sum(m.card for m in memories)
        in_containers = sum(gpu_jobs.container_memory(containers, usage, self.cgroup_root))
        folder = self.dir / "outside"
        line = {"t": int(now), "node": node.name, "outside": used - held - card - in_containers, "used": used,
                "jobs_held": held, "jobs_card": card, "jobs_containers": in_containers}
        oldest = time.strftime("%Y-%m-%d", time.localtime(now - OUTSIDE_KEEP_DAYS * DAY_S))
        try:
            folder.mkdir(exist_ok=True)
            with (folder / f"{time.strftime('%Y-%m-%d', time.localtime(now))}.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(line, sort_keys=True) + "\n")
            for path in folder.glob("*.jsonl"):
                if path.stem < oldest:
                    gpu_jobs.remove_quietly(path)
        except OSError as exc:
            errors.append(f"outside-use record: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--interval", type=float, default=5.0, help="seconds between steps"
    )
    parser.add_argument(
        "--once", action="store_true", help="run one step, print the heartbeat and exit"
    )
    arguments = parser.parse_args()

    site = site_config.load()
    guard = None
    if site.unified_memory:  # D44: the card shares the host's memory, so a job over its ask can starve the box
        guard = memory_guard.MemoryGuard(memory_guard.StopPolicy(), memory_guard.Actuator(gpu_jobs.STATE_DIR),
                                         log=lambda line: print(line, flush=True))
        guard.start()
    agent = Agent(site, gpu_jobs.STATE_DIR, guard=guard)
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
