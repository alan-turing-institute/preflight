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
import levels
import site_config

sys.path.insert(0, str(Path(__file__).resolve().parent / "install"))  # the installer ships render.py there

import render  # noqa: E402 - path set just above

BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
#: slurmctld must have been up this long before the nodes are re-registered (it needs time to read its state).
CTLD_SETTLE_S = 30.0
#: How long a service job may take to start once room is made for it; then the room is given back (step 3b).
ROOM_WAIT_S = 60.0
#: How long a service job that did not start is passed over before room is made for it again.
ROOM_RETRY_S = 300.0
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


#: The page's own drain of a part-time node; `accept` resumes a node only while its reason starts so.
ADMIN_REASON_PREFIX = "gpujobs-admin: "
CLOSED_FROM_PAGE = f"{ADMIN_REASON_PREFIX}closed from the page"
PART_TIME_REQUEST = re.compile(r"part_time-(.+)-(\d+)\.json")
#: The node `Extra` keys the health check reads (node_check.EXTRA_KEYS), by the request field that sets each.
THRESHOLD_FIELDS = {"close_below_gb": "close_below_mb", "push_back_below_gb": "push_back_below_mb"}
MAX_THRESHOLD_GB = 1000
RESUME_PROTECTED = frozenset({gpu_jobs.Partition.URGENT, gpu_jobs.Partition.SERVICE})


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
    suspended: list[str] = field(default_factory=list)  # cpu jobs the agent suspended for a service job (3b)
    held: dict[str, int] = field(default_factory=dict)  # job -> when (ns) the agent held it while one is suspended
    helping: dict[str, float] = field(default_factory=dict)  # service job -> when the agent first suspended for it
    given_up: dict[str, float] = field(default_factory=dict)  # service job -> when it did not start once room was made
    part_time_done: list[str] = field(default_factory=list)  # "<file>:<mtime_ns>" of part-time requests carried out
    part_time_extra: dict[str, dict[str, object]] = field(default_factory=dict)  # node -> health-check limits the page set
    part_time_resources: dict[str, dict[str, object]] = field(default_factory=dict)  # node -> jobs_memory_gb, keep_cpus


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
    ) -> None:
        self.site = site
        self.part_time_conf = part_time_conf
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
            if self.site.service is not None or self.state.suspended or self.state.held:  # undo even if removed
                self._make_room(queue, now, errors)
            self._pid_jobs = self._running_pids(queue)
            self._prune_oom_kills(queue)
        if self.site.part_time_nodes:  # needs no queue: only slurm's node records
            self._part_time(errors)
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
        """Requeue every `resume` job running on `node` (never an urgent or service one)."""
        done = call(self.scontrol, ["squeue", "-h", "-t", "R", "-w", node, "-o", "%i|%j|%P"])
        if done.returncode != 0:
            errors.append(f"squeue for {node} failed; will retry")
            return False
        ok = True
        for line in done.stdout.splitlines():
            job_id, _, rest = line.partition("|")
            name, _, partition = rest.rpartition("|")
            if (gpu_jobs.JOB_ID.fullmatch(job_id) and gpu_jobs.class_of(name) is gpu_jobs.JobClass.RESUME
                    and partition not in RESUME_PROTECTED):
                ok = self._scontrol_ok(["requeue", job_id], errors) and ok
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
        except (site_config.SiteError, TypeError) as exc:
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
