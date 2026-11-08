"""What the jobs page and `status.py` show about a job: plain-language reasons, chains, efficiency, history, use.

Design: `plan notes` section 3.4. Pure functions over what slurm
printed (`squeue --json`, `sacct --json`) and the records `gpu_jobs` reads; every slurm call goes through a `runner`
argument so a test feeds recorded text. Nothing here writes, escalates or guesses a runtime.

Stdlib only: the OOD app and the system `python3` tools import it.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

import gpu_jobs
from gpu_jobs import Event, EventAction, QueuedJob, Reason, State

#: Runs a command line and returns what it printed; "" when slurm does not answer.
Runner = Callable[[list[str]], str]

OTHER_PROJECT = "(other)"
SECONDS_PER_HOUR = 3600
MAX_CHAIN_DEPTH = 100  # the walk and the page's recursion over it stay far from Python's recursion limit
CLASS_AND_NAME = 2  # fields after the project in `P:class:name`
NO_PERMISSION = "no permission to read this log"
NO_LOG = "no log yet"
ERROR_LINE = re.compile(r"(?i)\b(error|exception|traceback|warning|killed|out of memory|oom)\b")

_NEVER = 2**32 - 2  # sacct's start time for a job that never started


def requested_mem_mb(job: QueuedJob | None, record: Mapping[str, object]) -> int | None:
    """The memory a job holds or waits for, in MB (0: the whole node): slurm's request as it stands, so a later
    `scontrol update MinMemoryNode=` shows; else the `--mem` submit.py recorded. None when neither says."""
    if job is not None and job.mem_mb is not None:
        return job.mem_mb
    try:
        return gpu_jobs.mem_mb(str(record["mem"])) if record.get("mem") else None
    except (ValueError, IndexError):
        return None


class Bucket(StrEnum):
    """How `project_usage` groups time."""

    DAY = "day"
    WEEK = "week"


_PAUSE_ACTIONS = {EventAction.PAUSE, EventAction.URGENT_HOLD}


@dataclass(frozen=True)
class ChainNode:
    """A job with the jobs it waits for (`before`) or that wait for it (`after`); only one side is filled per node."""

    job_id: str
    name: str = ""
    state: str = ""
    before: tuple[ChainNode, ...] = ()
    after: tuple[ChainNode, ...] = ()


@dataclass(frozen=True)
class Efficiency:
    """Asked against used. None means slurm or the job's record does not say."""

    job_id: str
    req_mem_bytes: int | None
    peak_cpu_mem_bytes: int | None
    peak_gpu_mib: float | None
    cpu_use: float | None  # TotalCPU / (Elapsed * NCPUS), 0..1 and above when threads oversubscribe


@dataclass(frozen=True)
class HistoryRow:
    job_id: str
    name: str
    project: str
    user: str
    partition: str
    state: str  # first word: "CANCELLED by 1000" is CANCELLED
    exit_code: str
    submit: datetime | None
    start: datetime | None
    end: datetime | None
    elapsed_s: int
    gpus: int
    req_mem_bytes: int | None
    max_rss_bytes: int | None = None
    node: str = ""  # where it ran (sacct's `nodes`); "" if it never started
    avg_rss_bytes: int | None = None  # average memory over its steps' samples (sacct's AveRSS)
    cpus: int | None = None  # cores it was given
    cpu_s: float | None = None  # CPU seconds it used (sacct's TotalCPU)


@dataclass(frozen=True)
class UsageRow:
    period: date  # the day, or the Monday of the week, the job STARTED in
    project: str
    gpu_hours: float
    core_hours: float = 0.0  # cores held x hours, every job that started (running ones so far)
    core_hours_ended: float = 0.0  # the same, jobs that have ended only: what `core_hours_used` is a share of
    core_hours_used: float = 0.0  # CPU time ended jobs really spent (sacct's TotalCPU, filled in when a job ends)


@dataclass(frozen=True)
class MachineShare:
    """What jobs held of the full-time machines over a window, and how much of the cores held they really used."""

    gpu_hours: float
    gpu_hours_possible: float
    core_hours: float
    core_hours_possible: float
    core_hours_ended: float
    core_hours_used: float


def run_command(argv: list[str]) -> str:
    """The default runner: stdout, or "" if the command is missing, slow or fails."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


def project_of(job_name: str) -> str:
    """The first field of a `P:class:name` job name; anything else is `(other)`."""
    project, *rest = job_name.split(":", 2)
    return project if project and len(rest) == CLASS_AND_NAME else OTHER_PROJECT


def _when(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _describe(job_id: str, queue: Mapping[str, QueuedJob], records: Mapping[str, Mapping[str, object]]) -> str:
    known = queue.get(job_id)
    name = known.name if known else str(records.get(job_id, {}).get("name") or "")
    return f"job {job_id} ({name})" if name else f"job {job_id}"


def _queued_ahead(job: QueuedJob, queue: Mapping[str, QueuedJob]) -> int:
    """Waiting jobs that start before this one: higher priority, or equal priority and submitted earlier."""
    def key(other: QueuedJob) -> tuple[int, int]:
        return other.priority, -int(other.job_id) if other.job_id.isdigit() else 0

    return sum(
        1
        for other in queue.values()
        if other.job_id != job.job_id
        and other.waiting
        and other.reason != Reason.DEPENDENCY
        and key(other) > key(job)
    )


def waiting_reason(
    job: QueuedJob,
    queue: Mapping[str, QueuedJob],
    records: Mapping[str, Mapping[str, object]],
    events: Sequence[Event],
) -> str:
    """A plain sentence for why a queued job is not running, "" when nothing holds it."""
    reason = job.reason
    if reason in {Reason.NONE, ""}:
        return ""
    if reason == Reason.DEPENDENCY:
        waits = " and ".join(_describe(dep, queue, records) for dep in job.depends_on)
        return f"waits for {waits} to finish" if waits else "waits for a job it depends on"
    if reason in {Reason.HELD_USER, Reason.HELD_REQUEUE}:
        paused = [e for e in events if e.job == job.job_id and e.action in _PAUSE_ACTIONS]
        by = f" by {paused[-1].actor} at {_when(paused[-1].at)}" if paused else ""
        return f"paused{by}; resume: resume.py {job.job_id}"
    if reason == Reason.HELD_ADMIN:
        return "held by an admin (requeue limit or admin hold)"
    if reason == Reason.RESOURCES:
        needs = [f"{job.gpus} GPU(s)"] if job.gpus else []
        mb = requested_mem_mb(job, records.get(job.job_id, {}))
        if mb is not None:  # --mem=0 is slurm's "all of the node's memory"
            needs.append("all of the box's memory" if mb == 0 else f"{mb * 1024**2 / 10**9:.1f} GB memory")  # slurm MB: 2^20 bytes
        return f"waiting for {' / '.join(needs)} to free" if needs else "waiting for resources to free"
    if reason == Reason.PRIORITY:
        return f"queued behind {_queued_ahead(job, queue)} job(s)"
    if reason == Reason.BEGIN_TIME:
        return "requeued; waiting for its eligible time"
    if reason in {Reason.PARTITION_CONFIG, Reason.ACCOUNT_NOT_ALLOWED}:
        return f"not allowed in partition {job.partition}"
    return f"slurm reason: {reason}"


def chain(
    job_id: str, queue: Mapping[str, QueuedJob], records: Mapping[str, Mapping[str, object]]
) -> ChainNode:
    """Upstream jobs (from squeue's dependency, else the record's `after`) and downstream (records' `after`)."""

    def after_of(other: str) -> tuple[str, ...]:
        recorded = records.get(other, {}).get("after")
        return tuple(str(item) for item in recorded) if isinstance(recorded, list) else ()

    def upstream_ids(other: str) -> tuple[str, ...]:
        queued = queue.get(other)
        return queued.depends_on if queued and queued.depends_on else after_of(other)

    def downstream_ids(other: str) -> tuple[str, ...]:
        ids = {j.job_id for j in queue.values() if other in j.depends_on}
        ids |= {j for j in records if other in after_of(j)}
        return tuple(sorted(ids, key=lambda i: (len(i), i)))

    def node(
        other: str, ids_of: Callable[[str], tuple[str, ...]], seen: set[str], depth: int = 0, *, up: bool
    ) -> ChainNode:
        seen.add(other)  # shared by the whole walk: a job shows once, so cycles and diamonds stay small
        queued = queue.get(other)
        name = queued.name if queued else str(records.get(other, {}).get("name") or "")
        links = (
            tuple(node(i, ids_of, seen, depth + 1, up=up) for i in ids_of(other) if i not in seen)
            if depth < MAX_CHAIN_DEPTH
            else ()
        )
        return ChainNode(other, name, queued.state if queued else "", links if up else (), () if up else links)

    root = node(job_id, upstream_ids, set(), up=True)
    below = node(job_id, downstream_ids, set(), up=False)
    return ChainNode(root.job_id, root.name, root.state, root.before, below.after)


def _field(record: object, *path: str) -> object:
    """`record[path[0]][path[1]]...`, None when any step is missing or not a mapping."""
    for key in path:
        record = record.get(key) if isinstance(record, dict) else None
    return record


def _tres_count(tres: object, kind: str, name: str = "") -> int | None:
    """The count of the first `{type, name, count}` entry of kind (and name, `gpu:a100` counting as `gpu`)."""
    for entry in tres if isinstance(tres, list) else ():
        if _field(entry, "type") == kind and str(_field(entry, "name")).split(":")[0] == name:
            count = _field(entry, "count")
            return count if isinstance(count, int) else None
    return None


def _time(epoch: object) -> datetime | None:
    """Slurm's epoch seconds as the local time its text output printed; 0 and the never-started marker are None."""
    return datetime.fromtimestamp(epoch) if isinstance(epoch, int) and 0 < epoch < _NEVER else None  # noqa: DTZ006 - local, as sacct printed


def _sacct(runner: Runner, *selection: str) -> list[dict[str, object]]:
    """The job records of `sacct --json -a <selection>`, steps nested in each; [] when slurm does not answer.

    JSON, not `-P` text: a job name may hold `|` and newlines, which in text shift columns or forge rows.
    """
    try:
        jobs = json.loads(runner(["sacct", "--json", "-a", *selection]))["jobs"]
    except (ValueError, KeyError, TypeError):
        return []
    return [job for job in jobs if isinstance(job, dict)] if isinstance(jobs, list) else []


def _peak_rss(job: Mapping[str, object]) -> int | None:
    """Largest max-RSS (bytes) over the job's steps; None when no step reports one."""
    steps = job.get("steps")
    peaks = [_tres_count(_field(step, "tres", "requested", "max"), "mem") for step in steps or () if isinstance(step, dict)]
    return max((peak for peak in peaks if peak), default=None)


def _avg_rss(job: Mapping[str, object]) -> int | None:
    """Largest average-RSS (bytes) over the job's steps; None when no step reports one."""
    steps = job.get("steps")
    means = [_tres_count(_field(step, "tres", "requested", "average"), "mem") for step in steps or () if isinstance(step, dict)]
    return max((mean for mean in means if mean), default=None)


def _mib(mebibytes: int | None) -> int | None:
    return mebibytes * 1024**2 if mebibytes else None


#: What sacct gives as the node list of a job that never ran; it is no node, so nothing is shown for it.
NO_NODE = "None assigned"


def _row(job: Mapping[str, object]) -> HistoryRow:
    name = str(job.get("name", ""))
    states = _field(job, "state", "current")
    elapsed = _field(job, "time", "elapsed")
    return HistoryRow(
        job_id=str(job.get("job_id", "")),
        name=name,
        project=project_of(name),
        user=str(job.get("user", "")),
        partition=str(job.get("partition", "")),
        state=str(states[0]) if isinstance(states, list) and states else "UNKNOWN",
        exit_code=f"{_field(job, 'exit_code', 'return_code', 'number') or 0}:"
        f"{_field(job, 'exit_code', 'signal', 'id', 'number') or 0}",
        submit=_time(_field(job, "time", "submission")),
        start=_time(_field(job, "time", "start")),
        end=_time(_field(job, "time", "end")),
        elapsed_s=elapsed if isinstance(elapsed, int) else 0,
        gpus=_tres_count(_field(job, "tres", "allocated"), "gres", "gpu") or 0,
        req_mem_bytes=_mib(_tres_count(_field(job, "tres", "requested"), "mem")),
        max_rss_bytes=_peak_rss(job),
        node="" if str(job.get("nodes") or "") == NO_NODE else str(job.get("nodes") or ""),
        avg_rss_bytes=_avg_rss(job),
        cpus=_tres_count(_field(job, "tres", "allocated"), "cpu"),
        cpu_s=_cpu_seconds(job),
    )


def _cpu_seconds(job: Mapping[str, object]) -> float | None:
    seconds, micros = _field(job, "time", "total", "seconds"), _field(job, "time", "total", "microseconds")
    return seconds + (micros if isinstance(micros, int) else 0) / 1e6 if isinstance(seconds, int) else None


def _rows(runner: Runner, since: str, until: str | None) -> list[HistoryRow]:
    return [_row(job) for job in _sacct(runner, "-S", since, *(["-E", until] if until else []))]


def finished_job(job_id: str, runner: Runner = run_command) -> HistoryRow | None:
    """Slurm's accounting record of one job (finished or not), None when slurm has none."""
    return next((_row(j) for j in _sacct(runner, "-j", job_id) if str(j.get("job_id")) == job_id), None)


def history(
    since: str,
    until: str | None = None,
    user: str | None = None,
    project: str | None = None,
    state: str | None = None,
    runner: Runner = run_command,
) -> list[HistoryRow]:
    """Finished and running jobs in the window (`YYYY-MM-DD[THH:MM:SS]`), filtered, with their peak CPU memory."""
    return [
        row
        for row in _rows(runner, since, until)
        if (user is None or row.user == user)
        and (project is None or row.project == project)
        and (state is None or row.state == state.upper())
    ]


def usage_rows(rows: Iterable[HistoryRow], bucket: Bucket = Bucket.DAY) -> list[UsageRow]:
    """GPU-hours (GPUs held x hours) and core-hours (cores held x hours; for ended jobs also the CPU time they spent)
    per project per day or week, by the day the job started. 4 cores held for an hour are 4 core-hours."""
    totals: dict[tuple[date, str], list[float]] = {}
    for row in rows:
        if row.start is None:
            continue
        day = row.start.date()
        period = day - timedelta(days=day.weekday()) if bucket == Bucket.WEEK else day
        hours = row.elapsed_s / SECONDS_PER_HOUR
        held = (row.cpus or 0) * hours
        ended = row.end is not None and row.state != State.RUNNING and row.cpu_s is not None
        add = totals.setdefault((period, row.project), [0.0, 0.0, 0.0, 0.0])
        add[0] += (row.gpus or 0) * hours
        add[1] += held
        add[2] += held if ended else 0.0
        add[3] += (row.cpu_s or 0.0) / SECONDS_PER_HOUR if ended else 0.0
    return [UsageRow(period, project, *values) for (period, project), values in sorted(totals.items())]


def project_usage(
    since: str, until: str | None = None, bucket: Bucket = Bucket.DAY, runner: Runner = run_command
) -> list[UsageRow]:
    """`usage_rows` of the jobs slurm's accounting has from `since` (to `until`)."""
    return usage_rows(_rows(runner, since, until), bucket)


def machine_share(rows: Iterable[HistoryRow], nodes: Mapping[str, tuple[int, int]], hours: float) -> MachineShare:
    """What jobs that started in a window of `hours` held on `nodes` (name -> (cores, GPUs)), against what those nodes
    had: GPU-hours and core-hours held, and the CPU time ended jobs spent of the cores they held. A job that ran on
    another node (a part-time machine) is left out of both sides."""
    gpu = core = ended = used = 0.0
    for row in rows:
        if row.start is None or not set(row.node.split(",")) <= set(nodes) or not row.node:
            continue
        job_hours = row.elapsed_s / SECONDS_PER_HOUR
        gpu += (row.gpus or 0) * job_hours
        core += (row.cpus or 0) * job_hours
        if row.end is not None and row.state != State.RUNNING and row.cpu_s is not None:
            ended += (row.cpus or 0) * job_hours
            used += row.cpu_s / SECONDS_PER_HOUR
    return MachineShare(gpu, sum(g for _c, g in nodes.values()) * hours, core,
                        sum(c for c, _g in nodes.values()) * hours, ended, used)


def efficiency(
    job_id: str,
    runner: Runner = run_command,
    record: Mapping[str, object] | None = None,
) -> Efficiency:
    """Memory asked against peak used, and CPU time against CPU time available.

    Peak GPU memory comes from the job's record (the wrapper's footprint samples), max over its legs.
    """
    job = next((j for j in _sacct(runner, "-j", job_id) if str(j.get("job_id")) == job_id), {})
    elapsed = _field(job, "time", "elapsed")
    cpus = _tres_count(_field(job, "tres", "allocated"), "cpu")
    cpu_seconds = _field(job, "time", "total", "seconds")
    cpu_micros = _field(job, "time", "total", "microseconds")
    available = elapsed * cpus if isinstance(elapsed, int) and cpus else 0
    used = cpu_seconds + (cpu_micros if isinstance(cpu_micros, int) else 0) / 1e6 if isinstance(cpu_seconds, int) else None
    owner = job.get("user") if isinstance(job.get("user"), str) else None  # a record someone else left is not read
    stored = record if record is not None else gpu_jobs.read_sidecar(job_id, owner or None) or {}
    legs = [leg for leg in (stored.get("legs") or [stored.get("last_leg")]) if isinstance(leg, dict)]
    gpu = [float(leg["peak_gpu_mib"]) for leg in legs if "peak_gpu_mib" in leg]
    return Efficiency(
        job_id,
        _mib(_tres_count(_field(job, "tres", "requested"), "mem")),
        _peak_rss(job),
        max(gpu) if gpu else None,
        used / available if used is not None and available else None,
    )


def log_tail(path: str, max_bytes: int = 262144, text: str | None = None, errors_only: bool = False) -> str:
    """The last `max_bytes` of a log, optionally only lines containing `text` and/or looking like errors."""
    try:
        # O_NONBLOCK: opening a FIFO for reading would otherwise wait for a writer, hanging the caller
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return NO_LOG
            size = handle.seek(0, 2)
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
            cut = size > max_bytes
    except PermissionError:
        return NO_PERMISSION
    except OSError:
        return NO_LOG
    lines = data.decode("utf-8", errors="replace").splitlines()
    if cut:
        lines = lines[1:]  # the first line began before the window
    if text:
        needle = text.lower()
        lines = [line for line in lines if needle in line.lower()]
    if errors_only:
        lines = [line for line in lines if ERROR_LINE.search(line)]
    return "\n".join(lines)


def can_act(job: QueuedJob, user: str, admin: bool) -> bool:
    """Owner or admin. A hint for the UI; slurm enforces."""
    return admin or bool(user) and job.user == user
