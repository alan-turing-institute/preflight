"""What the jobs page and `status.py` show about a job: plain-language reasons, chains, efficiency, history, use.

Design: `plan notes` section 3.4. Pure functions over text that slurm
printed (`squeue`, `sacct`) and the records `gpu_jobs` reads; every slurm call goes through a `runner`
argument so a test feeds recorded text. Nothing here writes, escalates or guesses a runtime.

Stdlib only: the OOD app and the system `python3` tools import it.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

import gpu_jobs
from gpu_jobs import Event, EventAction, QueuedJob

#: Runs a command line and returns what it printed; "" when slurm does not answer.
Runner = Callable[[list[str]], str]

OTHER_PROJECT = "(other)"
SECONDS_PER_HOUR = 3600
CLASS_AND_NAME = 2  # fields after the project in `P:class:name`
NO_PERMISSION = "no permission to read this log"
NO_LOG = "no log yet"
ERROR_LINE = re.compile(r"(?i)\b(error|exception|traceback|warning|killed|out of memory|oom)\b")

_HISTORY_FORMAT = "JobID,JobName,User,Partition,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocTRES,ReqMem"
_STEP_FORMAT = "JobID,MaxRSS,TotalCPU,NCPUS,ElapsedRaw,ReqMem"
_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_NO_TIME = {"", "None", "Unknown"}


class Bucket(StrEnum):
    """How `project_usage` groups time."""

    DAY = "day"
    WEEK = "week"


class Reason(StrEnum):
    """The squeue pending reasons that get their own sentence (`man squeue`, JOB REASON CODES)."""

    NONE = "None"
    DEPENDENCY = "Dependency"
    HELD_USER = "JobHeldUser"
    HELD_REQUEUE = "job requeued in held state"
    HELD_ADMIN = "JobHeldAdmin"
    RESOURCES = "Resources"
    PRIORITY = "Priority"
    BEGIN_TIME = "BeginTime"
    PARTITION_CONFIG = "PartitionConfig"
    ACCOUNT_NOT_ALLOWED = "AccountNotAllowed"


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


@dataclass(frozen=True)
class UsageRow:
    period: date  # the day, or the Monday of the week, the job STARTED in
    project: str
    gpu_hours: float


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
        mem = records.get(job.job_id, {}).get("mem")
        if mem:
            needs.append(f"{mem} memory")
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

    def node(other: str, ids_of: Callable[[str], tuple[str, ...]], seen: frozenset[str], *, up: bool) -> ChainNode:
        queued = queue.get(other)
        name = queued.name if queued else str(records.get(other, {}).get("name") or "")
        links = tuple(node(i, ids_of, seen | {other}, up=up) for i in ids_of(other) if i not in seen)
        return ChainNode(other, name, queued.state if queued else "", links if up else (), () if up else links)

    root = node(job_id, upstream_ids, frozenset(), up=True)
    below = node(job_id, downstream_ids, frozenset(), up=False)
    return ChainNode(root.job_id, root.name, root.state, root.before, below.after)


def parse_bytes(text: str) -> int | None:
    """`4G`, `100000M`, `528880K`, `4Gn` (old per-node suffix) -> bytes; None when empty or unreadable."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT])[nc]?", text.strip())
    return int(float(match[1]) * _UNITS[match[2]]) if match else None


def parse_cpu_seconds(text: str) -> float | None:
    """sacct's `[D-][HH:]MM:SS[.mmm]` -> seconds."""
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", text.strip())
    if not match:
        return None
    days, hours, minutes, seconds = match.groups(default="0")
    return ((int(days) * 24 + int(hours)) * 60 + int(minutes)) * 60 + float(seconds)


def parse_gres_gpus(alloc_tres: str) -> int:
    """GPUs in sacct's `AllocTRES` (`billing=4,cpu=4,gres/gpu=1,mem=4G,node=1`); 0 when none."""
    match = re.search(r"(?:^|,)gres/gpu(?::[^=,]*)?=(\d+)", alloc_tres)
    return int(match[1]) if match else 0


def _table(text: str) -> list[dict[str, str]]:
    """Rows of sacct's `-P` output keyed by the header line."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    return [dict(zip(header, line.split("|"), strict=False)) for line in lines[1:]]


def _time(text: str) -> datetime | None:
    if text in _NO_TIME:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _sacct(runner: Runner, fmt: str, since: str, until: str | None, *, steps: bool) -> list[dict[str, str]]:
    argv = ["sacct", "-a", "-P", "-S", since, *(["-E", until] if until else []), f"--format={fmt}"]
    return _table(runner(argv if steps else [*argv[:2], "-X", *argv[2:]]))


def _rows(runner: Runner, since: str, until: str | None) -> list[HistoryRow]:
    rows = []
    for row in _sacct(runner, _HISTORY_FORMAT, since, until, steps=False):
        name = row.get("JobName", "")
        elapsed = row.get("ElapsedRaw", "")
        rows.append(
            HistoryRow(
                job_id=row.get("JobID", ""),
                name=name,
                project=project_of(name),
                user=row.get("User", ""),
                partition=row.get("Partition", ""),
                state=(row.get("State") or "UNKNOWN").split()[0],
                exit_code=row.get("ExitCode", ""),
                submit=_time(row.get("Submit", "")),
                start=_time(row.get("Start", "")),
                end=_time(row.get("End", "")),
                elapsed_s=int(elapsed) if elapsed.isdigit() else 0,
                gpus=parse_gres_gpus(row.get("AllocTRES", "")),
                req_mem_bytes=parse_bytes(row.get("ReqMem", "")),
            )
        )
    return rows


def _peak_rss(steps: list[dict[str, str]]) -> dict[str, int]:
    """Largest MaxRSS over the steps of each job."""
    peaks: dict[str, int] = {}
    for row in steps:
        rss = parse_bytes(row.get("MaxRSS", ""))
        if rss is not None:
            job = row.get("JobID", "").split(".")[0]
            peaks[job] = max(peaks.get(job, 0), rss)
    return peaks


def history(
    since: str,
    until: str | None = None,
    user: str | None = None,
    project: str | None = None,
    state: str | None = None,
    runner: Runner = run_command,
) -> list[HistoryRow]:
    """Finished and running jobs in the window (`YYYY-MM-DD[THH:MM:SS]`), filtered, with their peak CPU memory."""
    peaks = _peak_rss(_sacct(runner, "JobID,MaxRSS", since, until, steps=True))
    return [
        HistoryRow(**{**row.__dict__, "max_rss_bytes": peaks.get(row.job_id)})
        for row in _rows(runner, since, until)
        if (user is None or row.user == user)
        and (project is None or row.project == project)
        and (state is None or row.state == state.upper())
    ]


def project_usage(
    since: str, until: str | None = None, bucket: Bucket = Bucket.DAY, runner: Runner = run_command
) -> list[UsageRow]:
    """GPU-hours per project per day or week: GPUs in `AllocTRES` x `ElapsedRaw` / 3600, by the day the job started."""
    totals: dict[tuple[date, str], float] = {}
    for row in _rows(runner, since, until):
        if not row.gpus or row.start is None:
            continue
        day = row.start.date()
        period = day - timedelta(days=day.weekday()) if bucket == Bucket.WEEK else day
        totals[period, row.project] = totals.get((period, row.project), 0.0) + row.gpus * row.elapsed_s / SECONDS_PER_HOUR
    return [UsageRow(period, project, hours) for (period, project), hours in sorted(totals.items())]


def efficiency(
    job_id: str,
    runner: Runner = run_command,
    record: Mapping[str, object] | None = None,
) -> Efficiency:
    """Memory asked against peak used, and CPU time against CPU time available.

    Peak GPU memory comes from the job's record (the wrapper's footprint samples), max over its legs.
    """
    rows = _table(runner(["sacct", "-a", "-P", "-j", job_id, f"--format={_STEP_FORMAT}"]))
    whole = next((r for r in rows if r.get("JobID") == job_id), {})
    elapsed, cpus = whole.get("ElapsedRaw", ""), whole.get("NCPUS", "")
    used = parse_cpu_seconds(whole.get("TotalCPU", ""))
    available = int(elapsed) * int(cpus) if elapsed.isdigit() and cpus.isdigit() else 0
    stored = record if record is not None else gpu_jobs.read_sidecar(job_id) or {}
    legs = [leg for leg in (stored.get("legs") or [stored.get("last_leg")]) if isinstance(leg, dict)]
    gpu = [float(leg["peak_gpu_mib"]) for leg in legs if "peak_gpu_mib" in leg]
    return Efficiency(
        job_id,
        parse_bytes(whole.get("ReqMem", "")),
        _peak_rss(rows).get(job_id),
        max(gpu) if gpu else None,
        used / available if used is not None and available else None,
    )


def log_tail(path: str, max_bytes: int = 262144, text: str | None = None, errors_only: bool = False) -> str:
    """The last `max_bytes` of a log, optionally only lines containing `text` and/or looking like errors."""
    try:
        with open(path, "rb") as handle:
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
