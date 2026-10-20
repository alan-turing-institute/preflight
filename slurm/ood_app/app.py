"""The GPU jobs page for Open OnDemand: a stdlib WSGI app that runs as the viewing user.

Design: `plan notes` section 3.7. Reads come from `jobs_view` and
`gpu_jobs`; pause, resume and kill run the same command-line tools a person would (`pause.py`, `resume.py`,
`kill.py`), so slurm's own rules decide what is allowed. Every URL in a page is relative, because OOD serves
a system app under `/pun/sys/<name>/`.
"""

from __future__ import annotations

import html
import json
import os
import pwd
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, urlencode, urlsplit

import gpu_jobs
import jobs_view
import site_config
from gpu_jobs import QueuedJob

APP_DIR = Path(__file__).resolve().parent
SLURM_DIR = APP_DIR.parent
TEMPLATES = APP_DIR / "templates"
STATIC = {"app.js": "text/javascript; charset=utf-8", "app.css": "text/css; charset=utf-8"}
JSON_TYPE = "application/json"
HTML_TYPE = "text/html; charset=utf-8"
MAX_BODY = 4096
MAX_SCRIPT_BYTES = 65536
NOT_SHOWN = "(job script not shown: not a regular file, or unreadable)"
ACTION_TIMEOUT_S = 60
DEFAULT_WINDOW_DAYS = 7
PROJECTS_DEFAULT_DAYS = 7
PRESET_DAYS = (7, 30, 90)
HISTORY_PAGE = 50
RECENT_ROWS = 8
PALETTE_SLOTS = 8  # the categorical slots c1..c8 in static/app.css; a ninth project and later are grey
#: what each submit.py class means, shown when hovering its label
CLASS_HELP = {
    "resume": "If paused or pre-empted it continues from its last checkpoint.",
    "restart": "Never pre-empted; if stopped it starts again from the beginning.",
    "urgent": "Submitted with urgent.py: it jumps the queue.",
}
#: project -> palette slot for the request being served (set in `handle`)
_SLOTS: ContextVar[Mapping[str, int] | None] = ContextVar("slots", default=None)
CLOCK_FIELDS = 3  # H:MM:SS
MINUTE_S, HOUR_MIN, DAY_H = 60, 60, 24
SMALLEST_HOURS = 0.05  # below this a total reads "under 0.1 h", not "0.0 h"

METER_WARN = 0.9
CANNOT_ACT = "Only the job's owner or an admin can pause or cancel it."
class Kind(StrEnum):
    """How a job's state is drawn: each is a badge class in static/app.css."""

    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    OOM = "oom"
    NODE_FAIL = "node-fail"
    PREEMPTED = "preempted"
    REQUEUED = "requeued"
    OTHER = "other"


#: slurm accounting state -> (kind, words); anything else is shown as slurm names it
RESULTS = {
    "COMPLETED": (Kind.COMPLETED, "Completed"),
    "FAILED": (Kind.FAILED, "Failed"),
    "CANCELLED": (Kind.CANCELLED, "Cancelled"),
    "TIMEOUT": (Kind.TIMEOUT, "Timed out"),
    "OUT_OF_MEMORY": (Kind.OOM, "Out of memory"),
    "NODE_FAIL": (Kind.NODE_FAIL, "Node failed"),
    "PREEMPTED": (Kind.PREEMPTED, "Preempted"),
    "REQUEUED": (Kind.REQUEUED, "Requeued"),
    "RUNNING": (Kind.RUNNING, "Running"),
    "PENDING": (Kind.WAITING, "Waiting"),
    "SUSPENDED": (Kind.PAUSED, "Paused"),
}
BAD_RESULTS = {Kind.FAILED, Kind.TIMEOUT, Kind.OOM, Kind.NODE_FAIL}
SECURITY_HEADERS = [
    ("X-Content-Type-Options", "nosniff"),
    ("Content-Security-Policy", "default-src 'self'; frame-ancestors 'self'"),
    ("Cache-Control", "no-store"),
]
JOB_ID = re.compile(r"\d+")
WHEN = re.compile(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?")
ROUTE_JOB = re.compile(r"/job/(\d+)")
ROUTE_API = re.compile(r"/api/job/([^/]+)/([a-z-]+)")


class Action(StrEnum):
    """The POST actions and the tool each runs."""

    PAUSE = "pause"
    RESUME = "resume"
    KILL = "kill"


class Tab(StrEnum):
    """The pages in the top bar: (relative path, label)."""

    QUEUE = "./"
    HISTORY = "history"
    PROJECTS = "projects"

    @property
    def path(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        return {"./": "Queue", "history": "History", "projects": "Use per project"}[self.value]


PREVIEW = "pause-preview"
LOG = "log"


class Refused(Exception):
    """A request the app will not act on; carries the HTTP status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


Run = Callable[[list[str]], tuple[int, str]]


def run_tool(argv: list[str]) -> tuple[int, str]:
    """Run a tool without a shell; (return code, stdout + stderr)."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=ACTION_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return 1, f"timed out after {ACTION_TIMEOUT_S} s"
    except OSError as exc:
        return 1, str(exc)
    return done.returncode, done.stdout + done.stderr


def _elapsed_by_job() -> dict[str, str]:
    out = jobs_view.run_command(["squeue", "-h", "-o", "%i|%M"])
    return dict(line.split("|", 1) for line in out.splitlines() if "|" in line)


def _viewer() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


def _site() -> site_config.Site | None:
    try:
        return site_config.load()
    except site_config.SiteError:
        return None


def _is_admin(user: str) -> bool:
    site = _site()
    return site is not None and site_config.is_admin(site, user)


def _host() -> str:
    site = _site()
    return site.name if site is not None else os.uname().nodename


def _keeper() -> Mapping[str, object]:
    return gpu_jobs.read_json(gpu_jobs.keeper_state_path()) or {}


def _metrics_on() -> bool:
    site = _site()
    return site is not None and site.metrics.mode != site_config.MetricsMode.OFF


@dataclass
class Backend:
    """Everything the app reads or runs, so a test can replace it with fakes."""

    queue: Callable[[], list[QueuedJob] | None] = gpu_jobs.read_queue
    record: Callable[..., Mapping[str, object] | None] = gpu_jobs.read_sidecar  # (job id, user=None)
    events: Callable[[str | None], list[gpu_jobs.Event]] = gpu_jobs.events
    elapsed: Callable[[], Mapping[str, str]] = _elapsed_by_job
    efficiency: Callable[..., jobs_view.Efficiency] = jobs_view.efficiency
    history: Callable[..., list[jobs_view.HistoryRow]] = jobs_view.history
    usage: Callable[..., list[jobs_view.UsageRow]] = jobs_view.project_usage
    log_tail: Callable[..., str] = jobs_view.log_tail
    viewer: Callable[[], str] = _viewer
    is_admin: Callable[[str], bool] = _is_admin
    metrics_on: Callable[[], bool] = _metrics_on
    host: Callable[[], str] = _host
    keeper: Callable[[], Mapping[str, object]] = _keeper
    live_memory: Callable[[str], Mapping[str, int]] = lambda job_id: _live_memory(job_id)
    finished: Callable[[str], jobs_view.HistoryRow | None] = jobs_view.finished_job
    now: Callable[[], datetime] = datetime.now
    project_order: Callable[[], list[str]] = lambda: _project_order()
    capacity: Callable[[], Mapping[str, int]] = gpu_jobs.capacity
    gpu_busy: Callable[[], float | None] = gpu_jobs.gpu_busy
    run: Run = run_tool
    python: str = field(default_factory=lambda: sys.executable)
    slurm_dir: Path = SLURM_DIR


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _template(name: str) -> Template:
    return Template((TEMPLATES / name).read_text(encoding="utf-8"))


def _records(backend: Backend, queue: list[QueuedJob]) -> dict[str, Mapping[str, object]]:
    return {j.job_id: rec for j in queue if (rec := backend.record(j.job_id, user=j.user))}


def job_rows(backend: Backend) -> list[dict[str, object]]:
    """One dict per unfinished job, the shape the table and `api/jobs` share."""
    queue = backend.queue() or []
    by_id = {j.job_id: j for j in queue}
    records = _records(backend, queue)
    events = backend.events(None)
    elapsed = backend.elapsed()
    rows = []
    for job in _ordered(queue):
        record = records.get(job.job_id, {})
        kind, label = _queue_state(job)
        rows.append(
            {
                "id": job.job_id,
                "short": _short(job.name),
                "kind": kind,
                "state_label": label,
                "elapsed_s": _seconds(elapsed.get(job.job_id, "")),
                "user": job.user,
                "project": jobs_view.project_of(job.name),
                "class": str(record.get("job_class") or job.partition),
                "name": job.name,
                "description": str(record.get("what") or ""),
                "state": job.state,
                "waiting": jobs_view.waiting_reason(job, by_id, records, events),
                "elapsed": elapsed.get(job.job_id, ""),
                "gpus": job.gpus,
                "cpus": job.cpus,
                "mem": str(record.get("mem") or ""),
            }
        )
    return rows


def _page(backend: Backend, title: str, body: str, *, root: str, tab: Tab | None = None) -> str:
    viewer = backend.viewer()
    tabs = "".join(
        f'<a href="{root}{t.path}"{" aria-current=\"page\"" if t is tab else ""}>{esc(t.label)}</a>' for t in Tab
    )
    role = '<span class="role">admin</span>' if backend.is_admin(viewer) else ""
    return _template("layout.html").substitute(
        title=esc(title), root=root, body=body, viewer=esc(viewer), role=role, tabs=tabs, host=esc(backend.host())
    )


# --- formatting: every number a person reads goes through one of these -------------------------------------------


def _seconds(squeue_elapsed: str) -> int | None:
    """squeue's %M (`M:SS`, `H:MM:SS` or `D-HH:MM:SS`) in seconds; None when it is not one of those."""
    days, _, clock = squeue_elapsed.rpartition("-")
    parts = clock.split(":")
    if not all(p.isdigit() for p in parts) or not 1 < len(parts) <= CLOCK_FIELDS or (days and not days.isdigit()):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds + int(days or 0) * 86400


def _duration(seconds: int | None) -> str:
    if seconds is None:
        return "—"
    if seconds < MINUTE_S:
        return f"{seconds} s"
    minutes, secs = divmod(seconds, MINUTE_S)
    if minutes < HOUR_MIN:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, HOUR_MIN)
    if hours < DAY_H:
        return f"{hours} h {minutes:02d} min"
    days, hours = divmod(hours, DAY_H)
    return f"{days} d {hours} h"


def _clock(moment: datetime | None, now: datetime) -> str:
    """'Today 16:23', 'Yesterday 09:10', 'Mon 13 Oct 16:23' (local time, as slurm prints it)."""
    if moment is None:
        return "—"
    days = (now.date() - moment.date()).days
    day = "Today" if days == 0 else "Yesterday" if days == 1 else moment.strftime("%a %-d %b")
    return f"{day} {moment:%H:%M}"


def _today() -> date:
    return date.today()  # noqa: DTZ011 - the box's local day, the same clock slurm prints times in


def _gib(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1024**3:.1f} GiB" if value >= 1024**3 else f"{value / 1024**2:.0f} MiB"


def _hours(value: float) -> str:
    return f"{value:.1f} h" if value >= SMALLEST_HOURS else "under 0.1 h"


def _why(text: str) -> str:
    """A waiting reason with each `job 259 (project:class:name)` shortened to `#259 name`."""
    return re.sub(r"job (\d+) \(([^)]*)\)", lambda m: f"#{m[1]} {_short(m[2])}", text)


def _exit_words(exit_code: str) -> str:
    """sacct's `code:signal` in words, '' for a clean exit: '1:0' is 'exit 1', '0:9' is 'signal 9'."""
    code, _, signal = exit_code.partition(":")
    words = [f"exit {code}"] if code not in {"", "0"} else []
    words += [f"signal {signal}"] if signal not in {"", "0"} else []
    return ", ".join(words)


def _class_pill(job_class: object) -> str:
    text = str(job_class or "—")
    help_text = CLASS_HELP.get(text)
    title = f' title="{esc(help_text)}"' if help_text else ""
    return f'<span class="pill"{title}>{esc(text)}</span>'



def _asked_memory(mem: object) -> str:
    """The `--mem` a job asked for, as submit.py recorded it: '48G', or '0', slurm's word for the whole node."""
    text = str(mem or "")
    if not text:
        return "—"
    if text == jobs_view.WHOLE_NODE_MEM:
        return "whole box"
    return re.sub(r"^(\d+)([KMGT])$", r"\1 \2B", text)


def _short(name: str) -> str:
    """The part of a `project:class:name` job name a person chose; the whole name otherwise."""
    parts = name.split(":", 2)
    return parts[2] if len(parts) == len(("project", "class", "name")) and parts[2] else name


def _project_order() -> list[str]:
    """Every project in slurm's accounting, in the order each first ran: new projects only ever append, so a
    project keeps its colour (`sacct -X -P`: only the name is used, and a name with `|` stays whole)."""
    out = jobs_view.run_command(["sacct", "-a", "-n", "-X", "-P", "-S", "2000-01-01", "-o", "JobID,JobName"])
    order: dict[str, None] = {}
    for line in out.splitlines():
        _, _, name = line.partition("|")
        order.setdefault(jobs_view.project_of(name), None)
    return list(order)


def _colour(project: str) -> str:
    """The project's palette slot, the same on every page; grey past the palette's eight."""
    slot = (_SLOTS.get() or {}).get(project)
    return f"c{slot + 1}" if slot is not None and slot < PALETTE_SLOTS else ""


def _project(project: str) -> str:
    return f'<span class="project"><span class="dot {_colour(project)}" aria-hidden="true"></span>{esc(project)}</span>'


def _badge(kind: str, label: str) -> str:
    return f'<span class="badge {kind}">{esc(label)}</span>'


def _queue_state(job: QueuedJob) -> tuple[str, str]:
    """(css kind, words) for a queued job."""
    if job.state in {gpu_jobs.State.RUNNING, gpu_jobs.State.CONFIGURING, gpu_jobs.State.COMPLETING}:
        return Kind.RUNNING, "Running"
    if job.state == gpu_jobs.State.SUSPENDED or job.reason in {gpu_jobs.Reason.HELD_USER, gpu_jobs.Reason.HELD_REQUEUE}:
        return Kind.PAUSED, "Paused"
    if job.reason == gpu_jobs.Reason.HELD_ADMIN:
        return Kind.PAUSED, "Held by admin"
    return Kind.WAITING, "Waiting"


def _result(state: str) -> tuple[str, str]:
    """(css kind, words) for slurm's accounting state."""
    return RESULTS.get(state, (Kind.OTHER, state.replace("_", " ").capitalize()))


def _live_memory(job_id: str) -> dict[str, int]:
    """The agent's latest per-job samples (`metrics/job_<id>.prom`): {'gpu': bytes, 'cpu': bytes}, either may be missing."""
    try:
        text = (gpu_jobs.STATE_DIR / "metrics" / f"job_{job_id}.prom").read_text(encoding="utf-8")
    except OSError:
        return {}
    found = {}
    for kind in ("gpu", "cpu"):
        match = re.search(rf"^gpujobs_job_{kind}_memory_bytes\{{[^}}]*\}} (\d+)$", text, re.MULTILINE)
        if match:
            found[kind] = int(match[1])
    return found


# --- queue page ------------------------------------------------------------------------------------------------------


def _ordered(queue: list[QueuedJob]) -> list[QueuedJob]:
    """Running first, then waiting jobs in the order slurm starts them (priority, then submission)."""
    def key(job: QueuedJob) -> tuple[int, int, int]:
        running = _queue_state(job)[0] == Kind.RUNNING
        return (0 if running else 1, -job.priority, int(job.job_id) if job.job_id.isdigit() else 0)

    return sorted(queue, key=key)


def _now_card(rows: list[dict[str, object]], keeper: Mapping[str, object], backend: Backend, root: str) -> str:
    running = sorted((r for r in rows if r["kind"] == Kind.RUNNING), key=lambda r: not r["gpus"])  # a GPU job leads
    waiting = [r for r in rows if r["kind"] == Kind.WAITING]
    blockers = [str(b) for b in keeper.get("blockers") or [] if b] if isinstance(keeper.get("blockers"), list) else []
    icon = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">{}</svg>'
    if running:
        job = running[0]
        live = backend.live_memory(str(job["id"]))
        more = f" and {len(running) - 1} more" if len(running) > 1 else ""
        figures = [
            ("Running for", _duration(job["elapsed_s"])),  # type: ignore[arg-type]
            ("GPU memory in use", _gib(live.get("gpu"))),
            ("CPU memory in use", _gib(live.get("cpu")) + (f" of {_asked_memory(job['mem'])} asked" if job["mem"] else "")),
        ]
        return (
            '<section class="card now" aria-label="The GPU now">'
            f'<div class="now-icon busy">{icon.format("<path d=\"M5 3l14 9-14 9V3z\"/>")}</div>'
            f'<div><div class="now-label">{"On the GPU now" if job["gpus"] else "Running now, without the GPU"}</div>'
            f'<div class="now-title"><a href="{root}job/{esc(job["id"])}">{esc(job["short"])}</a>{esc(more)}</div>'
            f'<div class="now-desc">{_project(str(job["project"]))} · {esc(job["description"] or job["name"])}</div></div>'
            '<div class="now-figures">'
            + "".join(f'<div class="figure"><div class="v num">{esc(v)}</div><div class="k">{esc(k)}</div></div>' for k, v in figures)
            + "</div></section>"
        )
    if blockers:
        title, desc, kind = "Waiting for the GPU", f"In use outside slurm by {', '.join(blockers)}; the next job starts when it is released.", "blocked"
    elif waiting:
        title, desc, kind = "Starting soon", f"{len(waiting)} job(s) waiting; the first one starts within a few seconds.", "blocked"
    else:
        title, desc, kind = "The GPU is free", "Nothing is running or waiting. A job submitted now starts straight away.", ""
    return (
        '<section class="card now" aria-label="The GPU now">'
        f'<div class="now-icon {kind}">{icon.format("<rect x=\"6\" y=\"5\" width=\"4\" height=\"14\"/><rect x=\"14\" y=\"5\" width=\"4\" height=\"14\"/>" if kind else "<path d=\"M20 6L9 17l-5-5\"/>")}</div>'
        f'<div><div class="now-label">The GPU now</div><div class="now-title">{esc(title)}</div><div class="now-desc">{esc(desc)}</div></div>'
        "</section>"
    )


def _running_note(running: int, on_gpu: int) -> str:
    """The Running tile's line: how many of the running jobs hold the GPU (a CPU-only job runs beside it)."""
    if not running:
        return "nothing running"
    return "on the GPU now" if on_gpu == running else f"{on_gpu} on the GPU" if on_gpu else "none on the GPU"


def _resource_tiles(have: Mapping[str, int], rows: list[dict[str, object]], busy: float | None = None) -> str:
    """What slurm has handed out of the box, and what is free (its own totals: memory is what slurm may give, not all
    of the machine's)."""
    if not have:
        return ""
    running = [r for r in rows if r["kind"] == Kind.RUNNING]
    by_gpu = sum(int(r["cpus"]) for r in running if r["gpus"])  # type: ignore[call-overload]
    by_cpu = sum(int(r["cpus"]) for r in running if not r["gpus"])  # type: ignore[call-overload]
    gib = 1024  # slurm counts memory in MB
    free_cpus, free_mem = have["cpus"] - have["cpus_used"], (have["mem_mb"] - have["mem_used_mb"]) / gib
    meters = [
        _meter("GPU", f"{busy:.0%} busy" if busy is not None else f"{have['gpus_used']} of {have['gpus']} in use",
               busy if busy is not None else _share(have["gpus_used"], have["gpus"]),
               f"{have['gpus_used']} of {have['gpus']} given to jobs · {have['gpus'] - have['gpus_used']} free"),
        _meter("Cores", f"{have['cpus_used']} of {have['cpus']} in use", _share(have["cpus_used"], have["cpus"]),
               f"{by_cpu} by CPU jobs · {by_gpu} by GPU jobs · {free_cpus} free"),
        _meter("Memory", f"{have['mem_used_mb'] / gib:.0f} of {have['mem_mb'] / gib:.0f} GiB given to jobs",
               _share(have["mem_used_mb"], have["mem_mb"]), f"{free_mem:.0f} GiB free"),
    ]
    return (
        '<section class="card box" aria-labelledby="h-box"><div class="card-head"><h2 id="h-box">This machine</h2>'
        '<span class="meta">now, out of what slurm can give</span></div><div class="box-meters">' + "".join(meters) + "</div></section>"
    )


def _share(used: int, total: int) -> float | None:
    return used / total if total else None


def _tiles(items: list[tuple[str, str, str]]) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="card tile"><div class="k">{esc(k)}</div><div class="v num">{esc(v)}</div><div class="note">{esc(n) or "&nbsp;"}</div></div>'
        for k, v, n in items
    ) + "</div>"


def _queue_table(rows: list[dict[str, object]], root: str) -> str:
    head = '<div class="card-head"><h2 id="h-queue">Queue</h2><span class="meta">{}</span></div>'
    if not rows:
        return head.format("empty") + '<div class="empty"><strong>Nothing queued</strong>The GPU takes the next job submitted with slurm/submit.py.</div>'
    body = []
    for position, r in enumerate(rows, 1):
        why = f'<div class="why">{esc(_why(str(r["waiting"])))}</div>' if r["waiting"] else ""
        time_text = _duration(r["elapsed_s"]) if r["kind"] == Kind.RUNNING else "—"  # type: ignore[arg-type]
        body.append(
            "<tr>"
            f'<td class="num muted">{position}</td>'
            f'<td class="lead"><a class="title" href="{root}job/{esc(r["id"])}">{esc(r["short"])}</a> <span class="jobid">#{esc(r["id"])}</span>'
            f'{f"<div class=\"desc\">{esc(r["description"])}</div>" if r["description"] else ""}</td>'
            f"<td class=\"inline\">{_project(str(r['project']))}</td>"
            f'<td class="inline">{_badge(str(r["kind"]), str(r["state_label"]))}{why}</td>'
            + (f'<td class="num inline" data-k="for">{esc(time_text)}</td>' if r["kind"] == Kind.RUNNING else '<td class="num hide-sm">—</td>')
            + f'<td class="hide-sm">{_class_pill(r["class"])}</td>'
            f'<td class="num hide-sm">{esc(r["gpus"])}</td>'
            f'<td class="num hide-sm">{esc(_asked_memory(r["mem"]))}</td>'
            f'<td class="hide-sm">{esc(r["user"])}</td>'
            "</tr>"
        )
    counts = f"{len(rows)} job(s)"
    return (
        head.format(counts)
        + '<div class="table-wrap"><table class="stack"><thead><tr><th scope="col" class="num">#</th><th scope="col">Job</th><th scope="col">Project</th>'
        '<th scope="col">State</th><th scope="col">Running for</th><th scope="col" class="hide-sm">Class</th>'
        '<th scope="col" class="hide-sm">GPUs</th><th scope="col" class="hide-sm">Memory asked</th><th scope="col" class="hide-sm">User</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _finished_rows(rows: list[jobs_view.HistoryRow], root: str, now: datetime, *, user_column: bool = False) -> str:
    out = []
    for r in rows:
        kind, label = _result(r.state)
        code = _exit_words(r.exit_code)
        exit_note = f' <span class="exit-bad small">{esc(code)}</span>' if code and kind != Kind.CANCELLED else ""
        out.append(
            "<tr>"
            f'<td class="lead"><a class="title" href="{root}job/{esc(r.job_id)}">{esc(_short(r.name))}</a> <span class="jobid">#{esc(r.job_id)}</span></td>'
            f'<td class="inline">{_project(r.project)}</td>'
            f'<td class="inline">{_badge(kind, label)}{exit_note}</td>'
            f'<td class="num inline" data-k="started">{esc(_clock(r.start, now))}</td>'
            f'<td class="num inline" data-k="took">{esc(_duration(r.elapsed_s) if r.start else "—")}</td>'
            f'<td class="num hide-sm">{esc(r.gpus)}</td>'
            f'<td class="num hide-sm">{esc(_gib(r.max_rss_bytes))}</td>'
            + (f'<td class="hide-sm">{esc(r.user)}</td>' if user_column else "")
            + "</tr>"
        )
    return "".join(out)


def _finished_table(rows: list[jobs_view.HistoryRow], root: str, now: datetime, *, user_column: bool = False) -> str:
    return (
        '<div class="table-wrap"><table class="stack"><thead><tr><th scope="col">Job</th><th scope="col">Project</th><th scope="col">Result</th>'
        '<th scope="col">Started</th><th scope="col">Took</th><th scope="col" class="hide-sm">GPUs</th>'
        '<th scope="col" class="hide-sm">Peak CPU memory</th>'
        + ('<th scope="col" class="hide-sm">User</th>' if user_column else "")
        + f"</tr></thead><tbody>{_finished_rows(rows, root, now, user_column=user_column)}</tbody></table></div>"
    )


def _is_finished(row: jobs_view.HistoryRow) -> bool:
    return row.end is not None and _result(row.state)[0] not in {Kind.RUNNING, Kind.WAITING}


def index_page(backend: Backend, root: str = "") -> str:
    rows = job_rows(backend)
    now = backend.now()
    day_ago = now - timedelta(days=1)
    recent = sorted(
        (r for r in backend.history(day_ago.strftime("%Y-%m-%dT%H:%M:%S"), None, None, None, None) if _is_finished(r) and r.end and r.end >= day_ago),
        key=lambda r: r.end or now,
        reverse=True,
    )
    failed = sum(1 for r in recent if _result(r.state)[0] in BAD_RESULTS)
    count = {kind: sum(1 for r in rows if r["kind"] == kind) for kind in (Kind.RUNNING, Kind.WAITING, Kind.PAUSED)}
    tiles = _tiles([
        ("Running", str(count[Kind.RUNNING]), _running_note(count[Kind.RUNNING], sum(1 for r in rows if r["kind"] == Kind.RUNNING and r["gpus"]))),
        ("Waiting", str(count[Kind.WAITING]), "in the queue" if count[Kind.WAITING] else "queue is empty"),
        ("Paused", str(count[Kind.PAUSED]), "resume from the job's page" if count[Kind.PAUSED] else ""),
        ("Finished, last 24 h", str(len(recent)), f"{failed} failed" if failed else "none failed"),
    ]) + _resource_tiles(backend.capacity(), rows, backend.gpu_busy())
    recent_html = (
        '<div class="card-head"><h2 id="h-recent">Finished in the last 24 hours</h2>'
        f'<a class="meta" href="{root}history">All history →</a></div>'
        + (_finished_table(recent[:RECENT_ROWS], root, now) if recent else '<div class="empty">No job finished in the last 24 hours.</div>')
    )
    body = _template("index.html").substitute(
        updated=now.strftime("%H:%M:%S"),
        now=_now_card(rows, backend.keeper(), backend, root),
        tiles=tiles,
        queue=_queue_table(rows, root),
        recent=recent_html,
    )
    return _page(backend, "Queue", body, root=root, tab=Tab.QUEUE)


# --- job page --------------------------------------------------------------------------------------------------------


def _node_lines(node: jobs_view.ChainNode, up: bool, root: str, lookup: Callable[[str], jobs_view.HistoryRow | None]) -> str:
    links = node.before if up else node.after
    name, state = node.name, node.state
    if not name or not state:  # a job no longer queued: slurm's accounting still knows its name and result
        row = lookup(node.job_id)
        name, state = name or (row.name if row else ""), state or (row.state if row else "")
    kind, label = _result(state) if state else (Kind.OTHER, "unknown")
    inner = "".join(_node_lines(child, up, root, lookup) for child in links)
    return (
        f'<li><a href="{root}job/{esc(node.job_id)}">{esc(_short(name) or "job " + node.job_id)}</a> '
        f'<span class="jobid">#{esc(node.job_id)}</span> {_badge(kind, label)}{f"<ul>{inner}</ul>" if inner else ""}</li>'
    )


def _chain_html(node: jobs_view.ChainNode, root: str, lookup: Callable[[str], jobs_view.HistoryRow | None]) -> str:
    if not node.before and not node.after:
        return '<p class="muted">Runs on its own: it waits for no job, and no job waits for it.</p>'
    parts = []
    if node.before:
        parts.append("<h3>Starts after</h3><ul>" + "".join(_node_lines(c, True, root, lookup) for c in node.before) + "</ul>")
    if node.after:
        parts.append("<h3>Then starts</h3><ul>" + "".join(_node_lines(c, False, root, lookup) for c in node.after) + "</ul>")
    return "".join(parts)


def _meter(label: str, value: str, fraction: float | None, note: str = "") -> str:
    bar = ""
    if fraction is not None:
        width = max(0.0, min(1.0, fraction)) * 100
        warn = " warn" if fraction > METER_WARN else ""
        bar = (
            '<svg viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true">'
            f'<rect class="track" width="100" height="8"/><rect class="fill{warn}" width="{width:.1f}" height="8"/></svg>'
        )
    return (
        f'<div class="meter"><div class="meter-row"><span class="k">{esc(label)}</span><span class="num">{esc(value)}</span></div>'
        f'{bar}{f"<div class=\"note\">{esc(note)}</div>" if note else ""}</div>'
    )


def _resources_html(eff: jobs_view.Efficiency, live: Mapping[str, int], kind: str) -> str:
    if kind in {Kind.WAITING, Kind.PAUSED} and eff.peak_gpu_mib is None and eff.peak_cpu_mem_bytes is None:
        return '<p class="muted">Not started yet: its memory and CPU use appear here once it runs.</p>'
    gpu_peak = None if eff.peak_gpu_mib is None else eff.peak_gpu_mib * 1024**2
    if kind == Kind.RUNNING:
        # slurm fills in peaks and CPU time when a job ends; while it runs, the agent's samples are what there is
        return "".join([
            _meter("GPU memory in use", _gib(live.get("gpu")), None, "sampled every few seconds"),
            _meter("CPU memory in use", _gib(live.get("cpu")), None),
            _meter("Peak GPU memory, earlier runs", _gib(gpu_peak), None) if gpu_peak is not None else "",
        ])
    parts = [_meter("Peak GPU memory", _gib(gpu_peak), None, "" if gpu_peak is not None else "not recorded for this job")]
    if eff.peak_cpu_mem_bytes is not None and eff.req_mem_bytes:
        share = eff.peak_cpu_mem_bytes / eff.req_mem_bytes
        parts.append(_meter("Peak CPU memory of asked", f"{_gib(eff.peak_cpu_mem_bytes)} of {_gib(eff.req_mem_bytes)}", share, f"{share:.0%} of the memory it asked for"))
    else:
        parts.append(_meter("Peak CPU memory", _gib(eff.peak_cpu_mem_bytes), None))
    if eff.cpu_use is not None:
        parts.append(_meter("CPU time used", f"{eff.cpu_use:.0%}", eff.cpu_use, "of the CPU time its cores could have given"))
    return "".join(parts)


EVENT_WORDS = {
    gpu_jobs.EventAction.PAUSE: "Paused",
    gpu_jobs.EventAction.RESUME: "Resumed",
    gpu_jobs.EventAction.KILL: "Cancelled",
    gpu_jobs.EventAction.URGENT_HOLD: "Held for urgent work",
    gpu_jobs.EventAction.RELEASE: "Released",
    gpu_jobs.EventAction.FAILED: "Failed",
}


def _timeline_html(row: jobs_view.HistoryRow | None, events: list[gpu_jobs.Event], now: datetime) -> str:
    items: list[tuple[datetime, str]] = []
    if row and row.submit:
        items.append((row.submit, "Submitted"))
    if row and row.start:
        items.append((row.start, "Started"))
    for event in events:
        moment = datetime.fromtimestamp(event.at / 1e9)  # noqa: DTZ006 - local, like slurm's times above
        words = EVENT_WORDS.get(event.action, event.action.replace("_", " ").capitalize())  # type: ignore[call-overload]
        items.append((moment, f"{words} by {event.actor}"))
    if row and row.end and _is_finished(row):
        items.append((row.end, f"{_result(row.state)[1]} after {_duration(row.elapsed_s)}"))
    if not items:
        return '<p class="muted">Slurm has no times for this job yet.</p>'
    items.sort(key=lambda item: item[0])
    return '<ul class="timeline">' + "".join(
        f'<li><span class="when num">{esc(_clock(moment, now))}</span><span>{esc(text)}</span></li>' for moment, text in items
    ) + "</ul>"


def _script_text(path: str) -> str:
    """The job script, read without ever blocking: the record's path may be a FIFO, a link or a device."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError:
        return NOT_SHOWN
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return NOT_SHOWN
        return os.read(fd, MAX_SCRIPT_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return NOT_SHOWN
    finally:
        os.close(fd)


def _command_text(record: Mapping[str, object]) -> str:
    command = record.get("command")
    text = shlex.join(str(c) for c in command) if isinstance(command, list) else ""
    script = record.get("script")
    if isinstance(script, str):
        text += ("\n\n" if text else "") + _script_text(script)
    return text or "no command recorded"


def _buttons_html(job: QueuedJob, steady: bool) -> str:
    ident = esc(job.job_id)
    paused = job.held
    pause = (  # a held job is paused already
        "" if paused else
        f'<button type="button" data-action="pause" data-job="{ident}" data-steady="{int(steady)}">Pause</button>'
    )
    resume = f'<button type="button" class="primary" data-action="resume" data-job="{ident}">Resume</button>' if paused else ""
    kill = f'<button type="button" class="danger" data-action="kill" data-job="{ident}">Cancel job</button>'
    return f'<div class="actions" aria-label="Actions on job {ident}">{pause}{resume}{kill}</div>'


def job_page(backend: Backend, job_id: str, root: str = "") -> str | None:
    queue = backend.queue() or []
    by_id = {j.job_id: j for j in queue}
    job = by_id.get(job_id)
    record = backend.record(job_id, user=job.user if job else None) or {}
    row = backend.finished(job_id)
    if job is None and not record and row is None:
        return None
    records = _records(backend, queue)
    records.setdefault(job_id, record)
    viewer = backend.viewer()
    now = backend.now()
    name = job.name if job else str(record.get("name") or (row.name if row else ""))
    project = str(record.get("project") or jobs_view.project_of(name))
    if job is not None:
        kind, label = _queue_state(job)
        why = jobs_view.waiting_reason(job, by_id, records, backend.events(None))
    elif row is not None:
        (kind, label), why = _result(row.state), ""
    else:
        kind, label, why = Kind.OTHER, "Finished", "slurm no longer has a record of it"
    user = job.user if job else (row.user if row else "")
    facts = [
        ("State", _badge(kind, label) + (f' <span class="muted small">{esc(_why(why))}</span>' if why else "")),
        ("Project", _project(project)),
        ("Description", esc(record.get("what") or "—")),
        ("Class", _class_pill(record.get("job_class") or (job.partition if job else row.partition if row else ""))),
        ("User", esc(user or "—")),
        ("Partition", esc((job.partition if job else row.partition if row else "") or "—")),
        ("GPUs", esc(record.get("gpus") or (job.gpus if job else row.gpus if row else "") or "—")),
        ("Memory asked", esc(_asked_memory(record.get("mem")) if record.get("mem") else _gib(row.req_mem_bytes) if row else "—")),
    ]
    if name and name != _short(name):
        facts.append(("Slurm name", f'<span class="mono">{esc(name)}</span>'))
    if row and _exit_words(row.exit_code) and kind != Kind.CANCELLED:
        facts.insert(1, ("Ended with", f'<span class="exit-bad">{esc(_exit_words(row.exit_code))}</span>'))
    jobstats = (
        # From the Jobstats README's example of its OOD helper at /pun/sys/jobstats/<jobid>.
        f'<a class="meta" href="{root.rstrip("/").rpartition("/")[0]}/jobstats/{esc(job_id)}">Jobstats report →</a>' if backend.metrics_on() else ""
    )
    buttons = f'<p class="muted small">{esc(CANNOT_ACT)}</p>' if job is not None else ""
    if job is not None and jobs_view.can_act(job, viewer, backend.is_admin(viewer)):
        # only a started steady job loses its run to a pause (pause.py)
        buttons = _buttons_html(job, job.partition == gpu_jobs.Partition.STEADY and job.state != gpu_jobs.State.PENDING)
    running = kind == Kind.RUNNING
    body = _template("job.html").substitute(
        job_id=esc(job_id),
        title=esc(_short(name) or f"Job {job_id}"),
        subtitle=f"Job #{esc(job_id)} · {esc(user or 'unknown user')}",
        facts="".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in facts),
        jobstats=jobstats,
        resources=_resources_html(backend.efficiency(job_id, record=record), backend.live_memory(job_id) if running else {}, kind),
        timeline=_timeline_html(row, backend.events(job_id), now),
        chain=_chain_html(jobs_view.chain(job_id, by_id, records), root, backend.finished),
        buttons=buttons,
        log_path=esc(record.get("log") or ""),
        command=esc(_command_text(record)),
    )
    return _page(backend, _short(name) or f"Job {job_id}", body, root=root, tab=None)


# --- history and projects --------------------------------------------------------------------------------------------


def _window(query: Mapping[str, list[str]], default_days: int = DEFAULT_WINDOW_DAYS) -> tuple[str, str | None]:
    def one(key: str) -> str:
        value = (query.get(key) or [""])[0]
        if value and not WHEN.fullmatch(value):
            raise Refused("400 Bad Request", f"{key} must look like YYYY-MM-DD")
        return value

    since = one("since") or (_today() - timedelta(days=default_days)).isoformat()
    return since, one("until") or None


def _presets(path: str, since: str, until: str | None, keep: Mapping[str, str]) -> str:
    links = []
    for days in PRESET_DAYS:
        start = (_today() - timedelta(days=days)).isoformat()
        params = urlencode({"since": start, **{k: v for k, v in keep.items() if v}})
        current = "true" if since == start and not until else "false"
        links.append(f'<a href="{path}?{esc(params)}" aria-current="{current}">{days} days</a>')
    return "".join(links)


def _options(values: Iterable[str], chosen: str, every: str, label: Callable[[str], str] = str) -> str:
    out = [f'<option value="">{esc(every)}</option>']
    out += [f'<option value="{esc(v)}"{" selected" if v == chosen else ""}>{esc(label(v))}</option>' for v in sorted(set(values))]
    return "".join(out)


def history_page(backend: Backend, query: Mapping[str, list[str]], root: str = "") -> str:
    since, until = _window(query)
    filters = {k: (query.get(k) or [""])[0] for k in ("user", "project", "state")}
    try:
        page = max(1, int((query.get("page") or ["1"])[0]))
    except ValueError:
        raise Refused("400 Bad Request", "page must be a number") from None
    every = backend.history(since, until, None, None, None)
    rows = [
        r for r in every
        if (not filters["user"] or r.user == filters["user"])
        and (not filters["project"] or r.project == filters["project"])
        and (not filters["state"] or r.state == filters["state"].upper())
    ]
    now = backend.now()
    rows.sort(key=lambda r: ((r.start or r.submit or now).timestamp() if r.start or r.submit else 0.0, int(r.job_id) if r.job_id.isdigit() else 0), reverse=True)
    pages = max(1, -(-len(rows) // HISTORY_PAGE))
    page = min(page, pages)
    shown = rows[(page - 1) * HISTORY_PAGE : page * HISTORY_PAGE]
    keep = {"since": since, "until": until or "", **filters}
    table = (
        _finished_table(shown, root, now, user_column=True) if shown
        else '<div class="empty"><strong>No jobs match</strong>Try a longer period or clear a filter.</div>'
    )

    def page_link(number: int, text: str) -> str:
        if number < 1 or number > pages or number == page:
            return f"<span>{esc(text)}</span>"
        return f'<a href="{root}history?{esc(urlencode({**{k: v for k, v in keep.items() if v}, "page": number}))}">{esc(text)}</a>'

    first = (page - 1) * HISTORY_PAGE + 1 if rows else 0
    pager = (
        f'<div class="pager"><span>{first}–{(page - 1) * HISTORY_PAGE + len(shown)} of {len(rows)} job(s)</span>'
        f'<span class="actions">{page_link(page - 1, "← Newer")}{page_link(page + 1, "Older →")}</span></div>'
    )
    body = _template("history.html").substitute(
        presets=_presets(f"{root}history", since, until, {k: v for k, v in filters.items()}),
        since=esc(since[:10]),
        until=esc((until or "")[:10]),
        project_options=_options((r.project for r in every), filters["project"], "Every project"),
        user_options=_options((r.user for r in every), filters["user"], "Every user"),
        state_options=_options((r.state for r in every), filters["state"].upper(), "Every result", lambda s: _result(s)[1]),
        table=table,
        pager=pager,
    )
    return _page(backend, "History", body, root=root, tab=Tab.HISTORY)


def _periods(since: str, until: str | None, bucket: jobs_view.Bucket) -> list[date]:
    start = date.fromisoformat(since[:10])
    end = date.fromisoformat(until[:10]) if until else _today()
    if bucket == jobs_view.Bucket.WEEK:
        start -= timedelta(days=start.weekday())
    step = timedelta(days=7 if bucket == jobs_view.Bucket.WEEK else 1)
    out = []
    while start <= end:
        out.append(start)
        start += step
    return out


def _nice_max(value: float) -> float:
    for nice in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000):
        if value <= nice:
            return float(nice)
    return float(int(value) + 1)


def _chart(rows: list[jobs_view.UsageRow], periods: list[date], projects: list[str], bucket: jobs_view.Bucket) -> str:
    """Stacked columns, one per period, one segment per project (largest at the base), with a 2 px gap between."""
    if not rows:
        return '<div class="empty">No GPU time in this period.</div>'
    by = {(r.period, r.project): r.gpu_hours for r in rows}
    totals = [sum(by.get((p, proj), 0.0) for proj in projects) for p in periods]
    top = _nice_max(max(totals))
    width, height, left, bottom = 1000, 260, 40, 24
    plot_h = height - bottom - 8
    slot = (width - left) / max(1, len(periods))
    bar_w = max(4.0, min(36.0, slot * 0.62))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="GPU-hours per {bucket.value} by project">']
    for tick in range(5):
        value = top * tick / 4
        y = 8 + plot_h - plot_h * tick / 4
        parts.append(f'<line class="grid-line" x1="{left}" x2="{width}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:g}</text>')
    label_every = max(1, -(-len(periods) // 12))
    for i, period in enumerate(periods):
        x = left + slot * i + (slot - bar_w) / 2
        y = 8 + plot_h
        for proj in projects:
            hours = by.get((period, proj), 0.0)
            if hours <= 0:
                continue
            h = plot_h * hours / top
            seg = max(0.0, h - 2)  # the 2 px surface gap between stacked segments
            y -= h
            tip = f"{proj}|{period:%a %-d %b}|{hours:.2f} GPU-hours"
            parts.append(
                f'<rect class="series bar {_colour(proj)}" x="{x:.1f}" y="{y + 2:.1f}" width="{bar_w:.1f}" height="{seg:.1f}" rx="2" data-tip="{esc(tip)}">'
                f"<title>{esc(proj)}, {period:%-d %b}: {hours:.2f} GPU-hours</title></rect>"
            )
        if i % label_every == 0:
            fmt = "%-d %b" if bucket == jobs_view.Bucket.DAY else "wk %-d %b"
            parts.append(f'<text class="axis" x="{x + bar_w / 2:.1f}" y="{height - 4}" text-anchor="middle">{period.strftime(fmt)}</text>')
    parts.append("</svg>")
    legend = "".join(f'<span><span class="dot {_colour(p)}"></span>{esc(p)}</span>' for p in projects)
    return f'<div class="chart">{"".join(parts)}</div><div class="legend">{legend}</div>'


def projects_page(backend: Backend, query: Mapping[str, list[str]], root: str = "") -> str:
    since, until = _window(query, PROJECTS_DEFAULT_DAYS)
    try:
        bucket = jobs_view.Bucket((query.get("bucket") or ["day"])[0])
    except ValueError:
        raise Refused("400 Bad Request", "bucket must be day or week") from None
    rows = backend.usage(since, until, bucket)
    totals: dict[str, float] = {}
    for r in rows:
        totals[r.project] = totals.get(r.project, 0.0) + r.gpu_hours
    projects = sorted(totals, key=lambda p: -totals[p])
    grand = sum(totals.values())
    periods = _periods(since, until, bucket)
    busiest = projects[0] if projects else "—"
    tiles = _tiles([
        ("GPU-hours", f"{grand:.1f}", f"since {date.fromisoformat(since[:10]):%-d %b}"),
        ("Busiest project", busiest, f"{totals[busiest]:.1f} GPU-hours ({totals[busiest] / grand:.0%})" if projects and grand else ""),
        ("Projects", str(len(projects)), "with GPU time in the period"),
        ("Busy share", f"{grand / max(1, len(periods) * (7 if bucket == jobs_view.Bucket.WEEK else 1) * 24):.0%}", "of the card's hours in the period"),
    ])
    peak = max(totals.values(), default=0.0) or 1.0
    totals_html = (
        '<div class="hbars">' + "".join(
            f"<div>{_project(p)}</div>"
            f'<svg viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true"><rect class="track" width="100" height="10"/>'
            f'<rect class="series {_colour(p)}" width="{max(1.0, 100 * totals[p] / peak):.1f}" height="10"/></svg>'
            f'<div class="num">{_hours(totals[p])}</div>'
            for p in projects
        ) + "</div>"
        if projects else '<div class="empty">No GPU time in this period.</div>'
    )
    buckets = "".join(
        f'<a href="{root}projects?{esc(urlencode({"since": since, **({"until": until} if until else {}), "bucket": b.value}))}" '
        f'aria-current="{"true" if b == bucket else "false"}">Per {b.value}</a>'
        for b in jobs_view.Bucket
    )
    table_rows = "".join(
        f'<tr><td class="num">{r.period:%a %-d %b %Y}</td><td>{_project(r.project)}</td><td class="num r">{r.gpu_hours:.2f}</td></tr>'
        for r in sorted(rows, key=lambda r: (r.period, -r.gpu_hours), reverse=True)
    )
    table = (
        '<div class="table-wrap"><table><thead><tr><th scope="col">Period</th><th scope="col">Project</th><th scope="col" class="r">GPU-hours</th></tr></thead>'
        f"<tbody>{table_rows}</tbody></table></div>"
    )
    body = _template("projects.html").substitute(
        presets=_presets(f"{root}projects", since, until, {"bucket": bucket.value}),
        buckets=buckets,
        tiles=tiles,
        window=esc(f"{date.fromisoformat(since[:10]):%-d %b} – {date.fromisoformat(until[:10]) if until else _today():%-d %b}"),
        totals=totals_html,
        bucket=esc(bucket.value),
        chart=_chart(rows, periods, projects, bucket),
        table=table,
    )
    return _page(backend, "Use per project", body, root=root, tab=Tab.PROJECTS)


def _check_post(environ: Mapping[str, object]) -> None:
    """The cross-site and content-type defences every action goes through."""
    if environ.get("REQUEST_METHOD") != "POST":
        raise Refused("403 Forbidden", "POST only")
    if str(environ.get("CONTENT_TYPE", "")).split(";")[0].strip().lower() != JSON_TYPE:
        raise Refused("403 Forbidden", "Content-Type must be application/json")
    origin = str(environ.get("HTTP_ORIGIN", ""))
    host = str(environ.get("HTTP_HOST", ""))
    if not origin or not host or urlsplit(origin).netloc != host:
        raise Refused("403 Forbidden", "Origin must be present and match Host")


def _read_json(environ: Mapping[str, object]) -> dict[str, object]:
    try:
        length = int(str(environ.get("CONTENT_LENGTH") or 0))
    except ValueError:
        length = -1
    if length < 0:
        raise Refused("400 Bad Request", "Content-Length must be a non-negative integer")
    if length > MAX_BODY:
        raise Refused("413 Payload Too Large", f"body is larger than {MAX_BODY} bytes")
    try:
        data = json.loads(environ["wsgi.input"].read(length) or b"{}")  # type: ignore[attr-defined]
    except (ValueError, KeyError, UnicodeDecodeError):
        raise Refused("400 Bad Request", "body must be JSON") from None
    if not isinstance(data, dict):
        raise Refused("400 Bad Request", "body must be a JSON object")
    return data


def run_action(backend: Backend, action: str, job_id: str, body: Mapping[str, object] | None = None, *, dry_run: bool = False) -> dict[str, object]:
    """Run one tool for one validated job id. Only the id and the boolean ever reach argv."""
    tool = "pause" if action == PREVIEW else action
    argv = [backend.python, str(backend.slurm_dir / f"{tool}.py"), job_id]
    if action == Action.PAUSE and body and body.get("lose_progress") is True:
        argv.append("--lose-progress")
    if dry_run:
        argv.append("--dry-run")
    code, output = backend.run(argv)
    return {"ok": code == 0, "output": output}


def _json(payload: object) -> tuple[str, str, bytes]:
    return "200 OK", JSON_TYPE, json.dumps(payload).encode()


def _api(backend: Backend, environ: Mapping[str, object], path: str, query: Mapping[str, list[str]]) -> tuple[str, str, bytes]:
    if path == "/api/jobs":
        return _json({"jobs": job_rows(backend)})
    match = ROUTE_API.fullmatch(path)
    if not match:
        raise Refused("404 Not Found", "no such route")
    job_id, verb = match.groups()
    if verb in {a.value for a in Action}:
        _check_post(environ)
        if not JOB_ID.fullmatch(job_id):
            raise Refused("403 Forbidden", "job id must be digits")
        return _json(run_action(backend, verb, job_id, _read_json(environ)))
    if environ.get("REQUEST_METHOD") != "GET" or not JOB_ID.fullmatch(job_id):
        raise Refused("404 Not Found", "no such route")
    if verb == PREVIEW:
        return _json(run_action(backend, PREVIEW, job_id, dry_run=True))
    if verb == LOG:
        job = next((j for j in backend.queue() or [] if j.job_id == job_id), None)
        record = backend.record(job_id, user=job.user if job else None) or {}
        path_text = str(record.get("log") or "")
        text = (query.get("text") or [""])[0] or None
        log = backend.log_tail(path_text, text=text, errors_only=query.get("errors") == ["1"]) if path_text else jobs_view.NO_LOG
        return _json({"log": log})
    raise Refused("404 Not Found", "no such route")


def handle(backend: Backend, environ: Mapping[str, object]) -> tuple[str, str, bytes]:
    path = str(environ.get("PATH_INFO") or "/")
    # Absolute links from the app's own mount point (Passenger's SCRIPT_NAME, /pun/sys/<app>): relative ones broke when
    # the page was opened without its trailing slash, which made the browser look for the stylesheet one level up.
    root = str(environ.get("SCRIPT_NAME") or "").rstrip("/") + "/"
    _SLOTS.set({project: slot for slot, project in enumerate(backend.project_order())})
    query = parse_qs(str(environ.get("QUERY_STRING", "")))
    method = environ.get("REQUEST_METHOD")
    if path.startswith("/api/"):
        return _api(backend, environ, path, query)
    if method != "GET":
        raise Refused("405 Method Not Allowed", "GET only")
    if path.startswith("/static/") and path[len("/static/"):] in STATIC:
        name = path[len("/static/"):]
        return "200 OK", STATIC[name], (APP_DIR / "static" / name).read_bytes()
    page: str | None
    if path == "/":
        page = index_page(backend, root)
    elif path == "/history":
        page = history_page(backend, query, root)
    elif path == "/projects":
        page = projects_page(backend, query, root)
    elif match := ROUTE_JOB.fullmatch(path):
        page = job_page(backend, match[1], root)
    else:
        page = None
    if page is None:
        raise Refused("404 Not Found", "no such page")
    return "200 OK", HTML_TYPE, page.encode()


def make_application(backend: Backend) -> Callable:
    def application(environ, start_response):
        try:
            status, content_type, payload = handle(backend, environ)
        except Refused as refusal:
            status, content_type = refusal.status, JSON_TYPE
            payload = json.dumps({"ok": False, "error": refusal.message}).encode()
        start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(payload))), *SECURITY_HEADERS])
        return [payload]

    return application


application = make_application(Backend())
