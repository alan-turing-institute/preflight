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
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, urlsplit

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
    for job in queue:
        record = records.get(job.job_id, {})
        rows.append(
            {
                "id": job.job_id,
                "user": job.user,
                "project": jobs_view.project_of(job.name),
                "class": str(record.get("job_class") or job.partition),
                "name": job.name,
                "description": str(record.get("what") or ""),
                "state": job.state,
                "waiting": jobs_view.waiting_reason(job, by_id, records, events),
                "elapsed": elapsed.get(job.job_id, ""),
                "gpus": job.gpus,
                "mem": str(record.get("mem") or ""),
            }
        )
    return rows


def _page(backend: Backend, title: str, body: str, *, root: str) -> str:
    viewer = backend.viewer()
    note = " (admin)" if backend.is_admin(viewer) else ""
    return _template("layout.html").substitute(
        title=esc(title), root=root, body=body, viewer=esc(viewer), admin_note=note
    )


def _cells(values: Iterable[object]) -> str:
    return "".join(f"<td>{esc(v)}</td>" for v in values)


def index_page(backend: Backend) -> str:
    rows = job_rows(backend)
    body_rows = "\n".join(
        "<tr>"
        + f'<td><a href="job/{esc(r["id"])}">{esc(r["id"])}</a></td>'
        + _cells(r[k] for k in ("user", "project", "class", "name", "description", "state", "waiting", "elapsed", "gpus", "mem"))
        + "</tr>"
        for r in rows
    )
    body = _template("index.html").substitute(count=len(rows), rows=body_rows)
    return _page(backend, "GPU jobs", body, root="")


def _node_lines(node: jobs_view.ChainNode, up: bool) -> str:
    links = node.before if up else node.after
    label = esc(f"job {node.job_id} {node.name} {node.state}".rstrip())
    inner = "".join(_node_lines(child, up) for child in links)
    return f"<li>{label}{f'<ul>{inner}</ul>' if inner else ''}</li>"


def _chain_html(node: jobs_view.ChainNode) -> str:
    if not node.before and not node.after:
        return "<p>No dependencies.</p>"
    parts = []
    if node.before:
        parts.append("<h3>Waits for</h3><ul>" + "".join(_node_lines(c, True) for c in node.before) + "</ul>")
    if node.after:
        parts.append("<h3>Waited for by</h3><ul>" + "".join(_node_lines(c, False) for c in node.after) + "</ul>")
    return "".join(parts)


def _gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 1024**3:.1f} GiB"


def _efficiency_html(eff: jobs_view.Efficiency) -> str:
    cpu = "unknown" if eff.cpu_use is None else f"{eff.cpu_use:.0%}"
    gpu = "unknown" if eff.peak_gpu_mib is None else f"{eff.peak_gpu_mib / 1024:.1f} GiB"
    rows = [
        ("Memory requested", _gib(eff.req_mem_bytes)),
        ("Peak CPU memory", _gib(eff.peak_cpu_mem_bytes)),
        ("Peak GPU memory", gpu),
        ("CPU time used of available", cpu),
    ]
    return "<table><tbody>" + "".join(f'<tr><th scope="row">{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in rows) + "</tbody></table>"


def _events_html(events: list[gpu_jobs.Event]) -> str:
    if not events:
        return "<p>No events.</p>"
    rows = "".join(
        f"<tr>{_cells((jobs_view._when(e.at), e.action, e.actor, json.dumps(e.fields, sort_keys=True) if e.fields else ''))}</tr>"
        for e in events
    )
    return (
        '<table><thead><tr><th scope="col">When</th><th scope="col">Action</th><th scope="col">By</th>'
        f'<th scope="col">Details</th></tr></thead><tbody>{rows}</tbody></table>'
    )


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
    pause = (
        f'<button type="button" data-action="pause" data-job="{ident}" data-steady="{int(steady)}">Pause job {ident}</button>'
    )
    resume = f'<button type="button" data-action="resume" data-job="{ident}">Resume job {ident}</button>'
    kill = f'<button type="button" class="danger" data-action="kill" data-job="{ident}">Kill job {ident}</button>'
    return pause + resume + kill


def job_page(backend: Backend, job_id: str) -> str | None:
    queue = backend.queue() or []
    by_id = {j.job_id: j for j in queue}
    job = by_id.get(job_id)
    record = backend.record(job_id, user=job.user if job else None) or {}
    if job is None and not record:
        return None
    records = _records(backend, queue)
    records.setdefault(job_id, record)
    viewer = backend.viewer()
    rows = {
        "Job": job_id,
        "Name": job.name if job else record.get("name", ""),
        "User": job.user if job else "",
        "State": job.state if job else "finished or unknown",
        "Waiting": jobs_view.waiting_reason(job, by_id, records, backend.events(None)) if job else "",
        "Partition": job.partition if job else "",
        "Project": record.get("project", ""),
        "Description": record.get("what", ""),
        "Class": record.get("job_class", ""),
        "GPUs": record.get("gpus", ""),
        "Memory": record.get("mem", ""),
        "Log file": record.get("log", ""),
    }
    record_rows = "".join(f'<tr><th scope="row">{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in rows.items())
    jobstats = (
        # From the Jobstats README's example of its OOD helper at /pun/sys/jobstats/<jobid>.
        f'<p><a href="../../jobstats/{esc(job_id)}">Per-job metrics (Jobstats)</a></p>' if backend.metrics_on() else ""
    )
    buttons = ""
    if job is not None and jobs_view.can_act(job, viewer, backend.is_admin(viewer)):
        buttons = _buttons_html(job, job.partition == gpu_jobs.Partition.STEADY)
    body = _template("job.html").substitute(
        job_id=esc(job_id),
        record_rows=record_rows,
        jobstats=jobstats,
        buttons=buttons or "<p>You cannot act on this job.</p>",
        command=esc(_command_text(record)),
        chain=_chain_html(jobs_view.chain(job_id, by_id, records)),
        efficiency=_efficiency_html(backend.efficiency(job_id, record=record)),
        events=_events_html(backend.events(job_id)),
    )
    return _page(backend, f"Job {job_id}", body, root="../")


def _window(query: Mapping[str, list[str]]) -> tuple[str, str | None]:
    def one(key: str) -> str:
        value = (query.get(key) or [""])[0]
        if value and not WHEN.fullmatch(value):
            raise Refused("400 Bad Request", f"{key} must look like YYYY-MM-DD")
        return value

    since = one("since") or (date.today() - timedelta(days=DEFAULT_WINDOW_DAYS)).isoformat()
    return since, one("until") or None


def history_page(backend: Backend, query: Mapping[str, list[str]]) -> str:
    since, until = _window(query)
    filters = {k: (query.get(k) or [""])[0] for k in ("user", "project", "state")}
    rows = backend.history(since, until, filters["user"] or None, filters["project"] or None, filters["state"] or None)
    cells = "\n".join(
        "<tr>"
        + _cells((r.job_id, r.name, r.project, r.user, r.state, r.exit_code, r.start or "", r.elapsed_s, r.gpus, _gib(r.max_rss_bytes)))
        + "</tr>"
        for r in rows
    )
    body = _template("history.html").substitute(
        since=esc(since), until=esc(until or ""), count=len(rows), rows=cells, **{k: esc(v) for k, v in filters.items()}
    )
    return _page(backend, "History", body, root="")


def projects_page(backend: Backend, query: Mapping[str, list[str]]) -> str:
    since, until = _window(query)
    try:
        bucket = jobs_view.Bucket((query.get("bucket") or ["day"])[0])
    except ValueError:
        raise Refused("400 Bad Request", "bucket must be day or week") from None
    rows = backend.usage(since, until, bucket)
    cells = "\n".join(f"<tr>{_cells((r.period, r.project, f'{r.gpu_hours:.2f}'))}</tr>" for r in rows)
    options = "".join(
        f'<option value="{b.value}"{" selected" if b == bucket else ""}>{b.value}</option>' for b in jobs_view.Bucket
    )
    body = _template("projects.html").substitute(since=esc(since), until=esc(until or ""), rows=cells, bucket_options=options)
    return _page(backend, "Use per project", body, root="")


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
        page = index_page(backend)
    elif path == "/history":
        page = history_page(backend, query)
    elif path == "/projects":
        page = projects_page(backend, query)
    elif match := ROUTE_JOB.fullmatch(path):
        page = job_page(backend, match[1])
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
