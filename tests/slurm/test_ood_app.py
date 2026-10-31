"""The OOD jobs app: every route answers on fake data, and the POST defences refuse what they must."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import signal
import sys
from datetime import date, datetime
from pathlib import Path
from wsgiref.util import setup_testing_defaults

import gpu_jobs
import jobs_view
import site_config
import pytest
from conftest import SLURM

sys.path.insert(0, str(SLURM / "ood_app"))
import app

HOST = "ood.example:8080"
MOUNT = "/pun/sys/gpujobs"  # where OOD serves the app (Passenger's SCRIPT_NAME)


def queued(job_id: str, user: str, name: str = "proj:normal:train", partition: str = "normal") -> gpu_jobs.QueuedJob:
    # 60 GiB: slurm's request, after an `scontrol update` the record's "4G" never saw
    return gpu_jobs.QueuedJob(job_id, partition, "RUNNING", "None", name, user=user, gpus=1, mem_mb=61440,
                              node="gpu-node-1")


class Fake:
    """Backend fakes plus a record of what ran."""

    def __init__(self, viewer: str = "alice", admin: bool = False, name: str = "proj:normal:train") -> None:
        self.ran: list[list[str]] = []
        self.headers: dict[str, str] = {}  # of the last response
        self.asked: list[tuple[str, str | None]] = []  # (job id, the slurm user the record must belong to)
        record = {"what": "a run", "job_class": "normal", "mem": "4G", "project": "proj", "command": ["python", "x.py"], "log": "/logs/x.log"}
        self.backend = app.Backend(
            queue=lambda: [queued("7", "alice", name), queued("8", "bob", partition="steady")],
            record=lambda job_id, user=None: self.asked.append((job_id, user)) or record,
            events=lambda _j: [gpu_jobs.Event("7", "pause", "alice", 1_700_000_000_000_000_000),
                               gpu_jobs.Event("7", "push_back", "root", 1_700_000_100_000_000_000,
                                              {"node": "laptop", "free_mb": 7300})],
            elapsed=lambda: {"7": "1:00"},
            efficiency=lambda job_id, record=None: jobs_view.Efficiency(job_id, 4 * 1024**3, None, None, 0.5),
            history=lambda *_a: [],
            usage=lambda *_a: [jobs_view.UsageRow(date(2026, 10, 19), "proj", 2.0)],
            log_tail=lambda path, text=None, errors_only=False: f"tail of {path} {text} {errors_only}",
            viewer=lambda: viewer,
            is_admin=lambda user: admin and user == viewer,
            metrics_on=lambda: True,
            run=lambda argv: (self.ran.append(argv) or 0, "done"),
            python="py",
            slurm_dir=Path("/slurm"),
            host=lambda: "box",
            keeper=lambda: {"blockers": []},
            idle=lambda: [{"text": "Waiting: #596 needs all memory; #611 p holds 24.0 GB (restart, cannot be moved) <b>"}],
            live_memory=lambda _j: {"gpu": 3 * 1024**3},
            finished=lambda _j: None,
            now=lambda: datetime(2026, 10, 19, 12, 0),  # noqa: DTZ001 - local, like the times sacct prints
            project_order=lambda: ["proj"],
            capacity=lambda: {"cpus": 20, "cpus_used": 8, "mem_mb": 102400, "mem_used_mb": 65536, "gpus": 1, "gpus_used": 1,
                              "nodes": 1, "here": 1},
            node_capacity=lambda: [],  # one machine: never the live cluster's
            machine=lambda: {"mem_total": 120 * 1024**3, "mem_used": 110 * 1024**3, "cores": 20, "cores_busy": 3.0},
            gpu_busy=lambda: 0.37,
        )
        self.application = app.make_application(self.backend)


def call(fake: Fake, path: str, method: str = "GET", body: object | None = None, headers: dict[str, str] | None = None):
    environ: dict[str, object] = {}
    setup_testing_defaults(environ)
    route, _, query = path.partition("?")
    payload = json.dumps(body).encode() if body is not None else b""
    environ.update(PATH_INFO=route, QUERY_STRING=query, REQUEST_METHOD=method, HTTP_HOST=HOST, SCRIPT_NAME=MOUNT)
    environ.update({"wsgi.input": io.BytesIO(payload), "CONTENT_LENGTH": str(len(payload))})
    if method == "POST":
        environ.update(CONTENT_TYPE="application/json", HTTP_ORIGIN=f"http://{HOST}")
    for key, value in (headers or {}).items():
        if value:
            environ[key] = value
        else:
            environ.pop(key, None)
    seen: list[str] = []

    def start_response(status: str, response_headers: list[tuple[str, str]]) -> None:
        seen.append(status)
        fake.headers = dict(response_headers)

    out = b"".join(fake.application(environ, start_response))
    return seen[0], out.decode()


@pytest.mark.parametrize(
    "path",
    [
        "/", "/api/jobs", "/job/7", "/job/8", "/history", "/projects?bucket=week", "/static/app.js", "/static/app.css",
        "/api/job/7/log?text=oom&errors=1", "/api/job/7/pause-preview",
    ],
)
def test_every_get_route_answers(path: str) -> None:
    fake = Fake()
    status, body = call(fake, path)
    assert status == "200 OK", body
    assert fake.headers["Cache-Control"] == "no-store" and fake.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in fake.headers["Content-Security-Policy"]
    # a queued job's record is asked for as the user slurm says owns the job, so a record another user wrote is ignored
    assert all(user == {"7": "alice", "8": "bob"}[job_id] for job_id, user in fake.asked)
    if path == "/":  # both jobs hold a GPU; a CPU-only job running beside them is not said to be on it
        assert "On the GPU now" in body and "on the GPU now" in body
        assert "Waiting: #596 needs all memory; #611 p holds 24.0 GB (restart, cannot be moved) &lt;b&gt;" in body  # the agent's idle finding, escaped
        assert (app._running_note(2, 0), app._running_note(2, 1)) == ("none on the GPU", "1 on the GPU")
        # each resource as reserved by jobs (of what slurm can give) beside really in use (of the whole machine), in GB
        assert "Reserved by jobs" in body and "1 of 1" in body and "37%" in body
        assert "8 of 20" in body and "3.0 of 20" in body and "12 left to reserve" in body
        assert "68.7 of 107.4 GB" in body and "38.7 GB left to reserve" in body
        assert "118.1 of 128.8 GB" in body and "10.7 GB available" in body
        assert "GiB" not in body and "MiB" not in body and "memory in use" not in body.split('class="card box"')[0]
        assert body.count("used warn") == 1 and "fill warn" not in body  # only real memory in use warns; a full reservation or a busy GPU does not
        assert "64.4 GB</td>" in body  # slurm's request now (60 x 2^30 bytes), not the record's 4G
        # where a running job is, as a small line inside the state cell; a pushed-back job's timeline says why it moved
        assert '<div class="why">on gpu-node-1</div>' in body
        assert "Pushed back: 7.3 GB free on laptop" in call(fake, "/job/7")[1]
        # who queued each job, and its level, at every width (not hide-sm)
        assert '<td class="inline" data-k="user">alice</td>' in body and '<td class="inline" data-k="level"><span class="pill level-normal"' in body
        # its own column: when slurm (last) queued a job, and for a waiting one how long ago; unknown shows a dash
        noon = datetime(2026, 10, 19, 12, 0)  # noqa: DTZ001 - local, like the fake's now
        waiting = {"kind": app.Kind.WAITING, "state": "PENDING", "submitted": int(datetime(2026, 10, 19, 9, 50).timestamp())}  # noqa: DTZ001
        assert 'Today 09:50<div class="why">(queued for 2 h 10 min)</div>' in app._queued_cell(waiting, noon)
        assert ">Today 09:50</td>" in app._queued_cell({**waiting, "kind": app.Kind.RUNNING, "state": "RUNNING"}, noon)
        assert "—" in app._queued_cell({**waiting, "submitted": 0}, noon)
        # a finished job: how long it waited to start; one cancelled before it started, until it ended
        at = lambda h, m: datetime(2026, 10, 19, h, m)  # noqa: DTZ001, E731 - local, like the times sacct prints
        row = jobs_view.HistoryRow("5", "p:normal:x", "p", "ann", "normal", "COMPLETED", "0:0", at(9, 0), at(10, 30), at(11, 0), 1800, 1, None)
        assert "Today 09:00<div class=\"why\">(queued for 1 h 30 min)</div>" in app._finished_rows([row], "/", noon)
        never = dataclasses.replace(row, state="CANCELLED", start=None, elapsed_s=0)
        assert "(queued for 2 h 00 min)" in app._finished_rows([never], "/", noon)
        assert "<th scope=\"col\">Running for</th><th scope=\"col\">Queued at</th>" in body
        # nvidia-smi and /proc read the portal's own host: no in-use bar unless that host is slurm's one node
        elsewhere = app._resource_tiles({"cpus": 1, "cpus_used": 0, "mem_mb": 1024, "mem_used_mb": 0, "gpus": 1, "gpus_used": 1,
                                         "nodes": 1, "here": 0}, [], 0.5, {"mem_total": 1, "mem_used": 1, "cores": 1, "cores_busy": 1})
        assert "Reserved by jobs" in elsewhere and "Busy" not in elsewhere and "In use" not in elsewhere
        # two machines: a strip for the cluster, then one each; only the page's own machine reads /proc, the other's reading
        # is the one its health check left in its comment
        total = {"cpus": 28, "cpus_used": 4, "mem_mb": 112000, "mem_used_mb": 0, "gpus": 1, "gpus_used": 0, "nodes": 2, "here": 1}
        dgx = {"name": "dgx", "here": 1, "cpus": 20, "cpus_used": 4, "mem_mb": 100000, "mem_used_mb": 0, "gpus": 1, "gpus_used": 0, "comment": ""}
        lap = {"name": "lap", "here": 0, "cpus": 8, "cpus_used": 0, "mem_mb": 12000, "mem_used_mb": 0, "gpus": 0, "gpus_used": 0,
               "comment": "available_mb=9000;total_mb=30000;cores=22;cores_busy=1.5"}
        running = [{"kind": app.Kind.RUNNING, "node": "lap", "cpus": 8, "gpus": 0}]
        both = app._resource_tiles(total, running, 0.5, {"mem_total": 2 * 10**9, "mem_used": 10**9, "cores": 20, "cores_busy": 3.0}, [dgx, lap])
        strips = both.split('<section')[1:]
        assert [x.split("</h2>")[0].rsplit(">", 1)[1] for x in strips] == ["Whole cluster", "dgx", "lap"]
        assert "3.0 of 20" in strips[1] and "1.5 of 22" in strips[2] and "21.0 of 30.0 GB" in strips[2]
        assert "4.5 of 42" in strips[0] and "22.0 of 32.0 GB" in strips[0] and "Busy</span><span class=\"num\">50%" in strips[0]  # summed
        assert ">GPU<" in strips[1] and ">GPU<" not in strips[2] and "not reported" not in both  # the laptop has none
        lap_silent = {**lap, "comment": ""}  # a machine that has not reported: said so on its strip and the cluster's, left out of the sum
        silent = app._resource_tiles(total, running, 0.5, {"mem_total": 2 * 10**9, "mem_used": 10**9, "cores": 20, "cores_busy": 3.0}, [dgx, lap_silent])
        assert silent.count("in use not reported by lap") == 2 and "3.0 of 20" in silent.split("<section")[1]


def test_unknown_things_are_not_found_or_bad() -> None:
    fake = Fake()
    assert call(fake, "/job/abc")[0].startswith("404")
    assert call(fake, "/projects?bucket=year")[0].startswith("400")
    assert call(fake, "/history?since=x;rm")[0].startswith("400")
    assert call(fake, "/api/job/7/pause")[0].startswith("403")  # GET on an action
    assert call(fake, "/", method="POST")[0].startswith("405")
    assert call(fake, "/api/job/7/pause", "POST", [1])[0].startswith("400")  # a JSON body that is not an object
    assert fake.ran == []


def test_action_posts_are_refused_unless_origin_json_and_id_are_right() -> None:
    fake = Fake()
    url = "/api/job/7/kill"
    body = {"x": 1}
    refused = [
        call(fake, url, "POST", body, {"HTTP_ORIGIN": ""}),
        call(fake, url, "POST", body, {"HTTP_ORIGIN": "http://evil.example"}),
        call(fake, url, "POST", body, {"HTTP_ORIGIN": "null"}),
        call(fake, url, "POST", body, {"HTTP_ORIGIN": f"http://{HOST}.evil.example"}),
        call(fake, url, "POST", body, {"HTTP_ORIGIN": f"http://evil.example/{HOST}"}),
        call(fake, url, "POST", body, {"HTTP_ORIGIN": f"http://{HOST}@evil.example"}),
        call(fake, url, "POST", body, {"CONTENT_TYPE": "text/plain"}),
        call(fake, "/api/job/7;rm/kill", "POST", body),
        call(fake, "/api/job/abc/kill", "POST", body),
        call(fake, "/api/part-time/lap/close", "POST", body, {"HTTP_ORIGIN": "http://evil.example"}),
    ]
    assert [status[:3] for status, _ in refused] == ["403"] * len(refused)
    assert fake.ran == []


def test_accepted_posts_run_exactly_the_expected_argv() -> None:
    fake = Fake()
    assert json.loads(call(fake, "/api/job/7/pause", "POST", {"lose_progress": True, "evil": "; rm"})[1]) == {"ok": True, "output": "done"}
    call(fake, "/api/job/7/pause", "POST", {"lose_progress": "yes"})
    call(fake, "/api/job/7/resume", "POST", {})
    call(fake, "/api/job/7/kill", "POST", {})
    call(fake, "/api/job/7/pause-preview")
    assert fake.ran == [
        ["py", "/slurm/pause.py", "7", "--lose-progress"],
        ["py", "/slurm/pause.py", "7"],
        ["py", "/slurm/resume.py", "7"],
        ["py", "/slurm/kill.py", "7"],
        ["py", "/slurm/pause.py", "7", "--dry-run"],
    ]


def test_part_time_card_is_the_admins_and_its_buttons_leave_requests_the_agent_reads(tmp_path: Path) -> None:
    site = site_config.parse({"name": "t", "nodes": [
        {"name": "head", "cpus": 8, "real_memory_mb": 60000},
        {"name": "lap", "cpus": 8, "real_memory_mb": 30000, "part_time": True, "address": "10.0.0.2", "keep_cpus": "0-3"}]})
    state = app.PartTimeState(closed=True, reason="gpujobs-check: 4 GB free", free_mb=3500, resume_jobs=2, extra={"close_below_mb": 7000},
                              copied_at=0)
    plain, boss = Fake(), Fake(viewer="root", admin=True)
    for fake in (plain, boss):
        fake.backend.site, fake.backend.requests_dir = (lambda: site), tmp_path
        fake.backend.part_time_state = lambda names: {"lap": state}
    assert (app._available_mb("available_mb=10284"), app._available_mb("a note"), app._available_mb("")) == (10284, None, None)
    assert "Part-time machines" not in call(plain, "/")[1]
    page = call(boss, "/")[1]
    assert "Part-time machines" in page and "3.5 GB" in page and 'data-jobs="2"' in page and 'name="close_below_gb" type="number" step="any" min="0" value="7"' in page
    assert "Job state last copied from this machine:" in page and "read from the shared state directory" not in page
    shared = dataclasses.replace(site, shared_state=True)
    boss.backend.site = lambda: shared
    assert "read from the shared state directory" in call(boss, "/")[1] and "last copied" not in call(boss, "/")[1]
    boss.backend.site = lambda: site
    assert call(plain, "/api/part-time/lap/close", "POST", {})[0][:3] == "403" and list(tmp_path.iterdir()) == []
    assert call(boss, "/api/part-time/head/close", "POST", {})[0][:3] == "404"  # a full node is not lent
    assert json.loads(call(boss, "/api/part-time/lap/thresholds", "POST", {"close_below_gb": 9, "evil": "x"})[1])["ok"]
    (request,) = tmp_path.glob("part_time-lap-*.json")
    assert json.loads(request.read_text()) == {"action": "thresholds", "close_below_gb": 9}
    request.unlink()
    for bad in (float("nan"), float("inf"), -float("inf")):  # sent as JSON's NaN/Infinity, which Python parses
        assert call(boss, "/api/part-time/lap/resources", "POST", {"jobs_memory_gb": bad})[0][:3] == "400"
    assert list(tmp_path.glob("part_time-*.json")) == []



def test_the_log_of_a_job_that_ran_on_a_part_time_machine_is_its_copied_back_one(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = Fake()
    fake.backend.log_tail = lambda path, text=None, errors_only=False: "" if path == "/copy/x.log" else jobs_view.NO_LOG
    monkeypatch.setattr(gpu_jobs, "remote_log", lambda job, log, user=None: Path("/copy/x.log") if job == "7" else None)
    assert json.loads(call(fake, "/api/job/7/log")[1]) == {"log": ""}
    assert json.loads(call(fake, "/api/job/8/log")[1]) == {"log": jobs_view.NO_LOG}


def test_a_fifo_script_and_a_bad_content_length_never_hang_or_run(tmp_path: Path) -> None:
    fifo = tmp_path / "job.sbatch"
    os.mkfifo(fifo)
    fake = Fake()
    fake.backend.record = lambda _j, user=None: {"script": str(fifo), "command": ["x"]}
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("the job page blocked on the script"))
    signal.alarm(5)
    try:
        status, page = call(fake, "/job/7")
    finally:
        signal.alarm(0)
    assert status == "200 OK" and app.NOT_SHOWN in page
    for length, refused in (("-1", "400"), ("x", "400"), (str(app.MAX_BODY + 1), "413")):
        status, _ = call(fake, "/api/job/7/pause", "POST", {}, {"CONTENT_LENGTH": length})
        assert status.startswith(refused)
    assert fake.ran == []
    assert call(Fake(), "/api/job/7/log?errors=0")[1].endswith("None False\"}")  # only errors=1 turns the filter on
    assert call(Fake(), "/api/job/7/log?errors=1")[1].endswith("None True\"}")


def test_values_are_html_escaped() -> None:
    fake = Fake(name="<script>alert(1)</script>:normal:x")
    for path in ("/", "/job/7"):
        _, page = call(fake, path)
        assert "<script>alert" not in page
        assert "&lt;script&gt;alert(1)" in page


@pytest.mark.parametrize(
    ("viewer", "admin", "shown"), [("alice", False, True), ("carol", False, False), ("carol", True, True)]
)
def test_buttons_follow_can_act(viewer: str, admin: bool, shown: bool) -> None:
    _, page = call(Fake(viewer, admin), "/job/7")
    assert ('data-action="kill"' in page) is shown
    assert (app.esc(app.CANNOT_ACT) in page) is not shown  # someone who cannot act is told why there are no buttons
    held = gpu_jobs.QueuedJob("9", "steady", "PENDING", "JobHeldUser", "n")
    assert 'data-action="pause"' not in app._buttons_html(held, steady=False)  # paused already
    assert 'data-action="resume"' in app._buttons_html(held, steady=False)
    # one slurm suspended for a service job (D20) resumes by itself: no Resume button, and the page says so
    suspended = gpu_jobs.QueuedJob("10", "cpu", "SUSPENDED", "None", "n")
    assert 'data-action="resume"' not in app._buttons_html(suspended, steady=False)
    assert app._queue_state(suspended)[1] == "Suspended (resumes by itself)"
    rows = [{"kind": app.Kind.PAUSED, "state": "PENDING"}, {"kind": app.Kind.PAUSED, "state": "SUSPENDED"}]
    assert app._paused_note(rows) == "resume from the job's page; 1 suspended, resumes by itself"
    assert 'data-action="resume"' not in page  # job 7 runs: nothing to resume
    assert "/pun/sys/jobstats/7" in page
    # links are absolute from the mount: opened without its trailing slash, the page still finds its stylesheet
    # and asks for it by the installed files' version, so a browser holding an older copy fetches the new one
    assert f'href="{MOUNT}/static/app.css?v={app._assets_version()}"' in page


def test_the_tables_show_memory_used_of_asked_and_cores_used_of_given() -> None:
    from datetime import datetime

    assert app._queue_memory({"mem_mb": 38000, "mem_used": 38_650_802_176}).startswith("38.7 GB of ")
    assert "of" not in app._queue_memory({"mem_mb": 38000, "mem_used": None})  # waiting: only what it asked for
    row = jobs_view.HistoryRow("5", "p:normal:x", "p", "ann", "normal", "COMPLETED", "0:0", datetime(2026, 1, 1), datetime(2026, 1, 1),
                               datetime(2026, 1, 1), 100, 0, 4 * 1024**3, 2 * 1024**3, cpus=4, cpu_s=150.0)
    assert app._cores_used(row) == "1.5 of 4"
    assert app._finished_memory(row) == f"{app._size(2 * 1024**3)} of {app._size(4 * 1024**3)}"
