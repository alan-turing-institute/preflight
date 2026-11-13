"""The OOD jobs app: every route answers on fake data, and the POST defences refuse what they must."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import re
import signal
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
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
SILENT_NOTES = 2  # a silent machine is said so on its own strip and on the cluster's


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
                                              {"node": "laptop", "free_mb": 7300}),
                               gpu_jobs.Event("7", "idle", "slurm", 1_700_000_200_000_000_000, {"message": "holds 8 cores"}),
                               gpu_jobs.Event("7", "idle", "alice", 1_700_000_300_000_000_000, {"message": "forged by alice"}),
                               gpu_jobs.Event("7", "memory_stop", "slurm", 1_700_000_400_000_000_000, {"message": "held 9 GB"}),
                               gpu_jobs.Event("7", "memory_stop", "alice", 1_700_000_500_000_000_000, {"message": "forged stop"})],
            elapsed=lambda: {"7": "1:00"},
            efficiency=lambda job_id, record=None: jobs_view.Efficiency(job_id, 4 * 1024**3, None, None, 0.5),
            history=lambda *_a: [],
            log_tail=lambda path, text=None, errors_only=False: f"tail of {path} {text} {errors_only}",
            viewer=lambda: viewer,
            is_admin=lambda user: admin and user == viewer,
            metrics_on=lambda: True,
            run=lambda argv: (self.ran.append(argv) or 0, "done"),
            python="py",
            slurm_dir=Path("/slurm"),
            host=lambda: "box",
            keeper=lambda: {"blockers": []},
            idle=lambda: [{"waiting": "596", "rule": "memory", "idle": "the GPU unreserved", "holders": [
                {"job": "611", "project": "p<b>", "amount": "24.0 GB", "job_class": "restart", "movable": False}]}],
            live_memory=lambda _j: {"gpu": 3 * 1024**3},
            finished=lambda _j: None,
            now=lambda: datetime(2026, 10, 19, 12, 0),  # noqa: DTZ001 - local, like the times sacct prints
            project_slots=lambda: {"proj": 0},
            capacity=lambda: {"cpus": 20, "cpus_used": 8, "mem_mb": 102400, "mem_used_mb": 65536, "gpus": 1, "gpus_used": 1,
                              "nodes": 1, "here": 1},
            node_capacity=lambda: [],  # one machine: never the live cluster's
            partition_tiers=lambda: None,  # never the live cluster's
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
        assert "Job #596 waits for memory with the GPU unreserved; held by #611 p&lt;b&gt; holds 24.0 GB (restart, cannot be moved)" in body  # the agent's idle finding, escaped
        assert (app._running_note(2, 0), app._running_note(2, 1)) == ("none on the GPU", "1 on the GPU")
        # the State cell lists each job waited on by name, one per line, its description on hover (2026-11-08)
        waiting = gpu_jobs.QueuedJob("9", "normal", "PENDING", "Dependency", "p:restart:x", dependency="afterok:2203:2206")
        upstream = gpu_jobs.QueuedJob("2203", "normal", "PENDING", "Resources", "p:restart:store<b>")
        waits = app._waits_on(waiting, {"2203": upstream}, {"2203": {"what": "put the <scores> in"}})
        assert app._waits_cell(waits, "/r/") == (
            '<div class="why">waits for:</div><ul class="waits">'
            '<li><a href="/r/job/2203" title="store&lt;b&gt; — put the &lt;scores&gt; in">store&lt;b&gt;</a> <span class="jobid">#2203</span></li>'
            '<li><a href="/r/job/2206" title="">job 2206</a> <span class="jobid">#2206</span></li></ul>')
        # one line each: the words a job's name shares with the waiting job's go, a long rest loses its middle (full on hover)
        assert app._middle(app._unshared("study-2-18-run-small-model-fold-36", "study-2-18-record-x")) == "…run-small…fold-36"
        assert app._unshared("store", "store") == "store"  # the last word always stays
        # more than three fold behind one line, kept open across the 10 s refresh by its key
        assert app._waits_cell(waits * 2, "/r/", "9").startswith(
            '<details class="waits" data-key="waits-9"><summary class="why">waits for 4 jobs</summary>')
        # each resource as reserved by jobs (of what slurm can give) beside really in use (of the whole machine), in GB
        assert "Reserved by jobs" in body and "1 of 1" in body and "37%" in body
        assert "8 of 20" in body and "3.0 of 20" in body and "12 left to reserve" in body
        assert "68.7 of 107.4 GB" in body and "38.7 GB left to reserve" in body
        assert "118.1 of 128.8 GB" in body and "10.7 GB available" in body
        assert "GiB" not in body and "MiB" not in body and "memory in use" not in body.split('class="card box"')[0]
        assert body.count("used warn") == 1 and "fill warn" not in body  # only real memory in use warns; a full reservation or a busy GPU does not
        assert "asked</span> 64.4 GB</div></td>" in body  # slurm's request now (60 x 2^30 bytes), not the record's 4G
        # where a running job is, as a small line inside the state cell; a pushed-back job's timeline says why it moved
        assert '<div class="why">on gpu-node-1</div>' in body
        job_page = call(fake, "/job/7")[1]
        assert "<dt>Checked before queuing</dt><dd>—</dd>" in job_page  # a record from before D47
        assert [app._checked(c) for c in ({"none": "one-off"}, {"command": "make check", "exit": 0, "seconds": 3.2},
                                          {"command": "make check", "reused_from": 1.0}, {"policy": "off"})] == [
            "no check: one-off", "`make check` passed in 3.2 s", "`make check` passed earlier (reused)", "not stated (policy off)"]
        assert "Pushed back: 7.3 GB free on laptop" in job_page
        assert "Idle finding: holds 8 cores" in job_page and "forged by alice" not in job_page  # only the agent's (D41)
        assert "Stopped for memory: held 9 GB" in job_page and "forged stop" not in job_page  # (D44)
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
        assert 'data-k="ended">Today 11:00</td>' in app._finished_rows([row], "/", noon)  # when it finished, beside Started
        never = dataclasses.replace(row, state="CANCELLED", start=None, elapsed_s=0)
        assert "(queued for 2 h 00 min)" in app._finished_rows([never], "/", noon)
        # D48: a finished job's memory is accounting's (an `scontrol update` never reaches the record's 4G), and a job
        # waiting for room is told, per node, what it lacks and who holds it
        gone = jobs_view.HistoryRow("12", "proj:normal:train", "proj", "alice", "normal", "COMPLETED", "0:0", at(9, 0),
                                    at(9, 5), at(10, 0), 3300, 1, 30 * 1024**3)
        backend = fake.backend
        fake.application = app.make_application(dataclasses.replace(backend, finished=lambda _j: gone))
        assert f"<dt>Memory asked</dt><dd>{app.esc(app._size(30 * 1024**3))}</dd>" in call(fake, "/job/12")[1]
        waiting_job = dataclasses.replace(queued("13", "alice"), state="PENDING", reason="Resources", node="", mem_mb=20000)
        row = {"name": "gpu-node-1", "state": "MIXED", "partitions": "normal", "cpus": 20, "cpus_used": 4, "mem_mb": 70000,
               "mem_used_mb": 61440, "gpus": 1, "gpus_used": 1, "shards": 0, "shards_used": 0}
        fake.application = app.make_application(dataclasses.replace(
            backend, queue=lambda: [queued("7", "alice"), waiting_job], node_capacity=lambda: [row],
            site=lambda: SimpleNamespace(nodes=[], cpu_job_cap=None, service=None)))
        assert "it lacks 12.0 GB memory and 1 GPU, held by #7 proj 64.4 GB" in call(fake, "/job/13")[1]
        fake.application = app.make_application(backend)
        assert re.search(r'>Running for</a></th><th scope="col"><a class="sort" [^>]*>Queued at</a></th>', body)  # side by side
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
        assert silent.count("in use not reported by lap") == SILENT_NOTES and "3.0 of 20" in silent.split("<section")[1]


    if path.startswith("/projects"):  # 4 cores for an hour are 4 core-hours; the machine tiles say what share was held
        ran = jobs_view.HistoryRow("9", "proj:resume:x", "proj", "u", "cpu", "COMPLETED", "0:0", None,
                                   datetime(2026, 10, 19, 1), datetime(2026, 10, 19, 2), 3600, 0, None, node="n",
                                   cpus=4, cpu_s=3600.0)
        page = app.projects_page(dataclasses.replace(fake.backend, history=lambda *_a: [ran], site=lambda: None),
                                 {"since": ["2026-10-19"], "until": ["2026-10-20"]})
        assert "Core-hours by project" in page and "4.0 h" in page and "25% used" in page and "GPU held" in page


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



def test_the_maintenance_banner_is_the_admins_and_its_buttons_leave_requests_for_the_restart_slot(tmp_path: Path) -> None:
    site = site_config.parse({"name": "t", "nodes": [{"name": "head", "cpus": 8, "real_memory_mb": 60000}],
                              "maintenance": {"status_file": "/slot/status.json", "request_dir": str(tmp_path)}})
    plain, boss = Fake(), Fake(viewer="root", admin=True)
    now = int(boss.backend.now().timestamp())
    status: dict[str, object] = {"updated": now, "state": "planned", "planned_at": now + 3600, "detail": "waits for jobs",
                                 "waiting": [{"kind": "kernel", "summary": "new kernel <b>6.17</b>"}]}
    for fake in (plain, boss):
        fake.backend.site = lambda: site
        fake.backend.maintenance_status = lambda _path: status
    assert "Updates waiting for a restart" not in call(plain, "/")[1]  # admins only (D3)
    page = call(boss, "/")[1]
    assert "A restart is planned for" in page and "new kernel &lt;b&gt;6.17&lt;/b&gt;" in page  # escaped: another program wrote it
    assert 'data-maintenance="restart_now"' in page and 'data-maintenance="skip_week"' in page
    for unavailable in (None, {**status, "updated": now - app.MAINTENANCE_STALE_S - 1}, {**status, "updated": now + 3600},
                        {**status, "state": "rebooting"}, {**status, "waiting": "kernel"}):
        boss.backend.maintenance_status = lambda _path, raw=unavailable: raw
        assert "Maintenance status unavailable" in call(boss, "/")[1]  # never read as "nothing waits"
    boss.backend.maintenance_status = lambda _path: {**status, "state": "none", "waiting": []}
    assert "Updates waiting for a restart" not in call(boss, "/")[1]  # nothing waits: no banner
    boss.backend.maintenance_status = lambda _path: status
    boss.backend.can_write = lambda _path: False  # an admin the slot's folder does not admit gets no buttons
    assert "data-maintenance" not in call(boss, "/")[1] and "not writable for you" in call(boss, "/")[1]
    assert call(plain, "/api/maintenance/restart_now", "POST", {})[0][:3] == "403" and list(tmp_path.iterdir()) == []
    assert call(boss, "/api/maintenance/reboot", "POST", {})[0][:3] == "404"
    assert json.loads(call(boss, "/api/maintenance/skip_week", "POST", {})[1])["ok"]
    (request,) = tmp_path.glob("root-*-skip_week.json")
    assert json.loads(request.read_text())["action"] == "skip_week"
    boss.backend.site = lambda: dataclasses.replace(site, maintenance=None)
    assert "Updates waiting" not in call(boss, "/")[1] and call(boss, "/api/maintenance/skip_week", "POST", {})[0][:3] == "404"


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


def test_the_tables_show_each_measure_on_its_own_line_in_one_order() -> None:
    from datetime import datetime

    line = lambda label, value: f'<div><span class="muted">{label}</span> {value}</div>'  # noqa: E731
    run = {"mem_mb": 38000, "mem_used": 3 * 1024**3, "mem_avg": 2 * 1024**3, "mem_peak": 4 * 1024**3,
           "cpus": 12, "cores_now": 7.26, "cores_used": 1.44, "cores_peak": 9.04}
    assert app._queue_memory(run) == line("now", app._size(3 * 1024**3)) + line("avg", app._size(2 * 1024**3)) + line("peak", app._size(4 * 1024**3)) + line("asked", app._queue_memory({"mem_mb": 38000}).split("</span> ")[1][:-6])
    assert app._queue_cores(run) == line("now", "7.3") + line("avg", "1.4") + line("peak", "9.0") + line("asked", "12")  # same order as memory
    assert app._queue_cores({"cpus": 4}) == line("asked", "4")  # waiting: only what it asked for, labelled the same way
    row = jobs_view.HistoryRow("5", "p:normal:x", "p", "ann", "normal", "COMPLETED", "0:0", datetime(2026, 1, 1), datetime(2026, 1, 1),
                               datetime(2026, 1, 1), 100, 0, 4 * 1024**3, 2 * 1024**3, avg_rss_bytes=1024**3, cpus=4, cpu_s=150.0)
    assert app._cores_used(row) == line("avg", "1.5") + line("asked", "4")
    assert app._finished_memory(row) == line("avg", app._size(1024**3)) + line("peak", app._size(2 * 1024**3)) + line("asked", app._size(4 * 1024**3))


def test_a_running_jobs_cpu_seconds_are_read_from_the_agents_metrics_file(tmp_path: Path) -> None:
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "job_7.prom").write_text('gpujobs_job_cpu_seconds_total{jobid="7"} 90.500\n', encoding="utf-8")
    old = gpu_jobs.STATE_DIR
    gpu_jobs.STATE_DIR = tmp_path
    try:
        assert app._live_memory("7") == {"cpu_ms": 90500}
    finally:
        gpu_jobs.STATE_DIR = old


def test_both_tables_sort_search_and_filter_from_the_address() -> None:
    fake = Fake()

    def queue_of(path: str) -> str:  # the queue card's table only: other cards name the jobs too
        return call(fake, path)[1].split('data-live="queue">', 1)[1].split("</section>", 1)[0]

    assert queue_of("/").index('jobid">#7') < queue_of("/").index('jobid">#8')  # slurm's order by default
    flipped = queue_of("/?sort=user&order=desc")
    assert flipped.index('jobid">#8') < flipped.index('jobid">#7')  # bob before alice
    assert '<td class="num muted">2</td>' in flipped.split('jobid">#8')[0]  # # stays slurm's place, whatever the order
    assert 'aria-sort="descending"' in flipped and "user=" not in flipped.split('aria-sort="descending"')[0][-200:]
    only_bob = queue_of("/?user=bob&sort=user&order=desc")
    assert 'jobid">#8' in only_bob and 'jobid">#7' not in only_bob
    assert "user=bob&amp;sort=user&amp;order=asc" in only_bob  # the heading flips the order and keeps the filter
    assert 'jobid">#7' in queue_of("/?q=ALICE+train") and 'jobid">#8' not in queue_of("/?q=ALICE+train")  # every word, any case
    assert "No jobs match" in queue_of("/?q=nobody") and "1 of 2 job(s)" in call(fake, "/?user=bob")[1]
    assert 'value="bob" selected' in call(fake, "/?user=bob")[1]  # the form shows the view it made

    at = lambda h: datetime(2026, 10, 19, h)  # noqa: DTZ001, E731
    rows = [jobs_view.HistoryRow("5", "p:normal:short", "p", "ann", "normal", "COMPLETED", "0:0", at(8), at(9), at(10), 60, 1, None),
            jobs_view.HistoryRow("6", "q:normal:long", "q", "bo", "normal", "FAILED", "1:0", at(8), at(9), at(11), 7200, 1, None),
            jobs_view.HistoryRow("4", "q:normal:never", "q", "bo", "normal", "CANCELLED", "0:0", at(7), None, at(9), 0, 1, None)]
    fake.backend = dataclasses.replace(fake.backend, history=lambda *_a: rows)
    fake.application = app.make_application(fake.backend)
    history = lambda path: call(fake, path)[1]  # noqa: E731
    took = history("/history?since=2026-10-01&sort=took&order=desc")
    assert took.index('jobid">#6') < took.index('jobid">#5') < took.index('jobid">#4')  # never started: last either way
    assert 'jobid">#4' in history("/history?since=2026-10-01&q=never") and 'jobid">#5' not in history("/history?since=2026-10-01&q=never")
    assert "q=never" in history("/history?since=2026-10-01&q=never&sort=job").split('class="pager"')[0]  # headings keep the search
    # the queue page's finished card has its own settings (prefix f): the two tables on one page keep apart
    both = call(fake, "/?fsort=took&forder=desc&user=bob")[1]
    card = both.split('data-live="recent">', 1)[1]
    assert card.index('jobid">#6') < card.index('jobid">#5')  # sorted by its own heading; the queue's user filter not applied
    assert 'name="fsort" value="took"' in both.split('aria-label="Queue search and filters"')[1].split("</form>")[0]


def test_a_colour_follows_its_project_and_the_rest_share_one_grey() -> None:
    day = lambda d: datetime(2026, 1, d)  # noqa: DTZ001, E731
    jobs = [(day(1), "old", 100.0), (day(1), jobs_view.OTHER_PROJECT, 999.0), *((day(1), f"p{i}", float(i)) for i in range(1, 9))]
    first = app.project_slots(jobs, day(2))
    assert sorted(first.values()) == list(range(8)) and set(first) == {"old", *(f"p{i}" for i in range(2, 9))}  # most used
    later = app.project_slots([*jobs, (day(20), "new", 50.0)], day(22))
    assert {p: later[p] for p in first if p != "p2"} == {p: s for p, s in first.items() if p != "p2"}  # nobody repainted
    assert later["new"] == first["p2"] and "p2" not in later  # the newcomer takes the colour the one it pushed out held
    app._SLOTS.set({"old": 0})
    rows = [jobs_view.UsageRow(day(1), p, 1.0, 1.0, 0.0, 0.0) for p in ("old", "p1", "p2")]
    chart = app._chart(rows, [day(1)], ["old", "p1", "p2"], jobs_view.Bucket.DAY)
    assert chart.count("other projects|") == 1 and "2.00 GPU-hours" in chart and "p1" not in chart  # one grey block
