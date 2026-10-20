"""The OOD jobs app: every route answers on fake data, and the POST defences refuse what they must."""

from __future__ import annotations

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
import pytest
from conftest import SLURM

sys.path.insert(0, str(SLURM / "ood_app"))
import app

HOST = "ood.example:8080"
MOUNT = "/pun/sys/gpujobs"  # where OOD serves the app (Passenger's SCRIPT_NAME)


def queued(job_id: str, user: str, name: str = "proj:normal:train", partition: str = "normal") -> gpu_jobs.QueuedJob:
    return gpu_jobs.QueuedJob(job_id, partition, "RUNNING", "None", name, user=user, gpus=1)


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
            events=lambda _j: [gpu_jobs.Event("7", "pause", "alice", 1_700_000_000_000_000_000)],
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
            live_memory=lambda _j: {"gpu": 3 * 1024**3},
            finished=lambda _j: None,
            now=lambda: datetime(2026, 10, 19, 12, 0),  # noqa: DTZ001 - local, like the times sacct prints
            project_order=lambda: ["proj"],
            capacity=lambda: {"cpus": 20, "cpus_used": 8, "mem_mb": 102400, "mem_used_mb": 65536, "gpus": 1, "gpus_used": 1},
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
        assert (app._running_note(2, 0), app._running_note(2, 1)) == ("none on the GPU", "1 on the GPU")
        assert "37% busy" in body and "1 of 1 given to jobs" in body  # how busy, not only whether it is given
        assert "8 of 20" in body and "12 free" in body and "64 of 100 GB" in body and "36 GB free" in body
        assert "of 4 GB asked" in body  # in the same units as the memory it uses


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
    assert 'data-action="resume"' not in page  # job 7 runs: nothing to resume
    assert "/pun/sys/jobstats/7" in page
    # links are absolute from the mount: opened without its trailing slash, the page still finds its stylesheet
    assert f'href="{MOUNT}/static/app.css"' in page
