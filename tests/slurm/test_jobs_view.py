"""jobs_view over `sacct --json` recorded read-only from this box on 2026-10-19 (tests/slurm/fixtures/jobs_view/).

Recorded GPU jobs list their GPU in `tres.allocated` as `{type: gres, name: gpu, count: 1}` (job 34); squeue's
`%b` shows `gres/gpu:1`. The fixture keeps only the fields jobs_view reads; the hand-made jobs are marked.
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Iterator
import dataclasses
from datetime import date, datetime
from pathlib import Path

import gpu_jobs
import jobs_view
import pytest
from gpu_jobs import Event, EventAction, QueuedJob
from jobs_view import Bucket

FIXTURES = Path(__file__).parent / "fixtures" / "jobs_view"
SACCT = (FIXTURES / "sacct.json").read_text()
QUEUE = {job.job_id: job for job in gpu_jobs.parse_squeue((FIXTURES / "squeue.json").read_text())}
LONDON = "GMT0BST,M3.5.0/1,M10.5.0"  # the zone the fixture's day buckets were recorded in


@pytest.fixture(autouse=True)
def london_time(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """sacct prints local times, so the day a job started in depends on the zone the test runs in."""
    monkeypatch.setenv("TZ", LONDON)
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def fake_sacct(argv: list[str]) -> str:
    """Serve the recorded `sacct --json` for any window or job selection."""
    assert argv[:2] == ["sacct", "--json"]
    return SACCT


def sacct_json(*jobs: tuple[int, str, str, int, int, int]) -> str:
    """Hand-made jobs `(id, name, user, start, elapsed, gpus)`, a COMPLETED record each."""
    return json.dumps(
        {
            "jobs": [
                {
                    "job_id": job_id,
                    "name": name,
                    "user": user,
                    "partition": "steady",
                    "state": {"current": ["COMPLETED"]},
                    "time": {"start": start, "elapsed": elapsed},
                    "tres": {"allocated": [{"type": "gres", "name": "gpu", "count": gpus}] if gpus else []},
                }
                for job_id, name, user, start, elapsed, gpus in jobs
            ]
        }
    )


def queued(reason: str, **fields: object) -> QueuedJob:
    values = {"job_id": "300", "partition": "steady", "state": "PENDING", "reason": reason, "name": "p:restart:x"}
    return QueuedJob(**{**values, **fields})  # type: ignore[arg-type]


def test_waiting_reason_sentences() -> None:
    records = {"300": {"mem": "100000M"}}
    pause = Event("300", EventAction.PAUSE, "user1", 1_792_400_000 * 10**9)
    other = Event("301", EventAction.PAUSE, "someone", 1_792_500_000 * 10**9)
    urgent = Event("300", EventAction.URGENT_HOLD, "root", 1_792_500_000 * 10**9, {"urgent": "timing: time 3 models"})
    released = Event("300", EventAction.RELEASE, "slurm", 1_792_600_000 * 10**9)
    jobs = (
        queued("Priority", job_id="1", priority=9, name="a"),
        queued("Priority", job_id="2", priority=5, name="b"),
        queued("Priority", job_id="300", priority=5),
        queued("Dependency", job_id="4", priority=99, dependency="afterok:1"),
    )
    ahead = {job.job_id: job for job in jobs}
    sentence = {
        "dependency": jobs_view.waiting_reason(QUEUE["217"], QUEUE, {}, []),
        "paused": jobs_view.waiting_reason(queued("job requeued in held state"), {}, records, [pause, other]),
        "newest": jobs_view.waiting_reason(queued("JobHeldUser"), {}, records, [pause, urgent]),
        "nobody": jobs_view.waiting_reason(queued("JobHeldUser"), {}, records, [other]),
        "admin": jobs_view.waiting_reason(queued("JobHeldAdmin"), {}, records, []),
        "urgent": jobs_view.waiting_reason(queued("JobHeldAdmin"), {}, records, [pause, urgent, other]),
        "released": jobs_view.waiting_reason(queued("JobHeldAdmin"), {}, records, [urgent, released]),
        "resources": jobs_view.waiting_reason(queued("Resources", gpus=1), {}, records, []),
        # slurm's request wins over the record submit.py wrote (`scontrol update MinMemoryNode=` changes only slurm's)
        "updated": jobs_view.waiting_reason(queued("Resources", gpus=1, mem_mb=61440), {}, records, []),
        "whole": jobs_view.waiting_reason(queued("Resources", mem_mb=0), {}, records, []),
        "priority": jobs_view.waiting_reason(ahead["300"], ahead, records, []),
        "begin": jobs_view.waiting_reason(queued("BeginTime"), {}, records, []),
        "partition": jobs_view.waiting_reason(queued("PartitionConfig"), {}, records, []),
        "none": jobs_view.waiting_reason(QUEUE["202"], QUEUE, {}, []),
        "other": jobs_view.waiting_reason(queued("QOSMaxCpuPerUserLimit"), {}, records, []),
    }
    assert sentence == {
        "dependency": "waits for job 216 (project-a:restart:swe32-multipatch-timing) to finish",
        "paused": "paused by user1 at 2026-10-19 08:53 UTC; resume: resume.py 300",
        "newest": "paused by root at 2026-10-20 12:40 UTC; resume: resume.py 300",
        "nobody": "paused; resume: resume.py 300",
        "admin": "held by an admin (requeue limit or admin hold)",
        "urgent": "held since 2026-10-20 12:40 UTC while urgent work (timing: time 3 models) runs, so the jobs it "
                  "pushed out start first; released automatically",
        "released": "held by an admin (requeue limit or admin hold)",  # an old urgent hold, released since
        "resources": "waiting for 1 GPU(s) / 104.9 GB memory to free",
        "updated": "waiting for 1 GPU(s) / 64.4 GB memory to free",
        "whole": "waiting for all of the box's memory to free",
        "priority": "queued behind 2 job(s)",  # job 1 (higher) and job 2 (equal priority, submitted earlier)
        "begin": "requeued; waiting for its eligible time",
        "partition": "not allowed in partition steady",
        "none": "",
        "other": "slurm reason: QOSMaxCpuPerUserLimit",
    }


def test_chain_walks_both_ways_and_stays_finite() -> None:
    node = jobs_view.chain("205", QUEUE, {"300": {"after": ["205"], "name": "finished-follower"}})

    def ids(start: jobs_view.ChainNode, side: str) -> list[str]:
        found = []
        while getattr(start, side):
            start = getattr(start, side)[0]
            found.append(start.job_id)
        return found

    assert ids(node, "before") == ["204", "203", "202"]
    assert ids(node.after[0], "after")[:3] == ["207", "208", "209"]
    assert ids(node.after[0], "after")[-1] == "217"
    assert {n.job_id for n in node.after} == {"206", "300"}
    assert node.after[1].name == "finished-follower"  # not in the queue: name and link come from the record
    assert node.before[0].state == "PENDING"
    loop = {
        "1": queued("Dependency", job_id="1", dependency="afterok:2"),
        "2": queued("Dependency", job_id="2", dependency="afterok:1"),
    }
    assert jobs_view.chain("1", loop, {}).before[0].before == ()
    depth = 40  # layers of two jobs, each depending on both jobs below it: 2**40 paths, 80 jobs
    layers = {
        str(2 * layer + side): queued(
            "Dependency",
            job_id=str(2 * layer + side),
            dependency=f"afterok:{2 * layer + 2}:{2 * layer + 3}" if layer < depth - 1 else "",
        )
        for layer in range(depth)
        for side in (1, 2)
    }
    top = jobs_view.chain("1", layers, {})

    def count(n: jobs_view.ChainNode) -> int:
        return 1 + sum(count(c) for c in n.before)

    assert count(top) <= 2 * depth  # each job once, never once per path
    deep = {str(i): queued("Dependency", job_id=str(i), dependency=f"afterok:{i + 1}") for i in range(1, 3000)}
    assert count(jobs_view.chain("1", deep, {})) == jobs_view.MAX_CHAIN_DEPTH + 1  # not a RecursionError


def test_project_usage_gpu_hours_per_day_and_week() -> None:
    def usage(bucket: Bucket, text: str | None = None) -> dict[tuple[date, str], float]:
        rows = jobs_view.project_usage("2026-10-15", None, bucket, (lambda _: text) if text else fake_sacct)
        return {(r.period, r.project): r.gpu_hours for r in rows if r.gpu_hours}

    day = usage(Bucket.DAY)
    assert day[date(2026, 10, 16), "stagec"] == pytest.approx((122 + 6 + 59 + 3 + 43) / 3600)  # job 33 "stagec:urgent" has no class field: (other)
    assert day[date(2026, 10, 17), "service-c"] == pytest.approx((1483 + 475) / 3600)
    assert day[date(2026, 10, 18), "service-c"] == pytest.approx(12512 / 3600)
    assert (date(2026, 10, 17), "project-a") in day  # job 57 started the 17th, ran 103 s
    week = usage(Bucket.WEEK)
    assert week[date(2026, 10, 12), "service-c"] == pytest.approx(14470 / 3600)  # Saturday and Sunday: one Monday
    assert week[date(2026, 10, 19), "project-a"] == pytest.approx(11047 / 3600)  # job 202
    started = 1_792_105_200  # 2026-10-16 01:00 BST
    forged = "p:restart:x|user1|steady|FAILED\n999|p:restart:y|user1|steady|FAILED|1:0"
    hand_made_ids = (1, 2, 3, 4)
    hand_made = sacct_json(
        (1, "p:restart:x", "u", started, 3600, 2),
        (2, "plain", "u", 0, 3600, 1),
        (3, "cpu:restart:y", "u", started, 3600, 0),
        (4, forged, "mallory", started, 1800, 1),
    )
    # two GPUs for an hour is 2 GPU-hours; the never-started job and the CPU-only job add nothing; a job name with
    # '|' and a newline is one row of its owner's, billed once, and shifts no column nor adds a row
    assert usage(Bucket.DAY, hand_made) == {(date(2026, 10, 16), "p"): 2.5}
    # core-hours: 4 cores held for an hour are 4; "used" counts ended jobs' CPU time against what those held
    ran = jobs_view.HistoryRow("9", "p:resume:x", "p", "u", "cpu", "COMPLETED", "0:0", None, datetime(2026, 10, 16, 1),
                               datetime(2026, 10, 16, 2), 3600, 0, None, node="box", cpus=4, cpu_s=3600.0)
    running = dataclasses.replace(ran, job_id="10", state="RUNNING", end=None, cpu_s=0.0)
    elsewhere = dataclasses.replace(ran, job_id="11", node="laptop")
    [row] = jobs_view.usage_rows([ran, running])
    assert (row.core_hours, row.core_hours_ended, row.core_hours_used) == (8.0, 4.0, 1.0)
    # the machine's share counts only its own jobs, against its own cores and GPUs over the window
    share = jobs_view.machine_share([ran, running, elsewhere], {"box": (8, 1)}, 10.0)
    assert (share.core_hours, share.core_hours_possible, share.core_hours_ended, share.gpu_hours_possible) == (8.0, 80.0, 4.0, 10.0)
    rows = jobs_view.history("2026-10-15", runner=lambda _: hand_made)
    assert [(r.job_id, r.user, r.state, r.name) for r in rows][-1] == ("4", "mallory", "COMPLETED", forged)
    assert len(rows) == len(hand_made_ids)  # the forged name adds no row


def test_history_filters_and_efficiency_on_recorded_json() -> None:
    rows = jobs_view.history("2026-10-15", user="user1", project="service-c", state="failed", runner=fake_sacct)
    assert [r.job_id for r in rows] == ["53", "224", "225"]
    assert (rows[0].gpus, rows[0].max_rss_bytes, rows[0].req_mem_bytes) == (1, 13861380 * 1024, 100000 * 1024**2)
    cancelled = jobs_view.history("2026-10-15", state="CANCELLED", runner=fake_sacct)
    assert "32" in [r.job_id for r in cancelled]  # "CANCELLED by 1000" counts as CANCELLED
    assert jobs_view.history("2026-10-15", user="nobody", runner=fake_sacct) == []
    eff = jobs_view.efficiency("57", fake_sacct, {"legs": [{"peak_gpu_mib": 100.0}, {"peak_gpu_mib": 18353.0}]})
    assert eff.req_mem_bytes == 100000 * 1024**2
    assert eff.peak_cpu_mem_bytes == 3174880 * 1024
    assert eff.peak_gpu_mib == pytest.approx(18353.0)
    assert eff.cpu_use == pytest.approx(101.333363 / (103 * 4))  # TotalCPU 101.333363 s over 4 CPUs for 103 s
    assert rows[0].exit_code == "2:0"
    running = next(r for r in jobs_view.history("2026-10-15", state="running", runner=fake_sacct))
    assert (running.job_id, running.end, running.start is not None) == ("239", None, True)  # end is 0 until it ends
    assert jobs_view.efficiency("999", lambda _: "", {}).cpu_use is None
    # a job cancelled before it ran: sacct says "None assigned", which is no node (the page showed "on None assigned")
    assert (jobs_view._row({"nodes": jobs_view.NO_NODE}).node, jobs_view._row({"nodes": "gpu-node-1"}).node) == ("", "gpu-node-1")


def test_log_tail_filters(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    log.write_text(
        "start\nWARNING: low disk\nstep 1 ok\nTraceback (most recent call last)\nstep 2 ok\nterrors are fine\n"
    )
    assert jobs_view.log_tail(str(log)).splitlines()[0] == "start"
    assert jobs_view.log_tail(str(log), errors_only=True).splitlines() == [
        "WARNING: low disk",
        "Traceback (most recent call last)",
    ]
    assert jobs_view.log_tail(str(log), text="step").splitlines() == ["step 1 ok", "step 2 ok"]
    assert jobs_view.log_tail(str(log), text="warning", errors_only=True) == "WARNING: low disk"
    assert jobs_view.log_tail(str(log), max_bytes=20) == "terrors are fine"  # the cut-off first line is dropped
    assert jobs_view.log_tail(str(tmp_path / "missing.log")) == jobs_view.NO_LOG
    fifo = tmp_path / "fifo.log"
    os.mkfifo(fifo)
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("log_tail blocked on a FIFO"))
    signal.alarm(5)
    try:
        assert jobs_view.log_tail(str(fifo)) == jobs_view.NO_LOG  # a blocking open() would wait for a writer
    finally:
        signal.alarm(0)


def test_unreadable_log_says_so(tmp_path: Path) -> None:
    log = tmp_path / "private.log"
    log.write_text("secret\n")
    log.chmod(0)
    try:
        log.read_bytes()
    except PermissionError:
        assert jobs_view.log_tail(str(log)) == "no permission to read this log"
    else:
        pytest.skip("running as a user that ignores file modes")


def test_can_act_owner_or_admin() -> None:
    job = queued("None", user="user1")
    assert jobs_view.can_act(job, "user1", admin=False)
    assert not jobs_view.can_act(job, "other", admin=False)
    assert jobs_view.can_act(job, "other", admin=True)
    assert not jobs_view.can_act(queued("None"), "", admin=False)  # an unknown owner never matches an empty name
