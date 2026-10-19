"""jobs_view over text recorded read-only from this box on 2026-10-19 (tests/slurm/fixtures/jobs_view/).

Recorded GPU jobs show their GPU in `sacct` AllocTRES as `gres/gpu=1` (job 34: `billing=4,cpu=4,gres/gpu=1,
mem=4G,node=1`); squeue's `%b` shows `gres/gpu:1`. Jobs 53/55/226.. are real; the two-GPU row is hand-made.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import gpu_jobs
import jobs_view
import pytest
from gpu_jobs import Event, EventAction, QueuedJob
from jobs_view import Bucket

FIXTURES = Path(__file__).parent / "fixtures" / "jobs_view"
SACCT_X = (FIXTURES / "sacct_x.txt").read_text()
SACCT_STEPS = (FIXTURES / "sacct_steps.txt").read_text()
QUEUE = {job.job_id: job for job in gpu_jobs.parse_squeue((FIXTURES / "squeue.json").read_text())}
HEADER = "JobID|JobName|User|Partition|State|ExitCode|Submit|Start|End|ElapsedRaw|AllocTRES|ReqMem\n"


def fake_sacct(argv: list[str]) -> str:
    """Serve the recorded -X table, or the recorded step table (MaxRSS, TotalCPU), by what was asked for."""
    assert argv[0] == "sacct"
    return SACCT_X if "-X" in argv else SACCT_STEPS


def queued(reason: str, **fields: object) -> QueuedJob:
    values = {"job_id": "300", "partition": "steady", "state": "PENDING", "reason": reason, "name": "p:restart:x"}
    return QueuedJob(**{**values, **fields})  # type: ignore[arg-type]


def test_waiting_reason_sentences() -> None:
    records = {"300": {"mem": "100000M"}}
    pause = Event("300", EventAction.PAUSE, "user1", 1_792_400_000 * 10**9)
    other = Event("301", EventAction.PAUSE, "someone", 1_792_500_000 * 10**9)
    urgent = Event("300", EventAction.URGENT_HOLD, "root", 1_792_500_000 * 10**9)
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
        "resources": jobs_view.waiting_reason(queued("Resources", gpus=1), {}, records, []),
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
        "resources": "waiting for 1 GPU(s) / 100000M memory to free",
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


def test_project_usage_gpu_hours_per_day_and_week() -> None:
    def usage(bucket: Bucket, text: str | None = None) -> dict[tuple[date, str], float]:
        rows = jobs_view.project_usage("2026-10-15", None, bucket, (lambda _: text) if text else fake_sacct)
        return {(r.period, r.project): r.gpu_hours for r in rows}

    day = usage(Bucket.DAY)
    assert day[date(2026, 10, 16), "stagec"] == pytest.approx((122 + 6 + 59 + 3 + 43) / 3600)  # job 33 "stagec:urgent" has no class field: (other)
    assert day[date(2026, 10, 17), "service-c"] == pytest.approx((1483 + 475) / 3600)
    assert day[date(2026, 10, 18), "service-c"] == pytest.approx(12512 / 3600)
    assert (date(2026, 10, 17), "project-a") in day  # job 57 started the 17th, ran 103 s
    week = usage(Bucket.WEEK)
    assert week[date(2026, 10, 12), "service-c"] == pytest.approx(14470 / 3600)  # Saturday and Sunday: one Monday
    assert week[date(2026, 10, 19), "project-a"] == pytest.approx(1956 / 3600)  # running job: elapsed so far
    two = (
        HEADER
        + "1|p:restart:x|u|steady|COMPLETED|0:0|2026-10-16T01:00:00|2026-10-16T01:00:00|2026-10-16T02:00:00|3600|gres/gpu=2,mem=1G|1G\n"
        + "2|plain|u|steady|COMPLETED|0:0|2026-10-16T01:00:00|None|2026-10-16T02:00:00|3600|gres/gpu=1|1G\n"
        + "3|cpu:restart:y|u|steady|COMPLETED|0:0|2026-10-16T01:00:00|2026-10-16T01:00:00|2026-10-16T02:00:00|3600|cpu=4|1G\n"
    )
    # two GPUs for an hour is 2 GPU-hours; the never-started job and the CPU-only job add nothing
    assert usage(Bucket.DAY, two) == {(date(2026, 10, 16), "p"): 2.0}


def test_history_filters_and_efficiency_on_recorded_text() -> None:
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
    assert eff.cpu_use == pytest.approx((60 + 41.333) / (103 * 4))  # TotalCPU 01:41.333 over 4 CPUs for 103 s
    assert jobs_view.efficiency("999", lambda _: "", {}).cpu_use is None


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
