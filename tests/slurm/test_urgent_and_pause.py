"""urgent.py's plan: what is in the way, what can be paused, and when it stops without submitting."""

from __future__ import annotations

from pathlib import Path

import gpu_jobs
import pytest
import urgent


def job(job_id: str, partition: str, state: str = "RUNNING") -> gpu_jobs.QueuedJob:
    return gpu_jobs.QueuedJob(job_id, partition, state, "None", f"p:x:{job_id}")


@pytest.fixture(autouse=True)
def sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(urgent, "elapsed", lambda _job: "1:00")
    for job_id, gpu in [("1", True), ("2", True), ("3", False), ("4", True)]:
        gpu_jobs.update_sidecar(job_id, gpu=gpu, what=f"job {job_id}")


def test_classes_of_what_is_in_the_way() -> None:
    queue = [job("1", "normal"), job("2", "steady"), job("3", "normal"), job("4", "urgent", "PENDING")]
    lines, victims, steady, wait_only = urgent.plan(
        queue, gpu=True, whole_box=False,
        leases=[("service-a", False), ("service-b", False), ("service-c", False), ("slurm", False)],
        unregistered=[(99, 500.0, "python patch4")],
    )
    assert victims == ["1"]  # job 3 holds no GPU, so a GPU-only urgent job leaves it alone
    assert steady == ["2"]
    assert wait_only == ["4", "lease:service-c", "pid:99"]
    text = "\n".join(lines)
    assert "PAUSABLE  service service-a" in text
    assert "PAUSABLE  service service-b" in text
    assert "loses all of that run" in text


def test_whole_box_puts_every_running_job_in_the_way() -> None:
    _, victims, _, _ = urgent.plan(
        [job("1", "normal"), job("3", "normal")], gpu=True, whole_box=True, leases=[], unregistered=[]
    )
    assert victims == ["1", "3"]


def test_stops_without_submitting_behind_a_wait_only_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: [job("2", "steady")])
    monkeypatch.setattr(urgent, "other_leases", list)
    monkeypatch.setattr(urgent.audit, "unregistered", list)
    monkeypatch.setattr(urgent, "free_memory_mb", lambda: None)
    submitted = []
    monkeypatch.setattr(urgent.subprocess, "run", lambda *a, **k: submitted.append(a))
    assert urgent.run(["--project", "p", "--what", "w", "--time", "0:10:00", "--", "true"]) == 1
    assert submitted == []


def test_a_memory_only_urgent_job_that_does_not_fit_puts_running_jobs_in_the_way() -> None:
    _, victims, _, _ = urgent.plan(
        [job("3", "normal")], gpu=False, whole_box=False, leases=[], unregistered=[], memory_short=True
    )
    assert victims == ["3"]


def test_memory_notation() -> None:
    assert (urgent.mem_mb("24G"), urgent.mem_mb("512M"), urgent.mem_mb("2048")) == (24576, 512, 2048)
